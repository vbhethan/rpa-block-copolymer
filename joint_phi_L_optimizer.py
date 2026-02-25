"""
Joint Optimization of Density Field and Box Dimensions
=======================================================

Optimizes the RPA free energy functional with respect to both the order
parameter field delta_phi (via projected gradient descent) and the box
dimensions L_1, ..., L_d (via standard gradient descent on log L).

Key insight: the constraint set for delta_phi is independent of the box
dimensions. Both constraints (local incompressibility and mass conservation)
are purely algebraic on the grid values, so the projection operator is
the same regardless of the box lengths. The box lengths are unconstrained
positive parameters that interact with the density only through the
functional evaluation (via the wavevectors q = 2*pi*n/L and the volume
prefactors).
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
import math
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Callable
from gen_initial_conditions import (
    generate_initial_hexagonal_ordered_structure,
    generate_hexagonal_A_in_BC_matrix,
)
from rpa import BlockCopolymerFreeEnergy


@dataclass
class JointOptimizationResult:
    """Container for joint optimization results."""

    F_history: list[float] = field(default_factory=list)
    grad_phi_norm_history: list[float] = field(default_factory=list)
    grad_box_norm_history: list[float] = field(default_factory=list)
    L_history: list[list[float]] = field(default_factory=list)
    converged: bool = False
    n_outer_iterations: int = 0
    message: str = ""


def backtracking_line_search_phi(
    model: nn.Module,
    delta_phi: torch.Tensor,
    direction: torch.Tensor,
    current_F: float,
    proj_grad: torch.Tensor,
    alpha_init: float = 1.0,
    beta: float = 0.5,
    c: float = 1e-4,
    min_alpha: float = 1e-14,
    Gamma_ij: torch.Tensor | None = None,
) -> tuple[float, torch.Tensor, float]:
    """
    Backtracking line search for the density field update.

    Checks both the Armijo condition and positivity of the full densities
    (to keep the log term well-defined).
    """
    slope = (proj_grad * direction).sum().item()
    if slope >= 0:
        return 0.0, delta_phi, current_F

    f_expanded = model._f_expanded() * model.phi_bar
    alpha = alpha_init

    while alpha > min_alpha:
        candidate = delta_phi + alpha * direction

        candidate = model._project_order_parameter(candidate)

        rho = candidate + f_expanded
        if rho.min().item() < 1e-10:
            alpha *= beta
            continue

        with torch.no_grad():
            F_candidate = model(candidate, Gamma_ij=Gamma_ij).item()

        if F_candidate <= current_F + c * alpha * slope:
            return alpha, candidate, F_candidate

        alpha *= beta

    return 0.0, delta_phi, current_F


def generate_initial_delta_phi(model: nn.Module) -> torch.Tensor:
    """
    Generate initial delta_phi for the density field.
    """
    phi_init = torch.randn(
        model.n_components, *model.grid_shape, dtype=model.real_dtype
    )
    phi_init = model._project_order_parameter(phi_init)
    return phi_init


def backtracking_line_search_box(
    model: nn.Module,
    delta_phi: torch.Tensor,
    current_F: float,
    grad_log_L: torch.Tensor,
    grad_scale: float = 1.0,
    alpha_init: float = 0.1,
    beta: float = 0.5,
    c: float = 1e-4,
    min_alpha: float = 1e-10,
    max_box_ratio: float = 2.0,
) -> tuple[float, float]:
    """
    Backtracking line search for box dimension update.

    Steps in log-L space, which naturally preserves positivity.
    Includes a guard against excessively large box changes in a single step.

    Parameters
    ----------
    grad_log_L : torch.Tensor
        Gradient w.r.t. log_L, shape (ndim,)
    """
    if grad_log_L.norm().item() <= 0:
        return 0.0, current_F

    direction = -grad_scale * grad_log_L
    slope = grad_log_L.dot(direction).item()

    if slope >= 0:
        return 0.0, current_F

    log_L_orig = model.log_L.data.clone()
    alpha = alpha_init

    while alpha > min_alpha:
        new_log_L = log_L_orig + alpha * direction

        if (new_log_L - log_L_orig).abs().max().item() > max_box_ratio:
            alpha *= beta
            continue

        model.log_L.data.copy_(new_log_L)

        with torch.no_grad():
            F_candidate = model(delta_phi).item()

        if F_candidate <= current_F + c * alpha * slope:
            return alpha, F_candidate

        alpha *= beta

    model.log_L.data.copy_(log_L_orig)
    return 0.0, current_F


def optimize_joint(
    model: nn.Module,
    n_outer: int = 500,
    n_inner_phi: int = 50,
    n_inner_box: int = 10,
    lr_phi: float = 0.5,
    lr_box: float = 0.1,
    box_grad_scale: float = 1e4,
    tol_grad_phi: float = 1e-6,
    tol_grad_box: float = 1e-7,
    use_line_search: bool = True,
    log_every: int = 10,
    callback: Optional[Callable] = None,
) -> JointOptimizationResult:
    """
    Alternating projected gradient descent for density + box optimization.

    The algorithm alternates between:
      1. Many inner steps optimizing delta_phi with fixed box (PGD)
      2. A few steps optimizing box lengths with fixed density (standard GD)

    This separation exploits the fact that:
      - The constraint set is independent of box dimensions
      - Gamma_ij only needs recomputing when the box changes
      - The density landscape is smoother than the box landscape

    Parameters
    ----------
    model : nn.Module
        The BlockCopolymerFreeEnergy module with optimize_box=True
    n_outer : int
        Number of outer (alternating) iterations
    n_inner_phi : int
        Number of PGD steps for density per outer iteration
    n_inner_box : int
        Number of GD steps for box per outer iteration
    lr_phi, lr_box : float
        Initial step sizes (used as starting points for line search)
    tol_grad_phi, tol_grad_box : float
        Convergence tolerances on projected gradient norms
    """
    if not model.optimize_box:
        raise ValueError("model.optimize_box must be True for joint optimization")

    result = JointOptimizationResult()

    for outer in range(n_outer):
        # =================================================================
        # Phase 1: Optimize density field with fixed box (PGD)
        # =================================================================
        delta_phi = model.delta_phi.data.clone()

        with torch.no_grad():
            K2 = model._compute_K2(model.L)
            Gamma_ij_cached = model._compute_gamma_ij(K2)

        for inner in range(n_inner_phi):
            delta_phi_var = delta_phi.clone().detach().requires_grad_(True)
            F_val = model(delta_phi_var, Gamma_ij=Gamma_ij_cached)
            (raw_grad,) = torch.autograd.grad(F_val, delta_phi_var)
            current_F = F_val.item()

            proj_grad = model._project_order_parameter(raw_grad)
            grad_phi_norm = proj_grad.norm().item()

            if grad_phi_norm < tol_grad_phi:
                break

            if use_line_search:
                _, delta_phi, _ = backtracking_line_search_phi(
                    model,
                    delta_phi,
                    -proj_grad,
                    current_F,
                    proj_grad,
                    alpha_init=lr_phi,
                    Gamma_ij=Gamma_ij_cached,
                )
            else:
                delta_phi = delta_phi - lr_phi * proj_grad
                delta_phi = model._project_order_parameter(delta_phi)
                rho = delta_phi + model._f_expanded() * model.phi_bar
                if rho.min().item() < 1e-10:
                    delta_phi = delta_phi * 0.9
                    delta_phi = model._project_order_parameter(delta_phi)

        model.delta_phi.data.copy_(delta_phi)

        # =================================================================
        # Phase 2: Optimize box dimensions with fixed density (standard GD)
        # =================================================================
        for inner_box in range(n_inner_box):
            model.log_L.requires_grad_(True)

            F_val = model(delta_phi.detach())
            F_val.backward()

            grad_log_L = model.log_L.grad.clone()
            current_F = F_val.item()

            model.log_L.grad = None

            grad_box_norm = grad_log_L.norm().item()

            if grad_box_norm < tol_grad_box:
                break

            if use_line_search:
                _, _ = backtracking_line_search_box(
                    model,
                    delta_phi,
                    current_F,
                    grad_log_L,
                    grad_scale=box_grad_scale,
                    alpha_init=lr_box,
                )
            else:
                model.log_L.data -= lr_box * box_grad_scale * grad_log_L

        # =================================================================
        # Logging and convergence check
        # =================================================================
        with torch.no_grad():
            F_current = model(delta_phi).item()

        result.F_history.append(F_current)
        result.grad_phi_norm_history.append(grad_phi_norm)
        result.grad_box_norm_history.append(grad_box_norm)
        result.L_history.append(model.L.tolist())

        if outer % log_every == 0:
            L_str = ", ".join(f"L{d}={v:.6f}" for d, v in enumerate(model.L.tolist()))
            print(
                f"Outer {outer:4d} | F = {F_current:.6e} | "
                f"|grad_phi| = {grad_phi_norm:.4e} | |grad_L| = {grad_box_norm:.4e} | "
                f"{L_str}"
            )

        if callback is not None:
            callback(outer, model, F_current, grad_phi_norm, grad_box_norm)

        if grad_phi_norm < tol_grad_phi and grad_box_norm < tol_grad_box:
            result.converged = True
            result.n_outer_iterations = outer
            result.message = (
                f"Converged at outer iteration {outer}: "
                f"|grad_phi| = {grad_phi_norm:.4e}, |grad_L| = {grad_box_norm:.4e}"
            )
            print(f"\n{result.message}")
            return result

    result.n_outer_iterations = n_outer
    result.message = (
        f"Did not converge after {n_outer} outer iterations. "
        f"|grad_phi| = {grad_phi_norm:.4e}, |grad_L| = {grad_box_norm:.4e}"
    )
    print(f"\n{result.message}")
    return result


# =============================================================================
# Period scanning utility
# =============================================================================


def scan_box_lengths(
    model: nn.Module,
    L_range: torch.Tensor,
    n_density_steps: int = 500,
    lr_phi: float = 0.5,
    tol_phi: float = 1e-6,
    isotropic: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Scan over box lengths to identify basins of attraction.

    For each candidate L, optimize the density field to convergence and
    record the resulting free energy. This helps identify the global
    minimum over L, which gradient descent alone may miss due to the
    multiple-minima structure of the box-length landscape.

    Parameters
    ----------
    model : nn.Module
        The BlockCopolymerFreeEnergy module
    L_range : torch.Tensor
        1D tensor of box lengths to scan
    n_density_steps : int
        Max PGD steps per box length
    isotropic : bool
        If True, set all box dimensions to the same L for each scan point

    Returns
    -------
    dict with 'L_values', 'F_values', 'L_optimal'
    """
    F_values = []
    for i, L_val_tensor in enumerate(L_range):
        L_val = L_val_tensor.item()

        if model.optimize_box:
            if isotropic:
                model.log_L.data.fill_(torch.log(L_val_tensor).item())
            else:
                model.log_L.data[0] = torch.log(L_val_tensor).item()

        with torch.no_grad():
            model.delta_phi.data.normal_(
                0, model.delta_phi.data.std().item() * 0.1 + 0.01
            )
            model.delta_phi.data = model._project_order_parameter(model.delta_phi.data)

        delta_phi = model.delta_phi.data.clone()
        for step in range(n_density_steps):
            delta_phi_var = delta_phi.clone().detach().requires_grad_(True)
            F_val = model(delta_phi_var)
            F_val.backward()
            proj_grad = model._project_order_parameter(delta_phi_var.grad.clone())

            if proj_grad.norm().item() < tol_phi:
                break

            _, delta_phi, _ = backtracking_line_search_phi(
                model,
                delta_phi,
                -proj_grad,
                F_val.item(),
                proj_grad,
                alpha_init=lr_phi,
            )

        with torch.no_grad():
            F_opt = model(delta_phi).item()
        F_values.append(F_opt)

        if verbose:
            print(f"  L = {L_val:.3f} | F = {F_opt:.8e} | steps = {step + 1}")

    F_tensor = torch.tensor(F_values)
    best_idx = F_tensor.argmin().item()

    if model.optimize_box:
        if isotropic:
            model.log_L.data.fill_(torch.log(L_range[best_idx]).item())
        else:
            model.log_L.data[0] = torch.log(L_range[best_idx]).item()

    return {
        "L_values": L_range.tolist(),
        "F_values": F_values,
        "L_optimal": L_range[best_idx].item(),
        "F_optimal": F_values[best_idx],
    }


if __name__ == "__main__":
    #  3 component block star copolymer
    N = 100
    chi_AB = 20
    chi_AC = 30
    chi_BC = 28
    chi_matrix = (
        torch.tensor(
            [
                [0.0, chi_AB, chi_AC],
                [chi_AB, 0.0, chi_BC],
                [chi_AC, chi_BC, 0.0],
            ],
            dtype=torch.float64,
        )
        * 1
    )
    f_A = 0.31
    f_B = 0.36
    f_C = 1 - f_A - f_B
    l_ij_matrix = torch.zeros((3, 3), dtype=torch.float64)
    # l_ij_matrix[0, 2] = f_B * N * 1.0**2
    # l_ij_matrix[2, 0] = f_B * N * 1.0**2
    block_fractions = torch.tensor([f_A, f_B, f_C], dtype=torch.float64)

    Lx = 20.0
    Ly = Lx * math.sqrt(3)

    grid_res = 32

    delta_phi = generate_hexagonal_A_in_BC_matrix(
        grid_shape=(grid_res, grid_res),
        block_fractions=block_fractions,
        Lx=Lx,
        Ly=Ly,
        amplitude=0.2,
    )

    # Add some noise to the delta_phi
    delta_phi = delta_phi + torch.randn_like(delta_phi) * 0.01

    model = BlockCopolymerFreeEnergy(
        N=N,
        chi_matrix=chi_matrix,
        l_ij_matrix=l_ij_matrix,
        block_fractions=block_fractions,
        optimize_box=True,
        grid_shape=(grid_res, grid_res),
        box_lengths=(Lx, Ly),
        init_amplitude=0.05,
    )

    model.delta_phi.data.copy_(delta_phi)

    print("initial box lengths:", model.L.tolist())
    print("initial free energy:", model(model.get_order_parameters()).item())

    result = optimize_joint(
        model,
        n_outer=500,
        lr_box=0.01,
        lr_phi=0.1,
        n_inner_phi=500,
        n_inner_box=5,
        tol_grad_phi=1e-5,
        tol_grad_box=1e-6,
        log_every=50,
    )
    print("final box lengths:", model.L.tolist())
    print("final free energy:", result.F_history[-1])
    # Save results for visualization with vis.py
    torch.save(model.get_order_parameters(), "delta_phi.pt")
    print("saved to delta_phi.pt")
    torch.save(model, "block_copolymer_model.pt")
    with open("joint_optimization_result.pkl", "wb") as f:
        pickle.dump(
            {
                "F_history": result.F_history,
                "grad_phi_norm_history": result.grad_phi_norm_history,
                "grad_box_norm_history": result.grad_box_norm_history,
                "L_history": result.L_history,
                "converged": result.converged,
                "n_outer_iterations": result.n_outer_iterations,
                "message": result.message,
            },
            f,
        )
