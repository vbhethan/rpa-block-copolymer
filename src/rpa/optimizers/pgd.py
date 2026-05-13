from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..free_energy import BlockCopolymerFreeEnergy
from ..simulation_io import SimulationData


def _model_real_dtype(model) -> torch.dtype:
    return model.delta_phi.dtype


def backtracking_line_search_phi(
    model: nn.Module,
    delta_phi: torch.Tensor,
    direction: torch.Tensor,
    current_F: float,
    proj_grad: torch.Tensor,
    alpha_init: float = 1.0,
    beta: float = 0.5,
    c: float = 1e-4,
    min_alpha: float = None,
    Gamma_ij: torch.Tensor | None = None,
) -> tuple[float, torch.Tensor, float]:
    """
    Backtracking line search for the density field update.

    Checks both the Armijo condition and positivity of the full densities
    (to keep the log term well-defined).
    """
    if min_alpha is None:
        min_alpha = 1e-14 if _model_real_dtype(model) == torch.float64 else 1e-7
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
            F_candidate = model(candidate, Gamma_ij=Gamma_ij, project=False).item()

        if F_candidate <= current_F + c * alpha * slope:
            return alpha, candidate, F_candidate

        alpha *= beta

    return 0.0, delta_phi, current_F


def backtracking_line_search_box(
    model: nn.Module,
    delta_phi: torch.Tensor,
    current_F: float,
    grad_log_L: torch.Tensor,
    grad_scale: float = 1.0,
    alpha_init: float = 0.1,
    beta: float = 0.5,
    c: float = 1e-4,
    min_alpha: float = None,
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
    if min_alpha is None:
        min_alpha = 1e-10 if _model_real_dtype(model) == torch.float64 else 1e-6
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


def optimize_phi_only(
    model: nn.Module,
    n_steps: int = 2000,
    lr_phi: float = 0.1,
    tol_F: float = 1e-6,
    patience: int = 50,
) -> SimulationData:
    """
    Projected gradient descent on delta_phi with box lengths held fixed.

    Convergence is declared when the free energy has not changed by more than
    ``tol_F`` for ``patience`` consecutive steps, which handles the small
    oscillations that appear near convergence better than a gradient norm test.

    Parameters
    ----------
    model : BlockCopolymerFreeEnergy
        Box lengths are not modified; optimize_box may be True or False.
    n_steps : int
        Maximum number of PGD steps.
    lr_phi : float
        Initial step size passed to the backtracking line search.
    tol_F : float
        Convergence threshold on |F - F_ref|.
    patience : int
        Number of consecutive steps within tol_F required to stop.

    Returns
    -------
    SimulationData
        Trajectory of F and phi at each step; box_lengths is fixed and
        broadcast across all frames.
    """
    delta_phi = model.delta_phi.data.detach().clone()

    with torch.no_grad():
        if model.optimize_box:
            K2 = model._compute_K2(model.L)
            Gamma_ij = model._compute_gamma_ij(K2)
        else:
            Gamma_ij = model._Gamma_ij_cached

    F_trajectory = []
    phi_trajectory = []
    converged = False
    grad_norm = float("inf")
    stall_count = 0
    F_ref = float("inf")

    with torch.no_grad():
        current_F = model(delta_phi, Gamma_ij=Gamma_ij, project=False).item()

    pbar = tqdm(range(n_steps))

    for step in pbar:
        pbar.set_description(
            f"Step {step:4d} | F = {current_F:.6e} | |grad_phi| = {grad_norm:.4e} | stall = {stall_count:02d}/{patience}"
        )
        delta_phi.requires_grad_(True)
        F_val = model(delta_phi, Gamma_ij=Gamma_ij, project=False)
        (raw_grad,) = torch.autograd.grad(F_val, delta_phi)
        current_F = F_val.item()
        delta_phi = delta_phi.detach()

        proj_grad = model._project_order_parameter(raw_grad)
        grad_norm = proj_grad.norm().item()

        F_trajectory.append(current_F)
        phi_trajectory.append(delta_phi.cpu().numpy())

        if abs(current_F - F_ref) < tol_F:
            stall_count += 1
        else:
            stall_count = 0
            F_ref = current_F

        if stall_count >= patience:
            converged = True
            break

        _, delta_phi, current_F = backtracking_line_search_phi(
            model,
            delta_phi,
            -proj_grad,
            current_F,
            proj_grad,
            alpha_init=lr_phi,
            Gamma_ij=Gamma_ij,
        )

    if converged:
        print(f"Converged at step {step} (F stable for {patience} steps)")
    else:
        print(f"Did not converge after {n_steps} steps")

    model.delta_phi.data.copy_(delta_phi)

    result = SimulationData.from_model(model)
    n = len(F_trajectory)
    result.F = np.array(F_trajectory)
    result.phi = np.array(phi_trajectory)
    box_lengths = model.L.detach().cpu().tolist()
    result.box_lengths = np.broadcast_to(
        np.array(box_lengths)[np.newaxis], (n, len(box_lengths))
    ).copy()
    result.converged = converged
    return result


def optimize_box_only(
    model: nn.Module,
    n_steps: int = 500,
    lr_box: float = 1.0,
    box_grad_scale: float = 10.0,
    tol_grad: float = 1e-6,
    log_every: int = 50,
) -> SimulationData:
    """
    Gradient descent on box lengths with delta_phi held fixed.

    Parameters
    ----------
    model : BlockCopolymerFreeEnergy
        Must have optimize_box=True. delta_phi is treated as a constant.
    n_steps : int
        Maximum number of gradient steps.
    lr_box : float
        Initial step size passed to the backtracking line search.
    box_grad_scale : float
        Scale applied to the log-L gradient direction before the line search.
    tol_grad : float
        Stop when ||grad log_L|| < tol_grad.
    log_every : int
        Print a status line every this many steps.

    Returns
    -------
    SimulationData
        Trajectory of F and box_lengths at each step; phi is the fixed field
        broadcast across all frames.
    """
    if not model.optimize_box:
        raise ValueError("model.optimize_box must be True")

    delta_phi = model.delta_phi.data.detach().clone()

    F_trajectory = []
    box_lengths_trajectory = []
    converged = False
    grad_norm = float("inf")

    with torch.no_grad():
        current_F = model(delta_phi, project=False).item()

    for step in tqdm(range(n_steps)):
        model.log_L.requires_grad_(True)
        F_val = model(delta_phi, project=False)
        F_val.backward()

        grad_log_L = model.log_L.grad.clone()
        current_F = F_val.item()
        model.log_L.grad = None

        grad_norm = grad_log_L.norm().item()

        F_trajectory.append(current_F)
        box_lengths_trajectory.append(model.L.detach().cpu().tolist())

        if step % log_every == 0:
            L_str = ", ".join(f"L{d}={v:.6f}" for d, v in enumerate(model.L.tolist()))
            print(
                f"Step {step:4d} | F = {current_F:.6e} | "
                f"|grad_L| = {grad_norm:.4e} | {L_str}"
            )

        if grad_norm < tol_grad:
            converged = True
            break

        _, current_F = backtracking_line_search_box(
            model,
            delta_phi,
            current_F,
            grad_log_L,
            grad_scale=box_grad_scale,
            alpha_init=lr_box,
        )

    if converged:
        print(f"Converged at step {step}")
    else:
        print(f"Did not converge after {n_steps} steps")

    result = SimulationData.from_model(model)
    n = len(F_trajectory)
    result.F = np.array(F_trajectory)
    result.box_lengths = np.array(box_lengths_trajectory)
    phi_np = delta_phi.cpu().numpy()
    result.phi = np.broadcast_to(phi_np[np.newaxis], (n, *phi_np.shape)).copy()
    result.converged = converged
    return result


def optimize_joint(
    model: nn.Module,
    n_outer: int = 500,
    n_inner_phi: int = 50,
    n_inner_box: int = 10,
    lr_phi: float = 0.5,
    lr_box: float = 0.1,
    box_grad_scale: float = 10,
    tol_grad_phi: float = 1e-6,
    tol_grad_box: float = 1e-7,
    use_line_search: bool = True,
    log_every: int = 10,
    callback: Optional[Callable] = None,
) -> SimulationData:
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

    # Initialize containers for the F, phi, and box_length trajectories to pass into SimulationData object later
    F_trajectory = []
    phi_trajectory = []
    box_lengths_trajectory = []
    converged = False

    for outer in tqdm(range(n_outer)):
        # =================================================================
        # Phase 1: Optimize density field with fixed box (PGD)
        # =================================================================
        delta_phi = model.delta_phi.data.clone()

        with torch.no_grad():
            K2 = model._compute_K2(model.L)
            Gamma_ij_cached = model._compute_gamma_ij(K2)

        for inner in range(n_inner_phi):
            # Enable grad in-place — no clone needed since delta_phi is a detached leaf
            delta_phi.requires_grad_(True)
            F_val = model(delta_phi, Gamma_ij=Gamma_ij_cached, project=False)
            (raw_grad,) = torch.autograd.grad(F_val, delta_phi)
            current_F = F_val.item()
            # Detach before arithmetic so line-search tensors stay out of the graph
            delta_phi = delta_phi.detach()

            proj_grad = model._project_order_parameter(raw_grad)
            grad_phi_norm = proj_grad.norm().item()

            if grad_phi_norm < tol_grad_phi:
                break

            if use_line_search:
                _, delta_phi, current_F = backtracking_line_search_phi(
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
                _, current_F = backtracking_line_search_box(
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
        F_trajectory.append(current_F)
        phi_trajectory.append(delta_phi.detach().cpu().numpy())
        box_lengths_trajectory.append(model.L.tolist())

        if outer % log_every == 0:
            L_str = ", ".join(f"L{d}={v:.6f}" for d, v in enumerate(model.L.tolist()))
            print(
                f"Outer {outer:4d} | F = {current_F:.6e} | "
                f"|grad_phi| = {grad_phi_norm:.4e} | |grad_L| = {grad_box_norm:.4e} | "
                f"{L_str}"
            )

        if callback is not None:
            callback(outer, model, current_F, grad_phi_norm, grad_box_norm)

        if grad_phi_norm < tol_grad_phi and grad_box_norm < tol_grad_box:
            converged = True
            break

    if converged:
        print("Converged at outer iteration", outer)
        converged = True
    else:
        print("Did not converge after", n_outer, "outer iterations")
    result = SimulationData.from_model(model)
    result.F = np.array(F_trajectory)
    result.phi = np.array(phi_trajectory)
    result.box_lengths = np.array(box_lengths_trajectory)
    result.converged = converged
    return result


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
