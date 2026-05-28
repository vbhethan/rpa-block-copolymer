from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from scipy.optimize import minimize as scipy_minimize
except ImportError as e:
    raise ImportError(
        "scipy is required for scipy_optim; install it with: pip install scipy"
    ) from e

from ..simulation_io import SimulationData


def _phi_fun_and_grad(
    x: np.ndarray,
    model: nn.Module,
    phi_shape: tuple,
    dtype: torch.dtype,
    device: torch.device,
    Gamma_ij: torch.Tensor | None = None,
) -> tuple[float, np.ndarray]:
    """
    Evaluate F(project(x)) and its gradient w.r.t. x.

    The projection is self-adjoint, so autograd through it yields the
    constraint-projected gradient automatically — no manual projection needed.
    """
    delta_phi = torch.from_numpy(x.reshape(phi_shape)).to(dtype=dtype, device=device)
    delta_phi.requires_grad_(True)
    delta_phi_proj = model._project_order_parameter(delta_phi)
    F_val = model(delta_phi_proj, Gamma_ij=Gamma_ij, project=False)
    F_val.backward()
    grad = delta_phi.grad.detach().cpu().numpy().flatten().astype(np.float64)
    return float(F_val.item()), grad


def _box_fun_and_grad(
    log_L_np: np.ndarray,
    model: nn.Module,
    delta_phi_tensor: torch.Tensor,
) -> tuple[float, np.ndarray]:
    """
    Evaluate F w.r.t. log_L and return its gradient.

    Mutates model.log_L.data in place (same pattern as pgd.backtracking_line_search_box).
    The .copy() on the returned gradient is mandatory: without it the array
    aliases the grad buffer that is zeroed on the next call.
    """
    model.log_L.data.copy_(
        torch.from_numpy(log_L_np).to(
            dtype=model.log_L.dtype, device=model.log_L.device
        )
    )
    model.log_L.grad = None
    F_val = model(delta_phi_tensor, project=False)
    F_val.backward()
    grad = model.log_L.grad.detach().cpu().numpy().astype(np.float64).copy()
    return float(F_val.item()), grad


def scipy_optimize_phi_only(
    model: nn.Module,
    maxiter: int = 2000,
    method: str = "CG",
    tol: float = 1e-6,
    options: Optional[dict] = None,
    silent: bool = False,
) -> SimulationData:
    """
    Scipy-based optimization of delta_phi with box lengths held fixed.

    Uses scipy.optimize.minimize (default: CG) with gradients from
    autograd. Constraints are handled via project-and-compose: the objective
    is F(project(x)), and autograd through the self-adjoint projection yields
    the correctly projected gradient automatically.

    Parameters
    ----------
    model : BlockCopolymerFreeEnergy
        Box lengths are not modified; optimize_box may be True or False.
    maxiter : int
        Maximum number of optimizer iterations (maps to options["maxiter"]).
    method : str
        scipy.optimize.minimize method. "CG" default, seemed to work best empirically...
    tol : float
        Gradient convergence tolerance (maps to options["gtol"] for CG).
    options : dict, optional
        Extra options forwarded to scipy.optimize.minimize. Keys here override
        the defaults set from maxiter and tol.
    silent : bool
        If True, suppress scipy's own output (sets options["disp"] = False).

    Returns
    -------
    SimulationData
        Trajectory recorded once per accepted optimizer iteration (via callback).
        box_lengths is fixed and broadcast across all frames.
    """
    phi_shape = model.delta_phi.shape
    dtype = model.delta_phi.dtype
    device = model.delta_phi.device

    with torch.no_grad():
        if model.optimize_box:
            K2 = model._compute_K2(model.L)
            Gamma_ij = model._compute_gamma_ij(K2)
        else:
            Gamma_ij = model._Gamma_ij_cached

    x0 = model.delta_phi.data.cpu().numpy().flatten().astype(np.float64)

    F_trajectory: list[float] = []
    phi_trajectory: list[np.ndarray] = []

    def _callback(xk: np.ndarray) -> None:
        dp = torch.from_numpy(xk.reshape(phi_shape)).to(dtype=dtype, device=device)
        with torch.no_grad():
            dp_proj = model._project_order_parameter(dp)
            F = model(dp_proj, Gamma_ij=Gamma_ij, project=False).item()
        F_trajectory.append(F)
        phi_trajectory.append(dp_proj.cpu().numpy())

    _options: dict = {"maxiter": maxiter, "gtol": tol, "disp": not silent}
    if options is not None:
        _options.update(options)

    res = scipy_minimize(
        fun=lambda x: _phi_fun_and_grad(x, model, phi_shape, dtype, device, Gamma_ij),
        x0=x0,
        jac=True,
        method=method,
        callback=_callback,
        options=_options,
    )

    x_final = torch.from_numpy(res.x.reshape(phi_shape)).to(dtype=dtype, device=device)
    model.delta_phi.data.copy_(model._project_order_parameter(x_final))

    result = SimulationData.from_model(model)
    n = len(F_trajectory)
    if n == 0:
        # optimizer terminated without a single accepted iterate (already converged)
        with torch.no_grad():
            F_now = model(model.delta_phi.data, Gamma_ij=Gamma_ij, project=False).item()
        F_trajectory.append(F_now)
        phi_trajectory.append(model.delta_phi.data.cpu().numpy())
        n = 1

    result.F = np.array(F_trajectory)
    result.phi = np.array(phi_trajectory)
    box_lengths = model.L.detach().cpu().tolist()
    result.box_lengths = np.broadcast_to(
        np.array(box_lengths)[np.newaxis], (n, len(box_lengths))
    ).copy()
    result.converged = bool(res.success)
    return result


def scipy_optimize_box_only(
    model: nn.Module,
    maxiter: int = 500,
    method: str = "CG",
    tol: float = 1e-7,
    options: Optional[dict] = None,
    silent: bool = False,
) -> SimulationData:
    """
    Scipy-based optimization of box dimensions with delta_phi held fixed.

    Parameters
    ----------
    model : BlockCopolymerFreeEnergy
        Must have optimize_box=True. delta_phi is treated as a constant.
    maxiter : int
        Maximum number of optimizer iterations.
    method : str
        scipy.optimize.minimize method. "CG" default, seemed to work best empirically...
        the number of variables equals the spatial dimension (2 or 3).
    tol : float
        Gradient convergence tolerance.
    options : dict, optional
        Extra options forwarded to scipy.optimize.minimize.
    silent : bool
        If True, suppress scipy's own output.

    Returns
    -------
    SimulationData
        Trajectory of F and box_lengths; phi is the fixed field broadcast
        across all frames.
    """
    if not model.optimize_box:
        raise ValueError("model.optimize_box must be True")

    delta_phi_tensor = model.delta_phi.data.detach().clone()
    x0 = model.log_L.data.cpu().numpy().astype(np.float64)

    F_trajectory: list[float] = []
    box_lengths_trajectory: list[list[float]] = []

    def _callback(xk: np.ndarray) -> None:
        model.log_L.data.copy_(
            torch.from_numpy(xk).to(dtype=model.log_L.dtype, device=model.log_L.device)
        )
        with torch.no_grad():
            F = model(delta_phi_tensor, project=False).item()
        F_trajectory.append(F)
        box_lengths_trajectory.append(model.L.detach().cpu().tolist())

    _options: dict = {"maxiter": maxiter, "gtol": tol, "disp": not silent}
    if options is not None:
        _options.update(options)

    res = scipy_minimize(
        fun=lambda x: _box_fun_and_grad(x, model, delta_phi_tensor),
        x0=x0,
        jac=True,
        method=method,
        callback=_callback,
        options=_options,
    )

    model.log_L.data.copy_(
        torch.from_numpy(res.x).to(dtype=model.log_L.dtype, device=model.log_L.device)
    )

    result = SimulationData.from_model(model)
    n = len(F_trajectory)
    if n == 0:
        with torch.no_grad():
            F_now = model(delta_phi_tensor, project=False).item()
        F_trajectory.append(F_now)
        box_lengths_trajectory.append(model.L.detach().cpu().tolist())
        n = 1

    result.F = np.array(F_trajectory)
    result.box_lengths = np.array(box_lengths_trajectory)
    phi_np = model.delta_phi.data.cpu().numpy()
    result.phi = np.broadcast_to(phi_np[np.newaxis], (n, *phi_np.shape)).copy()
    result.converged = bool(res.success)
    return result


def scipy_optimize_joint(
    model: nn.Module,
    n_outer: int = 500,
    n_inner_phi: int = 50,
    n_inner_box: int = 10,
    method_phi: str = "CG",
    method_box: str = "CG",
    tol_grad_phi: float = 1e-6,
    tol_grad_box: float = 1e-7,
    tol_F_outer: float = 1e-6,
    patience_outer: int = 10,
    log_every: int = 10,
    callback: Optional[Callable] = None,
) -> SimulationData:
    """
    Alternating scipy optimization for density + box (drop-in replacement for
    pgd.optimize_joint).

    Each outer iteration runs scipy_optimize_phi_only (up to n_inner_phi
    iterations, box fixed) then scipy_optimize_box_only (up to n_inner_box
    iterations, phi fixed). Outer convergence mirrors pgd.optimize_joint:
    gradient norms below tolerances OR free energy stalled for patience_outer
    consecutive outer iterations.

    Parameters
    ----------
    model : BlockCopolymerFreeEnergy
        Must have optimize_box=True.
    n_outer : int
        Number of outer (alternating) iterations.
    n_inner_phi : int
        Max scipy iterations per outer iteration for density optimization.
    n_inner_box : int
        Max scipy iterations per outer iteration for box optimization.
    method_phi : str
        scipy method for the phi inner loop (default "CG").
    method_box : str
        scipy method for the box inner loop (default "CG").
    tol_grad_phi : float
        Convergence tolerance on projected phi gradient norm (outer loop) and
        inner phi scipy gtol.
    tol_grad_box : float
        Convergence tolerance on box gradient norm (outer loop) and inner box
        scipy gtol.
    tol_F_outer : float
        Free-energy stall threshold for the outer loop.
    patience_outer : int
        Consecutive outer iterations within tol_F_outer required to stop.
    log_every : int
        Print status every this many outer iterations.
    callback : callable, optional
        Called at each outer iteration with signature
        (outer_iter, model, F, grad_phi_norm, grad_box_norm).

    Returns
    -------
    SimulationData
        One trajectory frame per outer iteration.
    """
    if not model.optimize_box:
        raise ValueError("model.optimize_box must be True for joint optimization")

    F_trajectory: list[float] = []
    phi_trajectory: list[np.ndarray] = []
    box_lengths_trajectory: list[list[float]] = []
    converged = False
    stall_count = 0
    F_ref = float("inf")

    for outer in tqdm(range(n_outer)):
        scipy_optimize_phi_only(
            model,
            maxiter=n_inner_phi,
            method=method_phi,
            tol=tol_grad_phi,
            silent=True,
        )

        scipy_optimize_box_only(
            model,
            maxiter=n_inner_box,
            method=method_box,
            tol=tol_grad_box,
            silent=True,
        )

        # Compute gradient norms for outer convergence (identical to pgd.optimize_joint)
        delta_phi = model.delta_phi.data.detach()
        with torch.no_grad():
            K2 = model._compute_K2(model.L)
            Gamma_ij = model._compute_gamma_ij(K2)
        delta_phi_var = delta_phi.clone().requires_grad_(True)
        F_val = model(delta_phi_var, Gamma_ij=Gamma_ij, project=False)
        (raw_grad,) = torch.autograd.grad(F_val, delta_phi_var)
        current_F = F_val.item()
        grad_phi_norm = model._project_order_parameter(raw_grad).norm().item()

        model.log_L.grad = None
        model(delta_phi).backward()
        grad_box_norm = model.log_L.grad.norm().item()
        model.log_L.grad = None

        F_trajectory.append(current_F)
        phi_trajectory.append(delta_phi.cpu().numpy())
        box_lengths_trajectory.append(model.L.detach().cpu().tolist())

        if outer % log_every == 0:
            L_str = ", ".join(f"L{d}={v:.6f}" for d, v in enumerate(model.L.tolist()))
            print(
                f"Outer {outer:4d} | F = {current_F:.6e} | "
                f"|grad_phi| = {grad_phi_norm:.4e} | |grad_L| = {grad_box_norm:.4e} | "
                f"{L_str}"
            )

        if callback is not None:
            callback(outer, model, current_F, grad_phi_norm, grad_box_norm)

        if abs(current_F - F_ref) < tol_F_outer:
            stall_count += 1
        else:
            stall_count = 0
            F_ref = current_F

        if grad_phi_norm < tol_grad_phi and grad_box_norm < tol_grad_box:
            converged = True
            break

        if stall_count >= patience_outer:
            converged = True
            break

    if converged:
        print("Converged at outer iteration", outer)
    else:
        print("Did not converge after", n_outer, "outer iterations")

    result = SimulationData.from_model(model)
    result.F = np.array(F_trajectory)
    result.phi = np.array(phi_trajectory)
    result.box_lengths = np.array(box_lengths_trajectory)
    result.converged = converged
    return result
