"""
Cahn-Hilliard (Model B) Dynamics Solver for Block Copolymer Systems
====================================================================

Time-evolution solver using the conserved-order-parameter equation of motion:

    ∂δφ_i/∂t = -M ∇² μ̃_i

where μ̃_i is the exchange chemical potential (incompressibility-enforcing)
and M is the mobility coefficient.

The free energy monotonically decreases along trajectories (Lyapunov property).
Conservation laws (zero-mean, incompressibility) are automatically preserved
by the equation structure.

Provides two time-stepping schemes:
  - Semi-implicit (linear part implicit): allows large dt
  - Forward Euler (fully explicit): for validation at small dt
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from rpa import BlockCopolymerFreeEnergy
from simulation_io import SimulationData


def compute_nonlinear_chemical_potential(
    delta_phi: torch.Tensor,
    f_expanded: torch.Tensor,
    phi_bar: float,
) -> torch.Tensor:
    """
    Compute the nonlinear part of the chemical potential in real space.

    μ_i^NL = μ_i^ref - μ_i^ref2

    where:
        μ_i^ref  = (1/(N*f_i)) * [ln(φ_i(r)) + 1]   (full reference entropy derivative)
        μ_i^ref2 = (1/N) * δφ_i / (f_i² * φ̄)        (quadratic expansion, already in Γ)

    The N prefactor cancels because Gamma_ij in rpa.py already absorbs 1/N scaling
    through the structure factor. We keep the expressions consistent with the
    free energy functional in rpa.py (which divides by vol but not by N in the
    real-space terms).

    Parameters
    ----------
    delta_phi : torch.Tensor
        Order parameter, shape (n_components, *grid_shape)
    f_expanded : torch.Tensor
        Block fractions broadcast-ready, shape (n_components, 1, ..., 1)
    phi_bar : float
        Mean density

    Returns
    -------
    mu_NL : torch.Tensor
        Nonlinear chemical potential, same shape as delta_phi
    """
    rho = delta_phi + f_expanded * phi_bar  # full density φ_i(r)

    # Reference entropy derivative: (1/f_i) * [ln(φ_i) + 1]
    mu_ref = (1.0 / f_expanded) * (torch.log(rho) + 1.0)

    # Quadratic expansion derivative: δφ_i / (f_i² * φ̄)
    mu_ref2 = delta_phi / (f_expanded**2 * phi_bar)

    return mu_ref - mu_ref2


def compute_exchange_chemical_potential(mu: torch.Tensor) -> torch.Tensor:
    """
    Compute exchange chemical potential to enforce incompressibility.

    μ̃_i = μ_i - (1/n) Σ_j μ_j

    Since Σ_i μ̃_i = 0 by construction, if Σ_i δφ_i = 0 at t=0,
    it remains zero for all time.

    Parameters
    ----------
    mu : torch.Tensor
        Chemical potential, shape (n_components, *grid_shape)

    Returns
    -------
    mu_exchange : torch.Tensor
        Exchange chemical potential, same shape
    """
    return mu - mu.mean(dim=0, keepdim=True)


def semi_implicit_step(
    delta_phi: torch.Tensor,
    model: BlockCopolymerFreeEnergy,
    dt: float,
    M: float,
    Gamma_ij: torch.Tensor,
    K2: torch.Tensor,
) -> torch.Tensor:
    """
    Semi-implicit time step for Cahn-Hilliard dynamics.

    Treats the linear part (RPA quadratic / Γ_ij term) implicitly and the
    nonlinear part (reference entropy minus its quadratic expansion) explicitly.

    Per-wavevector update:
        (I + Δt·M·q²·Γ(q)) · δφ̂^{n+1} = δφ̂^n - Δt·M·q²·μ̂̃^{NL}

    Parameters
    ----------
    delta_phi : torch.Tensor
        Current order parameter, shape (n_components, *grid_shape)
    model : BlockCopolymerFreeEnergy
        The RPA model (provides physical parameters)
    dt : float
        Time step
    M : float
        Mobility coefficient
    Gamma_ij : torch.Tensor
        Vertex function, shape (*grid_shape, n_components, n_components), complex
    K2 : torch.Tensor
        Squared wavevector grid, shape (*grid_shape)

    Returns
    -------
    delta_phi_new : torch.Tensor
        Updated order parameter (real-valued)
    """
    n = model.n_components
    ndim = model.ndim
    spatial_dims = model.spatial_dims
    f_expanded = model._f_expanded()

    # 1. Compute nonlinear chemical potential in real space
    mu_NL = compute_nonlinear_chemical_potential(delta_phi, f_expanded, model.phi_bar)

    # 2. Exchange chemical potential (enforce incompressibility)
    mu_NL_exchange = compute_exchange_chemical_potential(mu_NL)

    # 3. FFT the nonlinear part: shape (n_components, *grid_shape)
    mu_NL_hat = torch.fft.fftn(mu_NL_exchange, dim=spatial_dims)

    # 4. FFT the current field
    phi_hat = torch.fft.fftn(delta_phi, dim=spatial_dims)

    # Rearrange to (*grid_shape, n_components) for the linear solve
    # phi_hat: (n, *grid) -> (*grid, n)
    phi_hat = phi_hat.movedim(0, -1)
    mu_NL_hat = mu_NL_hat.movedim(0, -1)

    # Build RHS: δφ̂^n - Δt·M·q²·μ̂̃^{NL}
    # K2 has shape (*grid_shape), expand for broadcasting with n_components
    K2_expanded = K2.unsqueeze(-1)  # (*grid_shape, 1)
    rhs = phi_hat - dt * M * K2_expanded * mu_NL_hat  # (*grid_shape, n)

    # Build LHS matrix: I + Δt·M·q²·Γ(q)
    # Gamma_ij: (*grid_shape, n, n), complex
    # K2_expanded for matrix: (*grid_shape, 1, 1)
    K2_mat = K2.unsqueeze(-1).unsqueeze(-1)  # (*grid_shape, 1, 1)
    eye = torch.eye(n, dtype=model.complex_dtype, device=delta_phi.device)
    eye = eye.view(*([1] * ndim), n, n)
    A = eye + dt * M * K2_mat * Gamma_ij  # (*grid_shape, n, n)

    # 5. Solve the linear system at each wavevector
    # A @ phi_new = rhs  =>  phi_new = solve(A, rhs)
    rhs_unsqueezed = rhs.unsqueeze(-1)  # (*grid_shape, n, 1)
    phi_new_hat = torch.linalg.solve(A, rhs_unsqueezed).squeeze(-1)  # (*grid_shape, n)

    # 6. Zero out k=0 mode (conservation: zero mean)
    k0_idx = (0,) * ndim
    phi_new_hat[(*k0_idx, slice(None))] = 0

    # Rearrange back to (n_components, *grid_shape)
    phi_new_hat = phi_new_hat.movedim(-1, 0)

    # 7. IFFT back to real space
    delta_phi_new = torch.fft.ifftn(phi_new_hat, dim=spatial_dims).real

    # 8. Re-project onto the constraint manifold.
    # Unlike forward Euler (which applies the exchange projection explicitly to
    # the total chemical potential), the implicit linear solve A^{-1} rhs does
    # not guarantee sum_i delta_phi_i(r) = 0 for asymmetric systems where
    # sum_i Gamma_ij is not constant across j.  Projecting here costs O(n*N_grid)
    # and keeps the incompressibility error at machine precision.
    delta_phi_new = model._project_order_parameter(delta_phi_new)

    return delta_phi_new


def forward_euler_step(
    delta_phi: torch.Tensor,
    model: BlockCopolymerFreeEnergy,
    dt: float,
    M: float,
    Gamma_ij: torch.Tensor,
    K2: torch.Tensor,
) -> torch.Tensor:
    """
    Forward Euler (fully explicit) time step for Cahn-Hilliard dynamics.

    δφ̂_i^{n+1} = δφ̂_i^n - Δt·M·q²·μ̂̃_i^n

    where μ̃ is the full exchange chemical potential (linear + nonlinear parts).

    Parameters
    ----------
    delta_phi : torch.Tensor
        Current order parameter, shape (n_components, *grid_shape)
    model : BlockCopolymerFreeEnergy
        The RPA model
    dt : float
        Time step (must be small for stability)
    M : float
        Mobility coefficient
    Gamma_ij : torch.Tensor
        Vertex function, shape (*grid_shape, n_components, n_components), complex
    K2 : torch.Tensor
        Squared wavevector grid, shape (*grid_shape)

    Returns
    -------
    delta_phi_new : torch.Tensor
        Updated order parameter (real-valued)
    """
    n = model.n_components
    ndim = model.ndim
    spatial_dims = model.spatial_dims
    f_expanded = model._f_expanded()

    # 1. Nonlinear chemical potential
    mu_NL = compute_nonlinear_chemical_potential(delta_phi, f_expanded, model.phi_bar)
    mu_NL_exchange = compute_exchange_chemical_potential(mu_NL)

    # 2. Linear chemical potential (from Gamma_ij in Fourier space)
    phi_hat = torch.fft.fftn(delta_phi, dim=spatial_dims)
    # Rearrange: (n, *grid) -> (*grid, n)
    phi_hat_moved = phi_hat.movedim(0, -1)

    # μ̂_linear = Γ(q) · δφ̂(q)  =>  (*grid, n)
    # Gamma_ij: (*grid, n, n);  phi_hat_moved: (*grid, n)
    mu_linear_hat = torch.einsum("...ij,...j->...i", Gamma_ij, phi_hat_moved)

    # Back to (n, *grid)
    mu_linear_hat = mu_linear_hat.movedim(-1, 0)

    # Exchange projection on linear part
    mu_linear_hat = mu_linear_hat - mu_linear_hat.mean(dim=0, keepdim=True)

    # 3. FFT the nonlinear exchange chemical potential
    mu_NL_hat = torch.fft.fftn(mu_NL_exchange, dim=spatial_dims)

    # 4. Total exchange chemical potential in Fourier space
    mu_total_hat = mu_linear_hat + mu_NL_hat

    # 5. Explicit update in Fourier space
    K2_expanded = K2.unsqueeze(0)  # (1, *grid_shape)
    phi_hat_new = phi_hat - dt * M * K2_expanded * mu_total_hat

    # 6. Zero out k=0 (conservation)
    k0_idx = (slice(None),) + (0,) * ndim
    phi_hat_new[k0_idx] = 0

    # 7. IFFT
    delta_phi_new = torch.fft.ifftn(phi_hat_new, dim=spatial_dims).real

    return delta_phi_new


def simulate(
    model: BlockCopolymerFreeEnergy,
    delta_phi: torch.Tensor | None = None,
    n_steps: int = 1000,
    dt: float = 0.01,
    M: float = 1.0,
    method: str = "semi-implicit",
    save_every: int = 10,
    log_every: int = 100,
) -> SimulationData:
    """
    Run Cahn-Hilliard dynamics simulation.

    Parameters
    ----------
    model : BlockCopolymerFreeEnergy
        The RPA model (should have optimize_box=False for dynamics).
    delta_phi : torch.Tensor, optional
        Initial order parameter. If None, uses model's internal field.
    n_steps : int
        Total number of time steps.
    dt : float
        Time step size.
    M : float
        Mobility coefficient.
    method : str
        "semi-implicit" or "forward-euler".
    save_every : int
        Record a full snapshot every this many steps (frame 0 = initial state).
    log_every : int
        Print progress every this many steps. 0 = silent.

    Returns
    -------
    SimulationData
        Trajectory with phi, F, and box_lengths arrays (n_frames indexed).
    """
    if delta_phi is None:
        delta_phi = model._project_order_parameter(model.delta_phi.data.clone())
    else:
        delta_phi = model._project_order_parameter(delta_phi.clone())

    # Pre-compute fixed quantities
    K2 = model._compute_K2(model.L)
    if hasattr(model, "_Gamma_ij_cached") and model._Gamma_ij_cached is not None:
        Gamma_ij = model._Gamma_ij_cached
    else:
        Gamma_ij = model._compute_gamma_ij(K2)

    if method == "semi-implicit":
        step_fn = semi_implicit_step
    elif method == "forward-euler":
        step_fn = forward_euler_step
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Use 'semi-implicit' or 'forward-euler'."
        )

    L_np = model.L.detach().cpu().numpy().copy()
    f_expanded = model._f_expanded()

    t_frames: list[float] = []
    F_frames: list[float] = []
    phi_frames: list[np.ndarray] = []
    box_lengths_frames: list[np.ndarray] = []

    def _record(step: int, phi: torch.Tensor) -> None:
        t = step * dt
        with torch.no_grad():
            F = model(phi, Gamma_ij=Gamma_ij).item()
        t_frames.append(t)
        F_frames.append(F)
        phi_frames.append(phi.detach().cpu().numpy().copy())
        box_lengths_frames.append(L_np.copy())

        if log_every > 0 and (step == 0 or step % log_every == 0):
            cons_err = phi.mean(dim=model.spatial_dims).abs().max().item()
            incomp_err = phi.sum(dim=0).abs().max().item()
            print(
                f"  step {step:7d} | t = {t:.4e} | F = {F:.8e} | "
                f"cons = {cons_err:.2e} | incomp = {incomp_err:.2e}"
            )

    _record(0, delta_phi)

    n_steps_completed = 0
    for step in range(1, n_steps + 1):
        delta_phi = step_fn(delta_phi, model, dt, M, Gamma_ij, K2)
        n_steps_completed = step

        if step % save_every == 0 or step == n_steps:
            _record(step, delta_phi)
        elif log_every > 0 and step % log_every == 0:
            t = step * dt
            print(
                f"  step {step:7d} | t = {t:.4e} | F = {F_frames[-1]:.8e} (last saved frame)"
            )

        rho_min = (delta_phi + f_expanded * model.phi_bar).min().item()
        if rho_min < 0:
            print(
                f"\nWARNING: negative density at step {step} "
                f"(min rho = {rho_min:.4e}). Aborting — reduce dt."
            )
            break

    result = SimulationData.from_model(model)
    result.phi = np.stack(phi_frames, axis=0)
    result.F = np.array(F_frames, dtype=np.float64)
    result.box_lengths = np.stack(box_lengths_frames, axis=0)

    if log_every > 0:
        print(f"\nSimulation complete: {n_steps_completed} steps")
        print(f"  F_initial = {result.F[0]:.8e}")
        print(f"  F_final   = {result.F[-1]:.8e}")
        print(f"  ΔF        = {result.F[-1] - result.F[0]:.8e}")

    return result

