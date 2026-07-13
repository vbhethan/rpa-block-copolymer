import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..simulation_io import SimulationData


def picard_optimize_phi_only(
    model: nn.Module,
    n_steps: int = 5000,
    alpha: float = 0.05,
    tol: float = 1e-6,
    patience: int = 50,
    pin_mean: bool = True,
    anderson_m: int = 0,
    anderson_reg: float = 1e-10,
    safeguard_factor: float = 2.0,
    precondition: bool = True,
    precond_strength: float = 1.0,
    record_every: int = 1,
    silent: bool = False,
) -> SimulationData:
    """
    Grand-canonical fixed-point (Picard) iteration on the density field.

    Solves the Euler-Lagrange stationarity condition of the RPA free energy by
    repeated substitution rather than gradient descent. Splitting the free
    energy as ``F = F_mixing + F_res`` (ideal Flory-Huggins reference plus the
    residual ``F_res = Delta_F_int - F_mixing_2``), stationarity of the grand
    potential gives the per-component map, in the code's discretization:

        w_i(r)   = - f_i * n_grid * dF_res/d(delta_phi_i)(r)     ("field")
        g_i(r)   = f_i * phi_bar * exp(w_i(r))                   (rho_i,bulk = f_i*phi_bar)

    where ``g(rho)`` is the fixed-point map (optionally with the per-component
    spatial mean pinned to ``f_i * phi_bar``). Each iteration evaluates
    ``g(rho)`` and updates ``rho`` toward the fixed point ``rho = g(rho)``.

    The residual ``f = g(rho) - rho`` is optionally **preconditioned** in Fourier
    space and then optionally **Anderson-accelerated**:

    - **Preconditioning** (``precondition = True``, default): the ideal-chain
      vertex ``Gamma_ideal(k)`` grows like ``k^2``, so high-k modes have huge map
      gain and both simple mixing and Anderson are stiff/unstable on fine grids.
      The residual is filtered by ``P(k) = 1 / (1 + s * f_max^2 * gamma(k))``,
      where ``gamma(k)`` is the largest eigenvalue of ``Re Gamma(k)``. This damps
      exactly the stiff high-k modes while leaving the ordering (small/negative
      ``gamma``) modes at full strength, so the effective map gain is ``~(1-alpha)``
      uniformly in ``k``. The filter vanishes at the fixed point (``f = 0``), so
      the solution is unchanged. This is what makes fine 3D grids tractable.
    - **Anderson acceleration** (``anderson_m > 0``, off by default): combine the
      last ``anderson_m`` (preconditioned) residuals by a small least-squares
      solve to extrapolate toward the fixed point. In practice, on this k^2-stiff
      functional Anderson has been unreliable (it tends to amplify the stiff
      modes even after preconditioning) and *preconditioned simple mixing is the
      recommended, working path*; Anderson is kept available for experimentation
      and guarded by ``safeguard_factor``.

    Notes
    -----
    - Without preconditioning the stable ``alpha`` shrinks like ``1 / k_max^2``
      (impractical in 3D). Keep ``precondition = True`` for real systems; raise
      ``precond_strength`` if the free energy grows without bound, or lower
      ``alpha`` (strongly ordered cases like ``alpha ~ 0.1`` find deeper minima,
      near-disordered cases may need ``alpha ~ 0.03`` to stay stable).
    - **Positivity is why this beats gradient descent here.** The ``exp`` map keeps
      every density positive, so it reaches strongly-ordered states where PGD's
      ``log rho`` term diverges (PGD from a large initial amplitude goes to
      ``nan``; from a small one it stalls at the homogeneous saddle).
    - **Metastable fixed points are intended.** The iteration relaxes to whichever
      fixed point the initial density sits in the basin of, not necessarily the
      global minimum -- seed a candidate morphology (see
      ``rpa.initial_conditions``) to target a particular metastable phase, as in
      SCFT. A converged result above the homogeneous free energy is a valid
      metastable state, not a failure.
    - The box length is held fixed (phi-only). ``optimize_box`` may be True or
      False; either way ``Gamma_ij`` is evaluated once at the current box.
    - Local incompressibility ``sum_i rho_i(r) = phi_bar`` is NOT enforced (pure
      grand-canonical treatment); positivity is preserved (the map output is
      ``exp``-positive and Anderson steps that would drive a density non-positive
      fall back to simple mixing). The per-frame max drift
      ``max|sum_i rho_i - phi_bar|`` is exposed as
      ``result.incompressibility_drift`` for diagnostics.
    - The residual gradient is taken unprojected: the constraints are handled
      explicitly here, so projection would remove the very directions we act on.

    Parameters
    ----------
    model : BlockCopolymerFreeEnergy
        Box lengths are not modified.
    n_steps : int
        Maximum number of Picard iterations.
    alpha : float
        Mixing / Anderson damping parameter in (0, 1]. With Anderson enabled a
        moderate value (default 0.05) is usually fine; for pure simple mixing
        (``anderson_m = 0``) it must be much smaller on fine grids.
    tol : float
        Convergence threshold on the change in total free energy ``|F - F_ref|``
        (the pointwise residual can floor in a small limit cycle after F has
        settled, so convergence is declared on the free-energy stall instead, as
        in ``optimize_phi_only``). The raw residual ``max|g(rho) - rho|`` is still
        shown in the progress bar and drives the divergence safeguard.
    patience : int
        Number of consecutive steps within ``tol`` (of F) required to declare
        convergence.
    pin_mean : bool
        If True, rescale each component so ``<rho_i> = f_i * phi_bar`` every step
        (canonical mean composition). If False, the mean floats (chemical
        potential fixed by the homogeneous reference).
    anderson_m : int
        Anderson history depth (number of past residuals mixed in). Default 0
        (off): preconditioned simple mixing is the recommended path here. Set
        > 0 to experiment with Anderson acceleration.
    anderson_reg : float
        Tikhonov regularization added to the Anderson normal equations for
        numerical stability (relative to the mean diagonal).
    safeguard_factor : float
        If the residual grows beyond ``safeguard_factor`` times the best residual
        seen, the step is rejected: the iterate reverts to the best one and the
        Anderson history is cleared (a stable simple-mixing restart). This keeps
        an over-eager Anderson extrapolation from diverging.
    precondition : bool
        If True (default), filter the residual in Fourier space to damp the stiff
        high-k modes (see above). Disable only for small/mild problems.
    precond_strength : float
        Scale ``s`` in the filter ``P(k) = 1 / (1 + s * f_max^2 * gamma(k))``.
        Larger values damp high-k more aggressively (more stable, slightly
        slower); default 1.0.
    record_every : int
        Store a trajectory frame every ``record_every`` steps (the final step is
        always recorded). Because the iteration can take many steps, subsampling
        keeps the stored trajectory small. Default 1 (every step).
    silent : bool
        Suppress the progress bar and summary print.

    Returns
    -------
    SimulationData
        Trajectory of F (total free energy) and phi (= delta_phi) at each step;
        box_lengths is fixed and broadcast across all frames. The extra
        attribute ``incompressibility_drift`` (1D array, one value per frame) is
        attached for diagnostics.
    """
    with torch.no_grad():
        if model.optimize_box:
            K2 = model._compute_K2(model.L)
            Gamma_ij = model._compute_gamma_ij(K2)
        else:
            Gamma_ij = model._Gamma_ij_cached

    f_phi = model._f_expanded() * model.phi_bar  # f_i * phi_bar, shape (n, 1, ..., 1)
    f_col = model._f_expanded()  # f_i for the exponent prefactor
    n_grid = model.n_grid_points
    spatial_dims = model.spatial_dims

    # Fourier-space preconditioner P(k) that damps the stiff high-k modes.
    # gamma(k) = largest eigenvalue of Re Gamma(k); the map gain scales like
    # f^2 * gamma, so P(k) = 1 / (1 + s * f_max^2 * gamma(k)) flattens it.
    precond_filter = None
    if precondition:
        with torch.no_grad():
            gamma_k = torch.linalg.eigvalsh(Gamma_ij.real).amax(dim=-1)  # (*grid,)
            gamma_k = gamma_k.clamp_min(0.0)
            f_max2 = float((f_col**2).max())
            precond_filter = 1.0 / (1.0 + precond_strength * f_max2 * gamma_k)

    def apply_precond(residual: torch.Tensor) -> torch.Tensor:
        """Filter a real-space residual field (n, *grid) by P(k) in Fourier space."""
        if precond_filter is None:
            return residual
        r_hat = torch.fft.fftn(residual, dim=spatial_dims)
        r_hat = r_hat * precond_filter  # broadcasts over the component axis
        return torch.fft.ifftn(r_hat, dim=spatial_dims).real

    def picard_map(rho: torch.Tensor) -> torch.Tensor:
        """Fixed-point map g(rho): the exponential density update."""
        delta_phi = (rho - f_phi).detach().requires_grad_(True)
        F_res = model.residual_free_energy(delta_phi, Gamma_ij=Gamma_ij, project=False)
        (grad,) = torch.autograd.grad(F_res, delta_phi)
        # w_i = - f_i * n_grid * dF_res/d(delta_phi_i)
        w = -f_col * n_grid * grad
        if pin_mean:
            # Pin <rho_i> = f_i * phi_bar. Subtracting the per-component spatial
            # max before exponentiating cancels in the ratio (it is exact) but
            # keeps exp() from overflowing when the field is large.
            w = w - w.amax(dim=spatial_dims, keepdim=True)
            exp_w = torch.exp(w)
            return f_phi * exp_w / exp_w.mean(dim=spatial_dims, keepdim=True)
        return f_phi * torch.exp(w)

    rho = model.delta_phi.data.detach().clone() + f_phi

    # Anderson history: flattened iterates x, maps g, and residuals f = g - x.
    x_hist: list[torch.Tensor] = []
    g_hist: list[torch.Tensor] = []
    f_hist: list[torch.Tensor] = []

    # Safeguarding: keep the best (lowest-residual) iterate so a bad Anderson
    # extrapolation can be rejected and the history restarted from there.
    rho_best = rho.clone()
    best_res = float("inf")

    F_trajectory = []
    phi_trajectory = []
    drift_trajectory = []
    converged = False
    residual = float("inf")
    stall_count = 0
    F_ref = float("inf")

    pbar = tqdm(range(n_steps), disable=silent)

    for step in pbar:
        g = picard_map(rho)
        raw_res = g - rho
        residual = raw_res.abs().max().item()  # physical convergence metric

        # Safeguard: if the residual has blown up relative to the best seen,
        # reject the recent (Anderson) progress, revert to the best iterate, and
        # restart the history so the next step is a stable simple-mixing step.
        if residual > safeguard_factor * best_res:
            rho = rho_best.clone()
            x_hist.clear()
            g_hist.clear()
            f_hist.clear()
            g = picard_map(rho)
            raw_res = g - rho
            residual = raw_res.abs().max().item()
        if residual < best_res:
            best_res = residual
            rho_best = rho.clone()

        # Preconditioned residual and preconditioned map value g_pre = rho + f.
        f = apply_precond(raw_res)
        g_pre = rho + f

        x_flat = rho.reshape(-1)
        g_flat = g_pre.reshape(-1)
        f_flat = f.reshape(-1)

        used_anderson = False
        if anderson_m > 0 and len(x_hist) >= 1:
            # Build difference matrices over the stored history plus current point.
            m_k = min(anderson_m, len(x_hist))
            xs = x_hist[-m_k:] + [x_flat]
            gs = g_hist[-m_k:] + [g_flat]
            fs = f_hist[-m_k:] + [f_flat]
            dF = torch.stack([fs[i + 1] - fs[i] for i in range(m_k)], dim=1)  # (ndof, m_k)
            dX = torch.stack([xs[i + 1] - xs[i] for i in range(m_k)], dim=1)
            dG = torch.stack([gs[i + 1] - gs[i] for i in range(m_k)], dim=1)

            # Regularized normal equations: min_gamma ||f - dF gamma||.
            AtA = dF.T @ dF
            reg = anderson_reg * AtA.diagonal().mean().clamp_min(1e-30)
            AtA = AtA + reg * torch.eye(m_k, dtype=AtA.dtype, device=AtA.device)
            gamma = torch.linalg.solve(AtA, dF.T @ f_flat)

            x_bar = x_flat - dX @ gamma
            g_bar = g_flat - dG @ gamma
            rho_next = ((1 - alpha) * x_bar + alpha * g_bar).reshape(rho.shape)

            # Positivity safeguard: extrapolation can overshoot to rho <= 0.
            if torch.isfinite(rho_next).all() and rho_next.min() > 0:
                used_anderson = True

        if not used_anderson:
            rho_next = (1 - alpha) * rho + alpha * g_pre

        # Push current point into history (capped) before advancing.
        x_hist.append(x_flat)
        g_hist.append(g_flat)
        f_hist.append(f_flat)
        if len(x_hist) > anderson_m + 1:
            x_hist.pop(0)
            g_hist.pop(0)
            f_hist.pop(0)

        rho = rho_next
        delta_phi_mixed = rho - f_phi

        with torch.no_grad():
            F_total = model(delta_phi_mixed, Gamma_ij=Gamma_ij, project=False).item()
            drift = (rho.sum(dim=0) - model.phi_bar).abs().max().item()

        is_last = step == n_steps - 1
        if step % record_every == 0 or is_last:
            F_trajectory.append(F_total)
            phi_trajectory.append(delta_phi_mixed.cpu().numpy())
            drift_trajectory.append(drift)

        if not silent:
            pbar.set_description(
                f"Step {step:4d} | F = {F_total:.6e} | "
                f"|drho| = {residual:.4e} | drift = {drift:.2e} | "
                f"stall = {stall_count:02d}/{patience}"
            )

        # Divergence guard: too-large alpha (or a bad extrapolation) blows up.
        if not np.isfinite(F_total):
            if not silent:
                print(f"Diverged at step {step} (non-finite F); reduce alpha")
            break

        # Convergence on free-energy stall: the pointwise residual can floor
        # above tol in a small limit cycle even after F has settled, so (as in
        # optimize_phi_only) we declare convergence when F stops changing.
        if abs(F_total - F_ref) < tol:
            stall_count += 1
        else:
            stall_count = 0
            F_ref = F_total

        if stall_count >= patience:
            converged = True
            break

    if not silent:
        if converged:
            print(f"Converged at step {step} (F stable for {patience} steps)")
        else:
            print(f"Did not converge after {n_steps} steps")

    model.delta_phi.data.copy_(rho - f_phi)

    result = SimulationData.from_model(model)
    n = len(F_trajectory)
    result.F = np.array(F_trajectory)
    result.phi = np.array(phi_trajectory)
    box_lengths = model.L.detach().cpu().tolist()
    result.box_lengths = np.broadcast_to(
        np.array(box_lengths)[np.newaxis], (n, len(box_lengths))
    ).copy()
    result.converged = converged
    result.incompressibility_drift = np.array(drift_trajectory)
    return result
