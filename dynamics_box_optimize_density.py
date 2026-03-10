"""
Alternating Dynamics + Box-Length Optimization
===============================================

Jointly finds the equilibrium density profile and optimal box dimensions by
alternating between two phases:

  Phase 1 — Dynamics (density relaxation):
      Run Cahn-Hilliard (Model B) time-stepping for n_dynamics_steps with the
      box held fixed. The free energy decreases monotonically (Lyapunov property),
      driving the density toward a local minimum for the current box geometry.

  Phase 2 — Box optimization:
      With the density profile fixed, minimize F over the box lengths via
      gradient descent on log_L with Armijo backtracking (imported from
      optimize_box.py). The wavevectors q = 2π n/L shift, changing the
      interaction energy.

These two phases alternate for n_outer outer iterations.  Because each phase
is a descent step on the same free energy F, the outer loop is also a descent
scheme: F can only decrease (or stay constant) across iterations.

Why dynamics instead of PGD for the density step?
- Dynamics automatically respects conservation laws and incompressibility.
- Semi-implicit time stepping is numerically robust without a line search.
- The trajectory is physically interpretable, not just a gradient path.

Usage
-----
    python joint_phi_box_optimize.py
    python joint_phi_box_optimize.py --n-outer 20 --n-dynamics 2000 --dt 1e-5 --grid 32
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import math

import h5py
import numpy as np
import torch

from optimizers.dynamics.simulate import simulate
from optimizers.box_optimize import optimize_box_lengths
from simulation_io import SimulationData
from rpa import BlockCopolymerFreeEnergy


# ---------------------------------------------------------------------------
# Core alternating optimizer
# ---------------------------------------------------------------------------


def joint_optimize(
    model: BlockCopolymerFreeEnergy,
    delta_phi: torch.Tensor,
    n_outer: int = 10,
    # --- dynamics phase ---
    n_dynamics_steps: int = 2000,
    dt: float = 1e-5,
    M: float = 1.0,
    dynamics_method: str = "semi-implicit",
    # --- box optimization phase ---
    n_box_steps: int = 10000,
    lr_box: float = 1.0,
    tol_box: float = 1e-5,
    # --- convergence of outer loop ---
    tol_F: float = 1e-5,
    # --- trajectory recording ---
    save_every: int = 1,
    # --- logging ---
    log_every_outer: int = 1,
    log_dynamics: bool = False,
    log_box: int = 0,
) -> dict:
    """
    Alternate Cahn-Hilliard dynamics with box-length optimization.

    Parameters
    ----------
    model : BlockCopolymerFreeEnergy
        Must be created with ``optimize_box=True``.  ``model.log_L`` holds the
        current box lengths and is updated in-place by the box phase.
    delta_phi : torch.Tensor
        Initial order-parameter field, shape (n_components, *grid_shape).
        Updated in-place across outer iterations (the tensor itself is replaced,
        not mutated).
    n_outer : int
        Number of outer (alternating) iterations.
    n_dynamics_steps : int
        Cahn-Hilliard steps per dynamics phase.
    dt : float
        Time step for dynamics.
    M : float
        Mobility coefficient.
    dynamics_method : str
        ``"semi-implicit"`` (recommended) or ``"forward-euler"``.
    n_box_steps : int
        Max gradient steps per box-optimization phase.
    lr_box : float
        Initial Armijo line-search step size for box optimization.
    tol_box : float
        Gradient-norm tolerance for box convergence within each phase.
    tol_F : float
        Outer-loop convergence: stop if |ΔF| < tol_F between successive
        outer iterations (after both phases).
    log_every_outer : int
        Print a one-line outer-loop summary every this many outer iterations.
    log_dynamics : bool
        If True, print per-step dynamics progress (passed to simulate()).
    log_box : int
        Print box-optimization progress every this many steps (0 = silent).

    Returns
    -------
    dict with keys
        delta_phi_final  : torch.Tensor   — final order-parameter field
        L_final          : list[float]    — final box lengths
        F_initial        : float
        F_final          : float
        outer_idx        : list[int]      — outer iterations at which snapshots were taken
        F_after_dynamics : list[float]    — F after dynamics at each recorded outer iter
        F_after_box      : list[float]    — F after box-opt at each recorded outer iter
        L_frames         : list[list]     — box lengths at each recorded outer iter
        phi_frames       : list[ndarray]  — phi snapshots at each recorded outer iter
        converged        : bool
        n_outer_completed: int
    """
    if not model.optimize_box:
        raise ValueError("model must be created with optimize_box=True")

    # project initial field onto constraint manifold
    delta_phi = model._project_order_parameter(delta_phi.clone())

    # record initial free energy (at current box, before any stepping)
    with torch.no_grad():
        K2_init = model._compute_K2(model.L)
        Gamma_init = model._compute_gamma_ij(K2_init)
        F_initial = model(delta_phi, Gamma_ij=Gamma_init).item()

    outer_idx: list[int] = []
    F_after_dynamics: list[float] = []
    F_after_box: list[float] = []
    L_frames: list[list] = []
    phi_frames: list[np.ndarray] = []

    def _record(outer: int, F_dyn: float, F_box: float, phi: torch.Tensor) -> None:
        outer_idx.append(outer)
        F_after_dynamics.append(F_dyn)
        F_after_box.append(F_box)
        L_frames.append(model.L.detach().cpu().tolist())
        phi_frames.append(phi.detach().cpu().numpy().copy())

    # record outer=0: initial state before any phase
    _record(0, F_initial, F_initial, delta_phi)

    F_prev = F_initial
    converged = False
    n_outer_completed = 0

    print(f"{'Outer':>6}  {'Phase':>10}  {'F':>20}  {'Box lengths'}")
    print("-" * 72)

    for outer in range(1, n_outer + 1):
        # =================================================================
        # Phase 1: Cahn-Hilliard dynamics with fixed box
        # =================================================================
        sim_result = simulate(
            model,
            delta_phi=delta_phi,
            n_steps=n_dynamics_steps,
            dt=dt,
            M=M,
            method=dynamics_method,
            save_every=n_dynamics_steps,  # only record the final state
            log_every=n_dynamics_steps if log_dynamics else 0,
        )
        delta_phi = torch.from_numpy(sim_result.phi[-1]).to(torch.float64)
        F_dyn = float(sim_result.F[-1])

        # guard: stop if dynamics produced NaN or negative densities
        f_expanded = model._f_expanded()
        rho_min = (delta_phi + f_expanded * model.phi_bar).min().item()
        if torch.isnan(delta_phi).any() or math.isnan(F_dyn) or rho_min < 0:
            print(
                f"\nERROR: Dynamics phase in outer iteration {outer} produced an "
                f"invalid density field (min rho = {rho_min:.4e}, F = {F_dyn}).\n"
                f"Reduce --dt and retry."
            )
            n_outer_completed = outer - 1
            break

        if log_every_outer > 0 and outer % log_every_outer == 0:
            L_str = "  ".join(f"{v:.5f}" for v in model.L.tolist())
            print(f"{outer:>6}  {'dynamics':>10}  {F_dyn:>20.10e}  [{L_str}]")

        # =================================================================
        # Phase 2: Box optimization with fixed density
        # =================================================================
        box_result = optimize_box_lengths(
            model=model,
            delta_phi=delta_phi,
            n_steps=n_box_steps,
            lr=lr_box,
            tol=tol_box,
            log_every=log_box,
        )
        F_box = box_result["F_final"]

        if log_every_outer > 0 and outer % log_every_outer == 0:
            L_str = "  ".join(f"{v:.5f}" for v in box_result["L_final"])
            print(f"{outer:>6}  {'box opt':>10}  {F_box:>20.10e}  [{L_str}]")

        # =================================================================
        # Record snapshot every save_every outer iterations
        # =================================================================
        if outer % save_every == 0:
            _record(outer, F_dyn, F_box, delta_phi)

        # =================================================================
        # Outer-loop convergence check
        # =================================================================
        n_outer_completed = outer
        dF = abs(F_box - F_prev)
        F_prev = F_box

        if dF < tol_F:
            converged = True
            if outer_idx[-1] != outer:  # ensure final state is always captured
                _record(outer, F_dyn, F_box, delta_phi)
            print(
                f"\nOuter loop converged at iteration {outer}: |ΔF| = {dF:.4e} < {tol_F:.4e}"
            )
            break

    # capture the very last state if not already recorded (e.g. n_outer not divisible by save_every)
    if outer_idx[-1] != n_outer_completed:
        _record(
            n_outer_completed,
            F_after_dynamics[-1] if F_after_dynamics else F_initial,
            F_prev,
            delta_phi,
        )

    return {
        "delta_phi_final": delta_phi,
        "L_final": model.L.detach().cpu().tolist(),
        "F_initial": F_initial,
        "F_final": F_prev,
        "outer_idx": outer_idx,
        "F_after_dynamics": F_after_dynamics,
        "F_after_box": F_after_box,
        "L_frames": L_frames,
        "phi_frames": phi_frames,
        "converged": converged,
        "n_outer_completed": n_outer_completed,
    }


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------


def _save_hdf5(
    path: str,
    result: dict,
    model: BlockCopolymerFreeEnergy,
    args,
) -> None:
    """Write the optimization trajectory and metadata to an HDF5 file."""
    data = SimulationData.from_model(model)
    data.phi = np.stack(result["phi_frames"], axis=0)
    data.F = np.array(result["F_after_box"], dtype=np.float64)
    data.box_lengths = np.array(result["L_frames"], dtype=np.float64)

    data.to_hdf5(path)

    # Script-specific provenance attrs (not part of SimulationData schema)
    with h5py.File(path, "a") as f:
        f.attrs["dt"] = args.dt
        f.attrs["M"] = args.M
        f.attrs["dynamics_method"] = args.method
        f.attrs["n_outer"] = args.n_outer
        f.attrs["n_outer_completed"] = result["n_outer_completed"]
        f.attrs["n_dynamics_steps"] = args.n_dynamics
        f.attrs["n_box_steps"] = args.n_box_steps
        f.attrs["lr_box"] = args.lr_box
        f.attrs["tol_box"] = args.tol_box
        f.attrs["tol_F"] = args.tol_F
        f.attrs["save_every"] = args.save_every
        f.attrs["seed"] = args.seed
        f.attrs["init_amplitude"] = args.amplitude
        f.attrs["converged"] = result["converged"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Alternating Cahn-Hilliard dynamics + box-length optimization."
    )
    p.add_argument(
        "output",
        nargs="?",
        default="joint_optimization.h5",
        help="Output HDF5 file (default: joint_optimization.h5)",
    )
    # outer loop
    p.add_argument(
        "--n-outer",
        type=int,
        default=100,
        help="Number of outer (alternating) iterations (default: 100)",
    )
    p.add_argument(
        "--tol-F",
        type=float,
        default=1e-6,
        help="Outer convergence tolerance on |ΔF| (default: 1e-5)",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Record a snapshot every N outer iterations (default: 1)",
    )

    # dynamics phase
    p.add_argument(
        "--n-dynamics",
        type=int,
        default=2000,
        help="Cahn-Hilliard steps per dynamics phase (default: 2000)",
    )
    p.add_argument(
        "--dt", type=float, default=1e-5, help="Dynamics time step (default: 1e-5)"
    )
    p.add_argument(
        "--M", type=float, default=1.0, help="Mobility coefficient (default: 1.0)"
    )
    p.add_argument(
        "--method",
        default="semi-implicit",
        choices=["semi-implicit", "forward-euler"],
        help="Dynamics time-stepping scheme (default: semi-implicit)",
    )

    # box optimization phase
    p.add_argument(
        "--n-box-steps",
        type=int,
        default=1000,
        help="Max box-opt gradient steps per outer iteration (default: 10000)",
    )
    p.add_argument(
        "--lr-box",
        type=float,
        default=0.1,
        help="Box optimization initial step size (default: 1.0)",
    )
    p.add_argument(
        "--tol-box",
        type=float,
        default=1e-5,
        help="Box gradient-norm tolerance per phase (default: 1e-5)",
    )

    # system
    p.add_argument(
        "--grid", type=int, default=32, help="Grid resolution NxN (default: 32)"
    )
    p.add_argument(
        "--Lx", type=float, default=10.0, help="Initial box length x (default: 10.0)"
    )
    p.add_argument(
        "--Ly",
        type=float,
        default=math.sqrt(3) * 10.0,
        help="Initial box length y (default: 10.0)",
    )
    p.add_argument(
        "--chiN", type=float, default=26.0, help="chiN parameter (default: 26.0)"
    )
    p.add_argument(
        "--amplitude",
        type=float,
        default=0.05,
        help="Initial fluctuation amplitude (default: 0.05)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # restart
    p.add_argument(
        "--restart",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume from the final frame of an existing HDF5 file",
    )

    # logging
    p.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print outer-loop summary every N iterations (default: 1)",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    torch.manual_seed(args.seed)

    # --- build system ---
    if args.restart is not None:
        print(f"Restarting from {args.restart} (final frame)")
        sim_data = SimulationData.from_hdf5(args.restart)
        model = sim_data.build_model(optimize_box=True)
        initial_box = model.L.detach().cpu().tolist()
    else:
        N = 100
        chi_matrix = torch.ones((3, 3), dtype=torch.float64) * args.chiN
        chi_matrix.fill_diagonal_(0.0)
        block_fractions = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float64)
        l_ij_matrix = torch.zeros((3, 3), dtype=torch.float64)

        model = BlockCopolymerFreeEnergy(
            N=N,
            chi_matrix=chi_matrix,
            l_ij_matrix=l_ij_matrix,
            block_fractions=block_fractions,
            phi_bar=1.0,
            optimize_box=True,
            grid_shape=(args.grid, args.grid),
            box_lengths=(args.Lx, args.Ly),
            init_amplitude=args.amplitude,
        )
        initial_box = [args.Lx, args.Ly]

    delta_phi_init = model._project_order_parameter(model.delta_phi.data.clone())

    grid_str = "x".join(str(s) for s in model.grid_shape)
    box_str = "x".join(f"{v:.4f}" for v in initial_box)
    print("=" * 72)
    print("Alternating Dynamics + Box Optimization")
    print("=" * 72)
    if args.restart:
        print(f"  Restart from: {args.restart}")
    print(f"  Grid        : {grid_str}   Initial box: {box_str}")
    print(f"  chiN = {model.chi_matrix[0, 1].item():.1f},  N = {model.N}")
    print(
        f"  Dynamics    : {args.method},  dt = {args.dt},  {args.n_dynamics} steps/phase"
    )
    print(f"  Box opt     : {args.n_box_steps} steps/phase,  lr = {args.lr_box}")
    print(f"  Iterations  : {args.n_outer}  (save every {args.save_every})")
    print(f"  Output      : {args.output}")
    print()

    result = joint_optimize(
        model=model,
        delta_phi=delta_phi_init,
        n_outer=args.n_outer,
        n_dynamics_steps=args.n_dynamics,
        dt=args.dt,
        M=args.M,
        dynamics_method=args.method,
        n_box_steps=args.n_box_steps,
        lr_box=args.lr_box,
        tol_box=args.tol_box,
        tol_F=args.tol_F,
        save_every=args.save_every,
        log_every_outer=args.log_every,
        log_dynamics=False,
        log_box=0,
    )

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Iterations completed : {result['n_outer_completed']}")
    print(f"  Converged            : {result['converged']}")
    print(f"  F initial : {result['F_initial']:.10e}")
    print(f"  F final   : {result['F_final']:.10e}")
    print(f"  delta_F   : {result['F_final'] - result['F_initial']:.4e}")
    print()
    for d, (L0, Lf) in enumerate(zip(initial_box, result["L_final"])):
        print(f"  L{d}: {L0:.6f}  ->  {Lf:.6f}   ({(Lf / L0 - 1) * 100:+.3f} %)")

    print(f"\nSaving to {args.output} ...")
    _save_hdf5(args.output, result, model, args)
    n_frames = len(result["outer_idx"])
    print(
        f"  phi shape        : ({n_frames}, {model.n_components}, {', '.join(str(s) for s in model.grid_shape)})"
    )
    print(f"  box_lengths shape: ({n_frames}, {model.ndim})")
    print(f"  Frames saved     : {n_frames}")
    print(f"\nDone. Written to: {args.output}")


if __name__ == "__main__":
    main()
