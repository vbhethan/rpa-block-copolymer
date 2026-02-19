"""
Forward Cahn-Hilliard simulation with HDF5 output.

Runs a simulation from a random initial density field and saves a trajectory
to an HDF5 file containing:
  - t       : time at each recorded frame
  - F       : free energy at each recorded frame
  - phi     : order-parameter snapshots, shape (n_frames, n_components, *grid_shape)
  - conservation_error      : max |mean(delta_phi_i)| at each frame
  - incompressibility_error : max |sum_i delta_phi_i(r)| at each frame

HDF5 attributes store all simulation parameters.

Dependencies: PyTorch, h5py (pip install h5py)

Usage
-----
    python run_simulation.py                         # defaults
    python run_simulation.py out.h5 --n-steps 10000 --save-every 100
    python run_simulation.py out.h5 --method forward-euler --dt 1e-6
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import h5py
import numpy as np
import torch

from dynamics import forward_euler_step, semi_implicit_step
from rpa import BlockCopolymerFreeEnergy


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------


def _run(
    model: BlockCopolymerFreeEnergy,
    delta_phi: torch.Tensor,
    n_steps: int,
    dt: float,
    M: float,
    method: str,
    save_every: int,
    log_every: int,
) -> dict:
    """
    Integrate the Cahn-Hilliard equation and record snapshots.

    Parameters
    ----------
    model : BlockCopolymerFreeEnergy
        Must have ``optimize_box=False`` (Gamma_ij is pre-cached).
    delta_phi : torch.Tensor
        Initial order-parameter field, shape (n_components, *grid_shape).
        Will be projected onto the constraint manifold before stepping.
    n_steps : int
        Total number of time steps to take.
    dt : float
        Time step size.
    M : float
        Mobility coefficient.
    method : str
        ``"semi-implicit"`` or ``"forward-euler"``.
    save_every : int
        Record a frame every this many steps (frame 0 = initial state).
    log_every : int
        Print a one-line progress summary every this many steps (0 = silent).

    Returns
    -------
    dict with numpy arrays:
        t_frames, F_frames, phi_frames, conservation_error,
        incompressibility_error, n_steps_completed
    """
    # Project initial field onto constraint manifold
    delta_phi = model._project_order_parameter(delta_phi.clone())

    # Pre-compute fixed Fourier-space quantities
    K2 = model._compute_K2(model.L)
    if hasattr(model, "_Gamma_ij_cached") and model._Gamma_ij_cached is not None:
        Gamma_ij = model._Gamma_ij_cached
    else:
        Gamma_ij = model._compute_gamma_ij(K2)

    step_fn = {
        "semi-implicit": semi_implicit_step,
        "forward-euler": forward_euler_step,
    }[method]

    # Storage lists
    t_frames: list[float] = []
    F_frames: list[float] = []
    phi_frames: list[np.ndarray] = []
    cons_err_frames: list[float] = []
    incomp_err_frames: list[float] = []

    def _record(step: int, phi: torch.Tensor) -> None:
        t = step * dt
        with torch.no_grad():
            F = model(phi, Gamma_ij=Gamma_ij).item()
        cons_err = phi.mean(dim=model.spatial_dims).abs().max().item()
        incomp_err = phi.sum(dim=0).abs().max().item()

        t_frames.append(t)
        F_frames.append(F)
        phi_frames.append(phi.detach().cpu().numpy().copy())
        cons_err_frames.append(cons_err)
        incomp_err_frames.append(incomp_err)

        if log_every > 0 and (step == 0 or step % log_every == 0):
            print(
                f"  step {step:7d} | t = {t:.4e} | F = {F:.8e} | "
                f"cons = {cons_err:.2e} | incomp = {incomp_err:.2e}"
            )

    # --- initial frame ---
    _record(0, delta_phi)

    n_steps_completed = 0
    f_expanded = model._f_expanded()

    for step in range(1, n_steps + 1):
        delta_phi = step_fn(delta_phi, model, dt, M, Gamma_ij, K2)
        n_steps_completed = step

        if step % save_every == 0 or step == n_steps:
            _record(step, delta_phi)
        elif log_every > 0 and step % log_every == 0:
            # Print without recording a full frame
            t = step * dt
            F_latest = F_frames[-1]
            cons_latest = cons_err_frames[-1]
            incomp_latest = incomp_err_frames[-1]
            print(
                f"  step {step:7d} | t = {t:.4e} | F = {F_latest:.8e} "
                f"(last frame) | cons = {cons_latest:.2e} | incomp = {incomp_latest:.2e}"
            )

        # Safety: abort on negative density
        rho_min = (delta_phi + f_expanded * model.phi_bar).min().item()
        if rho_min < 0:
            print(
                f"\nWARNING: negative density at step {step} "
                f"(min rho = {rho_min:.4e}). Aborting — reduce dt."
            )
            break

    return {
        "t_frames": np.array(t_frames, dtype=np.float64),
        "F_frames": np.array(F_frames, dtype=np.float64),
        "phi_frames": np.stack(phi_frames, axis=0),  # (n_frames, n_comp, *grid)
        "conservation_error": np.array(cons_err_frames, dtype=np.float64),
        "incompressibility_error": np.array(incomp_err_frames, dtype=np.float64),
        "n_steps_completed": n_steps_completed,
    }


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------


def _save_hdf5(path: str, result: dict, model: BlockCopolymerFreeEnergy, args) -> None:
    """Write simulation result and metadata to an HDF5 file."""
    with h5py.File(path, "w") as f:
        # --- simulation metadata ---
        f.attrs["system"] = "3-component symmetric star copolymer"
        f.attrs["N"] = model.N
        f.attrs["b"] = model.b
        f.attrs["n_components"] = model.n_components
        f.attrs["block_fractions"] = model.f_vec.cpu().numpy()
        f.attrs["phi_bar"] = model.phi_bar
        f.attrs["grid_shape"] = list(model.grid_shape)
        f.attrs["box_lengths"] = model.L.cpu().numpy()
        f.attrs["dt"] = args.dt
        f.attrs["M"] = args.M
        f.attrs["method"] = args.method
        f.attrs["n_steps"] = args.n_steps
        f.attrs["n_steps_completed"] = result["n_steps_completed"]
        f.attrs["save_every"] = args.save_every
        f.attrs["seed"] = args.seed
        f.attrs["init_amplitude"] = args.amplitude

        # --- system parameter datasets ---
        kw = {"compression": "gzip", "compression_opts": 4}

        f.create_dataset("chi_matrix", data=model.chi_matrix.cpu().numpy())
        f.create_dataset("l_ij_matrix", data=model.l_ij_matrix.cpu().numpy())
        f.create_dataset("box_lengths", data=model.L.cpu().numpy())

        # --- trajectory datasets ---

        # time at each frame
        f.create_dataset("t", data=result["t_frames"], **kw)

        # free energy at each frame
        f.create_dataset("F", data=result["F_frames"], **kw)

        # order-parameter snapshots: (n_frames, n_components, *grid_shape)
        ds = f.create_dataset("phi", data=result["phi_frames"], **kw)
        ds.attrs["axes"] = "frame,component,x,y"

        # constraint diagnostics
        f.create_dataset("conservation_error", data=result["conservation_error"], **kw)
        f.create_dataset(
            "incompressibility_error", data=result["incompressibility_error"], **kw
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run Cahn-Hilliard simulation and save trajectory to HDF5."
    )
    p.add_argument(
        "output",
        nargs="?",
        default="simulation.h5",
        help="Output HDF5 file path (default: simulation.h5)",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=5000,
        help="Total number of time steps (default: 5000)",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Time step size (default: 0.01, use ~1e-6 for forward-euler)",
    )
    p.add_argument(
        "--M",
        type=float,
        default=1.0,
        help="Mobility coefficient (default: 1.0)",
    )
    p.add_argument(
        "--method",
        default="semi-implicit",
        choices=["semi-implicit", "forward-euler"],
        help="Time-stepping scheme (default: semi-implicit)",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Record a full-field snapshot every N steps (default: 50)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Print progress every N steps, 0 = silent (default: 500)",
    )
    p.add_argument(
        "--grid", type=int, default=32, help="Grid resolution (NxN, default: 32)"
    )
    p.add_argument(
        "--Lx", type=float, default=10.0, help="Box length x (default: 10.0)"
    )
    p.add_argument(
        "--Ly", type=float, default=10.0, help="Box length y (default: 10.0)"
    )
    p.add_argument("--chiN", type=float, default=26.0, help="chiN (default: 26.0)")
    p.add_argument(
        "--amplitude",
        type=float,
        default=0.05,
        help="Initial fluctuation amplitude (default: 0.05)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return p.parse_args()


def main():
    args = _parse_args()
    torch.manual_seed(args.seed)

    # --- Build model ---
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
        optimize_box=False,
        grid_shape=(args.grid, args.grid),
        box_lengths=(args.Lx, args.Ly),
        init_amplitude=args.amplitude,
    )

    print("=" * 64)
    print("Cahn-Hilliard Forward Simulation")
    print("=" * 64)
    print(f"  System     : 3-component symmetric star copolymer")
    print(f"  Grid       : {args.grid}x{args.grid}  Box: {args.Lx}x{args.Ly}")
    print(f"  chiN = {args.chiN},  N = {N},  M = {args.M}")
    print(f"  Method     : {args.method},  dt = {args.dt}")
    print(f"  Steps      : {args.n_steps}  (save every {args.save_every})")
    n_frames = args.n_steps // args.save_every + 1
    print(f"  Frames     : ~{n_frames}")
    print(f"  Output     : {args.output}")
    print()

    # Project random initial field onto constraint manifold
    delta_phi_init = model._project_order_parameter(model.delta_phi.data.clone())

    # --- Run ---
    result = _run(
        model=model,
        delta_phi=delta_phi_init,
        n_steps=args.n_steps,
        dt=args.dt,
        M=args.M,
        method=args.method,
        save_every=args.save_every,
        log_every=args.log_every,
    )

    # --- Save ---
    print(f"\nSaving to {args.output} ...")
    _save_hdf5(args.output, result, model, args)

    phi_shape = result["phi_frames"].shape
    print(f"  phi shape  : {phi_shape}  (n_frames x n_comp x Nx x Ny)")
    print(f"  F_initial  = {result['F_frames'][0]:.8e}")
    print(f"  F_final    = {result['F_frames'][-1]:.8e}")
    print(f"  delta_F    = {result['F_frames'][-1] - result['F_frames'][0]:.8e}")
    print(f"  Frames saved: {phi_shape[0]}")
    print(f"\nDone. Written to: {args.output}")


if __name__ == "__main__":
    main()
