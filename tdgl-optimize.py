"""
TDGL / Cahn-Hilliard Density Solver
=====================================

Reads a system definition and initial density field from an input HDF5 file,
runs Cahn-Hilliard (Model B / TDGL) dynamics to relax the density profile at
*fixed* box dimensions, and writes the resulting trajectory to an output HDF5.

The input HDF5 is expected to contain at least one frame (phi + box_lengths),
as produced by generate_input.py / initial_conditions.py.  If the file holds
multiple frames (e.g. from a previous run), the last frame is used as the
starting point.

Output HDF5 schema (from SimulationData.to_hdf5):
  phi          : (n_frames, n_components, *grid_shape)
  F            : (n_frames,)
  box_lengths  : (n_frames, ndim)
  block_fractions, chi_matrix, l_ij_matrix, plus scalar attrs

Additional run-specific attrs written after:
  dt, M, method, n_steps, n_steps_completed, save_every

Usage
-----
    python tdgl-optimize.py -i input.h5
    python tdgl-optimize.py -i input.h5 -o result.h5 --n-steps 10000 --dt 0.01
    python tdgl-optimize.py -i input.h5 --method forward-euler --dt 1e-6
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import h5py
import numpy as np
import torch

from rpa import BlockCopolymerFreeEnergy
from simulation_io import SimulationData
from optimizers.dynamics.simulate import forward_euler_step, semi_implicit_step


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
    Integrate the Cahn-Hilliard equation and collect trajectory snapshots.

    Parameters
    ----------
    model : BlockCopolymerFreeEnergy
        Must have ``optimize_box=False`` so Gamma_ij is pre-cached.
    delta_phi : torch.Tensor
        Initial order-parameter field, shape (n_components, *grid_shape).
    n_steps : int
        Total number of time steps.
    dt : float
        Time step size.
    M : float
        Mobility coefficient.
    method : str
        ``"semi-implicit"`` or ``"forward-euler"``.
    save_every : int
        Record a full snapshot every this many steps (frame 0 = initial state).
    log_every : int
        Print a one-line summary every this many steps (0 = silent).

    Returns
    -------
    dict with numpy arrays: t_frames, F_frames, phi_frames, box_lengths_frames,
    and n_steps_completed (int).
    """
    delta_phi = model._project_order_parameter(delta_phi.clone())

    K2 = model._compute_K2(model.L)
    if hasattr(model, "_Gamma_ij_cached") and model._Gamma_ij_cached is not None:
        Gamma_ij = model._Gamma_ij_cached
    else:
        Gamma_ij = model._compute_gamma_ij(K2)

    step_fn = {
        "semi-implicit": semi_implicit_step,
        "forward-euler": forward_euler_step,
    }[method]

    L_np = model.L.detach().cpu().numpy().copy()

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
    f_expanded = model._f_expanded()

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

    return {
        "t_frames": np.array(t_frames, dtype=np.float64),
        "F_frames": np.array(F_frames, dtype=np.float64),
        "phi_frames": np.stack(phi_frames, axis=0),
        "box_lengths_frames": np.stack(box_lengths_frames, axis=0),
        "n_steps_completed": n_steps_completed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run Cahn-Hilliard dynamics from an input HDF5 file (fixed box)."
    )
    p.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input HDF5 file (system definition + initial conditions)",
    )
    p.add_argument(
        "-o",
        "--output",
        default="result.h5",
        help="Output HDF5 file for the trajectory (default: result.h5)",
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
        default=1e-4,
        help="Time step size (default: 1e-4)",
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
        default=1000,
        help="Record a full snapshot every N steps (default: 1000)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Print progress every N steps, 0 = silent (default: 500)",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    # --- Load system definition and initial conditions ---
    print(f"Loading from {args.input} ...")
    sim_data = SimulationData.from_hdf5(args.input)
    model = sim_data.build_model(optimize_box=False)

    if sim_data.n_frames > 0:
        phi_init_np, _ = sim_data.final_state()
        delta_phi = torch.from_numpy(phi_init_np).to(torch.float64)
        print(f"  Initial conditions: last frame of input trajectory")
    else:
        delta_phi = model.delta_phi.data.clone()
        print(f"  Initial conditions: model default (random)")

    print(f"  Grid       : {model.grid_shape}")
    print(f"  Box        : {[f'{v:.4f}' for v in model.L.tolist()]}")
    print(f"  n_comp     : {model.n_components}")
    print(f"  Method     : {args.method},  dt = {args.dt}")
    print(f"  Steps      : {args.n_steps}  (save every {args.save_every})")
    n_frames = args.n_steps // args.save_every + 1
    print(f"  Frames     : ~{n_frames}")
    print()

    # --- Run ---
    result = _run(
        model=model,
        delta_phi=delta_phi,
        n_steps=args.n_steps,
        dt=args.dt,
        M=args.M,
        method=args.method,
        save_every=args.save_every,
        log_every=args.log_every,
    )

    # --- Save ---
    print(f"\nSaving to {args.output} ...")
    out = SimulationData.from_model(model)
    out.phi = result["phi_frames"]
    out.F = result["F_frames"]
    out.box_lengths = result["box_lengths_frames"]
    out.to_hdf5(args.output)

    with h5py.File(args.output, "a") as f:
        f.attrs["dt"] = args.dt
        f.attrs["M"] = args.M
        f.attrs["method"] = args.method
        f.attrs["n_steps"] = args.n_steps
        f.attrs["n_steps_completed"] = result["n_steps_completed"]
        f.attrs["save_every"] = args.save_every
        f.create_dataset("t", data=result["t_frames"])

    phi_shape = result["phi_frames"].shape
    print(f"  phi shape  : {phi_shape}  (n_frames x n_comp x grid...)")
    print(f"  F_initial  = {result['F_frames'][0]:.8e}")
    print(f"  F_final    = {result['F_frames'][-1]:.8e}")
    print(f"  delta_F    = {result['F_frames'][-1] - result['F_frames'][0]:.8e}")
    print(f"  Frames     : {phi_shape[0]}")
    print(f"\nDone. Written to: {args.output}")


if __name__ == "__main__":
    main()
