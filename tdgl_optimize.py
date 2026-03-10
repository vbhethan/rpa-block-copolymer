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
  dt, M, method, n_steps, save_every

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

from simulation_io import SimulationData
from optimizers.dynamics.simulate import simulate


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
        print("  Initial conditions: last frame of input trajectory")
    else:
        delta_phi = model.delta_phi.data.clone()
        print("  Initial conditions: model default (random)")

    print(f"  Grid       : {model.grid_shape}")
    print(f"  Box        : {[f'{v:.4f}' for v in model.L.tolist()]}")
    print(f"  n_comp     : {model.n_components}")
    print(f"  Method     : {args.method},  dt = {args.dt}")
    print(f"  Steps      : {args.n_steps}  (save every {args.save_every})")
    print(f"  Frames     : ~{args.n_steps // args.save_every + 1}")
    print()

    # --- Run ---
    result = simulate(
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
    result.to_hdf5(args.output)

    # Write run-specific provenance attrs and time axis
    t = np.arange(result.n_frames) * args.save_every * args.dt
    with h5py.File(args.output, "a") as f:
        f.attrs["dt"] = args.dt
        f.attrs["M"] = args.M
        f.attrs["method"] = args.method
        f.attrs["n_steps"] = args.n_steps
        f.attrs["save_every"] = args.save_every
        f.create_dataset("t", data=t)

    print(f"  phi shape  : {result.phi.shape}  (n_frames x n_comp x grid...)")
    print(f"  F_initial  = {result.F[0]:.8e}")
    print(f"  F_final    = {result.F[-1]:.8e}")
    print(f"  delta_F    = {result.F[-1] - result.F[0]:.8e}")
    print(f"  Frames     : {result.n_frames}")
    print(f"\nDone. Written to: {args.output}")


if __name__ == "__main__":
    main()
