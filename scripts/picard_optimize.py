"""
Grand-Canonical Fixed-Point (Picard) Optimization of the Density Field
======================================================================

Solves the RPA free-energy stationarity condition at fixed box length by
repeated substitution (Picard iteration) rather than gradient descent.

Splitting the free energy as ``F = F_mixing + F_res`` (ideal Flory-Huggins
reference plus the residual ``F_res = Delta_F_int - F_mixing_2``), the
grand-canonical Euler-Lagrange condition gives the per-component update

    rho_i(r) = f_i * phi_bar * exp(- f_i * n_grid * dF_res/d(delta_phi_i)(r)),

optionally with the per-component spatial mean pinned to ``f_i * phi_bar``. Each
iteration plugs the current density into the right-hand side and linearly mixes
the result with the previous iterate using ``--alpha``.

The box length is held fixed here (phi-only). Use ``scripts/optimize_density.py``
for joint density+box optimization.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import torch

from rpa import SimulationData
from rpa.optimizers import picard_optimize_phi_only


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Input hdf5 file containing model spec and initial conditions",
    )
    p.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="picard_result.h5",
        help="Output hdf5 file containing optimization result",
    )
    p.add_argument(
        "-n",
        "--n_steps",
        type=int,
        default=20000,
        help="Maximum number of Picard iterations",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Mixing parameter in (0, 1]. Strongly ordered cases may want ~0.1 "
        "(finds deeper minima); near-disordered cases may need ~0.03 to stay "
        "stable. Reduce if the free energy grows without bound.",
    )
    p.add_argument(
        "--precond_strength",
        type=float,
        default=1.0,
        help="Strength of the Fourier preconditioner that damps the stiff high-k "
        "modes (essential for fine 3D grids). Raise for more stability.",
    )
    p.add_argument(
        "--no_precondition",
        action="store_true",
        help="Disable the Fourier preconditioner (only for small/mild problems).",
    )
    p.add_argument(
        "--anderson_m",
        type=int,
        default=0,
        help="Anderson history depth (0 = preconditioned simple mixing, the "
        "recommended path; > 0 is experimental).",
    )
    p.add_argument(
        "--record_every",
        type=int,
        default=10,
        help="Store a trajectory frame every this many steps (final step always "
        "recorded). Larger values keep long-run output files small.",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=None,
        help="Convergence tolerance on the free-energy change |F - F_ref| "
        "(default: 1e-6 for float64, 1e-5 for float32)",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Consecutive steps within tol required to declare convergence",
    )
    p.add_argument(
        "--no_pin_mean",
        action="store_true",
        help="Let per-component means float (grand canonical). Default pins "
        "<rho_i> = f_i * phi_bar each step (canonical mean composition).",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Floating-point precision (float32 is faster but less accurate)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to use, e.g. 'cpu', 'cuda', 'cuda:0' (default: cpu)",
    )

    args = p.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    tol = args.tol if args.tol is not None else (1e-5 if dtype == torch.float32 else 1e-6)

    simulation_data = SimulationData.from_hdf5(args.input_file)

    model = simulation_data.build_model(
        optimize_box=False, dtype=dtype, device=args.device
    )

    print("box lengths (fixed):", model.L.tolist())
    print("initial free energy:", model(model.get_order_parameters()).item())

    result = picard_optimize_phi_only(
        model,
        n_steps=args.n_steps,
        alpha=args.alpha,
        tol=tol,
        patience=args.patience,
        pin_mean=not args.no_pin_mean,
        anderson_m=args.anderson_m,
        precondition=not args.no_precondition,
        precond_strength=args.precond_strength,
        record_every=args.record_every,
    )

    print("final free energy:", result.F[-1])
    print("max incompressibility drift:", result.incompressibility_drift.max())

    result.to_hdf5(args.output_file)
    print("saved to", args.output_file)
