"""
Box-Length-Only Optimization
=============================

CLI wrapper around rpa.optimizers.optimize_box_only. Optimizes only the
lattice parameters with the density field delta_phi held fixed at whatever
is stored in the input HDF5 file.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import torch

from rpa import SimulationData
from rpa.optimizers import optimize_box_only


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Optimize box lengths only, keeping delta_phi fixed."
    )
    p.add_argument("-i", "--input_file", type=str, required=True)
    p.add_argument("-o", "--output_file", type=str, default="result_box.h5")
    p.add_argument("-n", "--n_steps", type=int, default=500)
    p.add_argument("--lr_box", type=float, default=1.0)
    p.add_argument("--box_grad_scale", type=float, default=10.0)
    p.add_argument(
        "--tol_grad",
        type=float,
        default=1e-5,
        help="Convergence tolerance on ||grad log_L|| (default: 1e-5)"
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
    )
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    tol_grad = args.tol_grad

    simulation_data = SimulationData.from_hdf5(args.input_file)
    model = simulation_data.build_model(optimize_box=True, dtype=dtype, device=args.device)

    print("initial box lengths:", model.L.tolist())
    print("initial free energy:", model(model.get_order_parameters(), project=False).item())

    result = optimize_box_only(
        model,
        n_steps=args.n_steps,
        lr_box=args.lr_box,
        box_grad_scale=args.box_grad_scale,
        tol_grad=tol_grad,
        log_every=args.log_every,
    )

    result.to_hdf5(args.output_file)
    print("saved to", args.output_file)
