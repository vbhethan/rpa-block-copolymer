"""
Joint Optimization of Density Field and Box Dimensions
=======================================================

Optimizes the RPA free energy functional with respect to both the order
parameter field delta_phi (via projected gradient descent) and the box
dimensions L_1, ..., L_d (via standard gradient descent on log L).

Key insight: the constraint set for delta_phi is independent of the box
dimensions. Both constraints (local incompressibility and mass conservation)
are purely algebraic on the grid values, so the projection operator is
the same regardless of the box lengths. The box lengths are unconstrained
positive parameters that interact with the density only through the
functional evaluation (via the wavevectors q = 2*pi*n/L and the volume
prefactors).
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import torch

from rpa import SimulationData
from rpa.optimizers import optimize_joint


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
        default="result.h5",
        help="Output hdf5 file containing optimization result",
    )
    p.add_argument(
        "-n",
        "--n_outer",
        type=int,
        default=250,
        help="Number of outer iterations",
    )
    p.add_argument(
        "--lr_box",
        type=float,
        default=1.0,
        help="Learning rate for box lengths",
    )
    p.add_argument(
        "--lr_phi",
        type=float,
        default=0.1,
        help="Learning rate for density field",
    )
    p.add_argument(
        "--n_inner_phi",
        type=int,
        default=2000,
        help="Number of inner iterations for density field",
    )
    p.add_argument(
        "--n_inner_box",
        type=int,
        default=10,
        help="Number of inner iterations for box lengths",
    )
    p.add_argument(
        "--tol_grad_phi",
        type=float,
        default=None,
        help="Tolerance for gradient of density field (default: 1e-6 for float64, 1e-5 for float32)",
    )
    p.add_argument(
        "--tol_grad_box",
        type=float,
        default=None,
        help="Tolerance for gradient of box lengths (default: 1e-6 for float64, 1e-5 for float32)",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Floating-point precision (float32 is faster but less accurate)",
    )
    p.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Number of outer iterations between logging",
    )
    # p.add_argument(
    #     "--use_line_search",
    #     action="store_true",
    #     help="Use line search for density field update",
    # )
    p.add_argument(
        "--box_grad_scale",
        type=float,
        default=10,
        help="Scale for box-length gradient",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to use, e.g. 'cpu', 'cuda', 'cuda:0' (default: cpu)",
    )

    args = p.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    tol_grad_phi = (
        args.tol_grad_phi
        if args.tol_grad_phi is not None
        else (1e-5 if dtype == torch.float32 else 1e-6)
    )
    tol_grad_box = (
        args.tol_grad_box
        if args.tol_grad_box is not None
        else (1e-5 if dtype == torch.float32 else 1e-6)
    )

    simulation_data = SimulationData.from_hdf5(args.input_file)

    model = simulation_data.build_model(optimize_box=True, dtype=dtype, device=args.device)

    print("initial box lengths:", model.L.tolist())
    print("initial free energy:", model(model.get_order_parameters()).item())

    result = optimize_joint(
        model,
        n_outer=args.n_outer,
        lr_box=args.lr_box,
        lr_phi=args.lr_phi,
        n_inner_phi=args.n_inner_phi,
        n_inner_box=args.n_inner_box,
        tol_grad_phi=tol_grad_phi,
        tol_grad_box=tol_grad_box,
        log_every=args.log_every,
        use_line_search=True,
        box_grad_scale=args.box_grad_scale,
    )

    result.to_hdf5(args.output_file)
    print("saved to", args.output_file)
