"""
Box-Length-Only Optimization
=============================

Optimizes only the lattice parameters (box lengths) with the density field
delta_phi held fixed at whatever is stored in the input HDF5 file.

Useful as a post-processing step after density convergence, or to find the
optimal box size for a given morphology without re-running the full alternating
optimizer.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import numpy as np
import torch
from tqdm import tqdm

from rpa import SimulationData
from rpa.optimizers.pgd import backtracking_line_search_box


def optimize_box_only(
    model,
    n_steps: int = 500,
    lr_box: float = 1.0,
    box_grad_scale: float = 10.0,
    tol_grad: float = 1e-6,
    log_every: int = 50,
) -> SimulationData:
    """
    Gradient descent on box lengths with delta_phi fixed.

    Parameters
    ----------
    model : BlockCopolymerFreeEnergy
        Must have optimize_box=True. delta_phi is treated as a constant.
    n_steps : int
        Maximum number of gradient steps.
    lr_box : float
        Initial step size for backtracking line search.
    box_grad_scale : float
        Scale applied to the log-L gradient before the line search.
    tol_grad : float
        Stop when ||grad log_L|| < tol_grad.
    log_every : int
        Print a status line every this many steps.
    """
    if not model.optimize_box:
        raise ValueError("model.optimize_box must be True")

    delta_phi = model.delta_phi.data.detach().clone()

    F_trajectory = []
    box_lengths_trajectory = []
    converged = False
    grad_norm = float("inf")

    with torch.no_grad():
        current_F = model(delta_phi, project=False).item()

    for step in tqdm(range(n_steps)):
        model.log_L.requires_grad_(True)
        F_val = model(delta_phi, project=False)
        F_val.backward()

        grad_log_L = model.log_L.grad.clone()
        current_F = F_val.item()
        model.log_L.grad = None

        grad_norm = grad_log_L.norm().item()

        F_trajectory.append(current_F)
        box_lengths_trajectory.append(model.L.detach().cpu().tolist())

        if step % log_every == 0:
            L_str = ", ".join(f"L{d}={v:.6f}" for d, v in enumerate(model.L.tolist()))
            print(
                f"Step {step:4d} | F = {current_F:.6e} | "
                f"|grad_L| = {grad_norm:.4e} | {L_str}"
            )

        if grad_norm < tol_grad:
            converged = True
            break

        _, current_F = backtracking_line_search_box(
            model,
            delta_phi,
            current_F,
            grad_log_L,
            grad_scale=box_grad_scale,
            alpha_init=lr_box,
        )

    if converged:
        print(f"Converged at step {step}")
    else:
        print(f"Did not converge after {n_steps} steps")

    result = SimulationData.from_model(model)
    n = len(F_trajectory)
    result.F = np.array(F_trajectory)
    result.box_lengths = np.array(box_lengths_trajectory)
    # phi is fixed; store a single copy broadcast to n_frames for consistency
    phi_np = delta_phi.cpu().numpy()
    result.phi = np.broadcast_to(phi_np[np.newaxis], (n, *phi_np.shape)).copy()
    result.converged = converged
    return result


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
