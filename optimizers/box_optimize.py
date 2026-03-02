"""
Box-length optimization with a fixed density profile.
======================================================

Loads the last-frame density field from a dynamics HDF5 file, then
minimizes the RPA free energy with respect to the box lengths L_1, ..., L_d
while holding the grid values of delta_phi completely fixed.

Physically: the grid-point *values* of the order parameter are frozen, so
stretching the box stretches the physical density profile. The free energy
changes because the wavevectors q = 2π n / L shift, which moves the
interaction-energy integrand along the structure factor.

Optimization strategy:
- Work in log-L space (log_L = log L) so positivity is automatic.
- Gradient descent with Armijo backtracking line search.
- Convergence criterion: |grad log_L| < tol.

Usage
-----
    python optimize_box.py simulation.h5
    python optimize_box.py simulation.h5 --n-steps 500 --lr 0.1 --tol 1e-7
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import math

import h5py
import numpy as np
import torch

from rpa import BlockCopolymerFreeEnergy


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def optimize_box_lengths(
    model: BlockCopolymerFreeEnergy,
    delta_phi: torch.Tensor,
    n_steps: int = 300,
    lr: float = 0.1,
    beta: float = 0.5,
    c: float = 1e-4,
    min_alpha: float = 1e-12,
    max_log_step: float = 2.0,
    tol: float = 1e-7,
    log_every: int = 50,
) -> dict:
    """
    Minimize F(delta_phi; L) over box lengths L with delta_phi held fixed.

    Gradient descent on log_L = log(L) with Armijo backtracking line search.
    log_L parameterization keeps L > 0 automatically and avoids scale issues.

    Parameters
    ----------
    model : BlockCopolymerFreeEnergy
        Must have ``optimize_box=True``. ``model.log_L`` is the optimized
        parameter; its initial value sets the starting box lengths.
    delta_phi : torch.Tensor
        Fixed order-parameter field, shape (n_components, *grid_shape).
        Not modified.
    n_steps : int
        Maximum number of gradient steps.
    lr : float
        Initial step size for backtracking line search.
    beta : float
        Step-size reduction factor when Armijo condition fails (0 < beta < 1).
    c : float
        Armijo sufficient-decrease constant.
    min_alpha : float
        Minimum step size; give up line search below this.
    max_log_step : float
        Maximum allowed |Δlog_L|_∞ per step (guards against huge jumps).
    tol : float
        Convergence tolerance on the gradient norm |∂F/∂log_L|.
    log_every : int
        Print a summary line every this many steps (0 = silent).

    Returns
    -------
    dict with keys:
        L_initial, L_final, F_initial, F_final,
        F_history, L_history, grad_norm_history,
        converged, n_steps_completed
    """
    if not model.optimize_box:
        raise ValueError("model must be created with optimize_box=True")

    delta_phi = delta_phi.detach()

    # --- initial state ---
    with torch.no_grad():
        F0 = model(delta_phi).item()

    L_initial = model.L.detach().cpu().tolist()
    F_history = [F0]
    L_history = [model.L.detach().cpu().tolist()]
    grad_norm_history = []

    if log_every > 0:
        L_str = "  ".join(f"L{d}={v:.6f}" for d, v in enumerate(L_initial))
        print(f"  step {0:5d} | F = {F0:.10e} | {L_str}")

    converged = False
    n_steps_completed = 0

    for step in range(1, n_steps + 1):
        # --- compute gradient dF/d(log_L) ---
        model.log_L.requires_grad_(True)
        if model.log_L.grad is not None:
            model.log_L.grad.zero_()

        F_val = model(delta_phi)
        F_val.backward()

        grad = model.log_L.grad.detach().clone()
        model.log_L.requires_grad_(False)

        grad_norm = grad.norm().item()
        grad_norm_history.append(grad_norm)

        if grad_norm < tol:
            converged = True
            n_steps_completed = step
            break

        # --- Armijo backtracking line search in log_L space ---
        direction = -grad  # steepest descent
        slope = (grad * direction).sum().item()  # = -||grad||^2 < 0

        log_L_orig = model.log_L.data.clone()
        current_F = F_val.item()
        alpha = lr
        accepted = False

        while alpha > min_alpha:
            step_vec = alpha * direction
            if step_vec.abs().max().item() > max_log_step:
                alpha *= beta
                continue

            model.log_L.data = log_L_orig + step_vec

            with torch.no_grad():
                F_candidate = model(delta_phi).item()

            if F_candidate <= current_F + c * alpha * slope:
                accepted = True
                break

            alpha *= beta

        if not accepted:
            # restore and stop — line search exhausted
            model.log_L.data = log_L_orig
            n_steps_completed = step
            if log_every > 0:
                print(f"  Line search failed at step {step}; stopping.")
            break

        n_steps_completed = step
        F_new = F_candidate

        F_history.append(F_new)
        L_history.append(model.L.detach().cpu().tolist())

        if log_every > 0 and step % log_every == 0:
            L_str = "  ".join(
                f"L{d}={v:.6f}" for d, v in enumerate(model.L.detach().cpu().tolist())
            )
            print(
                f"  step {step:5d} | F = {F_new:.10e} | |grad| = {grad_norm:.4e} | {L_str}"
            )

    if converged and log_every > 0:
        L_str = "  ".join(
            f"L{d}={v:.6f}" for d, v in enumerate(model.L.detach().cpu().tolist())
        )
        print(
            f"  Converged at step {n_steps_completed} | F = {F_history[-1]:.10e} | "
            f"|grad| = {grad_norm_history[-1]:.4e} | {L_str}"
        )

    return {
        "L_initial": L_initial,
        "L_final": model.L.detach().cpu().tolist(),
        "F_initial": F0,
        "F_final": F_history[-1],
        "F_history": F_history,
        "L_history": L_history,
        "grad_norm_history": grad_norm_history,
        "converged": converged,
        "n_steps_completed": n_steps_completed,
    }


# ---------------------------------------------------------------------------
# HDF5 loader
# ---------------------------------------------------------------------------


def load_last_frame(path: str) -> tuple[torch.Tensor, dict]:
    """
    Load the last-frame density field and system parameters from a dynamics HDF5.

    Returns
    -------
    delta_phi : torch.Tensor
        Shape (n_components, *grid_shape), float64.
    params : dict
        All scalar/array parameters needed to reconstruct a
        BlockCopolymerFreeEnergy model.
    """
    with h5py.File(path, "r") as f:
        # last frame of the order-parameter trajectory
        phi_last = torch.from_numpy(f["phi"][-1]).double()  # (n_comp, *grid)

        params = {
            "N": int(f.attrs["N"]),
            "b": float(f.attrs["b"]),
            "phi_bar": float(f.attrs["phi_bar"]),
            "block_fractions": torch.from_numpy(
                np.array(f.attrs["block_fractions"])
            ).double(),
            "grid_shape": tuple(int(x) for x in f.attrs["grid_shape"]),
            "box_lengths": f["box_lengths"][:].tolist(),
            "chi_matrix": torch.from_numpy(f["chi_matrix"][:]).double(),
            "l_ij_matrix": torch.from_numpy(f["l_ij_matrix"][:]).double(),
        }

    return phi_last, params


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Optimize box lengths with a fixed density profile from a dynamics HDF5."
    )
    p.add_argument("h5file", help="Input HDF5 file from run_simulation.py")
    p.add_argument(
        "--n-steps", type=int, default=10000, help="Max gradient steps (default: 10000)"
    )
    p.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Initial line-search step size (default: 0.1)",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Gradient-norm convergence tolerance (default: 1e-5)",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print every N steps, 0=silent (default: 50)",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    # --- load ---
    print(f"Loading last frame from {args.h5file} ...")
    delta_phi, params = load_last_frame(args.h5file)

    print(f"  grid_shape   : {params['grid_shape']}")
    print(f"  n_components : {delta_phi.shape[0]}")
    print(f"  box_lengths  : {params['box_lengths']}")
    print(f"  N            : {params['N']}")
    print()

    # --- reconstruct model with optimize_box=True ---
    model = BlockCopolymerFreeEnergy(
        N=params["N"],
        b=params["b"],
        chi_matrix=params["chi_matrix"],
        l_ij_matrix=params["l_ij_matrix"],
        block_fractions=params["block_fractions"],
        phi_bar=params["phi_bar"],
        grid_shape=params["grid_shape"],
        box_lengths=params["box_lengths"],  # starting point
        optimize_box=True,
    )

    # --- optimize ---
    print("Optimizing box lengths ...")
    result = optimize_box_lengths(
        model=model,
        delta_phi=delta_phi,
        n_steps=args.n_steps,
        lr=args.lr,
        tol=args.tol,
        log_every=args.log_every,
    )

    # --- report ---
    print()
    print("=" * 60)
    print("Result")
    print("=" * 60)
    for d, (L0, Lf) in enumerate(zip(result["L_initial"], result["L_final"])):
        print(f"  L{d}: {L0:.6f}  ->  {Lf:.6f}   ({(Lf / L0 - 1) * 100:+.3f} %)")
    print(f"  F initial : {result['F_initial']:.10e}")
    print(f"  F final   : {result['F_final']:.10e}")
    print(f"  delta_F   : {result['F_final'] - result['F_initial']:.4e}")
    print(f"  Converged : {result['converged']}  ({result['n_steps_completed']} steps)")


if __name__ == "__main__":
    main()
