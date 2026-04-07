"""
Integration test: symmetric diblock copolymer lamellar structure -> PSCF I/O files.

System: AB symmetric diblock, f_A = f_B = 0.5, chiN = 11.0 (just above ODT ~10.5).
Morphology: lamellar, density varies along x only.
Output:  param.txt, c.rf, commands.txt written to this directory.
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Allow imports from the repo root regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math
import numpy as np
import torch
from rpa import BlockCopolymerFreeEnergy
from simulation_io import SimulationData
from pscf_io import write_param_file, write_C_RGRID_from_array, write_command_file

HERE = os.path.dirname(os.path.abspath(__file__))


def make_lamellar_model() -> BlockCopolymerFreeEnergy:
    """
    Build a symmetric AB diblock model with a single-mode lamellar delta_phi.

    chiN = 11.0  (above the mean-field ODT at ~10.5 for f=0.5)
    Lamellar period set to d* ~ 3.9 * Rg (Rg = b*sqrt(N/6) ~ 4.08 for N=100, b=1).
    Grid: (32, 4, 4) -- structure varies only along x; y/z are homogeneous.
    """
    N = 100
    b = 1.0
    f_A = 0.5

    chi = 11.0 / N  # chiN = 11.0

    chi_matrix = torch.tensor(
        [[0.0, chi], [chi, 0.0]], dtype=torch.float64
    )
    # l_ij_matrix: for a linear AB diblock the off-diagonal entry encodes the
    # center-to-center contour separation.  For f_A = f_B = 0.5, the midpoints
    # sit at s = 0.25 and s = 0.75 (normalized), giving a separation of 0.5.
    # In the distance factor exp(-l_ij * k^2 / 6), l_ij has units of N*b^2, so
    # l_AB = 0.5 * N * b^2 = 50.
    l_ij_matrix = torch.tensor(
        [[0.0, 50.0], [50.0, 0.0]], dtype=torch.float64
    )
    block_fractions = torch.tensor([f_A, 1.0 - f_A], dtype=torch.float64)

    # Lamellar period d* ~ 3.9 * Rg; Rg = sqrt(N * b^2 / 6) ~ 4.08
    Rg = math.sqrt(N * b**2 / 6)
    d_star = 3.9 * Rg  # ~ 15.9

    grid_shape = (32, 4, 4)
    # Box: one lamellar period in x, small transverse dimensions
    box_lengths = (d_star, 4.0 * b, 4.0 * b)

    model = BlockCopolymerFreeEnergy(
        N=N,
        b=b,
        block_fractions=block_fractions,
        chi_matrix=chi_matrix,
        l_ij_matrix=l_ij_matrix,
        phi_bar=1.0,
        grid_shape=grid_shape,
        box_lengths=box_lengths,
        init_amplitude=0.0,  # start with zero fluctuation, set manually below
    )

    # Manually impose a single-mode lamellar profile along x
    # delta_phi_A(x) = A * cos(2*pi * ix / Nx),  delta_phi_B = -delta_phi_A
    amplitude = 0.2
    Nx = grid_shape[0]
    ix = torch.arange(Nx, dtype=torch.float64)
    cos_profile = amplitude * torch.cos(2.0 * math.pi * ix / Nx)  # (Nx,)

    delta_phi = torch.zeros(2, *grid_shape, dtype=torch.float64)
    delta_phi[0] = cos_profile[:, None, None]   # A component
    delta_phi[1] = -cos_profile[:, None, None]  # B component (incompressibility)

    model.delta_phi = torch.nn.Parameter(delta_phi)
    return model


def make_simulation_data(model: BlockCopolymerFreeEnergy) -> SimulationData:
    """
    Wrap a single snapshot of *model* into a SimulationData trajectory frame.

    SimulationData.phi stores delta_phi (order parameter fluctuation),
    matching what the PGD optimizer records.
    """
    with torch.no_grad():
        F_val = model().item()
        delta_phi_np = model.get_order_parameters().cpu().numpy()  # (n, *grid)
        box_np = model.L.cpu().numpy()                             # (ndim,)

    sim = SimulationData.from_model(model)
    # Manually populate one trajectory frame
    sim.phi = delta_phi_np[np.newaxis, ...]             # (1, n, *grid)
    sim.F = np.array([F_val])                           # (1,)
    sim.box_lengths = box_np[np.newaxis, :]             # (1, ndim)
    return sim


def test_write_pscf_files() -> None:
    model = make_lamellar_model()
    sim = make_simulation_data(model)

    param_path = os.path.join(HERE, "param.txt")
    c_path = os.path.join(HERE, "c.rf")
    cmd_path = os.path.join(HERE, "commands.txt")

    # --- param file ---
    write_param_file(param_path, sim)

    # --- concentration field ---
    # SimulationData.phi stores delta_phi; convert to concentrations for PSCF.
    # phi_i(r) = f_i * phi_bar + delta_phi_i(r)
    delta_phi_last = sim.phi[-1]                                    # (n, Nx, Ny, Nz)
    f = sim.block_fractions[:, np.newaxis, np.newaxis, np.newaxis]  # (n, 1, 1, 1)
    concentrations = delta_phi_last + f * sim.phi_bar               # (n, Nx, Ny, Nz)

    write_C_RGRID_from_array(
        c_path,
        concentrations,
        sim.grid_shape,
        sim.box_lengths[-1].tolist(),
    )

    # --- command file ---
    write_command_file(cmd_path)

    # --- verify ---
    for path in [param_path, c_path, cmd_path]:
        assert os.path.exists(path), f"File not written: {path}"
        assert os.path.getsize(path) > 0, f"File is empty: {path}"

    # Sanity: concentrations should sum to phi_bar everywhere
    assert np.allclose(concentrations.sum(axis=0), sim.phi_bar, atol=1e-10), \
        "Concentrations do not satisfy incompressibility"

    print(f"  param file : {param_path}")
    print(f"  c-field    : {c_path}")
    print(f"  commands   : {cmd_path}")
    print(f"  F          = {sim.F[-1]:.6e}")
    print(f"  box_lengths= {sim.box_lengths[-1].tolist()}")
    print("test_write_pscf_files passed")


if __name__ == "__main__":
    test_write_pscf_files()
    print("\nAll tests passed!")
