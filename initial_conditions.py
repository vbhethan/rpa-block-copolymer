"""
Methods that generate initial conditions for block copolymer systems as starting points for optimization

Use the simulation_io.SimulationData class to write initial conditions to an HDF5 file which can be used to initialize a model and simulation
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from rpa import BlockCopolymerFreeEnergy
from simulation_io import SimulationData
from vis import plot_simulation_result
import math
import numpy as np
import matplotlib.pyplot as plt


def generate_random_normal_initial_conditions(
    model: BlockCopolymerFreeEnergy,
    amplitude: float = 0.1,
    box_lengths: tuple[float, float] = (10.0, 10.0 * math.sqrt(3)),
    grid_shape: tuple[int, int] = (32, 32),
    save_filename: str = None,
) -> SimulationData:
    """
    Generate uniform random initial conditions for a block copolymer system
    """
    n_components = model.n_components
    grid_shape = model.grid_shape
    phi = amplitude * torch.randn(n_components, *grid_shape)
    data = SimulationData.from_model(model)
    data.box_lengths = np.array([[box_lengths[0], box_lengths[1]]])
    data.phi = np.array([phi])
    if save_filename is not None:
        data.to_hdf5(save_filename)
    return data


def generate_hexagonal_A_in_BC_matrix(
    model: BlockCopolymerFreeEnergy,
    amplitude: float = 0.1,
    Lx: float = 10.0,
    n_periods_x: int = 1,
    n_periods_y: int = 1,
    save_filename: str = None,
    noise_level: float = 0.0,
) -> SimulationData:
    """
    Generate hexagonally packed A cylinders in a homogeneous B+C matrix.

    Component A is concentrated at hexagonal lattice sites (cylinder cores).
    Components B and C are depleted at those sites proportionally to their
    relative matrix fractions f_B/(f_B+f_C) and f_C/(f_B+f_C), so the
    B/C composition ratio is spatially uniform in the matrix.

    Incompressibility (delta_phi_A + delta_phi_B + delta_phi_C = 0) is
    satisfied by construction.

    For now, only for 3 component system

    Parameters
    ----------
    model: BlockCopolymerFreeEnergy
        Model to generate initial conditions for.
    amplitude : float
        Peak amplitude of δφ_A fluctuations (default 0.1).
    n_periods_x : int
        Number of hexagonal unit cells in the x direction.
    n_periods_y : int
        Number of hexagonal unit cells in the y direction.

    Returns
    -------
    SimulationData
        Simulation data containing the initial conditions.
    """
    Nx, Ny = model.grid_shape
    n_components = model.n_components
    if n_components != 3:
        raise ValueError(
            f"Hexagonal init requires exactly 3 components, got {n_components}"
        )

    Ly = Lx * math.sqrt(3)
    box_lengths = (Lx, Ly)
    # Coordinate grids (exclude endpoint for periodicity)
    x = torch.linspace(0, model.L[0], Nx + 1, dtype=torch.float64)[:-1]
    y = torch.linspace(0, model.L[1], Ny + 1, dtype=torch.float64)[:-1]
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Hexagonal lattice constant from box dimensions
    a = model.L[0] / n_periods_x

    # First star of the hexagonal reciprocal lattice (3 independent vectors)
    #   G1 = (2pi/a)(1, -1/sqrt(3))
    #   G2 = (2pi/a)(0,  2/sqrt(3))
    #   G3 = (2pi/a)(1,  1/sqrt(3))    [= G1 + G2]
    k0 = 2.0 * math.pi / a
    sqrt3 = math.sqrt(3.0)
    G = (
        torch.tensor(
            [
                [1.0, -1.0 / sqrt3],
                [0.0, 2.0 / sqrt3],
                [1.0, 1.0 / sqrt3],
            ],
            dtype=torch.float64,
        )
        * k0
    )

    # Single cylinder field: peaks at hexagonal lattice sites
    phi_cyl = torch.zeros(Nx, Ny, dtype=torch.float64)
    for j in range(3):
        phi_cyl += torch.cos(G[j, 0] * X + G[j, 1] * Y)

    # Normalize so the peak fluctuation has the desired amplitude
    phi_cyl = amplitude * phi_cyl / phi_cyl.abs().max()

    # A is enriched at cylinder sites; B, C are depleted proportionally
    # to their relative matrix fractions so the matrix is homogeneous.
    f_B = model.f_vec[1].item()
    f_C = model.f_vec[2].item()
    f_matrix = f_B + f_C

    model.box_lengths = box_lengths

    delta_phi = torch.zeros(n_components, Nx, Ny, dtype=torch.float64)
    delta_phi[0] = phi_cyl  # A: cylinder cores
    delta_phi[1] = -(f_B / f_matrix) * phi_cyl  # B: proportional depletion
    delta_phi[2] = -(f_C / f_matrix) * phi_cyl  # C: proportional depletion

    # Project onto constraint manifold (zero mean + incompressibility)
    delta_phi -= delta_phi.mean(dim=(1, 2), keepdim=True)
    delta_phi -= delta_phi.mean(dim=0, keepdim=True)
    if noise_level > 0.0:
        delta_phi += torch.randn_like(delta_phi) * noise_level

    data = SimulationData.from_model(model)
    data.phi = np.array([delta_phi])
    data.box_lengths = np.array([box_lengths])
    if save_filename is not None:
        data.to_hdf5(save_filename)
    return data


if __name__ == "__main__":
    N = 100
    f_A = 1 / 3
    f_B = 1 / 3
    f_C = 1 - f_A - f_B
    grid_shape = (32, 32)
    Lx = 20.0
    Ly = Lx * math.sqrt(3)
    box_lengths = (Lx, Ly)
    model = BlockCopolymerFreeEnergy(
        N=N,
        chi_matrix=torch.tensor(
            [[0.0, 26.0, 26.0], [26.0, 0.0, 26.0], [26.0, 26.0, 0.0]]
        ),
        l_ij_matrix=torch.tensor(
            [
                [0.0, 0.0, f_B * N * 1.0**2],
                [0.0, 0.0, 0.0],
                [f_B * N * 1.0**2, 0.0, 0.0],
            ]
        ),
        block_fractions=torch.tensor([f_A, f_B, f_C]),
        grid_shape=(32, 32),
        box_lengths=box_lengths,
    )
    simulation_data = generate_hexagonal_A_in_BC_matrix(model)
    print(simulation_data.phi.shape)
    print(simulation_data.box_lengths)
    fig, ax = plt.subplots()
    plot_simulation_result(simulation_data, fig, ax)
    plt.show()
