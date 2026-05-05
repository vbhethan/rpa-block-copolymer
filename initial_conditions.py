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
    box_lengths: tuple[float, ...] = (10.0, 10.0 * math.sqrt(3)),
    save_filename: str = None,
) -> SimulationData:
    """
    Generate uniform random initial conditions for a block copolymer system
    """
    n_components = model.n_components
    grid_shape = model.grid_shape
    phi = amplitude * torch.randn(n_components, *grid_shape)
    data = SimulationData.from_model(model)
    data.box_lengths = np.array([list(box_lengths)])
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
    dtype: torch.dtype = None,
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

    _dtype = dtype if dtype is not None else model.real_dtype
    Ly = Lx * math.sqrt(3)
    box_lengths = (Lx, Ly)
    # Coordinate grids (exclude endpoint for periodicity)
    x = torch.linspace(0, model.L[0], Nx + 1, dtype=_dtype)[:-1]
    y = torch.linspace(0, model.L[1], Ny + 1, dtype=_dtype)[:-1]
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
            dtype=_dtype,
        )
        * k0
    )

    # Single cylinder field: peaks at hexagonal lattice sites
    phi_cyl = torch.zeros(Nx, Ny, dtype=_dtype)
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

    delta_phi = torch.zeros(n_components, Nx, Ny, dtype=_dtype)
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


def generate_fourier_mode_initial_conditions(
    model: BlockCopolymerFreeEnergy,
    q_vectors: list[tuple[int, int, int]],
    amplitude: float = 0.1,
    box_lengths: tuple[float, ...] = None,
    noise_level: float = 0.0,
    save_filename: str = None,
    dtype: torch.dtype = None,
) -> SimulationData:
    """
    Generate initial conditions by populating specified Fourier modes.

    Sets delta_phi[0] = amplitude * sum_G cos(G·r) for the specified reciprocal
    lattice vectors G (given as integer Miller indices), then derives the
    remaining components to satisfy incompressibility.

    Parameters
    ----------
    model : BlockCopolymerFreeEnergy
        Model defining grid_shape, n_components, f_vec.
    q_vectors : list of (h, k, l) tuples
        Integer Miller indices of the modes to populate. All symmetry-related
        vectors in a star should be listed explicitly (e.g. all 6 {110} vectors
        for BCC).
    amplitude : float
        Peak amplitude of the first-component fluctuation.
    box_lengths : tuple of float, optional
        Box lengths. Defaults to model.L.
    noise_level : float
        Optional small random noise added after construction.
    save_filename : str, optional
        If given, saves the SimulationData to this HDF5 path.

    Returns
    -------
    SimulationData
    """
    _dtype = dtype if dtype is not None else model.real_dtype
    grid_shape = model.grid_shape
    ndim = model.ndim
    n_components = model.n_components

    if box_lengths is None:
        box_lengths = tuple(model.L.tolist())

    # Build real-space coordinate grids (exclude endpoint for periodicity)
    axes = [
        torch.linspace(0, 1, grid_shape[d] + 1, dtype=_dtype)[:-1] for d in range(ndim)
    ]
    grids = torch.meshgrid(*axes, indexing="ij")  # each: (*grid_shape)

    # Sum cosines over all specified Miller-index vectors
    phi_field = torch.zeros(grid_shape, dtype=_dtype)
    for hkl in q_vectors:
        phase = sum(
            hkl[d] * grids[d] for d in range(ndim)
        )  # fractional coords → phase in [0,1)
        phi_field = phi_field + torch.cos(2 * math.pi * phase)

    # Normalize to requested amplitude
    peak = phi_field.abs().max()
    if peak > 0:
        phi_field = amplitude * phi_field / peak

    # Build delta_phi for all components satisfying incompressibility:
    # delta_phi[i] = -(f_i / (1 - f_0)) * phi_field  for i > 0
    f = model.f_vec  # (n_components,)
    delta_phi = torch.zeros(n_components, *grid_shape, dtype=_dtype)
    delta_phi[0] = phi_field
    f_rest = 1.0 - f[0].item()
    for i in range(1, n_components):
        delta_phi[i] = -(f[i].item() / f_rest) * phi_field

    # Enforce zero mean + incompressibility via double projection
    delta_phi = delta_phi - delta_phi.mean(dim=tuple(range(1, ndim + 1)), keepdim=True)
    delta_phi = delta_phi - delta_phi.mean(dim=0, keepdim=True)

    if noise_level > 0.0:
        delta_phi = delta_phi + torch.randn_like(delta_phi) * noise_level

    data = SimulationData.from_model(model)
    data.phi = np.array([delta_phi.numpy()])
    data.box_lengths = np.array([list(box_lengths)])

    if save_filename is not None:
        data.to_hdf5(save_filename)
    return data


# BCC {110} star: 6 vectors ±permutations of (1,1,0)
BCC_110_STAR = [
    (1, 1, 0),
    (-1, 1, 0),
    (1, -1, 0),
    (1, 0, 1),
    (-1, 0, 1),
    (1, 0, -1),
    (0, 1, 1),
    (0, -1, 1),
    (0, 1, -1),
]

# BCC double-star {110} + {200}: adds the 3 (200)-type modes
BCC_110_200_STAR = BCC_110_STAR + [(2, 0, 0), (0, 2, 0), (0, 0, 2)]

# FCC {111} + {200} star (Fm3m symmetry seeds)
FCC_111_200_STAR = [
    (1, 1, 1),
    (-1, 1, 1),
    (1, -1, 1),
    (1, 1, -1),
    (2, 0, 0),
    (0, 2, 0),
    (0, 0, 2),
]

# Lamellar along z
LAMELLAR_Z = [(0, 0, 1)]


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
