import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import torch
from rpa import BlockCopolymerFreeEnergy
import matplotlib.pyplot as plt


def generate_hexagonal_A_in_BC_matrix(
    grid_shape,
    block_fractions: torch.Tensor,
    Lx: float,
    Ly: float,
    amplitude: float = 0.1,
    n_periods_x: int = 1,
    n_periods_y: int = 1,
) -> torch.Tensor:
    """
    Generate hexagonally packed A cylinders in a homogeneous B+C matrix.

    Component A is concentrated at hexagonal lattice sites (cylinder cores).
    Components B and C are depleted at those sites proportionally to their
    relative matrix fractions f_B/(f_B+f_C) and f_C/(f_B+f_C), so the
    B/C composition ratio is spatially uniform in the matrix.

    Incompressibility (delta_phi_A + delta_phi_B + delta_phi_C = 0) is
    satisfied by construction.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        (Nx, Ny) number of grid points.
    block_fractions : torch.Tensor
        Volume fractions [f_A, f_B, f_C], must have 3 components.
    Lx, Ly : float
        Box dimensions. Ideally Ly/Lx = (n_periods_y/n_periods_x)*sqrt(3).
    amplitude : float
        Peak amplitude of δφ_A fluctuations (default 0.1).
    n_periods_x : int
        Number of hexagonal unit cells in the x direction.
    n_periods_y : int
        Number of hexagonal unit cells in the y direction.

    Returns
    -------
    delta_phi : torch.Tensor
        Order parameter field of shape (3, Nx, Ny), satisfying zero mean
        and local incompressibility.
    """
    Nx, Ny = grid_shape
    n_components = len(block_fractions)
    if n_components != 3:
        raise ValueError(
            f"Hexagonal init requires exactly 3 components, got {n_components}"
        )

    expected_ratio = (n_periods_y / n_periods_x) * math.sqrt(3)
    actual_ratio = Ly / Lx
    if abs(actual_ratio - expected_ratio) / expected_ratio > 0.05:
        print(
            f"Warning: Ly/Lx = {actual_ratio:.4f}, but hexagonal commensurability "
            f"requires Ly/Lx ≈ {expected_ratio:.4f} for n_periods=({n_periods_x},{n_periods_y}). "
            f"Pattern may not tile smoothly."
        )

    # Coordinate grids (exclude endpoint for periodicity)
    x = torch.linspace(0, Lx, Nx + 1, dtype=torch.float64)[:-1]
    y = torch.linspace(0, Ly, Ny + 1, dtype=torch.float64)[:-1]
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Hexagonal lattice constant from box dimensions
    a = Lx / n_periods_x

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
    f_B = block_fractions[1].item()
    f_C = block_fractions[2].item()
    f_matrix = f_B + f_C

    delta_phi = torch.zeros(n_components, Nx, Ny, dtype=torch.float64)
    delta_phi[0] = phi_cyl  # A: cylinder cores
    delta_phi[1] = -(f_B / f_matrix) * phi_cyl  # B: proportional depletion
    delta_phi[2] = -(f_C / f_matrix) * phi_cyl  # C: proportional depletion

    # Project onto constraint manifold (zero mean + incompressibility)
    delta_phi -= delta_phi.mean(dim=(1, 2), keepdim=True)
    delta_phi -= delta_phi.mean(dim=0, keepdim=True)

    return delta_phi


def generate_initial_hexagonal_ordered_structure(
    grid_shape: tuple[int, int],
    block_fractions: torch.Tensor,
    Lx: float,
    Ly: float,
    amplitude: float = 0.1,
    n_periods_x: int = 1,
    n_periods_y: int = 1,
) -> torch.Tensor:
    """
    Generate an initial hexagonal ordered structure for 3 components.

    Uses plane-wave superposition with the first star of the hexagonal
    reciprocal lattice, with 120-degree phase shifts between components.

    For the pattern to be commensurate with periodic boundary conditions,
    the box aspect ratio should satisfy Ly/Lx = (n_periods_y / n_periods_x) * sqrt(3).

    Parameters
    ----------
    grid_shape : tuple[int, int]
        (Nx, Ny) number of grid points.
    block_fractions : torch.Tensor
        Volume fractions [f_A, f_B, f_C], must have 3 components.
    Lx, Ly : float
        Box dimensions. Ideally Ly/Lx = (n_periods_y/n_periods_x)*sqrt(3).
    amplitude : float
        Peak amplitude of the initial fluctuations (default 0.1).
    n_periods_x : int
        Number of hexagonal unit cells in the x direction.
    n_periods_y : int
        Number of hexagonal unit cells in the y direction.

    Returns
    -------
    delta_phi : torch.Tensor
        Order parameter field of shape (3, Nx, Ny), satisfying zero mean
        and local incompressibility.
    """
    Nx, Ny = grid_shape
    n_components = len(block_fractions)
    if n_components != 3:
        raise ValueError(
            f"Hexagonal init requires exactly 3 components, got {n_components}"
        )

    expected_ratio = (n_periods_y / n_periods_x) * math.sqrt(3)
    actual_ratio = Ly / Lx
    if abs(actual_ratio - expected_ratio) / expected_ratio > 0.05:
        print(
            f"Warning: Ly/Lx = {actual_ratio:.4f}, but hexagonal commensurability "
            f"requires Ly/Lx ≈ {expected_ratio:.4f} for n_periods=({n_periods_x},{n_periods_y}). "
            f"Pattern may not tile smoothly."
        )

    # Coordinate grids (exclude endpoint for periodicity)
    x = torch.linspace(0, Lx, Nx + 1, dtype=torch.float64)[:-1]
    y = torch.linspace(0, Ly, Ny + 1, dtype=torch.float64)[:-1]
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Hexagonal lattice constant from box dimensions
    a = Lx / n_periods_x

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

    # Build order parameter with 120-degree phase shifts between components
    delta_phi = torch.zeros(n_components, Nx, Ny, dtype=torch.float64)
    for m in range(n_components):
        phase_shift = 2.0 * math.pi * m / 3.0
        for j in range(3):
            delta_phi[m] += torch.cos(G[j, 0] * X + G[j, 1] * Y - phase_shift)

    # Normalize peak amplitude
    delta_phi = amplitude * delta_phi / delta_phi.abs().max()

    # Project onto constraint manifold (zero mean + incompressibility)
    delta_phi -= delta_phi.mean(dim=(1, 2), keepdim=True)
    delta_phi -= delta_phi.mean(dim=0, keepdim=True)

    return delta_phi


if __name__ == "__main__":
    # Quick visualization test
    Lx = 10.0
    Ly = Lx * math.sqrt(3)
    Nx, Ny = 128, 128
    f = torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3])

    delta_phi = generate_hexagonal_A_in_BC_matrix(
        grid_shape=(Nx, Ny),
        block_fractions=f,
        Lx=Lx,
        Ly=Ly,
        amplitude=0.1,
    )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    labels = ["A", "B", "C"]
    for i in range(3):
        im = axes[i].imshow(
            delta_phi[i].numpy().T,
            origin="lower",
            extent=[0, Lx, 0, Ly],
            cmap="RdBu_r",
        )
        axes[i].set_title(f"δφ_{labels[i]}")
        axes[i].set_aspect("equal")
        plt.colorbar(im, ax=axes[i], shrink=0.7)

    # Dominant component map
    rho = delta_phi + f.view(3, 1, 1)
    dominant = rho.argmax(dim=0).numpy().T
    axes[3].imshow(
        dominant, origin="lower", extent=[0, Lx, 0, Ly], cmap="Set1", vmin=0, vmax=2
    )
    axes[3].set_title("Dominant component")
    axes[3].set_aspect("equal")

    plt.tight_layout()
    plt.savefig("hex_init_test.png", dpi=150)
    plt.show()
    print("Shape:", delta_phi.shape)
    print(
        "Sum over components (should be ~0):", delta_phi.sum(dim=0).abs().max().item()
    )
    print("Mean per component:", [delta_phi[i].mean().item() for i in range(3)])
