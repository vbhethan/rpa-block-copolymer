"""
Generate a lamellar initial condition for a symmetric diblock copolymer.

The density profile follows:
    rho_A(x) = f_A + A * cos(2*pi*x / D)

where D is the lamellar period (= Lx, one full period in the box).
Incompressibility gives delta_phi_B = -delta_phi_A exactly.

Output: input_lam.h5
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math

import numpy as np
import torch

from rpa import BlockCopolymerFreeEnergy, SimulationData

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N = 100
CHI_AB = 26.0
f_A = 0.5
f_B = 1.0 - f_A

Rg = math.sqrt(N * 1.0**2 / 6)

DTYPE = torch.float32

# Lamellar period D sets Lx; Ly, Lz are arbitrary (lamellae are uniform there)
D = 1.0 * Rg  # lamellar period along x
Ly = D
Lz = D
box_lengths = (D, Ly, Lz)

grid_shape = (32, 32, 32)

AMPLITUDE = 0.3  # A in the formula above
NOISE = 0.001  # small symmetry-breaking noise

OUTPUT = "input_lam.h5"

# ---------------------------------------------------------------------------
# Build model (needed to construct SimulationData via from_model)
# ---------------------------------------------------------------------------

chi_matrix = torch.tensor([[0.0, CHI_AB], [CHI_AB, 0.0]], dtype=DTYPE)
l_ij_matrix = torch.zeros((2, 2), dtype=DTYPE)
block_fractions = torch.tensor([f_A, f_B], dtype=DTYPE)

model = BlockCopolymerFreeEnergy(
    N=N,
    chi_matrix=chi_matrix,
    l_ij_matrix=l_ij_matrix,
    block_fractions=block_fractions,
    phi_bar=1.0,
    grid_shape=grid_shape,
    box_lengths=box_lengths,
    optimize_box=False,
    dtype=DTYPE,
)

# ---------------------------------------------------------------------------
# Build lamellar field: delta_phi_A = A * cos(2*pi*x/D)
# ---------------------------------------------------------------------------

Nx, Ny, Nz = grid_shape
# fractional coordinates in [0, 1) along x
x = torch.linspace(0.0, 1.0, Nx + 1, dtype=DTYPE)[:-1]  # shape (Nx,)

# broadcast over y and z: shape (Nx, Ny, Nz)
cos_field = AMPLITUDE * torch.cos(2.0 * math.pi * x).reshape(Nx, 1, 1).expand(
    Nx, Ny, Nz
)

delta_phi = torch.zeros(2, Nx, Ny, Nz, dtype=DTYPE)
delta_phi[0] = cos_field  # delta_phi_A
delta_phi[1] = -cos_field  # delta_phi_B  (incompressibility)

if NOISE > 0.0:
    delta_phi += torch.randn_like(delta_phi) * NOISE

# Double-project to enforce exact constraints (zero mean + incompressibility)
delta_phi = delta_phi - delta_phi.mean(dim=(1, 2, 3), keepdim=True)
delta_phi = delta_phi - delta_phi.mean(dim=0, keepdim=True)

# ---------------------------------------------------------------------------
# Pack into SimulationData and save
# ---------------------------------------------------------------------------

data = SimulationData.from_model(model)
data.phi = delta_phi.numpy()[np.newaxis]  # shape (1, 2, Nx, Ny, Nz)
data.box_lengths = np.array([list(box_lengths)])  # shape (1, 3)

data.to_hdf5(OUTPUT)
print(f"Saved lamellar IC to {OUTPUT}")
print(f"  grid_shape  : {grid_shape}")
print(f"  box_lengths : {box_lengths}  (period D = {D})")
print(f"  amplitude   : {AMPLITUDE}")
print(
    f"  delta_phi range: [{delta_phi.min().item():.4f}, {delta_phi.max().item():.4f}]"
)
