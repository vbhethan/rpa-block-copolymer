"""
Generate a hexagonal initial condition for an asymmetric diblock copolymer.

A-minority cylinders aligned along z in a B matrix, using a rectangular unit
cell with Ly = sqrt(3) * Lx. The density profile follows:

    rho_A = f_A + A * [2*cos(2*pi*x/Lx)*cos(2*pi*y/Ly) + cos(4*pi*y/Ly)]

This is the sum of the 3 first-star reciprocal-lattice modes of the 2D
hexagonal lattice expressed in the rectangular unit cell. It places two
A-rich cylinder cores per cell at (x,y) = (0,0) and (Lx/2, Ly/2).

f_A = 0.35, chi*N = 35 places the system well inside the hexagonal phase
(the ODT from disorder → HEX is around chi*N ≈ 14–16 at this composition;
the HEX→LAM boundary is near f_A ≈ 0.40).

Incompressibility gives delta_phi_B = -delta_phi_A.

Output: input_hex.h5
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math

import numpy as np
import torch

from rpa import BlockCopolymerFreeEnergy, SimulationData

# ---------------------------------------------------------------------------
# Parameters  (f_A=0.35, chi*N=35 → well inside HEX)
# ---------------------------------------------------------------------------

N = 100
b = 1.0
f_A = 0.35
f_B = 1.0 - f_A
CHI_N = 35.0
CHI_AB = CHI_N / N

DTYPE = torch.float32

Rg = math.sqrt(N * b**2 / 6)

# Rectangular unit cell for hexagonal symmetry: Ly = sqrt(3) * Lx
# Lx is the hex lattice constant a (cylinder-to-cylinder distance).
# Equilibrium a ~ 2.5–3 Rg for chi*N = 35; use 2.5 Rg as initial guess.
Lx = 2.5 * Rg
Ly = math.sqrt(3) * Lx   # enforced by hexagonal symmetry
Lz = Lx                   # cylinders uniform along z; optimizer will relax

box_lengths = (Lx, Ly, Lz)

grid_shape = (32, 32, 32)

AMPLITUDE = 0.3    # A in the formula above (applied after normalizing peak to 1)
NOISE = 0.001

OUTPUT = "input_hex.h5"

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

chi_matrix = torch.tensor([[0.0, CHI_AB], [CHI_AB, 0.0]], dtype=DTYPE)
l_ij_matrix = torch.zeros((2, 2), dtype=DTYPE)
block_fractions = torch.tensor([f_A, f_B], dtype=DTYPE)

model = BlockCopolymerFreeEnergy(
    N=N,
    b=b,
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
# Build hexagonal field in the rectangular unit cell.
#
# Using fractional coordinates xf = x/Lx, yf = y/Ly ∈ [0, 1):
#
#   field = 2*cos(2π*xf)*cos(2π*yf) + cos(4π*yf)
#
# Peak = 3 at (xf,yf) = (0,0) and (0.5, 0.5)  →  two cylinders per cell
# Min  = -1.5  →  B-rich inter-cylinder region
# ---------------------------------------------------------------------------

Nx, Ny, Nz = grid_shape

xf = torch.linspace(0.0, 1.0, Nx + 1, dtype=DTYPE)[:-1]   # (Nx,)
yf = torch.linspace(0.0, 1.0, Ny + 1, dtype=DTYPE)[:-1]   # (Ny,)

# (Nx, Ny) grid via broadcasting
X = xf.reshape(Nx, 1).expand(Nx, Ny)
Y = yf.reshape(1, Ny).expand(Nx, Ny)

hex_2d = (
    2.0 * torch.cos(2.0 * math.pi * X) * torch.cos(2.0 * math.pi * Y)
    + torch.cos(4.0 * math.pi * Y)
)

# Normalize peak to 1, then scale to AMPLITUDE
hex_2d = AMPLITUDE * hex_2d / hex_2d.abs().max()

# Broadcast uniformly along z (cylinders are z-invariant)
hex_3d = hex_2d.unsqueeze(2).expand(Nx, Ny, Nz)   # (Nx, Ny, Nz)

delta_phi = torch.zeros(2, Nx, Ny, Nz, dtype=DTYPE)
delta_phi[0] = hex_3d    # delta_phi_A: A-rich cylinder cores
delta_phi[1] = -hex_3d   # delta_phi_B: incompressibility

if NOISE > 0.0:
    delta_phi += torch.randn_like(delta_phi) * NOISE

# Double-project: zero mean per component + incompressibility
delta_phi = delta_phi - delta_phi.mean(dim=(1, 2, 3), keepdim=True)
delta_phi = delta_phi - delta_phi.mean(dim=0, keepdim=True)

# ---------------------------------------------------------------------------
# Pack and save
# ---------------------------------------------------------------------------

data = SimulationData.from_model(model)
data.phi = delta_phi.numpy()[np.newaxis]            # (1, 2, Nx, Ny, Nz)
data.box_lengths = np.array([list(box_lengths)])    # (1, 3)

data.to_hdf5(OUTPUT)
print(f"Saved hexagonal IC to {OUTPUT}")
print(f"  grid_shape   : {grid_shape}")
print(f"  f_A          : {f_A},  chi*N = {CHI_N}")
print(f"  Rg           : {Rg:.3f}")
print(f"  Lx           : {Lx:.3f}  ({Lx/Rg:.2f} Rg)  [hex lattice constant a]")
print(f"  Ly           : {Ly:.3f}  ({Ly/Rg:.2f} Rg)  = sqrt(3)*Lx")
print(f"  Lz           : {Lz:.3f}  ({Lz/Rg:.2f} Rg)  [cylinder axis]")
print(f"  amplitude    : {AMPLITUDE}")
print(f"  delta_phi range: [{delta_phi.min().item():.4f}, {delta_phi.max().item():.4f}]")
