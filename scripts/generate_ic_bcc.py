"""
Generate a BCC initial condition for a strongly asymmetric diblock copolymer.

A-minority spheres on a BCC lattice in a B matrix, using a cubic unit cell.
The density profile populates the full {110} star:

    rho_A = f_A + A * sum_{i<j} [ cos(q*(x_i + x_j)) + cos(q*(x_i - x_j)) ]

where (x_1, x_2, x_3) = (x, y, z) and q = 2*pi/L.

Expanding the sum over the 3 coordinate pairs (i<j) gives 6 unique cosines
that together equal 2[cos(qx)cos(qy) + cos(qx)cos(qz) + cos(qy)cos(qz)],
placing density maxima at the BCC sites (corners + body center).

f_A = 0.25, chi*N = 40 places the system well inside the BCC phase
(ODT from disorder → BCC is around chi*N ≈ 13–15 at this composition).

Incompressibility gives delta_phi_B = -delta_phi_A.

Output: input_bcc.h5
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math

import numpy as np
import torch

from rpa import BlockCopolymerFreeEnergy, SimulationData

# ---------------------------------------------------------------------------
# Parameters  (f_A=0.25, chi*N=40 → well inside BCC)
# ---------------------------------------------------------------------------

N = 100
b = 1.0
f_A = 0.25
f_B = 1.0 - f_A
CHI_N = 40.0
CHI_AB = CHI_N / N

DTYPE = torch.float32

Rg = math.sqrt(N * b**2 / 6)

# BCC requires a cubic box: Lx = Ly = Lz = L (the BCC lattice constant).
# Equilibrium a ~ 2–3 Rg for chi*N = 40; use 2.5 Rg as initial guess.
L = 2.5 * Rg
box_lengths = (L, L, L)

grid_shape = (32, 32, 32)

AMPLITUDE = 0.3    # A in the formula above (applied after normalizing peak to 1)
NOISE = 0.001

OUTPUT = "input_bcc.h5"

# ---------------------------------------------------------------------------
# Build model
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
# Build BCC field: sum over coordinate pairs (i < j), each pair contributing
#   cos(2*pi*(xf_i + xf_j)) + cos(2*pi*(xf_i - xf_j))
#
# With fractional coordinates xf = x/L in [0,1):
#   pair (x,y): 2*cos(2*pi*xf)*cos(2*pi*yf)
#   pair (x,z): 2*cos(2*pi*xf)*cos(2*pi*zf)
#   pair (y,z): 2*cos(2*pi*yf)*cos(2*pi*zf)
#
# Total peak = 6 at (0,0,0) and (0.5,0.5,0.5) — the BCC sites.
# ---------------------------------------------------------------------------

Nx, Ny, Nz = grid_shape

xf = torch.linspace(0.0, 1.0, Nx + 1, dtype=DTYPE)[:-1]   # (Nx,)
yf = torch.linspace(0.0, 1.0, Ny + 1, dtype=DTYPE)[:-1]   # (Ny,)
zf = torch.linspace(0.0, 1.0, Nz + 1, dtype=DTYPE)[:-1]   # (Nz,)

cx = torch.cos(2.0 * math.pi * xf)   # (Nx,)
cy = torch.cos(2.0 * math.pi * yf)   # (Ny,)
cz = torch.cos(2.0 * math.pi * zf)   # (Nz,)

# Outer products broadcast to (Nx, Ny, Nz)
bcc_field = (
    2.0 * cx.reshape(Nx, 1, 1) * cy.reshape(1, Ny, 1)    # pair (x,y)
    + 2.0 * cx.reshape(Nx, 1, 1) * cz.reshape(1, 1, Nz)  # pair (x,z)
    + 2.0 * cy.reshape(1, Ny, 1) * cz.reshape(1, 1, Nz)  # pair (y,z)
)

# Normalize peak to 1, then scale to AMPLITUDE
bcc_field = AMPLITUDE * bcc_field / bcc_field.abs().max()

delta_phi = torch.zeros(2, Nx, Ny, Nz, dtype=DTYPE)
delta_phi[0] = bcc_field    # delta_phi_A: A-rich sphere sites
delta_phi[1] = -bcc_field   # delta_phi_B: incompressibility

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
print(f"Saved BCC IC to {OUTPUT}")
print(f"  grid_shape   : {grid_shape}")
print(f"  f_A          : {f_A},  chi*N = {CHI_N}")
print(f"  Rg           : {Rg:.3f}")
print(f"  L            : {L:.3f}  ({L/Rg:.2f} Rg)  [BCC lattice constant]")
print(f"  amplitude    : {AMPLITUDE}")
print(f"  delta_phi range: [{delta_phi.min().item():.4f}, {delta_phi.max().item():.4f}]")
