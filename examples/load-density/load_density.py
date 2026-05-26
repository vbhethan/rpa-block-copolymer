"""
Load phi_A density profiles from samples.pt into a BlockCopolymerFreeEnergy model.

samples.pt: shape (8, 1, 32, 32, 32), values in [-1, 1].
The single channel encodes phi_A; phi_B = 1 - phi_A.
Values map to physical density via:  phi_A = (x + 1) / 2
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from rpa import BlockCopolymerFreeEnergy

# ---------------------------------------------------------------------------
# Physical parameters — adjust to match the system that generated samples.pt
# ---------------------------------------------------------------------------
N = 100
f_A = 0.2
f_B = 1.0 - f_A
CHI_AB = 30.0
phi_bar = 1.0
grid_shape = (32, 32, 32)
box_lengths = (5.0, 5.0, 5.0)
DTYPE = torch.float64

block_fractions = torch.tensor([f_A, f_B], dtype=DTYPE)
chi_matrix = torch.tensor([[0.0, CHI_AB], [CHI_AB, 0.0]], dtype=DTYPE)
l_ij_matrix = torch.zeros((2, 2), dtype=DTYPE)

# ---------------------------------------------------------------------------
# Load samples and convert from [-1, 1] to physical phi_A in [0, 1]
# ---------------------------------------------------------------------------
samples = torch.load(
    "./examples/load-density/samples.pt", weights_only=False
)  # (8, 1, 32, 32, 32)
FRAME = 0

phi_A = (samples[FRAME, 0] + 1.0) / 2.0  # (32, 32, 32)
phi_B = 1.0 - phi_A
phi = torch.stack([phi_A, phi_B], dim=0)  # (2, 32, 32, 32)

# ---------------------------------------------------------------------------
# Build model and load the density profile
# ---------------------------------------------------------------------------
model = BlockCopolymerFreeEnergy(
    N=N,
    block_fractions=block_fractions,
    chi_matrix=chi_matrix,
    l_ij_matrix=l_ij_matrix,
    phi_bar=phi_bar,
    grid_shape=grid_shape,
    box_lengths=box_lengths,
    dtype=DTYPE,
)

model.set_density(phi)

F = model()
print(f"Free energy (frame {FRAME}): {F.item():.6f}")
print(f"Constraints: {model.check_constraints()}")
