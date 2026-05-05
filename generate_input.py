import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from simulation_io import SimulationData
from rpa import BlockCopolymerFreeEnergy
from initial_conditions import generate_fourier_mode_initial_conditions
from initial_conditions import BCC_110_STAR as INITIAL_CONDITION

from vis import plot_simulation_result


DTYPE = torch.float64  # change to torch.float32 to test single precision

# Define polymer parameters
N = 100
CHI_AB = 26
chi_matrix = (
    torch.tensor(
        [[0.0, CHI_AB], [CHI_AB, 0.0]],
        dtype=DTYPE,
    )
    * 1
)
f_A = 0.5
f_B = 1 - f_A
l_ij_matrix = torch.zeros((2, 2), dtype=DTYPE)
block_fractions = torch.tensor([f_A, f_B], dtype=DTYPE)

Lx = 5.0
Ly = 5.0
Lz = 5.0

box_lengths = (Lx, Ly, Lz)

grid_shape = (32, 32, 32)

model = BlockCopolymerFreeEnergy(
    N=N,
    chi_matrix=chi_matrix,
    l_ij_matrix=l_ij_matrix,
    block_fractions=block_fractions,
    grid_shape=grid_shape,
    box_lengths=box_lengths,
    dtype=DTYPE,
)

# BCC initial condition: populate the {110} star (9 vectors)
# Box should be cubic for BCC symmetry to be exact on the grid
input_data = generate_fourier_mode_initial_conditions(
    model=model,
    q_vectors=INITIAL_CONDITION,
    amplitude=0.1,
    box_lengths=box_lengths,
    noise_level=0.001,
    save_filename="input_data.h5",
)

# input_data = generate_random_normal_initial_conditions(
#     model=model,
#     box_lengths=box_lengths,
#     amplitude=0.01,
#     save_filename="input_data.h5",
# )
print(input_data.box_lengths)
print("saved to input_data.h5")
input_data.to_hdf5("input_data.h5")
# fig, ax = plt.subplots()
# plot_simulation_result(input_data, fig, ax)
# plt.show()
