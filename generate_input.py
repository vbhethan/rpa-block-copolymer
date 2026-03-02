import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from simulation_io import SimulationData
from rpa import BlockCopolymerFreeEnergy
from initial_conditions import (
    generate_hexagonal_A_in_BC_matrix,
    generate_random_normal_initial_conditions,
)
from vis import plot_simulation_result


# Define polymer parameters
N = 100
chi_AB = 28
chi_AC = 28
chi_BC = 28
chi_matrix = (
    torch.tensor(
        [
            [0.0, chi_AB, chi_AC],
            [chi_AB, 0.0, chi_BC],
            [chi_AC, chi_BC, 0.0],
        ],
        dtype=torch.float64,
    )
    * 1
)
f_A = 1 / 3
f_B = 1 / 3
f_C = 1 - f_A - f_B
l_ij_matrix = torch.zeros((3, 3), dtype=torch.float64)
l_ij_matrix[0, 2] = f_B * N * 1.0**2
l_ij_matrix[2, 0] = f_B * N * 1.0**2
block_fractions = torch.tensor([f_A, f_B, f_C], dtype=torch.float64)

Lx = 20.0
Ly = Lx * math.sqrt(3)
model = BlockCopolymerFreeEnergy(
    N=N,
    chi_matrix=chi_matrix,
    l_ij_matrix=l_ij_matrix,
    block_fractions=block_fractions,
    grid_shape=(32, 32),
    box_lengths=(Lx, Ly),
)

input_data = generate_hexagonal_A_in_BC_matrix(
    model=model, Lx=Lx, amplitude=0.1, save_filename="input_data.h5", noise_level=0.01
)
# input_data = generate_random_normal_initial_conditions(
#     model=model,
#     box_lengths=(Lx, Ly),
#     amplitude=0.001,
#     save_filename="input_data.h5",
# )
print("saved to input_data.h5")
input_data.to_hdf5("input_data.h5")
fig, ax = plt.subplots()
plot_simulation_result(input_data, fig, ax)
plt.show()
