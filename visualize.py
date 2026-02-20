import matplotlib.pyplot as plt
import numpy as np
import h5py

sim_file = "simulation.h5"

with h5py.File(sim_file, "r") as f:
    print(f.keys())
    block_fractions = f.attrs["block_fractions"]
    box_lengths = f["box_lengths"][:]
    t = f["t"][:]
    F = f["F"][:]
    phi = f["phi"][:]


def compute_dominant_component(
    delta_phi: np.ndarray, block_fractions: np.ndarray
) -> np.ndarray:
    rho = delta_phi + block_fractions[:, np.newaxis, np.newaxis]
    dominant_component = np.argmax(rho, axis=0)
    return dominant_component


if __name__ == "__main__":
    dominant_component = compute_dominant_component(phi[-1], block_fractions)
    print(dominant_component.shape)
    data_tiled = np.tile(dominant_component, (2, 2))
    Lx = box_lengths[0] * 2
    Ly = box_lengths[1] * 2
    plt.imshow(data_tiled, cmap="tab10", extent=[0, Ly, 0, Lx])
    plt.show()
