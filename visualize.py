import matplotlib.pyplot as plt
import numpy as np

from simulation_io import SimulationData

sim_file = "simulation.h5"

data = SimulationData.from_hdf5(sim_file)
block_fractions = data.block_fractions
box_lengths = data.box_lengths  # (n_frames, ndim)
F = data.F
phi = data.phi


def compute_dominant_component(
    delta_phi: np.ndarray, block_fractions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rho = delta_phi + block_fractions[:, np.newaxis, np.newaxis]
    dominant_component = np.argmax(rho, axis=0)
    # Works for any spatial dimensionality (1D, 2D, 3D)
    spatial_shape = dominant_component.shape
    flat_dom = dominant_component.ravel()
    flat_rho = rho.reshape(rho.shape[0], -1)
    dominant_rho = flat_rho[flat_dom, np.arange(flat_dom.size)].reshape(spatial_shape)
    alpha = (dominant_rho - dominant_rho.min()) / (
        dominant_rho.max() - dominant_rho.min() + 1e-12
    )
    return dominant_component, alpha


if __name__ == "__main__":
    dominant_component, alpha = compute_dominant_component(phi[-1], block_fractions)
    data_tiled = np.tile(dominant_component, (2, 2))
    alpha_tiled = np.tile(alpha, (2, 2))
    Lx = box_lengths[-1, 0] * 2
    Ly = box_lengths[-1, 1] * 2
    plt.imshow(
        data_tiled,
        cmap="tab10",
        vmin=0,
        vmax=10,
        extent=[0, Ly, 0, Lx],
        alpha=alpha_tiled,
    )
    # plt.colorbar()
    plt.show()
