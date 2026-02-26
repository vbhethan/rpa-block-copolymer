import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from rpa import BlockCopolymerFreeEnergy
from simulation_io import SimulationData


def load_delta_phi(filename: str) -> torch.Tensor:
    delta_phi = torch.load(filename)
    return delta_phi


def load_block_copolymer_model(filename: str) -> BlockCopolymerFreeEnergy:
    block_copolymer_model = torch.load(filename, weights_only=False)
    return block_copolymer_model


def compute_dominant_component(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    result = SimulationData.from_hdf5("optimization_result.h5")
    delta_phi = result.phi[-1]
    block_fractions = result.block_fractions
    rho = delta_phi + block_fractions[:, np.newaxis, np.newaxis]
    ndim = delta_phi.ndim - 1  # first axis is component

    model = result.build_model()
    block_fractions = model.f_vec
    chi_matrix = model.chi_matrix.detach().numpy().flatten()
    block_id = "_".join([f"{v:.3f}" for v in block_fractions.tolist()])
    chi_id = "_".join([f"{v:.3f}" for v in chi_matrix.tolist()])
    id_string = f"f_{block_id}_chi_{chi_id}"
    n_tiles = 3

    Lx = result.box_lengths[-1][0]
    Ly = result.box_lengths[-1][1]

    dominant_component, alpha = compute_dominant_component(rho)
    dom_tiled = np.tile(dominant_component, (n_tiles, n_tiles))
    alpha_tiled = np.tile(alpha, (n_tiles, n_tiles))
    plt.imshow(
        dom_tiled,
        cmap="tab10",
        alpha=alpha_tiled,
        extent=[0, Ly * n_tiles, 0, Lx * n_tiles],
        vmin=0,
        vmax=10,
    )
    plt.colorbar()
    plt.savefig(f"visualizations/vis_{id_string}.png")
    plt.show()
