from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch

from .free_energy import BlockCopolymerFreeEnergy
from .simulation_io import SimulationData


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


def plot_simulation_result(
    result: SimulationData, fig: plt.Figure = None, ax: plt.Axes = None
) -> tuple[plt.Figure, plt.Axes]:
    delta_phi = result.phi[-1]
    block_fractions = result.block_fractions
    rho = delta_phi + block_fractions[:, np.newaxis, np.newaxis]
    ndim = delta_phi.ndim - 1  # first axis is component

    model = result.build_model()
    block_fractions = model.f_vec
    chi_matrix = model.chi_matrix.detach().numpy().flatten()
    block_id = "_".join([f"{v:.3f}" for v in block_fractions.tolist()])
    chi_id = "_".join([f"{v:.3f}" for v in chi_matrix.tolist()])
    n_tiles = 3

    Lx = result.box_lengths[-1][0]
    Ly = result.box_lengths[-1][1]

    if fig is None:
        fig, ax = plt.subplots()

    dominant_component, alpha = compute_dominant_component(rho)
    dom_tiled = np.tile(dominant_component, (n_tiles, n_tiles))
    alpha_tiled = np.tile(alpha, (n_tiles, n_tiles))
    ax.imshow(
        dom_tiled,
        cmap="tab10",
        alpha=alpha_tiled,
        extent=[0, Ly * n_tiles, 0, Lx * n_tiles],
        vmin=0,
        vmax=10,
    )


def generate_annotation_str(result: SimulationData):
    block_fraction_string = np.array_str(result.block_fractions, precision=3)
    chi_matrix_string = np.array_str(result.chi_matrix, precision=3)
    return block_fraction_string, chi_matrix_string


def detect_trial_number(id_string: str) -> int:
    previous_trials = glob(f"visualizations/vis_{id_string}_*.png")
    if len(previous_trials) == 0:
        return 0
    else:
        return len(previous_trials)


