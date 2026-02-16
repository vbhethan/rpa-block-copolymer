import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from rpa_torch import BlockCopolymerFreeEnergy


def load_delta_phi(filename: str) -> torch.Tensor:
    delta_phi = torch.load(filename)
    return delta_phi


def load_block_copolymer_model(filename: str) -> BlockCopolymerFreeEnergy:
    block_copolymer_model = torch.load(filename, weights_only=False)
    return block_copolymer_model


def compute_dominant_component(
    block_copolymer_model: BlockCopolymerFreeEnergy, delta_phi: np.ndarray
) -> np.ndarray:
    rho = block_copolymer_model.get_densities(delta_phi).detach().numpy()
    dominant_component = np.argmax(rho, axis=0)
    return dominant_component


def tile_dominant_component(dominant_component: np.ndarray) -> np.ndarray:
    tile_size = 3
    tiled_dominant_component = np.tile(dominant_component, (tile_size, tile_size))
    return tiled_dominant_component


def visualize_dominant_component(dominant_component: np.ndarray) -> None:
    plt.imshow(dominant_component, cmap="viridis")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    delta_phi = load_delta_phi("delta_phi.pt")
    block_copolymer_model = load_block_copolymer_model("block_copolymer_model.pt")
    dominant_component = compute_dominant_component(block_copolymer_model, delta_phi)
    tiled_dominant_component = tile_dominant_component(dominant_component)
    visualize_dominant_component(tiled_dominant_component)
