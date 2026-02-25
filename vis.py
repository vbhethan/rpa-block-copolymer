import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from rpa import BlockCopolymerFreeEnergy
import pickle


def load_joint_optimization_result(filename: str) -> dict:
    with open(filename, "rb") as f:
        result = pickle.load(f)
    return result


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
    # Works for any spatial dimensionality (1D, 2D, 3D)
    spatial_shape = dominant_component.shape
    flat_dom = dominant_component.ravel()
    flat_rho = rho.reshape(rho.shape[0], -1)
    dominant_rho = flat_rho[flat_dom, np.arange(flat_dom.size)].reshape(spatial_shape)
    alpha = dominant_rho / (dominant_rho.max() + 1e-12)
    return dominant_component, alpha


def visualize_dominant_component(dominant_component: np.ndarray, alpha=None) -> None:
    if alpha is None:
        plt.imshow(dominant_component, cmap="viridis")
    else:
        plt.imshow(dominant_component, cmap="viridis", alpha=alpha)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    delta_phi = load_delta_phi("delta_phi.pt").detach().numpy()
    ndim = delta_phi.ndim - 1  # first axis is component

    model = load_block_copolymer_model("block_copolymer_model.pt")
    block_fractions = model.f_vec.detach().numpy()
    chi_matrix = model.chi_matrix.detach().numpy().flatten()
    block_id = "_".join([f"{v:.3f}" for v in block_fractions.tolist()])
    chi_id = "_".join([f"{v:.3f}" for v in chi_matrix.tolist()])
    id_string = f"f_{block_id}_chi_{chi_id}"
    print(id_string)
    n_tiles = 3

    if ndim == 2:
        data = np.argmax(delta_phi, axis=0)
        flat_dom = data.ravel()
        flat_phi = delta_phi.reshape(delta_phi.shape[0], -1)
        alpha = flat_phi[flat_dom, np.arange(flat_dom.size)].reshape(data.shape)
        alpha = alpha / (alpha.max() + 1e-12)
        data = np.tile(data, (n_tiles, n_tiles))
        alpha = np.tile(alpha, (n_tiles, n_tiles))

        result = load_joint_optimization_result("joint_optimization_result.pkl")
        L_last = result["L_history"][-1]
        Lx = L_last[0]
        Ly = L_last[1]
        Lx_tiled = Lx * n_tiles
        Ly_tiled = Ly * n_tiles
        plt.imshow(data, cmap="tab10", alpha=alpha, extent=[0, Ly_tiled, 0, Lx_tiled])
        plt.colorbar()
        plt.savefig(f"visualizations/vis_{id_string}.png")
        plt.show()
    elif ndim == 1:
        data = np.argmax(delta_phi, axis=0)
        result = load_joint_optimization_result("joint_optimization_result.pkl")
        L_last = result["L_history"][-1]
        x = np.linspace(0, L_last[0], data.shape[0])
        plt.plot(x, data)
        plt.xlabel("x")
        plt.ylabel("dominant component")
        plt.tight_layout()
        plt.savefig(f"visualizations/vis_{id_string}.png")
        plt.show()
    else:
        print(
            f"Visualization for {ndim}D not yet implemented; data shape = {delta_phi.shape}"
        )
