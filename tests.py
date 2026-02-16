import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from rpa_torch import BlockCopolymerFreeEnergy


def test_forward_and_gradients() -> None:
    torch.manual_seed(0)

    chi_matrix = torch.tensor(
        [
            [0.0, 26.0, 26.0],
            [26.0, 0.0, 26.0],
            [26.0, 26.0, 0.0],
        ],
        dtype=torch.float32,
    )
    l_ij_matrix = torch.zeros((3, 3), dtype=torch.float32)
    block_fractions = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32) / 3.0

    model = BlockCopolymerFreeEnergy(
        chi_matrix=chi_matrix,
        l_ij_matrix=l_ij_matrix,
        block_fractions=block_fractions,
        optimize_box=False,
        Nx=32,
        Ny=32,
        dx=1.0,
        dy=1.0,
    )

    delta_phi = torch.randn(
        model.n_components, model.Nx, model.Ny, dtype=torch.float32, requires_grad=True
    )

    F = model(delta_phi)
    grad = torch.autograd.grad(F, delta_phi)[0]

    assert F.ndim == 0, "Forward pass should return a scalar tensor."
    assert grad.shape == delta_phi.shape, "Gradient shape should match input shape."
    assert torch.isfinite(F).item(), "Free energy should be finite."
    assert torch.isfinite(grad).all().item(), "Gradients should be finite."

    print(f"F = {F.item():.6f}")
    print(f"grad shape = {tuple(grad.shape)}")
    print("forward + gradient test passed")


if __name__ == "__main__":
    test_forward_and_gradients()
