import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Allow running this file directly: python tests/test_rpa.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from rpa import BlockCopolymerFreeEnergy


def _make_model(grid_shape: tuple[int, ...], optimize_box: bool = False):
    """Helper to build a triblock model on a given grid shape."""
    chi_matrix = torch.tensor(
        [
            [0.0, 26.0, 26.0],
            [26.0, 0.0, 26.0],
            [26.0, 26.0, 0.0],
        ],
        dtype=torch.float64,
    )
    l_ij_matrix = torch.zeros((3, 3), dtype=torch.float64)
    block_fractions = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64) / 3.0

    model = BlockCopolymerFreeEnergy(
        chi_matrix=chi_matrix,
        l_ij_matrix=l_ij_matrix,
        block_fractions=block_fractions,
        optimize_box=optimize_box,
        grid_shape=grid_shape,
    )
    return model


def _run_forward_and_gradient(model):
    """Run a forward pass and gradient check on the given model."""
    delta_phi = model.get_order_parameters().detach().requires_grad_(True)

    F = model(delta_phi)
    grad = torch.autograd.grad(F, delta_phi)[0]

    assert F.ndim == 0, "Forward pass should return a scalar tensor."
    assert grad.shape == delta_phi.shape, "Gradient shape should match input shape."
    assert torch.isfinite(F).item(), "Free energy should be finite."
    assert torch.isfinite(grad).all().item(), "Gradients should be finite."

    return F, grad


def test_forward_and_gradients_2d() -> None:
    """Original 2D test case."""
    torch.manual_seed(0)
    model = _make_model(grid_shape=(32, 32))
    F, grad = _run_forward_and_gradient(model)

    assert model.ndim == 2
    assert model.grid_shape == (32, 32)
    assert grad.shape == (3, 32, 32)

    print(f"[2D] F = {F.item():.6f}, grad shape = {tuple(grad.shape)}")
    print("[2D] forward + gradient test passed")


def test_forward_and_gradients_1d() -> None:
    """1D grid test case."""
    torch.manual_seed(0)
    model = _make_model(grid_shape=(64,))
    F, grad = _run_forward_and_gradient(model)

    assert model.ndim == 1
    assert model.grid_shape == (64,)
    assert grad.shape == (3, 64)

    print(f"[1D] F = {F.item():.6f}, grad shape = {tuple(grad.shape)}")
    print("[1D] forward + gradient test passed")


def test_forward_and_gradients_3d() -> None:
    """3D grid test case (small grid to keep fast)."""
    torch.manual_seed(0)
    model = _make_model(grid_shape=(8, 8, 8))
    F, grad = _run_forward_and_gradient(model)

    assert model.ndim == 3
    assert model.grid_shape == (8, 8, 8)
    assert grad.shape == (3, 8, 8, 8)

    print(f"[3D] F = {F.item():.6f}, grad shape = {tuple(grad.shape)}")
    print("[3D] forward + gradient test passed")


def test_constraints_all_dims() -> None:
    """Check that projection satisfies constraints in all dims."""
    torch.manual_seed(42)
    for grid_shape in [(64,), (16, 16), (8, 8, 8)]:
        model = _make_model(grid_shape=grid_shape)
        info = model.check_constraints()
        for mean_val in info["phi_means"]:
            assert abs(mean_val) < 1e-12, f"Mean not zero for {grid_shape}"
        assert info["incompressibility_max_error"] < 1e-12, (
            f"Incompressibility violated for {grid_shape}"
        )
    print("constraint check passed for 1D, 2D, 3D")


def test_box_dimensions_property() -> None:
    """Check that get_box_dimensions returns correct structure."""
    model = _make_model(grid_shape=(16, 16), optimize_box=True)
    dims = model.get_box_dimensions()
    assert len(dims["L"]) == 2
    assert len(dims["spacing"]) == 2
    assert dims["ndim"] == 2
    assert dims["grid_shape"] == [16, 16]
    print("box dimensions property test passed")


if __name__ == "__main__":
    test_forward_and_gradients_1d()
    test_forward_and_gradients_2d()
    test_forward_and_gradients_3d()
    test_constraints_all_dims()
    test_box_dimensions_property()
    print("\nAll tests passed!")
