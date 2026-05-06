"""
GPU tests — skipped automatically when CUDA is unavailable.

Run on a CUDA host:
    pytest tests/test_gpu.py -v
"""

import numpy as np
import pytest
import torch

from rpa import BlockCopolymerFreeEnergy, SimulationData
from rpa.optimizers import optimize_joint


# ---------------------------------------------------------------------------
# Shared skip marker and helpers
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _make_model(
    grid_shape: tuple[int, ...],
    optimize_box: bool = False,
    dtype: torch.dtype = torch.float64,
    device: str = "cuda",
) -> BlockCopolymerFreeEnergy:
    chi_matrix = torch.tensor(
        [[0.0, 26.0, 26.0], [26.0, 0.0, 26.0], [26.0, 26.0, 0.0]], dtype=dtype
    )
    l_ij_matrix = torch.zeros((3, 3), dtype=dtype)
    block_fractions = torch.tensor([1.0, 1.0, 1.0], dtype=dtype) / 3.0
    return BlockCopolymerFreeEnergy(
        chi_matrix=chi_matrix,
        l_ij_matrix=l_ij_matrix,
        block_fractions=block_fractions,
        optimize_box=optimize_box,
        grid_shape=grid_shape,
        dtype=dtype,
        device=device,
    )


def _assert_on_cuda(tensor: torch.Tensor, name: str) -> None:
    assert tensor.is_cuda, f"{name} should be on CUDA, got device={tensor.device}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_cuda
def test_model_tensors_on_cuda() -> None:
    """All parameters and buffers land on CUDA after construction."""
    torch.manual_seed(0)
    model = _make_model(grid_shape=(8, 8, 8))
    _assert_on_cuda(model.delta_phi.data, "delta_phi")
    _assert_on_cuda(model.f_vec, "f_vec")
    _assert_on_cuda(model.chi_matrix, "chi_matrix")
    _assert_on_cuda(model.l_ij_matrix, "l_ij_matrix")
    _assert_on_cuda(model._Gamma_ij_cached, "_Gamma_ij_cached")


@requires_cuda
def test_forward_and_gradients_cuda() -> None:
    """Forward pass and autograd produce finite results on CUDA (all grid dims)."""
    torch.manual_seed(0)
    for grid_shape, label in [((64,), "1D"), ((16, 16), "2D"), ((8, 8, 8), "3D")]:
        model = _make_model(grid_shape=grid_shape)
        delta_phi = model.get_order_parameters().detach().requires_grad_(True)

        F = model(delta_phi)
        grad = torch.autograd.grad(F, delta_phi)[0]

        assert F.ndim == 0, f"[{label}] F should be scalar"
        _assert_on_cuda(F, f"[{label}] F")
        _assert_on_cuda(grad, f"[{label}] grad")
        assert torch.isfinite(F).item(), f"[{label}] F not finite"
        assert torch.isfinite(grad).all().item(), f"[{label}] grad not finite"


@requires_cuda
def test_forward_and_gradients_cuda_float32() -> None:
    """float32 forward pass and gradients on CUDA."""
    torch.manual_seed(0)
    for grid_shape, label in [((64,), "1D"), ((16, 16), "2D"), ((8, 8, 8), "3D")]:
        model = _make_model(grid_shape=grid_shape, dtype=torch.float32)
        delta_phi = model.get_order_parameters().detach().requires_grad_(True)

        F = model(delta_phi)
        grad = torch.autograd.grad(F, delta_phi)[0]

        assert F.dtype == torch.float32, f"[float32 {label}] wrong output dtype"
        assert torch.isfinite(F).item(), f"[float32 {label}] F not finite"
        assert torch.isfinite(grad).all().item(), f"[float32 {label}] grad not finite"


@requires_cuda
def test_constraints_cuda() -> None:
    """Projection satisfies incompressibility and zero-mean on CUDA."""
    torch.manual_seed(42)
    tol = 1e-12
    for grid_shape in [(64,), (16, 16), (8, 8, 8)]:
        model = _make_model(grid_shape=grid_shape)
        info = model.check_constraints()
        for mean_val in info["phi_means"]:
            assert abs(mean_val) < tol, f"Mean not zero for {grid_shape} on CUDA"
        assert info["incompressibility_max_error"] < tol, (
            f"Incompressibility violated for {grid_shape} on CUDA"
        )


@requires_cuda
def test_box_optimize_cuda() -> None:
    """optimize_box=True: log_L parameter and its gradient stay on CUDA."""
    torch.manual_seed(0)
    model = _make_model(grid_shape=(8, 8, 8), optimize_box=True)
    _assert_on_cuda(model.log_L.data, "log_L")
    _assert_on_cuda(model.delta_phi.data, "delta_phi")

    F = model()
    assert torch.isfinite(F).item()
    F.backward()
    assert model.log_L.grad is not None
    _assert_on_cuda(model.log_L.grad, "log_L.grad")


@requires_cuda
def test_build_model_cuda() -> None:
    """SimulationData.build_model places all tensors on CUDA."""
    data = SimulationData(
        N=100,
        b=1.0,
        block_fractions=np.array([0.5, 0.5]),
        chi_matrix=np.array([[0.0, 0.2], [0.2, 0.0]]),
        l_ij_matrix=np.zeros((2, 2)),
        phi_bar=1.0,
        grid_shape=(8, 8, 8),
    )
    model = data.build_model(optimize_box=True, device="cuda")
    _assert_on_cuda(model.delta_phi.data, "delta_phi")
    _assert_on_cuda(model.log_L.data, "log_L")
    _assert_on_cuda(model.f_vec, "f_vec")
    _assert_on_cuda(model.chi_matrix, "chi_matrix")


@requires_cuda
def test_build_model_cuda_with_trajectory() -> None:
    """build_model loads a trajectory frame onto CUDA correctly."""
    grid_shape = (8, 8, 8)
    n = 2
    phi = np.random.default_rng(0).standard_normal((1, n, *grid_shape))
    data = SimulationData(
        N=100,
        b=1.0,
        block_fractions=np.array([0.5, 0.5]),
        chi_matrix=np.array([[0.0, 0.2], [0.2, 0.0]]),
        l_ij_matrix=np.zeros((2, 2)),
        phi_bar=1.0,
        grid_shape=grid_shape,
        phi=phi,
        F=np.array([0.0]),
        box_lengths=np.array([[4.0, 4.0, 4.0]]),
    )
    model = data.build_model(optimize_box=False, device="cuda")
    _assert_on_cuda(model.delta_phi.data, "delta_phi (loaded from trajectory)")


@requires_cuda
def test_optimize_joint_cuda_returns_numpy() -> None:
    """optimize_joint on CUDA yields numpy arrays in SimulationData (not CUDA tensors)."""
    torch.manual_seed(0)
    model = _make_model(grid_shape=(8, 8, 8), optimize_box=True)

    result = optimize_joint(
        model,
        n_outer=3,
        n_inner_phi=5,
        n_inner_box=2,
        lr_phi=0.1,
        lr_box=0.1,
        log_every=10,
    )

    assert isinstance(result.phi, np.ndarray), "phi trajectory must be a numpy array"
    assert isinstance(result.F, np.ndarray), "F trajectory must be a numpy array"
    assert isinstance(result.box_lengths, np.ndarray), "box_lengths must be a numpy array"
    assert result.n_frames == 3
    assert np.isfinite(result.phi).all(), "phi trajectory contains non-finite values"
    assert np.isfinite(result.F).all(), "F trajectory contains non-finite values"
