import numpy as np
import torch

from rpa import BlockCopolymerFreeEnergy
from rpa.optimizers import optimize_phi_only, picard_optimize_phi_only


def _make_model(
    grid_shape: tuple[int, ...],
    chi: float = 26.0,
    dtype: torch.dtype = torch.float64,
):
    """Symmetric triblock fixture (equal block fractions), matching test_rpa."""
    chi_matrix = torch.tensor(
        [
            [0.0, chi, chi],
            [chi, 0.0, chi],
            [chi, chi, 0.0],
        ],
        dtype=dtype,
    )
    l_ij_matrix = torch.zeros((3, 3), dtype=dtype)
    block_fractions = torch.tensor([1.0, 1.0, 1.0], dtype=dtype) / 3.0

    return BlockCopolymerFreeEnergy(
        chi_matrix=chi_matrix,
        l_ij_matrix=l_ij_matrix,
        block_fractions=block_fractions,
        optimize_box=False,
        grid_shape=grid_shape,
        dtype=dtype,
    )


def _make_model_3d(
    grid_shape: tuple[int, ...],
    box: tuple[float, ...],
    chi: float = 30.0,
    dtype: torch.dtype = torch.float64,
):
    """Triblock on a 3D grid with an explicit box, for ordering tests."""
    chi_matrix = torch.full((3, 3), chi, dtype=dtype)
    chi_matrix.fill_diagonal_(0.0)
    return BlockCopolymerFreeEnergy(
        chi_matrix=chi_matrix,
        l_ij_matrix=torch.zeros((3, 3), dtype=dtype),
        block_fractions=torch.ones(3, dtype=dtype) / 3.0,
        optimize_box=False,
        grid_shape=grid_shape,
        box_lengths=box,
        dtype=dtype,
        init_amplitude=0.1,
    )


def test_residual_free_energy_matches_components() -> None:
    """residual_free_energy == Delta_F_int - F_mixing_2 for a fixed field."""
    torch.manual_seed(0)
    model = _make_model((16, 16))
    delta_phi = model.get_order_parameters()

    comps = model.get_energy_components(delta_phi)
    expected = comps["Delta_F_int"] - comps["F_mixing_2"]
    got = model.residual_free_energy(delta_phi, project=True)

    assert torch.allclose(got, expected)


def test_residual_gradient_zero_at_homogeneous() -> None:
    """F_res is quadratic in delta_phi, so its gradient vanishes at delta_phi=0."""
    model = _make_model((16, 16))
    delta_phi = torch.zeros_like(model.delta_phi.data).requires_grad_(True)
    F_res = model.residual_free_energy(delta_phi, project=False)
    (grad,) = torch.autograd.grad(F_res, delta_phi)
    assert grad.abs().max().item() < 1e-10


def test_homogeneous_is_fixed_point() -> None:
    """Starting from the homogeneous state, Picard leaves the field unchanged."""
    model = _make_model((16, 16))
    model.delta_phi.data.zero_()

    result = picard_optimize_phi_only(
        model, n_steps=5, alpha=0.5, silent=True
    )

    assert np.abs(result.phi[-1]).max() < 1e-10
    assert result.incompressibility_drift.max() < 1e-10


def test_return_contract() -> None:
    """Trajectory arrays are numpy, finite, and frame-major."""
    torch.manual_seed(1)
    model = _make_model((16, 16))

    result = picard_optimize_phi_only(
        model, n_steps=10, alpha=0.002, patience=100, silent=True
    )

    assert isinstance(result.F, np.ndarray)
    assert isinstance(result.phi, np.ndarray)
    assert isinstance(result.box_lengths, np.ndarray)
    assert result.F.shape == (10,)
    assert result.phi.shape == (10, 3, 16, 16)
    assert result.box_lengths.shape == (10, 2)
    assert np.isfinite(result.F).all()
    assert np.isfinite(result.phi).all()


def test_record_every_subsamples_trajectory() -> None:
    """record_every stores fewer frames but always includes the final step."""
    torch.manual_seed(1)
    model = _make_model((16, 16))

    result = picard_optimize_phi_only(
        model, n_steps=20, alpha=0.002, patience=100, record_every=5, silent=True
    )
    # frames at steps 0,5,10,15 plus the final step 19 -> 5 frames.
    assert result.n_frames == 5


def test_converges_to_pgd_minimum_when_disordered() -> None:
    """In a disordered regime Picard converges to the same (homogeneous) state
    the PGD optimizer finds, with strictly positive densities."""
    torch.manual_seed(2)
    model_pgd = _make_model((16, 16), chi=12.0)
    F_pgd = optimize_phi_only(model_pgd, n_steps=4000, silent=True).F[-1]

    torch.manual_seed(2)
    model = _make_model((16, 16), chi=12.0)
    result = picard_optimize_phi_only(
        model, n_steps=12000, alpha=0.002, tol=1e-7, patience=100, silent=True
    )

    assert result.converged, "Picard iteration should reach the stall criterion"
    assert np.isfinite(result.F[-1])
    assert abs(result.F[-1] - F_pgd) < 1e-3, (result.F[-1], F_pgd)

    # Densities rho_i = delta_phi_i + f_i * phi_bar must stay strictly positive.
    f = model.f_vec.detach().cpu().numpy().reshape(3, 1, 1)
    rho = result.phi[-1] + f * model.phi_bar
    assert rho.min() > 0.0


def test_finds_ordered_state_in_3d() -> None:
    """On a well-resolved ordered box the exp map breaks symmetry and relaxes
    into a microphase fixed point below the homogeneous free energy, keeping
    densities positive throughout. (PGD cannot reach this: from a large-amplitude
    start its log term diverges, from a small one it stalls at the saddle.)"""
    homog = 3.0 * np.log(1.0 / 3.0)  # -3.2958, the homogeneous free energy

    model = _make_model_3d((16, 16, 16), box=(7.5, 7.5, 7.5), chi=30.0)
    result = picard_optimize_phi_only(
        model,
        n_steps=2000,
        alpha=0.1,
        precondition=True,
        tol=1e-7,
        patience=200,
        record_every=500,
        silent=True,
    )

    assert np.isfinite(result.F[-1])
    assert result.F[-1] < homog - 0.02, (result.F[-1], homog)  # ordered
    assert np.abs(result.phi[-1]).max() > 0.2  # real structure formed

    f = model.f_vec.detach().cpu().numpy().reshape(3, 1, 1, 1)
    rho = result.phi[-1] + f * model.phi_bar
    assert rho.min() > 0.0  # positivity preserved by the exp map
