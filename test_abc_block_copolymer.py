import numpy as np
import pytest
from rpa import ABCBlockCopolymer, phi_grid


class TestABCBlockCopolymer:
    """Test suite for ABCBlockCopolymer free energy functional."""

    def setup_method(self):
        """Set up test fixtures."""
        # ABC block copolymer parameters
        self.n_components = 3
        self.block_lengths = np.array([10.0, 20.0, 30.0])  # A, B, C block lengths
        self.block_fractions = np.array([0.2, 0.3, 0.5])  # f_A, f_B, f_C
        self.kuhn_length = 1.0

        # Chi matrix for ABC triblock (symmetric)
        self.chi_matrix = np.array(
            [
                [0.0, 0.1, 0.2],  # chi_AA, chi_AB, chi_AC
                [0.1, 0.0, 0.15],  # chi_BA, chi_BB, chi_BC
                [0.2, 0.15, 0.0],  # chi_CA, chi_CB, chi_CC
            ]
        )

        # Grid parameters
        self.grid_shape = (32, 32)  # 2D grid
        self.box_length = 10.0

    def create_abc_copolymer(self, phi_values=None):
        """Helper method to create ABCBlockCopolymer instance."""
        # Create phi_grid
        grid = phi_grid(
            grid_shape=self.grid_shape,
            block_fractions=self.block_fractions,
            phi_values=phi_values,
            n_components=self.n_components,
            box_length=self.box_length,
        )

        # Create ABCBlockCopolymer with phi_grid
        copolymer = ABCBlockCopolymer(
            n_components=self.n_components,
            block_lengths=self.block_lengths,
            block_fractions=self.block_fractions,
            chi_matrix=self.chi_matrix,
            phi_grid=grid,
            kuhn_length=self.kuhn_length,
        )

        return copolymer

    def test_l_ij_matrix_computation(self):
        """Test that l_ij_matrix is computed correctly for ABC block copolymer."""
        copolymer = self.create_abc_copolymer()

        # Expected l_ij_matrix for ABC:
        # l_01 = 0 (A and B are adjacent)
        # l_02 = block_lengths[1] (A and C are separated by B)
        # l_12 = 0 (B and C are adjacent)
        expected_l_ij = np.zeros((3, 3))
        expected_l_ij[0, 1] = 0.0
        expected_l_ij[0, 2] = self.block_lengths[1]
        expected_l_ij[1, 2] = 0.0
        # Make symmetric
        expected_l_ij = expected_l_ij + expected_l_ij.T

        np.testing.assert_array_equal(copolymer.l_ij_matrix, expected_l_ij)

    def test_free_energy_functional_returns_scalar(self):
        """Test that free_energy_functional returns a scalar value."""
        copolymer = self.create_abc_copolymer()
        free_energy = copolymer.free_energy_functional()

        assert isinstance(free_energy, (float, np.floating))
        assert not np.isnan(free_energy)
        assert not np.isinf(free_energy)

    def test_free_energy_functional_components(self):
        """Test individual components of the free energy functional."""
        copolymer = self.create_abc_copolymer()

        ideal_entropy = copolymer.ideal_mixing_entropy()
        interaction_enthalpy = copolymer.compute_interaction_enthalpy()
        ideal_entropy_second_order = copolymer.ideal_mixing_entropy_second_order()

        total_free_energy = copolymer.free_energy_functional()
        expected_total = (
            ideal_entropy + interaction_enthalpy - ideal_entropy_second_order
        )

        np.testing.assert_almost_equal(total_free_energy, expected_total, decimal=10)

    # def test_free_energy_functional_with_uniform_phi(self):
    #     """Test free energy functional with uniform phi values (should be close to zero)."""
    #     # Create uniform phi values equal to block fractions
    #     phi_uniform = np.zeros((self.n_components, *self.grid_shape))
    #     for i in range(self.n_components):
    #         phi_uniform[i, :, :] = self.block_fractions[i]

    #     copolymer = self.create_abc_copolymer(phi_values=phi_uniform)
    #     free_energy = copolymer.free_energy_functional()

    #     # With uniform phi, delta_phi should be zero, so free energy should be small
    #     # (not exactly zero due to numerical precision and q=0 handling)
    #     assert abs(free_energy) < 1e-5

    def test_free_energy_functional_different_grid_sizes(self):
        """Test that free energy functional works with different grid sizes."""
        for grid_size in [(16, 16), (32, 32), (64, 64)]:
            self.grid_shape = grid_size
            copolymer = self.create_abc_copolymer()
            free_energy = copolymer.free_energy_functional()

            assert isinstance(free_energy, (float, np.floating))
            assert not np.isnan(free_energy)
            assert not np.isinf(free_energy)

    def test_free_energy_functional_different_chi_values(self):
        """Test free energy functional with different chi matrix values."""
        # Test with stronger interactions
        strong_chi = self.chi_matrix * 2.0
        grid = phi_grid(
            grid_shape=self.grid_shape,
            block_fractions=self.block_fractions,
            n_components=self.n_components,
            box_length=self.box_length,
        )
        copolymer_strong = ABCBlockCopolymer(
            n_components=self.n_components,
            block_lengths=self.block_lengths,
            block_fractions=self.block_fractions,
            chi_matrix=strong_chi,
            phi_grid=grid,
            kuhn_length=self.kuhn_length,
        )
        free_energy_strong = copolymer_strong.free_energy_functional()

        # Test with weaker interactions
        weak_chi = self.chi_matrix * 0.5
        grid = phi_grid(
            grid_shape=self.grid_shape,
            block_fractions=self.block_fractions,
            n_components=self.n_components,
            box_length=self.box_length,
        )
        copolymer_weak = ABCBlockCopolymer(
            n_components=self.n_components,
            block_lengths=self.block_lengths,
            block_fractions=self.block_fractions,
            chi_matrix=weak_chi,
            phi_grid=grid,
            kuhn_length=self.kuhn_length,
        )
        free_energy_weak = copolymer_weak.free_energy_functional()

        # Both should be valid free energy values
        assert isinstance(free_energy_strong, (float, np.floating))
        assert isinstance(free_energy_weak, (float, np.floating))
        assert not np.isnan(free_energy_strong)
        assert not np.isnan(free_energy_weak)

    def test_gamma_q_computation(self):
        """Test that gamma_q is computed correctly."""
        copolymer = self.create_abc_copolymer()

        # Check shape
        expected_shape = (*self.grid_shape, self.n_components, self.n_components)
        assert copolymer.gamma_q.shape == expected_shape

        # Check that gamma_q at q=0 is zero (or very small)
        # The first grid point typically corresponds to q=0
        gamma_at_zero = copolymer.gamma_q[0, 0]
        assert np.allclose(gamma_at_zero, 0, atol=1e-10)

    def test_interaction_enthalpy_computation(self):
        """Test interaction enthalpy computation."""
        copolymer = self.create_abc_copolymer()
        interaction_enthalpy = copolymer.compute_interaction_enthalpy()

        assert isinstance(interaction_enthalpy, (float, np.floating))
        assert not np.isnan(interaction_enthalpy)
        # Interaction enthalpy should be real (not complex)
        assert np.isrealobj(interaction_enthalpy)

    def test_ideal_mixing_entropy_computation(self):
        """Test ideal mixing entropy computation."""
        copolymer = self.create_abc_copolymer()
        ideal_entropy = copolymer.ideal_mixing_entropy()

        assert isinstance(ideal_entropy, (float, np.floating))
        assert not np.isnan(ideal_entropy)
        assert not np.isinf(ideal_entropy)

    def test_ideal_mixing_entropy_second_order_computation(self):
        """Test ideal mixing entropy second order computation."""
        copolymer = self.create_abc_copolymer()
        ideal_entropy_second = copolymer.ideal_mixing_entropy_second_order()

        assert isinstance(ideal_entropy_second, (float, np.floating))
        assert not np.isnan(ideal_entropy_second)
        assert not np.isinf(ideal_entropy_second)

    def test_abc_block_copolymer_initialization(self):
        """Test that ABCBlockCopolymer can be properly initialized with all required parameters."""
        grid = phi_grid(
            grid_shape=self.grid_shape,
            block_fractions=self.block_fractions,
            n_components=self.n_components,
            box_length=self.box_length,
        )

        copolymer = ABCBlockCopolymer(
            n_components=self.n_components,
            block_lengths=self.block_lengths,
            block_fractions=self.block_fractions,
            chi_matrix=self.chi_matrix,
            phi_grid=grid,
            kuhn_length=self.kuhn_length,
        )

        # Check that all attributes are set
        assert copolymer.n_components == self.n_components
        assert np.array_equal(copolymer.block_lengths, self.block_lengths)
        assert np.array_equal(copolymer.block_fractions, self.block_fractions)
        assert np.array_equal(copolymer.chi_matrix, self.chi_matrix)
        assert copolymer.phi_grid is not None
        assert copolymer.l_ij_matrix is not None
        assert copolymer.gamma_q is not None

    def test_free_energy_functional_consistency(self):
        """Test that free energy functional is consistent across multiple calls."""
        copolymer = self.create_abc_copolymer()

        free_energy_1 = copolymer.free_energy_functional()
        free_energy_2 = copolymer.free_energy_functional()

        # Should return the same value for the same state
        np.testing.assert_almost_equal(free_energy_1, free_energy_2, decimal=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
