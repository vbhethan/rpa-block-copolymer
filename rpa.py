import numpy as np


class phi_grid:
    def __init__(
        self,
        grid_shape,
        block_fractions,
        phi_values=None,
        n_components=None,
        box_length=1.0,
    ):
        self.grid_shape = grid_shape
        ndim = len(grid_shape)

        # Handle box_length: convert scalar to tuple or validate tuple
        if np.isscalar(box_length):
            self.box_length = tuple([box_length] * ndim)
        else:
            assert len(box_length) == ndim, (
                f"box_length must have length {ndim} to match grid_shape"
            )
            self.box_length = tuple(box_length)

        if phi_values is not None:
            n_components = phi_values.shape[0]
            assert phi_values.shape[1:] == grid_shape
            self.phi_values = phi_values
        else:
            self.phi_values = np.random.rand(*(n_components, *grid_shape))
        self.n_components = n_components

        # Compute the delta_phi_i on grid
        # Reshape block_fractions for proper broadcasting: (n_components,) -> (n_components, 1, 1, ...)
        block_fractions_reshaped = block_fractions.reshape(n_components, *([1] * ndim))
        self.delta_phi_i = self.phi_values - block_fractions_reshaped

        # Only transform over spatial dimensions (axes 1, 2, ..., ndim), not component axis (0)
        spatial_axes = tuple(range(1, ndim + 1))
        self.delta_phi_q = np.fft.fftn(self.delta_phi_i, axes=spatial_axes)

        # Generate the real space and k-space mesh grids
        self._compute_grids()

    def _compute_grids(self):
        """
        Compute and store real space and k-space grid coordinates.

        Stores:
        - self.r_grids: tuple of real space coordinate grids in physical units (r_x, r_y, ...)
                       where r_i = n_i * L_i / M_i, with n_i integer indices and M_i grid points
        - self.k_grids: tuple of k-space coordinate grids (k_x, k_y, ...)
        - self.k_magnitude: magnitude of k-vector at each grid point |k|

        For k-space: k = 2π * q / L where q are integer mode numbers and L is box_length
        """
        ndim = len(self.grid_shape)

        # Compute real space coordinate grids in physical units
        # r_i = n_i * L_i / M_i where n_i are integer indices (0, 1, ..., M_i-1)
        r_arrays = []
        for n, L in zip(self.grid_shape, self.box_length):
            # Integer indices: 0, 1, 2, ..., n-1
            n_1d = np.arange(n)
            # Convert to physical units: r = n * L / M
            r_1d = n_1d * L / n
            r_arrays.append(r_1d)

        if ndim == 1:
            self.r_grids = (r_arrays[0],)
        else:
            r_mesh = np.meshgrid(*r_arrays, indexing="ij")
            self.r_grids = tuple(r_mesh)

        # Compute k-space grids
        # For each dimension: grid_spacing = box_length / n
        # fftfreq(n, d) gives frequencies f = m/(n*d) where m are mode numbers
        # k = 2π * f = 2π * m / (n*d) = 2π * m / box_length
        k_arrays = []
        for n, L in zip(self.grid_shape, self.box_length):
            # Grid spacing for this dimension
            grid_spacing = L / n
            # Get frequency array for this dimension
            freq = np.fft.fftfreq(n, d=grid_spacing)
            # Convert to k-space: k = 2π * frequency
            k_1d = 2 * np.pi * freq
            k_arrays.append(k_1d)

        # Create meshgrids for multi-dimensional case
        if ndim == 1:
            self.k_grids = (k_arrays[0],)
            self.k_magnitude = np.abs(k_arrays[0])
        else:
            # Create coordinate grids
            k_mesh = np.meshgrid(*k_arrays, indexing="ij")
            self.k_grids = tuple(k_mesh)

            # Compute magnitude: |k| = sqrt(k_x^2 + k_y^2 + ...)
            self.k_magnitude = np.sqrt(sum(k**2 for k in self.k_grids))


class BlockCopolymer:
    """
    Base class for block copolymers.
    It is used to store the necessary information to pass to the free energy functional later
    The grid object for the density field is also contained in this class.
    """

    def __init__(
        self,
        n_components,
        block_lengths,
        block_fractions,
        chi_matrix,
        phi_grid,
        kuhn_length=1.0,
    ):
        self.block_fractions = block_fractions
        self.chi_matrix = chi_matrix
        self.kuhn_length = kuhn_length
        self.n_components = n_components
        self.block_lengths = block_lengths
        self.l_ij_matrix = self._compute_l_ij_matrix()

        self.phi_grid = phi_grid

        self.gamma_q = self.compute_gamma_q()

    def compute_gamma_q(self, N=None):
        """
        Compute gamma_q everywhere on the q-space grid.

        Parameters:
        -----------
        N : float, optional
            Total chain length. If None, computed as sum of block_lengths.

        Returns:
        --------
        gamma_q_grid : np.ndarray
            Array of shape (*grid_shape, n_components, n_components) containing
            gamma_q matrices at each grid point.
        """
        # Compute N if not provided
        if N is None:
            N = np.sum(self.block_lengths)

        # Get q-space grid (magnitude of k-vectors)
        q_grid = self.phi_grid.k_magnitude
        grid_shape = self.phi_grid.grid_shape

        # Initialize output array: (*grid_shape, n_components, n_components)
        gamma_q_grid = np.zeros((*grid_shape, self.n_components, self.n_components))

        # Iterate over all grid points
        for idx in np.ndindex(grid_shape):
            q = q_grid[idx]
            # Skip q=0 values (keep gamma_q as 0 for these points)
            if q < 1e-10:  # Use small epsilon to handle floating point precision
                continue
            # Compute gamma_q at this q value
            gamma_matrix = gamma_q(
                q,
                self.chi_matrix,
                self.l_ij_matrix,
                self.block_fractions,
                N,
                self.kuhn_length,
            )
            gamma_q_grid[idx] = gamma_matrix

        # Store as instance variable
        self.gamma_q = gamma_q_grid

        return gamma_q_grid

    def free_energy_functional(self):
        # compute ideal mixing entropy
        ideal_mixing_entropy = self.ideal_mixing_entropy()
        # compute the interaction enthalpy
        interaction_enthalpy = self.compute_interaction_enthalpy()
        # compute the ideal mixing entropy second order
        ideal_mixing_entropy_second_order = self.ideal_mixing_entropy_second_order()
        free_energy = (
            ideal_mixing_entropy
            + interaction_enthalpy
            - ideal_mixing_entropy_second_order
        )
        # Ensure return value is real (handle numerical precision issues)
        return np.real(free_energy)

    def compute_interaction_enthalpy(self):
        # compute the interaction enthalpy \sum_ij \int dq Gamma_ij(q)delta_phi_i(q) delta_phi_j(-q)
        # Since delta_phi(r) is real valued everywhere, use the conjugate for the (-q)
        delta_phi_q_conjugate = np.conj(self.phi_grid.delta_phi_q)
        volume = np.prod(self.phi_grid.box_length)
        N_points = np.prod(self.phi_grid.grid_shape)
        normalization = volume / (N_points**2)
        interaction_enthalpy = sum(
            sum(
                np.sum(
                    self.gamma_q[..., i, j]
                    * self.phi_grid.delta_phi_q[i]
                    * delta_phi_q_conjugate[j]
                )
                for j in range(self.n_components)
            )
            for i in range(self.n_components)
        )
        # Take real part to handle numerical precision issues (should be real)
        return 0.5 * np.real(interaction_enthalpy) * normalization

    def ideal_mixing_entropy(self):
        return sum(
            ideal_mixing_entropy(self.phi_grid.phi_values[i], self.block_fractions[i])
            for i in range(self.n_components)
        )

    def ideal_mixing_entropy_second_order(self):
        return sum(
            ideal_mixing_entropy_second_order(
                self.phi_grid.phi_values[i],
                self.phi_grid.delta_phi_i[i],
                self.block_fractions[i],
            )
            for i in range(self.n_components)
        )

    def _compute_l_ij_matrix(self):
        # Abstract method (TODO: look into implementing abstractmethod?)
        pass


class ABCBlockCopolymer(BlockCopolymer):
    def __init__(
        self,
        n_components,
        block_lengths,
        block_fractions,
        chi_matrix,
        phi_grid,
        kuhn_length=1.0,
    ):
        super().__init__(
            n_components,
            block_lengths,
            block_fractions,
            chi_matrix,
            phi_grid,
            kuhn_length,
        )
        # Recompute l_ij_matrix with ABC-specific values (overrides parent's computation)
        self.l_ij_matrix = self._compute_l_ij_matrix()
        # Recompute gamma_q with the correct l_ij_matrix
        self.gamma_q = self.compute_gamma_q()

    def _compute_l_ij_matrix(self):
        l_ij_matrix = np.zeros((self.n_components, self.n_components))
        l_ij_matrix[0, 1] = 0
        l_ij_matrix[0, 2] = self.block_lengths[1] * self.kuhn_length**2
        # The matrix is symmetric
        l_ij_matrix = l_ij_matrix + l_ij_matrix.T
        return l_ij_matrix


def ideal_mixing_entropy(phi_i, f_i):
    num_points = np.prod(phi_i.shape)
    # Avoid log(0) by only computing where phi_i > 0
    mask = phi_i > 0
    integrand = np.zeros_like(phi_i)
    integrand[mask] = phi_i[mask] / f_i * np.log(phi_i[mask])
    return np.sum(integrand) / num_points


def h_ii(xi_i, f_i, N):
    return (2 * N * f_i**2) / (xi_i**2) * (np.exp(-1 * xi_i) - 1 + xi_i)


def h_ij(q, xi_i, xi_j, l_ij, f_i, f_j, N):
    """
    l_ij: distance connecting the two blocks i and j: l_ij = M_ij * b^2
    """
    prefactor = (N * f_i * f_j) / (xi_i * xi_j)
    return (
        prefactor
        * (np.exp(-1 * xi_i) - 1)
        * (np.exp(-1 * xi_j) - 1)
        * np.exp(-l_ij * q**2 / 6)
    )


def gamma_q(q, chi_matrix, l_ij_matrix, block_fractions, N, kuhn_length):
    num_components = chi_matrix.shape[0]
    structure_factor_matrix_ideal = np.zeros((num_components, num_components))
    # Compute the diagonal elements
    for i in range(num_components):
        # Compute Rouse variable: xi_i = q^2 * N * f_i / 6
        xi_i = q**2 * N * block_fractions[i] * kuhn_length**2 / 6
        structure_factor_matrix_ideal[i, i] = h_ii(xi_i, block_fractions[i], N)
    # Compute the off-diagonal elements
    for i in range(num_components):
        for j in range(num_components):
            if i != j:
                # Compute Rouse variables for both blocks
                xi_i = q**2 * N * block_fractions[i] * kuhn_length**2 / 6
                xi_j = q**2 * N * block_fractions[j] * kuhn_length**2 / 6
                structure_factor_matrix_ideal[i, j] = h_ij(
                    q,
                    xi_i,
                    xi_j,
                    l_ij_matrix[i, j],
                    block_fractions[i],
                    block_fractions[j],
                    N,
                )
    # Add the chi matrix
    gamma_matrix = np.linalg.inv(structure_factor_matrix_ideal) + chi_matrix

    return gamma_matrix


def ideal_mixing_entropy_second_order(phi_i, delta_phi_i, f_i):
    """
    Compute the second-order term in the expansion of ideal mixing entropy.
    The second-order expansion of phi * log(phi) around f is: delta_phi^2 / (2 * f * phi)
    """
    num_points = np.prod(delta_phi_i.shape)
    # Second-order term: delta_phi^2 / (2 * f_i * phi_i)
    # Avoid division by zero where phi_i = 0
    mask = phi_i > 0
    integrand = np.zeros_like(delta_phi_i)
    integrand[mask] = (delta_phi_i[mask]) ** 2 / (2 * phi_i[mask])
    return np.sum(integrand) / num_points
