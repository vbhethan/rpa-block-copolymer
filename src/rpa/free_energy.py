"""
Block Copolymer Free Energy Module with Arbitrary Architectures

A PyTorch module for computing and optimizing the free energy functional
of block copolymer systems using the Random Phase Approximation (RPA).
Supports arbitrary chain architectures through distance matrices.
Supports 1D, 2D, and 3D spatial grids.
"""

import math

import torch
import torch.nn as nn

_REAL_TO_COMPLEX = {torch.float32: torch.complex64, torch.float64: torch.complex128}


class BlockCopolymerFreeEnergy(nn.Module):
    """
    Base class for block copolymer free energy with arbitrary architectures.

    The free energy functional consists of:
    - Interaction energy (computed in Fourier space using RPA)
    - Mixing entropy (real space)
    - 2nd order mixing entropy expansion (real space)

    This module supports arbitrary chain architectures through the distance
    matrix l_ij, which encodes the effective squared path length between blocks
    i and j: l_ij_matrix[i,j] = N * sum_{k on path i→j} f_k * b_k^2.
    For conformationally symmetric chains (all b_k equal) this reduces to
    l_ij * N * b^2 where l_ij is the dimensionless contour fraction.

    Physical constraints:
    1. Local incompressibility: sum_i rho_i(r) = phi_bar at every point
       This implies one component is determined by others.
    2. Global zero-mean: integral of phi_i over space must be zero for all i

    Parameters
    ----------
    N : int
        Total number of monomers in the polymer chain
    b : list of float, optional
        Kuhn lengths per block, length must equal n_components.
        Defaults to [1.0, ..., 1.0] when None.
    block_fractions : list or array
        Volume fractions of each block [f_A, f_B, ...], must sum to 1
    chi_matrix : array-like
        Flory-Huggins interaction parameter matrix (n x n)
        chi_ij represents the interaction between species i and j
    phi_bar : float
        Mean density of the polymer melt
    grid_shape : tuple of int
        Number of grid points along each spatial dimension.
        E.g. (128,) for 1D, (64, 64) for 2D, (32, 32, 32) for 3D.
    box_lengths : tuple of float, optional
        Box lengths along each spatial dimension (same length as grid_shape).
        Defaults to grid_shape[i] * b/2 for each dimension.
    init_amplitude : float, optional
        Amplitude of random initial fluctuations (default: 0.01)
    optimize_box : bool, optional
        If True, make the box lengths learnable parameters (default: False)
    """

    def __init__(
        self,
        N: int = 100,
        b: list[float] | None = None,
        block_fractions: list = None,
        chi_matrix: torch.Tensor = None,
        l_ij_matrix: torch.Tensor = None,
        phi_bar: float = 1.0,
        grid_shape: tuple[int, ...] = (64, 64),
        box_lengths: tuple[float, ...] | None = None,
        init_amplitude: float = 0.01,
        optimize_box: bool = False,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        if dtype not in _REAL_TO_COMPLEX:
            raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}")
        self.real_dtype = dtype
        self.complex_dtype = _REAL_TO_COMPLEX[dtype]

        # Validate inputs
        if block_fractions is None:
            raise ValueError("block_fractions must be provided")
        if chi_matrix is None:
            raise ValueError("chi_matrix must be provided")
        if l_ij_matrix is None:
            raise ValueError("l_ij_matrix must be provided")

        block_fractions = torch.as_tensor(block_fractions, dtype=self.real_dtype)
        chi_matrix = torch.as_tensor(chi_matrix, dtype=self.real_dtype)
        l_ij_matrix = torch.as_tensor(l_ij_matrix, dtype=self.real_dtype)

        # Validate block fractions sum to 1
        if not torch.allclose(block_fractions.sum(), block_fractions.new_tensor(1.0)):
            raise ValueError(
                f"block_fractions must sum to 1, got {block_fractions.sum()}"
            )
        if torch.any(block_fractions < 0):
            raise ValueError("block_fractions must be strictly positive")

        n_components = int(len(block_fractions))
        expected_shape = (n_components, n_components)
        if chi_matrix.shape != expected_shape:
            raise ValueError(
                f"chi_matrix must have shape {expected_shape}, got {chi_matrix.shape}"
            )
        if l_ij_matrix.shape != expected_shape:
            raise ValueError(
                f"l_ij_matrix must have shape {expected_shape}, got {l_ij_matrix.shape}"
            )
        if not torch.allclose(chi_matrix, chi_matrix.T):
            raise ValueError("chi_matrix must be symmetric")
        if not torch.allclose(l_ij_matrix, l_ij_matrix.T):
            raise ValueError("l_ij_matrix must be symmetric")

        # Store physical parameters
        self.N = N
        self.n_components = n_components
        self.phi_bar = phi_bar

        # Normalize and validate per-block Kuhn lengths
        if b is None:
            b_list = [1.0] * n_components
        else:
            b_list = [float(x) for x in b]
        if len(b_list) != n_components:
            raise ValueError(
                f"b must have length {n_components}, got {len(b_list)}"
            )
        if any(x <= 0 for x in b_list):
            raise ValueError("all entries of b must be positive")
        self.register_buffer("b", torch.tensor(b_list, dtype=self.real_dtype))

        # Store block fractions as buffer
        f_vec = block_fractions.clone().detach()
        self.register_buffer("f_vec", f_vec)

        # Store chi matrix as buffer (already scaled by N in input)
        chi_tensor = chi_matrix.clone().detach()
        self.register_buffer("chi_matrix", chi_tensor)

        # Grid parameters (dimension-agnostic)
        self.grid_shape = tuple(grid_shape)
        self.ndim = len(grid_shape)
        self.n_grid_points = math.prod(grid_shape)
        # Spatial dims of delta_phi tensor: axes 1..ndim (axis 0 is component)
        self.spatial_dims = tuple(range(1, self.ndim + 1))

        # Store grid shape as a buffer for convenience in computations
        self.register_buffer(
            "_grid_shape_tensor",
            torch.tensor(grid_shape, dtype=self.real_dtype),
        )

        # Compute initial box lengths
        if box_lengths is not None:
            if len(box_lengths) != self.ndim:
                raise ValueError(
                    f"box_lengths must have {self.ndim} entries to match "
                    f"grid_shape {grid_shape}, got {len(box_lengths)}"
                )
            self._L_init = tuple(float(v) for v in box_lengths)
        else:
            b_rms = float((self.b**2 @ self.f_vec).sqrt())
            self._L_init = tuple(float(n * b_rms / 2) for n in grid_shape)

        # Box optimization flag
        self.optimize_box = optimize_box

        if optimize_box:
            self.log_L = nn.Parameter(
                torch.tensor([math.log(v) for v in self._L_init], dtype=self.real_dtype)
            )
        else:
            self.register_buffer(
                "_log_L_fixed",
                torch.tensor(
                    [math.log(v) for v in self._L_init], dtype=self.real_dtype
                ),
            )

        # Pre-compute 1D normalized frequency vectors for each spatial axis
        for d, n in enumerate(grid_shape):
            freq = torch.fft.fftfreq(n, dtype=self.real_dtype)
            self.register_buffer(f"_kfreq_{d}", freq)

        # Register architecture distance matrix
        self.register_buffer("l_ij_matrix", l_ij_matrix.clone().detach())

        # Pre-compute and cache Gamma_ij if box is not optimized
        if not optimize_box:
            L_init_tensor = torch.tensor(self._L_init, dtype=self.real_dtype)
            K2 = self._compute_K2(L_init_tensor)
            Gamma_ij = self._compute_gamma_ij(K2)
            self.register_buffer("_Gamma_ij_cached", Gamma_ij)

        # Initialize learnable order parameter field delta_phi_i(r)
        # for all components. Constraints are enforced through projection
        # in _project_order_parameter.
        delta_phi_init = (
            torch.randn(self.n_components, *grid_shape, dtype=self.real_dtype)
            * init_amplitude
        )
        delta_phi_init = delta_phi_init - delta_phi_init.mean(
            dim=self.spatial_dims, keepdim=True
        )
        delta_phi_init = delta_phi_init - delta_phi_init.mean(dim=0, keepdim=True)
        self.delta_phi = nn.Parameter(delta_phi_init)

        self.to(device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def L(self) -> torch.Tensor:
        """Get current box lengths, shape (ndim,)."""
        if self.optimize_box:
            return torch.exp(self.log_L)
        else:
            return torch.exp(self._log_L_fixed)

    @property
    def spacing(self) -> torch.Tensor:
        """Get current grid spacings, shape (ndim,)."""
        return self.L / self._grid_shape_tensor

    @property
    def cell_vol(self) -> torch.Tensor:
        """Get volume of a single grid cell (product of spacings)."""
        return self.spacing.prod()

    @property
    def vol(self) -> torch.Tensor:
        """Get current box volume (product of all box lengths)."""
        return self.L.prod()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _f_expanded(self) -> torch.Tensor:
        """Return f_vec reshaped for broadcasting over spatial dims: (n, 1, ..., 1)."""
        return self.f_vec.view(-1, *([1] * self.ndim))

    def _compute_K2(self, L: torch.Tensor) -> torch.Tensor:
        """
        Compute k-space grid K2 = sum_d k_d^2 for given box dimensions.

        Parameters
        ----------
        L : torch.Tensor
            Box lengths, shape (ndim,)

        Returns
        -------
        K2 : torch.Tensor
            k-squared grid of shape (*grid_shape)
        """
        K2 = torch.zeros(self.grid_shape, dtype=self.real_dtype, device=L.device)
        for d in range(self.ndim):
            freq = getattr(self, f"_kfreq_{d}")
            k_d = 2 * math.pi * freq * self.grid_shape[d] / L[d]
            shape = [1] * self.ndim
            shape[d] = self.grid_shape[d]
            K2 = K2 + k_d.view(shape) ** 2
        return K2

    def _compute_gamma_ij(self, K2: torch.Tensor) -> torch.Tensor:
        """
        Compute the vertex function Gamma_ij(q) for all q vectors.

        Gamma_ij = (S^ideal_ij)^-1 + chi_ij

        where S^ideal is the ideal structure factor matrix computed from
        Debye functions with architecture-dependent distance factors.

        Parameters
        ----------
        K2 : torch.Tensor
            k-squared grid of shape (*grid_shape)

        Returns
        -------
        Gamma_ij : torch.Tensor
            Vertex function of shape (*grid_shape, n_components, n_components)
        """
        N = self.N
        n = self.n_components
        f = self.f_vec  # (n,)
        phi_bar = self.phi_bar
        ndim = self.ndim

        # Compute xi parameters for each component: xi_i = N * f_i * b_i^2 * K2 / 6
        # self.b has shape (n,); K2.unsqueeze(-1) has shape (*grid_shape, 1) → broadcasts to (*grid_shape, n)
        xi = K2.unsqueeze(-1) * N * f * self.b**2 / 6

        # Numerically stable small-x forms:
        # u(x) = (1 - exp(-x)) / x, u(0) = 1
        # v(x) = 2 * (exp(-x) - 1 + x) / x^2, v(0) = 1
        eps = 1e-4 if self.real_dtype == torch.float32 else 1e-8
        xi_safe = torch.where(xi > eps, xi, torch.ones_like(xi))
        u = torch.where(
            xi > eps,
            (1 - torch.exp(-xi)) / xi_safe,
            1 - xi / 2 + xi**2 / 6,
        )
        v = torch.where(
            xi > eps,
            2 * (torch.exp(-xi) - 1 + xi) / (xi_safe**2),
            1 - xi / 3 + xi**2 / 12,
        )

        # Diagonal Debye terms
        h_diag = N * f**2 * v

        # Compute off-diagonal terms h_ij(q) with distance factors
        # Shape: (*grid_shape, n, n)
        u_i = u.unsqueeze(-1)  # (*grid_shape, n, 1)
        u_j = u.unsqueeze(-2)  # (*grid_shape, 1, n)

        f_i = f.view(*([1] * ndim), -1, 1)  # (1,...,1, n, 1)
        f_j = f.view(*([1] * ndim), 1, -1)  # (1,...,1, 1, n)

        # Distance factor: exp(-l_ij * N * b^2 * K2 / 6)
        K2_expanded = K2.unsqueeze(-1).unsqueeze(-1)  # (*grid_shape, 1, 1)
        distance_factor = torch.exp(-self.l_ij_matrix * K2_expanded / 6)

        # Off-diagonal Debye terms with stable low-q limit
        h_matrix = N * f_i * f_j * u_i * u_j * distance_factor

        # Insert diagonal values
        eye = torch.eye(n, device=K2.device, dtype=torch.bool)
        eye = eye.view(*([1] * ndim), n, n).expand(*self.grid_shape, -1, -1)

        h_diag_expanded = torch.diag_embed(h_diag)  # (*grid_shape, n, n)

        h_full = torch.where(eye, h_diag_expanded, h_matrix)

        # Keep the structure factor Hermitian/symmetric before inversion.
        h_full = 0.5 * (h_full + h_full.transpose(-1, -2))

        # Construct ideal structure factor matrix: S^ideal_ij = (phi_bar / N) * h_ij
        S_ideal = (phi_bar / N) * h_full

        # Handle k=0 mode specially to avoid singular matrix
        k0_idx = (0,) * ndim
        S_ideal[k0_idx] = torch.eye(n, device=K2.device)

        # Compute inverse: Gamma^ideal_ij = (S^ideal)^-1
        Gamma_ideal_ij = torch.linalg.inv(S_ideal)

        # Add chi interaction: Gamma_ij = Gamma^ideal_ij - 2*chi_ij
        chi_expanded = self.chi_matrix.view(*([1] * ndim), n, n)
        Gamma_ij = Gamma_ideal_ij + chi_expanded

        # Convert to complex for Fourier space operations
        Gamma_ij = Gamma_ij.to(self.complex_dtype)

        # Zero out k=0 mode (will be excluded from sum anyway)
        Gamma_ij[(*k0_idx, slice(None), slice(None))] = 0

        return Gamma_ij

    def _project_order_parameter(self, delta_phi: torch.Tensor) -> torch.Tensor:
        """
        Project order parameter onto physical constraint manifold.

        Enforces:
        1) global zero-mean for each component
        2) local incompressibility sum_i delta_phi_i(r) = 0

        Parameters
        ----------
        delta_phi : torch.Tensor
            Input order parameter, shape (n_components, *grid_shape)

        Returns
        -------
        torch.Tensor
            Projected order parameter with same shape as input
        """
        expected_shape = (self.n_components, *self.grid_shape)
        if tuple(delta_phi.shape) != expected_shape:
            raise ValueError(
                f"delta_phi must have shape {expected_shape}, got {tuple(delta_phi.shape)}"
            )

        # Global zero-mean for each component (average over spatial dims)
        delta_phi = delta_phi - delta_phi.mean(dim=self.spatial_dims, keepdim=True)
        # Local incompressibility (sum over components equals zero pointwise)
        delta_phi = delta_phi - delta_phi.mean(dim=0, keepdim=True)
        return delta_phi

    def _get_order_parameter(
        self, delta_phi: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Return projected order parameter field used in free-energy calculation.

        If delta_phi is not provided, uses the internal learnable field.
        """
        if delta_phi is None:
            delta_phi = self.delta_phi
        return self._project_order_parameter(delta_phi)

    def forward(
        self,
        delta_phi: torch.Tensor | None = None,
        Gamma_ij: torch.Tensor | None = None,
        project: bool = True,
    ) -> torch.Tensor:
        """
        Compute the total free energy of the system.

        Parameters
        ----------
        delta_phi : torch.Tensor, optional
            Order parameter field with shape (n_components, *grid_shape).
            If None, uses the module's internal learnable `self.delta_phi`.
        Gamma_ij : torch.Tensor, optional
            Pre-computed vertex function. If provided, skips recomputation.
            Useful when the box dimensions are held fixed across many calls.
        project : bool
            If False, skip projection (caller guarantees field is already on
            the constraint manifold). Avoids redundant work in tight loops.

        Returns
        -------
        F_total : torch.Tensor
            Scalar tensor containing the total free energy
        """
        if project:
            delta_phi = self._get_order_parameter(delta_phi)
        elif delta_phi is None:
            delta_phi = self.delta_phi

        # Get current Gamma_ij
        if Gamma_ij is None:
            if self.optimize_box:
                K2 = self._compute_K2(self.L)
                Gamma_ij = self._compute_gamma_ij(K2)
            else:
                Gamma_ij = self._Gamma_ij_cached

        # Compute interaction energy in Fourier space
        Delta_F_int = self._compute_interaction_energy(delta_phi, Gamma_ij)

        # Compute mixing entropy terms in real space
        F_mixing = self._compute_mixing_entropy(delta_phi)
        F_mixing_2 = self._compute_mixing_entropy_quadratic(delta_phi)

        # Total free energy
        F_total = Delta_F_int + F_mixing - F_mixing_2

        return F_total

    def _compute_interaction_energy(
        self, delta_phi: torch.Tensor, Gamma_ij: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the interaction energy Delta F_int in Fourier space.

        Delta F_int = (1/2N) sum_{k != 0} Gamma_ij(k) * phi_i(k) * phi_j(-k)

        Parameters
        ----------
        delta_phi : torch.Tensor
            Order parameters for all components, shape (n_components, *grid_shape)
        Gamma_ij : torch.Tensor
            Vertex function, shape (*grid_shape, n_components, n_components)

        Returns
        -------
        Delta_F_int : torch.Tensor
            Scalar interaction energy
        """
        # N-dimensional Fourier transform over spatial dims
        phi_hat = torch.fft.fftn(delta_phi, dim=self.spatial_dims)

        # Move component dim to last: (*grid_shape, n_components)
        phi_hat = phi_hat.movedim(0, -1)

        # Flatten spatial dims for a single batched einsum
        M = self.n_grid_points
        n = self.n_components
        phi_flat = phi_hat.reshape(M, n)
        Gamma_flat = Gamma_ij.reshape(M, n, n)

        # Quadratic form per k-point
        Delta_F_int_k = 0.5 * torch.einsum(
            "mi,mij,mj->m", phi_flat, Gamma_flat, phi_flat.conj()
        )

        # Normalize: Fourier coefficients c_k = X[k] / n_grid_points,
        # so per-unit-volume quadratic form picks up 1/n_grid_points^2.
        # Delta_F_int = (
        #     (1 / self.N)
        #     * self.cell_vol
        #     / self.n_grid_points
        #     / self.vol
        #     * Delta_F_int_k.sum()
        # ).real

        Delta_F_int = Delta_F_int_k.sum().real / self.n_grid_points**2

        return Delta_F_int

    def _compute_mixing_entropy(self, delta_phi: torch.Tensor) -> torch.Tensor:
        """
        Compute the mixing entropy (Flory-Huggins reference free energy).

        F_mixing = (1/NV) sum_i integral phi_i(r) / f_i * ln(phi_i(r)) dr

        Parameters
        ----------
        delta_phi : torch.Tensor
            Order parameters for all components, shape (n_components, *grid_shape)

        Returns
        -------
        F_mixing : torch.Tensor
            Scalar mixing entropy
        """
        f_expanded = self._f_expanded()
        rho = delta_phi + f_expanded * self.phi_bar

        integrand = rho / f_expanded * torch.log(rho)
        # F_mixing = (1 / self.N) * (1 / self.vol) * integrand.sum() * self.cell_vol
        F_mixing = (1 / self.vol) * integrand.sum() * self.cell_vol

        return F_mixing

    def _compute_mixing_entropy_quadratic(
        self, delta_phi: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the 2nd order expansion of mixing entropy F_mix^(2).

        The entropy integrand is h(delta) = (rho/f)*ln(rho) where rho = f*phi_bar + delta.
        Its second derivative at delta=0 is h''(0) = 1/(f^2 * phi_bar), giving:

        F_mix^(2) = (1/NV) sum_i integral (delta_phi_i)^2 / (2 * f_i^2 * phi_bar) dr

        This quadratic term is subtracted to avoid double-counting with the
        ideal chain contribution already in Gamma_ideal.

        Parameters
        ----------
        delta_phi : torch.Tensor
            Order parameters for all components, shape (n_components, *grid_shape)

        Returns
        -------
        F_mixing_2 : torch.Tensor
            Scalar quadratic mixing entropy
        """
        f_expanded = self._f_expanded()

        integrand = delta_phi**2 / (2 * f_expanded**2 * self.phi_bar)
        # F_mixing_2 = (1 / self.N) * (1 / self.vol) * integrand.sum() * self.cell_vol
        F_mixing_2 = (1 / self.vol) * integrand.sum() * self.cell_vol

        return F_mixing_2

    def set_density(self, phi: torch.Tensor) -> None:
        """
        Set the density profile directly from physical densities phi_i(r).

        Computes delta_phi = phi - f_i * phi_bar and assigns it as the learnable
        parameter.  The model's projection step will enforce zero-mean and
        incompressibility constraints on the next forward pass.

        Parameters
        ----------
        phi : torch.Tensor
            Density profiles, shape (n_components, *grid_shape), values in [0, 1].
        """
        expected_shape = (self.n_components, *self.grid_shape)
        if tuple(phi.shape) != expected_shape:
            raise ValueError(
                f"phi must have shape {expected_shape}, got {tuple(phi.shape)}"
            )
        delta_phi = phi.to(self.real_dtype).to(self.f_vec.device) - self._f_expanded() * self.phi_bar
        self.delta_phi = nn.Parameter(delta_phi)

    def get_densities(self, delta_phi: torch.Tensor | None = None) -> torch.Tensor:
        """
        Get the current density profiles from order parameters.

        Enforces local incompressibility exactly: rho_0 + ... + rho_{n-1} = phi_bar
        at every point. The first n_components-1 densities are computed as
        rho_i = phi_i + f_i*phi_bar; the last is rho_last = phi_bar - sum(rho_i)
        so the sum is exact (no floating-point drift).

        Returns
        -------
        rho : torch.Tensor
            Density profiles for all components, shape (n_components, *grid_shape)
        """
        with torch.no_grad():
            delta_phi = self._get_order_parameter(delta_phi)
            f_expanded = self._f_expanded()
            rho = delta_phi + f_expanded * self.phi_bar
            n = self.n_components
            rho = torch.cat(
                [rho[: n - 1], (self.phi_bar - rho[: n - 1].sum(dim=0, keepdim=True))],
                dim=0,
            )
        return rho

    def get_order_parameters(
        self, delta_phi: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Get the current order parameters (constraints enforced).

        Returns
        -------
        delta_phi : torch.Tensor
            Order parameters for all components, shape (n_components, *grid_shape)
        """
        with torch.no_grad():
            projected = self._get_order_parameter(delta_phi)
        return projected.detach().clone()

    def get_energy_components(
        self, delta_phi: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Get individual energy components for analysis.

        Returns
        -------
        dict with keys 'Delta_F_int', 'F_mixing', 'F_mixing_2', 'F_total'
        """
        with torch.no_grad():
            delta_phi = self._get_order_parameter(delta_phi)

            if self.optimize_box:
                K2 = self._compute_K2(self.L)
                Gamma_ij = self._compute_gamma_ij(K2)
            else:
                Gamma_ij = self._Gamma_ij_cached

            Delta_F_int = self._compute_interaction_energy(delta_phi, Gamma_ij)
            F_mixing = self._compute_mixing_entropy(delta_phi)
            F_mixing_2 = self._compute_mixing_entropy_quadratic(delta_phi)
            F_total = Delta_F_int + F_mixing - F_mixing_2

        return {
            "Delta_F_int": Delta_F_int,
            "F_mixing": F_mixing,
            "F_mixing_2": F_mixing_2,
            "F_total": F_total,
        }

    def check_constraints(
        self, delta_phi: torch.Tensor | None = None
    ) -> dict[str, float]:
        """
        Check that physical constraints are satisfied.

        Returns
        -------
        dict with constraint violation metrics:
            - 'phi_means': mean of each component (should all be ~0)
            - 'incompressibility_max_error': max |sum(rho_i) - phi_bar|
        """
        with torch.no_grad():
            delta_phi = self._get_order_parameter(delta_phi)
            rho = self.get_densities(delta_phi)

            phi_means = [delta_phi[i].mean().item() for i in range(self.n_components)]
            incomp_error = (rho.sum(dim=0) - self.phi_bar).abs().max().item()

            return {
                "phi_means": phi_means,
                "incompressibility_max_error": incomp_error,
            }

    def gradient_wrt_input(
        self, delta_phi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute free energy and gradient dF/d(delta_phi) for an input field.

        Parameters
        ----------
        delta_phi : torch.Tensor
            Input order parameter, shape (n_components, *grid_shape)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (free_energy, gradient) where gradient has same shape as input
        """
        delta_phi_var = delta_phi.clone().detach().requires_grad_(True)
        F_total = self.forward(delta_phi_var)
        grad = torch.autograd.grad(F_total, delta_phi_var)[0]
        return F_total, grad

    def get_box_dimensions(self) -> dict[str, float | list[float]]:
        """
        Get current box dimensions.

        Returns
        -------
        dict with 'L' (list), 'spacing' (list), 'vol' (float),
              'grid_shape', and 'ndim'
        """
        with torch.no_grad():
            return {
                "L": self.L.tolist(),
                "spacing": self.spacing.tolist(),
                "vol": self.vol.item(),
                "grid_shape": list(self.grid_shape),
                "ndim": self.ndim,
            }
