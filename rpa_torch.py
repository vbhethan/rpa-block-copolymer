"""
Block Copolymer Free Energy Module with Arbitrary Architectures

A PyTorch module for computing and optimizing the free energy functional
of block copolymer systems using the Random Phase Approximation (RPA).
Supports arbitrary chain architectures through distance matrices.
"""

# Fix OpenMP conflict on macOS (must be before importing torch/numpy)
import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import numpy as np
import torch
import torch.nn as nn


class BlockCopolymerFreeEnergy(nn.Module):
    """
    Base class for block copolymer free energy with arbitrary architectures.

    The free energy functional consists of:
    - Interaction energy (computed in Fourier space using RPA)
    - Mixing entropy (real space)
    - 2nd order mixing entropy expansion (real space)

    This module supports arbitrary chain architectures through the distance
    matrix l_ij, which encodes the contour distance between blocks i and j.

    Physical constraints:
    1. Local incompressibility: sum_i rho_i(r) = phi_bar at every point
       This implies one component is determined by others.
    2. Global zero-mean: integral of phi_i over space must be zero for all i

    Parameters
    ----------
    N : int
        Total number of monomers in the polymer chain
    b : float
        Kuhn length (statistical segment length)
    block_fractions : list or array
        Volume fractions of each block [f_A, f_B, ...], must sum to 1
    chi_matrix : array-like
        Flory-Huggins interaction parameter matrix (n x n)
        chi_ij represents the interaction between species i and j
    phi_bar : float
        Mean density of the polymer melt
    Nx : int
        Number of grid points in x direction
    Ny : int
        Number of grid points in y direction
    dx : float, optional
        Grid spacing in x direction (default: b/2)
    dy : float, optional
        Grid spacing in y direction (default: b/2)
    init_amplitude : float, optional
        Amplitude of random initial fluctuations (default: 0.01)
    optimize_box : bool, optional
        If True, make the box length a learnable parameter (default: False)
    """

    def __init__(
        self,
        N: int = 100,
        b: float = 1.0,
        block_fractions: list = None,
        chi_matrix: torch.Tensor = None,
        l_ij_matrix: torch.Tensor = None,
        phi_bar: float = 1.0,
        Nx: int = 64,
        Ny: int = 64,
        dx: float = None,
        dy: float = None,
        init_amplitude: float = 0.01,
        optimize_box: bool = False,
    ):
        super().__init__()
        self.real_dtype = torch.float64
        self.complex_dtype = torch.complex128

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
        self.b = b
        self.n_components = n_components
        self.phi_bar = phi_bar

        # Store block fractions as buffer
        f_vec = block_fractions.clone().detach()
        self.register_buffer("f_vec", f_vec)

        # Store chi matrix as buffer (already scaled by N in input)
        chi_tensor = chi_matrix.clone().detach()
        self.register_buffer("chi_matrix", chi_tensor)

        # Grid parameters
        self.Nx = Nx
        self.Ny = Ny
        self._dx_init = dx if dx is not None else b / 2
        self._dy_init = dy if dy is not None else b / 2
        self._Lx_init = float(Nx * self._dx_init)
        self._Ly_init = float(Ny * self._dy_init)

        # Box optimization flag
        self.optimize_box = optimize_box

        if optimize_box:
            # Learnable box length parameters (use log for numerical stability)
            self.log_Lx = nn.Parameter(
                torch.tensor(math.log(self._Lx_init), dtype=self.real_dtype)
            )
            self.log_Ly = nn.Parameter(
                torch.tensor(math.log(self._Ly_init), dtype=self.real_dtype)
            )
        else:
            # Fixed box length (not learnable)
            self.register_buffer(
                "_log_Lx_fixed",
                torch.tensor(math.log(self._Lx_init), dtype=self.real_dtype),
            )
            self.register_buffer(
                "_log_Ly_fixed",
                torch.tensor(math.log(self._Ly_init), dtype=self.real_dtype),
            )

        # Pre-compute k-space grid structure (normalized frequencies, independent of L)
        kx_norm = torch.fft.fftfreq(Nx, dtype=self.real_dtype)
        ky_norm = torch.fft.fftfreq(Ny, dtype=self.real_dtype)
        KX_norm, KY_norm = torch.meshgrid(kx_norm, ky_norm, indexing="ij")
        self.register_buffer("KX_norm", KX_norm)
        self.register_buffer("KY_norm", KY_norm)

        # Register architecture distance matrix
        self.register_buffer("l_ij_matrix", l_ij_matrix.clone().detach())

        # Pre-compute and cache Gamma_ij if box is not optimized
        if not optimize_box:
            K2 = self._compute_K2(self._Lx_init, self._Ly_init)
            Gamma_ij = self._compute_gamma_ij(K2)
            self.register_buffer("_Gamma_ij_cached", Gamma_ij)

        # Initialize learnable order parameter field delta_phi_i(r)
        # for all components. Constraints are enforced through projection
        # in _project_order_parameter.
        delta_phi_init = (
            torch.randn(self.n_components, Nx, Ny, dtype=self.real_dtype)
            * init_amplitude
        )
        delta_phi_init = delta_phi_init - delta_phi_init.mean(
            dim=(-2, -1), keepdim=True
        )
        delta_phi_init = delta_phi_init - delta_phi_init.mean(dim=0, keepdim=True)
        self.delta_phi = nn.Parameter(delta_phi_init)

    @property
    def Lx(self) -> torch.Tensor:
        """Get current box length in x direction."""
        if self.optimize_box:
            return torch.exp(self.log_Lx)
        else:
            return torch.exp(self._log_Lx_fixed)

    @property
    def Ly(self) -> torch.Tensor:
        """Get current box length in y direction."""
        if self.optimize_box:
            return torch.exp(self.log_Ly)
        else:
            return torch.exp(self._log_Ly_fixed)

    @property
    def dx(self) -> torch.Tensor:
        """Get current grid spacing in x direction."""
        return self.Lx / self.Nx

    @property
    def dy(self) -> torch.Tensor:
        """Get current grid spacing in y direction."""
        return self.Ly / self.Ny

    @property
    def vol(self) -> torch.Tensor:
        """Get current box volume (area in 2D)."""
        return self.Lx * self.Ly

    def _compute_K2(self, Lx: torch.Tensor, Ly: torch.Tensor) -> torch.Tensor:
        """
        Compute k-space grid K2 = kx^2 + ky^2 for given box dimensions.

        Parameters
        ----------
        Lx, Ly : torch.Tensor
            Box lengths in x and y directions

        Returns
        -------
        K2 : torch.Tensor
            k-squared grid of shape (Nx, Ny)
        """
        kx = 2 * torch.pi * self.KX_norm * self.Nx / Lx
        ky = 2 * torch.pi * self.KY_norm * self.Ny / Ly
        return kx**2 + ky**2

    def _compute_gamma_ij(self, K2: torch.Tensor) -> torch.Tensor:
        """
        Compute the vertex function Gamma_ij(q) for all q vectors.

        Gamma_ij = (S^ideal_ij)^-1 + chi_ij

        where S^ideal is the ideal structure factor matrix computed from
        Debye functions with architecture-dependent distance factors.

        Parameters
        ----------
        K2 : torch.Tensor
            k-squared grid of shape (Nx, Ny)

        Returns
        -------
        Gamma_ij : torch.Tensor
            Vertex function of shape (Nx, Ny, n_components, n_components)
        """
        N, b = self.N, self.b
        n = self.n_components
        f = self.f_vec  # (n,)
        phi_bar = self.phi_bar

        # Compute xi parameters for each component: xi_i = N * f_i * b^2 * K2 / 6
        # Shape: (Nx, Ny, n)
        xi = K2.unsqueeze(-1) * N * f * b**2 / 6

        # Numerically stable small-x forms:
        # u(x) = (1 - exp(-x)) / x, u(0) = 1
        # v(x) = 2 * (exp(-x) - 1 + x) / x^2, v(0) = 1
        # We use short Taylor expansions for tiny x to avoid 0/0 and low-q artifacts.
        eps = 1e-8
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
        # h_ij = (N * f_i * f_j) / (xi_i * xi_j) * (exp(-xi_i)-1) * (exp(-xi_j)-1) * exp(-l_ij * K2 / 6)
        # Shape: (Nx, Ny, n, n)
        u_i = u.unsqueeze(-1)  # (Nx, Ny, n, 1)
        u_j = u.unsqueeze(-2)  # (Nx, Ny, 1, n)

        f_i = f.view(1, 1, -1, 1)  # (1, 1, n, 1)
        f_j = f.view(1, 1, 1, -1)  # (1, 1, 1, n)

        # Distance factor: exp(-l_ij * K2 / 6)
        # l_ij_matrix is (n, n), K2 is (Nx, Ny)
        K2_expanded = K2.unsqueeze(-1).unsqueeze(-1)  # (Nx, Ny, 1, 1)
        distance_factor = torch.exp(-self.l_ij_matrix * K2_expanded / 6)

        # Off-diagonal Debye terms with stable low-q limit
        h_matrix = N * f_i * f_j * u_i * u_j * distance_factor

        # Insert diagonal values
        # Create identity mask for diagonal
        eye = torch.eye(n, device=K2.device, dtype=torch.bool)
        eye = eye.view(1, 1, n, n).expand(self.Nx, self.Ny, -1, -1)

        # Expand h_diag to match h_matrix shape for diagonal insertion
        h_diag_expanded = torch.diag_embed(h_diag)  # (Nx, Ny, n, n)

        # Combine: use h_diag for diagonal, h_matrix for off-diagonal
        h_full = torch.where(eye, h_diag_expanded, h_matrix)

        # Keep the structure factor Hermitian/symmetric before inversion.
        h_full = 0.5 * (h_full + h_full.transpose(-1, -2))

        # Construct ideal structure factor matrix: S^ideal_ij = (phi_bar / N) * h_ij
        S_ideal = (phi_bar / N) * h_full

        # Handle k=0 mode specially to avoid singular matrix
        # Set S_ideal[0,0] to identity to make it invertible
        S_ideal[0, 0] = torch.eye(n, device=K2.device)

        # Compute inverse: Gamma^ideal_ij = (S^ideal)^-1
        Gamma_ideal_ij = torch.linalg.inv(S_ideal)

        # Add chi interaction: Gamma_ij = Gamma^ideal_ij + chi_ij
        Gamma_ij = Gamma_ideal_ij + self.chi_matrix.unsqueeze(0).unsqueeze(0)

        # Convert to complex for Fourier space operations
        Gamma_ij = Gamma_ij.to(self.complex_dtype)

        # Zero out k=0 mode (will be excluded from sum anyway)
        Gamma_ij[0, 0, :, :] = 0

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
            Input order parameter, shape (n_components, Nx, Ny)

        Returns
        -------
        torch.Tensor
            Projected order parameter with same shape as input
        """
        expected_shape = (self.n_components, self.Nx, self.Ny)
        if tuple(delta_phi.shape) != expected_shape:
            raise ValueError(
                f"delta_phi must have shape {expected_shape}, got {tuple(delta_phi.shape)}"
            )

        # Global zero-mean for each component
        delta_phi = delta_phi - delta_phi.mean(dim=(-2, -1), keepdim=True)
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
    ) -> torch.Tensor:
        """
        Compute the total free energy of the system.

        Parameters
        ----------
        delta_phi : torch.Tensor, optional
            Order parameter field with shape (n_components, Nx, Ny).
            If None, uses the module's internal learnable `self.delta_phi`.
        Gamma_ij : torch.Tensor, optional
            Pre-computed vertex function. If provided, skips recomputation.
            Useful when the box dimensions are held fixed across many calls.

        Returns
        -------
        F_total : torch.Tensor
            Scalar tensor containing the total free energy
        """
        # Get order parameters (projected to satisfy physical constraints)
        delta_phi = self._get_order_parameter(delta_phi)

        # Get current Gamma_ij
        if Gamma_ij is None:
            if self.optimize_box:
                K2 = self._compute_K2(self.Lx, self.Ly)
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
            Order parameters for all components, shape (n_components, Nx, Ny)
        Gamma_ij : torch.Tensor
            Vertex function, shape (Nx, Ny, n_components, n_components)

        Returns
        -------
        Delta_F_int : torch.Tensor
            Scalar interaction energy
        """
        # Fourier transform of order parameters: shape (n_components, Nx, Ny)
        phi_hat = torch.fft.fft2(delta_phi)

        # Rearrange to (Nx, Ny, n_components) for einsum
        phi_hat = phi_hat.permute(1, 2, 0)

        # Compute quadratic form: sum over k of phi_i(k) Gamma_ij(k) phi_j(-k)
        # Note: phi(-k) = phi(k)^* for real phi
        Delta_F_int_k = 0.5 * torch.einsum(
            "xyi,xyij,xyj->xy", phi_hat, Gamma_ij, phi_hat.conj()
        )

        # Sum over k-space (k=0 is already zeroed in Gamma_ij)
        # Normalize by grid factors and include 1/N prefactor
        Delta_F_int = (
            (1 / self.N)
            * (self.dx * self.dy)
            / (self.Nx * self.Ny)
            * Delta_F_int_k.sum()
        ).real

        return Delta_F_int

    def _compute_mixing_entropy(self, delta_phi: torch.Tensor) -> torch.Tensor:
        """
        Compute the mixing entropy (Flory-Huggins reference free energy).

        F_mixing = (1/NV) sum_i integral phi_i(r) / f_i * ln(phi_i(r)) dr

        Parameters
        ----------
        delta_phi : torch.Tensor
            Order parameters for all components, shape (n_components, Nx, Ny)

        Returns
        -------
        F_mixing : torch.Tensor
            Scalar mixing entropy
        """
        # Convert order parameters to volume fractions (densities)
        # rho_i(r) = phi_i(r) + f_i * phi_bar
        f_expanded = self.f_vec.view(-1, 1, 1)  # (n, 1, 1)
        rho = delta_phi + f_expanded * self.phi_bar

        # Clamp to avoid log of negative numbers during optimization
        # TODO: check if this is messing with gradients, going to try turning off for now
        # rho = torch.clamp(rho, min=1e-10)

        # Compute mixing entropy integral with 1/N prefactor
        # Sum over components and spatial points
        integrand = rho / f_expanded * torch.log(rho)
        F_mixing = (1 / self.N) * (1 / self.vol) * integrand.sum() * self.dx * self.dy

        return F_mixing

    def _compute_mixing_entropy_quadratic(
        self, delta_phi: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the 2nd order expansion of mixing entropy F_mix^(2).

        F_mix^(2) = (1/NV) sum_i integral (phi_i)^2 / (2 f_i) dr

        This quadratic term is subtracted to avoid double-counting with the
        ideal chain contribution already in Gamma_ideal.

        Parameters
        ----------
        delta_phi : torch.Tensor
            Order parameters for all components, shape (n_components, Nx, Ny)

        Returns
        -------
        F_mixing_2 : torch.Tensor
            Scalar quadratic mixing entropy
        """
        f_expanded = self.f_vec.view(-1, 1, 1)  # (n, 1, 1)

        # Compute quadratic term
        integrand = delta_phi**2 / (2 * f_expanded)
        F_mixing_2 = (1 / self.N) * (1 / self.vol) * integrand.sum() * self.dx * self.dy

        return F_mixing_2

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
            Density profiles for all components, shape (n_components, Nx, Ny)
        """
        with torch.no_grad():
            delta_phi = self._get_order_parameter(delta_phi)
            f_expanded = self.f_vec.view(-1, 1, 1)
            rho = delta_phi + f_expanded * self.phi_bar
            # Enforce exact incompressibility: last component = phi_bar - sum(others)
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
            Order parameters for all components, shape (n_components, Nx, Ny)
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
                K2 = self._compute_K2(self.Lx, self.Ly)
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
            Input order parameter, shape (n_components, Nx, Ny)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (free_energy, gradient) where gradient has same shape as input
        """
        delta_phi_var = delta_phi.clone().detach().requires_grad_(True)
        F_total = self.forward(delta_phi_var)
        grad = torch.autograd.grad(F_total, delta_phi_var)[0]
        return F_total, grad

    def get_box_dimensions(self) -> dict[str, float]:
        """
        Get current box dimensions.

        Returns
        -------
        dict with 'Lx', 'Ly', 'dx', 'dy', 'vol'
        """
        with torch.no_grad():
            return {
                "Lx": self.Lx.item(),
                "Ly": self.Ly.item(),
                "dx": self.dx.item(),
                "dy": self.dy.item(),
                "vol": self.vol.item(),
            }
