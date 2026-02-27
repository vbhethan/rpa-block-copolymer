"""
Simulation IO — HDF5-backed storage for system definitions and trajectories.

SimulationData is a pure data container holding:
  1. Physical system definition (everything needed to construct a
     BlockCopolymerFreeEnergy model).
  2. Trajectory output — a flat sequence of frames, each with phi, F,
     and box_lengths.

It does NOT store simulation run parameters (dt, M, method, etc.).
Scripts are free to write additional attributes into the HDF5 file
after calling to_hdf5().
"""

from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np
import torch

from rpa import BlockCopolymerFreeEnergy


@dataclass
class SimulationData:
    """
    Container for block copolymer system definition and trajectory data.

    Attributes — system definition
    -------------------------------
    N : int
        Total monomers per chain.
    b : float
        Kuhn length.
    block_fractions : np.ndarray, shape (n_components,)
    chi_matrix : np.ndarray, shape (n_components, n_components)
    l_ij_matrix : np.ndarray, shape (n_components, n_components)
    phi_bar : float
        Mean density.
    grid_shape : tuple[int, ...]
        Spatial grid resolution, e.g. (32, 32).

    Attributes — trajectory (all indexed on axis 0 by frame)
    ---------------------------------------------------------
    phi : np.ndarray, shape (n_frames, n_components, *grid_shape)
    F : np.ndarray, shape (n_frames,)
    box_lengths : np.ndarray, shape (n_frames, ndim)
    """

    # --- system definition ---
    N: int = 100
    b: float = 1.0
    block_fractions: np.ndarray = field(default_factory=lambda: np.empty(0))
    chi_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    l_ij_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    phi_bar: float = 1.0
    grid_shape: tuple[int, ...] = (64, 64)

    # --- trajectory ---
    phi: np.ndarray = field(default_factory=lambda: np.empty(0))
    F: np.ndarray = field(default_factory=lambda: np.empty(0))
    box_lengths: np.ndarray = field(default_factory=lambda: np.empty(0))

    # --- output metadata ---
    converged: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Number of stored trajectory frames."""
        if self.phi.shape[0] == 0:
            return 0
        return int(self.phi.shape[0])

    @property
    def n_components(self) -> int:
        return int(self.block_fractions.shape[0])

    @property
    def ndim(self) -> int:
        return len(self.grid_shape)

    # ------------------------------------------------------------------
    # Factory: from a live model
    # ------------------------------------------------------------------

    @classmethod
    def from_model(cls, model: BlockCopolymerFreeEnergy) -> SimulationData:
        """
        Extract the physical system definition from a BlockCopolymerFreeEnergy.

        The trajectory arrays are left empty (zero frames).
        """
        n = model.n_components
        ndim = model.ndim
        return cls(
            N=model.N,
            b=model.b,
            block_fractions=model.f_vec.detach().cpu().numpy().copy(),
            chi_matrix=model.chi_matrix.detach().cpu().numpy().copy(),
            l_ij_matrix=model.l_ij_matrix.detach().cpu().numpy().copy(),
            phi_bar=model.phi_bar,
            grid_shape=model.grid_shape,
            phi=np.empty((0, n, *model.grid_shape), dtype=np.float64),
            F=np.empty((0,), dtype=np.float64),
            box_lengths=np.empty((0, ndim), dtype=np.float64),
            converged=False,
        )

    # ------------------------------------------------------------------
    # Factory: from HDF5
    # ------------------------------------------------------------------

    @classmethod
    def from_hdf5(cls, path: str) -> SimulationData:
        """Load system definition and trajectory from an HDF5 file."""
        with h5py.File(path, "r") as f:
            grid_shape = tuple(int(v) for v in f.attrs["grid_shape"])

            data = cls(
                N=int(f.attrs["N"]),
                b=float(f.attrs["b"]),
                block_fractions=np.array(f["block_fractions"], dtype=np.float64),
                chi_matrix=np.array(f["chi_matrix"], dtype=np.float64),
                l_ij_matrix=np.array(f["l_ij_matrix"], dtype=np.float64),
                phi_bar=float(f.attrs["phi_bar"]),
                grid_shape=grid_shape,
                phi=np.array(f["phi"], dtype=np.float64) if "phi" in f else np.empty(0),
                F=np.array(f["F"], dtype=np.float64) if "F" in f else np.empty(0),
                box_lengths=np.array(f["box_lengths"], dtype=np.float64)
                if "box_lengths" in f
                else np.empty(0),
                converged=bool(
                    f.attrs["converged"] if "converged" in f.attrs else False
                ),
            )
        return data

    # ------------------------------------------------------------------
    # Write to HDF5
    # ------------------------------------------------------------------

    def to_hdf5(self, path: str) -> str:
        """
        Batch-write the system definition and trajectory to a new HDF5 file.

        Returns the path so callers can reopen it to add script-specific
        metadata.
        """
        kw = {"compression": "gzip", "compression_opts": 4}

        with h5py.File(path, "w") as f:
            # --- scalar attrs ---
            f.attrs["N"] = self.N
            f.attrs["b"] = self.b
            f.attrs["n_components"] = self.n_components
            f.attrs["phi_bar"] = self.phi_bar
            f.attrs["grid_shape"] = list(self.grid_shape)
            f.attrs["converged"] = self.converged

            # --- matrix / vector datasets ---
            f.create_dataset("block_fractions", data=self.block_fractions)
            f.create_dataset("chi_matrix", data=self.chi_matrix)
            f.create_dataset("l_ij_matrix", data=self.l_ij_matrix)

            # --- trajectory datasets ---
            if self.n_frames > 0:
                ds = f.create_dataset("phi", data=self.phi, **kw)
                ds.attrs["axes"] = ",".join(
                    ["frame", "component"] + [f"x{d}" for d in range(self.ndim)]
                )
                f.create_dataset("F", data=self.F, **kw)
                ds = f.create_dataset("box_lengths", data=self.box_lengths, **kw)
                ds.attrs["axes"] = "frame,spatial_dim"

        return path

    # ------------------------------------------------------------------
    # Reconstruct a model
    # ------------------------------------------------------------------

    def build_model(
        self,
        optimize_box: bool = False,
        frame: int = -1,
    ) -> BlockCopolymerFreeEnergy:
        """
        Reconstruct a BlockCopolymerFreeEnergy from stored system parameters.

        If the trajectory contains at least one frame, the model's delta_phi
        is initialized from ``self.phi[frame]`` and box_lengths from
        ``self.box_lengths[frame]``.  By default ``frame=-1`` (the last
        frame), so the returned model is ready to resume a simulation.

        If the trajectory is empty (zero frames), the model uses its default
        random initialization and the system's default box_lengths.

        Parameters
        ----------
        optimize_box : bool
            Whether box lengths should be optimizable parameters.
        frame : int
            Which trajectory frame to load (defaults to last frame, i.e. frame=-1).

        Returns
        -------
        BlockCopolymerFreeEnergy
        """
        has_trajectory = self.n_frames > 0

        if has_trajectory:
            box_lengths = tuple(float(v) for v in self.box_lengths[frame])
        else:
            box_lengths = None

        model = BlockCopolymerFreeEnergy(
            N=self.N,
            b=self.b,
            block_fractions=torch.from_numpy(self.block_fractions),
            chi_matrix=torch.from_numpy(self.chi_matrix),
            l_ij_matrix=torch.from_numpy(self.l_ij_matrix),
            phi_bar=self.phi_bar,
            grid_shape=self.grid_shape,
            box_lengths=box_lengths,
            optimize_box=optimize_box,
        )

        if has_trajectory:
            phi_tensor = torch.from_numpy(self.phi[frame].copy()).to(torch.float64)
            model.delta_phi = torch.nn.Parameter(phi_tensor)

        return model

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def final_state(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (phi, box_lengths) from the last trajectory frame.

        Returns
        -------
        phi : np.ndarray, shape (n_components, *grid_shape)
        box_lengths : np.ndarray, shape (ndim,)

        Raises
        ------
        ValueError
            If the trajectory is empty.
        """
        if self.n_frames == 0:
            raise ValueError("No trajectory frames stored.")
        return self.phi[-1].copy(), self.box_lengths[-1].copy()
