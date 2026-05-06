"""
Convert an RPA output HDF5 file to a PSCF++ C_RGRID field file.

Usage:
    python export_cgrid.py result.h5 [-o out/c.rf]

The last trajectory frame is used. The phi stored in the HDF5 is delta_phi
(density fluctuation); block_fractions are added back to get absolute
concentrations before writing.

For 1D/2D grids the spatial axes are padded to 3D with size-1 dimensions and
cell parameters of 1.0, since write_C_RGRID_from_array always writes a 3D
orthorhombic header.
"""

import argparse
import os

import numpy as np

from rpa import SimulationData
from rpa.pscf_io import write_C_RGRID_from_array


def _pad_to_3d(arr: np.ndarray, ndim: int) -> np.ndarray:
    """Append size-1 spatial axes until arr has shape (n_mon, Nx, Ny, Nz)."""
    for _ in range(3 - ndim):
        arr = arr[..., np.newaxis]
    return arr


def _cell_params_3d(box: np.ndarray) -> list[float]:
    """Pad box_lengths to length 3 with 1.0 for missing dimensions."""
    params = list(float(v) for v in box)
    while len(params) < 3:
        params.append(1.0)
    return params


def export(h5_path: str, out_path: str) -> None:
    sim = SimulationData.from_hdf5(h5_path)
    delta_phi, box = sim.final_state()

    # delta_phi shape: (n_components, *grid_shape)
    # block_fractions shape: (n_components,) — broadcast over spatial dims
    mean = sim.block_fractions.reshape((-1,) + (1,) * sim.ndim)
    phi_abs = delta_phi + mean

    fields = _pad_to_3d(phi_abs, sim.ndim)
    grid_shape_3d = fields.shape[1:]
    cell_params = _cell_params_3d(box)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_C_RGRID_from_array(out_path, fields, grid_shape_3d, cell_params)
    print(f"Wrote {out_path}  (mesh {grid_shape_3d}, cell {cell_params})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RPA HDF5 to PSCF++ C_RGRID")
    parser.add_argument("h5_file", help="Input HDF5 file from optimize_density.py")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .rf path (default: <h5_stem>_c.rf next to the input file)",
    )
    args = parser.parse_args()

    if args.output is None:
        stem = os.path.splitext(args.h5_file)[0]
        out_path = stem + "_c.rf"
    else:
        out_path = args.output

    export(args.h5_file, out_path)


if __name__ == "__main__":
    main()
