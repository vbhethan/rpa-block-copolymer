"""3D visualisation of diblock copolymer morphology."""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import matplotlib.pyplot as plt

from rpa import SimulationData
from rpa.viz_3d import visualize_3d_isosurface, visualize_3d_wireframe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", default="output.h5", help="HDF5 data file")
    parser.add_argument(
        "--frame", type=int, default=-1, help="Trajectory frame to show"
    )
    parser.add_argument(
        "--tiles",
        type=int,
        nargs=3,
        default=[1, 1, 1],
        metavar=("TX", "TY", "TZ"),
        help="Unit-cell repetitions along each axis",
    )
    parser.add_argument(
        "--mode",
        choices=["wireframe", "isosurface"],
        default="wireframe",
        help="Rendering mode (default: wireframe)",
    )
    parser.add_argument(
        "--level", type=float, default=0.5, help="Isosurface density level"
    )
    parser.add_argument("--color", default="green", help="Surface colour")

    iso = parser.add_argument_group("isosurface mode options")
    iso.add_argument("--alpha", type=float, default=0.6, help="Surface opacity")

    wire = parser.add_argument_group("wireframe mode options")
    wire.add_argument("--linewidth", type=float, default=0.3, help="Edge line width")

    args = parser.parse_args()

    data = SimulationData.from_hdf5(args.input)
    tiles = tuple(args.tiles)

    if args.mode == "wireframe":
        visualize_3d_wireframe(
            data,
            frame=args.frame,
            level=args.level,
            color=args.color,
            linewidth=args.linewidth,
            n_tiles=tiles,
        )
    else:
        visualize_3d_isosurface(
            data,
            frame=args.frame,
            level=args.level,
            color=args.color,
            alpha=args.alpha,
            n_tiles=tiles,
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
