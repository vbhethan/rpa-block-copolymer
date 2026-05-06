"""Render the last frame of an RPA result file as a 2D map of the dominant component."""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import matplotlib.pyplot as plt

from rpa import SimulationData
from rpa.viz import (
    detect_trial_number,
    generate_annotation_str,
    plot_simulation_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", default="output.h5", help="HDF5 result file"
    )
    args = parser.parse_args()

    result = SimulationData.from_hdf5(args.input)
    print(result.phi.shape)

    block_id = "_".join(f"{v:.3f}" for v in result.block_fractions.flatten().tolist())
    chi_id = "_".join(f"{v:.3f}" for v in result.chi_matrix.flatten().tolist())
    id_string = f"f_{block_id}_chi_{chi_id}"

    fig, ax = plt.subplots()
    plot_simulation_result(result, fig, ax)

    bf_str, chi_str = generate_annotation_str(result)
    print(f"Block fractions: {bf_str}\nChi matrix: {chi_str}")

    trial_number = detect_trial_number(id_string)
    if result.converged:
        out_path = f"visualizations/vis_{id_string}_{trial_number}.png"
        print(f"Saving visualization to {out_path}")
        fig.savefig(out_path)
    else:
        print("result was not converged, just showing the last frame")

    plt.show()


if __name__ == "__main__":
    main()
