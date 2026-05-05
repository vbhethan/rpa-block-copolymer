import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import matplotlib.pyplot as plt

from rpa import SimulationData

p = argparse.ArgumentParser()
p.add_argument("input", nargs="?", default="output.h5", help="HDF5 result file")
p.add_argument("-o", "--output", default=None, help="Save figure to this path instead of showing")
args = p.parse_args()

data = SimulationData.from_hdf5(args.input)
F = data.F

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(F)
ax.set_xlabel("Optimizer step")
ax.set_ylabel("Free energy $F$")
ax.set_title(args.input)

if args.output:
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"saved to {args.output}")
else:
    plt.show()
