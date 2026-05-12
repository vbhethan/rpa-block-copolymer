# rpa-block-copolymer

A PyTorch implementation of the Random Phase Approximation (RPA) free energy for block copolymer melts, with GPU-accelerated gradient-based optimization of both real-space density fields and box geometry.

## Goal of this codebase

Given a multi-component block copolymer system (chain architecture, composition fractions, Flory-Huggins χ parameters, and a candidate unit cell), this code minimizes the RPA free energy with respect to:

- **density fields** `δφ_i(r)`: the spatial fluctuations of each component on a real-space grid
- **box lengths** `L`: the simulation cell dimensions (optional, joint optimization)

The free energy combines an RPA interaction term evaluated in Fourier space via the vertex function Γ_ij(q) with the full nonlinear Flory-Huggins mixing entropy in real space. Incompressibility and mass conservation are enforced by a double projection after every gradient step.

## Workflow

See the `scripts` directory for the general workflow, including some example preparation of input `h5` files and optimizations of of the density fields, box size, and combined optimization.

Results of an optimization (polymer parameters, grid, χ matrix, density trajectory) are stored in HDF5 files. See the `SimulationData` dataclass (`src/rpa/simulation_io.py`) for details.

## Quickstart

Install with [pixi](https://pixi.sh):

```bash
pixi install
```

Generate an initial condition and run the optimizer:

```bash
pixi run gen-input      # writes input.h5
pixi run optimize       # reads input.h5, writes result.h5
pixi run test           # run the test suite
```

See `run-optimizer.sh` and `run-box-optimization.sh` for example CLI invocations with explicit hyperparameters. The `examples/` notebook walks through a complete optimization interactively.

You can also install this package in an environment using pip. 

`pip install -e .`

## Source layout

```
src/rpa/
  free_energy.py        # BlockCopolymerFreeEnergy (nn.Module) — the core physics
  optimizers/pgd.py     # optimize_joint(): alternating PGD with backtracking line search
  initial_conditions.py # structured ICs: BCC, FCC, hexagonal cylinders, lamellar, random
  simulation_io.py      # SimulationData dataclass, HDF5 serialization, build_model()
  pscf_io.py            # export fields to PSCF++ r-grid format
  viz.py / viz_3d.py    # 2D/3D density field visualization
```

