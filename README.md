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

You can also install this package in an environment using pip. 

`pip install -e .`


