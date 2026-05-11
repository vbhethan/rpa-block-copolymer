# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses [pixi](https://pixi.sh) for environment and task management (config in `pyproject.toml` under `[tool.pixi.*]`).

```bash
pixi run test              # run all tests
pixi run gen-input         # generate initial conditions HDF5 file
pixi run optimize          # run density optimization
```

Run specific tests:
```bash
pixi run pytest tests/test_rpa.py -v
pixi run pytest tests/ -v -k "test_forward_and_gradients_2d"
pixi run pytest tests/test_gpu.py -v   # GPU tests (skipped if no CUDA)
```

Run optimizer manually (see `run-optimizer.sh` for example flags):
```bash
python scripts/optimize_density.py -i input_data.h5 -o result.h5 \
    -n 250 --lr_box 1.0 --lr_phi 0.1 --n_inner_phi 2000 --n_inner_box 10 \
    --log_every 50 --box_grad_scale 10
```

## Architecture

### Data flow

```
generate_input.py  →  input.h5  →  optimize_density.py  →  result.h5  →  vis.py / vis_3d.py
```

HDF5 files (`.h5`) are the sole data exchange format. They store the full system definition (polymer parameters, chi matrix, l_ij matrix) plus a trajectory of density frames. The `SimulationData` dataclass (`src/rpa/simulation_io.py`) handles serialization and can reconstruct a live model via `build_model()`.

### Core module: `src/rpa/`

**`free_energy.py` — `BlockCopolymerFreeEnergy(nn.Module)`**

The central PyTorch module. The learnable parameter `delta_phi` has shape `(n_components, *grid_shape)` and represents fluctuations from the homogeneous reference: `rho_i(r) = f_i * phi_bar + delta_phi_i(r)`.

Two physical constraints are enforced by double projection in `_project_order_parameter()`:
1. Zero spatial mean per component (mass conservation)
2. Local incompressibility: `sum_i delta_phi_i(r) = 0` pointwise

The free energy has three terms (see `forward()`):
- `Delta_F_int`: RPA interaction energy computed in Fourier space via `Gamma_ij(q)`
- `F_mixing`: Flory-Huggins mixing entropy (real space, full nonlinear log term)
- `F_mixing_2`: Quadratic expansion of mixing entropy, subtracted to avoid double-counting with the ideal chain contribution already in `Gamma_ij`

The vertex function `Gamma_ij(q)` encodes the inverse ideal structure factor plus chi interactions. When the box is fixed (`optimize_box=False`), it is precomputed once and cached as `_Gamma_ij_cached`. When the box is optimizable, it is recomputed each forward pass from `log_L` (a learnable parameter storing `log` of box lengths to enforce positivity).

Chain architecture beyond simple diblock is controlled by `l_ij_matrix`, which encodes the contour distance between blocks `i` and `j`, entering as `exp(-l_ij * N * b^2 * k^2 / 6)` in the off-diagonal Debye structure factor terms.

**`optimizers/pgd.py` — `optimize_joint()`**

Alternating projected gradient descent: inner loops optimize `delta_phi` (with projection after each step), outer loop updates box lengths. Both use backtracking line search. Box dimensions are updated in log-space. Returns a `SimulationData` with the full optimization trajectory.

**`initial_conditions.py`**

Functions that generate structured starting configurations as `SimulationData`:
- `generate_fourier_mode_initial_conditions()`: populate specified reciprocal lattice vectors. Pre-defined stars: `BCC_110_STAR`, `BCC_110_200_STAR`, `FCC_111_200_STAR`, `LAMELLAR_Z`
- `generate_hexagonal_A_in_BC_matrix()`: hexagonally packed cylinders (3-component only)
- `generate_random_normal_initial_conditions()`: random Gaussian noise

**`pscf_io.py`**

Utilities to write PSCF++ r-grid field files (concentration or omega fields) from arrays produced by this code.

**`viz.py`, `viz_3d.py`**

Matplotlib-based visualization for 2D and 3D density fields.

### Key conventions

- Grid axes: component index is always axis 0; spatial dims are axes 1..ndim. The `spatial_dims` property returns `tuple(range(1, ndim+1))`.
- Dtype: all tensors use a single `real_dtype` (float32 or float64), with a paired `complex_dtype` for FFT results. Pass `dtype=torch.float32` to `BlockCopolymerFreeEnergy` for faster GPU runs.
- Device: pass `device="cuda"` or `device="cuda:0"` to place everything on GPU. `SimulationData.build_model(device=...)` forwards this to the constructor.
- `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` must be set before importing torch on macOS; `__init__.py` handles this automatically when the package is imported.
