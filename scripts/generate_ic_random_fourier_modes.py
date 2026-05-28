"""
Generate diverse initial conditions by superposing random Fourier modes
near the RPA characteristic wavevector q*.

Motivation
----------
Pure Gaussian noise spreads amplitude across all wavelengths; most of it is
physically irrelevant near the order-disorder transition. Fixed crystallographic
stars (BCC_110, HEX, ...) target the right wavelength but commit to a single
symmetry and basin. Random Fourier modes near q* concentrate energy at the
wavelength where the RPA predicts the instability while leaving the phase
relationships random — different seeds can nucleate different ordered phases.

Strategy
--------
1. Build the model and compute q* from the cached Gamma_ij: the wavevector
   at which det(Gamma_ij) is minimized over q > 0 (peak of the disordered-
   state structure factor).
2. Collect all integer Miller-index vectors whose physical |q| falls in a
   shell [q* - dq, q* + dq].
3. For each seed, draw N_MODES vectors from that pool (without replacement),
   assign each a random Gaussian amplitude and a uniform random phase, and
   build:
       field(r) = sum_n  A_n * cos(2π (h·xf + k·yf + l·zf) + φ_n)
4. Normalize the peak to AMPLITUDE, then derive the remaining components
   to satisfy incompressibility and project to zero mean.
5. Save one HDF5 file per seed: input_rfm_NNN.h5.

Output: input_rfm_000.h5, input_rfm_001.h5, ...
"""

import itertools
import math
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

from rpa import BlockCopolymerFreeEnergy, SimulationData

# ---------------------------------------------------------------------------
# System parameters
# ---------------------------------------------------------------------------

N = 100
b = 1.0
f_A = 0.35
f_B = 1.0 - f_A
CHI_N = 28.0
CHI_AB = CHI_N / N

DTYPE = torch.float32
Rg = math.sqrt(N * b**2 / 6)

# Box: 2 unit cells per side (will be relaxed by the optimizer).
# Use q* ~ 1.946/Rg as a first guess for the lattice spacing d = 2π/q*.
L_guess = 2.0 * math.pi / (1.946 / Rg) * 2
box_lengths = (L_guess, L_guess, L_guess)

grid_shape = (32, 32, 32)

# ---------------------------------------------------------------------------
# IC generation settings
# ---------------------------------------------------------------------------

N_SAMPLES = 5  # number of independent ICs to generate
N_MODES = 12  # Fourier modes to superpose per IC
AMPLITUDE = 0.1  # target peak amplitude of delta_phi[0]
DQ_FRAC = 0.15  # shell half-width as a fraction of q* (±15%)
N_MAX = 5  # max absolute value of any Miller index
OUTPUT_PREFIX = "input_rfm"

# ---------------------------------------------------------------------------
# Build model (optimize_box=False so Gamma_ij is cached)
# ---------------------------------------------------------------------------

chi_matrix = torch.tensor([[0.0, CHI_AB], [CHI_AB, 0.0]], dtype=DTYPE)
l_ij_matrix = torch.zeros((2, 2), dtype=DTYPE)
block_fractions = torch.tensor([f_A, f_B], dtype=DTYPE)

model = BlockCopolymerFreeEnergy(
    N=N,
    chi_matrix=chi_matrix,
    l_ij_matrix=l_ij_matrix,
    block_fractions=block_fractions,
    phi_bar=1.0,
    grid_shape=grid_shape,
    box_lengths=box_lengths,
    optimize_box=False,
    dtype=DTYPE,
)

# ---------------------------------------------------------------------------
# Find q* from the RPA structure factor
#
# q* minimizes det(Gamma_ij(q)) over q > 0, i.e., it is where the disordered-
# state structure factor S(q) ∝ 1/det(Gamma) is largest.
# ---------------------------------------------------------------------------

K2 = model._compute_K2(model.L)  # (*grid_shape)
k_mag = K2.sqrt()  # (*grid_shape)

Gamma = model._Gamma_ij_cached  # (*grid_shape, n_components, n_components)
G = Gamma.numpy().reshape(-1, model.n_components, model.n_components)
k_flat = k_mag.numpy().ravel()

# For 2×2: det = G00*G11 - G01*G10
det_vals = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] * G[:, 1, 0]

# Bin by rounded |q|, average det per shell, find minimum (excluding q=0)
valid = k_flat > 1e-6
k_rounded = np.round(k_flat[valid], 4)
det_valid = det_vals[valid]

unique_k = np.unique(k_rounded)
mean_det = {k: float(np.mean(det_valid[k_rounded == k])) for k in unique_k}
q_star = float(min(mean_det, key=mean_det.get))

print(f"q* = {q_star:.4f}  (q* Rg = {q_star * Rg:.4f})")

q_low = q_star * (1.0 - DQ_FRAC)
q_high = q_star * (1.0 + DQ_FRAC)

# ---------------------------------------------------------------------------
# Enumerate integer Miller-index vectors in the shell [q_low, q_high]
# ---------------------------------------------------------------------------

L_arr = model.L.numpy()
ndim = model.ndim

candidate_modes = []
for hkl in itertools.product(range(-N_MAX, N_MAX + 1), repeat=ndim):
    if all(h == 0 for h in hkl):
        continue
    q_phys = (
        2.0 * math.pi * math.sqrt(sum((hkl[d] / L_arr[d]) ** 2 for d in range(ndim)))
    )
    if q_low <= q_phys <= q_high:
        candidate_modes.append(hkl)

if len(candidate_modes) < N_MODES:
    raise RuntimeError(
        f"Only {len(candidate_modes)} modes in shell [{q_low:.3f}, {q_high:.3f}]; "
        f"increase DQ_FRAC or N_MAX."
    )

print(f"Shell [{q_low:.4f}, {q_high:.4f}]: {len(candidate_modes)} candidate modes")

# ---------------------------------------------------------------------------
# Pre-build real-space fractional-coordinate grids
# ---------------------------------------------------------------------------

Nx, Ny, Nz = grid_shape
xf = torch.linspace(0.0, 1.0, Nx + 1, dtype=DTYPE)[:-1]
yf = torch.linspace(0.0, 1.0, Ny + 1, dtype=DTYPE)[:-1]
zf = torch.linspace(0.0, 1.0, Nz + 1, dtype=DTYPE)[:-1]
coords = [
    xf.reshape(Nx, 1, 1).expand(Nx, Ny, Nz),
    yf.reshape(1, Ny, 1).expand(Nx, Ny, Nz),
    zf.reshape(1, 1, Nz).expand(Nx, Ny, Nz),
]

# ---------------------------------------------------------------------------
# Generate ICs
# ---------------------------------------------------------------------------

f_vec = model.f_vec  # (n_components,)
n_components = model.n_components
f_rest = 1.0 - f_vec[0].item()

for seed in range(N_SAMPLES):
    rng = np.random.default_rng(seed)

    # Sample N_MODES distinct modes from the candidate pool
    idx = rng.choice(len(candidate_modes), size=N_MODES, replace=False)
    modes = [candidate_modes[i] for i in idx]

    # Random Gaussian amplitudes + uniform phases
    amps = rng.standard_normal(N_MODES)
    phases = rng.uniform(0.0, 2.0 * math.pi, N_MODES)

    # Build superposition: sum_n amp_n * cos(2π (h·xf + k·yf + l·zf) + φ_n)
    field = torch.zeros(Nx, Ny, Nz, dtype=DTYPE)
    for hkl, amp, phi in zip(modes, amps, phases):
        phase_field = sum(hkl[d] * coords[d] for d in range(ndim))
        field = field + float(amp) * torch.cos(2.0 * math.pi * phase_field + phi)

    # Normalize peak to AMPLITUDE
    peak = field.abs().max()
    if peak > 0:
        field = AMPLITUDE * field / peak

    # Distribute across components to satisfy incompressibility:
    # delta_phi[0] = field, delta_phi[i>0] = -(f_i / f_rest) * field
    delta_phi = torch.zeros(n_components, Nx, Ny, Nz, dtype=DTYPE)
    delta_phi[0] = field
    for i in range(1, n_components):
        delta_phi[i] = -(f_vec[i].item() / f_rest) * field

    # Double-project: zero mean per component + pointwise incompressibility
    delta_phi = delta_phi - delta_phi.mean(dim=(1, 2, 3), keepdim=True)
    delta_phi = delta_phi - delta_phi.mean(dim=0, keepdim=True)

    # Save
    data = SimulationData.from_model(model)
    data.phi = delta_phi.numpy()[np.newaxis]  # (1, n_components, Nx, Ny, Nz)
    data.box_lengths = np.array([list(box_lengths)])  # (1, ndim)

    out_file = f"{OUTPUT_PREFIX}_{seed:03d}.h5"
    data.to_hdf5(out_file)
    print(
        f"  seed {seed:3d} → {out_file}  "
        f"delta_phi range [{delta_phi.min():.4f}, {delta_phi.max():.4f}]"
    )

print(
    f"\nGenerated {N_SAMPLES} ICs: {OUTPUT_PREFIX}_000.h5 … {OUTPUT_PREFIX}_{N_SAMPLES - 1:03d}.h5"
)
print(f"  grid_shape   : {grid_shape}")
print(
    f"  box_lengths  : ({L_guess:.3f}, {L_guess:.3f}, {L_guess:.3f})  ({L_guess / Rg:.2f} Rg each)"
)
print(f"  f_A          : {f_A},  chi*N = {CHI_N}")
print(f"  Rg           : {Rg:.3f}")
print(f"  q*           : {q_star:.4f}  (q* Rg = {q_star * Rg:.4f})")
print(f"  shell        : [{q_low:.4f}, {q_high:.4f}]  ({len(candidate_modes)} modes)")
print(f"  modes/IC     : {N_MODES}")
print(f"  amplitude    : {AMPLITUDE}")
