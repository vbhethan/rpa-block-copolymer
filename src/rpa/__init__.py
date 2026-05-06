"""RPA free-energy computation and optimization for block copolymer systems."""

import os

# Set before any submodule imports torch (must precede the first torch import).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .free_energy import BlockCopolymerFreeEnergy
from .simulation_io import SimulationData

__all__ = ["BlockCopolymerFreeEnergy", "SimulationData"]
