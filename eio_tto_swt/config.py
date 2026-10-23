"""Shared SWT and EIO numerical settings."""
import os

from pathlib import Path

ROOT = Path(__file__).resolve().parent

OUTPUT_ROOT = Path(os.environ.get('EIO_TTO_OUTPUT_ROOT', ROOT)).resolve()

EIO_MODEL = os.environ.get('EIO_TTO_EIO_MODEL', 'original')

if EIO_MODEL not in ('original', 'symmetrized'):
    raise ValueError("eio_model must be 'original' or 'symmetrized'")

MODEL_DIR = ROOT / 'model' if EIO_MODEL == 'original' else ROOT / 'eio_symmetry_clean' / 'model'

DATA_DIR = OUTPUT_ROOT / 'data'

FIGURE_DIR = OUTPUT_ROOT / 'figures'

EIO_REFERENCE = ROOT / 'reference' / 'eio' / 'surface_states_summary_nk24.hdf5'

EIO_NK = 24

N_WANN = 24

N_LAYER = 11

E_FERMI = 5.142059246119027

SURFACE_WEIGHT_CUTOFF = 0.5

ENERGY_WINDOW = 0.01

ORBITAL_HOPPINGS = (1.0, 1.0, 1.0)

ENERGY_BROADENING = 0.01

TTO_J_ZZ_EV = 0.00017

TTO_J_PM_RATIO = 0.2

TTO_J_PMPM_RATIO = 0.0

TTO_G_Z = 10.0

TTO_G_PERP = 0.0

TTO_DELTA1_PER_MEV = 1.0 / 1.4

TTO_DELTA2_PER_MEV = 1.0 / 1.4

TTO_DSSF_NK = 8

TTO_Q_PERP_MODE = 'slice'

TTO_N_Q_PERP = 8

TTO_DSSF_BROADENING_EV = 2e-05

TTO_CUBIC_LATTICE_ANGSTROM = 10.154

TTO_INTERFACE_SUBLATTICES = (1, 2, 3)

TTO_SPIN_WAVE_FIELDS = (0.15, 0.2, 0.25)

TTO_SPIN_WAVE_THETA_DEGREES = 90.0

TTO_SPIN_WAVE_PHI_OFFSET_DEGREES = 0.1

TTO_SPIN_WAVE_PHI_START_DEGREES = 0.0

TTO_SPIN_WAVE_PHI_STOP_DEGREES = 180.0

TTO_SPIN_WAVE_N_PHI = 91

TTO_SPIN_WAVE_REFLECT_NEGATIVE_PHI = True

TTO_SPIN_LENGTH = 0.5

TTO_SPIN_WAVE_REGULARIZATION = 1e-08

TTO_SPIN_WAVE_STABILITY_TILT = 0.01
