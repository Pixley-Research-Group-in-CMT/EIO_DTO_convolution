"""Shared parameters for the EIO/TTO dynamical interface calculation."""
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(os.environ.get("EIO_TTO_OUTPUT_ROOT", ROOT)).resolve()
# Keep every stage on the same directory layout so the numbered scripts can be
# run independently without passing paths between them.
MODEL_DIR = ROOT / "model"
DATA_DIR = OUTPUT_ROOT / "data"
FIGURE_DIR = OUTPUT_ROOT / "figures"
DTO_SNAPSHOTS = ROOT / "inputs" / "dto" / "gathered_snapshots_hex.hdf5"
DTO_OUT_OF_PLANE_SNAPSHOTS = (
    ROOT / "inputs" / "dto" / "gathered_snapshots_out_of_plane_hex.hdf5"
)
GMFT_IN_PLANE_MIN = ROOT.parent / "gMFT_in_plane_min"
EIO_REFERENCE = ROOT / "reference" / "eio" / "surface_states_summary_nk24.hdf5"
TRANSPORT_REFERENCE_DIR = ROOT / "reference" / "transport"
OUT_OF_PLANE_TRANSPORT_REFERENCE = (
    TRANSPORT_REFERENCE_DIR / "rho_out_of_plane_size4_nk24.csv"
)

# EIO slab and surface-state selection.
EIO_NK = 24
N_WANN = 24
N_LAYER = 11
E_FERMI = 5.142059246119027
SURFACE_WEIGHT_CUTOFF = 0.5
ENERGY_WINDOW = 0.01

# DTO snapshot sampling used by the archived 8x8 in-plane-field workflow.
# Each field contains 60 field directions, four Monte Carlo seeds, and a time
# series of spin configurations for every seed.
FIELDS = (0.1, 0.5, 1.0, 2.0, 4.0)
N_ANGLES = 60
TEMPERATURE = 0.5
N_SEEDS = 4
N_SNAPSHOTS = 20
SYSTEM_SIZE = 8
SITES_PER_SUBLATTICE = 64
EIO_DTO_LATTICE_RATIO = 7.26198664 / 7.071067811862696

# The archived production files were generated in two notebook states.
WARMUP_SNAPSHOTS_BY_FIELD = {
    0.1: 4,
    0.5: 10,
    1.0: 10,
    2.0: 10,
    4.0: 10,
}

# Historical interface and transport conventions.
ORBITAL_HOPPINGS = (1.0, 1.0, 1.0)
T2_NORMALIZATION = (SITES_PER_SUBLATTICE * 2) ** 0.5
ANGLE_SMOOTHING_SIGMA = 1.0
ENERGY_BROADENING = 0.01

# Dynamical TTO calculation.  The finite-field gMFT tables use J_zz as their
# energy unit; this explicit conversion puts their spinon frequencies in the
# eV units used by the EIO tight-binding calculation.  0.17 meV is the default
# TTO scale, and can be overridden from the stage-2 command line.
TTO_J_ZZ_EV = 0.17e-3
TTO_J_PM_RATIO = 0.20
TTO_J_PMPM_RATIO = 0.0
# Default field and angle grid for the independent high-field spin-wave scan.
TTO_FIELD_RATIOS = (0.01, 0.15, 0.20, 0.25)
TTO_DSSF_NK = 8
TTO_Q_PERP_MODE = "slice"
TTO_N_Q_PERP = 8
TTO_DSSF_BROADENING_EV = 0.02e-3
TTO_CUBIC_LATTICE_ANGSTROM = 10.154
TTO_INTERFACE_SUBLATTICES = (1, 2, 3)
TTO_SPIN_WAVE_FIELDS = (0.15, 0.20, 0.25)
TTO_SPIN_WAVE_THETA_DEGREES = 90.0
TTO_SPIN_WAVE_PHI_OFFSET_DEGREES = 0.1
# Evaluate only phi >= 0; the transport stage mirrors the result to -phi.
TTO_SPIN_WAVE_PHI_START_DEGREES = 0.0
TTO_SPIN_WAVE_PHI_STOP_DEGREES = 180.0
TTO_SPIN_WAVE_N_PHI = 91
TTO_SPIN_WAVE_REFLECT_NEGATIVE_PHI = True
TTO_SPIN_LENGTH = 0.5
TTO_SPIN_WAVE_REGULARIZATION = 1e-8
TTO_SPIN_WAVE_STABILITY_TILT = 1e-2

# Recovered 4x4 out-of-plane [111] field scan. Unlike the in-plane workflow,
# this scan varies field magnitude at two temperatures and has no angle axis.
OUT_OF_PLANE_FIELDS = (0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0,
                       6.0, 7.0, 8.0, 9.0)
OUT_OF_PLANE_TEMPERATURES = (0.5, 4.0)
OUT_OF_PLANE_N_SNAPSHOTS = 10
OUT_OF_PLANE_WARMUP_SNAPSHOTS = 5
OUT_OF_PLANE_SYSTEM_SIZE = 4
OUT_OF_PLANE_SITES_PER_SUBLATTICE = 16
OUT_OF_PLANE_FIELD_TO_TESLA = 1 / 6.72


def field_tag(field):
    """Match the floating-point labels in Zhengtao's HDF5 groups."""
    return str(float(field))


def condition_tag(scan, field, temperature=None):
    """Return a collision-free filename tag for one scan condition."""
    if scan == "in-plane":
        return f"h{field_tag(field)}"
    if temperature is None:
        raise ValueError("Out-of-plane outputs require a temperature")
    return f"out_of_plane_h{field_tag(field)}_t{field_tag(temperature)}"
