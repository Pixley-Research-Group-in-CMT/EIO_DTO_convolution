"""Shared parameters for the archived EIO/DTO interface calculation."""
from pathlib import Path


ROOT = Path(__file__).resolve().parent
# Keep every stage on the same directory layout so the numbered scripts can be
# run independently without passing paths between them.
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
DTO_SNAPSHOTS = ROOT / "inputs" / "dto" / "gathered_snapshots_hex.hdf5"
DTO_OUT_OF_PLANE_SNAPSHOTS = (
    ROOT / "inputs" / "dto" / "gathered_snapshots_out_of_plane_hex.hdf5"
)
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
