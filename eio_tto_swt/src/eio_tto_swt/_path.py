"""Shared numerical helpers for the three packaged SWT workflows."""
import numpy as np
from ._ordered_state import COMPONENTS
from ._geometry import FCC_PRIMITIVES
HIGH_SYMMETRY_LABELS = ('$\\Gamma$', 'X', 'W', 'K', '$\\Gamma$', 'L', 'U')
HIGH_SYMMETRY_NAMES = ('Gamma', 'X', 'W', 'K', 'Gamma', 'L', 'U')
HIGH_SYMMETRY_Q_REDUCED = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.75, 0.25, 0.5], [0.75, 0.375, 0.375], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.625, 0.25, 0.625]])

def reduced_to_cartesian(q_reduced, lattice_constant_angstrom):
    """Convert FCC primitive reduced coordinates to inverse Angstrom."""
    q_reduced = np.asarray(q_reduced, dtype=float)
    lattice_constant_angstrom = float(lattice_constant_angstrom)
    if q_reduced.shape[-1] != 3 or not np.all(np.isfinite(q_reduced)):
        raise ValueError('q_reduced must end with finite three-component momenta')
    if not np.isfinite(lattice_constant_angstrom) or lattice_constant_angstrom <= 0.0:
        raise ValueError('lattice_constant_angstrom must be finite and positive')
    direct = 0.5 * lattice_constant_angstrom * FCC_PRIMITIVES
    reciprocal = 2.0 * np.pi * np.linalg.inv(direct)
    return np.einsum('ij,...j->...i', reciprocal, q_reduced)

def fcc_high_symmetry_path(points_per_segment, lattice_constant_angstrom):
    """Sample Gamma-X-W-K-Gamma-L-U with uniform density per segment."""
    if int(points_per_segment) != points_per_segment or int(points_per_segment) < 1:
        raise ValueError('points_per_segment must be a positive integer')
    points_per_segment = int(points_per_segment)
    vertex_cartesian = reduced_to_cartesian(HIGH_SYMMETRY_Q_REDUCED, lattice_constant_angstrom)
    segment_lengths = np.linalg.norm(np.diff(vertex_cartesian, axis=0), axis=1)
    tick_positions = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    q_pieces = []
    distance_pieces = []
    interpolation = np.arange(points_per_segment, dtype=float) / points_per_segment
    for segment_index in range(len(segment_lengths)):
        q_start = HIGH_SYMMETRY_Q_REDUCED[segment_index]
        q_stop = HIGH_SYMMETRY_Q_REDUCED[segment_index + 1]
        q_pieces.append(q_start[None, :] + interpolation[:, None] * (q_stop - q_start)[None, :])
        distance_pieces.append(tick_positions[segment_index] + interpolation * segment_lengths[segment_index])
    q_reduced = np.concatenate(q_pieces + [HIGH_SYMMETRY_Q_REDUCED[-1:]], axis=0)
    path_distance = np.concatenate(distance_pieces + [tick_positions[-1:]])
    q_cartesian = reduced_to_cartesian(q_reduced, lattice_constant_angstrom)
    tick_indices = np.arange(len(HIGH_SYMMETRY_LABELS)) * points_per_segment
    return (q_reduced, q_cartesian, path_distance, tick_positions, tick_indices)

def energy_axis_from_bounds(maximum_mev, n_omega, component):
    """Return the energy axis, including two-magnon bandwidth when needed."""
    maximum_mev = float(maximum_mev)
    if not np.isfinite(maximum_mev) or maximum_mev <= 0.0:
        raise ValueError('maximum_mev must be finite and positive')
    if int(n_omega) != n_omega or int(n_omega) < 2:
        raise ValueError('n_omega must be an integer of at least 2')
    if component not in COMPONENTS:
        raise ValueError(f'Unknown component: {component}')
    multiplier = 2.0 if component in ('two_magnon', 'total') else 1.0
    return np.linspace(0.0, multiplier * maximum_mev, int(n_omega))

def coordinate_edges(centers):
    """Construct cell edges from a strictly increasing one-dimensional axis."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or len(centers) < 2 or (not np.all(np.diff(centers) > 0.0)):
        raise ValueError('centers must be a strictly increasing axis of length >= 2')
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(([centers[0] - 0.5 * (centers[1] - centers[0])], midpoints, [centers[-1] + 0.5 * (centers[-1] - centers[-2])]))

def phi_file_label(phi_degrees):
    """Return a filesystem-safe label that retains the evaluated angle."""
    if not np.isfinite(phi_degrees):
        raise ValueError('phi_degrees must be finite')
    return f'{float(phi_degrees):g}'.replace('-', 'm').replace('.', 'p')
