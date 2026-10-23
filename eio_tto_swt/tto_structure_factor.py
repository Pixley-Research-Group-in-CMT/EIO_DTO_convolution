"""Shared numerical helpers for the three packaged SWT workflows."""
import numpy as np

E_MU = np.asarray([
    [0.25, 0.25, 0.25],
    [0.25, -0.75, 0.25],
    [0.25, 0.25, -0.75],
    [-0.75, 0.25, 0.25],
])

FCC_PRIMITIVES = np.asarray([
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
])

def lorentzian_delta(x, eta):
    """Normalized broadened delta from Eq. (83)."""
    if eta <= 0:
        raise ValueError("The DSSF broadening eta must be positive")
    return eta / (np.pi * (np.asarray(x) ** 2 + eta ** 2))

def eio_to_tto_reduced_momentum(q_slab, cubic_lattice_angstrom):
    """Convert EIO slab Cartesian momenta (1/Angstrom) to TTO FCC r.l.v.

    The axes exactly reproduce ``eio_tb.transform_to_slab_coordinates``.  The
    result is in the same reduced coordinates used by the spin-wave solver.
    """
    q_slab = np.asarray(q_slab, dtype=float)
    x_axis = np.asarray([-1.0, 0.0, 1.0]) / np.sqrt(2.0)
    y_axis = np.asarray([1.0, -2.0, 1.0]) / np.sqrt(6.0)
    z_axis = np.asarray([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    slab_rotation = np.stack((x_axis, y_axis, z_axis))
    q_cubic = np.einsum("...i,ij->...j", q_slab, slab_rotation)
    direct = (cubic_lattice_angstrom / 2.0) * FCC_PRIMITIVES
    reciprocal = 2.0 * np.pi * np.linalg.inv(direct)
    return np.einsum("ij,...j->...i", np.linalg.inv(reciprocal), q_cubic)
