import numpy as np
import pytest

from swt_bz_maps import (
    FIELD_PHIS_DEGREES,
    FIELD_RATIOS,
    SUBLATTICES,
    build_parser,
    kagome_bz_grid,
)
from tto_structure_factor import FCC_PRIMITIVES


def test_kagome_bz_is_hexagonal_centered_and_perpendicular_to_111():
    lattice_constant = 10.154
    axis, edges, q, inside, vertices = kagome_bz_grid(5, lattice_constant)
    assert q.shape == (5, 5, 3)
    assert inside.shape == (5, 5)
    assert vertices.shape == (6, 2)
    assert np.isclose(axis[2], 0.0)
    assert np.allclose(q[2, 2], 0.0)
    assert inside[2, 2]
    assert not inside[0, 0]
    assert not inside[-1, -1]
    assert np.isclose(edges[0], -vertices[0, 0])
    assert np.isclose(edges[-1], vertices[0, 0])

    direct = 0.5 * lattice_constant * FCC_PRIMITIVES
    reciprocal = 2.0 * np.pi * np.linalg.inv(direct)
    q_cubic = np.einsum("ij,...j->...i", reciprocal, q[inside])
    normal = np.asarray([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    assert np.allclose(q_cubic @ normal, 0.0, atol=1e-12)


def test_kagome_bz_requires_an_odd_grid_and_positive_lattice_constant():
    with pytest.raises(ValueError, match="odd integer"):
        kagome_bz_grid(4, 10.154)
    with pytest.raises(ValueError, match="positive"):
        kagome_bz_grid(5, -10.154)


def test_map_scan_uses_only_interfacial_sublattices_and_fixed_fields():
    assert SUBLATTICES == (1, 2, 3)
    assert np.allclose(FIELD_PHIS_DEGREES, [0.1, 30.1])
    assert np.allclose(FIELD_RATIOS, [0.25, 0.45, 1.5])


def test_map_cli_no_longer_accepts_a_single_field_condition():
    parser = build_parser("static", "test")
    with pytest.raises(SystemExit):
        parser.parse_args(["--h-over-jzz", "0.5"])


def test_map_cli_exposes_slice_and_integrated_q_perp_modes():
    parser = build_parser("static", "test")
    defaults = parser.parse_args([])
    integrated = parser.parse_args([
        "--q-perp-mode", "integrated", "--n-q-perp", "12"
    ])
    assert defaults.q_perp_mode == "slice"
    assert defaults.n_q_perp == 8
    assert integrated.q_perp_mode == "integrated"
    assert integrated.n_q_perp == 12
