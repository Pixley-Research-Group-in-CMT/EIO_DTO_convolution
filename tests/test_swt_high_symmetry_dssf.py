import numpy as np
import pytest

from swt_bz_maps import COMPONENTS
from plot_swt_high_symmetry_dssf import (
    HIGH_SYMMETRY_Q_REDUCED,
    coordinate_edges,
    energy_axis_from_bounds,
    fcc_high_symmetry_path,
    phi_file_label,
    reduced_to_cartesian,
)


def test_path_contains_every_fcc_vertex_and_has_physical_distance():
    q_reduced, q_cartesian, distance, ticks, tick_indices = (
        fcc_high_symmetry_path(3, 10.0)
    )
    assert q_reduced.shape == (19, 3)
    assert q_cartesian.shape == q_reduced.shape
    assert np.allclose(q_reduced[tick_indices], HIGH_SYMMETRY_Q_REDUCED)
    assert np.allclose(distance[tick_indices], ticks)
    assert np.all(np.diff(distance) > 0.0)
    assert np.isclose(distance[0], 0.0)


def test_reduced_points_match_standard_conventional_fcc_coordinates():
    q_cartesian = reduced_to_cartesian(HIGH_SYMMETRY_Q_REDUCED, 1.0)
    conventional = q_cartesian / (2.0 * np.pi)
    assert np.allclose(conventional[1], [0.0, 1.0, 0.0])  # X
    assert np.allclose(conventional[2], [1.0, 0.5, 0.0])  # W
    assert np.allclose(conventional[3], [0.75, 0.75, 0.0])  # K
    assert np.allclose(conventional[5], [0.5, 0.5, 0.5])  # L
    assert np.allclose(conventional[6], [1.0, 0.25, 0.25])  # U


def test_energy_axis_and_coordinate_edge_validation():
    one_magnon = energy_axis_from_bounds(0.4, 5, "one_magnon")
    one_magnon_1 = energy_axis_from_bounds(0.4, 5, "one_magnon_1")
    two_magnon = energy_axis_from_bounds(0.4, 5, "two_magnon")
    assert np.allclose(one_magnon[[0, -1]], [0.0, 0.4])
    assert np.allclose(one_magnon_1, one_magnon)
    assert np.allclose(two_magnon[[0, -1]], [0.0, 0.8])
    assert np.allclose(
        coordinate_edges([0.0, 1.0, 3.0]), [-0.5, 0.5, 2.0, 4.0]
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        coordinate_edges([0.0, 0.0])
    with pytest.raises(ValueError, match="at least 2"):
        energy_axis_from_bounds(0.4, 1, "total")


def test_phi_file_labels_retain_the_actual_offset_angles():
    assert phi_file_label(0.1) == "0p1"
    assert phi_file_label(30.1) == "30p1"


def test_every_energy_ordered_one_magnon_mode_is_selectable():
    assert all(f"one_magnon_{mode}" in COMPONENTS for mode in range(1, 5))
    assert "bragg" not in COMPONENTS
