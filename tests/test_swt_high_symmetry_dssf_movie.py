from types import SimpleNamespace

import numpy as np
import pytest

from plot_swt_high_symmetry_dssf_movie import (
    field_axis,
    write_sublattice_video,
    write_sum_video,
)


def test_field_axis_includes_chosen_endpoints():
    fields = field_axis(0.25, 1.5, 6)
    assert np.isclose(fields[0], 0.25)
    assert np.isclose(fields[-1], 1.5)
    assert np.allclose(np.diff(fields), np.diff(fields)[0])


@pytest.mark.parametrize(
    "h_start,h_stop,n_frames,match",
    [
        (-0.1, 1.0, 3, "field range"),
        (1.0, 1.0, 3, "field range"),
        (0.1, 1.0, 1, "at least 2"),
    ],
)
def test_field_axis_rejects_invalid_ranges(h_start, h_stop, n_frames, match):
    with pytest.raises(ValueError, match=match):
        field_axis(h_start, h_stop, n_frames)


def test_small_field_movies_encode(tmp_path):
    values = np.arange(2 * 3 * 3 * 7, dtype=float).reshape(2, 3, 3, 7)
    omega_mev = np.linspace(0.0, 0.2, 3)
    path_distance = np.linspace(0.0, 1.0, 7)
    tick_positions = path_distance
    fields = np.asarray([0.25, 0.5])
    args = SimpleNamespace(
        vmax_percentile=100.0,
        j_zz_ev=0.0002,
        j_pm=0.2,
        j_pmpm=0.0,
        g_z=10.0,
        g_perp=0.0,
        delta1_per_mev=1.0 / 1.4,
        delta2_per_mev=1.0 / 1.4,
        component="one_magnon",
        fps=2.0,
        video_format="gif",
        dpi=40,
    )
    sublattice_path = tmp_path / "sublattices.gif"
    sum_path = tmp_path / "sum.gif"

    write_sublattice_video(
        values,
        omega_mev,
        path_distance,
        tick_positions,
        fields,
        0.1,
        args,
        sublattice_path,
    )
    write_sum_video(
        values,
        omega_mev,
        path_distance,
        tick_positions,
        fields,
        0.1,
        args,
        sum_path,
    )

    assert sublattice_path.stat().st_size > 0
    assert sum_path.stat().st_size > 0
