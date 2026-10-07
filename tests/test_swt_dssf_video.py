import numpy as np
import pytest
from matplotlib.animation import PillowWriter

from plot_swt_dssf_video import animation_writer, energy_axis_from_dispersion


def test_energy_axis_spans_all_condition_dispersion_bounds():
    bounds = np.asarray([
        [[0.03, 0.40], [0.02, 0.51], [0.04, 0.47]],
        [[0.01, 0.43], [0.025, 0.49], [0.035, 0.46]],
    ])
    omega = energy_axis_from_dispersion(bounds, 6)
    assert np.isclose(omega[0], 0.01)
    assert np.isclose(omega[-1], 0.51)
    assert np.all(np.diff(omega) > 0.0)


def test_energy_axis_and_portable_writer_validation():
    assert isinstance(animation_writer("gif", 8), PillowWriter)
    with pytest.raises(ValueError, match="at least 2"):
        energy_axis_from_dispersion(np.asarray([[[0.1, 0.2]]]), 1)
    with pytest.raises(ValueError, match="Unknown video format"):
        animation_writer("avi", 8)
