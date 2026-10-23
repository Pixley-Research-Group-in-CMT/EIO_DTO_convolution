import numpy as np
import pytest

from swt_transport_components import (
    combine_transport_components,
    transport_component_definition,
)


def test_transport_component_selection_never_needs_bragg():
    one_magnon = np.array([1.0, 2.0])
    two_magnon = np.array([3.0, 5.0])
    modes = {
        f"one_magnon_{mode}": np.asarray([0.1 * mode, 0.2 * mode])
        for mode in range(1, 5)
    }

    assert np.array_equal(
        combine_transport_components(one_magnon, two_magnon, "one_magnon"),
        one_magnon,
    )
    assert np.array_equal(
        combine_transport_components(one_magnon, two_magnon, "two_magnon"),
        two_magnon,
    )
    assert np.array_equal(
        combine_transport_components(one_magnon, two_magnon, "total"),
        one_magnon + two_magnon,
    )
    assert np.array_equal(
        combine_transport_components(
            one_magnon,
            two_magnon,
            "one_magnon_1",
            one_magnon_modes=modes,
        ),
        modes["one_magnon_1"],
    )
    assert np.array_equal(
        combine_transport_components(
            one_magnon,
            two_magnon,
            "one_magnon_4",
            one_magnon_modes=modes,
        ),
        modes["one_magnon_4"],
    )
    assert "Bragg excluded" in transport_component_definition("total")
    assert "mode 1" in transport_component_definition("one_magnon_1")


def test_transport_component_selection_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown transport component"):
        combine_transport_components(np.ones(2), np.ones(2), "bragg")


def test_transport_component_selection_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        combine_transport_components(np.ones(2), np.ones(3), "total")


def test_band_transport_requires_band_resolved_response():
    with pytest.raises(ValueError, match="band-resolved"):
        combine_transport_components(
            np.ones(2), np.ones(2), "one_magnon_2"
        )
