import numpy as np
import pandas as pd
from matplotlib.collections import QuadMesh

import plot_swt_three_field_results as plots
from plot_swt_three_field_results import (
    MIDDLE_FIELD_AMPLIFICATION,
    NORMALIZED_COLUMN,
    field_signals,
)


def test_only_middle_field_is_amplified_by_two():
    fields = np.asarray([0.25, 0.45, 1.5])
    data = pd.DataFrame({
        "h_over_J_zz": np.repeat(fields, 2),
        "phi_degrees": np.tile([0.0, 30.0], 3),
        NORMALIZED_COLUMN: np.tile([1.0, 2.0], 3),
    })
    signals = field_signals(data, fields)
    assert MIDDLE_FIELD_AMPLIFICATION == 2.0
    assert np.allclose(signals[0][1], [1.0, 2.0])
    assert np.allclose(signals[1][1], [2.0, 4.0])
    assert np.allclose(signals[2][1], [1.0, 2.0])


def test_heatmap_has_no_white_tick_grid_or_unpainted_cell_edges(
    monkeypatch, tmp_path
):
    fields = np.asarray([0.25, 0.45, 1.5])
    angles = np.asarray([-30.0, 0.0, 30.0])
    data = pd.DataFrame({
        "h_over_J_zz": np.repeat(fields, len(angles)),
        "phi_degrees": np.tile(angles, len(fields)),
        NORMALIZED_COLUMN: np.arange(9, dtype=float),
    })
    monkeypatch.setattr(plots.plt, "close", lambda figure: None)
    plots.plot_heatmap_rows(
        data, fields, angles, "test", tmp_path / "heatmap.pdf"
    )
    figure = plots.plt.gcf()
    for axis in figure.axes[:3]:
        assert not any(line.get_visible() for line in axis.get_xgridlines())
        mesh = next(
            collection for collection in axis.collections
            if isinstance(collection, QuadMesh)
        )
        assert mesh.get_edgecolors().size > 0
    plots.plt.close(figure)


def test_polar_titles_clear_ninety_degrees_and_radial_values_are_hidden(
    monkeypatch, tmp_path
):
    fields = np.asarray([0.25, 0.45, 1.5])
    angles = np.asarray([0.0, 30.0, 60.0])
    data = pd.DataFrame({
        "h_over_J_zz": np.repeat(fields, len(angles)),
        "phi_degrees": np.tile(angles, len(fields)),
        NORMALIZED_COLUMN: np.linspace(-0.2, 0.3, 9),
    })
    monkeypatch.setattr(plots.plt, "close", lambda figure: None)
    plots.plot_polar_panels(data, fields, "test", tmp_path / "polar.pdf")
    figure = plots.plt.gcf()
    for axis in figure.axes:
        assert axis.title.get_position()[1] >= 1.15
        assert not any(label.get_text() for label in axis.get_yticklabels())
    plots.plt.close(figure)
