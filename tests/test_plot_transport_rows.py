import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import QuadMesh
from matplotlib.animation import PillowWriter
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "05_plot_transport.py"
SPEC = importlib.util.spec_from_file_location("plot_transport_stage", MODULE_PATH)
plots = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plots)


def test_row_layout_uses_one_nonoverlaid_row_per_field(tmp_path):
    fields = np.asarray([0.25, 0.45, 1.5])
    angles = np.asarray([0.0, 30.0, 60.0])
    data = pd.DataFrame({
        "h_over_J_zz": np.repeat(fields, len(angles)),
        "phi_degrees": np.tile(angles, len(fields)),
        "rho": 1.0 + 0.1 * np.arange(9, dtype=float),
        plots.NORMALIZED_COLUMN: np.tile([0.0, 0.1, -0.05], len(fields)),
    })
    colors = plt.get_cmap("magma")(np.linspace(0.1, 0.9, len(fields)))
    figure = plots.plot_field_rows(
        data, fields, colors, "spin-wave", "SWT", "parameters"
    )

    data_axes = figure.axes[:4 * len(fields)]
    assert len(data_axes) == 12
    assert sum(axis.name == "polar" for axis in data_axes) == len(fields)
    for row_index, field in enumerate(fields):
        rho_axis, normalized_axis, density_axis, polar_axis = data_axes[
            4 * row_index:4 * row_index + 4
        ]
        assert len(rho_axis.lines) == 1
        assert len(normalized_axis.lines) == 1
        assert any(isinstance(item, QuadMesh) for item in density_axis.collections)
        assert polar_axis.name == "polar"
        assert any(f"={field:g}" in text.get_text() for text in rho_axis.texts)

    output = tmp_path / "rows.pdf"
    figure.savefig(output)
    assert output.stat().st_size > 0
    plt.close(figure)


def test_output_directory_can_be_overridden(tmp_path):
    args = plots.build_parser().parse_args([
        "--theory",
        "spin-wave",
        "--output-directory",
        str(tmp_path),
    ])

    assert args.output_directory == tmp_path


def test_movie_has_one_frame_per_field(tmp_path):
    fields = np.asarray([0.25, 0.45, 1.5])
    angles = np.asarray([0.0, 30.0, 60.0])
    data = pd.DataFrame({
        "h_over_J_zz": np.repeat(fields, len(angles)),
        "phi_degrees": np.tile(angles, len(fields)),
        "rho": 1.0 + 0.1 * np.arange(9, dtype=float),
        plots.NORMALIZED_COLUMN: np.concatenate([
            [0.0, 0.1, -0.05],
            [0.0, 0.2, -0.1],
            [0.0, 0.4, -0.2],
        ]),
    })
    figure, movie = plots.plot_field_movie(
        data, fields, "SWT", "parameters", fps=10.0
    )
    movie._func(0)
    first_limits = tuple(axis.get_ylim() for axis in figure.axes[:2])
    movie._func(2)
    last_limits = tuple(axis.get_ylim() for axis in figure.axes[:2])
    assert first_limits[0] != last_limits[0]
    assert first_limits[1] != last_limits[1]
    output = tmp_path / "fields.gif"
    movie.save(output, writer=PillowWriter(fps=10.0), dpi=60)
    plt.close(figure)
    with Image.open(output) as image:
        assert image.n_frames == len(fields)
