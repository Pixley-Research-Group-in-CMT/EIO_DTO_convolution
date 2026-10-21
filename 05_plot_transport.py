"""Plot EIO/TTO interface resistivity for either DSSF approximation."""
import argparse
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

from config import DATA_DIR, FIGURE_DIR


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
})

NORMALIZED_COLUMN = "normalized_resistivity"
NORMALIZED_LABEL = r"$[\rho(\phi)-\rho(0)]/\rho(0)$"


def ensure_normalized_resistivity(data):
    """Add the fractional-change convention to old or new output files."""
    data = data.copy()
    if NORMALIZED_COLUMN in data:
        return data
    if "delta_rho_over_rho_phi0" in data:
        data[NORMALIZED_COLUMN] = data["delta_rho_over_rho_phi0"]
        return data
    if "rho_over_rho_phi0" in data:
        data[NORMALIZED_COLUMN] = data["rho_over_rho_phi0"] - 1.0
        return data
    if "rho" not in data:
        raise ValueError("Transport table has no resistivity column to normalize")
    references = data.groupby("h_over_J_zz")["rho"].transform(
        lambda values: values.iloc[np.argmin(np.abs(
            data.loc[values.index, "phi_degrees"].to_numpy(dtype=float)
        ))]
    )
    data[NORMALIZED_COLUMN] = (data["rho"] - references) / references
    return data


def thirty_degree_ticks(minimum, maximum):
    """Return labeled multiples of 30 degrees within an angular interval."""
    start = 30.0 * np.ceil(float(minimum) / 30.0)
    stop = 30.0 * np.floor(float(maximum) / 30.0)
    return np.arange(start, stop + 15.0, 30.0)


def normalized_color_norm(values):
    """Use one finite color scale, centered on zero when signs differ."""
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if np.isclose(minimum, maximum):
        padding = max(1e-9, 0.01 * max(abs(minimum), 1.0))
        minimum -= padding
        maximum += padding
    if minimum < 0.0 < maximum:
        return mpl.colors.TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)
    return mpl.colors.Normalize(vmin=minimum, vmax=maximum)


def resistivity_density_grid(data):
    """Return normalized resistivity on the evaluated (phi, h) grid."""
    required = {"phi_degrees", "h_over_J_zz", NORMALIZED_COLUMN}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Transport table is missing columns: {sorted(missing_columns)}")

    coordinates = ["h_over_J_zz", "phi_degrees"]
    if data.duplicated(coordinates).any():
        raise ValueError("Transport table contains duplicate (h, phi) conditions")
    fields = np.sort(data["h_over_J_zz"].unique())
    angles = np.sort(data["phi_degrees"].unique())
    grid = data.pivot(
        index="h_over_J_zz", columns="phi_degrees", values=NORMALIZED_COLUMN
    ).reindex(index=fields, columns=angles)
    if grid.isna().any().any():
        missing = int(grid.isna().sum().sum())
        raise ValueError(f"Transport table is missing {missing} (h, phi) conditions")
    values = grid.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Normalized resistivity grid contains non-finite values")
    return angles, fields, values


def centered_cell_edges(coordinates, single_half_width):
    """Return cell edges for values sampled at the supplied coordinates."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 1 or coordinates.size == 0:
        raise ValueError("Cell coordinates must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Cell coordinates must be finite")
    if coordinates.size == 1:
        return np.array([
            coordinates[0] - single_half_width,
            coordinates[0] + single_half_width,
        ])

    spacing = np.diff(coordinates)
    if np.any(spacing <= 0.0):
        raise ValueError("Cell coordinates must be strictly increasing")
    edges = np.empty(coordinates.size + 1, dtype=float)
    edges[1:-1] = coordinates[:-1] + 0.5 * spacing
    edges[0] = coordinates[0] - 0.5 * spacing[0]
    edges[-1] = coordinates[-1] + 0.5 * spacing[-1]
    return edges


def add_resistivity_density(axis, data, norm=None):
    """Plot one normalized-resistivity cell per evaluated (phi, h) point."""
    angles, fields, density = resistivity_density_grid(data)
    angle_edges = centered_cell_edges(angles, single_half_width=0.5)
    field_half_width = max(0.05, 0.05 * abs(fields[0]))
    field_edges = centered_cell_edges(fields, single_half_width=field_half_width)
    image = axis.pcolormesh(
        angle_edges,
        field_edges,
        density,
        shading="flat",
        cmap="magma",
        norm=normalized_color_norm(density) if norm is None else norm,
    )
    axis.set_title("Fractional resistivity change", fontsize=8)
    axis.set_ylabel(r"Field energy $\mu_B|B|/J_{zz}$")
    if len(fields) > 1:
        minimum_spacing = 0.09 * (fields[-1] - fields[0])
        labeled_fields = [fields[0]]
        for field in fields[1:-1]:
            if (
                field - labeled_fields[-1] >= minimum_spacing
                and fields[-1] - field >= minimum_spacing
            ):
                labeled_fields.append(field)
        labeled_fields.append(fields[-1])
    else:
        labeled_fields = fields
    axis.set_yticks(labeled_fields)
    axis.set_yticks(fields, minor=True)
    axis.tick_params(axis="y", which="minor", length=2)
    axis.set_xlim(angle_edges[0], angle_edges[-1])
    axis.set_ylim(field_edges[0], field_edges[-1])
    return image


def add_normalized_resistivity_polar(
    axis, data, fields, colors, radial_limits=None
):
    """Plot signed fractional resistivity without reversing negative radii."""
    if radial_limits is None:
        radial_values = data[NORMALIZED_COLUMN].to_numpy(dtype=float)
        signed_minimum = min(0.0, float(np.min(radial_values)))
        signed_maximum = max(0.0, float(np.max(radial_values)))
    else:
        signed_minimum, signed_maximum = map(float, radial_limits)
    if np.isclose(signed_minimum, signed_maximum):
        signed_maximum = signed_minimum + 1e-3
    radial_offset = -signed_minimum
    radial_span = signed_maximum - signed_minimum

    for field, color in zip(fields, colors):
        selected = data[np.isclose(data["h_over_J_zz"], field)].sort_values(
            "phi_degrees"
        )
        axis.plot(
            np.deg2rad(selected["phi_degrees"].to_numpy(dtype=float)),
            selected[NORMALIZED_COLUMN].to_numpy(dtype=float) + radial_offset,
            marker="o",
            ms=2.4,
            lw=1.0,
            color=color,
        )

    radial_ticks = np.linspace(signed_minimum, signed_maximum, 4)
    axis.set_ylim(0.0, 1.08 * radial_span)
    axis.set_yticks(radial_ticks + radial_offset)
    axis.set_yticklabels([f"{value:.3g}" for value in radial_ticks])
    if signed_minimum < 0.0 < signed_maximum:
        full_circle = np.linspace(0.0, 2.0 * np.pi, 361)
        axis.plot(
            full_circle, np.full_like(full_circle, radial_offset),
            color="0.45", ls="--", lw=0.6,
        )
    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.set_thetagrids(np.arange(0, 360, 30))
    axis.set_rlabel_position(135)
    axis.grid(linewidth=0.5, alpha=0.5)
    axis.set_title("Fractional-change polar response", fontsize=8, pad=10)


def _configure_angle_axis(axis, data, show_xlabel=True):
    """Apply the common nominal-angle limits and ticks to a Cartesian axis."""
    angle_minimum = float(data["phi_degrees"].min())
    angle_maximum = float(data["phi_degrees"].max())
    axis.set_xlim(angle_minimum, angle_maximum)
    axis.set_xticks(thirty_degree_ticks(angle_minimum, angle_maximum))
    axis.tick_params(width=0.8, length=3, labelbottom=show_xlabel)
    if show_xlabel:
        axis.set_xlabel(r"In-plane field angle $\phi$ (degrees)")


def plot_overlaid_fields(data, fields, colors, theory, theory_label, parameters):
    """Return the original single-row figure with fields overlaid."""
    figure = plt.figure(figsize=(13.2, 3.35))
    grid = figure.add_gridspec(1, 4, width_ratios=(1.15, 1.15, 1.15, 1.0))
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[0, 2]),
        figure.add_subplot(grid[0, 3], projection="polar"),
    ]
    for field, color in zip(fields, colors):
        selected = data[np.isclose(data["h_over_J_zz"], field)].sort_values(
            "phi_degrees"
        )
        label = fr"$\mu_B|B|/J_{{zz}}={field:g}$"
        axes[0].plot(
            selected["phi_degrees"], selected["rho"], marker="o", ms=2.4,
            lw=1.0, color=color, label=label,
        )
        axes[1].plot(
            selected["phi_degrees"], selected[NORMALIZED_COLUMN],
            marker="o", ms=2.4, lw=1.0, color=color,
        )

    density_image = add_resistivity_density(axes[2], data)
    add_normalized_resistivity_polar(axes[3], data, fields, colors)
    colorbar = figure.colorbar(density_image, ax=axes[2], pad=0.03)
    colorbar.set_label(NORMALIZED_LABEL)
    axes[0].set_ylabel(r"Interface resistivity $\rho_{\mathrm{int}}$ (a.u.)")
    axes[0].legend(frameon=False, fontsize=6.5, handlelength=1.4, ncol=2)
    axes[1].set_ylabel(NORMALIZED_LABEL)
    if theory == "spinon":
        axes[0].set_yscale("log")
    for label, axis in zip(("a", "b", "c"), axes[:3]):
        _configure_angle_axis(axis, data)
        axis.text(
            0.01, 1.02, label, transform=axis.transAxes,
            va="bottom", fontweight="bold",
        )
    axes[3].text(
        -0.08, 1.06, "d", transform=axes[3].transAxes, fontweight="bold"
    )
    figure.suptitle(f"{theory_label}\n{parameters}", fontsize=8.5, y=0.98)
    figure.subplots_adjust(
        left=0.055, right=0.975, bottom=0.16, top=0.78, wspace=0.44,
    )
    return figure


def plot_field_rows(data, fields, colors, theory, theory_label, parameters):
    """Return a four-column figure with one non-overlaid row per field."""
    row_count = len(fields)
    figure = plt.figure(
        figsize=(13.2, max(3.8, 2.35 * row_count + 0.8)),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        row_count, 4, width_ratios=(1.15, 1.15, 1.15, 1.0)
    )
    axes = []
    density_norm = normalized_color_norm(
        data[NORMALIZED_COLUMN].to_numpy(dtype=float)
    )
    normalized_values = data[NORMALIZED_COLUMN].to_numpy(dtype=float)
    signed_minimum = min(0.0, float(np.min(normalized_values)))
    signed_maximum = max(0.0, float(np.max(normalized_values)))
    if np.isclose(signed_minimum, signed_maximum):
        signed_maximum = signed_minimum + 1e-3
    normalized_padding = 0.05 * (signed_maximum - signed_minimum)

    density_axes = []
    density_image = None
    for row_index, (field, color) in enumerate(zip(fields, colors)):
        selected = data[np.isclose(data["h_over_J_zz"], field)].sort_values(
            "phi_degrees"
        )
        row_axes = [
            figure.add_subplot(grid[row_index, 0]),
            figure.add_subplot(grid[row_index, 1]),
            figure.add_subplot(grid[row_index, 2]),
            figure.add_subplot(grid[row_index, 3], projection="polar"),
        ]
        axes.append(row_axes)
        row_axes[0].plot(
            selected["phi_degrees"], selected["rho"], marker="o", ms=2.4,
            lw=1.0, color=color,
        )
        row_axes[1].plot(
            selected["phi_degrees"], selected[NORMALIZED_COLUMN],
            marker="o", ms=2.4, lw=1.0, color=color,
        )
        density_image = add_resistivity_density(
            row_axes[2], selected, norm=density_norm
        )
        density_axes.append(row_axes[2])
        add_normalized_resistivity_polar(
            row_axes[3], selected, [field], [color],
            radial_limits=(signed_minimum, signed_maximum),
        )

        field_label = fr"$\mu_B|B|/J_{{zz}}={field:g}$"
        row_axes[0].text(
            0.02, 0.94, field_label, transform=row_axes[0].transAxes,
            va="top", ha="left", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
        row_axes[0].set_ylabel(r"$\rho_{\mathrm{int}}$ (a.u.)")
        row_axes[1].set_ylabel(NORMALIZED_LABEL)
        row_axes[1].set_ylim(
            signed_minimum - normalized_padding,
            signed_maximum + normalized_padding,
        )
        if theory == "spinon":
            row_axes[0].set_yscale("log")

        show_xlabel = row_index == row_count - 1
        for axis in row_axes[:2]:
            _configure_angle_axis(axis, data, show_xlabel=show_xlabel)
        _configure_angle_axis(row_axes[2], data, show_xlabel=show_xlabel)
        row_axes[2].set_yticks([])
        row_axes[2].set_ylabel("")
        row_axes[2].set_title("")
        row_axes[3].set_title("")
        row_axes[3].set_yticklabels([])

    column_titles = (
        "Interface resistivity",
        "Fractional resistivity change",
        "Evaluated-point heatmap",
        "Polar fractional change",
    )
    for label, axis, title in zip(("a", "b", "c", "d"), axes[0], column_titles):
        axis.set_title(title, fontsize=8, pad=10)
        axis.text(
            -0.08 if label == "d" else 0.01,
            1.06 if label == "d" else 1.02,
            label,
            transform=axis.transAxes,
            va="bottom",
            fontweight="bold",
        )

    colorbar = figure.colorbar(
        density_image, ax=density_axes, pad=0.02, shrink=0.92
    )
    colorbar.set_label(NORMALIZED_LABEL)
    figure.suptitle(f"{theory_label}\n{parameters}", fontsize=8.5)
    return figure


def padded_limits(values):
    """Return stable finite limits for one movie panel and frame."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 0.0, 1.0
    lower, upper = finite.min(), finite.max()
    span = upper - lower
    padding = 0.05 * (span if span else max(abs(lower), 1.0))
    return lower - padding, upper + padding


def plot_field_movie(data, fields, theory_label, parameters, fps):
    """Return an animation with one independently scaled field per frame."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("movie fps must be finite and positive")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)
    figure.suptitle(f"{theory_label}\n{parameters}", fontsize=9)
    axes[0].set_ylabel(r"Interface resistivity $\rho_{\mathrm{int}}$ (a.u.)")
    axes[1].set_ylabel(NORMALIZED_LABEL)
    for axis in axes:
        _configure_angle_axis(axis, data)
        axis.grid(alpha=0.2)
    absolute, = axes[0].plot([], [], marker="o", ms=2.4, lw=1.0)
    normalized, = axes[1].plot([], [], marker="o", ms=2.4, lw=1.0)
    frame_title = axes[0].set_title("")
    groups = [
        data[np.isclose(data["h_over_J_zz"], field)].sort_values("phi_degrees")
        for field in fields
    ]

    def update(frame):
        selected = groups[frame]
        absolute.set_data(selected["phi_degrees"], selected["rho"])
        normalized.set_data(
            selected["phi_degrees"], selected[NORMALIZED_COLUMN]
        )
        axes[0].set_ylim(*padded_limits(selected["rho"]))
        axes[1].set_ylim(*padded_limits(selected[NORMALIZED_COLUMN]))
        frame_title.set_text(
            rf"Frame {frame + 1}/{len(fields)}: "
            rf"$\mu_B|B|/J_{{zz}}={fields[frame]:g}$"
        )
        return absolute, normalized, frame_title

    movie = FuncAnimation(
        figure,
        update,
        frames=len(fields),
        interval=1000.0 / fps,
        repeat=True,
    )
    return figure, movie


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--theory", choices=("spinon", "spin-wave"), default="spinon"
    )
    parser.add_argument(
        "--field-layout",
        choices=("overlaid", "rows"),
        default=None,
        help="overlay all fields or draw one row per field",
    )
    parser.add_argument(
        "--plot-mode",
        choices=("overlay", "rows", "movie"),
        help="static field layout or one animated frame per field",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=FIGURE_DIR,
        help="directory in which to save the generated output",
    )
    parser.add_argument("--output-name", help="override the PDF or GIF filename")
    parser.add_argument("--movie-fps", type=float, default=2.0)
    # Kept so jobs submitted with the former plotting option still complete.
    parser.add_argument("--density-smoothing", type=float, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.theory == "spinon":
        data_path = DATA_DIR / "tto_transport_summary.csv"
        metadata_path = DATA_DIR / "tto_dynamic_scattering.hdf5"
        output_name = "tto_interface_resistivity.pdf"
        theory_label = "Spinon DSSF (Sec. 4.1)"
    else:
        data_path = DATA_DIR / "tto_spin_wave_transport_summary.csv"
        metadata_path = DATA_DIR / "tto_spin_wave_scattering.hdf5"
        output_name = "tto_spin_wave_interface_resistivity.pdf"
        theory_label = "Spin-wave DSSF (Sec. 4.2)"

    data = ensure_normalized_resistivity(pd.read_csv(data_path))
    with h5py.File(metadata_path) as metadata_h5:
        j_zz_mev = 1e3 * float(metadata_h5.attrs["J_zz_ev"])
        j_pm_ratio = float(metadata_h5.attrs["J_pm_over_J_zz"])
        j_pmpm_ratio = float(metadata_h5.attrs.get("J_pmpm_over_J_zz", 0.0))
        g_z = float(metadata_h5.attrs.get("g_z", 1.0))
        g_perp = float(metadata_h5.attrs.get("g_perp", 0.0))
        delta1_per_mev = float(metadata_h5.attrs.get("delta1_per_mev", 0.0))
        delta2_per_mev = float(metadata_h5.attrs.get("delta2_per_mev", 0.0))
    parameter_parts = [
        rf"J_{{zz}}={j_zz_mev:g}\,\mathrm{{meV}}",
        rf"J_{{\pm}}/J_{{zz}}={j_pm_ratio:g}",
    ]
    if args.theory == "spin-wave":
        parameter_parts.append(rf"J_{{\pm\pm}}/J_{{zz}}={j_pmpm_ratio:g}")
        parameter_parts.extend((
            rf"g_z={g_z:g}",
            rf"g_\perp={g_perp:g}",
            rf"\delta_1={delta1_per_mev:g}\,\mathrm{{meV}}^{{-1}}",
            rf"\delta_2={delta2_per_mev:g}\,\mathrm{{meV}}^{{-1}}",
        ))
    parameter_label = "$" + r",\quad ".join(parameter_parts) + "$"

    fields = np.sort(data["h_over_J_zz"].unique())
    colors = plt.get_cmap("magma")(np.linspace(0.1, 0.9, len(fields)))
    plot_mode = args.plot_mode or (
        "rows" if args.field_layout == "rows" else "overlay"
    )
    if plot_mode == "rows":
        figure = plot_field_rows(
            data, fields, colors, args.theory, theory_label, parameter_label
        )
        movie = None
    elif plot_mode == "overlay":
        figure = plot_overlaid_fields(
            data, fields, colors, args.theory, theory_label, parameter_label
        )
        movie = None
    else:
        figure, movie = plot_field_movie(
            data, fields, theory_label, parameter_label, args.movie_fps
        )

    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    if args.output_name:
        output_name = args.output_name
    elif plot_mode == "movie":
        output_name = Path(output_name).with_suffix(".gif").name
    output = output_directory / output_name
    if plot_mode == "movie":
        if output.suffix.lower() != ".gif":
            raise ValueError("movie output filename must end in .gif")
        movie.save(output, writer=PillowWriter(fps=args.movie_fps), dpi=120)
    else:
        if output.suffix.lower() != ".pdf":
            raise ValueError("static output filename must end in .pdf")
        figure.savefig(
            output,
            bbox_inches="tight",
            metadata={"CreationDate": None, "ModDate": None},
        )
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
