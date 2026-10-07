#!/usr/bin/env python3
"""Plot a downloaded three-field SWT run as heatmap rows and polar panels."""
import argparse
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Edit this path to the downloaded run directory, data directory, or CSV file.
DOWNLOADED_RESULTS = Path("~/Desktop/research/SWT_transport/run_251")

TRANSPORT_NAME = "tto_spin_wave_transport_summary.csv"
METADATA_NAME = "tto_spin_wave_scattering.hdf5"
HEATMAP_NAME = "tto_spin_wave_three_field_heatmap.pdf"
POLAR_NAME = "tto_spin_wave_three_field_polar.pdf"
NORMALIZED_COLUMN = "normalized_resistivity"
NORMALIZED_LABEL = r"$[\rho(\phi)-\rho(0)]/\rho(0)$"
MIDDLE_FIELD_AMPLIFICATION = 4.0


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
})


def resolve_run_directory(path):
    """Accept a run directory, its data directory, or its transport CSV."""
    path = path.expanduser().resolve()
    if path.is_file():
        if path.name != TRANSPORT_NAME:
            raise ValueError(f"Expected {TRANSPORT_NAME}, received {path.name}")
        data_directory = path.parent
        run_directory = data_directory.parent if data_directory.name == "data" else data_directory
    elif path.is_dir() and path.name == "data":
        data_directory = path
        run_directory = path.parent
    elif path.is_dir() and (path / "data").is_dir():
        run_directory = path
        data_directory = path / "data"
    else:
        raise ValueError(
            f"{path} is not a run directory, data directory, or {TRANSPORT_NAME}"
        )

    missing = [
        name for name in (TRANSPORT_NAME, METADATA_NAME)
        if not (data_directory / name).is_file()
    ]
    if missing:
        raise ValueError("Downloaded output is missing: " + ", ".join(missing))
    return run_directory, data_directory


def ensure_normalized_resistivity(data):
    """Support new outputs and older files containing the previous ratio."""
    data = data.copy()
    if NORMALIZED_COLUMN in data:
        return data
    if "delta_rho_over_rho_phi0" in data:
        data[NORMALIZED_COLUMN] = data["delta_rho_over_rho_phi0"]
        return data
    if "rho_over_rho_phi0" in data:
        data[NORMALIZED_COLUMN] = data["rho_over_rho_phi0"] - 1.0
        return data
    references = data.groupby("h_over_J_zz")["rho"].transform(
        lambda values: values.iloc[np.argmin(np.abs(
            data.loc[values.index, "phi_degrees"].to_numpy(dtype=float)
        ))]
    )
    data[NORMALIZED_COLUMN] = (data["rho"] - references) / references
    return data


def validate_three_field_grid(data):
    """Return three fields and one shared, sorted nominal-angle grid."""
    required = {"phi_degrees", "h_over_J_zz", NORMALIZED_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Transport table is missing columns: {sorted(missing)}")
    if data.duplicated(["h_over_J_zz", "phi_degrees"]).any():
        raise ValueError("Transport table contains duplicate (h, phi) rows")

    fields = np.sort(data["h_over_J_zz"].unique())
    if len(fields) != 3:
        raise ValueError(f"Exactly three magnetic fields are required; found {len(fields)}")
    angles = np.sort(data["phi_degrees"].unique())
    for field in fields:
        selected = np.sort(
            data.loc[np.isclose(data["h_over_J_zz"], field), "phi_degrees"].unique()
        )
        if not np.array_equal(selected, angles):
            raise ValueError(f"Field h/J_zz={field:g} does not have the full angle grid")
    return fields, angles


def centered_edges(values):
    """Return cell edges centered on a sorted coordinate array."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or np.any(np.diff(values) <= 0.0):
        raise ValueError("At least two strictly increasing angles are required")
    edges = np.empty(values.size + 1)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def thirty_degree_ticks(minimum, maximum):
    start = 30.0 * np.ceil(float(minimum) / 30.0)
    stop = 30.0 * np.floor(float(maximum) / 30.0)
    return np.arange(start, stop + 15.0, 30.0)


def color_norm(values):
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if np.isclose(minimum, maximum):
        padding = max(1e-9, 0.01 * max(abs(minimum), 1.0))
        minimum -= padding
        maximum += padding
    if minimum < 0.0 < maximum:
        return mpl.colors.TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)
    return mpl.colors.Normalize(vmin=minimum, vmax=maximum)


def field_signals(data, fields):
    """Return sorted plotting signals, amplifying only the middle one."""
    signals = []
    for index, field in enumerate(fields):
        selected = data[np.isclose(data["h_over_J_zz"], field)].sort_values(
            "phi_degrees"
        )
        values = selected[NORMALIZED_COLUMN].to_numpy(dtype=float)
        factor = MIDDLE_FIELD_AMPLIFICATION if index == 1 else 1.0
        signals.append((selected["phi_degrees"].to_numpy(dtype=float), factor * values))
    return signals


def parameter_label(metadata_path):
    with h5py.File(metadata_path) as metadata:
        j_zz_mev = 1e3 * float(metadata.attrs["J_zz_ev"])
        j_pm = float(metadata.attrs["J_pm_over_J_zz"])
        j_pmpm = float(metadata.attrs.get("J_pmpm_over_J_zz", 0.0))
    return (
        rf"$J_{{zz}}={j_zz_mev:g}\,\mathrm{{meV}},\ "
        rf"J_{{\pm}}/J_{{zz}}={j_pm:g},\ "
        rf"J_{{\pm\pm}}/J_{{zz}}={j_pmpm:g}$"
    )


def plot_heatmap_rows(data, fields, angles, title, output):
    signals = field_signals(data, fields)
    plotted_values = np.concatenate([values for _, values in signals])
    norm = color_norm(plotted_values)
    angle_edges = centered_edges(angles)
    figure, axes = plt.subplots(3, 1, figsize=(8.4, 5.6), sharex=True)
    image = None
    for index, (axis, field, (_, values)) in enumerate(zip(axes, fields, signals)):
        image = axis.pcolormesh(
            angle_edges, np.asarray([0.0, 1.0]), values[np.newaxis, :],
            cmap="magma", norm=norm, shading="flat",
            edgecolors="face", linewidth=0.1, antialiased=False, snap=True,
        )
        suffix = (
            f"\n(signal $\\times {MIDDLE_FIELD_AMPLIFICATION:g}$)"
            if index == 1 else ""
        )
        axis.set_ylabel(
            rf"$h/J_{{zz}}={field:g}$" + suffix,
            rotation=0, ha="right", va="center", labelpad=10,
        )
        axis.set_yticks([])
        axis.set_xlim(angle_edges[0], angle_edges[-1])
        axis.set_xticks(thirty_degree_ticks(angles[0], angles[-1]))
        for spine in axis.spines.values():
            spine.set_visible(True)
    axes[-1].set_xlabel(r"Nominal in-plane field angle $\phi$ (degrees)")
    colorbar = figure.colorbar(image, ax=axes, pad=0.025, fraction=0.035)
    colorbar.set_label(NORMALIZED_LABEL)
    figure.suptitle(
        "SWT fractional resistivity change\n" + title
        + f"\nMiddle-field signal is multiplied by "
        f"{MIDDLE_FIELD_AMPLIFICATION:g} for plotting only",
        fontsize=10,
    )
    figure.subplots_adjust(left=0.18, right=0.84, bottom=0.11, top=0.78, hspace=0.42)
    figure.savefig(output, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def plot_polar_panels(data, fields, title, output):
    signals = field_signals(data, fields)
    plotted_values = np.concatenate([values for _, values in signals])
    signed_minimum = min(0.0, float(np.min(plotted_values)))
    signed_maximum = max(0.0, float(np.max(plotted_values)))
    if np.isclose(signed_minimum, signed_maximum):
        signed_maximum = signed_minimum + 1e-3
    radial_offset = -signed_minimum
    radial_span = signed_maximum - signed_minimum
    radial_ticks = np.linspace(signed_minimum, signed_maximum, 4)
    colors = plt.get_cmap("magma")(np.linspace(0.12, 0.88, 3))

    figure, axes = plt.subplots(
        1, 3, figsize=(10.6, 4.0), subplot_kw={"projection": "polar"}
    )
    for index, (axis, field, (angles, values), color) in enumerate(
        zip(axes, fields, signals, colors)
    ):
        axis.plot(
            np.deg2rad(angles), values + radial_offset,
            color=color, lw=1.25, marker="o", ms=1.8,
        )
        if signed_minimum < 0.0 < signed_maximum:
            circle = np.linspace(0.0, 2.0 * np.pi, 361)
            axis.plot(
                circle, np.full_like(circle, radial_offset),
                color="0.45", ls="--", lw=0.6,
            )
        axis.set_ylim(0.0, 1.08 * radial_span)
        axis.set_yticks(radial_ticks + radial_offset)
        axis.set_yticklabels([])
        axis.set_theta_zero_location("E")
        axis.set_theta_direction(1)
        axis.set_thetagrids(np.arange(0, 360, 30), fontsize=7)
        axis.grid(linewidth=0.5, alpha=0.55)
        suffix = (
            rf" (signal $\times {MIDDLE_FIELD_AMPLIFICATION:g}$)"
            if index == 1 else ""
        )
        axis.set_title(
            rf"$h/J_{{zz}}={field:g}$" + suffix,
            fontsize=9,
            y=1.18,
            pad=0,
        )

    figure.suptitle(
        "SWT fractional resistivity change\n" + title
        + f"\nCommon radial scale; middle-field signal is multiplied by "
        f"{MIDDLE_FIELD_AMPLIFICATION:g}",
        fontsize=10,
    )
    figure.text(0.5, 0.025, "Radial coordinate: " + NORMALIZED_LABEL, ha="center")
    figure.subplots_adjust(left=0.04, right=0.98, bottom=0.12, top=0.70, wspace=0.34)
    figure.savefig(output, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", nargs="?", type=Path, default=DOWNLOADED_RESULTS,
        help="run directory, data directory, or transport-summary CSV",
    )
    args = parser.parse_args()
    try:
        run_directory, data_directory = resolve_run_directory(args.results)
        data = ensure_normalized_resistivity(
            pd.read_csv(data_directory / TRANSPORT_NAME)
        )
        fields, angles = validate_three_field_grid(data)
    except ValueError as error:
        parser.error(str(error))

    figures = run_directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    title = parameter_label(data_directory / METADATA_NAME)
    heatmap = figures / HEATMAP_NAME
    polar = figures / POLAR_NAME
    plot_heatmap_rows(data, fields, angles, title, heatmap)
    plot_polar_panels(data, fields, title, polar)
    print(f"Heatmap: {heatmap}")
    print(f"Polar: {polar}")


if __name__ == "__main__":
    main()
