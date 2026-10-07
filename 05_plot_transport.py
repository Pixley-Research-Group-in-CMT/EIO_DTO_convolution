"""Plot EIO/TTO interface resistivity for either DSSF approximation."""
import argparse

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
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


def add_resistivity_density(axis, data):
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
        norm=normalized_color_norm(density),
    )
    axis.set_title("Fractional resistivity change", fontsize=8)
    axis.set_ylabel(r"Field $h/J_{zz}$")
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


def add_normalized_resistivity_polar(axis, data, fields, colors):
    """Plot signed fractional resistivity without reversing negative radii."""
    radial_values = data[NORMALIZED_COLUMN].to_numpy(dtype=float)
    signed_minimum = min(0.0, float(np.min(radial_values)))
    signed_maximum = max(0.0, float(np.max(radial_values)))
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


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--theory", choices=("spinon", "spin-wave"), default="spinon")
# Kept so jobs submitted with the former plotting option still complete.
parser.add_argument("--density-smoothing", type=float, help=argparse.SUPPRESS)
args = parser.parse_args()

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
parameter_parts = [
    rf"J_{{zz}}={j_zz_mev:g}\,\mathrm{{meV}}",
    rf"J_{{\pm}}/J_{{zz}}={j_pm_ratio:g}",
]
if args.theory == "spin-wave":
    parameter_parts.append(rf"J_{{\pm\pm}}/J_{{zz}}={j_pmpm_ratio:g}")
parameter_label = "$" + r",\quad ".join(parameter_parts) + "$"

fields = np.sort(data["h_over_J_zz"].unique())
colors = plt.get_cmap("magma")(np.linspace(0.1, 0.9, len(fields)))
figure = plt.figure(figsize=(13.2, 3.35))
grid = figure.add_gridspec(1, 4, width_ratios=(1.15, 1.15, 1.15, 1.0))
axes = [
    figure.add_subplot(grid[0, 0]),
    figure.add_subplot(grid[0, 1]),
    figure.add_subplot(grid[0, 2]),
    figure.add_subplot(grid[0, 3], projection="polar"),
]
for field, color in zip(fields, colors):
    selected = data[np.isclose(data["h_over_J_zz"], field)].sort_values("phi_degrees")
    label = fr"$h/J_{{zz}}={field:g}$"
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
if args.theory == "spinon":
    axes[0].set_yscale("log")
for label, axis in zip(("a", "b", "c"), axes[:3]):
    axis.set_xlabel(r"In-plane field angle $\phi$ (degrees)")
    if axis is not axes[2]:
        axis.set_xlim(data["phi_degrees"].min(), data["phi_degrees"].max())
    axis.set_xticks(thirty_degree_ticks(
        data["phi_degrees"].min(), data["phi_degrees"].max()
    ))
    axis.tick_params(width=0.8, length=3)
    axis.text(0.01, 1.02, label, transform=axis.transAxes, va="bottom", fontweight="bold")
axes[3].text(-0.08, 1.06, "d", transform=axes[3].transAxes, fontweight="bold")

figure.suptitle(f"{theory_label}\n{parameter_label}", fontsize=8.5, y=0.98)
figure.subplots_adjust(
    left=0.055, right=0.975, bottom=0.16, top=0.78, wspace=0.44,
)

FIGURE_DIR.mkdir(exist_ok=True)
output = FIGURE_DIR / output_name
figure.savefig(output, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
plt.close(figure)
print(f"Wrote {output}")
