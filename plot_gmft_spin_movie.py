"""Animate winning gMFT link spins over the configured phi scan.

Each frame matches the reference's 2x4 heatmaps: signed s_z above |s_perp|,
field versus J_pm/Jzz. A single run supplies one J_pm column. Compatible
runs can supply additional columns. GIF encoding uses memory, with no PNG
or SVG frame files; the accompanying multipage PDF keeps vector figures.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def edges(values, single_width=.02):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.array([values[0]-single_width/2, values[0]+single_width/2])
    middle = (values[1:] + values[:-1])/2
    return np.r_[values[0]-(values[1]-values[0])/2, middle,
                 values[-1]+(values[-1]-values[-2])/2]


def configured_phi_angles(config):
    """Return endpoint-inclusive nominal angles for old and new run manifests."""
    return np.linspace(
        float(config.get("phi_start_degrees", 0.0)),
        float(config.get("phi_stop_degrees", 30.0)),
        int(config.get("n_phi", 30)) + 1,
    )


def load_grid(run_directories):
    runs = [Path(p).expanduser().resolve() for p in run_directories]
    configs = [json.loads((run / "manifest.json").read_text())["config"] for run in runs]
    phi_angles = configured_phi_angles(configs[0])
    physical = ("g_z", "g_perp", "delta1_per_mev", "delta2_per_mev", "j_zz_ev",
                "theta_degrees", "J_pmpm_ratio", "nk", "q_perp_mode", "n_q_perp")
    fields = np.sort(np.asarray(configs[0]["fields"], dtype=float))
    for config in configs[1:]:
        if config.get("phi_offset_degrees", 0.) != configs[0].get("phi_offset_degrees", 0.):
            raise ValueError("Combined runs must use the same phi offset")
        if any(config[k] != configs[0][k] for k in physical):
            raise ValueError("Combined runs must have identical field couplings, grids and projection settings")
        if not np.array_equal(fields, np.sort(config["fields"])):
            raise ValueError("Combined runs must have the same field values")
        other_phi_angles = configured_phi_angles(config)
        if len(phi_angles) != len(other_phi_angles) or not np.allclose(
            phi_angles, other_phi_angles
        ):
            raise ValueError("Combined runs must use the same nominal phi scan")
    jpm = np.asarray([c["J_pm_ratio"] for c in configs])
    if len(np.unique(jpm)) != len(jpm):
        raise ValueError("Provide at most one run per J_pm value")
    order = np.argsort(jpm)
    z = np.full((len(phi_angles), len(fields), len(runs), 4), np.nan)
    perp = np.full_like(z, np.nan)
    for column, index in enumerate(order):
        run = runs[index]
        table = pd.read_csv(run / "winning_saddles.csv")
        condition_path = run / "data/tto_gmft_conditions.csv"
        processed = pd.read_csv(condition_path) if condition_path.exists() else None
        for phi_index, phi in enumerate(phi_angles):
            for field_index, field in enumerate(fields):
                selected = table[np.isclose(table.phi_degrees, phi)
                    & np.isclose(table.h_over_J_zz, field, rtol=1e-10, atol=1e-14)]
                if len(selected) > 1:
                    raise ValueError(f"Duplicate winner at phi={phi}, field={field} in {run}")
                if selected.empty or selected.iloc[0].status != "ok":
                    continue
                if processed is not None:
                    response = processed[np.isclose(processed.phi_degrees, phi)
                        & np.isclose(processed.h_over_J_zz, field, rtol=1e-10, atol=1e-14)]
                    if len(response) == 1 and response.iloc[0].status != "ok":
                        continue
                with np.load(run / selected.iloc[0].saddle_file, allow_pickle=False) as saddle:
                    sz, sm = saddle["s_z"], saddle["s_minus"]
                    if sz.shape != (4,) or sm.shape != (4,) or not np.isfinite(sz).all() or not np.isfinite(sm).all():
                        raise ValueError("Winning saddle must contain four finite spin components")
                    if np.max(np.abs(sz.imag)) > 1e-10:
                        raise ValueError("s_z must be real")
                    z[phi_index, field_index, column] = sz.real
                    perp[phi_index, field_index, column] = np.abs(sm)
    return configs[0], fields, jpm[order], z, perp


def make_figure(config, fields, jpm, z, perp):
    phi_angles = configured_phi_angles(config)
    figure, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    xedges, yedges = edges(jpm), edges(fields)
    yedges[0] = max(0., yedges[0])
    meshes = []
    for row, (label, values) in enumerate(((r"s^z_\mu", z), (r"|s^\perp_\mu|", perp))):
        for mu in range(4):
            axis = axes[row, mu]
            mesh = axis.pcolormesh(xedges, yedges, values[0, :, :, mu],
                cmap="coolwarm", vmin=-.5, vmax=.5, shading="flat",
                edgecolors="black", linewidth=.35, rasterized=False)
            meshes.append(mesh)
            axis.set_title(rf"${label}$, $\mu={mu}$")
            axis.set_xticks(jpm)
            axis.set_yticks(fields)
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            if row == 1:
                axis.set_xlabel(r"$J_\pm/J_{zz}$")
            if mu == 0:
                axis.set_ylabel(r"$\mu_B|B|/J_{zz}$")
        colorbar = figure.colorbar(mesh, ax=list(axes[row]), shrink=.85,
                                  ticks=[-.5, -.25, 0., .25, .5])
        colorbar.solids.set_rasterized(False)
    title = figure.suptitle("")
    def update(phi_index):
        for i, mesh in enumerate(meshes):
            values = z if i < 4 else perp
            mesh.set_array(values[phi_index, :, :, i % 4].ravel())
        title.set_text(
            rf"Winning gMFT states: $\phi={phi_angles[phi_index]:g}^\circ$, "
            rf"$\theta={config['theta_degrees']:g}^\circ$, $g_z={config['g_z']:g}$"
            "\n" + rf"$g_\perp={config['g_perp']:g}$, "
            rf"$\delta_1={config['delta1_per_mev']:g}$, "
            rf"$\delta_2={config['delta2_per_mev']:g}$ meV$^{{-1}}$; "
            + config["q_perp_mode"] + " transport run")
    update(0)
    # Freeze geometry before switching between PDF and GIF renderers/DPI.
    # Repeated constrained-layout passes otherwise move the panels per frame.
    figure.canvas.draw()
    if hasattr(figure, "set_layout_engine"):
        figure.set_layout_engine("none")
    else:
        figure.set_constrained_layout(False)
    return figure, update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directories", nargs="+", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--fps", type=float, default=3.)
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("fps must be finite and positive")
    config, fields, jpm, z, perp = load_grid(args.run_directories)
    output = (args.output_directory or args.run_directories[0]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure, update = make_figure(config, fields, jpm, z, perp)
    gif, pdf = output / "gmft_spin_components.gif", output / "gmft_spin_components.pdf"
    writer = PillowWriter(fps=args.fps)
    with PdfPages(pdf) as pages, writer.saving(figure, str(gif), dpi=90):
        for phi_index in range(len(configured_phi_angles(config))):
            update(phi_index)
            pages.savefig(figure)
            writer.grab_frame()
    plt.close(figure)
    print(f"Wrote {gif}\nWrote {pdf}", flush=True)


if __name__ == "__main__":
    main()
