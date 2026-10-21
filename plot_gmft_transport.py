"""Regenerate a resistivity overlay or field movie from a gMFT transport run."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

from gmft_transport_job import configured_output_directories, resistivity_filename


def parameter_title(config):
    first = (rf"$J_\pm/J_{{zz}}={config['J_pm_ratio']:g}$, "
             rf"$J_{{\pm\pm}}/J_{{zz}}={config['J_pmpm_ratio']:g}$, "
             rf"$J_{{zz}}={1000*config['j_zz_ev']:g}$ meV, "
             rf"$\eta={1000*config['eta_ev']:g}$ meV")
    second = (rf"$g_z={config['g_z']:g}$, $g_\perp={config['g_perp']:g}$, "
              rf"$\delta_1={config['delta1_per_mev']:g}$ meV$^{{-1}}$, "
              rf"$\delta_2={config['delta2_per_mev']:g}$ meV$^{{-1}}$")
    projection = (r"$q_\perp=0$ slice" if config['q_perp_mode'] == 'slice'
                  else rf"$q_\perp$ integrated ($N_\perp={config['n_q_perp']}$)")
    third = (rf"$\theta={config['theta_degrees']:g}^\circ$, "
             rf"$n_k={config['nk']}$; " + projection)
    phi_start = float(config.get("phi_start_degrees", 0.0))
    phi_stop = float(config.get("phi_stop_degrees", 30.0))
    n_phi = int(config.get("n_phi", 30))
    angular_scan = (
        rf"Nominal $\phi={phi_start:g}^\circ\!\to{phi_stop:g}^\circ$: "
        rf"{n_phi} intervals, {n_phi + 1} angles"
    )
    title = "\n".join((first, second, third, angular_scan))
    if config.get("phi_offset_degrees", 0.):
        title += "\n" + rf"Positive branch evaluated at $\phi=\phi_{{nom}}+{config['phi_offset_degrees']:g}^\circ$"
    if config.get("convergence_criterion", "full") == "amplitudes":
        title += "; amplitude convergence (phases unchecked)"
    return title


def padded_limits(values):
    """Return stable limits shared by every movie frame."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 0.0, 1.0
    lower, upper = finite.min(), finite.max()
    span = upper - lower
    padding = 0.05 * (span if span else max(abs(lower), 1.0))
    return lower - padding, upper + padding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="override the plot destination saved in the run configuration",
    )
    args = parser.parse_args()
    run = args.run_directory.expanduser().resolve()
    table = pd.read_csv(run / "data/tto_transport_summary.csv")
    config = json.loads((run / "manifest.json").read_text())["config"]
    output_directory = (
        args.output_directory
        if args.output_directory is not None
        else configured_output_directories(config)[1]
    ).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    mode = config.get("resistivity_plot_mode", "overlay")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)
    figure.suptitle(parameter_title(config), fontsize=10)
    axes[0].set_ylabel(r"$\rho$ (interface units)")
    axes[1].set_ylabel(r"$[\rho(\phi)-\rho(0)]/\rho(0)$")
    for axis in axes:
        axis.set_xlabel(r"Nominal $\phi$ (degrees)" if config.get("phi_offset_degrees", 0.) else r"$\phi$ (degrees)")
        axis.set_xlim(table.phi_degrees.min(), table.phi_degrees.max())
        axis.grid(alpha=.2)
    output = output_directory / resistivity_filename(config)
    groups = [
        (float(field), group.sort_values("phi_degrees"))
        for field, group in table.groupby("h_over_J_zz", sort=True)
    ]
    if mode == "overlay":
        for field, group in groups:
            axes[0].plot(group.phi_degrees, group.rho, marker="o", ms=2.4,
                         lw=1.0, label=f"{field:g}")
            axes[1].plot(group.phi_degrees, group.delta_rho_over_rho_phi0,
                         marker="o", ms=2.4, lw=1.0)
        axes[0].legend(title=r"$\mu_B |B|/J_{zz}$")
        figure.savefig(output, bbox_inches="tight")
    else:
        axes[0].set_ylim(*padded_limits(table.rho))
        axes[1].set_ylim(*padded_limits(table.delta_rho_over_rho_phi0))
        absolute, = axes[0].plot([], [], marker="o", ms=2.4, lw=1.0)
        normalized, = axes[1].plot([], [], marker="o", ms=2.4, lw=1.0)
        frame_title = axes[0].set_title("")

        def update(frame):
            field, group = groups[frame]
            absolute.set_data(group.phi_degrees, group.rho)
            normalized.set_data(group.phi_degrees, group.delta_rho_over_rho_phi0)
            frame_title.set_text(
                rf"Frame {frame + 1}/{len(groups)}: $\mu_B |B|/J_{{zz}}={field:g}$"
            )
            return absolute, normalized, frame_title

        fps = float(config.get("resistivity_movie_fps", 2.0))
        movie = FuncAnimation(
            figure, update, frames=len(groups), interval=1000.0 / fps, repeat=True
        )
        movie.save(output, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
