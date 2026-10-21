"""Static winning spin components versus phi for one downloaded gMFT run.

Two rows (signed s_z and |s_perp|), four columns (mu=0,1,2,3).
Heatmap axes are phi and bare reduced field; colors are spin values.
Reads saved winners; no interpolation or new MF solves.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_gmft_spin_movie import configured_phi_angles, load_grid, edges


def make_figure(config, fields, jpm, z, perp):
    phi = configured_phi_angles(config)
    tick_stride = max(1, int(np.ceil(len(phi) / 7)))
    phi_ticks = phi[::tick_stride]
    if not np.isclose(phi_ticks[-1], phi[-1]):
        phi_ticks = np.r_[phi_ticks, phi[-1]]
    figure, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True,
                                sharey="row", constrained_layout=True)
    phi_edges, field_edges = edges(phi), edges(fields)
    field_edges[0] = max(0., field_edges[0])
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")
    for row, (label, values) in enumerate(((r"s^z_\mu", z), (r"|s^\perp_\mu|", perp))):
        for mu in range(4):
            axis = axes[row, mu]
            mesh = axis.pcolormesh(phi_edges, field_edges,
                np.ma.masked_invalid(values[:, :, 0, mu].T), shading="flat", cmap=cmap,
                vmin=-.5, vmax=.5, rasterized=False)
            axis.set(xlim=(phi_edges[0], phi_edges[-1]), xticks=phi_ticks,
                     ylim=(field_edges[0], field_edges[-1]), yticks=fields)
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            if row == 0:
                axis.set_title(rf"Sublattice $\mu={mu}$")
            else:
                axis.set_xlabel(r"$\phi$ (degrees)")
            if mu == 0:
                axis.set_ylabel(r"$h=\mu_B|B|/J_{zz}$")
        colorbar = figure.colorbar(mesh, ax=list(axes[row]), shrink=.9,
                                  ticks=[-.5, -.25, 0., .25, .5])
        colorbar.set_label(rf"${label}$")
        colorbar.solids.set_rasterized(False)
    figure.suptitle(
        rf"Winning gMFT states: $J_\pm/J_{{zz}}={float(jpm[0]):g}$, "
        rf"$\theta={config['theta_degrees']:g}^\circ$"
        "\n" + rf"$g_z={config['g_z']:g}$, $g_\perp={config['g_perp']:g}$, "
        rf"$\delta_1={config['delta1_per_mev']:g}$, "
        rf"$\delta_2={config['delta2_per_mev']:g}$ meV$^{{-1}}$; "
        + config["q_perp_mode"] + " transport run"
        + ("\n" + rf"Evaluated $\phi=\phi_{{nom}}+{config['phi_offset_degrees']:g}^\circ$"
           if config.get("phi_offset_degrees", 0.) else "")
        + ("; amplitude convergence (phases unchecked)"
           if config.get("convergence_criterion", "full") == "amplitudes" else ""))
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.expanduser().resolve()
    config, fields, jpm, z, perp = load_grid([run])
    figure = make_figure(config, fields, jpm, z, perp)
    output = run / "gmft_spin_components_vs_phi.pdf"
    figure.savefig(output)
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
