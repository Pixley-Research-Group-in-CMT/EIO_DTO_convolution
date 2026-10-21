"""Plot b_mu^x,y,z/Jzz in three rows, mu=1,2,3 in three columns.

Configuration deltas are dimensionless delta_i*Jzz, not inverse meV.
h is the bare reduced field mu_B|B|/Jzz. With H_Z=-b.S and the solver's
coefficient convention, bx=2 Re(b_plus), by=-2 Im(b_plus), bz=b_z.
Edit CONFIG below, then run: python plot_local_zeeman_components.py
No self-consistency calculation is needed.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent

# ---------------- Edit configuration here ----------------
CONFIG = {
    "gmft_source": "../gMFT_for_QSI",
    "g_z": 10.0,
    "g_perp": 0.0,
    "delta1_jzz": 0.0,       # Dimensionless delta_1 * Jzz, not inverse meV.
    "delta2_jzz": 0.0,       # Dimensionless delta_2 * Jzz.
    "h": 1.0,               # Bare mu_B * |B| / Jzz, before g factors.
    "theta_degrees": 90.0,
    "phi_start": 0.0,
    "phi_stop": 360.0,
    "n_phi": 721,
    "output_pdf": "../local_zeeman_components.pdf",  # Relative to this script.
}
# ---------------------------------------------------------


def evaluate(config):
    for key in ("g_z", "g_perp", "delta1_jzz", "delta2_jzz", "h",
                "theta_degrees", "phi_start", "phi_stop"):
        if not np.isfinite(config[key]):
            raise ValueError(key + " must be finite")
    if config["h"] < 0 or config["phi_stop"] <= config["phi_start"]:
        raise ValueError("Require h>=0 and phi_stop>phi_start")
    count = config["n_phi"]
    if isinstance(count, bool) or int(count) != count or count < 2:
        raise ValueError("n_phi must be an integer >=2")
    source = Path(config["gmft_source"]).expanduser()
    source = (ROOT / source).resolve()
    sys.path.insert(0, str(source))
    from gmft_for_qsi.gmft_field_general import local_zeeman_coefficients
    from gmft_for_qsi.lattice import magnetic_field_direction
    phi = np.linspace(config["phi_start"], config["phi_stop"], count)
    components = np.empty((count, 3, 4))
    for i, angle in enumerate(phi):
        bp, bz = local_zeeman_coefficients(dict(
            h_ext=config["h"] * magnetic_field_direction(
                theta=config["theta_degrees"], phi=angle),
            g_z=config["g_z"], g_perp=config["g_perp"],
            delta_1=config["delta1_jzz"], delta_2=config["delta2_jzz"]))
        components[i] = np.array([2*bp.real, -2*bp.imag, bz])
    return phi, components


def make_figure(config, phi, components):
    figure, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True,
                                sharey="row", constrained_layout=True)
    colors = ("#2166ac", "#d95f02", "#238b45")
    for row, component in enumerate(("x", "y", "z")):
        limit = float(np.max(np.abs(components[:, row, 1:])))
        limit = 1.1*limit if limit > 1e-14 else 1.
        for column, mu in enumerate((1, 2, 3)):
            axis = axes[row, column]
            axis.plot(phi, components[:, row, mu], color=colors[row], lw=1.6)
            axis.axhline(0., color="0.45", lw=.6)
            axis.set_xlim(phi[0], phi[-1])
            axis.set_ylim(-limit, limit)
            axis.grid(alpha=.2)
            if row == 0:
                axis.set_title(rf"Sublattice $\mu={mu}$")
            if column == 0:
                axis.set_ylabel(rf"$b_\mu^{component}/J_{{zz}}$")
            if row == 2:
                axis.set_xlabel(r"$\phi$ (degrees)")
            if phi[0] == 0 and phi[-1] == 360:
                axis.set_xticks(np.arange(0, 361, 60))
    figure.suptitle(
        rf"$h=\mu_B|B|/J_{{zz}}={config['h']:g}$, "
        rf"$\theta={config['theta_degrees']:g}^\circ$, "
        rf"$g_z={config['g_z']:g}$, $g_\perp={config['g_perp']:g}$"
        "\n" + rf"$\delta_1J_{{zz}}={config['delta1_jzz']:g}$, "
        rf"$\delta_2J_{{zz}}={config['delta2_jzz']:g}$")
    return figure


def main():
    config = CONFIG.copy()
    phi, components = evaluate(config)
    figure = make_figure(config, phi, components)
    output = (ROOT / Path(config["output_pdf"]).expanduser()).resolve()
    if output.suffix.lower() != ".pdf":
        raise ValueError("output_pdf must be a vector PDF filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
