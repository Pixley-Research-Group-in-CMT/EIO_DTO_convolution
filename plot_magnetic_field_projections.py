"""Bare unit-field projections in the solver's local x/y frames.

Uses theta=90 degrees and h(phi)=cos(phi)c1+sin(phi)c2. These are
geometrical projections, before any g factors or nonlinear field couplings.
All final figures are vector PDFs.
"""
import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def projections(gmft_source, phi_stop=60.):
    path = Path(gmft_source) / "gmft_for_qsi/lattice.py"
    spec = importlib.util.spec_from_file_location("gmft_lattice", path)
    lattice = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lattice)
    phi = np.linspace(0., phi_stop, int(np.ceil(phi_stop * 10)) + 1)
    field = np.array([lattice.magnetic_field_direction(theta=90., phi=p)
                      for p in phi])
    hx = field @ lattice.a_hat.T
    hy = field @ lattice.b_hat.T
    hz = field @ lattice.e_hat.T
    np.testing.assert_allclose(np.sum(field**2, axis=1), 1., atol=1e-14)
    np.testing.assert_allclose(hx**2 + hy**2 + hz**2, 1., atol=1e-14)
    for i in (0, len(phi)//2, len(phi)-1):
        _, minus, _ = lattice.get_local_field_components(field[i])
        np.testing.assert_allclose(hx[i]-1j*hy[i], minus, atol=1e-14)
    return phi, hx, hy


def panel(axis, phi, hx, hy, mu):
    axis.plot(phi, hx[:, mu], color="#2166ac", lw=1.8, label=rf"$h^x_{{{mu}}}$")
    axis.plot(phi, hy[:, mu], color="#d95f02", lw=1.8, label=rf"$h^y_{{{mu}}}$")
    axis.axhline(0., color="0.45", lw=.7)
    ticks = np.arange(0, 361, 60) if phi[-1] == 360 else np.linspace(0, phi[-1], 5)
    axis.set(xlim=(0, phi[-1]), ylim=(-1.05, 1.05), xticks=ticks,
             xlabel=r"$\phi$ (degrees)", ylabel="Local field projection",
             title=rf"Sublattice $\mu={mu}$")
    axis.grid(alpha=.2)
    axis.legend(frameon=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmft-source", type=Path, default=ROOT.parent / "gMFT_for_QSI")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "figures")
    parser.add_argument("--phi-stop", type=float, default=60.)
    args = parser.parse_args()
    if not np.isfinite(args.phi_stop) or not 0 < args.phi_stop <= 360:
        parser.error("phi-stop must be in (0, 360]")
    output = args.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    phi, hx, hy = projections(args.gmft_source.expanduser(), args.phi_stop)
    suffix = "" if args.phi_stop == 60 else f"_phi_0_{args.phi_stop:g}"
    title = r"Unit in-plane magnetic field: $h=1$, $\theta=90^\circ$"
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), constrained_layout=True)
    for mu, axis in enumerate(axes, 1):
        panel(axis, phi, hx, hy, mu)
    figure.suptitle(title)
    path = output / f"unit_field_local_xy_projections{suffix}.pdf"
    figure.savefig(path)
    plt.close(figure)
    print(path)
    for mu in (1, 2, 3):
        figure, axis = plt.subplots(figsize=(5, 3.8), constrained_layout=True)
        panel(axis, phi, hx, hy, mu)
        figure.suptitle(title, fontsize=11)
        path = output / f"unit_field_local_xy_mu_{mu}{suffix}.pdf"
        figure.savefig(path)
        plt.close(figure)
        print(path)


if __name__ == "__main__":
    main()
