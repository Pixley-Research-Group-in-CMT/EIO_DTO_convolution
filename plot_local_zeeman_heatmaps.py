"""Plot b_z/|b| and b_perp/|b| versus phi and a configured field list.

Edit the configuration below, then run: python plot_local_zeeman_heatmaps.py
Input fields use Jzz units; plotted ratios are dimensionless.
Zero-magnitude local vectors are masked because their direction is undefined.
No MF calculation is required.
With the solver's convention, bx=2 Re(b_plus), by=-2 Im(b_plus), so
b_perp=sqrt(bx**2+by**2)=2*abs(b_plus), not abs(b_plus).
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parent

# ---------------- Edit configuration here ----------------
GMFT_SOURCE = ROOT.parent / "gMFT_for_QSI"
G_Z = 10.0
G_PERP = 0.0
DELTA1_JZZ = 0.2               # Dimensionless delta_1 * Jzz (NOT inverse meV).
DELTA2_JZZ = 0.2               # Dimensionless delta_2 * Jzz.
FIELDS = [0.001, 0.1, 0.5, 1, 1.5, 5, 10, 20, 50]  # Bare mu_B * |B| / Jzz, before g factors.
THETA_DEGREES = 90.0
PHI_START = 0.0
PHI_STOP = 360.0               # Full rotation.
N_PHI = 721                    # Half-degree sampling.
SUBLATTICES = (0, 1, 2, 3)     # Use (1, 2, 3) for just the interfacial sites.
BZ_LIMIT = None                # None: normalized scale [-1,1]; otherwise positive limit.
BPERP_MAX = None               # None: normalized scale [0,1]; otherwise positive limit.
OUTPUT_PDF = "../local_zeeman_z_perp_heatmaps.pdf"
# ---------------------------------------------------------


def cell_edges(values, single_width=.02):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return values[0] + np.array([-.5, .5])*single_width
    middle = (values[:-1]+values[1:])/2
    return np.r_[values[0]-(values[1]-values[0])/2, middle,
                 values[-1]+(values[-1]-values[-2])/2]


def evaluate():
    fields = np.sort(np.asarray(FIELDS, dtype=float))
    if fields.ndim != 1 or not len(fields) or not np.isfinite(fields).all() or np.any(fields < 0):
        raise ValueError("FIELDS must be a nonempty list of finite nonnegative fields")
    if len(np.unique(fields)) != len(fields):
        raise ValueError("FIELDS must not contain duplicates")
    if not np.isfinite([G_Z, G_PERP, DELTA1_JZZ, DELTA2_JZZ,
                       THETA_DEGREES, PHI_START, PHI_STOP]).all():
        raise ValueError("Constants and angles must be finite")
    if PHI_STOP <= PHI_START or isinstance(N_PHI, bool) or int(N_PHI) != N_PHI or N_PHI < 2:
        raise ValueError("Require PHI_STOP>PHI_START and integer N_PHI>=2")
    if not SUBLATTICES or len(set(SUBLATTICES)) != len(SUBLATTICES) or any(mu not in (0, 1, 2, 3) for mu in SUBLATTICES):
        raise ValueError("SUBLATTICES must be distinct indices from 0,1,2,3")
    for limit in (BZ_LIMIT, BPERP_MAX):
        if limit is not None and (not np.isfinite(limit) or limit <= 0):
            raise ValueError("Color limits must be None or finite positive numbers")
    sys.path.insert(0, str(Path(GMFT_SOURCE).expanduser().resolve()))
    from gmft_for_qsi.gmft_field_general import local_zeeman_coefficients
    from gmft_for_qsi.lattice import magnetic_field_direction
    phi = np.linspace(PHI_START, PHI_STOP, N_PHI)
    directions = [magnetic_field_direction(theta=THETA_DEGREES, phi=p) for p in phi]
    bz = np.empty((len(fields), len(phi), 4))
    bperp = np.empty_like(bz)
    for i, field in enumerate(fields):
        for j, direction in enumerate(directions):
            bp, longitudinal = local_zeeman_coefficients(dict(
                h_ext=field*direction, g_z=G_Z, g_perp=G_PERP,
                delta_1=DELTA1_JZZ, delta_2=DELTA2_JZZ))
            bz[i, j] = longitudinal
            bperp[i, j] = 2*np.abs(bp)
    return phi, fields, bz, bperp


def normalized_components(bz, bperp):
    magnitude = np.hypot(bz, bperp)
    nz = np.full_like(magnitude, np.nan, dtype=float)
    nperp = np.full_like(magnitude, np.nan, dtype=float)
    np.divide(bz, magnitude, out=nz, where=magnitude > 0)
    np.divide(bperp, magnitude, out=nperp, where=magnitude > 0)
    return nz, nperp


def make_figure(phi, fields, bz, bperp):
    bz, bperp = normalized_components(bz, bperp)
    figure, axes = plt.subplots(2, len(SUBLATTICES), figsize=(3.5*len(SUBLATTICES), 6),
                                sharex=True, sharey=True, squeeze=False,
                                constrained_layout=True)
    pe, he = cell_edges(phi), cell_edges(fields)
    he[0] = max(0., he[0])
    zlimit = BZ_LIMIT if BZ_LIMIT is not None else 1.
    pmax = BPERP_MAX if BPERP_MAX is not None else 1.
    signed_cmap = plt.get_cmap("coolwarm").copy()
    signed_cmap.set_bad("white")
    gray_to_red = LinearSegmentedColormap.from_list(
        "coolwarm_positive", plt.get_cmap("coolwarm")(np.linspace(.5, 1., 256)))
    gray_to_red.set_bad("white")
    settings = ((bz, signed_cmap, -zlimit, zlimit, r"$b^z_\mu/|\mathbf{b}_\mu|$"),
                (bperp, gray_to_red, 0., pmax, r"$b^\perp_\mu/|\mathbf{b}_\mu|$"))
    for row, (values, cmap, vmin, vmax, label) in enumerate(settings):
        for column, mu in enumerate(SUBLATTICES):
            axis = axes[row, column]
            mesh = axis.pcolormesh(pe, he, np.ma.masked_invalid(values[:, :, mu]), shading="flat",
                                   cmap=cmap, vmin=vmin, vmax=vmax, rasterized=False)
            axis.set(xlim=(pe[0], pe[-1]), ylim=(he[0], he[-1]), yticks=fields)
            axis.set_xticks(np.linspace(phi[0], phi[-1], 7))
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            if row == 0:
                axis.set_title(rf"Sublattice $\mu={mu}$")
            else:
                axis.set_xlabel(r"$\phi$ (degrees)")
            if column == 0:
                axis.set_ylabel(r"$h=\mu_B|B|/J_{zz}$")
        colorbar = figure.colorbar(mesh, ax=list(axes[row]), shrink=.9)
        colorbar.set_label(label)
        colorbar.solids.set_rasterized(False)
    figure.suptitle(
        rf"Normalized local Zeeman fields: $g_z={G_Z:g}$, $g_\perp={G_PERP:g}$, "
        rf"$\delta_1J_{{zz}}={DELTA1_JZZ:g}$, $\delta_2J_{{zz}}={DELTA2_JZZ:g}$"
        "\n" + rf"$\theta={THETA_DEGREES:g}^\circ$; "
        + r"$b^\perp_\mu=\sqrt{(b^x_\mu)^2+(b^y_\mu)^2}$")
    return figure


def main():
    output = Path(OUTPUT_PDF).expanduser().resolve()
    if output.suffix.lower() != ".pdf":
        raise ValueError("OUTPUT_PDF must be a PDF filename")
    phi, fields, bz, bperp = evaluate()
    figure = make_figure(phi, fields, bz, bperp)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
