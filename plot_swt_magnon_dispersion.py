#!/usr/bin/env python3
"""Plot the four Sec. 4.2 magnon bands over the (111) kagome-plane BZ."""
import argparse
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from config import TTO_CUBIC_LATTICE_ANGSTROM
from swt_bz_maps import (
    FIELD_PHI_FILE_LABELS,
    FIELD_PHIS_DEGREES,
    FIELD_RATIOS,
    FIELD_THETA_DEGREES,
    kagome_bz_grid,
    prepare_ordered_state,
)
from tto_spin_wave import magnon_modes


N_MAGNON_BANDS = 4


def evaluate_magnon_bands(
    q_points,
    ordered_state,
    h_z,
    j_pm_ratio,
    *,
    j_pmpm_ratio,
    j_zz_ev,
    spin_length=0.5,
    regularization=1e-8,
):
    """Return the four energy-ordered bands in meV at each momentum."""
    q_points = np.asarray(q_points, dtype=float)
    if q_points.ndim != 2 or q_points.shape[1] != 3:
        raise ValueError("q_points must have shape (n_q, 3)")
    if not np.isfinite(j_zz_ev) or j_zz_ev <= 0.0:
        raise ValueError("j_zz_ev must be finite and positive")

    band_energy_mev = np.empty((N_MAGNON_BANDS, len(q_points)), dtype=float)
    max_imaginary = 0.0
    for momentum_index, momentum in enumerate(q_points):
        frequencies, _, imaginary = magnon_modes(
            momentum,
            ordered_state,
            h_z,
            j_pm_ratio,
            j_pmpm_ratio=j_pmpm_ratio,
            spin_length=spin_length,
            regularization=regularization,
        )
        band_energy_mev[:, momentum_index] = 1e3 * j_zz_ev * frequencies
        max_imaginary = max(max_imaginary, imaginary)
    return band_energy_mev, max_imaginary


def shared_energy_norm(values):
    """Return a zero-based energy scale shared by every displayed band."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or np.any(finite < -1e-12):
        raise ValueError("Magnon energies must be finite and nonnegative")
    maximum = float(np.max(finite))
    if maximum <= 0.0:
        maximum = 1.0
    return mpl.colors.Normalize(vmin=0.0, vmax=maximum)


def plot_band_grid(
    values,
    q_edges,
    bz_vertices,
    norm,
    args,
    field_phi_degrees,
    output_path,
):
    """Write one three-field by four-band vector PDF."""
    figure, axes = plt.subplots(
        len(FIELD_RATIOS),
        N_MAGNON_BANDS,
        figsize=(11.7, 8.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    boundary = np.vstack((bz_vertices, bz_vertices[0]))
    ticks = np.asarray((q_edges[0], 0.0, q_edges[-1]))
    image = None
    for field_index, field_ratio in enumerate(FIELD_RATIOS):
        for band_index in range(N_MAGNON_BANDS):
            axis = axes[field_index, band_index]
            image = axis.pcolormesh(
                q_edges,
                q_edges,
                values[field_index, band_index],
                cmap="magma",
                norm=norm,
                shading="flat",
            )
            axis.plot(boundary[:, 0], boundary[:, 1], color="0.35", lw=0.9)
            axis.set_aspect("equal")
            axis.set_xlim(q_edges[0], q_edges[-1])
            axis.set_ylim(q_edges[0], q_edges[-1])
            axis.set_xticks(ticks)
            axis.set_yticks(ticks)
            if field_index == 0:
                axis.set_title(rf"band $n={band_index + 1}$")
            if field_index == len(FIELD_RATIOS) - 1:
                axis.set_xlabel(r"$q_{[\bar{1}01]}$ ($\mathrm{\AA}^{-1}$)")
            if band_index == 0:
                axis.set_ylabel(
                    rf"$h/J_{{zz}}={field_ratio:g}$" + "\n"
                    + r"$q_{[1\bar{2}1]}$ ($\mathrm{\AA}^{-1}$)"
                )

    parameters = (
        rf"$(111)$ kagome-plane BZ, $\theta={FIELD_THETA_DEGREES:g}^\circ$, "
        rf"$\phi={field_phi_degrees:g}^\circ$, "
        rf"$J_{{zz}}={1e3 * args.j_zz_ev:g}$ meV, "
        rf"$J_{{\pm}}/J_{{zz}}={args.j_pm:g}$, "
        rf"$J_{{\pm\pm}}/J_{{zz}}={args.j_pmpm:g}$"
    )
    figure.suptitle(
        r"Magnon dispersion $\omega_n(\mathbf{q})$" + "\n" + parameters,
        fontsize=10,
    )
    colorbar = figure.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.84, pad=0.025
    )
    colorbar.set_label(r"magnon energy $\omega_n(\mathbf{q})$ (meV)")
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    figure.savefig(
        output_path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def write_hdf5(
    output_path,
    q_axis,
    q_grid,
    inside_bz,
    band_energy_mev,
    state_data,
    max_imaginary_frequency_ratio,
    args,
):
    """Save the four dispersion maps and ordered-state diagnostics."""
    with h5py.File(output_path, "w") as output:
        output.create_dataset("q_x_inv_angstrom", data=q_axis)
        output.create_dataset("q_y_inv_angstrom", data=q_axis)
        output.create_dataset("q_reduced", data=q_grid, compression="gzip")
        output.create_dataset("inside_2d_bz", data=inside_bz, compression="gzip")
        output.create_dataset("field_phi_degrees", data=FIELD_PHIS_DEGREES)
        output.create_dataset("h_over_J_zz", data=FIELD_RATIOS)
        output.create_dataset(
            "band_indices", data=np.arange(1, N_MAGNON_BANDS + 1)
        )
        output.create_dataset(
            "band_energy_mev", data=band_energy_mev, compression="gzip"
        )
        output.create_dataset(
            "max_imaginary_frequency_ratio",
            data=max_imaginary_frequency_ratio,
        )
        for name, values in state_data.items():
            output.create_dataset(name, data=values, compression="gzip")
        output.attrs.update(
            theory="linear spin wave, Sec. 4.2, Eqs. (88)-(95)",
            observable="four positive magnon energies",
            first_bz="hexagonal 2D BZ perpendicular to cubic [111]",
            momentum_axes="x || [-1,0,1], y || [1,-2,1], in inverse Angstrom",
            array_axes="field_phi, h_over_J_zz, band, q_y, q_x",
            band_ordering=(
                "ascending energy independently at each q; crossings are not "
                "tracked by eigenvector continuity"
            ),
            field_theta_degrees=FIELD_THETA_DEGREES,
            J_zz_ev=args.j_zz_ev,
            J_pm_over_J_zz=args.j_pm,
            J_pmpm_over_J_zz=args.j_pmpm,
            tto_cubic_lattice_angstrom=args.lattice_constant,
            spin_length=args.spin_length,
            regularization=args.regularization,
            stability_tilt=args.stability_tilt,
            random_starts=args.random_starts,
            n_q=args.n_q,
        )


def compute_and_plot(args):
    """Compute the fixed six-condition scan and write data plus two PDFs."""
    if not np.isfinite(args.j_zz_ev) or args.j_zz_ev <= 0.0:
        raise ValueError("j_zz_ev must be finite and positive")

    run_directory = args.output_root.expanduser().resolve()
    data_directory = run_directory / "data"
    figure_directory = run_directory / "figures"
    data_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    data_path = data_directory / "swt_magnon_dispersion.hdf5"
    figure_paths = tuple(
        figure_directory / f"swt_magnon_dispersion_phi_{label}.pdf"
        for label in FIELD_PHI_FILE_LABELS
    )
    existing = [path for path in (data_path,) + figure_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output already exists; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    q_axis, q_edges, q_grid, inside_bz, bz_vertices = kagome_bz_grid(
        args.n_q, args.lattice_constant
    )
    q_points = q_grid[inside_bz]
    scan_shape = (len(FIELD_PHIS_DEGREES), len(FIELD_RATIOS))
    band_energy_mev = np.full(
        scan_shape + (N_MAGNON_BANDS, args.n_q, args.n_q), np.nan
    )
    max_imaginary_frequency_ratio = np.empty(scan_shape)
    state_data = {
        "h_z_over_J_zz": np.empty(scan_shape + (4,)),
        "initial_spins_local": np.empty(scan_shape + (4, 3)),
        "ordered_spins_local": np.empty(scan_shape + (4, 3)),
        "initial_classical_energy": np.empty(scan_shape),
        "relaxed_classical_energy": np.empty(scan_shape),
        "min_classical_curvature": np.empty(scan_shape),
        "max_torque": np.empty(scan_shape),
        "stationarity_refinement_used": np.empty(scan_shape, dtype=bool),
    }

    for phi_index, field_phi_degrees in enumerate(FIELD_PHIS_DEGREES):
        for field_index, field_ratio in enumerate(FIELD_RATIOS):
            condition = (phi_index, field_index)
            print(
                "Preparing magnon bands for "
                f"phi={field_phi_degrees:g} deg, h/Jzz={field_ratio:g}",
                flush=True,
            )
            h_z, initial_spins, relaxed = prepare_ordered_state(
                args, field_ratio, field_phi_degrees
            )
            energies, max_imaginary = evaluate_magnon_bands(
                q_points,
                relaxed["unit_vectors_local"],
                h_z,
                args.j_pm,
                j_pmpm_ratio=args.j_pmpm,
                j_zz_ev=args.j_zz_ev,
                spin_length=args.spin_length,
                regularization=args.regularization,
            )
            band_energy_mev[condition][:, inside_bz] = energies
            max_imaginary_frequency_ratio[condition] = max_imaginary
            state_data["h_z_over_J_zz"][condition] = h_z
            state_data["initial_spins_local"][condition] = initial_spins
            state_data["ordered_spins_local"][condition] = relaxed["spins_local"]
            state_data["initial_classical_energy"][condition] = (
                relaxed["initial_energy"]
            )
            state_data["relaxed_classical_energy"][condition] = relaxed["energy"]
            state_data["min_classical_curvature"][condition] = (
                relaxed["min_classical_curvature"]
            )
            state_data["max_torque"][condition] = relaxed["max_torque"]
            state_data["stationarity_refinement_used"][condition] = relaxed.get(
                "stationarity_refinement_used", False
            )

    write_hdf5(
        data_path,
        q_axis,
        q_grid,
        inside_bz,
        band_energy_mev,
        state_data,
        max_imaginary_frequency_ratio,
        args,
    )
    norm = shared_energy_norm(band_energy_mev)
    for phi_index, (field_phi_degrees, figure_path) in enumerate(zip(
        FIELD_PHIS_DEGREES, figure_paths
    )):
        plot_band_grid(
            band_energy_mev[phi_index],
            q_edges,
            bz_vertices,
            norm,
            args,
            field_phi_degrees,
            figure_path,
        )
    print(f"Data: {data_path}")
    for figure_path in figure_paths:
        print(f"Figure: {figure_path}")
    return data_path, figure_paths


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot all four SWT magnon bands over the (111) 2D BZ."
    )
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    parser.add_argument("--j-pm", type=float, default=0.205)
    parser.add_argument("--j-pmpm", type=float, default=0.0)
    parser.add_argument("--j-zz-ev", type=float, default=0.00019)
    parser.add_argument("--n-q", type=int, default=101)
    parser.add_argument(
        "--lattice-constant", type=float, default=TTO_CUBIC_LATTICE_ANGSTROM
    )
    parser.add_argument("--spin-length", type=float, default=0.5)
    parser.add_argument("--regularization", type=float, default=1e-8)
    parser.add_argument("--stability-tilt", type=float, default=0.01)
    parser.add_argument("--random-starts", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    compute_and_plot(build_parser().parse_args())
