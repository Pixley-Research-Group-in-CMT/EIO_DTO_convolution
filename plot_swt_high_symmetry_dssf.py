#!/usr/bin/env python3
"""Plot the Sec. 4.2 SWT DSSF along the standard FCC symmetry path."""
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from config import TTO_CUBIC_LATTICE_ANGSTROM
from swt_bz_maps import (
    COMPONENTS,
    FIELD_THETA_DEGREES,
    SUBLATTICES,
    prepare_ordered_state,
    shared_norm,
)
from tto_spin_wave import (
    magnon_dispersion_bounds,
    spin_wave_structure_factor_spectrum,
)
from tto_structure_factor import FCC_PRIMITIVES


HIGH_SYMMETRY_LABELS = (r"$\Gamma$", "X", "W", "K", r"$\Gamma$", "L", "U")
HIGH_SYMMETRY_NAMES = ("Gamma", "X", "W", "K", "Gamma", "L", "U")
# FCC reciprocal coordinates in the primitive basis used by tto_spin_wave.py.
HIGH_SYMMETRY_Q_REDUCED = np.asarray([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.75, 0.25, 0.5],
    [0.75, 0.375, 0.375],
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [0.625, 0.25, 0.625],
])


def reduced_to_cartesian(q_reduced, lattice_constant_angstrom):
    """Convert FCC primitive reduced coordinates to inverse Angstrom."""
    q_reduced = np.asarray(q_reduced, dtype=float)
    lattice_constant_angstrom = float(lattice_constant_angstrom)
    if q_reduced.shape[-1] != 3 or not np.all(np.isfinite(q_reduced)):
        raise ValueError("q_reduced must end with finite three-component momenta")
    if not np.isfinite(lattice_constant_angstrom) or lattice_constant_angstrom <= 0.0:
        raise ValueError("lattice_constant_angstrom must be finite and positive")
    direct = 0.5 * lattice_constant_angstrom * FCC_PRIMITIVES
    reciprocal = 2.0 * np.pi * np.linalg.inv(direct)
    return np.einsum("ij,...j->...i", reciprocal, q_reduced)


def fcc_high_symmetry_path(points_per_segment, lattice_constant_angstrom):
    """Sample Gamma-X-W-K-Gamma-L-U with uniform density per segment."""
    if int(points_per_segment) != points_per_segment or int(points_per_segment) < 1:
        raise ValueError("points_per_segment must be a positive integer")
    points_per_segment = int(points_per_segment)
    vertex_cartesian = reduced_to_cartesian(
        HIGH_SYMMETRY_Q_REDUCED, lattice_constant_angstrom
    )
    segment_lengths = np.linalg.norm(np.diff(vertex_cartesian, axis=0), axis=1)
    tick_positions = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    q_pieces = []
    distance_pieces = []
    interpolation = np.arange(points_per_segment, dtype=float) / points_per_segment
    for segment_index in range(len(segment_lengths)):
        q_start = HIGH_SYMMETRY_Q_REDUCED[segment_index]
        q_stop = HIGH_SYMMETRY_Q_REDUCED[segment_index + 1]
        q_pieces.append(
            q_start[None, :] + interpolation[:, None] * (q_stop - q_start)[None, :]
        )
        distance_pieces.append(
            tick_positions[segment_index] + interpolation * segment_lengths[segment_index]
        )
    q_reduced = np.concatenate(q_pieces + [HIGH_SYMMETRY_Q_REDUCED[-1:]], axis=0)
    path_distance = np.concatenate(distance_pieces + [tick_positions[-1:]])
    q_cartesian = reduced_to_cartesian(q_reduced, lattice_constant_angstrom)
    tick_indices = np.arange(len(HIGH_SYMMETRY_LABELS)) * points_per_segment
    return q_reduced, q_cartesian, path_distance, tick_positions, tick_indices


def energy_axis_from_bounds(maximum_mev, n_omega, component):
    """Return the energy axis, including two-magnon bandwidth when needed."""
    maximum_mev = float(maximum_mev)
    if not np.isfinite(maximum_mev) or maximum_mev <= 0.0:
        raise ValueError("maximum_mev must be finite and positive")
    if int(n_omega) != n_omega or int(n_omega) < 2:
        raise ValueError("n_omega must be an integer of at least 2")
    if component not in COMPONENTS:
        raise ValueError(f"Unknown component: {component}")
    multiplier = 2.0 if component in ("two_magnon", "total") else 1.0
    return np.linspace(0.0, multiplier * maximum_mev, int(n_omega))


def coordinate_edges(centers):
    """Construct cell edges from a strictly increasing one-dimensional axis."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or len(centers) < 2 or not np.all(np.diff(centers) > 0.0):
        raise ValueError("centers must be a strictly increasing axis of length >= 2")
    midpoints = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate((
        [centers[0] - 0.5 * (centers[1] - centers[0])],
        midpoints,
        [centers[-1] + 0.5 * (centers[-1] - centers[-2])],
    ))


def phi_file_label(phi_degrees):
    """Return a filesystem-safe label that retains the evaluated angle."""
    if not np.isfinite(phi_degrees):
        raise ValueError("phi_degrees must be finite")
    return f"{float(phi_degrees):g}".replace("-", "m").replace(".", "p")


def _decorate_path_axis(axis, tick_positions, show_tick_labels):
    for position in tick_positions[1:-1]:
        axis.axvline(position, color="white", lw=0.45, alpha=0.55)
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(HIGH_SYMMETRY_LABELS if show_tick_labels else [])
    axis.set_xlim(tick_positions[0], tick_positions[-1])


def plot_sublattice_spectra(
    intensity, path_distance, tick_positions, omega_mev, fields,
    field_phi_degrees, args, output_path,
):
    """Write a three-field by three-sublattice DSSF intensity plot."""
    norm = shared_norm(intensity, args.vmax_percentile)
    x_edges = coordinate_edges(path_distance)
    y_edges = coordinate_edges(omega_mev)
    figure, axes = plt.subplots(
        len(fields), len(SUBLATTICES), figsize=(10.5, 8.2),
        sharex=True, sharey=True, constrained_layout=True,
    )
    image = None
    for field_index, field_ratio in enumerate(fields):
        for mu_index, mu in enumerate(SUBLATTICES):
            axis = axes[field_index, mu_index]
            image = axis.pcolormesh(
                x_edges, y_edges, intensity[field_index, mu_index],
                cmap="magma", norm=norm, shading="flat",
                edgecolors="none", linewidth=0.0, antialiased=False,
            )
            _decorate_path_axis(axis, tick_positions, field_index == len(fields) - 1)
            axis.set_ylim(omega_mev[0], omega_mev[-1])
            if field_index == 0:
                axis.set_title(rf"$\mu={mu}$")
            if mu_index == 0:
                axis.set_ylabel(
                    rf"$h/J_{{zz}}={field_ratio:g}$" + "\n" + r"$\omega$ (meV)"
                )
            if field_index == len(fields) - 1:
                axis.set_xlabel("wave-vector path")

    parameters = (
        rf"$\theta={FIELD_THETA_DEGREES:g}^\circ$, "
        rf"$\phi={field_phi_degrees:g}^\circ$, "
        rf"$J_{{zz}}={1e3 * args.j_zz_ev:g}$ meV, "
        rf"$J_{{\pm}}/J_{{zz}}={args.j_pm:g}$, "
        rf"$J_{{\pm\pm}}/J_{{zz}}={args.j_pmpm:g}$, "
        rf"component={args.component.replace('_', ' ')}"
    )
    figure.suptitle(r"$S_\mu(\mathbf{q},\omega)$" + "\n" + parameters, fontsize=10)
    colorbar = figure.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.86, pad=0.025
    )
    colorbar.set_label(r"$S_\mu(\mathbf{q},\omega)$ (eV$^{-1}$)")
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    if args.vmax_percentile < 100.0:
        colorbar.ax.set_title(
            f"clipped at {args.vmax_percentile:g}%", fontsize=7, pad=6
        )
    figure.savefig(
        output_path, bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def plot_sublattice_sum(
    intensity, path_distance, tick_positions, omega_mev, fields,
    field_phi_degrees, args, output_path,
):
    """Write the sum over mu=1,2,3 with one field per row."""
    summed = np.sum(intensity, axis=1)
    norm = shared_norm(summed, args.vmax_percentile)
    x_edges = coordinate_edges(path_distance)
    y_edges = coordinate_edges(omega_mev)
    figure, axes = plt.subplots(
        len(fields), 1, figsize=(7.2, 8.2),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    image = None
    for field_index, field_ratio in enumerate(fields):
        axis = axes[field_index]
        image = axis.pcolormesh(
            x_edges, y_edges, summed[field_index],
            cmap="magma", norm=norm, shading="flat",
            edgecolors="none", linewidth=0.0, antialiased=False,
        )
        _decorate_path_axis(axis, tick_positions, field_index == len(fields) - 1)
        axis.set_ylim(omega_mev[0], omega_mev[-1])
        axis.set_ylabel(
            rf"$h/J_{{zz}}={field_ratio:g}$" + "\n" + r"$\omega$ (meV)"
        )
    axes[-1].set_xlabel("wave-vector path")

    parameters = (
        rf"$\theta={FIELD_THETA_DEGREES:g}^\circ$, "
        rf"$\phi={field_phi_degrees:g}^\circ$, "
        rf"$J_{{zz}}={1e3 * args.j_zz_ev:g}$ meV, "
        rf"$J_{{\pm}}/J_{{zz}}={args.j_pm:g}$, "
        rf"$J_{{\pm\pm}}/J_{{zz}}={args.j_pmpm:g}$, "
        rf"component={args.component.replace('_', ' ')}"
    )
    figure.suptitle(
        r"$\sum_{\mu=1}^{3}S_\mu(\mathbf{q},\omega)$" + "\n" + parameters,
        fontsize=10,
    )
    colorbar = figure.colorbar(image, ax=axes.tolist(), shrink=0.86, pad=0.025)
    colorbar.set_label(
        r"$\sum_{\mu=1}^{3}S_\mu(\mathbf{q},\omega)$ (eV$^{-1}$)"
    )
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    if args.vmax_percentile < 100.0:
        colorbar.ax.set_title(
            f"clipped at {args.vmax_percentile:g}%", fontsize=7, pad=6
        )
    figure.savefig(
        output_path, bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def write_hdf5(
    output_path, q_reduced, q_cartesian, path_distance, tick_positions,
    tick_indices, omega_mev, fields, field_phis, intensity, dispersion_bounds_mev,
    state_data, max_imaginary_frequency_ratio, args,
):
    """Save the path spectra, sums, ordered states, and run metadata."""
    with h5py.File(output_path, "w") as output:
        output.create_dataset("q_reduced", data=q_reduced)
        output.create_dataset("q_cartesian_inv_angstrom", data=q_cartesian)
        output.create_dataset("path_distance_inv_angstrom", data=path_distance)
        output.create_dataset("high_symmetry_tick_positions", data=tick_positions)
        output.create_dataset("high_symmetry_tick_indices", data=tick_indices)
        output.create_dataset(
            "high_symmetry_labels",
            data=np.asarray(HIGH_SYMMETRY_NAMES, dtype=h5py.string_dtype("utf-8")),
        )
        output.create_dataset("omega_mev", data=omega_mev)
        output.create_dataset("field_phi_degrees", data=field_phis)
        output.create_dataset("h_over_J_zz", data=fields)
        output.create_dataset("sublattice_indices", data=SUBLATTICES)
        output.create_dataset("dispersion_minimum_mev", data=dispersion_bounds_mev[..., 0])
        output.create_dataset("dispersion_maximum_mev", data=dispersion_bounds_mev[..., 1])
        output.create_dataset(
            "max_imaginary_frequency_ratio", data=max_imaginary_frequency_ratio
        )
        output.create_dataset(
            "intensity", data=intensity, compression="gzip", shuffle=True
        )
        output.create_dataset(
            "intensity_mu_sum", data=np.sum(intensity, axis=2),
            compression="gzip", shuffle=True,
        )
        for name, values in state_data.items():
            output.create_dataset(name, data=values, compression="gzip")
        output.attrs.update(
            theory="linear spin wave, Sec. 4.2, Eqs. (88)-(100)",
            observable="S_mu(q, omega) along Gamma-X-W-K-Gamma-L-U",
            component=args.component,
            momentum_path="Gamma-X-W-K-Gamma-L-U in the bulk FCC BZ",
            array_axes=(
                "field_phi_degrees, h_over_J_zz, sublattice, omega, q_path"
            ),
            mu_sum_axes="field_phi_degrees, h_over_J_zz, omega, q_path",
            mu_sum_definition="sum over sublattice indices 1, 2, 3",
            field_theta_degrees=FIELD_THETA_DEGREES,
            J_zz_ev=args.j_zz_ev,
            J_pm_over_J_zz=args.j_pm,
            J_pmpm_over_J_zz=args.j_pmpm,
            lorentzian_eta_mev=args.eta_mev,
            tto_cubic_lattice_angstrom=args.lattice_constant,
            dssf_nk=args.dssf_nk,
            points_per_path_segment=args.points_per_segment,
            n_omega=args.n_omega,
            spin_length=args.spin_length,
            regularization=args.regularization,
            stability_tilt=args.stability_tilt,
            random_starts=args.random_starts,
            vmax_percentile=args.vmax_percentile,
        )


def compute_and_plot(args):
    """Compute both angle scans and write HDF5 plus four PDFs."""
    fields = np.asarray(args.fields, dtype=float)
    field_phis = np.asarray(args.field_phis, dtype=float)
    if fields.shape != (3,) or not np.all(np.isfinite(fields)) or np.any(fields < 0.0):
        raise ValueError("fields must contain exactly three finite nonnegative values")
    if field_phis.shape != (2,) or not np.all(np.isfinite(field_phis)):
        raise ValueError("field_phis must contain exactly two finite values")
    if not np.isfinite(args.j_zz_ev) or args.j_zz_ev <= 0.0:
        raise ValueError("j_zz_ev must be finite and positive")
    if not np.isfinite(args.eta_mev) or args.eta_mev <= 0.0:
        raise ValueError("eta_mev must be finite and positive")
    if args.dssf_nk <= 0 or args.omega_chunk_size <= 0:
        raise ValueError("dssf_nk and omega_chunk_size must be positive")
    if not 0.0 < args.vmax_percentile <= 100.0:
        raise ValueError("vmax_percentile must be in (0, 100]")

    run_directory = args.output_root.expanduser().resolve()
    data_directory = run_directory / "data"
    figure_directory = run_directory / "figures"
    data_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    data_path = data_directory / "swt_dssf_high_symmetry.hdf5"
    sublattice_figures = tuple(
        figure_directory
        / f"swt_dssf_high_symmetry_sublattices_phi_{phi_file_label(phi)}.pdf"
        for phi in field_phis
    )
    sum_figures = tuple(
        figure_directory
        / f"swt_dssf_high_symmetry_mu_sum_phi_{phi_file_label(phi)}.pdf"
        for phi in field_phis
    )
    outputs = (data_path,) + sublattice_figures + sum_figures
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output already exists; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    (
        q_reduced, q_cartesian, path_distance, tick_positions, tick_indices,
    ) = fcc_high_symmetry_path(args.points_per_segment, args.lattice_constant)

    scan_shape = (len(field_phis), len(fields))
    states = np.empty(scan_shape, dtype=object)
    dispersion_bounds_mev = np.empty(scan_shape + (2,), dtype=float)
    max_imaginary_frequency_ratio = np.zeros(scan_shape, dtype=float)
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
    for phi_index, field_phi_degrees in enumerate(field_phis):
        for field_index, field_ratio in enumerate(fields):
            condition = (phi_index, field_index)
            print(
                f"Preparing dispersion for phi={field_phi_degrees:g} deg, "
                f"h/Jzz={field_ratio:g}", flush=True,
            )
            h_z, initial_spins, relaxed = prepare_ordered_state(
                args, field_ratio, field_phi_degrees
            )
            bounds = magnon_dispersion_bounds(
                q_reduced, relaxed["unit_vectors_local"], h_z, args.j_pm,
                j_pmpm_ratio=args.j_pmpm, j_zz_ev=args.j_zz_ev,
                spin_length=args.spin_length, regularization=args.regularization,
            )
            states[condition] = (h_z, relaxed)
            dispersion_bounds_mev[condition] = 1e3 * np.asarray([
                bounds["minimum_ev"], bounds["maximum_ev"]
            ])
            max_imaginary_frequency_ratio[condition] = (
                bounds["max_imaginary_frequency_ratio"]
            )
            state_data["h_z_over_J_zz"][condition] = h_z
            state_data["initial_spins_local"][condition] = initial_spins
            state_data["ordered_spins_local"][condition] = relaxed["spins_local"]
            state_data["initial_classical_energy"][condition] = relaxed["initial_energy"]
            state_data["relaxed_classical_energy"][condition] = relaxed["energy"]
            state_data["min_classical_curvature"][condition] = (
                relaxed["min_classical_curvature"]
            )
            state_data["max_torque"][condition] = relaxed["max_torque"]
            state_data["stationarity_refinement_used"][condition] = relaxed.get(
                "stationarity_refinement_used", False
            )

    omega_mev = energy_axis_from_bounds(
        np.max(dispersion_bounds_mev[..., 1]), args.n_omega, args.component
    )
    print(
        f"DSSF energy range: {omega_mev[0]:.6g} to {omega_mev[-1]:.6g} meV",
        flush=True,
    )
    intensity = np.empty(
        scan_shape + (len(SUBLATTICES), len(omega_mev), len(q_reduced)),
        dtype=np.float32,
    )
    for phi_index, field_phi_degrees in enumerate(field_phis):
        for field_index, field_ratio in enumerate(fields):
            condition = (phi_index, field_index)
            h_z, relaxed = states[condition]
            print(
                f"Evaluating DSSF for phi={field_phi_degrees:g} deg, "
                f"h/Jzz={field_ratio:g}", flush=True,
            )
            result = spin_wave_structure_factor_spectrum(
                q_reduced, 1e-3 * omega_mev, relaxed["unit_vectors_local"],
                h_z, args.j_pm, j_pmpm_ratio=args.j_pmpm,
                j_zz_ev=args.j_zz_ev, eta_ev=1e-3 * args.eta_mev,
                nk=args.dssf_nk, spin_length=args.spin_length,
                regularization=args.regularization,
                interface_sublattices=SUBLATTICES, component=args.component,
                omega_chunk_size=args.omega_chunk_size,
            )
            intensity[condition] = result[args.component]
            max_imaginary_frequency_ratio[condition] = max(
                max_imaginary_frequency_ratio[condition],
                result["max_imaginary_frequency_ratio"],
            )

    write_hdf5(
        data_path, q_reduced, q_cartesian, path_distance, tick_positions,
        tick_indices, omega_mev, fields, field_phis, intensity,
        dispersion_bounds_mev,
        state_data, max_imaginary_frequency_ratio, args,
    )
    for phi_index, field_phi_degrees in enumerate(field_phis):
        plot_sublattice_spectra(
            intensity[phi_index], path_distance, tick_positions, omega_mev,
            fields, field_phi_degrees, args, sublattice_figures[phi_index],
        )
        plot_sublattice_sum(
            intensity[phi_index], path_distance, tick_positions, omega_mev,
            fields, field_phi_degrees, args, sum_figures[phi_index],
        )
    print(f"Data: {data_path}")
    for figure_path in sublattice_figures + sum_figures:
        print(f"Figure: {figure_path}")
    return data_path, sublattice_figures, sum_figures


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Plot SWT S_mu(q, omega) along Gamma-X-W-K-Gamma-L-U for three fields."
        )
    )
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    parser.add_argument("--fields", type=float, nargs=3, default=(0.25, 0.45, 1.5))
    parser.add_argument("--field-phis", type=float, nargs=2, default=(0.1, 30.1))
    parser.add_argument("--j-pm", type=float, default=0.205)
    parser.add_argument("--j-pmpm", type=float, default=0.0)
    parser.add_argument("--j-zz-ev", type=float, default=0.00019)
    parser.add_argument("--component", choices=COMPONENTS, default="one_magnon")
    parser.add_argument("--vmax-percentile", type=float, default=99.0)
    parser.add_argument("--points-per-segment", type=int, default=40)
    parser.add_argument("--n-omega", type=int, default=200)
    parser.add_argument("--eta-mev", type=float, default=0.02)
    parser.add_argument(
        "--lattice-constant", type=float, default=TTO_CUBIC_LATTICE_ANGSTROM
    )
    parser.add_argument("--dssf-nk", type=int, default=8)
    parser.add_argument("--spin-length", type=float, default=0.5)
    parser.add_argument("--regularization", type=float, default=1e-8)
    parser.add_argument("--stability-tilt", type=float, default=0.01)
    parser.add_argument("--random-starts", type=int, default=12)
    parser.add_argument("--omega-chunk-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    compute_and_plot(build_parser().parse_args())
