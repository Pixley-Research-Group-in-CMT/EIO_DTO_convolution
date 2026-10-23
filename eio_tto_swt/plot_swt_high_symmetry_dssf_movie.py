#!/usr/bin/env python3
"""Animate the SWT DSSF along the FCC symmetry path while varying field."""
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from config import TTO_CUBIC_LATTICE_ANGSTROM
from plot_swt_dssf_video import animation_writer
from plot_swt_high_symmetry_dssf import (
    HIGH_SYMMETRY_LABELS,
    HIGH_SYMMETRY_NAMES,
    coordinate_edges,
    energy_axis_from_bounds,
    fcc_high_symmetry_path,
    phi_file_label,
)
from swt_bz_maps import (
    COMPONENTS,
    FIELD_THETA_DEGREES,
    SUBLATTICES,
    prepare_ordered_state,
    shared_norm,
)
from tto_spin_wave import (
    magnon_dispersion_bounds,
    one_magnon_mode_index,
    spin_wave_structure_factor_spectrum,
)


def field_axis(mu_b_b_start, mu_b_b_stop, n_field_frames):
    """Return an inclusive, uniformly spaced field axis for the movie."""
    mu_b_b_start = float(mu_b_b_start)
    mu_b_b_stop = float(mu_b_b_stop)
    if not np.isfinite(mu_b_b_start) or not np.isfinite(mu_b_b_stop):
        raise ValueError("mu_b_b_start and mu_b_b_stop must be finite")
    if mu_b_b_start < 0.0 or mu_b_b_stop <= mu_b_b_start:
        raise ValueError(
            "field range must satisfy 0 <= mu_b_b_start < mu_b_b_stop"
        )
    if int(n_field_frames) != n_field_frames or int(n_field_frames) < 2:
        raise ValueError("n_field_frames must be an integer of at least 2")
    return np.linspace(mu_b_b_start, mu_b_b_stop, int(n_field_frames))


def _decorate_path_axis(axis, tick_positions):
    for position in tick_positions[1:-1]:
        axis.axvline(position, color="white", lw=0.45, alpha=0.55)
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(HIGH_SYMMETRY_LABELS)
    axis.set_xlim(tick_positions[0], tick_positions[-1])
    axis.set_xlabel("wave-vector path")


def _parameter_text(field_phi_degrees, args):
    return (
        rf"$\theta={FIELD_THETA_DEGREES:g}^\circ$, "
        rf"$\phi={field_phi_degrees:g}^\circ$, "
        rf"$J_{{zz}}={1e3 * args.j_zz_ev:g}$ meV, "
        rf"$J_{{\pm}}/J_{{zz}}={args.j_pm:g}$, "
        rf"$J_{{\pm\pm}}/J_{{zz}}={args.j_pmpm:g}$, "
        rf"$g_z={args.g_z:g}$, $g_\perp={args.g_perp:g}$, "
        rf"$\delta_1={args.delta1_per_mev:g}$ meV$^{{-1}}$, "
        rf"$\delta_2={args.delta2_per_mev:g}$ meV$^{{-1}}$, "
        rf"component={args.component.replace('_', ' ')}"
    )


def write_sublattice_video(
    values,
    omega_mev,
    path_distance,
    tick_positions,
    fields,
    field_phi_degrees,
    args,
    output_path,
):
    """Animate the three sublattice-resolved path spectra versus field."""
    norm = shared_norm(values, args.vmax_percentile)
    x_edges = coordinate_edges(path_distance)
    y_edges = coordinate_edges(omega_mev)
    figure, axes = plt.subplots(
        1,
        len(SUBLATTICES),
        figsize=(10.5, 3.7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    meshes = []
    for mu_index, mu in enumerate(SUBLATTICES):
        axis = axes[mu_index]
        mesh = axis.pcolormesh(
            x_edges,
            y_edges,
            values[0, mu_index],
            cmap="magma",
            norm=norm,
            shading="flat",
            edgecolors="none",
            linewidth=0.0,
            antialiased=False,
        )
        _decorate_path_axis(axis, tick_positions)
        axis.set_ylim(omega_mev[0], omega_mev[-1])
        axis.set_title(rf"$\mu={mu}$")
        meshes.append(mesh)
    axes[0].set_ylabel(r"$\omega$ (meV)")

    parameters = _parameter_text(field_phi_degrees, args)
    title = figure.suptitle("", fontsize=10)
    colorbar = figure.colorbar(
        meshes[-1], ax=axes.ravel().tolist(), shrink=0.88, pad=0.025
    )
    colorbar.set_label(r"$S_\mu(\mathbf{q},\omega)$ (eV$^{-1}$)")
    if args.vmax_percentile < 100.0:
        colorbar.ax.set_title(
            f"clipped at {args.vmax_percentile:g}%", fontsize=7, pad=6
        )

    def update(frame_index):
        for mu_index, mesh in enumerate(meshes):
            mesh.set_array(
                np.ma.masked_invalid(values[frame_index, mu_index]).ravel()
            )
        title.set_text(
            rf"$S_\mu(\mathbf{{q}},\omega)$, "
            rf"$\mu_B|B|/J_{{zz}}={fields[frame_index]:.5g}$"
            + "\n" + parameters
        )
        return tuple(meshes) + (title,)

    movie = FuncAnimation(
        figure,
        update,
        frames=len(fields),
        interval=1000.0 / args.fps,
        blit=False,
    )
    try:
        movie.save(
            output_path,
            writer=animation_writer(args.video_format, args.fps),
            dpi=args.dpi,
        )
    finally:
        plt.close(figure)


def write_sum_video(
    values,
    omega_mev,
    path_distance,
    tick_positions,
    fields,
    field_phi_degrees,
    args,
    output_path,
):
    """Animate the sum over mu=1,2,3 along the symmetry path."""
    summed = np.sum(values, axis=1)
    norm = shared_norm(summed, args.vmax_percentile)
    x_edges = coordinate_edges(path_distance)
    y_edges = coordinate_edges(omega_mev)
    figure, axis = plt.subplots(
        1, 1, figsize=(7.2, 4.6), constrained_layout=True
    )
    mesh = axis.pcolormesh(
        x_edges,
        y_edges,
        summed[0],
        cmap="magma",
        norm=norm,
        shading="flat",
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
    )
    _decorate_path_axis(axis, tick_positions)
    axis.set_ylim(omega_mev[0], omega_mev[-1])
    axis.set_ylabel(r"$\omega$ (meV)")

    parameters = _parameter_text(field_phi_degrees, args)
    title = figure.suptitle("", fontsize=10)
    colorbar = figure.colorbar(mesh, ax=axis, shrink=0.88, pad=0.025)
    colorbar.set_label(
        r"$\sum_{\mu=1}^{3}S_\mu(\mathbf{q},\omega)$ (eV$^{-1}$)"
    )
    if args.vmax_percentile < 100.0:
        colorbar.ax.set_title(
            f"clipped at {args.vmax_percentile:g}%", fontsize=7, pad=6
        )

    def update(frame_index):
        mesh.set_array(np.ma.masked_invalid(summed[frame_index]).ravel())
        title.set_text(
            r"$\sum_{\mu=1}^{3}S_\mu(\mathbf{q},\omega)$, "
            rf"$\mu_B|B|/J_{{zz}}={fields[frame_index]:.5g}$"
            + "\n" + parameters
        )
        return mesh, title

    movie = FuncAnimation(
        figure,
        update,
        frames=len(fields),
        interval=1000.0 / args.fps,
        blit=False,
    )
    try:
        movie.save(
            output_path,
            writer=animation_writer(args.video_format, args.fps),
            dpi=args.dpi,
        )
    finally:
        plt.close(figure)


def write_hdf5(
    output_path,
    q_reduced,
    q_cartesian,
    path_distance,
    tick_positions,
    tick_indices,
    omega_mev,
    fields,
    field_phis,
    intensity,
    dispersion_bounds_mev,
    state_data,
    max_imaginary_frequency_ratio,
    args,
):
    """Save every field frame, ordered state, and animation parameter."""
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
        output.create_dataset("mu_B_B_over_J_zz", data=fields)
        output.create_dataset("h_over_J_zz", data=fields)
        output.create_dataset("sublattice_indices", data=SUBLATTICES)
        output.create_dataset(
            "dispersion_minimum_mev", data=dispersion_bounds_mev[..., 0]
        )
        output.create_dataset(
            "dispersion_maximum_mev", data=dispersion_bounds_mev[..., 1]
        )
        output.create_dataset(
            "max_imaginary_frequency_ratio",
            data=max_imaginary_frequency_ratio,
        )
        output.create_dataset(
            "intensity", data=intensity, compression="gzip", shuffle=True
        )
        output.create_dataset(
            "intensity_mu_sum",
            data=np.sum(intensity, axis=2),
            compression="gzip",
            shuffle=True,
        )
        for name, values in state_data.items():
            output.create_dataset(name, data=values, compression="gzip")
        output.attrs.update(
            theory="linear spin wave, EIO_TTO Section 1, Eqs. (9)-(21)",
            observable="S_mu(q, omega) along Gamma-X-W-K-Gamma-L-U",
            component=args.component,
            animation_axis="mu_B_B_over_J_zz",
            momentum_path="Gamma-X-W-K-Gamma-L-U in the bulk FCC BZ",
            array_axes=(
                "field_phi_degrees, mu_B_B_over_J_zz, sublattice, omega, q_path"
            ),
            mu_sum_axes=(
                "field_phi_degrees, mu_B_B_over_J_zz, omega, q_path"
            ),
            mu_sum_definition="sum over sublattice indices 1, 2, 3",
            field_theta_degrees=FIELD_THETA_DEGREES,
            J_zz_ev=args.j_zz_ev,
            J_pm_over_J_zz=args.j_pm,
            J_pmpm_over_J_zz=args.j_pmpm,
            field_energy_definition="mu_B |B|",
            g_z=args.g_z,
            g_perp=args.g_perp,
            delta1_per_mev=args.delta1_per_mev,
            delta2_per_mev=args.delta2_per_mev,
            lorentzian_eta_mev=args.eta_mev,
            tto_cubic_lattice_angstrom=args.lattice_constant,
            dssf_nk=args.dssf_nk,
            points_per_path_segment=args.points_per_segment,
            n_omega=args.n_omega,
            n_field_frames=len(fields),
            fps=args.fps,
            video_format=args.video_format,
            spin_length=args.spin_length,
            regularization=args.regularization,
            stability_tilt=args.stability_tilt,
            random_starts=args.random_starts,
            vmax_percentile=args.vmax_percentile,
        )


def compute_and_animate(args):
    """Compute the field scan and write HDF5 plus four field movies."""
    fields = field_axis(
        args.mu_b_b_start, args.mu_b_b_stop, args.n_field_frames
    )
    field_phis = np.asarray(args.field_phis, dtype=float)
    if field_phis.shape != (2,) or not np.all(np.isfinite(field_phis)):
        raise ValueError("field_phis must contain exactly two finite values")
    if not np.isfinite(args.j_zz_ev) or args.j_zz_ev <= 0.0:
        raise ValueError("j_zz_ev must be finite and positive")
    if not np.isfinite(args.eta_mev) or args.eta_mev <= 0.0:
        raise ValueError("eta_mev must be finite and positive")
    if args.dssf_nk <= 0 or args.omega_chunk_size <= 0:
        raise ValueError("dssf_nk and omega_chunk_size must be positive")
    if args.n_omega < 2:
        raise ValueError("n_omega must be at least 2")
    if args.fps <= 0.0 or args.dpi <= 0:
        raise ValueError("fps and dpi must be positive")
    if not 0.0 < args.vmax_percentile <= 100.0:
        raise ValueError("vmax_percentile must be in (0, 100]")
    animation_writer(args.video_format, args.fps)

    run_directory = args.output_root.expanduser().resolve()
    data_directory = run_directory / "data"
    video_directory = run_directory / "videos"
    data_directory.mkdir(parents=True, exist_ok=True)
    video_directory.mkdir(parents=True, exist_ok=True)
    data_path = data_directory / "swt_dssf_high_symmetry_field_movie.hdf5"
    sublattice_videos = tuple(
        video_directory
        / (
            "swt_dssf_high_symmetry_sublattices_phi_"
            f"{phi_file_label(phi)}.{args.video_format}"
        )
        for phi in field_phis
    )
    sum_videos = tuple(
        video_directory
        / (
            "swt_dssf_high_symmetry_mu_sum_phi_"
            f"{phi_file_label(phi)}.{args.video_format}"
        )
        for phi in field_phis
    )
    outputs = (data_path,) + sublattice_videos + sum_videos
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output already exists; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    (
        q_reduced,
        q_cartesian,
        path_distance,
        tick_positions,
        tick_indices,
    ) = fcc_high_symmetry_path(args.points_per_segment, args.lattice_constant)

    scan_shape = (len(field_phis), len(fields))
    states = np.empty(scan_shape, dtype=object)
    dispersion_bounds_mev = np.empty(scan_shape + (2,), dtype=float)
    max_imaginary_frequency_ratio = np.zeros(scan_shape, dtype=float)
    state_data = {
        "bare_field_local_over_J_zz": np.empty(scan_shape + (4, 3)),
        "effective_field_local_over_J_zz": np.empty(scan_shape + (4, 3)),
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
                f"Preparing frame {field_index + 1}/{len(fields)} for "
                "phi="
                f"{field_phi_degrees:g} deg, mu_B|B|/Jzz={field_ratio:g}",
                flush=True,
            )
            h_z, initial_spins, relaxed = prepare_ordered_state(
                args, field_ratio, field_phi_degrees
            )
            bounds = magnon_dispersion_bounds(
                q_reduced,
                relaxed["unit_vectors_local"],
                h_z,
                args.j_pm,
                j_pmpm_ratio=args.j_pmpm,
                j_zz_ev=args.j_zz_ev,
                spin_length=args.spin_length,
                regularization=args.regularization,
                mode_index=one_magnon_mode_index(args.component),
            )
            states[condition] = (h_z, relaxed)
            dispersion_bounds_mev[condition] = 1e3 * np.asarray([
                bounds["minimum_ev"],
                bounds["maximum_ev"],
            ])
            max_imaginary_frequency_ratio[condition] = (
                bounds["max_imaginary_frequency_ratio"]
            )
            state_data["bare_field_local_over_J_zz"][condition] = (
                relaxed["bare_field_local_over_J_zz"]
            )
            state_data["effective_field_local_over_J_zz"][condition] = h_z
            state_data["initial_spins_local"][condition] = initial_spins
            state_data["ordered_spins_local"][condition] = relaxed["spins_local"]
            state_data["initial_classical_energy"][condition] = relaxed[
                "initial_energy"
            ]
            state_data["relaxed_classical_energy"][condition] = relaxed["energy"]
            state_data["min_classical_curvature"][condition] = relaxed[
                "min_classical_curvature"
            ]
            state_data["max_torque"][condition] = relaxed["max_torque"]
            state_data["stationarity_refinement_used"][condition] = relaxed.get(
                "stationarity_refinement_used", False
            )

    omega_mev = energy_axis_from_bounds(
        np.max(dispersion_bounds_mev[..., 1]), args.n_omega, args.component
    )
    print(
        f"DSSF energy range: {omega_mev[0]:.6g} to "
        f"{omega_mev[-1]:.6g} meV",
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
                f"Evaluating frame {field_index + 1}/{len(fields)} for "
                "phi="
                f"{field_phi_degrees:g} deg, mu_B|B|/Jzz={field_ratio:g}",
                flush=True,
            )
            result = spin_wave_structure_factor_spectrum(
                q_reduced,
                1e-3 * omega_mev,
                relaxed["unit_vectors_local"],
                h_z,
                args.j_pm,
                j_pmpm_ratio=args.j_pmpm,
                j_zz_ev=args.j_zz_ev,
                eta_ev=1e-3 * args.eta_mev,
                nk=args.dssf_nk,
                spin_length=args.spin_length,
                regularization=args.regularization,
                interface_sublattices=SUBLATTICES,
                component=args.component,
                omega_chunk_size=args.omega_chunk_size,
            )
            intensity[condition] = result[args.component]
            max_imaginary_frequency_ratio[condition] = max(
                max_imaginary_frequency_ratio[condition],
                result["max_imaginary_frequency_ratio"],
            )

    write_hdf5(
        data_path,
        q_reduced,
        q_cartesian,
        path_distance,
        tick_positions,
        tick_indices,
        omega_mev,
        fields,
        field_phis,
        intensity,
        dispersion_bounds_mev,
        state_data,
        max_imaginary_frequency_ratio,
        args,
    )
    for phi_index, field_phi_degrees in enumerate(field_phis):
        print(f"Encoding movies for phi={field_phi_degrees:g} deg", flush=True)
        write_sublattice_video(
            intensity[phi_index],
            omega_mev,
            path_distance,
            tick_positions,
            fields,
            field_phi_degrees,
            args,
            sublattice_videos[phi_index],
        )
        write_sum_video(
            intensity[phi_index],
            omega_mev,
            path_distance,
            tick_positions,
            fields,
            field_phi_degrees,
            args,
            sum_videos[phi_index],
        )
    print(f"Data: {data_path}")
    for video_path in sublattice_videos + sum_videos:
        print(f"Video: {video_path}")
    return data_path, sublattice_videos, sum_videos


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Animate SWT S_mu(q, omega) along Gamma-X-W-K-Gamma-L-U "
            "while varying mu_B |B| / J_zz."
        )
    )
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    parser.add_argument("--mu-b-b-start", type=float, default=0.25)
    parser.add_argument("--mu-b-b-stop", type=float, default=1.5)
    parser.add_argument("--n-field-frames", type=int, default=31)
    parser.add_argument("--field-phis", type=float, nargs=2, default=(0.1, 30.1))
    parser.add_argument("--j-pm", type=float, default=0.205)
    parser.add_argument("--j-pmpm", type=float, default=0.0)
    parser.add_argument("--g-z", type=float, default=10.0)
    parser.add_argument("--g-perp", type=float, default=0.0)
    parser.add_argument("--delta1-per-mev", type=float, default=1.0 / 1.4)
    parser.add_argument("--delta2-per-mev", type=float, default=1.0 / 1.4)
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
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--video-format", choices=("gif", "mp4"), default="gif")
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    compute_and_animate(build_parser().parse_args())
