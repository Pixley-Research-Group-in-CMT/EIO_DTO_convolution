#!/usr/bin/env python3
"""Animate the Sec. 4.2 spin-wave DSSF across the magnon bandwidth."""
import argparse
from pathlib import Path
import shutil

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np

from config import TTO_CUBIC_LATTICE_ANGSTROM
from swt_bz_maps import (
    COMPONENTS,
    FIELD_PHI_FILE_LABELS,
    FIELD_PHIS_DEGREES,
    FIELD_RATIOS,
    FIELD_THETA_DEGREES,
    SUBLATTICES,
    kagome_bz_grid,
    prepare_ordered_state,
    shared_norm,
)
from tto_spin_wave import (
    magnon_dispersion_bounds,
    one_magnon_mode_index,
    spin_wave_structure_factor_spectrum,
)


def animation_writer(video_format, fps):
    """Return a portable GIF writer or an FFmpeg-backed MP4 writer."""
    if video_format == "gif":
        return PillowWriter(fps=fps)
    if video_format != "mp4":
        raise ValueError(f"Unknown video format: {video_format}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError(
                "MP4 output requires ffmpeg. Use VIDEO_FORMAT=gif or install "
                "ffmpeg/imageio-ffmpeg in the cluster environment."
            ) from error
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_path
    return FFMpegWriter(
        fps=fps,
        metadata={"title": "TTO spin-wave dynamic structure factor"},
        extra_args=["-pix_fmt", "yuv420p"],
    )


def energy_axis_from_dispersion(dispersion_bounds_mev, n_frames):
    """Return a common frame axis spanning all displayed magnon bands."""
    bounds = np.asarray(dispersion_bounds_mev, dtype=float)
    if bounds.shape[-1] != 2 or not np.all(np.isfinite(bounds)):
        raise ValueError("dispersion_bounds_mev must end with finite min/max pairs")
    if int(n_frames) < 2:
        raise ValueError("n_omega_frames must be at least 2")
    minimum = float(np.min(bounds[..., 0]))
    maximum = float(np.max(bounds[..., 1]))
    if minimum < 0.0 or maximum <= minimum:
        raise ValueError("Magnon dispersion must have a positive energy range")
    return np.linspace(minimum, maximum, int(n_frames))


def write_video(
    values,
    omega_mev,
    q_edges,
    bz_vertices,
    field_phi_degrees,
    args,
    output_path,
):
    """Write one 3-field by 3-sublattice DSSF animation."""
    norm = shared_norm(values, args.vmax_percentile)
    figure, axes = plt.subplots(
        len(FIELD_RATIOS), len(SUBLATTICES),
        figsize=(9.0, 8.4), sharex=True, sharey=True,
        constrained_layout=True,
    )
    boundary = np.vstack((bz_vertices, bz_vertices[0]))
    ticks = np.asarray((q_edges[0], 0.0, q_edges[-1]))
    meshes = []
    for field_index, field_ratio in enumerate(FIELD_RATIOS):
        row = []
        for sublattice_index, mu in enumerate(SUBLATTICES):
            axis = axes[field_index, sublattice_index]
            mesh = axis.pcolormesh(
                q_edges,
                q_edges,
                values[field_index, sublattice_index, 0],
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
                axis.set_title(rf"$\mu={mu}$")
            if field_index == len(FIELD_RATIOS) - 1:
                axis.set_xlabel(r"$q_{[\bar{1}01]}$ ($\mathrm{\AA}^{-1}$)")
            if sublattice_index == 0:
                axis.set_ylabel(
                    rf"$\mu_B|B|/J_{{zz}}={field_ratio:g}$" + "\n"
                    + r"$q_{[1\bar{2}1]}$ ($\mathrm{\AA}^{-1}$)"
                )
            row.append(mesh)
        meshes.append(row)

    parameters = (
        rf"$(111)$ kagome-plane BZ, $\theta={FIELD_THETA_DEGREES:g}^\circ$, "
        rf"$\phi={field_phi_degrees:g}^\circ$, "
        rf"$J_{{zz}}={1e3 * args.j_zz_ev:g}$ meV, "
        rf"$J_{{\pm}}/J_{{zz}}={args.j_pm:g}$, "
        rf"$J_{{\pm\pm}}/J_{{zz}}={args.j_pmpm:g}$, "
        rf"$g_z={args.g_z:g}$, $g_\perp={args.g_perp:g}$, "
        rf"$\delta_1={args.delta1_per_mev:g}$ meV$^{{-1}}$, "
        rf"$\delta_2={args.delta2_per_mev:g}$ meV$^{{-1}}$, "
        rf"component={args.component.replace('_', ' ')}"
    )
    title = figure.suptitle("", fontsize=10)
    colorbar = figure.colorbar(
        meshes[-1][-1], ax=axes.ravel().tolist(), shrink=0.84, pad=0.025
    )
    colorbar.set_label(r"$S_\mu(\mathbf{q},\omega)$ (eV$^{-1}$)")
    if args.vmax_percentile < 100.0:
        colorbar.ax.set_title(
            f"clipped at {args.vmax_percentile:g}%", fontsize=7, pad=6
        )

    def update(frame_index):
        for field_index in range(len(FIELD_RATIOS)):
            for sublattice_index in range(len(SUBLATTICES)):
                frame = np.ma.masked_invalid(
                    values[field_index, sublattice_index, frame_index]
                )
                meshes[field_index][sublattice_index].set_array(frame.ravel())
        title.set_text(
            rf"$S_\mu(\mathbf{{q}},\omega)$, "
            rf"$\omega={omega_mev[frame_index]:.5g}$ meV" + "\n" + parameters
        )
        return tuple(mesh for row in meshes for mesh in row) + (title,)

    movie = FuncAnimation(
        figure,
        update,
        frames=len(omega_mev),
        interval=1000.0 / args.fps,
        blit=False,
    )
    writer = animation_writer(args.video_format, args.fps)
    try:
        movie.save(output_path, writer=writer, dpi=args.dpi)
    finally:
        plt.close(figure)


def write_hdf5(
    output_path,
    q_axis,
    q_grid,
    inside_bz,
    omega_mev,
    intensity,
    dispersion_bounds_mev,
    state_data,
    max_imaginary_frequency_ratio,
    args,
):
    """Save the selected DSSF spectrum and all animation metadata."""
    with h5py.File(output_path, "w") as output:
        output.create_dataset("q_x_inv_angstrom", data=q_axis)
        output.create_dataset("q_y_inv_angstrom", data=q_axis)
        output.create_dataset("q_reduced", data=q_grid, compression="gzip")
        output.create_dataset("inside_2d_bz", data=inside_bz, compression="gzip")
        output.create_dataset("omega_mev", data=omega_mev)
        output.create_dataset("field_phi_degrees", data=FIELD_PHIS_DEGREES)
        output.create_dataset("mu_B_B_over_J_zz", data=FIELD_RATIOS)
        output.create_dataset("h_over_J_zz", data=FIELD_RATIOS)
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
        for name, values in state_data.items():
            output.create_dataset(name, data=values, compression="gzip")
        output.create_dataset(
            "intensity", data=intensity, compression="gzip", shuffle=True
        )
        output.attrs.update(
            theory="linear spin wave, Sec. 4.2, Eqs. (88)-(100)",
            observable="S_mu(q, omega) across the sampled magnon bandwidth",
            component=args.component,
            first_bz="hexagonal 2D BZ perpendicular to cubic [111]",
            momentum_axes="x || [-1,0,1], y || [1,-2,1], in inverse Angstrom",
            array_axes=(
                "field_phi, mu_B_B_over_J_zz, sublattice, omega, q_y, q_x"
            ),
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
            n_q=args.n_q,
            n_omega_frames=args.n_omega_frames,
            fps=args.fps,
            video_format=args.video_format,
            vmax_percentile=args.vmax_percentile,
        )


def compute_and_animate(args):
    """Compute the six-condition DSSF spectrum and write two animations."""
    if args.j_zz_ev <= 0.0:
        raise ValueError("j_zz_ev must be positive")
    if args.eta_mev <= 0.0:
        raise ValueError("eta_mev must be positive")
    if args.dssf_nk <= 0:
        raise ValueError("dssf_nk must be positive")
    if args.n_omega_frames < 2:
        raise ValueError("n_omega_frames must be at least 2")
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
    data_path = data_directory / "swt_dssf.hdf5"
    video_paths = tuple(
        video_directory / f"swt_dssf_phi_{label}.{args.video_format}"
        for label in FIELD_PHI_FILE_LABELS
    )
    existing = [path for path in (data_path,) + video_paths if path.exists()]
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
    states = np.empty(scan_shape, dtype=object)
    dispersion_bounds_mev = np.empty(scan_shape + (2,))
    max_imaginary_frequency_ratio = np.zeros(scan_shape)
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

    for phi_index, field_phi_degrees in enumerate(FIELD_PHIS_DEGREES):
        for field_index, field_ratio in enumerate(FIELD_RATIOS):
            condition = (phi_index, field_index)
            print(
                "Preparing dispersion for "
                "phi="
                f"{field_phi_degrees:g} deg, mu_B|B|/Jzz={field_ratio:g}",
                flush=True,
            )
            h_z, initial_spins, relaxed = prepare_ordered_state(
                args, field_ratio, field_phi_degrees
            )
            bounds = magnon_dispersion_bounds(
                q_points,
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
                bounds["minimum_ev"], bounds["maximum_ev"]
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
            state_data["initial_classical_energy"][condition] = relaxed["initial_energy"]
            state_data["relaxed_classical_energy"][condition] = relaxed["energy"]
            state_data["min_classical_curvature"][condition] = (
                relaxed["min_classical_curvature"]
            )
            state_data["max_torque"][condition] = relaxed["max_torque"]
            state_data["stationarity_refinement_used"][condition] = relaxed.get(
                "stationarity_refinement_used", False
            )

    omega_mev = energy_axis_from_dispersion(
        dispersion_bounds_mev, args.n_omega_frames
    )
    print(
        f"Animation energy range: {omega_mev[0]:.6g} to "
        f"{omega_mev[-1]:.6g} meV",
        flush=True,
    )
    intensity_shape = scan_shape + (
        len(SUBLATTICES), len(omega_mev), args.n_q, args.n_q
    )
    intensity = np.full(intensity_shape, np.nan, dtype=np.float32)
    for phi_index, field_phi_degrees in enumerate(FIELD_PHIS_DEGREES):
        for field_index, field_ratio in enumerate(FIELD_RATIOS):
            condition = (phi_index, field_index)
            h_z, relaxed = states[condition]
            print(
                "Evaluating DSSF spectrum for "
                "phi="
                f"{field_phi_degrees:g} deg, mu_B|B|/Jzz={field_ratio:g}",
                flush=True,
            )
            result = spin_wave_structure_factor_spectrum(
                q_points,
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
            intensity[condition][:, :, inside_bz] = result[args.component]
            max_imaginary_frequency_ratio[condition] = max(
                max_imaginary_frequency_ratio[condition],
                result["max_imaginary_frequency_ratio"],
            )

    write_hdf5(
        data_path,
        q_axis,
        q_grid,
        inside_bz,
        omega_mev,
        intensity,
        dispersion_bounds_mev,
        state_data,
        max_imaginary_frequency_ratio,
        args,
    )
    for phi_index, (field_phi_degrees, video_path) in enumerate(zip(
        FIELD_PHIS_DEGREES, video_paths
    )):
        print(f"Encoding {video_path.name}", flush=True)
        write_video(
            intensity[phi_index],
            omega_mev,
            q_edges,
            bz_vertices,
            field_phi_degrees,
            args,
            video_path,
        )
    print(f"Data: {data_path}")
    for video_path in video_paths:
        print(f"Video: {video_path}")
    return data_path, video_paths


def build_parser():
    parser = argparse.ArgumentParser(
        description="Animate SWT S_mu(q, omega) across the magnon bandwidth."
    )
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    parser.add_argument("--j-pm", type=float, default=0.205)
    parser.add_argument("--j-pmpm", type=float, default=0.0)
    parser.add_argument("--g-z", type=float, default=10.0)
    parser.add_argument("--g-perp", type=float, default=0.0)
    parser.add_argument("--delta1-per-mev", type=float, default=1.0 / 1.4)
    parser.add_argument("--delta2-per-mev", type=float, default=1.0 / 1.4)
    parser.add_argument("--j-zz-ev", type=float, default=0.00019)
    parser.add_argument("--component", choices=COMPONENTS, default="one_magnon")
    parser.add_argument("--vmax-percentile", type=float, default=95.0)
    parser.add_argument("--n-omega-frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--video-format", choices=("gif", "mp4"), default="gif")
    parser.add_argument("--eta-mev", type=float, default=0.02)
    parser.add_argument("--n-q", type=int, default=101)
    parser.add_argument(
        "--lattice-constant", type=float, default=TTO_CUBIC_LATTICE_ANGSTROM
    )
    parser.add_argument("--dssf-nk", type=int, default=8)
    parser.add_argument("--spin-length", type=float, default=0.5)
    parser.add_argument("--regularization", type=float, default=1e-8)
    parser.add_argument("--stability-tilt", type=float, default=0.01)
    parser.add_argument("--random-starts", type=int, default=12)
    parser.add_argument("--omega-chunk-size", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    compute_and_animate(build_parser().parse_args())
