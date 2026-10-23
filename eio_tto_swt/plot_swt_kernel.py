#!/usr/bin/env python3
"""Show the Eq. (24) integrated SWT kernel side by side or as a field movie.

The energy-conservation delta sets omega_ij = epsilon_j - epsilon_i.
Both layouts show the momentum-only kernel at their configured fields.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from config import ROOT, ORBITAL_HOPPINGS
from eio_model_selection import eio_model_label, prepare_eio, eio_source_files
from eio_tb import transform_to_slab_coordinates
from plot_swt_dssf_video import animation_writer
from swt_bz_maps import prepare_ordered_state, shared_norm, projection_title
from tto_spin_wave import spin_wave_structure_factor
from tto_structure_factor import eio_to_tto_reduced_momentum
from tto_surface_projection import validate_q_perp_sampling, q_perp_samples_for_mode, surface_projected_dynamic_structure_factor

# Reuse the actual transport contraction rather than a second implementation.
_spec = importlib.util.spec_from_file_location("scattering_stage", ROOT / "03_compute_scattering.py")
_scattering = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scattering)


def transport_vertex(kf):
    kf = np.asarray(kf, dtype=float)
    norms = np.linalg.norm(kf, axis=1)
    if np.any(norms == 0):
        raise ValueError("Transport scattering angle is undefined at Gamma")
    vertex = 1.0 - np.clip(kf @ kf.T / np.outer(norms, norms), -1.0, 1.0)
    np.fill_diagonal(vertex, 0.0)
    return vertex


def contract_kernel(spectrum, weights, vertex):
    """Contract (mu, omega, i, j) to (omega, i, j), exactly as transport."""
    return np.einsum("mwij,mij->wij", spectrum, weights, optimize=True) * vertex


def transfer_map(kf):
    """Fold transfers into the EIO first BZ; group coincident plotted points.

    The DSSF is evaluated at the original transfers, before plotting folds.
    Group sums retain every ordered pair, including equivalent boundary points.
    """
    lattice = 5.1350 * np.asarray([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    geometry = transform_to_slab_coordinates(
        {"lattice_vectors": lattice, "wannier_centres": np.zeros((1, 3))},
        {"vectors": {}},
    )
    reciprocal = geometry["reciprocal_vectors"][:, :2]
    q = kf[:, None, :] - kf[None, :, :]
    shifts = np.asarray([reciprocal @ [i, j] for i in range(-2, 3) for j in range(-2, 3)])
    candidates = q.reshape(-1, 3)[:, None, :] - shifts[None, :, :]
    nearest = np.argmin(np.linalg.norm(candidates, axis=2), axis=1)
    folded = candidates[np.arange(len(candidates)), nearest]
    points, groups = np.unique(np.round(folded[:, :2], 10), axis=0, return_inverse=True)
    radius = np.linalg.norm(reciprocal[:, 0]) / np.sqrt(3.0)
    angles = np.arange(7) * np.pi / 3
    boundary = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    return q, points, groups, boundary


def group_kernel(kernel, groups, n_points):
    """Sum ordered-pair contributions at each displayed transfer (no interpolation)."""
    values = np.asarray(kernel).copy()
    grouped = np.zeros((len(values), n_points))
    for frame, pair_values in enumerate(values):
        np.add.at(grouped[frame], groups, pair_values.ravel())
    return grouped


def field_axis(start, stop, count):
    """Uniform endpoint-inclusive bare-field scan, as in swt_path_movie."""
    if not np.isfinite([start, stop]).all() or not 0 <= start < stop:
        raise ValueError("Require 0 <= mu_b_b_start < mu_b_b_stop, both finite")
    if isinstance(count, bool) or int(count) != count or count < 2:
        raise ValueError("n_field_frames must be an integer of at least 2")
    return np.linspace(start, stop, int(count))


def selected_fields(args):
    if args.plot_mode == "movie":
        return field_axis(args.mu_b_b_start, args.mu_b_b_stop, args.n_field_frames)
    fields = np.asarray(args.fields, dtype=float)
    if (fields.shape != (3,) or not np.isfinite(fields).all()
            or np.any(fields < 0) or len(np.unique(fields)) != 3):
        raise ValueError("side_by_side requires three distinct finite nonnegative fields")
    return fields


def parameter_text(args):
    parameters = (
        f"{eio_model_label({'eio_model': args.eio_model})}; total (one + two magnon); "
        + r"$\theta=90^\circ$"
        + "\n" + projection_title(args)
        + rf"; $J_\pm/J_{{zz}}={args.j_pm:g}$, $J_{{\pm\pm}}/J_{{zz}}={args.j_pmpm:g}$"
        + "\n" + rf"$g_z={args.g_z:g}$, $g_\perp={args.g_perp:g}$, "
        + rf"$\delta_1={args.delta1_per_mev:g}$, $\delta_2={args.delta2_per_mev:g}$ meV$^{{-1}}$"
    )
    selection = "all incoming states"

    return parameters, selection


def kernel_color_norm(values, percentile):
    # An entirely unstable scan still produces labeled gray panels.
    values = np.asarray(values)
    return shared_norm(values if np.isfinite(values).any() else np.zeros(1), percentile)


def set_panel_status(axis, artist, label, values):
    stable = bool(np.isfinite(values).all())
    artist.set_array(np.ma.masked_invalid(values))
    artist.set_visible(stable)
    axis.set_facecolor("white" if stable else "0.82")
    label.set_visible(not stable)


def decorate_map(axis, points, boundary, value, norm, args):
    artist = axis.scatter(points[:, 0], points[:, 1], c=value,
                          s=args.marker_size, cmap="Blues", norm=norm, edgecolors="none")
    axis.plot(boundary[:, 0], boundary[:, 1], color="0.2", lw=1)
    axis.set_aspect("equal")
    return artist


def add_colorbar(figure, artist, axes, args):
    colorbar = figure.colorbar(artist, ax=list(np.asarray(axes).ravel()), shrink=0.8, pad=0.02)
    colorbar.set_label(r"$\sum_{(k,k')\to q} K_{kk'}$ (arb. units)")
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    if args.vmax_percentile < 100:
        colorbar.ax.set_title(f"clipped at {args.vmax_percentile:g}%", fontsize=8)


def make_side_by_side(values, fields, points, boundary, args):
    """Three field rows and two phi columns, with one intensity scale."""
    norm = kernel_color_norm(values, args.vmax_percentile)
    figure, axes = plt.subplots(3, 2, figsize=(9, 11), sharex=True, sharey=True,
                                constrained_layout=True)
    for row, field in enumerate(fields):
        for column, phi in enumerate(args.phi_degrees):
            axis = axes[row, column]
            artist = decorate_map(axis, points, boundary, values[column, row], norm, args)
            label = axis.text(.5, .5, "Unstable", transform=axis.transAxes,
                              ha="center", va="center", color="0.25", fontsize=14)
            set_panel_status(axis, artist, label, values[column, row])
            axis.set_title(rf"$\phi={phi:g}^\circ$, $\mu_B|B|/J_{{zz}}={field:g}$")
            if row == 2:
                axis.set_xlabel(r"$q_x$ ($\mathrm{\AA}^{-1}$)")
            if column == 0:
                axis.set_ylabel(r"$q_y$ ($\mathrm{\AA}^{-1}$)")
    add_colorbar(figure, artist, axes, args)
    parameters, selection = parameter_text(args)
    figure.suptitle("Integrated SWT kernel (Eq. 24); " + selection + "\n" + parameters, fontsize=10)
    return figure


def make_movie(values, fields, points, boundary, args):
    """Two phi panels with a shared scale recomputed independently each frame."""
    norm = kernel_color_norm(values[:, 0], args.vmax_percentile)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True,
                                constrained_layout=True)
    artists, labels = [], []
    for column, (axis, phi) in enumerate(zip(axes, args.phi_degrees)):
        artists.append(decorate_map(axis, points, boundary, values[column, 0], norm, args))
        labels.append(axis.text(.5, .5, "Unstable", transform=axis.transAxes,
                                ha="center", va="center", color="0.25", fontsize=14))
        axis.set_title(rf"$\phi={phi:g}^\circ$")
        axis.set_xlabel(r"$q_x$ ($\mathrm{\AA}^{-1}$)")
    axes[0].set_ylabel(r"$q_y$ ($\mathrm{\AA}^{-1}$)")
    add_colorbar(figure, artists[0], axes, args)
    title = figure.suptitle("", fontsize=10)
    parameters, selection = parameter_text(args)

    def update(frame):
        frame_norm = kernel_color_norm(values[:, frame], args.vmax_percentile)
        for column, artist in enumerate(artists):
            artist.set_norm(frame_norm)
            set_panel_status(axes[column], artist, labels[column], values[column, frame])
        title.set_text("Integrated SWT kernel (Eq. 24); " + selection
                       + "\n" + rf"$\mu_B|B|/J_{{zz}}={fields[frame]:.5g}$; frame {frame+1}/{len(fields)}"
                       + "\n" + parameters)
        return (*artists, *labels, title)

    update(0)
    movie = FuncAnimation(figure, update, frames=len(fields), interval=1000 / args.fps)
    return figure, movie, update


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plot-mode", choices=("side_by_side", "movie"), default="movie")
    parser.add_argument("--fields", type=float, nargs=3, default=(.04, .045, .05),
                        help="three fields used only for side_by_side")
    parser.add_argument("--mu-b-b-start", type=float, default=0.04)
    parser.add_argument("--mu-b-b-stop", type=float, default=0.05)
    parser.add_argument("--n-field-frames", type=int, default=11)
    parser.add_argument("--eio-model", choices=("original", "symmetrized"), default="symmetrized")
    parser.add_argument("--eio-states", type=Path, help="optional existing nk24 EIO states; model must match")
    parser.add_argument("--phi-degrees", type=float, nargs=2, default=(0., 30.),
                        help="two angles shown as columns, evaluated at these exact angles")
    for name, default in (("j-pm", .2), ("j-pmpm", 0.), ("g-z", 10.), ("g-perp", 0.),
                          ("delta1-per-mev", 0.), ("delta2-per-mev", 0.), ("j-zz-ev", .00017),
                          ("eta-mev", .02), ("lattice-constant", 10.154),
                          ("spin-length", .5), ("regularization", 1e-8), ("stability-tilt", .01),
                          ("fps", 10.), ("vmax-percentile", 100.), ("marker-size", 20.)):
        parser.add_argument("--" + name, type=float, default=default)
    for name, default in (("dssf-nk", 8), ("n-q-perp", 8),
                          ("random-starts", 8), ("dpi", 120)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--q-perp-mode", choices=("slice", "integrated"), default="integrated")
    parser.add_argument("--video-format", choices=("gif", "mp4"), default="gif")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate(args):
    selected_fields(args)
    if not np.isfinite(args.phi_degrees).all() or len(set(args.phi_degrees)) != 2:
        raise ValueError("phi_degrees must contain two distinct finite angles")
    for name, value in vars(args).items():
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(name + " must be finite")
    for name in ("j_zz_ev", "eta_mev", "lattice_constant", "spin_length", "regularization",
                 "fps", "marker_size", "dssf_nk", "dpi", "stability_tilt"):
        if getattr(args, name) <= 0:
            raise ValueError(name + " must be positive")
    if args.random_starts < 0:
        raise ValueError("random_starts must be nonnegative")
    if not 0 < args.vmax_percentile <= 100:
        raise ValueError("vmax_percentile must be in (0, 100]")
    validate_q_perp_sampling(args.q_perp_mode, args.n_q_perp)


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate(args)
    run = args.output_root.expanduser().resolve()
    data = run / "data"
    plot_directory = run / ("movies" if args.plot_mode == "movie" else "figures")
    data.mkdir(parents=True, exist_ok=True)
    plot_directory.mkdir(parents=True, exist_ok=True)
    stem = "swt_kernel_movie" if args.plot_mode == "movie" else "swt_kernel_side_by_side"
    output = data / f"{stem}.hdf5"
    suffix = args.video_format if args.plot_mode == "movie" else "pdf"
    plot_path = plot_directory / f"{stem}.{suffix}"
    if not args.overwrite and any(p.exists() for p in (output, plot_path)):
        raise FileExistsError("Kernel output exists; choose a new run or pass --overwrite")
    # Archive the numerical sources and the selected hopping tensor for reuse.
    sources = [ROOT / name for name in (
        "plot_swt_kernel.py", "01_generate_eio.py", "03_compute_scattering.py",
        "config.py", "eio_tb.py", "swt_bz_maps.py", "swt_transport_components.py",
        "tto_spin_wave.py", "tto_structure_factor.py", "tto_surface_projection.py",
        "plot_swt_dssf_video.py",
    )] + eio_source_files({"eio_model": args.eio_model})
    hashes = {}
    for source in sources:
        target = run / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    (run / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    eio = data / "surface_states_summary_nk24.hdf5"
    if args.eio_states:
        source = args.eio_states.expanduser().resolve()
        with h5py.File(source) as h5:
            if h5.attrs.get("eio_model", "original") != args.eio_model:
                raise ValueError("eio_states model does not match --eio-model")
        if source != eio:
            shutil.copy2(source, eio)

    def stage(name, command):
        subprocess.run([sys.executable, "-u", *command], cwd=ROOT, check=True,
                       env=dict(os.environ, EIO_TTO_OUTPUT_ROOT=str(run), EIO_TTO_EIO_MODEL=args.eio_model))

    prepare_eio({"eio_model": args.eio_model}, data, stage)
    with h5py.File(eio) as h5:
        kf = h5["kf"][...].reshape(-1, 3)
        energies = h5["eigf"][...].reshape(-1)
        weights = np.asarray([
            sum(_scattering.local_eio_weights(h5[f"proj_sl{sl}"][...], np.asarray(ORBITAL_HOPPINGS))
                for sl in _scattering.EIO_NEIGHBORS[mu])
            for mu in range(3)
        ])
    vertex = transport_vertex(kf)
    q, points, groups, boundary = transfer_map(kf)
    q_reduced = eio_to_tto_reduced_momentum(q, args.lattice_constant)
    energy_transfer = energies[None, :] - energies[:, None]
    fields = selected_fields(args)
    kernels, maps, ordered, fields_local, imaginary, errors = [], [], [], [], [], []
    for phi in args.phi_degrees:
        phi_kernels, phi_maps, phi_ordered, phi_fields_local, phi_imaginary, phi_errors = [], [], [], [], [], []
        for field in fields:
            print(f"Computing kernel: phi={phi:g}, mu_B |B|/Jzz={field:g}", flush=True)
            h_local, initial, state = prepare_ordered_state(args, field, phi)
            solver_kwargs = dict(
                j_pmpm_ratio=args.j_pmpm, j_zz_ev=args.j_zz_ev, eta_ev=args.eta_mev * 1e-3,
                nk=args.dssf_nk, spin_length=args.spin_length, regularization=args.regularization,
                interface_sublattices=(1, 2, 3),
            )
            error = ""
            try:
                # Integrate delta(epsilon_j-epsilon_i-omega) over omega analytically.
                result = surface_projected_dynamic_structure_factor(
                    spin_wave_structure_factor, q_reduced, energy_transfer,
                    state["unit_vectors_local"], h_local, args.j_pm,
                    q_perp_mode=args.q_perp_mode, n_q_perp=args.n_q_perp, **solver_kwargs,
                )
                kernel = contract_kernel(result["total"][:, None], weights, vertex)[0]
                if not np.isfinite(kernel).all() or np.min(kernel) < -1e-12:
                    raise ValueError("Non-finite or negative transport kernel")
            except ValueError as exc:
                if not (str(exc).startswith("Dynamically unstable spin-wave state")
                        or str(exc) == "A positive magnon mode has non-positive bosonic norm"):
                    raise
                error = str(exc)
                print(f"WARNING: phi={phi:g}, field={field:g}: {error}; marking gray", flush=True)
                kernel = np.full(vertex.shape, np.nan)
                result = {"max_imaginary_frequency_ratio": np.nan}
            phi_errors.append(error)
            phi_kernels.append(kernel)
            phi_maps.append(group_kernel(kernel[None], groups, len(points))[0])
            phi_ordered.append(state["unit_vectors_local"])
            phi_fields_local.append(h_local)
            phi_imaginary.append(result["max_imaginary_frequency_ratio"])
        kernels.append(phi_kernels)
        maps.append(phi_maps)
        ordered.append(phi_ordered)
        fields_local.append(phi_fields_local)
        imaginary.append(phi_imaginary)
        errors.append(phi_errors)
    with h5py.File(output, "w") as h5:
        for name, values in dict(phi_degrees=args.phi_degrees, mu_B_B_over_J_zz=fields, kf=kf, eigf_ev=energies,
                                 energy_transfer_ev=energy_transfer,
                                 q_slab=q, plotted_q_xy=points, pair_to_point=groups.reshape(vertex.shape),
                                 bz_boundary=boundary, eio_bond_weights=weights, transport_vertex=vertex,
                                 kernel=kernels, plotted_kernel=maps, ordered_spins_local=ordered,
                                 effective_field_local_over_J_zz=fields_local,
                                 max_imaginary_frequency_ratio=imaginary,
                                 q_perp_reduced_offsets=q_perp_samples_for_mode(args.q_perp_mode,args.n_q_perp)).items():
            h5.create_dataset(name, data=values)
        h5.create_dataset("stable", data=np.asarray(errors) == "")
        h5.create_dataset("instability_error", data=np.asarray(errors, dtype=object),
                          dtype=h5py.string_dtype("utf-8"))
        h5.attrs.update(unstable_policy="NaN kernel; gray labeled panel; continue scan", component="total", incoming_states="all", plot_mode=args.plot_mode, source_sha256_json=json.dumps(hashes),
                        config_json=json.dumps({k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}),
                        eio_model=args.eio_model, kernel_axes="phi, field, incoming_k, outgoing_k",
                        plotted_axes="phi, field, plotted_q", animation_axis=("mu_B_B_over_J_zz" if args.plot_mode == "movie" else "none"),
                        kernel_definition="sum_mu W_mu,ij S_mu(k_i-k_j,omega) * (1-cos(theta_ij))",
                        grouping="sum of ordered pair kernels at coincident folded q; no interpolation",
                        energy_convention="Eq. 24: omega integral with delta; omega_ij = epsilon_j-epsilon_i",
                        omega_integral="delta-function integration over the full real omega axis; unit Jacobian",
                        bragg_included=False, interface_exchange_J_squared=1.0,
                        momentum_axes="x || [-1,0,1], y || [1,-2,1], in inverse Angstrom")
    if args.plot_mode == "movie":
        figure, movie, _ = make_movie(np.asarray(maps), fields, points, boundary, args)
        try:
            movie.save(plot_path, writer=animation_writer(args.video_format, args.fps), dpi=args.dpi)
        finally:
            plt.close(figure)
    else:
        figure = make_side_by_side(np.asarray(maps), fields, points, boundary, args)
        try:
            figure.savefig(plot_path, bbox_inches="tight",
                           metadata={"CreationDate": None, "ModDate": None})
        finally:
            plt.close(figure)
    print(f"Wrote {output}\nWrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
