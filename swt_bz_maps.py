"""Shared computation and plotting for interfacial SWT BZ maps."""
import argparse
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from config import TTO_CUBIC_LATTICE_ANGSTROM

from tto_spin_wave import (
    local_zeeman_components,
    relax_ordered_state,
    spin_wave_static_structure_factor,
    spin_wave_structure_factor,
    zeeman_minimizing_spins,
)
from tto_surface_projection import (
    Q_PERP_MODES,
    Q_PERP_REDUCED_DIRECTION,
    q_perp_samples_for_mode,
    surface_projected_dynamic_structure_factor,
    surface_projected_static_structure_factor,
    validate_q_perp_sampling,
)
from tto_structure_factor import FCC_PRIMITIVES, eio_to_tto_reduced_momentum


COMPONENTS = ("one_magnon", "two_magnon", "bragg", "total")
SUBLATTICES = (1, 2, 3)
FIELD_RATIOS = np.asarray((0.25, 0.45, 1.5))
FIELD_PHIS_DEGREES = np.asarray((0.1, 30.1))
FIELD_PHI_FILE_LABELS = ("0", "30")
FIELD_THETA_DEGREES = 90.0

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
})


def kagome_bz_grid(n_q, lattice_constant_angstrom):
    """Return the hexagonal 2D BZ perpendicular to the cubic [111] axis."""
    if n_q < 3 or n_q % 2 == 0:
        raise ValueError("n_q must be an odd integer of at least 3")
    lattice_constant_angstrom = float(lattice_constant_angstrom)
    if not np.isfinite(lattice_constant_angstrom) or lattice_constant_angstrom <= 0.0:
        raise ValueError("lattice_constant_angstrom must be finite and positive")

    direct = 0.5 * lattice_constant_angstrom * FCC_PRIMITIVES
    in_plane_1 = direct[1] - direct[0]
    in_plane_2 = direct[2] - direct[0]
    plane_cell = np.stack((
        in_plane_1,
        in_plane_2,
        np.cross(in_plane_1, in_plane_2),
    ))
    reciprocal = 2.0 * np.pi * np.linalg.inv(plane_cell)
    x_axis = in_plane_1 / np.linalg.norm(in_plane_1)
    y_axis = in_plane_2 - np.dot(in_plane_2, x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.stack((x_axis, y_axis, np.cross(x_axis, y_axis)))
    reciprocal_slab = rotation @ reciprocal
    hexagon_radius = np.linalg.norm(reciprocal_slab[:2, 0]) / np.sqrt(3.0)

    edges = np.linspace(-hexagon_radius, hexagon_radius, n_q + 1)
    axis = 0.5 * (edges[:-1] + edges[1:])
    q_x, q_y = np.meshgrid(axis, axis, indexing="xy")
    inside = (
        (np.abs(q_y) <= np.sqrt(3.0) * hexagon_radius / 2.0)
        & (np.sqrt(3.0) * np.abs(q_x) + np.abs(q_y)
           <= np.sqrt(3.0) * hexagon_radius)
    )
    q_slab = np.stack((q_x, q_y, np.zeros_like(q_x)), axis=-1)
    q_reduced = eio_to_tto_reduced_momentum(q_slab, lattice_constant_angstrom)
    vertices = hexagon_radius * np.asarray([
        [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0],
        [-0.5, np.sqrt(3.0) / 2.0], [-1.0, 0.0],
        [-0.5, -np.sqrt(3.0) / 2.0], [0.5, -np.sqrt(3.0) / 2.0],
    ])
    return axis, edges, q_reduced, inside, vertices


def prepare_ordered_state(args, field_ratio, field_phi_degrees):
    """Relax the four-sublattice classical state for one fixed field."""
    h_z = local_zeeman_components(
        field_ratio,
        theta_degrees=FIELD_THETA_DEGREES,
        phi_degrees=field_phi_degrees,
    )
    initial_spins = zeeman_minimizing_spins(h_z, spin_length=args.spin_length)
    relaxed = relax_ordered_state(
        initial_spins,
        h_z,
        args.j_pm,
        j_pmpm_ratio=args.j_pmpm,
        spin_length=args.spin_length,
        random_starts=args.random_starts,
        stability_tilt=args.stability_tilt,
    )
    return h_z, initial_spins, relaxed


def shared_norm(values, percentile):
    """Return a common linear intensity scale for all displayed panels."""
    values = np.asarray(values, dtype=float)
    if np.any(np.isinf(values)):
        raise ValueError("Structure-factor values cannot contain infinities")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Structure-factor values must be finite and nonempty")
    if np.any(finite < -1e-12):
        raise ValueError("Structure-factor values must be finite and nonnegative")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("vmax_percentile must be in (0, 100]")
    vmax = float(np.percentile(finite, percentile))
    if vmax <= 0.0:
        vmax = 1.0
    return mpl.colors.Normalize(vmin=0.0, vmax=vmax, clip=True)


def projection_title(args):
    """Return a compact description of the displayed perpendicular momentum."""
    if args.q_perp_mode == "slice":
        return r"$q_\perp=0$ slice"
    return rf"$q_\perp$ average ($N_\perp={args.n_q_perp}$)"


def plot_field_grid(
    values,
    q_edges,
    bz_vertices,
    mode,
    args,
    field_phi_degrees,
    output_path,
):
    """Write one shared-scale 3-field by 3-sublattice vector figure."""
    norm = shared_norm(values, args.vmax_percentile)
    figure, axes = plt.subplots(
        len(FIELD_RATIOS), len(SUBLATTICES),
        figsize=(9.0, 8.4), sharex=True, sharey=True,
        constrained_layout=True,
    )
    image = None
    boundary = np.vstack((bz_vertices, bz_vertices[0]))
    ticks = np.asarray((q_edges[0], 0.0, q_edges[-1]))
    for field_index, field_ratio in enumerate(FIELD_RATIOS):
        for sublattice_index, mu in enumerate(SUBLATTICES):
            axis = axes[field_index, sublattice_index]
            image = axis.pcolormesh(
                q_edges,
                q_edges,
                values[field_index, sublattice_index],
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
                    rf"$h/J_{{zz}}={field_ratio:g}$" + "\n"
                    + r"$q_{[1\bar{2}1]}$ ($\mathrm{\AA}^{-1}$)"
                )

    if mode == "dynamic":
        observable = rf"$S_\mu(\mathbf{{q}},\omega_0)$, $\omega_0={args.omega_mev:g}$ meV"
        colorbar_label = r"$S_\mu(\mathbf{q},\omega_0)$ (eV$^{-1}$)"
    else:
        observable = r"$S_\mu(\mathbf{q})=\int d\omega\,S_\mu(\mathbf{q},\omega)$"
        colorbar_label = r"$S_\mu(\mathbf{q})$"
    parameters = (
        rf"$(111)$ kagome-plane BZ, $\theta={FIELD_THETA_DEGREES:g}^\circ$, "
        rf"$\phi={field_phi_degrees:g}^\circ$, "
        rf"$J_{{zz}}={1e3 * args.j_zz_ev:g}$ meV, "
        rf"$J_{{\pm}}/J_{{zz}}={args.j_pm:g}$, "
        rf"$J_{{\pm\pm}}/J_{{zz}}={args.j_pmpm:g}$, "
        + projection_title(args)
    )
    figure.suptitle(observable + "\n" + parameters, fontsize=10)
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.84, pad=0.025)
    colorbar.set_label(colorbar_label)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    if args.vmax_percentile < 100.0:
        colorbar.ax.set_title(
            f"clipped at {args.vmax_percentile:g}%", fontsize=7, pad=6
        )
    figure.savefig(
        output_path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def plot_sublattice_sum(
    values,
    q_edges,
    bz_vertices,
    mode,
    args,
    field_phi_degrees,
    output_path,
):
    """Write the mu=1,2,3 sum as one shared-scale row of field maps."""
    summed_values = np.sum(values, axis=1)
    norm = shared_norm(summed_values, args.vmax_percentile)
    figure, axes = plt.subplots(
        1, len(FIELD_RATIOS), figsize=(9.0, 3.25),
        sharex=True, sharey=True, constrained_layout=True,
    )
    image = None
    boundary = np.vstack((bz_vertices, bz_vertices[0]))
    ticks = np.asarray((q_edges[0], 0.0, q_edges[-1]))
    for field_index, field_ratio in enumerate(FIELD_RATIOS):
        axis = axes[field_index]
        image = axis.pcolormesh(
            q_edges,
            q_edges,
            summed_values[field_index],
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
        axis.set_title(rf"$h/J_{{zz}}={field_ratio:g}$")
        axis.set_xlabel(r"$q_{[\bar{1}01]}$ ($\mathrm{\AA}^{-1}$)")
    axes[0].set_ylabel(r"$q_{[1\bar{2}1]}$ ($\mathrm{\AA}^{-1}$)")

    if mode == "dynamic":
        observable = (
            rf"$\sum_{{\mu=1}}^3 S_\mu(\mathbf{{q}},\omega_0)$, "
            rf"$\omega_0={args.omega_mev:g}$ meV"
        )
        colorbar_label = (
            r"$\sum_{\mu=1}^3 S_\mu(\mathbf{q},\omega_0)$ (eV$^{-1}$)"
        )
    else:
        observable = (
            r"$\sum_{\mu=1}^3 S_\mu(\mathbf{q})="
            r"\int d\omega\,\sum_{\mu=1}^3 S_\mu(\mathbf{q},\omega)$"
        )
        colorbar_label = r"$\sum_{\mu=1}^3 S_\mu(\mathbf{q})$"
    parameters = (
        rf"$(111)$ kagome-plane BZ, $\theta={FIELD_THETA_DEGREES:g}^\circ$, "
        rf"$\phi={field_phi_degrees:g}^\circ$, "
        rf"$J_{{zz}}={1e3 * args.j_zz_ev:g}$ meV, "
        rf"$J_{{\pm}}/J_{{zz}}={args.j_pm:g}$, "
        rf"$J_{{\pm\pm}}/J_{{zz}}={args.j_pmpm:g}$, "
        + projection_title(args)
    )
    figure.suptitle(observable + "\n" + parameters, fontsize=10)
    colorbar = figure.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.84, pad=0.025
    )
    colorbar.set_label(colorbar_label)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    if args.vmax_percentile < 100.0:
        colorbar.ax.set_title(
            f"clipped at {args.vmax_percentile:g}%", fontsize=7, pad=6
        )
    figure.savefig(
        output_path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def write_hdf5(
    output_path,
    mode,
    q_axis,
    q_grid,
    inside_bz,
    result,
    state_data,
    max_imaginary_frequency_ratio,
    args,
):
    """Save the plotted maps, all physical components, and run metadata."""
    with h5py.File(output_path, "w") as output:
        output.create_dataset("q_x_inv_angstrom", data=q_axis)
        output.create_dataset("q_y_inv_angstrom", data=q_axis)
        output.create_dataset("q_reduced", data=q_grid, compression="gzip")
        output.create_dataset("inside_2d_bz", data=inside_bz, compression="gzip")
        output.create_dataset(
            "q_perp_reduced_offsets",
            data=q_perp_samples_for_mode(args.q_perp_mode, args.n_q_perp),
        )
        output.create_dataset("field_phi_degrees", data=FIELD_PHIS_DEGREES)
        output.create_dataset("h_over_J_zz", data=FIELD_RATIOS)
        output.create_dataset("sublattice_indices", data=SUBLATTICES)
        for name, values in state_data.items():
            output.create_dataset(name, data=values, compression="gzip")
        output.create_dataset(
            "max_imaginary_frequency_ratio",
            data=max_imaginary_frequency_ratio,
        )
        for component in COMPONENTS:
            output.create_dataset(component, data=result[component], compression="gzip")
            output.create_dataset(
                f"{component}_mu_sum",
                data=np.sum(result[component], axis=2),
                compression="gzip",
            )
        output.attrs.update(
            theory="linear spin wave, Sec. 4.2, Eqs. (88)-(100)",
            observable=(
                "S_mu(q, omega) at fixed omega" if mode == "dynamic"
                else "integral d omega S_mu(q, omega)"
            ),
            first_bz="hexagonal 2D BZ perpendicular to cubic [111]",
            momentum_axes="x || [-1,0,1], y || [1,-2,1], in inverse Angstrom",
            array_axes="field_phi, h_over_J_zz, sublattice, q_y, q_x",
            mu_sum_axes="field_phi, h_over_J_zz, q_y, q_x",
            mu_sum_definition="sum over sublattice indices 1, 2, 3",
            plotted_component=args.component,
            field_theta_degrees=FIELD_THETA_DEGREES,
            J_zz_ev=args.j_zz_ev,
            J_pm_over_J_zz=args.j_pm,
            J_pmpm_over_J_zz=args.j_pmpm,
            tto_cubic_lattice_angstrom=args.lattice_constant,
            dssf_nk=args.dssf_nk,
            spin_length=args.spin_length,
            regularization=args.regularization,
            stability_tilt=args.stability_tilt,
            random_starts=args.random_starts,
            n_q=args.n_q,
            vmax_percentile=args.vmax_percentile,
            q_perp_mode=args.q_perp_mode,
            n_q_perp=args.n_q_perp,
            q_perp_reduced_direction=Q_PERP_REDUCED_DIRECTION,
            surface_projection=(
                "q_perp=0 slice" if args.q_perp_mode == "slice" else
                "uniform normalized average over one [111] reciprocal period"
            ),
        )
        if mode == "dynamic":
            output.attrs["omega_mev"] = args.omega_mev
            output.attrs["lorentzian_eta_mev"] = args.eta_mev


def compute_and_plot(mode, args):
    """Compute the fixed scan and write per-sublattice and summed PDFs."""
    if mode not in ("dynamic", "static"):
        raise ValueError(f"Unknown map mode: {mode}")
    if args.j_zz_ev <= 0.0:
        raise ValueError("j_zz_ev must be positive")
    if args.dssf_nk <= 0:
        raise ValueError("dssf_nk must be positive")
    if args.component not in COMPONENTS:
        raise ValueError(f"Unknown component: {args.component}")
    validate_q_perp_sampling(args.q_perp_mode, args.n_q_perp)
    if mode == "dynamic" and (args.omega_mev < 0.0 or args.eta_mev <= 0.0):
        raise ValueError("omega_mev must be nonnegative and eta_mev positive")

    run_directory = args.output_root.expanduser().resolve()
    data_directory = run_directory / "data"
    figure_directory = run_directory / "figures"
    data_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    stem = "swt_ssf" if mode == "dynamic" else "swt_sssf"
    data_path = data_directory / f"{stem}.hdf5"
    figure_paths = tuple(
        figure_directory / f"{stem}_phi_{label}.pdf"
        for label in FIELD_PHI_FILE_LABELS
    )
    sum_figure_paths = tuple(
        figure_directory / f"{stem}_mu_sum_phi_{label}.pdf"
        for label in FIELD_PHI_FILE_LABELS
    )
    existing = [
        path for path in (data_path,) + figure_paths + sum_figure_paths
        if path.exists()
    ]
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
    map_shape = scan_shape + (len(SUBLATTICES), args.n_q, args.n_q)
    mapped_result = {
        component: np.full(map_shape, np.nan) for component in COMPONENTS
    }
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
    omega_ev = None
    if mode == "dynamic":
        omega_ev = np.full(len(q_points), 1e-3 * args.omega_mev)

    for phi_index, field_phi_degrees in enumerate(FIELD_PHIS_DEGREES):
        for field_index, field_ratio in enumerate(FIELD_RATIOS):
            condition = (phi_index, field_index)
            print(
                "Relaxing ordered state for "
                f"phi={field_phi_degrees:g} deg, h/Jzz={field_ratio:g}",
                flush=True,
            )
            h_z, initial_spins, relaxed = prepare_ordered_state(
                args, field_ratio, field_phi_degrees
            )
            print(
                f"Evaluating {len(q_points)} points for sublattices 1, 2, 3",
                flush=True,
            )
            if mode == "dynamic":
                result = surface_projected_dynamic_structure_factor(
                    spin_wave_structure_factor,
                    q_points,
                    omega_ev,
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
                    q_perp_mode=args.q_perp_mode,
                    n_q_perp=args.n_q_perp,
                )
            else:
                result = surface_projected_static_structure_factor(
                    spin_wave_static_structure_factor,
                    q_points,
                    relaxed["unit_vectors_local"],
                    h_z,
                    args.j_pm,
                    j_pmpm_ratio=args.j_pmpm,
                    nk=args.dssf_nk,
                    spin_length=args.spin_length,
                    regularization=args.regularization,
                    sublattices=SUBLATTICES,
                    q_perp_mode=args.q_perp_mode,
                    n_q_perp=args.n_q_perp,
                )

            for component in COMPONENTS:
                mapped_result[component][condition][:, inside_bz] = result[component]
            max_imaginary_frequency_ratio[condition] = (
                result["max_imaginary_frequency_ratio"]
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

    write_hdf5(
        data_path, mode, q_axis, q_grid, inside_bz, mapped_result,
        state_data, max_imaginary_frequency_ratio, args
    )
    for phi_index, (field_phi_degrees, figure_path) in enumerate(zip(
        FIELD_PHIS_DEGREES, figure_paths
    )):
        plot_field_grid(
            mapped_result[args.component][phi_index], q_edges, bz_vertices,
            mode, args, field_phi_degrees, figure_path
        )
        plot_sublattice_sum(
            mapped_result[args.component][phi_index], q_edges, bz_vertices,
            mode, args, field_phi_degrees, sum_figure_paths[phi_index]
        )
    print(f"Data: {data_path}")
    for figure_path in figure_paths + sum_figure_paths:
        print(f"Figure: {figure_path}")
    return data_path, figure_paths, sum_figure_paths


def build_parser(mode, description):
    """Build the command-line interface shared by both map routines."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    parser.add_argument("--j-pm", type=float, default=0.205)
    parser.add_argument("--j-pmpm", type=float, default=0.0)
    parser.add_argument("--j-zz-ev", type=float, default=0.00019)
    parser.add_argument("--n-q", type=int, default=101)
    parser.add_argument(
        "--lattice-constant", type=float, default=TTO_CUBIC_LATTICE_ANGSTROM
    )
    parser.add_argument("--dssf-nk", type=int, default=8)
    parser.add_argument("--q-perp-mode", choices=Q_PERP_MODES, default="slice")
    parser.add_argument("--n-q-perp", type=int, default=8)
    parser.add_argument("--spin-length", type=float, default=0.5)
    parser.add_argument("--regularization", type=float, default=1e-8)
    parser.add_argument("--stability-tilt", type=float, default=0.01)
    parser.add_argument("--random-starts", type=int, default=12)
    parser.add_argument("--component", choices=COMPONENTS, default="total")
    parser.add_argument("--vmax-percentile", type=float, default=100.0)
    parser.add_argument("--overwrite", action="store_true")
    if mode == "dynamic":
        parser.add_argument("--omega-mev", type=float, default=0.05)
        parser.add_argument("--eta-mev", type=float, default=0.02)
    return parser
