"""Compute sample-resolved DTO spin Fourier amplitudes at q = k' - k."""
import argparse

import h5py
import numpy as np
import pandas as pd

from config import (DATA_DIR, DTO_OUT_OF_PLANE_SNAPSHOTS, DTO_SNAPSHOTS,
                    EIO_DTO_LATTICE_RATIO, EIO_NK, FIELDS, N_ANGLES, N_SEEDS,
                    N_SNAPSHOTS, OUT_OF_PLANE_FIELDS,
                    OUT_OF_PLANE_N_SNAPSHOTS,
                    OUT_OF_PLANE_SITES_PER_SUBLATTICE,
                    OUT_OF_PLANE_SYSTEM_SIZE, OUT_OF_PLANE_TEMPERATURES,
                    OUT_OF_PLANE_WARMUP_SNAPSHOTS, SITES_PER_SUBLATTICE,
                    SYSTEM_SIZE, TEMPERATURE, WARMUP_SNAPSHOTS_BY_FIELD,
                    condition_tag, field_tag)


def prepare_interface_coordinates(
    snapshot_h5, system_size, sites_per_sublattice
):
    """Align DTO with EIO and identify the three interfacial kagome sublattices.

    Returns the transformed coordinates, global site indices for each DTO
    sublattice, and the two-dimensional triangular Bravais lattice.
    """
    coordinates = snapshot_h5["coords"][...]
    theta = -30 * np.pi / 180
    rotation = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    coordinates = coordinates @ rotation * 10 * EIO_DTO_LATTICE_RATIO

    # Infer the in-plane Bravais vectors from the triangular layer beneath the
    # interface; this reproduces the geometry convention of the old notebooks.
    triangular = coordinates[coordinates[:, 2] < 2]
    length = 2 * triangular[:, 0].max()
    lattice = np.array([
        [length, 0, 0],
        [length / 2, np.sqrt(3) * length / 2, 0],
        [0, 0, length],
    ]) / system_size

    # The first kagome layer contains three translated triangular sublattices.
    # Classify sites by testing integer coordinates in the Bravais basis.
    kagome = np.where((coordinates[:, 2] > 2) & (coordinates[:, 2] < 5))[0]
    kagome_coordinates = coordinates[kagome]
    sublattices = []
    for target in ([0, 0, 0], [-1, -1, 0], [0, 2, 0]):
        origin = kagome_coordinates[
            np.argmin(np.linalg.norm(kagome_coordinates - target, axis=1))
        ]
        selected = []
        for local_index, position in enumerate(kagome_coordinates):
            m = (position[1] - origin[1]) / lattice[1, 1]
            if np.isclose(m, np.round(m), atol=1e-3):
                n = (
                    position[0] - origin[0] - m * lattice[1, 0]
                ) / lattice[0, 0]
                if np.isclose(n, np.round(n), atol=1e-3):
                    selected.append(local_index)
        sublattices.append(kagome[np.asarray(selected)])

    if not all(len(indices) == sites_per_sublattice for indices in sublattices):
        counts = [len(indices) for indices in sublattices]
        raise ValueError(
            f"Expected three {sites_per_sublattice}-site sublattices; found {counts}"
        )
    if len(np.unique(np.concatenate(sublattices))) != 3 * sites_per_sublattice:
        raise ValueError("The identified kagome sublattices overlap")
    return coordinates, np.asarray(sublattices), lattice


def read_configurations(
    snapshot_h5, scan, field, temperature, angle, site_indices,
    warmup_snapshots, n_snapshots,
):
    """Read one condition into (seed, snapshot, site, spin component)."""
    configurations = np.empty(
        (N_SEEDS, n_snapshots - warmup_snapshots, len(site_indices), 3),
        dtype=float,
    )
    tag = field_tag(field)
    for seed in range(N_SEEDS):
        for snapshot_index, timestep in enumerate(
            range(warmup_snapshots, n_snapshots)
        ):
            if scan == "in-plane":
                path = (
                    f"H{tag}/phi{angle}/kT{temperature}/seed{seed}/"
                    f"tstep{timestep}/configs"
                )
            else:
                path = (
                    f"H{tag}/kT{temperature}/seed{seed}/"
                    f"tstep{timestep}/configs"
                )
            configurations[seed, snapshot_index] = snapshot_h5[path][site_indices]
    return configurations


def local_moment_summary(configurations, field, temperature):
    """Reproduce the archived mean magnitude of each site's thermal moment."""
    thermal_spin = np.mean(configurations, axis=(0, 1))
    seed_spin = np.mean(configurations, axis=1)
    spin_error = np.std(seed_spin, axis=0) / np.sqrt(len(seed_spin))

    moment_squared = np.sum(thermal_spin ** 2, axis=1)
    moment = np.mean(np.sqrt(moment_squared))
    moment_squared_error = np.sqrt(
        np.sum(4 * spin_error ** 2 * thermal_spin ** 2, axis=1)
    )
    moment_error = np.mean(moment_squared_error) / (2 * moment)
    return {
        "field": field,
        "temperature": temperature,
        "mean_local_moment_squared": np.mean(moment_squared),
        "mean_local_moment_squared_error": np.mean(moment_squared_error),
        "mean_local_moment": moment,
        "mean_local_moment_error": moment_error,
    }


def compute_condition(
    scan, field, temperature, snapshots_path, system_size,
    sites_per_sublattice, n_angles, n_snapshots, warmup_snapshots,
    overwrite=False,
):
    """Compute sample-resolved DTO spin amplitudes for one scan condition."""
    tag = condition_tag(scan, field, temperature)
    eio_path = DATA_DIR / f"surface_states_summary_nk{EIO_NK}.hdf5"
    output_h5 = DATA_DIR / f"spin_fourier_{tag}.hdf5"
    output_csv = DATA_DIR / f"structure_factor_{tag}.csv"
    if (output_h5.exists() or output_csv.exists()) and not overwrite:
        raise FileExistsError(f"Outputs for {tag} exist; pass --overwrite")

    with h5py.File(eio_path) as eio_h5:
        kf = np.squeeze(eio_h5["kf"][...])
    # q[ik, ikprime] is the momentum transferred from the initial EIO state k
    # to the final state kprime.
    q_vectors = kf[None, :, :] - kf[:, None, :]
    nkf = len(kf)
    n_used_snapshots = n_snapshots - warmup_snapshots

    mode = "w" if overwrite else "x"
    structure_factors = np.empty((n_angles, 3, nkf, nkf), dtype=float)
    moment = None
    with h5py.File(snapshots_path) as snapshots_h5, h5py.File(
        output_h5, mode
    ) as output:
        coordinates, sublattices, lattice = prepare_interface_coordinates(
            snapshots_h5, system_size, sites_per_sublattice
        )
        # The geometric phase depends only on q and site position, so compute it
        # once and reuse it for every field angle, seed, and snapshot.
        phases = [
            np.exp(1j * np.einsum(
                "ijm,am->ija", q_vectors, coordinates[indices]
            ))
            for indices in sublattices
        ]
        output.create_dataset("kf", data=kf)
        output.create_dataset("q_vectors", data=q_vectors)
        output.create_dataset("sublattice_indices", data=sublattices)
        output.create_dataset("interface_lattice", data=lattice)
        fourier_h5 = output.create_dataset(
            "S_q",
            shape=(n_angles, 3, nkf, nkf, N_SEEDS, n_used_snapshots, 3),
            dtype=np.complex128,
            chunks=(1, 1, nkf, nkf, 1, n_used_snapshots, 3),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        output.attrs.update(
            scan=scan,
            field=field,
            temperature=temperature,
            n_angles=n_angles,
            n_seeds=N_SEEDS,
            snapshots_used=n_used_snapshots,
            warmup_snapshots=warmup_snapshots,
            fourier_normalization="1/sqrt(number of sites in one sublattice)",
            momentum_convention="q = kprime - k",
        )

        # Preserve individual seeds and snapshots here. They must remain
        # separate until the coherent scattering amplitude has been squared.
        for angle in range(n_angles):
            input_angle = angle if scan == "in-plane" else None
            for sublattice, indices in enumerate(sublattices):
                configurations = read_configurations(
                    snapshots_h5, scan, field, temperature, input_angle, indices,
                    warmup_snapshots, n_snapshots,
                )
                # S_q[ik, ikprime, seed, snapshot, spin_component]
                # = sum_R exp(i q.r_R) S_R / sqrt(number of sites).
                fourier = np.einsum(
                    "ija,snam->ijsnm", phases[sublattice], configurations,
                    optimize=True,
                ) / np.sqrt(len(indices))
                # This scalar diagnostic is averaged over the Monte Carlo
                # ensemble; the complex S_q amplitudes remain sample resolved.
                structure_factor = np.mean(
                    np.sum(np.abs(fourier) ** 2, axis=-1), axis=(2, 3)
                )
                fourier_h5[angle, sublattice] = fourier
                structure_factors[angle, sublattice] = structure_factor
            if angle == 0 or (angle + 1) % 10 == 0:
                print(f"{tag}: orientation {angle + 1}/{n_angles}", flush=True)
        output.create_dataset("structure_factor", data=structure_factors)

        if scan == "out-of-plane":
            all_sites = np.arange(len(coordinates))
            configurations = read_configurations(
                snapshots_h5, scan, field, temperature, None, all_sites,
                warmup_snapshots, n_snapshots,
            )
            moment = local_moment_summary(configurations, field, temperature)

    # Export the compact ensemble-averaged structure factor for inspection;
    # the much larger sample-resolved tensor remains in HDF5.
    pairs_per_sublattice = nkf * nkf
    frame = pd.DataFrame({
        "field": field,
        "angle": np.repeat(np.arange(n_angles), 3 * pairs_per_sublattice),
        "sublattice": np.tile(
            np.repeat(np.arange(3), pairs_per_sublattice), n_angles
        ),
        "ik": np.tile(np.repeat(np.arange(nkf), nkf), n_angles * 3),
        "ikprime": np.tile(np.arange(nkf), n_angles * 3 * nkf),
        "qx": np.tile(q_vectors[..., 0].ravel(), n_angles * 3),
        "qy": np.tile(q_vectors[..., 1].ravel(), n_angles * 3),
        "qz": np.tile(q_vectors[..., 2].ravel(), n_angles * 3),
        "structure_factor": structure_factors.ravel(),
    })
    if scan == "out-of-plane":
        frame.insert(1, "temperature", temperature)
        frame = frame.drop(columns="angle")
    frame.to_csv(output_csv, index=False)
    print(f"Wrote {output_h5}")
    print(f"Wrote {output_csv}")
    return moment


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--scan", choices=("in-plane", "out-of-plane"), default="in-plane"
)
parser.add_argument("--fields", nargs="+", type=float)
parser.add_argument("--temperatures", nargs="+", type=float)
parser.add_argument("--snapshots", type=str)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

if args.scan == "in-plane":
    fields = args.fields or FIELDS
    temperatures = args.temperatures or (TEMPERATURE,)
    if tuple(temperatures) != (TEMPERATURE,):
        raise ValueError(f"The archived in-plane scan is available only at T={TEMPERATURE}")
    snapshots = args.snapshots or str(DTO_SNAPSHOTS)
    settings = (SYSTEM_SIZE, SITES_PER_SUBLATTICE, N_ANGLES, N_SNAPSHOTS)
else:
    fields = args.fields or OUT_OF_PLANE_FIELDS
    temperatures = args.temperatures or OUT_OF_PLANE_TEMPERATURES
    snapshots = args.snapshots or str(DTO_OUT_OF_PLANE_SNAPSHOTS)
    settings = (
        OUT_OF_PLANE_SYSTEM_SIZE,
        OUT_OF_PLANE_SITES_PER_SUBLATTICE,
        1,
        OUT_OF_PLANE_N_SNAPSHOTS,
    )

DATA_DIR.mkdir(exist_ok=True)
moments = []
for selected_field in fields:
    for selected_temperature in temperatures:
        warmup = (
            WARMUP_SNAPSHOTS_BY_FIELD[selected_field]
            if args.scan == "in-plane"
            else OUT_OF_PLANE_WARMUP_SNAPSHOTS
        )
        result = compute_condition(
            args.scan, selected_field, selected_temperature, snapshots,
            *settings, warmup, overwrite=args.overwrite,
        )
        if result is not None:
            moments.append(result)

if moments:
    moment_output = DATA_DIR / "dto_magnetization_out_of_plane.csv"
    pd.DataFrame(moments).to_csv(moment_output, index=False)
    print(f"Wrote {moment_output}")
