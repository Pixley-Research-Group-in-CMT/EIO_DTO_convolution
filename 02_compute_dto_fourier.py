"""Compute sample-resolved DTO spin Fourier amplitudes at q = k' - k."""
import argparse

import h5py
import numpy as np
import pandas as pd

from config import (DATA_DIR, DTO_SNAPSHOTS, EIO_DTO_LATTICE_RATIO, EIO_NK,
                    FIELDS, N_ANGLES, N_SEEDS, N_SNAPSHOTS, SITES_PER_SUBLATTICE,
                    SYSTEM_SIZE, TEMPERATURE, WARMUP_SNAPSHOTS_BY_FIELD,
                    field_tag)


def prepare_interface_coordinates(snapshot_h5):
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
    ]) / SYSTEM_SIZE

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

    if not all(len(indices) == SITES_PER_SUBLATTICE for indices in sublattices):
        raise ValueError("Failed to identify three 64-site kagome sublattices")
    if len(np.unique(np.concatenate(sublattices))) != 3 * SITES_PER_SUBLATTICE:
        raise ValueError("The identified kagome sublattices overlap")
    return coordinates, np.asarray(sublattices), lattice


def compute_field(field, snapshots_path, overwrite=False):
    """Compute sample-resolved DTO spin amplitudes for one field magnitude."""
    tag = field_tag(field)
    eio_path = DATA_DIR / f"surface_states_summary_nk{EIO_NK}.hdf5"
    output_h5 = DATA_DIR / f"spin_fourier_h{tag}.hdf5"
    output_csv = DATA_DIR / f"structure_factor_h{tag}.csv"
    if (output_h5.exists() or output_csv.exists()) and not overwrite:
        raise FileExistsError(f"Outputs for field {tag} exist; pass --overwrite")

    with h5py.File(eio_path) as eio_h5:
        kf = np.squeeze(eio_h5["kf"][...])
    # q[ik, ikprime] is the momentum transferred from the initial EIO state k
    # to the final state kprime.
    q_vectors = kf[None, :, :] - kf[:, None, :]
    nkf = len(kf)
    warmup_snapshots = WARMUP_SNAPSHOTS_BY_FIELD[field]
    n_used_snapshots = N_SNAPSHOTS - warmup_snapshots

    mode = "w" if overwrite else "x"
    structure_factors = np.empty((N_ANGLES, 3, nkf, nkf), dtype=float)
    with h5py.File(snapshots_path) as snapshots_h5, h5py.File(output_h5, mode) as output:
        coordinates, sublattices, lattice = prepare_interface_coordinates(snapshots_h5)
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
            shape=(N_ANGLES, 3, nkf, nkf, N_SEEDS, n_used_snapshots, 3),
            dtype=np.complex128,
            chunks=(1, 1, nkf, nkf, 1, n_used_snapshots, 3),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        output.attrs.update(
            field=field,
            temperature=TEMPERATURE,
            n_angles=N_ANGLES,
            n_seeds=N_SEEDS,
            snapshots_used=n_used_snapshots,
            warmup_snapshots=warmup_snapshots,
            fourier_normalization="1/sqrt(number of sites in one sublattice)",
            momentum_convention="q = kprime - k",
        )

        # Preserve individual seeds and snapshots here. They must remain
        # separate until the coherent scattering amplitude has been squared.
        for angle in range(N_ANGLES):
            for sublattice, indices in enumerate(sublattices):
                configurations = np.empty(
                    (N_SEEDS, n_used_snapshots, len(indices), 3), dtype=float
                )
                for seed in range(N_SEEDS):
                    for snapshot_index, timestep in enumerate(
                        range(warmup_snapshots, N_SNAPSHOTS)
                    ):
                        path = (
                            f"H{tag}/phi{angle}/kT{TEMPERATURE}/seed{seed}/"
                            f"tstep{timestep}/configs"
                        )
                        configurations[seed, snapshot_index] = snapshots_h5[path][indices]

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
                print(f"field {tag}: angle {angle + 1}/{N_ANGLES}", flush=True)
        output.create_dataset("structure_factor", data=structure_factors)

    # Export the compact ensemble-averaged structure factor for inspection;
    # the much larger sample-resolved tensor remains in HDF5.
    pairs_per_sublattice = nkf * nkf
    pd.DataFrame({
        "field": field,
        "angle": np.repeat(np.arange(N_ANGLES), 3 * pairs_per_sublattice),
        "sublattice": np.tile(
            np.repeat(np.arange(3), pairs_per_sublattice), N_ANGLES
        ),
        "ik": np.tile(np.repeat(np.arange(nkf), nkf), N_ANGLES * 3),
        "ikprime": np.tile(np.arange(nkf), N_ANGLES * 3 * nkf),
        "qx": np.tile(q_vectors[..., 0].ravel(), N_ANGLES * 3),
        "qy": np.tile(q_vectors[..., 1].ravel(), N_ANGLES * 3),
        "qz": np.tile(q_vectors[..., 2].ravel(), N_ANGLES * 3),
        "structure_factor": structure_factors.ravel(),
    }).to_csv(output_csv, index=False)
    print(f"Wrote {output_h5}")
    print(f"Wrote {output_csv}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fields", nargs="+", type=float, default=FIELDS)
parser.add_argument("--snapshots", type=str, default=str(DTO_SNAPSHOTS))
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()
DATA_DIR.mkdir(exist_ok=True)
for selected_field in args.fields:
    compute_field(selected_field, args.snapshots, overwrite=args.overwrite)
