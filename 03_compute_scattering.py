"""Contract DTO spin amplitudes with EIO surface spinors."""
import argparse

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from config import (ANGLE_SMOOTHING_SIGMA, DATA_DIR, EIO_NK, FIELDS,
                    ORBITAL_HOPPINGS, SITES_PER_SUBLATTICE, T2_NORMALIZATION,
                    field_tag)


PAULI = np.array([
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
    [[1, 0], [0, -1]],
])
# Each DTO moment couples through two interface bonds to neighboring EIO
# sublattices. BOND_VECTORS supplies the corresponding phase displacement.
EIO_NEIGHBORS = {0: (1, 2), 1: (1, 3), 2: (2, 3)}

BOND_DISTANCE = 7.26198663999854 / (2 * np.sqrt(3))
COS_30 = np.cos(np.pi / 6)
SIN_30 = np.sin(np.pi / 6)
BOND_VECTORS = BOND_DISTANCE * np.array([
    [[0, 1, 0], [-COS_30, -SIN_30, 0]],
    [[COS_30, -SIN_30, 0], [0, 1, 0]],
    [[-COS_30, -SIN_30, 0], [COS_30, -SIN_30, 0]],
])


def compute_field(field, overwrite=False):
    """Build and ensemble-average |T(k,kprime)|^2 for one field magnitude."""
    tag = field_tag(field)
    eio_path = DATA_DIR / f"surface_states_summary_nk{EIO_NK}.hdf5"
    spin_path = DATA_DIR / f"spin_fourier_h{tag}.hdf5"
    output_h5 = DATA_DIR / f"scattering_h{tag}.hdf5"
    output_csv = DATA_DIR / f"scattering_h{tag}.csv"
    if (output_h5.exists() or output_csv.exists()) and not overwrite:
        raise FileExistsError(f"Outputs for field {tag} exist; pass --overwrite")

    orbital_hoppings = np.asarray(ORBITAL_HOPPINGS) ** 2

    with h5py.File(eio_path) as eio_h5, h5py.File(spin_path) as spin_h5:
        kf = np.squeeze(eio_h5["kf"][...])
        projections = {
            sublattice: eio_h5[f"proj_sl{sublattice}"][...]
            for sublattice in (1, 2, 3)
        }
        q_vectors = spin_h5["q_vectors"][...]
        spin_fourier = spin_h5["S_q"]
        n_angles, _, nkf, _, n_seeds, n_snapshots, _ = spin_fourier.shape
        expected_q = kf[None, :, :] - kf[:, None, :]
        if not np.allclose(q_vectors, expected_q):
            raise ValueError("EIO momenta do not match the DTO Fourier file")

        # t2_raw[angle, ik, ikprime] is the final sample-averaged scattering
        # strength. The inner amplitude keeps seed, snapshot, and spin indices.
        t2_raw = np.empty((n_angles, nkf, nkf), dtype=float)
        for angle in range(n_angles):
            for ik in range(nkf):
                amplitude = np.zeros(
                    (nkf, n_seeds, n_snapshots, 2, 2), dtype=complex
                )
                # Sum all sublattices, bonds, and orbitals at the amplitude
                # level. This retains their interference before taking |T|^2.
                for dto_sublattice in range(3):
                    spin = spin_fourier[angle, dto_sublattice, ik]
                    # Contract the three Cartesian spin components with the
                    # Pauli matrices, leaving incoming/outgoing spin indices.
                    spin_sigma = np.einsum(
                        "ksnm,mab->ksnab", spin, PAULI, optimize=True
                    )
                    for bond, eio_sublattice in enumerate(
                        EIO_NEIGHBORS[dto_sublattice]
                    ):
                        projection = projections[eio_sublattice]
                        phase = np.exp(1j * np.einsum(
                            "km,m->k", q_vectors[ik],
                            BOND_VECTORS[dto_sublattice][bond],
                        ))
                        amplitude += np.einsum(
                            "o,k,oa,kob,ksnab->ksnab",
                            orbital_hoppings,
                            phase,
                            projection[ik].conj(),
                            projection,
                            spin_sigma,
                            optimize=True,
                        )
                # Only after the coherent sum do we trace over spin and average
                # over statistically independent seeds and retained snapshots.
                amplitude_squared = np.sum(np.abs(amplitude) ** 2, axis=(-1, -2))
                t2_raw[angle, ik] = np.mean(
                    amplitude_squared, axis=(1, 2)
                ) / T2_NORMALIZATION
            if angle == 0 or (angle + 1) % 10 == 0:
                print(f"field {tag}: angle {angle + 1}/{n_angles}", flush=True)

    # Preserve both the raw result and the notebook's one-angle Gaussian
    # smoothing so collaborators can distinguish data from presentation.
    t2_smoothed = gaussian_filter1d(
        t2_raw, sigma=ANGLE_SMOOTHING_SIGMA, axis=0
    )
    mode = "w" if overwrite else "x"
    with h5py.File(output_h5, mode) as output:
        output.create_dataset("kf", data=kf)
        output.create_dataset("q_vectors", data=q_vectors)
        output.create_dataset("t2_raw", data=t2_raw)
        output.create_dataset("t2_smoothed", data=t2_smoothed)
        output.attrs.update(
            field=field,
            sites_per_dto_sublattice=SITES_PER_SUBLATTICE,
            historical_normalization=T2_NORMALIZATION,
            angular_smoothing_sigma=ANGLE_SMOOTHING_SIGMA,
            smoothing_mode="scipy.ndimage.gaussian_filter1d default reflect mode",
        )

    # Flatten the momentum-pair tensor into a human-readable companion CSV.
    pairs_per_angle = nkf * nkf
    pd.DataFrame({
        "field": field,
        "angle": np.repeat(np.arange(n_angles), pairs_per_angle),
        "ik": np.tile(np.repeat(np.arange(nkf), nkf), n_angles),
        "ikprime": np.tile(np.arange(nkf), n_angles * nkf),
        "qx": np.tile(q_vectors[..., 0].ravel(), n_angles),
        "qy": np.tile(q_vectors[..., 1].ravel(), n_angles),
        "qz": np.tile(q_vectors[..., 2].ravel(), n_angles),
        "t2_raw": t2_raw.ravel(),
        "t2_smoothed": t2_smoothed.ravel(),
    }).to_csv(output_csv, index=False)
    print(f"Wrote {output_h5}")
    print(f"Wrote {output_csv}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fields", nargs="+", type=float, default=FIELDS)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()
DATA_DIR.mkdir(exist_ok=True)
for selected_field in args.fields:
    compute_field(selected_field, overwrite=args.overwrite)
