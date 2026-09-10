"""Contract DTO spin amplitudes with EIO surface spinors."""
import argparse

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from config import (ANGLE_SMOOTHING_SIGMA, DATA_DIR, EIO_NK, FIELDS,
                    ORBITAL_HOPPINGS, SITES_PER_SUBLATTICE, T2_NORMALIZATION,
                    field_tag)


def interface_bond_vectors():
    lattice_constant = 7.26198663999854
    distance = lattice_constant / (2 * np.sqrt(3))
    cos30 = np.cos(np.pi / 6)
    sin30 = np.sin(np.pi / 6)
    return {
        0: ([0, distance, 0], [-distance * cos30, -distance * sin30, 0]),
        1: ([distance * cos30, -distance * sin30, 0], [0, distance, 0]),
        2: ([-distance * cos30, -distance * sin30, 0],
            [distance * cos30, -distance * sin30, 0]),
    }


def compute_field(field, overwrite=False):
    tag = field_tag(field)
    eio_path = DATA_DIR / f"surface_states_summary_nk{EIO_NK}.hdf5"
    spin_path = DATA_DIR / f"spin_fourier_h{tag}.hdf5"
    output_h5 = DATA_DIR / f"scattering_h{tag}.hdf5"
    output_csv = DATA_DIR / f"scattering_h{tag}.csv"
    if (output_h5.exists() or output_csv.exists()) and not overwrite:
        raise FileExistsError(f"Outputs for field {tag} exist; pass --overwrite")

    pauli = np.array([
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ])
    neighbors = {0: (1, 2), 1: (1, 3), 2: (2, 3)}
    bond_vectors = interface_bond_vectors()
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

        t2_raw = np.empty((n_angles, nkf, nkf), dtype=float)
        for angle in range(n_angles):
            for ik in range(nkf):
                amplitude = np.zeros(
                    (nkf, n_seeds, n_snapshots, 2, 2), dtype=complex
                )
                for dto_sublattice in range(3):
                    spin = spin_fourier[angle, dto_sublattice, ik]
                    spin_sigma = np.einsum(
                        "ksnm,mab->ksnab", spin, pauli, optimize=True
                    )
                    for bond, eio_sublattice in enumerate(
                        neighbors[dto_sublattice]
                    ):
                        projection = projections[eio_sublattice]
                        phase = np.exp(1j * np.einsum(
                            "km,m->k", q_vectors[ik],
                            bond_vectors[dto_sublattice][bond],
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
                amplitude_squared = np.sum(abs(amplitude) ** 2, axis=(-1, -2))
                t2_raw[angle, ik] = np.mean(
                    amplitude_squared, axis=(1, 2)
                ) / T2_NORMALIZATION
            print(f"field {tag}: angle {angle + 1}/{n_angles}", flush=True)

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

    rows = []
    for angle in range(n_angles):
        rows.append(pd.DataFrame({
            "field": field,
            "angle": angle,
            "ik": np.repeat(np.arange(nkf), nkf),
            "ikprime": np.tile(np.arange(nkf), nkf),
            "qx": q_vectors[..., 0].ravel(),
            "qy": q_vectors[..., 1].ravel(),
            "qz": q_vectors[..., 2].ravel(),
            "t2_raw": t2_raw[angle].ravel(),
            "t2_smoothed": t2_smoothed[angle].ravel(),
        }))
    pd.concat(rows, ignore_index=True).to_csv(output_csv, index=False)
    print(f"Wrote {output_h5}")
    print(f"Wrote {output_csv}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fields", nargs="+", type=float, default=FIELDS)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()
DATA_DIR.mkdir(exist_ok=True)
for selected_field in args.fields:
    compute_field(selected_field, overwrite=args.overwrite)
