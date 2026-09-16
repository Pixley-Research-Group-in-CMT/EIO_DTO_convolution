"""Contract DTO spin amplitudes with EIO surface spinors."""
import argparse

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from config import (ANGLE_SMOOTHING_SIGMA, DATA_DIR, EIO_NK, FIELDS,
                    ORBITAL_HOPPINGS, OUT_OF_PLANE_FIELDS,
                    OUT_OF_PLANE_SITES_PER_SUBLATTICE,
                    OUT_OF_PLANE_TEMPERATURES, SITES_PER_SUBLATTICE,
                    T2_NORMALIZATION, TEMPERATURE, condition_tag)


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


def compute_condition(
    scan, field, temperature, sites_per_sublattice, overwrite=False
):
    """Build and ensemble-average |T(k,kprime)|^2 for one scan condition."""
    tag = condition_tag(scan, field, temperature)
    eio_path = DATA_DIR / f"surface_states_summary_nk{EIO_NK}.hdf5"
    spin_path = DATA_DIR / f"spin_fourier_{tag}.hdf5"
    output_h5 = DATA_DIR / f"scattering_{tag}.hdf5"
    output_csv = DATA_DIR / f"scattering_{tag}.csv"
    if (output_h5.exists() or output_csv.exists()) and not overwrite:
        raise FileExistsError(f"Outputs for {tag} exist; pass --overwrite")

    orbital_hoppings = np.asarray(ORBITAL_HOPPINGS) ** 2
    normalization = (
        T2_NORMALIZATION
        if scan == "in-plane"
        else np.sqrt(sites_per_sublattice * 2)
    )

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

        # t2_raw[orientation, ik, ikprime] is the sample-averaged scattering
        # strength. The inner amplitude keeps seed, snapshot, and spin indices.
        t2_raw = np.empty((n_angles, nkf, nkf), dtype=float)
        t2_error = np.empty_like(t2_raw) if scan == "out-of-plane" else None
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
                t2_raw[angle, ik] = (
                    np.mean(amplitude_squared, axis=(1, 2)) / normalization
                )
                if t2_error is not None:
                    seed_means = np.mean(amplitude_squared, axis=2)
                    t2_error[angle, ik] = (
                        np.std(seed_means, axis=1)
                        / np.sqrt(n_seeds)
                        / normalization
                    )
            if angle == 0 or (angle + 1) % 10 == 0:
                print(f"{tag}: orientation {angle + 1}/{n_angles}", flush=True)

    # The out-of-plane scan has no angle coordinate to smooth. Preserve both
    # raw and smoothed arrays only for the published in-plane convention.
    if scan == "in-plane":
        t2_smoothed = gaussian_filter1d(
            t2_raw, sigma=ANGLE_SMOOTHING_SIGMA, axis=0
        )
        smoothing_mode = "scipy.ndimage.gaussian_filter1d default reflect mode"
    else:
        t2_smoothed = t2_raw.copy()
        smoothing_mode = "none"

    mode = "w" if overwrite else "x"
    with h5py.File(output_h5, mode) as output:
        output.create_dataset("kf", data=kf)
        output.create_dataset("q_vectors", data=q_vectors)
        output.create_dataset("t2_raw", data=t2_raw)
        output.create_dataset("t2_smoothed", data=t2_smoothed)
        if t2_error is not None:
            output.create_dataset("t2_error", data=t2_error)
        output.attrs.update(
            scan=scan,
            field=field,
            temperature=temperature,
            sites_per_dto_sublattice=sites_per_sublattice,
            historical_normalization=normalization,
            angular_smoothing_sigma=(
                ANGLE_SMOOTHING_SIGMA if scan == "in-plane" else 0.0
            ),
            smoothing_mode=smoothing_mode,
        )

    # Flatten the momentum-pair tensor into a human-readable companion CSV.
    pairs_per_angle = nkf * nkf
    frame = pd.DataFrame({
        "field": field,
        "angle": np.repeat(np.arange(n_angles), pairs_per_angle),
        "ik": np.tile(np.repeat(np.arange(nkf), nkf), n_angles),
        "ikprime": np.tile(np.arange(nkf), n_angles * nkf),
        "qx": np.tile(q_vectors[..., 0].ravel(), n_angles),
        "qy": np.tile(q_vectors[..., 1].ravel(), n_angles),
        "qz": np.tile(q_vectors[..., 2].ravel(), n_angles),
        "t2_raw": t2_raw.ravel(),
        "t2_smoothed": t2_smoothed.ravel(),
    })
    if scan == "out-of-plane":
        frame.insert(1, "temperature", temperature)
        frame = frame.drop(columns=["angle", "t2_smoothed"])
        frame["t2_error"] = t2_error.ravel()
    frame.to_csv(output_csv, index=False)
    print(f"Wrote {output_h5}")
    print(f"Wrote {output_csv}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--scan", choices=("in-plane", "out-of-plane"), default="in-plane"
)
parser.add_argument("--fields", nargs="+", type=float)
parser.add_argument("--temperatures", nargs="+", type=float)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

if args.scan == "in-plane":
    fields = args.fields or FIELDS
    temperatures = args.temperatures or (TEMPERATURE,)
    sites_per_sublattice = SITES_PER_SUBLATTICE
else:
    fields = args.fields or OUT_OF_PLANE_FIELDS
    temperatures = args.temperatures or OUT_OF_PLANE_TEMPERATURES
    sites_per_sublattice = OUT_OF_PLANE_SITES_PER_SUBLATTICE

DATA_DIR.mkdir(exist_ok=True)
for selected_field in fields:
    for selected_temperature in temperatures:
        compute_condition(
            args.scan, selected_field, selected_temperature,
            sites_per_sublattice, overwrite=args.overwrite,
        )
