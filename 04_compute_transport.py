"""Compute relaxation rates, conductivity, and interface resistivity."""
import argparse

import h5py
import numpy as np
import pandas as pd

from config import (DATA_DIR, E_FERMI, EIO_NK, ENERGY_BROADENING, FIELDS,
                    TRANSPORT_REFERENCE_DIR, field_tag)


def historical_delta(energy1, energy2, width):
    """Gaussian approximation used in the archived notebooks."""
    return (
        np.exp(-(energy1 - energy2) ** 2 / width ** 2)
        / np.sqrt(2 * np.pi * width ** 2)
    )


def compute_field(field):
    tag = field_tag(field)
    eio_path = DATA_DIR / f"surface_states_summary_nk{EIO_NK}.hdf5"
    scattering_path = DATA_DIR / f"scattering_h{tag}.hdf5"

    with h5py.File(eio_path) as eio_h5, h5py.File(scattering_path) as scattering_h5:
        kf = np.squeeze(eio_h5["kf"][...])
        energies = np.squeeze(eio_h5["eigf"][...])
        velocities = eio_h5["v_kf"][...]
        surface_weight = np.sum(
            abs(eio_h5["proj_sl1"][...]) ** 2
            + abs(eio_h5["proj_sl2"][...]) ** 2
            + abs(eio_h5["proj_sl3"][...]) ** 2,
            axis=(1, 2),
        )
        t2 = {
            "raw": scattering_h5["t2_raw"][...],
            "smoothed": scattering_h5["t2_smoothed"][...],
        }

    norms = np.linalg.norm(kf, axis=1)
    cos_theta = (kf @ kf.T) / np.outer(norms, norms)
    pair_delta = historical_delta(
        energies[:, None], energies[None, :], ENERGY_BROADENING
    )
    fermi_delta = historical_delta(E_FERMI, energies, ENERGY_BROADENING)

    transport_rows = []
    relaxation_rows = []
    for mode, matrix_elements in t2.items():
        mode_rows = []
        for angle in range(len(matrix_elements)):
            inverse_tau = np.sum(
                matrix_elements[angle] * pair_delta * (1 - cos_theta), axis=1
            )
            if np.any(inverse_tau <= 0):
                raise ValueError(
                    f"Non-positive scattering rate at field {tag}, angle {angle}"
                )
            sigma_terms = (
                fermi_delta * abs(velocities[:, 0]) ** 2 * surface_weight
                / inverse_tau
            )
            sigma = np.sum(sigma_terms)
            rho = 1 / sigma
            mode_rows.append({
                "field": field,
                "angle": angle,
                "mode": mode,
                "sigma": sigma,
                "rho": rho,
            })
            relaxation_rows.append(pd.DataFrame({
                "field": field,
                "angle": angle,
                "mode": mode,
                "ik": np.arange(len(kf)),
                "kx": kf[:, 0],
                "ky": kf[:, 1],
                "energy": energies,
                "vx_archived": velocities[:, 0],
                "surface_weight": surface_weight,
                "inverse_tau": inverse_tau,
                "tau": 1 / inverse_tau,
                "sigma_term": sigma_terms,
            }))

        mode_frame = pd.DataFrame(mode_rows)
        rho0 = mode_frame.loc[mode_frame["angle"] == 0, "rho"].iloc[0]
        mode_frame["delta_rho_over_rho0"] = mode_frame["rho"] / rho0 - 1
        mode_frame.to_csv(DATA_DIR / f"rho_h{tag}_{mode}.csv", index=False)
        transport_rows.append(mode_frame)

        if mode == "raw":
            reference_path = TRANSPORT_REFERENCE_DIR / f"rho_h{tag}_raw.csv"
            if reference_path.exists():
                reference = pd.read_csv(reference_path)
                max_error = np.max(abs(mode_frame["rho"] - reference["rho"]))
                relative_error = np.max(
                    abs((mode_frame["rho"] - reference["rho"]) / reference["rho"])
                )
                print(
                    f"field {tag}: maximum archived rho error = {max_error:.3e} "
                    f"({relative_error:.3e} relative)"
                )

    return transport_rows, relaxation_rows


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fields", nargs="+", type=float, default=FIELDS)
args = parser.parse_args()

all_transport = []
all_relaxation = []
for selected_field in args.fields:
    transport, relaxation = compute_field(selected_field)
    all_transport.extend(transport)
    all_relaxation.extend(relaxation)

pd.concat(all_transport, ignore_index=True).to_csv(
    DATA_DIR / "transport_summary.csv", index=False
)
pd.concat(all_relaxation, ignore_index=True).to_csv(
    DATA_DIR / "relaxation_times.csv", index=False
)
print(f"Wrote {DATA_DIR / 'transport_summary.csv'}")
print(f"Wrote {DATA_DIR / 'relaxation_times.csv'}")
