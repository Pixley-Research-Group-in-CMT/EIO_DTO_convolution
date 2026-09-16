"""Compute relaxation rates, conductivity, and interface resistivity."""
import argparse

import h5py
import numpy as np
import pandas as pd

from config import (DATA_DIR, E_FERMI, EIO_NK, ENERGY_BROADENING, FIELDS,
                    TRANSPORT_REFERENCE_DIR, field_tag)


def historical_delta(energy1, energy2, width):
    """Return the nonstandard Gaussian delta used in the archived notebooks.

    The exponent lacks the conventional factor of two. It is intentionally
    retained here so the cleaned workflow reproduces the historical curves.
    """
    return (
        np.exp(-(energy1 - energy2) ** 2 / width ** 2)
        / np.sqrt(2 * np.pi * width ** 2)
    )


def compute_field(field):
    """Convert one field's scattering matrix into relaxation and transport data."""
    tag = field_tag(field)
    eio_path = DATA_DIR / f"surface_states_summary_nk{EIO_NK}.hdf5"
    scattering_path = DATA_DIR / f"scattering_h{tag}.hdf5"

    with h5py.File(eio_path) as eio_h5, h5py.File(scattering_path) as scattering_h5:
        kf = np.squeeze(eio_h5["kf"][...])
        energies = np.squeeze(eio_h5["eigf"][...])
        velocities = eio_h5["v_kf"][...]
        # Weight transport by the probability carried on the three EIO
        # sublattices that touch the DTO interface.
        surface_weight = np.sum(
            np.abs(eio_h5["proj_sl1"][...]) ** 2
            + np.abs(eio_h5["proj_sl2"][...]) ** 2
            + np.abs(eio_h5["proj_sl3"][...]) ** 2,
            axis=(1, 2),
        )
        t2 = {
            "raw": scattering_h5["t2_raw"][...],
            "smoothed": scattering_h5["t2_smoothed"][...],
        }

    # Precompute the transport vertex and the two broadened energy constraints:
    # elastic scattering requires e_k = e_kprime, while conduction is sampled
    # near the fixed Fermi energy.
    norms = np.linalg.norm(kf, axis=1)
    cos_theta = (kf @ kf.T) / np.outer(norms, norms)
    pair_delta = historical_delta(
        energies[:, None], energies[None, :], ENERGY_BROADENING
    )
    fermi_delta = historical_delta(E_FERMI, energies, ENERGY_BROADENING)

    transport_rows = []
    relaxation_rows = []
    # Evaluate raw and angle-smoothed matrix elements independently. The
    # relaxation rate is sum_kprime |T|^2 delta(e-e') (1-cos theta).
    for mode, matrix_elements in t2.items():
        inverse_tau = np.asarray([
            np.sum(matrix_element * pair_delta * (1 - cos_theta), axis=1)
            for matrix_element in matrix_elements
        ])
        if np.any(inverse_tau <= 0):
            angle, ik = np.argwhere(inverse_tau <= 0)[0]
            raise ValueError(
                f"Non-positive scattering rate at field {tag}, angle {angle}, ik {ik}"
            )

        # Each retained EIO state contributes tau_k v_x^2, restricted to the
        # Fermi surface and weighted by its interface localization.
        sigma_terms = (
            fermi_delta[None, :]
            * np.abs(velocities[:, 0])[None, :] ** 2
            * surface_weight[None, :]
            / inverse_tau
        )
        sigma = np.asarray([np.sum(terms) for terms in sigma_terms])
        mode_frame = pd.DataFrame({
            "field": field,
            "angle": np.arange(len(matrix_elements)),
            "mode": mode,
            "sigma": sigma,
            "rho": 1 / sigma,
        })
        rho0 = mode_frame.loc[mode_frame["angle"] == 0, "rho"].iloc[0]
        mode_frame["delta_rho_over_rho0"] = mode_frame["rho"] / rho0 - 1
        mode_frame.to_csv(DATA_DIR / f"rho_h{tag}_{mode}.csv", index=False)
        transport_rows.append(mode_frame)

        n_angles, nkf = inverse_tau.shape
        relaxation_rows.append(pd.DataFrame({
            "field": field,
            "angle": np.repeat(np.arange(n_angles), nkf),
            "mode": mode,
            "ik": np.tile(np.arange(nkf), n_angles),
            "kx": np.tile(kf[:, 0], n_angles),
            "ky": np.tile(kf[:, 1], n_angles),
            "energy": np.tile(energies, n_angles),
            "vx_archived": np.tile(velocities[:, 0], n_angles),
            "surface_weight": np.tile(surface_weight, n_angles),
            "inverse_tau": inverse_tau.ravel(),
            "tau": 1 / inverse_tau.ravel(),
            "sigma_term": sigma_terms.ravel(),
        }))

        # The archived raw CSVs are numerical regression references, not extra
        # inputs to the transport calculation.
        if mode == "raw":
            reference_path = TRANSPORT_REFERENCE_DIR / f"rho_h{tag}_raw.csv"
            if reference_path.exists():
                reference = pd.read_csv(reference_path)
                max_error = np.max(np.abs(mode_frame["rho"] - reference["rho"]))
                relative_error = np.max(
                    np.abs(
                        (mode_frame["rho"] - reference["rho"]) / reference["rho"]
                    )
                )
                print(
                    f"field {tag}: maximum archived rho error = {max_error:.3e} "
                    f"({relative_error:.3e} relative)"
                )

    return transport_rows, relaxation_rows


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fields", nargs="+", type=float, default=FIELDS)
args = parser.parse_args()

# Collect all requested fields into two collaborator-friendly summary tables.
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
