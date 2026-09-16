"""Compute relaxation rates, conductivity, and interface resistivity."""
import argparse
import json

import h5py
import numpy as np
import pandas as pd

from config import (DATA_DIR, E_FERMI, EIO_NK, ENERGY_BROADENING, FIELDS,
                    OUT_OF_PLANE_FIELDS, OUT_OF_PLANE_TEMPERATURES,
                    OUT_OF_PLANE_TRANSPORT_REFERENCE, TEMPERATURE,
                    TRANSPORT_REFERENCE_DIR, condition_tag, field_tag)


def historical_delta(energy1, energy2, width):
    """Return the nonstandard Gaussian delta used in the archived notebooks.

    The exponent lacks the conventional factor of two. It is intentionally
    retained here so the cleaned workflow reproduces the historical curves.
    """
    return (
        np.exp(-(energy1 - energy2) ** 2 / width ** 2)
        / np.sqrt(2 * np.pi * width ** 2)
    )


def compute_condition(scan, field, temperature):
    """Convert one condition's scattering matrix into relaxation and transport."""
    tag = condition_tag(scan, field, temperature)
    eio_path = DATA_DIR / f"surface_states_summary_nk{EIO_NK}.hdf5"
    scattering_path = DATA_DIR / f"scattering_{tag}.hdf5"

    with h5py.File(eio_path) as eio_h5, h5py.File(
        scattering_path
    ) as scattering_h5:
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
        if scan == "in-plane":
            t2 = {
                "raw": scattering_h5["t2_raw"][...],
                "smoothed": scattering_h5["t2_smoothed"][...],
            }
            t2_error = None
        else:
            t2 = {"raw": scattering_h5["t2_raw"][...]}
            t2_error = scattering_h5["t2_error"][...]

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
                f"Non-positive scattering rate for {tag}, orientation {angle}, ik {ik}"
            )

        # Each retained EIO state contributes tau_k v_x^2, restricted to the
        # Fermi surface and weighted by its interface localization.
        conductivity_weight = (
            fermi_delta[None, :]
            * np.abs(velocities[:, 0])[None, :] ** 2
            * surface_weight[None, :]
        )
        sigma_terms = conductivity_weight / inverse_tau
        sigma = np.sum(sigma_terms, axis=1)
        mode_frame = pd.DataFrame({
            "field": field,
            "angle": np.arange(len(matrix_elements)),
            "mode": mode,
            "sigma": sigma,
            "rho": 1 / sigma,
        })

        inverse_tau_error = None
        if t2_error is not None:
            inverse_tau_error = np.asarray([
                np.sum(error * pair_delta * (1 - cos_theta), axis=1)
                for error in t2_error
            ])
            sigma_error = np.sum(
                inverse_tau_error / inverse_tau ** 2 * conductivity_weight,
                axis=1,
            )
            mode_frame["sigma_error"] = sigma_error
            mode_frame["rho_error"] = sigma_error / sigma ** 2

        if scan == "in-plane":
            rho0 = mode_frame.loc[mode_frame["angle"] == 0, "rho"].iloc[0]
            mode_frame["delta_rho_over_rho0"] = mode_frame["rho"] / rho0 - 1
        else:
            mode_frame.insert(1, "temperature", temperature)
            mode_frame = mode_frame.drop(columns="angle")

        mode_frame.to_csv(DATA_DIR / f"rho_{tag}_{mode}.csv", index=False)
        transport_rows.append(mode_frame)

        n_angles, nkf = inverse_tau.shape
        relaxation_frame = pd.DataFrame({
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
        })
        if inverse_tau_error is not None:
            relaxation_frame.insert(1, "temperature", temperature)
            relaxation_frame["inverse_tau_error"] = inverse_tau_error.ravel()
        relaxation_rows.append(relaxation_frame)

        # The archived raw CSVs are numerical regression references, not extra
        # inputs to the transport calculation.
        if scan == "in-plane" and mode == "raw":
            reference_path = TRANSPORT_REFERENCE_DIR / f"rho_h{field_tag(field)}_raw.csv"
            if reference_path.exists():
                reference = pd.read_csv(reference_path)
                max_error = np.max(np.abs(mode_frame["rho"] - reference["rho"]))
                relative_error = np.max(
                    np.abs(
                        (mode_frame["rho"] - reference["rho"]) / reference["rho"]
                    )
                )
                print(
                    f"field {field_tag(field)}: maximum archived rho error = "
                    f"{max_error:.3e} ({relative_error:.3e} relative)"
                )

    return transport_rows, relaxation_rows


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--scan", choices=("in-plane", "out-of-plane"), default="in-plane"
)
parser.add_argument("--fields", nargs="+", type=float)
parser.add_argument("--temperatures", nargs="+", type=float)
args = parser.parse_args()

if args.scan == "in-plane":
    fields = args.fields or FIELDS
    temperatures = args.temperatures or (TEMPERATURE,)
    transport_output = DATA_DIR / "transport_summary.csv"
    relaxation_output = DATA_DIR / "relaxation_times.csv"
else:
    fields = args.fields or OUT_OF_PLANE_FIELDS
    temperatures = args.temperatures or OUT_OF_PLANE_TEMPERATURES
    transport_output = DATA_DIR / "transport_out_of_plane.csv"
    relaxation_output = DATA_DIR / "relaxation_times_out_of_plane.csv"

# Collect all requested conditions into two collaborator-friendly summaries.
all_transport = []
all_relaxation = []
for selected_field in fields:
    for selected_temperature in temperatures:
        transport, relaxation = compute_condition(
            args.scan, selected_field, selected_temperature
        )
        all_transport.extend(transport)
        all_relaxation.extend(relaxation)

transport_frame = pd.concat(all_transport, ignore_index=True)
transport_frame.to_csv(transport_output, index=False)
pd.concat(all_relaxation, ignore_index=True).to_csv(
    relaxation_output, index=False
)
print(f"Wrote {transport_output}")
print(f"Wrote {relaxation_output}")

if args.scan == "out-of-plane" and OUT_OF_PLANE_TRANSPORT_REFERENCE.exists():
    reference = pd.read_csv(OUT_OF_PLANE_TRANSPORT_REFERENCE)
    reference = reference.sort_values(["magfield", "kT"]).reset_index(drop=True)
    generated = transport_frame.sort_values(["field", "temperature"]).reset_index(
        drop=True
    )
    if len(generated) != len(reference):
        print("Skipped full out-of-plane regression for a partial condition set")
    else:
        column_pairs = {
            "sigma": "sigma",
            "rho": "rho",
            "sigma_error": "sigma_err",
            "rho_error": "rho_err",
        }
        validation = {
            generated_name: float(np.max(np.abs(
                generated[generated_name] - reference[reference_name]
            )))
            for generated_name, reference_name in column_pairs.items()
        }
        validation_output = DATA_DIR / "out_of_plane_validation.json"
        validation_output.write_text(json.dumps(validation, indent=2) + "\n")
        print(json.dumps(validation, indent=2))
        print(f"Wrote {validation_output}")
