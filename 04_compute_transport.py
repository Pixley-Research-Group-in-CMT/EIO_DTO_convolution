"""Compute EIO/TTO relaxation rates and interface resistivity (Sec. 5)."""
import argparse

import h5py
import numpy as np
import pandas as pd

from config import DATA_DIR, E_FERMI, EIO_NK, ENERGY_BROADENING


def gaussian_delta(energy1, energy2, width):
    """Normalized Gaussian used only to select conducting EIO states."""
    return np.exp(-0.5 * (energy1 - energy2) ** 2 / width ** 2) / (
        np.sqrt(2 * np.pi) * width
    )

def reflect_negative_phi(frame, sort_columns):
    """Copy each positive-angle row to -phi without duplicating phi=0."""
    angles = frame["phi_degrees"].to_numpy(dtype=float)
    if not np.all(np.isfinite(angles)):
        raise ValueError("Field angles must be finite before reflection")
    if np.any(angles < -1e-12):
        raise ValueError(
            "Negative angles are already present; reflection expects phi >= 0"
        )
    missing_zero = [
        field
        for field, group in frame.groupby("h_over_J_zz", sort=False)
        if not np.any(np.isclose(group["phi_degrees"], 0.0, atol=1e-12))
    ]
    if missing_zero:
        raise ValueError(
            "Every field needs a phi=0 reference; missing for "
            f"h/J_zz={missing_zero}"
        )

    nonnegative = frame.copy()
    zero = np.isclose(nonnegative["phi_degrees"], 0.0, atol=1e-12)
    nonnegative.loc[zero, "phi_degrees"] = 0.0
    mirrored = nonnegative.loc[~zero].copy()
    mirrored["phi_degrees"] = -mirrored["phi_degrees"]
    return (
        pd.concat([nonnegative, mirrored], ignore_index=True)
        .sort_values(list(sort_columns), kind="stable")
        .reset_index(drop=True)
    )


THEORIES = {
    "spinon": ("tto_dynamic_scattering.hdf5", "tto_transport_summary.csv",
               "tto_relaxation_times.csv"),
    "spin-wave": ("tto_spin_wave_scattering.hdf5",
                  "tto_spin_wave_transport_summary.csv",
                  "tto_spin_wave_relaxation_times.csv"),
}


def compute(theory):
    scattering_name, transport_name, relaxation_name = THEORIES[theory]
    eio_path = DATA_DIR / f"surface_states_summary_nk{EIO_NK}.hdf5"
    scattering_path = DATA_DIR / scattering_name
    with h5py.File(eio_path) as eio_h5, h5py.File(scattering_path) as scattering_h5:
        kf = np.squeeze(eio_h5["kf"][...])
        energies = np.squeeze(eio_h5["eigf"][...])
        velocities = eio_h5["v_kf"][...]
        surface_weight = np.sum(
            sum(np.abs(eio_h5[f"proj_sl{i}"][...]) ** 2 for i in (1, 2, 3)),
            axis=(1, 2),
        )
        matrix_elements = scattering_h5["total"][...]
        phi = scattering_h5["phi_degrees"][...]
        fields = scattering_h5["h_over_J_zz"][...]
        should_reflect_negative_phi = bool(
            scattering_h5.attrs.get("reflect_negative_phi", False)
        )

    norms = np.linalg.norm(kf, axis=1)
    cos_theta = np.clip((kf @ kf.T) / np.outer(norms, norms), -1.0, 1.0)
    transport_vertex = 1.0 - cos_theta
    # Both Sec. 4.1 and 4.2 Bragg peaks are forward scattering at q=0.
    np.fill_diagonal(transport_vertex, 0.0)
    inverse_tau = np.sum(matrix_elements * transport_vertex[None, :, :], axis=2)
    if np.any(inverse_tau <= 0):
        condition, ik = np.argwhere(inverse_tau <= 0)[0]
        raise ValueError(f"Non-positive rate at condition {condition}, EIO state {ik}")

    fermi_delta = gaussian_delta(E_FERMI, energies, ENERGY_BROADENING)
    conductivity_weight = (
        fermi_delta[None, :]
        * np.abs(velocities[:, 0])[None, :] ** 2
        * surface_weight[None, :]
    )
    sigma_terms = conductivity_weight / inverse_tau
    sigma = np.sum(sigma_terms, axis=1)
    transport = pd.DataFrame({
        "phi_degrees": phi,
        "h_over_J_zz": fields,
        "sigma": sigma,
        "rho": 1.0 / sigma,
    })
    rho_phi0 = transport.groupby("h_over_J_zz")["rho"].transform(
        lambda values: values.iloc[np.argmin(np.abs(
            transport.loc[values.index, "phi_degrees"].to_numpy()
        ))]
    )
    transport["rho_phi0"] = rho_phi0
    normalized = (transport["rho"] - rho_phi0) / rho_phi0
    transport["normalized_resistivity"] = normalized
    # Keep literal legacy columns for downloaded-run compatibility while using
    # normalized_resistivity as the canonical plotting quantity.
    transport["rho_over_rho_phi0"] = transport["rho"] / rho_phi0
    transport["delta_rho_over_rho_phi0"] = normalized
    nk = len(kf)
    relaxation = pd.DataFrame({
        "phi_degrees": np.repeat(phi, nk),
        "h_over_J_zz": np.repeat(fields, nk),
        "ik": np.tile(np.arange(nk), len(phi)),
        "inverse_tau": inverse_tau.ravel(),
        "tau": (1.0 / inverse_tau).ravel(),
        "sigma_term": sigma_terms.ravel(),
    })
    if should_reflect_negative_phi:
        transport = reflect_negative_phi(
            transport, ("h_over_J_zz", "phi_degrees")
        )
        relaxation = reflect_negative_phi(
            relaxation, ("h_over_J_zz", "phi_degrees", "ik")
        )
        print(
            "Reflected nonnegative-angle transport results onto negative phi",
            flush=True,
        )

    transport.to_csv(DATA_DIR / transport_name, index=False)
    relaxation.to_csv(DATA_DIR / relaxation_name, index=False)
    print(f"Wrote {DATA_DIR / transport_name}")
    print(f"Wrote {DATA_DIR / relaxation_name}")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--theory", choices=tuple(THEORIES), default="spinon")
compute(parser.parse_args().theory)
