"""Dynamical TTO spin structure factor from Sec. 4.1 of ``papers/gMFT``.

The implementation uses the scalar, twofold-degenerate spinon dispersion of
the uniform zero-flux, J_zpm=0 saddle point.  It returns the continuum,
one-condensate, and two-condensate terms in Eqs. (85)--(87), evaluated at the
momentum and energy transfers required by the EIO surface states.
"""
from pathlib import Path

import numpy as np
import pandas as pd


# Diamond nearest-neighbour vectors expressed in the FCC primitive basis used
# by gMFT_for_QSI.  A reduced momentum q therefore contributes exp(2 pi i q.e).
E_MU = np.asarray([
    [0.25, 0.25, 0.25],
    [0.25, -0.75, 0.25],
    [0.25, 0.25, -0.75],
    [-0.75, 0.25, 0.25],
])
FCC_PRIMITIVES = np.asarray([
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
])


def lorentzian_delta(x, eta):
    """Normalized broadened delta from Eq. (83)."""
    if eta <= 0:
        raise ValueError("The DSSF broadening eta must be positive")
    return eta / (np.pi * (np.asarray(x) ** 2 + eta ** 2))


def reduced_k_grid(nk):
    """Uniform [0,1)^3 FCC Brillouin-zone grid used by the gMFT solver."""
    if nk <= 0:
        raise ValueError("nk must be positive")
    axis = np.arange(nk, dtype=float) / nk
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def spinon_potential(k_reduced, j_pm_ratio, s_minus):
    """Dimensionless v_k/J_zz from Eq. (53)."""
    k_reduced = np.asarray(k_reduced, dtype=float)
    s_minus = np.asarray(s_minus, dtype=complex)
    if k_reduced.shape[-1] != 3 or s_minus.shape != (4,):
        raise ValueError("k_reduced must end in 3 and s_minus must have shape (4,)")
    value = np.zeros(k_reduced.shape[:-1], dtype=complex)
    for mu in range(4):
        for nu in range(4):
            if mu == nu:
                continue
            phase = np.exp(
                2j * np.pi * np.einsum(
                    "...i,i->...", k_reduced, E_MU[mu] - E_MU[nu]
                )
            )
            value -= j_pm_ratio * np.conj(s_minus[mu]) * s_minus[nu] * phase
    imaginary_error = float(np.max(np.abs(value.imag)))
    if imaginary_error > 1e-9:
        raise ValueError(f"Spinon potential is not real (error {imaginary_error:.3e})")
    return value.real


def spinon_frequency(k_reduced, j_pm_ratio, s_minus, lambda_ratio, j_zz_ev):
    """Physical spinon frequency sqrt(2 Jzz (lambda+v)) in eV."""
    shifted = lambda_ratio + spinon_potential(k_reduced, j_pm_ratio, s_minus)
    if np.min(shifted) < -1e-8:
        raise ValueError(
            "Saved lambda lies below the spinon-potential minimum on the DSSF grid"
        )
    return j_zz_ev * np.sqrt(2.0 * np.maximum(shifted, 0.0))


def eio_to_tto_reduced_momentum(q_slab, cubic_lattice_angstrom):
    """Convert EIO slab Cartesian momenta (1/Angstrom) to TTO FCC r.l.v.

    The axes exactly reproduce ``eio_tb.transform_to_slab_coordinates``.  The
    result is in the same reduced coordinates used in the gMFT CSV's
    condensate momentum.
    """
    q_slab = np.asarray(q_slab, dtype=float)
    x_axis = np.asarray([-1.0, 0.0, 1.0]) / np.sqrt(2.0)
    y_axis = np.asarray([1.0, -2.0, 1.0]) / np.sqrt(6.0)
    z_axis = np.asarray([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    slab_rotation = np.stack((x_axis, y_axis, z_axis))
    q_cubic = np.einsum("...i,ij->...j", q_slab, slab_rotation)
    direct = (cubic_lattice_angstrom / 2.0) * FCC_PRIMITIVES
    reciprocal = 2.0 * np.pi * np.linalg.inv(direct)
    return np.einsum("ij,...j->...i", np.linalg.inv(reciprocal), q_cubic)


def load_gmft_scan(folder, j_pm_ratio, field_ratios=None):
    """Load one J_pm slice, taking each phi only from its matching directory."""
    folder = Path(folder)
    frames = []
    directory_phis = []
    for path in folder.glob("phi_*/field_diagnostics_min_energy.csv"):
        try:
            directory_phi = float(path.parent.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        directory_phis.append(directory_phi)
        frame = pd.read_csv(path)
        frame = frame[
            np.isclose(frame["phi_degrees"], directory_phi)
            & np.isclose(frame["J_pm_over_J_zz"], j_pm_ratio)
            & frame["status"].fillna("").eq("ok")
        ]
        if field_ratios is not None:
            requested = np.asarray(field_ratios, dtype=float)
            frame = frame[
                np.any(
                    np.isclose(frame["h_over_J_zz"].to_numpy()[:, None], requested[None, :]),
                    axis=1,
                )
            ]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No gMFT angle tables found under {folder}")
    result = pd.concat(frames, ignore_index=True)
    if result.empty:
        raise ValueError(f"No rows found for J_pm/J_zz={j_pm_ratio:g}")
    keys = ["phi_degrees", "h_over_J_zz", "J_pm_over_J_zz"]
    if result.duplicated(keys).any():
        duplicate = result.loc[result.duplicated(keys, keep=False), keys].iloc[0]
        raise ValueError(f"Duplicate gMFT condition: {duplicate.to_dict()}")
    if field_ratios is not None:
        found = result["h_over_J_zz"].unique()
        missing = [value for value in field_ratios if not np.any(np.isclose(found, value))]
        if missing:
            raise ValueError(f"Missing requested h/J_zz values: {missing}")
        expected_phis = np.sort(np.unique(directory_phis))
        for field in field_ratios:
            actual_phis = np.sort(
                result.loc[np.isclose(result["h_over_J_zz"], field), "phi_degrees"].unique()
            )
            missing_phis = expected_phis[
                ~np.any(np.isclose(expected_phis[:, None], actual_phis[None, :]), axis=1)
            ]
            if len(missing_phis):
                raise ValueError(
                    f"Incomplete converged scan for h/J_zz={field:g}; "
                    f"missing phi={missing_phis.tolist()}"
                )
    return result.sort_values(["h_over_J_zz", "phi_degrees"]).reset_index(drop=True)


def row_mean_field(row):
    """Extract the Sec. 4.1 saddle-point quantities from one CSV row."""
    s_minus = np.asarray(
        [row[f"s_x_mu{mu}"] - 1j * row[f"s_y_mu{mu}"] for mu in range(4)]
    )
    k_min_columns = [f"k_min_rlv_{axis}" for axis in range(3)]
    k_min = row[k_min_columns].to_numpy(dtype=float)
    if not bool(row["condensed"]):
        k_min = np.zeros(3)
    elif not np.isfinite(k_min).all():
        raise ValueError("Condensed gMFT row has no finite k_min")
    return {
        "j_pm_ratio": float(row["J_pm_over_J_zz"]),
        "lambda_ratio": float(row["lambda"]),
        "rho": float(row["rho"]),
        "condensed": bool(row["condensed"]),
        "s_minus": s_minus,
        "k_min": k_min,
    }


def dynamic_structure_factor(
    q_reduced,
    omega_ev,
    mean_field,
    *,
    j_zz_ev,
    eta_ev,
    nk=8,
    interface_sublattices=(1, 2, 3),
):
    """Evaluate Eqs. (85)--(87) for many (q, omega) transfer points.

    Returns arrays with shape ``(3, ...)`` for the three interfacial kagome
    sublattices.  The zero-temperature DSSF has support at positive omega; a
    Lorentzian supplies the finite numerical energy resolution.
    """
    q_reduced = np.asarray(q_reduced, dtype=float)
    omega_ev = np.asarray(omega_ev, dtype=float)
    if q_reduced.shape[:-1] != omega_ev.shape or q_reduced.shape[-1] != 3:
        raise ValueError("q_reduced must have shape omega_ev.shape + (3,)")
    original_shape = omega_ev.shape
    q = q_reduced.reshape(-1, 3)
    transfer = omega_ev.reshape(-1)
    k = reduced_k_grid(nk)
    s_minus = mean_field["s_minus"]
    j_pm_ratio = mean_field["j_pm_ratio"]
    lambda_ratio = mean_field["lambda_ratio"]
    rho = mean_field["rho"]
    k_min = mean_field["k_min"]

    omega_k = spinon_frequency(k, j_pm_ratio, s_minus, lambda_ratio, j_zz_ev)
    omega_kq = spinon_frequency(
        k[None, :, :] + q[:, None, :],
        j_pm_ratio,
        s_minus,
        lambda_ratio,
        j_zz_ev,
    )
    # A condensed soft mode has a tiny nonzero frequency because lambda and v
    # were rounded in the CSV.  Apply the same dimensionless 1e-10 tolerance
    # used by the source solver so that its weight is carried only by Eqs. 86-87.
    soft_frequency = j_zz_ev * np.sqrt(2e-10)
    active = (omega_k[None, :] > soft_frequency) & (omega_kq > soft_frequency)
    green_product = np.zeros_like(omega_kq)
    np.divide(
        j_zz_ev ** 2,
        omega_k[None, :] * omega_kq,
        out=green_product,
        where=active,
    )
    continuum_base = np.mean(
        green_product
        * lorentzian_delta(
            transfer[:, None] - omega_k[None, :] - omega_kq, eta_ev
        ),
        axis=1,
    )

    one_condensate_base = np.zeros(len(q), dtype=float)
    two_condensate_base = np.zeros(len(q), dtype=float)
    if rho > 0.0:
        omega_plus = spinon_frequency(
            k_min[None, :] + q, j_pm_ratio, s_minus, lambda_ratio, j_zz_ev
        )
        omega_minus = spinon_frequency(
            k_min[None, :] - q, j_pm_ratio, s_minus, lambda_ratio, j_zz_ev
        )
        for excitation in (omega_plus, omega_minus):
            prefactor = np.zeros_like(excitation)
            np.divide(
                rho * j_zz_ev,
                excitation,
                out=prefactor,
                where=excitation > soft_frequency,
            )
            one_condensate_base += prefactor * lorentzian_delta(
                transfer - excitation, eta_ev
            )
        reciprocal_zero = np.linalg.norm(q - np.round(q), axis=1) < 1e-10
        two_condensate_base[reciprocal_zero] = (
            nk ** 3 * rho ** 2
            * lorentzian_delta(transfer[reciprocal_zero], eta_ev)
        )

    weights = np.abs(s_minus[np.asarray(interface_sublattices)]) ** 2
    def weighted(base):
        return (weights[:, None] * base[None, :]).reshape((len(weights),) + original_shape)

    continuum = weighted(continuum_base)
    one_condensate = weighted(one_condensate_base)
    two_condensate = weighted(two_condensate_base)
    return {
        "continuum": continuum,
        "one_condensate": one_condensate,
        "two_condensate": two_condensate,
        "total": continuum + one_condensate + two_condensate,
    }
