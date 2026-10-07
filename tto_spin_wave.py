"""Linear spin-wave DSSF from Sec. 4.2 of ``papers/gMFT.pdf``.

For every field direction, a Zeeman-minimizing local-Ising configuration seeds
a multistart minimization of the physical classical spin Hamiltonian.  The BdG
expansion and Eqs. (99)--(100) are evaluated about that stable state.  The
bond-dependent J_pmpm tensor follows the gamma convention in Eqs. (1)--(2).
"""
import numpy as np
from scipy.optimize import least_squares, minimize

from tto_structure_factor import E_MU, lorentzian_delta


LOCAL_Z = np.asarray([
    [1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
]) / np.sqrt(3.0)
LOCAL_X = np.asarray([
    [-2.0, 1.0, 1.0],
    [-2.0, -1.0, -1.0],
    [2.0, 1.0, -1.0],
    [2.0, -1.0, 1.0],
]) / np.sqrt(6.0)
LOCAL_Y = np.cross(LOCAL_Z, LOCAL_X)
BOSON_METRIC = np.diag([1.0] * 4 + [-1.0] * 4)
MINUS_VECTOR = np.asarray([0.5, 0.5j, 0.0])
PLUS_VECTOR = np.conj(MINUS_VECTOR)
OMEGA = np.exp(2.0j * np.pi / 3.0)
GAMMA = np.asarray([
    [0.0, 1.0, OMEGA, OMEGA ** 2],
    [1.0, 0.0, OMEGA ** 2, OMEGA],
    [OMEGA, OMEGA ** 2, 0.0, 1.0],
    [OMEGA ** 2, OMEGA, 1.0, 0.0],
], dtype=complex)
FIELD_C1 = np.asarray([-1.0, -1.0, 2.0]) / np.sqrt(6.0)
FIELD_C2 = np.asarray([1.0, -1.0, 0.0]) / np.sqrt(2.0)
FIELD_C3 = np.asarray([1.0, 1.0, 1.0]) / np.sqrt(3.0)


def magnetic_field_direction(theta_degrees=90.0, phi_degrees=0.0):
    """Return the unit field direction in the convention of Eqs. (10)-(11)."""
    theta = np.deg2rad(float(theta_degrees))
    phi = np.deg2rad(float(phi_degrees))
    direction = (
        np.sin(theta) * np.cos(phi) * FIELD_C1
        + np.sin(theta) * np.sin(phi) * FIELD_C2
        + np.cos(theta) * FIELD_C3
    )
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 1e-14:
        raise ValueError("Magnetic-field direction must be finite and nonzero")
    return direction / norm


def local_zeeman_components(h_over_j_zz, theta_degrees=90.0, phi_degrees=0.0):
    """Project a dimensionless global field onto the four local Ising axes."""
    h_over_j_zz = float(h_over_j_zz)
    if not np.isfinite(h_over_j_zz) or h_over_j_zz < 0.0:
        raise ValueError("h/J_zz must be finite and nonnegative")
    direction = magnetic_field_direction(theta_degrees, phi_degrees)
    return h_over_j_zz * (LOCAL_Z @ direction)


def zeeman_minimizing_spins(h_z, spin_length=0.5, zero_tolerance=1e-12):
    """Return local-Ising spins minimizing ``-sum_mu h_z_mu S_z_mu``.

    A zero local field leaves that sublattice unconstrained by the Zeeman
    energy; ``+z`` is chosen as a deterministic representative.
    """
    h_z = np.asarray(h_z, dtype=float)
    if h_z.shape != (4,) or not np.all(np.isfinite(h_z)):
        raise ValueError("h_z must contain four finite local field components")
    orientation = np.where(h_z < -abs(float(zero_tolerance)), -1.0, 1.0)
    spins = np.zeros((4, 3), dtype=float)
    spins[:, 2] = float(spin_length) * orientation
    return spins


def shifted_k_grid(nk):
    """Midpoint BZ grid that avoids assigning finite weight to the Goldstone point."""
    axis = (np.arange(nk, dtype=float) + 0.5) / nk
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def _unit_vectors(parameters):
    theta = np.asarray(parameters[:4])
    phi = np.asarray(parameters[4:])
    return np.stack(
        (np.sin(theta) * np.cos(phi),
         np.sin(theta) * np.sin(phi),
         np.cos(theta)),
        axis=1,
    )


def _vector_derivatives(parameters):
    theta = np.asarray(parameters[:4])
    phi = np.asarray(parameters[4:])
    dtheta = np.stack(
        (np.cos(theta) * np.cos(phi),
         np.cos(theta) * np.sin(phi),
         -np.sin(theta)),
        axis=1,
    )
    dphi = np.stack(
        (-np.sin(theta) * np.sin(phi),
         np.sin(theta) * np.cos(phi),
         np.zeros(4)),
        axis=1,
    )
    return dtheta, dphi


def exchange_matrix(j_pm_ratio, j_pmpm_ratio=0.0, gamma=1.0):
    """Local Cartesian bond exchange in J_zz units for Eqs. (1)--(2).

    ``gamma`` is the complex phase for the ordered sublattice pair.  The
    returned real tensor acts on ``(S_x, S_y, S_z)`` in the two local frames.
    """
    gamma = complex(gamma)
    transverse = -2.0 * j_pm_ratio
    anisotropy_real = 2.0 * j_pmpm_ratio * gamma.real
    anisotropy_imaginary = -2.0 * j_pmpm_ratio * gamma.imag
    return np.asarray([
        [transverse + anisotropy_real, anisotropy_imaginary, 0.0],
        [anisotropy_imaginary, transverse - anisotropy_real, 0.0],
        [0.0, 0.0, 1.0],
    ])


def classical_energy_and_gradient(
    parameters,
    h_z,
    j_pm_ratio,
    *,
    j_pmpm_ratio=0.0,
    spin_length=0.5,
):
    """Classical four-sublattice energy and analytic angular gradient."""
    unit = _unit_vectors(parameters)
    energy = 0.0
    vector_gradient = np.zeros_like(unit)
    for mu in range(4):
        for nu in range(mu + 1, 4):
            # Each sublattice pair has two nearest-neighbour bonds per cell.
            exchange = exchange_matrix(
                j_pm_ratio, j_pmpm_ratio, GAMMA[mu, nu]
            )
            energy += 2.0 * spin_length ** 2 * unit[mu] @ exchange @ unit[nu]
            vector_gradient[mu] += 2.0 * spin_length ** 2 * exchange @ unit[nu]
            vector_gradient[nu] += (
                2.0 * spin_length ** 2 * exchange.T @ unit[mu]
            )
    energy -= spin_length * np.dot(h_z, unit[:, 2])
    vector_gradient[:, 2] -= spin_length * h_z
    dtheta, dphi = _vector_derivatives(parameters)
    gradient = np.concatenate((
        np.einsum("mi,mi->m", vector_gradient, dtheta),
        np.einsum("mi,mi->m", vector_gradient, dphi),
    ))
    return float(energy), gradient


def classical_tangent_hessian(
    unit_vectors_local,
    h_z,
    j_pm_ratio,
    *,
    j_pmpm_ratio=0.0,
    spin_length=0.5,
):
    """Return the constrained classical Hessian and local tangent bases."""
    unit = np.asarray(unit_vectors_local, dtype=float)
    unit = unit / np.linalg.norm(unit, axis=1)[:, None]
    bases = np.empty((4, 3, 2), dtype=float)
    vector_gradient = np.zeros_like(unit)
    hessian = np.zeros((8, 8), dtype=float)

    for mu in range(4):
        reference = np.eye(3)[np.argmin(np.abs(unit[mu]))]
        tangent_x = np.cross(unit[mu], reference)
        tangent_x /= np.linalg.norm(tangent_x)
        tangent_y = np.cross(unit[mu], tangent_x)
        bases[mu] = np.stack((tangent_x, tangent_y), axis=1)

    for mu in range(4):
        for nu in range(mu + 1, 4):
            exchange = exchange_matrix(
                j_pm_ratio, j_pmpm_ratio, GAMMA[mu, nu]
            )
            coupling = 2.0 * spin_length ** 2 * exchange
            vector_gradient[mu] += coupling @ unit[nu]
            vector_gradient[nu] += coupling.T @ unit[mu]
            block = bases[mu].T @ coupling @ bases[nu]
            hessian[2 * mu:2 * mu + 2, 2 * nu:2 * nu + 2] = block
            hessian[2 * nu:2 * nu + 2, 2 * mu:2 * mu + 2] = block.T

    vector_gradient[:, 2] -= spin_length * np.asarray(h_z)
    for mu in range(4):
        curvature = -np.dot(vector_gradient[mu], unit[mu])
        hessian[2 * mu:2 * mu + 2, 2 * mu:2 * mu + 2] = (
            curvature * np.eye(2)
        )
    return hessian, bases


def _tilted_local_spins(
    unit_vectors_local, tangent_bases, mode, amplitude, spin_length
):
    """Retract a small collective tangent-space displacement onto each spin."""
    displacement = np.einsum(
        "mai,mi->ma", tangent_bases, np.asarray(mode).reshape(4, 2)
    )
    tilted = np.asarray(unit_vectors_local) + float(amplitude) * displacement
    tilted /= np.linalg.norm(tilted, axis=1)[:, None]
    return float(spin_length) * tilted


def _parameters_from_local_spins(local_spins):
    unit = 2.0 * np.asarray(local_spins, dtype=float)
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    return np.concatenate((
        np.arccos(np.clip(unit[:, 2], -1.0, 1.0)),
        np.arctan2(unit[:, 1], unit[:, 0]),
    ))


def relax_ordered_state(
    local_spins,
    h_z,
    j_pm_ratio,
    *,
    j_pmpm_ratio=0.0,
    spin_length=0.5,
    random_starts=12,
    stability_tilt=1e-2,
):
    """Minimize from the physical seed, its unstable tilt, and random starts."""
    if not np.isfinite(stability_tilt) or stability_tilt <= 0.0:
        raise ValueError("stability_tilt must be finite and positive")
    initial_unit = np.asarray(local_spins, dtype=float)
    initial_unit /= np.linalg.norm(initial_unit, axis=1)[:, None]
    initial_parameters = _parameters_from_local_spins(local_spins)
    starts = [initial_parameters]

    seed_hessian, seed_bases = classical_tangent_hessian(
        initial_unit,
        h_z,
        j_pm_ratio,
        j_pmpm_ratio=j_pmpm_ratio,
        spin_length=spin_length,
    )
    seed_curvatures, seed_modes = np.linalg.eigh(seed_hessian)
    stability_tilt_used = seed_curvatures[0] < -1e-10
    if stability_tilt_used:
        for direction in (-1.0, 1.0):
            tilted_spins = _tilted_local_spins(
                initial_unit,
                seed_bases,
                seed_modes[:, 0],
                direction * stability_tilt,
                spin_length,
            )
            starts.append(_parameters_from_local_spins(tilted_spins))
    for seed in range(random_starts):
        rng = np.random.default_rng(1729 + seed)
        starts.append(np.concatenate((
            rng.uniform(0.0, np.pi, 4),
            rng.uniform(-np.pi, np.pi, 4),
        )))

    results = []
    for start in starts:
        results.append(minimize(
            lambda values: classical_energy_and_gradient(
                values,
                h_z,
                j_pm_ratio,
                j_pmpm_ratio=j_pmpm_ratio,
                spin_length=spin_length,
            ),
            start,
            method="BFGS",
            jac=True,
            options={"gtol": 1e-10, "maxiter": 2000},
        ))
    selected = min(results, key=lambda result: result.fun)

    # BFGS can stop with precision loss along an exact Goldstone direction.
    # A tiny residual torque then turns that zero mode into an artificial
    # imaginary Gamma-point frequency. Refine the stationarity equations
    # locally, retaining the result only when it lowers the residual without
    # increasing the classical energy.
    refined = least_squares(
        lambda values: classical_energy_and_gradient(
            values,
            h_z,
            j_pm_ratio,
            j_pmpm_ratio=j_pmpm_ratio,
            spin_length=spin_length,
        )[1],
        selected.x,
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=1000,
    )
    selected_energy, selected_gradient = classical_energy_and_gradient(
        selected.x,
        h_z,
        j_pm_ratio,
        j_pmpm_ratio=j_pmpm_ratio,
        spin_length=spin_length,
    )
    refined_energy, refined_gradient = classical_energy_and_gradient(
        refined.x,
        h_z,
        j_pm_ratio,
        j_pmpm_ratio=j_pmpm_ratio,
        spin_length=spin_length,
    )
    refinement_accepted = bool(
        np.isfinite(refined_energy)
        and np.all(np.isfinite(refined_gradient))
        and refined_energy <= selected_energy + 1e-10
        and np.max(np.abs(refined_gradient)) < np.max(np.abs(selected_gradient))
    )
    final_parameters = refined.x if refinement_accepted else selected.x

    relaxed_unit = _unit_vectors(final_parameters)
    final_energy, final_gradient = classical_energy_and_gradient(
        final_parameters,
        h_z,
        j_pm_ratio,
        j_pmpm_ratio=j_pmpm_ratio,
        spin_length=spin_length,
    )
    initial_energy, _ = classical_energy_and_gradient(
        initial_parameters,
        h_z,
        j_pm_ratio,
        j_pmpm_ratio=j_pmpm_ratio,
        spin_length=spin_length,
    )
    final_hessian, _ = classical_tangent_hessian(
        relaxed_unit,
        h_z,
        j_pm_ratio,
        j_pmpm_ratio=j_pmpm_ratio,
        spin_length=spin_length,
    )
    return {
        "unit_vectors_local": relaxed_unit,
        "spins_local": spin_length * relaxed_unit,
        "initial_energy": initial_energy,
        "energy": final_energy,
        "seed_min_classical_curvature": float(seed_curvatures[0]),
        "min_classical_curvature": float(np.linalg.eigvalsh(final_hessian)[0]),
        "stability_tilt_used": bool(stability_tilt_used),
        "max_torque": float(np.max(np.abs(final_gradient))),
        "max_spin_displacement": float(np.max(np.linalg.norm(
            relaxed_unit - initial_unit, axis=1
        ))),
        "optimizer_success": bool(
            selected.success or (refinement_accepted and refined.success)
        ),
        "optimizer_message": (
            f"BFGS: {selected.message}; stationarity refinement: {refined.message}"
            if refinement_accepted else str(selected.message)
        ),
        "stationarity_refinement_used": refinement_accepted,
    }


def quantization_rotations(unit_vectors_local):
    """Return R_mu^{ca}=tilde_axis_c dot local_axis_a from Sec. 4.2."""
    rotations = np.empty((4, 3, 3), dtype=float)
    reference = np.asarray([1.0, 1.0, 1.0])
    fallback = np.asarray([1.0, -1.0, 0.0])
    for mu in range(4):
        local_axes = np.stack((LOCAL_X[mu], LOCAL_Y[mu], LOCAL_Z[mu]))
        quant_z = unit_vectors_local[mu] @ local_axes
        quant_x = np.cross(quant_z, reference)
        if np.linalg.norm(quant_x) < 1e-10:
            quant_x = np.cross(quant_z, fallback)
        quant_x /= np.linalg.norm(quant_x)
        quant_y = np.cross(quant_z, quant_x)
        quant_y /= np.linalg.norm(quant_y)
        rotations[mu] = np.stack((quant_x, quant_y, quant_z)) @ local_axes.T
    return rotations


def spin_wave_matrix(
    k_reduced,
    unit_vectors_local,
    h_z,
    j_pm_ratio,
    *,
    j_pmpm_ratio=0.0,
    spin_length=0.5,
    regularization=1e-8,
):
    """Conventional bosonic BdG matrix equivalent to Eqs. (93)--(96)."""
    rotations = quantization_rotations(unit_vectors_local)
    local_exchange = np.empty((4, 4, 3, 3), dtype=float)
    for mu in range(4):
        for nu in range(4):
            local_exchange[mu, nu] = exchange_matrix(
                j_pm_ratio, j_pmpm_ratio, GAMMA[mu, nu]
            )
    rotated_exchange = np.einsum(
        "mai,mnij,nbj->mnab", rotations, local_exchange, rotations,
        optimize=True,
    )
    form_factor = np.zeros((4, 4), dtype=float)
    for mu in range(4):
        for nu in range(4):
            if mu != nu:
                form_factor[mu, nu] = 2.0 * np.cos(
                    np.pi * np.dot(k_reduced, E_MU[mu] - E_MU[nu])
                )
    exchange_k = form_factor[:, :, None, None] * rotated_exchange
    a_block = spin_length * 2.0 * np.einsum(
        "a,mnab,b->mn", MINUS_VECTOR, exchange_k, PLUS_VECTOR,
        optimize=True,
    )
    b_block = spin_length * 2.0 * np.einsum(
        "a,mnab,b->mn", MINUS_VECTOR, exchange_k, MINUS_VECTOR,
        optimize=True,
    )
    exchange_zero_zz = np.zeros((4, 4), dtype=float)
    for mu in range(4):
        for nu in range(4):
            if mu != nu:
                exchange_zero_zz[mu, nu] = 2.0 * rotated_exchange[mu, nu, 2, 2]
    diagonal = np.diag_indices(4)
    a_block[diagonal] -= spin_length * np.sum(exchange_zero_zz, axis=1)
    # The Zeeman term omitted from Eq. (95) contributes h_parallel n_mu.
    a_block[diagonal] += np.asarray(h_z) * rotations[:, 2, 2]
    a_block[diagonal] += regularization
    matrix = np.block([
        [a_block, b_block],
        [b_block.conj(), a_block.conj()],
    ])
    hermiticity_error = float(np.max(np.abs(matrix - matrix.conj().T)))
    if hermiticity_error > 1e-9:
        raise ValueError(f"Spin-wave matrix is not Hermitian: {hermiticity_error:.3e}")
    return matrix


def magnon_modes(
    k_reduced,
    unit_vectors_local,
    h_z,
    j_pm_ratio,
    *,
    j_pmpm_ratio=0.0,
    spin_length=0.5,
    regularization=1e-8,
):
    """Positive paraunitary modes of the Sec. 4.2 bosonic Hamiltonian."""
    matrix = spin_wave_matrix(
        k_reduced, unit_vectors_local, h_z, j_pm_ratio,
        j_pmpm_ratio=j_pmpm_ratio,
        spin_length=spin_length,
        regularization=regularization,
    )
    eigenvalues, eigenvectors = np.linalg.eig(BOSON_METRIC @ matrix)
    imaginary_error = float(np.max(np.abs(eigenvalues.imag)))
    if imaginary_error > 2e-5:
        raise ValueError(
            f"Dynamically unstable spin-wave state (imaginary frequency {imaginary_error:.3e})"
        )
    order = np.argsort(eigenvalues.real)
    positive = order[-4:]
    frequencies = np.maximum(eigenvalues.real[positive], 0.0)
    vectors = eigenvectors[:, positive]
    for band in range(4):
        norm = np.vdot(vectors[:, band], BOSON_METRIC @ vectors[:, band]).real
        if norm <= 1e-10:
            raise ValueError("A positive magnon mode has non-positive bosonic norm")
        vectors[:, band] /= np.sqrt(norm)
    band_order = np.argsort(frequencies)
    return frequencies[band_order], vectors[:, band_order], imaginary_error


def spin_wave_structure_factor(
    q_reduced,
    omega_ev,
    ordered_state,
    h_z,
    j_pm_ratio,
    *,
    j_pmpm_ratio=0.0,
    j_zz_ev,
    eta_ev,
    nk=8,
    spin_length=0.5,
    regularization=1e-8,
    interface_sublattices=(1, 2, 3),
):
    """Evaluate the one- and two-magnon DSSF in Eqs. (99)--(100)."""
    q_reduced = np.asarray(q_reduced, dtype=float)
    omega_ev = np.asarray(omega_ev, dtype=float)
    if q_reduced.shape[:-1] != omega_ev.shape or q_reduced.shape[-1] != 3:
        raise ValueError("q_reduced must have shape omega_ev.shape + (3,)")
    shape = omega_ev.shape
    q_flat = q_reduced.reshape(-1, 3)
    transfer = omega_ev.reshape(-1)
    interface = np.asarray(interface_sublattices, dtype=int)

    one_magnon = np.empty((len(interface), len(q_flat)), dtype=float)
    max_imaginary = 0.0
    for iq, (q_value, energy) in enumerate(zip(q_flat, transfer)):
        frequencies, vectors, imaginary = magnon_modes(
            q_value, ordered_state, h_z, j_pm_ratio,
            j_pmpm_ratio=j_pmpm_ratio,
            spin_length=spin_length, regularization=regularization,
        )
        max_imaginary = max(max_imaginary, imaginary)
        frequencies_ev = j_zz_ev * frequencies
        particle_weights = np.abs(vectors[:4]) ** 2
        one_magnon[:, iq] = 2.0 * spin_length * np.sum(
            particle_weights[interface]
            * lorentzian_delta(energy - frequencies_ev, eta_ev)[None, :],
            axis=1,
        )

    # Midpoint quadrature avoids the isolated Goldstone point at Gamma.  Its
    # thermodynamic integration weight is zero, while including it on a finite
    # grid gives a spurious regularization-dependent two-magnon contribution.
    k_grid = shifted_k_grid(nk)
    grid_frequencies = np.empty((len(k_grid), 4), dtype=float)
    grid_weights = np.empty((len(k_grid), 4, 4), dtype=float)
    for ik, k_value in enumerate(k_grid):
        frequencies, vectors, imaginary = magnon_modes(
            k_value, ordered_state, h_z, j_pm_ratio,
            j_pmpm_ratio=j_pmpm_ratio,
            spin_length=spin_length, regularization=regularization,
        )
        max_imaginary = max(max_imaginary, imaginary)
        grid_frequencies[ik] = j_zz_ev * frequencies
        grid_weights[ik] = np.abs(vectors[:4]) ** 2

    grid_indices = np.stack(np.unravel_index(
        np.arange(nk ** 3), (nk, nk, nk)
    ), axis=1)
    two_magnon = np.empty_like(one_magnon)
    for iq, (q_value, energy) in enumerate(zip(q_flat, transfer)):
        # The two-magnon BZ convolution is evaluated on the finite nk grid;
        # external q is mapped to the nearest periodic grid displacement.
        q_index = np.mod(np.rint(q_value * nk).astype(int), nk)
        partner_indices = np.mod(-grid_indices - q_index[None, :] - 1, nk)
        partner_flat = np.ravel_multi_index(partner_indices.T, (nk, nk, nk))
        pair_energy = (
            grid_frequencies[:, :, None]
            + grid_frequencies[partner_flat, None, :]
        )
        delta = lorentzian_delta(energy - pair_energy, eta_ev)
        for output_mu, mu in enumerate(interface):
            pair_weight = (
                grid_weights[:, mu, :, None]
                * grid_weights[partner_flat, mu, None, :]
            )
            two_magnon[output_mu, iq] = np.mean(
                np.sum(pair_weight * delta, axis=(1, 2))
            )

    bragg = np.zeros_like(one_magnon)
    reciprocal_zero = np.linalg.norm(q_flat - np.round(q_flat), axis=1) < 1e-10
    bragg[:, reciprocal_zero] = (
        nk ** 3 * spin_length ** 2
        * lorentzian_delta(transfer[reciprocal_zero], eta_ev)[None, :]
    )
    output_shape = (len(interface),) + shape
    one_magnon = one_magnon.reshape(output_shape)
    two_magnon = two_magnon.reshape(output_shape)
    bragg = bragg.reshape(output_shape)
    return {
        "one_magnon": one_magnon,
        "two_magnon": two_magnon,
        "bragg": bragg,
        "total": one_magnon + two_magnon + bragg,
        "max_imaginary_frequency_ratio": max_imaginary,
    }


def magnon_dispersion_bounds(
    k_reduced,
    ordered_state,
    h_z,
    j_pm_ratio,
    *,
    j_pmpm_ratio=0.0,
    j_zz_ev,
    spin_length=0.5,
    regularization=1e-8,
):
    """Return the minimum and maximum magnon energies on a momentum grid."""
    k_reduced = np.asarray(k_reduced, dtype=float)
    if (
        k_reduced.ndim < 2
        or k_reduced.shape[-1] != 3
        or k_reduced.size == 0
        or not np.all(np.isfinite(k_reduced))
    ):
        raise ValueError("k_reduced must contain finite three-component momenta")
    if not np.isfinite(j_zz_ev) or j_zz_ev <= 0.0:
        raise ValueError("j_zz_ev must be finite and positive")

    minimum = np.inf
    maximum = -np.inf
    max_imaginary = 0.0
    for k_value in k_reduced.reshape(-1, 3):
        frequencies, _, imaginary = magnon_modes(
            k_value,
            ordered_state,
            h_z,
            j_pm_ratio,
            j_pmpm_ratio=j_pmpm_ratio,
            spin_length=spin_length,
            regularization=regularization,
        )
        frequencies_ev = float(j_zz_ev) * frequencies
        minimum = min(minimum, float(np.min(frequencies_ev)))
        maximum = max(maximum, float(np.max(frequencies_ev)))
        max_imaginary = max(max_imaginary, imaginary)
    return {
        "minimum_ev": minimum,
        "maximum_ev": maximum,
        "max_imaginary_frequency_ratio": max_imaginary,
    }


def spin_wave_structure_factor_spectrum(
    q_reduced,
    omega_ev,
    ordered_state,
    h_z,
    j_pm_ratio,
    *,
    j_pmpm_ratio=0.0,
    j_zz_ev,
    eta_ev,
    nk=8,
    spin_length=0.5,
    regularization=1e-8,
    interface_sublattices=(1, 2, 3),
    component="total",
    omega_chunk_size=8,
):
    """Evaluate one selected DSSF component on a shared energy axis.

    Unlike :func:`spin_wave_structure_factor`, the magnon modes are calculated
    once per momentum and reused for every requested energy.  The returned
    intensity axes are ``(sublattice, omega) + q_reduced.shape[:-1]``.
    """
    components = ("one_magnon", "two_magnon", "bragg", "total")
    if component not in components:
        raise ValueError(f"Unknown spin-wave DSSF component: {component}")
    q_reduced = np.asarray(q_reduced, dtype=float)
    omega_ev = np.asarray(omega_ev, dtype=float)
    if (
        q_reduced.ndim < 1
        or q_reduced.shape[-1] != 3
        or not np.all(np.isfinite(q_reduced))
    ):
        raise ValueError("q_reduced must end with finite three-component momenta")
    if (
        omega_ev.ndim != 1
        or omega_ev.size == 0
        or not np.all(np.isfinite(omega_ev))
        or np.any(omega_ev < 0.0)
    ):
        raise ValueError("omega_ev must be a nonempty, finite, nonnegative axis")
    if not np.isfinite(eta_ev) or eta_ev <= 0.0:
        raise ValueError("eta_ev must be finite and positive")
    if int(nk) <= 0:
        raise ValueError("nk must be positive")
    if int(omega_chunk_size) <= 0:
        raise ValueError("omega_chunk_size must be positive")
    interface = np.asarray(interface_sublattices, dtype=int)
    if (
        interface.ndim != 1
        or len(interface) == 0
        or np.any((interface < 0) | (interface > 3))
    ):
        raise ValueError("interface_sublattices must contain indices between 0 and 3")

    q_shape = q_reduced.shape[:-1]
    q_flat = q_reduced.reshape(-1, 3)
    output_shape = (len(interface), len(omega_ev), len(q_flat))
    max_imaginary = 0.0
    pieces = []

    if component in ("one_magnon", "total"):
        one_magnon = np.empty(output_shape, dtype=float)
        for iq, q_value in enumerate(q_flat):
            frequencies, vectors, imaginary = magnon_modes(
                q_value,
                ordered_state,
                h_z,
                j_pm_ratio,
                j_pmpm_ratio=j_pmpm_ratio,
                spin_length=spin_length,
                regularization=regularization,
            )
            max_imaginary = max(max_imaginary, imaginary)
            frequencies_ev = j_zz_ev * frequencies
            particle_weights = np.abs(vectors[:4]) ** 2
            delta = lorentzian_delta(
                omega_ev[:, None] - frequencies_ev[None, :], eta_ev
            )
            one_magnon[:, :, iq] = 2.0 * spin_length * np.einsum(
                "mb,wb->mw", particle_weights[interface], delta,
                optimize=True,
            )
        pieces.append(one_magnon)

    if component in ("two_magnon", "total"):
        k_grid = shifted_k_grid(int(nk))
        grid_frequencies = np.empty((len(k_grid), 4), dtype=float)
        grid_weights = np.empty((len(k_grid), 4, 4), dtype=float)
        for ik, k_value in enumerate(k_grid):
            frequencies, vectors, imaginary = magnon_modes(
                k_value,
                ordered_state,
                h_z,
                j_pm_ratio,
                j_pmpm_ratio=j_pmpm_ratio,
                spin_length=spin_length,
                regularization=regularization,
            )
            max_imaginary = max(max_imaginary, imaginary)
            grid_frequencies[ik] = j_zz_ev * frequencies
            grid_weights[ik] = np.abs(vectors[:4]) ** 2
        grid_indices = np.stack(np.unravel_index(
            np.arange(int(nk) ** 3), (int(nk), int(nk), int(nk))
        ), axis=1)
        two_magnon = np.empty(output_shape, dtype=float)
        for iq, q_value in enumerate(q_flat):
            q_index = np.mod(np.rint(q_value * int(nk)).astype(int), int(nk))
            partner_indices = np.mod(
                -grid_indices - q_index[None, :] - 1, int(nk)
            )
            partner_flat = np.ravel_multi_index(
                partner_indices.T, (int(nk), int(nk), int(nk))
            )
            pair_energy = (
                grid_frequencies[:, :, None]
                + grid_frequencies[partner_flat, None, :]
            )
            for output_mu, mu in enumerate(interface):
                pair_weight = (
                    grid_weights[:, mu, :, None]
                    * grid_weights[partner_flat, mu, None, :]
                )
                for start in range(0, len(omega_ev), int(omega_chunk_size)):
                    stop = min(start + int(omega_chunk_size), len(omega_ev))
                    delta = lorentzian_delta(
                        omega_ev[start:stop, None, None, None]
                        - pair_energy[None, :, :, :],
                        eta_ev,
                    )
                    two_magnon[output_mu, start:stop, iq] = np.einsum(
                        "kij,wkij->w", pair_weight, delta, optimize=True
                    ) / len(k_grid)
        pieces.append(two_magnon)

    if component in ("bragg", "total"):
        bragg = np.zeros(output_shape, dtype=float)
        reciprocal_zero = (
            np.linalg.norm(q_flat - np.round(q_flat), axis=1) < 1e-10
        )
        profile = (
            int(nk) ** 3
            * spin_length ** 2
            * lorentzian_delta(omega_ev, eta_ev)
        )
        for iq in np.flatnonzero(reciprocal_zero):
            bragg[:, :, iq] = profile[None, :]
        pieces.append(bragg)

    intensity = pieces[0].copy()
    for values in pieces[1:]:
        intensity += values
    return {
        component: intensity.reshape(
            (len(interface), len(omega_ev)) + q_shape
        ),
        "max_imaginary_frequency_ratio": max_imaginary,
    }


def spin_wave_static_structure_factor(
    q_reduced,
    ordered_state,
    h_z,
    j_pm_ratio,
    *,
    j_pmpm_ratio=0.0,
    nk=8,
    spin_length=0.5,
    regularization=1e-8,
    sublattices=(0, 1, 2, 3),
):
    """Evaluate the energy-integrated spin structure factor from Eqs. (99)-(100).

    The broadened delta functions used by :func:`spin_wave_structure_factor`
    are integrated analytically.  This gives an energy-grid-independent static
    response and retains the one-magnon, two-magnon, and Bragg pieces.
    """
    q_reduced = np.asarray(q_reduced, dtype=float)
    if q_reduced.ndim < 1 or q_reduced.shape[-1] != 3:
        raise ValueError("q_reduced must end with three reduced coordinates")
    shape = q_reduced.shape[:-1]
    q_flat = q_reduced.reshape(-1, 3)
    selected_sublattices = np.asarray(sublattices, dtype=int)
    if (
        selected_sublattices.ndim != 1
        or len(selected_sublattices) == 0
        or np.any((selected_sublattices < 0) | (selected_sublattices > 3))
    ):
        raise ValueError("sublattices must contain indices between 0 and 3")

    one_magnon = np.empty((len(selected_sublattices), len(q_flat)), dtype=float)
    max_imaginary = 0.0
    for iq, q_value in enumerate(q_flat):
        _, vectors, imaginary = magnon_modes(
            q_value, ordered_state, h_z, j_pm_ratio,
            j_pmpm_ratio=j_pmpm_ratio,
            spin_length=spin_length, regularization=regularization,
        )
        max_imaginary = max(max_imaginary, imaginary)
        particle_weights = np.abs(vectors[:4]) ** 2
        one_magnon[:, iq] = (
            2.0 * spin_length
            * np.sum(particle_weights[selected_sublattices], axis=1)
        )

    k_grid = shifted_k_grid(nk)
    grid_weights = np.empty((len(k_grid), 4, 4), dtype=float)
    for ik, k_value in enumerate(k_grid):
        _, vectors, imaginary = magnon_modes(
            k_value, ordered_state, h_z, j_pm_ratio,
            j_pmpm_ratio=j_pmpm_ratio,
            spin_length=spin_length, regularization=regularization,
        )
        max_imaginary = max(max_imaginary, imaginary)
        grid_weights[ik] = np.abs(vectors[:4]) ** 2

    grid_indices = np.stack(np.unravel_index(
        np.arange(nk ** 3), (nk, nk, nk)
    ), axis=1)
    two_magnon = np.empty_like(one_magnon)
    for iq, q_value in enumerate(q_flat):
        q_index = np.mod(np.rint(q_value * nk).astype(int), nk)
        partner_indices = np.mod(-grid_indices - q_index[None, :] - 1, nk)
        partner_flat = np.ravel_multi_index(partner_indices.T, (nk, nk, nk))
        for output_mu, mu in enumerate(selected_sublattices):
            pair_weight = (
                grid_weights[:, mu, :, None]
                * grid_weights[partner_flat, mu, None, :]
            )
            two_magnon[output_mu, iq] = np.mean(
                np.sum(pair_weight, axis=(1, 2))
            )

    bragg = np.zeros_like(one_magnon)
    reciprocal_zero = np.linalg.norm(q_flat - np.round(q_flat), axis=1) < 1e-10
    bragg[:, reciprocal_zero] = nk ** 3 * spin_length ** 2

    output_shape = (len(selected_sublattices),) + shape
    one_magnon = one_magnon.reshape(output_shape)
    two_magnon = two_magnon.reshape(output_shape)
    bragg = bragg.reshape(output_shape)
    return {
        "one_magnon": one_magnon,
        "two_magnon": two_magnon,
        "bragg": bragg,
        "total": one_magnon + two_magnon + bragg,
        "max_imaginary_frequency_ratio": max_imaginary,
    }
