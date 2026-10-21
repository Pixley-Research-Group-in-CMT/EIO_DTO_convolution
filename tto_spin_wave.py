"""Linear spin-wave DSSF from Sec. 4.2 of ``papers/gMFT.pdf``.

For every field direction, a configuration aligned with the complete local
field in Eq. (1) of ``papers/EIO_TTO.pdf`` seeds a multistart minimization of
the physical classical spin Hamiltonian.  The input field coordinate is the
bare Zeeman energy ``mu_B |B| / J_zz``; the g factors and the quadratic
``delta_1``/``delta_2`` couplings are applied only after projection onto the
local axes.  The BdG expansion and Eqs. (99)--(100) are evaluated about that
stable state.  The bond-dependent J_pmpm tensor follows the positive
``+J_pmpm (gamma S+ S+ + gamma* S- S-)`` convention in Eqs. (1)--(3).
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
ONE_MAGNON_MODE_COMPONENTS = tuple(
    f"one_magnon_{mode}" for mode in range(1, 5)
)
SPIN_WAVE_COMPONENTS = (
    "one_magnon",
    *ONE_MAGNON_MODE_COMPONENTS,
    "two_magnon",
    "total",
)


def one_magnon_mode_index(component):
    """Return the zero-based energy-ordered band selected by a component."""
    if component in ONE_MAGNON_MODE_COMPONENTS:
        return ONE_MAGNON_MODE_COMPONENTS.index(component)
    return None


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


def local_magnetic_field(
    mu_b_b_over_j_zz,
    theta_degrees=90.0,
    phi_degrees=0.0,
):
    """Return the bare local components ``mu_B B . (x,y,z)_mu / J_zz``."""
    mu_b_b_over_j_zz = float(mu_b_b_over_j_zz)
    if not np.isfinite(mu_b_b_over_j_zz) or mu_b_b_over_j_zz < 0.0:
        raise ValueError("mu_B |B| / J_zz must be finite and nonnegative")
    direction = magnetic_field_direction(theta_degrees, phi_degrees)
    projections = np.stack((
        LOCAL_X @ direction,
        LOCAL_Y @ direction,
        LOCAL_Z @ direction,
    ), axis=1)
    return mu_b_b_over_j_zz * projections


def local_zeeman_components(
    mu_b_b_over_j_zz,
    theta_degrees=90.0,
    phi_degrees=0.0,
):
    """Return the bare local-z components retained for legacy diagnostics."""
    return local_magnetic_field(
        mu_b_b_over_j_zz,
        theta_degrees=theta_degrees,
        phi_degrees=phi_degrees,
    )[:, 2]


def local_zeeman_field(
    mu_b_b_over_j_zz,
    theta_degrees=90.0,
    phi_degrees=0.0,
    g_z=10.0,
    g_perp=0.0,
    delta1_j_zz=0.0,
    delta2_j_zz=0.0,
):
    """Return the effective local field ``b_mu/J_zz`` in Eq. (1).

    The input is ``mu_B |B| / J_zz``.  ``delta1_j_zz`` and
    ``delta2_j_zz`` mean ``delta_1 J_zz`` and ``delta_2 J_zz`` and are
    dimensionless.  If the deltas are specified in inverse meV, callers must
    multiply them by ``J_zz`` in meV.  The returned columns are the
    coefficients of ``-(S_x, S_y, S_z)``.
    """
    coefficients = np.asarray(
        (float(g_z), float(g_perp), float(delta1_j_zz), float(delta2_j_zz))
    )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("g factors and delta_i J_zz must be finite")
    if g_z < 0.0 or g_perp < 0.0:
        raise ValueError("g_z and g_perp must be nonnegative")
    bare = local_magnetic_field(
        mu_b_b_over_j_zz,
        theta_degrees=theta_degrees,
        phi_degrees=phi_degrees,
    )
    h_x, h_y, h_z = bare.T
    return np.column_stack((
        g_perp * h_x
        + delta1_j_zz * h_x * h_z
        + delta2_j_zz * (h_y ** 2 - h_x ** 2),
        g_perp * h_y
        + delta1_j_zz * h_y * h_z
        + 2.0 * delta2_j_zz * h_x * h_y,
        g_z * h_z,
    ))


def _as_local_field(field):
    """Coerce a legacy Ising field or a full local field to shape ``(4, 3)``."""
    field = np.asarray(field, dtype=float)
    if field.shape == (4,):
        local_field = np.zeros((4, 3), dtype=float)
        local_field[:, 2] = field
    elif field.shape == (4, 3):
        local_field = field
    else:
        raise ValueError(
            "local field must have shape (4,) for Ising-only coupling or "
            "(4, 3) for (x, y, z) coupling"
        )
    if not np.all(np.isfinite(local_field)):
        raise ValueError("local field components must be finite")
    return local_field


def zeeman_minimizing_spins(h_z, spin_length=0.5, zero_tolerance=1e-12):
    """Return independent spins minimizing the complete local Zeeman term.

    A legacy four-component input is interpreted as an Ising-only field.  A
    ``(4, 3)`` input includes the transverse coupling in Eq. (1).  A zero
    local field leaves that sublattice unconstrained; ``+z`` is selected as a
    deterministic representative.
    """
    local_field = _as_local_field(h_z)
    norms = np.linalg.norm(local_field, axis=1)
    spins = np.zeros_like(local_field)
    nonzero = norms > abs(float(zero_tolerance))
    spins[nonzero] = (
        float(spin_length) * local_field[nonzero] / norms[nonzero, None]
    )
    spins[~nonzero, 2] = float(spin_length)
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
    """Local Cartesian bond exchange in units of ``J_zz``.

    ``j_pmpm_ratio`` uses the note's positive
    ``+J_pmpm (gamma S+ S+ + gamma* S- S-)`` convention.  ``gamma`` is the
    bond phase in Eq. (2).  The returned real tensor acts on
    ``(S_x, S_y, S_z)`` in the two local frames.
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
    local_field = _as_local_field(h_z)
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
    energy -= spin_length * np.einsum("ma,ma->", local_field, unit)
    vector_gradient -= spin_length * local_field
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
    local_field = _as_local_field(h_z)
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

    vector_gradient -= spin_length * local_field
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
    local_field = _as_local_field(h_z)
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
    # The full Eq. (1) field term contributes the component of the effective
    # field parallel to the relaxed ordered spin, b_mu dot e_tilde_z, to the
    # boson number term.  Linear transverse terms vanish at stationarity.
    a_block[diagonal] += np.einsum(
        "ma,ma->m", local_field, rotations[:, 2, :]
    )
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
    one_magnon_modes = np.empty(
        (4, len(interface), len(q_flat)), dtype=float
    )
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
        mode_intensity = (
            2.0
            * spin_length
            * particle_weights[interface]
            * lorentzian_delta(energy - frequencies_ev, eta_ev)[None, :]
        )
        one_magnon_modes[:, :, iq] = mode_intensity.T
        one_magnon[:, iq] = np.sum(mode_intensity, axis=1)

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

    output_shape = (len(interface),) + shape
    one_magnon = one_magnon.reshape(output_shape)
    one_magnon_mode_result = {
        component: one_magnon_modes[mode].reshape(output_shape)
        for mode, component in enumerate(ONE_MAGNON_MODE_COMPONENTS)
    }
    two_magnon = two_magnon.reshape(output_shape)
    return {
        "one_magnon": one_magnon,
        **one_magnon_mode_result,
        "two_magnon": two_magnon,
        "total": one_magnon + two_magnon,
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
    mode_index=None,
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
    if mode_index is not None:
        if (
            isinstance(mode_index, (bool, np.bool_))
            or int(mode_index) != mode_index
            or not 0 <= int(mode_index) < 4
        ):
            raise ValueError("mode_index must be None or an integer from 0 to 3")
        mode_index = int(mode_index)

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
        if mode_index is not None:
            frequencies_ev = frequencies_ev[mode_index:mode_index + 1]
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
    if component not in SPIN_WAVE_COMPONENTS:
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

    selected_mode = one_magnon_mode_index(component)
    if component in ("one_magnon", "total") or selected_mode is not None:
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
            if selected_mode is None:
                one_magnon[:, :, iq] = 2.0 * spin_length * np.einsum(
                    "mb,wb->mw", particle_weights[interface], delta,
                    optimize=True,
                )
            else:
                one_magnon[:, :, iq] = (
                    2.0
                    * spin_length
                    * particle_weights[interface, selected_mode, None]
                    * delta[None, :, selected_mode]
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
    response and retains the one- and two-magnon pieces. The elastic Bragg
    contribution is excluded from every spin-wave structure-factor workflow.
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
    one_magnon_modes = np.empty(
        (4, len(selected_sublattices), len(q_flat)), dtype=float
    )
    max_imaginary = 0.0
    for iq, q_value in enumerate(q_flat):
        _, vectors, imaginary = magnon_modes(
            q_value, ordered_state, h_z, j_pm_ratio,
            j_pmpm_ratio=j_pmpm_ratio,
            spin_length=spin_length, regularization=regularization,
        )
        max_imaginary = max(max_imaginary, imaginary)
        particle_weights = np.abs(vectors[:4]) ** 2
        mode_intensity = 2.0 * spin_length * particle_weights[selected_sublattices]
        one_magnon_modes[:, :, iq] = mode_intensity.T
        one_magnon[:, iq] = np.sum(mode_intensity, axis=1)

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

    output_shape = (len(selected_sublattices),) + shape
    one_magnon = one_magnon.reshape(output_shape)
    one_magnon_mode_result = {
        component: one_magnon_modes[mode].reshape(output_shape)
        for mode, component in enumerate(ONE_MAGNON_MODE_COMPONENTS)
    }
    two_magnon = two_magnon.reshape(output_shape)
    return {
        "one_magnon": one_magnon,
        **one_magnon_mode_result,
        "two_magnon": two_magnon,
        "total": one_magnon + two_magnon,
        "max_imaginary_frequency_ratio": max_imaginary,
    }
