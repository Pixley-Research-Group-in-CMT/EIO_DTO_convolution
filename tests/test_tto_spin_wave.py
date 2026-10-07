import numpy as np

from tto_spin_wave import (
    GAMMA,
    classical_energy_and_gradient,
    classical_tangent_hessian,
    exchange_matrix,
    local_zeeman_components,
    magnon_dispersion_bounds,
    magnon_modes,
    relax_ordered_state,
    spin_wave_static_structure_factor,
    spin_wave_structure_factor,
    spin_wave_structure_factor_spectrum,
    zeeman_minimizing_spins,
)


def _polarized_test_state():
    spins = 0.5 * np.asarray([
        [0.6, -0.7, 0.3872983346],
        [0.6, -0.7, -0.3872983346],
        [0.6, -0.7, -0.3872983346],
        [0.6, -0.7, 0.3872983346],
    ])
    spins *= (0.5 / np.linalg.norm(spins, axis=1))[:, None]
    return spins


def test_generated_in_plane_field_matches_previous_geometry():
    field = 0.15
    h_z = local_zeeman_components(field, theta_degrees=90.0, phi_degrees=0.0)
    expected = field * np.asarray([
        0.0, -np.sqrt(2.0) / 3.0, -np.sqrt(2.0) / 3.0,
        2.0 * np.sqrt(2.0) / 3.0,
    ])
    assert np.allclose(h_z, expected, atol=1e-14)


def test_zeeman_seed_has_fixed_length_and_minimal_energy():
    h_z = local_zeeman_components(0.2, phi_degrees=15.0)
    spins = zeeman_minimizing_spins(h_z, spin_length=0.5)
    assert np.allclose(np.linalg.norm(spins, axis=1), 0.5)
    assert np.isclose(-np.dot(h_z, spins[:, 2]), -0.5 * np.sum(np.abs(h_z)))

    zero_field_spins = zeeman_minimizing_spins(np.zeros(4), spin_length=0.5)
    assert np.allclose(zero_field_spins[:, 2], 0.5)


def test_stability_guided_tilt_escapes_zeeman_pole():
    h_z = local_zeeman_components(0.5, theta_degrees=90.0, phi_degrees=17.0)
    seed = zeeman_minimizing_spins(h_z, spin_length=0.5)
    seed_hessian, _ = classical_tangent_hessian(
        2.0 * seed, h_z, 0.2, j_pmpm_ratio=0.1
    )
    assert np.linalg.eigvalsh(seed_hessian)[0] < 0.0

    relaxed = relax_ordered_state(
        seed, h_z, 0.2, j_pmpm_ratio=0.1, random_starts=0
    )
    assert relaxed["stability_tilt_used"]
    assert relaxed["energy"] < relaxed["initial_energy"]
    assert relaxed["min_classical_curvature"] > 0.0
    assert np.max(np.linalg.norm(relaxed["spins_local"][:, :2], axis=1)) > 1e-2

    _, _, imaginary = magnon_modes(
        np.zeros(3), relaxed["unit_vectors_local"], h_z, 0.2,
        j_pmpm_ratio=0.1,
    )
    assert imaginary < 2e-5


def test_stationarity_refinement_preserves_high_field_goldstone_mode():
    h_z = local_zeeman_components(6.0, theta_degrees=90.0, phi_degrees=162.1)
    seed = zeeman_minimizing_spins(h_z, spin_length=0.5)
    relaxed = relax_ordered_state(
        seed, h_z, 0.5, random_starts=3, stability_tilt=0.01
    )

    assert relaxed["stationarity_refinement_used"]
    assert relaxed["max_torque"] < 1e-12
    assert relaxed["min_classical_curvature"] > -1e-10
    _, _, imaginary = magnon_modes(
        np.zeros(3), relaxed["unit_vectors_local"], h_z, 0.5
    )
    assert imaginary < 2e-5


def test_classical_analytic_gradient():
    parameters = np.asarray([0.7, 1.1, 1.7, 2.2, -0.4, 0.2, 1.0, -1.3])
    h_z = np.asarray([0.0, -0.1, -0.1, 0.2])
    _, analytic = classical_energy_and_gradient(
        parameters, h_z, 0.2, j_pmpm_ratio=0.035
    )
    numeric = np.empty(8)
    step = 1e-6
    for index in range(8):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        numeric[index] = (
            classical_energy_and_gradient(
                plus, h_z, 0.2, j_pmpm_ratio=0.035
            )[0]
            - classical_energy_and_gradient(
                minus, h_z, 0.2, j_pmpm_ratio=0.035
            )[0]
        ) / (2.0 * step)
    assert np.allclose(analytic, numeric, atol=1e-8)


def test_j_pmpm_bond_exchange_follows_gamma_convention():
    j_pm = 0.2
    j_pmpm = 0.05
    gamma = GAMMA[0, 2]
    exchange = exchange_matrix(j_pm, j_pmpm, gamma)
    expected = np.asarray([
        [-2.0 * j_pm + 2.0 * j_pmpm * gamma.real,
         -2.0 * j_pmpm * gamma.imag, 0.0],
        [-2.0 * j_pmpm * gamma.imag,
         -2.0 * j_pm - 2.0 * j_pmpm * gamma.real, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.allclose(exchange, expected)
    assert np.allclose(exchange, exchange.T)

    spin_i = np.asarray([0.17, -0.23, 0.31])
    spin_j = np.asarray([-0.29, 0.11, -0.07])
    plus_i = spin_i[0] + 1.0j * spin_i[1]
    minus_i = np.conj(plus_i)
    plus_j = spin_j[0] + 1.0j * spin_j[1]
    minus_j = np.conj(plus_j)
    ladder_energy = (
        spin_i[2] * spin_j[2]
        - j_pm * (plus_i * minus_j + minus_i * plus_j)
        + j_pmpm * (gamma * plus_i * plus_j + np.conj(gamma) * minus_i * minus_j)
    )
    assert np.isclose(spin_i @ exchange @ spin_j, ladder_energy.real)
    assert np.isclose(ladder_energy.imag, 0.0)


def test_zero_j_pmpm_recovers_isotropic_transverse_exchange():
    expected = np.diag([-0.4, -0.4, 1.0])
    for gamma in GAMMA[GAMMA != 0.0]:
        assert np.allclose(exchange_matrix(0.2, 0.0, gamma), expected)


def test_relaxation_lowers_energy_and_modes_are_real():
    h_z = np.asarray([0.0, -0.12, -0.12, 0.24])
    relaxed = relax_ordered_state(
        _polarized_test_state(), h_z, 0.2,
        j_pmpm_ratio=0.03, random_starts=1,
    )
    assert relaxed["energy"] <= relaxed["initial_energy"] + 1e-10
    assert relaxed["max_torque"] < 1e-7
    frequencies, vectors, imaginary = magnon_modes(
        np.asarray([0.125, 0.25, 0.375]),
        relaxed["unit_vectors_local"], h_z, 0.2,
        j_pmpm_ratio=0.03,
    )
    assert np.all(frequencies >= 0.0)
    assert vectors.shape == (8, 4)
    assert imaginary < 2e-5


def test_spin_wave_dssf_components_sum():
    h_z = np.asarray([0.0, -0.12, -0.12, 0.24])
    relaxed = relax_ordered_state(
        _polarized_test_state(), h_z, 0.2,
        j_pmpm_ratio=0.03, random_starts=1,
    )
    q = np.asarray([[0.1, 0.2, 0.3], [0.25, -0.125, 0.0]])
    omega = np.asarray([1e-4, 2e-4])
    result = spin_wave_structure_factor(
        q, omega, relaxed["unit_vectors_local"], h_z, 0.2,
        j_pmpm_ratio=0.03, j_zz_ev=1.7e-4, eta_ev=2e-5, nk=2,
        interface_sublattices=(0, 1, 2, 3),
    )
    assert result["total"].shape == (4, 2)
    assert np.all(result["total"] >= 0.0)
    assert np.allclose(
        result["total"],
        result["one_magnon"] + result["two_magnon"] + result["bragg"],
    )


def test_spin_wave_dssf_spectrum_reuses_a_shared_energy_axis():
    h_z = np.asarray([0.0, -0.12, -0.12, 0.24])
    relaxed = relax_ordered_state(
        _polarized_test_state(), h_z, 0.2,
        j_pmpm_ratio=0.03, random_starts=1,
    )
    q = np.asarray([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    omega = np.asarray([1e-4, 2e-4])
    spectrum = spin_wave_structure_factor_spectrum(
        q, omega, relaxed["unit_vectors_local"], h_z, 0.2,
        j_pmpm_ratio=0.03, j_zz_ev=1.7e-4, eta_ev=2e-5, nk=2,
        interface_sublattices=(0, 1, 2, 3), component="total",
    )
    assert spectrum["total"].shape == (4, 2, 2)
    for energy_index, energy in enumerate(omega):
        fixed = spin_wave_structure_factor(
            q, np.full(len(q), energy),
            relaxed["unit_vectors_local"], h_z, 0.2,
            j_pmpm_ratio=0.03, j_zz_ev=1.7e-4, eta_ev=2e-5, nk=2,
            interface_sublattices=(0, 1, 2, 3),
        )
        assert np.allclose(spectrum["total"][:, energy_index], fixed["total"])

    bounds = magnon_dispersion_bounds(
        q, relaxed["unit_vectors_local"], h_z, 0.2,
        j_pmpm_ratio=0.03, j_zz_ev=1.7e-4,
    )
    assert 0.0 <= bounds["minimum_ev"] <= bounds["maximum_ev"]


def test_static_spin_wave_structure_factor_components_sum():
    h_z = np.asarray([0.0, -0.12, -0.12, 0.24])
    relaxed = relax_ordered_state(
        _polarized_test_state(), h_z, 0.2,
        j_pmpm_ratio=0.03, random_starts=1,
    )
    q = np.asarray([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    result = spin_wave_static_structure_factor(
        q, relaxed["unit_vectors_local"], h_z, 0.2,
        j_pmpm_ratio=0.03, nk=2, sublattices=(0, 1, 2, 3),
    )
    assert result["total"].shape == (4, 2)
    assert np.all(result["total"] >= 0.0)
    assert np.allclose(
        result["total"],
        result["one_magnon"] + result["two_magnon"] + result["bragg"],
    )
    assert np.allclose(result["bragg"][:, 0], 0.0)
    assert np.all(result["bragg"][:, 1] > 0.0)
