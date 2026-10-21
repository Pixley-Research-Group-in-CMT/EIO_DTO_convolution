import numpy as np
import pytest

from tto_spin_wave import (
    GAMMA,
    ONE_MAGNON_MODE_COMPONENTS,
    classical_energy_and_gradient,
    classical_tangent_hessian,
    exchange_matrix,
    local_magnetic_field,
    local_zeeman_components,
    local_zeeman_field,
    magnon_dispersion_bounds,
    magnon_modes,
    one_magnon_mode_index,
    quantization_rotations,
    relax_ordered_state,
    spin_wave_static_structure_factor,
    spin_wave_matrix,
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


def test_full_local_field_implements_every_eq_1_field_term():
    field = 0.37
    g_z = 9.7
    g_perp = 0.42
    delta1_j_zz = -0.13
    delta2_j_zz = 0.08
    bare = local_magnetic_field(
        field, theta_degrees=73.0, phi_degrees=21.0
    )
    full = local_zeeman_field(
        field,
        theta_degrees=73.0,
        phi_degrees=21.0,
        g_z=g_z,
        g_perp=g_perp,
        delta1_j_zz=delta1_j_zz,
        delta2_j_zz=delta2_j_zz,
    )
    h_x, h_y, h_z = bare.T
    expected = np.column_stack((
        g_perp * h_x
        + delta1_j_zz * h_x * h_z
        + delta2_j_zz * (h_y ** 2 - h_x ** 2),
        g_perp * h_y
        + delta1_j_zz * h_y * h_z
        + 2.0 * delta2_j_zz * h_x * h_y,
        g_z * h_z,
    ))
    assert full.shape == (4, 3)
    assert np.allclose(full, expected)
    assert np.allclose(np.linalg.norm(bare, axis=1), field)


def test_full_field_zeeman_seed_is_parallel_to_each_local_field():
    h_local = local_zeeman_field(
        0.2, phi_degrees=15.0, g_z=10.0, g_perp=0.8,
        delta1_j_zz=0.12, delta2_j_zz=0.09,
    )
    spins = zeeman_minimizing_spins(h_local, spin_length=0.5)
    assert np.allclose(np.linalg.norm(spins, axis=1), 0.5)
    assert np.allclose(
        np.einsum("ma,ma->m", h_local, spins),
        0.5 * np.linalg.norm(h_local, axis=1),
    )


def test_transverse_field_enters_quadratic_magnon_number_term():
    h_local = local_zeeman_field(
        0.4, theta_degrees=82.0, phi_degrees=19.0,
        g_z=10.0, g_perp=0.8,
        delta1_j_zz=0.12, delta2_j_zz=0.09,
    )
    ordered_state = 2.0 * _polarized_test_state()
    full_matrix = spin_wave_matrix(
        np.asarray([0.13, 0.21, -0.08]), ordered_state, h_local, 0.2
    )
    ising_matrix = spin_wave_matrix(
        np.asarray([0.13, 0.21, -0.08]), ordered_state,
        h_local[:, 2], 0.2,
    )
    rotations = quantization_rotations(ordered_state)
    expected_shift = np.einsum(
        "ma,ma->m",
        np.column_stack((h_local[:, :2], np.zeros(4))),
        rotations[:, 2, :],
    )
    difference = full_matrix - ising_matrix
    assert np.allclose(np.diag(difference)[:4], expected_shift)
    assert np.allclose(np.diag(difference)[4:], expected_shift)
    assert np.allclose(difference - np.diag(np.diag(difference)), 0.0)


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
    h_z = local_zeeman_field(
        0.3, theta_degrees=81.0, phi_degrees=13.0,
        g_z=10.0, g_perp=0.7,
        delta1_j_zz=0.12, delta2_j_zz=0.09,
    )
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
        result["one_magnon"] + result["two_magnon"],
    )
    assert "bragg" not in result
    assert np.allclose(
        result["one_magnon"],
        sum(result[component] for component in ONE_MAGNON_MODE_COMPONENTS),
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

    one_magnon = spin_wave_structure_factor_spectrum(
        q, omega, relaxed["unit_vectors_local"], h_z, 0.2,
        j_pmpm_ratio=0.03, j_zz_ev=1.7e-4, eta_ev=2e-5, nk=2,
        interface_sublattices=(0, 1, 2, 3), component="one_magnon",
    )["one_magnon"]
    two_magnon = spin_wave_structure_factor_spectrum(
        q, omega, relaxed["unit_vectors_local"], h_z, 0.2,
        j_pmpm_ratio=0.03, j_zz_ev=1.7e-4, eta_ev=2e-5, nk=2,
        interface_sublattices=(0, 1, 2, 3), component="two_magnon",
    )["two_magnon"]
    assert np.allclose(spectrum["total"], one_magnon + two_magnon)
    mode_sum = sum(
        spin_wave_structure_factor_spectrum(
            q, omega, relaxed["unit_vectors_local"], h_z, 0.2,
            j_pmpm_ratio=0.03, j_zz_ev=1.7e-4, eta_ev=2e-5, nk=2,
            interface_sublattices=(0, 1, 2, 3), component=component,
        )[component]
        for component in ONE_MAGNON_MODE_COMPONENTS
    )
    assert np.allclose(one_magnon, mode_sum)
    assert one_magnon_mode_index("one_magnon_1") == 0
    assert one_magnon_mode_index("one_magnon_4") == 3
    with pytest.raises(ValueError, match="Unknown spin-wave DSSF component"):
        spin_wave_structure_factor_spectrum(
            q, omega, relaxed["unit_vectors_local"], h_z, 0.2,
            j_pmpm_ratio=0.03, j_zz_ev=1.7e-4, eta_ev=2e-5,
            component="bragg",
        )

    bounds = magnon_dispersion_bounds(
        q, relaxed["unit_vectors_local"], h_z, 0.2,
        j_pmpm_ratio=0.03, j_zz_ev=1.7e-4,
    )
    assert 0.0 <= bounds["minimum_ev"] <= bounds["maximum_ev"]
    lowest_bounds = magnon_dispersion_bounds(
        q, relaxed["unit_vectors_local"], h_z, 0.2,
        j_pmpm_ratio=0.03, j_zz_ev=1.7e-4, mode_index=0,
    )
    highest_bounds = magnon_dispersion_bounds(
        q, relaxed["unit_vectors_local"], h_z, 0.2,
        j_pmpm_ratio=0.03, j_zz_ev=1.7e-4, mode_index=3,
    )
    assert lowest_bounds["minimum_ev"] >= bounds["minimum_ev"]
    assert highest_bounds["maximum_ev"] <= bounds["maximum_ev"]
    assert lowest_bounds["maximum_ev"] <= highest_bounds["maximum_ev"]


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
        result["one_magnon"] + result["two_magnon"],
    )
    assert np.allclose(
        result["one_magnon"],
        sum(result[component] for component in ONE_MAGNON_MODE_COMPONENTS),
    )
    assert "bragg" not in result
