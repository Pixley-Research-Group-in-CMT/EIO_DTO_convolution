import numpy as np

from tto_structure_factor import (
    dynamic_structure_factor,
    eio_to_tto_reduced_momentum,
    lorentzian_delta,
    spinon_potential,
)


def test_lorentzian_normalization():
    x = np.linspace(-1.0, 1.0, 200_001)
    eta = 1e-2
    assert np.isclose(np.trapezoid(lorentzian_delta(x, eta), x), 1.0, rtol=7e-3)


def test_spinon_potential_is_reciprocal_periodic():
    s_minus = np.asarray([0.4 + 0.1j, -0.2j, 0.3, -0.1 + 0.35j])
    k = np.asarray([[0.13, 0.27, 0.39], [0.72, 0.04, 0.51]])
    reciprocal_shift = np.asarray([1.0, -2.0, 3.0])
    assert np.allclose(
        spinon_potential(k, 0.2, s_minus),
        spinon_potential(k + reciprocal_shift, 0.2, s_minus),
    )


def test_slab_to_tto_zero_and_shape():
    q = np.zeros((2, 4, 3))
    converted = eio_to_tto_reduced_momentum(q, 10.154)
    assert converted.shape == q.shape
    assert np.array_equal(converted, np.zeros_like(converted))


def test_dssf_components_are_nonnegative_and_sum():
    mean_field = {
        "j_pm_ratio": 0.1,
        "lambda_ratio": 1.0,
        "rho": 0.0,
        "condensed": False,
        "s_minus": np.full(4, 0.5 + 0.0j),
        "k_min": np.zeros(3),
    }
    q = np.asarray([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]])
    omega = np.asarray([1e-4, 2e-4])
    result = dynamic_structure_factor(
        q, omega, mean_field, j_zz_ev=1.7e-4, eta_ev=2e-5, nk=4
    )
    assert result["total"].shape == (3, 2)
    assert all(np.all(values >= 0.0) for values in result.values())
    assert np.allclose(
        result["total"],
        result["continuum"]
        + result["one_condensate"]
        + result["two_condensate"],
    )
