import numpy as np
import pytest

from tto_surface_projection import (
    lift_q_perp,
    q_perp_reduced_offsets,
    q_perp_samples_for_mode,
    surface_projected_dynamic_structure_factor,
    surface_projected_static_structure_factor,
    validate_q_perp_sampling,
)


def test_q_perp_grid_spans_one_period_and_contains_zero():
    offsets = q_perp_reduced_offsets(8)
    assert np.allclose(offsets, np.arange(-4, 4) / 8)
    assert np.count_nonzero(np.isclose(offsets, 0.0)) == 1
    assert np.allclose(q_perp_samples_for_mode("slice", 8), [0.0])

    q_parallel = np.asarray([[0.2, -0.1, 0.3]])
    lifted = lift_q_perp(q_parallel, 4)
    assert lifted.shape == (4, 1, 3)
    assert np.allclose(
        lifted[:, 0] - q_parallel[0],
        q_perp_reduced_offsets(4)[:, None] * np.ones(3),
    )


def test_slice_mode_delegates_to_original_solver_unchanged():
    calls = []

    def solver(q, omega, scale):
        calls.append((q, omega))
        return {"total": scale * omega[None, ...]}

    q = np.zeros((2, 3))
    omega = np.asarray([0.1, 0.2])
    result = surface_projected_dynamic_structure_factor(
        solver, q, omega, 3.0, q_perp_mode="slice", n_q_perp=8
    )
    assert len(calls) == 1
    assert calls[0][0] is q
    assert calls[0][1] is omega
    assert np.allclose(result["total"], [[0.3, 0.6]])


def test_integrated_mode_averages_the_bulk_response_over_q_perp():
    def solver(q, omega):
        intensity = q[..., 0] ** 2 + omega
        return {
            "total": intensity[None, ...],
            "max_imaginary_frequency_ratio": 2e-8,
        }

    q = np.zeros((2, 3))
    omega = np.asarray([0.1, 0.2])
    result = surface_projected_dynamic_structure_factor(
        solver, q, omega, q_perp_mode="integrated", n_q_perp=4
    )
    offsets = q_perp_reduced_offsets(4)
    expected = np.mean(offsets ** 2) + omega
    assert result["total"].shape == (1, 2)
    assert np.allclose(result["total"], expected[None, :])
    assert result["max_imaginary_frequency_ratio"] == 2e-8


def test_static_integrated_mode_averages_the_bulk_response_over_q_perp():
    def solver(q, scale):
        return {
            "total": scale * q[..., 0][None, ...] ** 2,
            "max_imaginary_frequency_ratio": 3e-8,
        }

    q = np.zeros((2, 3))
    result = surface_projected_static_structure_factor(
        solver, q, 2.0, q_perp_mode="integrated", n_q_perp=4
    )
    expected = 2.0 * np.mean(q_perp_reduced_offsets(4) ** 2)
    assert result["total"].shape == (1, 2)
    assert np.allclose(result["total"], expected)
    assert result["max_imaginary_frequency_ratio"] == 3e-8


def test_q_perp_settings_are_validated():
    with pytest.raises(ValueError, match="one of"):
        validate_q_perp_sampling("average", 8)
    with pytest.raises(ValueError, match="positive integer"):
        validate_q_perp_sampling("slice", 0)
    with pytest.raises(ValueError, match="requires"):
        validate_q_perp_sampling("integrated", 1)
