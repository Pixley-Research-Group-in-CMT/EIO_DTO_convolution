import numpy as np
import pytest

import plot_swt_magnon_dispersion as dispersion


def test_evaluate_magnon_bands_returns_sorted_mev_energies(monkeypatch):
    def fake_modes(momentum, *args, **kwargs):
        frequencies = np.asarray([0.1, 0.2, 0.3, 0.4]) + momentum[0]
        return frequencies, np.zeros((8, 4)), 3e-8

    monkeypatch.setattr(dispersion, "magnon_modes", fake_modes)
    q_points = np.asarray([[0.0, 0.0, 0.0], [0.05, -0.1, 0.2]])
    energies, max_imaginary = dispersion.evaluate_magnon_bands(
        q_points,
        np.zeros((4, 3)),
        np.zeros(4),
        0.2,
        j_pmpm_ratio=0.03,
        j_zz_ev=2e-4,
    )

    assert energies.shape == (4, 2)
    assert np.all(np.diff(energies, axis=0) >= 0.0)
    assert np.allclose(energies[:, 0], [0.02, 0.04, 0.06, 0.08])
    assert np.isclose(max_imaginary, 3e-8)


def test_dispersion_validation_and_shared_scale():
    with pytest.raises(ValueError, match="shape"):
        dispersion.evaluate_magnon_bands(
            np.zeros((2, 2)), np.zeros((4, 3)), np.zeros(4), 0.2,
            j_pmpm_ratio=0.0, j_zz_ev=2e-4,
        )
    norm = dispersion.shared_energy_norm(
        np.asarray([0.0, 0.1, np.nan, 0.35])
    )
    assert norm.vmin == 0.0
    assert np.isclose(norm.vmax, 0.35)
