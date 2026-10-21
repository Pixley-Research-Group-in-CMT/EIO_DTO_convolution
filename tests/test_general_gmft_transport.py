import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from gmft_transport_job import (
    configured_output_directories,
    nominal_phi_angles,
    resistivity_filename,
    select_winners,
    validate,
)
from tto_general_gmft import dynamic_structure_factor, transverse_weight, spectrum
from tto_structure_factor import dynamic_structure_factor as scalar_dssf, reduced_k_grid, E_MU, spinon_potential, lorentzian_delta


def saddle(condensed=False):
    return dict(j_pm_ratio=.1, lambda_ratio=.3 if condensed else 1.,
                rho=.2 if condensed else 0., condensed=condensed,
                s_minus=np.full(4, .5 + 0j), b_plus=np.zeros(4, complex),
                k_min=np.zeros(3),
                condensate_k=np.zeros((1 if condensed else 0, 3)),
                condensate_diagonal=np.full((1 if condensed else 0, 2), .2))


@pytest.mark.parametrize("condensed", [False, True])
def test_general_response_reduces_to_scalar(condensed):
    mf = saddle(condensed)
    # Mesh-aligned transfers keep a condensed finite-grid saddle stable.
    q = np.array([[0., 0., 0.], [.25, .5, .75], [.5, 0, .5]])
    energy = np.array([0., 1e-4, 2e-4])
    kwargs = dict(j_zz_ev=1.7e-4, eta_ev=2e-5, nk=4)
    general = dynamic_structure_factor(q, energy, mf, chunk_size=1, **kwargs)
    scalar = scalar_dssf(q, energy, mf, **kwargs)
    for key in scalar:
        np.testing.assert_allclose(general[key], scalar[key], rtol=1e-10, atol=1e-10)


def test_trace_contraction_with_complex_projectors():
    rng = np.random.default_rng(43)
    matrices = []
    for _ in range(2):
        z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        matrices.append(z @ z.conj().T)
    a, b = matrices
    plus = np.array([[0, 1], [0, 0]])
    minus = plus.T
    exact = (np.trace(plus @ a @ minus @ b) + np.trace(minus @ a @ plus @ b)) / 2
    assert np.isclose(transverse_weight(np.diag(a), np.diag(b)), exact)


def test_transverse_field_changes_response_and_chunking_is_invariant():
    mf = saddle()
    q = np.array([[.1, .2, .3], [.3, .2, .1]])
    energy = np.array([1e-4, 2e-4])
    kwargs = dict(j_zz_ev=1.7e-4, eta_ev=2e-5, nk=3)
    old = dynamic_structure_factor(q, energy, mf, **kwargs)
    mf["b_plus"] = np.array([.1, -.04j, .02, -.03])
    one = dynamic_structure_factor(q, energy, mf, chunk_size=1, **kwargs)
    two = dynamic_structure_factor(q, energy, mf, chunk_size=2, **kwargs)
    assert not np.allclose(old["total"], one["total"])
    for key in one:
        np.testing.assert_allclose(one[key], two[key])
        assert np.all(one[key] >= 0)


def test_continuum_against_full_complex_vertex_trace():
    mf = saddle()
    mf["s_minus"] = np.array([.4 + .1j, -.3j, .25, -.2 + .15j])
    mf["b_plus"] = np.array([.1, -.04j, .02, -.03])
    q = np.array([.1, -.2, .3])
    energy, jzz, eta = 1e-4, 1.7e-4, 2e-5
    def diagonalize(k):
        v = spinon_potential(k, mf["j_pm_ratio"], mf["s_minus"])
        t = -np.exp(2j*np.pi*k @ E_MU.T) @ (mf["b_plus"] * mf["s_minus"].conj())
        vals, vecs = np.linalg.eigh(np.array([[v, t], [t.conjugate(), v]]))
        return jzz*np.sqrt(2*(mf["lambda_ratio"]+vals)), vecs
    kpoints = reduced_k_grid(2)
    exact = np.zeros(3)
    for k in kpoints:
        wa, ua = diagonalize(k+q)
        wb, ub = diagonalize(k)
        for m in range(2):
            pa = np.outer(ua[:, m], ua[:, m].conj())
            for n in range(2):
                pb = np.outer(ub[:, n], ub[:, n].conj())
                for index, mu in enumerate((1, 2, 3)):
                    phase = np.exp(2j*np.pi*(k+q/2) @ E_MU[mu])
                    plus = np.array([[0, mf["s_minus"][mu].conj()*phase], [0, 0]])
                    minus = np.array([[0, 0], [mf["s_minus"][mu]/phase, 0]])
                    trace = (np.trace(plus @ pa @ plus.conj().T @ pb)
                             + np.trace(minus @ pa @ minus.conj().T @ pb)).real / 2
                    exact[index] += jzz**2/(wa[m]*wb[n])*trace*lorentzian_delta(
                        energy-wa[m]-wb[n], eta)/len(kpoints)
    actual = dynamic_structure_factor(q[None, :], np.array([energy]), mf,
                                      j_zz_ev=jzz, eta_ev=eta, nk=2)
    np.testing.assert_allclose(actual["continuum"][:, 0], exact, rtol=1e-12)


def test_unsupported_exchange_rejected():
    config = json.loads((Path(__file__).parents[1] / "gmft_transport_job.json").read_text())
    validate(config)
    config["J_pmpm_ratio"] = .1
    with pytest.raises(ValueError, match="J_pmpm"):
        validate(config)


def test_configured_phi_scan_uses_n_phi_intervals_and_keeps_endpoints():
    config = {
        "phi_start_degrees": 0.0,
        "phi_stop_degrees": 60.0,
        "n_phi": 30,
    }

    np.testing.assert_allclose(nominal_phi_angles(config), np.arange(0.0, 62.0, 2.0))


def test_one_run_label_selects_matching_log_and_plot_directories(tmp_path):
    config = {
        "run_label": "scan_07",
        "log_root": str(tmp_path / "gMFT_transport_log"),
        "plot_root": str(tmp_path / "gMFT_transport_plots"),
    }

    run, plots = configured_output_directories(config)
    assert run == tmp_path / "gMFT_transport_log/scan_07"
    assert plots == tmp_path / "gMFT_transport_plots"
    assert resistivity_filename(config) == "scan_07.pdf"

    config["resistivity_plot_mode"] = "movie"
    assert resistivity_filename(config) == "scan_07.gif"

    config["run_label"] = "../scan_07"
    with pytest.raises(ValueError, match="run_label"):
        configured_output_directories(config)


def test_resistivity_plot_configuration_is_validated():
    config = json.loads((Path(__file__).parents[1] / "gmft_transport_job.json").read_text())
    config["resistivity_plot_mode"] = "invalid"
    with pytest.raises(ValueError, match="resistivity_plot_mode"):
        validate(config)
    config["resistivity_plot_mode"] = "movie"
    config["resistivity_movie_fps"] = 0
    with pytest.raises(ValueError, match="resistivity_movie_fps"):
        validate(config)


def test_resistivity_movie_contains_configured_field_frames(tmp_path):
    root = Path(__file__).parents[1]
    run = tmp_path / "run"
    data = run / "data"
    data.mkdir(parents=True)
    config = json.loads((root / "gmft_transport_job.json").read_text())
    config.update(
        run_label="field_movie",
        plot_root=str(tmp_path / "plots"),
        resistivity_plot_mode="movie",
        resistivity_movie_fps=10.0,
    )
    (run / "manifest.json").write_text(json.dumps({"config": config}))
    rows = []
    for field, scale in ((0.25, 1.0), (0.45, 1.5), (1.5, 2.0)):
        for phi in (-1.0, 0.0, 1.0):
            rows.append({
                "h_over_J_zz": field,
                "phi_degrees": phi,
                "rho": scale * (1.0 + phi * phi),
                "delta_rho_over_rho_phi0": phi * phi,
            })
    pd.DataFrame(rows).to_csv(data / "tto_transport_summary.csv", index=False)
    subprocess.run(
        [sys.executable, str(root / "plot_gmft_transport.py"), str(run)],
        check=True,
        capture_output=True,
        text=True,
    )
    output = tmp_path / "plots/field_movie.gif"
    assert output.is_file()
    assert output.read_bytes().startswith(b"GIF")
    with Image.open(output) as movie:
        assert movie.n_frames == 3


def test_minimum_uses_only_converged_starts_and_requires_every_angle():
    rows = []
    for phi in range(31):
        for start, energy, status in [(0, 1., "ok"), (1, .8, "ok"), (2, -100., "error")]:
            rows.append(dict(phi_degrees=phi, h_over_J_zz=.2, energy=energy,
                             status=status, start=start, J_pm_over_J_zz=.2))
    winners = select_winners(rows, [.2])
    assert len(winners) == 31
    assert all(r["start"] == 1 for r in winners)
    incomplete = select_winners(rows[3:], [.2])
    assert len(incomplete) == 31
    assert incomplete[0]["status"] == "not_converged"
    assert np.isnan(incomplete[0]["energy"])
    assert incomplete[0]["evaluated_phi_degrees"] == .05


def test_trial_uses_offset_field_and_records_actual_angle(tmp_path, monkeypatch):
    from gmft_transport_job import trial
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gMFT_for_QSI"))
    from gmft_for_qsi import gmft_field_general
    from gmft_for_qsi.lattice import magnetic_field_direction
    config = json.loads((Path(__file__).parents[1] / "gmft_transport_job.json").read_text())
    config["gmft_source"] = str(Path(__file__).resolve().parents[2] / "gMFT_for_QSI")
    config["phi_offset_degrees"] = .05
    captured = {}
    def solver(params, **kwargs):
        captured.update(params)
        raise RuntimeError("deliberate failed trial")
    monkeypatch.setattr(gmft_field_general, "solve_gmft", solver)
    for phi_index, phi in enumerate((0, 30)):
        row = trial((config, str(tmp_path), 0, phi_index, phi, 0))
        assert row["phi_degrees"] == phi
        assert row["evaluated_phi_degrees"] == phi + .05
        np.testing.assert_allclose(captured["h_ext"], config["fields"][0] *
            magnetic_field_direction(theta=config["theta_degrees"], phi=phi+.05))


def test_saved_solver_spectrum_and_condensate_normalization():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gMFT_for_QSI"))
    from gmft_for_qsi.gmft_field_general import solve_spinon_sector
    from gmft_for_qsi.lattice import gen_k_grid
    _, k = gen_k_grid(nk=3)
    mf = saddle()
    mf["b_plus"] = np.array([.1, -.04j, .02, -.03])
    result = solve_spinon_sector(dict(J_zz=1., J_pm=.1, b_plus=mf["b_plus"]), mf["s_minus"], k)
    mf["lambda_ratio"] = result["lambda"]
    frequency, _ = spectrum(k, mf)
    np.testing.assert_allclose(frequency, result["omega_k"], atol=1e-12)
    occupation = result["condensate_density_matrices"] / len(k)
    assert np.isclose(np.trace(occupation, axis1=1, axis2=2).real.sum() / 2, result["rho"])


def test_trial_saves_both_convergence_residuals(tmp_path):
    from gmft_transport_job import trial
    config = json.loads((Path(__file__).parents[1] / "gmft_transport_job.json").read_text())
    config.update(gmft_source=str(Path(__file__).resolve().parents[2] / "gMFT_for_QSI"),
                  fields=[.1], J_pm_ratio=.15, g_z=0., g_perp=.1,
                  delta1_per_mev=1., delta2_per_mev=0., convergence_criterion="amplitudes",
                  tol=1e-8, nk=8, phi_offset_degrees=.05)
    row = trial((config, str(tmp_path), 0, 0, 15, 0))
    assert row["status"] == "ok"
    assert row["convergence_criterion"] == "amplitudes"
    assert row["residual_amplitudes"] < config["tol"]
    assert row["residual_full"] > config["tol"]
    assert not row["fully_converged"]
    with np.load(tmp_path / row["saddle_file"]) as f:
        assert str(f["convergence_criterion"]) == "amplitudes"
        assert not f["fully_converged"]
