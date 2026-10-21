"""Verify missing MF points survive all stages as plot gaps, not zeros."""
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd

from plot_gmft_spin_movie import load_grid

ROOT = Path(__file__).resolve().parents[1]


def test_missing_points_end_to_end(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    with h5py.File(data / "surface_states_summary_nk24.hdf5", "w") as f:
        f["kf"] = np.array([[.1, 0., 0.], [0., .1, 0.], [-.1, -.1, 0.]])[:, None, :]
        f["eigf"] = np.array([5.1420, 5.1421, 5.1422])
        f["v_kf"] = np.array([[1., 0., 0.], [1., 1., 0.], [-1., 1., 0.]])
        for mu in (1, 2, 3):
            f[f"proj_sl{mu}"] = np.ones((3, 3, 2)) / np.sqrt(18)
    config = json.loads((ROOT / "gmft_transport_job.json").read_text())
    config.update(
        fields=[.01, .02, .03], J_pm_ratio=.2, nk=2,
        phi_offset_degrees=.05, phi_start_degrees=0.0,
        phi_stop_degrees=30.0, n_phi=30,
        run_label="test_run", log_root=str(tmp_path / "logs"),
        plot_root=str(tmp_path / "plots"),
        resistivity_plot_mode="overlay",
    )
    (tmp_path / "manifest.json").write_text(json.dumps(dict(config=config)))
    np.savez(tmp_path / "saddle.npz", nk=2, j_pm_ratio=.2, lambda_ratio=1.,
             s_minus=np.full(4, .5+0j), s_z=np.zeros(4, complex), b_plus=np.zeros(4, complex),
             rho=0., condensate_k=np.empty((0, 3)), condensate_diagonal=np.empty((0, 2)))
    with np.load(tmp_path / "saddle.npz") as saddle:
        unstable = {name: saddle[name] for name in saddle.files}
    unstable["lambda_ratio"] = -1.
    np.savez(tmp_path / "unstable.npz", **unstable)
    rows = []
    for phi in range(31):
        for field in config["fields"]:
            missing = field == .03 or phi == 15 or (field == .01 and phi == 0)
            rows.append(dict(phi_degrees=phi, evaluated_phi_degrees=phi+.05,
                h_over_J_zz=field, J_pm_over_J_zz=.2,
                status="not_converged" if missing else "ok",
                saddle_file=("" if missing else "unstable.npz"
                             if field == .02 and phi == 16 else "saddle.npz")))
    table = pd.DataFrame(rows)
    table.to_csv(tmp_path / "winning_saddles.csv", index=False)
    for phi in range(31):
        directory = tmp_path / "mean_fields" / f"phi_{phi}"
        directory.mkdir(parents=True)
        table[table.phi_degrees == phi].to_csv(directory / "field_diagnostics_min_energy.csv", index=False)
    env = dict(os.environ, EIO_TTO_OUTPUT_ROOT=str(tmp_path), MPLBACKEND="Agg")
    commands = [
        ["02_compute_tto_structure_factor.py", "--gmft-folder", str(tmp_path / "mean_fields"),
         "--general-saddles", str(tmp_path), "--fields", ".01", ".02", ".03", "--j-pm", ".2",
         "--dssf-nk", "2", "--allow-missing"],
        ["03_compute_scattering.py"], ["04_compute_transport.py"],
        ["plot_gmft_transport.py", str(tmp_path)],
        ["plot_gmft_spin_components.py", str(tmp_path)],
    ]
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True,
                       capture_output=True, text=True)
    result = pd.read_csv(data / "tto_transport_summary.csv")
    assert len(result) == 3*31
    for field in config["fields"]:
        group = result[np.isclose(result.h_over_J_zz, field)].sort_values("phi_degrees")
        np.testing.assert_array_equal(group.phi_degrees, np.arange(31))
    assert result[result.phi_degrees == 15].rho.isna().all()
    one = result[np.isclose(result.h_over_J_zz, .01)]
    assert np.isfinite(one[one.phi_degrees == 1].rho).all()
    assert one.delta_rho_over_rho_phi0.isna().all()  # Missing reference, no substitute angle.
    two = result[np.isclose(result.h_over_J_zz, .02)]
    assert np.isfinite(two[two.phi_degrees == 1].delta_rho_over_rho_phi0).all()
    assert two[two.phi_degrees == 16].rho.isna().all()
    assert two[two.phi_degrees == 16].mf_converged.all()
    assert not two[two.phi_degrees == 16].dssf_valid.any()
    three = result[np.isclose(result.h_over_J_zz, .03)]
    assert three.rho.isna().all()
    assert result.loc[result.phi_degrees == 0, "evaluated_phi_degrees"].eq(.05).all()
    _, _, _, z, perp = load_grid([tmp_path])
    assert np.isnan(z[15]).all() and np.isnan(perp[15]).all()
    assert np.isnan(z[:, 2]).all()
    assert np.isfinite(z[1, 0]).all()
    assert np.isnan(z[16, 1]).all() and np.isnan(perp[16, 1]).all()
    diagnostics = pd.read_csv(data / "tto_gmft_conditions.csv")
    failed = diagnostics[(diagnostics.phi_degrees == 16)
                         & np.isclose(diagnostics.h_over_J_zz, .02)].iloc[0]
    assert failed.mf_status == "ok" and failed.status == "unstable_dssf"
    warning = (tmp_path / "warnings.txt").read_text()
    assert "[DSSF] h=0.02, nominal_phi=16, evaluated_phi=16.05" in warning
    assert "min((lambda+v)/Jzz)=" in warning
    assert "[MF]" in warning
    with h5py.File(data / "tto_dynamic_structure_factor.hdf5") as f:
        assert len(f["converged"]) == 93
        assert "reflect_negative_phi" not in f.attrs
        assert np.isnan(f["total"][~f["converged"][...]]).all()
        assert np.count_nonzero(f["mf_converged"][...]) == np.count_nonzero(f["dssf_valid"][...])+1
    assert (tmp_path / "plots/test_run.pdf").is_file()
    assert not (tmp_path / "gmft_resistivity.pdf").exists()
    assert (tmp_path / "gmft_spin_components_vs_phi.pdf").is_file()
