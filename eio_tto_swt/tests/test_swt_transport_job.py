import json
from pathlib import Path

import numpy as np
import pytest

from swt_transport_job import (
    configured_output_directories,
    nominal_phi_angles,
    resistivity_filename,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]


def configured():
    return json.loads((ROOT / "swt_transport_job.json").read_text())


def test_swt_output_roots_and_plot_names(tmp_path):
    config = configured()
    config.update(
        run_label="scan_04",
        log_root=str(tmp_path / "SWT_transport_log"),
        plot_root=str(tmp_path / "SWT_transport_plots"),
        resistivity_plot_mode="rows",
    )
    run, plots = configured_output_directories(config)
    assert run == tmp_path / "SWT_transport_log/scan_04"
    assert plots == tmp_path / "SWT_transport_plots"
    assert resistivity_filename(config) == "scan_04.pdf"
    config["resistivity_plot_mode"] = "movie"
    assert resistivity_filename(config) == "scan_04.gif"


def test_swt_phi_count_counts_intervals():
    config = configured()
    config.update(phi_start_degrees=0.0, phi_stop_degrees=60.0, n_phi=30)
    np.testing.assert_allclose(nominal_phi_angles(config), np.arange(0, 62, 2))


def test_swt_configuration_validation():
    config = configured()
    validate(config)
    config["resistivity_plot_mode"] = "invalid"
    with pytest.raises(ValueError, match="resistivity_plot_mode"):
        validate(config)
    config = configured()
    config["reflect_negative_phi"] = True
    config["phi_start_degrees"] = 1.0
    with pytest.raises(ValueError, match="containing zero"):
        validate(config)
