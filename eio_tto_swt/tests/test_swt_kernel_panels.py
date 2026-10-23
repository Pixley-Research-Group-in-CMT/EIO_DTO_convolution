"""Check reconstruction and synchronized display of transport kernels."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from swt_kernel_panels import load_transport_kernels
from plot_swt_kernel import transfer_map, transport_vertex
from swt_transport_job import validate
import plot_swt_results

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('transport_plots_with_kernels', ROOT / '05_plot_transport.py')
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)


@pytest.fixture
def scattering(tmp_path):
    path = tmp_path / 'scattering.hdf5'
    # The saved pair weights intentionally vary with both field and angle.
    kf = np.array([[.1, 0, 0], [0, .1, 0], [-.1, 0, 0]])
    with h5py.File(path, 'w') as h5:
        h5['kf'] = kf
        h5['phi_degrees'] = [0., 30., 0., 30.]
        h5['evaluated_phi_degrees'] = [.1, 30.1, .1, 30.1]
        h5['mu_B_B_over_J_zz'] = [1., 1., 2., 2.]
        h5['total'] = np.arange(1, 5)[:, None, None] * np.ones((4, 3, 3))
        h5['converged'] = np.ones(4, dtype=bool)
        h5.attrs['transport_component'] = 'one_magnon_2'
    return path


def test_reconstruction_uses_saved_component_and_transport_vertex(scattering):
    kernels = load_transport_kernels(scattering, [1., 2.])
    with h5py.File(scattering) as h5:
        kf = h5['kf'][...]
    _, points, groups, _ = transfer_map(kf)
    expected = np.zeros(len(points))
    np.add.at(expected, groups, transport_vertex(kf).ravel())
    np.testing.assert_allclose(kernels.values[0, 0], expected)
    np.testing.assert_allclose(kernels.values[1, 1], 4 * expected)
    np.testing.assert_allclose(kernels.evaluated_phis, [[.1, 30.1], [.1, 30.1]])
    assert kernels.component == 'one_magnon_2'
    # Display grouping preserves the total rate summed over incoming states.
    assert kernels.values[0, 0].sum() == pytest.approx(transport_vertex(kf).sum())


def test_missing_or_reflected_only_angles_are_not_silently_substituted(scattering):
    with pytest.raises(ValueError, match='explicitly evaluated angles'):
        load_transport_kernels(scattering, [1.], [-30.])
    with pytest.raises(ValueError, match='unique'):
        load_transport_kernels(scattering, [1.], [0., 0.])
    with h5py.File(scattering, 'a') as h5:
        h5['converged'][1] = False
    kernels = load_transport_kernels(scattering, [1.])
    assert np.isfinite(kernels.values[0, 0]).all()
    assert np.isnan(kernels.values[0, 1]).all()


def resistivity_data():
    return pd.DataFrame({
        'h_over_J_zz': [1., 1., 2., 2.], 'phi_degrees': [0., 30., 0., 30.],
        'rho': [1., 1.1, 2., 2.4], plots.NORMALIZED_COLUMN: [0., .1, 0., .2],
    })


@pytest.mark.parametrize('layout', ['rows', 'overlay'])
def test_static_maps_are_below_resistivity_and_preserve_angles(scattering, tmp_path, layout):
    kernels = load_transport_kernels(scattering, [1., 2.])
    colors = plt.get_cmap('magma')([.2, .8])
    plot = plots.plot_field_rows if layout == 'rows' else plots.plot_overlaid_fields
    figure = plot(resistivity_data(), kernels.fields, colors, 'spin-wave', 'SWT', 'parameters', kernels=kernels)
    figure.canvas.draw()
    maps = [axis for axis in figure.axes if axis.get_title().startswith('Kernel:')]
    assert len(maps) == 4
    assert all('nominal' in axis.get_title() for axis in maps)
    if layout == 'rows':
        for row in range(2):
            assert maps[2 * row].get_position().y1 < figure.axes[4 * row].get_position().y0
    else:
        assert max(axis.get_position().y1 for axis in maps) < figure.axes[0].get_position().y0
    assert all(not artist.get_rasterized() for axis in maps for artist in axis.collections)
    output = tmp_path / f'kernel_{layout}.pdf'
    figure.savefig(output)
    assert output.stat().st_size > 0
    plt.close(figure)


def test_movie_updates_kernel_values_scales_and_field_labels(scattering, tmp_path):
    kernels = load_transport_kernels(scattering, [1., 2.])
    figure, movie = plots.plot_field_movie(resistivity_data(), kernels.fields, 'SWT', 'parameters', 2., kernels=kernels)
    maps = [axis for axis in figure.axes if axis.get_title().startswith('Kernel:')]
    movie._func(0)
    first = maps[0].collections[0].get_array().copy()
    first_clim = maps[0].collections[0].get_clim()
    movie._func(1)
    np.testing.assert_allclose(maps[0].collections[0].get_array(), 3 * first)
    assert maps[0].collections[0].get_clim() != first_clim
    assert maps[0].collections[0].get_clim() == maps[1].collections[0].get_clim()
    assert '=2$' in maps[0].get_title()
    output = tmp_path / 'combined.gif'
    movie.save(output, writer=PillowWriter(fps=2), dpi=35)
    plt.close(figure)
    with Image.open(output) as image:
        assert image.n_frames == 2


def test_job_rejects_unsampled_kernel_angles():
    config = json.loads((ROOT / 'swt_transport_job.json').read_text())
    config.update(show_kernel_plots=True, kernel_phi_degrees=[-30.])
    with pytest.raises(ValueError, match='explicitly sampled'):
        validate(config)
    config['show_kernel_plots'] = 'yes'
    with pytest.raises(ValueError, match='boolean'):
        validate(config)


@pytest.mark.parametrize('flag,enabled', [('--show-kernel-plots', True), ('--no-kernel-plots', False)])
def test_downloaded_run_can_override_saved_plot_option(tmp_path, monkeypatch, flag, enabled):
    data = tmp_path / 'data'
    data.mkdir()
    for name in plot_swt_results.REQUIRED_DATA_FILES:
        (data / name).touch()
    config = {'run_label': 'example', 'plot_root': str(tmp_path),
              'show_kernel_plots': not enabled, 'kernel_phi_degrees': [0., 30.]}
    (tmp_path / 'manifest.json').write_text(json.dumps({'config': config}))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        (tmp_path / 'example.pdf').write_bytes(b'test-output')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(plot_swt_results.subprocess, 'run', run)
    monkeypatch.setattr('sys.argv', ['plot_swt_results.py', str(tmp_path), flag])
    plot_swt_results.main()
    assert ('--show-kernel-plots' in calls[0]) is enabled
