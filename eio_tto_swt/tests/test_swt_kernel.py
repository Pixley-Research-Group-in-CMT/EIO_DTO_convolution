import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image
from matplotlib.animation import PillowWriter

from plot_swt_kernel import (
    build_parser, validate, transport_vertex, contract_kernel,
    transfer_map, group_kernel, make_movie, field_axis, _scattering,
)


def configured(tmp_path):
    return build_parser().parse_args([
        '--output-root', str(tmp_path), '--mu-b-b-start', '.25',
        '--mu-b-b-stop', '1.5', '--n-field-frames', '3',
    ])


def test_field_scan_validation(tmp_path):
    args = configured(tmp_path)
    validate(args)
    np.testing.assert_allclose(field_axis(.25, 1.5, 3), [.25, .875, 1.5])
    args.mu_b_b_stop = args.mu_b_b_start
    with pytest.raises(ValueError, match='mu_b_b_start'):
        validate(args)
    args = configured(tmp_path)
    args.n_field_frames = 1
    with pytest.raises(ValueError, match='at least 2'):
        validate(args)


def test_kernel_agrees_with_transport_contraction_and_group_sum():
    rng = np.random.default_rng(42)
    kf = np.array([[.2, .1, 0], [-.1, .2, 0], [-.2, -.1, 0]])
    projections = {i: rng.normal(size=(3, 3, 2)) + 1j*rng.normal(size=(3, 3, 2))
                   for i in (1, 2, 3)}
    weights = np.array([sum(_scattering.local_eio_weights(projections[sl], np.ones(3))
                            for sl in _scattering.EIO_NEIGHBORS[mu]) for mu in range(3)])
    spectrum = rng.random((3, 4, 3, 3))
    vertex = transport_vertex(kf)
    kernel = contract_kernel(spectrum, weights, vertex)
    for frame in range(4):
        expected = np.einsum('mij,mij->ij', spectrum[:, frame], weights) * vertex
        np.testing.assert_allclose(kernel[frame], expected)
    np.testing.assert_array_equal(np.diagonal(kernel, axis1=1, axis2=2), 0.)
    assert vertex[0, 2] == pytest.approx(2.)
    q, points, groups, boundary = transfer_map(kf)
    grouped = group_kernel(kernel, groups, len(points))
    np.testing.assert_allclose(grouped.sum(axis=1), kernel.sum(axis=(1, 2)))
    assert np.max(np.linalg.norm(points, axis=1)) <= np.linalg.norm(boundary[0]) + 1e-9


def test_movie_animates_integrated_kernel_over_field_with_independent_frame_scales(tmp_path):
    args = configured(tmp_path)
    kf = np.array([[.2, .1, 0], [-.1, .2, 0], [-.2, -.1, 0]])
    _, points, _, boundary = transfer_map(kf)
    values = np.arange(2*3*len(points), dtype=float).reshape(2, 3, len(points))
    fields = field_axis(.25, 1.5, 3)
    fig, movie, update = make_movie(values, fields, points, boundary, args)
    assert len(fig.axes) == 3  # Two angles and updating colorbar.
    first_limits = fig.axes[0].collections[0].get_clim()
    artist, second, *_ = update(2)
    assert '1.5' in fig._suptitle.get_text()
    assert 'Integrated' in fig._suptitle.get_text()
    assert 'symmetrized' in fig._suptitle.get_text()
    assert artist.get_clim() != first_limits
    assert artist.norm is second.norm
    assert artist.get_clim()[1] == np.max(values[:, 2])
    assert fig.axes[2].get_ylim()[1] == np.max(values[:, 2])
    np.testing.assert_array_equal(artist.get_array(), values[0, 2])
    path = tmp_path / 'kernel.gif'
    movie.save(path, writer=PillowWriter(fps=2), dpi=35)
    plt.close(fig)
    with Image.open(path) as im:
        assert im.n_frames == len(fields)


def test_side_by_side_has_three_fields_and_ignores_movie_range(tmp_path):
    from plot_swt_kernel import selected_fields, make_side_by_side
    args = configured(tmp_path)
    args.plot_mode = 'side_by_side'
    args.fields = [.04, .045, .05]
    args.mu_b_b_stop = args.mu_b_b_start  # Unused movie range.
    validate(args)
    fields = selected_fields(args)
    np.testing.assert_array_equal(fields, args.fields)
    kf = np.array([[.2, .1, 0], [-.1, .2, 0], [-.2, -.1, 0]])
    _, points, _, boundary = transfer_map(kf)
    values = np.arange(2*3*len(points), dtype=float).reshape(2, 3, len(points))
    fig = make_side_by_side(values, fields, points, boundary, args)
    assert len(fig.axes) == 7
    assert len({id(axis.collections[0].norm) for axis in fig.axes[:6]}) == 1
    for index, axis in enumerate(fig.axes[:6]):
        row, column = divmod(index, 2)
        np.testing.assert_array_equal(axis.collections[0].get_array(), values[column, row])
        assert f"={args.phi_degrees[column]:g}" in axis.get_title()
    output = tmp_path / 'kernel.pdf'
    fig.savefig(output)
    assert output.stat().st_size > 0
    plt.close(fig)
    args.fields = [.04, .04, .05]
    with pytest.raises(ValueError, match='three distinct'):
        validate(args)


def test_kernel_cannot_select_partial_components_or_incoming_states(tmp_path):
    for option in (['--component', 'one_magnon'], ['--incoming-index', '0']):
        with pytest.raises(SystemExit):
            build_parser().parse_args(['--output-root', str(tmp_path), *option])


def test_gray_panels_reset_and_all_unstable_scan_still_plots(tmp_path):
    from plot_swt_kernel import make_side_by_side
    args = configured(tmp_path)
    kf = np.array([[.2, .1, 0], [-.1, .2, 0], [-.2, -.1, 0]])
    _, points, _, boundary = transfer_map(kf)
    values = np.ones((2, 3, len(points)))
    values[0, 1] = np.nan
    fig, movie, update = make_movie(values, [.25, .5, 1.5], points, boundary, args)
    update(1)
    assert not fig.axes[0].collections[0].get_visible()
    assert fig.axes[0].texts[0].get_visible()
    assert fig.axes[1].collections[0].get_visible()
    update(2)
    assert fig.axes[0].collections[0].get_visible()
    assert not fig.axes[0].texts[0].get_visible()
    movie._draw_was_started = True
    plt.close(fig)
    fig = make_side_by_side(np.full_like(values, np.nan), [.25, .5, 1.5], points, boundary, args)
    assert all(axis.texts[0].get_visible() for axis in fig.axes[:6])
    plt.close(fig)


def test_instability_is_saved_and_scan_continues(tmp_path, monkeypatch):
    import h5py
    import plot_swt_kernel as kernel
    states = tmp_path / 'synthetic_states.hdf5'
    with h5py.File(states, 'w') as h5:
        angle = np.linspace(0, 2*np.pi, 15, endpoint=False)
        h5['kf'] = np.column_stack((.1*np.cos(angle), .1*np.sin(angle), np.zeros(15)))
        h5['eigf'] = np.zeros(15)
        for sl in (1, 2, 3):
            h5[f'proj_sl{sl}'] = np.ones((15, 3, 2), dtype=complex) / np.sqrt(18)
    calls = []
    def solver(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise ValueError('Dynamically unstable spin-wave state (imaginary frequency 2.646e-02)')
        return {'total': np.ones((3, 15, 15)), 'max_imaginary_frequency_ratio': 0.}
    monkeypatch.setattr(kernel, 'surface_projected_dynamic_structure_factor', solver)
    monkeypatch.setattr(kernel, 'prepare_ordered_state', lambda *args: (
        np.zeros((4, 3)), None, {'unit_vectors_local': np.ones((4, 3))}))
    kernel.main(['--output-root', str(tmp_path), '--eio-model', 'original',
                 '--eio-states', str(states),
                 '--mu-b-b-start', '.03', '--mu-b-b-stop', '.06', '--n-field-frames', '3', '--dpi', '30'])
    assert len(calls) == 6
    with h5py.File(tmp_path / 'data/swt_kernel_movie.hdf5') as h:
        np.testing.assert_array_equal(h['stable'][...], [[True, False, True], [True, True, True]])
        assert np.isnan(h['kernel'][0, 1]).all()
        assert np.isfinite(h['kernel'][0, 2]).all()
        assert 'Dynamically unstable' in h['instability_error'].asstr()[0, 1]
    with Image.open(tmp_path / 'movies/swt_kernel_movie.gif') as im:
        assert im.n_frames == 3
