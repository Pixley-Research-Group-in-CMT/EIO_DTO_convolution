"""Python interface to the three EIO/TTO spin-wave workflows.

All output paths are explicit. Importing this package never starts a calculation.
"""
from dataclasses import dataclass
from pathlib import Path
import shutil
from types import SimpleNamespace

import h5py
import numpy as np

from . import _surface_states, _structure_factor, _scattering, _transport
from . import _resistivity, _path_movie, _kernel


@dataclass(frozen=True)
class SpinWaveParameters:
    """Model and numerical settings; energies in eV unless labeled otherwise.

    Fields are bare mu_B |B| / J_zz. delta1_per_mev and delta2_per_mev are
    inverse meV; eta_mev is meV. nk controls the internal magnon grid.
    """
    j_pm: float = 0.2
    j_pmpm: float = 0.0
    j_zz_ev: float = 0.00017
    g_z: float = 10.0
    g_perp: float = 0.0
    delta1_per_mev: float = 0.0
    delta2_per_mev: float = 0.0
    eta_mev: float = 0.02
    nk: int = 8
    q_perp_mode: str = 'integrated'
    n_q_perp: int = 8
    lattice_constant: float = 10.154
    spin_length: float = 0.5
    regularization: float = 1e-8
    stability_tilt: float = 0.01
    random_starts: int = 8

    def __post_init__(self):
        for name, value in vars(self).items():
            if name != 'q_perp_mode' and not np.isfinite(value):
                raise ValueError(f'{name} must be finite')
        for name in ('j_zz_ev', 'eta_mev', 'lattice_constant', 'spin_length', 'regularization'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.stability_tilt < 0:
            raise ValueError('stability_tilt must be nonnegative')
        for name in ('nk', 'n_q_perp', 'random_starts'):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.q_perp_mode not in ('slice', 'integrated'):
            raise ValueError('q_perp_mode must be slice or integrated')
        if self.q_perp_mode == 'integrated' and self.n_q_perp < 2:
            raise ValueError('integrated mode requires n_q_perp >= 2')


@dataclass(frozen=True)
class TransportResult:
    """Locations of computed transport data, accepted by plot_resistivity."""
    output_dir: Path
    data_dir: Path
    summary_file: Path
    scattering_file: Path
    surface_states_file: Path


@dataclass(frozen=True)
class MovieResult:
    """Locations of saved numerical data and encoded movies."""
    data_file: Path
    movie_files: tuple


def _directory(path):
    return Path(path).expanduser().resolve()


def _model_directory(model):
    if model not in ('original', 'symmetrized'):
        raise ValueError('eio_model must be original or symmetrized')
    return Path(__file__).resolve().parent / 'models' / model


def generate_surface_states(output_dir, *, eio_model='symmetrized', nk=24, overwrite=False):
    """Compute EIO surface states and return the HDF5 path.

    ``output_dir`` is the directory receiving the HDF5 and validation JSON files.
    nk=24 is required by the transport and kernel workflows; nk=36 is also
    available for separate EIO calculations.
    """
    model_dir = _model_directory(eio_model)
    if isinstance(nk, bool) or nk not in (24, 36):
        raise ValueError('EIO nk must be 24 or 36')
    data = _directory(output_dir)
    data.mkdir(parents=True, exist_ok=True)
    return _surface_states.generate_surface_states(
        nk, overwrite=overwrite, data_dir=data, model_dir=model_dir, eio_model=eio_model,
    )


def _ensure_surface_states(data, model, existing=None):
    _model_directory(model)
    target = data / 'surface_states_summary_nk24.hdf5'
    source = _directory(existing) if existing is not None else target
    if source.exists():
        with h5py.File(source) as h5:
            if h5.attrs.get('eio_model', 'original') != model:
                raise ValueError('Saved EIO states use another model; choose a different output directory')
            if h5.attrs.get('nk', 24) != 24:
                raise ValueError('Transport and kernel workflows require nk24 EIO states')
            required = ('kf', 'eigf', 'proj_sl1', 'proj_sl2', 'proj_sl3', 'v_kf')
            if any(name not in h5 for name in required):
                raise ValueError('Saved EIO states are missing required electronic data')
        if target.exists() and source != target:
            raise FileExistsError(f'{target} already exists; choose another output directory')
        data.mkdir(parents=True, exist_ok=True)
        if source != target:
            shutil.copy2(source, target)
        return target
    if existing is not None:
        raise FileNotFoundError(source)
    return generate_surface_states(data, eio_model=model)


def _options(parameters):
    p = SpinWaveParameters() if parameters is None else parameters
    if not isinstance(p, SpinWaveParameters):
        raise TypeError('parameters must be SpinWaveParameters')
    return dict(j_pm=p.j_pm, j_pmpm=p.j_pmpm, j_zz_ev=p.j_zz_ev,
                g_z=p.g_z, g_perp=p.g_perp, delta1_per_mev=p.delta1_per_mev,
                delta2_per_mev=p.delta2_per_mev, dssf_nk=int(p.nk),
                q_perp_mode=p.q_perp_mode, n_q_perp=int(p.n_q_perp),
                eta_mev=p.eta_mev, spin_length=p.spin_length,
                regularization=p.regularization, stability_tilt=p.stability_tilt,
                lattice_constant=p.lattice_constant, random_starts=int(p.random_starts))


def _check_movie_options(options):
    for name in ('fps', 'dpi'):
        if not np.isfinite(options[name]) or options[name] <= 0:
            raise ValueError(f'{name} must be positive and finite')
    if isinstance(options['dpi'], bool) or int(options['dpi']) != options['dpi']:
        raise ValueError('dpi must be an integer')
    if options['video_format'] not in ('gif', 'mp4'):
        raise ValueError('video_format must be gif or mp4')
    if not 0 < options['vmax_percentile'] <= 100:
        raise ValueError('vmax_percentile must be in (0, 100]')
    # Check the optional encoder before starting expensive calculations.
    from ._animation import animation_writer
    animation_writer(options['video_format'], options['fps'])


def compute_transport(
    output_dir, *, fields=(0.25, 0.45, 1.5), phi_start=0.0, phi_stop=180.0,
    n_phi=90, phi_offset=0.1, theta=90.0, reflect_negative_phi=False,
    component='total', eio_model='symmetrized', eio_states=None,
    parameters=None, overwrite=False,
):
    """Compute SWT scattering, relaxation times and resistivity; return paths.

    n_phi counts intervals, so n_phi+1 nominal angles include both endpoints.
    eio_states can supply an existing compatible nk24 surface-state HDF5 file.
    Reuse the returned result with plot_resistivity to change the display.
    """
    options = _options(parameters)
    options.update(fields=tuple(fields), phi_start=phi_start, phi_stop=phi_stop,
                   n_phi=n_phi + 1, phi_offset=phi_offset, theta=theta,
                   reflect_negative_phi=reflect_negative_phi, transport_component=component,
                   eta_ev=options['eta_mev'] * 1e-3, overwrite=overwrite)
    if isinstance(n_phi, bool) or int(n_phi) != n_phi or n_phi < 1:
        raise ValueError('n_phi must be a positive number of intervals')
    if component not in _structure_factor.TRANSPORT_COMPONENTS:
        raise ValueError('Unknown transport component')
    if not isinstance(reflect_negative_phi, bool):
        raise ValueError('reflect_negative_phi must be boolean')
    args = SimpleNamespace(**options)
    _structure_factor.generate_conditions(args)  # Validate the scan first.
    run = _directory(output_dir)
    data = run / 'data'
    for name in ('tto_spin_wave_structure_factor.hdf5', 'tto_spin_wave_scattering.hdf5',
                 'tto_spin_wave_transport_summary.csv', 'tto_spin_wave_relaxation_times.csv'):
        if (data / name).exists() and not overwrite:
            raise FileExistsError(f'{data / name} exists; use overwrite=True or another directory')
    states = _ensure_surface_states(data, eio_model, eio_states)
    _structure_factor.compute(args, data_dir=data)
    _scattering.compute('spin-wave', overwrite=overwrite, data_dir=data)
    _transport.compute('spin-wave', data_dir=data)
    return TransportResult(run, data, data / 'tto_spin_wave_transport_summary.csv',
                           data / 'tto_spin_wave_scattering.hdf5', states)


def plot_resistivity(
    result, *, layout='rows', show_kernel_plots=False, kernel_phis=(0.0, 30.0),
    output=None, fps=2.0,
):
    """Plot saved transport data and return the vector PDF or GIF path.

    result accepts TransportResult, a run directory, or its data subdirectory.
    layout is 'rows', 'overlay', or 'movie'. Kernel angles must be explicitly
    sampled nominal field angles; actual offset angles appear in panel titles.
    """
    if layout not in ('rows', 'overlay', 'movie'):
        raise ValueError('layout must be rows, overlay, or movie')
    if not isinstance(show_kernel_plots, bool):
        raise ValueError('show_kernel_plots must be boolean')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')
    if isinstance(result, TransportResult):
        data, run = result.data_dir, result.output_dir
    else:
        run = _directory(result)
        data = run if run.name == 'data' else run / 'data'
        run = data.parent
    suffix = '.gif' if layout == 'movie' else '.pdf'
    output = _directory(output) if output is not None else run / 'figures' / f'resistivity{suffix}'
    if output.suffix.lower() != suffix:
        raise ValueError(f'{layout} output must end in {suffix}')
    args = SimpleNamespace(theory='spin-wave', plot_mode=layout, field_layout=None,
                           show_kernel_plots=show_kernel_plots, kernel_phis=tuple(kernel_phis),
                           movie_fps=fps, output_directory=output.parent, output_name=output.name)
    return _resistivity.render(args, data)


def make_dssf_path_movie(
    output_dir, *, field_start=0.25, field_stop=1.5, n_field_frames=31,
    phi_degrees=(0.1, 30.1), component='total', points_per_segment=40,
    n_omega=200, omega_chunk_size=8, vmax_percentile=99.0, fps=10.0,
    video_format='gif', dpi=100, parameters=None, overwrite=False,
):
    """Compute and animate the bulk FCC-path DSSF versus bare field.

    Returns MovieResult with an HDF5 file and sublattice/summed movie paths.
    Both field endpoints are included. There are two configured field angles.
    """
    options = _options(parameters)
    options.update(output_root=_directory(output_dir), mu_b_b_start=field_start,
                   mu_b_b_stop=field_stop, n_field_frames=n_field_frames,
                   field_phis=tuple(phi_degrees), component=component,
                   points_per_segment=points_per_segment, n_omega=n_omega,
                   omega_chunk_size=omega_chunk_size, vmax_percentile=vmax_percentile,
                   fps=fps, video_format=video_format, dpi=dpi, overwrite=overwrite)
    if component not in _structure_factor.TRANSPORT_COMPONENTS:
        raise ValueError('Unknown DSSF component')
    for name in ('points_per_segment', 'n_omega', 'omega_chunk_size'):
        value = options[name]
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    _check_movie_options(options)
    data, sublattices, summed = _path_movie.compute_and_animate(SimpleNamespace(**options))
    return MovieResult(data, tuple(sublattices) + tuple(summed))


def make_kernel_movie(
    output_dir, *, field_start=1.0, field_stop=1.1, n_field_frames=11,
    phi_degrees=(0.0, 30.0), eio_model='symmetrized', eio_states=None,
    vmax_percentile=100.0, marker_size=20.0, fps=10.0, video_format='gif',
    dpi=120, parameters=None, overwrite=False,
):
    """Compute the on-shell, total one- plus two-magnon kernel field movie.

    Uses all incoming surface states. Returns numerical HDF5 and movie paths.
    A frame's two angle panels share a color scale; scales vary across fields.
    """
    run = _directory(output_dir)
    options = _options(parameters)
    options.update(output_root=run, mu_b_b_start=field_start, mu_b_b_stop=field_stop,
                   n_field_frames=n_field_frames, phi_degrees=tuple(phi_degrees),
                   eio_model=eio_model, eio_states=None, plot_mode='movie',
                   vmax_percentile=vmax_percentile, marker_size=marker_size,
                   fps=fps, video_format=video_format, dpi=dpi, overwrite=overwrite)
    args = SimpleNamespace(**options)
    _kernel.validate(args)
    _check_movie_options(options)
    output = run / 'data/swt_kernel_movie.hdf5'
    movie = run / f'movies/swt_kernel_movie.{video_format}'
    if not overwrite and any(p.exists() for p in (output, movie)):
        raise FileExistsError('Kernel output exists; use overwrite=True or another directory')
    args.eio_states = _ensure_surface_states(run / 'data', eio_model, eio_states)
    data, movie = _kernel.compute_movie(args)
    return MovieResult(data, (movie,))
