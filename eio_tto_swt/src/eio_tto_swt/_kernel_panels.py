"""Kernel panels reconstructed from the scattering data used by resistivity."""
from dataclasses import dataclass
from types import SimpleNamespace
import h5py
import numpy as np
from ._kernel import add_colorbar, decorate_map, group_kernel, kernel_color_norm, set_panel_status, transfer_map, transport_vertex

@dataclass
class TransportKernels:
    fields: np.ndarray
    nominal_phis: np.ndarray
    evaluated_phis: np.ndarray
    values: np.ndarray
    points: np.ndarray
    boundary: np.ndarray
    component: str

def load_transport_kernels(path, fields, phi_degrees=(0.0, 30.0)):
    """Use exactly the saved transport component, momenta, and sampled angles.

    Scattering ``total`` already contains the local electronic bond weights.
    Multiplication by 1-cos(theta) recovers the same pair kernel used by stage 4.
    No angular interpolation, reflection, or new SWT evaluation is performed.
    """
    requested = np.asarray(phi_degrees, dtype=float)
    if requested.ndim != 1 or not len(requested) or (not np.isfinite(requested).all()) or (len(np.unique(requested)) != len(requested)):
        raise ValueError('kernel phi angles must be finite, nonempty, and unique')
    fields = np.asarray(fields, dtype=float)
    with h5py.File(path) as h5:
        kf = h5['kf'][...].reshape(-1, 3)
        phis = h5['phi_degrees'][...]
        field_key = 'mu_B_B_over_J_zz' if 'mu_B_B_over_J_zz' in h5 else 'h_over_J_zz'
        saved_fields = h5[field_key][...]
        evaluated = h5['evaluated_phi_degrees'][...] if 'evaluated_phi_degrees' in h5 else phis
        (_, points, groups, boundary) = transfer_map(kf)
        vertex = transport_vertex(kf)
        values = np.empty((len(fields), len(requested), len(points)))
        evaluated_phis = np.empty((len(fields), len(requested)))
        for (row, field) in enumerate(fields):
            for (column, phi) in enumerate(requested):
                matches = np.flatnonzero(np.isclose(saved_fields, field, rtol=1e-10, atol=1e-12) & np.isclose(phis, phi, rtol=0, atol=1e-08))
                if len(matches) != 1:
                    raise ValueError(f'Kernel requires one saved scattering condition at field={field:g}, nominal phi={phi:g}; found {len(matches)}. Choose kernel_phis from the explicitly evaluated angles in the run.')
                index = matches[0]
                pair_kernel = h5['total'][index] * vertex
                if any((name in h5 and (not bool(h5[name][index])) for name in ('converged', 'dssf_valid'))):
                    pair_kernel = np.full_like(pair_kernel, np.nan)
                values[row, column] = group_kernel(pair_kernel[None], groups, len(points))[0]
                evaluated_phis[row, column] = evaluated[index]
        component = h5.attrs.get('transport_component', 'total')
        if isinstance(component, bytes):
            component = component.decode('utf-8')
    return TransportKernels(fields, requested, evaluated_phis, values, points, boundary, component)

def draw_kernel_row(figure, subplot_spec, kernels, field_index):
    """Add one field's maps; return an updater for synchronized movies."""
    grid = subplot_spec.subgridspec(1, len(kernels.nominal_phis))
    axes = [figure.add_subplot(grid[0, col]) for col in range(len(kernels.nominal_phis))]
    args = SimpleNamespace(marker_size=20, vmax_percentile=100)
    norm = kernel_color_norm(kernels.values[field_index], 100)
    (artists, labels) = ([], [])
    for (col, axis) in enumerate(axes):
        artist = decorate_map(axis, kernels.points, kernels.boundary, kernels.values[field_index, col], norm, args)
        label = axis.text(0.5, 0.5, 'Unavailable', transform=axis.transAxes, ha='center', va='center', color='0.25')
        axis.set_xlabel('$q_x$ ($\\mathrm{\\AA}^{-1}$)')
        axis.set_ylabel('$q_y$ ($\\mathrm{\\AA}^{-1}$)')
        artists.append(artist)
        labels.append(label)
    add_colorbar(figure, artists[0], axes, args)

    def update(index):
        norm = kernel_color_norm(kernels.values[index], 100)
        for (col, (axis, artist, label)) in enumerate(zip(axes, artists, labels)):
            artist.set_norm(norm)
            set_panel_status(axis, artist, label, kernels.values[index, col])
            nominal = kernels.nominal_phis[col]
            actual = kernels.evaluated_phis[index, col]
            angle = f'$\\phi={actual:g}^\\circ$'
            if not np.isclose(nominal, actual, rtol=0, atol=1e-08):
                angle += f' (nominal ${nominal:g}^\\circ$)'
            axis.set_title(f'Kernel: $\\mu_B|B|/J_{{zz}}={kernels.fields[index]:g}$; ' + angle + '\n' + str(kernels.component).replace('_', ' '), fontsize=8)
        return tuple(artists + labels)
    update(field_index)
    return update
