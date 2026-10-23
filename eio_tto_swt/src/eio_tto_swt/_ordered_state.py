"""Shared numerical helpers for the three packaged SWT workflows."""
import matplotlib as mpl
import numpy as np
from .spin_wave import SPIN_WAVE_COMPONENTS, local_magnetic_field, local_zeeman_field, relax_ordered_state, zeeman_minimizing_spins
COMPONENTS = SPIN_WAVE_COMPONENTS
SUBLATTICES = (1, 2, 3)
FIELD_RATIOS = np.asarray((0.25, 0.45, 1.5))
FIELD_PHIS_DEGREES = np.asarray((0.1, 30.1))
FIELD_PHI_FILE_LABELS = ('0', '30')
FIELD_THETA_DEGREES = 90.0
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 9, 'axes.linewidth': 0.8, 'xtick.direction': 'out', 'ytick.direction': 'out', 'pdf.fonttype': 42})

def prepare_ordered_state(args, field_ratio, field_phi_degrees):
    """Relax the four-sublattice classical state for one fixed field."""
    bare_local = local_magnetic_field(field_ratio, theta_degrees=FIELD_THETA_DEGREES, phi_degrees=field_phi_degrees)
    j_zz_mev = 1000.0 * args.j_zz_ev
    h_local = local_zeeman_field(field_ratio, theta_degrees=FIELD_THETA_DEGREES, phi_degrees=field_phi_degrees, g_z=args.g_z, g_perp=args.g_perp, delta1_j_zz=args.delta1_per_mev * j_zz_mev, delta2_j_zz=args.delta2_per_mev * j_zz_mev)
    initial_spins = zeeman_minimizing_spins(h_local, spin_length=args.spin_length)
    relaxed = relax_ordered_state(initial_spins, h_local, args.j_pm, j_pmpm_ratio=args.j_pmpm, spin_length=args.spin_length, random_starts=args.random_starts, stability_tilt=args.stability_tilt)
    relaxed['bare_field_local_over_J_zz'] = bare_local
    return (h_local, initial_spins, relaxed)

def shared_norm(values, percentile):
    """Return a common linear intensity scale for all displayed panels."""
    values = np.asarray(values, dtype=float)
    if np.any(np.isinf(values)):
        raise ValueError('Structure-factor values cannot contain infinities')
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError('Structure-factor values must be finite and nonempty')
    if np.any(finite < -1e-12):
        raise ValueError('Structure-factor values must be finite and nonnegative')
    if not 0.0 < percentile <= 100.0:
        raise ValueError('vmax_percentile must be in (0, 100]')
    vmax = float(np.percentile(finite, percentile))
    if vmax <= 0.0:
        vmax = 1.0
    return mpl.colors.Normalize(vmin=0.0, vmax=vmax, clip=True)

def projection_title(args):
    """Return a compact description of the displayed perpendicular momentum."""
    if args.q_perp_mode == 'slice':
        return '$q_\\perp=0$ slice'
    return f'$q_\\perp$ average ($N_\\perp={args.n_q_perp}$)'
