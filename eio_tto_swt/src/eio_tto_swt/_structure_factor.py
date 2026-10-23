from pathlib import Path
'Evaluate the EIO_TTO Section 1 TTO spin-wave DSSF on the EIO transfers.'
import h5py
import numpy as np
import pandas as pd
from ._config import EIO_NK, TTO_INTERFACE_SUBLATTICES
from .projection import q_perp_samples_for_mode, surface_projected_dynamic_structure_factor
from .spin_wave import ONE_MAGNON_MODE_COMPONENTS, local_magnetic_field, local_zeeman_field, relax_ordered_state, spin_wave_structure_factor, zeeman_minimizing_spins
from ._geometry import eio_to_tto_reduced_momentum
from ._components import TRANSPORT_COMPONENTS, combine_transport_components, transport_component_definition

def generate_conditions(args):
    """Build the field scan without reading any mean-field result files."""
    fields = np.asarray(args.fields, dtype=float)
    if fields.ndim != 1 or len(fields) == 0 or (not np.all(np.isfinite(fields))):
        raise ValueError('At least one finite mu_B |B| / J_zz value is required')
    if np.any(fields < 0.0):
        raise ValueError('mu_B |B| / J_zz values must be nonnegative')
    if args.n_phi < 1:
        raise ValueError('--n-phi must be positive')
    if not np.isfinite(args.phi_offset):
        raise ValueError('--phi-offset must be finite')
    nominal_angles = np.linspace(args.phi_start, args.phi_stop, args.n_phi)
    evaluated_angles = nominal_angles + args.phi_offset
    if not np.all(np.isfinite(evaluated_angles)):
        raise ValueError('Field angles must be finite')
    if args.reflect_negative_phi:
        if np.any(nominal_angles < -1e-12):
            raise ValueError('--reflect-negative-phi requires a nonnegative angle grid')
        if not np.any(np.isclose(nominal_angles, 0.0, atol=1e-12)):
            raise ValueError('--reflect-negative-phi requires phi=0 in the angle grid')
    conditions = []
    j_zz_mev = 1000.0 * args.j_zz_ev
    for field in fields:
        for (phi, evaluated_phi) in zip(nominal_angles, evaluated_angles):
            bare_local = local_magnetic_field(field, theta_degrees=args.theta, phi_degrees=evaluated_phi)
            h_local = local_zeeman_field(field, theta_degrees=args.theta, phi_degrees=evaluated_phi, g_z=args.g_z, g_perp=args.g_perp, delta1_j_zz=args.delta1_per_mev * j_zz_mev, delta2_j_zz=args.delta2_per_mev * j_zz_mev)
            conditions.append({'theta_degrees': float(args.theta), 'phi_degrees': float(phi), 'evaluated_phi_degrees': float(evaluated_phi), 'mu_B_B_over_J_zz': float(field), 'bare_field_local': bare_local, 'h_local': h_local, 'initial_spins': zeeman_minimizing_spins(h_local, spin_length=args.spin_length)})
    return conditions

def compute(args, *, data_dir):
    DATA_DIR = Path(data_dir)
    if args.transport_component not in TRANSPORT_COMPONENTS:
        choices = ', '.join(TRANSPORT_COMPONENTS)
        raise ValueError(f'Unknown transport component {args.transport_component!r}; choose one of: {choices}')
    eio_path = DATA_DIR / f'surface_states_summary_nk{EIO_NK}.hdf5'
    output_path = DATA_DIR / 'tto_spin_wave_structure_factor.hdf5'
    conditions_path = DATA_DIR / 'tto_spin_wave_conditions.csv'
    if output_path.exists() and (not args.overwrite):
        raise FileExistsError(f'{output_path} exists; set overwrite=True')
    with h5py.File(eio_path) as eio_h5:
        kf = np.squeeze(eio_h5['kf'][...])
        energies = np.squeeze(eio_h5['eigf'][...])
    q_slab = kf[:, None, :] - kf[None, :, :]
    q_reduced = eio_to_tto_reduced_momentum(q_slab, args.lattice_constant)
    energy_transfer = energies[None, :] - energies[:, None]
    conditions = generate_conditions(args)
    components = ('one_magnon', *ONE_MAGNON_MODE_COMPONENTS, 'two_magnon', 'total')
    shape = (len(conditions), len(TTO_INTERFACE_SUBLATTICES), len(kf), len(kf))
    arrays = {component: np.empty(shape, dtype=float) for component in components}
    diagnostics = []
    for (condition_index, condition) in enumerate(conditions):
        initial_spins = condition['initial_spins']
        h_local = condition['h_local']
        relaxed = relax_ordered_state(initial_spins, h_local, args.j_pm, j_pmpm_ratio=args.j_pmpm, spin_length=args.spin_length, stability_tilt=args.stability_tilt)
        result = surface_projected_dynamic_structure_factor(spin_wave_structure_factor, q_reduced, energy_transfer, relaxed['unit_vectors_local'], h_local, args.j_pm, q_perp_mode=args.q_perp_mode, n_q_perp=args.n_q_perp, j_pmpm_ratio=args.j_pmpm, j_zz_ev=args.j_zz_ev, eta_ev=args.eta_ev, nk=args.dssf_nk, spin_length=args.spin_length, regularization=args.regularization, interface_sublattices=TTO_INTERFACE_SUBLATTICES)
        arrays['one_magnon'][condition_index] = result['one_magnon']
        for component in ONE_MAGNON_MODE_COMPONENTS:
            arrays[component][condition_index] = result[component]
        arrays['two_magnon'][condition_index] = result['two_magnon']
        arrays['total'][condition_index] = combine_transport_components(result['one_magnon'], result['two_magnon'], args.transport_component, one_magnon_modes={component: result[component] for component in ONE_MAGNON_MODE_COMPONENTS})
        diagnostic = {'theta_degrees': condition['theta_degrees'], 'phi_degrees': condition['phi_degrees'], 'evaluated_phi_degrees': condition['evaluated_phi_degrees'], 'mu_B_B_over_J_zz': condition['mu_B_B_over_J_zz'], 'J_pm_over_J_zz': args.j_pm, 'J_pmpm_over_J_zz': args.j_pmpm, 'g_z': args.g_z, 'g_perp': args.g_perp, 'delta1_per_mev': args.delta1_per_mev, 'delta2_per_mev': args.delta2_per_mev, 'q_perp_mode': args.q_perp_mode, 'n_q_perp': args.n_q_perp, 'transport_component': args.transport_component, 'initial_classical_energy': relaxed['initial_energy'], 'relaxed_classical_energy': relaxed['energy'], 'seed_min_classical_curvature': relaxed['seed_min_classical_curvature'], 'min_classical_curvature': relaxed['min_classical_curvature'], 'stability_tilt_used': relaxed['stability_tilt_used'], 'stability_tilt': args.stability_tilt, 'max_torque': relaxed['max_torque'], 'relaxation_converged': relaxed['max_torque'] < 1e-07, 'max_spin_displacement': relaxed['max_spin_displacement'], 'optimizer_success': relaxed['optimizer_success'], 'stationarity_refinement_used': relaxed['stationarity_refinement_used'], 'optimizer_message': relaxed['optimizer_message'], 'max_imaginary_frequency_ratio': result['max_imaginary_frequency_ratio']}
        for mu in range(4):
            for (axis_name, axis) in zip(('x', 'y', 'z'), range(3)):
                diagnostic[f'bare_h_{axis_name}_mu{mu}'] = condition['bare_field_local'][mu, axis]
                diagnostic[f'b_{axis_name}_mu{mu}'] = h_local[mu, axis]
                diagnostic[f'initial_s_{axis_name}_mu{mu}'] = initial_spins[mu, axis]
            for (axis_name, axis) in zip(('x', 'y', 'z'), range(3)):
                diagnostic[f'relaxed_s_{axis_name}_mu{mu}'] = relaxed['spins_local'][mu, axis]
        diagnostics.append(diagnostic)
        if condition_index == 0 or (condition_index + 1) % 10 == 0:
            print(f'Spin-wave DSSF condition {condition_index + 1}/{len(conditions)}', flush=True)
    mode = 'w' if args.overwrite else 'x'
    with h5py.File(output_path, mode) as output:
        output.create_dataset('kf', data=kf)
        output.create_dataset('q_slab', data=q_slab)
        output.create_dataset('q_tto_reduced', data=q_reduced)
        output.create_dataset('q_perp_reduced_offsets', data=q_perp_samples_for_mode(args.q_perp_mode, args.n_q_perp))
        output.create_dataset('energy_transfer_ev', data=energy_transfer)
        output.create_dataset('phi_degrees', data=[c['phi_degrees'] for c in conditions])
        output.create_dataset('evaluated_phi_degrees', data=[c['evaluated_phi_degrees'] for c in conditions])
        field_values = [c['mu_B_B_over_J_zz'] for c in conditions]
        output.create_dataset('mu_B_B_over_J_zz', data=field_values)
        output.create_dataset('h_over_J_zz', data=field_values)
        output.create_dataset('bare_field_local_over_J_zz', data=[c['bare_field_local'] for c in conditions])
        output.create_dataset('effective_field_local_over_J_zz', data=[c['h_local'] for c in conditions])
        for (component, values) in arrays.items():
            output.create_dataset(component, data=values, compression='gzip')
        output.attrs.update(theory='linear spin wave, EIO_TTO Section 1, Eqs. (9)-(21)', ordered_state='classical minimum seeded by Zeeman-minimizing spins', condition_source='generated internally; no external state-table input', momentum_convention='q = k - kprime', q_perp_mode=args.q_perp_mode, n_q_perp=args.n_q_perp, q_perp_reduced_direction=np.ones(3), surface_projection='q_perp=0 slice' if args.q_perp_mode == 'slice' else 'normalized integral over one [111] reciprocal period', energy_convention='omega = epsilon_kprime - epsilon_k', J_pm_over_J_zz=args.j_pm, J_pmpm_over_J_zz=args.j_pmpm, field_energy_definition='mu_B |B|', g_z=args.g_z, g_perp=args.g_perp, delta1_per_mev=args.delta1_per_mev, delta2_per_mev=args.delta2_per_mev, J_zz_ev=args.j_zz_ev, lorentzian_eta_ev=args.eta_ev, dssf_nk=args.dssf_nk, spin_length=args.spin_length, goldstone_regularization_ratio=args.regularization, stability_tilt=args.stability_tilt, tto_cubic_lattice_angstrom=args.lattice_constant, field_theta_degrees=args.theta, phi_offset_degrees=args.phi_offset, phi_start_degrees=args.phi_start, phi_stop_degrees=args.phi_stop, n_phi=args.n_phi, reflect_negative_phi=args.reflect_negative_phi, interface_sublattices=np.asarray(TTO_INTERFACE_SUBLATTICES), transport_component=args.transport_component, transport_total_definition=transport_component_definition(args.transport_component), bragg_included_in_transport=False)
    pd.DataFrame(diagnostics).to_csv(conditions_path, index=False)
    print(f'Wrote {output_path}')
    print(f'Wrote {conditions_path}')
