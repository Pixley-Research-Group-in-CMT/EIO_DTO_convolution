"""Project a bulk TTO structure factor onto the interfacial kagome layer."""
import numpy as np
Q_PERP_MODES = ('slice', 'integrated')
Q_PERP_REDUCED_DIRECTION = np.ones(3, dtype=float)

def validate_q_perp_sampling(mode, n_q_perp):
    """Validate and normalize the surface-projection settings."""
    if mode not in Q_PERP_MODES:
        raise ValueError(f'q_perp_mode must be one of {Q_PERP_MODES}, received {mode!r}')
    if isinstance(n_q_perp, (bool, np.bool_)):
        raise ValueError('n_q_perp must be a positive integer')
    integer = int(n_q_perp)
    if integer != n_q_perp or integer <= 0:
        raise ValueError('n_q_perp must be a positive integer')
    if mode == 'integrated' and integer < 2:
        raise ValueError('integrated q_perp mode requires n_q_perp >= 2')
    return (mode, integer)

def q_perp_reduced_offsets(n_q_perp):
    """Uniform samples of one [111] reciprocal period, including zero.

    In the FCC reduced basis used by the TTO solvers, a reciprocal translation
    normal to the (111) interface is ``(1, 1, 1)``.  The returned coefficients
    therefore parameterize shifts ``t * (1, 1, 1)`` over one periodic interval.
    Including ``t=0`` keeps the slice among the finite-grid samples.
    """
    (_, count) = validate_q_perp_sampling('slice', n_q_perp)
    return (np.arange(count, dtype=float) - count // 2) / count

def lift_q_perp(q_parallel_reduced, n_q_perp):
    """Lift in-plane momenta onto uniform points along the [111] direction."""
    q_parallel = np.asarray(q_parallel_reduced, dtype=float)
    if q_parallel.ndim < 1 or q_parallel.shape[-1] != 3 or (not np.all(np.isfinite(q_parallel))):
        raise ValueError('q_parallel_reduced must end with finite three-component momenta')
    offsets = q_perp_reduced_offsets(n_q_perp)
    offset_shape = (len(offsets),) + (1,) * q_parallel.ndim
    return q_parallel[None, ...] + offsets.reshape(offset_shape) * Q_PERP_REDUCED_DIRECTION

def q_perp_samples_for_mode(mode, n_q_perp):
    """Return the actual reduced offsets used by a selected projection mode."""
    (mode, count) = validate_q_perp_sampling(mode, n_q_perp)
    if mode == 'slice':
        return np.zeros(1, dtype=float)
    return q_perp_reduced_offsets(count)

def _average_projected_result(result, q_perp_axis, n_q_perp):
    projected = {}
    for (name, value) in result.items():
        array = np.asarray(value)
        if array.ndim == 0:
            projected[name] = value
            continue
        if array.shape[q_perp_axis] != n_q_perp:
            raise ValueError(f'Solver result {name!r} does not contain the q_perp axis at position {q_perp_axis}')
        projected[name] = np.mean(array, axis=q_perp_axis)
    return projected

def surface_projected_dynamic_structure_factor(solver, q_parallel_reduced, omega_ev, *solver_args, q_perp_mode='slice', n_q_perp=8, **solver_kwargs):
    """Evaluate a DSSF slice or its normalized surface projection.

    ``slice`` delegates directly to the original solver. ``integrated`` uses a
    periodic rectangle rule for

        S_surface(q_parallel, omega)
          = integral_(-1/2)^(1/2) dt
            S_bulk(q_parallel + t * (1,1,1), omega).

    The interval has unit length in reduced coordinates, so the numerical
    integral is the arithmetic mean over the configured samples.
    """
    (mode, count) = validate_q_perp_sampling(q_perp_mode, n_q_perp)
    q_parallel = np.asarray(q_parallel_reduced, dtype=float)
    omega = np.asarray(omega_ev, dtype=float)
    if q_parallel.shape[:-1] != omega.shape or q_parallel.shape[-1] != 3:
        raise ValueError('q_parallel_reduced must have shape omega_ev.shape + (3,)')
    if mode == 'slice':
        return solver(q_parallel, omega, *solver_args, **solver_kwargs)
    lifted_q = lift_q_perp(q_parallel, count)
    lifted_omega = np.broadcast_to(omega, (count,) + omega.shape)
    result = solver(lifted_q, lifted_omega, *solver_args, **solver_kwargs)
    return _average_projected_result(result, 1, count)
