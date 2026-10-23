"""Select the inelastic spin-wave components used by transport."""
import numpy as np
from .spin_wave import ONE_MAGNON_MODE_COMPONENTS
TRANSPORT_COMPONENTS = ('one_magnon', *ONE_MAGNON_MODE_COMPONENTS, 'two_magnon', 'total')

def combine_transport_components(one_magnon, two_magnon, component, *, one_magnon_modes=None):
    """Return the requested inelastic response without an elastic Bragg term."""
    if component not in TRANSPORT_COMPONENTS:
        choices = ', '.join(TRANSPORT_COMPONENTS)
        raise ValueError(f'Unknown transport component {component!r}; choose one of: {choices}')
    one_magnon = np.asarray(one_magnon)
    two_magnon = np.asarray(two_magnon)
    if one_magnon.shape != two_magnon.shape:
        raise ValueError('one- and two-magnon responses must have the same shape')
    if component == 'one_magnon':
        return one_magnon
    if component in ONE_MAGNON_MODE_COMPONENTS:
        if one_magnon_modes is None or component not in one_magnon_modes:
            raise ValueError(f'Transport component {component!r} requires its band-resolved one-magnon response')
        selected = np.asarray(one_magnon_modes[component])
        if selected.shape != one_magnon.shape:
            raise ValueError(f'Band-resolved response {component!r} must match one_magnon shape')
        return selected
    if component == 'two_magnon':
        return two_magnon
    return one_magnon + two_magnon

def transport_component_definition(component):
    """Describe the response stored as ``total`` for transport."""
    if component not in TRANSPORT_COMPONENTS:
        choices = ', '.join(TRANSPORT_COMPONENTS)
        raise ValueError(f'Unknown transport component {component!r}; choose one of: {choices}')
    if component == 'total':
        return 'one_magnon + two_magnon (Bragg excluded)'
    if component in ONE_MAGNON_MODE_COMPONENTS:
        band = ONE_MAGNON_MODE_COMPONENTS.index(component) + 1
        return f'one-magnon mode {band} only (Bragg excluded)'
    return f'{component} only (Bragg excluded)'
