"""EIO model selection and safe reuse of transport surface-state data."""

def validate_eio_model(config):
    model = config.get('eio_model', 'original')
    if model not in ('original', 'symmetrized'):
        raise ValueError("eio_model must be 'original' or 'symmetrized'")
    return model

def eio_model_label(config):
    """Human-readable model label, including the default for legacy runs."""
    return f'EIO model: {validate_eio_model(config)}'

def saved_eio_model(data):
    """Read the model actually used for saved states, then legacy run metadata."""
    import json
    import h5py
    states = data / 'surface_states_summary_nk24.hdf5'
    if states.exists():
        with h5py.File(states) as h5:
            return h5.attrs.get('eio_model', 'original')
    manifest = data.parent / 'manifest.json'
    if manifest.exists():
        return validate_eio_model(json.loads(manifest.read_text())['config'])
    return 'original'
