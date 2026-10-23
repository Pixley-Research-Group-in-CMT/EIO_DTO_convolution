"""EIO model selection and safe reuse of transport surface-state data."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
MODEL_FILES = ("cellindices.txt", "ham0.txt", "wannier90_centres.xyz")


def validate_eio_model(config):
    model = config.get("eio_model", "original")
    if model not in ("original", "symmetrized"):
        raise ValueError("eio_model must be 'original' or 'symmetrized'")
    return model


def eio_source_files(config):
    directory = (ROOT / "model" if validate_eio_model(config) == "original"
                 else ROOT / "eio_symmetry_clean/model")
    return [ROOT / "eio_model_selection.py", *[directory / name for name in MODEL_FILES]]


def prepare_eio(config, data, stage):
    """Keep legacy original caches usable, but never reuse them for clean runs."""
    import h5py
    model = validate_eio_model(config)
    output = data / "surface_states_summary_nk24.hdf5"

    def matches(path):
        with h5py.File(path) as h5:
            return h5.attrs.get("eio_model", "original") == model

    if output.exists():
        if not matches(output):
            raise ValueError("Existing EIO data uses a different model; choose a new run_label")
        return
    cache = ROOT / "data" / output.name
    if cache.exists() and matches(cache):
        shutil.copy2(cache, output)
    else:
        stage("eio", [str(ROOT / "01_generate_eio.py"), "--overwrite"])
    if not output.exists() or not matches(output):
        raise RuntimeError("EIO generation did not produce the selected model")


def eio_model_label(config):
    """Human-readable model label, including the default for legacy runs."""
    return f"EIO model: {validate_eio_model(config)}"


def saved_eio_model(data):
    """Read the model actually used for saved states, then legacy run metadata."""
    import json
    import h5py
    states = data / "surface_states_summary_nk24.hdf5"
    if states.exists():
        with h5py.File(states) as h5:
            return h5.attrs.get("eio_model", "original")
    manifest = data.parent / "manifest.json"
    if manifest.exists():
        return validate_eio_model(json.loads(manifest.read_text())["config"])
    return "original"
