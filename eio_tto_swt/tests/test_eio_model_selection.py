import h5py
import pytest

import eio_model_selection as selection


def test_selection_defaults_and_invalid_values():
    assert selection.validate_eio_model({}) == "original"
    assert selection.validate_eio_model({"eio_model": "symmetrized"}) == "symmetrized"
    with pytest.raises(ValueError, match="eio_model"):
        selection.validate_eio_model({"eio_model": "typo"})


@pytest.mark.parametrize("model", ["original", "symmetrized"])
def test_cache_selection_and_mismatched_run(tmp_path, monkeypatch, model):
    monkeypatch.setattr(selection, "ROOT", tmp_path)
    cache = tmp_path / "data/surface_states_summary_nk24.hdf5"
    cache.parent.mkdir()
    with h5py.File(cache, "w"):
        pass  # Legacy, untagged original cache.
    data = tmp_path / "run/data"
    data.mkdir(parents=True)
    calls = []

    def stage(name, command):
        calls.append(name)
        with h5py.File(data / cache.name, "w") as h5:
            h5.attrs["eio_model"] = model

    selection.prepare_eio({"eio_model": model}, data, stage)
    assert calls == ([] if model == "original" else ["eio"])
    other = "symmetrized" if model == "original" else "original"
    with pytest.raises(ValueError, match="different model"):
        selection.prepare_eio({"eio_model": other}, data, stage)
