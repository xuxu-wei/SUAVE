from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from suave import SUAVE, Schema


def make_dataset() -> tuple[pd.DataFrame, pd.Series, Schema]:
    X = pd.DataFrame(
        {
            "age": [65, 72, 58, 61, 70],
            "sofa_score": [5, 7, 4, 6, 8],
            "gender": [0, 1, 1, 0, 0],
        }
    )
    y = pd.Series([0, 1, 0, 1, 1], name="outcome")
    schema = Schema(
        {
            "age": {"type": "real"},
            "sofa_score": {"type": "real"},
            "gender": {"type": "cat", "n_classes": 2},
        }
    )
    return X, y, schema


def test_package_importable():
    import suave  # noqa: F401


@pytest.mark.parametrize("epochs", [1])
def test_fit_logs(caplog, epochs):
    caplog.set_level("INFO")
    X, y, schema = make_dataset()
    model = SUAVE(schema=schema)
    model.fit(X, y, epochs=epochs)
    assert any("Fit complete" in record.message for record in caplog.records)


def test_predict_proba_shape():
    X, y, schema = make_dataset()
    model = SUAVE(schema=schema, n_components=2)
    model.fit(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    uniform = np.full_like(probabilities, 1.0 / probabilities.shape[1])
    assert not np.allclose(probabilities, uniform)


def test_encode_returns_latent_means():
    X, y, schema = make_dataset()
    model = SUAVE(schema=schema, latent_dim=4, batch_size=2, n_components=2)
    model.fit(X, y)
    assert model._encoder is not None
    was_training = model._encoder.training
    latents = model.encode(X)
    assert latents.shape == (len(X), model.latent_dim)
    assert latents.dtype == np.float32
    assert model._encoder is not None
    assert model._encoder.training == was_training


def test_sample_generates_schema_aligned_dataframe():
    X, y, schema = make_dataset()
    model = SUAVE(schema=schema, latent_dim=4, batch_size=2, n_components=2)
    model.fit(X, y, epochs=1)
    samples = model.sample(3)
    assert isinstance(samples, pd.DataFrame)
    assert list(samples.columns) == list(schema.feature_names)
    assert len(samples) == 3
    numeric = samples.select_dtypes(include=[np.number])
    if not numeric.empty:
        assert not np.allclose(numeric.to_numpy(), 0.0)
    assert samples["gender"].dtype == "category"


def test_conditional_sampling_validates_labels():
    X, y, schema = make_dataset()
    model = SUAVE(schema=schema, latent_dim=4, batch_size=2, n_components=2)
    model.fit(X, y, epochs=1)
    requested = np.array([0, 1, 1])
    samples = model.sample(len(requested), conditional=True, y=requested)
    assert len(samples) == len(requested)
    with pytest.raises(ValueError):
        model.sample(2, conditional=True, y=np.array([2, 2]))


def test_save_load_predict_round_trip(tmp_path: Path):
    X, y, schema = make_dataset()
    model = SUAVE(schema=schema, latent_dim=4, batch_size=2, n_components=2)
    model.fit(X, y, epochs=1)
    model.calibrate(X, y)
    expected_probabilities = model.predict_proba(X)
    expected_latents = model.encode(X)
    save_path = tmp_path / "model.pt"
    model.save(save_path)

    reloaded = SUAVE.load(save_path)
    reloaded_probabilities = reloaded.predict_proba(X)
    np.testing.assert_allclose(expected_probabilities, reloaded_probabilities)
    np.testing.assert_allclose(expected_latents, reloaded.encode(X))
    samples = reloaded.sample(2)
    assert isinstance(samples, pd.DataFrame)
    assert list(samples.columns) == list(schema.feature_names)
    assert len(samples) == 2


def test_hivae_behaviour_disables_classifier():
    X, _, schema = make_dataset()
    model = SUAVE(schema=schema, behaviour="hivae", n_components=2)
    model.fit(X, epochs=1)
    latent = model.encode(X)
    assert latent.shape[0] == len(X)
    with pytest.raises(RuntimeError):
        model.predict_proba(X)


def test_hivae_behaviour_persists_after_save(tmp_path: Path):
    X, _, schema = make_dataset()
    model = SUAVE(schema=schema, behaviour="hivae", n_components=2)
    model.fit(X, epochs=1)
    save_path = tmp_path / "model.json"
    model.save(save_path)
    loaded = SUAVE.load(save_path)
    assert loaded.behaviour == "hivae"
    with pytest.raises(RuntimeError):
        loaded.predict_proba(X)


def test_hivae_temperature_without_annealing_is_reproducible():
    X, _, schema = make_dataset()
    model = SUAVE(
        schema=schema,
        behaviour="hivae",
        n_components=2,
        tau_start=1.0,
        tau_min=1.0,
        tau_decay=0.0,
    )
    model.fit(X, epochs=1)
    assert model._gumbel_temperature_for_epoch(0) == pytest.approx(1.0)
    assert model._inference_tau == pytest.approx(1.0)
    latents_first = model.encode(X)
    latents_second = model.encode(X)
    np.testing.assert_allclose(latents_first, latents_second)
