from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

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


def _serialise_for_legacy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {key: _serialise_for_legacy(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise_for_legacy(item) for item in value]
    return value


def _build_legacy_payload(payload: dict) -> dict:
    metadata = payload.get("metadata", {})
    modules = payload.get("modules", {})
    artefacts = payload.get("artefacts", {})

    legacy = {
        "schema": metadata.get("schema"),
        "behaviour": metadata.get("behaviour"),
        "latent_dim": metadata.get("latent_dim"),
        "n_components": metadata.get("n_components"),
        "hidden_dims": metadata.get("hidden_dims"),
        "dropout": metadata.get("dropout"),
        "learning_rate": metadata.get("learning_rate"),
        "batch_size": metadata.get("batch_size"),
        "kl_warmup_epochs": metadata.get("kl_warmup_epochs"),
        "val_split": metadata.get("val_split"),
        "stratify": metadata.get("stratify"),
        "random_state": metadata.get("random_state"),
        "tau_start": metadata.get("tau_start"),
        "tau_min": metadata.get("tau_min"),
        "tau_decay": metadata.get("tau_decay"),
        "inference_tau": metadata.get("inference_tau"),
        "modules": _serialise_for_legacy(modules),
        "artefacts": _serialise_for_legacy(artefacts),
    }

    if "prior" not in legacy and "prior" in legacy["modules"]:
        legacy["prior"] = legacy["modules"]["prior"]

    classes = legacy["artefacts"].get("classes")
    if classes is not None:
        legacy["classes"] = classes
    class_to_index = legacy["artefacts"].get("class_to_index")
    if class_to_index is not None:
        legacy["class_to_index"] = class_to_index

    return legacy


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


def test_predict_proba_uses_and_updates_cache():
    X, y, schema = make_dataset()
    model = SUAVE(schema=schema, n_components=2)
    model.fit(X, y, epochs=1)

    first = model.predict_proba(X)
    assert np.allclose(first.sum(axis=1), 1.0)
    assert model._cached_logits is not None
    assert model._cached_probabilities is not None
    assert model._logits_cache_key is not None
    assert model._probability_cache_key is not None

    cached_logits = model._cached_logits
    cached_probabilities = model._cached_probabilities

    with patch.object(
        SUAVE,
        "_compute_logits",
        side_effect=AssertionError(
            "_compute_logits should not run when cache is valid"
        ),
    ):
        second = model.predict_proba(X)
        third = model.predict(X)

    assert second is cached_probabilities
    assert model._cached_logits is cached_logits
    assert third.shape == (len(X),)


def test_predict_proba_refreshes_cache_for_new_frames():
    X, y, schema = make_dataset()
    model = SUAVE(schema=schema, n_components=2)
    model.fit(X, y, epochs=1)

    _ = model.predict_proba(X)
    previous_logits = model._cached_logits.copy()
    previous_probabilities = model._cached_probabilities.copy()
    previous_logit_key = model._logits_cache_key
    previous_proba_key = model._probability_cache_key

    shuffled = X.iloc[::-1].reset_index(drop=True)
    refreshed = model.predict_proba(shuffled)

    assert model._logits_cache_key != previous_logit_key
    assert model._probability_cache_key != previous_proba_key
    assert not np.shares_memory(model._cached_logits, previous_logits)
    assert not np.shares_memory(model._cached_probabilities, previous_probabilities)
    assert refreshed.shape == (len(X), 2)


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


def test_encode_return_components_dict():
    X, y, schema = make_dataset()
    model = SUAVE(schema=schema, latent_dim=4, batch_size=2, n_components=3)
    model.fit(X, y, epochs=1)
    encoded = model.encode(X, return_components=True)
    assert set(encoded) == {
        "mean",
        "assignments",
        "component_mu",
        "component_logvar",
    }
    n_samples = len(X)
    assert encoded["mean"].shape == (n_samples, model.latent_dim)
    assert encoded["assignments"].shape == (n_samples, model.n_components)
    assert encoded["component_mu"].shape == (n_samples, model.latent_dim)
    assert encoded["component_logvar"].shape == (n_samples, model.latent_dim)


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
    expected_logits = model._cached_logits.copy()
    expected_latents = model.encode(X)
    scaler_state = model._temperature_scaler_state.copy() if model._temperature_scaler_state else None
    tensor_attrs = [
        "_train_latent_mu",
        "_train_latent_logvar",
        "_train_component_logits",
        "_train_component_mu",
        "_train_component_logvar",
        "_train_component_probs",
    ]
    tensor_snapshots = {
        attr: getattr(model, attr).clone().detach() if getattr(model, attr) is not None else None
        for attr in tensor_attrs
    }
    target_indices_snapshot = None
    if model._train_target_indices is not None:
        target_indices_snapshot = model._train_target_indices.copy()
    save_path = tmp_path / "model.pt"
    model.save(save_path)

    reloaded = SUAVE.load(save_path)
    assert reloaded.behaviour == "suave"
    assert reloaded._classifier is not None
    assert reloaded._is_calibrated is True
    assert reloaded._temperature_scaler_state is not None
    if scaler_state is not None:
        assert reloaded._temperature_scaler_state["fitted"] is scaler_state["fitted"]
        assert reloaded._temperature_scaler_state["temperature"] == pytest.approx(
            scaler_state["temperature"]
        )
    np.testing.assert_allclose(expected_logits, reloaded._cached_logits)
    np.testing.assert_allclose(expected_probabilities, reloaded._cached_probabilities)
    reloaded_probabilities = reloaded.predict_proba(X)
    np.testing.assert_allclose(expected_probabilities, reloaded_probabilities)
    np.testing.assert_allclose(expected_latents, reloaded.encode(X))
    samples = reloaded.sample(2)
    assert isinstance(samples, pd.DataFrame)
    assert list(samples.columns) == list(schema.feature_names)
    assert len(samples) == 2
    if target_indices_snapshot is not None:
        np.testing.assert_array_equal(
            target_indices_snapshot, reloaded._train_target_indices
        )
    for attr, snapshot in tensor_snapshots.items():
        loaded_value = getattr(reloaded, attr)
        if snapshot is None:
            assert loaded_value is None
        else:
            assert loaded_value is not None
            assert torch.allclose(loaded_value, snapshot)


def test_save_payload_contains_classifier_temperature_and_prior(tmp_path: Path):
    X, y, schema = make_dataset()
    model = SUAVE(schema=schema, latent_dim=4, batch_size=2, n_components=2)
    model.fit(X, y, epochs=1)
    model.calibrate(X, y)
    expected_logits = model._cached_logits.copy()
    expected_probabilities = model._cached_probabilities.copy()
    save_path = tmp_path / "payload.pt"
    model.save(save_path)

    payload = torch.load(save_path, map_location="cpu")
    metadata = payload["metadata"]
    assert metadata["behaviour"] == "suave"
    modules = payload["modules"]
    classifier_payload = modules["classifier"]
    assert classifier_payload is not None
    assert classifier_payload["state_dict"]
    assert "linear.weight" in classifier_payload["state_dict"]
    assert modules["temperature_scaler"]["fitted"] is True
    prior_state = modules["prior"]
    for key in ("logits", "logvar", "mu"):
        assert key in prior_state
        assert isinstance(prior_state[key], torch.Tensor)
    artefacts = payload["artefacts"]
    assert artefacts["is_calibrated"] is True
    np.testing.assert_allclose(artefacts["cached_logits"], expected_logits)
    np.testing.assert_allclose(
        artefacts["cached_probabilities"], expected_probabilities
    )


def test_hivae_behaviour_disables_classifier():
    X, y, schema = make_dataset()
    model = SUAVE(schema=schema, behaviour="hivae", n_components=2)
    model.fit(X, epochs=1)
    latent = model.encode(X)
    assert latent.shape[0] == len(X)
    model._cached_logits = np.array([0.0])
    model._cached_probabilities = np.array([[0.5, 0.5]])
    with pytest.raises(RuntimeError):
        model._ensure_classifier_available("test")
    with pytest.raises(RuntimeError):
        model._compute_logits(X)
    with pytest.raises(RuntimeError):
        model.predict_proba(X)
    with pytest.raises(RuntimeError):
        model.predict(X)
    with pytest.raises(RuntimeError):
        model.calibrate(X, y)
    assert model._cached_logits is None
    assert model._cached_probabilities is None


def test_hivae_behaviour_persists_after_save(tmp_path: Path):
    X, _, schema = make_dataset()
    model = SUAVE(schema=schema, behaviour="hivae", n_components=2)
    model.fit(X, epochs=1)
    trained_tau = model._inference_tau
    save_path = tmp_path / "model.json"
    model.save(save_path)
    loaded = SUAVE.load(save_path)
    assert loaded.behaviour == "hivae"
    assert loaded._inference_tau == pytest.approx(trained_tau)
    assert loaded._classifier is None
    assert loaded._cached_logits is None
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


def test_hivae_inference_temperature_tracks_training_progress():
    X, _, schema = make_dataset()
    model = SUAVE(
        schema=schema,
        behaviour="hivae",
        n_components=2,
        tau_start=1.0,
        tau_min=0.5,
        tau_decay=0.2,
    )
    model.fit(X, epochs=3)
    expected_tau = model._gumbel_temperature_for_epoch(2)
    assert model._inference_tau == pytest.approx(expected_tau)


def test_hivae_prior_mean_layer_trains_and_samples():
    X, _, schema = make_dataset()
    model = SUAVE(
        schema=schema,
        behaviour="hivae",
        n_components=2,
        latent_dim=4,
        batch_size=2,
    )
    assert model._prior_mean_layer is not None
    initial_params = [
        param.detach().cpu().numpy().copy()
        for param in model._prior_mean_layer.parameters()
    ]
    model.fit(X, epochs=3)
    samples = model.sample(3)
    assert isinstance(samples, pd.DataFrame)
    assert len(samples) == 3
    updated_params = [
        param.detach().cpu().numpy() for param in model._prior_mean_layer.parameters()
    ]
    assert any(
        not np.allclose(initial, updated)
        for initial, updated in zip(initial_params, updated_params)
    )


def test_hivae_impute_assignment_modes_route_expected_weights():
    X, _, schema = make_dataset()
    model = SUAVE(
        schema=schema,
        behaviour="hivae",
        n_components=2,
        latent_dim=4,
        batch_size=2,
    )
    model.fit(X, epochs=1)
    assert model._decoder is not None

    aligned = X.loc[:, schema.feature_names].reset_index(drop=True)
    device = model._select_device()
    encoder_inputs = model._prepare_inference_inputs(aligned).to(device)
    with torch.no_grad():
        encoder_state = model._encoder.training
        model._encoder.eval()
        logits, mu, logvar = model._encoder(encoder_inputs)
        _, _, posterior_probs = model._mixture_posterior_statistics(
            logits,
            mu,
            logvar,
            temperature=model._inference_tau,
        )
        if encoder_state:
            model._encoder.train()
    hard_expected = F.one_hot(
        posterior_probs.argmax(dim=-1), num_classes=model.n_components
    ).float()

    def _capture(strategy: str) -> torch.Tensor:
        assert model._decoder is not None
        with patch.object(
            model._decoder, "forward", wraps=model._decoder.forward
        ) as spy:
            torch.manual_seed(0)
            model.impute(
                X,
                only_missing=False,
                assignment_strategy=strategy,
            )
            spy.assert_called_once()
            captured = spy.call_args[0][1].detach().cpu()
        return captured

    assignments_hard = _capture("hard")
    assert torch.allclose(assignments_hard, hard_expected.cpu())

    assignments_soft = _capture("soft")
    assert torch.allclose(assignments_soft, posterior_probs.cpu(), atol=1e-6)

    assignments_sample = _capture("sample")
    assert not torch.allclose(assignments_sample, hard_expected.cpu())
    assert not torch.allclose(assignments_sample, posterior_probs.cpu())
    assert torch.all(assignments_sample > 0.0)
    assert torch.allclose(
        assignments_sample.sum(dim=-1),
        torch.ones(assignments_sample.size(0)),
        atol=1e-6,
    )


def test_legacy_loader_respects_behaviour_modes(tmp_path: Path):
    X, y, schema = make_dataset()

    suave_model = SUAVE(schema=schema, latent_dim=4, batch_size=2, n_components=2)
    suave_model.fit(X, y, epochs=1)
    suave_model.calibrate(X, y)
    expected_suave_probs = suave_model.predict_proba(X)
    expected_suave_logits = suave_model._cached_logits.copy()
    expected_suave_latents = suave_model.encode(X)
    modern_suave_path = tmp_path / "modern_suave.pt"
    suave_model.save(modern_suave_path)
    suave_payload = torch.load(modern_suave_path, map_location="cpu")
    legacy_suave = _build_legacy_payload(suave_payload)
    legacy_suave_path = tmp_path / "legacy_suave.json"
    legacy_suave_path.write_text(json.dumps(legacy_suave))
    loaded_suave = SUAVE.load(legacy_suave_path)
    assert loaded_suave.behaviour == "suave"
    assert loaded_suave._classifier is not None
    np.testing.assert_allclose(expected_suave_logits, loaded_suave._cached_logits)
    np.testing.assert_allclose(
        expected_suave_probs, loaded_suave._cached_probabilities
    )
    np.testing.assert_allclose(expected_suave_latents, loaded_suave.encode(X))
    np.testing.assert_allclose(expected_suave_probs, loaded_suave.predict_proba(X))

    hivae_model = SUAVE(
        schema=schema, behaviour="hivae", latent_dim=4, batch_size=2, n_components=2
    )
    hivae_model.fit(X, epochs=1)
    expected_hivae_latents = hivae_model.encode(X)
    modern_hivae_path = tmp_path / "modern_hivae.pt"
    hivae_model.save(modern_hivae_path)
    hivae_payload = torch.load(modern_hivae_path, map_location="cpu")
    legacy_hivae = _build_legacy_payload(hivae_payload)
    legacy_hivae_path = tmp_path / "legacy_hivae.json"
    legacy_hivae_path.write_text(json.dumps(legacy_hivae))
    loaded_hivae = SUAVE.load(legacy_hivae_path)
    assert loaded_hivae.behaviour == "hivae"
    assert loaded_hivae._classifier is None
    assert loaded_hivae._cached_logits is None
    np.testing.assert_allclose(expected_hivae_latents, loaded_hivae.encode(X))
    with pytest.raises(RuntimeError):
        loaded_hivae.predict_proba(X)
