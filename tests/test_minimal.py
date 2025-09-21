from __future__ import annotations

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
    trained_tau = model._inference_tau
    save_path = tmp_path / "model.json"
    model.save(save_path)
    loaded = SUAVE.load(save_path)
    assert loaded.behaviour == "hivae"
    assert loaded._inference_tau == pytest.approx(trained_tau)
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
