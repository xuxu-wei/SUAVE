"""Tests for the public imputation interface exposed by :class:`SUAVE`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor
from unittest.mock import patch
from pandas import CategoricalDtype
from torch.nn import functional as F

from suave import data as data_utils
from suave.model import SUAVE
from suave.types import Schema


@pytest.fixture(scope="module")
def _schema() -> Schema:
    return Schema(
        {
            "age": {"type": "real"},
            "gender": {"type": "cat", "n_classes": 2},
            "cholesterol": {"type": "pos"},
        }
    )


def _make_dataset(n_samples: int, *, random_state: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    frame = pd.DataFrame(
        {
            "age": rng.normal(loc=60.0, scale=12.0, size=n_samples),
            "gender": rng.integers(0, 2, size=n_samples),
            "cholesterol": rng.lognormal(mean=np.log(180.0), sigma=0.1, size=n_samples),
        }
    )
    frame["age"] = frame["age"].clip(lower=18.0)
    frame["cholesterol"] = frame["cholesterol"].clip(lower=90.0)
    return frame


def _expected_latent_assignments(
    model: SUAVE, X: pd.DataFrame, *, assignment_strategy: str | None = None
) -> tuple[Tensor, Tensor]:
    aligned = X.loc[:, model.schema.feature_names]
    aligned_reset = aligned.reset_index(drop=True)
    mask = data_utils.build_missing_mask(aligned_reset)
    normalised = model._apply_training_normalization(aligned_reset)
    mask = (mask | normalised.isna()).reset_index(drop=True)
    _, data_tensors, mask_tensors = model._prepare_training_tensors(
        normalised, mask, update_layout=False
    )

    device = model._select_device()
    encoder_inputs = model._prepare_inference_inputs(aligned_reset).to(device)
    for feature_type in data_tensors:
        for column in data_tensors[feature_type]:
            data_tensors[feature_type][column] = data_tensors[feature_type][column].to(
                device
            )
            mask_tensors[feature_type][column] = mask_tensors[feature_type][column].to(
                device
            )

    with torch.no_grad():
        encoder_state = model._encoder.training
        model._encoder.eval()
        logits_enc, mu_enc, logvar_enc = model._encoder(encoder_inputs)
        posterior_mean, _, posterior_probs = model._mixture_posterior_statistics(
            logits_enc,
            mu_enc,
            logvar_enc,
            temperature=model._inference_tau if model.behaviour == "hivae" else None,
        )
        strategy = assignment_strategy
        if strategy is None:
            strategy = "hard" if model.behaviour == "suave" else "soft"
        strategy = strategy.lower()
        if strategy == "hard":
            component_indices = posterior_probs.argmax(dim=-1)
            assignments = F.one_hot(
                component_indices, num_classes=model.n_components
            ).float()
            selected_mu = model._gather_component_parameters(mu_enc, component_indices)
        elif strategy == "soft":
            assignments = posterior_probs
            selected_mu = posterior_mean
        elif strategy == "sample":
            tau = (
                max(float(model._inference_tau), 1e-6)
                if model.behaviour == "hivae"
                else max(float(model.gumbel_temperature), 1e-6)
            )
            assignments = F.gumbel_softmax(logits_enc, tau=tau, hard=False, dim=-1)
            selected_mu = (assignments.unsqueeze(-1) * mu_enc).sum(dim=1)
        else:
            raise AssertionError(f"Unknown assignment strategy '{strategy}'")
        if encoder_state:
            model._encoder.train()
    return selected_mu, assignments


@pytest.mark.parametrize("behaviour", ["hivae", "suave"])
@pytest.mark.parametrize("assignment_strategy", [None, "hard", "soft", "sample"])
def test_impute_matches_expected_latents(
    _schema: Schema, behaviour: str, assignment_strategy: str | None
) -> None:
    train = _make_dataset(64, random_state=42)
    test = train.copy()
    test.loc[::5, "cholesterol"] = np.nan
    test.loc[3, "age"] = np.nan

    model = SUAVE(
        schema=_schema,
        behaviour=behaviour,
        latent_dim=8,
        hidden_dims=(32, 16),
        batch_size=32,
    )
    if behaviour == "suave":
        targets = (train["age"] > train["age"].median()).astype(int).to_numpy()
        model.fit(train, targets, epochs=2, batch_size=32)
    else:
        model.fit(train, epochs=2, batch_size=32)

    if assignment_strategy == "sample":
        torch.manual_seed(0)
    expected_latents, expected_assignments = _expected_latent_assignments(
        model, test, assignment_strategy=assignment_strategy
    )
    if assignment_strategy == "sample":
        torch.manual_seed(0)
    kwargs = {"only_missing": False}
    if assignment_strategy is not None:
        kwargs["assignment_strategy"] = assignment_strategy
    assert model._decoder is not None
    with patch.object(model._decoder, "forward", wraps=model._decoder.forward) as spy:
        _ = model.impute(test, **kwargs)
        spy.assert_called_once()
        latent_arg, assignment_arg = spy.call_args[0][:2]
    assert torch.allclose(latent_arg, expected_latents)
    assert torch.allclose(assignment_arg, expected_assignments)


def test_impute_preserves_observed_entries(_schema: Schema) -> None:
    train = _make_dataset(40, random_state=1)
    model = SUAVE(schema=_schema, behaviour="hivae", latent_dim=6, hidden_dims=(16, 8))
    model.fit(train, epochs=1, batch_size=20)

    test = train.copy()
    test.loc[[0, 5, 10], "age"] = np.nan
    test.loc[[2, 8], "cholesterol"] = np.nan

    imputed = model.impute(test)

    mask = test.isna()
    for column in test.columns:
        observed_mask = ~mask[column]
        output_values = imputed.loc[observed_mask, column]
        input_values = test.loc[observed_mask, column]
        if isinstance(output_values.dtype, CategoricalDtype):
            expected = pd.Categorical(
                input_values,
                categories=output_values.cat.categories,
                ordered=output_values.cat.ordered,
            )
            np.testing.assert_array_equal(output_values.to_numpy(), expected.to_numpy())
        else:
            np.testing.assert_allclose(
                output_values.to_numpy(), input_values.to_numpy()
            )
    assert imputed.isna().sum().sum() == 0


def test_impute_raises_when_not_fitted(_schema: Schema) -> None:
    model = SUAVE(schema=_schema, behaviour="hivae")
    data = _make_dataset(5, random_state=3)
    with pytest.raises(RuntimeError):
        model.impute(data)
