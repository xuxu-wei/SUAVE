"""Tests for posterior predictive utilities exposed by :class:`SUAVE`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suave import SUAVE, Schema


def _make_mixed_dataset() -> tuple[pd.DataFrame, pd.Series, Schema]:
    X = pd.DataFrame(
        {
            "real": [1.0, 2.5, 3.1, 4.2, 5.5],
            "pos": [0.0, 1.5, 2.0, 3.0, 4.5],
            "count": [0, 1, 2, 3, 4],
            "cat": [0, 1, 0, 1, 1],
            "ordinal": [0, 1, 2, 1, 2],
        }
    )
    y = pd.Series([0, 1, 0, 1, 1], name="target")
    schema = Schema(
        {
            "real": {"type": "real"},
            "pos": {"type": "pos"},
            "count": {"type": "count"},
            "cat": {"type": "cat", "n_classes": 2},
            "ordinal": {"type": "ordinal", "n_classes": 3},
        }
    )
    return X, y, schema


@pytest.fixture(scope="module")
def trained_model() -> tuple[SUAVE, pd.DataFrame]:
    torch.manual_seed(0)
    np.random.seed(0)
    X, y, schema = _make_mixed_dataset()
    model = SUAVE(
        schema=schema,
        latent_dim=6,
        hidden_dims=(16,),
        dropout=0.0,
        n_components=2,
        batch_size=len(X),
    )
    model.fit(X, y, warmup_epochs=1, head_epochs=0, finetune_epochs=0)
    return model, X


def test_attribute_predict_proba_normalises(trained_model) -> None:
    model, X = trained_model
    cat_probs = model.predict_proba(X, attr="cat", L=32)
    assert isinstance(cat_probs, torch.Tensor)
    assert cat_probs.shape == (len(X), 2)
    torch.testing.assert_close(
        cat_probs.sum(dim=1), torch.ones(len(X)), atol=1e-4, rtol=1e-4
    )

    ordinal_probs = model.predict_proba(X, attr="ordinal", L=32)
    assert ordinal_probs.shape == (len(X), 3)
    torch.testing.assert_close(
        ordinal_probs.sum(dim=1), torch.ones(len(X)), atol=1e-4, rtol=1e-4
    )


def test_predict_confidence_interval_shapes(trained_model) -> None:
    model, X = trained_model
    stats_real = model.predict_confidence_interval(X, attr="real", L=128, ci=0.9)
    for key in ("point", "lower", "upper", "std"):
        tensor = stats_real[key]
        assert tensor.shape == (len(X),)
    assert torch.all(stats_real["lower"] <= stats_real["upper"])

    stats_pos = model.predict_confidence_interval(
        X, attr="pos", L=128, return_samples=True
    )
    samples_pos = stats_pos["samples"]
    median_pos = samples_pos.median(dim=1).values
    torch.testing.assert_close(stats_pos["point"], median_pos, atol=1e-4, rtol=1e-4)

    stats_count = model.predict_confidence_interval(
        X, attr="count", L=128, statistic="mean", return_samples=True
    )
    samples_count = stats_count["samples"]
    mean_count = samples_count.mean(dim=1)
    torch.testing.assert_close(stats_count["point"], mean_count, atol=1e-4, rtol=1e-4)


def test_predict_modes_align_with_confidence_interval(trained_model) -> None:
    model, X = trained_model
    probs = model.predict_proba(X, attr="cat", L=64)
    cat_point = model.predict(X, attr="cat", mode="point", L=64)
    torch.testing.assert_close(
        cat_point.float(), probs.argmax(dim=1).float(), atol=1e-4, rtol=1e-4
    )

    cat_sample = model.predict(X, attr="cat", mode="sample", L=32)
    assert cat_sample.shape == (len(X),)

    torch.manual_seed(123)
    stats_real = model.predict_confidence_interval(X, attr="real", L=64)
    torch.manual_seed(123)
    real_point = model.predict(X, attr="real", mode="point", L=64)
    torch.testing.assert_close(real_point, stats_real["point"], atol=1e-4, rtol=1e-4)

    real_sample = model.predict(X, attr="real", mode="sample", L=32)
    assert real_sample.shape == (len(X),)


def test_mask_argument_forces_attribute_missing(trained_model) -> None:
    model, X = trained_model
    torch.manual_seed(321)
    base = model.predict_confidence_interval(X, attr="real", L=64)["point"]
    perturbed = X.copy()
    perturbed["real"] = perturbed["real"] * 1000.0
    mask = pd.DataFrame(False, index=perturbed.index, columns=perturbed.columns)
    torch.manual_seed(321)
    forced = model.predict_confidence_interval(perturbed, attr="real", mask=mask, L=64)[
        "point"
    ]
    torch.testing.assert_close(base, forced, atol=1e-4, rtol=1e-4)
