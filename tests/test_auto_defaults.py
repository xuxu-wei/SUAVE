import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from suave import SUAVE, Schema


def _build_dataset(n_samples: int = 24) -> tuple[pd.DataFrame, pd.Series, Schema]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "age": rng.normal(65.0, 8.0, size=n_samples),
            "lactate": rng.normal(2.0, 0.4, size=n_samples),
            "gender": rng.integers(0, 3, size=n_samples),
        }
    )
    y = pd.Series(rng.integers(0, 2, size=n_samples), name="outcome")
    schema = Schema(
        {
            "age": {"type": "real"},
            "lactate": {"type": "real"},
            "gender": {"type": "cat", "n_classes": 3},
        }
    )
    return X, y, schema


def test_auto_defaults_applies_recommendations():
    X, y, schema = _build_dataset()
    model = SUAVE(schema=schema)
    model.fit(X, y)

    auto = model.auto_hyperparameters_
    assert auto is not None
    assert model.latent_dim == auto["latent_dim"]
    assert model.hidden_dims == tuple(auto["hidden_dims"])
    assert model.batch_size == auto["batch_size"]
    assert model.kl_warmup_epochs == auto["kl_warmup_epochs"]
    assert model.warmup_epochs == auto["warmup_epochs"]
    assert model.head_epochs == auto["head_epochs"]
    assert model.finetune_epochs == auto["finetune_epochs"]
    assert model.early_stop_patience == auto["early_stop_patience"]
    assert pytest.approx(model.dropout, rel=1e-6) == auto["dropout"]
    assert pytest.approx(model.learning_rate, rel=1e-9) == auto["learning_rate"]


def test_auto_defaults_respects_manual_overrides():
    X, y, schema = _build_dataset()
    model = SUAVE(
        schema=schema,
        latent_dim=12,
        hidden_dims=(64, 32),
        dropout=0.05,
        learning_rate=5e-4,
        batch_size=6,
        kl_warmup_epochs=7,
        warmup_epochs=9,
        head_epochs=4,
        finetune_epochs=6,
        early_stop_patience=2,
    )
    model.fit(X, y)

    assert model.latent_dim == 12
    assert model.hidden_dims == (64, 32)
    assert pytest.approx(model.dropout, rel=1e-6) == 0.05
    assert pytest.approx(model.learning_rate, rel=1e-9) == 5e-4
    assert model.batch_size == 6
    assert model.kl_warmup_epochs == 7
    assert model.warmup_epochs == 9
    assert model.head_epochs == 4
    assert model.finetune_epochs == 6
    assert model.early_stop_patience == 2
    assert model._auto_configured["latent_dim"] is False
    assert model._auto_configured["hidden_dims"] is False


def test_auto_defaults_disabled(tmp_path):
    X, y, schema = _build_dataset()
    model = SUAVE(schema=schema, auto_parameters=False)
    model.fit(X, y)

    assert model.auto_hyperparameters_ is None
    assert model.latent_dim == 32
    assert pytest.approx(model.dropout, rel=1e-6) == 0.1

    path = tmp_path / "auto_defaults.pt"
    model.save(path)
    restored = SUAVE.load(path)
    assert restored.auto_parameters is False
    assert restored.auto_hyperparameters_ is None


def test_auto_defaults_persist_after_serialisation(tmp_path):
    X, y, schema = _build_dataset()
    model = SUAVE(schema=schema)
    model.fit(X, y)

    path = tmp_path / "suave_auto.pt"
    model.save(path)
    restored = SUAVE.load(path)

    assert restored.auto_parameters is True
    assert restored.auto_hyperparameters_ == model.auto_hyperparameters_
    assert restored._auto_configured == model._auto_configured
