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
    model = SUAVE(schema=schema)
    model.fit(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    uniform = np.full_like(probabilities, 1.0 / probabilities.shape[1])
    assert not np.allclose(probabilities, uniform)


def test_hivae_behaviour_disables_classifier():
    X, _, schema = make_dataset()
    model = SUAVE(schema=schema, behaviour="hivae")
    model.fit(X, epochs=1)
    latent = model.encode(X)
    assert latent.shape[0] == len(X)
    with pytest.raises(RuntimeError):
        model.predict_proba(X)


def test_hivae_behaviour_persists_after_save(tmp_path: Path):
    X, _, schema = make_dataset()
    model = SUAVE(schema=schema, behaviour="hivae")
    model.fit(X, epochs=1)
    save_path = tmp_path / "model.json"
    model.save(save_path)
    loaded = SUAVE.load(save_path)
    assert loaded.behaviour == "hivae"
    with pytest.raises(RuntimeError):
        loaded.predict_proba(X)
