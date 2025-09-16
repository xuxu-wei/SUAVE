import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from suave.api import SUAVE
from suave.modules.calibration import (
    TemperatureScaler,
    calibration_curve,
    expected_calibration_error,
)
from suave.eval.tstr import tstr_auc, tstr_vs_trtr
from suave.eval.interpret import latent_feature_correlation, latent_projection


def test_generation_calibration_tstr():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + 0.1 * rng.normal(size=200) > 0).astype(int)

    model = SUAVE(input_dim=4, latent_dim=2)
    model.fit(X, y, epochs=30)

    df = model.generate(50, seed=1)
    assert df.shape == (50, 4)
    assert not df.isna().any().any()

    df_pos = model.generate(10, conditional={"y": np.ones(10)}, seed=2)
    assert df_pos.shape == (10, 4)
    assert not df_pos.isna().any().any()

    logits = model.predict_logits(X)
    probs = model.predict_proba(X)[:, 1]
    ece_before = expected_calibration_error(probs, y)
    scaler = TemperatureScaler().fit(logits, torch.as_tensor(y))
    probs_after = scaler.predict_proba(logits)[:, 1]
    ece_after = expected_calibration_error(probs_after, y)
    assert ece_after <= ece_before + 0.05

    bins, acc, conf = calibration_curve(probs, y, n_bins=10)
    assert len(bins) <= 10
    assert bins.shape == acc.shape == conf.shape

    auc = tstr_auc(model, X, y, X, y)
    auc_svm = tstr_auc(model, X, y, X, y, estimator="svm")
    auc_knn = tstr_auc(model, X, y, X, y, estimator="knn")
    for score in [auc, auc_svm, auc_knn]:
        assert 0.0 <= score <= 1.0

    res = tstr_vs_trtr(model, X, y, X, y, estimator="logreg", n_boot=5, seed=0)
    assert set(res.keys()) == {"tstr", "trtr", "delta", "diagnostics"}
    assert "auroc" in res["tstr"]
    assert res["diagnostics"]["missing"]["synthetic"].max() == 0.0

    Z = model.latent(X)
    corr, signif = latent_feature_correlation(Z, X)
    assert corr.shape == signif.shape
    emb = latent_projection(
        Z[:50], method="tsne", max_iter=250, perplexity=5, init="random", learning_rate="auto"
    )
    assert emb.shape == (50, 2)


def test_suave_fit_with_minibatches() -> None:
    rng = np.random.default_rng(42)
    torch.manual_seed(0)
    X = rng.normal(size=(64, 5))
    missing_mask = rng.random(X.shape) < 0.1
    X[missing_mask] = np.nan
    y = rng.integers(0, 2, size=64)

    model = SUAVE(input_dim=5, latent_dim=3)
    returned = model.fit(X, y, epochs=4, batch_size=16)
    assert returned is model
    assert np.isclose(model.label_distribution.sum(), 1.0)
    proba = model.predict_proba(X)
    assert proba.shape == (64, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(64), atol=1e-5)


def test_suave_minibatch_matches_full_batch() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] + 0.2 * rng.normal(size=40) > 0).astype(int)

    torch.manual_seed(1)
    full = SUAVE(input_dim=3, latent_dim=2)
    full.fit(X, y, epochs=5)
    full_proba = full.predict_proba(X)

    torch.manual_seed(1)
    mini = SUAVE(input_dim=3, latent_dim=2)
    mini.fit(X, y, epochs=5, batch_size=8)
    mini_proba = mini.predict_proba(X)

    assert full_proba.shape == mini_proba.shape == (40, 2)
    diff = np.abs(full_proba - mini_proba).mean()
    assert diff < 0.15


def test_suave_fit_invalid_batch_size() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(10, 2))
    y = rng.integers(0, 2, size=10)

    model = SUAVE(input_dim=2, latent_dim=2)
    with pytest.raises(ValueError):
        model.fit(X, y, batch_size=0)


def test_tstr_multiclass_support() -> None:
    rng = np.random.default_rng(123)
    X_train = rng.normal(size=(120, 5))
    y_train = rng.integers(0, 3, size=120)
    X_test = rng.normal(size=(60, 5))
    y_test = rng.integers(0, 3, size=60)

    class EchoGenerator:
        def __init__(self, data: np.ndarray):
            self._data = np.asarray(data)

        def generate(self, n: int, conditional=None, seed: int | None = None):  # noqa: ANN001
            if seed is None:
                idx = np.arange(n) % len(self._data)
            else:
                rng_local = np.random.default_rng(seed)
                idx = rng_local.integers(0, len(self._data), size=n)
            generated = self._data[idx].copy()
            return pd.DataFrame(generated)

    model = EchoGenerator(X_train)

    auc = tstr_auc(model, X_train, y_train, X_test, y_test, estimator="logreg")
    assert 0.0 <= auc <= 1.0

    results = tstr_vs_trtr(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        estimator="rf",
        n_boot=3,
        seed=0,
    )
    assert set(results.keys()) == {"tstr", "trtr", "delta", "diagnostics"}
    assert 0.0 <= results["tstr"]["auroc"] <= 1.0
    assert 0.0 <= results["tstr"]["auprc"] <= 1.0

    def mock_predict_scores(clf, X):  # noqa: ANN001
        return np.ones(X.shape[0])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("suave.eval.tstr._predict_scores", mock_predict_scores)
        with pytest.raises(ValueError, match="Multi-class evaluation requires probability"):
            tstr_auc(model, X_train, y_train, X_test, y_test, estimator="logreg")
