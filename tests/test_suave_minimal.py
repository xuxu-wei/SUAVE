import os
import sys

import numpy as np
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
