import os
import sys

import numpy as np
import torch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from suave.api import TabVAEClassifier
from suave.modules.calibration import (
    TemperatureScaler,
    expected_calibration_error,
    calibration_curve,
)
from suave.eval.tstr import (
    tstr_auc,
    tstr_vs_trtr,
    check_missingness,
    distribution_comparison,
)
from suave.eval.interpret import latent_feature_correlation, embed_latent


def test_generation_calibration_tstr():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + 0.1 * rng.normal(size=200) > 0).astype(int)

    model = TabVAEClassifier(input_dim=4, latent_dim=4)
    model.fit(X, y, epochs=30)

    df = model.generate(50, conditional={"y": 1}, seed=1)
    assert df.shape == (50, 5)
    assert (df["y"] == 1).all()
    assert not df.isna().any().any()

    logits = model.predict_logits(X)
    probs = model.predict_proba(X)[:, 1]
    ece_before = expected_calibration_error(probs, y)
    scaler = TemperatureScaler().fit(logits, torch.as_tensor(y))
    probs_after = scaler.predict_proba(logits)[:, 1]
    ece_after = expected_calibration_error(probs_after, y)
    assert ece_after <= ece_before + 0.05
    centers, acc, conf = calibration_curve(probs_after, y, n_bins=10)
    assert len(centers) == 10 and len(acc) == 10 and len(conf) == 10

    auc = tstr_auc(model, X, y, X, y)
    auc_svm = tstr_auc(model, X, y, X, y, estimator="svm")
    auc_knn = tstr_auc(model, X, y, X, y, estimator="knn")
    for score in [auc, auc_svm, auc_knn]:
        assert 0.0 <= score <= 1.0

    bench = tstr_vs_trtr(model, X, y, X, y, estimator="rf", n_boot=10, seed=0)
    assert "delta" in bench and "auroc" in bench["delta"]

    gen_full = model.generate(len(X), conditional={"y": y})
    miss = check_missingness(gen_full)
    assert (miss == 0).all()
    comp = distribution_comparison(
        pd.DataFrame(X, columns=[f"x{i}" for i in range(4)]),
        gen_full.drop(columns=["y"]),
    )
    assert set(comp.columns) == {
        "column",
        "real_mean",
        "synth_mean",
        "real_std",
        "synth_std",
    }

    Z = model.latent(X)
    corr, pval, sig = latent_feature_correlation(Z, X)
    assert corr.shape == (4, 4)
    assert pval.shape == (4, 4)
    assert sig.shape == (4, 4)
    emb = embed_latent(Z, method="tsne", random_state=0)
    assert emb.shape == (200, 2)
