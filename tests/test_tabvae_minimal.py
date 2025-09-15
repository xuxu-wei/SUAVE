import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from suave.api import TabVAEClassifier
from suave.modules.calibration import TemperatureScaler, expected_calibration_error
from suave.eval.tstr import tstr_auc


def test_generation_calibration_tstr():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + 0.1 * rng.normal(size=200) > 0).astype(int)

    model = TabVAEClassifier(input_dim=4, latent_dim=2)
    model.fit(X, y, epochs=30)

    df = model.generate(50, seed=1)
    assert df.shape == (50, 4)
    assert not df.isna().any().any()

    logits = model.predict_logits(X)
    probs = model.predict_proba(X)[:, 1]
    ece_before = expected_calibration_error(probs, y)
    scaler = TemperatureScaler().fit(logits, torch.as_tensor(y))
    probs_after = scaler.predict_proba(logits)[:, 1]
    ece_after = expected_calibration_error(probs_after, y)
    assert ece_after <= ece_before + 0.05

    auc = tstr_auc(model, X, y, X, y)
    auc_svm = tstr_auc(model, X, y, X, y, estimator="svm")
    auc_knn = tstr_auc(model, X, y, X, y, estimator="knn")
    for score in [auc, auc_svm, auc_knn]:
        assert 0.0 <= score <= 1.0
