"""Train-on-Synthetic-Test-on-Real (TSTR) evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def tstr_auc(model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Train a logistic regression on generated samples and evaluate on real data."""
    gen = model.generate(len(X_train))
    gen_X = gen.to_numpy()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(gen_X, y_train[: len(gen_X)])
    proba = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, proba))
