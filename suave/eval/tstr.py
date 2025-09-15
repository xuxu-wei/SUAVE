"""Train-on-Synthetic-Test-on-Real (TSTR) evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def tstr_auc(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    estimator: str = "logreg",
) -> float:
    """TSTR AUC using a simple downstream estimator.

    Parameters
    ----------
    model:
        Generator implementing ``generate``.
    X_train, y_train:
        Real training data used only to condition generation size and labels.
    X_test, y_test:
        Real test data for evaluation.
    estimator:
        Downstream model to train on synthetic data. Supported values are
        ``"logreg"`` (logistic regression), ``"svm"`` (RBF-kernel SVM) and
        ``"knn"`` (k-nearest neighbours). SVM and KNN are fitted on
        ``StandardScaler``-normalised features as required by the user.

    Returns
    -------
    float
        ROC-AUC of the estimator evaluated on real test data.
    """

    gen = model.generate(len(X_train))
    gen_X = gen.to_numpy()
    y_gen = y_train[: len(gen_X)]

    if estimator == "logreg":
        clf: Pipeline | LogisticRegression = LogisticRegression(max_iter=1000)
        Xtr = gen_X
        Xte = X_test
    elif estimator == "svm":
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, max_iter=200)),
            ]
        )
        Xtr = gen_X
        Xte = X_test
    elif estimator == "knn":
        clf = Pipeline(
            [("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]
        )
        Xtr = gen_X
        Xte = X_test
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown estimator: {estimator}")

    clf.fit(Xtr, y_gen)

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(Xte)
        if proba.ndim == 2 and proba.shape[1] > 1:
            scores = proba[:, 1]
        else:
            scores = proba.squeeze()
    else:
        scores = clf.decision_function(Xte)
        if scores.ndim != 1:
            exp_s = np.exp(scores)
            scores = exp_s[:, 1] / exp_s.sum(axis=1)

    return float(roc_auc_score(y_test, scores))
