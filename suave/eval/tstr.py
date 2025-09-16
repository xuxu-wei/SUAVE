"""Train-on-Synthetic-Test-on-Real (TSTR) evaluation."""

from __future__ import annotations

import numpy as np
from functools import partial

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:  # optional dependency
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional
    XGBClassifier = None


def _select_binary_scores(
    y_true: np.ndarray,
    proba: np.ndarray,
    class_order: np.ndarray | None = None,
) -> np.ndarray:
    """Return scores aligned with the positive class for binary problems."""

    if proba.ndim == 1 or proba.shape[1] == 1:
        return proba.squeeze()

    classes = np.unique(y_true)
    if classes.size < 2:
        return proba.squeeze()

    order = np.asarray(class_order) if class_order is not None else classes
    if order.ndim != 1:
        order = order.ravel()
    if order.shape[0] != proba.shape[1]:
        order = classes if classes.size == proba.shape[1] else np.arange(proba.shape[1])

    pos_label = classes[-1]
    matches = np.flatnonzero(order == pos_label)
    idx = int(matches[0]) if matches.size else proba.shape[1] - 1
    return proba[:, idx]


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

    class_order = getattr(clf, "classes_", None)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(Xte)
        scores = _select_binary_scores(y_test, proba, class_order)
    else:
        scores = clf.decision_function(Xte)
        if scores.ndim != 1:
            exp_s = np.exp(scores)
            norm = exp_s / exp_s.sum(axis=1, keepdims=True)
            scores = _select_binary_scores(y_test, norm, class_order)

    return float(roc_auc_score(y_test, scores))


def _estimator_factory(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=1000)
    if name == "rf":
        return RandomForestClassifier(n_estimators=100)
    if name == "xgboost":
        if XGBClassifier is None:  # pragma: no cover - optional
            raise ImportError("xgboost is required for estimator='xgboost'")
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    raise ValueError(name)


def _predict_scores(clf, X: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    class_order = getattr(clf, "classes_", None)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        return _select_binary_scores(y_true, proba, class_order)
    scores = clf.decision_function(X)
    if scores.ndim != 1:
        exp_s = np.exp(scores)
        norm = exp_s / exp_s.sum(axis=1, keepdims=True)
        return _select_binary_scores(y_true, norm, class_order)
    return scores


def _bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric,
    n_boot: int = 100,
    seed: int | None = None,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats_list = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats_list.append(metric(y_true[idx], scores[idx]))
    stats = np.sort(np.asarray(stats_list))
    lo = float(np.percentile(stats, 2.5))
    hi = float(np.percentile(stats, 97.5))
    return lo, hi


def _data_diagnostics(real: np.ndarray, synthetic: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    return {
        "missing": {
            "real": np.isnan(real).mean(axis=0),
            "synthetic": np.isnan(synthetic).mean(axis=0),
        },
        "mean": {
            "real": np.nanmean(real, axis=0),
            "synthetic": np.nanmean(synthetic, axis=0),
        },
        "std": {
            "real": np.nanstd(real, axis=0),
            "synthetic": np.nanstd(synthetic, axis=0),
        },
    }


def tstr_vs_trtr(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    estimator: str = "logreg",
    n_boot: int = 100,
    seed: int | None = None,
) -> dict[str, object]:
    """Compare TSTR and TRTR pipelines with confidence intervals."""

    gen = model.generate(len(X_train), conditional={"y": y_train})
    X_syn = gen.to_numpy()
    y_syn = y_train[: len(X_syn)]

    est_tstr = _estimator_factory(estimator)
    est_trtr = _estimator_factory(estimator)
    est_tstr.fit(X_syn, y_syn)
    est_trtr.fit(X_train, y_train)

    scores_tstr = _predict_scores(est_tstr, X_test, y_test)
    scores_trtr = _predict_scores(est_trtr, X_test, y_test)

    classes = np.unique(y_test)
    pos_label = classes[-1]

    auc_tstr = roc_auc_score(y_test, scores_tstr)
    auc_trtr = roc_auc_score(y_test, scores_trtr)
    pr_tstr = average_precision_score(y_test, scores_tstr, pos_label=pos_label)
    pr_trtr = average_precision_score(y_test, scores_trtr, pos_label=pos_label)

    ci_auc_tstr = _bootstrap_ci(y_test, scores_tstr, roc_auc_score, n_boot, seed)
    ci_auc_trtr = _bootstrap_ci(y_test, scores_trtr, roc_auc_score, n_boot, seed)
    ci_pr_tstr = _bootstrap_ci(
        y_test,
        scores_tstr,
        partial(average_precision_score, pos_label=pos_label),
        n_boot,
        seed,
    )
    ci_pr_trtr = _bootstrap_ci(
        y_test,
        scores_trtr,
        partial(average_precision_score, pos_label=pos_label),
        n_boot,
        seed,
    )

    diagnostics = _data_diagnostics(X_train, X_syn)

    return {
        "tstr": {
            "auroc": float(auc_tstr),
            "auprc": float(pr_tstr),
            "auroc_ci": ci_auc_tstr,
            "auprc_ci": ci_pr_tstr,
        },
        "trtr": {
            "auroc": float(auc_trtr),
            "auprc": float(pr_trtr),
            "auroc_ci": ci_auc_trtr,
            "auprc_ci": ci_pr_trtr,
        },
        "delta": {
            "auroc": float(auc_tstr - auc_trtr),
            "auprc": float(pr_tstr - pr_trtr),
        },
        "diagnostics": diagnostics,
    }


__all__ = ["tstr_auc", "tstr_vs_trtr"]
