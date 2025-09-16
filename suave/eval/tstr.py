"""Train-on-Synthetic-Test-on-Real (TSTR) evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

try:  # optional dependency
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional
    XGBClassifier = None


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

    scores = _predict_scores(clf, Xte)
    classes = _get_classes_from_estimator(clf, y_train, y_test)
    _validate_multiclass_scores(scores, classes)

    if len(classes) > 2:
        auc = roc_auc_score(
            y_test,
            scores,
            multi_class="ovr",
            average="macro",
            labels=classes,
        )
    else:
        auc = roc_auc_score(y_test, scores)

    return float(auc)


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


def _predict_scores(clf, X: np.ndarray) -> np.ndarray:
    """Return probability estimates or decision scores for ``clf`` on ``X``."""

    if hasattr(clf, "predict_proba"):
        proba = np.asarray(clf.predict_proba(X))
        if proba.ndim == 1:
            return proba
        if proba.shape[1] == 1:
            return proba[:, 0]
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba

    if not hasattr(clf, "decision_function"):
        raise ValueError("Estimator must implement predict_proba or decision_function")

    scores = np.asarray(clf.decision_function(X))
    if scores.ndim == 1:
        return scores
    if scores.ndim == 2 and scores.shape[1] == 1:
        return scores[:, 0]
    if scores.ndim == 2:
        stabilised = scores - scores.max(axis=1, keepdims=True)
        exp_s = np.exp(stabilised)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    raise ValueError(f"Unsupported decision_function output shape {scores.shape}")


def _collect_unique_labels(*arrays) -> np.ndarray:
    """Return sorted unique labels observed across the provided arrays."""

    flattened: list[np.ndarray] = []
    for arr in arrays:
        if arr is None:
            continue
        arr_np = np.asarray(arr)
        if arr_np.size == 0:
            continue
        flattened.append(arr_np.reshape(-1))
    if not flattened:
        raise ValueError("Cannot infer class labels from empty inputs.")
    return np.unique(np.concatenate(flattened))


def _get_classes_from_estimator(clf, *ys) -> np.ndarray:
    """Infer class labels from an estimator and fallback targets."""

    observed = _collect_unique_labels(*ys)
    if hasattr(clf, "classes_"):
        classes = np.asarray(clf.classes_)
        missing = np.setdiff1d(observed, classes)
        if missing.size:
            raise ValueError(
                "Estimator is missing classes present in evaluation data: "
                f"{missing.tolist()}"
            )
        return classes
    return observed


def _validate_multiclass_scores(scores: np.ndarray, classes: np.ndarray) -> None:
    """Validate that multi-class metrics receive properly shaped scores."""

    n_classes = len(classes)
    if n_classes <= 2:
        return
    if scores.ndim != 2 or scores.shape[1] != n_classes:
        raise ValueError(
            "Multi-class evaluation requires probability estimates for each class; "
            f"expected shape (n_samples, {n_classes}), got {scores.shape}."
        )


def _bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric,
    n_boot: int = 100,
    seed: int | None = None,
    metric_kwargs: dict[str, object] | None = None,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats_list = []
    metric_kwargs = metric_kwargs or {}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats_list.append(metric(y_true[idx], scores[idx], **metric_kwargs))
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

    scores_tstr = _predict_scores(est_tstr, X_test)
    scores_trtr = _predict_scores(est_trtr, X_test)

    classes_tstr = _get_classes_from_estimator(est_tstr, y_train, y_test)
    classes_trtr = _get_classes_from_estimator(est_trtr, y_train, y_test)
    _validate_multiclass_scores(scores_tstr, classes_tstr)
    _validate_multiclass_scores(scores_trtr, classes_trtr)

    if len(classes_tstr) > 2:
        auc_tstr = roc_auc_score(
            y_test,
            scores_tstr,
            multi_class="ovr",
            average="macro",
            labels=classes_tstr,
        )
        auc_trtr = roc_auc_score(
            y_test,
            scores_trtr,
            multi_class="ovr",
            average="macro",
            labels=classes_trtr,
        )
        y_test_bin_tstr = label_binarize(y_test, classes=classes_tstr)
        y_test_bin_trtr = label_binarize(y_test, classes=classes_trtr)
        pr_tstr = average_precision_score(
            y_test_bin_tstr, scores_tstr, average="macro"
        )
        pr_trtr = average_precision_score(
            y_test_bin_trtr, scores_trtr, average="macro"
        )
        ci_auc_tstr = _bootstrap_ci(
            y_test,
            scores_tstr,
            roc_auc_score,
            n_boot,
            seed,
            {"multi_class": "ovr", "average": "macro", "labels": classes_tstr},
        )
        ci_auc_trtr = _bootstrap_ci(
            y_test,
            scores_trtr,
            roc_auc_score,
            n_boot,
            seed,
            {"multi_class": "ovr", "average": "macro", "labels": classes_trtr},
        )
        ci_pr_tstr = _bootstrap_ci(
            y_test_bin_tstr,
            scores_tstr,
            average_precision_score,
            n_boot,
            seed,
            {"average": "macro"},
        )
        ci_pr_trtr = _bootstrap_ci(
            y_test_bin_trtr,
            scores_trtr,
            average_precision_score,
            n_boot,
            seed,
            {"average": "macro"},
        )
    else:
        auc_tstr = roc_auc_score(y_test, scores_tstr)
        auc_trtr = roc_auc_score(y_test, scores_trtr)
        pr_tstr = average_precision_score(y_test, scores_tstr)
        pr_trtr = average_precision_score(y_test, scores_trtr)
        ci_auc_tstr = _bootstrap_ci(y_test, scores_tstr, roc_auc_score, n_boot, seed)
        ci_auc_trtr = _bootstrap_ci(y_test, scores_trtr, roc_auc_score, n_boot, seed)
        ci_pr_tstr = _bootstrap_ci(y_test, scores_tstr, average_precision_score, n_boot, seed)
        ci_pr_trtr = _bootstrap_ci(y_test, scores_trtr, average_precision_score, n_boot, seed)

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
