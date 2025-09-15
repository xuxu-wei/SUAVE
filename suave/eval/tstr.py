"""Train-on-Synthetic-Test-on-Real (TSTR) evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
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

    gen = model.generate(len(X_train), conditional={"y": y_train})
    gen_X = gen.drop(columns=["y"]).to_numpy()
    y_gen = gen["y"].to_numpy()

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


def _get_estimator(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=1000)
    if name == "svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, max_iter=200)),
            ]
        )
    if name == "knn":
        return Pipeline([( "scaler", StandardScaler()), ("clf", KNeighborsClassifier())])
    if name == "rf":
        return RandomForestClassifier(n_estimators=100)
    if name == "xgb":
        try:  # pragma: no cover - optional dependency
            from xgboost import XGBClassifier

            return XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        except Exception as exc:  # pragma: no cover
            raise ImportError("xgboost is required for estimator='xgb'") from exc
    raise ValueError(f"Unknown estimator: {name}")


def _bootstrap_ci(metric_fn, y_true, y_score, n_boot: int, rng: np.random.Generator):
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(y_true[idx], y_score[idx]))
    lower, upper = np.percentile(stats, [2.5, 97.5])
    return lower, upper


def tstr_vs_trtr(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    estimator: str = "logreg",
    n_boot: int = 100,
    seed: int | None = None,
):
    """Compare TSTR against TRTR using a downstream estimator.

    Returns a dictionary with AUROC/AUPRC scores, 95% bootstrap confidence
    intervals and their differences (TSTR - TRTR).
    """

    rng = np.random.default_rng(seed)
    est = _get_estimator(estimator)

    # TRTR baseline
    est.fit(X_train, y_train)
    if hasattr(est, "predict_proba"):
        proba_real = est.predict_proba(X_test)
        if proba_real.ndim == 2 and proba_real.shape[1] > 1:
            score_real = proba_real[:, 1]
        else:
            score_real = proba_real.squeeze()
    else:
        score_real = est.decision_function(X_test)
    auroc_real = roc_auc_score(y_test, score_real)
    auprc_real = average_precision_score(y_test, score_real)
    ci_auroc_real = _bootstrap_ci(roc_auc_score, y_test, score_real, n_boot, rng)
    ci_auprc_real = _bootstrap_ci(average_precision_score, y_test, score_real, n_boot, rng)

    # TSTR
    gen = model.generate(len(X_train), conditional={"y": y_train})
    X_syn = gen.drop(columns=["y"]).to_numpy()
    y_syn = gen["y"].to_numpy()
    est_syn = _get_estimator(estimator)
    est_syn.fit(X_syn, y_syn)
    if hasattr(est_syn, "predict_proba"):
        proba_syn = est_syn.predict_proba(X_test)
        if proba_syn.ndim == 2 and proba_syn.shape[1] > 1:
            score_syn = proba_syn[:, 1]
        else:
            score_syn = proba_syn.squeeze()
    else:
        score_syn = est_syn.decision_function(X_test)
    auroc_syn = roc_auc_score(y_test, score_syn)
    auprc_syn = average_precision_score(y_test, score_syn)
    ci_auroc_syn = _bootstrap_ci(roc_auc_score, y_test, score_syn, n_boot, rng)
    ci_auprc_syn = _bootstrap_ci(average_precision_score, y_test, score_syn, n_boot, rng)

    return {
        "trtr": {
            "auroc": auroc_real,
            "auroc_ci": ci_auroc_real,
            "auprc": auprc_real,
            "auprc_ci": ci_auprc_real,
        },
        "tstr": {
            "auroc": auroc_syn,
            "auroc_ci": ci_auroc_syn,
            "auprc": auprc_syn,
            "auprc_ci": ci_auprc_syn,
        },
        "delta": {
            "auroc": auroc_syn - auroc_real,
            "auprc": auprc_syn - auprc_real,
        },
    }


def check_missingness(df: pd.DataFrame) -> pd.Series:
    """Return fraction of missing values per column."""

    return df.isna().mean()


def distribution_comparison(real: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    """Compare mean and std between real and synthetic data."""

    stats = []
    for col in real.columns:
        stats.append(
            {
                "column": col,
                "real_mean": float(real[col].mean()),
                "synth_mean": float(synth[col].mean()),
                "real_std": float(real[col].std()),
                "synth_std": float(synth[col].std()),
            }
        )
    return pd.DataFrame(stats)
