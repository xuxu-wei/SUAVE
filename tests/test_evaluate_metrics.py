"""Unit tests for evaluation helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip(
    "sklearn", reason="scikit-learn is required for evaluation metrics tests"
)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

from suave.evaluate import (
    classifier_two_sample_test,
    compute_auprc,
    compute_auroc,
    compute_brier,
    compute_ece,
    evaluate_classification,
    evaluate_trtr,
    evaluate_tstr,
    simple_membership_inference,
)


def test_evaluate_classification_binary_perfect_predictions() -> None:
    probabilities = np.array([[0.01, 0.99], [0.98, 0.02]])
    targets = np.array([1, 0])

    metrics = evaluate_classification(probabilities, targets)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(0.00025)

    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    bin_edges = np.linspace(0.0, 1.0, 11)
    bin_indices = np.minimum(np.digitize(confidences, bin_edges[1:], right=True), 9)
    expected_ece = 0.0
    for bin_index in range(10):
        mask = bin_indices == bin_index
        if not np.any(mask):
            continue
        expected_ece += abs(
            np.mean(confidences[mask]) - np.mean(predictions[mask] == targets[mask])
        ) * (np.sum(mask) / len(confidences))

    assert metrics["ece"] == pytest.approx(expected_ece)


def test_metrics_handle_single_class() -> None:
    probabilities = np.array([[0.8, 0.2], [0.7, 0.3]])
    targets = np.array([0, 0])

    metrics = evaluate_classification(probabilities, targets)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert math.isnan(metrics["auroc"])
    assert math.isnan(metrics["auprc"])
    assert metrics["brier"] == pytest.approx((0.2**2 + 0.3**2) / 2)
    assert metrics["ece"] >= 0.0


def test_metric_helpers_match_sklearn_multi_class() -> None:
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
            [0.2, 0.2, 0.6],
        ]
    )
    targets = np.array([0, 1, 2, 2])

    auroc = compute_auroc(probabilities, targets)
    auprc = compute_auprc(probabilities, targets)
    brier = compute_brier(probabilities, targets)

    expected_auroc = roc_auc_score(
        targets, probabilities, multi_class="ovr", average="macro"
    )
    expected_auprc = np.mean(
        [
            average_precision_score((targets == idx).astype(int), probabilities[:, idx])
            for idx in range(probabilities.shape[1])
        ]
    )
    one_hot = np.eye(probabilities.shape[1])[targets]
    expected_brier = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))

    assert auroc == pytest.approx(expected_auroc)
    assert auprc == pytest.approx(expected_auprc)
    assert brier == pytest.approx(expected_brier)


def test_compute_ece_known_value() -> None:
    probabilities = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.05, 0.95]])
    targets = np.array([0, 0, 1, 1])

    ece = compute_ece(probabilities, targets, n_bins=2)

    # Manually compute ECE for two bins [0, 0.5), [0.5, 1.0].
    confidences = np.max(probabilities, axis=1)
    bin_indices = (confidences >= 0.5).astype(int)

    expected = 0.0
    for bin_index in range(2):
        mask = bin_indices == bin_index
        if not np.any(mask):
            continue
        bin_confidence = np.mean(confidences[mask])
        bin_accuracy = np.mean(np.argmax(probabilities[mask], axis=1) == targets[mask])
        expected += abs(bin_confidence - bin_accuracy) * (
            np.sum(mask) / len(confidences)
        )
    assert ece == pytest.approx(expected)


def test_evaluate_accepts_one_dimensional_probabilities() -> None:
    probabilities = np.array([0.9, 0.2, 0.7, 0.1])
    targets = np.array([1, 0, 1, 0])

    metrics = evaluate_classification(probabilities, targets)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(np.mean((probabilities - targets) ** 2))


def _logistic_regression_factory() -> LogisticRegression:
    return LogisticRegression(max_iter=500, solver="lbfgs")


def _make_logistic_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, solver="lbfgs")),
        ]
    )


def test_tstr_and_trtr_workflow() -> None:
    X_syn = np.array([[0.0], [0.1], [1.0], [1.1]])
    y_syn = np.array([0, 0, 1, 1])
    X_real_train = np.array([[0.05], [0.15], [0.95], [1.05]])
    y_real_train = np.array([0, 0, 1, 1])
    X_real_test = np.array([[0.02], [0.12], [0.98], [1.08]])
    y_real_test = np.array([0, 0, 1, 1])

    tstr_metrics = evaluate_tstr(
        (X_syn, y_syn), (X_real_test, y_real_test), _logistic_regression_factory
    )
    trtr_metrics = evaluate_trtr(
        (X_real_train, y_real_train),
        (X_real_test, y_real_test),
        _logistic_regression_factory,
    )

    assert tstr_metrics["accuracy"] >= 0.75
    assert trtr_metrics["accuracy"] >= 0.75


def test_simple_membership_inference_baseline() -> None:
    train_probabilities = np.array([[0.05, 0.95], [0.15, 0.85], [0.02, 0.98]])
    train_targets = np.array([1, 1, 1])
    test_probabilities = np.array([[0.7, 0.3], [0.8, 0.2], [0.65, 0.35]])
    test_targets = np.array([0, 0, 0])

    results = simple_membership_inference(
        train_probabilities, train_targets, test_probabilities, test_targets
    )

    assert results["attack_auc"] == pytest.approx(1.0)
    assert results["attack_best_accuracy"] == pytest.approx(1.0)
    assert results["attack_best_threshold"] == pytest.approx(0.85)
    assert results["attack_majority_class_accuracy"] == pytest.approx(0.5)


def test_simple_membership_inference_identical_scores_majority_accuracy() -> None:
    train_probabilities = np.full((2, 2), 0.5)
    train_targets = np.ones(2, dtype=int)
    test_probabilities = np.full((8, 2), 0.5)
    test_targets = np.zeros(8, dtype=int)

    metrics = simple_membership_inference(
        train_probabilities, train_targets, test_probabilities, test_targets
    )

    majority_ratio = test_targets.size / (train_targets.size + test_targets.size)
    assert metrics["attack_best_accuracy"] == pytest.approx(majority_ratio)
    assert metrics["attack_majority_class_accuracy"] == pytest.approx(majority_ratio)
    assert metrics["attack_best_threshold"] > float(test_probabilities[:, 0].max())


def test_classifier_two_sample_test_produces_auc_and_intervals() -> None:
    pytest.importorskip("xgboost", reason="xgboost is required for C2ST evaluation")
    from xgboost import XGBClassifier

    rng = np.random.default_rng(0)
    real = rng.normal(size=(120, 4))
    synthetic = rng.normal(loc=0.2, size=(120, 4))

    results = classifier_two_sample_test(
        real,
        synthetic,
        model_factories={
            "xgboost": lambda: XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=0,
                tree_method="hist",
                n_jobs=-1,
                use_label_encoder=False,
            ),
            "logistic": _make_logistic_pipeline,
        },
        random_state=0,
        n_splits=4,
        n_bootstrap=50,
    )

    for key in (
        "xgboost_auc",
        "xgboost_auc_ci_low",
        "xgboost_auc_ci_high",
        "logistic_auc",
        "logistic_auc_ci_low",
        "logistic_auc_ci_high",
    ):
        assert key in results

    assert 0.0 <= results["xgboost_auc"] <= 1.0
    assert 0.0 <= results["logistic_auc"] <= 1.0
    assert results["xgboost_auc_ci_low"] <= results["xgboost_auc_ci_high"]
    assert results["logistic_auc_ci_low"] <= results["logistic_auc_ci_high"]
    assert results["xgboost_bootstrap_samples"] <= 50.0
    assert results["logistic_bootstrap_samples"] <= 50.0


def test_classifier_two_sample_test_filters_non_finite_rows() -> None:
    pytest.importorskip("xgboost", reason="xgboost is required for C2ST evaluation")
    from xgboost import XGBClassifier

    real = np.ones((10, 3))
    real[0, 0] = np.nan
    synthetic = np.ones((12, 3))
    synthetic[0, 1] = np.inf
    synthetic[1, 2] = np.nan

    results = classifier_two_sample_test(
        real,
        synthetic,
        model_factories={
            "xgboost": lambda: XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                n_estimators=50,
                max_depth=3,
                random_state=1,
                tree_method="hist",
                n_jobs=-1,
                use_label_encoder=False,
            ),
            "logistic": _make_logistic_pipeline,
        },
        random_state=1,
        n_splits=5,
        n_bootstrap=10,
    )

    assert results["n_real_samples"] == 9
    assert results["n_synthetic_samples"] == 10
    assert results["cv_splits"] >= 2
    assert results["cv_splits"] <= min(
        results["n_real_samples"], results["n_synthetic_samples"]
    )
