"""Unit tests for the lightweight evaluation metrics helper."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import label_binarize

from suave.eval.metrics import classification_metrics


def _binary_reference_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "auroc": roc_auc_score(y_true, proba),
        "auprc": average_precision_score(y_true, proba),
        "brier": np.mean((y_true - proba) ** 2),
        "nll": log_loss(y_true, proba),
    }


def test_classification_metrics_binary_vector() -> None:
    """Binary inputs with a probability vector should be supported."""

    y_true = np.array([0, 1, 1, 0, 1, 0])
    proba = np.array([0.2, 0.8, 0.7, 0.3, 0.9, 0.1])

    expected = _binary_reference_metrics(y_true, proba)
    result = classification_metrics(y_true, proba)

    for key, value in expected.items():
        assert result[key] == pytest.approx(value)


def test_classification_metrics_binary_matrix() -> None:
    """Binary probability matrices should reduce to the positive column."""

    y_true = np.array([0, 1, 1, 0, 1, 0])
    proba = np.array([0.2, 0.8, 0.7, 0.3, 0.9, 0.1])
    proba_matrix = np.column_stack([1 - proba, proba])

    expected = _binary_reference_metrics(y_true, proba)
    result = classification_metrics(y_true, proba_matrix)

    for key, value in expected.items():
        assert result[key] == pytest.approx(value)


def test_classification_metrics_multiclass() -> None:
    """Multi-class metrics use macro-averaged one-vs-rest reductions."""

    y_true = np.array([0, 1, 2, 1, 0, 2, 1, 0, 2, 0])
    proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
            [0.2, 0.2, 0.6],
            [0.1, 0.7, 0.2],
            [0.6, 0.3, 0.1],
            [0.2, 0.2, 0.6],
            [0.2, 0.6, 0.2],
            [0.8, 0.1, 0.1],
            [0.3, 0.3, 0.4],
            [0.6, 0.2, 0.2],
        ]
    )

    classes = np.sort(np.unique(y_true))
    y_bin = label_binarize(y_true, classes=classes)

    expected = {
        "auroc": roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
        "auprc": average_precision_score(y_bin, proba, average="macro"),
        "brier": np.mean(np.sum((proba - y_bin) ** 2, axis=1)),
        "nll": log_loss(y_true, proba, labels=classes),
    }
    result = classification_metrics(y_true, proba)

    for key, value in expected.items():
        assert result[key] == pytest.approx(value)


def test_classification_metrics_multiclass_invalid_probability_shape() -> None:
    """Multi-class labels require a probability matrix with matching columns."""

    y_true = np.array([0, 1, 2, 1])

    with pytest.raises(ValueError, match="Multi-class classification requires"):
        classification_metrics(y_true, np.array([0.2, 0.5, 0.3, 0.4]))

    with pytest.raises(ValueError, match="must have one column per unique class"):
        classification_metrics(y_true, np.array([[0.6, 0.4], [0.2, 0.8], [0.3, 0.7], [0.5, 0.5]]))
