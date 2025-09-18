"""Unit tests for evaluation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from suave.evaluate import evaluate_classification


def test_evaluate_binary_metrics() -> None:
    """Binary helper should compute standard metrics with 1D inputs."""

    probabilities = np.array([0.9, 0.2, 0.6, 0.3])
    targets = np.array([1, 0, 1, 0])

    metrics = evaluate_classification(probabilities, targets, num_bins=4)

    assert set(metrics) == {"accuracy", "auroc", "auprc", "brier", "ece"}
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(0.15)
    assert metrics["ece"] == pytest.approx(0.25)


def test_evaluate_multiclass_with_mask() -> None:
    """Multi-class metrics respect masks and match expected values."""

    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.2, 0.7],
            [0.6, 0.2, 0.2],
            [0.1, 0.6, 0.3],
        ]
    )
    targets = np.array([0, 1, 2, 0, 1])
    mask = np.array([True, True, True, False, True])

    metrics = evaluate_classification(probabilities, targets, mask=mask, num_bins=3)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(0.23)
    assert metrics["ece"] == pytest.approx(0.375)
