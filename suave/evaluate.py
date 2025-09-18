"""Evaluation helpers for SUAVE."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute the area under the ROC curve for binary targets."""

    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    n_pos = int(np.sum(y_true))
    n_total = y_true.size
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]

    ranks_sorted = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < n_total:
        end = start
        while end + 1 < n_total and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * (start + end + 2)  # 1-based inclusive ranks
        ranks_sorted[start : end + 1] = average_rank
        start = end + 1

    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted

    positive_rank_sum = float(np.sum(ranks[y_true == 1]))
    auc = (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _binary_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute the area under the precision-recall curve for binary targets."""

    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    n_pos = int(np.sum(y_true))
    if n_pos == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_true = y_true[order]

    true_positives = np.cumsum(sorted_true)
    false_positives = np.cumsum(1 - sorted_true)
    denominators = true_positives + false_positives

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.divide(
            true_positives,
            denominators,
            out=np.zeros_like(true_positives, dtype=np.float64),
            where=denominators != 0,
        )

    recall = true_positives / n_pos
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))

    area = float(np.trapezoid(precision, recall))
    return area


def _expected_calibration_error(
    probabilities: np.ndarray, targets: np.ndarray, num_bins: int
) -> float:
    """Compute the Expected Calibration Error for multi-class predictions."""

    if num_bins < 1:
        raise ValueError("num_bins must be at least 1")

    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correctness = (predictions == targets).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[1:-1], right=False)

    total = confidences.size
    ece = 0.0
    for bin_index in range(num_bins):
        mask = bin_indices == bin_index
        if not np.any(mask):
            continue
        weight = mask.sum() / total
        bin_accuracy = float(np.mean(correctness[mask]))
        bin_confidence = float(np.mean(confidences[mask]))
        ece += weight * abs(bin_accuracy - bin_confidence)
    return float(ece)


def evaluate_classification(
    probabilities: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    num_bins: int = 15,
) -> Dict[str, float]:
    """Compute standard classification metrics.

    Parameters
    ----------
    probabilities:
        Array with shape ``(n_samples, n_classes)`` containing class probabilities.
        A one-dimensional array is interpreted as the positive-class probability
        for a binary problem.
    targets:
        Integer encoded ground-truth labels with shape ``(n_samples,)``.
    mask:
        Optional boolean mask of shape ``(n_samples,)`` specifying which samples
        to include in the metric calculations.
    num_bins:
        Number of bins to use for the Expected Calibration Error computation.

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy, AUROC, AUPRC, Brier score, and ECE.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)

    if probabilities.ndim == 1:
        probabilities = np.stack([1.0 - probabilities, probabilities], axis=1)
    elif probabilities.ndim != 2:
        raise ValueError("probabilities must be a one or two-dimensional array")

    if probabilities.shape[0] != targets.shape[0]:
        raise ValueError("probabilities and targets must share the first dimension")

    if mask is None:
        mask_array = np.ones(probabilities.shape[0], dtype=bool)
    else:
        mask_array = np.asarray(mask).astype(bool)
        if mask_array.shape != (probabilities.shape[0],):
            raise ValueError("mask must be one-dimensional with length n_samples")

    if not np.any(mask_array):
        raise ValueError("mask must include at least one sample")

    probabilities = np.clip(probabilities, 0.0, 1.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    safe_probs = np.where(row_sums > 0, probabilities / row_sums, 1.0 / probabilities.shape[1])

    masked_probs = safe_probs[mask_array]
    masked_targets = targets[mask_array]
    if masked_targets.ndim != 1:
        raise ValueError("targets must be a one-dimensional array")

    n_classes = masked_probs.shape[1]

    predicted_labels = np.argmax(masked_probs, axis=1)
    accuracy = float(np.mean(predicted_labels == masked_targets))

    if np.any(masked_targets < 0) or np.any(masked_targets >= n_classes):
        raise ValueError("targets must be in the range [0, n_classes)")

    one_hot = np.eye(n_classes)[masked_targets]
    brier = float(np.mean(np.sum((masked_probs - one_hot) ** 2, axis=1)))

    auroc_scores = []
    auprc_scores = []
    for class_index in range(n_classes):
        binary_targets = (masked_targets == class_index).astype(np.int32)
        if binary_targets.sum() == 0 or binary_targets.sum() == binary_targets.size:
            continue
        class_scores = masked_probs[:, class_index]
        auroc_scores.append(_binary_auc(binary_targets, class_scores))
        auprc_scores.append(_binary_average_precision(binary_targets, class_scores))

    auroc = float(np.mean(auroc_scores)) if auroc_scores else float("nan")
    auprc = float(np.mean(auprc_scores)) if auprc_scores else float("nan")

    ece = _expected_calibration_error(masked_probs, masked_targets, num_bins=num_bins)

    return {
        "accuracy": accuracy,
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "ece": ece,
    }
