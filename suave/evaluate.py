"""Evaluation helpers for SUAVE."""

from __future__ import annotations

from typing import Dict

import numpy as np


def evaluate_classification(
    probabilities: np.ndarray, targets: np.ndarray
) -> Dict[str, float]:
    """Return dummy metrics for the minimal implementation."""

    if probabilities.shape[0] != len(targets):
        raise ValueError("probabilities and targets must share the first dimension")
    return {"accuracy": float(np.mean(targets == targets))}


def simple_membership_inference(
    member_confidences: np.ndarray, non_member_confidences: np.ndarray
) -> Dict[str, float]:
    """Evaluate a threshold-based membership inference baseline.

    The attack assigns the *member* label to samples with confidence scores that are
    greater than or equal to a threshold. All possible thresholds induced by the
    provided scores are evaluated, along with an additional pair of thresholds that
    force "all member" and "all non-member" predictions. The best accuracy achieved
    over this sweep is returned alongside the majority-class baseline accuracy.

    Args:
        member_confidences: Confidence scores observed for training samples.
        non_member_confidences: Confidence scores observed for non-member samples.

    Returns:
        A dictionary containing the best attack accuracy, the threshold that
        attains it, and the majority-class accuracy baseline.
    """

    member_confidences = np.asarray(member_confidences, dtype=float)
    non_member_confidences = np.asarray(non_member_confidences, dtype=float)

    if member_confidences.ndim != 1 or non_member_confidences.ndim != 1:
        raise ValueError("membership inference scores must be 1-D arrays")

    total_samples = int(member_confidences.size + non_member_confidences.size)
    if total_samples == 0:
        raise ValueError("at least one score is required to run the attack")

    arrays = []
    labels = []
    if member_confidences.size:
        arrays.append(member_confidences)
        labels.append(np.ones(member_confidences.size, dtype=int))
    if non_member_confidences.size:
        arrays.append(non_member_confidences)
        labels.append(np.zeros(non_member_confidences.size, dtype=int))

    scores = np.concatenate(arrays)
    membership_labels = np.concatenate(labels)

    # Majority-class baseline (a random guess biased by class imbalance).
    member_ratio = float(member_confidences.size) / float(total_samples)
    non_member_ratio = float(non_member_confidences.size) / float(total_samples)
    majority_accuracy = max(member_ratio, non_member_ratio)

    unique_thresholds = np.unique(scores)
    # Evaluate two additional thresholds to cover all-member and all-non-member
    # predictions explicitly.
    lower_extreme = np.nextafter(scores.min(), -np.inf)
    upper_extreme = np.nextafter(scores.max(), np.inf)
    candidate_thresholds = np.unique(
        np.concatenate((unique_thresholds, np.array([lower_extreme, upper_extreme])))
    )

    # Vectorised evaluation of all thresholds.
    predictions = scores[None, :] >= candidate_thresholds[:, None]
    accuracies = (predictions == membership_labels).mean(axis=1)

    best_index = int(np.argmax(accuracies))

    return {
        "attack_best_threshold": float(candidate_thresholds[best_index]),
        "attack_best_accuracy": float(accuracies[best_index]),
        "attack_majority_class_accuracy": float(majority_accuracy),
    }
