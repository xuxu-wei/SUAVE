"""Tests for evaluation metrics utilities."""

import numpy as np
import pytest

from suave.evaluate import simple_membership_inference


def test_simple_membership_inference_identical_scores_majority_accuracy() -> None:
    """When confidences tie, the best attack should match the majority class."""

    member_scores = np.full(2, 0.5)
    non_member_scores = np.full(8, 0.5)

    metrics = simple_membership_inference(member_scores, non_member_scores)

    majority_ratio = non_member_scores.size / (
        member_scores.size + non_member_scores.size
    )
    assert metrics["attack_best_accuracy"] == pytest.approx(majority_ratio)
    assert metrics["attack_majority_class_accuracy"] == pytest.approx(majority_ratio)
    assert metrics["attack_best_threshold"] > float(non_member_scores.max())
