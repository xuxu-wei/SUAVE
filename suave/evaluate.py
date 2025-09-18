"""Evaluation helpers for SUAVE."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _prepare_inputs(
    probabilities: np.ndarray, targets: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and format probability/target inputs.

    The helper normalises probability arrays into a two-dimensional matrix and
    ensures the target labels are integer encoded. It is primarily used by the
    metric helpers defined in this module.

    Args:
        probabilities: Raw probability estimates. For binary tasks the array may
            be one-dimensional containing the positive class probability. For
            multi-class tasks the array must be shaped ``(n_samples, n_classes)``
            with columns ordered according to the encoded class index.
        targets: Integer encoded class labels aligned with ``probabilities``.

    Returns:
        A tuple containing the probability matrix with shape
        ``(n_samples, n_classes)`` and the validated target vector.

    Raises:
        ValueError: If shapes do not align or labels fall outside the probability
            matrix columns.

    Example:
        >>> probs, labels = _prepare_inputs(np.array([0.2, 0.8]), np.array([0, 1]))
        >>> probs.shape
        (2, 2)
    """

    prob_array = np.asarray(probabilities, dtype=float)
    target_array = np.asarray(targets)

    if prob_array.ndim == 1:
        prob_array = np.stack([1.0 - prob_array, prob_array], axis=1)
    elif prob_array.ndim == 2 and prob_array.shape[1] == 1:
        prob_array = np.hstack([1.0 - prob_array, prob_array])
    elif prob_array.ndim != 2:
        raise ValueError("probabilities must be one- or two-dimensional")

    if not np.all(np.isfinite(prob_array)):
        raise ValueError("probabilities must contain only finite values")

    if target_array.ndim != 1:
        raise ValueError("targets must be a one-dimensional array")

    if not np.issubdtype(target_array.dtype, np.integer):
        raise ValueError("targets must be integer encoded starting at zero")

    if len(prob_array) != len(target_array):
        raise ValueError("probabilities and targets must share the first dimension")

    if prob_array.shape[0] == 0:
        return prob_array, target_array.astype(int, copy=False)

    if np.min(target_array) < 0:
        raise ValueError("targets must be non-negative integers")

    if np.max(target_array) >= prob_array.shape[1]:
        raise ValueError("targets reference an invalid class index")

    return prob_array, target_array.astype(int, copy=False)


def compute_auroc(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Compute the area under the ROC curve (AUROC).

    Args:
        probabilities: Probability estimates. One-dimensional arrays are
            interpreted as the positive class probability for binary problems.
        targets: Integer encoded target labels aligned with ``probabilities``.

    Returns:
        The AUROC value. ``numpy.nan`` is returned when the metric is undefined,
        such as when only a single class is present or scikit-learn raises a
        ``ValueError`` for degenerate inputs.

    Example:
        >>> probs = np.array([[0.1, 0.9], [0.8, 0.2]])
        >>> compute_auroc(probs, np.array([1, 0]))
        1.0
    """

    prob_array, target_array = _prepare_inputs(probabilities, targets)
    if prob_array.shape[0] == 0 or np.unique(target_array).size < 2:
        return float("nan")

    n_classes = prob_array.shape[1]
    try:
        if n_classes == 2:
            return float(roc_auc_score(target_array, prob_array[:, 1]))
        return float(
            roc_auc_score(
                target_array,
                prob_array,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return float("nan")


def compute_auprc(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Compute the area under the precision-recall curve (AUPRC).

    Args:
        probabilities: Probability estimates for each class.
        targets: Integer encoded target labels aligned with ``probabilities``.

    Returns:
        The macro-averaged AUPRC across classes. ``numpy.nan`` is returned when
        the score is undefined for the provided data.

    Example:
        >>> probs = np.array([[0.05, 0.95], [0.9, 0.1]])
        >>> compute_auprc(probs, np.array([1, 0]))
        1.0
    """

    prob_array, target_array = _prepare_inputs(probabilities, targets)
    if prob_array.shape[0] == 0:
        return float("nan")

    n_classes = prob_array.shape[1]
    if n_classes == 2:
        if np.unique(target_array).size < 2:
            return float("nan")
        try:
            return float(average_precision_score(target_array, prob_array[:, 1]))
        except ValueError:
            return float("nan")

    scores = []
    for class_index in range(n_classes):
        binary_target = (target_array == class_index).astype(int)
        if np.unique(binary_target).size < 2:
            continue
        try:
            scores.append(
                float(
                    average_precision_score(binary_target, prob_array[:, class_index])
                )
            )
        except ValueError:
            continue

    return float(np.mean(scores)) if scores else float("nan")


def compute_brier(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Compute the Brier score for binary and multi-class classifiers.

    Args:
        probabilities: Probability estimates for each class.
        targets: Integer encoded target labels aligned with ``probabilities``.

    Returns:
        The Brier score. Lower values indicate better calibrated probabilities.
        ``numpy.nan`` is returned for empty inputs.

    Example:
        >>> probs = np.array([[0.8, 0.2], [0.3, 0.7]])
        >>> round(compute_brier(probs, np.array([0, 1])), 3)
        0.065
    """

    prob_array, target_array = _prepare_inputs(probabilities, targets)
    if prob_array.shape[0] == 0:
        return float("nan")

    n_classes = prob_array.shape[1]
    if n_classes == 2:
        return float(brier_score_loss(target_array, prob_array[:, 1]))

    one_hot = np.eye(n_classes)[target_array]
    return float(np.mean(np.sum((prob_array - one_hot) ** 2, axis=1)))


def compute_ece(
    probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 10
) -> float:
    """Compute the expected calibration error (ECE).

    Args:
        probabilities: Probability estimates for each class.
        targets: Integer encoded target labels aligned with ``probabilities``.
        n_bins: Number of equally spaced bins used to evaluate calibration.

    Returns:
        The ECE value. ``numpy.nan`` is returned for empty inputs.

    Example:
        >>> probs = np.array([[0.2, 0.8], [0.7, 0.3]])
        >>> round(compute_ece(probs, np.array([1, 0]), n_bins=5), 2)
        0.25
    """

    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")

    prob_array, target_array = _prepare_inputs(probabilities, targets)
    if prob_array.shape[0] == 0:
        return float("nan")

    confidences = np.max(prob_array, axis=1)
    predictions = np.argmax(prob_array, axis=1)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.minimum(
        np.digitize(confidences, bin_edges[1:], right=True), n_bins - 1
    )

    ece = 0.0
    total = len(confidences)
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        if not np.any(mask):
            continue
        bin_confidence = float(np.mean(confidences[mask]))
        bin_accuracy = float(np.mean(predictions[mask] == target_array[mask]))
        ece += abs(bin_confidence - bin_accuracy) * (np.sum(mask) / total)

    return float(ece)


def evaluate_classification(
    probabilities: np.ndarray, targets: np.ndarray
) -> Dict[str, float]:
    """Evaluate common classification metrics for SUAVE models.

    Args:
        probabilities: Probability estimates shaped ``(n_samples, n_classes)`` or
            a one-dimensional array containing positive class probabilities for
            binary classification.
        targets: Integer encoded ground-truth labels aligned with
            ``probabilities``. Labels must be zero-indexed to match column
            positions.

    Returns:
        A dictionary containing accuracy, AUROC, AUPRC, Brier score, and ECE
        values. Metrics that are undefined for the provided data return
        ``numpy.nan``.

    Example:
        >>> probs = np.array([[0.1, 0.9], [0.8, 0.2]])
        >>> evaluate_classification(probs, np.array([1, 0]))
        {'accuracy': 1.0, 'auroc': 1.0, 'auprc': 1.0, 'brier': 0.024999999999999994, 'ece': 0.14999999999999997}
    """

    prob_array, target_array = _prepare_inputs(probabilities, targets)
    if prob_array.shape[0] == 0:
        nan_value = float("nan")
        return {
            "accuracy": nan_value,
            "auroc": nan_value,
            "auprc": nan_value,
            "brier": nan_value,
            "ece": nan_value,
        }

    predictions = np.argmax(prob_array, axis=1)
    accuracy = float(np.mean(predictions == target_array))

    return {
        "accuracy": accuracy,
        "auroc": compute_auroc(prob_array, target_array),
        "auprc": compute_auprc(prob_array, target_array),
        "brier": compute_brier(prob_array, target_array),
        "ece": compute_ece(prob_array, target_array),
    }


def evaluate_tstr(
    synthetic: Tuple[np.ndarray, np.ndarray],
    real: Tuple[np.ndarray, np.ndarray],
    model_factory: Callable[[], Any],
) -> Dict[str, float]:
    """Run a Train-on-Synthetic, Test-on-Real (TSTR) evaluation.

    Args:
        synthetic: Tuple ``(features, targets)`` used for training. The
            classifier produced by ``model_factory`` must support ``fit`` and
            ``predict_proba`` on these arrays.
        real: Tuple ``(features, targets)`` representing the held-out real data
            used for evaluation.
        model_factory: Callable returning an unfitted probabilistic classifier
            (e.g., ``lambda: LogisticRegression(max_iter=200)``).

    Returns:
        Classification metrics on the real evaluation split.

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> X_syn = np.array([[0.0], [1.0], [2.0], [3.0]])
        >>> y_syn = np.array([0, 0, 1, 1])
        >>> X_real = np.array([[0.2], [0.8], [2.2], [2.8]])
        >>> y_real = np.array([0, 0, 1, 1])
        >>> evaluate_tstr((X_syn, y_syn), (X_real, y_real),
        ...                lambda: LogisticRegression(max_iter=200))['accuracy']
        1.0
    """

    X_syn, y_syn = synthetic
    X_real, y_real = real

    model = model_factory()
    if not hasattr(model, "fit") or not hasattr(model, "predict_proba"):
        raise ValueError(
            "model_factory must return an object with fit and predict_proba"
        )

    model.fit(X_syn, y_syn)
    probabilities = model.predict_proba(X_real)
    return evaluate_classification(probabilities, y_real)


def evaluate_trtr(
    real_train: Tuple[np.ndarray, np.ndarray],
    real_test: Tuple[np.ndarray, np.ndarray],
    model_factory: Callable[[], Any],
) -> Dict[str, float]:
    """Run a Train-on-Real, Test-on-Real (TRTR) evaluation.

    Args:
        real_train: Tuple ``(features, targets)`` used for training.
        real_test: Tuple ``(features, targets)`` used for evaluation.
        model_factory: Callable returning an unfitted classifier supporting
            ``fit`` and ``predict_proba``.

    Returns:
        Classification metrics on the held-out real test split.

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
        >>> y_train = np.array([0, 0, 1, 1])
        >>> X_test = np.array([[0.1], [0.9], [2.1], [2.9]])
        >>> y_test = np.array([0, 0, 1, 1])
        >>> evaluate_trtr((X_train, y_train), (X_test, y_test),
        ...               lambda: LogisticRegression(max_iter=200))['accuracy']
        1.0
    """

    X_train, y_train = real_train
    X_test, y_test = real_test

    model = model_factory()
    if not hasattr(model, "fit") or not hasattr(model, "predict_proba"):
        raise ValueError(
            "model_factory must return an object with fit and predict_proba"
        )

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    return evaluate_classification(probabilities, y_test)


def simple_membership_inference(
    train_probabilities: np.ndarray,
    train_targets: np.ndarray,
    test_probabilities: np.ndarray,
    test_targets: np.ndarray,
) -> Dict[str, float]:
    """Run a simple membership-inference attack baseline.

    The attack scores each example by the predicted probability of its true
    class and evaluates a sweep over decision thresholds to separate members
    (training examples) from non-members (held-out examples). All thresholds
    induced by the observed scores are considered alongside extreme thresholds
    that force "all member" and "all non-member" predictions to ensure the
    majority-class baseline is explicitly evaluated.

    Args:
        train_probabilities: Probabilities predicted on the training data.
        train_targets: Integer encoded training labels.
        test_probabilities: Probabilities predicted on held-out data.
        test_targets: Integer encoded held-out labels.

    Returns:
        A dictionary containing ``attack_auc``, ``attack_best_threshold``,
        ``attack_best_accuracy``, and ``attack_majority_class_accuracy``. The
        scores are ``numpy.nan`` when the attack is undefined (e.g., only a
        single membership class is present).

    Example:
        >>> train_probs = np.array([[0.1, 0.9], [0.2, 0.8]])
        >>> train_targets = np.array([1, 1])
        >>> test_probs = np.array([[0.7, 0.3], [0.6, 0.4]])
        >>> test_targets = np.array([0, 0])
        >>> simple_membership_inference(train_probs, train_targets,
        ...                              test_probs, test_targets)['attack_auc']
        1.0
    """

    train_probs, train_labels = _prepare_inputs(train_probabilities, train_targets)
    test_probs, test_labels = _prepare_inputs(test_probabilities, test_targets)

    if train_probs.shape[1] != test_probs.shape[1]:
        raise ValueError(
            "train and test probabilities must have the same number of classes"
        )

    if train_probs.shape[0] == 0 or test_probs.shape[0] == 0:
        nan_value = float("nan")
        return {
            "attack_auc": nan_value,
            "attack_best_threshold": nan_value,
            "attack_best_accuracy": nan_value,
            "attack_majority_class_accuracy": nan_value,
        }

    def _true_class_confidence(
        prob_matrix: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        if prob_matrix.shape[1] == 2:
            confidences = np.where(labels == 1, prob_matrix[:, 1], prob_matrix[:, 0])
        else:
            confidences = prob_matrix[np.arange(len(labels)), labels]
        return confidences.astype(float, copy=False)

    member_scores = _true_class_confidence(train_probs, train_labels)
    non_member_scores = _true_class_confidence(test_probs, test_labels)

    scores = np.concatenate([member_scores, non_member_scores])
    membership_labels = np.concatenate(
        [
            np.ones_like(member_scores, dtype=int),
            np.zeros_like(non_member_scores, dtype=int),
        ]
    )

    total_samples = float(scores.size)
    member_ratio = float(member_scores.size) / total_samples
    non_member_ratio = float(non_member_scores.size) / total_samples
    majority_accuracy = float(max(member_ratio, non_member_ratio))

    if np.unique(membership_labels).size < 2:
        nan_value = float("nan")
        return {
            "attack_auc": nan_value,
            "attack_best_threshold": nan_value,
            "attack_best_accuracy": nan_value,
            "attack_majority_class_accuracy": majority_accuracy,
        }

    try:
        attack_auc = float(roc_auc_score(membership_labels, scores))
    except ValueError:
        attack_auc = float("nan")

    unique_thresholds = np.unique(scores)
    lower_extreme = np.nextafter(scores.min(), -np.inf)
    upper_extreme = np.nextafter(scores.max(), np.inf)
    candidate_thresholds = np.unique(
        np.concatenate((unique_thresholds, np.array([lower_extreme, upper_extreme])))
    )

    predictions = scores[None, :] >= candidate_thresholds[:, None]
    accuracies = (predictions == membership_labels).mean(axis=1)
    best_index = int(np.argmax(accuracies))

    return {
        "attack_auc": attack_auc,
        "attack_best_threshold": float(candidate_thresholds[best_index]),
        "attack_best_accuracy": float(accuracies[best_index]),
        "attack_majority_class_accuracy": majority_accuracy,
    }
