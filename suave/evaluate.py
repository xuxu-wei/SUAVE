"""Evaluation helpers for SUAVE's supervised branch."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mutual_info_score,
    roc_auc_score,
)


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


def kolmogorov_smirnov_statistic(real: np.ndarray, synthetic: np.ndarray) -> float:
    """Compute the Kolmogorov–Smirnov statistic for univariate samples.

    The Kolmogorov–Smirnov (KS) statistic measures the maximum distance between
    the empirical cumulative distribution functions of the real and synthetic
    samples. Values close to ``0.0`` indicate that the synthetic distribution is
    indistinguishable from the reference sample, whereas scores approaching
    ``1.0`` highlight large distributional shifts. A common heuristic is to flag
    KS statistics above ``0.1`` as a sign that the feature is not faithfully
    reproduced, although domain-specific tolerances should override this
    general threshold.

    Args:
        real: One-dimensional array of observations drawn from the reference
            dataset.
        synthetic: One-dimensional array sampled from the generative model.

    Returns:
        The KS statistic in ``[0, 1]``. ``numpy.nan`` is returned when either
        input is empty after removing non-finite values.

    Example:
        >>> import numpy as np
        >>> ks = kolmogorov_smirnov_statistic(
        ...     np.random.normal(size=256),
        ...     np.random.normal(size=256),
        ... )
        >>> ks < 0.1
        True
    """

    real = np.asarray(real, dtype=float)
    synthetic = np.asarray(synthetic, dtype=float)
    real = real[np.isfinite(real)]
    synthetic = synthetic[np.isfinite(synthetic)]
    if real.size == 0 or synthetic.size == 0:
        return float("nan")

    real_sorted = np.sort(real)
    synthetic_sorted = np.sort(synthetic)
    combined = np.concatenate([real_sorted, synthetic_sorted])
    cdf_real = np.searchsorted(real_sorted, combined, side="right") / real_sorted.size
    cdf_synth = (
        np.searchsorted(synthetic_sorted, combined, side="right")
        / synthetic_sorted.size
    )
    return float(np.max(np.abs(cdf_real - cdf_synth)))


def rbf_mmd(
    real: np.ndarray,
    synthetic: np.ndarray,
    *,
    random_state: int,
    max_samples: int = 5000,
) -> float:
    """Estimate the RBF maximum mean discrepancy between two samples.

    The maximum mean discrepancy (MMD) compares the first-order moments of the
    real and synthetic samples in a reproducing kernel Hilbert space. We use a
    radial basis function (RBF) kernel with the median heuristic for the
    bandwidth and subsample extremely large arrays for efficiency. Scores close
    to ``0.0`` indicate that the generator matches the reference feature well;
    higher scores denote a stronger distribution gap. Practitioners often
    compare MMD magnitudes across features: values above ``0.05``–``0.1``
    typically warrant inspection in structured clinical datasets.

    Args:
        real: One-dimensional array of reference observations.
        synthetic: One-dimensional array of synthetic observations.
        random_state: Seed used when subsampling large arrays prior to the
            kernel computation.
        max_samples: Maximum number of samples drawn from each array. Larger
            inputs are subsampled without replacement.

    Returns:
        The estimated MMD value. ``numpy.nan`` is returned when either input is
        empty after removing non-finite values.

    Example:
        >>> rng = np.random.default_rng(0)
        >>> real = rng.normal(loc=0.0, scale=1.0, size=200)
        >>> synth = rng.normal(loc=0.1, scale=1.1, size=200)
        >>> score = rbf_mmd(real, synth, random_state=0)
        >>> round(score, 3) >= 0.0
        True
    """

    real = np.asarray(real, dtype=float)
    synthetic = np.asarray(synthetic, dtype=float)
    real = real[np.isfinite(real)]
    synthetic = synthetic[np.isfinite(synthetic)]
    if real.size == 0 or synthetic.size == 0:
        return float("nan")

    rng = np.random.default_rng(random_state)
    if real.size > max_samples:
        real = rng.choice(real, size=max_samples, replace=False)
    if synthetic.size > max_samples:
        synthetic = rng.choice(synthetic, size=max_samples, replace=False)

    real = real[:, None]
    synthetic = synthetic[:, None]
    data = np.concatenate([real, synthetic], axis=0)
    squared_distances = (data - data.T) ** 2
    median_sq = float(np.median(squared_distances))
    bandwidth = np.sqrt(0.5 * median_sq) if median_sq > 1e-12 else 1.0

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        distances = (a - b.T) ** 2
        return np.exp(-distances / (2.0 * bandwidth**2))

    k_xx = kernel(real, real)
    k_yy = kernel(synthetic, synthetic)
    k_xy = kernel(real, synthetic)
    mmd = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return float(max(mmd, 0.0))


def mutual_information_feature(
    real: np.ndarray, synthetic: np.ndarray, n_bins: int = 10
) -> float:
    """Estimate the mutual information between dataset identity and a feature.

    The score quantifies how informative the feature is about whether a sample
    originated from the real dataset or from the generator. ``0.0`` denotes no
    detectable dependence, while higher scores indicate that a downstream
    classifier could reliably separate the distributions. Analysts commonly use
    ``n_bins`` in the ``10``–``20`` range and flag features whose mutual
    information exceeds ``0.1`` bits as candidates for debugging.

    Args:
        real: One-dimensional array of reference observations.
        synthetic: One-dimensional array of synthetic observations.
        n_bins: Number of quantile-based bins used to discretise the feature.

    Returns:
        Mutual information measured in bits. ``numpy.nan`` is returned when the
        metric is undefined (e.g., not enough unique values to form bins).

    Example:
        >>> import numpy as np
        >>> score = mutual_information_feature(
        ...     np.array([0.0, 0.2, 0.4, 0.6]),
        ...     np.array([0.7, 0.8, 0.9, 1.0]),
        ... )
        >>> score >= 0.0
        True
    """

    real = np.asarray(real, dtype=float)
    synthetic = np.asarray(synthetic, dtype=float)
    real = real[np.isfinite(real)]
    synthetic = synthetic[np.isfinite(synthetic)]
    if real.size == 0 or synthetic.size == 0:
        return float("nan")

    combined = np.concatenate([real, synthetic])
    quantiles = np.quantile(combined, np.linspace(0.0, 1.0, n_bins + 1))
    bin_edges = np.unique(quantiles)
    if bin_edges.size <= 1:
        return 0.0

    interior = bin_edges[1:-1]
    real_binned = np.digitize(real, interior, right=False)
    synthetic_binned = np.digitize(synthetic, interior, right=False)

    if np.unique(real_binned).size <= 1 and np.unique(synthetic_binned).size <= 1:
        return 0.0

    dataset_indicator = np.concatenate(
        [
            np.zeros(real_binned.size, dtype=int),
            np.ones(synthetic_binned.size, dtype=int),
        ]
    )
    feature_bins = np.concatenate([real_binned, synthetic_binned])
    if np.unique(feature_bins).size <= 1:
        return 0.0
    return float(mutual_info_score(dataset_indicator, feature_bins))
