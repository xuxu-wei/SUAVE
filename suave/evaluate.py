"""Evaluation helpers for SUAVE's supervised branch."""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .schema_inference import SchemaInferencer
from .types import Schema


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

    Parameters
    ----------
    probabilities : numpy.ndarray
        Probability estimates. One-dimensional arrays are interpreted as the
        positive class probability for binary problems.
    targets : numpy.ndarray
        Integer encoded target labels aligned with ``probabilities``.

    Returns
    -------
    float
        The AUROC value. ``numpy.nan`` is returned when the metric is
        undefined, such as when only a single class is present or scikit-learn
        raises a ``ValueError`` for degenerate inputs.

    Notes
    -----
    AUROC scores close to ``0.5`` indicate random guessing, values above
    ``0.8`` generally reflect strong separability, and scores below ``0.6`` are
    often considered inadequate for production use.

    Examples
    --------
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

    Parameters
    ----------
    probabilities : numpy.ndarray
        Probability estimates for each class.
    targets : numpy.ndarray
        Integer encoded target labels aligned with ``probabilities``.

    Returns
    -------
    float
        The macro-averaged AUPRC across classes. ``numpy.nan`` is returned when
        the score is undefined for the provided data.

    Notes
    -----
    Baseline AUPRC roughly matches the positive class prevalence; values above
    ``0.5`` typically signal useful precision-recall trade-offs, while scores
    near prevalence indicate the classifier is barely better than random.

    Examples
    --------
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

    Parameters
    ----------
    probabilities : numpy.ndarray
        Probability estimates for each class.
    targets : numpy.ndarray
        Integer encoded target labels aligned with ``probabilities``.

    Returns
    -------
    float
        The Brier score. Lower values indicate better calibrated probabilities.
        ``numpy.nan`` is returned for empty inputs.

    Notes
    -----
    Well-calibrated clinical risk models typically achieve Brier scores below
    ``0.1``. Values greater than ``0.2`` often flag poorly calibrated
    predictions that may require recalibration.

    Examples
    --------
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

    Parameters
    ----------
    probabilities : numpy.ndarray
        Probability estimates for each class.
    targets : numpy.ndarray
        Integer encoded target labels aligned with ``probabilities``.
    n_bins : int, default 10
        Number of equally spaced bins used to evaluate calibration.

    Returns
    -------
    float
        The ECE value. ``numpy.nan`` is returned for empty inputs.

    Notes
    -----
    ECE below ``0.05`` generally reflects well-calibrated predictions, whereas
    values above ``0.1`` highlight calibration drift worth investigating.

    Examples
    --------
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

    Parameters
    ----------
    probabilities : numpy.ndarray
        Probability estimates shaped ``(n_samples, n_classes)`` or a
        one-dimensional array containing positive class probabilities for
        binary classification.
    targets : numpy.ndarray
        Integer encoded ground-truth labels aligned with ``probabilities``.
        Labels must be zero-indexed to match column positions.

    Returns
    -------
    dict of str to float
        Dictionary containing accuracy, AUROC, AUPRC, Brier score, and ECE
        values. Metrics that are undefined for the provided data return
        ``numpy.nan``.

    Notes
    -----
    Accuracy and AUROC values above ``0.8`` usually indicate strong models,
    while Brier scores below ``0.1`` and ECE below ``0.05`` reflect reliable
    calibration. Deviations from these heuristics should trigger domain-specific
    review.

    Examples
    --------
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

    Parameters
    ----------
    synthetic : tuple of numpy.ndarray
        Tuple ``(features, targets)`` used for training. The classifier produced
        by ``model_factory`` must support :meth:`fit` and :meth:`predict_proba`
        on these arrays.
    real : tuple of numpy.ndarray
        Tuple ``(features, targets)`` representing the held-out real data used
        for evaluation.
    model_factory : Callable[[], Any]
        Callable returning an unfitted probabilistic classifier (e.g.,
        ``lambda: LogisticRegression(max_iter=200)``).

    Returns
    -------
    dict of str to float
        Classification metrics on the real evaluation split.

    Notes
    -----
    When synthetic data is high fidelity, TSTR accuracy and AUROC typically lag
    the real-data baseline by less than five percentage points; larger gaps
    highlight modelling issues or coverage mismatches.

    Examples
    --------
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

    Parameters
    ----------
    real_train : tuple of numpy.ndarray
        Tuple ``(features, targets)`` used for training.
    real_test : tuple of numpy.ndarray
        Tuple ``(features, targets)`` used for evaluation.
    model_factory : Callable[[], Any]
        Callable returning an unfitted classifier supporting :meth:`fit` and
        :meth:`predict_proba`.

    Returns
    -------
    dict of str to float
        Classification metrics on the held-out real test split.

    Notes
    -----
    TRTR acts as an upper bound for synthetic-data experiments. TSTR results
    trailing TRTR by more than ``5``–``10`` percentage points typically point to
    fidelity gaps in the generator or mismatched preprocessing.

    Examples
    --------
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

    Parameters
    ----------
    train_probabilities : numpy.ndarray
        Probabilities predicted on the training data.
    train_targets : numpy.ndarray
        Integer encoded training labels.
    test_probabilities : numpy.ndarray
        Probabilities predicted on held-out data.
    test_targets : numpy.ndarray
        Integer encoded held-out labels.

    Returns
    -------
    dict of str to float
        Dictionary containing ``attack_auc``, ``attack_best_threshold``,
        ``attack_best_accuracy``, and ``attack_majority_class_accuracy``. The
        scores are ``numpy.nan`` when the attack is undefined (e.g., only a
        single membership class is present).

    Notes
    -----
    Attack AUC values near ``0.5`` signal limited privacy leakage, while scores
    exceeding ``0.7`` suggest the model's outputs expose membership signals
    worth mitigating.

    Examples
    --------
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


def _prepare_c2st_inputs(
    real_features: np.ndarray, synthetic_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Validate inputs for the classifier two-sample test."""

    real = np.asarray(real_features, dtype=float)
    synthetic = np.asarray(synthetic_features, dtype=float)
    if real.ndim != 2 or synthetic.ndim != 2:
        raise ValueError("Inputs must be two-dimensional feature matrices")
    if real.shape[1] != synthetic.shape[1]:
        raise ValueError("Real and synthetic features must share the same columns")

    real_mask = np.all(np.isfinite(real), axis=1)
    synth_mask = np.all(np.isfinite(synthetic), axis=1)
    real = real[real_mask]
    synthetic = synthetic[synth_mask]
    if real.size == 0 or synthetic.size == 0:
        raise ValueError("Both datasets must contain at least one finite row")

    features = np.vstack([real, synthetic])
    labels = np.concatenate(
        [np.zeros(real.shape[0], dtype=int), np.ones(synthetic.shape[0], dtype=int)]
    )
    return features, labels, real.shape[0], synthetic.shape[0]


def _cross_validated_probabilities(
    model_factory: Callable[[], Any],
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    """Return out-of-fold probability estimates for the classifier two-sample test."""

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    probabilities = np.zeros(labels.shape[0], dtype=float)
    for train_indices, test_indices in cv.split(features, labels):
        model = model_factory()
        model.fit(features[train_indices], labels[train_indices])
        fold_probabilities = model.predict_proba(features[test_indices])
        probabilities[test_indices] = np.asarray(fold_probabilities)[:, 1]
    return probabilities


def _bootstrap_auc_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.95,
) -> Tuple[float, float, int]:
    """Bootstrap the ROC-AUC confidence interval."""

    samples: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, labels.size, size=labels.size)
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        sampled_scores = scores[indices]
        samples.append(float(roc_auc_score(sampled_labels, sampled_scores)))
    if not samples:
        return float("nan"), float("nan"), 0

    lower_percentile = (1.0 - alpha) / 2.0 * 100.0
    upper_percentile = (1.0 + alpha) / 2.0 * 100.0
    lower, upper = np.percentile(samples, [lower_percentile, upper_percentile])
    return float(lower), float(upper), len(samples)


def _normalise_model_name(model_name: str) -> str:
    """Return a normalised identifier for ``model_name`` metric keys."""

    slug = re.sub(r"[^0-9a-zA-Z]+", "_", model_name.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "model"


def classifier_two_sample_test(
    real_features: np.ndarray,
    synthetic_features: np.ndarray,
    *,
    model_factories: Mapping[str, Callable[[], Any]],
    random_state: int,
    n_splits: int = 5,
    n_bootstrap: int = 1000,
) -> Dict[str, float]:
    """Run a classifier two-sample test (C2ST) between real and synthetic data.

    The procedure trains discriminative models that attempt to separate rows
    drawn from the real dataset and a synthetic cohort. Well-aligned
    distributions should yield ROC-AUC scores near ``0.5``. Callers can provide
    one or more model factories whose :meth:`predict_proba` outputs are
    evaluated with stratified ``n_splits``-fold cross-validation. Bootstrap
    resampling of the out-of-fold predictions provides confidence intervals for
    each classifier's ROC-AUC estimate.

    Parameters
    ----------
    real_features : numpy.ndarray
        Two-dimensional array containing real samples.
    synthetic_features : numpy.ndarray
        Two-dimensional array with synthetic samples that share the same column
        order as ``real_features``.
    model_factories : Mapping[str, Callable[[], Any]]
        Mapping from a model identifier to a factory function that returns an
        unfitted estimator implementing :meth:`fit`/:meth:`predict_proba` for
        binary classification. Keys are converted into snake-case metric
        prefixes (e.g., ``"XGBoost"`` becomes ``"xgboost"``).
    random_state : int
        Seed controlling the cross-validation shuffling and the bootstrap
        sampling procedure.
    n_splits : int, default 5
        Number of stratified folds used to obtain out-of-fold predictions for
        each classifier. The value is capped at the smallest class count to
        maintain valid splits.
    n_bootstrap : int, default 1000
        Number of bootstrap replicates used to derive confidence intervals.
        Degenerate resamples that contain a single class are skipped.

    Returns
    -------
    dict of str to float
        Dictionary containing the ROC-AUC and 95% bootstrap confidence
        intervals for each supplied classifier, along with bookkeeping metadata
        detailing the sample counts and effective bootstrap iterations.

    Raises
    ------
    ValueError
        If the inputs are not two-dimensional, if they do not share the same
        number of columns, or if insufficient samples remain after filtering
        non-finite rows to form at least two cross-validation splits per class.

    Notes
    -----
    ROC-AUC values within ``0.45``–``0.55`` typically indicate well-matched
    distributions, while scores exceeding ``0.7`` signal large shifts that
    warrant closer inspection of the synthetic generator.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from xgboost import XGBClassifier
    >>> rng = np.random.default_rng(0)
    >>> real = rng.normal(size=(100, 3))
    >>> synth = rng.normal(loc=0.1, size=(100, 3))
    >>> metrics = classifier_two_sample_test(
    ...     real,
    ...     synth,
    ...     model_factories={
    ...         "xgboost": lambda: XGBClassifier(random_state=0),
    ...         "logistic": lambda: LogisticRegression(max_iter=200),
    ...     },
    ...     random_state=0,
    ...     n_bootstrap=10,
    ... )
    >>> sorted(metrics.keys())[:2]
    ['cv_splits', 'logistic_auc']
    """

    if not model_factories:
        raise ValueError("model_factories must contain at least one classifier")

    features, labels, n_real, n_synth = _prepare_c2st_inputs(
        real_features, synthetic_features
    )

    class_counts = np.bincount(labels)
    min_class = int(class_counts.min())
    if min_class < 2:
        raise ValueError("At least two samples per class are required for C2ST")
    splits = max(2, min(n_splits, min_class))

    rng = np.random.default_rng(random_state)

    results: Dict[str, float] = {
        "n_real_samples": int(n_real),
        "n_synthetic_samples": int(n_synth),
        "n_features": int(features.shape[1]),
        "cv_splits": int(splits),
    }

    for model_name, factory in model_factories.items():
        key_prefix = _normalise_model_name(model_name)
        scores = _cross_validated_probabilities(
            factory,
            features,
            labels,
            n_splits=splits,
            random_state=random_state,
        )
        auc = float(roc_auc_score(labels, scores))
        ci_low, ci_high, effective = _bootstrap_auc_interval(
            labels,
            scores,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        results[f"{key_prefix}_auc"] = auc
        results[f"{key_prefix}_auc_ci_low"] = ci_low
        results[f"{key_prefix}_auc_ci_high"] = ci_high
        results[f"{key_prefix}_bootstrap_samples"] = float(effective)

    return results


def kolmogorov_smirnov_statistic(real: np.ndarray, synthetic: np.ndarray) -> float:
    """Compute the Kolmogorov–Smirnov statistic for univariate samples.

    The Kolmogorov–Smirnov (KS) statistic measures the maximum distance between
    the empirical cumulative distribution functions of the real and synthetic
    samples. Values close to ``0.0`` indicate that the synthetic distribution is
    indistinguishable from the reference sample, whereas scores approaching
    ``1.0`` highlight large distributional shifts.

    Parameters
    ----------
    real : numpy.ndarray
        One-dimensional array of observations drawn from the reference dataset.
    synthetic : numpy.ndarray
        One-dimensional array sampled from the generative model.

    Returns
    -------
    float
        The KS statistic in ``[0, 1]``. ``numpy.nan`` is returned when either
        input is empty after removing non-finite values.

    Notes
    -----
    A common heuristic is to flag KS statistics above ``0.1`` as a sign that
    the feature is not faithfully reproduced, although domain-specific
    tolerances should override this general threshold.

    Examples
    --------
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
    real: Any,
    synthetic: Any,
    *,
    random_state: int,
    max_samples: int = 5000,
    schema: Optional[Schema | Mapping[str, Mapping[str, object]]] = None,
    feature_names: Optional[Sequence[str]] = None,
    kernel: Optional[str | Mapping[str, str]] = None,
    n_permutations: int = 0,
) -> Tuple[float, float]:
    r"""Estimate the maximum mean discrepancy (MMD) between two samples.

    The helper supports both per-feature comparisons (one-dimensional input)
    and global multi-feature tests. Continuous features default to an RBF
    kernel with the median heuristic (computed from non-diagonal pairwise
    distances), while categorical features use a Kronecker :math:`\delta`
    kernel unless overridden. When ``n_permutations`` is positive, a
    permutation test estimates a one-sided ``p``-value that the observed MMD
    arose under the null hypothesis that both samples share the same
    distribution.

    Parameters
    ----------
    real : Any
        Reference observations. Accepts NumPy arrays, pandas Series, or
        DataFrames. One-dimensional inputs are treated as a single feature
        column.
    synthetic : Any
        Synthetic observations with the same feature layout as ``real``.
    random_state : int
        Seed used for subsampling and the optional permutation test.
    max_samples : int, default 5000
        Maximum number of samples drawn from each dataset prior to computing
        pairwise kernels. Larger inputs are subsampled without replacement.
    schema : Schema or mapping, optional
        Optional :class:`~suave.types.Schema` or schema-like mapping describing
        column types. If omitted, a silent
        :class:`~suave.schema_inference.SchemaInferencer` run infers per-column
        types to choose default kernels.
    feature_names : Sequence[str], optional
        Explicit feature ordering when passing raw arrays without column
        labels.
    kernel : str or mapping, optional
        Kernel override. Provide a string (``"rbf"`` or ``"delta"``) to apply
        the same kernel to all features, or a mapping from column name to kernel
        identifier. Unspecified columns fall back to the schema-derived
        defaults.
    n_permutations : int, default 0
        Number of label permutations used to estimate the ``p``-value. Set to
        ``0`` to skip the permutation test and return ``numpy.nan`` for the
        ``p``-value.

    Returns
    -------
    tuple of float
        Tuple ``(mmd, p_value)``. ``mmd`` is the averaged maximum mean
        discrepancy across feature groups and ``p_value`` is the optional
        permutation test result. ``numpy.nan`` is returned for both entries when
        inputs are empty after removing non-finite rows.

    Notes
    -----
    Empirically, MMD scores below ``0.02`` typically indicate close alignment,
    values between ``0.02`` and ``0.05`` warrant monitoring, and scores above
    ``0.1`` often reveal significant divergences between real and synthetic
    distributions.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> real = rng.normal(loc=0.0, scale=1.0, size=200)
    >>> synth = rng.normal(loc=0.1, scale=1.1, size=200)
    >>> score, p_value = rbf_mmd(real, synth, random_state=0, n_permutations=10)
    >>> round(score, 3) >= 0.0
    True
    """

    # Align both inputs to data frames so feature order and labels match.
    real_frame, feature_names = _coerce_to_dataframe(real, feature_names)
    synthetic_frame, feature_names = _coerce_to_dataframe(synthetic, feature_names)

    if list(real_frame.columns) != list(synthetic_frame.columns):
        raise ValueError(
            "real and synthetic inputs must share the same feature columns"
        )

    # Derive a schema and kernel assignment for each feature column.
    schema_obj = _resolve_schema(schema, feature_names, real_frame, synthetic_frame)
    kernel_assignments = _resolve_kernel_assignments(feature_names, schema_obj, kernel)

    rng = np.random.default_rng(random_state)
    group_results: list[float] = []
    permutation_payloads: list[_PermutationPayload] = []

    for kernel_name, columns in kernel_assignments.items():
        if not columns:
            continue
        column_list = list(columns)
        real_subset = real_frame[column_list]
        synthetic_subset = synthetic_frame[column_list]

        if kernel_name == "rbf":
            real_array = _clean_continuous(real_subset.to_numpy(dtype=float))
            synthetic_array = _clean_continuous(synthetic_subset.to_numpy(dtype=float))
        elif kernel_name == "delta":
            real_array = _clean_categorical(real_subset)
            synthetic_array = _clean_categorical(synthetic_subset)
        else:
            raise ValueError(f"Unsupported kernel '{kernel_name}'")

        if real_array.size == 0 or synthetic_array.size == 0:
            continue

        # Subsample large arrays to keep kernel computations tractable.
        real_array = _subsample_array(real_array, max_samples, rng)
        synthetic_array = _subsample_array(synthetic_array, max_samples, rng)

        if real_array.size == 0 or synthetic_array.size == 0:
            continue

        bandwidth = None
        if kernel_name == "rbf":
            # Use the median heuristic to set the RBF bandwidth.
            bandwidth = _median_bandwidth(np.vstack([real_array, synthetic_array]))

        score = _compute_mmd(real_array, synthetic_array, kernel_name, bandwidth)
        group_results.append(score)

        if n_permutations > 0:
            # Cache permutation payloads so we reuse kernels efficiently.
            permutation_payloads.append(
                _PermutationPayload(
                    kernel=kernel_name,
                    real=real_array,
                    synthetic=synthetic_array,
                    bandwidth=bandwidth,
                )
            )

    if not group_results:
        return float("nan"), float("nan")

    observed = float(np.mean(group_results))
    if n_permutations <= 0 or not permutation_payloads:
        return observed, float("nan")

    extreme_count = 0
    for _ in range(int(n_permutations)):
        perm_scores: list[float] = []
        for payload in permutation_payloads:
            combined = np.vstack([payload.real, payload.synthetic])
            if combined.size == 0:
                continue
            indices = rng.permutation(combined.shape[0])
            split = payload.real.shape[0]
            perm_real = combined[indices[:split]]
            perm_synth = combined[indices[split:]]
            perm_score = _compute_mmd(
                perm_real, perm_synth, payload.kernel, payload.bandwidth
            )
            perm_scores.append(perm_score)
        if not perm_scores:
            continue
        # Aggregate feature-level permutation scores into a single sample.
        perm_value = float(np.mean(perm_scores))
        if perm_value >= observed:
            extreme_count += 1

    p_value = (extreme_count + 1.0) / (n_permutations + 1.0)
    return observed, float(min(max(p_value, 0.0), 1.0))


def energy_distance(
    real: Any,
    synthetic: Any,
    *,
    random_state: int,
    max_samples: int = 4000,
    schema: Optional[Schema | Mapping[str, Mapping[str, object]]] = None,
    feature_names: Optional[Sequence[str]] = None,
    n_permutations: int = 0,
) -> Tuple[float, float]:
    """Compute the energy distance between real and synthetic samples.

    The energy distance measures how dissimilar two probability distributions
    are under a given metric. For continuous columns we compute Euclidean
    distances after z-scoring the concatenated real and synthetic arrays. For
    categorical features we fall back to the Hamming distance, counting the
    number of entries that disagree between two samples. The helper supports
    both univariate inputs (single feature comparisons) and multivariate
    tables. When ``n_permutations`` is positive a permutation test estimates the
    ``p``-value associated with the observed statistic under the null
    hypothesis that both datasets are identically distributed.

    Parameters
    ----------
    real : Any
        Reference observations supplied as a NumPy array, pandas Series, or
        DataFrame. One-dimensional inputs are treated as a single feature.
    synthetic : Any
        Synthetic observations following the same layout as ``real``.
    random_state : int
        Seed controlling the optional subsampling step and permutation test.
    max_samples : int, default 4000
        Maximum number of rows drawn from each dataset before computing
        pairwise distances. Larger datasets are subsampled without replacement
        to keep the quadratic distance matrices tractable.
    schema : Schema or mapping, optional
        Optional :class:`~suave.types.Schema` (or schema-like mapping) that
        declares per-column types. When omitted a silent
        :class:`~suave.schema_inference.SchemaInferencer` pass infers the
        column taxonomy.
    feature_names : Sequence[str], optional
        Explicit ordering of feature names when raw arrays without column
        labels are provided.
    n_permutations : int, default 0
        Number of permutations used to approximate the ``p``-value. Set to ``0``
        to skip the permutation test and return ``numpy.nan`` for the ``p``
        statistic.

    Returns
    -------
    tuple of float
        Tuple ``(distance, p_value)``. ``distance`` is the averaged energy
        distance across the selected feature groups and ``p_value`` is the
        permutation-based significance estimate. ``numpy.nan`` values are
        returned when either dataset is empty after filtering invalid rows.

    Notes
    -----
    Practitioners typically interpret scores below ``0.2`` as strong
    alignment, values between ``0.2`` and ``0.5`` as moderate drift, and
    statistics above ``0.5`` as evidence that the generator struggles to match
    the reference distribution. The distance normalises the Euclidean and
    Hamming components by ``sqrt(n_continuous)`` and ``n_categorical`` to avoid
    either modality dominating when feature counts differ. Distances are
    accumulated in memory-friendly blocks so the helper scales to a few
    thousand samples per dataset. These heuristics should be adapted to the
    domain at hand.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> real = rng.normal(size=256)
    >>> synthetic = rng.normal(loc=0.5, size=256)
    >>> distance, p_value = energy_distance(
    ...     real,
    ...     synthetic,
    ...     random_state=0,
    ...     n_permutations=25,
    ... )
    >>> distance >= 0.0 and (np.isnan(p_value) or 0.0 <= p_value <= 1.0)
    True
    """

    if n_permutations < 0:
        raise ValueError("n_permutations must be non-negative")

    real_frame, feature_names = _coerce_to_dataframe(real, feature_names)
    synthetic_frame, feature_names = _coerce_to_dataframe(synthetic, feature_names)

    schema_obj = _resolve_schema(schema, feature_names, real_frame, synthetic_frame)
    assignments = _resolve_kernel_assignments(feature_names, schema_obj, None)

    continuous_columns = assignments.get("rbf", tuple())
    categorical_columns = assignments.get("delta", tuple())

    if not continuous_columns and not categorical_columns:
        raise ValueError("energy_distance requires at least one feature column")

    # Drop rows with missing or non-finite entries so distance computations
    # operate on fully observed samples.
    real_valid = _energy_valid_mask(real_frame, continuous_columns, categorical_columns)
    synth_valid = _energy_valid_mask(
        synthetic_frame, continuous_columns, categorical_columns
    )
    real_clean = real_frame.loc[real_valid, feature_names]
    synth_clean = synthetic_frame.loc[synth_valid, feature_names]

    if real_clean.empty or synth_clean.empty:
        return float("nan"), float("nan")

    rng = np.random.default_rng(random_state)

    category_levels: Dict[str, pd.Index] = {}
    if categorical_columns:
        category_levels = {
            name: pd.Categorical(
                pd.concat(
                    [real_clean[name], synth_clean[name]],
                    axis=0,
                    ignore_index=True,
                )
            ).categories
            for name in categorical_columns
        }

    real_arrays = _extract_energy_arrays(
        real_clean,
        continuous_columns,
        categorical_columns,
        max_samples,
        rng,
        category_levels,
    )
    synth_arrays = _extract_energy_arrays(
        synth_clean,
        continuous_columns,
        categorical_columns,
        max_samples,
        rng,
        category_levels,
    )

    if real_arrays.n_samples == 0 or synth_arrays.n_samples == 0:
        return float("nan"), float("nan")

    combined_continuous = np.vstack([
        real_arrays.continuous,
        synth_arrays.continuous,
    ])
    combined_categorical = np.vstack([
        real_arrays.categorical,
        synth_arrays.categorical,
    ])

    n_real = real_arrays.n_samples
    n_synth = synth_arrays.n_samples
    observed = _energy_distance_from_indices(
        combined_continuous,
        combined_categorical,
        np.arange(n_real),
        np.arange(n_real, n_real + n_synth),
    )

    if not np.isfinite(observed):
        return float("nan"), float("nan")

    observed = float(max(observed, 0.0))
    if n_permutations == 0:
        return observed, float("nan")

    total = n_real + n_synth
    extreme = 0
    for _ in range(int(n_permutations)):
        permuted = rng.permutation(total)
        perm_real = permuted[:n_real]
        perm_synth = permuted[n_real:]
        permuted_value = _energy_distance_from_indices(
            combined_continuous, combined_categorical, perm_real, perm_synth
        )
        if not np.isfinite(permuted_value):
            continue
        if permuted_value >= observed:
            extreme += 1

    p_value = (extreme + 1.0) / (n_permutations + 1.0)
    return observed, float(min(max(p_value, 0.0), 1.0))


class _PermutationPayload:
    """Container for cached data used during the permutation test."""

    def __init__(
        self,
        *,
        kernel: str,
        real: np.ndarray,
        synthetic: np.ndarray,
        bandwidth: Optional[float],
    ) -> None:
        self.kernel = kernel
        self.real = real
        self.synthetic = synthetic
        self.bandwidth = bandwidth


class _EnergyArrays:
    """Container bundling aligned continuous and categorical matrices."""

    def __init__(
        self,
        *,
        continuous: np.ndarray,
        categorical: np.ndarray,
    ) -> None:
        self.continuous = continuous
        self.categorical = categorical

    @property
    def n_samples(self) -> int:
        return int(self.continuous.shape[0])


def _coerce_to_dataframe(
    data: Any, feature_names: Optional[Sequence[str]]
) -> Tuple[pd.DataFrame, list[str]]:
    """Return a dataframe representation of ``data`` with column labels."""

    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        columns = list(frame.columns)
    elif isinstance(data, pd.Series):
        frame = data.to_frame()
        columns = list(frame.columns)
    else:
        array = np.asarray(data)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        elif array.ndim != 2:
            raise ValueError("Inputs must be one- or two-dimensional")
        if feature_names is None:
            columns = [f"feature_{idx}" for idx in range(array.shape[1])]
        else:
            columns = list(feature_names)
            if len(columns) != array.shape[1]:
                raise ValueError("feature_names length does not match array width")
        frame = pd.DataFrame(array, columns=columns)

    if feature_names is not None:
        missing = set(feature_names) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in provided data")
        columns = list(feature_names)
        frame = frame.loc[:, columns]
    return frame.reset_index(drop=True), columns


def _resolve_schema(
    schema: Optional[Schema | Mapping[str, Mapping[str, object]]],
    feature_names: Sequence[str],
    real_frame: pd.DataFrame,
    synthetic_frame: pd.DataFrame,
) -> Optional[Schema]:
    """Resolve a Schema instance from user input or silent inference."""

    if schema is None:
        inferencer = SchemaInferencer()
        combined = pd.concat([real_frame, synthetic_frame], axis=0, ignore_index=True)
        inference = inferencer.infer(
            combined, feature_columns=feature_names, mode="silent"
        )
        schema_obj = inference.schema
    elif isinstance(schema, Schema):
        schema_obj = schema
    else:
        schema_obj = Schema(schema)

    schema_obj.require_columns(feature_names)
    return schema_obj


def _resolve_kernel_assignments(
    feature_names: Sequence[str],
    schema: Optional[Schema],
    kernel_override: Optional[str | Mapping[str, str]],
) -> Dict[str, Tuple[str, ...]]:
    """Return a mapping of kernel name to the columns it should process."""

    per_column: Dict[str, str] = {}
    if isinstance(kernel_override, str):
        per_column = {name: kernel_override for name in feature_names}
    elif isinstance(kernel_override, Mapping):
        per_column = {str(name): str(kind) for name, kind in kernel_override.items()}

    assignments: Dict[str, str] = {}
    for name in feature_names:
        kernel_name = per_column.get(name)
        if kernel_name is None and schema is not None:
            column_type = schema[name].type
            if column_type in {"real", "pos", "count"}:
                kernel_name = "rbf"
            elif column_type in {"cat", "ordinal"}:
                kernel_name = "delta"
        if kernel_name is None:
            kernel_name = "rbf"
        kernel_name = kernel_name.lower()
        if kernel_name not in {"rbf", "delta"}:
            raise ValueError(f"Unsupported kernel '{kernel_name}' for column '{name}'")
        assignments[name] = kernel_name

    grouped: Dict[str, list[str]] = {"rbf": [], "delta": []}
    for name, kernel_name in assignments.items():
        grouped.setdefault(kernel_name, []).append(name)
    return {kernel: tuple(columns) for kernel, columns in grouped.items() if columns}


def _energy_valid_mask(
    frame: pd.DataFrame,
    continuous: Sequence[str],
    categorical: Sequence[str],
) -> np.ndarray:
    """Return a mask selecting rows with finite continuous and categorical data."""

    if frame.empty:
        return np.zeros(frame.shape[0], dtype=bool)

    mask = np.ones(frame.shape[0], dtype=bool)
    if continuous:
        cont_values = frame.loc[:, continuous].to_numpy(dtype=float, copy=True)
        mask &= np.all(np.isfinite(cont_values), axis=1)
    if categorical:
        cat_frame = frame.loc[:, categorical].replace([np.inf, -np.inf], np.nan)
        mask &= cat_frame.notna().all(axis=1).to_numpy()
    return mask.astype(bool)


def _clean_continuous(array: np.ndarray) -> np.ndarray:
    """Remove non-finite rows from a continuous-valued array."""

    if array.ndim == 1:
        array = array.reshape(-1, 1)
    mask = np.all(np.isfinite(array), axis=1)
    return array[mask]


def _clean_categorical(frame: pd.DataFrame) -> np.ndarray:
    """Return categorical values without missing entries."""

    cleaned = frame.replace([np.inf, -np.inf], np.nan).dropna()
    array = cleaned.to_numpy()
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _subsample_array(
    array: np.ndarray, max_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Subsample ``array`` without replacement when it exceeds ``max_samples``."""

    if array.shape[0] > max_samples:
        indices = rng.choice(array.shape[0], size=max_samples, replace=False)
        return array[indices]
    return array


def _subsample_indices(
    n_rows: int, max_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Return row indices after optional subsampling."""

    if n_rows <= max_samples:
        return np.arange(n_rows, dtype=int)
    return rng.choice(n_rows, size=max_samples, replace=False).astype(int)


def _extract_energy_arrays(
    frame: pd.DataFrame,
    continuous: Sequence[str],
    categorical: Sequence[str],
    max_samples: int,
    rng: np.random.Generator,
    category_levels: Mapping[str, pd.Index],
) -> _EnergyArrays:
    """Return aligned continuous and categorical matrices for energy distance."""

    n_rows = frame.shape[0]
    if n_rows == 0:
        return _EnergyArrays(
            continuous=np.empty((0, len(continuous)), dtype=float),
            categorical=np.empty((0, len(categorical)), dtype=int),
        )

    indices = _subsample_indices(n_rows, max_samples, rng)

    if continuous:
        cont = frame.loc[:, continuous].to_numpy(dtype=float, copy=True)[indices]
        if cont.ndim == 1:
            cont = cont.reshape(-1, 1)
    else:
        cont = np.empty((indices.size, 0), dtype=float)

    if categorical:
        cat = np.empty((indices.size, len(categorical)), dtype=int)
        for col_idx, name in enumerate(categorical):
            categories = category_levels.get(name)
            encoded = pd.Categorical(
                frame[name], categories=categories, ordered=False
            ).codes
            cat[:, col_idx] = encoded[indices]
    else:
        cat = np.empty((indices.size, 0), dtype=int)

    return _EnergyArrays(continuous=cont, categorical=cat)


def _energy_distance_from_indices(
    combined_continuous: np.ndarray,
    combined_categorical: np.ndarray,
    real_indices: np.ndarray,
    synthetic_indices: np.ndarray,
) -> float:
    """Compute the energy distance given index partitions."""

    real_cont = combined_continuous[real_indices]
    synth_cont = combined_continuous[synthetic_indices]
    real_cat = combined_categorical[real_indices]
    synth_cat = combined_categorical[synthetic_indices]
    return _energy_distance_statistic(real_cont, synth_cont, real_cat, synth_cat)


def _energy_distance_statistic(
    real_continuous: np.ndarray,
    synthetic_continuous: np.ndarray,
    real_categorical: np.ndarray,
    synthetic_categorical: np.ndarray,
) -> float:
    """Return the energy distance for the provided continuous/categorical arrays."""

    n_real = real_continuous.shape[0]
    n_synth = synthetic_continuous.shape[0]
    if n_real == 0 or n_synth == 0:
        return float("nan")

    if real_continuous.ndim == 1:
        real_continuous = real_continuous.reshape(-1, 1)
    if synthetic_continuous.ndim == 1:
        synthetic_continuous = synthetic_continuous.reshape(-1, 1)
    if real_categorical.ndim == 1:
        real_categorical = real_categorical.reshape(-1, 1)
    if synthetic_categorical.ndim == 1:
        synthetic_categorical = synthetic_categorical.reshape(-1, 1)

    real_standardised, synth_standardised = _standardise_continuous(
        real_continuous, synthetic_continuous
    )

    cross = _mean_cross_distance(
        real_standardised, real_categorical, synth_standardised, synthetic_categorical
    )
    if not np.isfinite(cross):
        return float("nan")

    real_within = _mean_within_distance(real_standardised, real_categorical)
    synth_within = _mean_within_distance(synth_standardised, synthetic_categorical)

    distance = 2.0 * cross - real_within - synth_within
    return float(max(distance, 0.0))


def _standardise_continuous(
    real: np.ndarray, synthetic: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score continuous arrays using the pooled mean and variance."""

    if real.size == 0 and synthetic.size == 0:
        return real.reshape(real.shape[0], 0), synthetic.reshape(synthetic.shape[0], 0)

    if real.ndim == 1:
        real = real.reshape(-1, 1)
    if synthetic.ndim == 1:
        synthetic = synthetic.reshape(-1, 1)

    if real.shape[1] == 0 and synthetic.shape[1] == 0:
        return np.empty((real.shape[0], 0), dtype=float), np.empty(
            (synthetic.shape[0], 0), dtype=float
        )

    combined = np.vstack([real, synthetic])
    if combined.size == 0:
        return np.empty((real.shape[0], 0), dtype=float), np.empty(
            (synthetic.shape[0], 0), dtype=float
        )

    mean = np.mean(combined, axis=0)
    std = np.std(combined, axis=0, ddof=0)
    mask = std > 0.0
    if not np.any(mask):
        return np.empty((real.shape[0], 0), dtype=float), np.empty(
            (synthetic.shape[0], 0), dtype=float
        )

    normalised_real = (real[:, mask] - mean[mask]) / std[mask]
    normalised_synth = (synthetic[:, mask] - mean[mask]) / std[mask]
    return normalised_real.astype(float), normalised_synth.astype(float)


def _mean_cross_distance(
    real_continuous: np.ndarray,
    real_categorical: np.ndarray,
    synthetic_continuous: np.ndarray,
    synthetic_categorical: np.ndarray,
) -> float:
    """Return the mean pairwise distance across the two datasets."""

    n_real = real_continuous.shape[0]
    n_synth = synthetic_continuous.shape[0]
    if n_real == 0 or n_synth == 0:
        return float("nan")

    return float(
        _mean_pairwise_distance(
            real_continuous,
            synthetic_continuous,
            real_categorical,
            synthetic_categorical,
            triangular=False,
        )
    )


def _mean_within_distance(
    continuous: np.ndarray, categorical: np.ndarray
) -> float:
    """Return the expected pairwise distance within a dataset."""

    n_samples = continuous.shape[0]
    if n_samples < 2:
        return 0.0

    return float(
        _mean_pairwise_distance(
            continuous,
            continuous,
            categorical,
            categorical,
            triangular=True,
        )
    )


def _mean_pairwise_distance(
    a_continuous: np.ndarray,
    b_continuous: np.ndarray,
    a_categorical: np.ndarray,
    b_categorical: np.ndarray,
    *,
    triangular: bool,
) -> float:
    """Return the mean combined distance without materialising full matrices."""

    n_a = a_continuous.shape[0]
    n_b = b_continuous.shape[0]
    if triangular:
        if n_a != n_b:
            raise ValueError("triangular distances require square inputs")
        if n_a < 2:
            return 0.0
    elif n_a == 0 or n_b == 0:
        return float("nan")

    block_size = 512
    d_cont = a_continuous.shape[1]
    d_cat = a_categorical.shape[1]
    cont_scale = math.sqrt(d_cont) if d_cont > 0 else 1.0
    cat_scale = float(d_cat) if d_cat > 0 else 1.0

    total = 0.0
    count = 0

    for i_start in range(0, n_a, block_size):
        i_end = min(i_start + block_size, n_a)
        a_cont_block = a_continuous[i_start:i_end]
        a_cat_block = a_categorical[i_start:i_end]

        if triangular:
            j_range = range(i_start, n_b, block_size)
        else:
            j_range = range(0, n_b, block_size)

        for j_start in j_range:
            j_end = min(j_start + block_size, n_b)
            b_cont_block = b_continuous[j_start:j_end]
            b_cat_block = b_categorical[j_start:j_end]

            block_rows = i_end - i_start
            block_cols = j_end - j_start
            block_dist = np.zeros((block_rows, block_cols), dtype=float)

            if d_cont > 0:
                diff = a_cont_block[:, None, :] - b_cont_block[None, :, :]
                cont = np.linalg.norm(diff, axis=-1)
                if cont_scale != 1.0:
                    cont = cont / cont_scale
                block_dist += cont

            if d_cat > 0:
                comparisons = a_cat_block[:, None, :] != b_cat_block[None, :, :]
                cat = comparisons.sum(axis=-1).astype(float)
                if cat_scale != 1.0:
                    cat = cat / cat_scale
                block_dist += cat

            if triangular and i_start == j_start:
                if block_rows <= 1:
                    continue
                tri = np.triu_indices(block_rows, k=1)
                block_values = block_dist[tri]
            else:
                block_values = block_dist.ravel()

            if block_values.size == 0:
                continue

            total += block_values.sum()
            count += block_values.size

    if count == 0:
        return 0.0 if triangular else float("nan")
    return total / count


def _median_bandwidth(data: np.ndarray) -> Optional[float]:
    """Return the median-based bandwidth for an RBF kernel."""

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[0] < 2:
        return None
    diff = data[:, None, :] - data[None, :, :]
    squared = np.sum(diff**2, axis=-1)
    upper = squared[np.triu_indices_from(squared, k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        return None
    return float(np.sqrt(np.median(positive)))


def _compute_mmd(
    real: np.ndarray,
    synthetic: np.ndarray,
    kernel: str,
    bandwidth: Optional[float],
) -> float:
    """Compute the (biased) MMD estimate for ``real`` and ``synthetic``."""

    if kernel == "rbf":
        if bandwidth is None or not np.isfinite(bandwidth) or bandwidth <= 0.0:
            return 0.0
        k_xx = _rbf_kernel(real, real, bandwidth)
        k_yy = _rbf_kernel(synthetic, synthetic, bandwidth)
        k_xy = _rbf_kernel(real, synthetic, bandwidth)
    elif kernel == "delta":
        k_xx = _delta_kernel(real, real)
        k_yy = _delta_kernel(synthetic, synthetic)
        k_xy = _delta_kernel(real, synthetic)
    else:  # pragma: no cover - guarded upstream
        raise ValueError(f"Unsupported kernel '{kernel}'")

    mmd = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return float(max(mmd, 0.0))


def _rbf_kernel(a: np.ndarray, b: np.ndarray, bandwidth: float) -> np.ndarray:
    """Return the RBF kernel matrix for ``a`` and ``b``."""

    diff = a[:, None, :] - b[None, :, :]
    squared = np.sum(diff**2, axis=-1)
    return np.exp(-squared / (2.0 * bandwidth**2))


def _delta_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the Kronecker delta kernel matrix for categorical features."""

    comparisons = a[:, None, :] == b[None, :, :]
    if comparisons.ndim == 2:
        return comparisons.astype(float)
    return comparisons.all(axis=-1).astype(float)


def mutual_information_feature(
    real: np.ndarray, synthetic: np.ndarray, n_bins: int = 10
) -> float:
    """Estimate the mutual information between dataset identity and a feature.

    The score quantifies how informative the feature is about whether a sample
    originated from the real dataset or from the generator. ``0.0`` denotes no
    detectable dependence, while higher scores indicate that a downstream
    classifier could reliably separate the distributions.

    Parameters
    ----------
    real : numpy.ndarray
        One-dimensional array of reference observations.
    synthetic : numpy.ndarray
        One-dimensional array of synthetic observations.
    n_bins : int, default 10
        Number of quantile-based bins used to discretise the feature.

    Returns
    -------
    float
        Mutual information measured in bits. ``numpy.nan`` is returned when the
        metric is undefined (e.g., not enough unique values to form bins).

    Notes
    -----
    Analysts commonly use ``n_bins`` in the ``10``–``20`` range. Mutual
    information below ``0.02`` bits usually indicates negligible leakage, while
    scores above ``0.1`` bits should trigger closer inspection of the feature.

    Examples
    --------
    >>> import numpy as np
    >>> score = mutual_information_feature(
    ...     np.array([0.0, 0.2, 0.4, 0.6]),
    ...     np.array([0.7, 0.8, 0.9, 1.0]),
    ... )
    >>> score >= 0.0
    True
    """

    # Sanitize inputs by coercing to arrays and dropping non-finite values.
    real = np.asarray(real, dtype=float)
    synthetic = np.asarray(synthetic, dtype=float)
    real = real[np.isfinite(real)]
    synthetic = synthetic[np.isfinite(synthetic)]
    if real.size == 0 or synthetic.size == 0:
        return float("nan")

    # Construct quantile-based bin edges shared across both datasets.
    combined = np.concatenate([real, synthetic])
    quantiles = np.quantile(combined, np.linspace(0.0, 1.0, n_bins + 1))
    bin_edges = np.unique(quantiles)
    if bin_edges.size <= 1:
        return 0.0

    # Assign each sample to a bin to approximate the continuous distribution.
    interior = bin_edges[1:-1]
    real_binned = np.digitize(real, interior, right=False)
    synthetic_binned = np.digitize(synthetic, interior, right=False)

    if np.unique(real_binned).size <= 1 and np.unique(synthetic_binned).size <= 1:
        return 0.0

    # Build the contingency table relating dataset identity and binned values.
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
