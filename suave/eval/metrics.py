"""Basic classification metrics used in tests.

The helper transparently handles binary and multi-class inputs. Probabilities can
either be provided as a one-dimensional array containing the positive class
scores or as a two-dimensional matrix with one column per class. Multi-class
support follows the one-vs-rest convention with macro averaging for the AUC and
average precision metrics. The function raises a :class:`ValueError` when given
an inconsistent combination of labels and probabilities (e.g. multi-class labels
with a one-dimensional probability vector).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Return a dictionary of common classification metrics.

    Parameters
    ----------
    y_true
        Ground-truth labels.
    proba
        Predicted probabilities, either as a one-dimensional array for binary
        classification or a two-dimensional array of shape ``(n_samples,
        n_classes)`` for multi-class problems.

    Returns
    -------
    dict
        Dictionary containing AUROC, AUPRC, Brier score and negative
        log-likelihood.

    Raises
    ------
    ValueError
        If multi-class targets are provided together with a one-dimensional
        probability array or when the probability matrix has a column count that
        does not match the number of observed classes.
    """

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    unique_classes = np.unique(y_true)
    proba_is_matrix = proba.ndim == 2
    # Detection covers both probability matrices and label sets with more than
    # two unique classes.
    has_multiclass_indicators = proba_is_matrix or unique_classes.size > 2

    if has_multiclass_indicators:
        if proba_is_matrix and proba.shape[1] == 2 and unique_classes.size <= 2:
            # Binary classification represented with two probability columns.
            positive_proba = proba[:, 1]
            return {
                "auroc": float(roc_auc_score(y_true, positive_proba)),
                "auprc": float(average_precision_score(y_true, positive_proba)),
                "brier": float(brier_score_loss(y_true, positive_proba)),
                "nll": float(log_loss(y_true, positive_proba)),
            }

        if not proba_is_matrix:
            raise ValueError(
                "Multi-class classification requires `proba` to be a 2D array "
                "with one column per class."
            )

        classes = np.sort(unique_classes)
        if proba.shape[1] != classes.size:
            raise ValueError(
                "`proba` must have one column per unique class observed in "
                "`y_true` to compute multi-class metrics."
            )

        y_true_bin = label_binarize(y_true, classes=classes)
        return {
            "auroc": float(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
            ),
            "auprc": float(
                average_precision_score(y_true_bin, proba, average="macro")
            ),
            "brier": float(np.mean(np.sum((proba - y_true_bin) ** 2, axis=1))),
            "nll": float(log_loss(y_true, proba, labels=classes)),
        }

    if proba_is_matrix:
        positive_proba = proba[:, 1]
    else:
        positive_proba = proba

    return {
        "auroc": float(roc_auc_score(y_true, positive_proba)),
        "auprc": float(average_precision_score(y_true, positive_proba)),
        "brier": float(brier_score_loss(y_true, positive_proba)),
        "nll": float(log_loss(y_true, positive_proba)),
    }
