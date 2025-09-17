"""Data utilities for the minimal SUAVE implementation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .types import Schema

_EPS = 1e-6


def split_train_val(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame | np.ndarray,
    val_split: float,
    stratify: bool,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series | pd.DataFrame | np.ndarray,
    pd.Series | pd.DataFrame | np.ndarray,
]:
    """Split training data into train and validation subsets.

    Parameters
    ----------
    X:
        Feature matrix with shape ``(n_samples, n_features)``.
    y:
        Targets aligned with ``X``. Accepts :class:`pandas.Series`,
        :class:`pandas.DataFrame`, or :class:`numpy.ndarray`.
    val_split:
        Proportion of samples that should end up in the validation split.  Must
        satisfy ``0 < val_split < 1``.
    stratify:
        If ``True`` the split preserves label frequencies using ``y``.

    Returns
    -------
    (X_train, X_val, y_train, y_val)
        Tuple of training and validation subsets.
    """

    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1 (exclusive)")

    n_samples = len(X)
    indices = np.arange(n_samples)

    if stratify:
        if y is None:
            raise ValueError("Stratified split requested but no targets provided")
        y_array = np.asarray(y)
        val_indices: list[int] = []
        rng = np.random.default_rng(0)
        for cls in np.unique(y_array):
            cls_indices = indices[y_array == cls]
            if len(cls_indices) == 0:
                continue
            n_val = max(1, int(round(len(cls_indices) * val_split)))
            n_val = min(n_val, len(cls_indices))
            selected = rng.choice(cls_indices, size=n_val, replace=False)
            val_indices.extend(selected.tolist())
        val_indices = np.array(sorted(set(val_indices)))
    else:
        rng = np.random.default_rng(0)
        n_val = int(round(n_samples * val_split))
        n_val = max(1, min(n_samples, n_val))
        val_indices = np.array(sorted(rng.choice(indices, size=n_val, replace=False)))

    train_mask = np.ones(n_samples, dtype=bool)
    train_mask[val_indices] = False
    train_indices = indices[train_mask]

    X_train = X.iloc[train_indices].reset_index(drop=True)
    X_val = X.iloc[val_indices].reset_index(drop=True)

    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_train = y.iloc[train_indices].reset_index(drop=True)
        y_val = y.iloc[val_indices].reset_index(drop=True)
    else:
        y = np.asarray(y)
        y_train = y[train_indices]
        y_val = y[val_indices]

    return X_train, X_val, y_train, y_val


def build_missing_mask(X: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean mask indicating missing entries in ``X``."""

    return X.isna()


def standardize(
    X: pd.DataFrame,
    schema: Schema,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float | list[object]]]]:
    """Standardise the dataframe using schema information.

    ``real`` columns are z-scored. ``cat`` columns are cast to categorical
    dtype; their categories are recorded but left untouched.

    Returns
    -------
    transformed, stats
        ``transformed`` is the normalised dataframe and ``stats`` contains
        per-column metadata for :func:`inverse_standardize`.
    """

    schema.require_columns(X.columns)
    transformed = X.copy()
    stats: Dict[str, Dict[str, float | list[object]]] = {}

    for column in schema.real_features:
        if column not in transformed:
            continue
        values = pd.to_numeric(transformed[column], errors="coerce")
        mean = float(values.mean()) if len(values) > 0 else 0.0
        std = float(values.std(ddof=0)) if len(values) > 0 else 1.0
        if std < _EPS:
            std = 1.0
        transformed[column] = (values - mean) / max(std, _EPS)
        stats[column] = {"type": "real", "mean": mean, "std": std}

    for column in schema.categorical_features:
        if column not in transformed:
            continue
        transformed[column] = transformed[column].astype("category")
        categories = transformed[column].cat.categories.tolist()
        stats[column] = {"type": "cat", "categories": categories}

    return transformed, stats


def inverse_standardize(
    X: pd.DataFrame,
    schema: Schema,
    stats: Dict[str, Dict[str, float | list[object]]],
) -> pd.DataFrame:
    """Undo :func:`standardize` using the stored statistics."""

    restored = X.copy()
    for column, meta in stats.items():
        if meta["type"] == "real":
            mean = float(meta["mean"])
            std = float(meta["std"])
            restored[column] = restored[column] * std + mean
        elif meta["type"] == "cat":
            categories = meta.get("categories", [])
            restored[column] = pd.Categorical(restored[column], categories=categories)
    return restored
