"""Data utilities for the minimal SUAVE implementation."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import warnings

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

    ``real`` columns are z-scored. ``pos`` columns are log transformed (using
    ``log1p``) and normalised with mean/standard deviation of the transformed
    values. ``count`` columns are log transformed (adding a one-offset when the
    observed minimum is zero). ``cat`` columns are cast to categorical dtype
    and ``ordinal`` columns are validated to fall inside the configured class
    range. The function returns the transformed dataframe alongside the
    metadata required to reconstruct the original values.

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

    for column in schema.positive_features:
        if column not in transformed:
            continue
        values = pd.to_numeric(transformed[column], errors="coerce")
        finite = values[values.notna()]
        if not finite.empty and (finite < -1.0 + _EPS).any():
            raise ValueError(
                f"Column '{column}' of type 'pos' must be >= -1 to apply log1p"
            )
        log_values = np.log1p(values)
        mean = float(np.nanmean(log_values)) if log_values.size else 0.0
        std = float(np.nanstd(log_values, ddof=0)) if log_values.size else 1.0
        if not np.isfinite(std) or std < _EPS:
            std = 1.0
        normalised = (log_values - mean) / max(std, _EPS)
        transformed[column] = normalised
        stats[column] = {"type": "pos", "mean_log": mean, "std_log": std}

    for column in schema.count_features:
        if column not in transformed:
            continue
        values = pd.to_numeric(transformed[column], errors="coerce")
        finite = values[values.notna()]
        if not finite.empty and (finite < 0).any():
            raise ValueError(f"Column '{column}' of type 'count' must be non-negative")
        offset = 1.0 if (not finite.empty and (finite <= 0).any()) else 0.0
        shifted = values + offset
        log_values = np.log(shifted)
        transformed[column] = log_values
        stats[column] = {"type": "count", "offset": offset}

    for column in schema.categorical_features:
        if column not in transformed:
            continue
        transformed[column] = transformed[column].astype("category")
        categories = transformed[column].cat.categories.tolist()
        stats[column] = {"type": "cat", "categories": categories}

    for column in schema.ordinal_features:
        if column not in transformed:
            continue
        spec = schema[column]
        n_classes = int(spec.n_classes or 0)
        original = transformed[column]
        series = pd.to_numeric(original, errors="coerce")
        invalid_coercion = original.notna() & series.isna()
        out_of_range = series.notna() & ~series.between(0, n_classes - 1)
        if invalid_coercion.any() or out_of_range.any():
            range_text = (
                f"[0, {n_classes - 1}]" if n_classes else "the configured ordinal range"
            )
            warnings.warn(
                (
                    f"Column '{column}' contains ordinal values outside {range_text} "
                    "or non-numeric entries; they will be treated as missing."
                ),
                UserWarning,
                stacklevel=2,
            )
            series[invalid_coercion] = np.nan
            series[out_of_range] = np.nan
        categories: Iterable[object]
        if isinstance(transformed[column].dtype, pd.CategoricalDtype):
            categories = transformed[column].cat.categories.tolist()
        else:
            categories = list(range(n_classes))
        transformed[column] = series
        stats[column] = {
            "type": "ordinal",
            "n_classes": n_classes,
            "categories": list(categories),
        }

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
        elif meta["type"] == "pos":
            mean_log = float(meta.get("mean_log", 0.0))
            std_log = float(meta.get("std_log", 1.0))
            values = pd.to_numeric(restored[column], errors="coerce")
            log_values = values * std_log + mean_log
            restored[column] = np.expm1(log_values)
        elif meta["type"] == "count":
            offset = float(meta.get("offset", 0.0))
            values = pd.to_numeric(restored[column], errors="coerce")
            restored[column] = np.exp(values) - offset
        elif meta["type"] == "ordinal":
            categories = meta.get("categories")
            if categories is not None:
                restored[column] = pd.Categorical(
                    restored[column], categories=categories, ordered=True
                )
    return restored
