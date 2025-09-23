"""Shared utilities for the MIMIC mortality modelling examples."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from suave import Schema, SchemaInferenceMode, SchemaInferencer  # noqa: E402

RANDOM_STATE: int = 42
TARGET_COLUMNS: Tuple[str, str] = ("in_hospital_mortality", "28d_mortality")
CATEGORICAL_FEATURES: Tuple[str, str, str] = (
    "sex",
    "CRRT",
    "Respiratory_Support",
)
CALIBRATION_SIZE: float = 0.2
VALIDATION_SIZE: float = 0.2

__all__ = [
    "RANDOM_STATE",
    "TARGET_COLUMNS",
    "CATEGORICAL_FEATURES",
    "CALIBRATION_SIZE",
    "VALIDATION_SIZE",
    "Schema",
    "SchemaInferenceMode",
    "SchemaInferencer",
    "load_dataset",
    "define_schema",
    "prepare_features",
    "split_train_validation_calibration",
    "schema_markdown_table",
    "format_float",
    "compute_auc",
    "to_numeric_frame",
    "kolmogorov_smirnov_statistic",
    "rbf_mmd",
    "mutual_information_feature",
]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a TSV file into a :class:`pandas.DataFrame`."""

    return pd.read_csv(path, sep="\t")


def define_schema(df: pd.DataFrame, feature_columns: Iterable[str], mode=SchemaInferenceMode.INFO) -> Schema:
    """Create a :class:`Schema` describing ``df``'s feature columns."""

    inferencer = SchemaInferencer(categorical_overrides=CATEGORICAL_FEATURES)
    result = inferencer.infer(
        df,
        feature_columns,
        mode=mode,
    )
    for message in result.messages:
        print(f"[schema] {message}")
    return result.schema


def prepare_features(df: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    """Return features aligned to ``feature_columns``."""

    return df.loc[:, list(feature_columns)].copy()


def split_train_validation_calibration(
    features: pd.DataFrame,
    targets: pd.Series,
    *,
    calibration_size: float,
    validation_size: float,
    random_state: int,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Split data into train, validation, and calibration subsets."""

    from sklearn.model_selection import train_test_split

    X_model, X_calibration, y_model, y_calibration = train_test_split(
        features,
        targets,
        test_size=calibration_size,
        stratify=targets,
        random_state=random_state,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_model,
        y_model,
        test_size=validation_size,
        stratify=y_model,
        random_state=random_state,
    )
    return (
        X_train.reset_index(drop=True),
        X_validation.reset_index(drop=True),
        X_calibration.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_validation.reset_index(drop=True),
        y_calibration.reset_index(drop=True),
    )


def schema_markdown_table(schema: Schema) -> str:
    """Return a Markdown table summarising ``schema``."""

    header = "| Column | Type | n_classes | y_dim |\n| --- | --- | --- | --- |"
    rows = [header]
    schema_dict: MutableMapping[str, MutableMapping[str, object]] = schema.to_dict()
    for name, spec in schema_dict.items():
        n_classes = spec.get("n_classes", "")
        y_dim = spec.get("y_dim", "")
        rows.append(f"| {name} | {spec['type']} | {n_classes} | {y_dim} |")
    return "\n".join(rows)


def format_float(value: Optional[float]) -> str:
    """Format floating point numbers for Markdown tables."""

    if value is None:
        return "nan"
    if isinstance(value, float) and not np.isfinite(value):
        return "nan"
    return f"{float(value):.3f}"


def compute_auc(probabilities: np.ndarray, targets: pd.Series | np.ndarray) -> float:
    """Return the ROC AUC given predicted probabilities and targets."""

    from sklearn.metrics import roc_auc_score

    prob_matrix = np.asarray(probabilities)
    if prob_matrix.ndim == 1:
        positive_probs = prob_matrix
    else:
        positive_probs = prob_matrix[:, -1]
    labels = np.asarray(targets)
    if np.unique(labels).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, positive_probs))
    except ValueError:
        return float("nan")


def to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns in ``df`` to numeric values."""

    numeric = df.copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric


def kolmogorov_smirnov_statistic(real: np.ndarray, synthetic: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov statistic between two samples."""

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
    """Compute the RBF maximum mean discrepancy between ``real`` and ``synthetic``."""

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
    """Estimate mutual information between dataset indicator and feature bins."""

    from sklearn.metrics import mutual_info_score

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
