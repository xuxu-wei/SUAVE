"""Supervised mortality modelling and analysis for the sepsis datasets."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from suave import SUAVE, Schema
from suave.evaluate import evaluate_tstr, evaluate_trtr, simple_membership_inference


RANDOM_STATE = 42
TARGET_COLUMNS = ("in_hospital_mortality", "28d_mortality")
CATEGORICAL_FEATURES = ("sex", "CRRT", "Respiratory_Support")
CALIBRATION_SIZE = 0.2
OUTPUT_DIR_NAME = "analysis_outputs"


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a TSV file into a :class:`pandas.DataFrame`."""

    return pd.read_csv(path, sep="\t")


def define_schema(df: pd.DataFrame, feature_columns: Iterable[str]) -> Schema:
    """Create a :class:`Schema` describing ``df``'s feature columns."""

    schema_dict: Dict[str, Mapping[str, object]] = {}
    for column in feature_columns:
        if column in CATEGORICAL_FEATURES:
            n_classes = int(df[column].dropna().nunique())
            schema_dict[column] = {"type": "cat", "n_classes": max(n_classes, 2)}
        else:
            schema_dict[column] = {"type": "real"}
    return Schema(schema_dict)


def prepare_features(df: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    """Return features aligned to ``feature_columns``."""

    return df.loc[:, list(feature_columns)].copy()


def split_train_calibration(
    features: pd.DataFrame,
    targets: pd.Series,
    *,
    calibration_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into model-training and calibration subsets."""

    X_train, X_cal, y_train, y_cal = train_test_split(
        features,
        targets,
        test_size=calibration_size,
        stratify=targets,
        random_state=random_state,
    )
    return (
        X_train.reset_index(drop=True),
        X_cal.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_cal.reset_index(drop=True),
    )


def compute_binary_metrics(
    probabilities: np.ndarray, targets: pd.Series | np.ndarray
) -> Dict[str, float]:
    """Compute AUROC, accuracy, specificity, sensitivity, and Brier score."""

    prob_matrix = np.asarray(probabilities)
    if prob_matrix.ndim == 1:
        positive_probs = prob_matrix
    else:
        positive_probs = prob_matrix[:, -1]
    labels = np.asarray(targets)
    predictions = (positive_probs >= 0.5).astype(int)

    metrics: Dict[str, float] = {}

    try:
        metrics["AUC"] = float(roc_auc_score(labels, positive_probs))
    except ValueError:
        metrics["AUC"] = float("nan")

    metrics["ACC"] = float(accuracy_score(labels, predictions))
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics["SPE"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    metrics["SEN"] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    metrics["Brier"] = float(brier_score_loss(labels, positive_probs))
    return metrics


def plot_calibration_curves(
    probability_map: Mapping[str, np.ndarray],
    label_map: Mapping[str, np.ndarray],
    *,
    target_name: str,
    output_path: Path,
    n_bins: int = 10,
) -> None:
    """Generate calibration curves with Brier scores annotated in the legend."""

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="tab:gray", label="Perfect calibration")

    for dataset_name, probs in probability_map.items():
        labels = label_map[dataset_name]
        if probs.ndim == 2:
            pos_probs = probs[:, -1]
        else:
            pos_probs = probs
        try:
            frac_pos, mean_pred = calibration_curve(labels, pos_probs, n_bins=n_bins)
        except ValueError:
            continue
        brier = brier_score_loss(labels, pos_probs)
        ax.plot(mean_pred, frac_pos, marker="o", label=f"{dataset_name} (Brier={brier:.3f})")

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration: {target_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_latent_space(
    model: SUAVE,
    feature_map: Mapping[str, pd.DataFrame],
    label_map: Mapping[str, pd.Series | np.ndarray],
    *,
    target_name: str,
    output_path: Path,
) -> None:
    """Project latent representations with PCA and create scatter plots."""

    latent_blocks: List[np.ndarray] = []
    dataset_keys: List[str] = []
    for name, features in feature_map.items():
        if features.empty:
            continue
        latents = model.encode(features)
        if latents.size == 0:
            continue
        latent_blocks.append(latents)
        dataset_keys.append(name)

    if not latent_blocks:
        return

    concatenated = np.vstack(latent_blocks)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(concatenated)

    offsets = np.cumsum([0] + [block.shape[0] for block in latent_blocks])
    fig, axes = plt.subplots(1, len(latent_blocks), figsize=(6 * len(latent_blocks), 5), sharex=True, sharey=True)
    if len(latent_blocks) == 1:
        axes = [axes]

    for idx, (ax, name) in enumerate(zip(axes, dataset_keys)):
        start, end = offsets[idx], offsets[idx + 1]
        subset = projected[start:end]
        labels = np.asarray(label_map[name])
        scatter = ax.scatter(
            subset[:, 0],
            subset[:, 1],
            c=labels,
            cmap="coolwarm",
            alpha=0.7,
            edgecolor="none",
        )
        ax.set_title(f"{name}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        legend = ax.legend(*scatter.legend_elements(), title="Label")
        ax.add_artist(legend)

    fig.suptitle(f"Latent space projection: {target_name}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


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
    cdf_synth = np.searchsorted(synthetic_sorted, combined, side="right") / synthetic_sorted.size
    return float(np.max(np.abs(cdf_real - cdf_synth)))


def rbf_mmd(
    real: np.ndarray, synthetic: np.ndarray, *, max_samples: int = 1000, random_state: int = 42
) -> float:
    """Compute the maximum mean discrepancy with an RBF kernel."""

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


def mutual_information_feature(real: np.ndarray, synthetic: np.ndarray, n_bins: int = 10) -> float:
    """Estimate mutual information between dataset indicator and feature bins."""

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
        [np.zeros(real_binned.size, dtype=int), np.ones(synthetic_binned.size, dtype=int)]
    )
    feature_bins = np.concatenate([real_binned, synthetic_binned])
    if np.unique(feature_bins).size <= 1:
        return 0.0
    return float(mutual_info_score(dataset_indicator, feature_bins))


def to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns in ``df`` to numeric values."""

    numeric = df.copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric


def make_logistic_pipeline() -> Pipeline:
    """Factory for the baseline classifier used in TSTR/TRTR."""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    )


def main() -> None:
    """Run the full supervised mortality analysis pipeline."""

    data_dir = Path(__file__).parent / "data" / "sepsis_mortality_dataset"
    output_dir = data_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(exist_ok=True)

    train_df = load_dataset(data_dir / "mimic-mortality-train.tsv")
    test_df = load_dataset(data_dir / "mimic-mortality-test.tsv")
    external_df = load_dataset(data_dir / "eicu-mortality-external_val.tsv")

    feature_columns = [column for column in train_df.columns if column not in TARGET_COLUMNS]
    schema = define_schema(train_df, feature_columns)

    metrics_records: List[Dict[str, object]] = []
    calibration_paths: Dict[str, Path] = {}
    latent_paths: Dict[str, Path] = {}
    membership_records: List[Dict[str, object]] = []

    models: Dict[str, SUAVE] = {}

    for target in TARGET_COLUMNS:
        if target not in train_df.columns:
            continue
        print(f"Training model for {target}…")
        X_full = prepare_features(train_df, feature_columns)
        y_full = train_df[target]

        X_train_model, X_calibration, y_train_model, y_calibration = split_train_calibration(
            X_full,
            y_full,
            calibration_size=CALIBRATION_SIZE,
            random_state=RANDOM_STATE,
        )

        model = SUAVE(
            schema=schema,
            latent_dim=16,
            hidden_dims=(128, 64),
            dropout=0.1,
            learning_rate=1e-3,
            batch_size=256,
            random_state=RANDOM_STATE,
        )
        model.fit(
            X_train_model,
            y_train_model,
            warmup_epochs=3,
            head_epochs=2,
            finetune_epochs=2,
        )
        model.calibrate(X_calibration, y_calibration)
        models[target] = model

        evaluation_datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
            "MIMIC test": (prepare_features(test_df, feature_columns), test_df[target]),
        }
        if target in external_df.columns:
            evaluation_datasets["eICU external"] = (
                prepare_features(external_df, feature_columns),
                external_df[target],
            )

        probability_map: Dict[str, np.ndarray] = {}
        label_map: Dict[str, np.ndarray] = {}
        for dataset_name, (features, labels) in evaluation_datasets.items():
            probs = model.predict_proba(features)
            probability_map[dataset_name] = probs
            label_map[dataset_name] = np.asarray(labels)
            metric_row = {
                "target": target,
                "dataset": dataset_name,
                **compute_binary_metrics(probs, labels),
            }
            metrics_records.append(metric_row)

        calibration_path = output_dir / f"calibration_{target}.png"
        plot_calibration_curves(probability_map, label_map, target_name=target, output_path=calibration_path)
        calibration_paths[target] = calibration_path

        latent_features = {
            name: features for name, (features, _) in evaluation_datasets.items()
        }
        latent_labels = {name: labels for name, (_, labels) in evaluation_datasets.items()}
        latent_path = output_dir / f"latent_{target}.png"
        plot_latent_space(model, latent_features, latent_labels, target_name=target, output_path=latent_path)
        latent_paths[target] = latent_path

        train_probabilities = model.predict_proba(X_train_model)
        test_probabilities = model.predict_proba(evaluation_datasets["MIMIC test"][0])
        membership = simple_membership_inference(
            train_probabilities,
            np.asarray(y_train_model),
            test_probabilities,
            np.asarray(evaluation_datasets["MIMIC test"][1]),
        )
        membership_records.append({"target": target, **membership})

    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = output_dir / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    membership_df = pd.DataFrame(membership_records)
    membership_path = output_dir / "membership_inference.csv"
    membership_df.to_csv(membership_path, index=False)

    in_hospital_model = models.get("in_hospital_mortality")
    if in_hospital_model is not None:
        print("Generating synthetic data for TSTR/TRTR comparisons…")
        X_train_full = prepare_features(train_df, feature_columns)
        y_train_full = train_df["in_hospital_mortality"]
        numeric_train = to_numeric_frame(X_train_full)

        rng = np.random.default_rng(RANDOM_STATE)
        synthetic_labels = rng.choice(y_train_full, size=len(y_train_full), replace=True)
        synthetic_features = in_hospital_model.sample(
            len(synthetic_labels), conditional=True, y=synthetic_labels
        )
        numeric_synthetic = to_numeric_frame(synthetic_features[feature_columns])

        numeric_test = to_numeric_frame(prepare_features(test_df, feature_columns))
        y_test = test_df["in_hospital_mortality"]

        tstr_metrics = evaluate_tstr(
            (numeric_synthetic.to_numpy(), np.asarray(synthetic_labels)),
            (numeric_test.to_numpy(), y_test.to_numpy()),
            make_logistic_pipeline,
        )
        trtr_metrics = evaluate_trtr(
            (numeric_train.to_numpy(), y_train_full.to_numpy()),
            (numeric_test.to_numpy(), y_test.to_numpy()),
            make_logistic_pipeline,
        )
        tstr_df = pd.DataFrame(
            [
                {"setting": "TSTR", **tstr_metrics},
                {"setting": "TRTR", **trtr_metrics},
            ]
        )
        tstr_path = output_dir / "tstr_trtr_comparison.csv"
        tstr_df.to_csv(tstr_path, index=False)

        distribution_rows: List[Dict[str, object]] = []
        for column in feature_columns:
            real_values = numeric_train[column].to_numpy()
            synthetic_values = numeric_synthetic[column].to_numpy()
            distribution_rows.append(
                {
                    "feature": column,
                    "ks": kolmogorov_smirnov_statistic(real_values, synthetic_values),
                    "mmd": rbf_mmd(real_values, synthetic_values),
                    "mutual_information": mutual_information_feature(real_values, synthetic_values),
                }
            )
        distribution_df = pd.DataFrame(distribution_rows)
        distribution_path = output_dir / "distribution_shift_metrics.csv"
        distribution_df.to_csv(distribution_path, index=False)

    print("Analysis complete.")
    print(f"Metric table saved to {metrics_path}")
    for target, path in calibration_paths.items():
        print(f"Calibration plot for {target}: {path}")
    for target, path in latent_paths.items():
        print(f"Latent space plot for {target}: {path}")
    print(f"Membership inference results saved to {membership_path}")
    if in_hospital_model is not None:
        print(f"TSTR/TRTR comparison saved to {tstr_path}")
        print(f"Distribution metrics saved to {distribution_path}")


if __name__ == "__main__":
    main()
