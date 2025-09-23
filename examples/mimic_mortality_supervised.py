"""Supervised mortality modelling and analysis for the sepsis datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


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


from suave import (
    SUAVE,
    Schema,
    SchemaInferenceMode,
    SchemaInferencer,
)
from suave.evaluate import evaluate_tstr, evaluate_trtr, simple_membership_inference


try:
    import optuna
except ImportError as exc:  # pragma: no cover - optuna provided via requirements
    raise RuntimeError(
        "Optuna is required for the mortality analysis. Install it via 'pip install optuna'."
    ) from exc



RANDOM_STATE = 42
TARGET_COLUMNS = ("in_hospital_mortality", "28d_mortality")
CATEGORICAL_FEATURES = ("sex", "CRRT", "Respiratory_Support")
CALIBRATION_SIZE = 0.2
VALIDATION_SIZE = 0.2
OUTPUT_DIR_NAME = "analysis_outputs"
HIDDEN_DIMENSION_OPTIONS: Dict[str, Tuple[int, int]] = {
    "compact": (96, 48),
    "small": (128, 64),
    "medium": (256, 128),
    "wide": (384, 192),
    "extra_wide": (512, 256),
}



def load_dataset(path: Path) -> pd.DataFrame:
    """Load a TSV file into a :class:`pandas.DataFrame`."""

    return pd.read_csv(path, sep="\t")

def define_schema(df: pd.DataFrame, feature_columns: Iterable[str]) -> Schema:
    """Create a :class:`Schema` describing ``df``'s feature columns."""

    inferencer = SchemaInferencer(categorical_overrides=CATEGORICAL_FEATURES)
    result = inferencer.infer(
        df,
        feature_columns,
        mode=SchemaInferenceMode.INFO,
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
        roauc = float(roc_auc_score(labels, positive_probs))
    except ValueError:
        roauc = float("nan")

    metrics["ROAUC"] = roauc
    metrics["AUC"] = roauc

    metrics["ACC"] = float(accuracy_score(labels, predictions))
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics["SPE"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    metrics["SEN"] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    metrics["Brier"] = float(brier_score_loss(labels, positive_probs))
    return metrics


def run_optuna_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    schema: Schema,
    *,
    random_state: int,
    n_trials: Optional[int],
    timeout: Optional[int],
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
) -> tuple["optuna.study.Study", Dict[str, object]]:
    """Perform Optuna hyperparameter optimisation for :class:`SUAVE`."""

    if n_trials is not None and n_trials <= 0:
        n_trials = None
    if timeout is not None and timeout <= 0:
        timeout = None

    def objective(trial: "optuna.trial.Trial") -> float:
        latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32, 48, 64, 96])
        hidden_key = trial.suggest_categorical(
            "hidden_dims", list(HIDDEN_DIMENSION_OPTIONS.keys())
        )
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 2e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 384, 512, 768])
        beta = trial.suggest_float("beta", 0.3, 4.0)
        warmup_epochs = trial.suggest_int("warmup_epochs", 2, 12)
        head_epochs = trial.suggest_int("head_epochs", 1, 8)
        finetune_epochs = trial.suggest_int("finetune_epochs", 1, 10)

        model = SUAVE(
            schema=schema,
            latent_dim=latent_dim,
            hidden_dims=HIDDEN_DIMENSION_OPTIONS[hidden_key],
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            beta=beta,
            random_state=random_state,
            auto_parameters=False,
            behaviour="supervised",
        )

        start_time = time.perf_counter()
        model.fit(
            X_train,
            y_train,
            warmup_epochs=warmup_epochs,
            head_epochs=head_epochs,
            finetune_epochs=finetune_epochs,
        )
        fit_seconds = time.perf_counter() - start_time
        validation_probs = model.predict_proba(X_validation)
        validation_metrics = compute_binary_metrics(validation_probs, y_validation)
        trial.set_user_attr("validation_metrics", validation_metrics)
        trial.set_user_attr("fit_seconds", fit_seconds)

        roauc = validation_metrics.get("ROAUC", float("nan"))
        if not np.isfinite(roauc):
            raise optuna.exceptions.TrialPruned("Non-finite validation ROAUC")
        return roauc

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage and study_name),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    if study.best_trial is None:
        raise RuntimeError("Optuna search did not produce a best trial")
    best_attributes: Dict[str, object] = {
        "trial_number": study.best_trial.number,
        "value": study.best_value,
        "params": dict(study.best_trial.params),
        "validation_metrics": study.best_trial.user_attrs.get("validation_metrics", {}),
        "fit_seconds": study.best_trial.user_attrs.get("fit_seconds"),
    }
    return study, best_attributes



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
    ax.plot(
        [0, 1], [0, 1], linestyle="--", color="tab:gray", label="Perfect calibration"
    )


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
        ax.plot(
            mean_pred, frac_pos, marker="o", label=f"{dataset_name} (Brier={brier:.3f})"
        )


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
    fig, axes = plt.subplots(
        1,
        len(latent_blocks),
        figsize=(6 * len(latent_blocks), 5),
        sharex=True,
        sharey=True,
    )

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
    cdf_synth = (
        np.searchsorted(synthetic_sorted, combined, side="right")
        / synthetic_sorted.size
    )

    return float(np.max(np.abs(cdf_real - cdf_synth)))


def rbf_mmd(
    real: np.ndarray,
    synthetic: np.ndarray,
    *,
    max_samples: int = 1000,
    random_state: int = 42,

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


def mutual_information_feature(
    real: np.ndarray, synthetic: np.ndarray, n_bins: int = 10
) -> float:

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
        [
            np.zeros(real_binned.size, dtype=int),
            np.ones(synthetic_binned.size, dtype=int),
        ]

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


def parse_arguments() -> argparse.Namespace:
    """Return parsed command line arguments for the analysis script."""

    parser = argparse.ArgumentParser(
        description="Run the SUAVE mortality modelling analysis with Optuna tuning.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=30,
        help="Maximum number of Optuna trials per target (default: 30).",
    )
    parser.add_argument(
        "--optuna-timeout",
        type=int,
        default=7200,
        help="Maximum tuning time in seconds per target (default: 7200, i.e. 2 hours).",
    )
    parser.add_argument(
        "--optuna-study-prefix",
        type=str,
        default=None,
        help="Optional prefix for Optuna study names when using persistent storage.",
    )
    parser.add_argument(
        "--optuna-storage",
        type=str,
        default=None,
        help="Optional Optuna storage URL for persistent studies.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full supervised mortality analysis pipeline."""

    args = parse_arguments()

    data_dir = Path(__file__).parent / "data" / "sepsis_mortality_dataset"
    output_dir = data_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(exist_ok=True)

    train_df = load_dataset(data_dir / "mimic-mortality-train.tsv")
    test_df = load_dataset(data_dir / "mimic-mortality-test.tsv")
    external_df = load_dataset(data_dir / "eicu-mortality-external_val.tsv")

    feature_columns = [
        column for column in train_df.columns if column not in TARGET_COLUMNS
    ]
    schema = define_schema(train_df, feature_columns)
    schema_table = schema_markdown_table(schema)


    metrics_records: List[Dict[str, object]] = []
    calibration_paths: Dict[str, Path] = {}
    latent_paths: Dict[str, Path] = {}
    membership_records: List[Dict[str, object]] = []
    optuna_reports: Dict[str, Dict[str, object]] = {}

    models: Dict[str, SUAVE] = {}
    tstr_results: Optional[pd.DataFrame] = None
    tstr_path: Optional[Path] = None
    distribution_df: Optional[pd.DataFrame] = None
    distribution_path: Optional[Path] = None


    for target in TARGET_COLUMNS:
        if target not in train_df.columns:
            continue
        print(f"Training model for {target}…")
        X_full = prepare_features(train_df, feature_columns)
        y_full = train_df[target]

        (
            X_train_model,
            X_validation,
            X_calibration,
            y_train_model,
            y_validation,
            y_calibration,
        ) = split_train_validation_calibration(
            X_full,
            y_full,
            calibration_size=CALIBRATION_SIZE,
            validation_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
        )

        study_name = (
            f"{args.optuna_study_prefix}_{target}" if args.optuna_study_prefix else None
        )
        study, best_info = run_optuna_search(
            X_train_model,
            y_train_model,
            X_validation,
            y_validation,
            schema,
            random_state=RANDOM_STATE,
            n_trials=args.optuna_trials,
            timeout=args.optuna_timeout,
            study_name=study_name,
            storage=args.optuna_storage,
        )

        best_params = dict(best_info.get("params", {}))
        hidden_key = str(best_params.get("hidden_dims", "medium"))
        hidden_dims = HIDDEN_DIMENSION_OPTIONS.get(
            hidden_key, HIDDEN_DIMENSION_OPTIONS["medium"]
        )
        model = SUAVE(
            schema=schema,
            latent_dim=int(best_params.get("latent_dim", 16)),
            hidden_dims=hidden_dims,
            dropout=float(best_params.get("dropout", 0.1)),
            learning_rate=float(best_params.get("learning_rate", 1e-3)),
            batch_size=int(best_params.get("batch_size", 256)),
            beta=float(best_params.get("beta", 1.5)),
            random_state=RANDOM_STATE,
            auto_parameters=False,
            behaviour="supervised",

        )
        model.fit(
            X_train_model,
            y_train_model,
            warmup_epochs=int(best_params.get("warmup_epochs", 3)),
            head_epochs=int(best_params.get("head_epochs", 2)),
            finetune_epochs=int(best_params.get("finetune_epochs", 2)),

        )
        model.calibrate(X_calibration, y_calibration)
        models[target] = model

        evaluation_datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
            "Train": (X_train_model, y_train_model),
            "Validation": (X_validation, y_validation),
            "MIMIC test": (prepare_features(test_df, feature_columns), test_df[target]),
        }
        if target in external_df.columns:
            evaluation_datasets["eICU external"] = (
                prepare_features(external_df, feature_columns),
                external_df[target],
            )

        probability_map: Dict[str, np.ndarray] = {}
        label_map: Dict[str, np.ndarray] = {}
        dataset_metric_map: Dict[str, Dict[str, float]] = {}

        for dataset_name, (features, labels) in evaluation_datasets.items():
            probs = model.predict_proba(features)
            probability_map[dataset_name] = probs
            label_map[dataset_name] = np.asarray(labels)
            metrics = compute_binary_metrics(probs, labels)
            dataset_metric_map[dataset_name] = metrics
            metric_row = {
                "target": target,
                "dataset": dataset_name,
                **metrics,
            }
            metrics_records.append(metric_row)

        calibration_path = output_dir / f"calibration_{target}.png"
        plot_calibration_curves(
            probability_map, label_map, target_name=target, output_path=calibration_path
        )
        calibration_paths[target] = calibration_path

        latent_features = {
            name: features for name, (features, _) in evaluation_datasets.items()
        }
        latent_labels = {
            name: labels for name, (_, labels) in evaluation_datasets.items()
        }
        latent_path = output_dir / f"latent_{target}.png"
        plot_latent_space(
            model,
            latent_features,
            latent_labels,
            target_name=target,
            output_path=latent_path,
        )
        latent_paths[target] = latent_path

        train_probabilities = probability_map["Train"]
        test_probabilities = probability_map["MIMIC test"]

        membership = simple_membership_inference(
            train_probabilities,
            np.asarray(y_train_model),
            test_probabilities,
            np.asarray(evaluation_datasets["MIMIC test"][1]),
        )
        membership_records.append({"target": target, **membership})

        trial_rows: List[Dict[str, object]] = []
        for trial in study.trials:
            record: Dict[str, object] = {
                "trial_number": trial.number,
                "value": trial.value,
            }
            record.update(trial.params)
            val_metrics = trial.user_attrs.get("validation_metrics")
            if isinstance(val_metrics, Mapping):
                for metric_name, metric_value in val_metrics.items():
                    record[f"validation_{metric_name.lower()}"] = metric_value
            fit_seconds = trial.user_attrs.get("fit_seconds")
            if fit_seconds is not None:
                record["fit_seconds"] = fit_seconds
            trial_rows.append(record)
        trials_df = pd.DataFrame(trial_rows)
        trials_path = output_dir / f"optuna_trials_{target}.csv"
        if not trials_df.empty:
            trials_df.to_csv(trials_path, index=False)
        else:
            trials_path.write_text("trial_number,value\n")

        optuna_reports[target] = {
            "best": best_info,
            "best_params": best_params,
            "metrics": dataset_metric_map,
            "trials_csv": trials_path,
        }

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
        synthetic_labels = rng.choice(
            y_train_full, size=len(y_train_full), replace=True
        )

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
        tstr_results = pd.DataFrame(

            [
                {"setting": "TSTR", **tstr_metrics},
                {"setting": "TRTR", **trtr_metrics},
            ]
        )
        tstr_path = output_dir / "tstr_trtr_comparison.csv"
        tstr_results.to_csv(tstr_path, index=False)


        distribution_rows: List[Dict[str, object]] = []
        for column in feature_columns:
            real_values = numeric_train[column].to_numpy()
            synthetic_values = numeric_synthetic[column].to_numpy()
            distribution_rows.append(
                {
                    "feature": column,
                    "ks": kolmogorov_smirnov_statistic(real_values, synthetic_values),
                    "mmd": rbf_mmd(real_values, synthetic_values),
                    "mutual_information": mutual_information_feature(
                        real_values, synthetic_values
                    ),

                }
            )
        distribution_df = pd.DataFrame(distribution_rows)
        distribution_path = output_dir / "distribution_shift_metrics.csv"
        distribution_df.to_csv(distribution_path, index=False)

    summary_lines: List[str] = [
        "# Mortality modelling report",
        "",
        "## Schema",
        schema_table,
        "",
        "## Model selection and performance",
    ]

    if not optuna_reports:
        summary_lines.append("No models were trained.")

    for target, report in optuna_reports.items():
        best = report["best"]
        best_params = report["best_params"]
        metrics_map: Mapping[str, Dict[str, float]] = report["metrics"]
        summary_lines.append(f"### {target}")
        best_value = best.get("value")
        value_text = (
            f"{best_value:.4f}" if isinstance(best_value, (int, float)) else "n/a"
        )
        summary_lines.append(
            f"Best Optuna trial #{best.get('trial_number')} with validation ROAUC {value_text}"
        )
        summary_lines.append("Best parameters:")
        summary_lines.append("```json")
        summary_lines.append(json.dumps(best_params, indent=2, ensure_ascii=False))
        summary_lines.append("```")
        summary_lines.append("| Dataset | AUC | ACC | SPE | SEN | Brier |")
        summary_lines.append("| --- | --- | --- | --- | --- | --- |")
        for dataset_name, metrics in metrics_map.items():
            summary_lines.append(
                "| {dataset} | {auc} | {acc} | {spe} | {sen} | {brier} |".format(
                    dataset=dataset_name,
                    auc=format_float(metrics.get("AUC")),
                    acc=format_float(metrics.get("ACC")),
                    spe=format_float(metrics.get("SPE")),
                    sen=format_float(metrics.get("SEN")),
                    brier=format_float(metrics.get("Brier")),
                )
            )
        summary_lines.append(
            f"Optuna trials logged at: {report['trials_csv'].relative_to(output_dir)}"
        )
        summary_lines.append(
            f"Calibration plot: {calibration_paths[target].relative_to(output_dir)}"
        )
        summary_lines.append(
            f"Latent projection: {latent_paths[target].relative_to(output_dir)}"
        )
        summary_lines.append("")

    if tstr_results is not None:
        summary_lines.append("## TSTR vs TRTR")
        summary_lines.append("| Setting | Accuracy | AUROC | AUPRC | Brier | ECE |")
        summary_lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, row in tstr_results.iterrows():
            summary_lines.append(
                "| {setting} | {acc:.3f} | {auroc:.3f} | {auprc:.3f} | {brier:.3f} | {ece:.3f} |".format(
                    setting=row["setting"],
                    acc=row.get("accuracy", np.nan),
                    auroc=row.get("auroc", row.get("auc", np.nan)),
                    auprc=row.get("auprc", np.nan),
                    brier=row.get("brier", np.nan),
                    ece=row.get("ece", np.nan),
                )
            )
        summary_lines.append("")

    if distribution_df is not None and distribution_path is not None:
        summary_lines.append("## Distribution shift and privacy")
        distribution_top = distribution_df.sort_values("ks", ascending=False).head(10)
        summary_lines.append("Top 10 features by KS statistic:")
        summary_lines.append("| Feature | KS | MMD | Mutual information |")
        summary_lines.append("| --- | --- | --- | --- |")
        for _, row in distribution_top.iterrows():
            summary_lines.append(
                "| {feature} | {ks:.3f} | {mmd:.3f} | {mi:.3f} |".format(
                    feature=row["feature"],
                    ks=row.get("ks", np.nan),
                    mmd=row.get("mmd", np.nan),
                    mi=row.get("mutual_information", np.nan),
                )
            )
        summary_lines.append(
            f"Full distribution metrics: {distribution_path.relative_to(output_dir)}"
        )
    else:
        summary_lines.append("## Distribution shift and privacy")

    if not membership_df.empty:
        summary_lines.append("Membership inference results:")
        summary_lines.append(
            "| Target | Attack AUC | Best accuracy | Threshold | Majority baseline |"
        )
        summary_lines.append("| --- | --- | --- | --- | --- |")
        for _, row in membership_df.iterrows():
            summary_lines.append(
                "| {target} | {auc:.3f} | {best_acc:.3f} | {threshold:.3f} | {majority:.3f} |".format(
                    target=row["target"],
                    auc=row.get("attack_auc", np.nan),
                    best_acc=row.get("attack_best_accuracy", np.nan),
                    threshold=row.get("attack_best_threshold", np.nan),
                    majority=row.get("attack_majority_class_accuracy", np.nan),
                )
            )
        summary_lines.append(
            f"Membership metrics saved to: {membership_path.relative_to(output_dir)}"
        )
    else:
        summary_lines.append("No membership inference metrics were recorded.")

    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")


    print("Analysis complete.")
    print(f"Metric table saved to {metrics_path}")
    for target, path in calibration_paths.items():
        print(f"Calibration plot for {target}: {path}")
    for target, path in latent_paths.items():
        print(f"Latent space plot for {target}: {path}")
    print(f"Membership inference results saved to {membership_path}")
    if (
        in_hospital_model is not None
        and tstr_path is not None
        and distribution_path is not None
    ):
        print(f"TSTR/TRTR comparison saved to {tstr_path}")
        print(f"Distribution metrics saved to {distribution_path}")
    print(f"Summary written to {summary_path}")



if __name__ == "__main__":
    main()
