"""Unsupervised mortality modelling and analysis for the sepsis datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from suave import SUAVE, Schema, SchemaInferenceMode, SchemaInferencer  # noqa: E402
from suave.evaluate import (  # noqa: E402
    evaluate_tstr,
    evaluate_trtr,
    simple_membership_inference,
)

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
OUTPUT_DIR_NAME = "analysis_outputs_unsupervised"
HIDDEN_DIMENSION_OPTIONS: Dict[str, Tuple[int, int]] = {
    "compact": (128, 64),
    "balanced": (256, 128),
    "widened": (384, 192),
    "extended": (512, 256),
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


def compute_auc(probabilities: np.ndarray, targets: pd.Series | np.ndarray) -> float:
    """Return the ROC AUC given predicted probabilities and targets."""

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


def make_latent_classifier() -> Pipeline:
    """Return the logistic regression pipeline used on latent representations."""

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def make_logistic_pipeline() -> Pipeline:
    """Factory for the baseline classifier used in TSTR/TRTR."""

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    )


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
    """Perform Optuna hyperparameter optimisation for unsupervised :class:`SUAVE`."""

    if n_trials is not None and n_trials <= 0:
        n_trials = None
    if timeout is not None and timeout <= 0:
        timeout = None

    rng = np.random.default_rng(random_state)

    def objective(trial: "optuna.trial.Trial") -> float:
        latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32, 64, 128])
        hidden_key = trial.suggest_categorical(
            "hidden_dims", list(HIDDEN_DIMENSION_OPTIONS.keys())
        )
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 2e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
        beta = trial.suggest_float("beta", 0.25, 4.0)
        kl_warmup_epochs = trial.suggest_int("kl_warmup_epochs", 2, 25)
        warmup_epochs = trial.suggest_int("warmup_epochs", 10, 60)
        n_components = trial.suggest_int("n_components", 1, 4)
        tau_start = trial.suggest_float("tau_start", 0.5, 5.0)
        tau_min = trial.suggest_float("tau_min", 1e-4, 0.5, log=True)
        tau_decay = trial.suggest_float("tau_decay", 1e-4, 0.1, log=True)

        model = SUAVE(
            schema=schema,
            behaviour="unsupervised",
            latent_dim=latent_dim,
            hidden_dims=HIDDEN_DIMENSION_OPTIONS[hidden_key],
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            beta=beta,
            n_components=n_components,
            tau_start=tau_start,
            tau_min=tau_min,
            tau_decay=tau_decay,
            random_state=random_state,
            auto_parameters=False,
        )

        start_time = time.perf_counter()
        model.fit(
            X_train,
            warmup_epochs=warmup_epochs,
            kl_warmup_epochs=kl_warmup_epochs,
        )
        fit_seconds = time.perf_counter() - start_time

        latent_classifier = make_latent_classifier()
        train_latents = model.encode(X_train)
        val_latents = model.encode(X_validation)

        if train_latents.size == 0 or val_latents.size == 0:
            raise optuna.exceptions.TrialPruned("Empty latent representations")

        if np.unique(y_train).size < 2 or np.unique(y_validation).size < 2:
            raise optuna.exceptions.TrialPruned("Insufficient class diversity")

        latent_classifier.fit(train_latents, np.asarray(y_train))
        val_probs = latent_classifier.predict_proba(val_latents)
        val_auc = compute_auc(val_probs, y_validation)
        if not np.isfinite(val_auc):
            raise optuna.exceptions.TrialPruned("Non-finite validation AUC")

        numeric_train = to_numeric_frame(X_train)
        numeric_val = to_numeric_frame(X_validation)
        train_means = numeric_train.mean(axis=0)
        train_means = train_means.fillna(0.0)
        numeric_train = numeric_train.fillna(train_means)
        numeric_val = numeric_val.fillna(train_means)

        try:
            synthetic_features = model.sample(len(X_train))
        except Exception as exc:
            raise optuna.exceptions.TrialPruned(f"Sampling failed: {exc}") from exc
        if not isinstance(synthetic_features, pd.DataFrame):
            synthetic_features = pd.DataFrame(synthetic_features, columns=X_train.columns)
        synthetic_features = synthetic_features.reindex(columns=X_train.columns)
        numeric_synth = to_numeric_frame(synthetic_features).fillna(train_means)

        synthetic_latents = model.encode(synthetic_features)
        synth_probs = latent_classifier.predict_proba(synthetic_latents)
        if synth_probs.ndim == 1:
            positive_probs = synth_probs
        else:
            positive_probs = synth_probs[:, -1]
        if not np.all(np.isfinite(positive_probs)):
            raise optuna.exceptions.TrialPruned("Non-finite synthetic probabilities")
        rng_local = np.random.default_rng(rng.integers(0, 1_000_000))
        synthetic_labels = rng_local.binomial(1, np.clip(positive_probs, 1e-4, 1 - 1e-4))
        if np.unique(synthetic_labels).size < 2:
            raise optuna.exceptions.TrialPruned("Synthetic labels lacked class diversity")

        try:
            tstr_metrics = evaluate_tstr(
                (numeric_synth.to_numpy(), synthetic_labels),
                (numeric_val.to_numpy(), y_validation.to_numpy()),
                make_logistic_pipeline,
            )
            trtr_metrics = evaluate_trtr(
                (numeric_train.to_numpy(), y_train.to_numpy()),
                (numeric_val.to_numpy(), y_validation.to_numpy()),
                make_logistic_pipeline,
            )
        except ValueError as exc:
            raise optuna.exceptions.TrialPruned(f"Classification failed: {exc}") from exc

        tstr_auc = tstr_metrics.get("auroc")
        trtr_auc = trtr_metrics.get("auroc")
        if not (np.isfinite(tstr_auc) and np.isfinite(trtr_auc)):
            raise optuna.exceptions.TrialPruned("Non-finite TSTR/TRTR AUC")

        delta_auc = float(tstr_auc - trtr_auc)

        trial.set_user_attr("validation_auc", float(val_auc))
        trial.set_user_attr("fit_seconds", fit_seconds)
        trial.set_user_attr(
            "train_auc",
            compute_auc(latent_classifier.predict_proba(train_latents), y_train),
        )
        trial.set_user_attr("tstr_auc", float(tstr_auc))
        trial.set_user_attr("trtr_auc", float(trtr_auc))
        trial.set_user_attr("delta_auc", delta_auc)
        return delta_auc

    sampler = optuna.samplers.TPESampler(seed=rng.integers(0, 1_000_000))
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
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
        "validation_auc": study.best_trial.user_attrs.get("validation_auc"),
        "fit_seconds": study.best_trial.user_attrs.get("fit_seconds"),
        "tstr_auc": study.best_trial.user_attrs.get("tstr_auc"),
        "trtr_auc": study.best_trial.user_attrs.get("trtr_auc"),
        "delta_auc": study.best_trial.user_attrs.get("delta_auc", study.best_value),
    }
    return study, best_attributes


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


def format_float(value: Optional[float]) -> str:
    """Format floating point numbers for Markdown tables."""

    if value is None:
        return "nan"
    if isinstance(value, float) and not np.isfinite(value):
        return "nan"
    return f"{float(value):.3f}"


def parse_arguments() -> argparse.Namespace:
    """Return parsed command line arguments for the analysis script."""

    parser = argparse.ArgumentParser(
        description="Run the unsupervised SUAVE mortality analysis with Optuna tuning.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=60,
        help="Maximum number of Optuna trials per target (default: 60).",
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
    """Run the full unsupervised mortality analysis pipeline."""

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
    membership_records: List[Dict[str, object]] = []
    optuna_reports: Dict[str, Dict[str, object]] = {}

    latent_models: Dict[str, Pipeline] = {}
    suave_models: Dict[str, SUAVE] = {}

    tstr_results: Optional[pd.DataFrame] = None
    tstr_path: Optional[Path] = None
    distribution_df: Optional[pd.DataFrame] = None
    distribution_path: Optional[Path] = None

    for target in TARGET_COLUMNS:
        if target not in train_df.columns:
            continue
        print(f"Training unsupervised model for {target}…")
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

        X_train_model = pd.concat([X_train_model, X_calibration], ignore_index=True)
        y_train_model = pd.concat([y_train_model, y_calibration], ignore_index=True)

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
        hidden_key = str(best_params.get("hidden_dims", "balanced"))
        hidden_dims = HIDDEN_DIMENSION_OPTIONS.get(
            hidden_key, HIDDEN_DIMENSION_OPTIONS["balanced"]
        )
        model = SUAVE(
            schema=schema,
            behaviour="unsupervised",
            latent_dim=int(best_params.get("latent_dim", 32)),
            hidden_dims=hidden_dims,
            dropout=float(best_params.get("dropout", 0.1)),
            learning_rate=float(best_params.get("learning_rate", 1e-3)),
            batch_size=int(best_params.get("batch_size", 256)),
            beta=float(best_params.get("beta", 1.5)),
            n_components=int(best_params.get("n_components", 1)),
            tau_start=float(best_params.get("tau_start", 1.0)),
            tau_min=float(best_params.get("tau_min", 0.1)),
            tau_decay=float(best_params.get("tau_decay", 0.01)),
            random_state=RANDOM_STATE,
            auto_parameters=False,
        )
        model.fit(
            X_train_model,
            warmup_epochs=int(best_params.get("warmup_epochs", 30)),
            kl_warmup_epochs=int(best_params.get("kl_warmup_epochs", 10)),
        )
        suave_models[target] = model

        latent_classifier = make_latent_classifier()
        train_latents = model.encode(X_train_model)

        evaluation_datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
            "Train": (X_train_model, y_train_model),
            "Validation": (X_validation, y_validation),
            "MIMIC test": (
                prepare_features(test_df, feature_columns),
                test_df[target],
            ),
        }
        if target in external_df.columns:
            evaluation_datasets["eICU external"] = (
                prepare_features(external_df, feature_columns),
                external_df[target],
            )

        latent_classifier.fit(train_latents, np.asarray(y_train_model))
        latent_models[target] = latent_classifier

        for dataset_name, (features, labels) in evaluation_datasets.items():
            latents = model.encode(features)
            probs = latent_classifier.predict_proba(latents)
            auc = compute_auc(probs, labels)
            metrics_records.append(
                {
                    "target": target,
                    "dataset": dataset_name,
                    "auc": auc,
                }
            )

        train_probs = latent_classifier.predict_proba(train_latents)
        test_latents = model.encode(evaluation_datasets["MIMIC test"][0])
        test_probs = latent_classifier.predict_proba(test_latents)
        membership = simple_membership_inference(
            train_probs,
            np.asarray(y_train_model),
            test_probs,
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
            validation_auc = trial.user_attrs.get("validation_auc")
            if validation_auc is not None:
                record["validation_auc"] = validation_auc
            fit_seconds = trial.user_attrs.get("fit_seconds")
            if fit_seconds is not None:
                record["fit_seconds"] = fit_seconds
            train_auc = trial.user_attrs.get("train_auc")
            if train_auc is not None:
                record["train_auc"] = train_auc
            tstr_auc = trial.user_attrs.get("tstr_auc")
            if tstr_auc is not None:
                record["tstr_auc"] = tstr_auc
            trtr_auc = trial.user_attrs.get("trtr_auc")
            if trtr_auc is not None:
                record["trtr_auc"] = trtr_auc
            delta_auc = trial.user_attrs.get("delta_auc")
            if delta_auc is not None:
                record["delta_auc"] = delta_auc
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
            "metrics": {
                row["dataset"]: row["auc"]
                for row in metrics_records
                if row["target"] == target
            },
            "trials_csv": trials_path,
        }

    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = output_dir / "evaluation_metrics_unsupervised.csv"
    metrics_df.to_csv(metrics_path, index=False)

    membership_df = pd.DataFrame(membership_records)
    membership_path = output_dir / "membership_inference_unsupervised.csv"
    membership_df.to_csv(membership_path, index=False)

    primary_target = "in_hospital_mortality"
    if primary_target in suave_models and primary_target in latent_models:
        print("Generating synthetic data for TSTR/TRTR comparisons…")
        model = suave_models[primary_target]
        latent_classifier = latent_models[primary_target]

        X_train_full = prepare_features(train_df, feature_columns)
        y_train_full = train_df[primary_target]
        numeric_train = to_numeric_frame(X_train_full)
        train_means = numeric_train.mean(axis=0)
        train_means = train_means.fillna(0.0)
        numeric_train = numeric_train.fillna(train_means)

        synthetic_features = model.sample(len(X_train_full))
        synthetic_features = synthetic_features[feature_columns]
        numeric_synthetic = to_numeric_frame(synthetic_features)
        numeric_synthetic = numeric_synthetic.fillna(train_means)

        synthetic_latents = model.encode(synthetic_features)
        synthetic_probs = latent_classifier.predict_proba(synthetic_latents)[:, 1]
        rng = np.random.default_rng(RANDOM_STATE)
        synthetic_labels = rng.binomial(1, synthetic_probs)

        numeric_test = to_numeric_frame(prepare_features(test_df, feature_columns))
        numeric_test = numeric_test.fillna(train_means)
        y_test = test_df[primary_target]

        tstr_metrics = evaluate_tstr(
            (numeric_synthetic.to_numpy(), synthetic_labels),
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
        tstr_path = output_dir / "tstr_trtr_comparison_unsupervised.csv"
        tstr_results.to_csv(tstr_path, index=False)

        distribution_rows: List[Dict[str, object]] = []
        for column in feature_columns:
            real_values = numeric_train[column].to_numpy()
            synthetic_values = numeric_synthetic[column].to_numpy()
            distribution_rows.append(
                {
                    "feature": column,
                    "ks": kolmogorov_smirnov_statistic(real_values, synthetic_values),
                    "mmd": rbf_mmd(
                        real_values, synthetic_values, random_state=RANDOM_STATE
                    ),
                    "mutual_information": mutual_information_feature(
                        real_values, synthetic_values
                    ),
                }
            )
        distribution_df = pd.DataFrame(distribution_rows)
        distribution_path = output_dir / "distribution_shift_metrics_unsupervised.csv"
        distribution_df.to_csv(distribution_path, index=False)

    summary_lines: List[str] = [
        "# Unsupervised mortality modelling report",
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
        metrics_map: Mapping[str, float] = report["metrics"]
        summary_lines.append(f"### {target}")
        best_value = best.get("value")
        value_text = (
            f"{best_value:.4f}" if isinstance(best_value, (int, float)) else "n/a"
        )
        summary_lines.append(
            f"Best Optuna trial #{best.get('trial_number')} with delta AUC (TSTR-TRTR) {value_text}"
        )
        summary_lines.append("Best parameters:")
        summary_lines.append("```json")
        summary_lines.append(json.dumps(best_params, indent=2, ensure_ascii=False))
        summary_lines.append("```")
        if best.get("tstr_auc") is not None and best.get("trtr_auc") is not None:
            summary_lines.append(
                "TSTR AUC: {tstr} | TRTR AUC: {trtr} | Delta: {delta}".format(
                    tstr=format_float(best.get("tstr_auc")),
                    trtr=format_float(best.get("trtr_auc")),
                    delta=format_float(best.get("delta_auc")),
                )
            )
        summary_lines.append("| Dataset | AUC |")
        summary_lines.append("| --- | --- |")
        for dataset_name in [
            "Train",
            "Validation",
            "MIMIC test",
            "eICU external",
        ]:
            if dataset_name not in metrics_map:
                continue
            summary_lines.append(
                "| {dataset} | {auc} |".format(
                    dataset=dataset_name,
                    auc=format_float(metrics_map.get(dataset_name)),
                )
            )
        summary_lines.append(
            f"Optuna trials logged at: {report['trials_csv'].relative_to(output_dir)}"
        )
        summary_lines.append("")

    if tstr_results is not None:
        summary_lines.append("## TSTR vs TRTR")
        summary_lines.append("| Setting | Accuracy | AUC | AUPRC | Brier | ECE |")
        summary_lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, row in tstr_results.iterrows():
            summary_lines.append(
                "| {setting} | {acc:.3f} | {auc:.3f} | {auprc:.3f} | {brier:.3f} | {ece:.3f} |".format(
                    setting=row["setting"],
                    acc=row.get("accuracy", np.nan),
                    auc=row.get("auroc", np.nan),
                    auprc=row.get("auprc", np.nan),
                    brier=row.get("brier", np.nan),
                    ece=row.get("ece", np.nan),
                )
            )
        summary_lines.append("")

    summary_lines.append("## Distribution shift and privacy")
    if distribution_df is not None and distribution_path is not None:
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
        summary_lines.append("Distribution metrics were not computed.")

    if not membership_df.empty:
        summary_lines.append("Membership inference results:")
        summary_lines.append(
            "| Target | attack_auc | attack_accuracy | attack_threshold |"
        )
        summary_lines.append("| --- | --- | --- | --- |")
        for _, row in membership_df.iterrows():
            summary_lines.append(
                "| {target} | {auc:.3f} | {accuracy:.3f} | {threshold:.3f} |".format(
                    target=row["target"],
                    auc=row.get("attack_auc", np.nan),
                    accuracy=row.get("attack_best_accuracy", np.nan),
                    threshold=row.get("attack_best_threshold", np.nan),
                )
            )
        summary_lines.append(
            f"Membership metrics saved to: {membership_path.relative_to(output_dir)}"
        )
    else:
        summary_lines.append("No membership inference metrics were recorded.")

    summary_path = output_dir / "summary_unsupervised.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("Analysis complete.")
    print(f"Metric table saved to {metrics_path}")
    print(f"Membership inference results saved to {membership_path}")
    if tstr_path is not None and distribution_path is not None:
        print(f"TSTR/TRTR comparison saved to {tstr_path}")
        print(f"Distribution metrics saved to {distribution_path}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
