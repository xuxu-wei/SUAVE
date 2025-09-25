# %% [markdown]
# # MIMIC mortality (evaluation)
#
# This notebook-style script loads the artefacts produced by the Optuna
# optimisation pipeline, ensures a calibrated SUAVE model is available, and runs
# the downstream evaluation suite (classical baselines, prognosis metrics,
# synthetic-vs-real analysis, and reporting).

# %%

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

EXAMPLES_DIR = Path(__file__).resolve().parent
if not EXAMPLES_DIR.exists():
    raise RuntimeError(
        "Run this notebook from the repository root so 'examples' is available."
    )
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from mimic_mortality_utils import (  # noqa: E402
    RANDOM_STATE,
    TARGET_COLUMNS,
    BENCHMARK_COLUMNS,
    VALIDATION_SIZE,
    build_prediction_dataframe,
    build_suave_model,
    build_tstr_training_sets,
    compute_binary_metrics,
    dataframe_to_markdown,
    define_schema,
    evaluate_transfer_baselines,
    extract_positive_probabilities,
    fit_isotonic_calibrator,
    load_dataset,
    load_or_create_iteratively_imputed_features,
    make_baseline_model_factories,
    plot_benchmark_curves,
    plot_calibration_curves,
    plot_latent_space,
    plot_transfer_metric_bars,
    prepare_features,
    render_dataframe,
    schema_to_dataframe,
    to_numeric_frame,
)
from cls_eval import evaluate_predictions, write_results_to_excel_unique  # noqa: E402

from suave.evaluate import (  # noqa: E402
    kolmogorov_smirnov_statistic,
    mutual_information_feature,
    rbf_mmd,
)

from suave import SUAVE  # noqa: E402
from suave.evaluate import simple_membership_inference  # noqa: E402


# %% [markdown]
# ## Analysis configuration
#
# Define the label of interest and locations for cached outputs from the
# optimisation script.

# %%

TARGET_LABEL = "in_hospital_mortality"

analysis_config = {
    "optuna_trials": 60,
    "optuna_timeout": 3600 * 48,
    "optuna_study_prefix": "supervised",
    "optuna_storage": None,
    "output_dir_name": "analysis_outputs_supervised",
}


# %% [markdown]
# ## Data loading and schema definition
#
# Load train/test/external splits, construct the schema, and validate the
# requested target label. Schema corrections are added here so that the
# downstream modelling code receives explicit type information.

# %%

DATA_DIR = (EXAMPLES_DIR / "data" / "sepsis_mortality_dataset").resolve()
OUTPUT_DIR = EXAMPLES_DIR / analysis_config["output_dir_name"]
OUTPUT_DIR.mkdir(exist_ok=True)
analysis_config["optuna_storage"] = (
    f"sqlite:///{OUTPUT_DIR}/{analysis_config['optuna_study_prefix']}_optuna.db"
)

train_df = load_dataset(DATA_DIR / "mimic-mortality-train.tsv")
test_df = load_dataset(DATA_DIR / "mimic-mortality-test.tsv")
external_df = load_dataset(DATA_DIR / "eicu-mortality-external_val.tsv")

if TARGET_LABEL not in TARGET_COLUMNS:
    raise ValueError(
        f"Target label '{TARGET_LABEL}' is not one of the configured targets: {TARGET_COLUMNS}"
    )

FEATURE_COLUMNS = [
    column
    for column in train_df.columns
    if column not in TARGET_COLUMNS + BENCHMARK_COLUMNS
]
schema = define_schema(train_df, FEATURE_COLUMNS, mode="interactive")

# Manual schema corrections ensure columns with ambiguous types are treated
# appropriately during modelling.
schema.update(
    {
        "BMI": {"type": "real"},
        "Respiratory_Support": {"type": "ordinal", "n_classes": 5},
        "LYM%": {"type": "real"},
    }
)

schema_df = schema_to_dataframe(schema).sort_values("Column").reset_index(drop=True)
render_dataframe(schema_df, title="Schema overview", floatfmt=None)


# %%


def make_study_name(prefix: Optional[str], target_label: str) -> Optional[str]:
    """Return the Optuna study name for ``target_label`` given ``prefix``."""

    if not prefix:
        return None
    return f"{prefix}_{target_label}"


def load_optuna_results(
    output_dir: Path,
    target_label: str,
    *,
    study_prefix: Optional[str],
    storage: Optional[str],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load the best Optuna trial metadata and parameters if available."""

    best_info_path = output_dir / f"optuna_best_info_{target_label}.json"
    best_params_path = output_dir / f"optuna_best_params_{target_label}.json"

    best_info: Dict[str, Any] = (
        json.loads(best_info_path.read_text()) if best_info_path.exists() else {}
    )
    best_params: Dict[str, Any] = (
        json.loads(best_params_path.read_text()) if best_params_path.exists() else {}
    )

    study_name = make_study_name(study_prefix, target_label)
    if (not best_info or not best_params) and storage and study_name:
        try:
            import optuna  # type: ignore
        except ImportError:
            optuna = None  # type: ignore
        if optuna is not None:  # pragma: no cover - optuna available in examples env
            study = optuna.load_study(study_name=study_name, storage=storage)
            feasible_trials = [
                trial for trial in study.trials if trial.values is not None
            ]
            if feasible_trials:

                def sort_key(trial: "optuna.trial.FrozenTrial") -> Tuple[float, float]:
                    values = trial.values or (float("nan"), float("inf"))
                    primary = values[0]
                    secondary = values[1]
                    return (
                        primary,
                        -secondary if np.isfinite(secondary) else float("-inf"),
                    )

                best_trial = max(feasible_trials, key=sort_key)
                best_info = {
                    "trial_number": best_trial.number,
                    "values": tuple(best_trial.values or ()),
                    "params": dict(best_trial.params),
                    "validation_metrics": best_trial.user_attrs.get(
                        "validation_metrics", {}
                    ),
                    "fit_seconds": best_trial.user_attrs.get("fit_seconds"),
                    "tstr_metrics": best_trial.user_attrs.get("tstr_metrics", {}),
                    "trtr_metrics": best_trial.user_attrs.get("trtr_metrics", {}),
                    "tstr_trtr_delta_auc": best_trial.user_attrs.get(
                        "tstr_trtr_delta_auc"
                    ),
                }
                best_params = dict(best_trial.params)
    if not best_info:
        best_info = {}
    if not best_params and isinstance(best_info.get("params"), Mapping):
        best_params = dict(best_info["params"])
    return best_info, best_params


def resolve_suave_fit_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Return fit-time keyword arguments derived from Optuna parameters."""

    return {
        "warmup_epochs": int(params.get("warmup_epochs", 3)),
        "kl_warmup_epochs": int(params.get("kl_warmup_epochs", 0)),
        "head_epochs": int(params.get("head_epochs", 2)),
        "finetune_epochs": int(params.get("finetune_epochs", 2)),
        "joint_decoder_lr_scale": float(params.get("joint_decoder_lr_scale", 0.1)),
        "early_stop_patience": int(params.get("early_stop_patience", 10)),
    }


def extract_calibrator_estimator(calibrator: Any) -> Optional[SUAVE]:
    """Return the underlying SUAVE estimator from ``calibrator`` if present."""

    if calibrator is None:
        return None
    candidate = getattr(calibrator, "base_estimator", None)
    if isinstance(candidate, SUAVE):
        return candidate
    candidate = getattr(calibrator, "estimator", None)
    if isinstance(candidate, SUAVE):
        return candidate
    calibrated_list = getattr(calibrator, "calibrated_classifiers_", None)
    if calibrated_list:
        base_est = getattr(calibrated_list[0], "base_estimator", None)
        if isinstance(base_est, SUAVE):
            return base_est
    return None


class _TSTRSuaveEstimator:
    """Wrapper exposing ``SUAVE`` with a scikit-learn-style interface."""

    def __init__(self, base_model: SUAVE, fit_kwargs: Mapping[str, Any]):
        self._model = base_model
        self._fit_kwargs = dict(fit_kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_TSTRSuaveEstimator":
        self._model.fit(X, y, **self._fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def classes_(self) -> Optional[np.ndarray]:
        return getattr(self._model, "classes_", None)


# %% [markdown]
# ## Prepare modelling datasets
#
# Split the training cohort into train and validation folds for the selected
# label. The validation fold later supports calibration if a saved model is
# unavailable.

# %%

X_full = prepare_features(train_df, FEATURE_COLUMNS)
y_full = train_df[TARGET_LABEL]

X_train_model, X_validation, y_train_model, y_validation = train_test_split(
    X_full,
    y_full,
    test_size=VALIDATION_SIZE,
    stratify=y_full,
    random_state=RANDOM_STATE,
)

X_train_model = X_train_model.reset_index(drop=True)
X_validation = X_validation.reset_index(drop=True)
y_train_model = y_train_model.reset_index(drop=True)
y_validation = y_validation.reset_index(drop=True)

# Prepare holdout datasets once to reuse across later cells.
X_test = prepare_features(test_df, FEATURE_COLUMNS)
y_test = test_df[TARGET_LABEL]

external_features: Optional[pd.DataFrame]
external_labels: Optional[pd.Series]
if TARGET_LABEL in external_df.columns:
    external_features = prepare_features(external_df, FEATURE_COLUMNS)
    external_labels = external_df[TARGET_LABEL]
else:
    external_features = None
    external_labels = None


# %% [markdown]
# ## Classical model benchmarks
#
# Fit a suite of scikit-learn classifiers as quick baselines before evaluating
# SUAVE. These provide a reference point for MIMIC test performance and, when
# available, eICU external validation.

# %%

baseline_feature_frames: Dict[str, pd.DataFrame] = {
    "Train": X_train_model,
    "MIMIC test": X_test,
}
if external_features is not None:
    baseline_feature_frames["eICU external"] = external_features

(
    baseline_imputed_features,
    baseline_imputed_paths,
    baseline_loaded_from_cache,
) = load_or_create_iteratively_imputed_features(
    baseline_feature_frames,
    output_dir=OUTPUT_DIR,
    target_label=TARGET_LABEL,
    reference_key="Train",
)

if baseline_loaded_from_cache:
    print("Loaded iterative-imputed baseline features from disk.")
else:
    print("Saved iterative-imputed baseline features:")
    for name, path in baseline_imputed_paths.items():
        print(f"  - {name}: {path}")

baseline_models = {
    "Logistic regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
            ),
        ]
    ),
    "KNN": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=25)),
        ]
    ),
    "Decision tree": Pipeline(
        [
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    ),
    "Random forest": Pipeline(
        [
            (
                "classifier",
                RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
            ),
        ]
    ),
    "SVM (RBF)": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True)),
        ]
    ),
}

baseline_label_sets: Dict[str, pd.Series] = {
    "Train": y_train_model,
    "MIMIC test": y_test,
}
if external_labels is not None:
    baseline_label_sets["eICU external"] = external_labels

baseline_evaluation_sets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
    name: (baseline_imputed_features[name], labels)
    for name, labels in baseline_label_sets.items()
}

baseline_rows: List[Dict[str, object]] = []
metric_columns = ["AUC", "ACC", "SPE", "SEN", "Brier"]
baseline_probability_map: Dict[str, Dict[str, np.ndarray]] = {
    name: {} for name in baseline_evaluation_sets.keys()
}
model_abbreviation_lookup = {
    "Logistic regression": "LR",
    "KNN": "KNN",
    "Decision tree": "DT",
    "Random forest": "RF",
    "SVM (RBF)": "SVM",
}
model_abbreviation_lookup["SUAVE"] = "SUAVE"

train_features_imputed = baseline_imputed_features["Train"]
train_labels = baseline_label_sets["Train"]

for model_name, estimator in baseline_models.items():
    fitted_estimator = estimator.fit(train_features_imputed, train_labels)
    for dataset_name, (features, labels) in baseline_evaluation_sets.items():
        probabilities = fitted_estimator.predict_proba(features)
        baseline_probability_map[dataset_name][model_name] = (
            extract_positive_probabilities(probabilities)
        )
        metrics = compute_binary_metrics(probabilities, labels)
        row = {
            "Model": model_name,
            "Dataset": dataset_name,
            "Notes": "",
        }
        row.update(
            {column: metrics.get(column, float("nan")) for column in metric_columns}
        )
        baseline_rows.append(row)

    if "eICU external" not in baseline_evaluation_sets:
        baseline_rows.append(
            {
                "Model": model_name,
                "Dataset": "eICU external",
                "Notes": "Target not available in eICU split.",
                **{column: float("nan") for column in metric_columns},
            }
        )

baseline_df = pd.DataFrame(baseline_rows)
baseline_order = ["Model", "Dataset", *metric_columns, "Notes"]
baseline_df = baseline_df.loc[:, baseline_order]
baseline_path = OUTPUT_DIR / f"baseline_models_{TARGET_LABEL}.csv"
baseline_df.to_csv(baseline_path, index=False)
render_dataframe(
    baseline_df,
    title=f"Classical baseline performance for {TARGET_LABEL}",
    floatfmt=".3f",
)


# %% [markdown]
# ## Load optimisation artefacts
#
# Retrieve the best Optuna trial information saved by the optimisation script.

# %%

optuna_best_info, optuna_best_params = load_optuna_results(
    OUTPUT_DIR,
    TARGET_LABEL,
    study_prefix=analysis_config.get("optuna_study_prefix"),
    storage=analysis_config.get("optuna_storage"),
)
optuna_trials_path = OUTPUT_DIR / f"optuna_trials_{TARGET_LABEL}.csv"

if not optuna_best_params:
    print(
        "Optuna best parameters were not found on disk; subsequent steps will "
        "rely on defaults unless the storage backend is available."
    )


# %% [markdown]
# ## Ensure calibrated SUAVE model
#
# Load the trained SUAVE model and isotonic calibrator if they were saved by the
# optimisation pipeline. When the artefacts are unavailable, retrain the model
# using the best Optuna parameters and calibrate on the validation split.

# %%

model_path = OUTPUT_DIR / f"suave_best_{TARGET_LABEL}.pt"
calibrator_path = OUTPUT_DIR / f"isotonic_calibrator_{TARGET_LABEL}.joblib"

model: Optional[SUAVE] = None
calibrator: Optional[Any] = None

if calibrator_path.exists():
    calibrator = joblib.load(calibrator_path)
    model = extract_calibrator_estimator(calibrator)
    if model is not None:
        print(
            f"Loaded isotonic calibrator and embedded SUAVE model from {calibrator_path}."
        )

if model is None and model_path.exists():
    model = SUAVE.load(model_path)
    print(f"Loaded SUAVE model from {model_path}.")

if model is None and optuna_best_params:
    print(
        "Training SUAVE with best Optuna parameters because no saved model was found…"
    )
    model = build_suave_model(optuna_best_params, schema, random_state=RANDOM_STATE)
    fit_kwargs = resolve_suave_fit_kwargs(optuna_best_params)
    model.fit(
        X_train_model,
        y_train_model,
        **fit_kwargs,
    )
else:
    if model is None:
        raise RuntimeError(
            "Unable to load or reconstruct the SUAVE model. Ensure the optimisation "
            "script has been executed to produce the necessary artefacts."
        )

if calibrator is None:
    calibrator = fit_isotonic_calibrator(model, X_validation, y_validation)
    print("Fitted a new isotonic calibrator on the validation split.")
else:
    embedded = extract_calibrator_estimator(calibrator)
    if embedded is None:
        print(
            "Calibrator did not contain a usable SUAVE estimator; refitting calibrator."
        )
        calibrator = fit_isotonic_calibrator(model, X_validation, y_validation)
    else:
        model = embedded


# %% [markdown]
# ## Prognosis prediction and evaluation
#
# Evaluate the trained model on train/validation/test/eICU cohorts, generate
# calibration curves, and run a membership inference baseline.

# %%

evaluation_datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
    "Train": (X_train_model, y_train_model),
    "Validation": (X_validation, y_validation),
    "MIMIC test": (X_test, y_test),
}
if external_features is not None and external_labels is not None:
    evaluation_datasets["eICU external"] = (external_features, external_labels)

probability_map: Dict[str, np.ndarray] = {}
label_map: Dict[str, np.ndarray] = {}
metrics_rows: List[Dict[str, object]] = []

for dataset_name, (features, labels) in evaluation_datasets.items():
    if calibrator is not None:
        probs = calibrator.predict_proba(features)
    else:
        probs = model.predict_proba(features)
    probability_map[dataset_name] = probs
    label_map[dataset_name] = np.asarray(labels)
    metrics = compute_binary_metrics(probs, labels)
    metrics_rows.append({"target": TARGET_LABEL, "dataset": dataset_name, **metrics})

metrics_df = pd.DataFrame(metrics_rows)
ordered_metric_columns = [
    "target",
    "dataset",
    "AUC",
    "ACC",
    "SPE",
    "SEN",
    "Brier",
]
existing_columns = [
    column for column in ordered_metric_columns if column in metrics_df.columns
]
if existing_columns:
    metrics_df = metrics_df.loc[:, existing_columns]
metrics_path = OUTPUT_DIR / "evaluation_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
render_dataframe(
    metrics_df,
    title=f"Evaluation metrics for {TARGET_LABEL}",
    floatfmt=".3f",
)

calibration_path = OUTPUT_DIR / f"calibration_{TARGET_LABEL}.png"
plot_calibration_curves(
    probability_map, label_map, target_name=TARGET_LABEL, output_path=calibration_path
)

train_probabilities = probability_map["Train"]
test_probabilities = probability_map["MIMIC test"]
membership_metrics = simple_membership_inference(
    train_probabilities,
    np.asarray(y_train_model),
    test_probabilities,
    np.asarray(y_test),
)
membership_df = pd.DataFrame([{"target": TARGET_LABEL, **membership_metrics}])
membership_path = OUTPUT_DIR / "membership_inference.csv"
membership_df.to_csv(membership_path, index=False)
render_dataframe(
    membership_df,
    title="Membership inference baseline",
    floatfmt=".3f",
)


# %% [markdown]
# ## Benchmark ROC and calibration curves
#
# Visualise the discriminative and calibration performance of the classical
# baselines alongside SUAVE across the train, test, and external cohorts.

# %%

benchmark_datasets = ["Train", "MIMIC test", "eICU external"]
benchmark_curve_paths: List[Path] = []

for dataset_name in benchmark_datasets:
    if dataset_name not in label_map:
        print(f"Skipping {dataset_name} because ground-truth labels are unavailable.")
        continue

    model_probabilities: Dict[str, np.ndarray] = {}
    suave_probs = extract_positive_probabilities(probability_map[dataset_name])
    model_probabilities["SUAVE"] = suave_probs

    for baseline_name, baseline_probs in baseline_probability_map.get(
        dataset_name, {}
    ).items():
        model_probabilities[baseline_name] = baseline_probs

    if not model_probabilities:
        print(f"No model probabilities available for {dataset_name}.")
        continue

    figure_path = plot_benchmark_curves(
        dataset_name,
        label_map[dataset_name],
        model_probabilities,
        output_dir=OUTPUT_DIR,
        target_label=TARGET_LABEL,
        abbreviation_lookup=model_abbreviation_lookup,
    )
    if figure_path is not None:
        benchmark_curve_paths.append(figure_path)


# %% [markdown]
# ## Bootstrap benchmarking
#
# Derive confidence intervals for each cohort using the reusable classification
# evaluation helpers. The exported artefacts include per-dataset Excel sheets,
# summary tables, and optional warnings when probability columns require
# renormalisation.

# %%

bootstrap_results: Dict[str, Dict[str, pd.DataFrame]] = {}
bootstrap_overall_frames: List[pd.DataFrame] = []
bootstrap_per_class_frames: List[pd.DataFrame] = []
bootstrap_warnings_frames: List[pd.DataFrame] = []

model_classes_array = getattr(model, "_classes", None)
if model_classes_array is None or len(model_classes_array) == 0:
    model_classes_array = np.unique(np.asarray(y_train_model))
class_value_list = list(model_classes_array)
class_name_strings = [str(value) for value in class_value_list]
positive_label_name = class_name_strings[-1] if len(class_name_strings) == 2 else None
for dataset_name, (features, labels) in evaluation_datasets.items():
    dataset_probabilities = probability_map[dataset_name]
    positive_probs = extract_positive_probabilities(dataset_probabilities)
    if len(class_value_list) == 2:
        negative_label = class_value_list[0]
        positive_label = class_value_list[-1]
        dataset_predictions = np.where(
            positive_probs >= 0.5, positive_label, negative_label
        )
    else:
        dataset_predictions = model.predict(features)
    prediction_df = build_prediction_dataframe(
        dataset_probabilities,
        labels,
        dataset_predictions,
        class_name_strings,
    )

    results = evaluate_predictions(
        prediction_df,
        label_col="label",
        pred_col="y_pred",
        positive_label=positive_label_name,
        bootstrap_n=1000,
        random_state=RANDOM_STATE,
    )
    bootstrap_results[dataset_name] = results

    overall_df = results["overall"].copy()
    overall_df.insert(0, "Dataset", dataset_name)
    overall_df.insert(0, "Target", TARGET_LABEL)
    bootstrap_overall_frames.append(overall_df)

    per_class_df = results["per_class"].copy()
    per_class_df.insert(0, "Dataset", dataset_name)
    per_class_df.insert(0, "Target", TARGET_LABEL)
    bootstrap_per_class_frames.append(per_class_df)

    warnings_df = results.get("warnings")
    if warnings_df is not None and not warnings_df.empty:
        warnings_copy = warnings_df.copy()
        warnings_copy.insert(0, "Dataset", dataset_name)
        warnings_copy.insert(0, "Target", TARGET_LABEL)
        bootstrap_warnings_frames.append(warnings_copy)


bootstrap_overall_df = pd.concat(bootstrap_overall_frames, ignore_index=True)
bootstrap_per_class_df = pd.concat(bootstrap_per_class_frames, ignore_index=True)
bootstrap_overall_path = OUTPUT_DIR / f"bootstrap_overall_{TARGET_LABEL}.csv"
bootstrap_per_class_path = OUTPUT_DIR / f"bootstrap_per_class_{TARGET_LABEL}.csv"
bootstrap_overall_df.to_csv(bootstrap_overall_path, index=False)
bootstrap_per_class_df.to_csv(bootstrap_per_class_path, index=False)

bootstrap_warning_path: Optional[Path]
if bootstrap_warnings_frames:
    bootstrap_warning_df = pd.concat(bootstrap_warnings_frames, ignore_index=True)
    bootstrap_warning_path = OUTPUT_DIR / f"bootstrap_warnings_{TARGET_LABEL}.csv"
    bootstrap_warning_df.to_csv(bootstrap_warning_path, index=False)
else:
    bootstrap_warning_path = None

bootstrap_excel_path = OUTPUT_DIR / f"bootstrap_{TARGET_LABEL}.xlsx"
write_results_to_excel_unique(
    bootstrap_results,
    str(bootstrap_excel_path),
    include_warnings_sheet=True,
)

summary_metric_candidates = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "recall_macro",
    "specificity_macro",
    "sensitivity_pos",
    "specificity_pos",
    "roc_auc",
    "pr_auc",
]
summary_columns: List[str] = ["Target", "Dataset"]
for metric_name in summary_metric_candidates:
    if metric_name in bootstrap_overall_df.columns:
        summary_columns.append(metric_name)
        low_col = f"{metric_name}_ci_low"
        high_col = f"{metric_name}_ci_high"
        if low_col in bootstrap_overall_df.columns:
            summary_columns.append(low_col)
        if high_col in bootstrap_overall_df.columns:
            summary_columns.append(high_col)

bootstrap_summary_df = bootstrap_overall_df.loc[:, summary_columns]
bootstrap_summary_path = OUTPUT_DIR / f"bootstrap_summary_{TARGET_LABEL}.csv"
bootstrap_summary_df.to_csv(bootstrap_summary_path, index=False)
render_dataframe(
    bootstrap_summary_df,
    title=f"Bootstrap performance with confidence intervals for {TARGET_LABEL}",
    floatfmt=".3f",
)


# %% [markdown]
# ## TSTR/TRTR comparison
#
# Compare models trained on synthetic versus real data. The analysis is only
# relevant when the current task models in-hospital mortality, matching the
# publication results.

# %%

tstr_summary_df: Optional[pd.DataFrame] = None
tstr_plot_df: Optional[pd.DataFrame] = None
tstr_summary_path: Optional[Path] = None
tstr_plot_path: Optional[Path] = None
tstr_figure_paths: List[Path] = []
distribution_df: Optional[pd.DataFrame] = None
distribution_path: Optional[Path] = None
distribution_top: Optional[pd.DataFrame] = None

if TARGET_LABEL != "in_hospital_mortality":
    print(
        "Skipping TSTR/TRTR comparison because it is defined for the in-hospital "
        "mortality task."
    )
else:
    print("Generating synthetic data for TSTR/TRTR comparisons…")
    training_sets = build_tstr_training_sets(
        model,
        FEATURE_COLUMNS,
        X_full,
        y_full,
        random_state=RANDOM_STATE,
    )
    evaluation_sets_numeric: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
        "MIMIC test": (to_numeric_frame(X_test), y_test.reset_index(drop=True)),
    }
    if external_features is not None and external_labels is not None:
        evaluation_sets_numeric["eICU external"] = (
            to_numeric_frame(external_features),
            external_labels.reset_index(drop=True),
        )

    model_factories = dict(make_baseline_model_factories(RANDOM_STATE))
    if optuna_best_params:
        suave_fit_kwargs = resolve_suave_fit_kwargs(optuna_best_params)

        def make_suave_transfer_estimator() -> _TSTRSuaveEstimator:
            base_model = build_suave_model(
                optuna_best_params,
                schema,
                random_state=RANDOM_STATE,
            )
            return _TSTRSuaveEstimator(base_model, suave_fit_kwargs)

        model_factories["SUAVE (Optuna best)"] = make_suave_transfer_estimator
    else:
        print(
            "Skipping SUAVE TSTR/TRTR baseline because no Optuna parameters are available."
        )
    (
        tstr_summary_df,
        tstr_plot_df,
        _,
    ) = evaluate_transfer_baselines(
        training_sets,
        evaluation_sets_numeric,
        model_factories=model_factories,
        bootstrap_n=1000,
        random_state=RANDOM_STATE,
    )
    tstr_summary_path = OUTPUT_DIR / f"tstr_trtr_summary_{TARGET_LABEL}.csv"
    tstr_plot_path = OUTPUT_DIR / f"tstr_trtr_plot_data_{TARGET_LABEL}.csv"
    tstr_summary_df.to_csv(tstr_summary_path, index=False)
    tstr_plot_df.to_csv(tstr_plot_path, index=False)
    render_dataframe(
        tstr_summary_df,
        title="TSTR/TRTR supervised evaluation",
        floatfmt=".3f",
    )

    training_order = list(training_sets.keys())
    model_order = list(model_factories.keys())
    for evaluation_name in evaluation_sets_numeric.keys():
        for metric_name in ("accuracy", "roc_auc"):
            figure_path = plot_transfer_metric_bars(
                tstr_plot_df,
                metric=metric_name,
                evaluation_dataset=evaluation_name,
                training_order=training_order,
                model_order=model_order,
                output_dir=OUTPUT_DIR,
                target_label=TARGET_LABEL,
            )
            if figure_path is not None:
                tstr_figure_paths.append(figure_path)

    real_features_numeric = training_sets["TRTR (real)"][0]
    synthesis_features_numeric = training_sets["TSTR synthesis"][0]
    distribution_rows: List[Dict[str, object]] = []
    for column in FEATURE_COLUMNS:
        real_values = real_features_numeric[column].to_numpy()
        synthetic_values = synthesis_features_numeric[column].to_numpy()
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
    distribution_path = OUTPUT_DIR / "distribution_shift_metrics.csv"
    distribution_df.to_csv(distribution_path, index=False)
    distribution_top = (
        distribution_df.sort_values("ks", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    render_dataframe(
        distribution_top,
        title="Top distribution shift features (KS)",
        floatfmt=".3f",
    )


# %% [markdown]
# ## Latent space interpretation
#
# Project latent representations using PCA for qualitative assessment of class
# separation across cohorts.

# %%

latent_features = {
    name: features for name, (features, _) in evaluation_datasets.items()
}
latent_labels = {name: labels for name, (_, labels) in evaluation_datasets.items()}
latent_path = OUTPUT_DIR / f"latent_{TARGET_LABEL}.png"
plot_latent_space(
    model,
    latent_features,
    latent_labels,
    target_name=TARGET_LABEL,
    output_path=latent_path,
)


# %% [markdown]
# ## Reporting
#
# Collate metrics, Optuna summary, and artifact locations into a Markdown
# summary mirroring the original analysis output.

# %%

summary_lines: List[str] = [
    "# Mortality modelling report",
    "",
    "## Schema",
    dataframe_to_markdown(schema_df, floatfmt=None),
    "",
    "## Model selection and performance",
    f"### {TARGET_LABEL}",
]

best_values = optuna_best_info.get("values") if optuna_best_info else None
if isinstance(best_values, (list, tuple)) and best_values:
    best_roauc = best_values[0]
    roauc_text = f"{best_roauc:.4f}" if np.isfinite(best_roauc) else "n/a"
    delta_text: Optional[str]
    if len(best_values) > 1 and np.isfinite(best_values[1]):
        delta_text = f" (ΔAUC {best_values[1]:.4f})"
    else:
        delta_text = None
else:
    roauc_text = "n/a"
    delta_text = None

if optuna_best_info:
    summary_line = (
        f"Best Optuna trial #{optuna_best_info.get('trial_number')} "
        f"with validation ROAUC {roauc_text}"
    )
    if delta_text:
        summary_line += delta_text
    summary_lines.append(summary_line)
else:
    summary_lines.append("Best Optuna trial: information unavailable.")

summary_lines.append("Best parameters:")
summary_lines.append("```json")
summary_lines.append(json.dumps(optuna_best_params, indent=2, ensure_ascii=False))
summary_lines.append("```")

metrics_summary_df = metrics_df.rename(
    columns={"target": "Target", "dataset": "Dataset"}
)
metric_column_order = [
    "Target",
    "Dataset",
    "AUC",
    "ACC",
    "SPE",
    "SEN",
    "Brier",
]
existing_metric_columns = [
    column for column in metric_column_order if column in metrics_summary_df.columns
]
if existing_metric_columns:
    metrics_summary_df = metrics_summary_df.loc[:, existing_metric_columns]
summary_lines.append(dataframe_to_markdown(metrics_summary_df, floatfmt=".3f"))
summary_lines.append(
    f"Optuna trials logged at: {optuna_trials_path.relative_to(OUTPUT_DIR)}"
)
summary_lines.append(f"Calibration plot: {calibration_path.relative_to(OUTPUT_DIR)}")
summary_lines.append(f"Latent projection: {latent_path.relative_to(OUTPUT_DIR)}")
summary_lines.append("")

summary_lines.append("Bootstrap evaluation artefacts:")
summary_lines.append(
    f"- Summary table: {bootstrap_summary_path.relative_to(OUTPUT_DIR)}"
)
summary_lines.append(
    f"- Overall metrics: {bootstrap_overall_path.relative_to(OUTPUT_DIR)}"
)
summary_lines.append(
    f"- Per-class metrics: {bootstrap_per_class_path.relative_to(OUTPUT_DIR)}"
)
summary_lines.append(
    f"- Excel workbook: {bootstrap_excel_path.relative_to(OUTPUT_DIR)}"
)
if bootstrap_warning_path is not None:
    summary_lines.append(
        f"- Warnings: {bootstrap_warning_path.relative_to(OUTPUT_DIR)}"
    )
summary_lines.append("")

if tstr_summary_df is not None and tstr_summary_path is not None:
    summary_lines.append("## TSTR/TRTR supervised baselines")
    summary_lines.append(dataframe_to_markdown(tstr_summary_df, floatfmt=".3f"))
    summary_lines.append("")
    summary_lines.append("Artefacts:")
    summary_lines.append(
        f"- Summary table: {tstr_summary_path.relative_to(OUTPUT_DIR)}"
    )
    if tstr_plot_path is not None:
        summary_lines.append(f"- Plot data: {tstr_plot_path.relative_to(OUTPUT_DIR)}")
    for figure_path in tstr_figure_paths:
        summary_lines.append(f"- Figure: {figure_path.relative_to(OUTPUT_DIR)}")
    summary_lines.append("")

summary_lines.append("## Distribution shift and privacy")
if distribution_df is not None and distribution_path is not None:
    summary_lines.append(
        f"- Distribution metrics: {distribution_path.relative_to(OUTPUT_DIR)}"
    )
if membership_path.exists():
    summary_lines.append(
        f"- Membership inference: {membership_path.relative_to(OUTPUT_DIR)}"
    )
summary_lines.append(f"- Baseline metrics: {baseline_path.relative_to(OUTPUT_DIR)}")
for figure_path in benchmark_curve_paths:
    summary_lines.append(f"- Benchmark curves: {figure_path.relative_to(OUTPUT_DIR)}")
summary_lines.append("")

summary_path = OUTPUT_DIR / f"evaluation_summary_{TARGET_LABEL}.md"
summary_path.write_text("\n".join(summary_lines))
print(f"Summary written to {summary_path}")
