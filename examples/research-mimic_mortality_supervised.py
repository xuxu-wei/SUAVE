"""MIMIC mortality evaluation workflow.

Usage
-----
Interactive sessions (e.g. IPython, Jupyter)
    * Detect the active Optuna study and render the Pareto front for manual
      inspection.
    * Prompt for a trial identifier; pressing Enter reuses the most recently
      saved model when available, otherwise the chosen Pareto trial is trained.

Script mode (command line execution)
    * Accepts an optional ``trial_id`` positional argument (or ``--trial-id``)
      to force loading/training a specific Optuna trial.
    * Without an argument, attempts to load the most recent saved model; if
      absent, automatically trains the preferred Pareto-front trial or falls
      back to stored best parameters.
"""

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
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
    CLINICAL_SCORE_BENCHMARK_STRATEGY,
    VALIDATION_SIZE,
    PARETO_MAX_ABS_DELTA_AUC,
    PARETO_MIN_VALIDATION_ROAUC,
    DATA_DIR,
    VAR_GROUP_DICT,
    PATH_GRAPH_GROUP_COLORS,
    PATH_GRAPH_NODE_COLORS,
    PATH_GRAPH_NODE_GROUPS,
    PATH_GRAPH_NODE_LABELS,
    build_analysis_config,
    prepare_analysis_output_directories,
    parse_script_arguments,
    summarise_pareto_trials,
    build_prediction_dataframe,
    build_suave_model,
    build_tstr_training_sets,
    compute_binary_metrics,
    dataframe_to_markdown,
    define_schema,
    evaluate_transfer_baselines,
    extract_positive_probabilities,
    fit_isotonic_calibrator,
    is_interactive_session,
    load_dataset,
    load_or_create_iteratively_imputed_features,
    iteratively_impute_clinical_scores,
    ModelLoadingPlan,
    FORCE_UPDATE_FLAG_DEFAULTS,
    make_baseline_model_factories,
    make_study_name,
    read_bool_env_flag,
    resolve_model_loading_plan,
    confirm_model_loading_plan_selection,
    resolve_suave_fit_kwargs,
    resolve_analysis_output_root,
    plot_benchmark_curves,
    plot_calibration_curves,
    plot_latent_space,
    plot_transfer_metric_bars,
    prepare_features,
    render_dataframe,
    schema_to_dataframe,
    to_numeric_frame,
    record_model_manifest,
)
from cls_eval import evaluate_predictions, write_results_to_excel_unique  # noqa: E402

from suave.evaluate import (  # noqa: E402
    classifier_two_sample_test,
    energy_distance,
    mutual_information_feature,
    rbf_mmd,
    simple_membership_inference,
)

from suave import SUAVE  # noqa: E402
from suave.plots import (  # noqa: E402
    compute_feature_latent_correlation,
    plot_feature_latent_correlation_bubble,
    plot_feature_latent_correlation_heatmap,
    plot_feature_latent_outcome_path_graph,
)


# %% [markdown]
# ## Analysis configuration
#
# Define the label of interest and locations for cached outputs from the
# optimisation script.

# %%

TARGET_LABEL = "in_hospital_mortality"

analysis_config = build_analysis_config()

FORCE_UPDATE_BENCHMARK_MODEL = read_bool_env_flag(
    "FORCE_UPDATE_BENCHMARK_MODEL",
    FORCE_UPDATE_FLAG_DEFAULTS["FORCE_UPDATE_BENCHMARK_MODEL"],
)
FORCE_UPDATE_TSTR_MODEL = read_bool_env_flag(
    "FORCE_UPDATE_TSTR_MODEL",
    FORCE_UPDATE_FLAG_DEFAULTS["FORCE_UPDATE_TSTR_MODEL"],
)
FORCE_UPDATE_TRTR_MODEL = read_bool_env_flag(
    "FORCE_UPDATE_TRTR_MODEL",
    FORCE_UPDATE_FLAG_DEFAULTS["FORCE_UPDATE_TRTR_MODEL"],
)
FORCE_UPDATE_SUAVE = read_bool_env_flag(
    "FORCE_UPDATE_SUAVE",
    FORCE_UPDATE_FLAG_DEFAULTS["FORCE_UPDATE_SUAVE"],
)
INCLUDE_SUAVE_TRANSFER = read_bool_env_flag("INCLUDE_SUAVE_TRANSFER", False)

IS_INTERACTIVE = is_interactive_session()
CLI_REQUESTED_TRIAL_ID: Optional[int] = None
if not IS_INTERACTIVE:
    CLI_REQUESTED_TRIAL_ID = parse_script_arguments(sys.argv[1:])


def _sanitise_path_component(value: str) -> str:
    """Return a filesystem-friendly representation of ``value``."""

    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    return cleaned or "model"


# %% [markdown]
# ## Data loading and schema definition
#
# Load train/test/external splits, construct the schema, and validate the
# requested target label. Schema corrections are added here so that the
# downstream modelling code receives explicit type information.

# %%

OUTPUT_DIR = resolve_analysis_output_root(analysis_config["output_dir_name"])

analysis_dirs = prepare_analysis_output_directories(
    OUTPUT_DIR,
    (
        "data_schema",
        "feature_engineering",
        "optuna",
        "suave_model",
        "calibration_uncertainty",
        "evaluation_reports",
        "bootstrap_analysis",
        "baseline_models",
        "tstr_trtr",
        "distribution_shift",
        "privacy_assessment",
        "visualisations",
    ),
)

DATA_SCHEMA_DIR = analysis_dirs["data_schema"]
FEATURE_ENGINEERING_DIR = analysis_dirs["feature_engineering"]
OPTUNA_DIR = analysis_dirs["optuna"]
SUAVE_MODEL_DIR = analysis_dirs["suave_model"]
CALIBRATION_DIR = analysis_dirs["calibration_uncertainty"]
EVALUATION_DIR = analysis_dirs["evaluation_reports"]
BOOTSTRAP_DIR = analysis_dirs["bootstrap_analysis"]
BASELINE_MODELS_DIR = analysis_dirs["baseline_models"]
TSTR_TRTR_DIR = analysis_dirs["tstr_trtr"]
DISTRIBUTION_SHIFT_DIR = analysis_dirs["distribution_shift"]
PRIVACY_ASSESSMENT_DIR = analysis_dirs["privacy_assessment"]
VISUALISATIONS_DIR = analysis_dirs["visualisations"]

analysis_config["optuna_storage"] = (
    f"sqlite:///{OPTUNA_DIR}/{analysis_config['optuna_study_prefix']}_optuna.db"
)

train_df = load_dataset(DATA_DIR / "mimic-mortality-train.tsv")
test_df = load_dataset(DATA_DIR / "mimic-mortality-test.tsv")
external_df = load_dataset(DATA_DIR / "eicu-mortality-external_val.tsv")

if TARGET_LABEL not in TARGET_COLUMNS:
    raise ValueError(
        f"Target label '{TARGET_LABEL}' is not one of the configured targets: {TARGET_COLUMNS}"
    )

available_benchmark_columns = [
    column for column in BENCHMARK_COLUMNS if column in train_df.columns
]

FEATURE_COLUMNS = [
    column
    for column in train_df.columns
    if column not in TARGET_COLUMNS + BENCHMARK_COLUMNS
]

schema = define_schema(train_df, FEATURE_COLUMNS, 
                       mode="interactive" if IS_INTERACTIVE else "info"
                       )

# Manual schema corrections ensure columns with ambiguous types are treated
# appropriately during modelling.
schema.update(
    {
        "BMI": {"type": "pos"},
    }
)

schema_df = schema_to_dataframe(schema).reset_index(drop=True)
render_dataframe(schema_df, title="Schema overview", floatfmt=None)


model_loading_plan: ModelLoadingPlan = resolve_model_loading_plan(
    target_label=TARGET_LABEL,
    analysis_config=analysis_config,
    model_dir=SUAVE_MODEL_DIR,
    optuna_dir=OPTUNA_DIR,
    schema=schema,
    is_interactive=IS_INTERACTIVE,
    cli_requested_trial_id=CLI_REQUESTED_TRIAL_ID,
    force_update_suave=FORCE_UPDATE_SUAVE,
)

optuna_best_info = model_loading_plan.optuna_best_info
optuna_best_params = model_loading_plan.optuna_best_params
model_manifest = model_loading_plan.model_manifest
pareto_trials = model_loading_plan.pareto_trials

optuna_storage_uri = analysis_config.get("optuna_storage")
optuna_storage_path = None
if isinstance(optuna_storage_uri, str) and optuna_storage_uri.startswith("sqlite:///"):
    optuna_storage_path = Path(optuna_storage_uri.replace("sqlite:///", "", 1))

missing_optuna = (
    model_loading_plan.optuna_study is None
    or not model_loading_plan.optuna_best_params
    or (optuna_storage_path is not None and not optuna_storage_path.exists())
)


# %%
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

    requires_schema_aligned_features = True

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

(
    X_train_model,
    X_validation,
    y_train_model,
    y_validation,
    benchmark_train,
    benchmark_validation,
) = train_test_split(
    X_full,
    y_full,
    train_df.loc[:, available_benchmark_columns],
    test_size=VALIDATION_SIZE,
    stratify=y_full,
    random_state=RANDOM_STATE,
)

X_train_model = X_train_model.reset_index(drop=True)
X_validation = X_validation.reset_index(drop=True)
y_train_model = y_train_model.reset_index(drop=True)
y_validation = y_validation.reset_index(drop=True)
benchmark_train = benchmark_train.reset_index(drop=True)
benchmark_validation = benchmark_validation.reset_index(drop=True)

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

benchmark_frames: Dict[str, pd.DataFrame] = {
    "Train": benchmark_train,
    "Validation": benchmark_validation,
    "MIMIC test": test_df.loc[:, available_benchmark_columns].reset_index(drop=True),
}
if external_features is not None and TARGET_LABEL in external_df.columns:
    external_benchmark_columns = [
        column for column in available_benchmark_columns if column in external_df.columns
    ]
    if external_benchmark_columns:
        benchmark_frames["eICU external"] = (
            external_df.loc[:, external_benchmark_columns].reset_index(drop=True)
        )

# %% [markdown]
# ## Classical model benchmarks
#
# Fit a suite of scikit-learn classifiers as quick baselines before evaluating
# SUAVE. These provide a reference point for MIMIC test performance and, when
# available, eICU external validation.

# %%

baseline_feature_frames: Dict[str, pd.DataFrame] = {
    "Train": X_train_model,
    "Validation": X_validation,
    "MIMIC test": X_test,
}
if external_features is not None:
    baseline_feature_frames["eICU external"] = external_features

clinical_score_frames: Dict[str, pd.DataFrame] = {
    name: frame.copy() for name, frame in benchmark_frames.items()
}

clinical_score_strategy = CLINICAL_SCORE_BENCHMARK_STRATEGY.lower()
use_imputed_clinical_scores = clinical_score_strategy == "imputed"

if use_imputed_clinical_scores and available_benchmark_columns:
    clinical_score_frames = iteratively_impute_clinical_scores(
        clinical_score_frames,
        baseline_feature_frames,
        columns=available_benchmark_columns,
        reference_key="Train",
    )

(
    baseline_imputed_features,
    baseline_imputed_paths,
    baseline_loaded_from_cache,
) = load_or_create_iteratively_imputed_features(
    baseline_feature_frames,
    output_dir=FEATURE_ENGINEERING_DIR,
    target_label=TARGET_LABEL,
    reference_key="Train",
)

if baseline_loaded_from_cache:
    print("Loaded iterative-imputed baseline features from disk.")
else:
    print("Saved iterative-imputed baseline features:")
    for name, path in baseline_imputed_paths.items():
        print(f"  - {name}: {path}")

baseline_models: Dict[str, Pipeline] = {
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
    "Gradient boosting": Pipeline(
        [
            (
                "classifier",
                GradientBoostingClassifier(random_state=RANDOM_STATE),
            ),
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
    "Validation": y_validation,
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
    "Gradient boosting": "GB",
    "Random forest": "RF",
    "SVM (RBF)": "SVM",
}
model_abbreviation_lookup["SUAVE"] = "SUAVE"

train_features_imputed = baseline_imputed_features["Train"]
train_labels = baseline_label_sets["Train"]

baseline_model_cache_path = (
    BASELINE_MODELS_DIR / f"baseline_estimators_{TARGET_LABEL}.joblib"
)

if baseline_model_cache_path.exists() and not FORCE_UPDATE_BENCHMARK_MODEL:
    baseline_models = joblib.load(baseline_model_cache_path)
    print(
        "Loaded cached classical baseline models from",
        baseline_model_cache_path,
    )
else:
    for estimator in baseline_models.values():
        estimator.fit(train_features_imputed, train_labels)
    joblib.dump(baseline_models, baseline_model_cache_path)
    print("Saved classical baseline models to", baseline_model_cache_path)

for model_name, estimator in baseline_models.items():
    for dataset_name, (features, labels) in baseline_evaluation_sets.items():
        probabilities = estimator.predict_proba(features)
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
baseline_path = BASELINE_MODELS_DIR / f"baseline_models_{TARGET_LABEL}.csv"
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

optuna_trials_path = OPTUNA_DIR / f"optuna_trials_{TARGET_LABEL}.csv"

if IS_INTERACTIVE and pareto_trials:
    pareto_summary = summarise_pareto_trials(
        pareto_trials,
        manifest=model_manifest,
        model_dir=SUAVE_MODEL_DIR,
    )
    render_dataframe(
        pareto_summary,
        title="Pareto-optimal Optuna trials",
        floatfmt=".4f",
    )

    model_loading_plan = confirm_model_loading_plan_selection(
        model_loading_plan,
        is_interactive=IS_INTERACTIVE,
        model_dir=SUAVE_MODEL_DIR,
    )


# %% [markdown]
# ## Ensure calibrated SUAVE model
#
# Load the trained SUAVE model and isotonic calibrator if they were saved by the
# optimisation pipeline. When the artefacts are unavailable, retrain the model
# using the best Optuna parameters and calibrate on the validation split.

# %%

selected_trial_number = model_loading_plan.selected_trial_number
selected_model_path = model_loading_plan.selected_model_path
selected_calibrator_path = model_loading_plan.selected_calibrator_path
selected_params: Dict[str, Any] = dict(model_loading_plan.selected_params)

if not selected_params and optuna_best_params:
    selected_params = dict(optuna_best_params)

model: Optional[SUAVE] = model_loading_plan.preloaded_model
calibrator: Optional[Any] = None

if model is not None and selected_model_path and selected_model_path.exists():
    if selected_trial_number is not None:
        print(
            f"Reusing cached SUAVE model for Optuna trial #{selected_trial_number} from {selected_model_path}."
        )
    else:
        print(f"Reusing cached SUAVE model from {selected_model_path}.")
    if missing_optuna:
        print(
            "Optuna tuning artefacts were unavailable; using the saved SUAVE model as a local backup."
        )

if selected_calibrator_path and selected_calibrator_path.exists():
    calibrator = joblib.load(selected_calibrator_path)
    embedded = extract_calibrator_estimator(calibrator)
    if embedded is not None:
        model = embedded
        if selected_trial_number is not None:
            print(
                f"Loaded isotonic calibrator for Optuna trial #{selected_trial_number} from {selected_calibrator_path}."
            )
        else:
            print(f"Loaded isotonic calibrator from {selected_calibrator_path}.")
    else:
        print("Loaded calibrator did not embed a SUAVE estimator; it will be refitted.")
        calibrator = None

if model is None and selected_model_path and selected_model_path.exists():
    model = SUAVE.load(selected_model_path)
    if selected_trial_number is not None:
        print(
            f"Loaded SUAVE model for Optuna trial #{selected_trial_number} from {selected_model_path}."
        )
    else:
        print(f"Loaded SUAVE model from {selected_model_path}.")
    if missing_optuna:
        print(
            "Optuna tuning artefacts were unavailable; using the saved SUAVE model as a local backup."
        )

model_was_trained = False

if model is None:
    fit_params = selected_params or {}
    if not fit_params and missing_optuna:
        print(
            "Warning: Optuna tuning artefacts were not found; falling back to default SUAVE hyperparameters."
        )
    if selected_trial_number is not None:
        print(
            f"Training SUAVE for Optuna trial #{selected_trial_number} because no saved model artefacts were available…"
        )
    else:
        if fit_params:
            print(
                "Training SUAVE with fallback hyperparameters because no saved model was available…"
            )
        else:
            print(
                "Training SUAVE with default hyperparameters because no saved model was available…"
            )
    model = build_suave_model(fit_params, schema, random_state=RANDOM_STATE)
    fit_kwargs = resolve_suave_fit_kwargs(fit_params)
    model.fit(
        X_train_model,
        y_train_model,
        plot_monitor=IS_INTERACTIVE,
        **fit_kwargs,
    )
    model_was_trained = True

if model_was_trained:
    model_output_path = selected_model_path or (
        SUAVE_MODEL_DIR / f"suave_best_{TARGET_LABEL}.pt"
    )
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_output_path)
    selected_model_path = model_output_path
    if selected_trial_number is not None:
        print(
            f"Saved SUAVE model for Optuna trial #{selected_trial_number} to {model_output_path}."
        )
    else:
        print(f"Saved SUAVE model to {model_output_path}.")

calibrator_was_fitted = False

if calibrator is None:
    calibrator = fit_isotonic_calibrator(model, X_validation, y_validation)
    calibrator_was_fitted = True
    print("Fitted a new isotonic calibrator on the validation split.")
else:
    embedded = extract_calibrator_estimator(calibrator)
    if embedded is None:
        print(
            "Calibrator did not contain a usable SUAVE estimator; refitting calibrator."
        )
        calibrator = fit_isotonic_calibrator(model, X_validation, y_validation)
        calibrator_was_fitted = True
    else:
        model = embedded

if calibrator_was_fitted:
    calibrator_output_path = selected_calibrator_path or (
        CALIBRATION_DIR / f"isotonic_calibrator_{TARGET_LABEL}.joblib"
    )
    calibrator_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, calibrator_output_path)
    selected_calibrator_path = calibrator_output_path
    if selected_trial_number is not None:
        print(
            "Saved isotonic calibrator for Optuna trial "
            f"#{selected_trial_number} to {calibrator_output_path}."
        )
    else:
        print(f"Saved isotonic calibrator to {calibrator_output_path}.")

if (
    (model_was_trained or calibrator_was_fitted)
    and selected_model_path is not None
    and selected_calibrator_path is not None
):
    manifest_values: List[float] = []

    if selected_trial_number is not None:
        matching_trial = next(
            (trial for trial in pareto_trials if trial.number == selected_trial_number),
            None,
        )
        if matching_trial is not None and matching_trial.values is not None:
            manifest_values = [float(value) for value in matching_trial.values]

    if not manifest_values:
        previous_values = model_loading_plan.model_manifest.get("values")
        if isinstance(previous_values, Sequence) and not isinstance(
            previous_values, (str, bytes)
        ):
            manifest_values = [float(value) for value in previous_values]

    if not manifest_values:
        best_info_values = optuna_best_info.get("values")
        if isinstance(best_info_values, Sequence) and not isinstance(
            best_info_values, (str, bytes)
        ):
            manifest_values = [float(value) for value in best_info_values]

    manifest_path = record_model_manifest(
        SUAVE_MODEL_DIR,
        TARGET_LABEL,
        trial_number=selected_trial_number,
        values=manifest_values,
        params=selected_params,
        model_path=selected_model_path,
        calibrator_path=selected_calibrator_path,
        study_name=make_study_name(
            analysis_config.get("optuna_study_prefix"), TARGET_LABEL
        ),
        storage=analysis_config.get("optuna_storage"),
    )
    print(f"Updated SUAVE model manifest at {manifest_path}.")


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
metrics_path = EVALUATION_DIR / "evaluation_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
render_dataframe(
    metrics_df,
    title=f"Evaluation metrics for {TARGET_LABEL}",
    floatfmt=".3f",
)

calibration_path = CALIBRATION_DIR / f"calibration_{TARGET_LABEL}.png"
plot_calibration_curves(
    probability_map, label_map, target_name=TARGET_LABEL, output_path=calibration_path
)

# %% [markdown]
# ## Benchmark ROC and calibration curves
#
# Visualise the discriminative and calibration performance of the classical
# baselines alongside SUAVE across the train, test, and external cohorts.

# %%

benchmark_datasets = ["Train", "Validation", "MIMIC test", "eICU external"]
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
        output_dir=EVALUATION_DIR,
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
#
# %%

clinical_score_models: Dict[str, LogisticRegression] = {}
if available_benchmark_columns:
    for column in available_benchmark_columns:
        if column not in benchmark_train.columns:
            continue
        training_scores = (
            clinical_score_frames.get("Train", benchmark_train)
            if use_imputed_clinical_scores
            else benchmark_train
        )
        if column not in training_scores.columns:
            continue
        score_frame = training_scores[[column]].reset_index(drop=True)
        score_labels = y_train_model
        if use_imputed_clinical_scores:
            score_labels = score_labels.reset_index(drop=True)
        else:
            mask = score_frame[column].notna()
            if not mask.any():
                continue
            score_frame = score_frame.loc[mask].reset_index(drop=True)
            score_labels = score_labels.loc[mask].reset_index(drop=True)
        if score_frame.empty or score_labels.empty:
            continue
        try:
            estimator = LogisticRegression(max_iter=1000, solver="lbfgs")
            estimator.fit(score_frame, score_labels)
        except ValueError as exc:
            print(
                f"Skipping calibration for clinical score '{column}' because "
                f"fitting failed: {exc}"
            )
            continue
        clinical_score_models[column] = estimator


def _format_metric_with_ci(
    value: object, low: Optional[object], high: Optional[object]
) -> str:
    """Format ``value`` with its confidence interval if available."""

    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        if value is None:
            return "NA"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"

    for bound in (low, high):
        try:
            if pd.isna(bound):
                return f"{val:.3f}"
        except TypeError:
            if bound is None:
                return f"{val:.3f}"
    try:
        return f"{val:.3f} ({float(low):.3f}, {float(high):.3f})"
    except (TypeError, ValueError):
        return f"{val:.3f}"


model_prediction_frames: Dict[str, Dict[str, pd.DataFrame]] = {}

model_classes_array = getattr(model, "_classes", None)
if model_classes_array is None or len(model_classes_array) == 0:
    model_classes_array = np.unique(np.asarray(y_train_model))
class_value_list = list(model_classes_array)
class_name_strings = [str(value) for value in class_value_list]
positive_label_name = class_name_strings[-1] if len(class_name_strings) == 2 else None

suave_prediction_tables: Dict[str, pd.DataFrame] = {}
for dataset_name, (features, labels) in evaluation_datasets.items():
    dataset_probabilities = probability_map[dataset_name]
    positive_probs = extract_positive_probabilities(dataset_probabilities)
    if len(class_value_list) == 2 and positive_probs is not None:
        negative_label = class_value_list[0]
        positive_label = class_value_list[-1]
        dataset_predictions = np.where(
            positive_probs >= 0.5, positive_label, negative_label
        )
    else:
        dataset_predictions = model.predict(features)
    suave_prediction_tables[dataset_name] = build_prediction_dataframe(
        dataset_probabilities,
        labels,
        dataset_predictions,
        class_name_strings,
    )
model_prediction_frames["SUAVE"] = suave_prediction_tables

for baseline_name in baseline_models.keys():
    baseline_tables: Dict[str, pd.DataFrame] = {}
    for dataset_name, probability_lookup in baseline_probability_map.items():
        baseline_probs = probability_lookup.get(baseline_name)
        labels = label_map.get(dataset_name)
        if baseline_probs is None or labels is None:
            continue
        baseline_prob_array = np.asarray(baseline_probs)
        if len(class_name_strings) == 2 and baseline_prob_array.ndim == 1:
            negative_label = class_value_list[0]
            positive_label = class_value_list[-1]
            probability_matrix = np.column_stack(
                [1.0 - baseline_prob_array, baseline_prob_array]
            )
            predictions = np.where(
                baseline_prob_array >= 0.5, positive_label, negative_label
            )
            class_names_for_baseline = class_name_strings
        else:
            estimator = baseline_models.get(baseline_name)
            feature_frame = baseline_feature_frames.get(dataset_name)
            if estimator is None or feature_frame is None:
                continue
            probability_matrix = estimator.predict_proba(feature_frame)
            predictions = estimator.predict(feature_frame)
            class_names_for_baseline = [
                str(cls) for cls in getattr(estimator, "classes_", [])
            ]
        baseline_tables[dataset_name] = build_prediction_dataframe(
            probability_matrix,
            labels,
            predictions,
            class_names_for_baseline,
        )
    if baseline_tables:
        model_prediction_frames[baseline_name] = baseline_tables

for score_name, estimator in clinical_score_models.items():
    score_tables: Dict[str, pd.DataFrame] = {}
    estimator_class_names = [str(cls) for cls in getattr(estimator, "classes_", [])]
    if not estimator_class_names:
        continue
    for dataset_name, score_frame in benchmark_frames.items():
        if dataset_name not in label_map or score_name not in score_frame.columns:
            continue
        if use_imputed_clinical_scores:
            dataset_scores = clinical_score_frames.get(dataset_name)
            if dataset_scores is None or score_name not in dataset_scores.columns:
                continue
            input_frame = dataset_scores[[score_name]].reset_index(drop=True)
            label_series = pd.Series(
                label_map[dataset_name], index=score_frame.index
            ).reset_index(drop=True)
        else:
            raw_scores = score_frame[[score_name]]
            mask = raw_scores[score_name].notna()
            if not mask.any():
                continue
            input_frame = raw_scores.loc[mask].reset_index(drop=True)
            label_series = pd.Series(
                label_map[dataset_name], index=raw_scores.index
            ).loc[mask].reset_index(drop=True)
        if input_frame.empty or label_series.empty:
            continue
        probability_matrix = estimator.predict_proba(input_frame)
        predictions = estimator.predict(input_frame)
        score_tables[dataset_name] = build_prediction_dataframe(
            probability_matrix,
            label_series,
            predictions,
            estimator_class_names,
        )
    if score_tables:
        model_prediction_frames[score_name] = score_tables


bootstrap_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
bootstrap_overall_frames: List[pd.DataFrame] = []
bootstrap_per_class_frames: List[pd.DataFrame] = []
bootstrap_warnings_frames: List[pd.DataFrame] = []
bootstrap_overall_record_frames: List[pd.DataFrame] = []
bootstrap_per_class_record_frames: List[pd.DataFrame] = []

for model_name, dataset_tables in model_prediction_frames.items():
    model_dir = BOOTSTRAP_DIR / _sanitise_path_component(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_results: Dict[str, Dict[str, pd.DataFrame]] = {}
    model_overall_frames: List[pd.DataFrame] = []
    model_per_class_frames: List[pd.DataFrame] = []
    model_overall_record_frames: List[pd.DataFrame] = []
    model_per_class_record_frames: List[pd.DataFrame] = []
    model_warning_frames: List[pd.DataFrame] = []

    for dataset_name, prediction_df in dataset_tables.items():
        results = evaluate_predictions(
            prediction_df,
            label_col="label",
            pred_col="y_pred",
            positive_label=positive_label_name,
            bootstrap_n=1000,
            random_state=RANDOM_STATE,
        )
        model_results[dataset_name] = results

        dataset_slug = _sanitise_path_component(dataset_name.lower())

        overall_df = results["overall"].copy()
        overall_augmented = overall_df.copy()
        overall_augmented.insert(0, "Dataset", dataset_name)
        overall_augmented.insert(0, "Model", model_name)
        overall_augmented.insert(0, "Target", TARGET_LABEL)
        bootstrap_overall_frames.append(overall_augmented)
        model_overall_frames.append(overall_augmented)
        overall_df.to_csv(model_dir / f"{dataset_slug}_overall.csv", index=False)

        per_class_df = results["per_class"].copy()
        per_class_augmented = per_class_df.copy()
        per_class_augmented.insert(0, "Dataset", dataset_name)
        per_class_augmented.insert(0, "Model", model_name)
        per_class_augmented.insert(0, "Target", TARGET_LABEL)
        bootstrap_per_class_frames.append(per_class_augmented)
        model_per_class_frames.append(per_class_augmented)
        per_class_df.to_csv(model_dir / f"{dataset_slug}_per_class.csv", index=False)

        overall_records_df = results.get("bootstrap_overall_records")
        if overall_records_df is not None and not overall_records_df.empty:
            overall_records_copy = overall_records_df.copy()
            overall_records_copy.insert(0, "Dataset", dataset_name)
            overall_records_copy.insert(0, "Model", model_name)
            overall_records_copy.insert(0, "Target", TARGET_LABEL)
            bootstrap_overall_record_frames.append(overall_records_copy)
            model_overall_record_frames.append(overall_records_copy)
            overall_records_df.to_csv(
                model_dir / f"{dataset_slug}_bootstrap_overall_records.csv",
                index=False,
            )

        per_class_records_df = results.get("bootstrap_per_class_records")
        if per_class_records_df is not None and not per_class_records_df.empty:
            per_class_records_copy = per_class_records_df.copy()
            per_class_records_copy.insert(0, "Dataset", dataset_name)
            per_class_records_copy.insert(0, "Model", model_name)
            per_class_records_copy.insert(0, "Target", TARGET_LABEL)
            bootstrap_per_class_record_frames.append(per_class_records_copy)
            model_per_class_record_frames.append(per_class_records_copy)
            per_class_records_df.to_csv(
                model_dir / f"{dataset_slug}_bootstrap_per_class_records.csv",
                index=False,
            )

        warnings_df = results.get("warnings")
        if warnings_df is not None and not warnings_df.empty:
            warnings_copy = warnings_df.copy()
            warnings_copy.insert(0, "Dataset", dataset_name)
            warnings_copy.insert(0, "Model", model_name)
            warnings_copy.insert(0, "Target", TARGET_LABEL)
            bootstrap_warnings_frames.append(warnings_copy)
            model_warning_frames.append(warnings_copy)
            warnings_df.to_csv(
                model_dir / f"{dataset_slug}_warnings.csv", index=False
            )

    if model_results:
        bootstrap_results[model_name] = model_results

    if model_overall_frames:
        combined_overall_df = pd.concat(model_overall_frames, ignore_index=True)
        combined_overall_df.to_csv(
            model_dir / "bootstrap_overall.csv", index=False
        )
    else:
        combined_overall_df = pd.DataFrame()

    if model_per_class_frames:
        combined_per_class_df = pd.concat(
            model_per_class_frames, ignore_index=True
        )
        combined_per_class_df.to_csv(
            model_dir / "bootstrap_per_class.csv", index=False
        )

    if model_overall_record_frames:
        combined_overall_records_df = pd.concat(
            model_overall_record_frames, ignore_index=True
        )
        combined_overall_records_df.to_csv(
            model_dir / "bootstrap_overall_records.csv", index=False
        )

    if model_per_class_record_frames:
        combined_per_class_records_df = pd.concat(
            model_per_class_record_frames, ignore_index=True
        )
        combined_per_class_records_df.to_csv(
            model_dir / "bootstrap_per_class_records.csv", index=False
        )

    if model_warning_frames:
        combined_warning_df = pd.concat(model_warning_frames, ignore_index=True)
        combined_warning_df.to_csv(
            model_dir / "bootstrap_warnings.csv", index=False
        )

    if not combined_overall_df.empty:
        summary_df = combined_overall_df.copy()
        metric_name_map = {
            "roc_auc": "AUROC",
            "accuracy": "ACC",
            "f1_micro": "F1_micro",
            "pr_auc": "AUPRC",
        }
        rename_map = {}
        for metric_key, display_name in metric_name_map.items():
            if metric_key in summary_df.columns:
                rename_map[metric_key] = display_name
            low_col = f"{metric_key}_ci_low"
            high_col = f"{metric_key}_ci_high"
            if low_col in summary_df.columns:
                rename_map[low_col] = f"{display_name}_ci_low"
            if high_col in summary_df.columns:
                rename_map[high_col] = f"{display_name}_ci_high"
        if rename_map:
            summary_df = summary_df.rename(columns=rename_map)

        summary_columns = ["Target", "Model", "Dataset"]
        for display_name in ["AUROC", "ACC", "F1_micro", "AUPRC"]:
            if display_name in summary_df.columns:
                summary_columns.append(display_name)
                low_col = f"{display_name}_ci_low"
                high_col = f"{display_name}_ci_high"
                if low_col in summary_df.columns:
                    summary_columns.append(low_col)
                if high_col in summary_df.columns:
                    summary_columns.append(high_col)
                ci_label = f"{display_name} (95% CI)"
                summary_df[ci_label] = summary_df.apply(
                    lambda row, name=display_name: _format_metric_with_ci(
                        row.get(name),
                        row.get(f"{name}_ci_low"),
                        row.get(f"{name}_ci_high"),
                    ),
                    axis=1,
                )
                summary_columns.append(ci_label)
        summary_df = summary_df.loc[:, summary_columns]
        summary_df.to_csv(model_dir / "metric_summary.csv", index=False)

    if model_results:
        excel_path = model_dir / (
            _sanitise_path_component(model_name.lower()) + "_bootstrap.xlsx"
        )
        write_results_to_excel_unique(
            model_results,
            str(excel_path),
            include_warnings_sheet=True,
        )


bootstrap_overall_df: pd.DataFrame
if bootstrap_overall_frames:
    bootstrap_overall_df = pd.concat(bootstrap_overall_frames, ignore_index=True)
else:
    bootstrap_overall_df = pd.DataFrame()

bootstrap_per_class_df: pd.DataFrame
if bootstrap_per_class_frames:
    bootstrap_per_class_df = pd.concat(bootstrap_per_class_frames, ignore_index=True)
else:
    bootstrap_per_class_df = pd.DataFrame()

bootstrap_overall_path = BOOTSTRAP_DIR / f"bootstrap_overall_{TARGET_LABEL}.csv"
bootstrap_per_class_path = BOOTSTRAP_DIR / f"bootstrap_per_class_{TARGET_LABEL}.csv"
bootstrap_overall_df.to_csv(bootstrap_overall_path, index=False)
bootstrap_per_class_df.to_csv(bootstrap_per_class_path, index=False)

bootstrap_overall_records_path: Optional[Path]
if bootstrap_overall_record_frames:
    bootstrap_overall_records_df = pd.concat(
        bootstrap_overall_record_frames, ignore_index=True
    )
    bootstrap_overall_records_path = (
        BOOTSTRAP_DIR / f"bootstrap_overall_records_{TARGET_LABEL}.csv"
    )
    bootstrap_overall_records_df.to_csv(
        bootstrap_overall_records_path, index=False
    )
else:
    bootstrap_overall_records_path = None

bootstrap_per_class_records_path: Optional[Path]
if bootstrap_per_class_record_frames:
    bootstrap_per_class_records_df = pd.concat(
        bootstrap_per_class_record_frames, ignore_index=True
    )
    bootstrap_per_class_records_path = (
        BOOTSTRAP_DIR / f"bootstrap_per_class_records_{TARGET_LABEL}.csv"
    )
    bootstrap_per_class_records_df.to_csv(
        bootstrap_per_class_records_path, index=False
    )
else:
    bootstrap_per_class_records_path = None

bootstrap_warning_path: Optional[Path]
if bootstrap_warnings_frames:
    bootstrap_warning_df = pd.concat(bootstrap_warnings_frames, ignore_index=True)
    bootstrap_warning_path = BOOTSTRAP_DIR / f"bootstrap_warnings_{TARGET_LABEL}.csv"
    bootstrap_warning_df.to_csv(bootstrap_warning_path, index=False)
else:
    bootstrap_warning_path = None

metrics_label_map = {
    "roc_auc": "AUROC",
    "accuracy": "ACC",
    "f1_micro": "F1_micro",
    "pr_auc": "AUPRC",
}
bootstrap_summary_df = bootstrap_overall_df.copy()
rename_map = {}
for metric_key, display_name in metrics_label_map.items():
    if metric_key in bootstrap_summary_df.columns:
        rename_map[metric_key] = display_name
    low_col = f"{metric_key}_ci_low"
    high_col = f"{metric_key}_ci_high"
    if low_col in bootstrap_summary_df.columns:
        rename_map[low_col] = f"{display_name}_ci_low"
    if high_col in bootstrap_summary_df.columns:
        rename_map[high_col] = f"{display_name}_ci_high"
if rename_map:
    bootstrap_summary_df = bootstrap_summary_df.rename(columns=rename_map)

summary_columns = ["Target", "Model", "Dataset"]
for display_name in metrics_label_map.values():
    if display_name in bootstrap_summary_df.columns:
        summary_columns.append(display_name)
        low_col = f"{display_name}_ci_low"
        high_col = f"{display_name}_ci_high"
        if low_col in bootstrap_summary_df.columns:
            summary_columns.append(low_col)
        if high_col in bootstrap_summary_df.columns:
            summary_columns.append(high_col)
        ci_label = f"{display_name} (95% CI)"
        bootstrap_summary_df[ci_label] = bootstrap_summary_df.apply(
            lambda row, name=display_name: _format_metric_with_ci(
                row.get(name), row.get(f"{name}_ci_low"), row.get(f"{name}_ci_high")
            ),
            axis=1,
        )
        summary_columns.append(ci_label)

bootstrap_summary_df = bootstrap_summary_df.loc[:, summary_columns]

required_datasets = ["Train", "Validation", "MIMIC test", "eICU external"]
bootstrap_summary_df = bootstrap_summary_df[
    bootstrap_summary_df["Dataset"].isin(required_datasets)
].copy()
bootstrap_summary_df["Dataset"] = pd.Categorical(
    bootstrap_summary_df["Dataset"],
    categories=required_datasets,
    ordered=True,
)
bootstrap_summary_df = bootstrap_summary_df.sort_values(
    ["Model", "Dataset"]
).reset_index(drop=True)

bootstrap_summary_path = BOOTSTRAP_DIR / f"bootstrap_summary_{TARGET_LABEL}.csv"
bootstrap_summary_df.to_csv(bootstrap_summary_path, index=False)
render_dataframe(
    bootstrap_summary_df,
    title=(
        "Bootstrap performance with confidence intervals for "
        f"{TARGET_LABEL}"
    ),
    floatfmt=".3f",
)
# %% [markdown]
# ## TSTR/TRTR comparison
#
# Compare models trained on synthetic versus real data. This block mirrors the
# published SUAVE protocol and therefore only runs for the in-hospital
# mortality target, which is the cohort studied in the manuscript.

# %%

tstr_summary_df: Optional[pd.DataFrame] = None
tstr_plot_df: Optional[pd.DataFrame] = None
tstr_summary_path: Optional[Path] = None
tstr_plot_path: Optional[Path] = None
tstr_figure_paths: List[Path] = []
distribution_df: Optional[pd.DataFrame] = None
distribution_path: Optional[Path] = None
distribution_top: Optional[pd.DataFrame] = None
tstr_nested_results: Optional[
    Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]
] = None
tstr_bootstrap_overall_records_path: Optional[Path] = None
tstr_bootstrap_per_class_records_path: Optional[Path] = None

transfer_results_cache_path = (
    TSTR_TRTR_DIR / f"tstr_trtr_results_{TARGET_LABEL}.joblib"
)

if TARGET_LABEL != "in_hospital_mortality":
    print(
        "Skipping TSTR/TRTR comparison because it is defined for the in-hospital "
        "mortality task."
    )
else:
    print("Generating synthetic data for TSTR/TRTR comparisons…")
    training_sets_numeric, training_sets_raw = build_tstr_training_sets(
        model,
        FEATURE_COLUMNS,
        X_full,
        y_full,
        random_state=RANDOM_STATE,
        return_raw=True,
    )
    evaluation_sets_numeric: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
        "MIMIC test": (to_numeric_frame(X_test), y_test.reset_index(drop=True)),
    }
    evaluation_sets_raw: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
        "MIMIC test": (X_test.reset_index(drop=True), y_test.reset_index(drop=True)),
    }
    if external_features is not None and external_labels is not None:
        evaluation_sets_numeric["eICU external"] = (
            to_numeric_frame(external_features),
            external_labels.reset_index(drop=True),
        )
        evaluation_sets_raw["eICU external"] = (
            external_features.reset_index(drop=True),
            external_labels.reset_index(drop=True),
        )

    model_factories = dict(make_baseline_model_factories(RANDOM_STATE))
    if INCLUDE_SUAVE_TRANSFER and optuna_best_params:
        suave_fit_kwargs = resolve_suave_fit_kwargs(optuna_best_params)

        def make_suave_transfer_estimator() -> _TSTRSuaveEstimator:
            base_model = build_suave_model(
                optuna_best_params,
                schema,
                random_state=RANDOM_STATE,
            )
            return _TSTRSuaveEstimator(base_model, suave_fit_kwargs)

        model_factories["SUAVE (Optuna best)"] = make_suave_transfer_estimator
    elif INCLUDE_SUAVE_TRANSFER and not optuna_best_params:
        print(
            "Skipping SUAVE TSTR/TRTR baseline because no Optuna parameters are available."
        )

    should_use_cached_transfer = (
        transfer_results_cache_path.exists()
        and not FORCE_UPDATE_TSTR_MODEL
        and not FORCE_UPDATE_TRTR_MODEL
    )

    cached_transfer_payload: Optional[Dict[str, Any]] = None
    if should_use_cached_transfer:
        payload = joblib.load(transfer_results_cache_path)
        cached_model_order = payload.get("model_order")
        expected_model_order = list(model_factories.keys())
        if list(cached_model_order or []) == expected_model_order:
            cached_transfer_payload = payload
            tstr_summary_df = cached_transfer_payload.get("summary_df")
            tstr_plot_df = cached_transfer_payload.get("plot_df")
            tstr_nested_results = cached_transfer_payload.get("nested_results")
            print(
                "Loaded cached TSTR/TRTR evaluation results from",
                transfer_results_cache_path,
            )
        else:
            print(
                "Discarding cached TSTR/TRTR evaluation results because the model roster changed.",
            )
    if cached_transfer_payload is None:
        (
            tstr_summary_df,
            tstr_plot_df,
            tstr_nested_results,
        ) = evaluate_transfer_baselines(
            training_sets_numeric,
            evaluation_sets_numeric,
            model_factories=model_factories,
            bootstrap_n=1000,
            random_state=RANDOM_STATE,
            raw_training_sets=training_sets_raw,
            raw_evaluation_sets=evaluation_sets_raw,
        )
        transfer_payload = {
            "summary_df": tstr_summary_df,
            "plot_df": tstr_plot_df,
            "nested_results": tstr_nested_results,
            "training_order": list(training_sets_numeric.keys()),
            "model_order": list(model_factories.keys()),
        }
        joblib.dump(transfer_payload, transfer_results_cache_path)
        print("Saved TSTR/TRTR evaluation results to", transfer_results_cache_path)
    transfer_overall_df: Optional[pd.DataFrame] = None
    transfer_per_class_df: Optional[pd.DataFrame] = None
    render_dataframe(
        tstr_summary_df,
        title="TSTR/TRTR supervised evaluation",
        floatfmt=".3f",
    )

    if cached_transfer_payload is not None:
        training_order = cached_transfer_payload.get(
            "training_order", list(training_sets_numeric.keys())
        )
        model_order = list(model_factories.keys())
    else:
        training_order = list(training_sets_numeric.keys())
        model_order = list(model_factories.keys())
    for evaluation_name in evaluation_sets_numeric.keys():
        for metric_name in ("accuracy", "roc_auc"):
            figure_path = plot_transfer_metric_bars(
                tstr_plot_df,
                metric=metric_name,
                evaluation_dataset=evaluation_name,
                training_order=training_order,
                model_order=model_order,
                output_dir=TSTR_TRTR_DIR,
                target_label=TARGET_LABEL,
            )
            if figure_path is not None:
                tstr_figure_paths.append(figure_path)

    real_features_numeric = training_sets_numeric["TRTR (real)"][0]
    synthesis_features_numeric = training_sets_numeric["TSTR synthesis"][0]

    if tstr_nested_results is not None:
        transfer_overall_records: List[pd.DataFrame] = []
        transfer_per_class_records: List[pd.DataFrame] = []
        for training_name, model_map in tstr_nested_results.items():
            for model_name, evaluation_map in model_map.items():
                for evaluation_name, result_map in evaluation_map.items():
                    overall_records = result_map.get("bootstrap_overall_records")
                    if overall_records is not None and not overall_records.empty:
                        overall_copy = overall_records.copy()
                        overall_copy.insert(0, "evaluation_dataset", evaluation_name)
                        overall_copy.insert(0, "model", model_name)
                        overall_copy.insert(0, "training_dataset", training_name)
                        transfer_overall_records.append(overall_copy)
                    per_class_records = result_map.get("bootstrap_per_class_records")
                    if per_class_records is not None and not per_class_records.empty:
                        per_class_copy = per_class_records.copy()
                        per_class_copy.insert(0, "evaluation_dataset", evaluation_name)
                        per_class_copy.insert(0, "model", model_name)
                        per_class_copy.insert(0, "training_dataset", training_name)
                        transfer_per_class_records.append(per_class_copy)
        if transfer_overall_records:
            transfer_overall_df = pd.concat(transfer_overall_records, ignore_index=True)
        if transfer_per_class_records:
            transfer_per_class_df = pd.concat(
                transfer_per_class_records, ignore_index=True
            )

    tstr_excel_path = TSTR_TRTR_DIR / "TSTR_TRTR_eval.xlsx"
    with pd.ExcelWriter(tstr_excel_path) as writer:
        tstr_summary_df.to_excel(writer, sheet_name="summary", index=False)
        tstr_plot_df.to_excel(writer, sheet_name="plot_data", index=False)
        if transfer_overall_df is not None:
            transfer_overall_df.to_excel(
                writer, sheet_name="bootstrap_overall", index=False
            )
        if transfer_per_class_df is not None:
            transfer_per_class_df.to_excel(
                writer, sheet_name="bootstrap_per_class", index=False
            )
    print("Saved TSTR/TRTR evaluation workbook to", tstr_excel_path)

    c2st_model_factories = make_baseline_model_factories(RANDOM_STATE)
    c2st_metrics = classifier_two_sample_test(
        real_features_numeric.to_numpy(),
        synthesis_features_numeric.to_numpy(),
        model_factories=c2st_model_factories,
        random_state=RANDOM_STATE,
        n_bootstrap=1000,
    )
    c2st_primary = pd.DataFrame(
        [
            {
                "target": TARGET_LABEL,
                "gbdt_auc": c2st_metrics.get("gbdt_auc", float("nan")),
                "gbdt_auc_ci_low": c2st_metrics.get("gbdt_auc_ci_low", float("nan")),
                "gbdt_auc_ci_high": c2st_metrics.get("gbdt_auc_ci_high", float("nan")),
                "gbdt_bootstrap_samples": c2st_metrics.get(
                    "gbdt_bootstrap_samples", float("nan")
                ),
                "n_real_samples": c2st_metrics.get("n_real_samples", float("nan")),
                "n_synthetic_samples": c2st_metrics.get(
                    "n_synthetic_samples", float("nan")
                ),
                "n_features": c2st_metrics.get("n_features", float("nan")),
                "cv_splits": c2st_metrics.get("cv_splits", float("nan")),
            }
        ]
    )
    secondary_rows: List[Dict[str, object]] = []
    for model_name in c2st_model_factories:
        if model_name == "GBDT":
            continue
        prefix = model_name.lower().replace(" ", "_")
        auc_key = f"{prefix}_auc"
        if auc_key not in c2st_metrics:
            continue
        secondary_rows.append(
            {
                "target": TARGET_LABEL,
                "model": model_name,
                "auc": c2st_metrics.get(auc_key, float("nan")),
                "auc_ci_low": c2st_metrics.get(f"{prefix}_auc_ci_low", float("nan")),
                "auc_ci_high": c2st_metrics.get(f"{prefix}_auc_ci_high", float("nan")),
                "bootstrap_samples": c2st_metrics.get(
                    f"{prefix}_bootstrap_samples", float("nan")
                ),
            }
        )
    c2st_secondary = pd.DataFrame(secondary_rows)
    render_dataframe(
        c2st_primary,
        title="Classifier two-sample test (C2ST) - GBDT",
        floatfmt=".3f",
    )
    if not c2st_secondary.empty:
        render_dataframe(
            c2st_secondary,
            title="Classifier two-sample test (C2ST) - secondary models",
            floatfmt=".3f",
        )

    c2st_workbook_path = DISTRIBUTION_SHIFT_DIR / "C2ST-distribution_shift.xlsx"
    with pd.ExcelWriter(c2st_workbook_path) as writer:
        c2st_primary.to_excel(writer, sheet_name="gbdt_primary", index=False)
        if not c2st_secondary.empty:
            c2st_secondary.to_excel(
                writer, sheet_name="secondary_models", index=False
            )
        pd.DataFrame([{k: v for k, v in c2st_metrics.items()}]).to_excel(
            writer, sheet_name="raw_metrics", index=False
        )
    print("Saved C2ST results to", c2st_workbook_path)

    global_mmd, global_mmd_p_value = rbf_mmd(
        real_features_numeric,
        synthesis_features_numeric,
        random_state=RANDOM_STATE,
        n_permutations=200,
    )
    global_energy, global_energy_p_value = energy_distance(
        real_features_numeric,
        synthesis_features_numeric,
        random_state=RANDOM_STATE,
        n_permutations=200,
    )
    distribution_overall_df = pd.DataFrame(
        [
            {
                "target": TARGET_LABEL,
                "global_mmd": global_mmd,
                "global_mmd_p_value": global_mmd_p_value,
                "global_energy_distance": global_energy,
                "global_energy_p_value": global_energy_p_value,
                **c2st_metrics,
            }
        ]
    )
    render_dataframe(
        distribution_overall_df,
        title="Distribution shift overview",
        floatfmt=".3f",
    )

    distribution_rows: List[Dict[str, object]] = []
    for column in FEATURE_COLUMNS:
        real_values = real_features_numeric[column].to_numpy()
        synthetic_values = synthesis_features_numeric[column].to_numpy()
        mmd_value, mmd_p = rbf_mmd(
            real_values,
            synthetic_values,
            random_state=RANDOM_STATE,
            n_permutations=200,
        )
        energy_value, _ = energy_distance(
            real_values,
            synthetic_values,
            random_state=RANDOM_STATE,
            n_permutations=0,
        )
        distribution_rows.append(
            {
                "feature": column,
                "mmd": mmd_value,
                "mmd_p_value": mmd_p,
                "energy_distance": energy_value,
                "mutual_information": mutual_information_feature(
                    real_values, synthetic_values
                ),
            }
        )
    distribution_df = pd.DataFrame(distribution_rows)
    distribution_path = DISTRIBUTION_SHIFT_DIR / "metrics_distribution_shift.xlsx"
    with pd.ExcelWriter(distribution_path) as writer:
        distribution_overall_df.to_excel(writer, sheet_name="overall", index=False)
        distribution_df.to_excel(writer, sheet_name="per_feature", index=False)
    distribution_top = (
        distribution_df.sort_values("mutual_information", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    render_dataframe(
        distribution_top,
        title="Top distribution shift features (mutual information)",
        floatfmt=".3f",
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
    membership_path = PRIVACY_ASSESSMENT_DIR / "membership_inference.xlsx"
    with pd.ExcelWriter(membership_path) as writer:
        membership_df.to_excel(writer, sheet_name="summary", index=False)
    render_dataframe(
        membership_df,
        title="Membership inference baseline",
        floatfmt=".3f",
    )


# %% [markdown]
# ## Latent space correlation analysis
#
# Quantify the relationship between latent representations and clinical
# features using Spearman correlation. The multilayer path graph provides the
# primary interpretation of latent-feature-target associations, with heatmaps
# and bubble charts serving as supplementary summaries.

# %%

latent_correlation_base = (
    VISUALISATIONS_DIR / f"latent_clinical_correlation_{TARGET_LABEL}"
)
overall_corr_path = latent_correlation_base.with_name(
    f"{latent_correlation_base.name}_correlations.csv"
)
overall_pval_path = latent_correlation_base.with_name(
    f"{latent_correlation_base.name}_pvalues.csv"
)
overall_bubble_base = latent_correlation_base.with_name(
    f"{latent_correlation_base.name}_bubble"
)
overall_corr_heatmap_base = latent_correlation_base.with_name(
    f"{latent_correlation_base.name}_corr_heatmap"
)
overall_pval_heatmap_base = latent_correlation_base.with_name(
    f"{latent_correlation_base.name}_pvalue_heatmap"
)

overall_corr, overall_pvals = compute_feature_latent_correlation(
    model,
    X_train_model,
    targets=y_train_model,
    target_name=TARGET_LABEL,
    variables=list(FEATURE_COLUMNS) + [TARGET_LABEL],
)
overall_corr.to_csv(overall_corr_path)
overall_pvals.to_csv(overall_pval_path)

overall_path_graph_base = latent_correlation_base.with_name(
    f"{latent_correlation_base.name}_path_graph"
)
overall_path_fig, _overall_path_ax = plot_feature_latent_outcome_path_graph(
    model,
    X_train_model,
    y=y_train_model,
    target_name=TARGET_LABEL,
    node_label_mapping=PATH_GRAPH_NODE_LABELS,
    node_color_mapping=PATH_GRAPH_NODE_COLORS,
    node_group_mapping=PATH_GRAPH_NODE_GROUPS,
    group_color_mapping=PATH_GRAPH_GROUP_COLORS,
    edge_label_top_k=15,
    figure_kwargs={"figsize": (14, 8)},
)
overall_path_graph_path = overall_path_graph_base.with_suffix(".png")
overall_path_fig.savefig(overall_path_graph_path, dpi=300, bbox_inches="tight")
overall_path_fig.savefig(
    overall_path_graph_base.with_suffix(".pdf"), bbox_inches="tight"
)
plt.close(overall_path_fig)

overall_bubble_fig, _overall_bubble_ax = plot_feature_latent_correlation_bubble(
    model,
    X_train_model,
    targets=y_train_model,
    target_name=TARGET_LABEL,
    variables=list(FEATURE_COLUMNS) + [TARGET_LABEL],
    title=f"Latent correlations ({TARGET_LABEL}) – bubble chart",
    output_path=overall_bubble_base,
    correlations=overall_corr,
    p_values=overall_pvals,
)
plt.close(overall_bubble_fig)
overall_bubble_path = overall_bubble_base.with_suffix(".png")

overall_corr_fig, _overall_corr_ax = plot_feature_latent_correlation_heatmap(
    model,
    X_train_model,
    targets=y_train_model,
    target_name=TARGET_LABEL,
    variables=list(FEATURE_COLUMNS) + [TARGET_LABEL],
    title=f"Latent correlations ({TARGET_LABEL}) – correlation heatmap",
    output_path=overall_corr_heatmap_base,
    correlations=overall_corr,
    p_values=overall_pvals,
)
plt.close(overall_corr_fig)
overall_corr_heatmap_path = overall_corr_heatmap_base.with_suffix(".png")

overall_pval_fig, _overall_pval_ax = plot_feature_latent_correlation_heatmap(
    model,
    X_train_model,
    targets=y_train_model,
    target_name=TARGET_LABEL,
    variables=list(FEATURE_COLUMNS) + [TARGET_LABEL],
    title=f"Latent correlations ({TARGET_LABEL}) – p-value heatmap",
    value="pvalue",
    output_path=overall_pval_heatmap_base,
    correlations=overall_corr,
    p_values=overall_pvals,
)
plt.close(overall_pval_fig)
overall_pval_heatmap_path = overall_pval_heatmap_base.with_suffix(".png")

feature_only_corr = overall_corr.drop(index=TARGET_LABEL, errors="ignore")
has_latent_correlations = not feature_only_corr.empty
if not has_latent_correlations:
    print("Latent-clinical correlation heatmap could not be generated.")
else:
    top_correlated = (
        feature_only_corr.abs().max(axis=1).sort_values(ascending=False).head(10)
    )
    render_dataframe(
        top_correlated.rename("max_abs_correlation")
        .reset_index()
        .rename(columns={"index": "variable"}),
        title="Top latent-clinical correlations (absolute)",
        floatfmt=".3f",
    )

available_features = set(FEATURE_COLUMNS)
latent_group_outputs: list[tuple[str, Path, Path, Path, Path, Path]] = []
for group_name, candidate_columns in VAR_GROUP_DICT.items():
    missing = sorted(set(candidate_columns) - available_features)
    if missing:
        print(f"Skipping unavailable variables for {group_name}: {', '.join(missing)}")
    group_features = [
        column for column in candidate_columns if column in available_features
    ]
    if not group_features:
        continue

    group_base = latent_correlation_base.with_name(
        f"{latent_correlation_base.name}_{group_name}"
    )
    group_corr, group_pvals = compute_feature_latent_correlation(
        model,
        X_train_model,
        targets=y_train_model,
        target_name=TARGET_LABEL,
        variables=group_features + [TARGET_LABEL],
    )
    corr_path = group_base.with_name(f"{group_base.name}_correlations.csv")
    pval_path = group_base.with_name(f"{group_base.name}_pvalues.csv")
    group_corr.to_csv(corr_path)
    group_pvals.to_csv(pval_path)

    bubble_base = group_base.with_name(f"{group_base.name}_bubble")
    corr_heatmap_base = group_base.with_name(f"{group_base.name}_corr_heatmap")
    pval_heatmap_base = group_base.with_name(f"{group_base.name}_pvalue_heatmap")

    bubble_fig, _bubble_ax = plot_feature_latent_correlation_bubble(
        model,
        X_train_model,
        targets=y_train_model,
        target_name=TARGET_LABEL,
        variables=group_features + [TARGET_LABEL],
        title=(
            f"Latent correlations ({TARGET_LABEL}) – "
            f"{group_name.replace('_', ' ').title()} bubble"
        ),
        output_path=bubble_base,
        correlations=group_corr,
        p_values=group_pvals,
    )
    plt.close(bubble_fig)

    corr_fig, _corr_ax = plot_feature_latent_correlation_heatmap(
        model,
        X_train_model,
        targets=y_train_model,
        target_name=TARGET_LABEL,
        variables=group_features + [TARGET_LABEL],
        title=(
            f"Latent correlations ({TARGET_LABEL}) – "
            f"{group_name.replace('_', ' ').title()} correlation"
        ),
        output_path=corr_heatmap_base,
        correlations=group_corr,
        p_values=group_pvals,
    )
    plt.close(corr_fig)

    pval_fig, _pval_ax = plot_feature_latent_correlation_heatmap(
        model,
        X_train_model,
        targets=y_train_model,
        target_name=TARGET_LABEL,
        variables=group_features + [TARGET_LABEL],
        title=(
            f"Latent correlations ({TARGET_LABEL}) – "
            f"{group_name.replace('_', ' ').title()} p-values"
        ),
        value="pvalue",
        output_path=pval_heatmap_base,
        correlations=group_corr,
        p_values=group_pvals,
    )
    plt.close(pval_fig)

    latent_group_outputs.append(
        (
            group_name,
            bubble_base.with_suffix(".png"),
            corr_heatmap_base.with_suffix(".png"),
            pval_heatmap_base.with_suffix(".png"),
            corr_path,
            pval_path,
        )
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
latent_path = VISUALISATIONS_DIR / f"latent_{TARGET_LABEL}.png"
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
if has_latent_correlations:
    summary_lines.append(
        f"Latent-clinical bubble: {overall_bubble_path.relative_to(OUTPUT_DIR)}"
    )
    summary_lines.append(
        f"Latent-clinical correlation heatmap: {overall_corr_heatmap_path.relative_to(OUTPUT_DIR)}"
    )
    summary_lines.append(
        f"Latent-clinical p-value heatmap: {overall_pval_heatmap_path.relative_to(OUTPUT_DIR)}"
    )
    summary_lines.append(
        f"Latent-clinical correlations: {overall_corr_path.relative_to(OUTPUT_DIR)}"
    )
    summary_lines.append(
        f"Latent-clinical p-values: {overall_pval_path.relative_to(OUTPUT_DIR)}"
    )
    if latent_group_outputs:
        summary_lines.append("Latent-clinical group artefacts:")
        for (
            group_name,
            bubble_path,
            corr_heatmap_path,
            pval_heatmap_path,
            corr_path,
            pval_path,
        ) in latent_group_outputs:
            summary_lines.append(
                f"- {group_name}: bubble={bubble_path.relative_to(OUTPUT_DIR)}, "
                f"corr_heatmap={corr_heatmap_path.relative_to(OUTPUT_DIR)}, "
                f"pvalue_heatmap={pval_heatmap_path.relative_to(OUTPUT_DIR)}, "
                f"correlations={corr_path.relative_to(OUTPUT_DIR)}, "
                f"p_values={pval_path.relative_to(OUTPUT_DIR)}"
            )
else:
    summary_lines.append("Latent-clinical visualisations: unavailable")
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
if bootstrap_overall_records_path is not None:
    summary_lines.append(
        f"- Overall bootstrap samples: {bootstrap_overall_records_path.relative_to(OUTPUT_DIR)}"
    )
if bootstrap_per_class_records_path is not None:
    summary_lines.append(
        f"- Per-class bootstrap samples: {bootstrap_per_class_records_path.relative_to(OUTPUT_DIR)}"
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
    if tstr_bootstrap_overall_records_path is not None:
        summary_lines.append(
            "- Overall bootstrap samples: "
            f"{tstr_bootstrap_overall_records_path.relative_to(OUTPUT_DIR)}"
        )
    if tstr_bootstrap_per_class_records_path is not None:
        summary_lines.append(
            "- Per-class bootstrap samples: "
            f"{tstr_bootstrap_per_class_records_path.relative_to(OUTPUT_DIR)}"
        )
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
