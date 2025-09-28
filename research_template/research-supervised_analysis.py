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
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from analysis_config import (
    ANALYSIS_SUBDIRECTORIES,
    BASELINE_DATASET_LABELS,
    BASELINE_DATASET_ORDER,
    BASELINE_MODEL_ABBREVIATIONS,
    BASELINE_MODEL_PIPELINE_BUILDERS,
    BENCHMARK_COLUMNS,
    CLINICAL_SCORE_BENCHMARK_STRATEGY,
    DATA_DIR,
    DATASET_FILENAMES,
    FORCE_UPDATE_FLAG_DEFAULTS,
    PARETO_MAX_ABS_DELTA_AUC,
    PARETO_MIN_VALIDATION_ROAUC,
    RANDOM_STATE,
    TARGET_COLUMNS,
    TARGET_LABEL,
    VALIDATION_SIZE,
    VAR_GROUP_DICT,
    PATH_GRAPH_GROUP_COLORS,
    PATH_GRAPH_NODE_COLORS,
    PATH_GRAPH_NODE_GROUPS,
    PATH_GRAPH_NODE_LABELS,
)

EXAMPLES_DIR = Path(__file__).resolve().parent
if not EXAMPLES_DIR.exists():
    raise RuntimeError(
        "Run this notebook from the repository root so 'examples' is available."
    )
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from analysis_utils import (  # noqa: E402
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
    make_baseline_model_factories,
    make_study_name,
    resolve_model_loading_plan,
    confirm_model_loading_plan_selection,
    resolve_suave_fit_kwargs,
    resolve_analysis_output_root,
    plot_benchmark_curves,
    plot_calibration_curves,
    plot_latent_space,
    plot_transfer_metric_boxes,
    load_tstr_training_sets_from_tsv,
    save_tstr_training_sets_to_tsv,
    collect_transfer_bootstrap_records,
    prepare_features,
    render_dataframe,
    schema_to_dataframe,
    to_numeric_frame,
    load_model_manifest,
    record_model_manifest,
    DISTRIBUTION_SHIFT_OVERALL_NOTE,
    DISTRIBUTION_SHIFT_PER_FEATURE_NOTE,
    _interpret_feature_shift,
    _interpret_global_shift,
)
from cls_eval import (  # noqa: E402
    evaluate_predictions,
    write_results_to_excel_unique,
    export_three_line_tables,
    _format_three_line_ci,
    _write_three_line_workbook,
    _make_three_line_sheet_name,
)

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
IS_INTERACTIVE = is_interactive_session()

DATASET_NAME_MAP = BASELINE_DATASET_LABELS
DATASET_ORDER = tuple(BASELINE_DATASET_ORDER)

TRAIN_DATASET_NAME = DATASET_NAME_MAP.get("train", "Train")
VALIDATION_DATASET_NAME = DATASET_NAME_MAP.get("validation", "Validation")
INTERNAL_TEST_DATASET_NAME = DATASET_NAME_MAP.get("internal_test", "Internal test")
EXTERNAL_DATASET_NAME = DATASET_NAME_MAP.get("external_validation")


CLI_REQUESTED_TRIAL_ID: Optional[int] = None
if not IS_INTERACTIVE:
    CLI_REQUESTED_TRIAL_ID = parse_script_arguments(sys.argv[1:])

if IS_INTERACTIVE:
    FORCE_UPDATE_BENCHMARK_MODEL = False  # Retrain classical baseline models.
    FORCE_UPDATE_TSTR_MODEL = False  # Refit downstream models on TSTR sets.
    FORCE_UPDATE_TRTR_MODEL = False  # Refit downstream models on TRTR sets.
    FORCE_UPDATE_SYNTHETIC_DATA = False  # Regenerate synthetic training TSV artefacts.
    FORCE_UPDATE_C2ST_MODEL = False  # Retrain two-sample test discriminators.
    FORCE_UPDATE_DISTRIBUTION_SHIFT = False  # Refresh distribution-shift analytics.
    FORCE_UPDATE_SUAVE = False  # Reload the persisted SUAVE generator artefact.
    FORCE_UPDATE_BOOTSTRAP = False  # Regenerate global bootstrap summaries.
    FORCE_UPDATE_TSTR_BOOTSTRAP = False  # Recompute cached TSTR bootstrap replicates.
    FORCE_UPDATE_TRTR_BOOTSTRAP = False  # Recompute cached TRTR bootstrap replicates.
else:
    FORCE_UPDATE_BENCHMARK_MODEL = FORCE_UPDATE_FLAG_DEFAULTS.get(
        "FORCE_UPDATE_BENCHMARK_MODEL", False
    )  # Retrain classical baseline models.
    FORCE_UPDATE_TSTR_MODEL = FORCE_UPDATE_FLAG_DEFAULTS.get(
        "FORCE_UPDATE_TSTR_MODEL", False
    )  # Refit downstream models on TSTR sets.
    FORCE_UPDATE_TRTR_MODEL = FORCE_UPDATE_FLAG_DEFAULTS.get(
        "FORCE_UPDATE_TRTR_MODEL", False
    )  # Refit downstream models on TRTR sets.
    FORCE_UPDATE_SYNTHETIC_DATA = FORCE_UPDATE_FLAG_DEFAULTS.get(
        "FORCE_UPDATE_SYNTHETIC_DATA", False
    )  # Regenerate synthetic training TSV artefacts.
    FORCE_UPDATE_C2ST_MODEL = FORCE_UPDATE_FLAG_DEFAULTS.get(
        "FORCE_UPDATE_C2ST_MODEL", False
    )  # Retrain two-sample test discriminators.
    FORCE_UPDATE_DISTRIBUTION_SHIFT = FORCE_UPDATE_FLAG_DEFAULTS.get(
        "FORCE_UPDATE_DISTRIBUTION_SHIFT", False
    )  # Refresh distribution-shift analytics.
    FORCE_UPDATE_SUAVE = FORCE_UPDATE_FLAG_DEFAULTS.get(
        "FORCE_UPDATE_SUAVE", False
    )  # Reload the persisted SUAVE generator artefact.
    FORCE_UPDATE_BOOTSTRAP = FORCE_UPDATE_FLAG_DEFAULTS.get(
        "FORCE_UPDATE_BOOTSTRAP", False
    )  # Regenerate global bootstrap summaries.
    FORCE_UPDATE_TSTR_BOOTSTRAP = FORCE_UPDATE_FLAG_DEFAULTS.get(
        "FORCE_UPDATE_TSTR_BOOTSTRAP", False
    )  # Recompute cached TSTR bootstrap replicates.
    FORCE_UPDATE_TRTR_BOOTSTRAP = FORCE_UPDATE_FLAG_DEFAULTS.get(
        "FORCE_UPDATE_TRTR_BOOTSTRAP", False
    )  # Recompute cached TRTR bootstrap replicates.

INCLUDE_SUAVE_TRANSFER = False


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
    tuple(ANALYSIS_SUBDIRECTORIES.keys()),
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
INTERPRETATION_DIR = analysis_dirs["interpretation"]

analysis_config["optuna_storage"] = (
    f"sqlite:///{OPTUNA_DIR}/{analysis_config['optuna_study_prefix']}_optuna.db"
)

dataset_files = DATASET_FILENAMES

train_file = dataset_files.get("train")
if not train_file:
    raise ValueError(
        "DATASET_FILENAMES['train'] must be set to the training dataset file name."
    )
test_file = dataset_files.get("internal_test")
if not test_file:
    raise ValueError(
        "DATASET_FILENAMES['internal_test'] must be set to the test dataset file name."
    )

train_df = load_dataset(DATA_DIR / train_file)
test_df = load_dataset(DATA_DIR / test_file)

external_file = dataset_files.get("external_validation")
external_df: Optional[pd.DataFrame]
if external_file:
    external_df = load_dataset(DATA_DIR / external_file)
else:
    external_df = None

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

schema = define_schema(
    train_df,
    FEATURE_COLUMNS,
    mode="interactive" if IS_INTERACTIVE else "info",
)

schema_df = schema_to_dataframe(schema).reset_index(drop=True)
render_dataframe(schema_df, title="Schema overview", floatfmt=None)

schema_dir = DATA_SCHEMA_DIR
schema_dir.mkdir(parents=True, exist_ok=True)
schema_excel_path = schema_dir / f"schema_{TARGET_LABEL}.xlsx"
schema_df.to_excel(schema_excel_path, index=False)
print("Saved schema workbook to", schema_excel_path)


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
suave_manifest_signature: Optional[str] = None
if isinstance(model_manifest, Mapping) and model_manifest:
    try:
        suave_manifest_signature = hashlib.sha256(
            json.dumps(model_manifest, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError):
        suave_manifest_signature = None
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

external_features: Optional[pd.DataFrame] = None
external_labels: Optional[pd.Series] = None
if external_df is not None and TARGET_LABEL in external_df.columns:
    external_features = prepare_features(external_df, FEATURE_COLUMNS)
    external_labels = external_df[TARGET_LABEL]

benchmark_frames: Dict[str, pd.DataFrame] = {
    TRAIN_DATASET_NAME: benchmark_train,
    VALIDATION_DATASET_NAME: benchmark_validation,
    INTERNAL_TEST_DATASET_NAME: (
        test_df.loc[:, available_benchmark_columns].reset_index(drop=True)
    ),
}
if external_df is not None and external_features is not None:
    external_benchmark_columns = [
        column for column in available_benchmark_columns if column in external_df.columns
    ]
    if external_benchmark_columns:
        external_name = EXTERNAL_DATASET_NAME or "External validation"
        benchmark_frames[external_name] = (
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
    TRAIN_DATASET_NAME: X_train_model,
    VALIDATION_DATASET_NAME: X_validation,
    INTERNAL_TEST_DATASET_NAME: X_test,
}
if external_features is not None:
    external_name = EXTERNAL_DATASET_NAME or "External validation"
    baseline_feature_frames[external_name] = external_features

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
        reference_key=TRAIN_DATASET_NAME,
    )

(
    baseline_imputed_features,
    baseline_imputed_paths,
    baseline_loaded_from_cache,
) = load_or_create_iteratively_imputed_features(
    baseline_feature_frames,
    output_dir=FEATURE_ENGINEERING_DIR,
    target_label=TARGET_LABEL,
    reference_key=TRAIN_DATASET_NAME,
)

if baseline_loaded_from_cache:
    print("Loaded iterative-imputed baseline features from disk.")
else:
    print("Saved iterative-imputed baseline features:")
    for name, path in baseline_imputed_paths.items():
        print(f"  - {name}: {path}")

baseline_models: Dict[str, Pipeline] = {
    name: builder(RANDOM_STATE)
    for name, builder in BASELINE_MODEL_PIPELINE_BUILDERS.items()
}

baseline_label_sets: Dict[str, pd.Series] = {
    TRAIN_DATASET_NAME: y_train_model,
    VALIDATION_DATASET_NAME: y_validation,
    INTERNAL_TEST_DATASET_NAME: y_test,
}
if external_labels is not None:
    external_name = EXTERNAL_DATASET_NAME or "External validation"
    baseline_label_sets[external_name] = external_labels

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
    model_name: BASELINE_MODEL_ABBREVIATIONS.get(model_name, model_name)
    for model_name in baseline_models.keys()
}
model_abbreviation_lookup["SUAVE"] = BASELINE_MODEL_ABBREVIATIONS.get(
    "SUAVE", "SUAVE"
)

train_features_imputed = baseline_imputed_features[TRAIN_DATASET_NAME]
train_labels = baseline_label_sets[TRAIN_DATASET_NAME]

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

    if EXTERNAL_DATASET_NAME and EXTERNAL_DATASET_NAME not in baseline_evaluation_sets:
        baseline_rows.append(
            {
                "Model": model_name,
                "Dataset": EXTERNAL_DATASET_NAME,
                "Notes": (
                    f"Target not available in {EXTERNAL_DATASET_NAME} split."
                    if EXTERNAL_DATASET_NAME
                    else "Target not available in external split."
                ),
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
    model_manifest = load_model_manifest(SUAVE_MODEL_DIR, TARGET_LABEL)


# %% [markdown]
# ## Prognosis prediction and evaluation
#
# Evaluate the trained model on train/validation/test/eICU cohorts, generate
# calibration curves, and run a membership inference baseline.

# %%

evaluation_datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
    TRAIN_DATASET_NAME: (X_train_model, y_train_model),
    VALIDATION_DATASET_NAME: (X_validation, y_validation),
    INTERNAL_TEST_DATASET_NAME: (X_test, y_test),
}
if external_features is not None and external_labels is not None:
    external_name = EXTERNAL_DATASET_NAME or "External validation"
    evaluation_datasets[external_name] = (external_features, external_labels)

evaluation_cache_path = DATA_SCHEMA_DIR / f"evaluation_datasets_{TARGET_LABEL}.joblib"
joblib.dump(
    {
        "target_label": TARGET_LABEL,
        "feature_columns": list(FEATURE_COLUMNS),
        "datasets": evaluation_datasets,
    },
    evaluation_cache_path,
)
print("Saved evaluation datasets to", evaluation_cache_path)

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

benchmark_datasets = [
    name
    for name in (
        TRAIN_DATASET_NAME,
        INTERNAL_TEST_DATASET_NAME,
        EXTERNAL_DATASET_NAME,
    )
    if name is not None
]
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
            clinical_score_frames.get(TRAIN_DATASET_NAME, benchmark_train)
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
            show_progress=True,
            progress_desc=f"Bootstrap | {model_name} @ {dataset_name}",
        )
        model_results[dataset_name] = results
        dataset_slug = _sanitise_path_component(dataset_name.lower())
        dataset_cache_path = model_dir / f"{dataset_slug}_bootstrap.joblib"

        cached_results: Optional[Dict[str, pd.DataFrame]] = None
        if dataset_cache_path.exists() and not FORCE_UPDATE_BOOTSTRAP:
            try:
                potential_results = joblib.load(dataset_cache_path)
            except Exception as error:
                print(
                    "Failed to load cached bootstrap metrics for",
                    f"{model_name} @ {dataset_name}:",
                    error,
                )
            else:
                required_keys = {"overall", "per_class"}
                if isinstance(potential_results, dict) and required_keys.issubset(
                    potential_results.keys()
                ):
                    cached_results = potential_results
                    print(
                        "Loaded cached bootstrap metrics for",
                        f"{model_name} @ {dataset_name}",
                    )
                else:
                    print(
                        "Discarding cached bootstrap metrics for",
                        f"{model_name} @ {dataset_name} due to missing keys.",
                    )

        if cached_results is None:
            results = evaluate_predictions(
                prediction_df,
                label_col="label",
                pred_col="y_pred",
                positive_label=positive_label_name,
                bootstrap_n=1000,
                random_state=RANDOM_STATE,
                show_progress=True,
                progress_desc=f"Bootstrap | {model_name} @ {dataset_name}",
            )
            joblib.dump(results, dataset_cache_path)
            print("Saved bootstrap metrics to", dataset_cache_path)
        else:
            results = cached_results

        model_results[dataset_name] = results

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
    bootstrap_overall_df = pd.DataFrame(columns=["Target", "Model", "Dataset"])

bootstrap_per_class_df: pd.DataFrame
if bootstrap_per_class_frames:
    bootstrap_per_class_df = pd.concat(bootstrap_per_class_frames, ignore_index=True)
else:
    bootstrap_per_class_df = pd.DataFrame(columns=["Target", "Model", "Dataset"])

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
    bootstrap_warning_df = pd.DataFrame(
        columns=["Target", "Model", "Dataset", "message"]
    )
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

available_datasets = set(bootstrap_summary_df["Dataset"].unique())
required_datasets = [
    name
    for key in DATASET_ORDER
    for name in [DATASET_NAME_MAP.get(key)]
    if name in available_datasets and name is not None
]
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

benchmark_excel_path = BOOTSTRAP_DIR / (
    f"bootstrap_benchmark_{TARGET_LABEL}.xlsx"
)
with pd.ExcelWriter(benchmark_excel_path) as writer:
    bootstrap_summary_df.to_excel(writer, sheet_name="Summary", index=False)
    bootstrap_overall_df.to_excel(writer, sheet_name="overall", index=False)
    bootstrap_per_class_df.to_excel(writer, sheet_name="Perclass", index=False)
    bootstrap_warning_df.to_excel(writer, sheet_name="Warnings", index=False)

print("Saved consolidated bootstrap benchmark workbook to", benchmark_excel_path)

three_line_benchmark_path = benchmark_excel_path.with_name(
    f"report_{benchmark_excel_path.stem}.xlsx"
)

perclass_index_columns = ["Model"]
if "class" in bootstrap_per_class_df.columns:
    perclass_index_columns.append("class")
elif "Class" in bootstrap_per_class_df.columns:
    perclass_index_columns.append("Class")

export_three_line_tables(
    {
        "Summary": bootstrap_summary_df,
        "overall": bootstrap_overall_df,
        "Perclass": bootstrap_per_class_df,
    },
    three_line_benchmark_path,
    index_columns={
        "Summary": ["Model"],
        "overall": ["Model"],
        "Perclass": perclass_index_columns,
    },
    dataset_column="Dataset",
    dataset_order=required_datasets,
    drop_columns=("Target",),
    decimals=3,
    ci_label_text="95%",
)

print("Saved formatted three-line benchmark workbook to", three_line_benchmark_path)
# %% [markdown]
# ## SUAVE latent interpretation
#
# Summarise latent-feature-target relationships before exploring transfer
# learning. Correlation tables and visualisations are written to the
# interpretation directory so analysts can reconcile explainability artefacts
# with downstream evaluations.

# %%

latent_correlation_base = (
    INTERPRETATION_DIR / f"latent_clinical_correlation_{TARGET_LABEL}"
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

latent_features = {
    name: features for name, (features, _) in evaluation_datasets.items()
}
latent_labels = {name: labels for name, (_, labels) in evaluation_datasets.items()}
latent_path = INTERPRETATION_DIR / f"latent_{TARGET_LABEL}.png"
plot_latent_space(
    model,
    latent_features,
    latent_labels,
    target_name=TARGET_LABEL,
    output_path=latent_path,
)
# %% [markdown]
# ## TSTR/TRTR, distribution shift, and privacy checks
#
# Compare models trained on synthetic versus real data while reproducing the
# protocol's distribution shift diagnostics (C2ST, MMD, energy distance,
# mutual information) and membership inference baseline. The workflow applies to
# any configured target when the necessary resources are available.

# %%

tstr_summary_df: Optional[pd.DataFrame] = None
tstr_plot_df: Optional[pd.DataFrame] = None
tstr_summary_path: Optional[Path] = None
tstr_plot_path: Optional[Path] = None
tstr_figure_paths: List[Path] = []
distribution_df: Optional[pd.DataFrame] = None
distribution_overall_df: Optional[pd.DataFrame] = None
distribution_path: Optional[Path] = None
distribution_top: Optional[pd.DataFrame] = None
tstr_nested_results: Optional[
    Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]
] = None
tstr_bootstrap_overall_records_path: Optional[Path] = None
tstr_bootstrap_per_class_records_path: Optional[Path] = None
membership_path = PRIVACY_ASSESSMENT_DIR / "membership_inference.xlsx"

training_cache_dir = TSTR_TRTR_DIR / "training_sets"
training_cache_dir.mkdir(parents=True, exist_ok=True)
bootstrap_cache_dir = TSTR_TRTR_DIR / "bootstrap_cache"
bootstrap_cache_dir.mkdir(parents=True, exist_ok=True)
training_sets_numeric: Optional[Dict[str, Tuple[pd.DataFrame, pd.Series]]] = None
training_sets_raw: Optional[Dict[str, Tuple[pd.DataFrame, pd.Series]]] = None
training_manifest_signature: Optional[str] = None
cached_training_sets: Optional[
    Tuple[
        Dict[str, Tuple[pd.DataFrame, pd.Series]],
        Dict[str, Tuple[pd.DataFrame, pd.Series]],
        str,
    ]
] = None
if not FORCE_UPDATE_SYNTHETIC_DATA:
    cached_training_sets = load_tstr_training_sets_from_tsv(
        training_cache_dir,
        target_label=TARGET_LABEL,
        feature_columns=FEATURE_COLUMNS,
    )

if cached_training_sets is not None:
    training_sets_numeric, training_sets_raw, training_manifest_signature = (
        cached_training_sets
    )
    print(
        "Loaded cached TSTR/TRTR training sets from",
        training_cache_dir,
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
    manifest_path, training_manifest_signature = save_tstr_training_sets_to_tsv(
        training_sets_raw,
        output_dir=training_cache_dir,
        target_label=TARGET_LABEL,
        feature_columns=FEATURE_COLUMNS,
        random_state=RANDOM_STATE,
    )
    print("Saved TSTR/TRTR training sets to", manifest_path)
evaluation_sets_numeric: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
    INTERNAL_TEST_DATASET_NAME: (
        to_numeric_frame(X_test), y_test.reset_index(drop=True)
    ),
}
evaluation_sets_raw: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
    INTERNAL_TEST_DATASET_NAME: (
        X_test.reset_index(drop=True), y_test.reset_index(drop=True)
    ),
}
if external_features is not None and external_labels is not None:
    external_name = EXTERNAL_DATASET_NAME or "External validation"
    evaluation_sets_numeric[external_name] = (
        to_numeric_frame(external_features),
        external_labels.reset_index(drop=True),
    )
    evaluation_sets_raw[external_name] = (
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

training_order = list(training_sets_numeric.keys())
model_order = list(model_factories.keys())


def _is_trtr_dataset(name: str) -> bool:
    return name.upper().startswith("TRTR")


trtr_training_sets_numeric = {
    name: value
    for name, value in training_sets_numeric.items()
    if _is_trtr_dataset(name)
}
trtr_training_sets_raw = {
    name: value
    for name, value in training_sets_raw.items()
    if _is_trtr_dataset(name)
}
tstr_training_sets_numeric = {
    name: value
    for name, value in training_sets_numeric.items()
    if not _is_trtr_dataset(name)
}
tstr_training_sets_raw = {
    name: value
    for name, value in training_sets_raw.items()
    if not _is_trtr_dataset(name)
}

tstr_results_cache_path = TSTR_TRTR_DIR / f"tstr_results_{TARGET_LABEL}.joblib"
trtr_results_cache_path = TSTR_TRTR_DIR / f"trtr_results_{TARGET_LABEL}.joblib"

tstr_plot_df = None
tstr_nested_results = None
tstr_bootstrap_df: Optional[pd.DataFrame] = None
trtr_summary_df: Optional[pd.DataFrame] = None
trtr_plot_df: Optional[pd.DataFrame] = None
trtr_nested_results = None
trtr_bootstrap_df: Optional[pd.DataFrame] = None

if tstr_training_sets_numeric:
    cached_tstr: Optional[Dict[str, Any]] = None
    if tstr_results_cache_path.exists() and not FORCE_UPDATE_TSTR_MODEL:
        payload = joblib.load(tstr_results_cache_path)
        if (
            list(payload.get("feature_columns") or []) == list(FEATURE_COLUMNS)
            and list(payload.get("training_order") or [])
            == list(tstr_training_sets_numeric.keys())
            and list(payload.get("evaluation_order") or [])
            == list(evaluation_sets_numeric.keys())
            and list(payload.get("model_order") or []) == model_order
            and payload.get("training_manifest_signature")
            == training_manifest_signature
        ):
            cached_tstr = payload
            print("Loaded cached TSTR evaluation results from", tstr_results_cache_path)
    if cached_tstr is not None:
        tstr_summary_df = cached_tstr.get("summary_df")
        tstr_plot_df = cached_tstr.get("plot_df")
        tstr_nested_results = cached_tstr.get("nested_results")
        tstr_bootstrap_df = cached_tstr.get("bootstrap_df")
    if (
        tstr_summary_df is None
        or tstr_plot_df is None
        or tstr_nested_results is None
        or tstr_bootstrap_df is None
    ):
        tstr_bootstrap_metadata = {
            "transfer_mode": "TSTR",
            "training_manifest_signature": training_manifest_signature,
            "data_generator_signature": suave_manifest_signature,
        }
        (
            tstr_summary_df,
            tstr_plot_df,
            tstr_nested_results,
        ) = evaluate_transfer_baselines(
            tstr_training_sets_numeric,
            evaluation_sets_numeric,
            model_factories=model_factories,
            bootstrap_n=1000,
            random_state=RANDOM_STATE,
            raw_training_sets=tstr_training_sets_raw,
            raw_evaluation_sets=evaluation_sets_raw,
            bootstrap_cache_dir=bootstrap_cache_dir,
            bootstrap_cache_metadata=tstr_bootstrap_metadata,
            force_update_bootstrap=FORCE_UPDATE_TSTR_BOOTSTRAP,
        )
        tstr_bootstrap_df = collect_transfer_bootstrap_records(tstr_nested_results)
        joblib.dump(
            {
                "summary_df": tstr_summary_df,
                "plot_df": tstr_plot_df,
                "nested_results": tstr_nested_results,
                "bootstrap_df": tstr_bootstrap_df,
                "training_order": list(tstr_training_sets_numeric.keys()),
                "evaluation_order": list(evaluation_sets_numeric.keys()),
                "model_order": model_order,
                "feature_columns": list(FEATURE_COLUMNS),
                "training_manifest_signature": training_manifest_signature,
            },
            tstr_results_cache_path,
        )
        print("Saved TSTR evaluation results to", tstr_results_cache_path)

if trtr_training_sets_numeric:
    cached_trtr: Optional[Dict[str, Any]] = None
    if trtr_results_cache_path.exists() and not FORCE_UPDATE_TRTR_MODEL:
        payload = joblib.load(trtr_results_cache_path)
        if (
            list(payload.get("feature_columns") or []) == list(FEATURE_COLUMNS)
            and list(payload.get("training_order") or [])
            == list(trtr_training_sets_numeric.keys())
            and list(payload.get("evaluation_order") or [])
            == list(evaluation_sets_numeric.keys())
            and list(payload.get("model_order") or []) == model_order
            and payload.get("training_manifest_signature")
            == training_manifest_signature
        ):
            cached_trtr = payload
            print("Loaded cached TRTR evaluation results from", trtr_results_cache_path)
    if cached_trtr is not None:
        trtr_summary_df = cached_trtr.get("summary_df")
        trtr_plot_df = cached_trtr.get("plot_df")
        trtr_nested_results = cached_trtr.get("nested_results")
        trtr_bootstrap_df = cached_trtr.get("bootstrap_df")
    if (
        trtr_summary_df is None
        or trtr_plot_df is None
        or trtr_nested_results is None
        or trtr_bootstrap_df is None
    ):
        trtr_bootstrap_metadata = {
            "transfer_mode": "TRTR",
            "training_manifest_signature": training_manifest_signature,
            "data_generator_signature": suave_manifest_signature,
        }
        (
            trtr_summary_df,
            trtr_plot_df,
            trtr_nested_results,
        ) = evaluate_transfer_baselines(
            trtr_training_sets_numeric,
            evaluation_sets_numeric,
            model_factories=model_factories,
            bootstrap_n=1000,
            random_state=RANDOM_STATE,
            raw_training_sets=trtr_training_sets_raw,
            raw_evaluation_sets=evaluation_sets_raw,
            bootstrap_cache_dir=bootstrap_cache_dir,
            bootstrap_cache_metadata=trtr_bootstrap_metadata,
            force_update_bootstrap=FORCE_UPDATE_TRTR_BOOTSTRAP,
        )
        trtr_bootstrap_df = collect_transfer_bootstrap_records(trtr_nested_results)
        joblib.dump(
            {
                "summary_df": trtr_summary_df,
                "plot_df": trtr_plot_df,
                "nested_results": trtr_nested_results,
                "bootstrap_df": trtr_bootstrap_df,
                "training_order": list(trtr_training_sets_numeric.keys()),
                "evaluation_order": list(evaluation_sets_numeric.keys()),
                "model_order": model_order,
                "feature_columns": list(FEATURE_COLUMNS),
                "training_manifest_signature": training_manifest_signature,
            },
            trtr_results_cache_path,
        )
        print("Saved TRTR evaluation results to", trtr_results_cache_path)

transfer_summary_frames: List[pd.DataFrame] = []
transfer_plot_frames: List[pd.DataFrame] = []
transfer_bootstrap_frames: List[pd.DataFrame] = []

if tstr_summary_df is not None:
    transfer_summary_frames.append(tstr_summary_df.copy())
if trtr_summary_df is not None:
    transfer_summary_frames.append(trtr_summary_df.copy())
if tstr_plot_df is not None:
    transfer_plot_frames.append(tstr_plot_df.copy())
if trtr_plot_df is not None:
    transfer_plot_frames.append(trtr_plot_df.copy())
if tstr_bootstrap_df is not None and not tstr_bootstrap_df.empty:
    transfer_bootstrap_frames.append(tstr_bootstrap_df.copy())
if trtr_bootstrap_df is not None and not trtr_bootstrap_df.empty:
    transfer_bootstrap_frames.append(trtr_bootstrap_df.copy())

combined_summary_df = (
    pd.concat(transfer_summary_frames, ignore_index=True)
    if transfer_summary_frames
    else pd.DataFrame()
)
combined_plot_df = (
    pd.concat(transfer_plot_frames, ignore_index=True)
    if transfer_plot_frames
    else pd.DataFrame()
)
combined_bootstrap_df = (
    pd.concat(transfer_bootstrap_frames, ignore_index=True)
    if transfer_bootstrap_frames
    else pd.DataFrame()
)

if not combined_summary_df.empty:
    render_dataframe(
        combined_summary_df,
        title="TSTR/TRTR supervised evaluation",
        floatfmt=".3f",
    )

transfer_overall_records: List[pd.DataFrame] = []
transfer_per_class_records: List[pd.DataFrame] = []
for nested in (tstr_nested_results, trtr_nested_results):
    if not nested:
        continue
    for training_name, model_map in nested.items():
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

transfer_overall_df = (
    pd.concat(transfer_overall_records, ignore_index=True)
    if transfer_overall_records
    else None
)
transfer_per_class_df = (
    pd.concat(transfer_per_class_records, ignore_index=True)
    if transfer_per_class_records
    else None
)

if not combined_bootstrap_df.empty:
    for evaluation_name in evaluation_sets_numeric.keys():
        for metric_name in ("accuracy", "roc_auc"):
            figure_path = plot_transfer_metric_boxes(
                combined_bootstrap_df,
                metric=metric_name,
                evaluation_dataset=evaluation_name,
                training_order=training_order,
                model_order=model_order,
                output_dir=TSTR_TRTR_DIR,
                target_label=TARGET_LABEL,
            )
            if figure_path is not None:
                tstr_figure_paths.append(figure_path)
                tstr_plot_path = figure_path

tstr_excel_path = TSTR_TRTR_DIR / "TSTR_TRTR_eval.xlsx"
with pd.ExcelWriter(tstr_excel_path) as writer:
    if not combined_summary_df.empty:
        combined_summary_df.to_excel(writer, sheet_name="summary", index=False)
    if not combined_plot_df.empty:
        combined_plot_df.to_excel(writer, sheet_name="metrics", index=False)
    if not combined_bootstrap_df.empty:
        combined_bootstrap_df.to_excel(writer, sheet_name="bootstrap", index=False)
    if tstr_summary_df is not None and not tstr_summary_df.empty:
        tstr_summary_df.to_excel(writer, sheet_name="tstr_summary", index=False)
    if trtr_summary_df is not None and not trtr_summary_df.empty:
        trtr_summary_df.to_excel(writer, sheet_name="trtr_summary", index=False)
    if transfer_overall_df is not None:
        transfer_overall_df.to_excel(
            writer, sheet_name="bootstrap_overall", index=False
        )
    if transfer_per_class_df is not None:
        transfer_per_class_df.to_excel(
            writer, sheet_name="bootstrap_per_class", index=False
        )
print("Saved TSTR/TRTR evaluation workbook to", tstr_excel_path)

summary_three_line_tables: Dict[str, pd.DataFrame] = {}
if not combined_summary_df.empty:
    metric_triplets: List[Tuple[str, str, str]] = []
    for metric in ("accuracy", "roc_auc"):
        low_col = f"{metric}_ci_low"
        high_col = f"{metric}_ci_high"
        if (
            metric in combined_summary_df.columns
            and low_col in combined_summary_df.columns
            and high_col in combined_summary_df.columns
        ):
            metric_triplets.append((metric, low_col, high_col))

    metric_labels: Dict[str, Tuple[str, str]] = {
        "accuracy": ("ACC", "ACC (95% CI)"),
        "roc_auc": ("AUC", "AUC (95% CI)"),
    }

    evaluation_order = [
        name
        for name in combined_summary_df["evaluation_dataset"].dropna().unique()
        if isinstance(name, str)
    ]

    for evaluation_name in evaluation_order:
        evaluation_subset = combined_summary_df[
            combined_summary_df["evaluation_dataset"] == evaluation_name
        ].copy()
        if evaluation_subset.empty:
            continue
        for metric, low_col, high_col in metric_triplets:
            metric_short, metric_label = metric_labels.get(metric, (
                metric.replace("_", " ").upper(),
                f"{metric.replace('_', ' ').upper()} (95% CI)",
            ))
            formatted = evaluation_subset.apply(
                lambda row, m=metric, lo=low_col, hi=high_col: _format_three_line_ci(
                    row.get(m),
                    row.get(lo),
                    row.get(hi),
                    decimals=3,
                    thousand_sep=False,
                    fill_value="NA",
                ),
                axis=1,
            )
            value_frame = evaluation_subset.loc[:, ["training_dataset", "model"]].copy()
            value_frame[metric_label] = formatted
            pivot = value_frame.pivot(
                index="training_dataset", columns="model", values=metric_label
            )
            if pivot.empty:
                continue
            pivot = pivot.reindex(index=training_order).reindex(columns=model_order)
            pivot = pivot.fillna("NA")
            pivot.insert(0, "training_dataset", pivot.index)
            pivot = pivot.reset_index(drop=True)
            column_tuples: List[Tuple[str, str]] = []
            for column in pivot.columns:
                if column == "training_dataset":
                    column_tuples.append(("", "Training dataset"))
                else:
                    column_tuples.append((str(column), metric_label))
            formatted_table = pivot.astype(str)
            formatted_table.columns = pd.MultiIndex.from_tuples(column_tuples)
            sheet_name = _make_three_line_sheet_name(evaluation_name, metric_short)
            if sheet_name in summary_three_line_tables:
                suffix = 2
                while True:
                    candidate = _make_three_line_sheet_name(
                        evaluation_name, metric_short, suffix=suffix
                    )
                    if candidate not in summary_three_line_tables:
                        sheet_name = candidate
                        break
                    suffix += 1
            summary_three_line_tables[sheet_name] = formatted_table

if summary_three_line_tables:
    summary_three_line_path = TSTR_TRTR_DIR / "report_TSTR_TRTR_eval_summary.xlsx"
    _write_three_line_workbook(summary_three_line_tables, summary_three_line_path)
    print(
        "Saved formatted TSTR/TRTR summary workbook to",
        summary_three_line_path,
    )

real_features_numeric = training_sets_numeric.get(
    "TRTR (real)", (pd.DataFrame(), pd.Series(dtype=float))
)[0]
synthesis_features_numeric = training_sets_numeric.get(
    "TSTR synthesis", (pd.DataFrame(), pd.Series(dtype=float))
)[0]

c2st_model_factories = make_baseline_model_factories(RANDOM_STATE)
c2st_cache_path = (
    DISTRIBUTION_SHIFT_DIR / f"c2st_metrics_{TARGET_LABEL}.joblib"
)
c2st_metrics: Optional[Dict[str, float]] = None
c2st_results_df: Optional[pd.DataFrame] = None
expected_c2st_models = list(c2st_model_factories.keys())
if (
    c2st_cache_path.exists()
    and not FORCE_UPDATE_SYNTHETIC_DATA
    and not FORCE_UPDATE_C2ST_MODEL
):
    cached_c2st = joblib.load(c2st_cache_path)
    if (
        list(cached_c2st.get("feature_columns") or [])
        == list(FEATURE_COLUMNS)
        and list(cached_c2st.get("model_order") or []) == expected_c2st_models
    ):
        c2st_metrics = cached_c2st.get("metrics")
        c2st_results_df = cached_c2st.get("results_df")
        print("Loaded cached C2ST metrics from", c2st_cache_path)
    else:
        print(
            "Discarding cached C2ST metrics because the configuration changed."
        )
if c2st_metrics is None:
    c2st_metrics = classifier_two_sample_test(
        real_features_numeric.to_numpy(),
        synthesis_features_numeric.to_numpy(),
        model_factories=c2st_model_factories,
        random_state=RANDOM_STATE,
        n_bootstrap=1000,
    )
    rows: List[Dict[str, object]] = []
    n_real = c2st_metrics.get("n_real_samples", float("nan"))
    n_synth = c2st_metrics.get("n_synthetic_samples", float("nan"))
    for model_name in expected_c2st_models:
        prefix = model_name.lower().replace(" ", "_")
        auc_key = f"{prefix}_auc"
        if auc_key not in c2st_metrics:
            continue
        rows.append(
            {
                "target": TARGET_LABEL,
                "model": model_name,
                "auc": c2st_metrics.get(auc_key, float("nan")),
                "auc_ci_low": c2st_metrics.get(
                    f"{prefix}_auc_ci_low", float("nan")
                ),
                "auc_ci_high": c2st_metrics.get(
                    f"{prefix}_auc_ci_high", float("nan")
                ),
                "bootstrap_samples": c2st_metrics.get(
                    f"{prefix}_bootstrap_samples", float("nan")
                ),
                "n_real_samples": n_real,
                "n_synthetic_samples": n_synth,
            }
        )
    c2st_results_df = pd.DataFrame(rows)
    if not c2st_results_df.empty:
        c2st_results_df = c2st_results_df[
            [
                "target",
                "model",
                "auc",
                "auc_ci_low",
                "auc_ci_high",
                "bootstrap_samples",
                "n_real_samples",
                "n_synthetic_samples",
            ]
        ]
    joblib.dump(
        {
            "feature_columns": list(FEATURE_COLUMNS),
            "model_order": expected_c2st_models,
            "metrics": c2st_metrics,
            "results_df": c2st_results_df,
        },
        c2st_cache_path,
    )
    print("Saved C2ST metrics to", c2st_cache_path)
if c2st_results_df is None:
    c2st_results_df = pd.DataFrame()
render_dataframe(
    c2st_results_df,
    title="Classifier two-sample test (C2ST)",
    floatfmt=".3f",
)

c2st_workbook_path = DISTRIBUTION_SHIFT_DIR / "c2st_metrics.xlsx"
with pd.ExcelWriter(c2st_workbook_path) as writer:
    c2st_results_df.to_excel(writer, sheet_name="metrics", index=False)
print("Saved C2ST results to", c2st_workbook_path)


distribution_cache_path = (
    DISTRIBUTION_SHIFT_DIR / f"distribution_metrics_{TARGET_LABEL}.joblib"
)
distribution_overall_df: Optional[pd.DataFrame] = None
distribution_df: Optional[pd.DataFrame] = None
if (
    distribution_cache_path.exists()
    and not FORCE_UPDATE_SYNTHETIC_DATA
    and not FORCE_UPDATE_DISTRIBUTION_SHIFT
):
    cached_distribution = joblib.load(distribution_cache_path)
    if (
        list(cached_distribution.get("feature_columns") or [])
        == list(FEATURE_COLUMNS)
    ):
        distribution_overall_df = cached_distribution.get("overall_df")
        distribution_df = cached_distribution.get("per_feature_df")
        print("Loaded cached distribution-shift metrics from", distribution_cache_path)
    else:
        print(
            "Discarding cached distribution-shift metrics because the configuration changed."
        )
if distribution_overall_df is None or distribution_df is None:
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
    feature_rows: List[Dict[str, object]] = []
    feature_iterator = tqdm(
        FEATURE_COLUMNS,
        desc="Distribution shift | per feature",
        leave=False,
    )
    for column in feature_iterator:
        feature_iterator.set_postfix_str(column)
        real_values = real_features_numeric[column].to_numpy()
        synthetic_values = synthesis_features_numeric[column].to_numpy()
        mmd_value, _ = rbf_mmd(
            real_values,
            synthetic_values,
            random_state=RANDOM_STATE,
            n_permutations=0,
        )
        energy_value, _ = energy_distance(
            real_values,
            synthetic_values,
            random_state=RANDOM_STATE,
            n_permutations=0,
        )
        mi_value = mutual_information_feature(real_values, synthetic_values)
        feature_rows.append(
            {
                "feature": column,
                "rbf_mmd": mmd_value,
                "energy_distance": energy_value,
                "mutual_information": mi_value,
            }
        )
    distribution_df = pd.DataFrame(feature_rows)
    average_mutual_info = (
        float(distribution_df["mutual_information"].mean())
        if not distribution_df.empty
        else float("nan")
    )
    overall_rows = [
        {
            "metric": "rbf_mmd",
            "value": global_mmd,
            "p_value": global_mmd_p_value,
            "interpretation": _interpret_global_shift(
                "rbf_mmd", global_mmd, global_mmd_p_value
            ),
        },
        {
            "metric": "energy_distance",
            "value": global_energy,
            "p_value": global_energy_p_value,
            "interpretation": _interpret_global_shift(
                "energy_distance", global_energy, global_energy_p_value
            ),
        },
        {
            "metric": "mutual_information",
            "value": average_mutual_info,
            "p_value": float("nan"),
            "interpretation": _interpret_global_shift(
                "mutual_information", average_mutual_info, float("nan")
            ),
        },
    ]
    distribution_overall_df = pd.DataFrame(overall_rows)
    joblib.dump(
        {
            "feature_columns": list(FEATURE_COLUMNS),
            "overall_df": distribution_overall_df,
            "per_feature_df": distribution_df,
        },
        distribution_cache_path,
    )
    print("Saved distribution-shift metrics to", distribution_cache_path)

feature_interpretation_rows: List[Dict[str, object]] = []
if distribution_df is not None:
    for row in distribution_df.itertuples():
        feature_interpretation_rows.append(
            {
                "feature": row.feature,
                "rbf_mmd": row.rbf_mmd,
                "energy_distance": row.energy_distance,
                "mutual_information": row.mutual_information,
                "interpretation": _interpret_feature_shift(
                    row.rbf_mmd, row.energy_distance, row.mutual_information
                ),
            }
        )
distribution_feature_df = pd.DataFrame(feature_interpretation_rows)
if distribution_overall_df is None:
    distribution_overall_df = pd.DataFrame(
        columns=["metric", "value", "p_value", "interpretation"]
    )

render_dataframe(
    distribution_overall_df,
    title="Distribution shift overview",
    floatfmt=".3f",
)
distribution_path = DISTRIBUTION_SHIFT_DIR / "distribution_metrics.xlsx"
with pd.ExcelWriter(distribution_path, engine="openpyxl") as writer:
    distribution_overall_df.to_excel(writer, sheet_name="overall", index=False)
    overall_sheet = writer.sheets["overall"]
    overall_note_row = distribution_overall_df.shape[0] + 3
    overall_sheet.cell(
        row=overall_note_row,
        column=1,
        value=DISTRIBUTION_SHIFT_OVERALL_NOTE,
    )

    distribution_feature_df.to_excel(writer, sheet_name="per_feature", index=False)
    feature_sheet = writer.sheets["per_feature"]
    feature_note_row = distribution_feature_df.shape[0] + 3
    feature_sheet.cell(
        row=feature_note_row,
        column=1,
        value=DISTRIBUTION_SHIFT_PER_FEATURE_NOTE,
    )
if not distribution_feature_df.empty:
    distribution_top = (
        distribution_feature_df.sort_values("mutual_information", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
else:
    distribution_top = distribution_feature_df
render_dataframe(
    distribution_top,
    title="Top distribution shift features (mutual information)",
    floatfmt=".3f",
)

train_probabilities = probability_map[TRAIN_DATASET_NAME]
test_probabilities = probability_map[INTERNAL_TEST_DATASET_NAME]
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
    summary_lines.append(
        f"Latent-clinical path graph: {overall_path_graph_path.relative_to(OUTPUT_DIR)}"
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
    summary_lines.append("Latent-clinical interpretation assets: unavailable")
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
