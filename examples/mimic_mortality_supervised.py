# %% [markdown]
# # MIMIC mortality (supervised)
# 
# This notebook reproduces the supervised SUAVE mortality analysis with Optuna-based hyperparameter tuning.

# %%

import sys
import json
from pathlib import Path
import time
from typing import Dict, List, Mapping, Optional, Tuple

from IPython.display import display
from tabulate import tabulate

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

EXAMPLES_DIR = Path().resolve()
if not EXAMPLES_DIR.exists():
    raise RuntimeError("Run this notebook from the repository root so 'examples' is available.")
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from mimic_mortality_utils import (
    RANDOM_STATE,
    TARGET_COLUMNS,
    CALIBRATION_SIZE,
    VALIDATION_SIZE,
    Schema,
    define_schema,

    kolmogorov_smirnov_statistic,
    load_dataset,
    mutual_information_feature,
    prepare_features,
    rbf_mmd,
    split_train_validation_calibration,
    to_numeric_frame,
)
from cls_eval import evaluate_predictions, write_results_to_excel_unique

from suave import SUAVE
from suave.evaluate import evaluate_tstr, evaluate_trtr, simple_membership_inference

try:
    import optuna
except ImportError as exc:  # pragma: no cover - optuna provided via requirements
    raise RuntimeError(
        "Optuna is required for the mortality analysis. Install it via 'pip install optuna'."
    ) from exc


def is_interactive_session() -> bool:
    """Return ``True`` when running inside an interactive IPython session."""

    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return get_ipython() is not None


def render_dataframe(
    df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    floatfmt: Optional[str] = ".3f",
) -> None:
    """Display ``df`` using ``display`` when interactive, otherwise print via tabulate."""

    if title:
        print(title)
    if df.empty:
        print("(empty table)")
        return
    if is_interactive_session():
        display(df)
    else:
        tabulate_kwargs = {"headers": "keys", "tablefmt": "github", "showindex": False}
        if floatfmt is not None:
            tabulate_kwargs["floatfmt"] = floatfmt
        print(tabulate(df, **tabulate_kwargs))


def dataframe_to_markdown(df: pd.DataFrame, *, floatfmt: Optional[str] = ".3f") -> str:
    """Return a GitHub-flavoured Markdown table representing ``df``."""

    if df.empty:
        return "_No data available._"
    tabulate_kwargs = {"headers": "keys", "tablefmt": "github", "showindex": False}
    if floatfmt is not None:
        tabulate_kwargs["floatfmt"] = floatfmt
    return tabulate(df, **tabulate_kwargs)


def schema_to_dataframe(schema: Schema) -> pd.DataFrame:
    """Convert a :class:`Schema` into a tidy :class:`pandas.DataFrame`."""

    schema_records: List[Dict[str, object]] = []
    for column, spec in schema.to_dict().items():
        schema_records.append(
            {
                "Column": column,
                "Type": spec.get("type", ""),
                "n_classes": spec.get("n_classes", ""),
                "y_dim": spec.get("y_dim", ""),
            }
        )
    return pd.DataFrame(schema_records)


#%% [markdown]
# ## Analysis configuration
#
# Define the label to model and tuning/runtime parameters. Setting the label up
# front avoids looping over every possible prediction task so that the analysis
# remains focused on a single clinical question.

# %%

# Select the target label to model. Choose from the columns listed in
# ``TARGET_COLUMNS`` loaded from ``mimic_mortality_utils``.
TARGET_LABEL = "in_hospital_mortality"

# Configuration for Optuna search and output artifacts.
analysis_config = {
    "optuna_trials": 60,
    "optuna_timeout": 3600 * 48,
    "optuna_study_prefix": "supervised",
    "optuna_storage": None,
    "output_dir_name": "analysis_outputs_supervised",
}

#%% [markdown]
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

FEATURE_COLUMNS = [column for column in train_df.columns if column not in TARGET_COLUMNS]
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

HIDDEN_DIMENSION_OPTIONS: Dict[str, Tuple[int, int]] = {
    "compact": (96, 48),
    "small": (128, 64),
    "medium": (256, 128),
    "wide": (384, 192),
    "extra_wide": (512, 256),
}

HEAD_HIDDEN_DIMENSION_OPTIONS: Dict[str, Tuple[int, int]] = {
    "compact": (32,),
    "small": (48,),
    "medium": (48, 32),
    "wide": (96, 48, 16),
    "extra_wide": (64, 128, 64, 16),
}
def make_logistic_pipeline() -> Pipeline:
    """Factory for the baseline classifier used in TSTR/TRTR."""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
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
        latent_dim = trial.suggest_categorical("latent_dim", [6, 8, 16, 32])
        hidden_key = trial.suggest_categorical("hidden_dims", list(HIDDEN_DIMENSION_OPTIONS.keys()))
        head_hidden_key = trial.suggest_categorical("head_hidden_dims", list(HEAD_HIDDEN_DIMENSION_OPTIONS.keys()))
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 2e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
        beta = trial.suggest_float("beta", 0.5, 4.0)
        warmup_epochs = trial.suggest_int("warmup_epochs", 2, 60)
        head_epochs = trial.suggest_int("head_epochs", 1, 50)
        finetune_epochs = trial.suggest_int("finetune_epochs", 3, 20)
        joint_decoder_lr_scale = trial.suggest_float("joint_decoder_lr_scale", 1e-3, 0.3)

        model = SUAVE(
            schema=schema,
            latent_dim=latent_dim,
            hidden_dims=HIDDEN_DIMENSION_OPTIONS[hidden_key],
            head_hidden_dims=HEAD_HIDDEN_DIMENSION_OPTIONS[head_hidden_key],
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            beta=beta,
            random_state=random_state,
            behaviour="supervised",
        )

        start_time = time.perf_counter()
        model.fit(
            X_train,
            y_train,
            warmup_epochs=warmup_epochs,
            head_epochs=head_epochs,
            finetune_epochs=finetune_epochs,
            joint_decoder_lr_scale=joint_decoder_lr_scale,
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


#%% [markdown]
# ## Prepare modelling datasets
#
# Create train/validation/calibration splits for the selected label. The split
# mirrors the original notebook so that Optuna tuning and calibration operate on
# disjoint subsets.

# %%

X_full = prepare_features(train_df, FEATURE_COLUMNS)
y_full = train_df[TARGET_LABEL]

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


#%% [markdown]
# ## Classical model benchmarks
#
# Fit a suite of scikit-learn classifiers as quick baselines before training
# SUAVE. These provide a reference point for MIMIC test performance and, when
# available, eICU external validation.

# %%

baseline_models = {
    "Logistic regression": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
            ),
        ]
    ),
    "KNN": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=25)),
        ]
    ),
    "Decision tree": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    ),
    "Random forest": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
            ),
        ]
    ),
    "SVM (RBF)": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True)),
        ]
    ),
}

baseline_evaluation_sets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
    "MIMIC test": (X_test, y_test)
}
if external_features is not None and external_labels is not None:
    baseline_evaluation_sets["eICU external"] = (external_features, external_labels)

baseline_rows: List[Dict[str, object]] = []
metric_columns = ["AUC", "ACC", "SPE", "SEN", "Brier"]

for model_name, estimator in baseline_models.items():
    fitted_estimator = estimator.fit(X_train_model, y_train_model)
    for dataset_name, (features, labels) in baseline_evaluation_sets.items():
        probabilities = fitted_estimator.predict_proba(features)
        metrics = compute_binary_metrics(probabilities, labels)
        row = {
            "Model": model_name,
            "Dataset": dataset_name,
            "Notes": "",
        }
        row.update({column: metrics.get(column, float("nan")) for column in metric_columns})
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


#%% [markdown]
# ## Hyperparameter search with Optuna
#
# Optimise SUAVE hyperparameters on the training/validation split. The trial
# history is persisted to CSV for later inspection.

# %%

study_name = (
    f"{analysis_config['optuna_study_prefix']}_{TARGET_LABEL}"
    if analysis_config["optuna_study_prefix"]
    else None
)
study, optuna_best_info = run_optuna_search(
    X_train_model,
    y_train_model,
    X_validation,
    y_validation,
    schema,
    random_state=RANDOM_STATE,
    n_trials=analysis_config["optuna_trials"],
    timeout=analysis_config["optuna_timeout"],
    study_name=study_name,
    storage=analysis_config["optuna_storage"],
)

optuna_best_params = dict(optuna_best_info.get("params", {}))

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

optuna_trials_df = pd.DataFrame(trial_rows)
optuna_trials_path = OUTPUT_DIR / f"optuna_trials_{TARGET_LABEL}.csv"
if not optuna_trials_df.empty:
    optuna_trials_df.to_csv(optuna_trials_path, index=False)
else:
    optuna_trials_path.write_text("trial_number,value")


#%% [markdown]
# ## Model training
#
# Instantiate SUAVE with the best Optuna configuration and fit/calibrate on the
# appropriate subsets.

# %%

hidden_key = str(optuna_best_params.get("hidden_dims", "medium"))
head_hidden_key = str(optuna_best_params.get("head_hidden_dims", "medium"))
hidden_dims = HIDDEN_DIMENSION_OPTIONS.get(hidden_key, HIDDEN_DIMENSION_OPTIONS["medium"])
head_hidden_dims = HEAD_HIDDEN_DIMENSION_OPTIONS.get(
    head_hidden_key, HEAD_HIDDEN_DIMENSION_OPTIONS["medium"]
)

model = SUAVE(
    schema=schema,
    latent_dim=int(optuna_best_params.get("latent_dim", 16)),
    hidden_dims=hidden_dims,
    head_hidden_dims=head_hidden_dims,
    dropout=float(optuna_best_params.get("dropout", 0.1)),
    learning_rate=float(optuna_best_params.get("learning_rate", 1e-3)),
    batch_size=int(optuna_best_params.get("batch_size", 256)),
    beta=float(optuna_best_params.get("beta", 1.5)),
    random_state=RANDOM_STATE,
    behaviour="supervised",
)

model.fit(
    X_train_model,
    y_train_model,
    warmup_epochs=int(optuna_best_params.get("warmup_epochs", 3)),
    head_epochs=int(optuna_best_params.get("head_epochs", 2)),
    finetune_epochs=int(optuna_best_params.get("finetune_epochs", 2)),
    joint_decoder_lr_scale=float(optuna_best_params.get("joint_decoder_lr_scale", 0.1)),
)
model.calibrate(X_calibration, y_calibration)


#%% [markdown]
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
existing_columns = [column for column in ordered_metric_columns if column in metrics_df.columns]
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


#%% [markdown]
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


def build_prediction_dataframe(
    probabilities: np.ndarray,
    labels: pd.Series | np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Assemble a dataframe compatible with :func:`evaluate_predictions`."""

    prob_matrix = np.asarray(probabilities)
    if prob_matrix.ndim == 1:
        if len(class_name_strings) == 2:
            negative_name, positive_name = class_name_strings[0], class_name_strings[-1]
            proba_dict = {
                f"pred_proba_{negative_name}": 1.0 - prob_matrix,
                f"pred_proba_{positive_name}": prob_matrix,
            }
        else:
            proba_dict = {"pred_proba_0": prob_matrix}
    else:
        if prob_matrix.shape[1] == len(class_name_strings) and len(class_name_strings) > 0:
            proba_dict = {
                f"pred_proba_{class_name_strings[idx]}": prob_matrix[:, idx]
                for idx in range(prob_matrix.shape[1])
            }
        else:
            proba_dict = {
                f"pred_proba_{idx}": prob_matrix[:, idx]
                for idx in range(prob_matrix.shape[1])
            }

    base_df = pd.DataFrame(
        {
            "label": np.asarray(labels),
            "y_pred": np.asarray(predictions),
        }
    )
    if proba_dict:
        proba_df = pd.DataFrame(proba_dict)
        base_df = pd.concat([base_df.reset_index(drop=True), proba_df], axis=1)
    else:
        base_df = base_df.reset_index(drop=True)
    return base_df


for dataset_name, (features, labels) in evaluation_datasets.items():
    dataset_predictions = model.predict(features)
    dataset_probabilities = probability_map[dataset_name]
    prediction_df = build_prediction_dataframe(
        dataset_probabilities,
        labels,
        dataset_predictions,
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


#%% [markdown]
# ## TSTR/TRTR comparison
#
# Compare models trained on synthetic versus real data. The analysis is only
# relevant when the current task models in-hospital mortality, matching the
# publication results.

# %%

tstr_results: Optional[pd.DataFrame] = None
tstr_path: Optional[Path] = None
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
    numeric_train = to_numeric_frame(X_full)

    rng = np.random.default_rng(RANDOM_STATE)
    synthetic_labels = rng.choice(y_full, size=len(y_full), replace=True)

    synthetic_features = model.sample(
        len(synthetic_labels), conditional=True, y=synthetic_labels
    )
    numeric_synthetic = to_numeric_frame(synthetic_features[FEATURE_COLUMNS])

    numeric_test = to_numeric_frame(X_test)

    tstr_metrics = evaluate_tstr(
        (numeric_synthetic.to_numpy(), np.asarray(synthetic_labels)),
        (numeric_test.to_numpy(), y_test.to_numpy()),
        make_logistic_pipeline,
    )
    trtr_metrics = evaluate_trtr(
        (numeric_train.to_numpy(), y_full.to_numpy()),
        (numeric_test.to_numpy(), y_test.to_numpy()),
        make_logistic_pipeline,
    )
    tstr_results = pd.DataFrame(
        [
            {"setting": "TSTR", **tstr_metrics},
            {"setting": "TRTR", **trtr_metrics},
        ]
    )
    tstr_path = OUTPUT_DIR / "tstr_trtr_comparison.csv"
    tstr_results.to_csv(tstr_path, index=False)
    render_dataframe(tstr_results, title="TSTR vs TRTR", floatfmt=".3f")

    distribution_rows: List[Dict[str, object]] = []
    for column in FEATURE_COLUMNS:
        real_values = numeric_train[column].to_numpy()
        synthetic_values = numeric_synthetic[column].to_numpy()
        distribution_rows.append(
            {
                "feature": column,
                "ks": kolmogorov_smirnov_statistic(real_values, synthetic_values),
                "mmd": rbf_mmd(real_values, synthetic_values, random_state=RANDOM_STATE),
                "mutual_information": mutual_information_feature(
                    real_values, synthetic_values
                ),
            }
        )
    distribution_df = pd.DataFrame(distribution_rows)
    distribution_path = OUTPUT_DIR / "distribution_shift_metrics.csv"
    distribution_df.to_csv(distribution_path, index=False)
    distribution_top = (
        distribution_df.sort_values("ks", ascending=False).head(10).reset_index(drop=True)
    )
    render_dataframe(
        distribution_top,
        title="Top distribution shift features (KS)",
        floatfmt=".3f",
    )


#%% [markdown]
# ## Latent space interpretation
#
# Project latent representations using PCA for qualitative assessment of class
# separation across cohorts.

# %%

latent_features = {name: features for name, (features, _) in evaluation_datasets.items()}
latent_labels = {name: labels for name, (_, labels) in evaluation_datasets.items()}
latent_path = OUTPUT_DIR / f"latent_{TARGET_LABEL}.png"
plot_latent_space(
    model,
    latent_features,
    latent_labels,
    target_name=TARGET_LABEL,
    output_path=latent_path,
)


#%% [markdown]
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

best_value = optuna_best_info.get("value")
value_text = f"{best_value:.4f}" if isinstance(best_value, (int, float)) else "n/a"
summary_lines.append(
    f"Best Optuna trial #{optuna_best_info.get('trial_number')} with validation ROAUC {value_text}"
)
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
summary_lines.append(
    f"Calibration plot: {calibration_path.relative_to(OUTPUT_DIR)}"
)
summary_lines.append(
    f"Latent projection: {latent_path.relative_to(OUTPUT_DIR)}"
)
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

if tstr_results is not None and tstr_path is not None:
    summary_lines.append("## TSTR vs TRTR")
    tstr_summary_df = tstr_results.rename(columns={"setting": "Setting"})
    summary_lines.append(dataframe_to_markdown(tstr_summary_df, floatfmt=".3f"))
    summary_lines.append("")

summary_lines.append("## Distribution shift and privacy")
if distribution_df is not None and distribution_path is not None:
    base_distribution_df = (
        distribution_top if distribution_top is not None else distribution_df
    )
    distribution_summary_df = base_distribution_df.rename(
        columns={
            "feature": "Feature",
            "ks": "KS",
            "mmd": "MMD",
            "mutual_information": "Mutual information",
        }
    )
    distribution_columns = [
        "Feature",
        "KS",
        "MMD",
        "Mutual information",
    ]
    existing_distribution_columns = [
        column for column in distribution_columns if column in distribution_summary_df.columns
    ]
    if existing_distribution_columns:
        distribution_summary_df = distribution_summary_df.loc[:, existing_distribution_columns]
    distribution_summary_df = distribution_summary_df.reset_index(drop=True)
    summary_lines.append("Top 10 features by KS statistic:")
    summary_lines.append(dataframe_to_markdown(distribution_summary_df, floatfmt=".3f"))
    summary_lines.append(
        f"Full distribution metrics: {distribution_path.relative_to(OUTPUT_DIR)}"
    )
else:
    summary_lines.append("Distribution metrics were not computed.")

if membership_df.empty:
    summary_lines.append("No membership inference metrics were recorded.")
else:
    membership_summary_df = membership_df.rename(
        columns={
            "target": "Target",
            "attack_auc": "Attack AUC",
            "attack_best_accuracy": "Best accuracy",
            "attack_best_threshold": "Threshold",
            "attack_majority_class_accuracy": "Majority baseline",
        }
    )
    membership_columns = [
        "Target",
        "Attack AUC",
        "Best accuracy",
        "Threshold",
        "Majority baseline",
    ]
    existing_membership_columns = [
        column for column in membership_columns if column in membership_summary_df.columns
    ]
    if existing_membership_columns:
        membership_summary_df = membership_summary_df.loc[
            :, existing_membership_columns
        ]
    membership_summary_df = membership_summary_df.reset_index(drop=True)
    summary_lines.append("Membership inference results:")
    summary_lines.append(dataframe_to_markdown(membership_summary_df, floatfmt=".3f"))
    summary_lines.append(
        f"Membership metrics saved to: {membership_path.relative_to(OUTPUT_DIR)}"
    )

summary_path = OUTPUT_DIR / "summary.md"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

print("Analysis complete.")
print(f"Metric table saved to {metrics_path}")
print(f"Calibration plot saved to {calibration_path}")
print(f"Latent space plot saved to {latent_path}")
print(f"Membership inference results saved to {membership_path}")
if tstr_results is not None and tstr_path is not None and distribution_path is not None:
    print(f"TSTR/TRTR comparison saved to {tstr_path}")
    print(f"Distribution metrics saved to {distribution_path}")
print(f"Summary written to {summary_path}")


# %%



