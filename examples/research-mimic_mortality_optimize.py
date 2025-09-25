# %% [markdown]
# # MIMIC mortality – hyperparameter optimisation
#
# This script runs Optuna to tune SUAVE hyperparameters for the selected
# mortality target and fits an isotonic calibrator on the internal validation
# split. The resulting artefacts (model checkpoint, calibrator, and Optuna
# summary) are written to the analysis output directory for downstream
# evaluation.

# %%

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EXAMPLES_DIR = Path(__file__).resolve().parent
if not EXAMPLES_DIR.exists():
    raise RuntimeError(
        "Run this script from the repository so the 'examples' directory is available."
    )
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from mimic_mortality_utils import (  # noqa: E402
    HIDDEN_DIMENSION_OPTIONS,
    HEAD_HIDDEN_DIMENSION_OPTIONS,
    RANDOM_STATE,
    TARGET_COLUMNS,
    BENCHMARK_COLUMNS,
    VALIDATION_SIZE,
    Schema,
    build_suave_model,
    compute_binary_metrics,
    define_schema,
    fit_isotonic_calibrator,
    load_dataset,
    make_logistic_pipeline,
    prepare_features,
    render_dataframe,
    schema_to_dataframe,
    to_numeric_frame,
)

from suave.evaluate import evaluate_tstr, evaluate_trtr  # noqa: E402

try:
    import optuna
except ImportError as exc:  # pragma: no cover - optuna provided via requirements
    raise RuntimeError(
        "Optuna is required for the mortality optimisation. Install it via 'pip install optuna'."
    ) from exc


# %% [markdown]
# ## Analysis configuration
#
# Configure the analysis outputs and Optuna storage. Artefacts are written to
# ``analysis_outputs_supervised`` so they can be reused by the downstream
# evaluation script.

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

# %%

DATA_DIR = (EXAMPLES_DIR / "data" / "sepsis_mortality_dataset").resolve()
OUTPUT_DIR = EXAMPLES_DIR / analysis_config["output_dir_name"]
OUTPUT_DIR.mkdir(exist_ok=True)
analysis_config["optuna_storage"] = (
    f"sqlite:///{OUTPUT_DIR}/{analysis_config['optuna_study_prefix']}_optuna.db"
)

train_df = load_dataset(DATA_DIR / "mimic-mortality-train.tsv")
test_df = load_dataset(DATA_DIR / "mimic-mortality-test.tsv")

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
schema.update(
    {
        "BMI": {"type": "real"},
        "Respiratory_Support": {"type": "ordinal", "n_classes": 5},
        "LYM%": {"type": "real"},
    }
)

schema_df = schema_to_dataframe(schema).sort_values("Column").reset_index(drop=True)
render_dataframe(schema_df, title="Schema overview", floatfmt=None)


# %% [markdown]
# ## Optuna study helper


def run_optuna_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    *,
    feature_columns: Sequence[str],
    schema: Schema,
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

    def objective(trial: "optuna.trial.Trial") -> Tuple[float, float]:
        trial.suggest_categorical("latent_dim", [8, 16, 24, 32, 48, 64])
        trial.suggest_categorical("n_components", [1, 2, 4, 6])
        trial.suggest_categorical("hidden_dims", list(HIDDEN_DIMENSION_OPTIONS.keys()))
        trial.suggest_categorical(
            "head_hidden_dims", list(HEAD_HIDDEN_DIMENSION_OPTIONS.keys())
        )
        trial.suggest_float("beta", 0.1, 6.0)
        trial.suggest_categorical("use_classification_loss_weight", [True, False])
        if trial.params.get("use_classification_loss_weight"):
            trial.suggest_float("classification_loss_weight", 1.0, 200.0, log=True)
        trial.suggest_float("dropout", 0.0, 0.7)
        trial.suggest_float("learning_rate", 1e-5, 5e-2, log=True)
        trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])
        trial.suggest_int("warmup_epochs", 1, 100)
        trial.suggest_int("kl_warmup_epochs", 0, 80)
        trial.suggest_int("head_epochs", 1, 80)
        trial.suggest_int("finetune_epochs", 1, 60)
        trial.suggest_int("early_stop_patience", 3, 30)
        trial.suggest_float("joint_decoder_lr_scale", 1e-4, 1.0, log=True)

        model = build_suave_model(trial.params, schema, random_state=random_state)

        start_time = time.perf_counter()
        model.fit(
            X_train,
            y_train,
            warmup_epochs=int(trial.params.get("warmup_epochs", 3)),
            kl_warmup_epochs=int(trial.params.get("kl_warmup_epochs", 0)),
            head_epochs=int(trial.params.get("head_epochs", 2)),
            finetune_epochs=int(trial.params.get("finetune_epochs", 2)),
            joint_decoder_lr_scale=float(
                trial.params.get("joint_decoder_lr_scale", 0.1)
            ),
            early_stop_patience=int(trial.params.get("early_stop_patience", 10)),
        )
        fit_seconds = time.perf_counter() - start_time
        validation_probs = model.predict_proba(X_validation)
        validation_metrics = compute_binary_metrics(validation_probs, y_validation)
        trial.set_user_attr("validation_metrics", validation_metrics)
        trial.set_user_attr("fit_seconds", fit_seconds)

        roauc = validation_metrics.get("ROAUC", float("nan"))
        if not np.isfinite(roauc):
            raise optuna.exceptions.TrialPruned("Non-finite validation ROAUC")

        try:
            numeric_train = to_numeric_frame(X_train.loc[:, feature_columns])
            numeric_validation = to_numeric_frame(X_validation.loc[:, feature_columns])

            rng = np.random.default_rng(random_state + trial.number)
            synthetic_labels = rng.choice(y_train, size=len(y_train), replace=True)
            synthetic_samples = model.sample(
                len(synthetic_labels), conditional=True, y=synthetic_labels
            )
            if not isinstance(synthetic_samples, pd.DataFrame):
                synthetic_features = pd.DataFrame(
                    synthetic_samples, columns=feature_columns
                )
            else:
                synthetic_features = synthetic_samples.loc[:, feature_columns].copy()
            numeric_synthetic = to_numeric_frame(synthetic_features)

            tstr_metrics = evaluate_tstr(
                (
                    numeric_synthetic.to_numpy(),
                    np.asarray(synthetic_labels),
                ),
                (
                    numeric_validation.to_numpy(),
                    y_validation.to_numpy(),
                ),
                make_logistic_pipeline,
            )
            trtr_metrics = evaluate_trtr(
                (
                    numeric_train.to_numpy(),
                    y_train.to_numpy(),
                ),
                (
                    numeric_validation.to_numpy(),
                    y_validation.to_numpy(),
                ),
                make_logistic_pipeline,
            )
            trial.set_user_attr("tstr_metrics", tstr_metrics)
            trial.set_user_attr("trtr_metrics", trtr_metrics)

            delta_auc = abs(
                float(trtr_metrics.get("auroc", float("nan")))
                - float(tstr_metrics.get("auroc", float("nan")))
            )
            if not np.isfinite(delta_auc):
                raise ValueError("Non-finite TSTR/TRTR delta AUC")
        except Exception as error:
            trial.set_user_attr("tstr_trtr_error", repr(error))
            raise optuna.exceptions.TrialPruned(
                f"Failed to compute TSTR/TRTR delta AUC: {error}"
            ) from error

        trial.set_user_attr("tstr_trtr_delta_auc", delta_auc)
        return roauc, delta_auc

    study = optuna.create_study(
        directions=("maximize", "minimize"),
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage and study_name),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    feasible_trials = [trial for trial in study.trials if trial.values is not None]
    if not feasible_trials:
        raise RuntimeError("Optuna search did not produce any completed trials")

    def sort_key(trial: "optuna.trial.Trial") -> Tuple[float, float]:
        values = trial.values or (float("nan"), float("inf"))
        primary = values[0]
        secondary = values[1]
        return (primary, -secondary if np.isfinite(secondary) else float("-inf"))

    best_trial = max(feasible_trials, key=sort_key)
    best_attributes: Dict[str, object] = {
        "trial_number": best_trial.number,
        "values": tuple(best_trial.values or ()),
        "params": dict(best_trial.params),
        "validation_metrics": best_trial.user_attrs.get("validation_metrics", {}),
        "fit_seconds": best_trial.user_attrs.get("fit_seconds"),
        "tstr_metrics": best_trial.user_attrs.get("tstr_metrics", {}),
        "trtr_metrics": best_trial.user_attrs.get("trtr_metrics", {}),
        "tstr_trtr_delta_auc": best_trial.user_attrs.get("tstr_trtr_delta_auc"),
    }
    return study, best_attributes


# %% [markdown]
# ## Train/validation split for optimisation

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


# %% [markdown]
# ## Execute Optuna search

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
    feature_columns=FEATURE_COLUMNS,
    schema=schema,
    random_state=RANDOM_STATE,
    n_trials=analysis_config["optuna_trials"],
    timeout=analysis_config["optuna_timeout"],
    study_name=study_name,
    storage=analysis_config["optuna_storage"],
)

trial_rows: list[dict[str, object]] = []
for trial in study.trials:
    values = trial.values or (float("nan"), float("nan"))
    trial_rows.append(
        {
            "trial_number": trial.number,
            "validation_roauc": values[0],
            "tstr_trtr_delta_auc": values[1],
            **trial.params,
        }
    )

optuna_trials_df = pd.DataFrame(trial_rows)
optuna_trials_path = OUTPUT_DIR / f"optuna_trials_{TARGET_LABEL}.csv"
if not optuna_trials_df.empty:
    optuna_trials_df.to_csv(optuna_trials_path, index=False)
else:
    optuna_trials_path.write_text("trial_number,value")


# %% [markdown]
# ## Serialise Optuna artefacts

# %%


def _json_ready(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(val) for val in value]
    return value


best_params = dict(optuna_best_info.get("params", {}))
best_info_path = OUTPUT_DIR / f"optuna_best_info_{TARGET_LABEL}.json"
best_params_path = OUTPUT_DIR / f"optuna_best_params_{TARGET_LABEL}.json"
best_info_path.write_text(
    json.dumps(_json_ready(optuna_best_info), indent=2, ensure_ascii=False)
)
best_params_path.write_text(
    json.dumps(_json_ready(best_params), indent=2, ensure_ascii=False)
)


# %% [markdown]
# ## Train best SUAVE model and calibrator

# %%

if not best_params:
    raise RuntimeError(
        "Optuna did not return best parameters; cannot train final model"
    )

model = build_suave_model(best_params, schema, random_state=RANDOM_STATE)

model.fit(
    X_train_model,
    y_train_model,
    warmup_epochs=int(best_params.get("warmup_epochs", 3)),
    kl_warmup_epochs=int(best_params.get("kl_warmup_epochs", 0)),
    head_epochs=int(best_params.get("head_epochs", 2)),
    finetune_epochs=int(best_params.get("finetune_epochs", 2)),
    joint_decoder_lr_scale=float(best_params.get("joint_decoder_lr_scale", 0.1)),
    early_stop_patience=int(best_params.get("early_stop_patience", 10)),
)

isotonic_calibrator = fit_isotonic_calibrator(model, X_validation, y_validation)

model_path = OUTPUT_DIR / f"suave_best_{TARGET_LABEL}.pt"
calibrator_path = OUTPUT_DIR / f"isotonic_calibrator_{TARGET_LABEL}.joblib"

model.save(model_path)
joblib.dump(isotonic_calibrator, calibrator_path)


# %% [markdown]
# ## Summary

# %%

summary_lines = [
    "# Optimisation summary",
    f"Target label: {TARGET_LABEL}",
    f"Best trial: #{optuna_best_info.get('trial_number')}",
    f"Validation ROAUC: {optuna_best_info.get('values', (float('nan'),))[0]:.4f}",
    f"TSTR/TRTR ΔAUC: {optuna_best_info.get('values', (float('nan'), float('nan')))[1]:.4f}",
    "\nBest parameters:",
    json.dumps(_json_ready(best_params), indent=2, ensure_ascii=False),
    "",
    "Saved artefacts:",
    f"- Optuna trials: {optuna_trials_path.relative_to(OUTPUT_DIR)}",
    f"- Best info: {best_info_path.relative_to(OUTPUT_DIR)}",
    f"- Best params: {best_params_path.relative_to(OUTPUT_DIR)}",
    f"- SUAVE model: {model_path.relative_to(OUTPUT_DIR)}",
    f"- Isotonic calibrator: {calibrator_path.relative_to(OUTPUT_DIR)}",
]

summary_path = OUTPUT_DIR / f"optimisation_summary_{TARGET_LABEL}.md"
summary_path.write_text("\n".join(summary_lines))
render_dataframe(
    pd.DataFrame([_json_ready(optuna_best_info.get("validation_metrics", {}))]),
    title="Best validation metrics",
)
