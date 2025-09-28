# %% [markdown]
# # MIMIC mortality – hyperparameter optimisation
#
# This script runs Optuna to tune SUAVE hyperparameters for the selected
# mortality target and fits an isotonic calibrator on the internal validation
# split. The resulting artefacts (model checkpoint, calibrator, and Optuna
# summary) are written to the research output directory for downstream
# evaluation. The workflow is identical whether executed interactively or as a
# command-line script.

# %%

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
    PARETO_MAX_ABS_DELTA_AUC,
    PARETO_MIN_VALIDATION_ROAUC,
    DATA_DIR,
    build_analysis_config,
    choose_preferred_pareto_trial,
    RANDOM_STATE,
    TARGET_COLUMNS,
    BENCHMARK_COLUMNS,
    VALIDATION_SIZE,
    Schema,
    build_suave_model,
    collect_manual_and_optuna_overview,
    define_schema,
    evaluate_candidate_model_performance,
    fit_isotonic_calibrator,
    is_interactive_session,
    load_dataset,
    load_manual_model_manifest,
    load_manual_tuning_overrides,
    load_optuna_results,
    make_logistic_pipeline,
    prepare_features,
    prepare_analysis_output_directories,
    prompt_manual_override_action,
    resolve_analysis_output_root,
    render_dataframe,
    schema_to_dataframe,
    to_numeric_frame,
    record_model_manifest,
    run_manual_override_training,
    _save_figure_multiformat,
    make_study_name,
    resolve_suave_fit_kwargs,
)

from suave.evaluate import evaluate_tstr, evaluate_trtr  # noqa: E402

try:
    import optuna
    from optuna.visualization import matplotlib as optuna_visualisation
except ImportError as exc:  # pragma: no cover - optuna provided via requirements
    raise RuntimeError(
        "Optuna is required for the mortality optimisation. Install it via 'pip install optuna'."
    ) from exc


# %% [markdown]
# ## Analysis configuration
#
# Configure the analysis outputs and Optuna storage. Artefacts are written to
# ``research_outputs_supervised`` so they can be reused by the downstream
# evaluation script.

# %%

TARGET_LABEL = "in_hospital_mortality"

analysis_config = build_analysis_config()

IS_INTERACTIVE = is_interactive_session()


# %% [markdown]
# ## Data loading and schema definition

# %%

OUTPUT_DIR = resolve_analysis_output_root(analysis_config["output_dir_name"])

analysis_dirs = prepare_analysis_output_directories(
    OUTPUT_DIR,
    (
        "optuna",
        "suave_model",
        "calibration_uncertainty",
    ),
)

OPTUNA_DIR = analysis_dirs["optuna"]
SUAVE_MODEL_DIR = analysis_dirs["suave_model"]
CALIBRATION_DIR = analysis_dirs["calibration_uncertainty"]

analysis_config["optuna_storage"] = (
    f"sqlite:///{OPTUNA_DIR}/{analysis_config['optuna_study_prefix']}_optuna.db"
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
schema = define_schema(train_df, FEATURE_COLUMNS, mode="info")
schema.update(
    {
        "age": {"type": "real"},
    }
)

schema_df = schema_to_dataframe(schema).reset_index(drop=True)
render_dataframe(schema_df, title="Schema overview", floatfmt=None)


# %% [markdown]
# ## Optuna study helper


def _target_accessor(index: int) -> Callable[["optuna.trial.FrozenTrial"], float]:
    """Return a safe accessor for the multi-objective study targets."""

    def _target(trial: "optuna.trial.FrozenTrial") -> float:
        if trial.values is None or index >= len(trial.values):
            raise optuna.exceptions.TrialPruned(
                "Trial lacks the requested objective value"
            )
        value = trial.values[index]
        if value is None or not np.isfinite(value):
            raise optuna.exceptions.TrialPruned("Objective value is not finite")
        return float(value)

    return _target


def _save_optuna_figure(figure_or_axes: "plt.Figure", output_path: Path) -> Path:
    """Persist Optuna diagnostic figures in multiple formats."""

    # Optuna's Matplotlib helpers return ``Axes`` instances while our utility
    # expects a ``Figure``. Normalise the object so saving succeeds regardless
    # of the exact return type.
    figure = getattr(figure_or_axes, "figure", figure_or_axes)
    if not hasattr(figure, "savefig"):
        raise TypeError(
            "Unsupported Matplotlib object returned from Optuna visualisation"
        )

    _save_figure_multiformat(figure, output_path.with_suffix(""))
    plt.close(figure)
    return output_path.with_suffix(".jpg")


def _generate_optuna_diagnostics(
    study: "optuna.study.Study",
    *,
    target_label: str,
    output_dir: Path,
) -> Mapping[str, Path]:
    """Create Optuna diagnostic plots summarising the optimisation dynamics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics: Dict[str, Path] = {}

    objective_specs = [
        ("validation_roauc", _target_accessor(0), "Validation ROAUC"),
        ("tstr_trtr_delta_auc", _target_accessor(1), "TSTR/TRTR ΔAUC"),
    ]

    for suffix, target_fn, display_name in objective_specs:
        try:
            importance_fig = optuna_visualisation.plot_param_importances(
                study, target=target_fn, target_name=display_name
            )
        except Exception as error:
            print(f"Skipping parameter importance for {display_name}: {error}")
        else:
            base_path = output_dir / f"param_importance_{suffix}_{target_label}"
            diagnostics[f"param_importance_{suffix}"] = _save_optuna_figure(
                importance_fig, base_path
            )

        try:
            history_fig = optuna_visualisation.plot_optimization_history(
                study, target=target_fn, target_name=display_name
            )
        except Exception as error:
            print(f"Skipping optimisation history for {display_name}: {error}")
        else:
            base_path = output_dir / f"optimization_history_{suffix}_{target_label}"
            diagnostics[f"optimization_history_{suffix}"] = _save_optuna_figure(
                history_fig, base_path
            )

    try:
        pareto_fig = optuna_visualisation.plot_pareto_front(
            study, target_names=[spec[2] for spec in objective_specs]
        )
    except Exception as error:
        print(f"Skipping Pareto front plot: {error}")
    else:
        base_path = output_dir / f"pareto_front_{target_label}"
        diagnostics["pareto_front"] = _save_optuna_figure(pareto_fig, base_path)

    return diagnostics


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
    target_label: str,
    diagnostics_dir: Path,
) -> tuple[
    "optuna.study.Study",
    Dict[str, object],
    list[Dict[str, object]],
    Mapping[str, Path],
]:
    """Perform Optuna hyperparameter optimisation for :class:`SUAVE`."""

    if n_trials is not None and n_trials <= 0:
        n_trials = None
    if timeout is not None and timeout <= 0:
        timeout = None

    def objective(trial: "optuna.trial.Trial") -> Tuple[float, float]:
        trial.suggest_categorical("latent_dim", [6, 8, 16, 24, 32, 48, 64])
        trial.suggest_categorical("n_components", [1, 2, 3, 4, 5, 6, 7, 8])
        trial.suggest_categorical("hidden_dims", list(HIDDEN_DIMENSION_OPTIONS.keys()))
        trial.suggest_categorical(
            "head_hidden_dims", list(HEAD_HIDDEN_DIMENSION_OPTIONS.keys())
        )
        trial.suggest_float("beta", 0.5, 6.0)
        trial.suggest_categorical("use_classification_loss_weight", [True, False])
        if trial.params.get("use_classification_loss_weight"):
            trial.suggest_float("classification_loss_weight", 1.0, 1000.0, log=True)
        trial.suggest_float("dropout", 0.0, 0.5)
        trial.suggest_float("learning_rate", 1e-5, 5e-2, log=True)
        trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
        trial.suggest_int("warmup_epochs", 2, 60)
        trial.suggest_int("kl_warmup_epochs", 0, 20)
        trial.suggest_int("head_epochs", 10, 80)
        trial.suggest_int("finetune_epochs", 1, 30)
        trial.suggest_int("early_stop_patience", 3, 8)
        trial.suggest_float("joint_decoder_lr_scale", 1e-3, 0.3, log=True)

        model = build_suave_model(trial.params, schema, random_state=random_state)

        start_time = time.perf_counter()
        fit_kwargs = resolve_suave_fit_kwargs(trial.params)
        model.fit(
            X_train,
            y_train,
            **fit_kwargs,
        )
        fit_seconds = time.perf_counter() - start_time
        try:
            evaluation = evaluate_candidate_model_performance(
                model,
                feature_columns=feature_columns,
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                random_state=random_state + trial.number,
            )
        except Exception as error:
            trial.set_user_attr("tstr_trtr_error", repr(error))
            raise optuna.exceptions.TrialPruned(
                f"Failed to evaluate candidate model: {error}"
            ) from error

        validation_metrics = evaluation["validation_metrics"]
        tstr_metrics = evaluation["tstr_metrics"]
        trtr_metrics = evaluation["trtr_metrics"]
        delta_auc = evaluation["delta_auc"]
        values = evaluation["values"]

        trial.set_user_attr("validation_metrics", validation_metrics)
        trial.set_user_attr("fit_seconds", fit_seconds)
        trial.set_user_attr("tstr_metrics", tstr_metrics)
        trial.set_user_attr("trtr_metrics", trtr_metrics)
        trial.set_user_attr("tstr_trtr_delta_auc", delta_auc)

        if not np.isfinite(values[0]):
            raise optuna.exceptions.TrialPruned("Non-finite validation ROAUC")
        if not np.isfinite(values[1]):
            raise optuna.exceptions.TrialPruned("Non-finite TSTR/TRTR delta AUC")

        return float(values[0]), float(values[1])

    study = optuna.create_study(
        directions=("maximize", "minimize"),
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage and study_name),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    diagnostics = _generate_optuna_diagnostics(
        study, target_label=target_label, output_dir=diagnostics_dir
    )

    feasible_trials = [trial for trial in study.trials if trial.values is not None]
    if not feasible_trials:
        raise RuntimeError("Optuna search did not produce any completed trials")

    pareto_trials = [trial for trial in study.best_trials if trial.values is not None]
    if not pareto_trials:
        pareto_trials = feasible_trials

    def _trial_objectives(trial: "optuna.trial.FrozenTrial") -> Tuple[float, float]:
        values = trial.values or (float("nan"), float("nan"))
        primary = float(values[0])
        secondary = float(values[1]) if len(values) > 1 else float("nan")
        return primary, secondary

    best_trial = choose_preferred_pareto_trial(
        pareto_trials,
        min_validation_roauc=PARETO_MIN_VALIDATION_ROAUC,
        max_abs_delta_auc=PARETO_MAX_ABS_DELTA_AUC,
    )
    if best_trial is None:
        best_trial = max(pareto_trials, key=lambda trial: _trial_objectives(trial)[0])

    def _trial_metadata(trial: "optuna.trial.FrozenTrial") -> Dict[str, object]:
        metadata: Dict[str, object] = {
            "trial_number": trial.number,
            "values": tuple(trial.values or ()),
            "params": dict(trial.params),
            "validation_metrics": trial.user_attrs.get("validation_metrics", {}),
            "fit_seconds": trial.user_attrs.get("fit_seconds"),
            "tstr_metrics": trial.user_attrs.get("tstr_metrics", {}),
            "trtr_metrics": trial.user_attrs.get("trtr_metrics", {}),
            "tstr_trtr_delta_auc": trial.user_attrs.get("tstr_trtr_delta_auc"),
        }
        if "tstr_trtr_error" in trial.user_attrs:
            metadata["tstr_trtr_error"] = trial.user_attrs["tstr_trtr_error"]
        return metadata

    pareto_metadata = [_trial_metadata(trial) for trial in pareto_trials]
    best_metadata = next(
        (
            metadata
            for metadata in pareto_metadata
            if metadata.get("trial_number") == best_trial.number
        ),
        _trial_metadata(best_trial),
    )
    best_metadata["diagnostic_paths"] = {
        name: str(path) for name, path in diagnostics.items()
    }
    if best_metadata not in pareto_metadata:
        pareto_metadata.append(best_metadata)

    return study, best_metadata, pareto_metadata, diagnostics


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

manual_config = analysis_config.get("interactive_manual_tuning", {})
manual_action = "optuna"

if IS_INTERACTIVE and manual_config:
    try:
        manual_summary, manual_ranked = collect_manual_and_optuna_overview(
            target_label=TARGET_LABEL,
            model_dir=SUAVE_MODEL_DIR,
            optuna_dir=OPTUNA_DIR,
            study_prefix=analysis_config.get("optuna_study_prefix"),
            storage=analysis_config.get("optuna_storage"),
        )
    except Exception as error:  # pragma: no cover - diagnostic aid
        print(f"Failed to prepare manual tuning overview: {error}")
        manual_summary = pd.DataFrame()
        manual_ranked = pd.DataFrame()

    if not manual_summary.empty:
        render_dataframe(
            manual_summary,
            title="Manual override and Pareto summary",
            floatfmt=".4f",
        )
    if not manual_ranked.empty:
        render_dataframe(
            manual_ranked,
            title="Manual/Pareto candidates (top 50)",
            floatfmt=".4f",
        )

    manual_action = prompt_manual_override_action()

    if manual_action == "train":
        manual_overrides = load_manual_tuning_overrides(manual_config, SUAVE_MODEL_DIR)
        _base_info, base_params = load_optuna_results(
            OPTUNA_DIR,
            TARGET_LABEL,
            study_prefix=analysis_config.get("optuna_study_prefix"),
            storage=analysis_config.get("optuna_storage"),
        )
        if not manual_overrides and not base_params:
            print(
                "Manual overrides were empty and no Optuna parameters were found; continuing with Optuna search."
            )
            manual_action = "optuna"
        else:
            try:
                manual_result = run_manual_override_training(
                    target_label=TARGET_LABEL,
                    manual_overrides=manual_overrides,
                    base_params=base_params,
                    schema=schema,
                    feature_columns=FEATURE_COLUMNS,
                    X_train=X_train_model,
                    y_train=y_train_model,
                    X_validation=X_validation,
                    y_validation=y_validation,
                    model_dir=SUAVE_MODEL_DIR,
                    calibration_dir=CALIBRATION_DIR,
                    random_state=RANDOM_STATE,
                )
            except Exception as error:  # pragma: no cover - diagnostic path
                print(f"Manual override training failed: {error}")
                manual_action = "optuna"
            else:
                validation_value, delta_value = manual_result["values"]
                print(
                    "\nManual override training summary:"
                    f"\n  Validation ROAUC: {validation_value:.4f}"
                    f"\n  TSTR/TRTR ΔAUC: {delta_value:.4f}"
                )
                if manual_result["params"]:
                    print("Applied hyper-parameters:")
                    for key, value in sorted(manual_result["params"].items()):
                        print(f"  - {key}: {value}")
                print(
                    f"Saved manual SUAVE model to {manual_result['model_path']}."
                )
                print(
                    "Saved manual calibrator to"
                    f" {manual_result['calibrator_path']}."
                )
                print(
                    f"Updated manual manifest at {manual_result['manifest_path']}."
                )
                raise SystemExit(0)

    if manual_action == "reuse":
        manual_manifest = load_manual_model_manifest(SUAVE_MODEL_DIR, TARGET_LABEL)
        if manual_manifest:
            print(
                "Manual SUAVE artefacts detected on disk; Optuna search will be skipped. "
                "Run the evaluation script to load the manual override."
            )
            raise SystemExit(0)
        print(
            "Manual manifest was not found; continuing with Optuna search."
        )

study_name = make_study_name(analysis_config["optuna_study_prefix"], TARGET_LABEL)

study, optuna_best_info, optuna_pareto_info, optuna_diagnostics = run_optuna_search(
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
    target_label=TARGET_LABEL,
    diagnostics_dir=OPTUNA_DIR / "figures",
)

best_trial_values = optuna_best_info.get("values", (float("nan"), float("nan")))
print(
    "\nBest Optuna trial:"
    f" #{optuna_best_info.get('trial_number', 'N/A')}"
    f" | Validation ROAUC: {best_trial_values[0]:.4f}"
    f" | TSTR/TRTR ΔAUC: {best_trial_values[1]:.4f}"
)
if optuna_best_info.get("params"):
    print("Selected hyperparameters:")
    for key, value in sorted(optuna_best_info["params"].items()):
        print(f"  - {key}: {value}")

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
optuna_trials_path = OPTUNA_DIR / f"optuna_trials_{TARGET_LABEL}.csv"
if not optuna_trials_df.empty:
    optuna_trials_df.to_csv(optuna_trials_path, index=False)
    top_trials = (
        optuna_trials_df.sort_values("validation_roauc", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    render_dataframe(
        top_trials,
        title="Top Optuna trials by validation ROAUC",
    )
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
best_info_path = OPTUNA_DIR / f"optuna_best_info_{TARGET_LABEL}.json"
best_params_path = OPTUNA_DIR / f"optuna_best_params_{TARGET_LABEL}.json"

preferred_trial_number = optuna_best_info.get("trial_number")

pareto_info_payload: list[dict[str, object]] = []
for entry in optuna_pareto_info:
    enriched_entry = dict(entry)
    enriched_entry["is_preferred"] = (
        entry.get("trial_number") == preferred_trial_number
    )
    pareto_info_payload.append(enriched_entry)

best_info_payload = {
    "preferred_trial_number": preferred_trial_number,
    "preferred_trial": optuna_best_info,
    "pareto_front": pareto_info_payload,
}

pareto_params_payload: list[dict[str, object]] = []
for entry in optuna_pareto_info:
    params_dict = dict(entry.get("params", {}))
    pareto_params_payload.append(
        {
            "trial_number": entry.get("trial_number"),
            "params": params_dict,
            "is_preferred": entry.get("trial_number") == preferred_trial_number,
        }
    )

best_params_payload = {
    "preferred_trial_number": preferred_trial_number,
    "preferred_params": best_params,
    "pareto_front": pareto_params_payload,
}

best_info_path.write_text(
    json.dumps(_json_ready(best_info_payload), indent=2, ensure_ascii=False)
)
best_params_path.write_text(
    json.dumps(_json_ready(best_params_payload), indent=2, ensure_ascii=False)
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
    **resolve_suave_fit_kwargs(best_params),
)

isotonic_calibrator = fit_isotonic_calibrator(model, X_validation, y_validation)

model_path = SUAVE_MODEL_DIR / f"suave_best_{TARGET_LABEL}.pt"
calibrator_path = CALIBRATION_DIR / f"isotonic_calibrator_{TARGET_LABEL}.joblib"

model.save(model_path)
joblib.dump(isotonic_calibrator, calibrator_path)

manifest_path = record_model_manifest(
    SUAVE_MODEL_DIR,
    TARGET_LABEL,
    trial_number=optuna_best_info.get("trial_number"),
    values=best_trial_values,
    params=best_params,
    model_path=model_path,
    calibrator_path=calibrator_path,
    study_name=study.study_name,
    storage=analysis_config.get("optuna_storage"),
)


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
    f"- Pareto front size: {len(optuna_pareto_info)} trials",
    f"- Pareto front info: {best_info_path.relative_to(OUTPUT_DIR)}",
    f"- Pareto front params: {best_params_path.relative_to(OUTPUT_DIR)}",
    f"- Model manifest: {manifest_path.relative_to(OUTPUT_DIR)}",
    f"- SUAVE model: {model_path.relative_to(OUTPUT_DIR)}",
    f"- Isotonic calibrator: {calibrator_path.relative_to(OUTPUT_DIR)}",
]

if optuna_diagnostics:
    summary_lines.append("Optuna diagnostics:")
    for name, figure_path in sorted(optuna_diagnostics.items()):
        summary_lines.append(
            f"- {name.replace('_', ' ').title()}: {figure_path.relative_to(OUTPUT_DIR)}"
        )

summary_path = OPTUNA_DIR / f"optimisation_summary_{TARGET_LABEL}.md"
summary_path.write_text("\n".join(summary_lines))
render_dataframe(
    pd.DataFrame([_json_ready(optuna_best_info.get("validation_metrics", {}))]),
    title="Best validation metrics",
)
