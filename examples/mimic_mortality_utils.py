"""Shared utilities for the MIMIC mortality modelling examples."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import math
import os
import sys
from datetime import datetime, timezone
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import joblib
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from tabulate import tabulate
from tqdm.auto import tqdm
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    roc_curve,
)
from sklearn.pipeline import Pipeline


# =============================================================================
# === Configuration constants and global options ==============================
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from suave import Schema, SchemaInferencer, SUAVE  # noqa: E402
from suave.evaluate import (  # noqa: E402
    compute_auroc,
    evaluate_classification,
    evaluate_tstr,
    evaluate_trtr,
    kolmogorov_smirnov_statistic,
    mutual_information_feature,
    rbf_mmd,
)
from cls_eval import evaluate_predictions  # noqa: E402


# Interpretation notes appended to distribution-shift workbooks so users can
# review the heuristics applied by the helper functions below.
DISTRIBUTION_SHIFT_OVERALL_NOTE = (
    "Interpretation guide: rbf_mmd and energy_distance use permutation tests; "
    "treat p < 0.05 as evidence of a significant distribution difference. "
    "Mutual_information lacks a permutation p-value, so assess magnitudes "
    "against domain expectations."
)
DISTRIBUTION_SHIFT_PER_FEATURE_NOTE = (
    "Interpretation guide: feature-level heuristics highlight potential shifts "
    "when rbf_mmd > 0.05, energy_distance > 0.1, or mutual_information > 0.1."
)


RANDOM_STATE: int = 20201021
DEFAULT_PLOT_THEME: Optional[str] = "paper"
PLOT_LATIN_FONT_FAMILY: str = "Times New Roman"
PLOT_CHINESE_FONT_FAMILY: Optional[str] = "Microsoft YaHei"
TARGET_COLUMNS: Tuple[str, str] = ("in_hospital_mortality", "28d_mortality")
BENCHMARK_COLUMNS = (
    "APS_III",
    "APACHE_IV",
    "SAPS_II",
    "OASIS",
)  # do not include in training. Only use for benchamrk validation.

#: Strategy for evaluating clinical score benchmarks.
#: ``"imputed"`` (default) applies iterative imputation before evaluation, any
#: other value keeps observed scores and skips rows with missing values. Keep
#: this comment in sync with ``analysis_config.py``.
CLINICAL_SCORE_BENCHMARK_STRATEGY: str = "imputed"

VALIDATION_SIZE: float = 0.2

DATA_DIR: Path = (EXAMPLES_DIR / "data" / "sepsis_mortality_dataset").resolve()

# Thresholds governing which Optuna trials are considered viable for persistence.
PARETO_MIN_VALIDATION_ROAUC: float = 0.81
PARETO_MAX_ABS_DELTA_AUC: float = 0.035

HIDDEN_DIMENSION_OPTIONS: Dict[str, Tuple[int, ...]] = {
    "lean": (64, 32),
    "compact": (96, 48),
    "small": (128, 64),
    "medium": (256, 128),
    "wide": (384, 192),
    "extra_wide": (512, 256),
    "ultra_wide": (640, 320),
}

HEAD_HIDDEN_DIMENSION_OPTIONS: Dict[str, Tuple[int, ...]] = {
    "minimal": (16,),
    "compact": (32,),
    "small": (48,),
    "medium": (48, 32),
    "wide": (96, 48, 16),
    "extra_wide": (64, 128, 64, 16),
    "deep": (128, 64, 32),
}

INTERACTIVE_MANUAL_TUNING: Dict[str, object] = {
    "module": "manual_param_setting",
    "attribute": "manual_param_setting",
    # ``override_on_history`` toggles whether manual overrides build on the
    # persisted Optuna parameters (``True``) or start from SUAVE defaults (``False``).
    "override_on_history": False,
}


DEFAULT_ANALYSIS_CONFIG: Dict[str, object] = {
    "optuna_trials": 50,
    "optuna_timeout": 3600 * 48,
    "optuna_study_prefix": "supervised",
    "optuna_storage": None,
    "output_dir_name": "research_outputs_supervised",
    "plot_theme": DEFAULT_PLOT_THEME,
    "tstr_models": ("Logistic regression",),
    "tstr_metric_labels": {
        "accuracy": "Accuracy",
        "roc_auc": "AUROC",
        "delta_accuracy": "ΔAccuracy",
        "delta_roc_auc": "ΔAUROC",
    },
    "training_color_palette": None,
    "interactive_manual_tuning": INTERACTIVE_MANUAL_TUNING,
}

#: Default script-mode flags that determine whether cached artefacts should be
#: regenerated. Interactive runs always keep these set to ``False``; CLI
#: executions adopt the values below. ``FORCE_UPDATE_SUAVE`` is only consulted
#: when Optuna artefacts are unavailable, allowing callers to refresh the
#: locally persisted SUAVE model that otherwise acts as a fallback.
FORCE_UPDATE_FLAG_DEFAULTS: Dict[str, bool] = {
    "FORCE_UPDATE_BENCHMARK_MODEL": True,  # Retrain cached classical baselines.
    "FORCE_UPDATE_TSTR_MODEL": True,  # Refit downstream models on TSTR sets.
    "FORCE_UPDATE_TRTR_MODEL": True,  # Refit downstream models on TRTR sets.
    "FORCE_UPDATE_SYNTHETIC_DATA": True,  # Regenerate synthetic training TSV artefacts.
    "FORCE_UPDATE_C2ST_MODEL": True,  # Retrain two-sample test discriminators.
    "FORCE_UPDATE_DISTRIBUTION_SHIFT": True,  # Refresh distribution-shift analytics.
    "FORCE_UPDATE_SUAVE": False,  # Reload the persisted SUAVE generator artefact.
    "FORCE_UPDATE_BOOTSTRAP": True,  # Regenerate global bootstrap summaries.
    "FORCE_UPDATE_TSTR_BOOTSTRAP": True,  # Recompute cached TSTR bootstrap replicates.
    "FORCE_UPDATE_TRTR_BOOTSTRAP": True,  # Recompute cached TRTR bootstrap replicates.
}

ANALYSIS_SUBDIRECTORIES: Dict[str, str] = {
    "data_schema": "01_data_and_schema",
    "feature_engineering": "02_feature_engineering",
    "optuna": "03_optuna_search",
    "suave_model": "04_suave_training",
    "calibration_uncertainty": "05_calibration_uncertainty",
    "evaluation_reports": "06_evaluation_metrics",
    "bootstrap_analysis": "07_bootstrap_analysis",
    "baseline_models": "08_baseline_models",
    "interpretation": "09_interpretation",
    "tstr_trtr": "10_tstr_trtr_transfer",
    "distribution_shift": "11_distribution_shift",
    "privacy_assessment": "12_privacy_assessment",
}


# =============================================================================
# === Plot theming utilities ==================================================
# =============================================================================


def configure_plot_theme(
    theme: Optional[str] = DEFAULT_PLOT_THEME,
    *,
    base_font: str = PLOT_LATIN_FONT_FAMILY,
    chinese_font: Optional[str] = PLOT_CHINESE_FONT_FAMILY,
) -> None:
    """Apply a consistent plotting theme across evaluation figures."""

    plt.rcdefaults()
    if theme is not None:
        sns.set_theme(context=theme)
    else:
        plt.style.use("default")

    font_families: List[str] = [base_font]
    if chinese_font and chinese_font not in font_families:
        font_families.append(chinese_font)
    plt.rcParams["font.family"] = font_families
    plt.rcParams["font.sans-serif"] = font_families
    plt.rcParams["axes.unicode_minus"] = False


configure_plot_theme()


# =============================================================================
# === Palette helpers =========================================================
# =============================================================================


def build_training_color_map(
    training_order: Sequence[str],
    *,
    palette: str | Sequence[str] | None = None,
) -> Dict[str, str]:
    """Return a colour mapping for training datasets using a Seaborn palette."""

    if not training_order:
        return {}

    if palette is None:
        colors = sns.color_palette(None, n_colors=len(training_order))
    else:
        colors = sns.color_palette(palette, n_colors=len(training_order))

    if not colors:
        colors = sns.color_palette("deep", n_colors=len(training_order))

    return {name: colors[idx % len(colors)] for idx, name in enumerate(training_order)}


# =============================================================================
# === Feature grouping and visualisation metadata ============================
# =============================================================================


# fmt: off
VAR_GROUP_DICT: Dict[str, List[str]] = {
    "basic_feature_and_organ_support": [
        "sex", "age", "BMI", "temperature", "heart_rate", "respir_rate",
        "GCS", "CRRT", "Respiratory_Support",
    ],
    "BP_and_perfusion": ["SBP", "MAP", "Lac", "septic_shock"],
    "respiratory_and_bg": [
        "SPO2", "PaO2", "PaO2/FiO2", "PaCO2", "HCO3-", "PH",
    ],
    "blood_routine": [
        "RBC", "Hb", "HCT", "WBC", "NE%", "LYM%",
    ],
    "coagulation": ["PLT", "PT", "APTT", "Fg"],
    "biochem_lab": [
        "ALT", "AST", "STB", "BUN", "Scr", "Glu", "K+", "Na+",
    ],
}
# fmt: on


PATH_GRAPH_GROUP_COLORS: Dict[str, str] = {
    "Demographics & Vitals": "#1f77b4",
    "Hemodynamics & Perfusion": "#d62728",
    "Organ Support & Neurology": "#9467bd",
    "Hematology and Immunology": "#2ca02c",
    "Hepatic Function": "#bcbd22",
    "Renal Function": "#17becf",
    "Metabolic & Electrolytes": "#ff7f0e",
    "Coagulation": "#8c564b",
    "Respiratory and Blood Gas": "#7f7f7f",
    "Outcome": "#e377c2",
    "Latent": "#4c72b0",
}


PATH_GRAPH_NODE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "age": {"group": "Demographics & Vitals", "label": "Age"},
    "sex": {"group": "Demographics & Vitals", "label": "Male sex"},
    "BMI": {"group": "Demographics & Vitals", "label": r"BMI",},
    "temperature": {"group": "Demographics & Vitals", "label": "Temperature"},
    "heart_rate": {"group": "Demographics & Vitals","label": "Heart rate",},
    "respir_rate": {"group": "Demographics & Vitals","label": "Respiratory rate",},
    "SBP": {"group": "Hemodynamics & Perfusion", "label": "SBP"},
    "DBP": {"group": "Hemodynamics & Perfusion", "label": "DBP"},
    "MAP": {"group": "Hemodynamics & Perfusion", "label": "MAP"},
    "Lac": {"group": "Hemodynamics & Perfusion","label": "Serum lactate",},
    "SOFA_cns": {"group": "Organ Support & Neurology", "label": "SOFA CNS"},
    "CRRT": {"group": "Organ Support & Neurology", "label": "CRRT"},
    "Respiratory_Support": {"group": "Organ Support & Neurology","label": "Respiratory support",},
    "WBC": {"group": "Hematology and Immunology", "label": "WBC"},
    "Hb": {"group": "Hematology and Immunology", "label": "Hb"},
    "NE%": {"group": "Hematology and Immunology", "label": "NE%"},
    "LYM%": {"group": "Hematology and Immunology", "label": "LYM%"},
    "PLT": {"group": "Hematology and Immunology", "label": "PLT"},
    "ALT": {"group": "Hepatic Function", "label": "ALT"},
    "AST": {"group": "Hepatic Function", "label": "AST"},
    "STB": {"group": "Hepatic Function", "label": "TBil"},
    "BUN": {"group": "Renal Function", "label": "BUN"},
    "Scr": {"group": "Renal Function", "label": "SCr"},
    "Glu": {"group": "Metabolic & Electrolytes", "label": "Glucose"},
    "K+": {"group": "Metabolic & Electrolytes", "label": r"$\mathrm{K}^{+}$"},
    "Na+": {"group": "Metabolic & Electrolytes", "label": r"$\mathrm{Na}^{+}$"},
    "HCO3-": {"group": "Metabolic & Electrolytes", "label": r"$\mathrm{HCO}_{3}^{-}$"},
    "Fg": {"group": "Coagulation", "label": "Fibrinogen"},
    "PT": {"group": "Coagulation", "label": "PT"},
    "APTT": {"group": "Coagulation", "label": "APTT"},
    "PH": {"group": "Respiratory and Blood Gas", "label": "pH"},
    "PaO2": {"group": "Respiratory and Blood Gas", "label": r"$\mathrm{PaO}_{2}$"},
    "PaO2/FiO2": {"group": "Respiratory and Blood Gas","label": r"$\mathrm{PaO}_{2}/\mathrm{FiO}_{2}$ ratio",},
    "PaCO2": {"group": "Respiratory and Blood Gas", "label": r"$\mathrm{PaCO}_{2}$"},
    "in_hospital_mortality": {"group": "Outcome","label": "In-hospital mortality",},
}


PATH_GRAPH_NODE_LABELS: Dict[str, str] = {
    node_id: metadata["label"]
    for node_id, metadata in PATH_GRAPH_NODE_DEFINITIONS.items()
}


PATH_GRAPH_NODE_GROUPS: Dict[str, str] = {
    node_id: metadata["group"]
    for node_id, metadata in PATH_GRAPH_NODE_DEFINITIONS.items()
}


PATH_GRAPH_NODE_COLORS: Dict[str, str] = {
    node_id: PATH_GRAPH_GROUP_COLORS[metadata["group"]]
    for node_id, metadata in PATH_GRAPH_NODE_DEFINITIONS.items()
}


# =============================================================================
# === Analysis configuration helpers ==========================================
# =============================================================================


def build_analysis_config(**overrides: object) -> Dict[str, object]:
    """Return the default analysis configuration with optional overrides."""

    config = dict(DEFAULT_ANALYSIS_CONFIG)
    config.update(overrides)
    config.setdefault("optuna_storage", None)
    config.setdefault("interactive_manual_tuning", INTERACTIVE_MANUAL_TUNING)
    return config


def prepare_analysis_output_directories(
    output_root: Path, keys: Sequence[str]
) -> Dict[str, Path]:
    """Create and return standardised analysis output subdirectories."""

    directories: Dict[str, Path] = {}
    for key in keys:
        try:
            subdir_name = ANALYSIS_SUBDIRECTORIES[key]
        except KeyError as error:  # pragma: no cover - guard against typos
            raise KeyError(f"Unknown analysis subdirectory key: {key}") from error
        path = output_root / subdir_name
        path.mkdir(parents=True, exist_ok=True)
        directories[key] = path
        if subdir_name == ANALYSIS_SUBDIRECTORIES.get("suave_model"):
            _initialise_manual_param_script(path)
    return directories


def _initialise_manual_param_script(directory: Path) -> None:
    """Ensure ``manual_param_setting.py`` exists with a default placeholder."""

    script_path = directory / "manual_param_setting.py"
    default_content = (
        '"""Manual hyper-parameter overrides for SUAVE training.\n\n'
        "Populate ``manual_param_setting`` with overrides when interactive tuning is enabled.\n"
        '"""\n\n'
        "manual_param_setting: dict = {}\n"
    )

    if script_path.exists():
        try:
            existing = script_path.read_text(encoding="utf-8")
        except OSError:
            existing = None
        if existing and existing.strip():
            return

    script_path.write_text(default_content, encoding="utf-8")


def resolve_analysis_output_root(
    output_dir: Optional[Union[str, Path]] = None,
    *,
    create: bool = True,
) -> Path:
    """Return the root directory for mortality analysis outputs.

    Parameters
    ----------
    output_dir:
        Optional custom directory name or path. Relative paths are resolved
        against the examples directory. When omitted, the default output
        directory configured in :data:`DEFAULT_ANALYSIS_CONFIG` is used.
    create:
        Whether to create the directory (and parents) if it does not exist.

    Returns
    -------
    pathlib.Path
        The resolved output directory path.
    """

    if output_dir is None:
        output_dir = str(DEFAULT_ANALYSIS_CONFIG["output_dir_name"])

    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = EXAMPLES_DIR / output_path

    if create:
        output_path.mkdir(parents=True, exist_ok=True)

    return output_path


__all__ = [
    "RANDOM_STATE",
    "TARGET_COLUMNS",
    "BENCHMARK_COLUMNS",
    "CLINICAL_SCORE_BENCHMARK_STRATEGY",
    "VALIDATION_SIZE",
    "DATA_DIR",
    "DISTRIBUTION_SHIFT_OVERALL_NOTE",
    "DISTRIBUTION_SHIFT_PER_FEATURE_NOTE",
    "Schema",
    "SchemaInferencer",
    "HIDDEN_DIMENSION_OPTIONS",
    "HEAD_HIDDEN_DIMENSION_OPTIONS",
    "PARETO_MIN_VALIDATION_ROAUC",
    "PARETO_MAX_ABS_DELTA_AUC",
    "VAR_GROUP_DICT",
    "PATH_GRAPH_GROUP_COLORS",
    "PATH_GRAPH_NODE_DEFINITIONS",
    "PATH_GRAPH_NODE_LABELS",
    "PATH_GRAPH_NODE_GROUPS",
    "PATH_GRAPH_NODE_COLORS",
    "choose_preferred_pareto_trial",
    "load_model_manifest",
    "load_manual_model_manifest",
    "load_optuna_results",
    "manifest_artifact_paths",
    "manifest_artifacts_exist",
    "record_model_manifest",
    "record_manual_model_manifest",
    "collect_manual_and_optuna_overview",
    "load_manual_tuning_overrides",
    "prompt_manual_override_action",
    "evaluate_candidate_model_performance",
    "run_manual_override_training",
    "build_prediction_dataframe",
    "compute_auc",
    "compute_binary_metrics",
    "dataframe_to_markdown",
    "define_schema",
    "extract_positive_probabilities",
    "fit_isotonic_calibrator",
    "format_float",
    "is_interactive_session",
    "kolmogorov_smirnov_statistic",
    "load_dataset",
    "load_or_create_iteratively_imputed_features",
    "iteratively_impute_clinical_scores",
    "make_logistic_pipeline",
    "make_random_forest_pipeline",
    "make_gradient_boosting_pipeline",
    "make_baseline_model_factories",
    "build_training_color_map",
    "mutual_information_feature",
    "plot_benchmark_curves",
    "plot_calibration_curves",
    "plot_latent_space",
    "plot_transfer_metric_boxes",
    "prepare_features",
    "render_dataframe",
    "_interpret_global_shift",
    "_interpret_feature_shift",
    "rbf_mmd",
    "make_study_name",
    "DEFAULT_ANALYSIS_CONFIG",
    "INTERACTIVE_MANUAL_TUNING",
    "ANALYSIS_SUBDIRECTORIES",
    "build_analysis_config",
    "prepare_analysis_output_directories",
    "resolve_analysis_output_root",
    "parse_script_arguments",
    "load_optuna_study",
    "render_optuna_parameter_grid",
    "ModelLoadingPlan",
    "summarise_pareto_trials",
    "resolve_model_loading_plan",
    "confirm_model_loading_plan_selection",
    "schema_markdown_table",
    "schema_to_dataframe",
    "slugify_identifier",
    "to_numeric_frame",
    "build_tstr_training_sets",
    "load_tstr_training_sets_from_tsv",
    "save_tstr_training_sets_to_tsv",
    "collect_transfer_bootstrap_records",
    "compute_transfer_delta_bootstrap",
    "evaluate_transfer_baselines",
    "build_suave_model",
    "resolve_suave_fit_kwargs",
    "resolve_classification_loss_weight",
    "plot_transfer_metric_bars",
]


# =============================================================================
# === Session and rendering helpers ==========================================
# =============================================================================


def is_interactive_session() -> bool:
    """Return ``True`` when executed inside an interactive IPython session."""

    try:
        from IPython import get_ipython
    except ImportError:  # pragma: no cover - optional dependency
        return False
    return get_ipython() is not None


def _save_figure_multiformat(
    figure: "plt.Figure",
    base_path: Path,
    *,
    dpi: int = 300,
    use_tight_layout: bool = False,
) -> None:
    """Persist ``figure`` to PNG, SVG, PDF, and JPG variants."""

    save_kwargs: Dict[str, object] = {"dpi": dpi}
    if use_tight_layout:
        save_kwargs["bbox_inches"] = "tight"
    for suffix in (".png", ".svg", ".pdf", ".jpg"):
        figure.savefig(base_path.with_suffix(suffix), **save_kwargs)


def render_dataframe(
    df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    floatfmt: Optional[str] = ".3f",
) -> None:
    """Render a dataframe either via ``display`` or ``tabulate``."""

    if title:
        print(title)
    if df.empty:
        print("(empty table)")
        return
    if is_interactive_session():
        with pd.option_context("display.max_columns", None):
            display(df)
        return
    tabulate_kwargs = {"headers": "keys", "tablefmt": "github", "showindex": False}
    if floatfmt is not None:
        tabulate_kwargs["floatfmt"] = floatfmt
    print(tabulate(df, **tabulate_kwargs))


def _interpret_global_shift(metric_name: str, value: float, p_value: float) -> str:
    """Return a short narrative for global distribution-shift diagnostics."""

    if np.isnan(value):
        return "Metric unavailable."
    if metric_name in {"rbf_mmd", "energy_distance"}:
        if not np.isnan(p_value) and p_value < 0.05:
            return "Significant distribution difference detected (p < 0.05)."
        if not np.isnan(p_value):
            return "No significant difference detected (p ≥ 0.05)."
        return "Inspect magnitude relative to domain expectations."
    if metric_name == "mutual_information":
        return (
            "Average feature-level mutual information; higher values indicate "
            "stronger dependency."
        )
    return "Review metric in context."


def _interpret_feature_shift(
    mmd_value: float, energy_value: float, mi_value: float
) -> str:
    """Return guidance for per-feature distribution shift metrics."""

    messages: List[str] = []
    if not np.isnan(mmd_value) and mmd_value > 0.05:
        messages.append("MMD > 0.05 suggests a noticeable shift.")
    if not np.isnan(energy_value) and energy_value > 0.1:
        messages.append("Energy distance > 0.1 indicates distribution divergence.")
    if not np.isnan(mi_value) and mi_value > 0.1:
        messages.append(
            "Mutual information > 0.1 highlights dependency differences."
        )
    if not messages:
        return "No pronounced shift detected."
    return " ".join(messages)


# =============================================================================
# === Model manifest and Optuna metadata helpers ==============================
# =============================================================================


def _normalise_manifest_path(path: Path, base_dir: Path) -> str:
    """Return ``path`` as a relative string when nested under ``base_dir``."""

    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        try:
            return os.path.relpath(path, base_dir)
        except ValueError:
            return str(path)


def manifest_artifact_paths(
    manifest: Mapping[str, object],
    model_dir: Path,
) -> Dict[str, Optional[Path]]:
    """Return resolved artefact paths declared in ``manifest``.

    Parameters
    ----------
    manifest
        Metadata loaded via :func:`load_model_manifest`.
    model_dir
        Directory that stores serialised SUAVE artefacts.

    Returns
    -------
    dict
        Mapping with ``model`` and ``calibrator`` keys pointing to resolved
        :class:`~pathlib.Path` instances when available.

    Examples
    --------
    >>> manifest = {"model_path": "suave_best_label.pt"}
    >>> paths = manifest_artifact_paths(manifest, Path("/tmp"))
    >>> paths["model"].name
    'suave_best_label.pt'
    """

    resolved: Dict[str, Optional[Path]] = {"model": None, "calibrator": None}

    for key, label in (("model_path", "model"), ("calibrator_path", "calibrator")):
        raw = manifest.get(key)
        if not raw:
            continue
        candidate = Path(str(raw))
        if not candidate.is_absolute():
            candidate = model_dir / candidate
        resolved[label] = candidate
    return resolved


def manifest_artifacts_exist(manifest: Mapping[str, object], model_dir: Path) -> bool:
    """Return ``True`` when artefact paths declared in ``manifest`` exist.

    Parameters
    ----------
    manifest
        Metadata loaded via :func:`load_model_manifest`.
    model_dir
        Directory that stores serialised SUAVE artefacts.

    Examples
    --------
    >>> manifest = {"model_path": "model.pt"}
    >>> manifest_artifacts_exist(manifest, Path("."))
    False
    """

    paths = manifest_artifact_paths(manifest, model_dir)
    model_path = paths.get("model")
    calibrator_path = paths.get("calibrator")
    return bool(
        model_path
        and model_path.exists()
        and calibrator_path
        and calibrator_path.exists()
    )


def load_model_manifest(model_dir: Path, target_label: str) -> Dict[str, object]:
    """Load the metadata describing the latest persisted SUAVE artefacts.

    Parameters
    ----------
    model_dir
        Directory storing SUAVE checkpoints and calibrators.
    target_label
        Name of the prediction target the artefacts were trained on.

    Returns
    -------
    dict
        Manifest metadata. An empty dictionary is returned when no manifest is
        present on disk.

    Examples
    --------
    >>> tmp = Path("/tmp/model_manifest")
    >>> _ = tmp.mkdir(parents=True, exist_ok=True)
    >>> load_model_manifest(tmp, "mortality")
    {}
    """

    manifest_path = model_dir / f"suave_model_manifest_{target_label}.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


def record_model_manifest(
    model_dir: Path,
    target_label: str,
    *,
    trial_number: Optional[Union[str, int]],
    values: Sequence[float],
    params: Mapping[str, object],
    model_path: Path,
    calibrator_path: Path,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
) -> Path:
    """Persist metadata describing the saved SUAVE model artefacts.

    Parameters
    ----------
    model_dir
        Directory storing SUAVE checkpoints and calibrators.
    target_label
        Prediction target associated with the artefacts.
    trial_number
        Optuna trial identifier or override label used to train the artefacts.
    values
        Objective values reported by Optuna for the selected trial.
    params
        Hyperparameters used to fit the SUAVE model.
    model_path
        Filesystem path to the serialised SUAVE checkpoint.
    calibrator_path
        Filesystem path to the isotonic calibrator joblib file.
    study_name
        Optional Optuna study name to aid traceability.
    storage
        Optional Optuna storage backend URI.

    Returns
    -------
    Path
        Location of the manifest file written to disk.

    Examples
    --------
    >>> tmp = Path("/tmp/model_manifest_example")
    >>> _ = tmp.mkdir(parents=True, exist_ok=True)
    >>> model = tmp / "model.pt"
    >>> calibrator = tmp / "calibrator.joblib"
    >>> _ = model.write_text("dummy")
    >>> _ = calibrator.write_text("dummy")
    >>> manifest = record_model_manifest(
    ...     tmp,
    ...     "mortality",
    ...     trial_number=1,
    ...     values=[0.9, 0.01],
    ...     params={"lr": 1e-3},
    ...     model_path=model,
    ...     calibrator_path=calibrator,
    ... )
    >>> manifest.exists()
    True
    """

    manifest_path = model_dir / f"suave_model_manifest_{target_label}.json"
    manifest: Dict[str, object] = {
        "target_label": target_label,
        "trial_number": trial_number,
        "values": [float(value) for value in values],
        "params": dict(params),
        "model_path": _normalise_manifest_path(model_path, model_dir),
        "calibrator_path": _normalise_manifest_path(calibrator_path, model_dir),
        "study_name": study_name,
        "storage": storage,
        # ``datetime.utcnow`` is deprecated in Python 3.12; ``now(timezone.utc)``
        # keeps the timestamp timezone-aware while remaining backwards
        # compatible with older interpreters.
        "saved_at": datetime.now(tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest_path


def load_manual_model_manifest(model_dir: Path, target_label: str) -> Dict[str, object]:
    """Load metadata describing manually managed SUAVE artefacts.

    Parameters
    ----------
    model_dir
        Directory storing SUAVE checkpoints and calibrators.
    target_label
        Name of the prediction target the artefacts correspond to.

    Returns
    -------
    dict
        Manifest metadata. An empty dictionary is returned when no manual
        manifest is present on disk.

    Examples
    --------
    >>> tmp = Path("/tmp/manual_manifest")
    >>> _ = tmp.mkdir(parents=True, exist_ok=True)
    >>> load_manual_model_manifest(tmp, "mortality")
    {}
    """

    manifest_path = model_dir / f"suave_manual_manifest_{target_label}.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


def record_manual_model_manifest(
    model_dir: Path,
    target_label: str,
    *,
    model_path: Path,
    calibrator_path: Optional[Path] = None,
    params: Optional[Mapping[str, object]] = None,
    values: Optional[Sequence[float]] = None,
    validation_metrics: Optional[Mapping[str, object]] = None,
    tstr_metrics: Optional[Mapping[str, object]] = None,
    trtr_metrics: Optional[Mapping[str, object]] = None,
    description: Optional[str] = None,
) -> Path:
    """Persist metadata describing manually managed SUAVE artefacts.

    Parameters
    ----------
    model_dir
        Directory storing SUAVE checkpoints and calibrators.
    target_label
        Prediction target associated with the artefacts.
    model_path
        Filesystem path to the serialised SUAVE checkpoint.
    calibrator_path
        Optional filesystem path to an isotonic calibrator produced alongside
        the manual model.
    params
        Optional hyperparameter mapping used to construct the manual artefacts.
    values
        Optional Optuna-style objective tuple ``(validation_roauc, delta_auc)``
        describing the manual artefacts. When provided the values are stored in
        the manifest for downstream summaries.
    validation_metrics
        Optional dictionary of validation metrics computed for the manual
        artefacts.
    tstr_metrics
        Optional dictionary containing transfer-to-synthetic-to-real (TSTR)
        evaluation metrics for the manual artefacts.
    trtr_metrics
        Optional dictionary containing transfer-to-real-to-real (TRTR)
        evaluation metrics for the manual artefacts.
    description
        Optional free-form text describing the origin of the manual artefacts.

    Returns
    -------
    Path
        Location of the manual manifest file written to disk.

    Examples
    --------
    >>> tmp = Path("/tmp/manual_manifest_example")
    >>> _ = tmp.mkdir(parents=True, exist_ok=True)
    >>> model = tmp / "manual_model.pt"
    >>> calibrator = tmp / "manual_calibrator.joblib"
    >>> _ = model.write_text("dummy")
    >>> _ = calibrator.write_text("dummy")
    >>> manifest = record_manual_model_manifest(
    ...     tmp,
    ...     "mortality",
    ...     model_path=model,
    ...     calibrator_path=calibrator,
    ...     params={"lr": 1e-3},
    ... )
    >>> manifest.exists()
    True
    """

    manifest_path = model_dir / f"suave_manual_manifest_{target_label}.json"
    manifest: Dict[str, object] = {
        "target_label": target_label,
        "model_path": _normalise_manifest_path(model_path, model_dir),
        "saved_at": datetime.now(tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "source": "manual",
        "trial_number": "manual",
    }
    if calibrator_path is not None:
        manifest["calibrator_path"] = _normalise_manifest_path(
            calibrator_path, model_dir
        )
    if params is not None:
        manifest["params"] = dict(params)
    if values is not None:
        manifest["values"] = [float(value) for value in values]

    def _serialise_metrics(
        metrics: Optional[Mapping[str, object]]
    ) -> Optional[Dict[str, float]]:
        if not isinstance(metrics, Mapping):
            return None
        serialised: Dict[str, float] = {}
        for key, raw_value in metrics.items():
            try:
                serialised[str(key)] = float(raw_value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
        return serialised or None

    validation_payload = _serialise_metrics(validation_metrics)
    if validation_payload is not None:
        manifest["validation_metrics"] = validation_payload

    tstr_payload = _serialise_metrics(tstr_metrics)
    if tstr_payload is not None:
        manifest["tstr_metrics"] = tstr_payload

    trtr_payload = _serialise_metrics(trtr_metrics)
    if trtr_payload is not None:
        manifest["trtr_metrics"] = trtr_payload

    if description:
        manifest["description"] = description
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest_path


# =============================================================================
# === Optuna study discovery and selection utilities =========================
# =============================================================================


def make_study_name(prefix: Optional[str], target_label: str) -> Optional[str]:
    """Return the Optuna study name for ``target_label`` given ``prefix``."""

    if not prefix:
        return None
    return f"{prefix}_{target_label}"


def parse_script_arguments(argv: Sequence[str]) -> Optional[Union[str, int]]:
    """Parse ``argv`` and return the requested Optuna trial identifier.

    Parameters
    ----------
    argv
        Command-line arguments excluding the executable name.

    Returns
    -------
    Optional[Union[str, int]]
        The requested Optuna trial identifier or ``"manual"`` override, if
        provided.

    Examples
    --------
    >>> parse_script_arguments(["--trial-id", "12"])
    12
    >>> parse_script_arguments(["manual"])
    'manual'
    >>> parse_script_arguments([]) is None
    True
    """

    parser = argparse.ArgumentParser(
        description=(
            "Select an Optuna trial for SUAVE model loading/training, or pass"
            " 'manual' to load the manual tuning manifest."
        ),
    )

    def _trial_argument(value: str) -> Union[str, int]:
        if value.lower() == "manual":
            return "manual"
        try:
            return int(value)
        except ValueError as error:  # pragma: no cover - argparse normalises
            raise argparse.ArgumentTypeError(
                "Provide an integer Optuna trial ID or the keyword 'manual'."
            ) from error

    parser.add_argument(
        "trial_id",
        nargs="?",
        type=_trial_argument,
        help=(
            "Optuna trial identifier to load or train; pass 'manual' to load the"
            " manual tuning manifest."
        ),
    )
    parser.add_argument(
        "--trial-id",
        dest="trial_id_flag",
        type=_trial_argument,
        help=(
            "Optuna trial identifier to load or train; pass 'manual' to load the"
            " manual tuning manifest."
        ),
    )
    args = parser.parse_args(list(argv))
    if args.trial_id_flag is not None:
        return args.trial_id_flag
    return args.trial_id


def load_optuna_study(
    *,
    study_prefix: Optional[str],
    target_label: str,
    storage: Optional[str],
) -> Optional["optuna.study.Study"]:
    """Return the persisted Optuna study when available."""

    study_name = make_study_name(study_prefix, target_label)
    if not storage or not study_name:
        return None
    try:
        import optuna  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency in notebooks
        return None

    try:
        return optuna.load_study(study_name=study_name, storage=storage)
    except Exception as error:  # pragma: no cover - diagnostic logging only
        print(
            f"Failed to load Optuna study '{study_name}' from storage '{storage}': {error}"
        )
        return None


def render_optuna_parameter_grid(
    study: "optuna.study.Study",
    *,
    objective_targets: Sequence[tuple[str, Callable[["optuna.trial.FrozenTrial"], float]]],
) -> None:
    """Render slice, parallel, and importance plots for ``study`` objectives."""

    if not objective_targets:
        return

    try:
        from optuna.visualization import (
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice,
        )
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError as error:  # pragma: no cover - optional dependency guard
        print(f"Optuna visualisations unavailable: {error}")
        return

    plot_specs: Sequence[tuple[str, Callable[..., "plotly.graph_objs.Figure"]]] = (
        ("Parameter slice", plot_slice),
        ("Parallel coordinate", plot_parallel_coordinate),
        ("Parameter importance", plot_param_importances),
    )

    for plot_title, plot_fn in plot_specs:
        for objective_name, target_fn in objective_targets:
            try:
                figure = plot_fn(study, target=target_fn, target_name=objective_name)
            except Exception as error:  # pragma: no cover - diagnostic aid
                figure = go.Figure()
                figure.add_annotation(text=f"Unable to render:<br>{error}", showarrow=False)
                figure.update_xaxes(visible=False)
                figure.update_yaxes(visible=False)

            for trace in getattr(figure, "data", []):
                try:
                    trace.showlegend = False
                except ValueError:
                    pass

            figure.update_layout(
                title_text=f"{plot_title} — {objective_name}",
                height=450,
                showlegend=False,
            )
            pio.show(figure)


def load_optuna_results(
    output_dir: Path,
    target_label: str,
    *,
    study_prefix: Optional[str],
    storage: Optional[str],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load the best Optuna trial metadata and parameters when available.

    Parameters
    ----------
    output_dir
        Directory storing Optuna artefacts produced by the optimisation
        workflow.
    target_label
        Prediction target associated with the optimisation run.
    study_prefix
        Optional Optuna study prefix used to construct the study name.
    storage
        Optional Optuna storage URI for recovering metadata when JSON files are
        absent.

    Returns
    -------
    tuple(dict, dict)
        Tuple containing the best trial metadata and best parameter dictionary.

    Examples
    --------
    >>> info, params = load_optuna_results(Path("/tmp"), "mortality", study_prefix="demo", storage=None)
    >>> isinstance(info, dict), isinstance(params, dict)
    (True, True)
    """

    best_info_path = output_dir / f"optuna_best_info_{target_label}.json"
    best_params_path = output_dir / f"optuna_best_params_{target_label}.json"

    best_info_raw: object = (
        json.loads(best_info_path.read_text()) if best_info_path.exists() else {}
    )
    best_params_raw: object = (
        json.loads(best_params_path.read_text()) if best_params_path.exists() else {}
    )

    pareto_front_info: List[Dict[str, Any]] = []
    best_info: Dict[str, Any] = {}
    preferred_trial_number: Optional[int] = None

    if isinstance(best_info_raw, Mapping):
        pareto_candidates = best_info_raw.get("pareto_front")
        if isinstance(pareto_candidates, list):
            pareto_front_info = [
                dict(candidate)
                for candidate in pareto_candidates
                if isinstance(candidate, Mapping)
            ]
        preferred_trial = best_info_raw.get("preferred_trial")
        if isinstance(preferred_trial, Mapping):
            best_info = dict(preferred_trial)
        elif not pareto_front_info:
            best_info = dict(best_info_raw)
        preferred_trial_number = best_info_raw.get("preferred_trial_number")
    elif isinstance(best_info_raw, list):
        pareto_front_info = [
            dict(candidate)
            for candidate in best_info_raw
            if isinstance(candidate, Mapping)
        ]

    if pareto_front_info:
        if preferred_trial_number is not None:
            matching_info = next(
                (
                    candidate
                    for candidate in pareto_front_info
                    if candidate.get("trial_number") == preferred_trial_number
                ),
                None,
            )
        else:
            matching_info = next(
                (
                    candidate
                    for candidate in pareto_front_info
                    if candidate.get("is_preferred")
                ),
                None,
            )
        if matching_info is None and pareto_front_info:
            matching_info = pareto_front_info[0]
        if matching_info is not None:
            if not best_info:
                best_info = dict(matching_info)
            preferred_trial_number = matching_info.get("trial_number")
        best_info = dict(best_info) if best_info else {}
        best_info.setdefault("pareto_front", pareto_front_info)
        if preferred_trial_number is not None:
            best_info.setdefault("preferred_trial_number", preferred_trial_number)
    else:
        best_info = dict(best_info) if best_info else {}

    pareto_params_info: List[Dict[str, Any]] = []
    best_params: Dict[str, Any] = {}

    if isinstance(best_params_raw, Mapping):
        pareto_candidates = best_params_raw.get("pareto_front")
        if isinstance(pareto_candidates, list):
            for candidate in pareto_candidates:
                if not isinstance(candidate, Mapping):
                    continue
                params_mapping = candidate.get("params")
                candidate_dict = {
                    "trial_number": candidate.get("trial_number"),
                    "params": dict(params_mapping)
                    if isinstance(params_mapping, Mapping)
                    else {},
                    "is_preferred": bool(candidate.get("is_preferred")),
                }
                pareto_params_info.append(candidate_dict)
            preferred_params = best_params_raw.get("preferred_params")
            if isinstance(preferred_params, Mapping):
                best_params = dict(preferred_params)
        else:
            best_params = dict(best_params_raw)
    elif isinstance(best_params_raw, list):
        for candidate in best_params_raw:
            if not isinstance(candidate, Mapping):
                continue
            params_mapping = candidate.get("params")
            candidate_dict = {
                "trial_number": candidate.get("trial_number"),
                "params": dict(params_mapping)
                if isinstance(params_mapping, Mapping)
                else {},
                "is_preferred": bool(candidate.get("is_preferred")),
            }
            pareto_params_info.append(candidate_dict)

    if not best_params and preferred_trial_number is not None:
        preferred_params_candidate = next(
            (
                candidate
                for candidate in pareto_params_info
                if candidate.get("trial_number") == preferred_trial_number
            ),
            None,
        )
        if preferred_params_candidate is not None and isinstance(
            preferred_params_candidate.get("params"), Mapping
        ):
            best_params = dict(preferred_params_candidate["params"])

    if not best_params and pareto_params_info:
        preferred_candidate = next(
            (
                candidate
                for candidate in pareto_params_info
                if candidate.get("is_preferred")
            ),
            None,
        )
        if preferred_candidate is None:
            preferred_candidate = pareto_params_info[0]
        if isinstance(preferred_candidate.get("params"), Mapping):
            best_params = dict(preferred_candidate["params"])

    if not best_params and isinstance(best_info.get("params"), Mapping):
        best_params = dict(best_info["params"])

    if pareto_params_info:
        best_info.setdefault("pareto_front_params", pareto_params_info)

    study_name = make_study_name(study_prefix, target_label)
    if (not best_info or not best_params) and storage and study_name:
        try:
            import optuna  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency in examples
            optuna = None  # type: ignore
        if optuna is not None:  # pragma: no cover - optuna available in examples env
            study = optuna.load_study(study_name=study_name, storage=storage)
            feasible_trials = [
                trial for trial in study.trials if trial.values is not None
            ]
            if feasible_trials:

                def sort_key(trial: "optuna.trial.FrozenTrial") -> tuple[float, float]:
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


def _coerce_float(value: object) -> float:
    """Return ``value`` converted to ``float`` when possible."""

    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _format_artifact_path(path: Optional[Path], base_dir: Path) -> str:
    """Return ``path`` relative to ``base_dir`` when feasible."""

    if path is None:
        return ""
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _extract_trial_metrics(trial: object) -> Tuple[float, float]:
    """Return ``(validation_roauc, delta_auc)`` for ``trial`` or manifests."""

    validation = float("nan")
    delta = float("nan")

    if hasattr(trial, "values"):
        values = getattr(trial, "values")  # type: ignore[attr-defined]
    elif isinstance(trial, Mapping):
        raw_values = cast(Mapping[str, object], trial).get("values")
        values = raw_values if isinstance(raw_values, Sequence) else None
    else:
        values = None

    if values:
        try:
            numeric = [float(item) for item in values]
        except (TypeError, ValueError):
            numeric = []
        if numeric:
            validation = float(numeric[0])
            if len(numeric) > 1:
                delta = float(numeric[1])

    if np.isnan(validation) and isinstance(trial, Mapping):
        metrics = cast(Mapping[str, object], trial).get("validation_metrics")
        if isinstance(metrics, Mapping):
            for key in ("ROAUC", "roauc", "AUROC", "auroc"):
                if key in metrics:
                    validation = _coerce_float(metrics[key])
                    break

    if np.isnan(delta) and isinstance(trial, Mapping):
        mapping = cast(Mapping[str, object], trial)
        delta_value = mapping.get("tstr_trtr_delta_auc")
        if delta_value is not None:
            delta = _coerce_float(delta_value)
        else:
            tstr_metrics = mapping.get("tstr_metrics")
            trtr_metrics = mapping.get("trtr_metrics")
            if isinstance(tstr_metrics, Mapping) and isinstance(trtr_metrics, Mapping):
                tstr_auc = _coerce_float(
                    tstr_metrics.get("auroc", tstr_metrics.get("ROAUC"))
                )
                trtr_auc = _coerce_float(
                    trtr_metrics.get("auroc", trtr_metrics.get("ROAUC"))
                )
                if np.isfinite(tstr_auc) and np.isfinite(trtr_auc):
                    delta = abs(float(trtr_auc) - float(tstr_auc))

    return validation, delta


TRIAL_SUMMARY_BASE_COLUMNS = [
    "Source",
    "Saved locally",
    "Trial ID",
    "Model path",
    "Validation ROAUC",
    "TSTR/TRTR ΔAUC",
]

TRIAL_SUMMARY_ALLOWED_METRICS = {"Validation ROAUC"}


def _format_metric_column(prefix: str, metric_key: object) -> str:
    """Return a human-readable column name for metric ``metric_key``."""

    raw_key = str(metric_key)
    if raw_key.isupper():
        formatted_key = raw_key
    else:
        formatted_key = raw_key.replace("_", " ").upper()
    prefix_label = prefix.strip()
    if prefix_label:
        return f"{prefix_label} {formatted_key}"
    return formatted_key


def _collect_metric_columns(source: object) -> Dict[str, float]:
    """Return flattened metric columns stored on ``source``."""

    metric_columns: Dict[str, float] = {}
    containers: list[Mapping[str, object]] = []

    if hasattr(source, "user_attrs"):
        user_attrs = getattr(source, "user_attrs")  # type: ignore[attr-defined]
        if isinstance(user_attrs, Mapping):
            containers.append(cast(Mapping[str, object], user_attrs))

    if isinstance(source, Mapping):
        containers.append(cast(Mapping[str, object], source))

    metric_prefixes = (
        ("validation_metrics", "Validation"),
        ("tstr_metrics", "TSTR"),
        ("trtr_metrics", "TRTR"),
    )

    for container in containers:
        for attr_name, prefix in metric_prefixes:
            raw_metrics = container.get(attr_name)
            if not isinstance(raw_metrics, Mapping):
                continue
            for metric_key, metric_value in raw_metrics.items():
                column_name = _format_metric_column(prefix, metric_key)
                if column_name not in TRIAL_SUMMARY_ALLOWED_METRICS:
                    continue
                metric_columns[column_name] = _coerce_float(metric_value)

    return metric_columns


def _collect_param_columns(source: object) -> Dict[str, object]:
    """Return Optuna-style parameter columns stored on ``source``."""

    if hasattr(source, "params"):
        params = getattr(source, "params")  # type: ignore[attr-defined]
        if isinstance(params, Mapping):
            return {str(key): value for key, value in params.items()}

    if isinstance(source, Mapping):
        raw_params = cast(Mapping[str, object], source).get("params")
        if isinstance(raw_params, Mapping):
            return {str(key): value for key, value in raw_params.items()}

    return {}


def _append_unique(sequence: list[str], value: str) -> None:
    """Append ``value`` to ``sequence`` if it has not been seen yet."""

    if value not in sequence:
        sequence.append(value)


def _compose_trial_columns(
    metric_columns: Sequence[str], param_columns: Sequence[str]
) -> list[str]:
    """Return ordered DataFrame columns without duplicates."""

    columns = list(TRIAL_SUMMARY_BASE_COLUMNS)
    for column_name in metric_columns:
        if column_name not in columns:
            columns.append(column_name)
    for column_name in param_columns:
        if column_name not in columns:
            columns.append(column_name)
    return columns


def _build_trial_summary_rows(
    trials: Sequence[object],
    *,
    manifest: Mapping[str, Any],
    manual_manifest: Mapping[str, Any],
    model_dir: Path,
    capture_params: bool = False,
) -> tuple[list[Dict[str, object]], list[str], list[str]]:
    """Return DataFrame-ready rows describing manual and Pareto artefacts."""

    rows: List[Dict[str, object]] = []
    metric_columns: list[str] = []
    param_columns: list[str] = []

    def _attach_metrics(row: Dict[str, object], source: object) -> None:
        for column_name, value in _collect_metric_columns(source).items():
            row[column_name] = value
            _append_unique(metric_columns, column_name)

    def _attach_params(row: Dict[str, object], source: object) -> None:
        if not capture_params:
            return
        for param_name, param_value in _collect_param_columns(source).items():
            row[param_name] = param_value
            _append_unique(param_columns, param_name)

    if manual_manifest:
        manual_identifier = manual_manifest.get("trial_number", "manual")
        manual_paths = manifest_artifact_paths(manual_manifest, model_dir)
        manual_model_path = manual_paths.get("model")
        manual_saved = manifest_artifacts_exist(manual_manifest, model_dir)
        validation, delta = _extract_trial_metrics(manual_manifest)
        manual_row: Dict[str, object] = {
            "Source": "Manual override",
            "Saved locally": "✅" if manual_saved else "❌",
            "Trial ID": manual_identifier,
            "Model path": _format_artifact_path(manual_model_path, model_dir),
            "Validation ROAUC": validation,
            "TSTR/TRTR ΔAUC": delta,
        }
        _attach_metrics(manual_row, manual_manifest)
        _attach_params(manual_row, manual_manifest)
        rows.append(manual_row)

    saved_trial_number = manifest.get("trial_number") if manifest else None
    manifest_paths = manifest_artifact_paths(manifest, model_dir) if manifest else {}
    saved_model_path = manifest_paths.get("model") if manifest_paths else None

    for trial in trials:
        if trial is None:
            continue
        if hasattr(trial, "number"):
            identifier: object = getattr(trial, "number")  # type: ignore[attr-defined]
        elif isinstance(trial, Mapping):
            identifier = cast(Mapping[str, object], trial).get("trial_number")
        else:
            identifier = None
        validation, delta = _extract_trial_metrics(trial)
        saved_locally = bool(
            identifier is not None and saved_trial_number == identifier
        )
        pareto_row: Dict[str, object] = {
            "Source": "Optuna Pareto",
            "Saved locally": "✅" if saved_locally else "❌",
            "Trial ID": identifier if identifier is not None else "",
            "Model path": _format_artifact_path(saved_model_path, model_dir)
            if saved_locally
            else "",
            "Validation ROAUC": validation,
            "TSTR/TRTR ΔAUC": delta,
        }
        _attach_metrics(pareto_row, trial)
        _attach_params(pareto_row, trial)
        rows.append(pareto_row)

    return rows, metric_columns, param_columns


def _build_manual_optuna_ranked_table(
    *,
    manual_manifest: Mapping[str, Any],
    model_manifest: Optional[Mapping[str, Any]],
    pareto_candidates: Sequence[Mapping[str, Any]],
    trials_df: pd.DataFrame,
    model_dir: Path,
) -> pd.DataFrame:
    """Return a ranked table prioritising manual overrides then Pareto trials."""

    rows, metric_columns, param_columns = _build_trial_summary_rows(
        list(pareto_candidates),
        manifest=model_manifest or {},
        manual_manifest=manual_manifest,
        model_dir=model_dir,
        capture_params=True,
    )

    seen_identifiers = {
        str(row.get("Trial ID")) for row in rows if row.get("Trial ID") not in {None, ""}
    }
    for candidate in pareto_candidates:
        seen_identifiers.add(str(candidate.get("trial_number")))

    if not trials_df.empty:
        for _, record in trials_df.iterrows():
            trial_identifier = record.get("trial_number")
            identifier_key = str(trial_identifier)
            if identifier_key in seen_identifiers:
                continue
            record_dict = record.to_dict()
            fallback_validation, fallback_delta = _extract_trial_metrics(record_dict)
            validation = record.get("validation_roauc", fallback_validation)
            if pd.isna(validation):
                validation = fallback_validation
            delta = record.get("tstr_trtr_delta_auc", fallback_delta)
            if pd.isna(delta):
                delta = fallback_delta
            row: Dict[str, object] = {
                "Source": "Optuna study",
                "Saved locally": "",
                "Trial ID": trial_identifier,
                "Model path": "",
                "Validation ROAUC": validation,
                "TSTR/TRTR ΔAUC": delta,
            }

            for column_name, value in _collect_metric_columns(record_dict).items():
                row[column_name] = value
                _append_unique(metric_columns, column_name)

            for column_name, value in _collect_param_columns(record_dict).items():
                row[column_name] = value
                _append_unique(param_columns, column_name)

            for column_name in record.index:
                if column_name in {
                    "trial_number",
                    "validation_roauc",
                    "tstr_trtr_delta_auc",
                    "params",
                }:
                    continue
                value = record.get(column_name)
                _append_unique(param_columns, str(column_name))
                row[column_name] = value

            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=_compose_trial_columns(metric_columns, param_columns))

    source_order = {"Manual override": 0, "Optuna Pareto": 1, "Optuna study": 2}

    def _sort_key(entry: Dict[str, object]) -> Tuple[int, float, float]:
        group = source_order.get(str(entry.get("Source")), 3)
        validation = _coerce_float(entry.get("Validation ROAUC"))
        delta = _coerce_float(entry.get("TSTR/TRTR ΔAUC"))
        validation_key = -validation if np.isfinite(validation) else float("inf")
        delta_key = abs(delta) if np.isfinite(delta) else float("inf")
        return (group, validation_key, delta_key)

    ranked_rows = sorted(rows, key=_sort_key)[:50]

    columns = _compose_trial_columns(metric_columns, param_columns)
    return pd.DataFrame(ranked_rows, columns=columns)


def collect_manual_and_optuna_overview(
    *,
    target_label: str,
    model_dir: Path,
    optuna_dir: Path,
    study_prefix: Optional[str],
    storage: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return manual manifest and Optuna trial summaries as dataframes."""

    manual_manifest = load_manual_model_manifest(model_dir, target_label)
    model_manifest = load_model_manifest(model_dir, target_label)
    best_info, _ = load_optuna_results(
        optuna_dir,
        target_label,
        study_prefix=study_prefix,
        storage=storage,
    )

    pareto_raw = best_info.get("pareto_front", [])
    pareto_candidates: List[Mapping[str, Any]] = []
    if isinstance(pareto_raw, Sequence):
        for entry in pareto_raw:
            if isinstance(entry, Mapping):
                pareto_candidates.append(dict(entry))

    summary_rows, summary_metric_columns, _ = _build_trial_summary_rows(
        pareto_candidates,
        manifest=model_manifest,
        manual_manifest=manual_manifest,
        model_dir=model_dir,
    )
    summary_columns = _compose_trial_columns(summary_metric_columns, [])
    summary_df = (
        pd.DataFrame(summary_rows, columns=summary_columns)
        if summary_rows
        else pd.DataFrame(columns=summary_columns)
    )

    trials_path = optuna_dir / f"optuna_trials_{target_label}.csv"
    if trials_path.exists():
        try:
            trials_df = pd.read_csv(trials_path)
        except Exception:
            trials_df = pd.DataFrame()
    else:
        trials_df = pd.DataFrame()

    ranked_df = _build_manual_optuna_ranked_table(
        manual_manifest=manual_manifest,
        model_manifest=model_manifest,
        pareto_candidates=pareto_candidates,
        trials_df=trials_df,
        model_dir=model_dir,
    )

    return summary_df, ranked_df


def summarise_pareto_trials(
    trials: Sequence["optuna.trial.FrozenTrial"],
    *,
    manifest: Mapping[str, Any],
    model_dir: Path,
    manual_manifest: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Return a tidy summary of Pareto-optimal Optuna trials."""

    rows, metric_columns, _ = _build_trial_summary_rows(
        list(trials),
        manifest=manifest,
        manual_manifest=manual_manifest or {},
        model_dir=model_dir,
    )
    columns = _compose_trial_columns(metric_columns, [])
    return pd.DataFrame(rows, columns=columns)


def load_manual_tuning_overrides(
    manual_config: Mapping[str, Any], manual_dir: Path
) -> Dict[str, Any]:
    """Return manual hyper-parameter overrides defined by ``manual_config``.

    Parameters
    ----------
    manual_config
        Manual tuning configuration specifying the module and attribute to
        import.
    manual_dir
        Directory searched for a ``<module>.py`` file that contains the manual
        overrides.

    Raises
    ------
    FileNotFoundError
        If the configured module cannot be imported from disk.
    RuntimeError
        If the configured attribute is missing or does not provide a mapping.
    """

    if not isinstance(manual_config, Mapping):
        return {}

    module_name = manual_config.get("module")
    attribute_name = str(manual_config.get("attribute", "manual_param_setting"))
    if not module_name:
        return {}

    module_path = manual_dir / f"{module_name}.py"

    module: Any
    if module_path.exists():
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError(f"Could not load manual overrides from {module_path!s}")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(module_name, module)
        spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            raise FileNotFoundError(
                f"Manual override module '{module_name}' was not found at {module_path!s}."
            ) from error

    if not hasattr(module, attribute_name):
        raise RuntimeError(
            f"Manual override attribute '{attribute_name}' was not found in module '{module_name}'."
        )

    raw_overrides = getattr(module, attribute_name)
    if not isinstance(raw_overrides, Mapping):
        raise RuntimeError(
            f"Manual override attribute '{attribute_name}' on module '{module_name}' must be a mapping."
        )

    return dict(raw_overrides)


def prompt_manual_override_action(
    *, input_fn: Callable[[str], str] = input
) -> str:
    """Return the manual override action selected by the user.

    The helper keeps the interactive experience identical across the example
    and research template scripts while remaining straightforward to test. When
    the caller runs in a non-interactive environment (``EOFError``), the helper
    defaults to continuing with the automatic Optuna search.

    Examples
    --------
    >>> prompt_manual_override_action(  # doctest: +SKIP
    ...     input_fn=lambda prompt: "manual",
    ... )
    'reuse'

    Parameters
    ----------
    input_fn
        Callable compatible with ``input`` that returns user responses. Tests
        can inject a stub to exercise the various decision branches.

    Returns
    -------
    str
        ``"train"`` when the user opts to fit overrides immediately,
        ``"reuse"`` to load saved manual artefacts, and ``"optuna"`` when the
        user chooses the automatic Optuna flow or leaves the prompt blank.
    """

    prompt = (
        "Manual tuning is enabled. Enter 'y' to train using the manual overrides, "
        "'manual' to reuse existing manual artefacts, 'n' to skip manual overrides, "
        "or press Enter to continue with Optuna: "
    )

    while True:
        try:
            response = input_fn(prompt)
        except EOFError:
            return "optuna"
        except KeyboardInterrupt:
            print("\nManual override selection interrupted.")
            try:
                confirm = input_fn(
                    "Cancel manual override selection and continue with Optuna? "
                    "Enter 'y' to confirm or press Enter to resume selection: "
                )
            except (KeyboardInterrupt, EOFError):
                print("\nContinuing with Optuna search.")
                return "optuna"
            if confirm.strip().lower() in {"y", "yes"}:
                print("Continuing with Optuna search.")
                return "optuna"
            print("Resuming manual override prompt.")
            continue

        lowered = response.strip().lower()
        if lowered in {"", "n", "no"}:
            return "optuna"
        if lowered in {"y", "yes"}:
            return "train"
        if lowered == "manual":
            return "reuse"
        print("Unrecognised response. Please enter 'y', 'n', 'manual', or press Enter.")


def evaluate_candidate_model_performance(
    model: SUAVE,
    *,
    feature_columns: Sequence[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    random_state: int,
    probability_fn: Optional[Callable[[SUAVE, pd.DataFrame], np.ndarray]] = None,
) -> Dict[str, Any]:
    """Compute validation and transfer metrics for a trained SUAVE candidate.

    Parameters
    ----------
    model
        Fitted SUAVE model to be assessed.
    feature_columns
        List of feature names used to build numeric matrices.
    X_train, y_train
        Training split used for TRTR evaluation.
    X_validation, y_validation
        Validation split used for calibration and TSTR evaluation.
    random_state
        Seed governing the synthetic label bootstrap used for TSTR metrics.
    probability_fn
        Optional callable that returns class probabilities for ``X_validation``.
        Defaults to ``model.predict_proba``.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing validation, TSTR, TRTR metrics alongside the
        Optuna objective values (validation ROAUC and |ΔAUC|).

    Examples
    --------
    >>> # ``model`` and dataset initialisation omitted for brevity
    >>> metrics = evaluate_candidate_model_performance(  # doctest: +SKIP
    ...     model,
    ...     feature_columns=["age", "apache"],
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     X_validation=X_val,
    ...     y_validation=y_val,
    ...     random_state=0,
    ... )
    >>> sorted(metrics.keys())  # doctest: +SKIP
    ['delta_auc', 'tstr_metrics', 'trtr_metrics', 'validation_metrics', 'values']
    """

    proba_fn = probability_fn or (lambda candidate, frame: candidate.predict_proba(frame))
    validation_probabilities = np.asarray(proba_fn(model, X_validation))
    validation_metrics = compute_binary_metrics(validation_probabilities, y_validation)

    roauc = _coerce_float(
        validation_metrics.get("ROAUC", validation_metrics.get("roauc", float("nan")))
    )
    if not np.isfinite(roauc):
        raise ValueError("Non-finite validation ROAUC")

    numeric_train = to_numeric_frame(X_train.loc[:, feature_columns])
    numeric_validation = to_numeric_frame(X_validation.loc[:, feature_columns])

    rng = np.random.default_rng(random_state)
    synthetic_labels = rng.choice(y_train, size=len(y_train), replace=True)
    synthetic_samples = model.sample(
        len(synthetic_labels), conditional=True, y=synthetic_labels
    )
    if isinstance(synthetic_samples, pd.DataFrame):
        synthetic_features = synthetic_samples.loc[:, feature_columns].copy()
    else:
        synthetic_features = pd.DataFrame(synthetic_samples, columns=feature_columns)
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

    trtr_auc = _coerce_float(trtr_metrics.get("auroc", trtr_metrics.get("ROAUC")))
    tstr_auc = _coerce_float(tstr_metrics.get("auroc", tstr_metrics.get("ROAUC")))
    delta_auc = (
        abs(float(trtr_auc) - float(tstr_auc))
        if np.isfinite(trtr_auc) and np.isfinite(tstr_auc)
        else float("nan")
    )

    return {
        "validation_metrics": validation_metrics,
        "tstr_metrics": tstr_metrics,
        "trtr_metrics": trtr_metrics,
        "delta_auc": delta_auc,
        "values": (
            roauc,
            delta_auc,
        ),
    }


def run_manual_override_training(
    *,
    target_label: str,
    manual_overrides: Mapping[str, Any],
    base_params: Optional[Mapping[str, Any]],
    override_on_history: bool = False,
    schema: Schema,
    feature_columns: Sequence[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    model_dir: Path,
    calibration_dir: Path,
    random_state: int,
) -> Dict[str, Any]:
    """Train and persist a manual SUAVE model using ``manual_overrides``.

    Parameters
    ----------
    override_on_history
        When ``True``, merge ``base_params`` from the latest Optuna history
        before applying manual overrides. Leave as ``False`` to rely solely on
        the overrides (falling back to SUAVE defaults when empty).
    """

    merged_params: Dict[str, Any] = {}
    if override_on_history and base_params:
        merged_params.update(dict(base_params))
    merged_params.update(dict(manual_overrides))

    model = build_suave_model(merged_params, schema, random_state=random_state)
    model.fit(X_train, y_train, plot_monitor=is_interactive_session(), **resolve_suave_fit_kwargs(merged_params))

    calibrator = fit_isotonic_calibrator(model, X_validation, y_validation)

    manual_model_path = model_dir / f"suave_manual_{target_label}.pt"
    manual_calibrator_path = (
        calibration_dir / f"isotonic_manual_calibrator_{target_label}.joblib"
    )
    manual_model_path.parent.mkdir(parents=True, exist_ok=True)
    manual_calibrator_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(manual_model_path)
    joblib.dump(calibrator, manual_calibrator_path)

    evaluation = evaluate_candidate_model_performance(
        model,
        feature_columns=feature_columns,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        random_state=random_state,
        probability_fn=lambda _model, features: calibrator.predict_proba(features),
    )
    validation_metrics = evaluation["validation_metrics"]
    tstr_metrics = evaluation["tstr_metrics"]
    trtr_metrics = evaluation["trtr_metrics"]
    values = evaluation["values"]

    manifest_path = record_manual_model_manifest(
        model_dir,
        target_label,
        model_path=manual_model_path,
        calibrator_path=manual_calibrator_path,
        params=merged_params,
        values=values,
        validation_metrics=validation_metrics,
        tstr_metrics=tstr_metrics,
        trtr_metrics=trtr_metrics,
        description="Interactive manual tuning run",
    )

    return {
        "model_path": manual_model_path,
        "calibrator_path": manual_calibrator_path,
        "manifest_path": manifest_path,
        "validation_metrics": validation_metrics,
        "tstr_metrics": tstr_metrics,
        "trtr_metrics": trtr_metrics,
        "values": values,
        "params": merged_params,
    }


# =============================================================================
# === Optuna-driven model resolution ==========================================
# =============================================================================


@dataclass
class ModelLoadingPlan:
    """Container describing how the SUAVE model should be initialised."""

    optuna_best_info: Dict[str, Any]
    optuna_best_params: Dict[str, Any]
    model_manifest: Dict[str, Any]
    manual_model_manifest: Dict[str, Any]
    optuna_study: Optional["optuna.study.Study"]
    pareto_trials: List["optuna.trial.FrozenTrial"]
    selected_trial_number: Optional[Union[str, int]]
    selected_model_path: Optional[Path]
    selected_calibrator_path: Optional[Path]
    selected_params: Dict[str, Any]
    preloaded_model: Optional[SUAVE]


# =============================================================================
# === Optuna trial scoring ====================================================
# =============================================================================


def choose_preferred_pareto_trial(
    trials: Sequence["optuna.trial.FrozenTrial"],
    *,
    min_validation_roauc: float = PARETO_MIN_VALIDATION_ROAUC,
    max_abs_delta_auc: float = PARETO_MAX_ABS_DELTA_AUC,
) -> Optional["optuna.trial.FrozenTrial"]:
    """Return the trial that best satisfies the Pareto-front constraints.

    Parameters
    ----------
    trials
        Pareto-optimal trials emitted by Optuna.
    min_validation_roauc
        Minimum acceptable validation ROC-AUC threshold.
    max_abs_delta_auc
        Maximum permissible absolute difference between TSTR/TRTR ROC-AUC.

    Examples
    --------
    >>> choose_preferred_pareto_trial([]) is None
    True
    """

    if not trials:
        return None

    def _objective_values(trial: "optuna.trial.FrozenTrial") -> Tuple[float, float]:
        values = trial.values or (float("nan"), float("nan"))
        primary = float(values[0])
        secondary = float(values[1]) if len(values) > 1 else float("nan")
        return primary, secondary

    def _within_constraints(trial: "optuna.trial.FrozenTrial") -> bool:
        primary, secondary = _objective_values(trial)
        return (
            primary > min_validation_roauc
            and np.isfinite(primary)
            and np.isfinite(secondary)
            and abs(secondary) < max_abs_delta_auc
        )

    constrained = [trial for trial in trials if _within_constraints(trial)]
    if constrained:
        return max(constrained, key=lambda trial: _objective_values(trial)[0])

    eligible: List[Tuple[float, "optuna.trial.FrozenTrial"]] = []
    for trial in trials:
        primary, _ = _objective_values(trial)
        if np.isfinite(primary) and primary > min_validation_roauc:
            # Track each eligible trial alongside its validation ROC-AUC so we can
            # deterministically recover the strongest performer.
            eligible.append((primary, trial))
    if eligible:
        return max(eligible, key=lambda item: item[0])[1]

    return max(trials, key=lambda trial: _objective_values(trial)[0])


def resolve_model_loading_plan(
    *,
    target_label: str,
    analysis_config: Mapping[str, Any],
    model_dir: Path,
    optuna_dir: Path,
    schema: Schema,
    is_interactive: bool,
    cli_requested_trial_id: Optional[Union[str, int]] = None,
    force_update_suave: bool = False,
    pareto_min_validation_roauc: float = PARETO_MIN_VALIDATION_ROAUC,
    pareto_max_abs_delta_auc: float = PARETO_MAX_ABS_DELTA_AUC,
) -> ModelLoadingPlan:
    """Determine how the evaluation workflow should obtain a SUAVE model.

    Parameters
    ----------
    target_label
        Name of the prediction task under evaluation.
    analysis_config
        Dictionary containing Optuna configuration such as ``optuna_storage``
        and ``optuna_study_prefix``.
    model_dir
        Directory storing persisted SUAVE checkpoints and calibrators.
    optuna_dir
        Directory containing Optuna metadata exported by the optimisation
        pipeline.
    schema
        Freshly inferred schema describing the current dataset layout.
    is_interactive
        When ``True`` the user is prompted to select an Optuna trial
        interactively; otherwise command-line arguments and sensible defaults
        drive the selection.
    cli_requested_trial_id
        Optional Optuna trial identifier or ``"manual"`` override supplied via
        command-line arguments.
    force_update_suave
        Boolean flag indicating whether cached SUAVE artefacts should be
        retrained when Optuna outputs are unavailable. When Optuna metadata can
        be loaded the flag has no effect, preserving Pareto trial selection.
    pareto_min_validation_roauc
        Minimum acceptable validation ROC-AUC used when choosing a Pareto
        candidate automatically.
    pareto_max_abs_delta_auc
        Maximum tolerated absolute TSTR/TRTR ROC-AUC difference when selecting
        Pareto trials.

    Returns
    -------
    ModelLoadingPlan
        Dataclass describing the chosen artefacts and fallback hyperparameters.

    Examples
    --------
    >>> schema = Schema({})
    >>> config = {"optuna_study_prefix": None, "optuna_storage": None}
    >>> plan = resolve_model_loading_plan(
    ...     target_label="mortality",
    ...     analysis_config=config,
    ...     model_dir=Path("/tmp"),
    ...     optuna_dir=Path("/tmp"),
    ...     schema=schema,
    ...     is_interactive=False,
    ... )
    >>> isinstance(plan.selected_params, dict)
    True
    """

    optuna_best_info, optuna_best_params = load_optuna_results(
        optuna_dir,
        target_label,
        study_prefix=analysis_config.get("optuna_study_prefix"),
        storage=analysis_config.get("optuna_storage"),
    )
    model_manifest = load_model_manifest(model_dir, target_label)
    manual_model_manifest = load_manual_model_manifest(model_dir, target_label)

    if not optuna_best_params:
        print(
            "Optuna best parameters were not found on disk; subsequent steps will rely on defaults unless the storage backend is available."
        )

    optuna_study = load_optuna_study(
        study_prefix=analysis_config.get("optuna_study_prefix"),
        target_label=target_label,
        storage=analysis_config.get("optuna_storage"),
    )

    pareto_trials: List["optuna.trial.FrozenTrial"] = []
    if optuna_study is not None:
        pareto_trials = [
            trial for trial in optuna_study.best_trials if trial.values is not None
        ]
        if not pareto_trials:
            pareto_trials = [
                trial for trial in optuna_study.trials if trial.values is not None
            ]

    manifest_paths = manifest_artifact_paths(model_manifest, model_dir)
    saved_model_path = manifest_paths.get("model")
    saved_calibrator_path = manifest_paths.get("calibrator")
    saved_trial_number = model_manifest.get("trial_number")

    manual_manifest_paths = manifest_artifact_paths(manual_model_manifest, model_dir)
    manual_model_path = manual_manifest_paths.get("model")
    manual_calibrator_path = manual_manifest_paths.get("calibrator")
    manual_identifier: Optional[Union[str, int]] = manual_model_manifest.get(
        "trial_number"
    )
    if manual_identifier is None and manual_model_path is not None:
        manual_identifier = "manual"
    manual_model_available = bool(
        manual_model_path is not None and manual_model_path.exists()
    )

    legacy_model_path = model_dir / f"suave_best_{target_label}.pt"
    legacy_calibrator_path = model_dir / f"isotonic_calibrator_{target_label}.joblib"

    pareto_lookup = {trial.number: trial for trial in pareto_trials}
    all_trials_lookup = (
        {trial.number: trial for trial in optuna_study.trials if trial.values is not None}
        if optuna_study is not None
        else {}
    )

    selected_trial_number: Optional[Union[str, int]] = None
    selected_model_path: Optional[Path] = None
    selected_calibrator_path: Optional[Path] = None
    selected_trial: Optional["optuna.trial.FrozenTrial"] = None

    requested_id = cli_requested_trial_id
    if requested_id is not None:
        if requested_id == "manual":
            if manual_model_available:
                selected_trial_number = manual_identifier or "manual"
                selected_model_path = manual_model_path
                selected_calibrator_path = manual_calibrator_path
                if not (
                    manual_calibrator_path is not None
                    and manual_calibrator_path.exists()
                ):
                    print(
                        "Manual override selected but calibrator artefacts were not found; an isotonic calibrator will be fitted."
                    )
            else:
                print(
                    "Manual override requested but suave_manual_manifest was not found; proceeding with Optuna selection."
                )
        else:
            if (
                saved_trial_number == requested_id
                and saved_model_path
                and saved_model_path.exists()
            ):
                selected_trial_number = requested_id
                selected_model_path = saved_model_path
                selected_calibrator_path = saved_calibrator_path
            else:
                selected_trial = all_trials_lookup.get(requested_id)
                if selected_trial is None:
                    print(
                        f"Requested Optuna trial #{requested_id} could not be located; proceeding with fallback selection."
                    )
                else:
                    selected_trial_number = requested_id

    if selected_model_path is None and selected_trial is None:
        if manual_model_available:
            selected_trial_number = manual_identifier or "manual"
            selected_model_path = manual_model_path
            selected_calibrator_path = manual_calibrator_path
            if not (
                manual_calibrator_path is not None
                and manual_calibrator_path.exists()
            ):
                print(
                    "Manual override manifest detected but calibrator artefacts were not found; an isotonic calibrator will be fitted."
                )
        elif saved_model_path and saved_model_path.exists():
            selected_trial_number = saved_trial_number
            selected_model_path = saved_model_path
            selected_calibrator_path = saved_calibrator_path
        elif legacy_model_path.exists() or legacy_calibrator_path.exists():
            selected_model_path = (
                legacy_model_path if legacy_model_path.exists() else None
            )
            selected_calibrator_path = (
                legacy_calibrator_path if legacy_calibrator_path.exists() else None
            )
        elif pareto_trials:
            selected_trial = choose_preferred_pareto_trial(
                pareto_trials,
                min_validation_roauc=pareto_min_validation_roauc,
                max_abs_delta_auc=pareto_max_abs_delta_auc,
            )
            if selected_trial is not None:
                selected_trial_number = selected_trial.number
        elif optuna_best_params:
            print(
                "Optuna study unavailable; using stored best parameters for training."
            )

    optuna_outputs_available = bool(optuna_study is not None or optuna_best_params)
    if force_update_suave and not optuna_outputs_available:
        if selected_model_path is not None or selected_calibrator_path is not None:
            print(
                "FORCE_UPDATE_SUAVE enabled and Optuna artefacts are unavailable; "
                "the saved SUAVE model will be retrained."
            )
        if selected_trial_number == saved_trial_number:
            selected_trial_number = None
        selected_model_path = None
        selected_calibrator_path = None

    manual_params = manual_model_manifest.get("params")
    selected_params: Dict[str, Any] = {}
    if selected_trial is not None:
        selected_params = dict(selected_trial.params)
    elif (
        selected_trial_number == saved_trial_number
        and isinstance(model_manifest.get("params"), Mapping)
    ):
        selected_params = dict(model_manifest["params"])
    elif (
        isinstance(manual_params, Mapping)
        and selected_trial_number
        in {manual_identifier, "manual"}
    ):
        selected_params = dict(manual_params)
    elif optuna_best_params:
        selected_params = dict(optuna_best_params)

    preloaded_model: Optional[SUAVE] = None
    if selected_model_path and selected_model_path.exists():
        try:
            preloaded_model = SUAVE.load(selected_model_path)
        except Exception as error:  # pragma: no cover - defensive logging
            print(f"Failed to load SUAVE model from {selected_model_path}: {error}")
            preloaded_model = None
        else:
            stored_schema = getattr(preloaded_model, "schema", None)
            if stored_schema is None:
                print(
                    "Warning: saved SUAVE model does not include schema metadata; continuing with the newly inferred schema."
                )
            else:
                if stored_schema.to_dict() != schema.to_dict():
                    print(
                        "Warning: schema embedded in the saved SUAVE model differs from the freshly inferred definition; consider retraining before continuing."
                    )

    return ModelLoadingPlan(
        optuna_best_info=dict(optuna_best_info),
        optuna_best_params=dict(optuna_best_params),
        model_manifest=dict(model_manifest),
        manual_model_manifest=dict(manual_model_manifest),
        optuna_study=optuna_study,
        pareto_trials=list(pareto_trials),
        selected_trial_number=selected_trial_number,
        selected_model_path=selected_model_path,
        selected_calibrator_path=selected_calibrator_path,
        selected_params=selected_params,
        preloaded_model=preloaded_model,
    )


def confirm_model_loading_plan_selection(
    plan: ModelLoadingPlan,
    *,
    is_interactive: bool,
    model_dir: Path,
) -> ModelLoadingPlan:
    """Confirm or adjust ``plan`` based on interactive user input.

    Parameters
    ----------
    plan
        The provisional loading strategy returned by :func:`resolve_model_loading_plan`.
    is_interactive
        Flag indicating whether the current session expects human interaction.
    model_dir
        Directory containing cached SUAVE model artefacts.

    Returns
    -------
    ModelLoadingPlan
        The original plan when no confirmation is required, otherwise a copy
        updated with the user's preferred trial selection.

    Examples
    --------
    >>> empty_plan = ModelLoadingPlan(
    ...     optuna_best_info={},
    ...     optuna_best_params={},
    ...     model_manifest={},
    ...     manual_model_manifest={},
    ...     optuna_study=None,
    ...     pareto_trials=[],
    ...     selected_trial_number=None,
    ...     selected_model_path=None,
    ...     selected_calibrator_path=None,
    ...     selected_params={},
    ...     preloaded_model=None,
    ... )
    >>> confirm_model_loading_plan_selection(
    ...     empty_plan,
    ...     is_interactive=True,
    ...     model_dir=Path("/tmp"),
    ... ) is empty_plan
    True
    """

    if not is_interactive or not plan.pareto_trials:
        return plan

    manifest_paths = manifest_artifact_paths(plan.model_manifest, model_dir)
    saved_model_path = manifest_paths.get("model")
    saved_calibrator_path = manifest_paths.get("calibrator")
    saved_trial_number = plan.model_manifest.get("trial_number")

    manual_manifest_paths = manifest_artifact_paths(
        plan.manual_model_manifest, model_dir
    )
    manual_model_path = manual_manifest_paths.get("model")
    manual_calibrator_path = manual_manifest_paths.get("calibrator")
    manual_identifier: Optional[Union[str, int]] = plan.manual_model_manifest.get(
        "trial_number"
    )
    if manual_identifier is None and manual_model_path is not None:
        manual_identifier = "manual"
    manual_available = bool(
        manual_model_path is not None and manual_model_path.exists()
    )
    manual_params: Dict[str, Any] = (
        dict(plan.manual_model_manifest["params"])
        if isinstance(plan.manual_model_manifest.get("params"), Mapping)
        else {}
    )

    default_hint = "a new training run"
    if manual_available and plan.selected_trial_number in {
        manual_identifier,
        "manual",
    }:
        default_hint = "the manual override"
    elif (
        saved_trial_number is not None
        and saved_model_path is not None
        and saved_model_path.exists()
    ):
        default_hint = f"trial #{saved_trial_number}"

    manual_prompt = (
        " or type 'manual' to load the manual override"
        if manual_available
        else ""
    )
    prompt = (
        "Enter the Optuna trial ID from the Pareto front to load or train"
        f"{manual_prompt} (press Enter to reuse {default_hint}): "
    )

    pareto_lookup = {trial.number: trial for trial in plan.pareto_trials}

    selected_trial_number = plan.selected_trial_number
    selected_model_path = plan.selected_model_path
    selected_calibrator_path = plan.selected_calibrator_path
    selected_params = dict(plan.selected_params)
    preloaded_model = plan.preloaded_model

    if (
        manual_available
        and selected_trial_number in {manual_identifier, "manual"}
        and not selected_params
        and manual_params
    ):
        selected_params = dict(manual_params)

    while True:
        try:
            response = input(prompt).strip()
        except EOFError:  # pragma: no cover - interactive safety net
            response = ""

        if not response:
            break

        lowered = response.lower()
        if manual_available and lowered == "manual":
            selected_trial = None
            selected_trial_number = manual_identifier or "manual"
            selected_model_path = manual_model_path
            selected_calibrator_path = manual_calibrator_path
            selected_params = dict(manual_params)
            if not (
                manual_calibrator_path is not None
                and manual_calibrator_path.exists()
            ):
                print(
                    "Manual override selected but calibrator artefacts were not found; an isotonic calibrator will be fitted."
                )
            if not (
                manual_model_path is not None
                and plan.selected_model_path == manual_model_path
            ):
                preloaded_model = None
            break

        try:
            candidate_id = int(response)
        except ValueError:
            print(
                "Please enter a valid integer trial identifier from the listed Pareto front or the keyword 'manual'."
            )
            continue

        if candidate_id not in pareto_lookup:
            print(
                "The specified trial is not part of the Pareto front; choose one of the displayed IDs."
            )
            continue

        selected_trial = pareto_lookup[candidate_id]
        selected_trial_number = candidate_id

        if (
            saved_trial_number == candidate_id
            and saved_model_path is not None
            and saved_model_path.exists()
        ):
            selected_model_path = saved_model_path
            selected_calibrator_path = saved_calibrator_path
            preloaded_model = plan.preloaded_model
        else:
            selected_model_path = None
            selected_calibrator_path = None
            preloaded_model = None

        selected_params = dict(selected_trial.params)
        break

    return replace(
        plan,
        selected_trial_number=selected_trial_number,
        selected_model_path=selected_model_path,
        selected_calibrator_path=selected_calibrator_path,
        selected_params=selected_params,
        preloaded_model=preloaded_model,
    )


def dataframe_to_markdown(df: pd.DataFrame, *, floatfmt: Optional[str] = ".3f") -> str:
    """Return a GitHub-flavoured Markdown representation of ``df``."""

    if df.empty:
        return "_No data available._"
    tabulate_kwargs = {"headers": "keys", "tablefmt": "github", "showindex": False}
    if floatfmt is not None:
        tabulate_kwargs["floatfmt"] = floatfmt
    return tabulate(df, **tabulate_kwargs)


def schema_to_dataframe(schema: Schema) -> pd.DataFrame:
    """Convert a :class:`Schema` into a tidy dataframe."""

    records: List[Dict[str, object]] = []
    for column, spec in schema.to_dict().items():
        records.append(
            {
                "Column": column,
                "Type": spec.get("type", ""),
                "n_classes": spec.get("n_classes", ""),
                "y_dim": spec.get("y_dim", ""),
            }
        )
    return pd.DataFrame(records)


def slugify_identifier(value: str) -> str:
    """Return a filesystem-friendly identifier derived from ``value``."""

    cleaned = [char.lower() if char.isalnum() else "_" for char in value.strip()]
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _normalise_bootstrap_metadata(
    metadata: Optional[Mapping[str, object]],
) -> Dict[str, object]:
    """Return a copy of ``metadata`` without ``None`` values sorted by key."""

    normalised: Dict[str, object] = {}
    if metadata is None:
        return normalised
    for key in sorted(metadata):
        value = metadata[key]
        if value is None:
            continue
        normalised[key] = value
    return normalised


def _build_bootstrap_cache_path(
    cache_root: Path,
    training_name: str,
    model_name: str,
    evaluation_name: str,
) -> Path:
    """Return the cache path for a bootstrap evaluation entry."""

    return (
        cache_root
        / slugify_identifier(training_name)
        / slugify_identifier(model_name)
        / f"{slugify_identifier(evaluation_name)}.joblib"
    )


def _build_prediction_signature(
    probabilities: np.ndarray, predictions: np.ndarray
) -> str:
    """Return a deterministic hash of ``probabilities`` and ``predictions``."""

    prob_array = np.asarray(probabilities)
    pred_array = np.asarray(predictions)
    return joblib.hash((prob_array, pred_array))


def _load_bootstrap_cache_entry(
    cache_path: Path, expected_metadata: Mapping[str, object]
) -> Optional[Dict[str, Any]]:
    """Load cached bootstrap results when ``expected_metadata`` matches."""

    try:
        payload = joblib.load(cache_path)
    except Exception:
        return None

    if not isinstance(payload, Mapping):
        return None

    cached_metadata = _normalise_bootstrap_metadata(payload.get("metadata"))
    if cached_metadata != _normalise_bootstrap_metadata(expected_metadata):
        return None

    results = payload.get("results")
    if not isinstance(results, Mapping):
        return None

    required_keys = {
        "overall",
        "per_class",
        "overall_records",
        "per_class_records",
        "bootstrap_overall_records",
        "bootstrap_per_class_records",
        "warnings",
    }
    if not required_keys.issubset(results.keys()):
        return None
    return {key: results[key] for key in required_keys}


def _save_bootstrap_cache_entry(
    cache_path: Path,
    *,
    metadata: Mapping[str, object],
    results: Mapping[str, pd.DataFrame],
) -> None:
    """Persist bootstrap evaluation artefacts to ``cache_path``."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "metadata": _normalise_bootstrap_metadata(metadata),
            "results": {key: results.get(key) for key in results},
        },
        cache_path,
    )


def load_or_create_iteratively_imputed_features(
    feature_sets: Mapping[str, pd.DataFrame],
    *,
    output_dir: Path,
    target_label: str,
    reference_key: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Path], bool]:
    """Load cached iterative imputations or fit a new imputer."""

    if reference_key not in feature_sets:
        raise KeyError(
            f"Reference key '{reference_key}' missing from feature sets: {list(feature_sets)}"
        )

    dataset_paths: Dict[str, Path] = {
        name: output_dir
        / f"iterative_imputed_{slugify_identifier(name)}_{slugify_identifier(target_label)}.csv"
        for name in feature_sets
    }

    loaded_features: Dict[str, pd.DataFrame] = {}
    load_successful = True
    for name, path in dataset_paths.items():
        features = feature_sets[name]
        if not path.exists():
            load_successful = False
            break
        cached = pd.read_csv(path, index_col=0)
        column_match = list(cached.columns) == list(features.columns)
        length_match = len(cached) == len(features)
        if not (column_match and length_match):
            load_successful = False
            break
        try:
            cached = cached.loc[features.index]
        except KeyError:
            load_successful = False
            break
        loaded_features[name] = cached

    if load_successful:
        return loaded_features, dataset_paths, True

    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    imputer = IterativeImputer(max_iter=100, tol=1e-2)
    imputer.fit(feature_sets[reference_key])

    imputed_features: Dict[str, pd.DataFrame] = {}
    for name, features in feature_sets.items():
        transformed = imputer.transform(features)
        imputed_df = pd.DataFrame(
            transformed,
            columns=features.columns,
            index=features.index,
        )
        path = dataset_paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        imputed_df.to_csv(path)
        imputed_features[name] = imputed_df

    return imputed_features, dataset_paths, False


def iteratively_impute_clinical_scores(
    score_frames: Mapping[str, pd.DataFrame],
    feature_frames: Mapping[str, pd.DataFrame],
    *,
    columns: Sequence[str],
    reference_key: str,
) -> Dict[str, pd.DataFrame]:
    """Return copies of ``score_frames`` with missing values imputed per score."""

    if reference_key not in score_frames:
        raise KeyError(
            f"Reference key '{reference_key}' missing from score frames: {list(score_frames)}"
        )
    if reference_key not in feature_frames:
        raise KeyError(
            f"Reference key '{reference_key}' missing from feature frames: {list(feature_frames)}"
        )

    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    imputed_frames = {name: frame.copy() for name, frame in score_frames.items()}

    reference_scores = score_frames[reference_key]
    available_columns = [
        column for column in columns if column in reference_scores.columns
    ]

    if not available_columns:
        return imputed_frames

    reference_features = feature_frames[reference_key]

    for column in available_columns:
        reference_column = reference_scores[column]
        if reference_column.isna().all():
            continue
        imputer = IterativeImputer(max_iter=100, tol=1e-2)
        training_matrix = pd.concat(
            [
                reference_features.reset_index(drop=True),
                reference_column.to_frame().reset_index(drop=True),
            ],
            axis=1,
        )
        imputer.fit(training_matrix)

        for dataset_name, score_frame in score_frames.items():
            if column not in score_frame.columns:
                continue
            feature_frame = feature_frames.get(dataset_name)
            if feature_frame is None:
                continue
            dataset_matrix = pd.concat(
                [
                    feature_frame.reset_index(drop=True),
                    score_frame[[column]].reset_index(drop=True),
                ],
                axis=1,
            )
            transformed = imputer.transform(dataset_matrix)
            imputed_series = pd.Series(
                transformed[:, -1], index=score_frame.index, name=column
            )
            imputed_frames[dataset_name][column] = imputed_series

    return imputed_frames


def make_logistic_pipeline(random_state: Optional[int] = None) -> Pipeline:
    """Factory for the baseline classifier used in TSTR/TRTR evaluations."""

    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    classifier = LogisticRegression()
    if random_state is not None:
        classifier.set_params(random_state=random_state)

    return Pipeline(
        [
            ("imputer", IterativeImputer(max_iter=100, tol=1e-2)),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a TSV file into a :class:`pandas.DataFrame`."""

    return pd.read_csv(path, sep="\t")


def make_random_forest_pipeline(random_state: Optional[int] = None) -> Pipeline:
    """Return a random forest pipeline with iterative imputation."""

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    classifier = RandomForestClassifier()
    if random_state is not None:
        classifier.set_params(random_state=random_state)

    return Pipeline(
        [
            ("imputer", IterativeImputer(max_iter=100, tol=1e-2)),
            ("classifier", classifier),
        ]
    )


def make_gradient_boosting_pipeline(
    random_state: Optional[int] = None,
) -> Pipeline:
    """Return a gradient boosting pipeline with iterative imputation."""

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    classifier = GradientBoostingClassifier()
    if random_state is not None:
        classifier.set_params(random_state=random_state)

    return Pipeline(
        [
            ("imputer", IterativeImputer(max_iter=100, tol=1e-2)),
            ("classifier", classifier),
        ]
    )


def make_baseline_model_factories(
    random_state: int,
    selected_models: Optional[Sequence[str]] = None,
) -> Dict[str, Callable[[], Pipeline]]:
    """Return model factories for the supervised transfer comparison."""

    all_factories: Dict[str, Callable[[], Pipeline]] = {
        "Logistic regression": lambda: make_logistic_pipeline(random_state),
        "Random forest": lambda: make_random_forest_pipeline(random_state),
        "GBDT": lambda: make_gradient_boosting_pipeline(random_state),
    }
    if selected_models is None:
        return all_factories

    filtered: Dict[str, Callable[[], Pipeline]] = {}
    for name in selected_models:
        if name in all_factories:
            filtered[name] = all_factories[name]
    return filtered if filtered else all_factories


def define_schema(
    df: pd.DataFrame, feature_columns: Iterable[str], mode: str = "info"
) -> Schema:
    """Create a :class:`Schema` describing ``df``'s feature columns."""

    inferencer = SchemaInferencer()
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


def _normalize_zero_indexed_labels(targets: pd.Series | np.ndarray) -> np.ndarray:
    """Return ``targets`` as an integer array with labels mapped to start at zero."""

    labels = np.asarray(targets)
    if labels.size == 0:
        if labels.dtype != int and not np.issubdtype(labels.dtype, np.integer):
            labels = labels.astype(int, copy=False)
        return labels

    if labels.dtype != int and not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(int, copy=False)

    unique = np.unique(labels)
    if np.array_equal(unique, np.arange(unique.size)):
        return labels

    return np.searchsorted(unique, labels)


def compute_auc(probabilities: np.ndarray, targets: pd.Series | np.ndarray) -> float:
    """Return the ROC AUC given predicted probabilities and targets."""

    labels = _normalize_zero_indexed_labels(targets)

    try:
        return float(compute_auroc(probabilities, labels))
    except ValueError:
        return float("nan")


def to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns in ``df`` to numeric values."""

    numeric = df.copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric


def _ensure_feature_frame(
    samples: pd.DataFrame | np.ndarray,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Return ``samples`` restricted to ``feature_columns`` as a dataframe."""

    if isinstance(samples, pd.DataFrame):
        missing = [
            column for column in feature_columns if column not in samples.columns
        ]
        if missing:
            raise KeyError(
                "Sampled dataframe is missing expected feature columns: " f"{missing}"
            )
        frame = samples.loc[:, list(feature_columns)].copy()
    else:
        frame = pd.DataFrame(samples, columns=list(feature_columns))
    return frame


def _generate_balanced_labels(
    labels: Sequence[object],
    total_samples: int,
    *,
    random_state: int,
) -> np.ndarray:
    """Generate a balanced label vector using ``labels`` as candidates."""

    unique = np.unique(np.asarray(labels))
    if unique.size == 0:
        raise ValueError("Cannot balance labels when no classes are present")

    base = total_samples // unique.size
    remainder = total_samples % unique.size
    counts = {value: base for value in unique}

    rng = np.random.default_rng(random_state)
    if remainder > 0:
        extras = rng.choice(unique, size=remainder, replace=False)
        for value in extras:
            counts[value] += 1

    balanced = np.concatenate([np.full(counts[value], value) for value in unique])
    shuffle_rng = np.random.default_rng(random_state + 1)
    shuffle_rng.shuffle(balanced)
    return balanced


def build_tstr_training_sets(
    model: SUAVE,
    feature_columns: Sequence[str],
    real_features: pd.DataFrame,
    real_labels: pd.Series,
    *,
    random_state: int,
    return_raw: bool = False,
) -> Union[
    Dict[str, Tuple[pd.DataFrame, pd.Series]],
    Tuple[
        Dict[str, Tuple[pd.DataFrame, pd.Series]],
        Dict[str, Tuple[pd.DataFrame, pd.Series]],
    ],
]:
    """Construct real and synthetic training sets for TSTR/TRTR evaluation.

    Parameters
    ----------
    model:
        The fitted :class:`SUAVE` model used to generate synthetic samples.
    feature_columns:
        Ordered collection of feature column names to preserve in each dataset.
    real_features:
        Feature frame from the source domain (e.g., MIMIC train split).
    real_labels:
        Corresponding label series aligned with ``real_features``.
    random_state:
        Seed controlling synthetic sampling reproducibility.
    return_raw:
        When ``True``, also return schema-aligned (non-numeric) feature frames for
        each training set. These are required when re-training SUAVE models that
        expect categorical values rather than numeric casts.

    Returns
    -------
    datasets : Dict[str, Tuple[pd.DataFrame, pd.Series]]
        Mapping from dataset name to a tuple of numeric feature frame and label
        series.
    raw_datasets : Dict[str, Tuple[pd.DataFrame, pd.Series]]
        Only returned when ``return_raw`` is ``True``. Contains the same data as
        ``datasets`` but with schema-aligned feature frames prior to
        ``to_numeric_frame`` conversion.
    """

    feature_columns = list(feature_columns)
    raw_real_features = real_features.loc[:, feature_columns].reset_index(drop=True)
    real_label_series = pd.Series(real_labels).reset_index(drop=True)
    real_label_series.name = real_labels.name

    raw_datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
        "TRTR (real)": (
            raw_real_features.copy(),
            real_label_series.copy(),
        )
    }
    datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
        name: (
            to_numeric_frame(features).reset_index(drop=True),
            labels.copy(),
        )
        for name, (features, labels) in raw_datasets.items()
    }

    n_train = len(real_label_series)
    label_array = real_label_series.to_numpy()

    def sample_features(
        n_samples: int,
        *,
        conditional: bool,
        labels: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        sampled = model.sample(
            n_samples,
            conditional=conditional,
            y=labels if conditional else None,
        )
        frame = _ensure_feature_frame(sampled, feature_columns)
        return frame.reset_index(drop=True)

    # Conditional sampling that mirrors the empirical class distribution.
    unconditional_labels = np.random.default_rng(random_state).choice(
        label_array,
        size=n_train,
        replace=True,
    )
    synthesis_features = sample_features(
        n_train,
        conditional=True,
        labels=np.asarray(unconditional_labels),
    )
    synthesis_labels = pd.Series(unconditional_labels, name=real_label_series.name)
    datasets["TSTR"] = (
        to_numeric_frame(synthesis_features.copy()).reset_index(drop=True),
        synthesis_labels.copy(),
    )
    raw_datasets["TSTR"] = (
        synthesis_features,
        synthesis_labels,
    )

    balanced_labels = _generate_balanced_labels(
        label_array,
        n_train,
        random_state=random_state + 10,
    )
    balance_features = sample_features(
        len(balanced_labels),
        conditional=True,
        labels=np.asarray(balanced_labels),
    )
    balance_labels = pd.Series(balanced_labels, name=real_label_series.name)
    datasets["TSTR balance"] = (
        to_numeric_frame(balance_features.copy()).reset_index(drop=True),
        balance_labels.copy(),
    )
    raw_datasets["TSTR balance"] = (
        balance_features,
        balance_labels,
    )

    label_counts = real_label_series.value_counts().sort_index()
    target_count = int(label_counts.max()) if not label_counts.empty else 0
    augmented_features_raw = [raw_real_features.copy()]
    augmented_labels = [real_label_series.copy()]
    for value, count in label_counts.items():
        deficit = target_count - int(count)
        if deficit <= 0:
            continue
        class_labels = np.full(deficit, value)
        synthetic_block = sample_features(
            deficit,
            conditional=True,
            labels=class_labels,
        )
        augmented_features_raw.append(synthetic_block)
        augmented_labels.append(pd.Series(class_labels, name=real_label_series.name))

    raw_augmented = pd.concat(augmented_features_raw, ignore_index=True)
    augmented_labels_series = pd.concat(
        augmented_labels, ignore_index=True
    ).reset_index(drop=True)
    datasets["TSTR augment"] = (
        to_numeric_frame(raw_augmented).reset_index(drop=True),
        augmented_labels_series,
    )
    raw_datasets["TSTR augment"] = (
        raw_augmented,
        augmented_labels_series.copy(),
    )

    five_x = n_train * 5
    five_x_labels = np.random.default_rng(random_state + 20).choice(
        label_array,
        size=five_x,
        replace=True,
    )
    five_x_features = sample_features(
        five_x,
        conditional=True,
        labels=np.asarray(five_x_labels),
    )
    five_x_series = pd.Series(five_x_labels, name=real_label_series.name)
    datasets["TSTR 5x"] = (
        to_numeric_frame(five_x_features.copy()).reset_index(drop=True),
        five_x_series.copy(),
    )
    raw_datasets["TSTR 5x"] = (
        five_x_features,
        five_x_series,
    )

    five_x_balanced = _generate_balanced_labels(
        label_array,
        five_x,
        random_state=random_state + 30,
    )
    five_x_balance_features = sample_features(
        len(five_x_balanced),
        conditional=True,
        labels=np.asarray(five_x_balanced),
    )
    five_x_balance_labels = pd.Series(five_x_balanced, name=real_label_series.name)
    datasets["TSTR 5x balance"] = (
        to_numeric_frame(five_x_balance_features.copy()).reset_index(drop=True),
        five_x_balance_labels.copy(),
    )
    raw_datasets["TSTR 5x balance"] = (
        five_x_balance_features,
        five_x_balance_labels,
    )

    ten_x = n_train * 10
    ten_x_labels = np.random.default_rng(random_state + 40).choice(
        label_array,
        size=ten_x,
        replace=True,
    )
    ten_x_features = sample_features(
        ten_x,
        conditional=True,
        labels=np.asarray(ten_x_labels),
    )
    ten_x_series = pd.Series(ten_x_labels, name=real_label_series.name)
    datasets["TSTR 10x"] = (
        to_numeric_frame(ten_x_features.copy()).reset_index(drop=True),
        ten_x_series.copy(),
    )
    raw_datasets["TSTR 10x"] = (
        ten_x_features,
        ten_x_series,
    )

    ten_x_balanced = _generate_balanced_labels(
        label_array,
        ten_x,
        random_state=random_state + 50,
    )
    ten_x_balance_features = sample_features(
        len(ten_x_balanced),
        conditional=True,
        labels=np.asarray(ten_x_balanced),
    )
    ten_x_balance_labels = pd.Series(
        ten_x_balanced, name=real_label_series.name
    )
    datasets["TSTR 10x balance"] = (
        to_numeric_frame(ten_x_balance_features.copy()).reset_index(drop=True),
        ten_x_balance_labels.copy(),
    )
    raw_datasets["TSTR 10x balance"] = (
        ten_x_balance_features,
        ten_x_balance_labels,
    )

    if return_raw:
        return datasets, raw_datasets
    return datasets


def save_tstr_training_sets_to_tsv(
    raw_datasets: Mapping[str, Tuple[pd.DataFrame, pd.Series]],
    *,
    output_dir: Path,
    target_label: str,
    feature_columns: Sequence[str],
    random_state: Optional[int] = None,
) -> Tuple[Path, str]:
    """Persist TSTR/TRTR training sets to TSV files with a JSON manifest.

    Returns
    -------
    manifest_path, manifest_signature:
        The JSON manifest path and a SHA256 signature used to validate
        downstream caches.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"manifest_{slugify_identifier(target_label)}.json"
    manifest: Dict[str, object] = {
        "target_label": target_label,
        "feature_columns": list(feature_columns),
        "datasets": [],
    }
    if random_state is not None:
        manifest["random_state"] = int(random_state)
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()

    for dataset_name, (features, labels) in raw_datasets.items():
        dataset_frame = features.loc[:, feature_columns].reset_index(drop=True).copy()
        label_series = pd.Series(labels).reset_index(drop=True)
        dataset_frame[target_label] = label_series.values
        filename = f"{slugify_identifier(dataset_name)}.tsv"
        dataset_frame.to_csv(output_dir / filename, index=False)
        manifest["datasets"].append({
            "name": dataset_name,
            "filename": filename,
        })

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)

    signature = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return manifest_path, signature


def load_tstr_training_sets_from_tsv(
    output_dir: Path,
    *,
    target_label: str,
    feature_columns: Sequence[str],
) -> Optional[Tuple[
    Dict[str, Tuple[pd.DataFrame, pd.Series]],
    Dict[str, Tuple[pd.DataFrame, pd.Series]],
    str,
]]:
    """Load cached TSTR/TRTR training sets from TSV files if available.

    Returns
    -------
    datasets, raw_datasets, manifest_signature:
        Numeric and raw training sets, along with the SHA256 signature of the
        cached manifest. ``None`` is returned when the manifest is missing or
        incompatible with the current configuration.
    """

    output_dir = Path(output_dir)
    manifest_path = output_dir / f"manifest_{slugify_identifier(target_label)}.json"
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    cached_features = manifest.get("feature_columns")
    if list(cached_features or []) != list(feature_columns):
        return None

    datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    raw_datasets: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    for entry in manifest.get("datasets", []):
        name = entry.get("name")
        filename = entry.get("filename")
        if not name or not filename:
            continue
        dataset_path = output_dir / str(filename)
        if not dataset_path.exists():
            return None
        frame = pd.read_csv(dataset_path)
        if target_label not in frame.columns:
            return None
        labels = frame[target_label].reset_index(drop=True)
        features = frame.drop(columns=[target_label]).reset_index(drop=True)
        raw_datasets[name] = (features, labels)
        datasets[name] = (to_numeric_frame(features), labels.copy())

    if not datasets:
        return None

    signature = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return datasets, raw_datasets, signature


def collect_transfer_bootstrap_records(
    nested_results: Mapping[str, Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]]],
    *,
    metrics: Sequence[str] = ("accuracy", "roc_auc"),
) -> pd.DataFrame:
    """Flatten bootstrap metrics from transfer evaluations into long format."""

    rows: List[Dict[str, object]] = []
    for training_name, model_map in nested_results.items():
        for model_name, evaluation_map in model_map.items():
            for evaluation_name, result_map in evaluation_map.items():
                bootstrap_df = result_map.get("bootstrap_overall_records")
                if bootstrap_df is None or bootstrap_df.empty:
                    continue
                iteration = (
                    bootstrap_df["iteration"].to_numpy()
                    if "iteration" in bootstrap_df.columns
                    else np.arange(len(bootstrap_df))
                )
                for metric in metrics:
                    if metric not in bootstrap_df.columns:
                        continue
                    metric_values = bootstrap_df[metric].to_numpy()
                    for idx, value in zip(iteration, metric_values):
                        if pd.isna(value):
                            continue
                        rows.append(
                            {
                                "training_dataset": training_name,
                                "evaluation_dataset": evaluation_name,
                                "model": model_name,
                                "iteration": int(idx) if not pd.isna(idx) else None,
                                "metric": metric,
                                "value": float(value),
                            }
                        )
    return pd.DataFrame(rows)


def compute_transfer_delta_bootstrap(
    bootstrap_df: pd.DataFrame,
    *,
    baseline_training_dataset: str,
    metrics: Sequence[str] = ("accuracy", "roc_auc"),
) -> pd.DataFrame:
    """Return bootstrap differences relative to ``baseline_training_dataset``."""

    if bootstrap_df.empty:
        return pd.DataFrame(columns=bootstrap_df.columns)

    required = {"training_dataset", "evaluation_dataset", "model", "metric", "value"}
    if not required.issubset(bootstrap_df.columns):
        return pd.DataFrame(columns=bootstrap_df.columns)

    base_df = bootstrap_df[
        bootstrap_df["training_dataset"] == baseline_training_dataset
    ].copy()
    if base_df.empty:
        return pd.DataFrame(columns=bootstrap_df.columns)

    comparison_df = bootstrap_df[
        bootstrap_df["training_dataset"] != baseline_training_dataset
    ].copy()
    if comparison_df.empty:
        return pd.DataFrame(columns=bootstrap_df.columns)

    metric_set = set(metrics)
    comparison_df = comparison_df[
        comparison_df["metric"].isin(metric_set)
    ].copy()
    base_df = base_df[base_df["metric"].isin(metric_set)].copy()
    if comparison_df.empty or base_df.empty:
        return pd.DataFrame(columns=bootstrap_df.columns)

    merge_keys = ["evaluation_dataset", "model", "metric"]
    if "iteration" in base_df.columns and "iteration" in comparison_df.columns:
        merge_keys.append("iteration")

    merged = comparison_df.merge(
        base_df[
            merge_keys
            + (["value"] if "value" in base_df.columns else [])
            + (["training_dataset"] if "training_dataset" in base_df.columns else [])
        ].rename(columns={"value": "baseline_value"}),
        on=merge_keys,
        how="inner",
        suffixes=("", "_baseline"),
    )

    if merged.empty:
        return pd.DataFrame(columns=bootstrap_df.columns)

    merged["value"] = merged["value"] - merged["baseline_value"]
    merged["metric"] = merged["metric"].map(lambda name: f"delta_{name}")
    drop_cols = ["baseline_value"]
    baseline_training_col = "training_dataset_baseline"
    if baseline_training_col in merged.columns:
        drop_cols.append(baseline_training_col)
    merged = merged.drop(columns=drop_cols)
    return merged


def evaluate_transfer_baselines(
    training_sets: Mapping[str, Tuple[pd.DataFrame, pd.Series]],
    evaluation_sets: Mapping[str, Tuple[pd.DataFrame, pd.Series]],
    *,
    model_factories: Mapping[str, Callable[[], Pipeline]],
    bootstrap_n: int,
    random_state: int,
    raw_training_sets: Optional[Mapping[str, Tuple[pd.DataFrame, pd.Series]]] = None,
    raw_evaluation_sets: Optional[Mapping[str, Tuple[pd.DataFrame, pd.Series]]] = None,
    bootstrap_cache_dir: Optional[Path] = None,
    bootstrap_cache_metadata: Optional[Mapping[str, object]] = None,
    force_update_bootstrap: bool = False,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]
]:
    """Train classical models on each training set and evaluate with bootstraps.

    Parameters
    ----------
    training_sets, evaluation_sets:
        Numeric feature frames paired with label series for each dataset.
    model_factories:
        Mapping from model name to a callable producing a scikit-learn compatible
        estimator.
    bootstrap_n:
        Number of bootstrap samples for metric confidence intervals.
    random_state:
        Seed for reproducible bootstrapping.
    raw_training_sets, raw_evaluation_sets:
        Optional schema-aligned feature frames keyed identically to
        ``training_sets`` and ``evaluation_sets``. Estimators declaring the
        attribute ``requires_schema_aligned_features`` will be trained and
        evaluated using these raw frames.
    bootstrap_cache_dir:
        Directory used to persist per-dataset bootstrap artefacts. When
        provided, cached entries are reused whenever ``force_update_bootstrap``
        is ``False`` and the stored metadata matches ``bootstrap_cache_metadata``
        together with the current prediction signature.
    bootstrap_cache_metadata:
        Additional provenance information recorded alongside each cache entry
        (for example the TSTR manifest signature or the SUAVE model identifier).
    force_update_bootstrap:
        Skip cache reuse for this call when set to ``True``.
    """

    summary_rows: List[Dict[str, object]] = []
    long_rows: List[Dict[str, object]] = []
    nested_results: Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]] = {}

    cache_root = Path(bootstrap_cache_dir) if bootstrap_cache_dir else None
    cache_metadata = _normalise_bootstrap_metadata(bootstrap_cache_metadata)

    training_items = list(training_sets.items())
    training_progress = tqdm(
        training_items,
        desc="TSTR/TRTR | training datasets",
        leave=False,
    )
    for training_name, (train_X_numeric, train_y_numeric) in training_progress:
        training_progress.set_postfix_str(training_name)
        nested_results.setdefault(training_name, {})
        raw_training = (
            raw_training_sets.get(training_name)
            if raw_training_sets is not None
            else None
        )
        model_items = list(model_factories.items())
        model_progress = tqdm(
            model_items,
            desc=f"Models @ {training_name}",
            leave=False,
        )
        for model_name, factory in model_progress:
            model_progress.set_postfix_str(model_name)
            estimator = factory()
            use_raw_features = getattr(
                estimator, "requires_schema_aligned_features", False
            )
            if use_raw_features:
                if raw_training is None:
                    raise ValueError(
                        "Raw training data missing for estimator requiring schema-aligned"
                        f" features on training set '{training_name}'."
                    )
                train_X, train_y = raw_training
            else:
                train_X, train_y = train_X_numeric, train_y_numeric

            estimator.fit(train_X, train_y)
            nested_results[training_name].setdefault(model_name, {})
            train_columns = list(train_X.columns)
            classes = getattr(estimator, "classes_", None)
            if classes is None:
                classes = np.unique(np.asarray(train_y))
            class_names = [str(value) for value in classes]
            positive_label = class_names[-1] if len(class_names) == 2 else None

            evaluation_items = list(evaluation_sets.items())
            evaluation_progress = tqdm(
                evaluation_items,
                desc=f"Evaluate @ {training_name} | {model_name}",
                leave=False,
            )
            for evaluation_name, (
                eval_X_numeric,
                eval_y_numeric,
            ) in evaluation_progress:
                evaluation_progress.set_postfix_str(evaluation_name)
                if use_raw_features:
                    if (
                        raw_evaluation_sets is None
                        or evaluation_name not in raw_evaluation_sets
                    ):
                        raise ValueError(
                            "Raw evaluation data missing for estimator requiring schema-aligned"
                            f" features on evaluation set '{evaluation_name}'."
                        )
                    eval_X, eval_y = raw_evaluation_sets[evaluation_name]
                else:
                    eval_X, eval_y = eval_X_numeric, eval_y_numeric

                if eval_X.empty or len(eval_y) == 0:
                    continue
                aligned_eval = eval_X.loc[:, train_columns]
                probabilities = estimator.predict_proba(aligned_eval)
                predictions = estimator.predict(aligned_eval)
                prediction_df = build_prediction_dataframe(
                    probabilities,
                    eval_y,
                    predictions,
                    class_names,
                )

                prediction_signature = _build_prediction_signature(
                    probabilities, predictions
                )

                metadata = dict(cache_metadata)
                metadata.update(
                    {
                        "training_dataset": training_name,
                        "evaluation_dataset": evaluation_name,
                        "model": model_name,
                        "bootstrap_n": int(bootstrap_n),
                        "prediction_signature": prediction_signature,
                    }
                )

                results: Dict[str, pd.DataFrame]
                cache_path: Optional[Path] = None
                cached_results: Optional[Dict[str, pd.DataFrame]] = None
                if cache_root is not None:
                    cache_path = _build_bootstrap_cache_path(
                        cache_root,
                        training_name,
                        model_name,
                        evaluation_name,
                    )
                    if cache_path.exists() and not force_update_bootstrap:
                        cached_results = _load_bootstrap_cache_entry(
                            cache_path, metadata
                        )

                if cached_results is None:
                    results = evaluate_predictions(
                        prediction_df,
                        label_col="label",
                        pred_col="y_pred",
                        positive_label=positive_label,
                        bootstrap_n=bootstrap_n,
                        random_state=random_state,
                        show_progress=True,
                        progress_desc=(
                            "Bootstrap | "
                            f"{model_name} | {training_name}→{evaluation_name}"
                        ),
                    )
                    if cache_path is not None:
                        _save_bootstrap_cache_entry(
                            cache_path, metadata=metadata, results=results
                        )
                else:
                    results = dict(cached_results)
                nested_results[training_name][model_name][evaluation_name] = results

                overall_df = results.get("overall", pd.DataFrame())
                if overall_df.empty:
                    continue
                row: Dict[str, object] = {
                    "training_dataset": training_name,
                    "evaluation_dataset": evaluation_name,
                    "model": model_name,
                }
                for metric in ("accuracy", "roc_auc"):
                    if metric in overall_df.columns:
                        value = overall_df.at[0, metric]
                        row[metric] = float(value)
                        low_col = f"{metric}_ci_low"
                        high_col = f"{metric}_ci_high"
                        row[low_col] = (
                            float(overall_df.at[0, low_col])
                            if low_col in overall_df.columns
                            else float("nan")
                        )
                        row[high_col] = (
                            float(overall_df.at[0, high_col])
                            if high_col in overall_df.columns
                            else float("nan")
                        )
                        long_rows.append(
                            {
                                "training_dataset": training_name,
                                "evaluation_dataset": evaluation_name,
                                "model": model_name,
                                "metric": metric,
                                "estimate": float(value),
                                "ci_low": (
                                    float(overall_df.at[0, low_col])
                                    if low_col in overall_df.columns
                                    else float("nan")
                                ),
                                "ci_high": (
                                    float(overall_df.at[0, high_col])
                                    if high_col in overall_df.columns
                                    else float("nan")
                                ),
                            }
                        )
                    else:
                        row[metric] = float("nan")
                        row[f"{metric}_ci_low"] = float("nan")
                        row[f"{metric}_ci_high"] = float("nan")
                summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    long_df = pd.DataFrame(long_rows)
    return summary_df, long_df, nested_results


def _generate_balanced_labels(
    labels: Sequence[object],
    total_samples: int,
    *,
    random_state: int,
) -> np.ndarray:
    """Generate a balanced label vector using ``labels`` as candidates."""

    unique = np.unique(np.asarray(labels))
    if unique.size == 0:
        raise ValueError("Cannot balance labels when no classes are present")

    base = total_samples // unique.size
    remainder = total_samples % unique.size
    counts = {value: base for value in unique}

    rng = np.random.default_rng(random_state)
    if remainder > 0:
        extras = rng.choice(unique, size=remainder, replace=False)
        for value in extras:
            counts[value] += 1

    balanced = np.concatenate([np.full(counts[value], value) for value in unique])
    shuffle_rng = np.random.default_rng(random_state + 1)
    shuffle_rng.shuffle(balanced)
    return balanced


def extract_positive_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Return the positive-class probabilities as a 1-D array."""

    prob_matrix = np.asarray(probabilities)
    if prob_matrix.ndim == 1:
        return prob_matrix
    return prob_matrix[:, -1]


def compute_binary_metrics(
    probabilities: np.ndarray, targets: pd.Series | np.ndarray
) -> Dict[str, float]:
    """Compute AUROC, accuracy, specificity, sensitivity, and Brier score."""

    labels = _normalize_zero_indexed_labels(targets)

    try:
        classification = evaluate_classification(probabilities, labels)
    except ValueError:
        classification = {
            "accuracy": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
        }

    positive_probs = extract_positive_probabilities(probabilities)
    predictions = (positive_probs >= 0.5).astype(int, copy=False)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")

    return {
        "ROAUC": classification.get("auroc", float("nan")),
        "AUC": classification.get("auroc", float("nan")),
        "ACC": classification.get("accuracy", float("nan")),
        "SPE": specificity,
        "SEN": sensitivity,
        "Brier": classification.get("brier", float("nan")),
    }


class IsotonicProbabilityCalibrator:
    """Lightweight isotonic calibrator compatible with SUAVE classifiers."""

    def __init__(
        self,
        base_estimator: SUAVE,
        isotonic: IsotonicRegression,
        classes: np.ndarray,
    ) -> None:
        self.base_estimator = base_estimator
        self.estimator = base_estimator
        self.isotonic_ = isotonic
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self.base_estimator.predict_proba(X))
        positive = extract_positive_probabilities(raw)
        calibrated = np.clip(self.isotonic_.predict(positive), 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(self.predict_proba(X))
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - simple proxy
        try:
            base = object.__getattribute__(self, "base_estimator")
        except AttributeError as exc:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            ) from exc
        if base is None or base is self:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            )
        return getattr(base, name)


def fit_isotonic_calibrator(
    model: SUAVE,
    features: pd.DataFrame,
    targets: pd.Series | np.ndarray,
) -> IsotonicProbabilityCalibrator:
    """Return an isotonic calibrator tailored for SUAVE probability outputs."""

    labels = _normalize_zero_indexed_labels(targets)
    unique = np.unique(labels)
    if unique.size < 2:
        raise ValueError("Isotonic calibration requires at least two classes.")
    if unique.size > 2:
        raise ValueError("Isotonic calibration currently supports binary tasks only.")

    raw_probabilities = np.asarray(model.predict_proba(features))
    positive = extract_positive_probabilities(raw_probabilities)

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(positive, labels.astype(float, copy=False))

    classes = getattr(model, "classes_", np.array([0, 1]))
    return IsotonicProbabilityCalibrator(model, isotonic, np.asarray(classes))


def plot_calibration_curves(
    probability_map: Mapping[str, np.ndarray],
    label_map: Mapping[str, np.ndarray],
    *,
    target_name: str,
    output_path: Path,
    n_bins: int = 10,
) -> None:
    """Generate calibration curves with Brier scores annotated in the legend."""

    calibration_records: List[Tuple[str, np.ndarray, np.ndarray, float]] = []
    for dataset_name, probs in probability_map.items():
        labels = label_map[dataset_name]
        pos_probs = extract_positive_probabilities(probs)
        try:
            frac_pos, mean_pred = calibration_curve(labels, pos_probs, n_bins=n_bins)
        except ValueError:
            continue
        brier = brier_score_loss(labels, pos_probs)
        calibration_records.append((dataset_name, mean_pred, frac_pos, brier))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")

    if not calibration_records:
        ax.text(0.5, 0.5, "Insufficient variation", ha="center", va="center")
        axis_min, axis_max = 0.0, 1.0
    else:
        x_values = np.concatenate([record[1] for record in calibration_records])
        y_values = np.concatenate([record[2] for record in calibration_records])
        combined = np.concatenate([x_values, y_values])
        axis_min = float(np.nanmin(combined))
        axis_max = float(np.nanmax(combined))
        if axis_min == axis_max:
            axis_min -= 0.05
            axis_max += 0.05
        padding = max((axis_max - axis_min) * 0.05, 1e-3)
        axis_min -= padding
        axis_max += padding

        reference = np.linspace(axis_min, axis_max, 2)
        ax.plot(reference, reference, linestyle="--", color="tab:gray", label="Perfect calibration")

        for dataset_name, mean_pred, frac_pos, brier in calibration_records:
            ax.plot(
                mean_pred,
                frac_pos,
                marker="o",
                label=f"{dataset_name} (Brier={brier:.3f})",
            )

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed probability")
    ax.set_title(f"Calibration: {target_name}")
    ax.legend()
    fig.tight_layout()
    _save_figure_multiformat(fig, output_path.with_suffix(""))
    plt.close(fig)


def plot_latent_space(
    model: "SUAVE",
    feature_map: Mapping[str, pd.DataFrame],
    label_map: Mapping[str, Sequence[object]],
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
    _save_figure_multiformat(fig, output_path.with_suffix(""))
    plt.close(fig)


def plot_benchmark_curves(
    dataset_name: str,
    y_true: np.ndarray,
    model_probability_lookup: Mapping[str, np.ndarray],
    *,
    output_dir: Path,
    target_label: str,
    abbreviation_lookup: Optional[Mapping[str, str]] = None,
    n_bins: int = 10,
) -> Optional[Mapping[str, Path]]:
    """Plot benchmark ROC and calibration curves for the supplied dataset."""

    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        print(f"Skipping {dataset_name} curves because only one class is present.")
        return None

    roc_fig, roc_ax = plt.subplots(figsize=(6, 6))
    roc_ax.set_aspect("equal", adjustable="box")
    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    roc_ax.set_title(f"ROC – {dataset_name}")
    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate")

    calibration_records: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for model_name, probs in model_probability_lookup.items():
        abbrev = (
            model_name
            if abbreviation_lookup is None
            else abbreviation_lookup.get(model_name, model_name)
        )
        positive_probs = extract_positive_probabilities(probs)
        fpr, tpr, _ = roc_curve(y_true, positive_probs)
        roc_ax.plot(fpr, tpr, label=abbrev)

        try:
            frac_pos, mean_pred = calibration_curve(
                y_true, positive_probs, n_bins=n_bins, strategy="quantile"
            )
        except ValueError:
            print(
                f"Calibration curve for {model_name} on {dataset_name} skipped due to insufficient variation."
            )
        else:
            calibration_records.append((abbrev, mean_pred, frac_pos))

    roc_ax.legend(loc="lower right")
    roc_fig.tight_layout()

    dataset_slug = dataset_name.lower().replace(" ", "_")
    roc_path = output_dir / f"benchmark_roc_{dataset_slug}_{target_label}.png"
    _save_figure_multiformat(roc_fig, roc_path.with_suffix(""), use_tight_layout=True)
    plt.close(roc_fig)
    print(f"Saved benchmark ROC curves for {dataset_name} to {roc_path}")

    calibration_paths: Optional[Path] = None
    if calibration_records:
        cal_fig, cal_ax = plt.subplots(figsize=(6, 6))
        cal_ax.set_aspect("equal", adjustable="box")

        x_values = np.concatenate([record[1] for record in calibration_records])
        y_values = np.concatenate([record[2] for record in calibration_records])
        combined = np.concatenate([x_values, y_values])
        axis_min = float(np.nanmin(combined))
        axis_max = float(np.nanmax(combined))
        if axis_min == axis_max:
            axis_min -= 0.05
            axis_max += 0.05
        padding = max((axis_max - axis_min) * 0.05, 1e-3)
        axis_min -= padding
        axis_max += padding

        reference = np.linspace(axis_min, axis_max, 2)
        cal_ax.plot(reference, reference, linestyle="--", color="tab:gray", label="Perfect calibration")

        for abbrev, mean_pred, frac_pos in calibration_records:
            cal_ax.plot(mean_pred, frac_pos, marker="o", label=abbrev)

        cal_ax.set_xlim(axis_min, axis_max)
        cal_ax.set_ylim(axis_min, axis_max)
        cal_ax.set_title(f"Calibration – {dataset_name}")
        cal_ax.set_xlabel("Mean predicted probability")
        cal_ax.set_ylabel("Observed probability")
        cal_ax.legend(loc="best")
        cal_fig.tight_layout()

        calibration_paths = (
            output_dir / f"benchmark_calibration_{dataset_slug}_{target_label}.png"
        )
        _save_figure_multiformat(
            cal_fig, calibration_paths.with_suffix(""), use_tight_layout=True
        )
        plt.close(cal_fig)
        print(
            f"Saved benchmark calibration curves for {dataset_name} to {calibration_paths}"
        )

    result: Dict[str, Path] = {"roc": roc_path}
    if calibration_paths is not None:
        result["calibration"] = calibration_paths
    return result


def plot_transfer_metric_boxes(
    bootstrap_df: pd.DataFrame,
    *,
    metric: str,
    evaluation_dataset: str,
    training_order: Sequence[str],
    model_order: Sequence[str],
    output_dir: Path,
    target_label: str,
    color_map: Optional[Mapping[str, str]] = None,
    metric_labels: Optional[Mapping[str, str]] = None,
    y_padding: float = 0.02,
    minor_tick: float = 0.05,
    major_tick: float = 0.1,
) -> Optional[Path]:
    """Plot box plots of transfer metrics grouped by model and training dataset."""

    subset = bootstrap_df[
        (bootstrap_df["metric"] == metric)
        & (bootstrap_df["evaluation_dataset"] == evaluation_dataset)
    ]
    if subset.empty:
        print(
            f"Skipping {metric} box plot for {evaluation_dataset} because no data was provided."
        )
        return None

    metric_labels = metric_labels or {}
    available_training = subset["training_dataset"].unique()
    training_order = [name for name in training_order if name in available_training]
    available_models = subset["model"].unique()
    model_order = [name for name in model_order if name in available_models]
    if not training_order or not model_order:
        print(
            f"Skipping {metric} box plot for {evaluation_dataset} due to missing model or dataset coverage."
        )
        return None

    model_count = len(model_order)
    dataset_count = len(training_order)
    if color_map is None:
        color_map = build_training_color_map(training_order)

    fig_width = max(6.0, model_count * 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))
    legend_handles: Dict[str, Patch] = {}
    collected_values: List[float] = []

    if model_count == 1:
        model_name = model_order[0]
        model_subset = subset[subset["model"] == model_name]
        positions: List[float] = []
        box_data: List[np.ndarray] = []
        colors: List[str] = []
        labels: List[str] = []
        for idx, dataset_name in enumerate(training_order):
            dataset_values = model_subset[
                model_subset["training_dataset"] == dataset_name
            ]["value"].dropna()
            if dataset_values.empty:
                continue
            positions.append(float(idx))
            values = dataset_values.to_numpy()
            box_data.append(values)
            collected_values.extend(values.tolist())
            color = color_map.get(dataset_name, "#1f77b4")
            colors.append(color)
            labels.append(dataset_name)
            if dataset_name not in legend_handles:
                legend_handles[dataset_name] = Patch(
                    facecolor=color,
                    alpha=0.6,
                    label=dataset_name,
                )
        if not box_data:
            plt.close(fig)
            print(
                f"Skipping {metric} box plot for {evaluation_dataset} because no bootstrap samples were found."
            )
            return None
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            boxprops={"linewidth": 1.0, "edgecolor": "black"},
            whiskerprops={"linewidth": 1.0, "color": "black"},
            capprops={"linewidth": 1.0, "color": "black"},
            medianprops={"linewidth": 1.0, "color": "black"},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_xlabel("Training dataset")
    else:
        box_positions: List[float] = []
        box_data: List[np.ndarray] = []
        box_colors: List[str] = []
        width = 0.8 / max(dataset_count, 1)
        for model_idx, model_name in enumerate(model_order):
            model_subset = subset[subset["model"] == model_name]
            if model_subset.empty:
                continue
            for dataset_idx, dataset_name in enumerate(training_order):
                dataset_values = model_subset[
                    model_subset["training_dataset"] == dataset_name
                ]["value"].dropna()
                if dataset_values.empty:
                    continue
                offset = (dataset_idx - (dataset_count - 1) / 2) * width
                position = float(model_idx) + offset
                box_positions.append(position)
                values = dataset_values.to_numpy()
                box_data.append(values)
                collected_values.extend(values.tolist())
                color = color_map.get(dataset_name, "#1f77b4")
                box_colors.append(color)
                if dataset_name not in legend_handles:
                    legend_handles[dataset_name] = Patch(
                        facecolor=color,
                        alpha=0.6,
                        label=dataset_name,
                    )
        if not box_data:
            plt.close(fig)
            print(
                f"Skipping {metric} box plot for {evaluation_dataset} because no bootstrap samples were found."
            )
            return None
        bp = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            boxprops={"linewidth": 1.0, "edgecolor": "black"},
            whiskerprops={"linewidth": 1.0, "color": "black"},
            capprops={"linewidth": 1.0, "color": "black"},
            medianprops={"linewidth": 1.0, "color": "black"},
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticks(np.arange(len(model_order)))
        ax.set_xticklabels(model_order, rotation=0, ha="center")
        ax.set_xlabel("Model")

    display_label = metric_labels.get(metric, metric.replace("_", " ").upper())
    ax.set_ylabel(display_label)
    ax.set_title(f"{display_label} – {evaluation_dataset}")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if collected_values:
        finite = np.asarray([value for value in collected_values if np.isfinite(value)])
        if finite.size:
            data_min = float(finite.min())
            data_max = float(finite.max())
        else:
            data_min = data_max = 0.0
        major_min = math.floor(data_min / major_tick) * major_tick
        major_max = math.ceil(data_max / major_tick) * major_tick
        if major_min == major_max:
            major_max = major_min + major_tick
        pad = max(y_padding, abs(major_max - major_min) * 0.05)
        y_min = major_min - pad
        y_max = major_max + pad
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(MultipleLocator(major_tick))
        ax.yaxis.set_minor_locator(MultipleLocator(minor_tick))

    if legend_handles:
        ax.legend(
            handles=list(legend_handles.values()),
            title="Training dataset",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0, 0.85, 1))

    dataset_slug = slugify_identifier(evaluation_dataset)
    figure_path = (
        output_dir
        / f"tstr_trtr_{dataset_slug}_{metric.lower()}_{slugify_identifier(target_label)}.png"
    )
    _save_figure_multiformat(fig, figure_path.with_suffix(""), use_tight_layout=True)
    plt.close(fig)
    print(
        f"Saved {display_label} box plot for {evaluation_dataset} to {figure_path}"
    )
    return figure_path


def plot_transfer_metric_bars(
    summary_df: pd.DataFrame,
    *,
    metric: str,
    evaluation_dataset: str,
    training_order: Sequence[str],
    model_order: Sequence[str],
    output_dir: Path,
    target_label: str,
    color_map: Optional[Mapping[str, str]] = None,
    metric_labels: Optional[Mapping[str, str]] = None,
    y_limit: Tuple[float, float] = (0.5, 1.0),
) -> Optional[Path]:
    """Plot grouped bar charts for transfer metrics without error bars."""

    if metric not in summary_df.columns:
        return None

    subset = summary_df[
        (summary_df["evaluation_dataset"] == evaluation_dataset)
        & summary_df[metric].notna()
    ]
    if subset.empty:
        return None

    metric_labels = metric_labels or {}
    available_training = subset["training_dataset"].unique()
    training_order = [name for name in training_order if name in available_training]
    available_models = subset["model"].unique()
    model_order = [name for name in model_order if name in available_models]
    if not training_order or not model_order:
        return None

    model_count = len(model_order)
    dataset_count = len(training_order)
    if color_map is None:
        color_map = build_training_color_map(training_order)

    fig_width = max(6.0, model_count * 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))
    legend_handles: Dict[str, Patch] = {}

    display_label = metric_labels.get(metric, metric.replace("_", " ").upper())

    if model_count == 1:
        model_name = model_order[0]
        model_subset = subset[subset["model"] == model_name]
        heights: List[float] = []
        positions: List[float] = []
        labels: List[str] = []
        colors: List[str] = []
        for idx, dataset_name in enumerate(training_order):
            dataset_subset = model_subset[
                model_subset["training_dataset"] == dataset_name
            ]
            if dataset_subset.empty:
                continue
            height = float(dataset_subset.iloc[0][metric])
            heights.append(height)
            positions.append(float(idx))
            labels.append(dataset_name)
            color = color_map.get(dataset_name, "#1f77b4")
            colors.append(color)
            if dataset_name not in legend_handles:
                legend_handles[dataset_name] = Patch(
                    facecolor=color,
                    alpha=0.8,
                    label=dataset_name,
                )
        if not heights:
            plt.close(fig)
            return None
        ax.bar(positions, heights, color=colors, width=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_xlabel("Training dataset")
    else:
        base_positions = np.arange(model_count)
        width = 0.8 / max(dataset_count, 1)
        for dataset_idx, dataset_name in enumerate(training_order):
            offsets = (dataset_idx - (dataset_count - 1) / 2) * width
            heights: List[float] = []
            positions: List[float] = []
            for model_name, base_pos in zip(model_order, base_positions):
                selection = subset[
                    (subset["model"] == model_name)
                    & (subset["training_dataset"] == dataset_name)
                ]
                if selection.empty:
                    heights.append(np.nan)
                    positions.append(base_pos + offsets)
                    continue
                heights.append(float(selection.iloc[0][metric]))
                positions.append(base_pos + offsets)
            color = color_map.get(dataset_name, "#1f77b4")
            ax.bar(positions, heights, width=width, color=color, alpha=0.8, label=dataset_name)
            if dataset_name not in legend_handles:
                legend_handles[dataset_name] = Patch(
                    facecolor=color,
                    alpha=0.8,
                    label=dataset_name,
                )
        ax.set_xticks(base_positions)
        ax.set_xticklabels(model_order, rotation=0, ha="center")
        ax.set_xlabel("Model")

    ax.set_ylabel(display_label)
    ax.set_title(f"{display_label} – {evaluation_dataset}")
    ax.set_ylim(*y_limit)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if legend_handles:
        ax.legend(
            handles=list(legend_handles.values()),
            title="Training dataset",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0, 0.85, 1))

    dataset_slug = slugify_identifier(evaluation_dataset)
    figure_path = (
        output_dir
        / f"tstr_trtr_{dataset_slug}_{metric.lower()}_bars_{slugify_identifier(target_label)}.png"
    )
    _save_figure_multiformat(fig, figure_path.with_suffix(""), use_tight_layout=True)
    plt.close(fig)
    print(
        f"Saved {display_label} bar chart for {evaluation_dataset} to {figure_path}"
    )
    return figure_path


def build_prediction_dataframe(
    probabilities: np.ndarray,
    labels: Sequence[object],
    predictions: Sequence[object],
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Assemble a dataframe compatible with :func:`evaluate_predictions`."""

    prob_matrix = np.asarray(probabilities)
    class_names = list(class_names)
    if prob_matrix.ndim == 1:
        if len(class_names) == 2:
            negative_name, positive_name = class_names[0], class_names[-1]
            proba_dict = {
                f"pred_proba_{negative_name}": 1.0 - prob_matrix,
                f"pred_proba_{positive_name}": prob_matrix,
            }
        else:
            proba_dict = {"pred_proba_0": prob_matrix}
    else:
        if prob_matrix.shape[1] == len(class_names) and len(class_names) > 0:
            proba_dict = {
                f"pred_proba_{class_names[idx]}": prob_matrix[:, idx]
                for idx in range(prob_matrix.shape[1])
            }
        else:
            proba_dict = {
                f"pred_proba_{idx}": prob_matrix[:, idx]
                for idx in range(prob_matrix.shape[1])
            }

    label_array = np.asarray(labels)
    prediction_array = np.asarray(predictions)
    if prediction_array.dtype != label_array.dtype:
        if {
            prediction_array.dtype.kind,
            label_array.dtype.kind,
        } <= {"U", "S"}:  # Prefer the wider string dtype when both are fixed-width
            # ``itemsize`` reflects the byte-width of each string slot, so use the
            # larger allocation to avoid truncation (e.g. "Deceased" -> "Decea").
            target_dtype = (
                label_array.dtype
                if label_array.dtype.itemsize >= prediction_array.dtype.itemsize
                else prediction_array.dtype
            )
            label_array = label_array.astype(target_dtype, copy=False)
            prediction_array = prediction_array.astype(target_dtype, copy=False)
        else:
            try:
                label_array = label_array.astype(prediction_array.dtype, copy=False)
            except (TypeError, ValueError):
                try:
                    prediction_array = prediction_array.astype(label_array.dtype, copy=False)
                except (TypeError, ValueError):
                    label_array = label_array.astype(object)
                    prediction_array = prediction_array.astype(object)
    base_df = pd.DataFrame(
        {
            "label": label_array,
            "y_pred": prediction_array,
        }
    )
    if proba_dict:
        proba_df = pd.DataFrame(proba_dict)
        base_df = pd.concat([base_df.reset_index(drop=True), proba_df], axis=1)
    else:
        base_df = base_df.reset_index(drop=True)
    return base_df


def resolve_suave_fit_kwargs(params: Mapping[str, object]) -> Dict[str, object]:
    """Map Optuna trial parameters to :class:`SUAVE.fit` keyword arguments."""

    return {
        "warmup_epochs": int(params.get("warmup_epochs", 3)),
        "kl_warmup_epochs": int(params.get("kl_warmup_epochs", 0)),
        "head_epochs": int(params.get("head_epochs", 2)),
        "finetune_epochs": int(params.get("finetune_epochs", 2)),
        "joint_decoder_lr_scale": float(params.get("joint_decoder_lr_scale", 0.1)),
        "early_stop_patience": int(params.get("early_stop_patience", 10)),
    }


def resolve_classification_loss_weight(params: Mapping[str, object]) -> Optional[float]:
    """Normalise ``classification_loss_weight`` from Optuna parameters."""

    use_weight = params.get("use_classification_loss_weight")
    if isinstance(use_weight, str):
        use_weight = use_weight.lower() in {"1", "true", "yes"}
    elif isinstance(use_weight, (np.bool_,)):
        use_weight = bool(use_weight)
    if not use_weight:
        return None
    weight = params.get("classification_loss_weight")
    if weight is None:
        return 1.0
    if isinstance(weight, (np.floating, np.integer)):
        return float(weight)
    return float(weight)


def build_suave_model(
    params: Mapping[str, object],
    schema: Schema,
    *,
    random_state: int,
) -> SUAVE:
    """Instantiate :class:`SUAVE` using Optuna-style parameters."""

    hidden_key = str(params.get("hidden_dims", "medium"))
    head_hidden_key = str(params.get("head_hidden_dims", "medium"))
    hidden_dims = HIDDEN_DIMENSION_OPTIONS.get(
        hidden_key, HIDDEN_DIMENSION_OPTIONS["medium"]
    )
    head_hidden_dims = HEAD_HIDDEN_DIMENSION_OPTIONS.get(
        head_hidden_key, HEAD_HIDDEN_DIMENSION_OPTIONS["medium"]
    )
    classification_loss_weight = resolve_classification_loss_weight(params)
    return SUAVE(
        schema=schema,
        latent_dim=int(params.get("latent_dim", 16)),
        n_components=int(params.get("n_components", 1)),
        hidden_dims=hidden_dims,
        head_hidden_dims=head_hidden_dims,
        dropout=float(params.get("dropout", 0.1)),
        learning_rate=float(params.get("learning_rate", 1e-3)),
        batch_size=int(params.get("batch_size", 256)),
        beta=float(params.get("beta", 1.5)),
        classification_loss_weight=classification_loss_weight,
        decoder_refine_mode=params.get("decoder_refine_mode", 'decoder_only'),
        decoder_refine_epochs=params.get("decoder_refine_epochs", None),
        random_state=random_state,
        behaviour="supervised",
    )
