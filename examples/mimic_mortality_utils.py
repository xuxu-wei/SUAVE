"""Shared utilities for the MIMIC mortality modelling examples."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
import json
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
)

import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
from tabulate import tabulate

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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
    kolmogorov_smirnov_statistic,
    mutual_information_feature,
    rbf_mmd,
)
from cls_eval import evaluate_predictions  # noqa: E402


RANDOM_STATE: int = 20201021
TARGET_COLUMNS: Tuple[str, str] = ("in_hospital_mortality", "28d_mortality")
BENCHMARK_COLUMNS = (
    "APS_III",
    "APACHE_IV",
    "SAPS_II",
    "OASIS",
)  # do not include in training. Only use for benchamrk validation.

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

DEFAULT_ANALYSIS_CONFIG: Dict[str, object] = {
    "optuna_trials": 60,
    "optuna_timeout": 3600 * 48,
    "optuna_study_prefix": "supervised",
    "optuna_storage": None,
    "output_dir_name": "research_outputs_supervised",
}

ANALYSIS_SUBDIRECTORIES: Dict[str, str] = {
    "schema": "01_schema_validation",
    "features": "02_feature_engineering",
    "optuna": "03_optuna_search",
    "model": "04_suave_model",
    "evaluation": "05_evaluation_metrics",
    "bootstrap": "06_bootstrap_analysis",
    "baseline": "07_baseline_models",
    "transfer": "08_transfer_learning",
    "distribution": "09_distribution_shift",
    "privacy": "10_privacy_assessment",
    "visualisation": "11_visualizations",
}


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
    "Hematology": "#2ca02c",
    "Hepatic Function": "#bcbd22",
    "Renal Function": "#17becf",
    "Metabolic & Electrolytes": "#ff7f0e",
    "Coagulation": "#8c564b",
    "Arterial Blood Gas": "#7f7f7f",
    "Outcome": "#e377c2",
    "Latent": "#4c72b0",
}


PATH_GRAPH_NODE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "age": {"group": "Demographics & Vitals", "label": "Age (years)"},
    "sex": {"group": "Demographics & Vitals", "label": "Male sex (indicator)"},
    "BMI": {
        "group": "Demographics & Vitals",
        "label": r"Body mass index (kg/m$^2$)",
    },
    "temperature": {"group": "Demographics & Vitals", "label": "Temperature (°C)"},
    "heart_rate": {
        "group": "Demographics & Vitals",
        "label": "Heart rate (beats/min)",
    },
    "respir_rate": {
        "group": "Demographics & Vitals",
        "label": "Respiratory rate (breaths/min)",
    },
    "SBP": {"group": "Hemodynamics & Perfusion", "label": "SBP"},
    "DBP": {"group": "Hemodynamics & Perfusion", "label": "DBP"},
    "MAP": {"group": "Hemodynamics & Perfusion", "label": "MAP"},
    "Lac": {
        "group": "Hemodynamics & Perfusion",
        "label": "Serum lactate (mmol/L)",
    },
    "SOFA_cns": {"group": "Organ Support & Neurology", "label": "SOFA CNS"},
    "CRRT": {"group": "Organ Support & Neurology", "label": "CRRT"},
    "Respiratory_Support": {
        "group": "Organ Support & Neurology",
        "label": "Respiratory support level",
    },
    "WBC": {"group": "Hematology", "label": "WBC"},
    "Hb": {"group": "Hematology", "label": "Hb"},
    "NE%": {"group": "Hematology", "label": "NE%"},
    "LYM%": {"group": "Hematology", "label": "LYM%"},
    "PLT": {"group": "Hematology", "label": "PLT"},
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
    "APTT": {"group": "Coagulation", "label": "aPTT"},
    "PH": {"group": "Arterial Blood Gas", "label": "Arterial pH"},
    "PaO2": {"group": "Arterial Blood Gas", "label": r"$\mathrm{PaO}_{2}$ (mmHg)"},
    "PaO2/FiO2": {
        "group": "Arterial Blood Gas",
        "label": r"$\mathrm{PaO}_{2}/\mathrm{FiO}_{2}$ ratio",
    },
    "PaCO2": {"group": "Arterial Blood Gas", "label": r"$\mathrm{PaCO}_{2}$ (mmHg)"},
    "in_hospital_mortality": {
        "group": "Outcome",
        "label": "In-hospital mortality",
    },
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
    return directories


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
    "VALIDATION_SIZE",
    "DATA_DIR",
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
    "manifest_artifact_paths",
    "manifest_artifacts_exist",
    "record_model_manifest",
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
    "make_logistic_pipeline",
    "make_random_forest_pipeline",
    "make_xgboost_pipeline",
    "make_baseline_model_factories",
    "mutual_information_feature",
    "plot_benchmark_curves",
    "plot_calibration_curves",
    "plot_latent_space",
    "plot_transfer_metric_bars",
    "prepare_features",
    "render_dataframe",
    "rbf_mmd",
    "make_study_name",
    "DEFAULT_ANALYSIS_CONFIG",
    "ANALYSIS_SUBDIRECTORIES",
    "build_analysis_config",
    "prepare_analysis_output_directories",
    "resolve_analysis_output_root",
    "parse_script_arguments",
    "load_optuna_study",
    "summarise_pareto_trials",
    "schema_markdown_table",
    "schema_to_dataframe",
    "slugify_identifier",
    "to_numeric_frame",
    "build_tstr_training_sets",
    "evaluate_transfer_baselines",
    "build_suave_model",
    "resolve_suave_fit_kwargs",
    "resolve_classification_loss_weight",
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
        display(df)
        return
    tabulate_kwargs = {"headers": "keys", "tablefmt": "github", "showindex": False}
    if floatfmt is not None:
        tabulate_kwargs["floatfmt"] = floatfmt
    print(tabulate(df, **tabulate_kwargs))


# =============================================================================
# === Model manifest and Optuna metadata helpers ==============================
# =============================================================================


def _normalise_manifest_path(path: Path, base_dir: Path) -> str:
    """Return ``path`` as a relative string when nested under ``base_dir``."""

    try:
        return str(path.relative_to(base_dir))
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
    trial_number: Optional[int],
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
        Optuna trial identifier used to train the artefacts.
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
        "saved_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
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


def parse_script_arguments(argv: Sequence[str]) -> Optional[int]:
    """Parse ``argv`` and return the requested Optuna trial identifier.

    Parameters
    ----------
    argv
        Command-line arguments excluding the executable name.

    Returns
    -------
    Optional[int]
        The requested Optuna trial identifier, if provided.

    Examples
    --------
    >>> parse_script_arguments(["--trial-id", "12"])
    12
    >>> parse_script_arguments([]) is None
    True
    """

    parser = argparse.ArgumentParser(
        description="Select an Optuna trial for SUAVE model loading/training.",
    )
    parser.add_argument(
        "trial_id",
        nargs="?",
        type=int,
        help="Optuna trial identifier to load or train.",
    )
    parser.add_argument(
        "--trial-id",
        dest="trial_id_flag",
        type=int,
        help="Optuna trial identifier to load or train.",
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


def summarise_pareto_trials(
    trials: Sequence["optuna.trial.FrozenTrial"],
    *,
    manifest: Mapping[str, Any],
    model_dir: Path,
) -> pd.DataFrame:
    """Return a tidy summary of Pareto-optimal Optuna trials."""

    if not trials:
        return pd.DataFrame()

    saved_trial_number = None
    if manifest and manifest_artifacts_exist(manifest, model_dir):
        saved_trial_number = manifest.get("trial_number")

    rows: List[Dict[str, object]] = []
    for trial in trials:
        values = trial.values or (float("nan"), float("nan"))
        validation_roauc = float(values[0])
        delta_auc = float(values[1]) if len(values) > 1 else float("nan")
        saved_locally = bool(saved_trial_number == trial.number)
        rows.append(
            {
                "Saved locally": "✅" if saved_locally else "❌",
                "Trial ID": trial.number,
                "Validation ROAUC": validation_roauc,
                "TSTR/TRTR ΔAUC": delta_auc,
            }
        )
    return pd.DataFrame(rows)


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

    imputer = IterativeImputer(max_iter=50, tol=1e-2)
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


def make_logistic_pipeline(random_state: Optional[int] = None) -> Pipeline:
    """Factory for the baseline classifier used in TSTR/TRTR evaluations."""

    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("imputer", IterativeImputer(max_iter=50, tol=1e-2)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=200, random_state=random_state),
            ),
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

    return Pipeline(
        [
            ("imputer", IterativeImputer(max_iter=50, tol=1e-2)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_xgboost_pipeline(random_state: Optional[int] = None) -> Pipeline:
    """Return an XGBoost pipeline with iterative imputation."""

    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    try:
        from xgboost import XGBClassifier
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "xgboost is required for the mortality TSTR evaluation."
        ) from error

    return Pipeline(
        [
            ("imputer", IterativeImputer(max_iter=50, tol=1e-2)),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="auc",
                    reg_lambda=1.0,
                    random_state=random_state,
                    tree_method="hist",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_baseline_model_factories(
    random_state: int,
) -> Dict[str, Callable[[], Pipeline]]:
    """Return model factories for the supervised transfer comparison."""

    return {
        "Logistic regression": lambda: make_logistic_pipeline(random_state),
        "Random forest": lambda: make_random_forest_pipeline(random_state),
        "XGBoost": lambda: make_xgboost_pipeline(random_state),
    }


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
    datasets["TSTR synthesis"] = (
        to_numeric_frame(synthesis_features.copy()).reset_index(drop=True),
        synthesis_labels.copy(),
    )
    raw_datasets["TSTR synthesis"] = (
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
    datasets["TSTR synthesis-balance"] = (
        to_numeric_frame(balance_features.copy()).reset_index(drop=True),
        balance_labels.copy(),
    )
    raw_datasets["TSTR synthesis-balance"] = (
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
    datasets["TSTR synthesis-augment"] = (
        to_numeric_frame(raw_augmented).reset_index(drop=True),
        augmented_labels_series,
    )
    raw_datasets["TSTR synthesis-augment"] = (
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
    datasets["TSTR synthesis-5x"] = (
        to_numeric_frame(five_x_features.copy()).reset_index(drop=True),
        five_x_series.copy(),
    )
    raw_datasets["TSTR synthesis-5x"] = (
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
    datasets["TSTR synthesis-5x balance"] = (
        to_numeric_frame(five_x_balance_features.copy()).reset_index(drop=True),
        five_x_balance_labels.copy(),
    )
    raw_datasets["TSTR synthesis-5x balance"] = (
        five_x_balance_features,
        five_x_balance_labels,
    )

    if return_raw:
        return datasets, raw_datasets
    return datasets


def evaluate_transfer_baselines(
    training_sets: Mapping[str, Tuple[pd.DataFrame, pd.Series]],
    evaluation_sets: Mapping[str, Tuple[pd.DataFrame, pd.Series]],
    *,
    model_factories: Mapping[str, Callable[[], Pipeline]],
    bootstrap_n: int,
    random_state: int,
    raw_training_sets: Optional[Mapping[str, Tuple[pd.DataFrame, pd.Series]]] = None,
    raw_evaluation_sets: Optional[Mapping[str, Tuple[pd.DataFrame, pd.Series]]] = None,
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
    """

    summary_rows: List[Dict[str, object]] = []
    long_rows: List[Dict[str, object]] = []
    nested_results: Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]] = {}

    for training_name, (train_X_numeric, train_y_numeric) in training_sets.items():
        nested_results.setdefault(training_name, {})
        raw_training = (
            raw_training_sets.get(training_name)
            if raw_training_sets is not None
            else None
        )
        for model_name, factory in model_factories.items():
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

            for evaluation_name, (
                eval_X_numeric,
                eval_y_numeric,
            ) in evaluation_sets.items():
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

                results = evaluate_predictions(
                    prediction_df,
                    label_col="label",
                    pred_col="y_pred",
                    positive_label=positive_label,
                    bootstrap_n=bootstrap_n,
                    random_state=random_state,
                )
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


def fit_isotonic_calibrator(
    model: SUAVE,
    features: pd.DataFrame,
    targets: pd.Series | np.ndarray,
) -> CalibratedClassifierCV:
    """Wrap ``model`` with an isotonic :class:`CalibratedClassifierCV`."""

    calibrator = CalibratedClassifierCV(
        base_estimator=model,
        method="isotonic",
        cv="prefit",
    )
    calibrator.fit(features, np.asarray(targets))
    return calibrator


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
        pos_probs = extract_positive_probabilities(probs)
        try:
            frac_pos, mean_pred = calibration_curve(labels, pos_probs, n_bins=n_bins)
        except ValueError:
            continue
        brier = brier_score_loss(labels, pos_probs)
        ax.plot(
            mean_pred,
            frac_pos,
            marker="o",
            label=f"{dataset_name} (Brier={brier:.3f})",
        )

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
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
) -> Optional[Path]:
    """Plot ROC and calibration curves for the supplied dataset."""

    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        print(f"Skipping {dataset_name} curves because only one class is present.")
        return None

    fig, (roc_ax, cal_ax) = plt.subplots(1, 2, figsize=(12, 5))

    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    roc_ax.set_title(f"ROC – {dataset_name}")
    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate")

    cal_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    cal_ax.set_title(f"Calibration – {dataset_name}")
    cal_ax.set_xlabel("Mean predicted probability")
    cal_ax.set_ylabel("Fraction of positives")

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
            cal_ax.plot(mean_pred, frac_pos, marker="o", label=abbrev)

    roc_ax.legend(loc="lower right")
    cal_ax.legend(loc="upper left")
    fig.suptitle(f"Benchmark ROC & calibration – {dataset_name}")
    fig.tight_layout()

    dataset_slug = dataset_name.lower().replace(" ", "_")
    figure_path = output_dir / f"benchmark_curves_{dataset_slug}_{target_label}.png"
    _save_figure_multiformat(fig, figure_path.with_suffix(""), use_tight_layout=True)
    plt.close(fig)
    print(f"Saved benchmark curves for {dataset_name} to {figure_path}")
    return figure_path


def plot_transfer_metric_bars(
    metric_df: pd.DataFrame,
    *,
    metric: str,
    evaluation_dataset: str,
    training_order: Sequence[str],
    model_order: Sequence[str],
    output_dir: Path,
    target_label: str,
) -> Optional[Path]:
    """Plot grouped bar charts with error bars for TSTR/TRTR comparisons."""

    subset = metric_df[
        (metric_df["metric"] == metric)
        & (metric_df["evaluation_dataset"] == evaluation_dataset)
    ]
    if subset.empty:
        print(
            f"Skipping {metric} bars for {evaluation_dataset} because no data was provided."
        )
        return None

    training_order = list(training_order)
    model_order = list(model_order)
    x_positions = np.arange(len(training_order), dtype=float)
    width = 0.8 / max(len(model_order), 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, model_name in enumerate(model_order):
        model_subset = (
            subset[subset["model"] == model_name]
            .set_index("training_dataset")
            .reindex(training_order)
        )
        estimates = model_subset["estimate"].to_numpy()
        lower = estimates - model_subset["ci_low"].to_numpy()
        upper = model_subset["ci_high"].to_numpy() - estimates
        lower = np.nan_to_num(lower, nan=0.0, posinf=0.0, neginf=0.0)
        upper = np.nan_to_num(upper, nan=0.0, posinf=0.0, neginf=0.0)
        offsets = (idx - (len(model_order) - 1) / 2) * width
        ax.bar(
            x_positions + offsets,
            estimates,
            width=width,
            label=model_name,
            yerr=np.vstack([lower, upper]),
            capsize=4,
            alpha=0.9,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(training_order, rotation=20, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{metric.upper()} – {evaluation_dataset}")
    ax.legend()
    fig.tight_layout()

    dataset_slug = slugify_identifier(evaluation_dataset)
    figure_path = (
        output_dir
        / f"tstr_trtr_{dataset_slug}_{metric.lower()}_{slugify_identifier(target_label)}.png"
    )
    _save_figure_multiformat(fig, figure_path.with_suffix(""), use_tight_layout=True)
    plt.close(fig)
    print(f"Saved {metric.upper()} bars for {evaluation_dataset} to {figure_path}")
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
        random_state=random_state,
        behaviour="supervised",
    )
