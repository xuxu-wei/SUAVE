"""Hard-coded configuration block for the research workflow template.

This module centralises every project-specific constant that is expected to
change when porting the research workflow to a new dataset. Users should edit
these values instead of modifying :mod:`analysis_utils` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


TEMPLATE_ROOT = Path(__file__).resolve().parent


# =============================================================================
# === Dataset, modelling and evaluation defaults ==============================
# =============================================================================


RANDOM_STATE: int = 20201021
TARGET_COLUMNS: Tuple[str, ...] = ("in_hospital_mortality", "28d_mortality")
BENCHMARK_COLUMNS: Tuple[str, ...] = (
    "APS_III",
    "APACHE_IV",
    "SAPS_II",
    "OASIS",
)

CLINICAL_SCORE_BENCHMARK_STRATEGY: str = "imputed"

VALIDATION_SIZE: float = 0.2

DATA_DIR: Path = (TEMPLATE_ROOT / "datasets").resolve()

# Thresholds governing which Optuna trials are considered viable for
# persistence.
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
    "optuna_trials": 5,
    "optuna_timeout": 3600 * 48,
    "optuna_study_prefix": "supervised",
    "optuna_storage": None,
    "output_dir_name": "research_outputs_supervised",
}

#: Default environment flags that determine whether cached artefacts should be
#: regenerated. ``FORCE_UPDATE_SUAVE`` is only consulted when Optuna artefacts
#: are unavailable, allowing callers to refresh the locally persisted SUAVE
#: model that otherwise acts as a fallback.
FORCE_UPDATE_FLAG_DEFAULTS: Dict[str, bool] = {
    "FORCE_UPDATE_BENCHMARK_MODEL": False,
    "FORCE_UPDATE_TSTR_MODEL": True,
    "FORCE_UPDATE_TRTR_MODEL": True,
    "FORCE_UPDATE_SUAVE": False,
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
    "tstr_trtr": "09_tstr_trtr_transfer",
    "distribution_shift": "10_distribution_shift",
    "privacy_assessment": "11_privacy_assessment",
    "visualisations": "12_visualizations",
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


__all__ = [
    "ANALYSIS_SUBDIRECTORIES",
    "BENCHMARK_COLUMNS",
    "CLINICAL_SCORE_BENCHMARK_STRATEGY",
    "DATA_DIR",
    "DEFAULT_ANALYSIS_CONFIG",
    "FORCE_UPDATE_FLAG_DEFAULTS",
    "HEAD_HIDDEN_DIMENSION_OPTIONS",
    "HIDDEN_DIMENSION_OPTIONS",
    "PATH_GRAPH_GROUP_COLORS",
    "PATH_GRAPH_NODE_COLORS",
    "PATH_GRAPH_NODE_DEFINITIONS",
    "PATH_GRAPH_NODE_GROUPS",
    "PATH_GRAPH_NODE_LABELS",
    "PARETO_MAX_ABS_DELTA_AUC",
    "PARETO_MIN_VALIDATION_ROAUC",
    "RANDOM_STATE",
    "TARGET_COLUMNS",
    "VALIDATION_SIZE",
    "VAR_GROUP_DICT",
]

