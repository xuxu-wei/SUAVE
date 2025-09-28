"""Hard-coded configuration block for the research workflow template.

This module centralises every project-specific constant that is expected to
change when porting the research workflow to a new dataset. Users should edit
these values instead of modifying :mod:`analysis_utils` directly.

Alignment checklist
-------------------
* 在运行分析之前，请先确认不同数据集（训练、测试、外部验证等）中的变量集合已经对齐；只有
  :data:`BENCHMARK_COLUMNS` 中登记的临床评分允许在不同数据集中缺失。
* 模型特征是通过排除 :data:`BENCHMARK_COLUMNS` 和 :data:`TARGET_COLUMNS` 得到的，因此数据文件
  中不应再包含其它非对齐变量；否则这些额外列会被视为模型特征。
* :data:`TARGET_COLUMNS` 仅用于在构建特征时排除目标变量，真正用于训练与评估的标签由
  :data:`TARGET_LABEL` 指定。

Emoji legend
------------
🟢 Update for every new project.
🟡 Adjust when the default does not match your setup.
🔴 Advanced setting – change only if you understand the downstream impact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 🔴 Template root path inferred automatically so other defaults can reuse it.
TEMPLATE_ROOT = Path(__file__).resolve().parent


# =============================================================================
# === Dataset, modelling and evaluation defaults ==============================
# =============================================================================

# 🟢 Directory containing raw tabular datasets used by the workflow.
DATA_DIR: Path = (TEMPLATE_ROOT / "datasets").resolve()

# 🟢 Primary target label for supervised analysis. Must appear in ``TARGET_COLUMNS``.
TARGET_LABEL: str = "in_hospital_mortality"

# 🟢 Ordered tuple of candidate target columns for the current project.
#     Only :data:`TARGET_LABEL` participates in modelling.
#     （仅 :data:`TARGET_LABEL` 会用于建模，其余条目用于避免误把目标列当成特征。）
TARGET_COLUMNS: Tuple[str, ...] = ("in_hospital_mortality", "28d_mortality")

# 🟢 File names for the canonical dataset splits. Replace the placeholders with
#     your own file names.
#     （若没有外部验证集，可删除 ``"external_validation"`` 条目，并同步更新
#     :data:`BASELINE_DATASET_LABELS` 与 :data:`BASELINE_DATASET_ORDER`。）
DATASET_FILENAMES: Dict[str, Optional[str]] = {
    "train": "mimic-mortality-train.tsv",
    "internal_test": "mimic-mortality-test.tsv",
    # Validation features are derived via ``train_test_split`` from the train file.
    "external_validation": "eicu-mortality-external_val.tsv",
}

# 🟡 Human-readable labels for standard dataset splits used in reports。键值必须
#     与 :data:`DATASET_FILENAMES`、下游缓存文件命名保持一致。若无外部验证集，请移除
#     ``"external_validation"`` 项并在 :data:`BASELINE_DATASET_ORDER` 中同步删除。
BASELINE_DATASET_LABELS: Dict[str, str] = {
    "train": "Train",
    "validation": "Validation",
    "internal_test": "Test",
    "external_validation": "External cohort",
}

# 🟡 Preferred ordering of dataset labels when generating tables and figures。
#     该顺序会影响生成的表格/图像，亦用于遍历基线模型缓存。请确保仅包含实际存在的数据集键。
BASELINE_DATASET_ORDER: Tuple[str, ...] = (
    "train",
    "validation",
    "internal_test",
    "external_validation",
)

# 🟡 Random seed reused across dataset splits, baseline models, and Optuna.
RANDOM_STATE: int = 20201021

# 🟢 Clinical score columns that act as reference benchmarks.
#     （这些列会被排除在特征之外，仅在评估或缺失值处理阶段使用。若研究中没有临床评分，
#     请将该元组设为 ``()`` 并跳过相关对照分析。）
BENCHMARK_COLUMNS: Tuple[str, ...] = (
    "APS_III",
    "APACHE_IV",
    "SAPS_II",
    "OASIS",
)

# 🟡 Strategy used when comparing clinical scores with model predictions.
#     Available options:
#     ``"imputed"`` – default iterative imputation before evaluating scores.
#     Any other value – keep observed scores and skip rows with missing values.
#     （修改时请同步阅读 `examples/mimic_mortality_utils.py` 中的注释，确保主流程行为一致。）
CLINICAL_SCORE_BENCHMARK_STRATEGY: str = "imputed"

# 🟡 Fraction of the training cohort reserved for internal validation.
VALIDATION_SIZE: float = 0.2

# 🟡 Minimum validation AUROC for an Optuna trial to be considered viable.
PARETO_MIN_VALIDATION_ROAUC: float = 0.81

# 🟡 Maximum absolute AUROC gap between train and validation for Pareto members.
PARETO_MAX_ABS_DELTA_AUC: float = 0.035

# 🟡 Candidate hidden-layer widths for the shared SUAVE backbone.
HIDDEN_DIMENSION_OPTIONS: Dict[str, Tuple[int, ...]] = {
    "lean": (64, 32),
    "compact": (96, 48),
    "small": (128, 64),
    "medium": (256, 128),
    "wide": (384, 192),
    "extra_wide": (512, 256),
    "ultra_wide": (640, 320),
}

# 🟡 Candidate hidden-layer widths for the prediction heads.
HEAD_HIDDEN_DIMENSION_OPTIONS: Dict[str, Tuple[int, ...]] = {
    "minimal": (16,),
    "compact": (32,),
    "small": (48,),
    "medium": (48, 32),
    "wide": (96, 48, 16),
    "extra_wide": (64, 128, 64, 16),
    "deep": (128, 64, 32),
}

# 🟡 Default configuration passed to :func:`build_analysis_config`.
DEFAULT_ANALYSIS_CONFIG: Dict[str, object] = {
    "optuna_trials": 5,
    "optuna_timeout": 3600 * 48,
    "optuna_study_prefix": "supervised",
    "optuna_storage": None,
    "output_dir_name": "research_outputs_supervised",
}

# 🟡 Script-mode defaults that force regeneration of cached artefacts。
#     （交互式运行会强制将这些标志视为 False；命令行模式按照此处的默认值执行。）
#     - ``FORCE_UPDATE_SYNTHETIC_DATA`` 会跳过合成数据缓存并重新采样。
#     - ``FORCE_UPDATE_C2ST_MODEL`` 会重新训练 C2ST 分类器。
#     - ``FORCE_UPDATE_DISTRIBUTION_SHIFT`` 会重新计算全局/逐特征漂移指标。
FORCE_UPDATE_FLAG_DEFAULTS: Dict[str, bool] = {
    "FORCE_UPDATE_BENCHMARK_MODEL": False,  # Retrain cached classical baselines.
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

# 🟡 Canonical sub-directory names for artefacts written during the analysis.
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
# === Baseline model configuration ============================================
# =============================================================================


def _build_logistic_regression_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=500, random_state=random_state),
            ),
        ]
    )


def _build_knn_pipeline(random_state: int) -> Pipeline:
    """Return a KNN baseline pipeline (``random_state`` ignored)."""

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=25)),
        ]
    )


def _build_gradient_boosting_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        [
            (
                "classifier",
                GradientBoostingClassifier(random_state=random_state),
            ),
        ]
    )


def _build_random_forest_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        [
            (
                "classifier",
                RandomForestClassifier(n_estimators=200, random_state=random_state),
            ),
        ]
    )


def _build_svm_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=random_state)),
        ]
    )


# 🔴 Mapping of baseline model names to builder callables.
#     Each callable accepts ``random_state`` so downstream utilities can keep
#     stochastic pipelines reproducible. Only adjust this section when you know
#     which estimators should appear in the benchmark tables and caches.
BASELINE_MODEL_PIPELINE_BUILDERS: Dict[str, Callable[[int], Pipeline]] = {
    "Logistic regression": _build_logistic_regression_pipeline,
    "KNN": _build_knn_pipeline,
    "Gradient boosting": _build_gradient_boosting_pipeline,
    "Random forest": _build_random_forest_pipeline,
    "SVM (RBF)": _build_svm_pipeline,
}

# 🔴 Abbreviations used in plots and tables for the configured baseline models.
BASELINE_MODEL_ABBREVIATIONS: Dict[str, str] = {
    "Logistic regression": "LR",
    "KNN": "KNN",
    "Gradient boosting": "GB",
    "Random forest": "RF",
    "SVM (RBF)": "SVM",
}


# =============================================================================
# === Feature grouping and visualisation metadata ============================
# =============================================================================

# fmt: off
# 🟢 Assign original feature names to semantic groups for feature engineering
#     and visualisation. Update these mappings to reflect your dataset.
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

# 🟡 Colour palette applied to feature groups in path graphs.
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

# 🟢 Node metadata reused across reports and visualisations.
PATH_GRAPH_NODE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "age": {"group": "Demographics & Vitals", "label": "Age"},
    "sex": {"group": "Demographics & Vitals", "label": "Male sex"},
    "BMI": {"group": "Demographics & Vitals", "label": r"BMI"},
    "temperature": {"group": "Demographics & Vitals", "label": "Temperature"},
    "heart_rate": {"group": "Demographics & Vitals", "label": "Heart rate"},
    "respir_rate": {"group": "Demographics & Vitals", "label": "Respiratory rate"},
    "SBP": {"group": "Hemodynamics & Perfusion", "label": "SBP"},
    "DBP": {"group": "Hemodynamics & Perfusion", "label": "DBP"},
    "MAP": {"group": "Hemodynamics & Perfusion", "label": "MAP"},
    "Lac": {"group": "Hemodynamics & Perfusion", "label": "Serum lactate"},
    "SOFA_cns": {"group": "Organ Support & Neurology", "label": "SOFA CNS"},
    "CRRT": {"group": "Organ Support & Neurology", "label": "CRRT"},
    "Respiratory_Support": {
        "group": "Organ Support & Neurology",
        "label": "Respiratory support",
    },
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
    "HCO3-": {
        "group": "Metabolic & Electrolytes",
        "label": r"$\mathrm{HCO}_{3}^{-}$",
    },
    "Fg": {"group": "Coagulation", "label": "Fibrinogen"},
    "PT": {"group": "Coagulation", "label": "PT"},
    "APTT": {"group": "Coagulation", "label": "APTT"},
    "PH": {"group": "Respiratory and Blood Gas", "label": "pH"},
    "PaO2": {
        "group": "Respiratory and Blood Gas",
        "label": r"$\mathrm{PaO}_{2}$",
    },
    "PaO2/FiO2": {
        "group": "Respiratory and Blood Gas",
        "label": r"$\mathrm{PaO}_{2}/\mathrm{FiO}_{2}$ ratio",
    },
    "PaCO2": {
        "group": "Respiratory and Blood Gas",
        "label": r"$\mathrm{PaCO}_{2}$",
    },
    "in_hospital_mortality": {
        "group": "Outcome",
        "label": "In-hospital mortality",
    },
}

# 🔴 Convenience lookup for node labels derived from the metadata above.
PATH_GRAPH_NODE_LABELS: Dict[str, str] = {
    node_id: metadata["label"]
    for node_id, metadata in PATH_GRAPH_NODE_DEFINITIONS.items()
}

# 🔴 Convenience lookup for node groups derived from the metadata above.
PATH_GRAPH_NODE_GROUPS: Dict[str, str] = {
    node_id: metadata["group"]
    for node_id, metadata in PATH_GRAPH_NODE_DEFINITIONS.items()
}

# 🔴 Map nodes to colours by combining node groups with the palette above.
PATH_GRAPH_NODE_COLORS: Dict[str, str] = {
    node_id: PATH_GRAPH_GROUP_COLORS[metadata["group"]]
    for node_id, metadata in PATH_GRAPH_NODE_DEFINITIONS.items()
}


__all__ = [
    "ANALYSIS_SUBDIRECTORIES",
    "BASELINE_DATASET_LABELS",
    "BASELINE_DATASET_ORDER",
    "BASELINE_MODEL_ABBREVIATIONS",
    "BASELINE_MODEL_PIPELINE_BUILDERS",
    "BENCHMARK_COLUMNS",
    "CLINICAL_SCORE_BENCHMARK_STRATEGY",
    "DATA_DIR",
    "DATASET_FILENAMES",
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
    "TARGET_LABEL",
    "TARGET_COLUMNS",
    "VALIDATION_SIZE",
    "VAR_GROUP_DICT",
]
