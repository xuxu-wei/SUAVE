"""Hard-coded configuration block for the research workflow template.

This module centralises every project-specific constant that is expected to
change when porting the research workflow to a new dataset. Users should edit
these values instead of modifying :mod:`analysis_utils` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


#: 模板根目录路径，供派生目录配置复用。
#: 建议修改：否 —— 由脚本自动计算。
TEMPLATE_ROOT = Path(__file__).resolve().parent


# =============================================================================
# === Dataset, modelling and evaluation defaults ==============================
# =============================================================================

#: 控制所有随机流程（如数据拆分、模型训练）稳定性的随机数种子。
#: 建议修改：按需 —— 若希望与原模板保持完全一致，可保留默认值。
RANDOM_STATE: int = 20201021
#: 研究目标变量的列名集合，按照重要性或优先级排序。
#: 建议修改：是 —— 应替换为当前研究任务的真实标签列。
TARGET_COLUMNS: Tuple[str, ...] = ("in_hospital_mortality", "28d_mortality")
#: 数据集中可作为传统临床评分基线的列名集合。
#: 建议修改：是 —— 根据实际可用的评分或对照指标进行增删。
BENCHMARK_COLUMNS: Tuple[str, ...] = (
    "APS_III",
    "APACHE_IV",
    "SAPS_II",
    "OASIS",
)

#: 当比较临床评分与模型预测时采用的缺失值处理策略。
#: 建议修改：按需 —— 保持为 "imputed" 可复用模板逻辑，若采用其他策略需同步更新分析代码。
CLINICAL_SCORE_BENCHMARK_STRATEGY: str = "imputed"

#: 验证集在训练集中的占比，用于数据拆分。
#: 建议修改：按需 —— 根据样本量与实验需求调节。
VALIDATION_SIZE: float = 0.2

#: 模板默认假设的原始数据目录，所有输入数据应放置于此。
#: 建议修改：是 —— 指向当前项目的数据存储路径。
DATA_DIR: Path = (TEMPLATE_ROOT / "datasets").resolve()

# Thresholds governing which Optuna trials are considered viable for
# persistence.
#: Optuna 试验被视为合格所需达到的最小验证 ROC AUC。
#: 建议修改：按需 —— 可依据任务难度与性能期望调整。
PARETO_MIN_VALIDATION_ROAUC: float = 0.81
#: 允许的训练/验证 AUC 最大绝对差值，用于筛除过拟合配置。
#: 建议修改：按需 —— 若样本较少或度量差异较大，可适度放宽。
PARETO_MAX_ABS_DELTA_AUC: float = 0.035

#: 主干网络隐藏层宽度的候选配置集合，键为描述性名称。
#: 建议修改：按需 —— 新任务可增删配置以匹配特征复杂度。
HIDDEN_DIMENSION_OPTIONS: Dict[str, Tuple[int, ...]] = {
    "lean": (64, 32),
    "compact": (96, 48),
    "small": (128, 64),
    "medium": (256, 128),
    "wide": (384, 192),
    "extra_wide": (512, 256),
    "ultra_wide": (640, 320),
}

#: 预测头部网络隐藏层宽度的候选配置集合。
#: 建议修改：按需 —— 按任务需求调整深度与宽度。
HEAD_HIDDEN_DIMENSION_OPTIONS: Dict[str, Tuple[int, ...]] = {
    "minimal": (16,),
    "compact": (32,),
    "small": (48,),
    "medium": (48, 32),
    "wide": (96, 48, 16),
    "extra_wide": (64, 128, 64, 16),
    "deep": (128, 64, 32),
}

#: 超参数搜索与输出目录的默认配置项。
#: 建议修改：按需 —— 若需调整搜索预算、存储位置或命名规则，请更新对应键值。
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
#: 控制是否强制重新生成各类缓存产物的默认布尔开关。
#: 建议修改：按需 —— 仅在需要强制刷新缓存时将对应值设为 True。
FORCE_UPDATE_FLAG_DEFAULTS: Dict[str, bool] = {
    "FORCE_UPDATE_BENCHMARK_MODEL": False,
    "FORCE_UPDATE_TSTR_MODEL": True,
    "FORCE_UPDATE_TRTR_MODEL": True,
    "FORCE_UPDATE_SUAVE": False,
}

#: 约定各分析阶段的子目录命名，便于生成统一的输出结构。
#: 建议修改：按需 —— 若本地目录结构不同，可按需调整命名。
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
#: 将原始变量划分至功能性分组的映射，用于特征工程或可视化。
#: 建议修改：是 —— 应根据当前数据的变量体系进行重新组织。
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


#: 展示路径图（path graph）时为各变量组指定的颜色。
#: 建议修改：按需 —— 为保持视觉一致可沿用默认配色。
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


#: 节点 ID 与其标签/分组的定义，供路径图或报告复用。
#: 建议修改：是 —— 应与 VAR_GROUP_DICT 中的变量保持一致。
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


#: 提取节点标签的便捷映射，通常无需手动改动。
#: 建议修改：否 —— 当 PATH_GRAPH_NODE_DEFINITIONS 更新时会自动同步。
PATH_GRAPH_NODE_LABELS: Dict[str, str] = {
    node_id: metadata["label"]
    for node_id, metadata in PATH_GRAPH_NODE_DEFINITIONS.items()
}


#: 提取节点所属分组的便捷映射，依赖 PATH_GRAPH_NODE_DEFINITIONS。
#: 建议修改：否 —— 通常保持自动推导即可。
PATH_GRAPH_NODE_GROUPS: Dict[str, str] = {
    node_id: metadata["group"]
    for node_id, metadata in PATH_GRAPH_NODE_DEFINITIONS.items()
}


#: 将节点映射至颜色的便捷字典，用于绘图。
#: 建议修改：否 —— 若需调整颜色，请在 PATH_GRAPH_GROUP_COLORS 中操作。
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

