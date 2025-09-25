# 研究协议（Research Protocol）

本文档汇总 SUAVE 项目的研究流程约定，帮助在不同数据集上复现实验、整理结果并产出可审计的研究报告。

## MIMIC-IV 住院死亡率（监督）分析方案

以下流程对齐 `examples/research-mimic_mortality_supervised.py` 的实现，目标是在 MIMIC-IV 住院患者的死亡率预测任务上复现 SUAVE 研究级评估，并生成可直接纳入技术报告的成果物。

### 0. 分析脚本与角色定位

1. `research-mimic_mortality_supervised.py`：主流程脚本，负责串联数据加载、特征工程、模型训练、评估与归档。
2. `cls_eval.py`：封装分类任务常用的指标计算、概率校准与报告生成功能，被主流程脚本引用以保持评估一致性。
3. `mimic_mortality_utils.py`：提供数据读取、特征构建、schema 维护与基线模型工厂等工具函数，支撑监督/超参优化脚本复用同一套处理逻辑。
4. `research-mimic_mortality_optimize.py`：在一致的预处理基础上运行 Optuna 超参搜索，并将最优 trial 供主流程脚本直接复现。

### 1. 研究目标与核心指标

1. 主要目标：基于 ICU 入科时刻前 24h 内的结构化特征预测 `in_hospital_mortality`。
2. 判定指标：AUROC/AUPRC 作为判别性能主指标，Brier Score 与校准曲线衡量概率校准，ECE 作为补充。
3. 次要分析：eICU 外部验证集的一致性评估、合成数据 TSTR/TRTR、潜空间可视化及隐私基线。

### 2. 数据来源与管理

1. 训练/内部验证/测试使用 `examples/data/sepsis_mortality_dataset/` 中基于 **MIMIC-IV** 抽样生成的 `mimic-mortality-train.tsv` 与 `mimic-mortality-test.tsv`，样本均从入院 24 小时内满足 Sepsis-3 标准的患者筛选得到。
2. 外部验证集来自 eICU（`eicu-mortality-external_val.tsv`），采用与 MIMIC-IV 相同的 Sepsis-3 纳入标准以保证队列可比性。
3. MIMIC-IV 测试集与 eICU 数据仅用于最终评估，不参与超参选择或模型选择；所有 TSV 均需保留原始列名，脚本在 `analysis_outputs_supervised/` 下派生的缓存、插补产物与评估文件应纳入审计归档。

### 3. 准备阶段

1. 确认目标标签（默认 `in_hospital_mortality`）存在于 `TARGET_COLUMNS` 列表，并记录所有候选标签以备扩展分析。
2. 建立输出目录 `examples/analysis_outputs_supervised/`，若 Optuna 存储不存在则自动创建；必要时备份历史运行的 best trial JSON。
3. 若已有 Optuna trial 或 SUAVE 模型缓存，需在研究日志中记录其生成配置，以便差异分析。

### 4. 数据加载与 Schema 校验

1. 调用 `load_dataset` 读取 MIMIC-IV 训练/测试及 eICU 外部验证集的 TSV，确保无缺失列并保留原始数据类型。
2. 通过 `define_schema(train_df, FEATURE_COLUMNS, mode="interactive")` 生成初始 schema；对 BMI、呼吸支持等级（`Respiratory_Support`）和淋巴细胞百分比（`LYM%`）执行人工修正，以保证类型一致性。
3. 使用 `schema_to_dataframe` 与 `render_dataframe` 输出列类型摘要并保存截图/Markdown，作为数据说明附件的一部分。

### 5. 特征构建与内部验证划分

1. 采用 `prepare_features` 对训练集进行特征工程；随后调用 `train_test_split`（`VALIDATION_SIZE`、`RANDOM_STATE`）在训练集内部创建分层验证集。
2. 对测试集与外部验证集复用同一特征处理函数，确保列顺序与训练集匹配，并将结果缓存至 `baseline_feature_frames`。
3. 若后续分析扩展到其他标签，需要复用相同的特征转换并记录生成时间及校验摘要。

### 6. 基线模型与对照实验

1. 借助 `load_or_create_iteratively_imputed_features` 对各评估集执行迭代插补，生成可复用的 `*_imputed.joblib` 文件。
2. 构建 Logistic Regression（带标准化）、KNN、Decision Tree、Random Forest 与 RBF-SVM 等 `Pipeline`，通过 `evaluate_transfer_baselines` 统一训练并评估。
3. 输出的指标 CSV 与 Markdown 表格需包含训练/验证/测试/eICU 全量指标，若外部验证缺失标签需在脚注注明处理策略。

### 7. SUAVE 模型构建、调参与训练

1. 若存在历史最优 trial，优先读取 `optuna_best_params_{label}.json`；否则调用 `build_suave_model` 以默认超参初始化模型，并记录关键参数（latent_dim、beta、dropout 等）。
2. 训练顺序遵循脚本：先执行 VAE warm-up，再进行分类头独立训练，最后 joint fine-tuning，必要时启用分类损失权重自动调节。
3. 对于每个阶段，记录训练轮数、早停标准、最优模型路径以及 GPU/CPU 运行时长，用于技术报告的实验设置章节。

### 8. 概率校准与不确定性量化

1. 通过 `fit_isotonic_calibrator` 在内部验证集上拟合等渗校准器，必要时回退至逻辑回归温度缩放，并保存校准对象。
2. 使用 `evaluate_predictions` 对训练、验证、MIMIC-IV 测试与 eICU 集执行 bootstrap（默认 1000 次）以估计指标置信区间，并生成 Excel 汇总。
3. 运行 `simple_membership_inference` 获得隐私攻击基线，将结果写入研究报告的风险评估章节。

### 9. 合成数据（TSTR/TRTR）与分布漂移分析

1. 调用 `build_tstr_training_sets` 创建真实与合成训练集，并通过 `make_baseline_model_factories` 构建一致的下游分类器族。
2. 使用 `plot_transfer_metric_bars`、`kolmogorov_smirnov_statistic`、`rbf_mmd` 与 `mutual_information_feature` 评估分布一致性。
3. 所有 TRTR/TSTR 指标、KS/MMD/互信息结果应保存为 CSV 与可视化 PNG，纳入附录及复现包。

### 10. 潜空间可视化、报告生成与归档

1. 借助 `plot_latent_space` 对潜空间执行 PCA/UMAP（视脚本配置）投影，输出训练、验证、测试与外部集的可视化比较。
2. 使用 `dataframe_to_markdown`、`render_dataframe` 与 `write_results_to_excel_unique` 汇总评估结果，最终通过 `build_prediction_dataframe` 与 `evaluate_predictions` 生成 `evaluation_summary_{label}.md` 技术报告草稿。
3. 在归档目录保留所有原始数据引用、模型权重（`.pt`/`.joblib`）、插补缓存、图表、Markdown 报告及运行日志，以支持第三方审计与论文附录撰写。
