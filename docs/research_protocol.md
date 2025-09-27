# 研究协议（Research Protocol）

本文档汇总 SUAVE 项目的研究流程约定，帮助在不同数据集上复现实验、整理结果并产出可审计的研究报告。核心实现围绕四个示例脚本协同展开：

- `examples/research-mimic_mortality_supervised.py`：监督学习主流程脚本，串联数据加载、特征工程、模型训练、校准与报告生成。
- `examples/cls_eval.py`：分类评估工具，提供标准化指标计算与图表输出，支撑内部验证与外部评估。
- `examples/mimic_mortality_utils.py`：特征构建、缺失值处理与 schema 校验等通用函数库，确保数据处理步骤可复用。
- `examples/research-mimic_mortality_optimize.py`：Optuna 超参搜索入口，负责调度试验、记录最佳结果并供主流程读取。

## MIMIC-IV 住院死亡率（监督）分析方案

以下流程对齐 `examples/research-mimic_mortality_supervised.py` 的实现，目标是在 MIMIC-IV 住院患者的死亡率预测任务上复现 SUAVE 研究级评估，并生成可直接纳入技术报告的成果物。

### 1. 研究目标与核心指标

1. 主要目标：基于 ICU 入科时刻前 24h 内的结构化特征预测 `in_hospital_mortality`。
2. 判定指标：AUROC/AUPRC 作为判别性能主指标，Brier Score 与校准曲线衡量概率校准，ECE 作为补充。
3. 次要分析：eICU 外部验证集的一致性评估、合成数据 TSTR/TRTR、潜空间可视化及隐私基线。

### 2. 数据来源与管理

1. 训练/内部验证/测试使用 `examples/data/sepsis_mortality_dataset/` 中基于 **MIMIC-IV** 抽样生成的 `mimic-mortality-train.tsv` 与 `mimic-mortality-test.tsv`，其样本均为入院 24h 内满足 Sepsis-3 标准的住院患者。
2. 外部验证集来自 eICU-CRD（`eicu-mortality-external_val.tsv`），采用与 MIMIC-IV 相同的纳入标准，同样聚焦于入院 24h 内满足 Sepsis-3 标准的患者。
3. MIMIC-IV 测试集与 eICU 外部验证集仅用于最终评估，不参与 Optuna 搜索或其他超参选择流程。
4. 所有 TSV 均需保留原始列名；主流程在 `examples/research_outputs_supervised/` 下派生的缓存、插补产物与评估文件应纳入审计归档，根目录仅保留研究整体的 Markdown 总结。
5. `examples/research_outputs_supervised/` 采用分阶段目录结构：`01_schema_validation/`、`02_feature_engineering/`、`03_optuna_search/`、`04_suave_model/`、`05_evaluation_metrics/`、`06_bootstrap_analysis/`、`07_baseline_models/`、`08_transfer_learning/`、`09_distribution_shift/`、`10_privacy_assessment/` 与 `11_visualizations/`。各阶段产物应写入对应子目录，便于审计检索。

### 3. 准备阶段

1. 确认目标标签（默认 `in_hospital_mortality`）存在于 `TARGET_COLUMNS` 列表，并记录所有候选标签以备扩展分析。
2. 建立输出目录 `examples/research_outputs_supervised/`，若 Optuna 存储不存在则自动创建；必要时备份历史运行的 best trial JSON。
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
2. 构建 Logistic Regression（带标准化）、KNN、Decision Tree、Random Forest 与 RBF-SVM 等 `Pipeline`，通过 `evaluate_transfer_baselines` 统一训练并评估；所有基线共享 `compute_binary_metrics` 统计 AUC、ACC、SPE、SEN 与 Brier，并写入 `baseline_models_{label}.csv`。
3. 脚本在 `07_baseline_models/` 下缓存拟合后的 `baseline_estimators_{label}.joblib`，二次运行时默认复用；若需强制重新训练，可在执行前设置 `FORCE_UPDATE_BENCHMARK_MODEL=1`（默认为 `0`）。相关布尔参数统一通过 `mimic_mortality_utils.read_bool_env_flag` 解析，以便批量实验脚本复用。
4. 在临床基准方面，保留传统 ICU 评分（如数据集中现成的 SOFA 及相关器官支持指标）作为零参数对照：当 `mimic_mortality_utils.py` 中登记了此类 `baseline_probability_map` 项时，同样纳入基线汇总并在 `Notes` 中标记“临床评分”。
5. 输出的指标 CSV 与 Markdown 表格需包含训练/验证/测试/eICU 全量指标，若外部验证缺失标签需在脚注注明处理策略。


### 7. SUAVE 模型构建、调参与训练

1. 若存在历史最优 trial，优先读取 `optuna_best_params_{label}.json`；否则调用 `build_suave_model` 以默认超参初始化模型，并记录关键参数（latent_dim、beta、dropout 等）。
2. 训练顺序遵循脚本：先执行 VAE warm-up，再进行分类头独立训练，最后 joint fine-tuning，必要时启用分类损失权重自动调节。
3. 对于每个阶段，记录训练轮数、早停标准、最优模型路径以及 GPU/CPU 运行时长，用于技术报告的实验设置章节。
4. Optuna 搜索完成后需导出参数重要性、最优值收敛轨迹与多目标帕累托前沿图（均保存为 PNG/SVG/PDF/JPG），便于后续调参与审计复核；默认输出位于 `03_optuna_search/figures/`。
5. Trial 级别的搜索记录需写入 `optuna_trials_{label}.csv` 并在日志中展示前 10 个验证集 AUROC 最优的 trial，确保调参轨迹透明可追溯。
6. `research-mimic_mortality_supervised.py` 在交互模式下会读取 Optuna 帕累托前沿并列出各 trial 的验证集 AUROC、TSTR/TRTR ΔAUC 与本地模型保存状态，等待人工输入 trial ID 以加载或重新训练；脚本模式可通过 `--trial-id`（或位置参数）指定目标 trial，若未提供则优先加载最近一次保存的模型，缺失时再按照硬阈值（AUROC>0.81、|ΔAUC|<0.035）自动选取帕累托前沿解重训模型。
7. Optuna 优化脚本会在 `04_suave_model/` 下生成 `suave_model_manifest_{label}.json`，记录 trial 编号、目标函数值与模型/校准器路径；主流程在加载前需校验 manifest 所指向的 artefact 是否存在，不满足时回退至最近一次保存的权重或触发重新训练；当触发重新训练时会自动将新的 SUAVE 权重写入 `suave_best_{label}.pt` 以恢复后续运行的缓存链路。

### 8. 概率校准与不确定性量化

1. 通过 `fit_isotonic_calibrator` 在内部验证集上拟合专为 SUAVE 封装的等渗校准器（内部使用 `IsotonicRegression`，兼容缺失 `decision_function` 的估计器），必要时回退至逻辑回归温度缩放，并保存校准对象；若缓存缺失或校准器与模型不匹配，主流程会重新训练并自动序列化新的校准 artefact。
2. 使用 `evaluate_predictions` 对训练、验证、MIMIC-IV 测试与 eICU 集执行 bootstrap（默认 1000 次）以估计指标置信区间，并生成 Excel 汇总；除表格主列的 AUC、ACC、SPE、SEN、Brier 外，Excel 中还会给出 `accuracy`、`balanced_accuracy`、`f1_macro`、`recall_macro`、`specificity_macro`、`sensitivity_pos`、`specificity_pos`、`roc_auc`、`pr_auc` 及其置信区间，以支撑不同风险偏好的诊断分析。
3. 评估函数会同步导出 bootstrap 的原始采样记录：总体指标写入 `bootstrap_overall_records_{label}.csv`，逐类指标写入 `bootstrap_per_class_records_{label}.csv`，用于复核任意抽样迭代的轨迹。
4. 将生成的 `evaluation_metrics.csv`、校准曲线及相关图表保存为 PNG/SVG/PDF/JPG 四种格式，并写入 `05_evaluation_metrics/` 子目录，同时在实验日志中登记路径，便于后续报告引用与审计复核。

### 9. 合成数据（TSTR/TRTR）与分布漂移分析

1. 调用 `build_tstr_training_sets` 创建 `TRTR (real)`、`TSTR synthesis`、`TSTR synthesis-balance`、`TSTR synthesis-augment`、`TSTR synthesis-5x` 与 `TSTR synthesis-5x balance` 等方案，并在评估阶段对照 MIMIC-IV 测试集及（若标签可用）eICU 外部验证集。
2. 通过 `make_baseline_model_factories` 注册 `Logistic regression`、`Random forest` 与 `XGBoost` 三类下游分类器，对每个训练方案分别拟合并在 `evaluate_transfer_baselines` 中统计 `accuracy` 与 `roc_auc`（含置信区间）。
3. `evaluate_transfer_baselines` 的结果会缓存在 `08_transfer_learning/` 中的 `tstr_trtr_results_{label}.joblib`；若需跳过缓存直接重训，可设置 `FORCE_UPDATE_TSTR_MODEL=1` 或 `FORCE_UPDATE_TRTR_MODEL=1`（默认均为 `1`，即始终更新）。缓存中还包含训练/模型顺序元数据，复用时需保持与现有目录结构兼容。
4. 运行结束后会额外导出 bootstrap 抽样记录：整体指标写入 `tstr_trtr_bootstrap_overall_records_{label}.csv`，逐类指标写入 `tstr_trtr_bootstrap_per_class_records_{label}.csv`，便于二次分析 TSTR/TRTR 的不确定性分布。
5. 分布漂移的主要结局指标采用 `classifier_two_sample_test`：以 `TRTR (real)` 与 `TSTR synthesis` 作为两类样本，使用 XGBoost 分类器执行 C2ST 并报告 ROC-AUC 及 95% bootstrap 置信区间；同一流程下的逻辑回归 AUC 作为次要敏感性指标。
6. 次要结局指标包括全局 `rbf_mmd` 与 `energy_distance`，均基于置换检验给出 `p` 值，用于量化生成数据与真实分布之间的整体偏移程度。
7. `rbf_mmd`、`energy_distance` 与 `mutual_information_feature` 继续按特征逐列计算，定位非单调差异或潜在信息泄露风险的列，并与 C2ST 结果交叉验证。
8. `plot_transfer_metric_bars` 负责绘制 TSTR/TRTR 方案在 `accuracy`、`roc_auc` 等指标上的分组柱状图（含置信区间），展示不同生成数据训练集对下游分类器的影响；其输出与分布漂移指标无直接对应关系，应单列在报告的“生成数据迁移性能”小节中说明。
9. 在生成数据性能分析后运行 `simple_membership_inference`，补充隐私攻击基线并记录攻击 AUC/阈值，支撑隐私风险评估章节。
10. 所有 TRTR/TSTR 指标、C2ST 结果及分布漂移指标需导出至同一个 Excel 工作簿：`overall` 工作表整合 C2ST 与全局 MMD/能量距离统计，`per_feature` 工作表罗列逐列 MMD/能量距离/互信息；配套的分布漂移可视化图表需在 `09_distribution_shift/` 中以 PNG/SVG/PDF/JPG 四种格式保存，纳入附录及复现包。


### 10. 潜空间可视化、报告生成与归档

1. 借助 `plot_latent_space` 对潜空间执行 PCA/UMAP（视脚本配置）投影，输出训练、验证、测试与外部集的可视化比较。
2. 使用 `dataframe_to_markdown`、`render_dataframe` 与 `write_results_to_excel_unique` 汇总评估结果，最终通过 `build_prediction_dataframe` 与 `evaluate_predictions` 生成 `evaluation_summary_{label}.md` 技术报告草稿。
3. 在归档目录保留所有原始数据引用、模型权重（`.pt`/`.joblib`）、插补缓存、图表、Markdown 报告及运行日志，以支持第三方审计与论文附录撰写。
4. 在潜空间解释章节调用 `suave.plots.compute_feature_latent_correlation` 计算潜空间与临床特征（含目标标签）的 Spearman 相关性，依托
   statsmodels 进行 FDR/Bonferroni/Holm 校正。将 `compute_feature_latent_correlation` 的结果导出为相关系数/`p` 值 CSV，并以
   `plot_feature_latent_outcome_path_graph` 绘制特征→潜变量→结局的多层次路径图（显著性水平 0.05，临床特征按定义的 group/color 映射着色），
   作为潜空间解释章节的核心可视化。`plot_feature_latent_correlation_heatmap(..., value="correlation")`、
   `plot_feature_latent_correlation_heatmap(..., value="pvalue")` 与 `plot_feature_latent_correlation_bubble` 作为补充结果，仍按
   VAR_GROUP_DICT 分组生成热图与气泡图，用于量化隐空间与关键临床变量的相关性分布。
