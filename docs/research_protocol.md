# 研究协议（Research Protocol）

本文档汇总 SUAVE 项目的研究流程约定，帮助在不同数据集上复现实验、整理结果并产出可审计的研究报告。核心实现围绕四个示例脚本协同展开：

- `examples/research-mimic_mortality_supervised.py`：监督学习主流程脚本，串联数据加载、特征工程、模型训练、校准与报告生成。
- `examples/cls_eval.py`：分类评估工具，提供标准化指标计算与图表输出，支撑内部验证与外部评估。
- `examples/mimic_mortality_utils.py`：特征构建、缺失值处理与 schema 校验等通用函数库，确保数据处理步骤可复用。
- `examples/research-mimic_mortality_optimize.py`：Optuna 超参搜索入口，负责调度试验、记录最佳结果并供主流程读取。

## MIMIC-IV 住院死亡率（监督）分析方案

以下流程对齐 `examples/research-mimic_mortality_supervised.py` 的实现，目标是在 MIMIC-IV 住院患者的死亡率预测任务上复现 SUAVE 研究级评估，并生成可直接纳入技术报告的成果物。

### 1. 研究目标与核心指标

**目的**：界定住院死亡率预测任务的建模目标与主要评估维度，为后续实验提供统一的成效判断标准。
**结果解读**：以 AUROC/AUPRC 评估判别性能，结合 Brier Score、ECE 与校准曲线理解概率输出的可靠性。
**输入**：依赖 `examples/mimic_mortality_utils.py` 中的目标标签枚举以及研究脚本的评估配置。
**输出**：形成研究日志中的目标声明，未生成额外文件。

1. 主要目标：基于 ICU 入科时刻前 24h 内的结构化特征预测 `in_hospital_mortality`。
2. 判定指标：AUROC/AUPRC 作为判别性能主指标，Brier Score 与校准曲线衡量概率校准，ECE 作为补充。
3. 次要分析：eICU 外部验证集的一致性评估、合成数据 TSTR/TRTR、潜空间可视化及隐私基线。

### 2. 数据来源与管理

**目的**：列出训练与验证所需的数据集，并定义研究输出的目录结构，使实验 artefact 可追踪、可复现。
**结果解读**：数据完整性通过 schema 校验与缺失值报告进行确认，目录命名映射到后续章节以辅助审计。
**输入**：期望在 `examples/data/sepsis_mortality_dataset/` 下提供三份 TSV 数据集；若存在补充数据，需要同步更新 schema 定义。
**输出**：在 `examples/research_outputs_supervised/` 下创建分阶段目录，并记录数据加载日志。

1. 训练/内部验证/测试使用 `examples/data/sepsis_mortality_dataset/` 中基于 **MIMIC-IV** 抽样生成的 `mimic-mortality-train.tsv` 与 `mimic-mortality-test.tsv`，其样本均为入院 24h 内满足 Sepsis-3 标准的住院患者。
2. 外部验证集来自 eICU-CRD（`eicu-mortality-external_val.tsv`），采用与 MIMIC-IV 相同的纳入标准，同样聚焦于入院 24h 内满足 Sepsis-3 标准的患者。
3. MIMIC-IV 测试集与 eICU 外部验证集仅用于最终评估，不参与 Optuna 搜索或其他超参选择流程。
4. 所有 TSV 均需保留原始列名；主流程在 `examples/research_outputs_supervised/` 下派生的缓存、插补产物与评估文件应纳入审计归档，根目录仅保留研究整体的 Markdown 总结。
5. `examples/research_outputs_supervised/` 采用与章节编号一致的分阶段目录结构：`01_data_and_schema/`、`02_feature_engineering/`、`03_optuna_search/`、`04_suave_training/`、`05_calibration_uncertainty/`、`06_evaluation_metrics/`、`07_bootstrap_analysis/`、`08_baseline_models/`、`09_tstr_trtr_transfer/`、`10_distribution_shift/`、`11_privacy_assessment/` 与 `12_visualizations/`。各阶段产物应写入对应子目录，便于审计检索。

### 3. 准备阶段

**目的**：初始化实验配置、输出目录与环境变量，确保后续步骤具备可重入性与可追溯性。
**结果解读**：通过目录与配置检查确认缓存是否可复用，以及需要强制刷新的 artefact。
**输入**：读取 `examples/mimic_mortality_utils.py` 中的默认配置，并可根据命令行参数或环境变量覆写。
**输出**：在研究日志中记录配置概况与缓存状态，无新增文件。

1. 确认目标标签（默认 `in_hospital_mortality`）存在于 `TARGET_COLUMNS` 列表，并记录所有候选标签以备扩展分析。
2. 建立输出目录 `examples/research_outputs_supervised/`，若 Optuna 存储不存在则自动创建；必要时备份历史运行的 best trial JSON。
3. 若已有 Optuna trial 或 SUAVE 模型缓存，需在研究日志中记录其生成配置，以便差异分析。

### 4. 数据加载与 Schema 校验

**目的**：验证原始特征的列名、类型与取值范围，为特征工程建立可靠的输入基线。
**结果解读**：Schema 校验通过说明数据与脚本期望一致；若存在类型冲突需在日志中列明并修正。
**输入**：读取 `01_data_and_schema/` 中缓存的 TSV 或直接访问原始数据集。
**输出**：在 `01_data_and_schema/` 中保存 schema DataFrame、Markdown 与可选截图。

1. 调用 `load_dataset` 读取 MIMIC-IV 训练/测试及 eICU 外部验证集的 TSV，确保无缺失列并保留原始数据类型。
2. 通过 `define_schema(train_df, FEATURE_COLUMNS, mode="interactive")` 生成初始 schema；对 BMI、呼吸支持等级（`Respiratory_Support`）和淋巴细胞百分比（`LYM%`）执行人工修正，以保证类型一致性。
3. 使用 `schema_to_dataframe` 与 `render_dataframe` 输出列类型摘要并保存截图/Markdown，作为数据说明附件的一部分。

### 5. 特征构建与内部验证划分

**目的**：在统一的特征工程流程下衍生模型输入，并构建稳定的内部验证集。
**结果解读**：特征缓存的成功生成意味着后续模型可直接加载 `.joblib` 文件进行训练；若特征缺失需回到上一阶段修复。
**输入**：读取 `02_feature_engineering/` 中的缓存或运行 `prepare_features` 生成新特征。
**输出**：在 `02_feature_engineering/` 中写入特征矩阵、`baseline_feature_frames` 及验证划分日志。

1. 采用 `prepare_features` 对训练集进行特征工程；随后调用 `train_test_split`（`VALIDATION_SIZE`、`RANDOM_STATE`）在训练集内部创建分层验证集。
2. 对测试集与外部验证集复用同一特征处理函数，确保列顺序与训练集匹配，并将结果缓存至 `baseline_feature_frames`。
3. 若后续分析扩展到其他标签，需要复用相同的特征转换并记录生成时间及校验摘要。

### 6. 基线模型与对照实验

**目的**：提供与 SUAVE 模型独立的下游分类基线，衡量数据集与合成数据的可用性。
**结果解读**：AUC、准确率与 Brier Score 用于比较不同基线模型及临床评分的表现差异。
**输入**：从 `02_feature_engineering/` 读取插补前特征，并在 `load_or_create_iteratively_imputed_features` 中生成迭代插补结果。
**输出**：在 `08_baseline_models/` 写入 `baseline_estimators_{label}.joblib` 与 `baseline_models_{label}.csv`。

1. 借助 `load_or_create_iteratively_imputed_features` 对各评估集执行迭代插补，并将结果保存为 `02_feature_engineering/` 目录下的 `iterative_imputed_{dataset}_{label}.csv`。首次运行会以训练集为参考拟合插补器；后续运行若检测到缓存与原始特征形状匹配，则直接读取 CSV 而无需重新训练。`08_baseline_models/` 仅存放模型权重与指标，不重复缓存这些迭代插补特征。
2. 构建 Logistic Regression（带标准化）、Random Forest 与 GBDT `Pipeline`，通过 `evaluate_transfer_baselines` 统一训练并评估；所有基线共享 `compute_binary_metrics` 统计 AUC、ACC、SPE、SEN 与 Brier，并写入 `baseline_models_{label}.csv`。`mimic_mortality_utils.make_logistic_pipeline`、`make_random_forest_pipeline` 与 `make_gradient_boosting_pipeline` 均保持 scikit-learn 默认超参数，仅在需要时设置 `random_state`，确保与 C2ST/TSTR/TRTR 下游模型一致。上述流水线会在内部再执行一次迭代插补（同样采用 `max_iter=100`、`tol=1e-2`），从而保证独立运行时也具备缺失值鲁棒性。
3. 脚本在 `08_baseline_models/` 下缓存拟合后的 `baseline_estimators_{label}.joblib`，二次运行时默认复用；若需强制重新训练，可在执行前设置 `FORCE_UPDATE_BENCHMARK_MODEL=1`（默认为 `0`）。相关布尔参数统一通过 `mimic_mortality_utils.read_bool_env_flag` 解析，以便批量实验脚本复用。SUAVE 主模型及其校准器始终使用 `prepare_features` 产出的原始特征（即未经过该迭代插补管线），保持与生成器训练阶段的输入一致。
4. 在临床基准方面，保留传统 ICU 评分（如数据集中现成的 SOFA 及相关器官支持指标）作为零参数对照：当 `mimic_mortality_utils.py` 中登记了此类 `baseline_probability_map` 项时，同样纳入基线汇总并在 `Notes` 中标记“临床评分”。`CLINICAL_SCORE_BENCHMARK_STRATEGY` 默认设为 `"imputed"`，针对每个评分独立地使用建模特征驱动的 `IterativeImputer` 进行缺失填补，确保不在评分之间泄露信息；若调至 `"observed"`，则在评估阶段跳过缺失评分的样本，维持原始分布。
5. 输出的指标 CSV 与 Markdown 表格需包含训练/验证/测试/eICU 全量指标，若外部验证缺失标签需在脚注注明处理策略。


### 7. SUAVE 模型构建、调参与训练

**目的**：使用 Optuna 搜索的最优配置训练 SUAVE 主模型，并输出可复用的检查点与调参记录。
**结果解读**：验证集 ROAUC 与 TSTR/TRTR ΔAUC 指标用于选择最终 trial；训练日志与 manifest 确认 artefact 完整性。
**输入**：读取 `03_optuna_search/` 中的 `optuna_best_params_{label}.json`、`optuna_best_info_{label}.json` 与 `optuna_trials_{label}.csv`；若缺失则根据默认配置重新搜索。
**输出**：在 `04_suave_training/` 写入 `suave_best_{label}.pt`、`suave_model_manifest_{label}.json` 及 Optuna 图表。

1. 若存在历史最优 trial，优先读取 `optuna_best_params_{label}.json`；否则调用 `build_suave_model` 以默认超参初始化模型，并记录关键参数（latent_dim、beta、dropout 等）。该 JSON 现包含完整的帕累托前沿（`pareto_front`）及推荐 trial 编号（`preferred_trial_number`），脚本可按需选择不同 trial 的超参组合。
2. 训练顺序遵循脚本：先执行 VAE warm-up，再进行分类头独立训练，最后 joint fine-tuning，必要时启用分类损失权重自动调节。
3. 对于每个阶段，记录训练轮数、早停标准、最优模型路径以及 GPU/CPU 运行时长，用于技术报告的实验设置章节。
4. Optuna 搜索完成后需导出参数重要性、最优值收敛轨迹与多目标帕累托前沿图（均保存为 PNG/SVG/PDF/JPG），便于后续调参与审计复核；默认输出位于 `03_optuna_search/figures/`。
5. Trial 级别的搜索记录需写入 `optuna_trials_{label}.csv` 并在日志中展示前 10 个验证集 AUROC 最优的 trial，确保调参轨迹透明可追溯。帕累托前沿的完整元数据与指标同步保存于 `optuna_best_info_{label}.json`，供后续复核和多 trial 对比分析。
6. `research-mimic_mortality_supervised.py` 在交互模式下会读取 Optuna 帕累托前沿并列出各 trial 的验证集 AUROC、TSTR/TRTR ΔAUC 与本地模型保存状态，等待人工输入 trial ID 以加载或重新训练；脚本模式可通过 `--trial-id`（或位置参数）指定目标 trial，若未提供则优先加载最近一次保存的模型，缺失时再按照硬阈值（AUROC>0.81、|ΔAUC|<0.035）自动选取帕累托前沿解重训模型。
7. Optuna 优化脚本会在 `04_suave_training/` 下生成 `suave_model_manifest_{label}.json`，记录 trial 编号、目标函数值与模型/校准器路径；主流程在加载前需校验 manifest 所指向的 artefact 是否存在，不满足时回退至最近一次保存的权重或触发重新训练；当触发重新训练时会自动将新的 SUAVE 权重写入 `suave_best_{label}.pt` 以恢复后续运行的缓存链路。
8. 若 Optuna study 与最佳参数均缺失，可设置环境变量 `FORCE_UPDATE_SUAVE=1` 强制刷新本地备份模型；该开关仅在 Optuna artefact 不可用时生效，用于确保重新训练覆盖旧的 SUAVE 权重。

### 8. 分类/校准评估与不确定性量化（Bootstrap）

**目的**：量化 SUAVE 输出概率的可靠性，并汇总多样指标的置信区间与可视化，同时复核基线分类器表现。
**结果解读**：校准曲线平滑且 ECE 较低说明概率可信；bootstrap 置信区间用于评估模型稳定性及分类性能波动。
**输入**：读取 `04_suave_training/` 下的模型权重、`05_calibration_uncertainty/` 中已有的校准器（若存在）以及 `02_feature_engineering/` 下的特征缓存。SUAVE 与校准器直接使用 `prepare_features` 输出的原始特征；若需评估基线模型，则加载 `iterative_imputed_{dataset}_{label}.csv` 作为对照。
**输出**：在 `05_calibration_uncertainty/` 保存校准器与曲线，在 `06_evaluation_metrics/` 导出指标表与 Excel。

**目的**：量化 SUAVE 输出概率的可靠性，并汇总多样指标的置信区间与可视化。
**结果解读**：校准曲线平滑且 ECE 较低说明概率可信；bootstrap 置信区间用于评估模型稳定性。
**输入**：读取 `04_suave_training/` 下的模型权重及 `08_baseline_models/` 中的插补特征。
**输出**：在 `05_calibration_uncertainty/` 保存校准器与曲线，在 `06_evaluation_metrics/` 导出指标表与 Excel。

1. 通过 `fit_isotonic_calibrator` 在内部验证集上拟合专为 SUAVE 封装的等渗校准器（内部使用 `IsotonicRegression`，兼容缺失 `decision_function` 的估计器），必要时回退至逻辑回归温度缩放，并保存校准对象；若缓存缺失或校准器与模型不匹配，主流程会重新训练并自动序列化新的校准 artefact。
2. 使用 `evaluate_predictions` 对训练、验证、MIMIC-IV 测试与 eICU 集执行 bootstrap（默认 1000 次）以估计指标置信区间，并生成 Excel 汇总；除表格主列的 AUC、ACC、SPE、SEN、Brier 外，Excel 中还会给出 `accuracy`、`balanced_accuracy`、`f1_macro`、`recall_macro`、`specificity_macro`、`sensitivity_pos`、`specificity_pos`、`roc_auc`、`pr_auc` 及其置信区间，以支撑不同风险偏好的诊断分析。若需要并排比较基线分类器，可从迭代插补 CSV 读取概率输出或重新调用 `evaluate_transfer_baselines`。
3. 评估函数会同步导出 bootstrap 的原始采样记录：总体指标写入 `bootstrap_overall_records_{label}.csv`，逐类指标写入 `bootstrap_per_class_records_{label}.csv`，用于复核任意抽样迭代的轨迹。
4. 将生成的 `evaluation_metrics.csv` 与 Excel 汇总保存至 `06_evaluation_metrics/`，校准曲线及相关图表保存为 PNG/SVG/PDF/JPG 四种格式并写入 `05_calibration_uncertainty/`，同时在实验日志中登记路径，便于后续报告引用与审计复核。

### 9. 合成数据 - TSTR/TRTR

**目的**：评估 SUAVE 生成数据对下游监督任务的实用性，并对比真实数据训练的基线表现。
**结果解读**：重点关注 `roc_auc` 与 `accuracy` 的差异；若合成数据与真实数据性能接近，说明生成器具备迁移价值。
**输入**：使用 `build_tstr_training_sets` 生成的训练集缓存、`02_feature_engineering/` 中的迭代插补特征（`iterative_imputed_{dataset}_{label}.csv`）以及 `INCLUDE_SUAVE_TRANSFER` 环境变量。
**输出**：在 `09_tstr_trtr_transfer/` 缓存 `tstr_trtr_results_{label}.joblib`、`TSTR_TRTR_eval.xlsx` 与可视化结果。

1. 调用 `build_tstr_training_sets` 创建 `TRTR (real)`、`TSTR synthesis`、`TSTR synthesis-balance`、`TSTR synthesis-augment`、`TSTR synthesis-5x` 与 `TSTR synthesis-5x balance` 等方案，并在评估阶段对照 MIMIC-IV 测试集及（若标签可用）eICU 外部验证集。
2. 通过 `make_baseline_model_factories` 注册 `Logistic regression`、`Random forest` 与 `GBDT` 三类下游分类器，对每个训练方案分别拟合并在 `evaluate_transfer_baselines` 中统计 `accuracy` 与 `roc_auc`（含置信区间）。
3. 若需将 SUAVE 纳入迁移评估，可在运行脚本前设置 `INCLUDE_SUAVE_TRANSFER=1`，前提是 Optuna 已产出可用的最优超参；默认行为仅评估传统基线模型，以避免在 TSTR/TRTR 分析阶段重复拟合 SUAVE。
4. `evaluate_transfer_baselines` 的结果会缓存在 `09_tstr_trtr_transfer/tstr_trtr_results_{label}.joblib`；若需跳过缓存直接重训，可设置 `FORCE_UPDATE_TSTR_MODEL=1` 或 `FORCE_UPDATE_TRTR_MODEL=1`（默认均为 `1`，即始终更新）。缓存中还包含训练/模型顺序元数据，复用时需保持与现有目录结构兼容。
5. 运行结束后会额外导出 bootstrap 抽样记录，并统一写入 `09_tstr_trtr_transfer/TSTR_TRTR_eval.xlsx`：`summary` 工作表包含下游模型的汇总 AUC/accuracy 与置信区间，`plot_data` 存储绘图所需的长表结构，`bootstrap_overall` 与 `bootstrap_per_class` 分别保留整体与分层自助采样结果。
6. `plot_transfer_metric_bars` 负责绘制 TSTR/TRTR 方案在 `accuracy`、`roc_auc` 等指标上的分组柱状图（含置信区间），展示不同生成数据训练集对下游分类器的影响；其输出与分布漂移指标无直接对应关系，应单列在报告的“生成数据迁移性能”小节中说明。


### 10. 合成数据 - 分布漂移分析

**目的**：量化生成数据与真实数据之间的分布差异，定位潜在的失真特征。
**结果解读**：GBDT C2ST AUC 作为主要指标，若接近 0.5 表示难以区分；全局/逐列统计的 `p` 值指导进一步修正。
**输入**：复用 `09_tstr_trtr_transfer/` 中的训练数据拆分、`make_baseline_model_factories` 输出的下游模型以及 `suave.evaluate` 中的分布漂移函数。
**输出**：在 `10_distribution_shift/` 保存 `C2ST-distribution_shift.xlsx` 与 `metrics_distribution_shift.xlsx`，并在 `11_privacy_assessment/` 记录隐私攻击结果。

1. 分布漂移的主要结局指标采用 `classifier_two_sample_test`：以 `TRTR (real)` 与 `TSTR synthesis` 作为两类样本，复用 `make_baseline_model_factories` 中的 Logistic、Random Forest 与 GBDT pipeline（保持默认超参）。其中 **GBDT ROC-AUC** 为首要指标，逻辑回归与随机森林的 ROC-AUC 作为敏感性补充。
2. 次要结局指标包括全局 `rbf_mmd` 与 `energy_distance`，均基于置换检验给出 `p` 值，用于量化生成数据与真实分布之间的整体偏移程度。
3. `rbf_mmd`、`energy_distance` 与 `mutual_information_feature` 继续按特征逐列计算，定位非单调差异或潜在信息泄露风险的列，并与 C2ST 结果交叉验证。
4. 分布相似性相关的所有产物分布在两个工作簿中：`10_distribution_shift/C2ST-distribution_shift.xlsx` 汇总 C2ST 主指标（GBDT）与次要模型（Logistic、Random forest）的置信区间；`10_distribution_shift/metrics_distribution_shift.xlsx` 保存全局/逐列的 MMD、能量距离与互信息；配套的分布漂移可视化图表需在同一目录以 PNG/SVG/PDF/JPG 四种格式保存，纳入附录及复现包。
5. 在生成数据性能分析后运行 `simple_membership_inference`，补充隐私攻击基线并记录攻击 AUC/阈值，结果导出至 `11_privacy_assessment/membership_inference.xlsx`。


### 11. 潜空间可视化、报告生成与归档

**目的**：整合模型性能与潜空间解释结果，形成可交付的研究报告与可视化资产。
**结果解读**：潜空间图用于定性评估生成特征区分度，相关性图揭示潜变量与临床特征的联系。
**输入**：读取 `04_suave_training/`、`05_calibration_uncertainty/` 与 `06_evaluation_metrics/` 的模型与指标结果。
**输出**：在 `12_visualizations/` 保存图像，在输出根目录生成 `evaluation_summary_{label}.md` 与关联 CSV。

1. 借助 `plot_latent_space` 对潜空间执行 PCA/UMAP（视脚本配置）投影，输出训练、验证、测试与外部集的可视化比较，图像写入 `12_visualizations/`。
2. 使用 `dataframe_to_markdown`、`render_dataframe` 与 `write_results_to_excel_unique` 汇总评估结果，最终通过 `build_prediction_dataframe` 与 `evaluate_predictions` 生成 `evaluation_summary_{label}.md` 技术报告草稿，并在 `06_evaluation_metrics/` 同步保存 Excel/Markdown 版本。
3. 在归档目录保留所有原始数据引用、模型权重（`.pt`/`.joblib`）、插补缓存、图表、Markdown 报告及运行日志，以支持第三方审计与论文附录撰写。
4. 在潜空间解释章节调用 `suave.plots.compute_feature_latent_correlation` 计算潜空间与临床特征（含目标标签）的 Spearman 相关性，依托 statsmodels 进行 FDR/Bonferroni/Holm 校正。将 `compute_feature_latent_correlation` 的结果导出为相关系数/`p` 值 CSV，并以 `plot_feature_latent_outcome_path_graph` 绘制特征→潜变量→结局的多层次路径图（显著性水平 0.05，临床特征按定义的 group/color 映射着色），作为潜空间解释章节的核心可视化。`plot_feature_latent_correlation_heatmap(..., value="correlation")`、`plot_feature_latent_correlation_heatmap(..., value="pvalue")` 与 `plot_feature_latent_correlation_bubble` 作为补充结果，仍按 VAR_GROUP_DICT 分组生成热图与气泡图，用于量化隐空间与关键临床变量的相关性分布。
