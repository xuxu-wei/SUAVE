# 研究协议（Research Protocol）

本文档汇总 SUAVE 项目的研究流程约定，帮助在不同数据集上复现实验、整理结果并产出可审计的研究报告。

## MIMIC-III 住院死亡率（监督）分析方案

以下步骤基于 `examples/research-mimic_mortality_supervised.py`，该脚本整合 Optuna 搜索结果、经典基线、SUAVE 模型校准与合成数据评估，最终生成标准化的 Markdown 报告与配套制品。

### 准备阶段

1. 设定目标标签（默认 `in_hospital_mortality`）与输出目录，并记录 Optuna 试验的数量、超时时间和存储后端。
2. 确保 `examples/data/sepsis_mortality_dataset/` 下的训练、测试与外部验证（eICU）切分可用。
3. 若提供 Optuna SQLite 存储，脚本可直接回读最优试验与参数；否则将使用磁盘缓存或默认值重新训练。

### 数据加载与 Schema 校验

1. 读取 train/test/external TSV 数据框，校验目标列是否在 `TARGET_COLUMNS` 白名单中。
2. 调用 `define_schema(..., mode="interactive")` 生成初始 schema，并对 BMI、呼吸支持等级等存在歧义的列执行手动修正。
3. 使用 `schema_to_dataframe` 与 `render_dataframe` 输出列类型概览，便于人工复核。

### 数据集切分与特征工程

1. 通过 `prepare_features` 对训练样本构建特征矩阵，结合目标标签调用 `train_test_split`（分层抽样）划分内部验证集。
2. 对测试集与外部 eICU 数据应用相同的特征处理流程，确保列顺序与训练集一致。
3. 将结果缓存为 `baseline_feature_frames`，供后续基线模型与 SUAVE 共同复用。

### 基线模型基准

1. 借助 `load_or_create_iteratively_imputed_features` 完成各评估集的多重迭代插补，并缓存到磁盘以复用。
2. 训练并评估 Logistic Regression、KNN、Decision Tree、Random Forest 与 RBF-SVM 等 scikit-learn 管线，记录 AUC/ACC/SPE/SEN/Brier 指标。
3. 保存基线指标 CSV，并通过 `render_dataframe` 生成 Markdown 表格，必要时补充外部验证缺失标签的说明。

### SUAVE 模型与校准

1. 优先尝试加载磁盘上的最佳 SUAVE 模型与等渗（isotonic）校准器；若仅存在校准器则从中提取底层估计器。
2. 如均不可用且存在 Optuna 最优参数，则调用 `build_suave_model` 重建模型并依次执行 warm-up、head 与 joint fine-tuning 阶段。
3. 使用 `fit_isotonic_calibrator` 在验证集上拟合或更新校准器，确保预测概率可靠。

### 预测评估与置信区间

1. 对训练、验证、MIMIC 测试和（如适用）eICU 外部集计算校准后概率，生成校准曲线与 ROC/PR 基准图。
2. 调用 `evaluate_predictions` 进行自举（bootstrap）评估，输出整体与按类别的置信区间，并生成 Excel 汇总。
3. 运行 `simple_membership_inference` 记录隐私攻击基线，所有路径写入 `analysis_outputs_supervised/` 目录。

### 合成数据评估（TSTR/TRTR）

1. 针对目标标签构建真实与合成训练集（`build_tstr_training_sets`），必要时跳过非住院死亡率任务。
2. 组合 `make_baseline_model_factories` 产生统一的下游分类器，对真实与合成训练集分别进行训练/评估。
3. 汇总并渲染 TSTR/TRTR 指标、导出柱状图数据，同时计算 KS、RBF-MMD 与互信息衡量合成分布漂移。

### 潜空间可视化与报告生成

1. 使用 `plot_latent_space` 对各评估集的潜在表示执行 PCA 投影，检查类间可分性。
2. 汇总 schema、Optuna 最优结果、预测指标、引导自举摘要、TSTR/TRTR 与分布漂移产物，最终写入 `evaluation_summary_{label}.md`。
3. 保留所有 CSV/PNG/Markdown 路径，方便后续打包与论文附录引用。
