# 研究工作流模板

本目录汇总了 SUAVE 项目在 `examples/` 中沉淀的研究脚本，并将其整理为可复用的“模板工程”。通过集中化的配置文件，你只需调整少量硬编码常量即可在新的临床数据上复用整套流程。

## 目录概览

| 文件 | 作用 | 备注 |
| --- | --- | --- |
| `analysis_config.py` | 集中定义数据集路径、标签名称、特征分组、图形配色、Optuna 搜索阈值等常量。 | 迁移到新数据集时，仅需修改此文件。导入时会被其它模块读取并作为单一事实来源。 |
| `analysis_utils.py` | 研究流程的核心工具函数，涵盖 schema 校验、特征工程、模型训练、校准、评估与报告导出。 | 除非需要修改整体流程，否则保持逻辑不变；此文件从 `analysis_config.py` 读取全部配置。 |
| `cls_eval.py` | 分类评估与 Bootstrap 汇总工具。 | 依赖 `pandas`、`numpy`，以及 `openpyxl`/`xlsxwriter` 等表格写入库。 |
| `research-suave_optimize.py` | SUAVE 模型的 Optuna 调参入口。 | 读写最优 Trial、帕累托前沿与调参图表。 |
| `research-supervised_analysis.py` | 主分析脚本：加载 Artefact、执行校准与评估、生成总结报告。 | 支持 `--trial-id` 参数与 `FORCE_UPDATE_*` 环境变量。 |
| `datasets/` | 训练集、验证集、测试集、外部评估集等原始/加工数据的占位目录。 | 请放置与 `analysis_config.py` 中 schema 对应的 TSV/CSV 文件。 |

## 快速开始

1. **准备数据**
   - 将数据集复制或软链接到 `datasets/`。
   - 确认训练集、测试集及（如有）外部验证集的列名、数据类型、缺失值处理策略完全一致；只有 `BENCHMARK_COLUMNS` 中登记的临床评分允许在部分数据集中缺失。
   - 由于特征列是通过排除 `BENCHMARK_COLUMNS` 与 `TARGET_COLUMNS` 得到的，请确保数据文件中除目标列与临床评分外不包含额外信息，否则这些列会自动进入模型特征。
   - 若研究不涉及临床评分，请将 `analysis_config.py` 中的 `BENCHMARK_COLUMNS` 设为 `()`（空元组），并在报告中注明已跳过相关基准对照。

2. **定制配置**
   - 编辑 `analysis_config.py`，更新 `DATA_DIR`、`DATASET_FILENAMES`、标签名称、特征分组、输出目录、Optuna 搜索范围等。若研究不包含外部验证集，可删除 `DATASET_FILENAMES`、`BASELINE_DATASET_LABELS`、`BASELINE_DATASET_ORDER` 中的 `external_validation` 项。
   - 根据研究目标设置 `TARGET_LABEL`（实际建模用的标签）与 `TARGET_COLUMNS`（用于排除的所有目标变量集合），保持两者一致以避免误用。
   - 检查 `DEFAULT_ANALYSIS_CONFIG` 以同步存储路径、缓存目录与运行超参数。
   - 如需调整绘图主题或字体，可编辑 `PLOT_THEME`、`PLOT_LATIN_FONT_FAMILY`、`PLOT_CHINESE_FONT_FAMILY`；将 `PLOT_THEME` 设为 `None` 即可还原 Matplotlib 默认外观。

3. **安装依赖**
   - 按项目根目录的 `README.md` 或 `README-CN.md` 安装 SUAVE 及其可选依赖（Optuna、statsmodels、绘图后端等）。

4. **运行 Optuna 搜索**
   - 执行 `python research-suave_optimize.py` 生成帕累托前沿、最优 Trial JSON 与调参可视化。目录结构遵循 `analysis_config.py` 的 `ANALYSIS_SUBDIRECTORIES` 定义。

5. **执行主分析**
   - 运行 `python research-supervised_analysis.py [--trial-id N]` 以加载或训练目标模型、拟合校准器并完成下游评估。交互模式会提示选择 Trial，脚本模式可通过参数或环境变量控制缓存策略。
   - 若需刷新缓存，可按需设置 `FORCE_UPDATE_BENCHMARK_MODEL`、`FORCE_UPDATE_SYNTHETIC_DATA`、`FORCE_UPDATE_TSTR_MODEL`、`FORCE_UPDATE_TRTR_MODEL`、`FORCE_UPDATE_C2ST_MODEL`、`FORCE_UPDATE_DISTRIBUTION_SHIFT`。脚本模式下这些变量默认开启（值为 `1`）以保证批处理稳定性，交互模式默认关闭（值为 `0`）以节省时间。

6. **整理与归档**
   - 所有 Artefact 默认存储在 `resolve_analysis_output_root()` 指向的目录（通常为 `research_outputs_supervised/`）。请保留分阶段子目录以确保审计可追溯性。

## 使用注意事项

- **单一配置源**：务必将数据集路径、标签、特征分组等修改集中在 `analysis_config.py`。若在运行后更改配置，请清理相关缓存以避免 schema 不一致。
- **Schema 校验**：在执行耗时步骤前，先调用 `define_schema` 与 `schema_to_dataframe` 生成最新的 schema 记录。更新特征或列类型时需同步刷新。
- **缓存管理**：修改影响特征工程的设置后，删除 `02_feature_engineering/` 等目录中的旧缓存，防止旧特征沿用。
- **可选依赖**：潜空间图或路径图等高级可视化可能需要 `networkx`、`pygraphviz` 等额外库。根据需求安装或在脚本中禁用相关段落。
- **可重复性**：`RANDOM_STATE` 控制 Optuna、数据划分、基线模型等多个随机流程。若需更新，请在研究日志中记录原因与时间。
- **临床评分策略**：`CLINICAL_SCORE_BENCHMARK_STRATEGY="imputed"` 时会对 `BENCHMARK_COLUMNS` 进行迭代插补后再评估，设置为其他值则保持原始观测并跳过缺失样本；如需调整，请同步检查主流程与 `analysis_config.py` 中的注释。
- **⚠️ 深度修改提醒**：TSTR/TRTR 基线模型工厂、分布漂移评估函数与绘图参数主要位于 `research-supervised_analysis.py` 和 `analysis_utils.py` 中。若需调整，请先确认依赖关系并备份原始实现。

## 通用监督学习分析流程

以下步骤整合自原版研究协议，适用于任意结构化临床数据的监督学习研究。请将示例路径替换为你在 `analysis_config.py` 中配置的实际目录，并在每一步维护详尽的研究日志。

### 1. 研究目标与核心指标
- **目的**：明确研究问题、建模目标与评估维度，为实验提供统一的成效标准。
- **结果解读**：通常以 AUROC/AUPRC 衡量判别能力，结合 Brier Score、期望校准误差（ECE）与校准曲线评估概率可靠性。
- **输入**：目标标签枚举与评估配置，来自 `analysis_config.py` 或对应工具函数。
- **输出**：在研究日志中记录目标定义与评估方案，为后续报告撰写奠定基础。

### 2. 数据来源与管理
- **目的**：列出训练、验证、测试、外部评估等必要数据集，并规划 Artefact 的目录结构，确保研究可复现、可审计。
- **结果解读**：schema 校验与缺失值报告用于确认数据完整性；目录命名需与后续章节对应，方便追踪产出。
- **输入**：`datasets/` 下的 TSV/CSV 数据、schema 定义、必要的元数据。
- **输出**：在 `01_data_and_schema/` 记录数据加载日志、schema DataFrame 与 Markdown 摘要。
- **执行要点**：
  1. 确定训练集、内部验证集、测试集与外部验证集的来源及纳入标准，保持与配置文件一致。
  2. 保留原始列名与数据类型，必要时在日志中记录转换或派生字段。
  3. 按章节编号划分输出目录（默认 `01_data_and_schema/` → … → `12_privacy_assessment/`），其中 `09_interpretation/` 存放解释性 artefact，`10_tstr_trtr_transfer/`、`11_distribution_shift/` 与 `12_privacy_assessment/` 分别记录迁移、分布漂移与隐私分析。

### 3. 准备阶段
- **目的**：初始化实验配置、输出目录、环境变量，保证流程可重入。
- **结果解读**：通过目录与缓存检查确认是否可以复用历史 Artefact。
- **输入**：`analysis_config.py` 中的默认设置与命令行参数。
- **输出**：在研究日志中记录配置摘要与缓存状态。
- **执行要点**：
  1. 确认目标标签存在于配置的 `TARGET_COLUMNS`，并列出潜在的扩展标签。
  2. 自动创建输出目录与 Optuna 存储路径；必要时备份既有最优 Trial 信息。
  3. 若检测到已有模型或调参缓存，记录对应配置以便差异分析。

### 4. 数据加载与 Schema 校验
- **目的**：验证列名、类型、取值范围，为特征工程建立可信输入。
- **结果解读**：校验通过说明数据与预期一致；若发现冲突需在日志中说明并修复。
- **输入**：`01_data_and_schema/` 中的 TSV/CSV 或原始数据集。
- **输出**：`schema_{label}.xlsx`、`evaluation_datasets_{label}.joblib` 及相关可视化。
- **执行要点**：
  1. 使用 `load_dataset` 读取训练/验证/测试/外部集，确保列齐全。
  2. 调用 `define_schema(..., mode="interactive")` 生成 schema，必要时手动调整数值范围或类别映射。
  3. 借助 `schema_to_dataframe`、`render_dataframe` 导出列摘要，同时在 `01_data_and_schema/` 落盘 `schema_{label}.xlsx`。
  4. 将评估阶段实际使用的 `evaluation_datasets` 序列化为 `evaluation_datasets_{label}.joblib`，方便复核训练/验证/外部集的特征矩阵与标签。

### 5. 特征构建与内部验证划分
- **目的**：在统一流程下生成模型输入，并构建稳定的内部验证集。
- **结果解读**：成功生成特征缓存意味着后续模型可直接加载；若失败需回溯数据阶段。
- **输入**：`02_feature_engineering/` 中的缓存或 `prepare_features` 生成的新特征。
- **输出**：特征矩阵、验证划分日志、`baseline_feature_frames`。
- **执行要点**：
  1. 对训练集运行 `prepare_features`，并使用固定 `VALIDATION_SIZE`、`RANDOM_STATE` 进行分层划分。
  2. 对测试集与外部集复用同一转换逻辑，确保列顺序一致。
  3. 记录所有派生特征、缺失值处理策略与生成时间。

### 6. 基线模型与对照实验
- **目的**：构建与 SUAVE 独立的分类基线，用于衡量数据质量与合成数据贡献。
- **结果解读**：对比各基线的 AUC、准确率、Brier Score，以评估数据可用性。
- **输入**：迭代插补特征、基线模型工厂函数。
- **输出**：`08_baseline_models/` 下的 `baseline_estimators_{label}.joblib`、`baseline_models_{label}.csv`。
- **执行要点**：
  0. `BASELINE_DATASET_LABELS` 决定结果表中的数据集名称，`BASELINE_DATASET_ORDER` 控制输出顺序；如移除外部验证集，请同步修改两者及 `DATASET_FILENAMES`。
  1. 使用 `load_or_create_iteratively_imputed_features` 生成或复用插补特征，并记录缺失处理策略。
  2. 通过 `evaluate_transfer_baselines` 训练 Logistic 回归、随机森林、GBDT 等基线，统一统计指标。
  3. 若集成临床评分或专家基准，需注明数据来源与缺失处理方式。

### 7. SUAVE 模型构建、调参与训练
- **目的**：利用 Optuna 搜索结果训练最优 SUAVE 模型，并生成可复用的 Artefact。
- **结果解读**：验证集 AUROC 及迁移实验 ΔAUC 等指标用于选择最终模型。
- **输入**：`03_optuna_search/` 中的最优参数、trial CSV 与图表。
- **输出**：`04_suave_training/` 下的模型权重、manifest 与训练日志。
- **执行要点**：
  1. 若存在历史最优 Trial，优先加载对应 JSON；否则使用默认超参重新搜索。
  2. 记录每个训练阶段（预训练、分类头、联合微调）的轮数、早停标准与耗时。
  3. 导出参数重要性、收敛曲线、帕累托前沿图，并保存到 `03_optuna_search/figures/`。
  4. 模板会在 `04_suave_training/` 下生成 `manual_param_setting.py`，用于登记交互式手动调参的覆盖项；如需生效，请将 `build_analysis_config()` 返回的 `interactive_manual_tuning` 配置指向该模块并填写 `manual_param_setting` 字典。
     > 🟡 提醒：`analysis_config.INTERACTIVE_MANUAL_TUNING` 仅为手动覆写钩子，不建议在不了解字段作用时修改，以免干扰批量流程的默认行为。
  5. 交互式运行可输入 `manual` 直接加载 `suave_manual_manifest_{label}.json` 中登记的模型与校准器；命令行同样支持 `--trial-id manual`。未指定 trial 时脚本会优先检查手动 manifest，再回退至最近保存的自动 trial，最后依据帕累托阈值自动挑选候选。

### 8. 分类、校准与不确定性分析
- **目的**：量化模型概率输出的可靠性，并汇总各指标的置信区间与可视化。
- **结果解读**：平滑的校准曲线和较低的 ECE 表示概率可信；Bootstrap 区间衡量指标稳定性。
- **输入**：`04_suave_training/` 中的模型、`05_calibration_uncertainty/` 中的校准器、`02_feature_engineering/` 中的特征。
- **输出**：校准对象、曲线图、指标表格与 Excel 汇总。
- **执行要点**：
  1. 通过 `fit_isotonic_calibrator` 在内部验证集拟合校准器，必要时回退到温度缩放。
  2. 使用 `evaluate_predictions` 对所有数据集执行 Bootstrap，生成 CSV/Excel 以及抽样记录。
  3. `plot_benchmark_curves` 会为每个数据集分别写出 `benchmark_roc_{dataset}_{label}` 与 `benchmark_calibration_{dataset}_{label}`（PNG/SVG/PDF/JPG，默认位于 `06_evaluation_metrics/`），并统一应用 Seaborn `paper` 主题与 Times New Roman（含微软雅黑回退）字体，使 ROC/校准图保持 1:1 坐标比例。`plot_calibration_curves` 与基准校准图共享同一主题，纵轴标签更新为 “Observed probability”，坐标范围依据当前分箱概率自适应调整；如需恢复 Matplotlib 默认样式，可将 `DEFAULT_ANALYSIS_CONFIG["plot_theme"]` 或 `analysis_config.PLOT_THEME` 设为 `None`。

### 9. 潜空间相关性与解释
- **目的**：在执行迁移评估前审视 SUAVE 潜空间与临床特征、结局之间的耦合关系，为报告准备可追溯的解释性 artefact。
- **结果解读**：相关矩阵与 `p` 值识别潜变量与关键特征的关联强度，路径图揭示潜在因果结构，潜空间投影用于比较不同数据集的分布差异。
- **输入**：`VAR_GROUP_DICT` 定义的特征分组、训练集潜空间嵌入、`evaluation_datasets` 缓存。
- **输出**：`09_interpretation/` 下的 `latent_clinical_correlation_{label}` 系列 CSV/图像，以及 `latent_{label}.png` 潜空间投影。
- **执行要点**：
  1. 使用 `compute_feature_latent_correlation` 生成整体相关矩阵与 `p` 值，泡泡图/热图统一以相关系数着色（`plt.cm.RdBu_r`，0 为色谱中点），气泡大小按 `-log10(p)` 缩放并隐藏 `p≥0.1` 的关联，`p` 值热图在保持颜色的同时根据数值自动选择精度（0.049–0.051 与 0.001–0.01 区间保留三位小数，小于 0.001 显示 `<0.001`，大于 0.99 显示 `>0.99`）。轴标签继承 `PATH_GRAPH_NODE_DEFINITIONS` 的中文/LaTeX 标注，潜变量刻度渲染为 `$z_{n}$` 并水平放置，色条位于图像下方；所有图像以 PNG/JPG/SVG/PDF 四种格式写入 `09_interpretation/`。
  2. 依照 `VAR_GROUP_DICT` 分组重复相关性分析，若特征缺失脚本会打印 `Skipping unavailable variables` 以提醒补齐或记录。
  3. 调用 `plot_latent_space` 比较训练、验证、测试及外部验证集的潜空间分布，图像保存在 `latent_{label}.png`。

### 10. 合成数据 TSTR/TRTR 评估
- **目的**：评估生成数据对监督任务的迁移能力，与真实数据训练的基线做比较。
- **结果解读**：关注真实 vs. 合成训练的指标差异；差距越小，说明生成器迁移价值越高。
- **输入**：`build_tstr_training_sets` 生成的训练方案、迭代插补特征、基线模型工厂。
- **输出**：`10_tstr_trtr_transfer/` 下的结果缓存与 Excel/图表。
- **执行要点**：
  1. 按既定方案构建 `TRTR (real)`、`TSTR`、`TSTR balance`、`TSTR augment`、`TSTR 5x`、`TSTR 5x balance`、`TSTR 10x`、`TSTR 10x balance` 等训练集。
  2. 统一使用 `evaluate_transfer_baselines` 计算 Accuracy、ROC-AUC 及置信区间；默认仅使用 `analysis_config.TSTR_BASELINE_MODELS`（示例脚本为 `analysis_config["tstr_models"]`）列出的经典模型，当列表仅包含 1 个模型时，箱线图的横轴按训练数据集展开，若配置多个模型则横轴切换为模型名称、箱体按数据集着色。
  3. `plot_transfer_metric_boxes` 按 `analysis_config.TSTR_METRIC_LABELS` 设置纵轴标签，默认隐藏离群点并启用 0.1/0.05 的主次刻度；`plot_transfer_metric_bars` 额外绘制 Accuracy/AUROC 无误差棒条形图（纵轴固定 (0.5, 1)），同时输出 ΔAccuracy/ΔAUROC 箱线图便于对比。
  4. 需要纳入 SUAVE 迁移评估时，确保已有最优 Trial 并设置 `INCLUDE_SUAVE_TRANSFER=1`。
  5. 所有 TSTR/TRTR 图表默认沿用当前 Seaborn 主题的调色板；如需自定义配色，可在 `analysis_config.TRAINING_COLOR_PALETTE`（示例脚本为 `analysis_config["training_color_palette"]`）传入调色板名称或颜色序列，以保持不同环境下的颜色一致性。

### 11. 合成数据分布漂移分析
- **目的**：量化生成数据与真实数据的分布差异，定位潜在失真。
- **结果解读**：C2ST ROC-AUC 接近 0.5 表示难以区分；MMD、能量距离与互信息提供全局/逐列视角。
- **输入**：TSTR/TRTR 数据拆分、基线模型工厂、分布漂移评估函数。
- **输出**：`11_distribution_shift/` 下的 `c2st_metrics.xlsx`、`distribution_metrics.xlsx`（`overall` 与 `per_feature` 工作表尾部附有判读提示，内容与 `_interpret_global_shift` / `_interpret_feature_shift` 保持一致）以及相关图表。
- **执行要点**：
  1. 使用 `classifier_two_sample_test` 评估多种分类器的区分能力。
  2. 结合 `rbf_mmd`、`energy_distance`、`mutual_information_feature` 获取全局与逐列指标。
  3. 将所有结果导出为 Excel/图像，并在研究日志记录关键信息。

### 12. 报告生成与归档
- **目的**：整合模型性能、潜空间解释（参见第 9 节）与迁移评估结果，形成最终的 Markdown 报告与归档材料。
- **结果解读**：`evaluation_summary_{label}.md` 汇总最优 Trial、关键指标及主要 artefact 路径，方便撰写技术报告或提交审计。
- **输入**：`06_evaluation_metrics/` 指标表、`07_bootstrap_analysis/` 区间统计、`09_interpretation/` 解释性输出、`10_tstr_trtr_transfer/` 与 `11_distribution_shift/` 的迁移评估结果。
- **输出**：输出根目录下的 `evaluation_summary_{label}.md` 与关联 CSV/图像。
- **执行要点**：
  1. 使用 `dataframe_to_markdown`、`render_dataframe`、`write_results_to_excel_unique` 汇总评估指标，并在 `06_evaluation_metrics/` 保留 Excel/Markdown 副本。
  2. 执行脚本末尾的汇总逻辑，将 Optuna trial、校准曲线、潜空间解释 artefact 以及 TSTR/分布漂移路径写入 `evaluation_summary_{label}.md`。
  3. 在归档目录保留模型权重、插补缓存、解释性 CSV/图像、TSTR/TRTR 工作簿与运行日志，确保第三方复核可追溯。

该模板保持数据集无关性，只要在 `analysis_config.py` 中完成适配，即可在新的临床研究任务上复用完整流程。

## 预期输出

模板执行后会在 `resolve_analysis_output_root()` 指向的目录生成分阶段 Artefact。下表列出关键产物及缓存所包含的原始数据结构，便于在新数据集上复核。

| 阶段目录 | 主要输出产物 | 缓存数据结构说明 |
| --- | --- | --- |
| `01_data_and_schema/` | 原始数据快照（可选）<br>`schema_{label}.csv`、`schema_summary_{label}.md` | TSV/CSV 快照保持原始列名及数据类型；schema CSV 含 `Column`、`Type`、`n_classes`、`y_dim` 字段，Markdown 版本提供相同信息的表格文本。 |
| `02_feature_engineering/` | `*_features_{label}.parquet`、`baseline_feature_frames/`、`iterative_imputed_{dataset}_{label}.csv` | 特征帧以 `FEATURE_COLUMNS` 顺序保存数值化数据；`baseline_feature_frames/` 记录训练/验证划分索引；迭代插补 CSV 与原始特征列一致，仅值经过插补。 |
| `03_optuna_search/` | `optuna_trials_{label}.csv`、`optuna_best_info_{label}.json`、`optuna_best_params_{label}.json`、`figures/` | Trial CSV 汇总 `trial_number`、目标值、耗时等列；`optuna_best_info` JSON 存储 `preferred_trial_number`、`preferred_trial`（含 `values`、`params`、`validation_metrics`、`tstr_metrics`、`trtr_metrics`、`diagnostic_paths`）及 `pareto_front` 元数据；`optuna_best_params` JSON 汇集 `preferred_params` 与帕累托 trial 的参数字典。 |
| `04_suave_training/` | `suave_best_{label}.pt`、`suave_model_manifest_{label}.json`、训练日志 | 模型权重使用 PyTorch 序列化；manifest JSON 包含 `target_label`、`trial_number`、`values`、`params`、`model_path`、`calibrator_path`、`study_name`、`storage`、`saved_at`。 |
| `05_calibration_uncertainty/` | `isotonic_calibrator_{label}.joblib`、`calibration_curve_{dataset}_{label}.*` | Joblib 中保存拟合后的等渗/温度缩放对象及其内部状态；图像文件按数据集输出曲线，不附加脚本指示。 |
| `06_evaluation_metrics/` | `evaluation_metrics_{label}.csv`、`evaluation_metrics_{label}.xlsx`、`evaluation_summary_{label}.md` | 指标 CSV/Excel 覆盖训练、验证、测试、外部评估（如有）各 split；工作簿包含 `metrics`、长表及 bootstrap 明细；Markdown 摘要罗列关键文件路径。 |
| `07_bootstrap_analysis/` | `*_bootstrap.joblib` | 每个 joblib 为字典：`metadata` 记录训练/评估数据集、模型名、`bootstrap_n`、`prediction_signature`；`results` 提供 `overall`、`per_class`、`overall_records`、`per_class_records`、`bootstrap_overall_records`、`bootstrap_per_class_records`、`warnings` DataFrame。 |
| `08_baseline_models/` | `baseline_estimators_{label}.joblib`、`baseline_models_{label}.csv` | Joblib 字典的键为基线模型名称，值为已拟合 Pipeline；CSV 包含 `AUC`、`ACC`、`SPE`、`SEN`、`Brier` 等列及备注。 |
| `09_tstr_trtr_transfer/` | `training_sets/manifest_{label}.json` 与 TSV<br>`tstr_trtr_results_{label}.joblib`、`TSTR_TRTR_eval.xlsx`、`bootstrap_cache/` | Manifest JSON 记录 `target_label`、`feature_columns`、`datasets` 列表（名称+文件名）、可选 `random_state`、`generated_at`；TSTR/TRTR joblib 存储 `summary_df`、`plot_df`、`nested_results`、`bootstrap_df` 及训练/评估顺序、特征列、manifest 签名；Excel 汇总 `summary`、`metrics`、`bootstrap`、`tstr_summary`、`trtr_summary` 等工作表；`bootstrap_cache/` 条目结构同第 7 行。 |
| `10_distribution_shift/` | `c2st_metrics_{label}.joblib`、`c2st_metrics.xlsx`、`distribution_metrics_{label}.joblib`、`distribution_metrics.xlsx` | C2ST joblib 含 `feature_columns`、`model_order`、`metrics` 字典与 `results_df`；分布漂移 joblib 提供 `overall_df`、`per_feature_df`；对应 Excel 在 `overall`、`per_feature` 工作表末尾附解释文本。 |
| `11_privacy_assessment/` | `membership_inference.xlsx` | 工作簿包含 `summary`、`metrics`、`bootstrap` 工作表，记录攻击 AUC、阈值与抽样明细。 |
| `12_visualizations/` | 潜空间、指标曲线、箱线图等图像 | 图像按 PNG/SVG/PDF/JPG 等格式输出，文件名标记数据集与指标。 |
| 输出根目录 | `evaluation_summary_{label}.md`、日志 | Markdown 摘要列出各阶段 Artefact 路径与关键信息；日志文件记录运行时间与缓存命中状态。 |

## 缓存机制

### 缓存判定信息

- `07_bootstrap_analysis/`：主分析脚本会将每个“模型 × 数据集”的 bootstrap 结果保存为 `*_bootstrap.joblib`，其中包含总体/分层指标与抽样记录。命中缓存时直接读取；若 `FORCE_UPDATE_BOOTSTRAP=True` 则重新计算。
- `10_tstr_trtr_transfer/training_sets/`：`build_tstr_training_sets` 会生成 TSV 与 `manifest_{label}.json`，manifest 记录特征列、生成时间以及 SUAVE manifest 的 SHA256。若签名与当前配置不一致或启用了 `FORCE_UPDATE_SYNTHETIC_DATA`，训练集会被重建，后续依赖同一签名的缓存也会失效。
- `10_tstr_trtr_transfer/tstr_results_{label}.joblib`、`trtr_results_{label}.joblib`：存储真实/合成训练下的基线预测结果与指标，并携带 `training_manifest_signature`、`data_generator_signature` 等元数据。只有当签名匹配且未启用 `FORCE_UPDATE_TSTR_MODEL` / `FORCE_UPDATE_TRTR_MODEL` 时才会复用。
- `10_tstr_trtr_transfer/bootstrap_cache/`：`evaluate_transfer_baselines` 在完成一次 bootstrap 后立即写入缓存，校验字段包括 `training_manifest_signature`、`data_generator_signature`、`prediction_signature` 与 `bootstrap_n`。当预测发生变化或启用 `FORCE_UPDATE_TSTR_BOOTSTRAP`、`FORCE_UPDATE_TRTR_BOOTSTRAP` 时会重新采样。
- `11_distribution_shift/`：两类缓存分别存放在 `c2st_metrics_{label}.joblib` 与 `distribution_metrics_{label}.joblib` 中，记录特征列、模型顺序及统计结果。若配置改变或设置了 `FORCE_UPDATE_C2ST_MODEL`、`FORCE_UPDATE_DISTRIBUTION_SHIFT`，脚本会放弃缓存并重新计算。
- SUAVE 生成器 artefact：默认读取 `04_suave_training/` 下的 `suave_best_{label}.pt` 与 manifest。当需要覆盖旧模型时，可启用 `FORCE_UPDATE_SUAVE` 强制重新训练（前提是 Optuna artefact 不可用或显式请求刷新）。

### FORCE_UPDATE 参数对照

| 参数 | 控制内容与关联缓存 |
| --- | --- |
| `FORCE_UPDATE_BENCHMARK_MODEL` | 覆盖 `08_baseline_models/` 下的 `baseline_estimators_{label}.joblib` 与相关指标，确保传统基线与最新特征一致。 |
| `FORCE_UPDATE_BOOTSTRAP` | 忽略 `07_bootstrap_analysis/` 中的缓存，重新执行 SUAVE 与基线的 bootstrap 评估。 |
| `FORCE_UPDATE_SYNTHETIC_DATA` | 重新生成合成训练 TSV 与 manifest，并使依赖 `training_manifest_signature` 的缓存全部失效。 |
| `FORCE_UPDATE_TSTR_MODEL` | 重新拟合 TSTR 基线模型并覆盖 `tstr_results_{label}.joblib`。 |
| `FORCE_UPDATE_TRTR_MODEL` | 重新拟合 TRTR 基线模型并覆盖 `trtr_results_{label}.joblib`。 |
| `FORCE_UPDATE_TSTR_BOOTSTRAP` | 禁用 `bootstrap_cache/` 中与 TSTR 相关的缓存条目，基于最新预测重新生成 bootstrap 明细。 |
| `FORCE_UPDATE_TRTR_BOOTSTRAP` | 禁用 `bootstrap_cache/` 中与 TRTR 相关的缓存条目，确保真实训练结果的 bootstrap 指标更新。 |
| `FORCE_UPDATE_C2ST_MODEL` | 跳过 `c2st_metrics_{label}.joblib` 缓存，重新训练 C2ST 分类器并输出最新统计。 |
| `FORCE_UPDATE_DISTRIBUTION_SHIFT` | 重新计算全局与逐特征的分布漂移指标，覆盖 `distribution_metrics_{label}.joblib`。 |
| `FORCE_UPDATE_SUAVE` | 当 Optuna 产物缺失或需要替换生成器时，强制放弃已有 `suave_best_{label}.pt`，触发重新训练。 |

默认开关由脚本顶部或 `FORCE_UPDATE_FLAG_DEFAULTS` 控制：批处理流程通常将耗时步骤设为 `True` 以确保输出最新；交互式分析则倾向复用缓存以节省时间。调整参数时请在研究日志中记录原因与时间，便于后续审计与复现。

## 预期输出

下表罗列模板主流程在各阶段产生的核心 artefact、缓存位置及人工复现方法。在代码示例中，请先通过 `resolve_analysis_output_root(DEFAULT_ANALYSIS_CONFIG["output_dir_name"])` 计算得到 `OUTPUT_DIR`。

| 分析流程 | 输出产物名称 | 类型（报表、图像） | 描述 | 原始数据 | 缓存的原始数据 | 缓存数据结构说明 |
| --- | --- | --- | --- | --- | --- | --- |
| 8. 分类/校准评估与不确定性量化（Bootstrap） | Benchmark ROC曲线（逐数据集） | 图像 | 每个数据集写出 `benchmark_roc_{dataset}_{label}`，比较 SUAVE 与经典基线的 ROC 表现，图像统一使用 Seaborn `paper` 主题并保持 1:1 坐标比例 | 各数据集的预测概率与标签映射（`probability_map`、`baseline_probability_map`、`label_map`） | `OUTPUT_DIR / "01_data_and_schema" / f"evaluation_datasets_{label}.joblib"`<br>`OUTPUT_DIR / "05_calibration_uncertainty" / f"isotonic_calibrator_{label}.joblib"`<br>`OUTPUT_DIR / "08_baseline_models" / f"baseline_estimators_{label}.joblib"` | <pre><code class="language-python">from pathlib import Path
import joblib

from analysis_config import DEFAULT_ANALYSIS_CONFIG
from analysis_utils import resolve_analysis_output_root

label = "in_hospital_mortality"
output_root = resolve_analysis_output_root(DEFAULT_ANALYSIS_CONFIG["output_dir_name"])
cache_path = output_root / "01_data_and_schema" / f"evaluation_datasets_{label}.joblib"
payload = joblib.load(cache_path)
datasets = payload["datasets"]
calibrator = joblib.load(output_root / "05_calibration_uncertainty" / f"isotonic_calibrator_{label}.joblib")
baselines = joblib.load(output_root / "08_baseline_models" / f"baseline_estimators_{label}.joblib")
for name, (features, labels) in datasets.items():
    suave_probs = calibrator.predict_proba(features)
    print(name, suave_probs.shape, labels.shape)
</code></pre> |
| 8. 分类/校准评估与不确定性量化（Bootstrap） | 校准曲线（逐数据集） | 图像 | `plot_calibration_curves` 生成的图像和 `benchmark_calibration_{dataset}_{label}` 采用相同主题与 1:1 坐标比例，纵轴标签为 “Observed probability”，坐标范围依据分箱概率自适应调整 | 经过校准的预测概率与真实标签（`probability_map`、`label_map`） | `OUTPUT_DIR / "01_data_and_schema" / f"evaluation_datasets_{label}.joblib"`<br>`OUTPUT_DIR / "05_calibration_uncertainty" / f"isotonic_calibrator_{label}.joblib"` | <pre><code class="language-python">from pathlib import Path
import joblib
import numpy as np

from analysis_config import DEFAULT_ANALYSIS_CONFIG
from analysis_utils import resolve_analysis_output_root

label = "in_hospital_mortality"
output_root = resolve_analysis_output_root(DEFAULT_ANALYSIS_CONFIG["output_dir_name"])
payload = joblib.load(output_root / "01_data_and_schema" / f"evaluation_datasets_{label}.joblib")
datasets = payload["datasets"]
calibrator = joblib.load(output_root / "05_calibration_uncertainty" / f"isotonic_calibrator_{label}.joblib")
probability_map = {name: calibrator.predict_proba(features) for name, (features, _) in datasets.items()}
label_map = {name: np.asarray(labels) for name, (_, labels) in datasets.items()}
print(probability_map.keys(), label_map["Train"].shape)
</code></pre> |
| 8. 分类/校准评估与不确定性量化（Bootstrap） | bootstrap benchmark excel报表 | 报表 | 汇总各模型在 Train/Validation/MIMIC/eICU 的 bootstrap 置信区间、原始记录与告警 | `evaluate_predictions` 生成的 bootstrap 结果字典（`overall`、`per_class`、`bootstrap_*_records`） | `OUTPUT_DIR / "07_bootstrap_analysis" / "SUAVE"` 下的 `*_bootstrap.joblib` | <pre><code class="language-python">from pathlib import Path
import joblib

from analysis_config import DEFAULT_ANALYSIS_CONFIG
from analysis_utils import resolve_analysis_output_root

label = "in_hospital_mortality"
output_root = resolve_analysis_output_root(DEFAULT_ANALYSIS_CONFIG["output_dir_name"])
cache_dir = output_root / "07_bootstrap_analysis" / "SUAVE"
for cache_path in sorted(cache_dir.glob("*_bootstrap.joblib")):
    payload = joblib.load(cache_path)
    print(cache_path.name, payload.keys())
</code></pre> |
| 10. 合成数据 - TSTR/TRTR | TSTR/TRTR箱线图 | 图像 | `plot_transfer_metric_boxes` 生成的 Accuracy/AUROC 与 ΔAccuracy/ΔAUROC 箱线图；单模型时按训练数据集排布，多模型时横轴展示模型、箱体按数据集着色 | TSTR/TRTR bootstrap 明细表（`combined_bootstrap_df`、`delta_bootstrap_df`） | `OUTPUT_DIR / "10_tstr_trtr_transfer" / f"tstr_results_{label}.joblib"`<br>`OUTPUT_DIR / "10_tstr_trtr_transfer" / f"trtr_results_{label}.joblib"` | <pre><code class="language-python">from pathlib import Path
import joblib

from analysis_config import DEFAULT_ANALYSIS_CONFIG
from analysis_utils import resolve_analysis_output_root

label = "in_hospital_mortality"
output_root = resolve_analysis_output_root(DEFAULT_ANALYSIS_CONFIG["output_dir_name"])
tstr_payload = joblib.load(output_root / "10_tstr_trtr_transfer" / f"tstr_results_{label}.joblib")
trtr_payload = joblib.load(output_root / "10_tstr_trtr_transfer" / f"trtr_results_{label}.joblib")
for name, payload in {"tstr": tstr_payload, "trtr": trtr_payload}.items():
    bootstrap_df = payload.get("bootstrap_df")
    if bootstrap_df is not None:
        print(name, bootstrap_df.head())
</code></pre> |
| 10. 合成数据 - TSTR/TRTR | TSTR/TRTR条形图 | 图像 | `plot_transfer_metric_bars` 生成的 Accuracy/AUROC 无误差棒条形图，纵轴固定在 (0.5, 1)，便于比较各训练方案的绝对表现 | TSTR/TRTR 指标摘要表（`combined_summary_df`） | `OUTPUT_DIR / "10_tstr_trtr_transfer" / f"tstr_results_{label}.joblib"`<br>`OUTPUT_DIR / "10_tstr_trtr_transfer" / f"trtr_results_{label}.joblib"` | <pre><code class="language-python">from pathlib import Path
import joblib

from analysis_config import DEFAULT_ANALYSIS_CONFIG
from analysis_utils import resolve_analysis_output_root

label = "in_hospital_mortality"
output_root = resolve_analysis_output_root(DEFAULT_ANALYSIS_CONFIG["output_dir_name"])
summary_df = joblib.load(output_root / "10_tstr_trtr_transfer" / f"tstr_results_{label}.joblib").get("summary_df")
print(summary_df[["training_dataset", "model", "accuracy", "roc_auc"]].head())
</code></pre> |
| 10. 合成数据 - TSTR/TRTR | TSTR_TRTR_eval报表 | 报表 | `TSTR_TRTR_eval.xlsx` 汇总 TSTR/TRTR 指标长表、图表输入与 bootstrap（含 `bootstrap_delta`）记录 | TSTR/TRTR 评估结果（`summary_df`、`plot_df`、`bootstrap_df`、`nested_results`） | `OUTPUT_DIR / "10_tstr_trtr_transfer" / f"tstr_results_{label}.joblib"`<br>`OUTPUT_DIR / "10_tstr_trtr_transfer" / f"trtr_results_{label}.joblib"` | <pre><code class="language-python">from pathlib import Path
import joblib

from analysis_config import DEFAULT_ANALYSIS_CONFIG
from analysis_utils import resolve_analysis_output_root

label = "in_hospital_mortality"
output_root = resolve_analysis_output_root(DEFAULT_ANALYSIS_CONFIG["output_dir_name"])
for stem in ("tstr_results", "trtr_results"):
    payload = joblib.load(output_root / "10_tstr_trtr_transfer" / f"{stem}_{label}.joblib")
    print(stem, payload.keys())
</code></pre> |
| 11. 合成数据 - 分布漂移分析 | c2st_metrics.xlsx报表 | 报表 | 记录 C2ST 分类器在真实 vs 合成特征上的 AUC 及置信区间 | C2ST 统计与明细（`metrics`、`results_df`） | `OUTPUT_DIR / "11_distribution_shift" / f"c2st_metrics_{label}.joblib"` | <pre><code class="language-python">from pathlib import Path
import joblib

from analysis_config import DEFAULT_ANALYSIS_CONFIG
from analysis_utils import resolve_analysis_output_root

label = "in_hospital_mortality"
output_root = resolve_analysis_output_root(DEFAULT_ANALYSIS_CONFIG["output_dir_name"])
payload = joblib.load(output_root / "11_distribution_shift" / f"c2st_metrics_{label}.joblib")
print(payload.keys())
print(payload["results_df"].head())
</code></pre> |
| 11. 合成数据 - 分布漂移分析 | distribution_metrics.xlsx报表 | 报表 | 汇总全局/逐特征的 MMD、能量距离、互信息统计及判读备注 | 分布漂移结果（`overall_df`、`per_feature_df`） | `OUTPUT_DIR / "11_distribution_shift" / f"distribution_metrics_{label}.joblib"` | <pre><code class="language-python">from pathlib import Path
import joblib

from analysis_config import DEFAULT_ANALYSIS_CONFIG
from analysis_utils import resolve_analysis_output_root

label = "in_hospital_mortality"
output_root = resolve_analysis_output_root(DEFAULT_ANALYSIS_CONFIG["output_dir_name"])
payload = joblib.load(output_root / "11_distribution_shift" / f"distribution_metrics_{label}.joblib")
print(payload["overall_df"].head())
print(payload["per_feature_df"].head())
</code></pre> |
| 11. 合成数据 - 分布漂移分析 | membership_inference.xlsx报表 | 报表 | 基于 SUAVE 训练/测试概率对比的成员推断基线指标 | 训练/测试概率向量与标签（`probability_map["Train"]`、`probability_map["MIMIC test"]`、`y_train_model`、`y_test`） | `OUTPUT_DIR / "01_data_and_schema" / f"evaluation_datasets_{label}.joblib"`<br>`OUTPUT_DIR / "05_calibration_uncertainty" / f"isotonic_calibrator_{label}.joblib"` | <pre><code class="language-python">from pathlib import Path
import joblib

from analysis_config import DEFAULT_ANALYSIS_CONFIG
from analysis_utils import resolve_analysis_output_root

label = "in_hospital_mortality"
output_root = resolve_analysis_output_root(DEFAULT_ANALYSIS_CONFIG["output_dir_name"])
payload = joblib.load(output_root / "01_data_and_schema" / f"evaluation_datasets_{label}.joblib")
datasets = payload["datasets"]
calibrator = joblib.load(output_root / "05_calibration_uncertainty" / f"isotonic_calibrator_{label}.joblib")
train_probs = calibrator.predict_proba(datasets["Train"][0])
test_probs = calibrator.predict_proba(datasets["MIMIC test"][0])
print(train_probs.shape, test_probs.shape)
</code></pre> |
| 9. 潜空间相关性与解释 | 潜空间投影比较图 | 图像 | `plot_latent_space` 输出的 SUAVE 潜空间可视化（PCA/UMAP 对比） | 各评估数据集的潜空间输入特征与标签字典（`latent_features`、`latent_labels`） | `OUTPUT_DIR / "01_data_and_schema" / f"evaluation_datasets_{label}.joblib"`<br>`OUTPUT_DIR / "04_suave_training" / f"suave_model_manifest_{label}.json"` | <pre><code class="language-python">from pathlib import Path
import joblib, json

from analysis_config import DEFAULT_ANALYSIS_CONFIG
from analysis_utils import resolve_analysis_output_root

label = "in_hospital_mortality"
output_root = resolve_analysis_output_root(DEFAULT_ANALYSIS_CONFIG["output_dir_name"])
manifest = json.loads((output_root / "04_suave_training" / f"suave_model_manifest_{label}.json").read_text())
print(manifest.keys())
payload = joblib.load(output_root / "01_data_and_schema" / f"evaluation_datasets_{label}.joblib")
print(payload["datasets"].keys())
</code></pre> |
| 9. 潜空间相关性与解释 | 特征-预测目标-潜空间相关性气泡图 | 图像 | `plot_feature_latent_correlation_bubble` 绘制的总体相关性气泡图，颜色表示相关系数（RdBu_r，0 为中点），气泡大小按 `-log10(p)` 缩放并隐藏 `p≥0.1` 的关联，特征/结局标签来自 `PATH_GRAPH_NODE_DEFINITIONS`，潜变量刻度渲染为 `$z_{n}$`；图像输出 PNG/JPG/SVG/PDF 四种格式 | 潜变量-特征-结局的相关矩阵与显著性矩阵（`overall_corr`、`overall_pvals`） | `OUTPUT_DIR / "09_interpretation" / f"latent_clinical_correlation_{label}_correlations.csv"`<br>`OUTPUT_DIR / "09_interpretation" / f"latent_clinical_correlation_{label}_pvalues.csv"` | <pre><code class="language-python">from pathlib import Path
import pandas as pd

from analysis_config import DEFAULT_ANALYSIS_CONFIG
from analysis_utils import resolve_analysis_output_root

label = "in_hospital_mortality"
output_root = resolve_analysis_output_root(DEFAULT_ANALYSIS_CONFIG["output_dir_name"])
corr_path = output_root / "09_interpretation" / f"latent_clinical_correlation_{label}_correlations.csv"
pval_path = output_root / "09_interpretation" / f"latent_clinical_correlation_{label}_pvalues.csv"
print(pd.read_csv(corr_path, index_col=0).head())
print(pd.read_csv(pval_path, index_col=0).head())
</code></pre> |
| 9. 潜空间相关性与解释 | 特征→潜变量→结局的多层次路径图 | 图像 | `plot_feature_latent_outcome_path_graph` 生成的多层次路径网络图 | SUAVE 模型与训练特征/标签（`model`、`X_train_model`、`y_train_model`） | `OUTPUT_DIR / "01_data_and_schema" / f"evaluation_datasets_{label}.joblib"`<br>`OUTPUT_DIR / "04_suave_training" / f"suave_model_manifest_{label}.json"` | <pre><code class="language-python">from pathlib import Path
import joblib, json

from analysis_config import DEFAULT_ANALYSIS_CONFIG
from analysis_utils import resolve_analysis_output_root

label = "in_hospital_mortality"
output_root = resolve_analysis_output_root(DEFAULT_ANALYSIS_CONFIG["output_dir_name"])
payload = joblib.load(output_root / "01_data_and_schema" / f"evaluation_datasets_{label}.joblib")
train_features, train_labels = payload["datasets"]["Train"]
print(train_features.shape, train_labels.shape)
manifest = json.loads((output_root / "04_suave_training" / f"suave_model_manifest_{label}.json").read_text())
print(manifest["model_path"])
</code></pre> |
