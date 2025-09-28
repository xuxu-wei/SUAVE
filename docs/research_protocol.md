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
5. `examples/research_outputs_supervised/` 采用与章节编号一致的分阶段目录结构：`01_data_and_schema/`、`02_feature_engineering/`、`03_optuna_search/`、`04_suave_training/`、`05_calibration_uncertainty/`、`06_evaluation_metrics/`、`07_bootstrap_analysis/`、`08_baseline_models/`、`09_interpretation/`、`10_tstr_trtr_transfer/`、`11_distribution_shift/` 与 `12_privacy_assessment/`。各阶段产物应写入对应子目录，便于审计检索。

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
**输出**：在 `01_data_and_schema/` 中保存 schema DataFrame 工作簿与可选截图。

1. 调用 `load_dataset` 读取 MIMIC-IV 训练/测试及 eICU 外部验证集的 TSV，确保无缺失列并保留原始数据类型。
2. 通过 `define_schema(train_df, FEATURE_COLUMNS, mode="interactive")` 生成初始 schema；对 BMI、呼吸支持等级（`Respiratory_Support`）和淋巴细胞百分比（`LYM%`）执行人工修正，以保证类型一致性。
 3. 使用 `schema_to_dataframe` 与 `render_dataframe` 输出列类型摘要，并在 `01_data_and_schema/` 写入 `schema_{label}.xlsx`，保证审计时可直接查看结构化描述。
 4. 将 `evaluation_datasets` 字典缓存为 `evaluation_datasets_{label}.joblib`，其中包含模型评估阶段实际使用的特征矩阵与标签；如需手动加载，可执行：
    <pre><code class="language-python">import joblib
from pathlib import Path

label = "in_hospital_mortality"  # 根据当前任务调整标签
schema_root = Path("examples/research_outputs_supervised/01_data_and_schema")
cache_path = schema_root / f"evaluation_datasets_{label}.joblib"
payload = joblib.load(cache_path)
datasets = payload["datasets"]
train_features, train_labels = datasets["Train"]
print(train_features.head())
print(train_labels.head())</code></pre>

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
3. 脚本在 `08_baseline_models/` 下缓存拟合后的 `baseline_estimators_{label}.joblib`，二次运行时默认复用；若需强制重新训练，可在执行前将脚本顶部的 `FORCE_UPDATE_BENCHMARK_MODEL` 设为 `True`，或在 `analysis_config.py` / `mimic_mortality_utils.py` 中调整 `FORCE_UPDATE_FLAG_DEFAULTS`。交互式运行默认保留缓存，命令行批处理则按照该字典的设定决定是否覆盖。SUAVE 主模型及其校准器始终使用 `prepare_features` 产出的原始特征（即未经过该迭代插补管线），保持与生成器训练阶段的输入一致。
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
6. `research-mimic_mortality_supervised.py` 在交互模式下会读取 Optuna 帕累托前沿并列出各 trial 的验证集 AUROC、TSTR/TRTR ΔAUC 与本地模型保存状态：
 > 等待人工输入 trial ID 以加载或重新训练，输入manual可加载手动调参模型；
 > 脚本模式可通过 `--trial-id`（或位置参数）指定目标 trial, --trial-id manual 会加载手动调参模型；
 > 若未提供参数，脚本会先检查 `suave_manual_manifest_{label}.json` 是否存在手动覆盖，其次复用最近一次保存的自动 trial，仍缺失时再按照硬阈值（AUROC>0.81、|ΔAUC|<0.035）自动选取帕累托前沿解重训模型。
7. 在创建 `04_suave_training/` 目录时，脚本会自动生成（或补全）`manual_param_setting.py` 占位文件，并写入 `manual_param_setting: dict = {}` 默认体以便登记人工覆写的超参；若启用 `build_analysis_config()` 返回的 `interactive_manual_tuning` 配置，可在该字典中填入新的学习率、权重衰减等局部参数并即时生效。若模块文件或 `manual_param_setting` 属性缺失，脚本会直接报错并终止运行，提示补全手动覆写脚本。
8. Optuna 优化脚本会在 `04_suave_training/` 下生成 `suave_model_manifest_{label}.json`，记录 trial 编号、目标函数值与模型/校准器路径；主流程在加载前需校验 manifest 所指向的 artefact 是否存在，不满足时回退至最近一次保存的权重或触发重新训练；当触发重新训练时会自动将新的 SUAVE 权重写入 `suave_best_{label}.pt` 以恢复后续运行的缓存链路。若以手动模式覆写模型与校准器，脚本也会同步生成/更新 `suave_manual_manifest_{label}.json`，确保后续运行能优先加载指定的手动 artefact。启用 `interactive_manual_tuning` 后，交互式运行优化脚本会在执行 Optuna 之前展示手动模型及既有 Pareto trial 摘要，并接受 `y/yes`（按当前覆写配置直接训练并登记手动模型）、`manual`（跳过搜索复用磁盘上的手动 artefact）或 `n/no`/回车（继续自动搜索）的输入；若在提示过程中触发键盘中断，脚本会提示确认是否直接回退至 Optuna 搜索。最新流程在渲染该 Pareto 摘要前即复制当前的手动 manifest，使交互分支被跳过时也能沿用最新的手动配置。
9. 手动覆写与 Optuna trial 的指标计算均调用 `examples/mimic_mortality_utils.py`（研究模板对应 `research_template/analysis_utils.py`）中的 `evaluate_candidate_model_performance`，该函数统一完成验证集指标、TSTR/TRTR 评估与 ΔAUC 计算。当需要调整模型评估逻辑时，只需修改这一函数即可同时覆盖手动与自动流程。
10. 若 Optuna study 与最佳参数均缺失，可设置环境变量 `FORCE_UPDATE_SUAVE=1` 强制刷新本地备份模型；该开关仅在 Optuna artefact 不可用时生效，用于确保重新训练覆盖旧的 SUAVE 权重。

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
4. 将生成的 `evaluation_metrics.csv` 与 Excel 汇总保存至 `06_evaluation_metrics/`，并在实验日志中登记路径，便于后续报告引用与审计复核。`plot_benchmark_curves` 会针对每个数据集分别输出 `benchmark_roc_{dataset}_{label}` 与 `benchmark_calibration_{dataset}_{label}` 图像（PNG/SVG/PDF/JPG 四种格式），统一写入 `06_evaluation_metrics/`。图像默认采用 Seaborn `paper` 主题与 Times New Roman 字体（含微软雅黑回退），ROC 与校准图保持 1:1 坐标比例。若需恢复 Matplotlib 默认主题，可在 `DEFAULT_ANALYSIS_CONFIG["plot_theme"]`（模板项目请修改 `analysis_config.PLOT_THEME`）设为 `None`。校准曲线的纵轴标签更新为 “Observed probability”，横纵坐标范围依据当前分箱的预测概率自适应扩展，避免固定在 `(0, 1)` 导致的留白。

### 9. 潜空间相关性与解释

**目的**：在迁移评估前审查 SUAVE 潜空间与临床特征、目标标签之间的耦合关系，沉淀可追踪的解释性 artefact。
**结果解读**：相关系数及 `p` 值用于定位潜变量与关键临床变量的关联强度，路径图提供多层次因果假设，潜空间投影帮助识别不同数据集在生成特征空间的分布差异。
**输入**：训练集潜空间嵌入、`VAR_GROUP_DICT` 中的特征分组映射、评估阶段构建的 `evaluation_datasets` 缓存。
**输出**：在 `09_interpretation/` 写入 `latent_clinical_correlation_{label}` 前缀的 CSV/图像，以及 `latent_{label}.png` 投影图。

1. 使用 `compute_feature_latent_correlation` 生成整体相关矩阵与 `p` 值，并导出 `*_correlations.csv`、`*_pvalues.csv`。泡泡图与热图的着色统一由相关系数驱动（`plt.cm.RdBu_r`，以 0 为色谱中点），气泡半径按 `-log10(p)` 线性缩放且仅保留 `p<0.1` 的关联，`p` 值热图在颜色保持相关系数的同时采用动态精度显示显著性（0.049–0.051 与 0.001–0.01 区间保留三位小数，小于 0.001 显示 `<0.001`，大于 0.99 显示 `>0.99`）。轴标签继承 `PATH_GRAPH_NODE_DEFINITIONS` 的中文/LaTeX 标注，潜变量刻度使用 `$z_{n}$` 并水平置于刻度正上方，色条统一置于图像下方。所有图像以 PNG、JPG、SVG、PDF 四种格式写入 `09_interpretation/`。
2. 依照 `VAR_GROUP_DICT` 为每个临床分组重复相关性分析，输出分组级别的 CSV 与配套图像；若特征缺失，脚本会打印 `Skipping unavailable variables` 以提醒补齐或记录。
3. 调用 `plot_latent_space` 将训练、验证、测试与 eICU 数据集的潜空间嵌入投影到统一图像，留存于 `latent_{label}.png` 以支持质性审查。

### 10. 合成数据 - TSTR/TRTR

**目的**：评估 SUAVE 生成数据对下游监督任务的实用性，并对比真实数据训练的基线表现。
**结果解读**：重点关注 `roc_auc` 与 `accuracy` 的差异；若合成数据与真实数据性能接近，说明生成器具备迁移价值。
**输入**：使用 `build_tstr_training_sets` 生成的训练集缓存、`02_feature_engineering/` 中的迭代插补特征（`iterative_imputed_{dataset}_{label}.csv`）以及 `INCLUDE_SUAVE_TRANSFER` 环境变量。
**输出**：在 `10_tstr_trtr_transfer/` 缓存 `tstr_trtr_results_{label}.joblib`、`TSTR_TRTR_eval.xlsx` 与可视化结果。

1. 调用 `build_tstr_training_sets` 创建 `TRTR (real)`、`TSTR`、`TSTR balance`、`TSTR augment`、`TSTR 5x`、`TSTR 5x balance`、`TSTR 10x` 与 `TSTR 10x balance` 等方案，并在评估阶段对照 MIMIC-IV 测试集及（若标签可用）eICU 外部验证集。
2. 通过 `make_baseline_model_factories` 注册 `Logistic regression`、`Random forest` 与 `GBDT` 三类下游分类器，对每个训练方案分别拟合并在 `evaluate_transfer_baselines` 中统计 `accuracy` 与 `roc_auc`（含置信区间）。
3. 若需将 SUAVE 纳入迁移评估，可在运行脚本前设置 `INCLUDE_SUAVE_TRANSFER=1`，前提是 Optuna 已产出可用的最优超参；默认行为仅评估 `analysis_config["tstr_models"]`（模板脚本同名配置项）列出的传统基线，以避免在 TSTR/TRTR 分析阶段重复拟合 SUAVE。配置项允许用户通过元组挑选参与 TSTR 的模型；当仅包含 1 个模型时，箱线图横轴展示训练数据集，若配置多个模型则横轴切换为模型名称、箱体按数据集着色。
4. `build_tstr_training_sets` 会在 `10_tstr_trtr_transfer/training_sets/` 生成 TSV + JSON manifest（`manifest_{label}.json`），复用 SUAVE 采样得到的各类训练集；`evaluate_transfer_baselines` 则分别缓存 `tstr_results_{label}.joblib` 与 `trtr_results_{label}.joblib`，避免重复训练下游基线。脚本模式下默认强制重算（`FORCE_UPDATE_SYNTHETIC_DATA=1`、`FORCE_UPDATE_TSTR_MODEL=1`、`FORCE_UPDATE_TRTR_MODEL=1`），交互模式默认复用缓存（上述变量默认为 `0`）。如需刷新单独阶段，可将 `FORCE_UPDATE_SYNTHETIC_DATA` 置为 `1` 以重建 TSV 并自动失效旧的 TSTR/TRTR 结果，或分别设置 `FORCE_UPDATE_TSTR_MODEL`、`FORCE_UPDATE_TRTR_MODEL` 以重新拟合下游模型。

5. 运行结束后会额外导出 bootstrap 抽样记录，并统一写入 `10_tstr_trtr_transfer/TSTR_TRTR_eval.xlsx`：`summary` 汇总各模型在所有训练方案下的 Accuracy/AUROC 与置信区间，`metrics` 保留长表结构以便绘图，`bootstrap` 集成原始抽样指标与 ΔAccuracy/ΔAUROC 长表（相对 `TRTR (real)` 的差值），`bootstrap_delta` 独立存放差值明细，`tstr_summary`/`trtr_summary` 拆分合成与真实训练集的结果，`bootstrap_overall` 与 `bootstrap_per_class` 则保留整体与分层自助采样明细。
6. `plot_transfer_metric_boxes` 在生成 Accuracy/AUROC/ΔAccuracy/ΔAUROC 箱线图时会按 `analysis_config["tstr_metric_labels"]`（模板项目改写 `analysis_config.TSTR_METRIC_LABELS`）设置纵轴标签，默认启用 0.1 间隔的主刻度、0.05 的次刻度并隐藏离群点；新增的 `plot_transfer_metric_bars` 则绘制无误差棒的绝对指标条形图，纵轴固定在 (0.5, 1)。
6. `plot_transfer_metric_boxes` 会针对每个评估数据集绘制箱线图：横轴按模型分组，箱体颜色区分训练数据来源，当仅评估单个模型时横轴改为数据集标签。箱线图基于 bootstrap 样本的分布，更直观地展示生成数据对下游性能的影响；其输出与分布漂移指标无直接对应关系，应单列在报告的“生成数据迁移性能”小节中说明。

7. TSTR/TRTR 可视化默认沿用当前 Seaborn 主题的调色板，脚本会根据训练数据集数量自动循环颜色；若需自定义配色，可在 `analysis_config["training_color_palette"]`（模板项目修改 `analysis_config.TRAINING_COLOR_PALETTE`）传入调色板名称或颜色序列，便于在不同环境中保持一致的图例颜色。


### 10. 合成数据 - 分布漂移分析

**目的**：量化生成数据与真实数据之间的分布差异，定位潜在的失真特征。
**结果解读**：GBDT C2ST AUC 作为主要指标，若接近 0.5 表示难以区分；全局/逐列统计的 `p` 值指导进一步修正。
**输入**：复用 `10_tstr_trtr_transfer/` 中的训练数据拆分、`make_baseline_model_factories` 输出的下游模型以及 `suave.evaluate` 中的分布漂移函数。
**输出**：在 `11_distribution_shift/` 保存 `c2st_metrics.xlsx` 与 `distribution_metrics.xlsx`，并在 `12_privacy_assessment/` 记录隐私攻击结果。

1. 分布漂移的主要结局指标采用 `classifier_two_sample_test`：以 `TRTR (real)` 与 `TSTR` 作为两类样本，复用 `make_baseline_model_factories` 中的 Logistic、Random Forest 与 GBDT pipeline（保持默认超参）。其中 **GBDT ROC-AUC** 为首要指标，逻辑回归与随机森林的 ROC-AUC 作为敏感性补充。分析过程中会显示针对模型/特征的进度条，便于追踪耗时步骤。
2. 次要结局指标包括全局 `rbf_mmd` 与 `energy_distance`，均基于置换检验给出 `p` 值，用于量化生成数据与真实分布之间的整体偏移程度。
3. `rbf_mmd`、`energy_distance` 与 `mutual_information_feature` 继续按特征逐列计算，定位非单调差异或潜在信息泄露风险的列，并与 C2ST 结果交叉验证。
4. 分布相似性相关的所有产物分布在两个工作簿中：`11_distribution_shift/c2st_metrics.xlsx` 汇总所有模型的 C2ST ROC-AUC、置信区间、bootstrap 次数以及真实/合成样本量；`11_distribution_shift/distribution_metrics.xlsx` 的 `overall` 工作表报告全局 MMD、能量距离与互信息（附解释列），`per_feature` 工作表逐列列出 `rbf_mmd`、能量距离、互信息与对应解读。每个工作表尾部额外空一行，并写入指标判读提示（p 值阈值与特征级启发式），与 `analysis_utils._interpret_global_shift` / `_interpret_feature_shift` 中的逻辑保持一致。配套的分布漂移可视化图表需在同一目录以 PNG/SVG/PDF/JPG 四种格式保存，纳入附录及复现包。脚本模式默认重算 C2ST 和漂移统计（`FORCE_UPDATE_C2ST_MODEL=1`、`FORCE_UPDATE_DISTRIBUTION_SHIFT=1`），交互模式默认复用缓存（上述变量默认为 `0`）；如需刷新结果，可显式设置相应环境变量。
5. 在生成数据性能分析后运行 `simple_membership_inference`，补充隐私攻击基线并记录攻击 AUC/阈值，结果导出至 `12_privacy_assessment/membership_inference.xlsx`。


### 12. 报告生成与归档

**目的**：整合模型性能、潜空间解释（参见第 9 节）与迁移评估结果，生成可交付的 Markdown 报告与归档材料。
**结果解读**：`evaluation_summary_{label}.md` 汇总最优 Trial、指标表、解释性 artefact 路径与 TSTR/分布漂移结论，为技术报告或审计提供统一入口。
**输入**：`06_evaluation_metrics/` 中的指标表、`07_bootstrap_analysis/` 的区间统计、`09_interpretation/` 的解释性 artefact、`10_tstr_trtr_transfer/` 与 `11_distribution_shift/` 的迁移分析结果。
**输出**：输出根目录下的 `evaluation_summary_{label}.md` 与关联 CSV/图像；根据需要将最终图表整理至 `reports/` 或论文附录。

1. 调用 `dataframe_to_markdown`、`render_dataframe` 与 `write_results_to_excel_unique` 汇总评估结果，并在 `06_evaluation_metrics/` 保留 Excel/Markdown 版本。
2. 执行脚本结尾的汇总逻辑，将 Optuna trial 信息、校准曲线、潜空间解释 artefact（`09_interpretation/`）以及 TSTR/分布漂移输出路径逐条写入 `evaluation_summary_{label}.md`。
3. 在归档目录保留所有原始数据引用、模型权重（`.pt`/`.joblib`）、插补缓存、解释性 CSV/图像、TSTR/TRTR 工作簿以及运行日志，确保第三方复核可追溯。

## 预期输出

下表汇总各阶段在 `examples/research_outputs_supervised/` 下的主要产物，并记录缓存文件所保存的原始数据结构，便于审计时核对。

| 阶段目录 | 主要输出产物 | 缓存数据结构说明 |
| --- | --- | --- |
| `01_data_and_schema/` | `mimic-mortality-*.tsv` 原始快照<br>`schema_{label}.csv`、`schema_summary_{label}.md` | TSV 文件保留原始列名及字符串/数值类型，按行顺序写入；schema CSV 含 `Column`、`Type`、`n_classes`、`y_dim` 四列描述字段属性，对应 Markdown 版本提供同样字段的表格文本。 |
| `02_feature_engineering/` | `train_features_{label}.parquet`、`validation_features_{label}.parquet` 等特征帧<br>`baseline_feature_frames/` 与 `iterative_imputed_{dataset}_{label}.csv` | 特征帧以列名等于 `FEATURE_COLUMNS` 的 DataFrame 形式存储（Parquet/CSV 中保留浮点与类别编码）；`baseline_feature_frames/` 目录内的缓存按数据集划分子文件，记录分层划分后的索引列表；`iterative_imputed_*.csv` 继承原始特征列顺序并包含插补后的数值。 |
| `03_optuna_search/` | `optuna_trials_{label}.csv`、`optuna_best_info_{label}.json`、`optuna_best_params_{label}.json`、`figures/` | CSV 汇总 trial 指标列（`trial_number`、`values[0]`、`values[1]`、`fit_seconds` 等）；`optuna_best_info` JSON 记录 `preferred_trial_number`、`preferred_trial`（含 `values`、`params`、`validation_metrics`、`tstr_metrics`、`trtr_metrics`、`diagnostic_paths`）及 `pareto_front` 列表；`optuna_best_params` JSON 聚合 `preferred_params` 与帕累托前沿各 trial 的参数字典。图像目录保存 Pareto、收敛曲线等静态文件。 |
| `04_suave_training/` | `suave_best_{label}.pt`、`suave_model_manifest_{label}.json`、训练日志 | 模型权重以 PyTorch 序列化格式存储；manifest JSON 含 `target_label`、`trial_number`、`values`、`params`、`model_path`、`calibrator_path`、`study_name`、`storage` 与 `saved_at` 字段，用于指向对应 artefact。 |
| `05_calibration_uncertainty/` | `isotonic_calibrator_{label}.joblib`、`calibration_curve_{dataset}_{label}.png/svg` | 校准器 joblib 保存拟合后的等渗回归对象及其内部参数；配套图像按照数据集名称输出曲线文件，不含额外脚本元信息。 |
| `06_evaluation_metrics/` | `evaluation_metrics_{label}.csv`、`evaluation_metrics_{label}.xlsx`、`evaluation_summary_{label}.md` | CSV/Excel 均包含训练、验证、测试、外部验证四类数据集的指标列，工作簿含 `metrics` 主表及长表/抽样附表；Markdown 摘要列出对应文件路径与关键指标。 |
| `07_bootstrap_analysis/` | `*_bootstrap.joblib` | 每个 joblib 载荷均为包含 `metadata` 与 `results` 的字典；`results` 内含 `overall`、`per_class`、`overall_records`、`per_class_records`、`bootstrap_overall_records`、`bootstrap_per_class_records`、`warnings` DataFrame，`metadata` 记录 `training_dataset`、`evaluation_dataset`、`model`、`bootstrap_n`、`prediction_signature` 等字段。 |
| `08_baseline_models/` | `baseline_estimators_{label}.joblib`、`baseline_models_{label}.csv` | 模型缓存以字典形式保存 Pipeline 对象，键为基线模型名称；指标 CSV 列出各模型在不同数据集上的 `AUC`、`ACC`、`SPE`、`SEN`、`Brier` 与备注列。 |
| `09_tstr_trtr_transfer/` | `training_sets/manifest_{label}.json` 与对应 TSV<br>`tstr_trtr_results_{label}.joblib`、`TSTR_TRTR_eval.xlsx`、`bootstrap_cache/` | Manifest JSON 含 `target_label`、`feature_columns`、`datasets`（名称与文件名映射）、可选 `random_state` 与时间戳；TSTR/TRTR joblib 存储 `summary_df`、`plot_df`、`nested_results`、`bootstrap_df` 及训练/评估顺序、特征列、训练 manifest 签名；Excel 汇总 `summary`、`metrics`、`bootstrap`、`tstr_summary`、`trtr_summary` 等工作表；`bootstrap_cache/` 目录内条目沿用 07 节描述的结构。 |
| `10_distribution_shift/` | `c2st_metrics_{label}.joblib`、`c2st_metrics.xlsx`、`distribution_metrics_{label}.joblib`、`distribution_metrics.xlsx` | C2ST joblib 包含 `feature_columns`、`model_order`、`metrics`（按模型存储 ROC-AUC 及区间）与 `results_df`；对应 Excel 提供 `metrics` 工作表。分布漂移 joblib 保存 `overall_df` 与 `per_feature_df` DataFrame；Excel 的 `overall`、`per_feature` 工作表在表尾追加解释性注记。 |
| `11_privacy_assessment/` | `membership_inference.xlsx` | 工作簿含 `summary`、`metrics`、`bootstrap` 等工作表，记录攻击 AUC、阈值及抽样明细。 |
| `12_visualizations/` | 潜空间投影、基线曲线、箱线图等图像 | 图像以 PNG/SVG/PDF/JPG 四种格式落盘，文件名包含数据集与指标信息，便于报告引用。 |
| 输出根目录 | `evaluation_summary_{label}.md`、运行日志 | Markdown 摘要罗列各阶段 artefact 路径与关键结论；日志文件记录脚本执行时间、环境变量与缓存命中信息。 |

## 缓存机制

### 缓存判定信息

- `07_bootstrap_analysis/`：针对 SUAVE 主模型与临床基线的分类/校准评估，`evaluate_predictions` 会按“模型 × 数据集”落盘 `*_bootstrap.joblib`，其中包含 `overall`、`per_class` 等 DataFrame。命中缓存时直接读取，避免重复触发 bootstrap 进度条；若设置 `FORCE_UPDATE_BOOTSTRAP=True` 则忽略缓存重新计算。
- `10_tstr_trtr_transfer/training_sets/`：`build_tstr_training_sets` 保存 `manifest_{label}.json` 与对应的 TSV，manifest 内记录特征列、生成时间与 SUAVE manifest 的 SHA256。脚本会校验 manifest 与当前配置/生成器签名一致，若 `FORCE_UPDATE_SYNTHETIC_DATA=True` 或签名不匹配则重建训练集并刷新后续缓存。
- `10_tstr_trtr_transfer/tstr_results_{label}.joblib`、`trtr_results_{label}.joblib`：存储真实/合成训练下的基线模型预测表与指标。缓存载荷包含 `training_manifest_signature` 与当前 SUAVE manifest 的 SHA256（`data_generator_signature`），命中缓存需与最新训练集一致；可通过 `FORCE_UPDATE_TSTR_MODEL`、`FORCE_UPDATE_TRTR_MODEL` 控制重新拟合。
- `10_tstr_trtr_transfer/bootstrap_cache/`：`evaluate_transfer_baselines` 针对每个“训练方案 × 评估集 × 模型”单独缓存 bootstrap 明细。校验字段包含 `training_manifest_signature`、`data_generator_signature`、`prediction_signature`（基于概率与预测的哈希）及 `bootstrap_n`。任一信息变化或显式启用 `FORCE_UPDATE_TSTR_BOOTSTRAP` / `FORCE_UPDATE_TRTR_BOOTSTRAP` 时会重新执行 bootstrap。
- `11_distribution_shift/`：`classifier_two_sample_test` 与分布漂移统计分别写入 `c2st_metrics_{label}.joblib` 与 `distribution_metrics_{label}.joblib`。缓存载荷记录特征列顺序、模型顺序及相关指标，若检测到配置变化或设置了 `FORCE_UPDATE_C2ST_MODEL`、`FORCE_UPDATE_DISTRIBUTION_SHIFT`，则重新计算。
- SUAVE 模型目录（`04_suave_training/`）与 Optuna artefact：默认读取已有的 `suave_best_{label}.pt` 与 manifest；当设置 `FORCE_UPDATE_SUAVE=True` 且 Optuna 产物缺失时，会跳过缓存并强制重新训练模型。

### FORCE_UPDATE 参数对照

| 参数 | 控制内容与关联缓存 |
| --- | --- |
| `FORCE_UPDATE_BENCHMARK_MODEL` | 重新训练并覆盖 `08_baseline_models/` 下的 `baseline_estimators_{label}.joblib` 及基线指标表，避免沿用旧的临床/传统模型。 |
| `FORCE_UPDATE_BOOTSTRAP` | 忽略 `07_bootstrap_analysis/` 中的 `*_bootstrap.joblib`，强制重新计算 SUAVE 与基线的分类/校准 bootstrap 指标。 |
| `FORCE_UPDATE_SYNTHETIC_DATA` | 重建 `10_tstr_trtr_transfer/training_sets/` 的 TSV 与 manifest，同步失效依赖该签名的 TSTR/TRTR 结果与 bootstrap 缓存。 |
| `FORCE_UPDATE_TSTR_MODEL` | 重新拟合并写入 `tstr_results_{label}.joblib`，常用于生成器或训练集更新后刷新合成数据基线。 |
| `FORCE_UPDATE_TRTR_MODEL` | 重新拟合并写入 `trtr_results_{label}.joblib`，确保真实数据训练的对照基线与最新特征/标签一致。 |
| `FORCE_UPDATE_TSTR_BOOTSTRAP` | 对 TSTR 结果禁用 `bootstrap_cache/` 复用，依据最新预测重新计算并缓存 bootstrap 明细。 |
| `FORCE_UPDATE_TRTR_BOOTSTRAP` | 对 TRTR 结果禁用 `bootstrap_cache/` 复用，确保真实训练的 bootstrap 指标与当前预测一致。 |
| `FORCE_UPDATE_C2ST_MODEL` | 忽略 `c2st_metrics_{label}.joblib`，重新训练 C2ST 分类器并生成分布漂移 Excel。 |
| `FORCE_UPDATE_DISTRIBUTION_SHIFT` | 强制刷新 `distribution_metrics_{label}.joblib` 及相关 Excel，重新评估全局与逐特征分布差异。 |
| `FORCE_UPDATE_SUAVE` | 在 Optuna 产物缺失或需替换生成器时，指示脚本放弃加载已有 `suave_best_{label}.pt`，触发模型重训。 |

上述参数的默认值由脚本顶部或 `FORCE_UPDATE_FLAG_DEFAULTS` 决定：批处理模式通常将耗时阶段设为 `True` 以确保结果最新，而交互式探索默认复用缓存以缩短迭代时间。根据研究需要调整开关时，请同步记录触发原因与时间以便审计追踪。

## 预期输出

| 分析流程 | 输出产物名称 | 类型（报表、图像） | 描述 | 原始数据 | 缓存的原始数据 | 缓存数据结构说明 |
| --- | --- | --- | --- | --- | --- | --- |
| 8. 分类/校准评估与不确定性量化（Bootstrap） | Benchmark ROC曲线（逐数据集） | 图像 | 每个数据集写出 `benchmark_roc_{dataset}_{label}`，比较 SUAVE 与经典基线的 ROC 表现，图像统一使用 Seaborn `paper` 主题并保持 1:1 坐标比例 | 各数据集的预测概率与标签映射（`probability_map`、`baseline_probability_map`、`label_map`） | `examples/research_outputs_supervised/01_data_and_schema/evaluation_datasets_in_hospital_mortality.joblib`<br>`examples/research_outputs_supervised/05_calibration_uncertainty/isotonic_calibrator_in_hospital_mortality.joblib`<br>`examples/research_outputs_supervised/08_baseline_models/baseline_estimators_in_hospital_mortality.joblib` | <pre><code class="language-python">from pathlib import Path
import joblib

payload = joblib.load(Path("examples/research_outputs_supervised/01_data_and_schema/evaluation_datasets_in_hospital_mortality.joblib"))
datasets = payload["datasets"]
calibrator = joblib.load(Path("examples/research_outputs_supervised/05_calibration_uncertainty/isotonic_calibrator_in_hospital_mortality.joblib"))
baselines = joblib.load(Path("examples/research_outputs_supervised/08_baseline_models/baseline_estimators_in_hospital_mortality.joblib"))
for name, (features, labels) in datasets.items():
    suave_probs = calibrator.predict_proba(features)
    print(name, suave_probs.shape, labels.shape)
</code></pre> |
| 8. 分类/校准评估与不确定性量化（Bootstrap） | 校准曲线（逐数据集） | 图像 | `plot_calibration_curves` 生成的图像和 `benchmark_calibration_{dataset}_{label}` 采用相同主题与 1:1 坐标比例，纵轴标签为 “Observed probability”，坐标范围依据分箱概率自适应调整 | 经过校准的预测概率与真实标签（`probability_map`、`label_map`） | `examples/research_outputs_supervised/01_data_and_schema/evaluation_datasets_in_hospital_mortality.joblib`<br>`examples/research_outputs_supervised/05_calibration_uncertainty/isotonic_calibrator_in_hospital_mortality.joblib` | <pre><code class="language-python">from pathlib import Path
import joblib
import numpy as np

payload = joblib.load(Path("examples/research_outputs_supervised/01_data_and_schema/evaluation_datasets_in_hospital_mortality.joblib"))
datasets = payload["datasets"]
calibrator = joblib.load(Path("examples/research_outputs_supervised/05_calibration_uncertainty/isotonic_calibrator_in_hospital_mortality.joblib"))
probability_map = {name: calibrator.predict_proba(features) for name, (features, _) in datasets.items()}
label_map = {name: np.asarray(labels) for name, (_, labels) in datasets.items()}
print(probability_map.keys(), label_map["Train"].shape)
</code></pre> |
| 8. 分类/校准评估与不确定性量化（Bootstrap） | bootstrap benchmark excel报表 | 报表 | 汇总各模型在 Train/Validation/MIMIC/eICU 的 bootstrap 置信区间、原始记录与告警 | `evaluate_predictions` 生成的 bootstrap 结果字典（`overall`、`per_class`、`bootstrap_*_records`） | `examples/research_outputs_supervised/07_bootstrap_analysis/SUAVE/`*`*_bootstrap.joblib` | <pre><code class="language-python">from pathlib import Path
import joblib

cache_dir = Path("examples/research_outputs_supervised/07_bootstrap_analysis/SUAVE")
for cache_path in sorted(cache_dir.glob("*_bootstrap.joblib")):
    payload = joblib.load(cache_path)
    print(cache_path.name, payload.keys())
</code></pre> |
| 10. 合成数据 - TSTR/TRTR | TSTR/TRTR箱线图 | 图像 | `plot_transfer_metric_boxes` 生成的 Accuracy/AUROC 与 ΔAccuracy/ΔAUROC 箱线图；单模型时按训练数据集排布，多模型时横轴展示模型、箱体按数据集着色 | TSTR/TRTR bootstrap 明细表（`combined_bootstrap_df`、`delta_bootstrap_df`） | `examples/research_outputs_supervised/10_tstr_trtr_transfer/tstr_results_in_hospital_mortality.joblib`<br>`examples/research_outputs_supervised/10_tstr_trtr_transfer/trtr_results_in_hospital_mortality.joblib` | <pre><code class="language-python">from pathlib import Path
import joblib

tstr_payload = joblib.load(Path("examples/research_outputs_supervised/10_tstr_trtr_transfer/tstr_results_in_hospital_mortality.joblib"))
tstr_bootstrap = tstr_payload.get("bootstrap_df")
if tstr_bootstrap is not None:
    print(tstr_bootstrap.head())
trtr_payload = joblib.load(Path("examples/research_outputs_supervised/10_tstr_trtr_transfer/trtr_results_in_hospital_mortality.joblib"))
trtr_bootstrap = trtr_payload.get("bootstrap_df")
if trtr_bootstrap is not None:
    print(trtr_bootstrap.head())
</code></pre> |
| 10. 合成数据 - TSTR/TRTR | TSTR/TRTR条形图 | 图像 | `plot_transfer_metric_bars` 生成的 Accuracy/AUROC 无误差棒条形图，纵轴固定在 (0.5, 1)，便于比较各训练方案的绝对表现 | TSTR/TRTR 指标摘要表（`combined_summary_df`） | `examples/research_outputs_supervised/10_tstr_trtr_transfer/tstr_results_in_hospital_mortality.joblib`<br>`examples/research_outputs_supervised/10_tstr_trtr_transfer/trtr_results_in_hospital_mortality.joblib` | <pre><code class="language-python">from pathlib import Path
import joblib

payload = joblib.load(Path("examples/research_outputs_supervised/10_tstr_trtr_transfer/tstr_results_in_hospital_mortality.joblib"))
summary_df = payload.get("summary_df")
print(summary_df[["training_dataset", "model", "accuracy", "roc_auc"]].head())
</code></pre> |
| 10. 合成数据 - TSTR/TRTR | TSTR_TRTR_eval报表 | 报表 | `TSTR_TRTR_eval.xlsx` 汇总 TSTR/TRTR 指标长表、图表输入与 bootstrap（含 `bootstrap_delta`）记录 | TSTR/TRTR 评估结果（`summary_df`、`plot_df`、`bootstrap_df`、`nested_results`） | `examples/research_outputs_supervised/10_tstr_trtr_transfer/tstr_results_in_hospital_mortality.joblib`<br>`examples/research_outputs_supervised/10_tstr_trtr_transfer/trtr_results_in_hospital_mortality.joblib` | <pre><code class="language-python">from pathlib import Path
import joblib

for path in [
    Path("examples/research_outputs_supervised/10_tstr_trtr_transfer/tstr_results_in_hospital_mortality.joblib"),
    Path("examples/research_outputs_supervised/10_tstr_trtr_transfer/trtr_results_in_hospital_mortality.joblib"),
]:
    payload = joblib.load(path)
    print(path.name, payload.keys())
</code></pre> |
| 11. 合成数据 - 分布漂移分析 | c2st_metrics.xlsx报表 | 报表 | 记录 C2ST 分类器在真实 vs 合成特征上的 AUC 及置信区间 | C2ST 统计与明细（`metrics`、`results_df`） | `examples/research_outputs_supervised/11_distribution_shift/c2st_metrics_in_hospital_mortality.joblib` | <pre><code class="language-python">from pathlib import Path
import joblib

payload = joblib.load(Path("examples/research_outputs_supervised/11_distribution_shift/c2st_metrics_in_hospital_mortality.joblib"))
print(payload.keys())
print(payload["results_df"].head())
</code></pre> |
| 11. 合成数据 - 分布漂移分析 | distribution_metrics.xlsx报表 | 报表 | 汇总全局/逐特征的 MMD、能量距离、互信息统计及判读备注 | 分布漂移结果（`overall_df`、`per_feature_df`） | `examples/research_outputs_supervised/11_distribution_shift/distribution_metrics_in_hospital_mortality.joblib` | <pre><code class="language-python">from pathlib import Path
import joblib

payload = joblib.load(Path("examples/research_outputs_supervised/11_distribution_shift/distribution_metrics_in_hospital_mortality.joblib"))
print(payload["overall_df"].head())
print(payload["per_feature_df"].head())
</code></pre> |
| 11. 合成数据 - 分布漂移分析 | membership_inference.xlsx报表 | 报表 | 基于 SUAVE 训练/测试概率对比的成员推断基线指标 | 训练/测试概率向量与标签（`probability_map["Train"]`、`probability_map["MIMIC test"]`、`y_train_model`、`y_test`） | `examples/research_outputs_supervised/01_data_and_schema/evaluation_datasets_in_hospital_mortality.joblib`<br>`examples/research_outputs_supervised/05_calibration_uncertainty/isotonic_calibrator_in_hospital_mortality.joblib` | <pre><code class="language-python">from pathlib import Path
import joblib

payload = joblib.load(Path("examples/research_outputs_supervised/01_data_and_schema/evaluation_datasets_in_hospital_mortality.joblib"))
datasets = payload["datasets"]
calibrator = joblib.load(Path("examples/research_outputs_supervised/05_calibration_uncertainty/isotonic_calibrator_in_hospital_mortality.joblib"))
train_probs = calibrator.predict_proba(datasets["Train"][0])
test_probs = calibrator.predict_proba(datasets["MIMIC test"][0])
print(train_probs.shape, test_probs.shape)
</code></pre> |
| 9. 潜空间相关性与解释 | 潜空间投影比较图 | 图像 | `plot_latent_space` 输出的 SUAVE 潜空间可视化（PCA/UMAP 对比） | 各评估数据集的潜空间输入特征与标签字典（`latent_features`、`latent_labels`） | `examples/research_outputs_supervised/01_data_and_schema/evaluation_datasets_in_hospital_mortality.joblib`<br>`examples/research_outputs_supervised/04_suave_training/suave_model_manifest_in_hospital_mortality.json` | <pre><code class="language-python">from pathlib import Path
import joblib, json

manifest = json.loads(Path("examples/research_outputs_supervised/04_suave_training/suave_model_manifest_in_hospital_mortality.json").read_text())
print(manifest.keys())
payload = joblib.load(Path("examples/research_outputs_supervised/01_data_and_schema/evaluation_datasets_in_hospital_mortality.joblib"))
print(payload["datasets"].keys())
</code></pre> |
| 9. 潜空间相关性与解释 | 特征-预测目标-潜空间相关性气泡图 | 图像 | `plot_feature_latent_correlation_bubble` 绘制的总体相关性气泡图，颜色表示相关系数（RdBu_r，0 为中点），气泡大小按 `-log10(p)` 缩放且省略 `p≥0.1` 的关联，标签来源 `PATH_GRAPH_NODE_DEFINITIONS`，潜变量刻度渲染为 `$z_{n}$`；图像输出 PNG/JPG/SVG/PDF 四种格式 | 潜变量-特征-结局的相关矩阵与显著性矩阵（`overall_corr`、`overall_pvals`） | `examples/research_outputs_supervised/09_interpretation/latent_clinical_correlation_in_hospital_mortality_correlations.csv`<br>`examples/research_outputs_supervised/09_interpretation/latent_clinical_correlation_in_hospital_mortality_pvalues.csv` | <pre><code class="language-python">from pathlib import Path
import pandas as pd

corr_path = Path("examples/research_outputs_supervised/09_interpretation/latent_clinical_correlation_in_hospital_mortality_correlations.csv")
pval_path = Path("examples/research_outputs_supervised/09_interpretation/latent_clinical_correlation_in_hospital_mortality_pvalues.csv")
print(pd.read_csv(corr_path, index_col=0).head())
print(pd.read_csv(pval_path, index_col=0).head())
</code></pre> |
| 9. 潜空间相关性与解释 | 特征→潜变量→结局的多层次路径图 | 图像 | `plot_feature_latent_outcome_path_graph` 生成的多层次路径网络图 | SUAVE 模型与训练特征/标签（`model`、`X_train_model`、`y_train_model`） | `examples/research_outputs_supervised/01_data_and_schema/evaluation_datasets_in_hospital_mortality.joblib`<br>`examples/research_outputs_supervised/04_suave_training/suave_model_manifest_in_hospital_mortality.json` | <pre><code class="language-python">from pathlib import Path
import joblib, json

payload = joblib.load(Path("examples/research_outputs_supervised/01_data_and_schema/evaluation_datasets_in_hospital_mortality.joblib"))
train_features, train_labels = payload["datasets"]["Train"]
print(train_features.shape, train_labels.shape)
manifest = json.loads(Path("examples/research_outputs_supervised/04_suave_training/suave_model_manifest_in_hospital_mortality.json").read_text())
print(manifest["model_path"])
</code></pre> |
