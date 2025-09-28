# 一、总体路线

1. **MVP（先跑通端到端）** ✅
   - [x] 包结构与安装脚手架
   - [x] HI-VAE（PyTorch实现）：以hivae2的tf实现为骨架，改写为pytorch实现并对齐到我们设计的项目结构。似然头包括 `real`(Gaussian) 、 `cat/bernoulli`(Categorical/Bernoulli)`、`pos`(LogNormal)、`count`(Poisson) 、`ordinal`（cumulative link）
   - [x] `SUAVE` 高层 API（一个类就够）：`fit() / predict() / predict_proba() / calibrate() / sample() / save() / load()`
   - [x] `fit()` 内部完成 **train 内部切分验证集**（如 `val_split=0.1`）
   - [x] 分类头：冻结解码器 → 训练 head → **轻联合微调**（warm-start → head → light joint FT）
   - [x] 温度缩放校准 + 基础可视化（ROC/PR、可靠性图、`plot_feature_latent_correlation_bubble` 潜变量-特征气泡图，p 值由 statsmodels 校正）

2. **小成本，大增强** ✅
   - [x] 追加似然头
   - [x] 自动化：自动化识别数据类型和生成schema（`suave.schema_inference` + `suave.interactive.schema_builder`）
   - [x] 启发式超参推荐与序列化（`suave.defaults`），覆盖 latent_dim/hidden_dim/batch_size/epoch 调度
- [x] 条件生成（CVAE 开关 `conditional=True`）：`fit(..., y=...)` 时启用可控采样
   - [x] 可解释性：beta-VAE
- [x] 类不平衡处理：`class_weight/focal` + 条件过采样（已支持 `class_weight`，**`focal` 与条件过采样待补充**）
   - [x] 分类损失权重自动调节
- [x] Schema自动推断
   
3. **可能需要大幅修改的增强**
   - [ ] 添加 SUAVE 半监督支持（Warmup 阶段无监督允许无标签样本、分类训练和联合微调阶段仅接受有标签样本）
   - [x] 无监督模式下的`predict()/predict_proba()`方法实现，参考HIVAE论文（通过 `attr=` 指定要推断的属性）
   - [ ] 显式建模缺失模式，并允许生成带缺失的数据
- [ ] 添加回归任务支持
   
4. **评测闭环** ✅
   - [x] **TSTR/TRTR** 评测
   - [x] 简单 **MIA**（membership inference）基线（影子模型/置信阈值法）
   - [x] 结果打包与示例 notebook（研究作 example）
   - [x] 研究级 MIMIC/eICU 分析脚本整合：Optuna 最优试验、基线模型、SUAVE 校准、TSTR/TRTR、分布漂移与报告导出（`examples/research-mimic_mortality_supervised.py`）

------

# 二、包结构与API

```
suave/
  __init__.py
  defaults.py             # 超参启发式推荐与序列化
  types.py                 # Schema&枚举（用户手动提供，先不做自动推断）
  data.py                  # 缺失mask、标准化/反标准化、train内部分割
  schema_inference.py      # 列类型启发式推断、info/interactive 模式
  modules/
    encoder.py             # MLP Encoder
    decoder.py             # 多头解码（real/cat 起步；pos/count/ordinal 后续加）
    distributions.py       # torch.distributions 封装 & NLL
    heads.py               # 分类头（MLP/Logistic）
    losses.py              # ELBO(重构NLL+KL)、CE、Focal、对齐正则（预留）
    calibrate.py           # 温度缩放
    prior.py               # 混合先验（含可学习均值）
  model.py                 # SUAVE 主类（下面给接口）
  sampling.py              # 条件采样/批量生成
  evaluate.py              # ROC/PR/Brier/ECE/可靠性图；TSTR/TRTR；MIA基线
  plots.py                 # 可视化
  interactive/
    __init__.py
    schema_builder.py      # 交互式 schema 构建/校对工具
examples/
  sepsis_minimal.py        # 端到端最小示例（你的研究）
```


### `SUAVE` 主类（关键参数与训练阶段）

- `schema`: 列名 → {`type`, `n_classes`} 的显式声明，可在 `fit()` 时覆盖；若缺省会调用 `SchemaInferencer` 给出 info/review 模式提示。
- `behaviour`: 选择 `"supervised"` 或 `"unsupervised"` 流程，控制是否启用分类头与联合微调。
- `latent_dim`、`hidden_dims`、`classification_loss_weight`、`dropout`、`learning_rate`、`batch_size`、`warmup_epochs`、`head_epochs`、`finetune_epochs` 等参数缺省为 `None` 时，会在 `fit()` 前通过 `defaults.recommend_hyperparameters` 自动补全，并记录到 `auto_hyperparameters_`。
- `head_hidden_dims` 与 `joint_decoder_lr_scale` 支持自定义分类头容量与联合微调时 decoder/prior 的学习率缩放。
- 训练默认内部划分验证集（`val_split`、`stratify`），并在 `fit()` 的 warm-up → head-only → joint 三阶段中自动调节分类损失权重与早停策略。

```python
import numpy as np
from suave import SUAVE

model = SUAVE(
    schema=schema,
    behaviour="supervised",
    latent_dim=None,              # None 触发启发式 latent_dim 推断
    hidden_dims=None,             # None 触发多层隐藏单元推荐
    classification_loss_weight=None,  # None 时将基于 ELBO/CE 自动调节
    dropout=None,
    learning_rate=None,
    batch_size=None,
    warmup_epochs=None,
    head_epochs=None,
    finetune_epochs=None,
    val_split=0.1,
    stratify=True,
)

model.fit(
    X_train,
    y_train,
    plot_monitor=True,            # 可视化训练/验证曲线
)

proba = model.predict_proba(X_test)
model.calibrate()                 # 未显式传入验证集时复用内部切分
latent = model.encode(X_train)
synthetic = model.sample(1000, conditional=True, y=np.random.choice(model.classes_, size=1000))
model.save("path/to/ckpt")
reloaded = SUAVE.load("path/to/ckpt")
```

**评测（独立函数）**：`evaluate.evaluate_classification`、`evaluate_tstr`、`evaluate_trtr`、`simple_membership_inference` 等函数可直接消费 `predict_proba` 的输出，补充 ROC/PR/Brier/ECE、TSTR/TRTR 与隐私攻击基线结果。

------

# 三、任务计划

## 1) 准备工作（一次性）

- **仓库初始化**

  ```bash
  mkdir suave && cd suave
  python -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or cpu
  pip install numpy pandas scikit-learn matplotlib scipy tqdm torchmetrics
  pip install black ruff pytest
  git init && echo ".venv\n__pycache__\n*.ipynb_checkpoints\n" > .gitignore
  ```

- 新建空文件树：

  ```bash
  mkdir -p suave/modules examples && touch suave/__init__.py suave/types.py suave/data.py suave/model.py \
    suave/modules/{encoder.py,decoder.py,distributions.py,heads.py,losses.py,calibrate.py} \
    suave/{sampling.py,evaluate.py,plots.py} examples/sepsis_minimal.py
  ```


## 2) 主干任务

### ✅ Task-0｜包骨架 & 最小 API（空实现 + 文档 + 单测）

- 目标：创建最小可运行的包架构，提供 `SUAVE` 空实现、类型约定与数据处理占位，确保 `pytest -q`、`black .` 与 `ruff .` 可通过。
- 交付：模块化目录结构、`SUAVE` 类接口（`fit/predict/predict_proba/calibrate/encode/sample/save/load`）、schema 约定、最小示例与占位单测。
- 验收：导入包即可运行最小流程，分类头暂为空实现但具有完整 docstring 与类型注解。

### ✅ Task-1｜迁移 HI-VAE 核心训练（real+cat）

- 目标：在 PyTorch 中复刻 HI-VAE 的 encoder/decoder 与重构 NLL，打通 `real` 与 `cat` 类型的 ELBO 训练路径。
- 交付：`EncoderMLP`、`LikelihoodHead` 框架与 `RealHead/CatHead` 实现，`losses.elbo`、`losses.kl_warmup`、`distributions` 工具与 mask-aware 训练循环。
- 验收：在混合 real/cat 玩具数据上 ELBO 稳定下降，遇到未实现列类型抛出清晰错误，相关单测覆盖 NLL 与 mask 行为。

### ✅ Task-2｜分类头、轻联合微调与校准

- 目标：扩展监督模式训练计划，引入分类头阶段、轻联合微调与温度缩放校准，初步支持类不平衡设置。
- 交付：`LogisticHead/MLPHead`（含 `class_weight` 与 `focal_gamma` 支持）、三阶段训练调度、`TemperatureScaler` 实现与可靠性评估工具。
- 验收：`fit()` 可自动拆分验证集完成三阶段训练，`calibrate()` 使 ECE 明显下降，分类评估指标在示例数据上可追踪。

### ✅ Task-3｜追加似然头（pos/count/ordinal）与数值稳定

- 目标：补齐剩余分布头，保证数值稳定性与采样一致性，使 ELBO 可在多类型列上收敛。
- 交付：`PosHead/CountHead/OrdinalHead` forward & 采样逻辑、Poisson/LogNormal/Ordinal NLL 推导、针对阈值与 rate 的 softplus/clamp 处理、对应单测。
- 验收：启用新增列类型后 ELBO 可下降，单测校验数值一致性与缺失掩码处理。

### ✅ Task-4｜评测闭环（TSTR/TRTR、MIA 基线）

- 目标：构建合成数据评测与隐私基线，完善可视化与结果打包流程。
- 交付：TSTR/TRTR 评测函数、membership inference 基线、ROC/PR/可靠性图工具、可复用的基线模型工厂。
- 验收：API 在小数据上可运行至结束，返回包含 AUROC/AUPRC/Brier 等关键指标的 JSON 摘要。

### ⏳ Task-5｜文档与示例强化

- 目标：补全用户文档、示例脚本与 schema 工具集成，突出条件生成与评测流程。
- 交付：完善 docstring、`examples/sepsis_minimal.py` 的端到端流程、schema dump/load 辅助工具、README 中的使用指南。
- 验收：文档示例可直接运行生成校准曲线与潜变量可视化，用户可按 README 指引完成条件生成与 TSTR 评测。
