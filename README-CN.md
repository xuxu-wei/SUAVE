# SUAVE: Supervised, Unified, Augmented Variational Embedding

SUAVE 是一款以 Schema 为核心的变分自编码器，面向混合类型表格数据，同时支持生成式建模与有监督预测。项目直接受到 HI-VAE 及其后续层次潜变量研究的启发，并在此基础上强化了显式 Schema、分阶段训练以及概率校准等现代化工作流。

## 核心特性

- **Schema 驱动的输入。** 用户需通过 `Schema` 明确声明每一列特征，使模型在训练前即可获知数据类型与类别数量。
- **分阶段优化。** 训练流程遵循“预热 → 分类头 → 联合微调 → 解码器细化”的调度，并配合 KL 退火以获得稳定的收敛表现。
- **透明的自动化。** 启发式默认值会依据数据统计量自适应批大小、学习率与训练时长，同时所有超参仍可被显式覆盖。
- **缺失值感知的生成解码。** 标准化工具与解码器头会在实数、分类、正数、计数与序数特征上持续传递掩码，从而一致地处理缺失数据。
- **内置校准与评估。** 模型提供温度缩放、Brier 得分、ECE 等评估指标，帮助在下游场景中生成可靠概率。

## 安装

SUAVE 依赖 Python 3.10+ 与 PyTorch。**推荐**在安装此包前，先安装适合您系统环境的 PyTorch。请参考 [PyTorch官方指南](https://pytorch.org/get-started/locally/) 进行安装。
例如，windows 平台下使用 pip 命令安装Cuda 12.1版本对应的 PyTorch:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install suave-ml
```
## API 概览

| 方法 | 作用 |
| ---- | ---- |
| `fit(X, y=None, **schedule)` | 采用分阶段优化训练生成模型与（可选的）分类头，并自动划分内部验证集。 |
| `predict(X, attr=None, **options)` | 返回类别标签或指定特征的后验预测；在无监督模式下必须指定 `attr`。 |
| `predict_proba(X, attr=None, **options)` | 输出已校准的类别概率，或在给定分类/序数特征时返回后验预测分布，并缓存结果以避免重复编码。 |
| `predict_confidence_interval(X, attr, **options)` | 针对实数/正数/计数特征给出后验统计量（点估计 + 置信区间，可选原始样本）。 |
| `calibrate(X, y)` | 在保留集 logits 上学习温度缩放参数，并在后续预测中复用。 |
| `encode(X, return_components=False)` | 将数据映射至潜空间，可选地返回混合分量分配与对应统计量。 |
| `sample(n, conditional=False, y=None)` | 生成合成样本，可按类别标签进行条件采样。 |
| `impute(X, only_missing=True)` | 重建缺失或被掩码的单元格并与原始数据合并。 |
| `save(path)` / `SUAVE.load(path)` | 保存与恢复模型权重、Schema 元数据和校准状态，以便部署。 |



## 快速上手

```python
import pandas as pd
from suave import SUAVE, SchemaInferencer

# 1. 读取数据并通过交互式界面审阅 Schema
train_X = pd.read_csv("data/train_features.csv")
train_y = pd.read_csv("data/train_labels.csv")["label"]
schema_result = SchemaInferencer().infer(train_X, mode="interactive")  # 打开 UI 并协助构建 schema
schema = schema_result.schema

# 2. 使用审阅后的 Schema 训练模型
model = SUAVE(schema=schema)
model.fit(train_X, train_y)

# 3. 获取预测结果
probabilities = model.predict_proba(train_X.tail(5))
labels = model.predict(train_X.tail(5))
```

若跳过第 1 步，`SUAVE.fit` 会在 `mode="info"` 下自动推断 schema，便于快速验证流程；对于生产数据，推荐开启交互式审阅以突显需要手动检查的列。

完整示例可参考 [`examples/demo-mimic_mortality_supervised.ipynb`](examples/demo-mimic_mortality_supervised.ipynb)。

## API 参考

以下示例覆盖最常用的工作流程。除特别说明外，接口均可接受 pandas DataFrame 或 NumPy 数组。

### Schema 定义

```python
from suave.types import Schema

schema = Schema(
    {
        "age": {"type": "real"},
        "gender": {"type": "cat", "n_classes": 2},
        "lactate": {"type": "pos"},
        "icu_visits": {"type": "count"},
    }
)
```

Schema 支持动态更新并校验数据列：

```python
schema.update({"qsofa": {"type": "ordinal", "n_classes": 4}})
schema.require_columns(["age", "gender", "qsofa"])
```

### 模型训练

```python
from suave import SUAVE

model = SUAVE(schema=schema, latent_dim=32, beta=1.5)
model.fit(train_X,train_y)
```

最终的解码器精修阶段默认与预热轮数相同，如专注于分类头可通过``decoder_refine_epochs=0`` 禁用该阶段。

当 ``behaviour="unsupervised"`` 时可省略 ``y``，训练会退化为仅包含预热阶段，因为分类头与解码器细化会被禁用：

```python
unsupervised = SUAVE(schema=schema, behaviour="unsupervised")
unsupervised.fit(train_X, epochs=50)
```

### 概率预测

```python
from suave import data as suave_data

# 监督模式下的类别预测
proba = model.predict_proba(test_X)
preds = model.predict(test_X)

# 针对单个特征的后验查询
mask = suave_data.build_missing_mask(test_X)
gender_probs = model.predict_proba(test_X, attr="gender", mask=mask)
glucose_point = model.predict(test_X, attr="glucose")
glucose_samples = model.predict(test_X, attr="glucose", mode="sample", L=128)

# 连续型特征的置信区间估计
age_stats = model.predict_confidence_interval(test_X, "age", L=256)
```

模型会基于输入指纹缓存分类概率，避免重复的编码计算。当提供 `attr` 时将切换至生成式解码器以返回单个特征的后验预测；若输入为外部填补的数据，请传入 `mask` 以保留原始缺失模式。对于实数/正数/计数特征，可使用 `predict_confidence_interval` 获取点估计与置信区间，而在 `predict` 中设置 `mode="sample"` 则会返回蒙特卡洛样本。在无监督模式下，由于分类头被禁用，需显式指定 `attr`。

**监督与无监督预测的差异**

- 当不提供 ``attr`` 时，``predict`` 与 ``predict_proba`` 依赖已经训练好的分类头（默认监督模式）；若在无标签数据上训练后直接调用，将因无法生成 logits 而报错。
- 提供 ``attr`` 时，两种模式都会走生成式解码路径。``predict_proba`` 仅支持分类/序数特征，而 ``predict`` 在遇到实数/正数/计数特征时会回退到 ``predict_confidence_interval`` 的结果。
- `predict_confidence_interval` 始终走解码器路径（因此必须指定``attr``），仅支持实数/正数/计数特征，可在两种模式下返回后验统计；当不存在分类头时，这是连续变量的推荐入口。
- ``behaviour="unsupervised"`` 会禁用分类头，因此必须在 ``predict`` 与 ``predict_proba`` 中显式传入 ``attr``，仅返回解码器给出的生成式统计量。
- 对解码器相关的调用传入 ``mask`` 可以确保掩码单元保持缺失状态；当输入 `X` 本身保留了原始 ``NaN`` 标记时可省略``mask``。

### 概率校准与评估

```python
model.calibrate(val_X, val_y)
calibrated = model.predict_proba(test_X)
```

温度缩放在保留集 logits 上训练，并在后续预测中自动复用。

```python
from suave.evaluate import compute_auroc, compute_auprc, compute_brier, compute_ece

auroc = compute_auroc(proba, val_y.to_numpy())
auprc = compute_auprc(proba, val_y.to_numpy())
brier = compute_brier(proba, val_y.to_numpy())
ece = compute_ece(proba, val_y.to_numpy(), n_bins=15)
```

各指标函数会自动校验概率矩阵的形状，在输入退化时返回 `numpy.nan`。

### 生成数据质量评估

```python
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from suave.evaluate import (
    evaluate_trtr,
    evaluate_tstr,
    classifier_two_sample_test,
    mutual_information_feature,
    rbf_mmd,
    simple_membership_inference,
)

# 对比真实数据与合成数据的迁移表现
tstr_scores = evaluate_tstr((X_syn, y_syn), (X_test, y_test), LogisticRegression)
trtr_scores = evaluate_trtr((X_train, y_train), (X_test, y_test), LogisticRegression)

# 执行 C2ST 分布检验
real_matrix = real_features.values
synthetic_matrix = synthetic_features.values
c2st = classifier_two_sample_test(
    real_matrix,
    synthetic_matrix,
    model_factories={
        "xgboost": lambda: XGBClassifier(random_state=0),
        "logistic": lambda: LogisticRegression(max_iter=200),
    },
    random_state=0,
    n_bootstrap=200,
)

# 检查特征分布的一致性
mmd_labs, mmd_labs_p = rbf_mmd(
    real_labs.values, synthetic_labs.values, random_state=0, n_permutations=200
)
mi_unit = mutual_information_feature(real_unit.values, synthetic_unit.values)

# 评估隐私泄露风险
attack = simple_membership_inference(train_probs, train_labels, test_probs, test_labels)
```

`evaluate_tstr` / `evaluate_trtr` 可以搭配任意监督模型验证迁移性能；`classifier_two_sample_test` 接收一个模型工厂映射（默认组合 XGBoost 主模型与逻辑回归敏感性分析），而 RBF-MMD、按维度归一的欧氏 + 汉明能量距离（支持置换检验 `p` 值）以及互信息则聚焦单个特征的差异程度。C2ST AUC 接近 `0.5`、MMD/能量距离接近 `0.0`、互信息接近 `0` 通常表示较好的拟合。成员推断攻击给出区分训练样本与保留样本的 AUROC 与准确率，用于监控潜在的隐私泄露。

### 潜变量编码

```python
z = model.encode(test_X)
components = model.encode(test_X, return_components=True)
```

后者会额外返回混合分量分配及对应的后验统计量。

### 潜变量-特征相关分析

```python
from suave.plots import plot_feature_latent_correlation_bubble

fig, ax = plot_feature_latent_correlation_bubble(model, train_X, targets=train_y)
```

气泡大小代表 Spearman 相关系数的绝对值，颜色表示（经校正的）P
值；指定 ``output_path`` 时会将图像写入磁盘（如
``outputs/latent_correlations.png``）。

### 数据采样

```python
synthetic = model.sample(100)
conditional = model.sample(50, conditional=True, y=preds[:50])
```

生成结果会自动反标准化，并恢复分类特征的原始取值。

### 缺失值填补

```python
# 仅填补在标准化阶段被标记为缺失的单元格
completed = model.impute(test_X, only_missing=True)

# 无监督模式同样复用该接口
unsup_completed = unsupervised.impute(test_X, only_missing=True)
```

``impute`` 会在掩码单元上运行解码器（包括未见过的分类取值与越界的序数编码），并将重建结果合并回原始
DataFrame，确保下游流程接收到完整特征。

### 持久化

```python
path = model.save("artifacts/sepsis.suave")
restored = SUAVE.load(path)
restored.predict_proba(test_X)
```

保存的模型文件包含 schema 信息、参数权重与校准状态，可直接复现部署推理。
