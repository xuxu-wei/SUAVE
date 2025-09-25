![GitHub Repo stars](https://img.shields.io/github/stars/xuxu-wei/SUAVE)![Static Badge](https://img.shields.io/badge/English-README-blue?link=https%3A%2F%2Fgithub.com%2Fxuxu-wei%2FSUAVE%2Fblob%2Fmain%2FREADME.md)![PyPI](https://img.shields.io/pypi/v/suave-ml)

# SUAVE: Supervised, Unified, Augmented Variational Embedding

SUAVE 是一款以 Schema 为核心的变分自编码器，面向混合类型表格数据，同时支持生成式建模与有监督预测。项目直接受到 HI-VAE 及其后续层次潜变量研究的启发，并在此基础上强化了显式 Schema、分阶段训练以及概率校准等现代化工作流。

## 核心特性

- **Schema 驱动的输入。** 用户需通过 `Schema` 明确声明每一列特征，使模型在训练前即可获知数据类型与类别数量。
- **分阶段优化。** 训练流程遵循“预热 → 分类头 → 联合微调”的调度，并配合 KL 退火以获得稳定的收敛表现。
- **透明的自动化。** 启发式默认值会依据数据统计量自适应批大小、学习率与训练时长，同时所有超参仍可被显式覆盖。
- **缺失值感知的生成解码。** 标准化工具与解码器头会在实数、分类、正数、计数与序数特征上持续传递掩码，从而一致地处理缺失数据。
- **内置校准与评估。** 模型提供温度缩放、Brier 得分、ECE 等评估指标，帮助在下游场景中生成可靠概率。

## API 全景

| 方法 | 作用 |
| ---- | ---- |
| `fit(X, y=None, **schedule)` | 采用分阶段优化训练生成模型与（可选的）分类头，并自动划分内部验证集。 |
| `predict(X)` | 在经过校准后返回最可能的类别标签。 |
| `predict_proba(X)` | 输出已校准的类别概率，并通过缓存机制避免重复的编码计算。 |
| `calibrate(X, y)` | 在保留集 logits 上学习温度缩放参数，并在后续预测中复用。 |
| `encode(X, return_components=False)` | 将数据映射至潜空间，可选地返回混合分量分配与对应统计量。 |
| `sample(n, conditional=False, y=None)` | 生成合成样本，可按类别标签进行条件采样。 |
| `impute(X, only_missing=True)` | 重建缺失或被掩码的单元格并与原始数据合并。 |
| `save(path)` / `SUAVE.load(path)` | 保存与恢复模型权重、Schema 元数据和校准状态，以便部署。 |

## 安装

```bash
pip install -r requirements.txt
```

SUAVE 依赖 Python 3.9+ 与 PyTorch。

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

完整示例可参考 [`examples/mimic_mortality_supervised.ipynb`](examples/sepsis_minimal.py)。

## API 概览

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
model.fit(train_X, train_y, warmup_epochs=20, head_epochs=5, finetune_epochs=10)
```

当 ``behaviour="unsupervised"`` 时可省略 ``y``，训练仅包含预热阶段：

```python
unsupervised = SUAVE(schema=schema, behaviour="unsupervised")
unsupervised.fit(train_X, epochs=50)
```

### 概率预测

```python
proba = model.predict_proba(test_X)
preds = model.predict(test_X)
```

模型会基于输入指纹缓存 logits，从而避免重复的编码计算。

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

### 潜变量编码

```python
z = model.encode(test_X)
components = model.encode(test_X, return_components=True)
```

后者会额外返回混合分量分配及对应的后验统计量。

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
