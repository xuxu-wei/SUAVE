# AGENTS.md — SUAVE 重构蓝图（Codex-Ready）

## 0) 任务目标（一体三用，Mode 0）

- **目标**：以**表格VAE + 分类头**替换/统一现有实现，在**同一潜空间**内完成
  1. **院内/28天死亡预测**（判别+校准），
  2. **合成数据生成**（**Mode 0**：生成样本**无缺失**，用于TSTR），
  3. **潜变量可解释性**（与临床特征相关性热图）。
- **关键约束**：
  - **保留主要对外API功能**与**现有benchmark测试**可用性（如 `pytest tests/test_benchmarks.py -s` 能跑通）。
  - **缺失掩码只用于 Encoder 输入与重构损失的 mask**；**不训练/输出 mask 头**；生成样本**不含缺失**。
  - 优先保证**训练稳定性**、**可重复性**、**外部验证流程**与**校准评估**落地。

------

## 1) 开工须知（Agent 执行顺序）

1. **扫描测试以锁定公共API**
   - 解析 `tests/`（尤其 `tests/test_benchmarks.py`）找出**被导入的模块/类/函数名**与**调用签名**。
   - 建立“**适配层**”：若新实现的签名不同，提供**同名包装**以免破坏测试。
2. **按本蓝图落库**（见 §2 目录结构与模块职责），**不改动**测试中显式导入路径；必要时在原路径下保留 shim。
3. **实现 SUAVE+Classifier（Mode 0）**，训练与评测脚手架（见 §3–§6）。
4. **回归测试**：确保 `pytest -q` 全绿；新增单测覆盖生成无缺失、校准、TSTR 最小工作示例。
5. **Benchmark 监控**：每次代码修改后运行 `pytest tests/test_benchmarks.py -s`，记录分类性能变化。
   - Benchmark 允许运行 40 分钟，若超过 45 分钟应终止测试。
   - 内部评估需安装 `autogluon` 以运行 benchmark；用户使用时则无此强制依赖。
6. **清理与文档**：补 `README`/`docs/` 与示例配置；在CI中加入基本单测。
7. **参数回归检查**：修改 SUAVE/SUAVE 损失函数等核心逻辑后，务必复核模型类与测试用例的默认参数设置，避免性能意外下降。

------

## 2) 目录结构与模块职责（建议）

> 若现有仓库结构不同，请新增以下模块并用**适配层**对齐测试导入。

```
suave/
  api/
    __init__.py
    model.py                # 适配层：保留旧API名称/签名，内部调用新实现
  models/
    __init__.py
    suave.py               # SUAVE 主模型（Encoder/Decoder/Classifier/forward）
  modules/
    losses.py               # 重构NLL（混合数据类型）；KL退火；分类CE/加权/focal
    calibration.py          # Temperature scaling + ECE + reliability plot utils
  data/
    schema.py               # 变量模式(连续/二元/多类/计数)与缺失掩码组织；标准化/嵌入
    preprocessing.py        # 训练/评估一致的预处理与DataModule
  trainers/
    train_suave.py         # 训练循环/early-stop/日志；保存/加载
  eval/
    metrics.py              # AUROC/AUPRC/Brier/NLL/ECE
    tstr.py                 # TSTR vs TRTR 管线（简化版）
    interpret.py            # 潜变量与临床特征相关性（Spearman/BH-FDR）
  cli/
    train.py                # CLI 入口（fit / evaluate / generate）
    generate.py
  utils/
    seed.py                 # 可重复性设置
    io.py                   # 模型/配置/标准化器保存与加载
configs/
  suave_default.yaml       # 超参与数据schema示例（无时序）
tests/
  test_suave_minimal.py    # 新增：无缺失生成、校准、TSTR 冒烟测试
```

------

## 3) 模型规格（SUAVE+Classifier，Mode 0）

### 3.1 输入与Schema

- **变量类型**：
  - 连续：标准化（Robust/StandardScaler）
  - 二元：{0,1}
  - 多类：用**Embedding**（避免高维 one-hot）
  - 计数（可选）：Poisson/NB
- **缺失处理**：
  - 训练：保留**缺失指示器 mask**，与特征拼接给 Encoder；
  - 重构损失：对缺失位置**mask掉**；
  - **不训练任何 mask 输出头**。
- **拼接**：连续 + 二元 + 多类嵌入 → 向量 `x`；另有 `m`（同维mask）。

### 3.2 Encoder（MLP）

- 结构：MLP 256→256→128（ReLU/SiLU + BatchNorm + Dropout 0.3）；
- 输出：`mu(x, m), log_sigma(x, m)` → 采样 `z ~ N(mu, diag(sigma^2))`；
- 推荐潜维 **K=32**（网格扫 16/32/64）。

### 3.3 Decoder（MLP，多头）

- 共享层后**按变量类型分头**：
  - 连续：Gaussian（`mu, log_sigma`）或 Huber 回归（可选）
  - 二元：Bernoulli（`logit`）
  - 多类：Categorical（`logits`）
  - 计数：Poisson / NB（可选）
- **Mode 0 生成**：仅从 `z` 解码**值**，**不产生缺失**。

### 3.4 分类头（Predictor）

- 输入：`[z, encoder_last]` 或 `z`；
- 结构：MLP 128→64（Dropout 0.3） → `p(y=1)`；
- **温度缩放**：验证集拟合 `T`（冻结参数），推理用 `logits/T`。

### 3.5 损失与训练

- **ELBO**：
  - 重构NLL = 各变量类型的 log-likelihood 求和；**按mask屏蔽缺失项**；
  - KL：对 `q(z|x)` 与 `N(0, I)`；**KL退火**（前10–20 epoch线性从0→β=0.7）；
  - 可选：对重构项按**变量标准差逆**加权。
- **分类损失**：`CE(y, p)`（可加 `pos_weight` 或 focal γ=2）。
- **总损失**：`L = (NLL + β·KL) + λ·CE`（推荐 λ 从 0.5→1.0 退火）。
- 训练细节：AdamW(lr=3e-4, wd=1e-4), batch=256, epoch=100, early-stop(patience=10), grad-clip=1.0。

------

## 4) 生成（Generation）API — 仅 Mode 0

- **生成要求**：输出**无缺失**表格样本，列名/类型与训练前处理一致；可返回 `pandas.DataFrame`。

- **API 约定**（适配旧接口）：

  ```python
  model.generate(n_samples: int, conditional: Optional[Dict[str, Any]] = None, seed: Optional[int] = None) -> DataFrame
  ```

  - `conditional=None`：从标准正态或**后验聚合先验**采样 `z`；
  - 若传入 `conditional`（如 `{label: 1}`），采用**类条件**（CVAE小改动，若保持纯VAE则忽略并记录warning）。

- **确保**：`assert not df.isna().any().any()`；必要时对超出物理学意义范围的变量进行**后处理裁剪**（由 `schema` 提供上下限）。

------

## 5) 评测与外部验证

### 5.1 分类与校准

- 指标：**AUROC、AUPRC、Brier、NLL、ECE（10–20分箱）**；
- 报告**温度缩放**前后 ECE/NLL 改善；提供**可靠性图**绘制工具（存 PNG）。

### 5.2 TSTR（合成训练→真实测试） vs TRTR

- `eval/tstr.py`：
  - 训练：LogReg / XGBoost / MLP（三者至少一类）
  - 测试：真实 held-out 测试集
  - 报告：AUROC/AUPRC 置信区间与 Δ（TSTR − TRTR）
  - **断言**：TSTR 训练数据**无缺失**，其统计分布与真实训练集**基本匹配**（KS/MMD 报告）。

### 5.3 潜变量解释

- `eval/interpret.py`：
  - 相关性：Spearman / 点二列 / MI，**BH-FDR** 多重校正；
  - 输出：相关性矩阵与**热图**（PNG/CSV）；
  - 可选：UMAP/t-SNE 对 `z` 可视化（着色 `y` 与关键临床特征）。

------

## 6) 对外API与适配层（必须保留/兼容）

> 智能体需**先读取测试**以确定**真实API**；以下为**功能级**对齐清单（名称以现有测试为准）：

- **模型构造**：`Model(**config)` 或 `get_model(config)`
- **训练**：`fit(X_train, y_train, X_val=None, y_val=None, **kwargs)`
- **推理**：`predict_proba(X) -> np.ndarray`，`predict(X, threshold=0.5)`
- **生成**：`generate(n_samples, conditional=None)`（**无缺失**）
- **潜变量**：`latent(X) -> np.ndarray`
- **持久化**：`save(path)` / `load(path)`（包含标准化器/温度缩放参数）
- **基线/评估入口**：若测试中有 `run_benchmarks()` / `evaluate()` 等，需保持原路径与签名；内部调用新实现。

> 若旧实现存在 `suave.api.SUAVEModel` 之类对外类，请保留同名类，内部持有 `SUAVE` 实例并**完全转发**。

------

## 7) 新增/调整测试（在不破坏既有测试前提下）

1. `tests/test_suave_minimal.py`（新增）
   - **生成无缺失**：训练极小数据后 `df_gen = model.generate(256)`，断言 `df_gen.isna().sum().sum()==0`。
   - **校准**：温度缩放前后 ECE 下降或 NLL 改善（容忍小幅波动，断言“非劣”）。
   - **TSTR冒烟**：用 `generate` 合成训练集 vs TRTR 对比，断言流程可跑、指标可产出。
2. 确保 `pytest tests/test_benchmarks.py -s` **不需改动**即可通过。若其导入路径与签名发生冲突，**仅在实现端加适配层**，**不得改测试**。

------

## 8) 训练与复现实用脚本（CLI）

- `cli/train.py`
  - 接受 `--config configs/suave_default.yaml`；
  - 关键参数：latent_dim、β退火、λ退火、class_weight、early_stop；
  - 保存：`artifacts/` 下 `model.pt`、`scaler.pkl`、`tempscaler.pkl`、`config.yaml`、`metrics.json`。
- `cli/generate.py`
  - 载入模型，输出 `generated.csv`；**断言无缺失**；打印基本统计。

------

## 9) 配置样例（`configs/suave_default.yaml` 摘要）

```yaml
model:
  type: SUAVEClassifier
  latent_dim: 32
  encoder: [256, 256, 128]
  decoder: [256, 256, 128]
  clf_head: [128, 64]
  dropout: 0.3
  beta_anneal: {start: 0.0, end: 0.7, epochs: 15}
  lambda_anneal: {start: 0.5, end: 1.0, epochs: 15}
  loss:
    recon_weighting: "inv_std"   # or "uniform"
    focal: {use: false, gamma: 2.0}
train:
  optimizer: {name: adamw, lr: 3e-4, weight_decay: 1e-4}
  batch_size: 256
  max_epochs: 100
  early_stop: {patience: 10, metric: "val_auprc", mode: "max"}
data:
  schema: "configs/schema_example.yaml"   # 定义每列类型、上下限、类别词典
  target: "in_hospital_mortality"         # 或 28d_mortality
  imputation: "simple"                    # 训练时仍保留缺失指示器
eval:
  calibration: {ece_bins: 15, temperature_scaling: true}
  tstr: {estimators: ["logreg", "xgboost"]}
```

------

## 10) 代码风格与性能预算

- **风格**：PEP8 + type hints；模块/函数含docstring；关键数学处写公式；
- **日志**：CSV/JSON日志最简实现，TensorBoard 可选；
- **随机性**：统一 `utils/seed.py` 设置 `numpy/torch/random`；
- **性能**：单卡/CPU均可训；单元测试在10分钟内完成；生成端内存友好（分批）。

------

## 11) 迁移与兼容（必做）

- **保留旧路径/名称**：在旧模块下保留**同名类/函数**，内部仅 `from suave.models.suave import SUAVEClassifier as _Impl` 并转发。
- **数据前处理对齐**：若旧代码有标准化/编码器，务必在 `io.py` 一并持久化，`load()` 时恢复**完全一致**的前处理状态。
- **弃用标记**：对确需移除的内部函数，加 `@deprecated` 注解并在下个版本删除；但**不得破坏测试**。

------

## 12) 完成判定（Acceptance）

- 所有现有测试 **不修改** 的前提下通过（含 `tests/test_benchmarks.py -s`）。
- 新增测试通过（无缺失生成、校准、TSTR冒烟）。
- `generate(n)` 返回**无缺失**样本（Mode 0）。
- `predict_proba`/`predict`、`fit`、`save/load`、`latent` 等**主要API**可用。
- 产出 `metrics.json` 含：AUROC/AUPRC/Brier/NLL/ECE（校准前后）。

------

## 13) 风险与回退

- **KL塌缩** → 开启 KL 退火 / free-bits；
- **重构项主导** → 重构权重归一化；
- **类不平衡** → `pos_weight` 或 focal；
- **生成值异常** → 按 `schema` 裁剪；
- **外检性能低/过拟合** → 温度缩放 + 仅阈值再校准（不fine-tune）；
- **接口不匹配** → 以**适配层**优先，**不改测试**。

------

## 14) 提交与CI

- 提交信息规范：`feat(models): add SUAVEClassifier (mode-0 generation)`
- 开一个最小CI：`pytest -q` + flake8（或 ruff）+ mypy（非阻塞）。

------

### 附：关键函数伪代码（供实现对照）

```python
# forward(x, mask):
# x: concatenated numeric/binary/embeddings; mask: 1=observed, 0=missing
mu, log_sigma = encoder(torch.cat([x, mask], dim=-1))
z = reparameterize(mu, log_sigma)

# decoder outputs per variable type
recon_params = decoder(z)  # dict: {col_name: params}
nll = 0.0
for col in columns:
    lp = log_likelihood(recon_params[col], x[col], var_type[col])
    nll += (-lp * mask[col]).sum() / mask[col].sum().clamp_min(1.0)

kl = kl_normal(mu, log_sigma)  # batch-wise mean
logits = clf_head(torch.cat([z, encoder_last], dim=-1))
ce = weighted_ce(logits, y)

loss = (nll + beta*kl) + lambda_*ce
# generate(n):
z = torch.randn(n, latent_dim, device=device)
recon_params = decoder(z)
x_gen = sample_from_params(recon_params, var_types, schema)
assert not has_nan(x_gen)
return to_dataframe(x_gen, schema)
```

------

**到此为止，Codex 请按本蓝图逐步实施、加适配层确保测试不变、提交可运行代码与最小文档。**