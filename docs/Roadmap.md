## 进度快照

- ✅ MVP 核心链路（HI-VAE 迁移 + SUAVE API + 三阶段训练 + 校准 + 条件采样）已跑通并落地到主干。
- ✅ Task-0 ～ Task-4 的实现、测试与评测工具已全部交付并在当前仓库中维护。
- ⏳ Task-5（文档、示例与 schema dump/load）仍在待办列表，需要按原规划补齐。
- 🆕 TODO 已登记：为 SUAVE 增强半监督流程（Warmup 无监督 + 分类阶段有监督）。

------

# 一、总体路线（两周内可跑通）

**MVP → 低成本增强 → 评测闭环** 的三段式推进，每段都能独立交付。

1. **MVP（先跑通端到端）** ✅
   - [x] 包结构与安装脚手架
   - [x] **HI-VAE（PyTorch实现）**：将hivae2的tf实现完整改写为pytorch实现并对齐到我们设计的项目结构。似然头包括 `real`(Gaussian) 、 `cat/bernoulli`(Categorical/Bernoulli)`、`pos`(LogNormal)、`count`(Poisson) 、`ordinal`（cumulative link）
   - [x] `SUAVE` 高层 API（一个类就够）：`fit() / predict() / predict_proba() / calibrate() / sample() / save() / load()`
   - [x] `fit()` 内部完成 **train 内部切分验证集**（如 `val_split=0.1`）
   - [x] 分类头：冻结解码器 → 训练 head → **轻联合微调**（warm-start → head → light joint FT）
   - [x] 温度缩放校准 + 基础可视化（ROC/PR、可靠性图、潜变量相关热图）

2. **低成本增强（按需启用）**
   - [x] 追加似然头
   - [ ] 自动化：自动化识别数据类型和生成schema

   - [x] 条件生成（CVAE 开关 `conditional=True`）：`fit(..., y=...)` 时启用可控采样

   - [x] 可解释性：beta-VAE

   - [ ] 类不平衡：`class_weight/focal` + 条件过采样（已支持 `class_weight`，`focal` 与条件过采样待补充）

3. **可能需要大幅修改的增强**
   - [ ] 添加 SUAVE 半监督支持（Warmup 阶段无监督、分类训练阶段有监督）
- [ ] 无监督模式下的`predict()/predict_proba()`方法实现，参考HIVAE论文
   
4. **评测闭环** ✅
   - [x] **TSTR/TRTR** 脚手架（独立评测器）
   - [x] 简单 **MIA**（membership inference）基线（影子模型/置信阈值法）
   - [ ] 结果打包与示例 notebook（你的研究作 example）

------

# 二、包结构与API（codex先按此骨架生成）

```
suave/
  __init__.py
  types.py                 # Schema&枚举（用户手动提供，先不做自动推断）
  data.py                  # 缺失mask、标准化/反标准化、train内部分割
  modules/
    encoder.py             # MLP Encoder
    decoder.py             # 多头解码（real/cat 起步；pos/count/ordinal 后续加）
    distributions.py       # torch.distributions 封装 & NLL
    heads.py               # 分类头（MLP/Logistic）
    losses.py              # ELBO(重构NLL+KL)、CE、Focal、对齐正则（预留）
    calibrate.py           # 温度缩放
  model.py                 # SUAVE 主类（下面给接口）
  sampling.py              # 条件采样/批量生成
  evaluate.py              # ROC/PR/Brier/ECE/可靠性图；TSTR/TRTR；MIA基线
  plots.py                 # 可视化
examples/
  sepsis_minimal.py        # 端到端最小示例（你的研究）
```

### `SUAVE` 主类（超参全在方法/构造器里）

```python
m = SUAVE(
    schema: Dict[str, Dict],              # 用户手动提供：{"col":{"type":"real|cat|pos|count|ordinal","nclass":...}}
    latent_dim: int = 32,
    beta: float = 1.5,                    # β-VAE权重
    conditional: bool = False,            # 是否CVAE
    hidden_dims: Tuple[int,...] = (256,128),
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    kl_warmup_epochs: int = 8,
    val_split: float = 0.1,               # 训练内部划验证集（从train切）
    class_weight: Optional[Dict[int,float]] = None,
    focal_gamma: Optional[float] = None,
    device: str = "auto",                 # "auto"|"cpu"|"cuda"
)

m.fit(
    X_train: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    max_epochs: int = 60,
    batch_size: int = 256,
    joint_ft_epochs: int = 10,            # 轻联合微调轮数
    freeze_decoder_on_head: bool = True,
    early_stop_patience: int = 5,
    random_state: int = 42,
)

proba = m.predict_proba(X_test)           # 预测概率
m.calibrate(X_val=None, y_val=None)       # 若未传则使用fit时的内部验证集
Z = m.encode(X_train)                     # 潜表示
X_syn = m.sample(n=10000, y=1)            # 条件生成（若conditional=True）
m.save("path/to/ckpt"); m2 = SUAVE.load("path/to/ckpt")

# 评测（独立函数）
evaluate.calibration_curves(y_true, proba)
evaluate.tstr(X_syn, X_test, y_test, clf="xgboost")
evaluate.mia_baseline(m, X_train)
```

> 备注：如需配置文件，**只在数据目录**允许 `schema.json`（`SUAVE.dump_schema(data_dir)` / `SUAVE.load_schema(data_dir)`），避免用户修改包内文件。

------

# 三、与 codex 的协作方式

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

- 新建空文件树（让 codex 往里填实现）：

  ```bash
  mkdir -p suave/modules examples && touch suave/__init__.py suave/types.py suave/data.py suave/model.py \
    suave/modules/{encoder.py,decoder.py,distributions.py,heads.py,losses.py,calibrate.py} \
    suave/{sampling.py,evaluate.py,plots.py} examples/sepsis_minimal.py
  ```

## 2) **AGENTS.md**（放根目录，codex 先读它再写代码）

见根目录。

## 3) **首批 codex 任务与指令模板**

> ### ✅ Task-0｜包骨架 & 最小 API（空实现 + 文档 + 单测）
>
> **Prompt 给 codex：**
>
> > 目标：创建最小可运行的包架构，仅提供空实现和文档，占位单测能跑通导入与最小流程。
> >
> > - 生成目录：
> >
> >   ```
> >   suave/
> >     __init__.py
> >     types.py
> >     data.py
> >     model.py
> >     modules/
> >       encoder.py
> >       decoder.py
> >       distributions.py
> >       heads.py
> >       losses.py
> >       calibrate.py
> >     sampling.py
> >     evaluate.py
> >     plots.py
> >   examples/sepsis_minimal.py
> >   third_party/hivae_tf/  # 已预置官方TF源码（只读）
> >   ```
> >
> > - `suave.model.SUAVE`：定义构造器与空方法 `fit / predict_proba / calibrate / encode / sample / save / load`，写完整 docstring（参数、形状、返回值、示例）。
> >
> > - `suave.types.Schema`：手动 schema 约定（列名→{type, nclass?}），v1 不做自动推断。
> >
> > - `suave.data`：实现
> >
> >   - `split_train_val(X, y, val_split: float, stratify: bool)`（从 train 内部分出 val）
> >   - 缺失掩码构造（bool mask）
> >   - 标准化/反标准化仅覆盖 `real / cat` 两类
> >
> > - `examples/sepsis_minimal.py`：演示从 CSV 读数据、构造 schema、实例化模型、调用 `fit→predict→calibrate→plot`（可生成空图占位）。
> >
> > - 基础单测（pytest）：
> >
> >   - 包可导入；`SUAVE` 可实例化；
> >   - `fit` 跑通日志打印（占位训练循环可仅 sleep/print）；
> >   - `predict_proba` 返回形状与输入行数一致。
>
> **验收**：`pytest -q` 通过；`black . && ruff .` 无报错。
>
> ------
>
> ### ✅ Task-1｜对照 HI-VAE(TF) 迁移 Encoder/Decoder（real+cat）与 ELBO
>
> **核心变更**：不是凭空实现，而是**逐函数对照** `third_party/hivae_tf/hivae/loglik_models_missing_normalize.py` 与 `VAE_functions.py`，在 PyTorch 中复现 **real / pos / count / cat / ordinal** 的**接口雏形**，但 **本任务只启用 real + cat 训练路径**，其余先实现为**可调用但默认禁用**（返回 `NotImplementedError` 或 `@experimental` 提示）。ELBO 先覆盖 mask-aware 重构 + KL；Warm-start 仅训 ELBO。
>
> **Prompt 给 codex：**
>
> > 目标：在 PyTorch 中**对照迁移** HI-VAE(TF) 的解码与 NLL 计算，先打通 `real + cat` 的端到端训练（仅 ELBO）。
> >
> > **阅读参考（只读）：**
> >  `third_party/hivae_tf/hivae/loglik_models_missing_normalize.py`
> >  `third_party/hivae_tf/hivae/VAE_functions.py`
> >
> > **实现要求：**
> >
> > 1. `modules/encoder.py`
> >    - `EncoderMLP(input_dim: int, hidden=(256,128), dropout=0.1, out_dim: int)`：返回 `mu_z, logvar_z`（Normal 近似后验）。
> >    - 数值稳定：对 `logvar_z` clamp 到 [-15, 15] 范围。
> > 2. `modules/decoder.py`（仿 HI-VAE 的“按列类型选择头”机制）
> >    - 统一接口 `class LikelihoodHead(nn.Module)`：
> >      - `forward(x, params, norm_stats, mask) -> dict`，键包括：
> >        - `"log_px"`（按样本聚合的对数似然，**已乘以 mask**）
> >        - `"log_px_missing"`（缺失项 loglik，用于评估）
> >        - `"params"`（分布参数，反标准化后）
> >        - `"sample"`（`torch.distributions` 采样得到的样本）
> >    - 实现 `RealHead`（Gaussian）：softplus 确保方差正，反标准化（对齐 TF 中的 affine transform）；
> >    - 实现 `CatHead`（Categorical）：输出 logits，NLL= `F.cross_entropy(logits, onehot_labels, reduction='none')` 的负号；采样用 `Categorical(logits=...)`；
> >    - 占位头（先不在训练图中启用）：
> >       `PosHead`（LogNormal：对数域NLL + 反标准化 + exp-1 采样）、
> >       `CountHead`（Poisson：rate=softplus）、
> >       `OrdinalHead`（cumulative-link：阈值单调性用 `softplus`+`cumsum`）。
> >       —— 保留 forward 骨架与 TODO 注释，单测暂跳过。
> > 3. `modules/distributions.py`
> >    - 封装 `nll_gaussian(x, mu, var, mask)`、`nll_categorical(x_onehot, logits, mask)`；
> >    - 提供 `sample_gaussian(mu, var)`、`sample_categorical(logits)`；
> >    - 常数 `EPS=1e-6`，`softplus(var_raw)+EPS` 保正。
> > 4. `modules/losses.py`
> >    - `elbo(recon_terms: List[Tensor], kl: Tensor) -> Tensor`，`recon_terms` 是各列 loglik 之和（已含 mask）；
> >    - `kl_warmup(step, total_steps, beta_target)` 线性退火；
> >    - `kl_normal(mu, logvar)`。
> > 5. `model.SUAVE.fit`（Warm-start 版）
> >    - 仅优化 ELBO（重构 + β·KL），**支持 mask-aware**；
> >    - KL 线性退火 0→β（`kl_warmup_epochs`）；
> >    - 训练循环 + `tqdm` 日志；
> >    - 保存：`self._norm_stats_per_col`（模仿 TF 的 normalization parameters）；
> >    - 仅启用 `real + cat` 列；其它类型遇到时抛出清晰错误，提示“启用后续 Task-4”。
> >
> > **数值一致性与单测：**
> >
> > - 在 `tests/test_nll_math.py` 写**闭式对照**（不是对 TF 求值）：
> >   - 随机小张量上验证：Gaussian/Categorical 的 NLL 与手写公式一致（容差 `1e-6`）。
> > - 在 `tests/test_elbo_sanity.py`：
> >   - 合成数据（混合 real/cat 列）训练 5–10 个 epoch，验证 `NLL` 下降、`KL > 0`。
> > - 在 `tests/test_masking.py`：构造缺失掩码，确认只在观测项累计 NLL。
> >
> > **风格与文档：**
> >  所有公开函数写 type hints 与示例；对照 TF 代码处增加行内注释 `# parity: loglik_real TF lines XX-YY`。
>
> **验收**：
>  ELBO 训练可在混合 real/cat 的玩具数据上稳定下降；单测全部通过；遇到 pos/count/ordinal 列能给出明确错误信息与启用提示。
>
> ------
>
> ### ✅ Task-2（微调）｜分类头 + 轻联合微调 + 不平衡 & 校准
>
> **Prompt 给 codex：**
>
> > - `modules/heads.py`：实现 `LogisticHead` 与可选 `MLPHead`；支持 `class_weight` 与 `focal(gamma)`。
> > - `model.SUAVE.fit`：加入阶段 B（冻结解码器训练 head）与阶段 C（全模型 small-lr 轻联合微调；解码器 lr 更小）。
> > - `modules/calibrate.py`：温度缩放（验证集拟合 T，最小化 NLL）。
> > - `evaluate.py`：实现 `auroc/auprc/brier/ece` 与可靠性曲线。
> > - 单测：
> >   - 小数据上温度缩放后 `ECE` 明显下降；
> >   - 阶段 C 后验证 `AUPRC` 不下降（相对阶段 B）。
>
> ------
>
> ### ✅ Task-3（增强）｜追加似然头（pos/count/ordinal）与数值稳定
>
> **Prompt 给 codex：**
>
> > - 完成 `PosHead/CountHead/OrdinalHead` 的 forward，实现：
> >   - `pos`：LogNormal（对 log(1+x) 计算高斯 NLL，采样时 `exp(sample)-1`，对负值 clamp 0）；
> >   - `count`：Poisson（rate=softplus(raw)+EPS；`torch.poisson`/`Poisson(rate)` 采样；NLL 用 `log_poisson_loss` 等价式）；
> >   - `ordinal`：cumulative-link（分段阈值 `softplus`+`cumsum` 单调，类别概率用差分 sigmoid）。
> > - 单测：
> >   - 在 `tests/test_heads_math.py` 用手写闭式或稳定实现对照（容差 `1e-4`）；
> >   - 随机数据 sanity-check：开启相应列后，ELBO 可下降。
>
> ------
>
> ### ✅ Task-4｜评测闭环（TSTR/TRTR、MIA基线）
>
> **Prompt 给 codex：**
>
> > - `evaluate.tstr(X_syn, X_test, y_test, clf='xgboost')` 与 `evaluate.trtr(...)`：训练独立分类器（XGBoost/LightGBM/LogReg 任选其一）并返回 JSON 摘要（AUROC/AUPRC/Brier）。
> > - `evaluate.mia_baseline(model, X_train)`：实现影子模型/置信阈值攻击占位，无外网依赖。
> > - 产出图表：分布学（直方/核密度）、可靠性曲线、PR/ROC。
> > - 单测：API 正常返回关键字段；在极小数据上可运行至结束。
>
> ------
>
> ### ⏳ Task-5｜文档与示例
>
> **Prompt 给 codex：**
>
> > - 完善 docstring；
> > - 在 `examples/sepsis_minimal.py` 展示：训练→校准→外检→潜变量相关热图（`plots`）；
> > - 提供 `SUAVE.dump_schema(data_dir)` 与 `SUAVE.load_schema(data_dir)`（如需 `schema.json`，只写到**数据目录**）；
> > - 确保 README 片段：如何准备 schema、如何开启条件生成、如何做 TSTR。

## 4) 代码评审清单（你用来验收 codex 产物）

- `fit()` 可仅凭 train（内部切 val）跑完三阶段；日志包含 NLL/KL/AUROC/AUPRC/Brier/ECE
- `predict()` 输出 shape 正确、无 NaN
- `predict_proba()` 输出 shape 正确、无 NaN
- `calibrate()` 后 ECE 明显下降
- `sample()` 可条件采样（若开启）
- `save/load()` 往返一致
- 单测通过：`pytest -q`
- 风格：`black . && ruff .` 零错误

------

# 四、初始命令（从空仓开始）

```bash
# 1) 建仓&环境
mkdir suave && cd suave
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio  # 按需选CPU/CUDA源
pip install numpy pandas scikit-learn matplotlib scipy tqdm torchmetrics xgboost
pip install black ruff pytest

# 2) 生成空文件树（见上）
# 3) 打开 codex，贴入 Task-0 Prompt（上文），让其一次写完骨架与最小示例
# 4) 本地跑
python examples/sepsis_minimal.py
pytest -q
black . && ruff .
```

------

## 附：给 codex 的“分布映射”速查（短表）

| HI-VAE TF 名     | PyTorch 目标                                            |
| ---------------- | ------------------------------------------------------- |
| `loglik_real`    | Gaussian：`Normal(mu, sqrt(var))`                       |
| `loglik_pos`     | LogNormal：对 `log(1+x)` 高斯 NLL；采样 `exp(sample)-1` |
| `loglik_cat`     | Categorical：`Categorical(logits)`                      |
| `loglik_ordinal` | Cumulative-link：`softplus` 阈值 + `cumsum`             |
| `loglik_count`   | Poisson：`Poisson(rate=softplus(raw))`                  |

> 缺失处理：所有 NLL 都要 **乘 mask**；评估需返回 `log_px_missing`。

