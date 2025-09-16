## 0) 目的与适用范围

本文件规范**智能代码代理 / 协作开发者（下文统称“Agent”）**在 SUAVE 仓库中的通用工作方式：

- 面向**临床表格数据**的统一潜空间框架（VAE + 分类头），支持**预测、合成、解释**三类工作流；
- 保持对现有 **`SUAVE` 主类**与 README 快速开始示例的**完全兼容**（`from suave import SUAVE` / `fit/predict_proba/generate/latent` 等）。
- 所有**具体开发任务**由“任务 Prompt”下发；本文件只定义**角色、约束、质量门槛、回归守护与提交规范**。
- 任何改动不得破坏 `tests/` 下的基准用例（尤其是分类性能监控用例）。
- **环境初始化提示**：启动任务前请按照 AutoGluon 官网推荐命令安装全量 AutoGluon 依赖：

  ```bash
  pip install -U pip
  pip install -U setuptools wheel
  pip install autogluon --extra-index-url https://download.pytorch.org/whl/cpu
  ```

------

## 1) Agent 角色与通用承诺

**角色**：全栈 ML 系统开发协作者（代码、文档、实验、评估）。
 **承诺**：

1. **API 兼容优先**：不破坏用户面向的公开 API 与 README 用法（`SUAVE.fit/predict_proba/generate/latent` 等）。
2. **可复现实验**：每次改动需可**本地复现**并附最小脚本/命令。
3. **回归守护**：任何与**模型结构/损失/训练流程**有关的改动，**必须**触发基准运行与回归比对（见 §3）。
4. **默认安全开关**：隐私相关（MIA/DP-SGD）默认关闭，仅在任务明确要求时启用。
5. **文档先行**：新增/变更功能同步更新示例与简要文档（示例优先使用合成数据，避免真实数据分发风险）。

------

## 2) 任务 Prompt 模板（推荐）

当你要让 Agent 做一项具体工作时，请使用以下模板：

```
[Goal]
用一句话说明预期产出（例如：为 SUAVE 增加温度缩放校准并导出 ECE/可靠性图）。

[Context]
- 受影响模块：suave/* （尽量列出）
- 相关API：SUAVE.fit / predict_proba / generate / latent（保持兼容）  # 来自 README 的公开接口
- 评测基线：`python tools/benchmark.py --epochs 100 --latent-dim 8 --batch-size 128`（全量 hard + 缺失率任务；需保持默认训练配置以发挥 SUAVE 性能）
- 冒烟校验：`pytest tests/test_benchmarks_smoke.py -s`（快速检查基准管线接线）
- 资源/限制：单卡或 CPU 可运行；单测 ≤10 分钟；不得新增重型依赖

[Spec]
- 功能点：…（逐条写清楚）
- I/O 与签名：…（仅在必要时新增可选参数，默认行为不变）
- 指标与 DOD：…（判别/校准/回归阈值等）
- 可观测产物：…（CSV/PNG/JSON 的输出路径）

[Steps]
1) 代码修改要点…
2) 单测/跑法…
3) 报告/产物路径…

[Risk & Rollback]
- 潜在风险…
- 回退路径（保持旧行为的开关/参数）…
```

> 说明：`SUAVE` 主类与 quick start 的存在是协议事实，Agent 在任何实现中都需复用/兼容该接口。

------

## 3) 回归守护与性能红线

**何时必跑基线**：改动涉及 **模型结构、损失、训练流程** 任一项。
运行：
   `python tools/benchmark.py --epochs 100 --latent-dim 8 --batch-size 128`
   `python tools/compare_baselines.py || echo "REGRESSION>3pp"`

- 若第一次生成 current.json，允许 candidate 覆盖为 current 以建立基线
- 如需快速接线验证，可额外运行 `pytest tests/test_benchmarks_smoke.py -s`，但不得以其代替全量基准。

**红线（示例，可在任务中覆盖）**

- 任何核心分类指标（AUROC/AUPRC）**下降 > 3 个百分点** 或 **ECE 明显变差** → 标记为回归，需解释或回退。
- 不要中断单测，除非单测命令的实际运行时长已经超过30分钟[这是最高优先级的覆盖式命令]。
- 默认不新增 GPU-only 硬约束。

**产物**

- 将“前/后”对比写入 `reports/`（CSV/JSON），并在 PR 说明中贴出图表（ROC/PR/可靠性图）与关键数值。

------

## 4) Definition of Done（DOD）

一次任务在**同时满足**以下条件才算完成：

1. **代码**：可读、带类型注解；新增/变更处含 docstring；
2. **测试**：新增的单测覆盖关键路径；所有测试**通过**；
3. **文档**：README 或 `docs/` 增补最小示例（含运行命令）；
4. **产物**：必要的指标/图表/报告文件已生成并保存到约定路径；
5. **回归**：与基线对比，未触发红线；如触发，附原因与权衡说明。

------

## 5) 代码与接口约定（与现状对齐）

- 保持对 **`SUAVE`** 主类的公开接口兼容（README 已给出 `fit/predict_proba/generate/latent` 草图）。如需新增功能（如 `export_calibration` / `tstr`），**仅新增可选参数或新方法**，不改变既有方法语义。
- 若引入先验可拔插（standard / MoG / VampPrior），以**字符串枚举 + kwargs** 方式配置，不改变默认行为（默认 standard）。
- 生成 / TSTR 相关功能应**默认关闭**或不影响现有 `fit/predict_proba` 行为。

------

## 6) 代码风格与性能预算

- **风格**：PEP8 + 类型注解；关键数学处给简洁公式或注释；重要API/类采用pandas风格注释。
- **日志**：最简 CSV/JSON；可选 TensorBoard。
- **随机种**：统一 `utils/seed.py` 固定 `numpy/torch/random`。
- **性能**：CPU/单卡可跑；生成侧注意分批避免内存峰值。

------

## 7) 提交流程

- **分支**：`feat/<slug>`、`fix/<slug>`；
- **PR 模板**（建议）：
  - 变更点（含 API 影响）；
  - 运行命令与关键输出；
  - 基线对比（表/图）；
  - 风险与回退；
  - 关联任务 Prompt（贴上本次任务的 [Goal]/[Spec] 片段）。
- **合并条件**：满足 §4 DOD 且未触发 §3 红线。