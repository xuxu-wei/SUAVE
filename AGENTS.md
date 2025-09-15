# AGENTS.md

## 0) 使命 / 范围

在 **SUAVE** 中实现并默认启用**纯表格（非时序）**的统一生成–判别框架 **TabVAE+Classifier**，支撑三大能力：

1. **预测**（二分类/多分类，多任务可选，含**校准**）；
2. **合成**（TSTR vs TRTR 效用评估，分布学指标）；
3. **解释**（潜变量–临床特征**相关热图**+显著性）。
    并提供可选**先验升级**（VampPrior / MoG）与**隐私审计占位**（MIA）。
    必须**保留现有主要 API/用法**（见 README 中 `from suave import SuaveClassifier` 等），并通过现有基准测试（`tests/test_benchmarks.py`）。([GitHub](https://github.com/xuxu-wei/SUAVE))

------

## 1) 开工须知

1. **Benchmark 监控**：每次代码修改后运行：
    `pytest tests/test_benchmarks.py -s`

   `python tools/compare_baselines.py || echo "REGRESSION>3pp"`，记录分类性能变化。

   - 若第一次生成 current.json，允许 candidate 覆盖为 current 以建立基线。

   - 每当涉及 **SUAVE 模型结构、损失函数或训练流程** 的改动时，务必运行该测试，将结果写入 `reports/baselines/candidate.json`，并与 `current.json` 对比监控性能；若任一性能指标回落超过 **3 个百分点**（>0.03），必须发出警告信息。
   - 不要中断测试，除非测试已运行了 45 分钟以上。
   - 内部评估需安装 `autogluon` 以运行 benchmark；用户使用时则无此强制依赖。

2. **清理与文档**：补 `README`/`docs/` 与示例配置

3. **参数回归检查**：修改 SUAVE/SUAVE 损失函数等核心逻辑后，务必复核模型类与测试用例的默认参数设置，并运行**Benchmark 监控**避免性能意外下降。

## 2) 成功标准（验收门槛）

- **API 兼容**
- **功能完成**：
  - SUAVE（多分布头、缺失 mask）+ 分类头多任务；
  - **温度缩放**校准与 **ECE/可靠性图** 导出；
  - **合成器**（先验/后验采样，类条件可选）与 **TSTR 管线**；
  - **先验可拔插**（标准正态 / **MoG** / **VampPrior**）；
  - **潜变量×特征**相关热图与 FDR 显著性表；
  - **MIA 占位**（影子模型+置信攻击接口，默认关闭）。
- **测试通过**：保留并通过现有基准测试；新增最小单测覆盖核心模块。

## 3) 代码风格与性能预算

- **风格**：PEP8 + type hints；模块/函数含docstring；关键数学处写公式；
- **日志**：CSV/JSON日志最简实现，TensorBoard 可选；
- **随机性**：统一 `utils/seed.py` 设置 `numpy/torch/random`；
- **性能**：单卡/CPU均可训；单元测试在10分钟内完成；生成端内存友好（分批）。

---

📌 备注

- 请**严格保留**现有主要 API 与 README 展示流程（以免下游用户断裂）。([GitHub](https://github.com/xuxu-wei/SUAVE))
- 任何新增功能的默认值应**不改变**现有行为（如：`prior="standard"`，`calibrate=True` 仅在 `fit(val=...)` 时生效）。
- 隐私/MIA 仅提供**研究占位**，默认关闭；开启需用户显式传参。
- 文档与示例优先用**合成数据**演示，避免真实数据分发风险。