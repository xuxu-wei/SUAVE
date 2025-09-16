# SUAVE Feature Checklist

本清单依据 README 描述的核心目标以及 `configs/suave_default.yaml` 中的研发计划，逐项核对当前仓库的实现情况。

## 模型层
- [ ] 多分布解码器，覆盖连续 / 分类 / 计数变量。（计划在 README 中强调需支持混合类型；当前 `Decoder` 仅实现高斯头并在文件头明确写明“Only continuous variables are implemented”。）
- [x] 缺失感知的重构损失与 mask 处理。（`fit` 中对 NaN 位置构造 mask，并在 `gaussian_nll` 内按 mask 计算 NLL；推理路径 `predict_logits` / `predict_proba` / `latent` 同样在前向前构造 `mask` 并以 `torch.nan_to_num` 替换缺失值，保持预测阶段鲁棒性。）
- [x] 类条件合成 / CVAE 模式。（`generate` 支持 `conditional={"y": ...}`，并按标签生成 one-hot 条件。）
- [ ] 部署友好的校准链路。（README 要求模型原生输出经校准概率，但 `predict_proba` 仅做 softmax，未集成温度缩放等校准步骤。）

## 训练与优化
- [ ] 支持通过配置文件自定义编码器/解码器/分类头结构。（YAML 中给出多层宽度，但模型中写死了 256-256-128 等架构。）
- [ ] 重构损失按变量尺度加权（如 `inv_std`）。（配置中指定 `recon_weighting`，当前实现仅统一平均。）
- [ ] Focal Loss 等分类损失变体的可选开关。（配置中给出 `focal`，代码只调用标准交叉熵。）
- [ ] 小批量训练 / 可调 batch size。（`fit` 采用全量张量，无 DataLoader 支持 `batch_size`。）
- [ ] 早停与验证指标监控。（配置要求 `early_stop`，训练循环中未实现早停逻辑或验证评估。）
- [ ] CLI 训练脚本按配置加载真实数据流水线。（`suave/cli/train.py` 读取配置后仍使用随机数据，且未消费数据/优化参数配置。）

## 数据处理
- [ ] Schema 驱动的数据列类型解析与绑定。（`TableSchema` 只定义数据类，无读取或与训练流程的整合。）
- [ ] 提供示例 schema (`configs/schema_example.yaml`)。（配置引用了该文件，但仓库中不存在实际样例。）

## 评估与解释
- [x] 校准评估工具（ECE、温度缩放）。（`suave/modules/calibration.py` 实现了温度缩放和 ECE / 可靠性曲线计算。）
- [x] TSTR 评估（logistic / SVM / KNN）。（`tstr_auc` 支持在合成数据上训练这些估计器并输出 ROC-AUC。）
- [ ] TSTR 评估的 XGBoost 支持。（计划文件列出 `xgboost`，`tstr_auc` 尚未接入该选项，只在 `_estimator_factory` 中占位。）
- [x] 潜变量解释工具（相关性 + 降维投影）。（`latent_feature_correlation`、`latent_projection` 可用于解释。）

> 注：若条目被勾选，表示对应功能已在当前仓库中落地；未勾选则代表需要进一步实现或补全。

## TODO

### 开发优先级评估与任务规划

下表对 `checklist.md` 中尚未完成的特性进行影响分析与优先级排序，评估指标依序为：分类性能 → 生成质量 → 潜空间质量。

| 特性 | 分类 | 生成 | 潜空间 | 优先级 |
|---|---|---|---|---|
| 连续/分类/计数的多分布解码器 | ◑ | ● | ● | 高 |
| 部署级概率校准链路 | ● | ○ | ○ | 高 |
| 可配置的编码器/解码器/分类头结构 | ● | ◑ | ◑ | 高 |
| Focal Loss 等分类损失变体 | ● | ○ | ○ | 高 |
| 按变量尺度加权的重构损失 | ◑ | ◑ | ◑ | 中 |
| 小批量训练 & 可调 batch size | ◑ | ◑ | ◑ | 中 |
| 早停与验证指标监控 | ◑ | ◑ | ◑ | 中 |
| Schema 驱动的数据列类型解析 | ◑ | ◑ | ◑ | 中 |
| CLI 训练脚本读取真实数据 | ○ | ○ | ○ | 低 |
| 示例 schema 文件 | ○ | ○ | ○ | 低 |
| TSTR 评估的 XGBoost 支持 | ○ | ○ | ○ | 低 |

> ●=高影响, ◑=中影响, ○=低影响

---

## 高优先级任务

**1. 连续/分类/计数的多分布解码器**  
缺失此功能将导致离散变量重构与生成失真，从而削弱潜空间表示质量并限制分类泛化能力。

:::task-stub{title="支持混合类型的多分布解码器"}
1. 在 `suave/models/suave.py` 的 `Decoder` 中新增分类(softmax)与计数(泊松/负二项)头；按列类型输出对应参数。
2. 引入 `configs/suave_default.yaml` 中的列类型映射，训练阶段根据 schema 选择合适的 NLL 计算。
3. 扩展 `gaussian_nll` 或新增 `categorical_nll`、`poisson_nll` 等函数，并在 `fit` 中按列聚合。
4. 更新 `generate` 逻辑，对离散变量从相应分布采样。
5. README 与示例脚本展示混合类型数据的用法。
:::

**2. 部署级概率校准链路**  
目前 `predict_proba` 仅返回 softmax，缺乏温度缩放等校准，概率输出在临床场景不可靠。

:::task-stub{title="集成温度缩放的概率校准"}
1. 在 `suave/models/suave.py` 中加入可选 `calibrator` 成员，引用 `suave/modules/calibration.py` 的 `TemperatureScaler`。
2. 在 `fit` 结束后（或通过 `fit_calibration` 方法）使用验证集 logits/labels 训练温度参数。
3. `predict_proba` 调用前先通过 `calibrator.forward` 处理 logits。
4. 配置文件 `configs/suave_default.yaml` 中新增 `eval.calibration` 开关与参数。
5. 为校准流程编写单元测试并记录 ECE/可靠性曲线。
:::

**3. 可配置的编码器/解码器/分类头结构**  
当前架构写死，无法按数据规模/复杂度调整，影响分类、生成与潜空间表现。

:::task-stub{title="从配置文件加载网络结构"}
1. 在 `SUAVE.__init__` 解析 `encoder`, `decoder`, `clf_head` 列表，根据层宽动态构建 `nn.Sequential`。
2. `configs/suave_default.yaml` 中保留层宽列表，允许用户自定义。
3. 对 `suave/cli/train.py` 和 README 示例同步说明如何在 YAML 中修改结构。
4. 新增测试验证不同层数可正确构建并训练。
:::

**4. Focal Loss 等分类损失变体**  
缺失对类不平衡的适应，直接削弱分类性能。

:::task-stub{title="Focal Loss 可选开关"}
1. 在 `suave/modules/losses.py` 实现 `focal_loss(logits, targets, gamma)`。
2. 在 `SUAVE.fit` 根据配置 `loss.focal.use` 选择 `F.cross_entropy` 或 `focal_loss`。
3. 更新 `configs/suave_default.yaml` 的 `loss.focal` 参数解释。
4. 为平衡/不平衡场景各写一个单测，验证损失与梯度计算。
:::

---

## 中优先级任务

**5. 按变量尺度加权的重构损失**  
各列尺度差异过大可能主导损失，影响三类任务的训练稳定性。

:::task-stub{title="按变量尺度加权的重构损失"}
1. 在 `SUAVE.fit` 前计算训练集各列标准差或使用 schema 中提供的 `inv_std` 权重。
2. 修改 `gaussian_nll` 或包裹函数，在乘以 `mask` 后再按权重加权求和。
3. YAML 中 `loss.recon_weighting` 支持 `none`/`inv_std`，默认保持 `none`。
:::

**6. 小批量训练 & 可调 batch size**  
全量训练限制数据规模与泛化；按批次训练可提升训练效率及稳定性。

:::task-stub{title="引入 DataLoader 支持的批训练"}
1. 在 `SUAVE.fit` 重构为使用 `torch.utils.data.TensorDataset` + `DataLoader`，添加 `batch_size`、`shuffle` 参数。
2. 训练循环迭代 mini-batches，维护 `self._last_epoch` 等日志。
3. 更新 `configs/suave_default.yaml` 的 `train.batch_size`，并在 README 示例中展示。
:::

**7. 早停与验证指标监控**  
缺乏验证集监控易导致过拟合，影响三类性能。

:::task-stub{title="加入早停与验证评估"}
1. `SUAVE.fit` 接受 `X_val`, `y_val`，每个 epoch 计算验证 AUROC/AUPRC。
2. 根据 `early_stop` 配置记录最佳指标并在超过 `patience` 时停止训练。
3. 在 `reports/` 输出训练/验证曲线 CSV。
:::

**8. Schema 驱动的数据列类型解析**  
没有自动绑定列类型会增加数据预处理风险，间接影响全部任务。

:::task-stub{title="实现 schema 解析与绑定"}
1. 在 `suave/data/` 新增 `TabularPreprocessor`，读取 YAML schema，生成 `TableSchema` 对象。
2. 提供 `fit_transform`/`transform` 方法完成缺失值处理、类别编码等，并返回 `X, y, schema_info`。
3. `SUAVE.fit` 和 CLI 脚本调用 `TabularPreprocessor`，使模型根据 schema 知道变量类型。
:::

---

## 低优先级任务

**9. CLI 训练脚本读取真实数据**  
当前 CLI 用随机数据，仅影响用户体验。

:::task-stub{title="CLI 训练脚本接入真实数据"}
1. 在 `suave/cli/train.py` 中读取 `data.path`、`data.schema` 等配置，利用 `TabularPreprocessor` 加载 CSV/Parquet。
2. 允许命令行传入 `--config` 后自动训练并保存模型/指标。
3. 为脚本添加文档说明与示例配置。
:::

**10. 示例 schema 文件**  
仅文档作用，但有助于用户理解数据格式。

:::task-stub{title="提供示例 schema"}
1. 在 `configs/` 添加 `schema_example.yaml`，列举连续/分类/计数字段及目标列。
2. 在 README 与 CLI 示例中引用该文件。
:::

**11. TSTR 评估的 XGBoost 支持**  
影响评估覆盖度，但不直接影响模型性能。

:::task-stub{title="TSTR 支持 XGBoost 估计器"}
1. 在 `suave/eval/tstr.py` 的 `tstr_auc` 中增加 `elif estimator == "xgboost"` 分支，调用 `_estimator_factory`。
2. 配置 `eval.tstr.estimators` 支持 `xgboost`；文档注明需安装 `xgboost`。
3. 添加单元测试（跳过无 xgboost 环境时）。
:::

---

## 开发路线建议

1. **阶段一（分类可靠性）**：完成高优先级任务（多分布解码器、校准链路、可配置网络、Focal Loss），确保分类与生成的基础能力。
2. **阶段二（训练稳定性与数据抽象）**：实现中优先级任务（重构加权、批训练、早停、schema 解析），提升扩展性与泛化。
3. **阶段三（工具链与评估拓展）**：处理低优先级任务（CLI 真数据、示例 schema、XGBoost TSTR），完善使用体验与评估覆盖。

按此路线推进，可逐步提升 SUAVE 在分类、生成及潜空间建模方面的整体性能与可用性。
