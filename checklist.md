# SUAVE Feature Checklist

本清单依据 README 描述的核心目标以及 `configs/suave_default.yaml` 中的研发计划，逐项核对当前仓库的实现情况。

## 模型层
- [ ] 多分布解码器，覆盖连续 / 分类 / 计数变量。（计划在 README 中强调需支持混合类型；当前 `Decoder` 仅实现高斯头并在文件头明确写明“Only continuous variables are implemented”。）
- [x] 缺失感知的重构损失与 mask 处理。（`fit` 中对 NaN 位置构造 mask，并在 `gaussian_nll` 内按 mask 计算 NLL。）
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
