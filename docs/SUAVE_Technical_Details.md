# SUAVE 技术说明

## 1. 核心理念：面向临床的“三能一体”模型
SUAVE（Supervised, Unified, Augmented Variational Embedding）旨在为临床智能提供可落地的生成式—判别式统一框架，其核心动机体现在“三能一体”中：

1. **临床预测模型构建能力** —— 通过监督头与校准模块，面向死亡率、再入院风险、并发症预测等任务提供可靠概率输出。
2. **异质数据合成能力** —— 在显式 Schema 的约束下统一处理实值、分类、计数、序数等多模态表格特征，以生成隐私友好的合成数据，支持数据共享、隐私保护和数据增强。
3. **内生可解释性能力** —— 利用层次化潜变量、显式分布参数与特征级掩码，使潜在空间和重建因子可以直接映射到临床语义，满足医生对可解释性的需求。

## 2. 设计思路

### 2.1 Schema-First 工作流
- 用户以 `Schema` 声明每个特征的类型（`real`、`cat`、`pos`、`count`、`ordinal` 等）及类别数量。
- 预处理管线使用 Schema 完成标准化/嵌入、缺失值掩码与后处理恢复，形成统一的数据接口。

### 2.2 分阶段训练调度
1. **Warm-up**：仅优化生成模型，逐步增加 KL 权重（KL annealing）稳定训练。
2. **Head 训练**：冻结生成器，训练分类头以最小化监督损失。
3. **联合微调**：小学习率联合优化生成器与分类头，实现更好的生成—判别折衷。

### 2.3 自洽的校准与评估
- 内置温度缩放（Temperature Scaling）对 logits 进行后验校准。
- 提供 AUROC、AUPRC、Brier Score、ECE 等评估函数，确保预测可靠性。

## 3. 关键创新点

| 维度 | 创新描述 |
| ---- | -------- |
| **统一 Schema 驱动的异质数据建模** | 通过显式 Schema 与类型特定的解码头（高斯、对数正态、泊松、分类 softmax 等），无缝覆盖多种特征分布，并保持缺失值一致性。 |
| **生成与监督的同域潜空间** | 层次化潜变量共享于合成与分类任务，支持条件采样、特征重建以及概率预测的一致表示。 |
| **面向临床的解释性路径** | 潜变量分量与参数直接对应到特征簇，结合互信息、KS、MMD 等指标形成可审计报告。 |
| **多阶段训练 + 校准的一站式实现** | 从预处理、训练、评估到合成数据质量审计均有统一 API，降低临床数据科学家的上手门槛。 |

## 4. 模型架构

### 4.1 组件概览
1. **Encoder (`modules/encoder.py`)**：对不同类型特征进行嵌入后，经多层感知机输出潜变量的均值 `\mu` 与尺度参数 `\sigma`。
2. **Latent Prior**：采用多分量高斯先验或标准正态先验 `p(z)`，支持临床群体的亚群划分。
3. **Decoder (`modules/decoder.py`)**：针对每类特征输出参数（实值的均值/方差、分类概率、泊松率等），并在有缺失掩码的情况下执行重建。
4. **Supervised Head (`modules/heads.py`)**：在潜空间上添加线性或多层感知机分类器。
5. **Calibration (`modules/calibrate.py`)**：对分类头 logits 进行温度缩放。
6. **Evaluation (`evaluate.py`)**：计算性能与隐私指标。

### 4.2 前向与训练流程
- **编码**：`x` 经标准化与缺失掩码处理后输入编码器，得到潜在分布 `q_\phi(z \mid x)`。
- **采样与重构**：从 `q_\phi(z \mid x)` 中采样 `z`，经解码器生成参数化分布 `p_\theta(x \mid z)` 并计算重建损失。
- **监督预测**：同一 `z` 作为分类头输入，输出 logits 并计算监督损失与校准。
- **合成数据**：从先验 `p(z)` 采样，经解码器生成合成样本；在条件采样时，利用分类头的逆向约束或条件先验采样特定类别的 `z`。

## 5. 数学公式

### 5.1 潜变量后验与先验
编码器给出潜变量的均值 `\mu_\phi(x)` 与对数方差 `\log \sigma_\phi^2(x)`，近似后验为：
\[
q_\phi(z \mid x) = \mathcal{N}\big(z;\, \mu_\phi(x), \operatorname{diag}(\sigma_\phi^2(x))\big).
\]
若使用混合先验，则：
\[
p(z) = \sum_{k=1}^K \pi_k \mathcal{N}(z; \mu_k, \Sigma_k), \qquad \sum_k \pi_k = 1.
\]

### 5.2 证据下界（ELBO）
SUAVE 的无监督训练目标采用掩码感知的 ELBO：
\[
\mathcal{L}_{\text{ELBO}}(x) = \mathbb{E}_{q_\phi(z \mid x)}\big[\log p_\theta(x_{\text{obs}} \mid z)\big] - \beta \, D_{\mathrm{KL}}\big(q_\phi(z \mid x) \parallel p(z)\big),
\]
其中 `x_{\text{obs}}` 表示通过掩码选择的观测特征，`\beta` 是 KL 退火系数。对于实值特征，重建项为高斯似然：
\[
\log p_\theta(x^{(r)} \mid z) = -\frac{1}{2}\sum_i m_i \left[ \frac{\big(x^{(r)}_i - \mu^{(r)}_{\theta,i}(z)\big)^2}{\sigma^{2,(r)}_{\theta,i}(z)} + \log \sigma^{2,(r)}_{\theta,i}(z) + \log(2\pi) \right],
\]
其中 `m_i` 为缺失掩码。分类特征使用 softmax 分布，计数特征使用泊松或负二项分布，以此类推。

### 5.3 监督损失与温度校准
对于带标签的样本 `(x, y)`，分类头产生 logits `f_\psi(z)`，监督损失为：
\[
\mathcal{L}_{\text{sup}}(x, y) = - \sum_{c=1}^C y_c \log \operatorname{softmax}_c(f_\psi(z)).
\]
温度缩放在校准阶段优化标量 `T>0`：
\[
\hat{y} = \operatorname{softmax}\Big(\frac{f_\psi(z)}{T}\Big), \qquad T^* = \arg\min_T \Big(-\sum_{(x,y)\in \mathcal{D}_{\text{val}}} y^\top \log \hat{y}\Big).
\]

### 5.4 总体训练目标
结合生成与监督任务，最终目标（在联合微调阶段）为：
\[
\mathcal{J}(x, y) = \mathcal{L}_{\text{ELBO}}(x) + \lambda \mathcal{L}_{\text{sup}}(x, y) + \gamma \mathcal{R}_{\text{reg}},
\]
其中 `\lambda` 控制生成与分类的权衡，`\mathcal{R}_{\text{reg}}` 表示可选的正则项（如权重衰减或对齐约束），`\gamma` 为其权重。

## 6. 应用与展望
- **临床预测**：支持 ICU 死亡率、早期预警等任务，通过校准的概率提升决策透明度。
- **数据共享与隐私保护**：生成合成数据集以便跨机构共享，同时结合成员推断评估监测隐私风险。
- **可解释分析**：通过潜变量分量、互信息、KS/MMD 指标构建可视化报告，帮助医生理解模型行为。

未来计划请参阅 [docs/Roadmap.md](Roadmap.md)。
