Purpose：本文件为自动化编码代理（如 CODEX 代理）提供可执行的改造蓝图与验收标准。鉴于当前仓库不包含 MIMIC/eICU 原始数据、且已有 pytest tests/test_benchmarks.py -s 可用于分类性能观测，一切改动需先通过 P0 性能守门，确保分类性能不降、并优先查明“性能异常低”的根因。

0) Operating Guardrails（硬性约束）

P0 性能守门：任何改动前后均需运行：
  pytest tests/test_benchmarks.py -s
  将 AUROC/AUPRC/ACC/F1 等关键指标写入 reports/baselines/current.json（改动前的“基线”）与 reports/baselines/candidate.json（改动后）。若候选较基线任一核心指标下降超过 1 个百分点且无标记 ALLOW_REGRESSION=1 环境变量，则 PR 失败。

外部验证纯净性（为后续论文准备）：eICU 仅用于外部评估；任何训练/预训练/调参阶段不得使用 eICU（除非特意开展“无监督域适配”附录实验并在 eICU 内再留独立 holdout）。

可复现：固定随机种子；保存配置、版本、数据切分与评估输出；所有区间指标使用 bootstrap 95% CI（B=1000，后续可调）。

缺失与数值稳定：重构/似然仅对观测项计损（partial loss）；所有方差/率参数经 softplus；全局 eps=1e-8。

P0 — 分类性能诊断与稳定（先于一切改造）

目标：找出并修复导致 SUAVE 分类性能异常低的根因，在不引入外部数据前提下，通过现有基准测试稳定提升/守住性能。

P0.1 即刻基线捕获

运行：pytest tests/test_benchmarks.py -s
把逐折/逐种子指标写入：
  reports/baselines/current.json（首次运行即为“基线”）
  reports/figures/P0_learning_curves.png（训练/验证 loss、AUROC/AUPRC 随 epoch 变化）
  在 reports/md/P0_summary.md 生成可读总结（数据集名、样本量、阳性率、当前指标、种子方差）。

P0.2 故障树（最可能根因 → 探针 → 修复）

逐条执行与记录；每完成一条，重新跑基准并追加到 P0_summary.md。

A. 任务损失被重构/KL“淹没”
  探针：逐 epoch 打印/记录三类损失占比：L_cls : L_recon : beta*KL；把 beta_kl、lambda_recon 暂时设低（如 beta_kl=0.1, lambda_recon=0.5）。
  修复：
    启用 KL 退火（线性/循环；10–30 epoch 拉满）；
    引入 free bits（每潜维 ≥ 0.5 nat）；
    将早期训练的 lambda_cls 设为 2.0、随后余弦退火到 1.0。

B. 损失函数/激活不匹配
  探针：检查分类头是否 BCEWithLogitsLoss + 未手动 sigmoid；多标签 vs 单标签设置是否一致；正例比例与 pos_weight 是否传入。
  修复：统一为 BCEWithLogitsLoss(pos_weight=...)；若已做 sigmoid，改用 BCELoss 或移除 sigmoid。对严重不平衡，考虑 FocalLoss(γ=2.0)（作为对照试验）。

C. 评价与阈值问题
  探针：AUPRC 是否基于真实阳性率计算；阈值 0.5 是否合理。
  修复：报告 ROC/AUPRC 为主；阈值型指标（F1/精召）在验证集上调阈，并在报告中注明。

D. 数据切分/泄漏
  探针：确认基准测试中 train/val/test 是否 按标签分层、不交叉患者；任何标准化/缺失填补是否在训练集拟合、在验证/测试集仅变换。
  修复：将 StandardScaler/Imputer 放入 Pipeline 并仅在训练集 fit；保证随机种子与分层。

E. 优化/正则超参
  探针：LR 是否过高/过低（观测训练损失震荡或停滞）；Dropout/Weight decay 是否过强。
  修复：
    开启 OneCycleLR（max_lr=1e-3 起步）；
    Dropout 先降到 p=0.1–0.2；
    Weight decay 1e-5 起步；
    批大小 128–256；混合精度开启但监控数值稳定。

F. 梯度/连通性
  探针：在分类头与编码器关键层注册 backward hook，验证 grad_norm>0；检查是否梯度在 KL 或重构分支被“截断”。
  修复：确保分类头取自 潜空间 z（或其线性投影）而非重构路径的中间层；必要时在 z 上加一层 LayerNorm。

G. 训练时增强泄漏到验证
  探针：确认任何扰动/掩码增强只用于训练，不作用于验证/测试。
  修复：把增强封装在 train=True 的分支中。

H. 类别不平衡采样
  探针：统计阳性率，观测小批中阳性是否长期稀缺。
  修复：WeightedRandomSampler 或 BalancedBatchSampler；与 pos_weight 二选一或联合，观察稳定性。

I. 随机标签实验（泄漏自检）
  探针：将训练标签随机打乱；若 AUROC 明显高于 0.5，则高度可疑（泄漏或评估缺陷）。
  修复：逐条排查特征工程与评估流程，直到随机标签 AUROC≈0.5。

P0.3 最低验收线（P0 Gate）
  重新运行基准测试，任一核心指标不低于当前基线的 -1pp；
  生成对比报告：reports/md/P0_summary.md（含改动项、学习曲线、损失占比、最终指标与 CI）。
  通过后，方可进入后续 Epics（A–G）。

后续 Epics（在 P0 通过后依次推进）
  以下与此前规划保持一致，但全程受 P0 性能守门 约束；每个 Epic 完成后必须再次运行 pytest tests/test_benchmarks.py -s 并与 reports/baselines/current.json 对比; 每个Epic完成后必须检查下一个Epic，并完善下一个Epic的技术方案。

  Epic A — 异质似然解码器 + 自动 Schema（Heterogeneous Likelihoods）
    目标：为每列变量选择匹配的分布（Gaussian/LogNormal/Bernoulli/Categorical/Poisson/NegBin/ZINB/Beta/Ordinal），重构用 NLL、生成用采样；自动推断类型并允许 metadata.yaml 覆盖。
    新增模块：suave/schema/auto_schema.py、suave/models/hetero_decoder.py、suave/losses/hetero_nll.py。
    训练：partial NLL（仅观测项）、softplus 保证参数正值；
    生成：新增 model.sample(n, cond=None, seed=42)。
    验收：单测覆盖类型识别/数值稳定/采样出表；不得降低 P0 指标。

  Epic B — 半监督（M2 + 一致性/伪标签；仅用 MIMIC 内部未标注）
    接口：semisup 配置块；unlabeled_ratio=2.0、pseudo_thresh=0.9、lambda_cons=0.5 起步。
    实现：有标签优化 ELBO+CE；无标签优化 E_{q(y|x)}[ELBO]-α·H[q(y|x)] + 一致性。
    验收：不触碰 eICU；P0 指标不降，且建议报告稳定性（多种子平均±SD）。

  Epic C — 条件生成（类/站点条件，便于 TSTR）
    解码输入拼接 [z, cond_embed] 或类条件先验 p(z|y)；
    cond={"y":0/1, "site":"mimic"}；
    单测：不同 cond 生成样本的分布差异（KS）显著。

  Epic D — 域鲁棒（可选，作为附录）
    DANN/MMD/HSIC 放在 z 上；如用 eICU 无标签，必须另留 eICU holdout；标注为“UA 附录”。
    验收：附录实验不影响主文外部验证纯净性。

  Epic E — 评估套件（校准/TSTR/外部/隐私）
    校准：temperature_scale()、calibration_report()；
    TSTR/TRTR：统一管线与 CLI；
    外部：分中心/亚群森林图；
    隐私：最近邻重合率、成员推断基线；
    验收：生成 reports/figures/ 与 reports/tables/ 资产；P0 指标不降。

  Epic F — 可解释与潜表型
    latent_report()：潜变量—临床特征相关/回归（FDR 校正）、UMAP、聚类稳定性（NMI/ARI）。
    验收：报告生成且不影响 P0。

  Epic G — 文档/测试/打包
    扩充 README.md 示例；新增/完善单测（覆盖率≥80%）；CHANGELOG.md 记录变更。

工程与接口规范（更新版）
  训练日志：保存到 reports/diagnostics/（csv）；至少含 epoch, L_total, L_cls, L_recon, KL, auroc, auprc。
  随机性：所有 DataLoader/NumPy/PyTorch 种子统一自 config.seed。
  评估：默认 bootstrap B=1000 输出均值与 95% CI。
  命令：
    # 基线对比（P0 必做）
    pytest tests/test_benchmarks.py -s
    # 生成与校准示例（在相关 Epic 完成后）
    python -m suave.cli.tstr --synthetic syn.parquet --real real.parquet
    python -m suave.cli.calibrate --split val --method temperature

  数值稳定：所有对数/方差参数用 softplus 与 clamp(min=1e-6)；严禁在概率上取 log(0)。
  缺失：训练时 NLL 仅对观测项；编码器可拼接缺失掩码。

性能守门实现细节（代理须创建）
  脚本：tools/compare_baselines.py
    输入：reports/baselines/current.json 与 reports/baselines/candidate.json
    规则：若 (AUROC,AUPRC) 任一下降 > 0.01，则返回非零退出码。
  Pytest 插桩：在 tests/test_benchmarks.py 末尾追加写盘逻辑（不改变现有断言）：
    若 reports/baselines/current.json 不存在，保存当前结果为 current.json；
    若存在，保存为 candidate.json 并调用 tools/compare_baselines.py。
  CI 友好：允许通过环境变量 ALLOW_REGRESSION=1 暂时放行（需在 PR 描述里说明原因）。

  配置起点（保持不变动你现有接口的同时，利于 P0 排查）
    training:
      seed: 2025
      batch_size: 256
      lr: 1e-3
      weight_decay: 1e-5
      max_epochs: 150
      onecycle: true
    model:
      latent_dim: 32
      beta_kl: 0.5         # 开启 KL 退火；free_bits=0.5 nat/dim
      lambda_recon: 1.0
      lambda_cls: 1.0
    eval:
      bootstraps: 1000
      metrics: [auroc, auprc, brier]

验收清单（每个 PR 必填勾选项）
  已运行 pytest tests/test_benchmarks.py -s 并更新 reports/baselines/*.json；
  与基线比较未触发性能回退门槛（或已说明并设置 ALLOW_REGRESSION=1）；
  若修改了损失/优化/评估逻辑，提供 reports/figures/P0_learning_curves.png 与 P0_summary.md；
  新增/修改代码有单测；
  文档（README/CHANGELOG）已更新相关用法或变更。

Maintainers’ Notes
  本文件优先级：P0 > 其他 Epics。只有当 P0 守门通过后，代理才可继续实现异质似然、半监督等功能。
  若现有目录结构与文中建议路径不一致，代理应在 reports/repo_map.md 中说明适配方案，但不得删改 P0 守门机制。
