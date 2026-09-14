# 1. 报告元数据

- 审稿日期：2026-09-14
- 审阅对象：`output/pdf/nca_anonymous_draft.pdf`
- 论文题目：*Are Modality Quality Scores Trustworthy? Audited, Fixed-Budget Auxiliary Learning for Multimodal Sentiment Analysis*
- 暂定目标期刊：*Neural Computing and Applications*（NCA），Original Research Article
- 审稿模式：完整科学审稿、模拟多审稿人评议、写作与图表审查、NCA 投稿准备度检查
- 稿件规模：23 页，正文约 8,200 词，27 条参考文献，6 张表、5 幅图
- 审稿边界：以匿名 PDF 中可见证据为准；代码、实验日志和项目文件仅用于核对实现与稿件陈述，不把未写入论文的内部材料当作论文证据

## 本版相对上一稿的实质进步

1. 方法第 5.4 节已明确写出 `stopgrad(q)`，消除了任务辅助损失是否会通过权重路径训练可靠性头的关键歧义（p. 8）。
2. 摘要和第 7.6 节已将 MOSEI 定位为 frozen-protocol replication，而非外部预注册意义上的 confirmatory study。
3. 作者内部 drafting ledger 已从投稿 PDF 删除。
4. SAM-LML 的正式页码已修正。
5. 负面跨扰动结果、文本主导、三种子统计限制和未完成控制均被保留，科研披露较诚实。

# 2. 编辑初筛与可送审性

## 初筛判断

- **主题适配：通过。** 多模态学习、可靠性估计、样本级辅助监督分配和情感分析均符合 NCA 范围。
- **基本论文结构：通过。** 摘要、引言、相关工作、方法、实验、讨论、局限、结论和声明齐全。
- **双盲匿名：基本通过。** PDF 正文显示匿名作者和隐藏单位；PDF 元数据无作者、标题等身份字段，未发现明显自我暴露。
- **排版与技术生成：基本通过。** 23 页均可渲染，字体嵌入，未发现正文裁切、图像缺失或交叉引用失效。
- **投稿材料完整性：未通过。** Funding、Competing interests、Author contributions 仍是占位符；Data/Code Availability 仍以未来时表述（p. 19）。
- **科学证据闭环：未通过。** 跨样本分数可比性和最终训练日程下的 Permuted/Inverse 控制均缺失。

如果今天提交，我预计编辑不一定会因为格式或语言直接拒稿，但可能以“收益很小、关键对照不完整、可靠性解释未被验证”为由拒绝送外审。若进入完整外审，更可能得到 **Major Revision**；若按只能接收/拒绝的会议式决策，我倾向 **Weak Reject / Borderline Negative**。

NCA 官方指南要求双盲稿件去除作者及可识别信息，并要求原始研究包含 Data Availability Statement。当前匿名处理方向正确，但声明仍需在提交前写成最终事实，而非计划。[NCA Submission Guidelines](https://link.springer.com/journal/521/submission-guidelines)

# 3. 论文概述与贡献图谱

论文把“质量分数是否真的测量质量”作为独立研究问题。作者首先审计一个 MFON 扩展原型，发现范数代理与有效序列长度高度相关，并随高斯扰动增强而上升。随后，作者通过 clean–mild–strong 特征三元组训练视觉和音频可靠性头，保留逐样本 KL/InfoNCE 损失，并在每个批次内按可靠性重新分配辅助权重，同时严格保持平均辅助预算不变。MOSI 用于开发，MOSEI 用于冻结协议复现。结果表明 Learned 相对 Constant 在两个数据集的 MAE、Corr 和 Loss 均值上方向较好，但幅度很小，分类指标混合，且 MOSEI 上没有稳定优于 repaired MFON。跨扰动实验进一步表明，高斯退化识别并未自动迁移到 dropout、shift 或视觉缺失。

四项贡献可以更准确地归纳为：

1. 对样本级质量信号进行粒度、退化单调性、混杂、作用性和权重塌缩审计。
2. 对旧范数代理和过早 batch reduction 的失败进行可复现诊断。
3. 提出带 `stop-gradient` 的逐样本、有限批次精确预算守恒分配。
4. 提供 MOSI 开发证据、MOSEI 冻结协议复现、跨扰动负面结果和效率边界。

论文最有价值的部分是“审计原则 + 精确预算控制 + 负面边界”的组合。排序学习、合成扰动监督、质量调权和样本重加权本身已有大量先例，因此新颖性不应落在单个组件上。

# 4. 相关工作与新颖性核对

当前稿件已经覆盖 QMF、SAM-LML、QA-MoE、MFON 等主要多模态近邻，但仍有两类定位缺口。

第一类是一般样本重加权、课程学习和辅助任务加权。固定总量下重新分配样本损失与元重加权、self-paced/curriculum learning、hard-example mining 有直接概念关系。至少应讨论 [Ren et al., ICML 2018](https://proceedings.mlr.press/v80/ren18a.html) 以及多模态课程学习工作 [Curriculum Learning Meets Weakly Supervised Multimodal Correlation Learning, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.209/)。需要说明本文为什么选择“更可靠样本得到更多辅助监督”，而不是“更困难样本得到更多监督”。

第二类是 2026 年可靠性/缺失模态工作。稿件已经引用 [QA-MoE](https://aclanthology.org/2026.acl-long.1461/)，但未引用 [DEAR: Distributional Error-Aware Reliability for Robust Multimodal Sentiment Analysis with Missing Modalities](https://aclanthology.org/2026.findings-acl.1517/)。DEAR 的目标、输入缺失设定和推理路由与本文不同，但它直接涉及 reliability estimation，必须在相关工作中明确区别“重建/缺失模态可靠性与推理门控”和“本文的训练期辅助损失分配”。

现有最接近的方法边界可写成：

| 工作 | 质量/可靠性信号 | 使用阶段 | 主要目标 | 与本文的关键差异 |
| --- | --- | --- | --- | --- |
| [QMF](https://proceedings.mlr.press/v202/zhang23ar.html) | 不确定性/质量 | 推理融合 | 低质量多模态鲁棒性 | 本文不做推理门控，强调训练期审计与预算控制 |
| [SAM-LML](https://aclanthology.org/2025.emnlp-main.1084/) | 原始/扰动/噪声排序 | 训练与表示学习 | 低质量模态学习 | 本文按样本分配辅助损失并严格守恒预算 |
| [QA-MoE](https://aclanthology.org/2026.acl-long.1461/) | 连续质量谱 | 推理专家路由 | 鲁棒融合 | 本文当前没有专家路由或推理期自适应 |
| DEAR | 分布误差感知可靠性 | 缺失模态恢复/路由 | 缺失模态 MSA | 本文只训练辅助分配，且跨缺失视觉的识别失败 |

当前证据支持“组合式审计与控制框架具有区分度”，不支持“首次提出固定预算质量分配”或“通用质量估计器”的强新颖性表述。

# 5. 预期评审结果

- **建议：Major Revision**
- **会议式总体评分：5/10**
- **信心：4/5**
- **最强正面证据：** 精确预算守恒、对失败的主动审计、冻结开发边界、跨扰动负面结果和谨慎的成本结论。
- **最强负面证据：** 可靠性头只接受同一样本内部的退化排序监督，而实际权重分配比较不同干净样本的绝对分数。
- **第二负面证据：** MOSEI 中 Learned 相对 Constant 的优势很小且指标混合，最终配置的 Permuted/Inverse 对照未完成。

# 6. 主要优点与主要问题

## 6.1 主要优点

1. **研究问题清楚且具有方法论价值。** “一个变量叫 quality，不代表它已经被验证为质量”是一个真实、可迁移的警示。
2. **固定预算控制很强。** 第 5.4 节给出有限批次恒等式，且本版明确 `stopgrad`；这排除了动态方案通过减少辅助监督总量获益的替代解释（p. 8）。
3. **开发与复现边界披露充分。** MOSI 测试诊断参与设计、MOSEI 在冻结协议下运行、checkpoint 仅按 validation loss 选择，均写得清楚（pp. 9–10）。
4. **没有隐藏负面结果。** 视觉 timestep dropout/shift AUROC 接近随机、视觉缺失不稳定、音频扰动对任务影响极弱，这些都被正面报告并写入结论（pp. 13, 16–19）。
5. **任务收益表述克制。** 作者没有宣称普遍优于 MFON，也没有把单次微基准中的推理延迟差异解释为加速。
6. **总体写作成熟。** 主线清楚，限制与主张边界贯穿全文，摘要准确反映正负结果。

## 6.2 M1：样本内排序不保证样本间绝对分数可比较

**严重性：Major，若验证失败可能升级为 Fatal。**

第 5.2–5.3 节的监督只要求同一原始样本满足 `q_clean > q_mild > q_strong`；第 5.4 节却在一个 mini-batch 内用不同干净样本的 `q_i` 归一化并分配辅助权重（pp. 7–8）。共享头会把输出放在同一数值域，但排序损失没有要求不同内容、说话者、长度或情感强度下的分数具有同一标度。

模型完全可能同时做到：每个样本内部正确识别高斯扰动；不同干净样本之间的分数排序主要由内容或长度决定。MOSEI 干净声学分数与长度相关约 0.219、与能量相关约 −0.271（p. 13），说明该风险不是纯理论问题。高 clean/corrupt AUROC 证明的是训练扰动族的区分能力，不能证明 clean-vs-clean 排名适合分配监督。

**必须补的证据：**

- 在不同样本上以相同扰动严重度测试分数标度、校准误差和 rank consistency；
- 做 speaker/content-disjoint 校准，排除说话者或语义内容捷径；
- 直接检验干净 `q_i` 与辅助目标保真度的关系，例如 frozen-teacher 误差、正对可靠性、辅助梯度与验证梯度的一致性，或 leave-one-sample-out 验证贡献；
- 若不成立，将 `q` 改称 Gaussian-degradation response score，并撤回跨样本“可靠性”解释。

## 6.3 M2：最终日程下没有验证分配方向与分数对应关系

**严重性：Major。**

作者在第 5.5–5.6 节正确承认“高可靠性应获得更多辅助监督”是经验假设，并明确写出 final-schedule Inverse/Difficulty-aware 和 Permuted 尚未完成（pp. 8–9）。这份诚实披露不能替代决定性实验。

Learned vs Constant 只能说明非均匀方案与均匀方案产生了不同均值，不能证明改进来自正确的 score–sample correspondence，也不能证明正向分配优于难例优先。早期单种子 Reversed/Permuted 采用不同日程，不能与最终三种子矩阵混用（pp. 11–12）。

**最低修复：** 在完全相同的冻结配置和种子上完成 Constant、Learned、Permuted、Inverse 四组；最好加入基于辅助损失或梯度的 Difficulty-aware。只有 Learned 稳定优于 Permuted 和 Inverse，才能把作用归因于分数对应关系及方向。

## 6.4 M3：三种子效应很小，稳定性证据不足

**严重性：Major。**

MOSEI Learned 相对 Constant 的 MAE 改善 0.0025、Corr 提升 0.0007、Loss 改善 0.0040；Non0 F1、Acc-5 和 Acc-7 反而略差（pp. 12–14）。这些差值接近或小于跨种子标准差。Learned 相对 repaired MFON 的 MAE 仍差 0.0004，Loss 差 0.0010，只在 Corr 上好 0.0009。

论文已经说明三种子不支持显著性，这是正确的；但 “repeatable” 或 “confirmation” 仍容易被理解成稳定效果。Fig. 2 只显示均值与 SD，无法判断每个相同种子的配对方向。

**最低修复：** 报告逐种子完整表和 Learned−Constant 配对差值；用配对点/连线和差值区间替代主要柱状图；至少增加到 5 个独立种子。若资源受限，结论只能保留为 “descriptive mean direction under three seeds”。

## 6.5 M4：相对 MFON 的组件贡献没有完全拆开

**严重性：Major。**

P4 Constant 仍训练 reliability head 和 reliability loss，仅将分配设为均匀；repaired MFON 没有这条训练路径。Learned vs Constant 能隔离分配，但 P4 系列 vs MFON 同时混入可靠性监督、额外参数和共享表征上的梯度影响。

在 MOSEI 上，Constant 的 MAE/Loss 明显差于 MFON，Learned 主要恢复该损失，而不是形成清楚的净提升（p. 13）。因此，需要组件表：repaired MFON；逐样本 loss 实现但无 reliability loss；加 reliability loss 且 Constant；加 Learned。训练曲线还应显示 task loss 与 reliability loss 是否冲突。

## 6.6 M5：外部有效性不足，且基础模型对音视频质量不敏感

**严重性：Major。**

两个数据集都属于英文视频情感分析，使用同一 MFON 骨干及近似特征体系。更关键的是，严重音频/视觉扰动几乎不改变预测，第 7.5 和 8.2 节明确将模型描述为 text-dominant（pp. 12, 17）。在几乎不依赖音视频的模型上研究音视频可靠性分配，会削弱实际意义。

建议至少做一项：

- 在结构不同、音视频贡献更强的第二骨干上复用完整审计；
- 按 modality utility 分层，检查 Learned 的收益是否集中在真正依赖视觉/音频的样本；
- 用同一特征和扰动协议复现一个公开质量感知方法，说明审计结论不是 MFON 特有的实现现象。

## 6.7 M6：训练扰动与“通用质量”概念不匹配

**严重性：Moderate–Major。**

训练只使用特征级 Gaussian corruption，而 held-out timestep dropout、temporal shift 和视觉缺失多项接近随机（p. 13）。这些负面结果非常有价值，但也证明当前头不是一般质量估计器。

有两条合理路线：一是用混合扰动和真实媒体级噪声/遮挡/错位训练并做未见扰动测试；二是把标题、方法名和主张收缩到 “auditing intervention-response scores for fixed-budget auxiliary allocation”。以现有证据，第二条更稳妥。

# 7. 缺失或需要强化的相关工作

1. 一般样本重加权、元重加权、hard-example mining 和 self-paced learning。
2. 多任务/辅助任务加权，尤其是样本维度与任务维度权重的区别及梯度冲突。
3. DEAR 2026，与其 reliability estimation、缺失模态处理和推理期机制做实质对照。
4. SAM-LML 的监督对象、扰动族、排序粒度、预算控制、使用阶段和鲁棒性目标，不应只用一两句概括。
5. QMF/QA-MoE 与本文的“推理期路由 vs 训练期辅助分配”边界需要在一张相关工作表中一次讲清。

# 8. 主张—证据审计

| 主张 | PDF 位置 | 当前证据 | 判断 | 修复 |
| --- | --- | --- | --- | --- |
| 旧范数代理违反质量单调性并受长度混杂 | pp. 10–11 | `r=0.8694`；severity–score 约 `+0.969`；99.85% strongest > clean | 强，限于该代理与协议 | 保留完整样本数和计算方式 |
| 可靠性头识别训练内 Gaussian 退化 | pp. 11, 13, 15 | MOSI/MOSEI 高 AUROC、负 Spearman | 强但范围窄 | 明确称 in-family detection |
| 干净 `q` 可跨样本比较 | pp. 7–8 | 无直接校准 | 缺失，核心前提未验证 | 跨样本校准与目标保真度实验 |
| 固定预算排除了辅助总量变化 | p. 8 | 数学恒等式、`stopgrad`、实现测试 | 强 | 保持现有表述 |
| Learned 优于 Constant | pp. 11–14 | 两数据集三种子均值，部分指标有利 | 描述性支持，稳定性不足 | 逐种子配对与更多种子 |
| 改善来自正确的分数对应关系和方向 | pp. 9, 11–12, 16 | 早期单种子旧日程控制；最终控制缺失 | 不成立 | final-schedule Permuted/Inverse |
| Learned 普遍优于 MFON | pp. 12–13 | MOSEI MAE/Loss 略差，指标混合 | 不支持；正文已基本避免 | 继续保持否定边界 |
| 分数具有跨扰动通用质量含义 | pp. 13, 16–18 | 多个 held-out AUROC 约 0.5 | 被反证 | 收缩命名或扩大训练扰动 |
| 方法实现鲁棒融合 | pp. 12, 17–19 | 无推理门控；A/V 扰动对任务影响小 | 不支持；正文已明确否定 | 不扩大主张 |
| 额外资源成本较小 | pp. 14, 16–17 | 单卡单批 20 次微基准 | 参数/显存结论成立，速度结论局部 | 提供多批次不确定性或维持窄结论 |

# 9. 实验、统计与可复现性审计

## 9.1 实验设计

- MOSI 开发、MOSEI 冻结协议复现的划分清楚，但两个数据集同域，不能代表跨场景泛化。
- Constant 是必要且设计良好的控制；最需要补的是 final-schedule Permuted/Inverse，而不是再添加大量弱基线。
- 若篇幅有限，第二优先级是组件消融和 modality-utility 分层，第三优先级才是更多 leaderboard 模型。

## 9.2 统计报告

- 三个种子只支持描述性结论；建议至少 5 个，条件允许时 10 个。
- 由于方法共享种子与数据顺序，应报告 paired delta，而不应只报告各自 mean±SD。
- 可以给出 paired bootstrap interval，但必须说明 resampling unit 是 seed 还是 example；不要把同一 checkpoint 的 4,659 个测试样本当作 4,659 次独立训练重复。
- Gaussian 多严重度样本来自同一原始样本，可靠性区间应按原始样本或 speaker 聚类。
- Has0/Non0 应在正文明确阈值、零标签处理和 F1 averaging 方式。

## 9.3 可复现性

- 优点：固定种子、validation checkpoint selection、37 项实现测试、padding-preserving corruption 和预算守恒都有说明。
- 不足：PDF 仍说匿名代码包“will be prepared”，因此读者目前无法检查逐种子结果、冻结时间线和配置。
- 匿名工件至少应包括：环境文件、数据获取/预处理说明、所有配置、命令、逐种子日志、表图生成脚本、测试入口和 checkpoint hash；不应重新分发受限数据或凭证。

# 10. 模拟多审稿人评议

## Reviewer A：多模态可靠性建模

- 分数：6/10；信心 4/5。
- 正面：审计框架、预算守恒和负面结果形成了可信的方法论故事。
- 负面：更像测量/审计论文，预测收益不足以支撑强方法论文。
- 提分条件：验证 clean-vs-clean 可比性并补齐最终作用性控制。

## Reviewer B：经验机器学习与统计

- 分数：4/10；信心 5/5。
- 正面：披露完整，未进行显著性夸大。
- 负面：三种子效应小于波动，分类结果混合，配对趋势不可见。
- 提分条件：逐种子配对、更多种子、效应区间。

## Reviewer C：优化与多任务学习

- 分数：5/10；信心 4/5。
- 正面：有限批次预算控制与 `stopgrad` 设计合理。
- 负面：目标保真度假设未验证，难例优先可能同样有效，组件梯度未拆分。
- 提分条件：Inverse/Difficulty、梯度一致性和组件消融。

## Reviewer D：多模态情感分析

- 分数：5/10；信心 4/5。
- 正面：MOSI/MOSEI 协议透明，指标报告完整。
- 负面：MFON 高度文本主导，视听质量信号对任务的重要性偏弱。
- 提分条件：第二骨干或按视听 utility 分层。

## Reviewer E：写作与呈现

- 分数：7/10（仅写作）；信心 5/5。
- 正面：摘要、讨论和局限互相一致，主张谨慎。
- 负面：P4 工程代号偏多；“confirmation”残留；浮动体切断讨论段落；若干表图过密。
- 提分条件：统一术语、重排浮动体、用配对图替代柱状图。

## 评审组共识

固定预算控制和诚实的失败审计构成可发表潜力。决定论文命运的不是再润色一轮，而是补齐这条推理链：

`样本内退化排序` → `样本间干净分数可比` → `分数对应辅助目标保真度` → `正确对应和正向分配优于打乱/反向` → `配对任务收益稳定`

当前第一步和预算合同强，第二至第四步缺失或偏弱，第五步只有小幅描述性证据。

# 11. 问题—行动表

| ID | 严重性 | 位置 | 影响 | 必须行动 | 预期分数影响 |
| --- | --- | --- | --- | --- | --- |
| C1 | Major/Fatal risk | pp. 7–8, 13 | Soundness | 跨样本校准与目标保真度 | 成立可使 Soundness +1；失败需改写核心解释 |
| C2 | Major | pp. 9, 16, 18 | Evidence | final Constant/Learned/Permuted/Inverse | 稳定优于两者可使 Evidence +1 |
| C3 | Major | pp. 11–14 | Statistics | 逐种子配对、区间、≥5 seeds | 可使总体 +0.5–1 |
| C4 | Major | pp. 9, 13 | Attribution | 完整组件消融 | 解释 P4 相对 MFON 的真实来源 |
| C5 | Major | pp. 12, 17–18 | Significance | 第二骨干或 modality-utility 分层 | 可使 Significance +1 |
| C6 | Moderate–Major | 标题、pp. 13, 17–19 | Claim scope | 收缩 quality/reliability 或扩大训练扰动 | 降低过度主张风险 |
| C7 | Moderate | pp. 3, 12, 15, 18 | Consistency | 将残余 confirmation 统一为 frozen-protocol replication | 消除设计措辞争议 |
| C8 | Moderate | Fig. 2, p. 14 | Presentation | 显示每种子点和配对线，避免截断轴放大差异 | 提升统计可读性 |
| C9 | Moderate | pp. 15–16 | Reading flow | 移动 Fig. 4/Table 6，避免切断 Discussion 段落 | 提升专业完成度 |
| C10 | Moderate | Table 5, p. 13；Fig. 1, p. 7 | Readability | 放大表格/标签，减少密度 | 提升印刷可读性 |
| C11 | Moderate | Related Work | Novelty | 加入 DEAR、一般重加权和课程学习 | 降低漏引与新颖性风险 |
| C12 | Submission blocker | p. 19 | Compliance | 完成声明与匿名工件，改为事实时态 | 清除投稿材料阻塞 |
| C13 | Minor | pp. 20–23 | Format | 去除 DOI/URL 重复，压缩参考文献留白 | 改善版面 |

# 12. AC / Meta-review

这篇论文比常见的“增加质量头并报告小幅提升”更可信。作者愿意公开开发污染、跨扰动失败、文本主导和不显著性边界，并用严格预算守恒消除一个常被忽视的混杂。这些做法本身值得肯定，也使审计型定位具有发表价值。

然而，核心方法当前存在测量对象与使用对象不一致的问题：训练目标证明的是同一样本在不同 Gaussian 严重度下的排序，分配机制需要的却是不同干净样本之间的绝对排序。论文没有证明高 `q` 的样本拥有更可信的 teacher target 或 contrastive positive，也没有在最终日程下证明正向、正确对应的分配优于 inverse/permuted。因而，当前不能把小幅任务差异可靠归因于“可靠性驱动分配”。

我的建议是大修。若作者完成 C1–C3，即使最终发现跨样本可靠性或正向分配不成立，这篇工作仍可能以高质量负面研究和审计框架成立，但必须相应收缩方法命名与结论。如果作者只做语言和格式修改，科学评分不会改变。

# 13. 定量评分

| 维度 | 分数（1–5） | 信心 | 主要依据 | 提升条件 |
| --- |:---:|:---:| --- | --- |
| Novelty | 3 | 4 | 审计与预算控制组合有区分度；单个组件已有先例 | 完整近邻定位与第二系统验证 |
| Soundness | 2 | 4 | 预算数学正确；跨样本标度和目标保真度未验证 | C1、C2 成立可升至 3–4 |
| Evidence | 3 | 5 | 两数据集、三种子、压力测试；决定性控制缺失 | final controls、配对统计 |
| Significance | 3 | 4 | 审计问题重要；任务增益和适用域较窄 | 第二骨干或强 utility 子集 |
| Clarity | 4 | 5 | 叙事和限制清楚；术语与浮动体仍有问题 | 统一措辞并修复版面 |
| Reproducibility | 3 | 4 | 协议和实现检查充分；匿名工件尚未提供 | 发布可运行匿名包与逐种子输出 |
| Ethics/Limitations | 5 | 5 | 数据、用途和局限披露充分 | 保持现有边界 |

- **总体：5/10**
- **建议：Major Revision / Borderline Negative**
- **审稿信心：4/5**

升至 6/10 的最低条件是：C1 不失败，且最终配置下 Learned 在配对结果中稳定优于 Permuted 和 Inverse。升至 7/10 还需要更充分种子以及第二骨干、外部方法或 modality-utility 子集中的可迁移证据。

# 14. 给作者的决定性问题

1. 只用样本内排序损失时，两个不同干净样本的绝对 `q` 为什么具有可比意义？
2. 干净 `q` 是否与 frozen-teacher 误差、contrastive positive 质量或梯度一致性相关？
3. 最终 P4 schedule 下，Learned 是否在每个种子上优于 Permuted 和 Inverse？
4. P4 Constant 相对 MFON 的退化来自 reliability loss、额外参数、逐样本 loss 修复，还是训练路径冲突？
5. Learned 的收益是否集中在真正依赖视觉/音频的样本，而不是文本足以完成预测的样本？
6. 若混合扰动训练后仍无法跨腐败泛化，作者是否愿意把核心对象改称 intervention-response score？
7. “frozen” 的时间戳和不可变配置能否随匿名工件公开？
8. 为什么 Fig. 2 不直接展示九项指标的逐种子配对差值？

# 15. 分数修订标准

## 提高分数

- 跨样本校准及辅助目标保真度证据支持 `q` 的实际使用语义；
- final-schedule Learned/Constant/Permuted/Inverse 多种子对照闭合因果解释；
- 逐种子配对结果表明改善不是由单个 seed 驱动；
- 第二骨干、外部方法或高 modality-utility 子集复现关键结论；
- 统一 frozen-protocol 术语，完成声明与匿名工件。

## 降低分数或明确拒绝

- 干净 `q` 与辅助目标保真度无关，或主要由长度/内容混杂决定；
- Permuted/Inverse 与 Learned 无稳定差异；
- 逐种子结果显示均值由单个异常 seed 驱动；
- 最终稿仍把 Gaussian 检测包装成通用质量估计或鲁棒融合；
- 声明或匿名工件在投稿时仍为占位状态。

# 16. 修改优先级、写作与排版检查

## P0：决定科学结论是否成立

1. 做跨样本 `q` 校准和目标保真度实验。
2. 完成 final-schedule Constant/Learned/Permuted/Inverse 多种子矩阵。
3. 公开逐种子配对结果、区间与完整指标。

## P1：决定贡献是否足够

4. 做 MFON → per-sample loss → +reliability loss Constant → +Learned 的组件消融。
5. 在第二骨干或高视听 utility 子集上验证。
6. 补 DEAR、一般样本重加权、课程学习和辅助损失加权定位。

## P2：投稿前必须修复

7. 把 p. 3、p. 12、p. 15、p. 18 残留的 `confirmation` 统一改为 `frozen-protocol replication` 或 `replication`。
8. 重画 Fig. 2：显示逐种子 paired points/lines；当前截断纵轴的柱状图会放大 0.001–0.004 量级差异。
9. 修复 pp. 15–16 的阅读流：Discussion 段落在 p. 15 末尾被 Fig. 4 与 Table 6 整页插入打断，到 p. 16 底部才续接。
10. 放大 p. 13 的 Table 5 和 p. 7 的 Fig. 1；两者在正常页面尺寸下偏密。
11. 完成 Funding、Competing interests、Author contributions，并将 Data/Code Availability 改为已经落实的事实陈述。
12. 清理参考文献中的 DOI/URL 重复；p. 23 只有两条文献且留白很大，可通过 bibliography 设置改善。

## 写作质量评分

| 写作维度 | 评分（1–5） | 评语 |
| --- |:---:| --- |
| Storyline | 4 | 问题—失败审计—修复—边界清晰 |
| Contribution clarity | 4 | 贡献边界较诚实，但 P4 工程代号和 broad quality 命名略模糊 |
| Paragraph logic | 4 | 段落推进良好；浮动体造成一次严重断裂 |
| Claim–evidence alignment | 3 | 大多数边界准确；跨样本 reliability 仍强于证据 |
| Method explanation | 4 | 公式完整，`stopgrad` 已解释 |
| Experiment narration | 4 | 协议透明；逐种子配对未可视化 |
| Related work positioning | 3 | 多模态近邻较好；一般重加权与 DEAR 缺失 |
| Terminology consistency | 3 | frozen-protocol 与 confirmation 混用 |
| Figure/table readability | 3 | Fig. 1、Table 5 偏密；Fig. 2 表达选择需改 |
| Reviewer-facing risk control | 4 | 主动披露局限；投稿声明仍未完成 |

## 已执行检查

- 完成 23 页逐页渲染与视觉审查；未发现裁切、叠字、缺图或不可读字体。
- PDF 未加密，无表单或脚本；字体均嵌入/子集化；元数据未暴露作者身份。
- LaTeX 日志未见 undefined citation/reference；存在多处 underfull box 和一个 bookmark-level warning，主要影响版面整洁而非正确性。
- 数字抽查覆盖 MOSI/MOSEI 主结果、跨扰动表、效率表及正文差值；未发现改变结论的数值矛盾。
- 未执行：重新训练 checkpoint、从原始数据独立复算全部指标、验证尚未提供的匿名代码包。

## 未解决的核心风险

跨样本可靠性可比性、辅助目标保真度、最终日程作用性、三种子稳定性、第二骨干可迁移性。这五项不能通过语言润色替代。
