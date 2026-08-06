# 检索记录

## 范围

- 检索日期：2026-07-20
- 主题：多模态情感分析、低质量/缺失/噪声模态、可靠性估计、动态融合、样本级贡献、课程学习、提示学习、可信度诊断
- 时间重点：2023--2026，同时保留 2022 年鲁棒性诊断作为基础锚点
- 优先来源：ACL Anthology、PMLR、CVF Open Access、AAAI、IJCAI、arXiv 原始页面

## 代表性查询

1. `multimodal sentiment reliability fusion low-quality missing noisy modality`
2. `sample-level modality valuation uncertainty dynamic fusion`
3. `quality-aware mixture of experts multimodal sentiment 2026`
4. `decision-level dependence reliability score permutation multimodal`
5. `counterfactual reliability alignment multimodal sentiment`
6. `temporal segment-level modality reliability multimodal sentiment`
7. `quality-aware knowledge distillation sample weighted multimodal`
8. `conformal prediction multimodal sentiment MOSI MOSEI`

## 纳入规则

- 与 MOSI、MOSEI、SIMS 或一般低质量多模态学习直接相关。
- 优先顶会/主流档案中的正式论文；预印本单独标注，不把预印本当作已录用顶会工作。
- 能够改变本项目创新性判断、基线选择或实验协议的论文优先。

## 排除或降权规则

- 仅有搜索聚合页、无法核验原文的条目不作为主要证据。
- 与情感分析距离较远、只有通用“注意力/融合”关键词的论文不进入核心表。
- 未来刊期、低可信转载或元数据冲突的结果不用于关键判断。

## 仍需精读的问题

1. QA-MoE、EBMC、CPSC 对质量/可靠性的监督来源和推理路径有何具体差异。
2. SAM-LML 的排序监督与“质量代理有效性审计”之间还有多少可区分空间。
3. 2026 leakage-safe diagnostic 的分数置换协议能否扩展到训练阶段梯度与辅助损失。
4. MRUF 的留一贡献监督是否已经同时区分可靠性和任务效用。
5. 各论文是否公开代码、是否能在同一数据划分和特征上公平复现。

## 边界声明

检索无法证明“世界上不存在相同想法”。当前结论是：通用质量加权路线已经高度拥挤，原 Q-DAMFON 的创新性主张风险很高；新主张必须在进一步精读和代码复现后才能确定。

