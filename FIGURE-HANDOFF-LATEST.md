# MFON/P4 小论文图表制作交接（最新版）

更新时间：2026-08-16

## 1. 任务目标

为当前多模态情感分析小论文制作可投稿 Neural Computing and Applications（NCA，当前
工作目标为 CCF-C）的学术图表。不要从零开始，不要改动已冻结的方法和超参数。先阅读
本文件及下列权威项目文件，再设计、生成、视觉检查并集成图表：

- `docs/small-paper-draft-v2-en.md`：英文完整初稿；
- `docs/small-paper-draft-v2-zh.md`：中文对应稿；
- `paper/springer-nca/body.tex`：NCA LaTeX 正文；
- `docs/experiment-log.md`：实验事实与数值来源；
- `MOSEI-HANDOFF-LATEST.md`：MOSEI 实验实时边界；
- `docs/reference-verification-20260811.md`：引用核验状态；
- `paper/springer-nca/submission-status.md`：投稿缺口。

开始工作时使用 `nature-figure` skill。所有定量图必须由已核验数据或用户随后提供的正式
结果生成；没有原始逐样本数据时不得伪造散点、置信区间、分布或误差条。

## 2. 论文的真实技术故事

论文不是图像模型，也不声称推理阶段动态鲁棒融合。基础模型 MFON 属于既有工作。本项目
的增量是一个面向训练阶段辅助监督的审计与控制框架：

1. 审计模态质量分数的样本粒度、退化单调性、混杂敏感性、作用性和非坍缩；
2. 由 clean--mild--strong 有序干预学习视觉和音频可靠性；
3. 在损失归约前保持逐样本辅助损失；
4. 在每个有限 batch 内严格保持辅助监督平均预算；
5. 用 P4 Constant 与 P4 Learned 区分“总预算增加”和“样本分配改变”。

结论边界：现有证据支持“经过审计的可靠性可指导训练阶段固定预算辅助监督分配”，不支持
全面性能提升、统计显著性、通用可靠性或推理阶段抗噪融合。

## 3. 当前已完成证据

### MOSI

- 修复版 MFON、P4 Constant、P4 Learned 的 seeds 1111/1112/1113 公平比较已冻结；
- P4 Learned 三种子可靠性审计已完成；
- 论文现有表 1 和表 2 中的均值与样本标准差可以用于正式图表；
- Learned 相对 Constant 的均值优势集中在二分类、MAE、Corr 和 Loss；Acc-5 基本相同，
  Acc-7 低 0.0015，不能画成“全面提升”。

### MOSEI

- seed 1111 音频和视觉 encoder 已完成；
- 两轮 P4 Learned smoke 的训练、保存、重载和完整测试门已通过；
- 两轮 smoke 指标仅是工程验证，严禁进入论文性能图；
- 正式 seed 1111 repaired baseline 已完成：Has0 Acc-2=0.8150、Has0 F1=0.8205、
  Non0 Acc-2=0.8635、Non0 F1=0.8634、Acc-5=0.5426、Acc-7=0.5267、
  MAE=0.5372、Corr=0.7721、Loss=0.500200593969192；
- seed 1111 P4 Constant 正在服务器训练，实验名
  `p5_mosei_p4_constant_true_budget`；P4 Learned 尚未启动；
- 在 Constant 和 Learned 均完成前，不制作 MOSEI 方法比较图。

## 4. 图表优先级

### F1：方法总览图（现在即可制作，最高优先级）

建议表现：文本/视觉/音频输入与冻结单模态 encoder；视觉和音频的
clean--mild--strong 干预；可靠性头输出逐样本 `q_v`、`q_a`；精确 batch 预算归一化；
逐样本蒸馏和对比辅助损失；干净特征进入 MFON 主任务融合路径。

必须清楚区分：可靠性控制训练辅助损失，不直接控制推理融合权重。不得暗示测试时动态
切换模态。

### F2：五部分质量审计框架（现在即可制作）

用紧凑流程图或矩阵展示：sample granularity、degradation monotonicity、confound
sensitivity、actionability、non-collapse。可同时标出早期原型的三项已证实失败：batch
聚合、损失提前归约、范数代理在强扰动下反向增大。

### F3：MOSI 可靠性审计结果（取决于可用数据）

若只有论文表 1 的汇总值，制作带明确均值与样本标准差的点图或小多图；不得生成看似逐样本
的曲线或分布。若仓库或用户提供逐严重度原始审计文件，再制作严重度--可靠性曲线、AUROC
或 confound 诊断图。

### F4：MOSI 匹配方法比较（现在即可制作）

优先使用小多图或方向统一的 Learned-minus-Constant 差值图，避免把 Acc、MAE、Corr、
Loss 混在同一纵轴。图注必须说明三种子、均值和样本标准差；对 MAE/Loss 使用“越低越好”
方向。不得隐藏 Acc-7 的负向差异。

### F5：MOSEI 跨数据集比较（暂缓）

只有 seed 1111 baseline、Constant、Learned 三组正式重载测试全部完成后，才建立第一版
MOSEI 比较表/图；只有多种子完成后，才画均值和标准差。严禁使用两轮 smoke 值补空位。

## 5. 交付规范

- 建议新建 `paper/figures/`，每张图同时保留可编辑源文件、生成脚本和数据文件；
- 优先输出矢量 PDF/SVG，并另存 300 dpi PNG 预览；文字和线条不得栅格化；
- 采用色盲友好配色，保证灰度可区分，避免红绿作为唯一编码；
- 正文字号缩放后仍清晰，设计时同时检查单栏与双栏宽度；
- 图内术语与中英文稿术语账本一致；论文正式图以英文为主，可额外生成中文说明版；
- 每张定量图附数据来源、生成命令和图注草稿；
- 视觉 QA 后再插入 `paper/springer-nca/body.tex`，不得只生成图片而不核对版面；
- 不提交服务器绝对路径、checkpoint、日志、数据集、密钥或个人信息。

## 6. 第一轮工作顺序与完成门槛

1. 阅读上述文件，建立图表清单和版式草图；
2. 先完成 F1 方法总览，并输出英文投稿版和中文审阅版；
3. 再完成 F2；
4. 从论文已冻结表格建立独立 CSV，完成 F3/F4 中不依赖缺失原始数据的部分；
5. 对每张图做尺寸、字体、颜色、数值和结论边界检查；
6. 集成进 NCA LaTeX，并更新中英文 Markdown 图注；
7. 提交前运行数字一致性检查并通过 GitHub Desktop 同步。

第一轮通过条件：F1/F2 至少有可审阅成稿；所有使用数值的图均能追溯到
`docs/experiment-log.md` 或冻结论文表；没有使用 smoke 指标；没有暗示推理阶段可靠性融合；
文件可编辑、可复现且适合 NCA 版面。

## 7. 项目同步状态

本交接创建前的已同步主分支提交为：

```text
2755085 Record MOSEI seed-1111 baseline
```

本项目没有 `ccfa.yaml`，当前以 `PROJECT-CONTEXT-LATEST.md`、
`MOSEI-HANDOFF-LATEST.md` 和本文件作为阶段状态来源。
