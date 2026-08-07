# MFON多模态情感分析项目上下文

> 最后更新：2026-08-06
>
> 本文件是新Codex对话的首要入口。不要从零开始，不要把未完成实验写成已证明结论。

## 1. 项目位置

- 本地工作区：`/Users/augustus/projects/论文`
- 本地代码：`/Users/augustus/projects/论文/MFON`
- 服务器代码：`/home/jovyan/projects/MFON`
- 当前数据集：CMU-MOSI
- 基础模型：MFON（COLING 2025）
- 当前方法暂称：审计驱动的干预式可靠性预算学习
- 旧名`Q-DAMFON`已停用；不把MFON基础结构宣称为原创。

## 2. 当前方法

1. 视觉和音频分别学习逐样本可靠性。
2. 为每个样本生成clean/mild/strong人工退化三联对，使可靠性分数随退化强度下降。
3. 音频使用专用时序头，描述相邻帧差分、差分能量、滞后相关和峰度。
4. 将MFON的KL蒸馏和InfoNCE保留为逐样本损失。
5. 每个mini-batch的辅助监督平均预算严格固定，仅根据可靠性重新分配给不同样本。
6. 可靠性头从第1轮学习完整退化，主情感任务所见退化在10轮内渐进增强。
7. 作用性对照保持模型、退化、可靠性监督和平均预算一致，只改变分数与样本的对应方式。

## 3. 已完成修复

- 去掉冻结编码器输出的`.squeeze()`，防止batch size 1丢维。
- `position_embedding.py`改用`torch.arange`生成离散位置，修复`index_select`越界。
- 固定预算、可靠性、退化审计与P2对照共25项CPU测试通过。

## 4. 主要基线

MOSI seed 1111修复版MFON：

```text
Has0 Acc-2=0.8265, Has0 F1=0.8253
Non0 Acc-2=0.8476, Non0 F1=0.8471
Acc-5=0.5029, Acc-7=0.4388
MAE=0.7279, Corr=0.7959
```

修复版MFON三种子：

```text
Has0 Acc-2=0.8270±0.0009
Acc-7=0.4505±0.0125
MAE=0.7258±0.0028
Corr=0.7943±0.0015
```

## 5. 可靠性审计

25轮 learned 最佳checkpoint，MOSI完整测试集686样本：

```text
vision: Spearman=-0.965883, AUROC=0.998882,
        highest_below_clean=0.998542
audio:  Spearman=-0.815835, AUROC=0.945113,
        highest_below_clean=0.998542
```

混淆相关：

```text
corr(q_v, vision_length)= 0.087347
corr(q_v, vision_energy)= 0.007884
corr(q_a, audio_length)= 0.252213
corr(q_a, audio_energy)=-0.024787
```

结论：视觉/音频可靠性均能稳定识别人工退化；能量捷径基本消失，但音频长度相关0.252需要跨数据集复核。

## 6. MOSI seed 1111正式结果

| 预算分配 | Has0 Acc-2 | Has0 F1 | Non0 Acc-2 | Non0 F1 | Acc-5 | Acc-7 | MAE↓ | Corr↑ | Loss↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Learned | 0.8353 | 0.8345 | 0.8552 | 0.8550 | 0.5117 | 0.4446 | 0.7171 | 0.7972 | 0.9567 |
| Constant | 0.8309 | 0.8300 | 0.8491 | 0.8487 | 0.5087 | 0.4504 | 0.7221 | 0.7900 | 0.9845 |
| Reversed | 0.8338 | 0.8335 | 0.8521 | 0.8524 | 0.4985 | 0.4242 | 0.7393 | 0.7993 | 1.0266 |
| Permuted | 0.8294 | 0.8284 | 0.8476 | 0.8471 | 0.5219 | 0.4577 | 0.7190 | 0.7940 | 0.9812 |
| Oracle | 0.8338 | 0.8326 | 0.8537 | 0.8531 | 0.5146 | 0.4548 | 0.7164 | 0.7923 | 0.9769 |

当前诚实结论：

- Learned相对Constant在二分类、Acc-5、MAE、Corr和Loss上更好，但Acc-7较低。
- Learned相对Reversed在除Corr外的主要指标上更好，Acc-7 +0.0204，MAE -0.0222。
- Learned相对Permuted在二分类、MAE、Corr和Loss上更好，Permuted在Acc-5/7上更好。
- Learned相对Oracle在二分类、Corr和Loss上更好；Oracle在Acc-5/7和MAE上更好。
- 单种子证据总体支持主要回归/二分类目标，但作用性证据呈现指标权衡，不支持“所有指标稳定提升”。

## 7. 当前立即任务

MOSI seed 1112的Learned正式实验已完成，结果未复现seed 1111的提升：

| Method, seed 1112 | Has0 Acc-2 | Has0 F1 | Non0 Acc-2 | Non0 F1 | Acc-5 | Acc-7 | MAE | Corr | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Repaired baseline | 0.8265 | 0.8258 | 0.8460 | 0.8459 | 0.4985 | 0.4490 | 0.7269 | 0.7941 | TBD |
| Learned | 0.8163 | 0.8168 | 0.8247 | 0.8257 | 0.4636 | 0.4111 | 0.7802 | 0.7855 | 1.0828 |

Learned在全部已知指标上均差于baseline，属于实质性跨种子失败，
不应解释为普通随机波动。完整686样本可靠性审计显示：视觉
Spearman=-0.963491、AUROC=0.994929；音频Spearman=-0.789313、AUROC=0.916853。
可靠性头仍能稳定识别退化，因此当前问题定位为可靠性动态分配的优化效果不稳定。
下一个诊断实验是seed 1112 Constant；暂不运行Permuted。

seed 1112 Constant已完成：Has0 Acc-2=0.8105、Has0 F1=0.8110、Non0 Acc-2=0.8186、
Non0 F1=0.8196、Acc-5=0.4723、Acc-7=0.4155、MAE=0.7695、Corr=0.7870、
Loss=1.0607。Constant相对Learned在MAE、Corr、Loss和Acc-5/7上更好，但二分类更差；
两者均明显差于seed 1112 repaired baseline。因此动态分配不是唯一问题，
主任务特征混入人工退化是当前首要待验证原因。

已新增 `--reliability-task-corrupt-scale`。设为0时，可靠性头仍在干净/退化对上学习，
但情感主任务始终使用干净音视频。本地27项测试全部通过。下一实验是
seed 1112 Constant + clean task path，用于严格隔离主任务污染效应。

P3 Constant + clean task path正式结果为：Has0 Acc-2=0.8134、Has0 F1=0.8139、
Non0 Acc-2=0.8216、Non0 F1=0.8227、Acc-5=0.4752、Acc-7=0.4169、
MAE=0.7708、Corr=0.7869、Loss=1.0607。该结果与P2 Constant几乎一致，
说明主任务污染不是主要原因。

旧“固定预算”warmup在前10轮将辅助权重均值从0.05递增到0.5，而
repaired baseline从第1轮就使用0.5，因此并非真正的等预算对照。已新增
`--budget-warmup-mode allocation`：总预算始终为0.5，只将样本间分配从均匀逐步过渡到
可靠性感知。历史默认 `scale` 保持不变；本地29项测试通过。这是训练损失
分配路线的最后一轮修复。

Oracle两轮烟雾训练、checkpoint保存、重新加载和完整测试均已通过。第2轮真实日志为：

```text
q_v=0.934053, q_a=0.935742
q_v_std=0.074186, q_a_std=0.073381
w_v=w_a=w_nce_v=w_nce_a=0.1
w_v_std=0.007948, w_a_std=0.007847
```

完整测试结果为：Has0 Acc-2=0.8192、Has0 F1=0.8193、Non0 Acc-2=0.8415、
Non0 F1=0.8422、Acc-5=0.4723、Acc-7=0.3892、MAE=0.8588、Corr=0.7383、
Loss=1.3551。这些是工程烟雾结果，不与25轮正式结果作论文结论。

匹配的25轮Oracle正式对照已完成：Has0 Acc-2=0.8338、Has0 F1=0.8326、
Non0 Acc-2=0.8537、Non0 F1=0.8531、Acc-5=0.5146、Acc-7=0.4548、
MAE=0.7164、Corr=0.7923、Loss=0.9769。

已清理smoke checkpoint，可用磁盘空间恢复到6.7GB；seed 1112/1113的四个
单模态encoder checkpoint均存在。

## 8. Oracle之后的决策

1. seed 1111作用性证据呈现指标权衡，总体有利但不足以单独形成稳定结论。
2. 在MOSI seeds 1112/1113上优先复现Learned和Permuted对照，再决定是否扩展到5种子。
3. 然后扩展MOSEI和SIMS，完成跨数据集、鲁棒性、消融、效率和统计显著性。
4. 小论文当前不可宣称已达到CCF-B投稿证据要求。

## 9. 关键文件

- `small-paper-draft-v1.md`：小论文草稿
- `small-paper-references.bib`：真实参考文献元数据
- `experiment-log.md`：真实实验日志
- `experiment-plan-v4.md`：实验证据计划
- `method-evolution-and-ownership-20260729.md`：方法演化和原创边界
- `HANDOFF-SERVER-EXPERIMENTS-20260731.md`：旧服务器交接，历史还原用
- `MFON/`：本地代码，不应包含数据集、checkpoint、大日志或`__pycache__`

## 10. 数据与完整性边界

- 只使用用户实际运行的数字。
- 不伪造平均值、显著性、SOTA排名或跨数据集结果。
- 不将MFON、SAM-LML、QMF等已有思想宣称为本文首创。
- 不上传MOSI/MOSEI/SIMS数据、BERT权重、服务器checkpoint、密钥或个人附件。
