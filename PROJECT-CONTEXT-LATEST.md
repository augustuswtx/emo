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

当前诚实结论：

- Learned相对Constant在二分类、Acc-5、MAE、Corr和Loss上更好，但Acc-7较低。
- Learned相对Reversed在除Corr外的主要指标上更好，Acc-7 +0.0204，MAE -0.0222。
- Learned相对Permuted在二分类、MAE、Corr和Loss上更好，Permuted在Acc-5/7上更好。
- 单种子证据总体支持主要回归/二分类目标，但不支持“所有指标稳定提升”。

## 7. 当前立即任务

Oracle两轮烟雾训练已通过并保存checkpoint。第2轮真实日志为：

```text
q_v=0.934053, q_a=0.935742
q_v_std=0.074186, q_a_std=0.073381
w_v=w_a=w_nce_v=w_nce_a=0.1
w_v_std=0.007948, w_a_std=0.007847
```

当前先测试Oracle烟雾checkpoint的加载和推理：

```bash
cd /home/jovyan/projects/MFON

python run_experiment.py \
  --dataset MOSI \
  --stage test-fusion \
  --seed 1111 \
  --use-budgeted-aux \
  --use-interventional-reliability \
  --warmup-epoch 10 \
  --reliability-task-warmup-epoch 10 \
  --reliability-allocation-control oracle \
  --exp-name p2_oracle_smoke \
  2>&1 | tee mosi_p2_oracle_smoke_test_1111.log
```

加载通过后，再运行25轮`interventional_rel_p2_oracle`。

## 8. Oracle之后的决策

1. 完成Oracle后，判断作用性证据是否总体成立。
2. 若成立，在MOSI seeds 1112/1113上优先复现Learned和最有信息量的对照，再决定是否扩展到5种子。
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
