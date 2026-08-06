# MFON可靠性预算学习服务器实验交接（2026-07-31）

请直接接管这个项目，不要从零开始设计，也不要把未完成结果写成论文结论。用户是零基础，请使用通俗中文，并给出可以直接运行的完整命令。

## 一、项目与当前目标

- 服务器项目目录：`/home/jovyan/projects/MFON`
- 当前数据集：CMU-MOSI
- 当前基础模型：MFON（COLING 2025）
- 当前方法暂称：审计驱动的干预式可靠性预算学习（旧名Q-DAMFON已停用）
- 当前实验目标：证明学习到的视觉/音频可靠性不仅能识别退化，而且在固定辅助监督总预算下，正确分数分配优于常数、错配和反转分数。
- 当前不是投稿完成状态。只有一个正式种子结果，不得声称达到CCF-B标准或稳定优于MFON。

## 二、已完成的重要修复

1. `model.py`取消冻结编码器输出的`.squeeze()`，避免batch size为1时维度丢失。
2. `position_embedding.py`改用`torch.arange`生成离散位置，避免连续特征触发`index_select`越界。
3. KL和InfoNCE改为逐样本损失后再加权。
4. 每个mini-batch内固定辅助监督平均预算，避免通过整体缩小辅助权重获得虚假收益。
5. 视觉使用通用时序统计可靠性头；音频使用专门的时序差分、差分能量和滞后相关可靠性头。
6. 可靠性头从首轮观察完整clean/mild/strong退化，主情感任务所见退化在10轮内渐进增强。

## 三、已有真实结果

### 1. 修复版MFON，MOSI seed 1111

```text
Has0 Acc-2=0.8265, Has0 F1=0.8253
Non0 Acc-2=0.8476, Non0 F1=0.8471
Acc-5=0.5029, Acc-7=0.4388
MAE=0.7279, Corr=0.7959
```

修复版MFON三种子均值：

```text
Has0 Acc-2=0.8270±0.0009
Acc-7=0.4505±0.0125
MAE=0.7258±0.0028
Corr=0.7943±0.0015
```

### 2. P1失败结论

- 通用视觉头pilot AUROC约0.981。
- 通用音频头pilot AUROC约0.501，近似随机。
- 过早使用完整主任务扰动导致2轮MAE 1.1387、Corr 0.6365。
- P1已淘汰，不运行25轮。

### 3. P1.1两轮与全量审计

两轮clean测试（686样本）：

```text
Has0 Acc-2=0.8309, Acc-7=0.4038
MAE=0.7881, Corr=0.7584
```

两轮checkpoint完整686样本退化审计：

```text
vision: Spearman=-0.944177, AUROC=0.970536,
        highest_below_clean=0.998542
audio:  Spearman=-0.555608, AUROC=0.853704,
        highest_below_clean=0.928571
```

混淆相关：

```text
corr(q_v, vision_length)=-0.044788
corr(q_v, vision_energy)=-0.139763
corr(q_a, audio_length)= 0.213261
corr(q_a, audio_energy)=-0.110135
```

### 4. P1.1 25轮 learned 分配，seed 1111

实验名：

```text
interventional_rel_temporal_warmup_pilot
```

第25轮日志：

```text
mean w_v/w_a/w_nce_v/w_nce_a=0.5
q_gap_v=0.175127
q_gap_a=0.145695
loss_rank=0.003032
task_corruption_progress=1.0
```

最佳验证checkpoint的完整clean测试：

```text
Has0_acc_2=0.8353
Has0_F1_score=0.8345
Non0_acc_2=0.8552
Non0_F1_score=0.8550
Mult_acc_5=0.5117
Mult_acc_7=0.4446
MAE=0.7171
Corr=0.7972
Loss=0.956682082428529
```

相对seed 1111修复版MFON，八项指标全部向好，但仍然只是单种子pilot。

## 四、当前立即要做的事

服务器目前应仍是P1.1代码。先审计25轮最佳checkpoint，不要先上传P2补丁：

```bash
cd /home/jovyan/projects/MFON

python audit_model_quality.py \
  --dataset MOSI \
  --seed 1111 \
  --exp-name interventional_rel_temporal_warmup_pilot \
  --q-type learned \
  --split test \
  2>&1 | tee mosi_interventional_rel_temporal_warmup_pilot_audit_full_1111.log
```

提取结果：

```bash
grep -E \
"^Audit |^clean metrics|^clean confounds|^vision audit|^audio audit" \
mosi_interventional_rel_temporal_warmup_pilot_audit_full_1111.log
```

通过条件：

1. 视觉、音频Spearman均为负；
2. 视觉AUROC保持约0.90以上；
3. 音频AUROC明显高于0.50，优先要求约0.75以上；
4. 无明显长度或能量捷径；
5. checkpoint正常加载，clean指标与上面的25轮测试一致。

若审计失败，停止P2并诊断。若通过，进入下一节。

## 五、P2等预算作用性补丁

原始Mac上的补丁：

```text
/Users/augustus/projects/论文/mfon_actionability_controls_p2_20260730.tar.gz
```

SHA256：

```text
91307163f8767afce7b2eaee1c9c3a8d6011113593ebbf02ef8319b1e9b7c689
```

用户需要把该文件上传到服务器：

```text
/home/jovyan/projects/mfon_actionability_controls_p2_20260730.tar.gz
```

确认和解压：

```bash
cd /home/jovyan/projects/MFON
sha256sum ../mfon_actionability_controls_p2_20260730.tar.gz
tar -xzf ../mfon_actionability_controls_p2_20260730.tar.gz
python -m unittest discover -s tests -v
```

必须显示：

```text
Ran 25 tests
OK
```

补丁提供：

```text
--reliability-allocation-control learned
--reliability-allocation-control constant
--reliability-allocation-control permuted
--reliability-allocation-control reversed
--reliability-allocation-control oracle
```

这些设置保持网络、退化日程、可靠性监督以及KL/InfoNCE平均预算一致，只改变辅助监督分给哪些样本。

## 六、先做两轮constant烟雾测试

```bash
python run_experiment.py \
  --dataset MOSI --stage train-fusion --seed 1111 --epochs 2 \
  --use-budgeted-aux --use-interventional-reliability \
  --warmup-epoch 10 --reliability-task-warmup-epoch 10 \
  --reliability-allocation-control constant \
  --exp-name p2_constant_smoke \
  2>&1 | tee mosi_p2_constant_smoke_1111.log
```

通过条件：

1. 无Traceback、NaN、Inf；
2. `q_v_std=q_a_std=0`；
3. 第2轮四类平均权重均为0.1；
4. `w_v_std=w_a_std=0`；
5. checkpoint保存并能测试。

测试命令：

```bash
python run_experiment.py \
  --dataset MOSI --stage test-fusion --seed 1111 \
  --use-budgeted-aux --use-interventional-reliability \
  --warmup-epoch 10 --reliability-task-warmup-epoch 10 \
  --reliability-allocation-control constant \
  --exp-name p2_constant_smoke \
  2>&1 | tee mosi_p2_constant_smoke_test_1111.log
```

烟雾测试指标只检查路径，不与25轮模型直接比较。

## 七、25轮作用性对照顺序

按成本和信息量排序：

1. `constant`
2. `reversed`
3. `permuted`
4. `oracle`

模板：

```bash
python run_experiment.py \
  --dataset MOSI --stage train-fusion --seed 1111 --epochs 25 \
  --use-budgeted-aux --use-interventional-reliability \
  --warmup-epoch 10 --reliability-task-warmup-epoch 10 \
  --reliability-allocation-control CONTROL \
  --exp-name interventional_rel_p2_CONTROL \
  2>&1 | tee mosi_interventional_rel_p2_CONTROL_1111.log
```

将`CONTROL`分别替换为`constant`、`reversed`、`permuted`、`oracle`。每次训练完成后使用完全相同的参数运行`--stage test-fusion`。

## 八、决策规则

- learned必须至少在主要指标整体上优于constant，否则“可靠性分配有效”不成立。
- learned应优于reversed/permuted，否则分数与样本的正确对应关系没有被证明。
- Oracle应提供合理正对照；若Oracle也无效，说明固定预算分配对当前模型可能没有足够作用。
- 单个指标小幅领先不算通过，要看Acc-2/F1、Acc-7、MAE、Corr整体方向。
- 完成作用性对照前不要跑seed 1112/1113，不要扩展MOSEI/SIMS。

## 九、操作纪律

1. 每次先检查磁盘：`df -h .`。
2. 长训练放入`tmux`。
3. 每个实验使用不同`exp-name`和日志名，不覆盖已有checkpoint。
4. 当前程序默认只保存验证集最佳checkpoint，日志第25轮不等于最终测试checkpoint来自第25轮。
5. 服务器重启会终止训练；当前没有中途续训功能。
6. 不删除旧实验，除非用户明确同意并先列出文件。
7. 不把2轮烟雾测试和25轮结果直接当作公平性能比较。
8. 不伪造、不补齐、不推测任何实验数字。

## 十、本地论文材料

原始Mac工作区：

```text
/Users/augustus/projects/论文
```

关键文件：

```text
small-paper-draft-v1.md
small-paper-references.bib
experiment-plan-v4.md
experiment-log.md
method-evolution-and-ownership-20260729.md
SERVER-P2-ACTIONABILITY-INSTRUCTIONS.md
```

当前没有`ccfa.yaml`。新的服务器结果应先写入实验日志，再谨慎同步论文草稿；必须明确区分真实完成、pilot和TBD。
