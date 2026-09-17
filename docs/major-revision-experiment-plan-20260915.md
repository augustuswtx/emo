# NCA 大修实验执行方案（C1--C5）

> 状态日期：2026-09-17。本文档只登记已完成证据和待运行协议，不把 TBD 写成结果。所有新训练由用户在服务器手动启动；本地修改不会自动发起 GPU 任务。

## 1. 审稿问题与最低闭环

| ID | 核心问题 | 最低证据 | 当前状态 | 论文处理 |
|---|---|---|---|---|
| C1 | 样本内排序不保证跨样本可比 | 干净分数与辅助目标保真度的相关、偏相关、pairwise concordance；三种子 | 已完成；视觉反向、音频弱相关 | 称 degradation-response score；撤回跨样本保真度解释 |
| C2 | 分配方向与对应关系未验证 | final-schedule Learned / Constant / Permuted / Inverse，同种子配对 | 1111 四组完成测试；1112--1113 Permuted/Inverse 待定 | 不宣称正向可靠性分配最优 |
| C3 | 三种子效应小 | 逐种子点、配对差值；扩展 1114--1115 | 配对图已完成；新种子 TBD | 现阶段只作描述性均值 |
| C4 | 相对 MFON 的组件归因不清 | Baseline / per-sample-only / +reliability Constant / Learned | Baseline、Constant、Learned done；per-sample-only TBD | 不把 P4--MFON 差异归因于分配 |
| C5 | 文本主导、外部有效性弱 | modality-utility 分层；第二骨干为可选高成本项 | 推理审计 TBD | 不宣称鲁棒融合或模型无关性 |

## 2. P0：无需重训的检查点审计

### C1 跨样本可比性与目标保真度

新增脚本：`audit_cross_sample_validity.py`。该分析是在审稿后新增的，统一标为 **post-hoc exploratory**。它在冻结 checkpoint 上读取每个样本的视觉/音频分数、KL 和 InfoNCE，采用以下修正版口径：

- 主分析为跨样本 `q` 与 `-KL` 的 Spearman、Pearson、控制长度和能量后的残差 Spearman，以及成对排序一致率（0.5 为无方向）；
- InfoNCE 的负样本集合依赖当前 mini-batch，因此 `q`--InfoNCE 只在各自生成它的批次内部比较，不跨批拼接排序；
- 真实训练尺度的 `0.3*KL + 0.001*InfoNCE` 同样只报告批内 Spearman 与批内成对一致率；
- 分数与长度、能量的相关。

这里的 `-KL` 是跨样本主代理；InfoNCE 及按真实系数组合的代理只在批内比较。它们都不是外部真值。只有三种子方向一致、偏相关仍为正且 concordance 明显高于 0.5，才可谨慎支持跨样本分配语义；否则保留负结果并撤回“可靠性越高、辅助目标越可信”的解释。

三种子 validation 全量审计（每个种子 `n=1871`）已经完成。视觉 `q`--`-KL` Spearman 为 `-0.315423±0.069633`，控制长度和能量后的残差 Spearman 为 `-0.248627±0.064859`，pairwise concordance 为 `0.393666±0.023143`；方向在三个种子上一致为负。音频对应结果为 `0.092174±0.042066`、`0.072239±0.067445` 和 `0.529795±0.014286`，仅表现为弱关联。批内真实加权代理的 Spearman 为视觉 `-0.290099±0.047795`、音频 `0.064885±0.047447`。因此 C1 不支持跨样本辅助目标保真度假设，结果保留为事后探索性负面证据。由于 validation 已经否定支持性门槛，不追加 test-split 审计，避免无必要地扩大事后测试集使用。

先在 validation split 做 seed 1111 smoke，不启动训练：

```bash
cd /home/jovyan/projects/MFON
python -m py_compile audit_cross_sample_validity.py
python -m unittest discover -s tests -p 'test_cross_sample_validity.py' -v
python audit_cross_sample_validity.py \
  --dataset MOSEI --seed 1111 \
  --exp-name p5_mosei_p4_learned_true_budget \
  --split valid --max-batches 5
```

smoke 无报错后先运行三种子 validation 全量（仍为只读审计）：

```bash
cd /home/jovyan/projects/MFON
for s in 1111 1112 1113; do
  nohup env PYTHONUNBUFFERED=1 python audit_cross_sample_validity.py \
    --dataset MOSEI --seed "$s" \
    --exp-name p5_mosei_p4_learned_true_budget --split valid \
    > "mosei_cross_sample_validity_valid_${s}.log" 2>&1 &
  wait $!
done
```

validation 结果已经锁定为负面解释；不运行 test split，也不得将该分析称为原冻结协议或预注册验证。

### C5 modality-utility 分层

对同一测试样本计算完整输入、视觉特征置零、音频特征置零时的预测。定义视觉/音频**零化敏感度代理**为相应特征置零后绝对误差的增加；按每个 seed 的 repaired MFON checkpoint 给出的代理值分为低/中/高三层。分别比较 Learned 与 Constant 的 MAE、Corr 和 Loss。该代理依赖置零扰动和测试标签，不能视为真实模态效用或因果贡献。分层阈值不得用 Learned 的结果选择。

通过条件：高敏感度层的样本数、阈值和全部指标完整报告；不要求结果必须正向。若高敏感度层仍无优势，论文保留文本主导限制，并不再投入第二骨干训练。

只读实现：`MFON/audit_modality_utility.py`；纯数值助手与测试分别位于 `MFON/modality_utility_stats.py` 和 `MFON/tests/test_modality_utility.py`。审计标记为 **post-hoc exploratory**。分层采用修复版 MFON 的逐样本误差增量排序，等效用值以数据集索引稳定打破并列，输出会标记边界是否存在并列。它不修改模型和 checkpoint，也不以 Learned/Constant 结果选择分层。服务器先运行：

```bash
cd /home/jovyan/projects/MFON
python -m unittest discover -s tests -p 'test_modality_utility.py' -v
python audit_modality_utility.py --seed 1111 --split valid --max-batches 5 \
  --output c5_modality_utility_valid_smoke_1111.json \
  > c5_modality_utility_valid_smoke_1111.log 2>&1
```

小批量输出必须显示 Baseline、Constant、Learned 各收集相同样本数，六个分层均非空，所有 MAE/Loss 有限。还应检查边界并列标记；若大量样本效用相同，则如实报告该模态分层的解释限制。检查后再对三个冻结种子依次运行完整 test；每种子只运行一次、使用唯一日志与 JSON 文件。test 分层涉及真实标签，是事后探索性亚组分析，不能用于重新选择模型或反向宣称预注册。

## 3. P1：final-schedule 作用性矩阵

严格复用已冻结参数：25 epochs、validation-loss checkpoint selection、`warmup=10`、`budget-warmup-mode=allocation`、`task-corrupt-scale=0`、相同单模态 encoder 和 seed。

| Variant | `reliability-allocation-control` | 回答问题 | Seeds |
|---|---|---|---|
| Constant | constant | 等预算均匀分配 | 1111--1113 done |
| Learned | learned | 正向分数分配 | 1111--1113 done |
| Permuted | permuted | 破坏 score--sample 对应 | 1111 done；1112--1113 TBD |
| Inverse | inverse | 低分/困难样本优先 | 1111 测试指标已取得；1112--1113 TBD |

单元命令模板（`CONTROL` 替换为 `permuted` 或 `inverse`；同一时间只跑一个）：

```bash
cd /home/jovyan/projects/MFON
SEED=1111
CONTROL=permuted
EXP="p5_mosei_p4_${CONTROL}_true_budget"
nohup env PYTHONUNBUFFERED=1 python run_experiment.py \
  --dataset MOSEI --stage train-fusion --seed "$SEED" --epochs 25 \
  --use-budgeted-aux --use-interventional-reliability \
  --warmup-epoch 10 --budget-warmup-mode allocation \
  --reliability-task-warmup-epoch 10 \
  --reliability-task-corrupt-scale 0 \
  --reliability-allocation-control "$CONTROL" \
  --exp-name "$EXP" \
  > "mosei_${EXP}_${SEED}.log" 2>&1 & echo $!
```

预先固定的判据：Learned 必须在多数种子上同时优于 Permuted 和 Inverse 的 MAE/Corr 配对方向，才支持“正确对应 + 正向分配”。若只优于 Constant，则结论收缩为“非均匀分配改变结果”；若 Permuted/Inverse 相当或更好，则撤回可靠性驱动的因果解释。seed 1111 的 Learned MAE 为 0.5388，优于 Permuted 0.5399 和 Inverse 0.5405；Corr 为 0.7742，高于 Permuted 0.7740，却低于 Inverse 0.7743。单种子差异很小，不能据此宣称判据通过；C1 的跨样本目标保真度负结果仍然成立。

## 4. P1：组件消融

| 组件 | Baseline | Per-sample only | Reliability Constant | Learned |
|---|:---:|:---:|:---:|:---:|
| 修复版 MFON | ✓ | ✓ | ✓ | ✓ |
| 未归约逐样本辅助损失 |  | ✓ | ✓ | ✓ |
| Reliability ranking/invariance loss |  |  | ✓ | ✓ |
| 非均匀 learned allocation |  |  |  | ✓ |

“Per-sample only” 使用 Constant 分配且将 reliability loss 权重置 0；这样保留相同逐样本实现和固定预算，但不让 reliability loss 更新共享训练路径：

```bash
python run_experiment.py \
  --dataset MOSEI --stage train-fusion --seed 1111 --epochs 25 \
  --use-budgeted-aux --use-interventional-reliability \
  --reliability-loss-weight 0 \
  --warmup-epoch 10 --budget-warmup-mode allocation \
  --reliability-task-warmup-epoch 10 --reliability-task-corrupt-scale 0 \
  --reliability-allocation-control constant \
  --exp-name p5_mosei_per_sample_only_true_budget
```

先做 seed 1111。若 Per-sample only 与 Baseline 几乎一致而 Reliability Constant 退化，则说明损失冲突主要来自 reliability supervision；再扩展 1112--1113。若 seed 1111 已显示实现异常，停止并排查，不批量运行。

## 5. P2：扩展到五种子

C1 已失败，因此 seeds 1114、1115 暂停，不以增加种子数量掩盖跨样本语义不成立。只有 C2 显示 Learned 对 Permuted/Inverse 的一致趋势且论文仍需要更稳定的任务效应估计时，才重新评估扩展；每个新 seed 仍需匹配的单模态 encoders 和所有进入统计主表的变体，不得只补 Learned。

五种子报告：逐种子原值、配对差值、均值、样本 SD、配对差值的 bootstrap 95% CI。统计重采样单位是 seed；测试样本 bootstrap 只能用于 checkpoint 内预测不确定性，不能冒充五次独立训练。

## 6. 结果表模板

| Seed | Constant MAE | Learned MAE | Permuted MAE | Inverse MAE | Learned-Constant | Learned-Permuted | Learned-Inverse |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1111 | 0.5406 | 0.5388 | 0.5399 | 0.5405 | -0.0018 | -0.0011 | -0.0017 |
| 1112 | 0.5274 | 0.5283 | TBD | TBD | +0.0009 | TBD | TBD |
| 1113 | 0.5342 | 0.5277 | TBD | TBD | -0.0065 | TBD | TBD |
| 1114 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 1115 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Corr、Loss 与所有分类指标使用相同格式完整报告，不能只挑有利指标。

## 7. 资源顺序与停止条件

1. C1 只读审计已完成，结论为视觉反向、音频弱关联。
2. seed 1111 Permuted 和 Inverse 已按冻结日程训练与重载测试；归档日志和 checkpoint 哈希，不重复运行。
3. 单种子主终点并未同时支持 Learned 优于 Inverse；在决定是否投入约 16 小时串行 GPU 时间扩展 1112--1113 前，先保持机制主张收缩，并完成不需重训的 C5 分层。
4. 运行 seed 1111 Per-sample only，判断是否值得扩展组件消融；启动前重新满足 12 GiB 磁盘门槛。
6. 只有前述证据链支持方法解释，才训练 1114--1115；否则以负面审计论文收缩主张。

任何阶段出现磁盘低于 5 GiB、checkpoint 路径冲突、配置不一致或日志异常时立即停止，不覆盖现有 checkpoint。
