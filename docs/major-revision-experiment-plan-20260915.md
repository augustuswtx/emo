# NCA 大修实验执行方案（C1--C5）

> 状态日期：2026-09-15。本文档只登记已完成证据和待运行协议，不把 TBD 写成结果。所有新训练必须由用户在服务器上明确启动；本地修改不会自动发起 GPU 任务。

## 1. 审稿问题与最低闭环

| ID | 核心问题 | 最低证据 | 当前状态 | 论文处理 |
|---|---|---|---|---|
| C1 | 样本内排序不保证跨样本可比 | 干净分数与辅助目标保真度的相关、偏相关、pairwise concordance；三种子 | 工具已实现，服务器结果 TBD | 完成前称 degradation-response score |
| C2 | 分配方向与对应关系未验证 | final-schedule Learned / Constant / Permuted / Inverse，同种子配对 | 1111--1113 Learned/Constant done；其余 TBD | 不宣称正向可靠性分配最优 |
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

这里的 `-(KL + InfoNCE)` 是“当前 checkpoint 下辅助目标一致性代理”，不是外部真值。只有三种子方向一致、偏相关仍为正且 concordance 明显高于 0.5，才可谨慎支持跨样本分配语义；否则保留负结果并撤回“可靠性越高、辅助目标越可信”的解释。

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

validation 结果完成并锁定解释规则后，才对 test split 各运行一次；test 结果仍必须明确写为事后探索性分析，不得称为原冻结协议或预注册验证。

### C5 modality-utility 分层

对同一测试样本计算完整输入、缺失视觉、缺失音频预测。定义视觉效用为删除视觉后绝对误差的增加，音频同理；按每个基线 checkpoint 的效用分位数分为低/中/高三层。分别比较 Learned 与 Constant 的 MAE、Corr 和 Loss。分层阈值不得用 Learned 的结果选择，主报告使用 repaired MFON seed 对应 checkpoint 定义的层。

通过条件：高效用层的样本数、阈值和全部指标完整报告；不要求结果必须正向。若高效用层仍无优势，论文保留文本主导限制，并不再投入第二骨干训练。

## 3. P1：final-schedule 作用性矩阵

严格复用已冻结参数：25 epochs、validation-loss checkpoint selection、`warmup=10`、`budget-warmup-mode=allocation`、`task-corrupt-scale=0`、相同单模态 encoder 和 seed。

| Variant | `reliability-allocation-control` | 回答问题 | Seeds |
|---|---|---|---|
| Constant | constant | 等预算均匀分配 | 1111--1113 done |
| Learned | learned | 正向分数分配 | 1111--1113 done |
| Permuted | permuted | 破坏 score--sample 对应 | 1111--1113 TBD |
| Inverse | inverse | 低分/困难样本优先 | 1111--1113 TBD |

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

预注册判据：Learned 必须在多数种子上同时优于 Permuted 和 Inverse 的 MAE/Corr 配对方向，才支持“正确对应 + 正向分配”。若只优于 Constant，则结论收缩为“非均匀分配改变结果”；若 Permuted/Inverse 相当或更好，则撤回可靠性驱动的因果解释。

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

仅在 C1 不失败且 C2 至少显示 Learned 对 Permuted/Inverse 的一致趋势后，新增 seeds 1114、1115。每个新 seed 需要匹配的单模态 encoders 和所有进入统计主表的变体。不得只补 Learned。

五种子报告：逐种子原值、配对差值、均值、样本 SD、配对差值的 bootstrap 95% CI。统计重采样单位是 seed；测试样本 bootstrap 只能用于 checkpoint 内预测不确定性，不能冒充五次独立训练。

## 6. 结果表模板

| Seed | Constant MAE | Learned MAE | Permuted MAE | Inverse MAE | Learned-Constant | Learned-Permuted | Learned-Inverse |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1111 | 0.5406 | 0.5388 | TBD | TBD | -0.0018 | TBD | TBD |
| 1112 | 0.5274 | 0.5283 | TBD | TBD | +0.0009 | TBD | TBD |
| 1113 | 0.5342 | 0.5277 | TBD | TBD | -0.0065 | TBD | TBD |
| 1114 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 1115 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Corr、Loss 与所有分类指标使用相同格式完整报告，不能只挑有利指标。

## 7. 资源顺序与停止条件

1. 先运行 C1 只读审计（小时级以内，不训练）。
2. 再运行 seed 1111 Permuted、Inverse；每次只跑一个。
3. 若两项能区分机制，再扩展至 seeds 1112--1113。
4. 运行 seed 1111 Per-sample only，判断是否值得扩展组件消融。
5. 做 modality-utility 分层。
6. 只有前述证据链支持方法解释，才训练 1114--1115；否则以负面审计论文收缩主张。

任何阶段出现磁盘低于 5 GiB、checkpoint 路径冲突、配置不一致或日志异常时立即停止，不覆盖现有 checkpoint。
