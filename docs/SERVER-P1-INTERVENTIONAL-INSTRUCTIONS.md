# MOSI P1.1 音频时序可靠性与渐进扰动服务器操作

服务器目录：`/home/jovyan/projects/MFON`

## 1. 上传并检查补丁

将 `mfon_interventional_rel_p11_20260729.tar.gz` 上传到
`/home/jovyan/projects`。进入项目后检查：

```bash
cd /home/jovyan/projects/MFON
ls -lh ../mfon_interventional_rel_p11_20260729.tar.gz
```

## 2. 解压指定文件

```bash
tar -xzf ../mfon_interventional_rel_p11_20260729.tar.gz
```

该补丁只更新MOSI P1.1路径、公共固定预算模块、审计脚本、README和测试。它保留视觉可靠性头，把音频侧改为时序可靠性头，并让主任务扰动在10轮内逐渐增强。

## 3. CPU闸门

```bash
python -m unittest discover -s tests -v
```

预期结果：

```text
Ran 23 tests
OK
```

不满足时不要启动GPU训练。

## 4. 两轮GPU烟雾测试

先确认磁盘空间：

```bash
df -h .
```

建议`Avail`至少5GB。随后进入tmux：

```bash
tmux new -s interventional_p11
```

在tmux内运行：

```bash
cd /home/jovyan/projects/MFON

CUDA_LAUNCH_BLOCKING=1 python run_experiment.py \
  --dataset MOSI \
  --stage train-fusion \
  --seed 1111 \
  --epochs 2 \
  --use-budgeted-aux \
  --use-interventional-reliability \
  --warmup-epoch 10 \
  --reliability-task-warmup-epoch 10 \
  --exp-name interventional_rel_temporal_warmup_smoke \
  2>&1 | tee mosi_interventional_rel_temporal_warmup_smoke_1111.log
```

通过条件：

1. 两轮均无`traceback`、`nan`或`inf`；
2. 两轮均输出`Budgeted auxiliary epoch`；
3. 两轮均输出`Interventional reliability epoch`；
4. 四类预算均值分别约为0.05和0.10；
5. `loss_reliability`和`loss_rank`为有限值；
6. `q_gap_v`和`q_gap_a`应随训练向正值发展；
7. `task_corruption_progress`在第1、2轮应分别为0.1和0.2；
8. checkpoint成功保存。

## 5. checkpoint加载

```bash
python run_experiment.py \
  --dataset MOSI \
  --stage test-fusion \
  --seed 1111 \
  --use-budgeted-aux \
  --use-interventional-reliability \
  --warmup-epoch 10 \
  --reliability-task-warmup-epoch 10 \
  --exp-name interventional_rel_temporal_warmup_smoke \
  2>&1 | tee mosi_interventional_rel_temporal_warmup_smoke_test_1111.log
```

两轮测试指标只用于检查加载和推理，不与25轮基线比较。

## 6. 五批可靠性审计

```bash
python audit_model_quality.py \
  --dataset MOSI \
  --seed 1111 \
  --exp-name interventional_rel_temporal_warmup_smoke \
  --q-type learned \
  --split test \
  --max-batches 5 \
  2>&1 | tee mosi_interventional_rel_temporal_warmup_audit_pilot_1111.log
```

提取关键结果：

```bash
grep -E \
"^Audit |^clean metrics|^clean confounds|^vision audit|^audio audit" \
mosi_interventional_rel_temporal_warmup_audit_pilot_1111.log
```

视觉至少应维持P1的明显负相关和高AUROC；音频必须明显高于P1的AUROC 0.501，且clean两轮指标不再出现同等程度的恶化，才进行完整测试集审计。不要直接启动25轮训练。
