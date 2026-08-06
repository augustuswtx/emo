# MOSI 固定预算 P0 服务器操作

## 1. 上传代码包

把本地文件 `mfon_budgeted_aux_p0_20260720.tar.gz` 上传到服务器目录：

`/home/jovyan/projects/MFON`

## 2. 进入目录并备份

```bash
cd /home/jovyan/projects/MFON

tar -czf backup_before_budgeted_aux_$(date +%Y%m%d_%H%M%S).tar.gz \
  run_experiment.py \
  MOSI/config.py MOSI/models/model.py MOSI/train/TVA_train.py \
  MOSEI/config.py MOSEI/models/model.py MOSEI/train/TVA_train.py \
  SIMS/config.py SIMS/models/model.py SIMS/train/TVA_train.py
```

## 3. 解压新代码并运行测试

```bash
tar -xzf mfon_budgeted_aux_p0_20260720.tar.gz
python -m unittest discover -s tests -v
```

应看到 `Ran 8 tests` 和 `OK`。测试不通过时不要启动训练。

## 4. 启动可断线恢复的实时训练

先检查服务器是否安装 `tmux`：

```bash
tmux -V
```

如果能显示版本，创建训练会话：

```bash
tmux new -s budget_p0
```

进入 tmux 后运行 2 个 epoch 的烟雾测试：

```bash
cd /home/jovyan/projects/MFON

CUDA_LAUNCH_BLOCKING=1 python run_experiment.py \
  --dataset MOSI --stage train-fusion --seed 1111 \
  --epochs 2 --use-budgeted-aux --q-type align --warmup-epoch 10 \
  --exp-name budgeted_aux_align_smoke \
  2>&1 | tee mosi_budgeted_aux_align_smoke_1111.log
```

此时当前终端会实时显示训练日志。按 `Ctrl-b`，松开后按 `d`，可以退出显示但保持训练继续。重新连接后恢复实时画面：

```bash
tmux attach -t budget_p0
```

只查看日志而不进入 tmux：

```bash
tail -f mosi_budgeted_aux_align_smoke_1111.log
```

## 5. 烟雾测试检查

日志必须满足：

1. 没有 traceback、CUDA assert、`nan` 或 `inf`；
2. 每个 epoch 出现 `Budgeted auxiliary epoch`；
3. 第 1 个 epoch 的 `w_v`、`w_a`、`w_nce_v`、`w_nce_a` 均值约为 `0.05`；
4. `q_v_std`、`q_a_std`、`w_v_std`、`w_a_std` 不应全部严格为 `0`；
5. 保存 `MOSI/save_models/all_model/MOSI/1111/budgeted_aux_align_smoke/TVA_fusion_model.pt`。

## 6. 完整 P0

烟雾测试通过后运行完整 25 个 epoch：

```bash
CUDA_LAUNCH_BLOCKING=1 python run_experiment.py \
  --dataset MOSI --stage train-fusion --seed 1111 \
  --use-budgeted-aux --q-type align --warmup-epoch 10 \
  --exp-name budgeted_aux_align_p0 \
  2>&1 | tee mosi_budgeted_aux_align_p0_1111.log
```

训练完成后测试：

```bash
python run_experiment.py \
  --dataset MOSI --stage test-fusion --seed 1111 \
  --use-budgeted-aux --q-type align --warmup-epoch 10 \
  --exp-name budgeted_aux_align_p0 \
  2>&1 | tee mosi_budgeted_aux_align_p0_test_1111.log
```

暂时不要同时启用 `--use-alw`、`--use-css` 或 `--use-dpg`。这个 P0 只验证逐样本损失和预算守恒修复。
