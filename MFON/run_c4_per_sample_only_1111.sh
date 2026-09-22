#!/usr/bin/env bash
# Start the single-seed C4 attribution run only after its server preflight.
set -euo pipefail

cd /home/jovyan/projects/MFON
experiment=p5_mosei_per_sample_only_true_budget
log=mosei_p5_per_sample_only_true_budget_1111.log
checkpoint_dir="MOSEI/save_models/all_model/MOSEI/1111/$experiment"

python - <<'PY'
import os
import sys

stat = os.statvfs('/home/jovyan')
free_gib = stat.f_bavail * stat.f_frsize / 2**30
print('可用空间：%.2f GiB' % free_gib)
if free_gib < 12:
    sys.exit('低于预定的 12 GiB 启动门槛，未启动训练')
PY

for path in "$log" "$checkpoint_dir"; do
  if [ -e "$path" ]; then
    echo "目标已存在，停止以免覆盖：$path"
    exit 1
  fi
done
for path in \
  data/MOSEI/unaligned_50.pkl \
  MOSEI/save_models/uni_fea_encoder/MOSEI/1111/best_loss_audio_encoder.pt \
  MOSEI/save_models/uni_fea_encoder/MOSEI/1111/best_loss_vision_encoder.pt; do
  if [ ! -f "$path" ]; then
    echo "所需文件不存在，未启动：$path"
    exit 1
  fi
done
if pgrep -af '^python (run_experiment|audit_.*)[.]py'; then
  echo '已有训练或审计在运行，未启动'
  exit 1
fi
nvidia-smi -L >/dev/null
if nvidia-smi --query-compute-apps=pid --format=csv,noheader |
    grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'GPU 上已有计算进程，未启动'
  exit 1
fi
python -m py_compile run_experiment.py budgeted_auxiliary.py

nohup env PYTHONUNBUFFERED=1 python run_experiment.py \
  --dataset MOSEI --stage train-fusion --seed 1111 --epochs 25 \
  --use-budgeted-aux --use-interventional-reliability \
  --reliability-loss-weight 0 \
  --warmup-epoch 10 --budget-warmup-mode allocation \
  --reliability-task-warmup-epoch 10 \
  --reliability-task-corrupt-scale 0 \
  --reliability-allocation-control constant \
  --exp-name "$experiment" \
  > "$log" 2>&1 < /dev/null &
echo "C4 已启动，PID=$!；日志：$log"
