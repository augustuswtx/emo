#!/usr/bin/env bash
# Test the completed C4 seed-1111 checkpoint without overwriting prior output.
set -euo pipefail

cd /home/jovyan/projects/MFON
experiment=p5_mosei_per_sample_only_true_budget
train_log=mosei_p5_per_sample_only_true_budget_1111.log
test_log=mosei_p5_per_sample_only_true_budget_1111.test.log
checkpoint="MOSEI/save_models/all_model/MOSEI/1111/$experiment/TVA_fusion_model.pt"

for path in "$train_log" "$checkpoint"; do
  if [ ! -s "$path" ]; then
    echo "训练来源缺少或为空，未测试：$path"
    exit 1
  fi
done
if ! grep -q 'Budgeted auxiliary epoch 25:' "$train_log"; then
  echo '训练日志未显示完成第 25 轮，未测试'
  exit 1
fi
if [ -e "$test_log" ]; then
  echo "测试日志已存在，停止以免覆盖：$test_log"
  exit 1
fi
if pgrep -af '^python (run_experiment|audit_.*)[.]py'; then
  echo '已有训练或审计在运行，未测试'
  exit 1
fi
nvidia-smi -L >/dev/null
if nvidia-smi --query-compute-apps=pid --format=csv,noheader |
    grep -Eq '^[[:space:]]*[0-9]+'; then
  echo 'GPU 上已有计算进程，未测试'
  exit 1
fi
python -m py_compile run_experiment.py budgeted_auxiliary.py
sha256sum "$checkpoint"

nohup bash -c 'env PYTHONUNBUFFERED=1 python run_experiment.py "$@"; result=$?; echo "test_exit_code=$result"; exit "$result"' c4-test \
  --dataset MOSEI --stage test-fusion --seed 1111 --epochs 25 \
  --use-budgeted-aux --use-interventional-reliability \
  --reliability-loss-weight 0 \
  --warmup-epoch 10 --budget-warmup-mode allocation \
  --reliability-task-warmup-epoch 10 \
  --reliability-task-corrupt-scale 0 \
  --reliability-allocation-control constant \
  --exp-name "$experiment" \
  > "$test_log" 2>&1 < /dev/null &
echo "C4 测试已启动，PID=$!；日志：$test_log"
