#!/usr/bin/env bash
set -euo pipefail

cd /home/jovyan/projects/MFON
archive=mfon_visual_meta_reuse_20260922.tar.gz
script=audit_visual_meta_reuse.py
log=visual_meta_reuse_seed1111_12pairs_20260922.log
output=visual_meta_reuse_seed1111_12pairs_20260922.json
reference=visual_online_cost_seed1111_3pairs_20260922.json

for path in "$script" "$log" "$output"; do
  if [ -e "$path" ]; then
    echo "已有文件，停止：$path"
    exit 1
  fi
done
test -s "$reference"
for path in audit_visual_online_cost.py audit_visual_directional_difference.py; do
  test -f "$path"
done
if pgrep -f '^python (run_experiment|audit_visual_online_cost|audit_visual_meta_reuse|audit_task_aligned_step|audit_visual_directional_difference)[.]py' >/dev/null; then
  echo '已有相关实验在运行，停止'
  exit 1
fi

tar -xzf "$archive" -C /home/jovyan/projects MFON/audit_visual_meta_reuse.py
python -m py_compile "$script"
nohup env PYTHONUNBUFFERED=1 python "$script" \
  --seed 1111 --pairs 12 --sample-order-seed 314159 \
  --exclude-cost-json "$reference" --output "$output" \
  > "$log" 2>&1 < /dev/null &
echo "梯度复用审计 PID=$!；日志：$log"
