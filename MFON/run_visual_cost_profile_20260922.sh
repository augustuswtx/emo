#!/usr/bin/env bash
set -euo pipefail

cd /home/jovyan/projects/MFON
archive=mfon_visual_cost_profile_20260922.tar.gz
script=audit_visual_online_cost.py
backup=audit_visual_online_cost.py.before_profile_20260922
log=visual_online_cost_profile_seed1111_3pairs_20260922.log
output=visual_online_cost_profile_seed1111_3pairs_20260922.json

for path in "$backup" "$log" "$output"; do
  if [ -e "$path" ]; then
    echo "已有文件，停止：$path"
    exit 1
  fi
done
if pgrep -f '^python (run_experiment|audit_visual_online_cost|audit_task_aligned_step|audit_visual_directional_difference)[.]py' >/dev/null; then
  echo '已有相关实验在运行，停止'
  exit 1
fi
printf '%s  %s\n' \
  '945e8deb0251cacf6de0fc7a6dd78f686e5d576150c03fbdbaf44bbacb04895d' \
  "$script" | sha256sum -c -

cp -a "$script" "$backup"
tar -xzf "$archive" -C /home/jovyan/projects MFON/audit_visual_online_cost.py
python -m py_compile "$script"
python -m unittest discover -s tests -p 'test_visual_online_cost.py' -q

nohup env PYTHONUNBUFFERED=1 python "$script" \
  --seed 1111 --pairs 3 --output "$output" \
  > "$log" 2>&1 < /dev/null &
echo "分段计时审计 PID=$!；日志：$log"
