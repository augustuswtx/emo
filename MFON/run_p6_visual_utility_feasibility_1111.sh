#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

dataset_json="visual_utility_seed1111_final_64pairs_20260928.json"
analysis_json="visual_utility_predictor_seed1111_final_20260928.json"
log_file="visual_utility_feasibility_seed1111_final_20260928.log"

for path in "$dataset_json" "$analysis_json" "$log_file"; do
    if [ -e "$path" ]; then
        echo "输出已存在，停止：$path"
        exit 1
    fi
done

python -m py_compile \
    audit_visual_utility_dataset.py \
    analyze_visual_utility_predictor.py \
    tests/test_visual_utility_dataset.py \
    tests/test_visual_utility_predictor.py

python -m unittest discover -s tests -p 'test_visual_utility_dataset.py' -v
python -m unittest discover -s tests -p 'test_visual_utility_predictor.py' -v

{
    python audit_visual_utility_dataset.py \
        --seed 1111 \
        --checkpoint-stage final \
        --meta-batch-size 4 \
        --aux-batch-size 32 \
        --pairs 64 \
        --sample-order-seed 161803 \
        --exclude-first 1024 \
        --output "$dataset_json"

    python analyze_visual_utility_predictor.py \
        --input "$dataset_json" \
        --output "$analysis_json"
} > "$log_file" 2>&1

echo "完成：$dataset_json"
echo "完成：$analysis_json"
echo "日志：$log_file"

