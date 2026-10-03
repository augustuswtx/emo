#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

reference="visual_utility_seed1111_final_64pairs_20260928.json"
dataset_json="visual_utility_stability_seed1111_final_24groups_20261003.json"
analysis_json="visual_utility_stability_analysis_seed1111_final_20261003.json"
log_file="visual_utility_stability_seed1111_final_20261003.log"

test -s "$reference" || { echo "缺少 P6 结果：$reference"; exit 1; }
for path in "$dataset_json" "$analysis_json" "$log_file"; do
    if [ -e "$path" ]; then
        echo "输出已存在，停止：$path"
        exit 1
    fi
done

python -m py_compile \
    audit_visual_utility_stability.py \
    analyze_visual_utility_stability.py \
    tests/test_visual_utility_stability.py \
    tests/test_visual_utility_stability_analysis.py

python -m unittest discover -s tests -p 'test_visual_utility_stability.py' -v
python -m unittest discover -s tests -p 'test_visual_utility_stability_analysis.py' -v

{
    python audit_visual_utility_stability.py \
        --seed 1111 \
        --checkpoint-stage final \
        --meta-batch-size 4 \
        --aux-batch-size 32 \
        --groups 24 \
        --meta-contexts 6 \
        --sample-order-seed 314159 \
        --exclude-first 1024 \
        --exclude-json "$reference" \
        --output "$dataset_json"

    python analyze_visual_utility_stability.py \
        --input "$dataset_json" \
        --output "$analysis_json"
} > "$log_file" 2>&1

echo "完成：$dataset_json"
echo "完成：$analysis_json"
echo "日志：$log_file"
