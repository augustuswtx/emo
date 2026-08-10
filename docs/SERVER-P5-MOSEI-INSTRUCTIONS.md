# P5 MOSEI P4 Port Instructions

Package:

```text
mfon_mosei_p4_port_20260810.tar.gz
SHA256 9cc49c6978290e5442ac55b460c74752c2ac3fceebeb06c0232776dd330d8994
```

## 1. Preflight before upload

```bash
cd /home/jovyan/projects/MFON
ls -lh data/MOSEI/unaligned_50.pkl
for s in 1111 1112 1113; do
  echo "===== seed $s ====="
  find MOSEI/save_models/uni_fea_encoder/MOSEI/$s -maxdepth 1 -type f -ls 2>/dev/null
done
df -h /home/jovyan
```

Do not start fusion training until the MOSEI dataset and both audio/vision
encoder checkpoints exist for the selected seed.

## 2. Install and verify

Upload the package into `/home/jovyan/projects/MFON`, then run:

```bash
cd /home/jovyan/projects/MFON
sha256sum mfon_mosei_p4_port_20260810.tar.gz

mkdir -p backup_before_mosei_p5_20260810
cp run_experiment.py audit_model_quality.py \
  budgeted_auxiliary.py interventional_reliability.py \
  backup_before_mosei_p5_20260810/
cp MOSEI/config.py MOSEI/models/model.py MOSEI/train/TVA_train.py \
  backup_before_mosei_p5_20260810/

tar -xzf mfon_mosei_p4_port_20260810.tar.gz
python -m unittest discover -s tests -v
python -m py_compile run_experiment.py audit_model_quality.py \
  MOSEI/config.py MOSEI/models/model.py MOSEI/train/TVA_train.py
python run_experiment.py --help | grep -E "budget-warmup-mode|interventional-reliability|reliability-task-corrupt-scale"
```

Expected: 33 tests pass, syntax compilation is silent, and all three CLI flags
are listed.

## 3. Two-epoch MOSEI smoke

Only after preflight and tests pass:

```bash
cd /home/jovyan/projects/MFON && nohup env PYTHONUNBUFFERED=1 \
python run_experiment.py --dataset MOSEI --stage train-fusion --seed 1111 \
  --epochs 2 --use-budgeted-aux --use-interventional-reliability \
  --warmup-epoch 10 --budget-warmup-mode allocation \
  --reliability-task-warmup-epoch 10 --reliability-task-corrupt-scale 0 \
  --reliability-allocation-control learned \
  --exp-name p5_mosei_p4_learned_smoke \
  > mosei_p5_p4_learned_smoke_1111.log 2>&1 & echo $!
```

The epoch-2 log must show dataset `MOSEI`, all four mean budgets equal to the
MOSEI baseline values (`w_v=w_a=0.3`, `w_nce_v=w_nce_a=0.001`), non-zero
Learned allocation standard deviations, positive reliability gaps, and
`task_corruption_progress=0`. These budget values intentionally differ from
MOSI because the original MOSEI configuration uses `delta_va=0.3` and
`delta_nce=0.001`.

Do not launch a 25-epoch run until the smoke checkpoint saves and reloads.
