Pytorch implementation for Coling2025 paper: Modal Feature Optimization Network with Prompt for Multimodal Sentiment Analysis.

![Image text](https://github.com/123sprouting/MFON/blob/main/MFON-structure.jpg)



The training and testing of the MOSI, MOSEI, and SIMS datasets are implemented in three files with the same name.

### Setup the environment

We work with a conda environment.

```
conda env create -f environment.yaml
conda activate pytorch
```

### Data Download

- Get datasets from public link:https://github.com/thuiar/Self-MM and  change the raw_data_path  to your local path(In config.py).

Or download the MMSA processed feature files used by this project:

```bash
../.venv/bin/python download_mmsa_data.py --dataset all
```

If Google Drive rate-limits a file, use the official BaiduYun link from MMSA and put files at:

- `data/MOSI/unaligned_50.pkl`
- `data/MOSEI/unaligned_50.pkl`
- `data/SIMS/unaligned_39.pkl`

### Pretrained model:
链接:https://pan.baidu.com/s/1_EcyiHYXtSFTjG5Ro7E1SQ 
提取码:2xx5

### Running the code

Take MOSI for example:
1. cd MOSI
2. python main.py 

### Running MFON / DAMFON experiments

The original dataset-specific `main.py` files are preserved. A shared runner is also provided for controlled experiments:

```bash
# Train unimodal encoders required by fusion training
python run_experiment.py --dataset MOSI --stage train-unimodal

# Baseline MFON fusion training
python run_experiment.py --dataset MOSI --stage train-fusion

# DAMFON ALW training with alignment-based quality
python run_experiment.py --dataset MOSI --stage train-fusion --use-alw --q-type align --warmup-epoch 10

# DAMFON DPG-only training
python run_experiment.py --dataset MOSI --stage train-fusion --use-dpg

# DAMFON ALW + DPG training
python run_experiment.py --dataset MOSI --stage train-fusion --use-alw --use-dpg --q-type align --warmup-epoch 10

# DAMFON full training: ALW + DPG + CSS
python run_experiment.py --dataset MOSI --stage train-fusion --use-alw --use-dpg --use-css --q-type align --warmup-epoch 10 --css-epoch 20

# Two-epoch fixed-budget smoke test (run this first)
python run_experiment.py --dataset MOSI --stage train-fusion --seed 1111 \
  --epochs 2 --use-budgeted-aux --q-type align --warmup-epoch 10 \
  --exp-name budgeted_aux_align_smoke

# Full fixed-budget per-sample auxiliary pilot
python run_experiment.py --dataset MOSI --stage train-fusion --seed 1111 \
  --use-budgeted-aux --q-type align --warmup-epoch 10 \
  --exp-name budgeted_aux_align_p0

# Test the saved fusion model
python run_experiment.py --dataset MOSI --stage test-fusion
```

Supported datasets: `MOSI`, `MOSEI`, `SIMS`. Supported ALW quality functions: `align`, `norm`, `conf`.

`--use-alw` is the legacy adaptive-weight experiment and cannot be combined with
`--use-budgeted-aux`. The budgeted path keeps KL and InfoNCE losses per sample,
redistributes a fixed batch-level auxiliary budget, and logs mean quality and
mean weights after every epoch.

Run the CPU contract tests before uploading changes to the server:

```bash
python -m unittest discover -s tests -v
```

After the fixed-budget smoke checkpoint loads successfully, audit whether its
quality proxy decreases under modality-specific degradation:

```bash
python audit_model_quality.py \
  --dataset MOSI --seed 1111 \
  --exp-name budgeted_aux_align_smoke --q-type align \
  --split test --max-batches 5
```

The five-batch run is a pipeline check. Remove `--max-batches 5` for the full
test-split audit. A credible reliability proxy should generally have a negative
`spearman(severity,q)`, AUROC above 0.5, and a substantial
`fraction(highest_below_clean)`. These are diagnostic gates, not performance
claims.

The P1.1 interventional-reliability pilot is currently limited to MOSI. It
learns a visual reliability head and an audio-specific temporal reliability
head from clean, mild-corruption, and strong-corruption triplets. The learned
scores redistribute the same fixed auxiliary budget. Full corruptions supervise
the reliability ranking from the first epoch, while the corruption visible to
the main sentiment task is blended in gradually over ten epochs:

```bash
python run_experiment.py \
  --dataset MOSI --stage train-fusion --seed 1111 --epochs 2 \
  --use-budgeted-aux --use-interventional-reliability \
  --warmup-epoch 10 --reliability-task-warmup-epoch 10 \
  --exp-name interventional_rel_temporal_warmup_smoke
```

The smoke log must contain both `Budgeted auxiliary epoch` and
`Interventional reliability epoch`. A healthy initial run preserves the
expected budget means, has finite losses, and develops positive `q_gap_v` and
`q_gap_a` values rather than treating corrupted inputs as more reliable.
`task_corruption_progress` should be `0.1` after epoch 1 and `0.2` after epoch
2. The corresponding `q_task_gap_v` and `q_task_gap_a` may be smaller because
the main task only receives the blended corruption.

After the smoke checkpoint loads, audit the learned heads with the same protocol
used for the weak proxies:

```bash
python audit_model_quality.py \
  --dataset MOSI --seed 1111 \
  --exp-name interventional_rel_temporal_warmup_smoke --q-type learned \
  --split test --max-batches 5
```

After the converged `learned` pilot, run equal-budget actionability controls by
adding exactly one of:

```text
--reliability-allocation-control constant
--reliability-allocation-control permuted
--reliability-allocation-control reversed
--reliability-allocation-control oracle
```

`constant` uses uniform allocation. `permuted` cyclically shifts the learned
scores within each batch without sampling extra randomness. `reversed` assigns
the same score multiset in reverse reliability rank. `oracle` uses the known
effective synthetic-corruption severity. All controls preserve the exact mean
KL and InfoNCE budgets, keep reliability-head supervision active, and change
only which samples receive the auxiliary supervision.

To train the reliability heads on ordered corruptions while keeping the
sentiment task inputs clean, set:

```bash
--reliability-task-corrupt-scale 0
```

The default is `1`, which preserves the original progressive task-corruption
behavior. Values between `0` and `1` scale the final corruption strength
without disabling reliability-head supervision.

The original budget warmup scales the mean auxiliary budget from zero to its
configured value. To keep the mean budget equal to the MFON baseline from the
first epoch and warm only the sample-to-sample redistribution, use:

```bash
--budget-warmup-mode allocation
```

The default `scale` mode preserves all earlier experiment behavior.
