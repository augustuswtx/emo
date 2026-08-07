# P4 True Fixed-Budget Schedule

Purpose: match the repaired MFON auxiliary-budget mean from epoch 1 and warm
only the sample-to-sample reliability allocation.

## Install

Upload `mfon_fixed_budget_schedule_p4_20260807.tar.gz` into
`/home/jovyan/projects/MFON`, then run:

```bash
cd /home/jovyan/projects/MFON
tar -xzf mfon_fixed_budget_schedule_p4_20260807.tar.gz
python -m unittest discover -s tests -v
python run_experiment.py --help | grep budget-warmup-mode
```

Expected test count: 29.

## Two-epoch smoke

```bash
python run_experiment.py --dataset MOSI --stage train-fusion --epochs 2 --seed 1112 --use-budgeted-aux --use-interventional-reliability --warmup-epoch 10 --budget-warmup-mode allocation --reliability-task-warmup-epoch 10 --reliability-task-corrupt-scale 0 --reliability-allocation-control constant --exp-name p4_constant_true_budget_smoke 2>&1 | tee mosi_p4_constant_true_budget_smoke_1112.log
```

At epoch 2, all four mean weights must be `0.5`, their standard deviations must
be zero, and `task_corruption_progress` must be zero.

## Full diagnostic

After checkpoint load/inference succeeds, run:

```bash
python run_experiment.py --dataset MOSI --stage train-fusion --seed 1112 --use-budgeted-aux --use-interventional-reliability --warmup-epoch 10 --budget-warmup-mode allocation --reliability-task-warmup-epoch 10 --reliability-task-corrupt-scale 0 --reliability-allocation-control constant --exp-name p4_constant_true_budget 2>&1 | tee mosi_p4_constant_true_budget_1112.log
```

Use the same arguments with `--stage test-fusion` for final inference.

## Stop rule

- Recovery toward the repaired seed-1112 baseline justifies a Learned run with
  the same schedule.
- Continued material degradation ends the training-loss allocation branch.
  Do not add further loss weights, warmups, or corruption hyperparameters.
