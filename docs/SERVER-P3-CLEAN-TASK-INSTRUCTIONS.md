# P3 Clean-Task Diagnostic

Purpose: keep interventional reliability supervision active while preventing
synthetic corruption from entering the sentiment task path.

## Install

Upload `mfon_clean_task_control_p3_20260806.tar.gz` into
`/home/jovyan/projects/MFON`, then run:

```bash
cd /home/jovyan/projects/MFON
tar -xzf mfon_clean_task_control_p3_20260806.tar.gz
python -m unittest discover -s tests -v
python run_experiment.py --help | grep reliability-task-corrupt-scale
```

Expected test count: 27.

## Two-epoch smoke

```bash
python run_experiment.py --dataset MOSI --stage train-fusion --epochs 2 --seed 1112 --use-budgeted-aux --use-interventional-reliability --warmup-epoch 10 --reliability-task-warmup-epoch 10 --reliability-task-corrupt-scale 0 --reliability-allocation-control constant --exp-name p3_constant_clean_task_smoke 2>&1 | tee mosi_p3_constant_clean_task_smoke_1112.log
```

The epoch log must report `task_corruption_progress: 0.0`, while reliability
rank loss and clean-corrupt quality gaps remain non-zero.

## Full diagnostic

Run only after the smoke checkpoint saves and loads successfully:

```bash
python run_experiment.py --dataset MOSI --stage train-fusion --seed 1112 --use-budgeted-aux --use-interventional-reliability --warmup-epoch 10 --reliability-task-warmup-epoch 10 --reliability-task-corrupt-scale 0 --reliability-allocation-control constant --exp-name p3_constant_clean_task 2>&1 | tee mosi_p3_constant_clean_task_1112.log
```

Use the same arguments with `--stage test-fusion` for final inference.

## Decision gate

- Recovery toward the repaired seed-1112 baseline implicates task-path
  corruption.
- Continued degradation implicates the broader budgeted auxiliary or
  reliability-loss training setup.
- Do not start Permuted or seed 1113 P2 runs before this result is reviewed.
