# Server P6: visual utility predictor feasibility

Date: 2026-09-28

Package: `mfon_visual_utility_feasibility_20260928.tar.gz`  
SHA-256: `7d4ba9cf6b06293ed53152fa9cab298bb4ff4bc3b3e4e4d2ad12aa6eaa16004f`

## Purpose

This is the first go/no-go screen for the proposed CCF-B method. It collects 2,048
visual sample records from 64 disjoint meta/auxiliary batch pairs at the existing
seed-1111 final checkpoint. Each record contains an exact local visual utility label
and cheap forward-pass features. A deterministic linear predictor is evaluated on
held-out complete batch pairs.

This run cannot establish Gate B by itself because it uses one seed and one final
checkpoint. It answers the cheaper preliminary question: is there enough
cross-batch signal to justify collecting multiple checkpoints and seeds?

## Expected files

- `visual_utility_seed1111_final_64pairs_20260928.json`
- `visual_utility_predictor_seed1111_final_20260928.json`
- `visual_utility_feasibility_seed1111_final_20260928.log`

## Install and run

Upload the package into `~/projects/MFON`, then run the single command below. It
refuses to overwrite the four installed files or any output file.

```bash
bash -c 'set -e; cd ~/projects/MFON; printf "%s  %s\n" 7d4ba9cf6b06293ed53152fa9cab298bb4ff4bc3b3e4e4d2ad12aa6eaa16004f mfon_visual_utility_feasibility_20260928.tar.gz | sha256sum -c -; for f in audit_visual_utility_dataset.py analyze_visual_utility_predictor.py tests/test_visual_utility_dataset.py tests/test_visual_utility_predictor.py run_p6_visual_utility_feasibility_1111.sh p6_launcher_20260928.log; do test ! -e "$f" || { echo "已有文件，停止：$f"; exit 1; }; done; tar -xzf mfon_visual_utility_feasibility_20260928.tar.gz -C ~/projects; nohup bash run_p6_visual_utility_feasibility_1111.sh > p6_launcher_20260928.log 2>&1 < /dev/null & echo "P6已启动，PID=$!"'
```

Live view:

```bash
tail -n 20 -f ~/projects/MFON/visual_utility_feasibility_seed1111_final_20260928.log
```

## Stop rule

Do not start a 25-epoch allocation run from this result. If the full feature model is
near chance or fails most numeric checks, inspect leakage and feature definitions once;
then stop the positive method route if the failure repeats on a fresh sample. If it
passes, collect seed 2222 and early/middle checkpoints before implementing training.

## Result summary command

```bash
cd ~/projects/MFON && python -c 'import json; d=json.load(open("visual_utility_predictor_seed1111_final_20260928.json")); print("train/test",d["train_samples"],d["test_samples"]); [print(k,"AUROC",round(v["auroc"],3),"AUPRC-margin",round(v["auprc_above_prevalence"],3),"rho",round(v["spearman_score_utility"],3),"harm-reduction",round(v["top_quarter_relative_harmful_reduction"],3)) for k,v in d["models"].items()]; print("numeric checks",d["internal_gate_b"])'
```
