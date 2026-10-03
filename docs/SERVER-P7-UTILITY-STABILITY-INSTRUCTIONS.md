# Server P7: repeated-meta visual utility stability

Date: 2026-10-03

Package: `mfon_visual_utility_stability_20261003.tar.gz`

SHA-256: `01b2fa61f9518f1c035f4d998b835c994e72ecb443ce34c48fa27993b73dcb8b`

## Purpose

P6 showed that forward features from an auxiliary sample do not predict a utility
label defined against one random meta batch. P7 is the single allowed repair screen.
It uses fresh samples, evaluates each auxiliary sample against six independent meta
batches, measures sign stability, and predicts mean cross-context utility with a
locked linear baseline and one fixed 16-unit MLP.

This is a read-only audit. It does not update an optimizer or write a checkpoint.
It requires the completed P6 file
`visual_utility_seed1111_final_64pairs_20260928.json` and excludes every sample index
recorded there.

## Expected outputs

- `visual_utility_stability_seed1111_final_24groups_20261003.json`
- `visual_utility_stability_analysis_seed1111_final_20261003.json`
- `visual_utility_stability_seed1111_final_20261003.log`

## Run

Upload the package to `~/projects/MFON`, then run:

```bash
bash -c 'set -e; cd ~/projects/MFON; printf "%s  %s\n" 01b2fa61f9518f1c035f4d998b835c994e72ecb443ce34c48fa27993b73dcb8b mfon_visual_utility_stability_20261003.tar.gz | sha256sum -c -; for f in audit_visual_utility_stability.py analyze_visual_utility_stability.py tests/test_visual_utility_stability.py tests/test_visual_utility_stability_analysis.py run_p7_visual_utility_stability_1111.sh p7_launcher_20261003.log visual_utility_stability_seed1111_final_24groups_20261003.json visual_utility_stability_analysis_seed1111_final_20261003.json visual_utility_stability_seed1111_final_20261003.log; do test ! -e "$f" || { echo "已有文件，停止：$f"; exit 1; }; done; test -s visual_utility_seed1111_final_64pairs_20260928.json; if pgrep -af "^python (run_experiment|audit_.*)\\.py"; then echo "已有实验运行，停止"; exit 1; fi; tar -xzf mfon_visual_utility_stability_20261003.tar.gz -C ~/projects; nohup bash run_p7_visual_utility_stability_1111.sh > p7_launcher_20261003.log 2>&1 < /dev/null & echo "P7已启动，PID=$!"'
```

Live log:

```bash
tail --retry -n 20 -F ~/projects/MFON/visual_utility_stability_seed1111_final_20261003.log
```

## Stop rule

Do not start a five-epoch allocation run unless `all_repair_checks_pass` is true.
If the repair gate fails, stop utility-predictor architecture and threshold tuning.
The next paper decision is then a broader audit route, not another predictor search
on the same checkpoint.
