# MFON/P4 manuscript figures

This directory contains reproducible manuscript figures for the current
Neural Computing and Applications working draft.

## Rebuild

Run from the repository root:

```bash
python3 paper/figures/scripts/make_f1_method_overview.py
```

The script reads `paper/figures/data/f1_method_overview.json` and writes the
English submission figure and Chinese review figure as editable SVG, vector
PDF, and 300 dpi PNG.

## Evidence rules

- Quantitative figures may use only frozen manuscript tables or values in
  `docs/experiment-log.md`.
- Two-epoch MOSEI smoke metrics are engineering evidence and must never be
  plotted as performance results.
- No sample-level distribution, scatter, confidence interval, or error bar is
  synthesized when sample-level source data are unavailable.
- F1 is schematic and contains no empirical values.

See `figure-plan.md` for the figure inventory and layout contract, and `qa/`
for visual and scientific-boundary checks.
