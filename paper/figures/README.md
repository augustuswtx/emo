# MFON/P4 manuscript figures

This directory contains reproducible manuscript figures for the current
Neural Computing and Applications working draft.

## Rebuild

Run from the repository root:

```bash
python3 paper/figures/scripts/make_f1_method_overview.py
python3 paper/figures/scripts/make_quantitative_figures.py
```

The first script reads `paper/figures/data/f1_method_overview.json`; the second
reads the frozen F2--F5 CSV files in `paper/figures/data/`. Both write English
submission figures and Chinese review figures as editable SVG, vector PDF, and
300 dpi PNG.

## Evidence rules

- Quantitative figures may use only frozen manuscript tables or values in
  `docs/experiment-log.md`.
- Two-epoch MOSEI smoke metrics are engineering evidence and must never be
  plotted as performance results.
- No sample-level distribution, scatter, confidence interval, or error bar is
  synthesized when sample-level source data are unavailable.
- F1 is schematic and contains no empirical values.
- F2--F4 report mean and sample standard deviation over the three frozen MOSEI
  seeds. F5 is a single-session microbenchmark and therefore has no uncertainty
  bars.
- Error bars are sample standard deviations, not confidence intervals or
  significance tests.

See `figure-plan.md` for the figure inventory and layout contract, and `qa/`
for visual and scientific-boundary checks.
