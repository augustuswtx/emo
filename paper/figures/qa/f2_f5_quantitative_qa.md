# F2--F5 quantitative-figure QA

Date: 2026-09-15

## Evidence and statistics

| Check | Result |
|---|---|
| F2 values match the frozen MOSEI three-seed clean-test table | Pass |
| F3 values match the full-test Gaussian audit and confound summary | Pass |
| F4 values match the four full-test held-out-corruption audits | Pass |
| F5 values match the uniform seed-1111 efficiency audit | Pass |
| F2 uses paired seed-level points and favorable-direction deltas without synthesized uncertainty | Pass |
| F3--F4 error bars are identified as sample SD over seeds 1111--1113 | Pass |
| F5 omits uncertainty bars and identifies the single-session protocol | Pass |
| No confidence interval, p value, or significance claim is synthesized | Pass |
| Negative and near-chance cross-corruption findings remain visible | Pass |
| Lower-is-better sign reversal is limited to and disclosed in F2d | Pass |

## Visual and export checks

| Check | Result |
|---|---|
| Python/matplotlib is the exclusive drawing and export backend | Pass |
| English and Chinese variants share geometry and numeric content | Pass |
| Method and modality colors are consistent across panels | Pass |
| Hatching makes method/modality identity available without color | Pass |
| Panel labels are lowercase, bold, and consistently positioned | Pass |
| SVG keeps labels as editable text (`svg.fonttype=none`) | Pass |
| PDF is vector and PNG is exported at 300 dpi | Pass |
| Final width is 183 mm and full-resolution previews were inspected | Pass |
| Source values are available as standalone CSV files | Pass |

## Claim boundaries

- F2 exposes the three paired seed trajectories and supports only favourable
  descriptive mean directions over Constant on MAE, correlation, and loss,
  not stability, statistical significance, or comprehensive superiority.
- F3 supports in-family Gaussian degradation detection, not a general quality
  estimator.
- F4 distinguishes degradation detectability from task utility and documents
  failed cross-corruption transfer.
- F5 supports low measured overhead under one controlled session, not a general
  inference speedup.

## Rebuild command

```bash
python3 paper/figures/scripts/make_quantitative_figures.py
```
