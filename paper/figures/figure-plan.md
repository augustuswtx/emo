# Executable figure plan and layout contract

Target: Neural Computing and Applications, double-column research article.
Backend: Python (`matplotlib`) exclusively for drawing, export, previews, and
visual QA.

## Shared layout specification

- Primary width: 183 mm (double column); inspect a 89 mm downscaled copy for
  legibility even when the final placement is double column.
- Typeface: Arial/Helvetica/DejaVu Sans for English; PingFang SC fallback for
  Chinese. Final body text target is 6.5--8 pt.
- Background: white. Use a cool blue family for the clean MFON path, violet
  for reliability learning, amber for auxiliary supervision, and neutral grey
  for boundaries and controls.
- Red/green is never the only encoding. Path identity is also encoded by solid
  versus dashed borders, arrow style, band labels, and direct labels.
- Required exports: editable SVG, vector PDF, and 300 dpi PNG preview. Every
  quantitative figure additionally requires a standalone CSV and source note.
- Figure legends must state data scope, seed count, spread definition, and
  claim boundaries. Three seeds are descriptive and do not establish
  statistical significance.

## Figure inventory

| ID | Role and core conclusion | Archetype | Source | Status / gate |
|---|---|---|---|---|
| F1 | Reliability controls only training-time auxiliary-loss allocation; clean features remain on the MFON task path and inference has no reliability-gated fusion. | Schematic-led composite | Frozen method equations and implementation | First review version produced in English and Chinese |
| F2 | A quality score is trustworthy only after passing five complementary audits: granularity, monotonicity, confounds, actionability, and non-collapse. | Compact audit flow/matrix | Manuscript Section 4 and failure analysis | Next |
| F3 | Frozen MOSI reliability heads track synthetic degradation strongly, with a residual audio-length confound. | Quantitative grid using aggregate point estimates | Frozen Table 1 / experiment log | Allowed; no sample-level curves or distributions without raw audit files |
| F4 | Under equal budget, Learned-minus-Constant gains are concentrated in binary/regression metrics and are not uniform across fine-grained classification. | Direction-normalized small multiples | Frozen Table 2 / experiment log | Allowed; preserve the negative Acc-7 result |
| F5 | Cross-dataset MOSEI method comparison. | Quantitative grid | Formal reloaded tests only | Blocked until seed-1111 Constant and Learned formal results are both complete; smoke metrics prohibited |

## F1 figure contract

Core conclusion: Audited reliability redistributes an exactly fixed batch
budget across per-sample auxiliary losses during training, while the clean
MFON feature-fusion path alone produces sentiment predictions at inference.

- Archetype: schematic-led composite.
- Hero evidence: uninterrupted clean text/vision/audio to MFON fusion and
  sentiment prediction path.
- Supporting mechanism: ordered clean--mild--strong interventions train visual
  and acoustic reliability heads; clean scores are normalized under an exact
  finite-batch budget and weight unreduced KL/InfoNCE losses.
- Controls visible in the figure: training-only boundary, frozen teachers,
  per-sample losses before reduction, and `mean_i w_i^m = delta_m`.
- Reviewer risk: an arrow from reliability to fusion would falsely imply
  inference-time dynamic gating. The figure therefore contains no such arrow
  and includes an explicit boundary statement.
- Statistics/source data: not applicable; F1 is a method schematic and contains
  no empirical values.
- Export size: 183 x 112 mm; editable text in SVG/PDF; 300 dpi PNG preview.
