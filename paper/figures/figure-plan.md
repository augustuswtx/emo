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
| F2 | On frozen MOSEI, Learned improves the prespecified regression endpoints over equal-budget Constant, but secondary metrics are mixed and the method does not uniformly beat repaired MFON. | Quantitative grid with direction-normalized delta panel | Frozen three-seed clean-test table | Complete in English and Chinese; mean ± sample SD, with no significance claim |
| F3 | MOSEI reliability scores strongly detect and rank the Gaussian corruption used in training, while retaining measurable correlations with simple feature statistics. | Quantitative audit grid | Frozen full-test Gaussian audits | Complete in English and Chinese; confounds remain descriptive |
| F4 | Gaussian-trained reliability does not generalize uniformly to held-out corruptions, and corruption detectability differs from task sensitivity. | Paired horizontal-bar audit | Frozen full-test held-out-corruption audits | Complete in English and Chinese; preserves near-chance and below-chance findings |
| F5 | P4 adds negligible optimized-parameter and peak-memory overhead without changing the deployed inference footprint. | Quantitative efficiency grid | Uniform seed-1111 efficiency microbenchmark | Complete in English and Chinese; single-session timings are not a speedup claim |

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

## F2--F5 quantitative contracts

### F2: frozen MOSEI task results

- Hero evidence: MAE, correlation, and loss means with sample-SD error bars.
- Supporting evidence: favourable-direction Learned-minus-Constant deltas for
  all reported endpoints; lower-is-better metrics are sign-reversed only in
  this delta panel and are labelled explicitly.
- Reviewer risk: the visual must not imply statistical significance or uniform
  superiority over repaired MFON.

### F3: Gaussian reliability and confounds

- Hero evidence: clean/corrupt AUROC and monotonicity under the Gaussian
  intervention used in reliability training.
- Supporting evidence: fraction below clean at maximum severity and clean-score
  correlations with length, energy, and absolute sentiment label.
- Reviewer risk: high in-family detection must not be presented as general
  quality estimation; correlation audits cannot exclude unmeasured confounds.

### F4: held-out corruption stress audit

- Hero evidence: modality-specific AUROC against the 0.5 chance reference.
- Supporting evidence: matched task-prediction change measured by corrupt-minus-
  clean MAE.
- Reviewer risk: audio missingness is easy to detect while barely changing task
  predictions; visual missingness is unstable across seeds. Detection and task
  utility must remain separate claims.

### F5: efficiency and resource cost

- Hero evidence: additional optimized parameters as a percentage of the MFON
  optimizer scope.
- Supporting evidence: inference and forward-plus-backward latency, throughput,
  and peak allocated memory.
- Reviewer risk: the timing run is one sequential single-GPU session with no
  uncertainty estimate. Small latency differences are descriptive, not a
  general speed comparison.
