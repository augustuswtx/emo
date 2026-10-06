# Historical CCF-B experiment plan: sparse-supervised task-utility allocation

Date: 2026-09-28

Status: superseded on 2026-10-06 after P6 and P7 failed every predictor gate.

> Do not execute P2 or P3 in this document. The active experiment plan is
> [the actionability-audit plan](ccfb-actionability-audit-plan-20261006.md).

## 1. Claim-evidence matrix

| Intended claim | Required evidence | Failure interpretation |
|---|---|---|
| Reliability is not update utility | Correlations and ranking tests across datasets, seeds, checkpoints, and score families | If strongly aligned, simplify the method and explain why earlier negative results differed |
| Exact visual utility is locally valid | First-order sign versus reversible actual step at early/mid/late checkpoints | If unstable, do not train a utility predictor |
| Utility can be amortized cheaply | Held-out checkpoint/time prediction, calibration, coverage, overhead | If weak, abandon positive method track |
| Correct allocation causes gains | Correct versus gate-only, permuted, inverse, Constant under matched budgets | If tied, gains are not attributable to assignment |
| Method generalizes | Three datasets and two fusion backbones | If MFON-only, narrow claims and venue target |
| Method improves robustness | Multiple unseen corruption families and clean data | If robustness trades off clean performance, report the frontier rather than claiming universal improvement |

## 2. Datasets and splits

| Dataset | Role | Metrics |
|---|---|---|
| CMU-MOSEI | Primary development and full study; large scale and diverse speakers/topics | MAE, Corr, Acc-2/F1 with and without zero, Acc-5, Acc-7 |
| CMU-MOSI | Small English replication and rapid ablation; not the sole evidence source | Same standard regression/classification metrics |
| Current CH-SIMS/SIMS | Chinese transfer and analysis using its independent unimodal annotations | Dataset-standard regression/classification metrics plus modality-specific analyses |
| CH-SIMS v2.0 | New locked confirmation set after the method and thresholds are frozen | Same task metrics, allocation diagnostics, and raw/realistic degradation where media are available |

Use official train/validation/test splits. Utility predictor labels come from disjoint
subsets of the training split. Validation selects checkpoints and locked
hyperparameters; the test split is used once per frozen configuration.

### Dataset decision

Do not replace the three datasets already integrated in the repository. Replacing
them would discard comparable MFON results without solving the central evidence gap.
Instead, assign them different roles and add CH-SIMS v2.0 as an independent
confirmation dataset only after the method is frozen.

The current MOSI/MOSEI/SIMS test results have already been inspected during method
development, so they cannot be described as untouched confirmation. The new dataset
or version must remain unopened for final method selection. MuSe-CaR is a useful
future real-world application set, but its continuous emotion, engagement, topic, and
trustworthiness tasks are not identical to the present sentence-level sentiment
regression protocol. It should not replace the core datasets or be added merely to
increase the dataset count.

If raw media cannot be obtained, restrict claims to pre-extracted feature-level
degradation. A claim about real audio/video corruption requires at least one raw-media
experiment, preferably on CH-SIMS v2.0 or a legally obtained raw subset of MOSEI.

## 3. Backbones

1. **MFON:** primary backbone because the frozen unimodal teachers and auxiliary KL
   already exist.
2. **MISA or MulT:** select one after a two-day integration estimate. Add the same
   frozen-teacher representation loss so that only the allocation mechanism changes.

Do not replace BERT, fusion layers, and allocation simultaneously. Backbone upgrades
are a separate experiment, not part of the central causal comparison.

## 4. Baselines

### Central matched controls

- Repaired MFON without the new auxiliary allocation;
- fixed Constant auxiliary weight;
- current learned degradation-reliability allocation;
- per-sample-only variant without reliability loss;
- gate-only uniform;
- proposed correct utility allocation;
- proposed Permuted allocation;
- proposed Inverse allocation.

### Strong external methods

- OGM-GE;
- PMR or AGM;
- MMPareto;
- SLMV;
- CMoB;
- G2D;
- ARL;
- one global task/KD gradient method, preferably DTO-KD or a faithful PCGrad-style
  control.

Prioritize official code and identical feature inputs. Mark results copied from papers
as non-comparable unless the protocol is identical. At least four recent strong
baselines should be rerun in the same environment.

## 5. Predictor evaluation

### Features

Evaluate nested feature groups:

1. `q` only;
2. `q + KL + student/teacher alignment`;
3. group 2 plus task residual, confidence, norms, length, energy, and epoch;
4. group 3 without `q`, testing whether the old reliability score contributes.

### Splits

- Hold out later training steps from earlier steps.
- Hold out one checkpoint stage.
- Hold out one seed for a transfer diagnostic.
- Never randomly split records from the same batch across train and test.

### Metrics

- AUROC and AUPRC for positive utility;
- Spearman and Kendall rank correlation for signed utility;
- expected calibration error and reliability diagram;
- precision, recall, coverage, and harmful-update rate under abstention;
- batch-wise and checkpoint-wise variance;
- predictor parameters and forward time.

## 6. Training protocol

1. Start with visual KL only; audio KL and both InfoNCE losses remain uniform.
2. Generate exact labels every `K=16` batches in the first pilot. Compare `K` in
   `{8,16,32}` only after feasibility passes.
3. Use all samples from a refresh auxiliary batch to update the utility predictor.
4. Stop gradients from the predictor features into the fusion backbone.
5. Maintain a bounded rolling label buffer stratified by epoch and utility sign.
6. Lock the confidence threshold on training-held-out records before full runs.
7. Preserve the visual KL mean coefficient on active batches and log realized weighted
   KL and its gradient norm.
8. Use the same checkpoint selection rule for every arm.

## 7. Ablations

| Ablation | Question |
|---|---|
| No sparse exact labels | Does ordinary end-to-end score learning suffice? |
| No uncertainty / no abstention | Is conservative rejection necessary? |
| No fixed-budget normalization | Are gains caused by a larger auxiliary coefficient? |
| Gate-only uniform | Is sample assignment useful beyond turning KL off? |
| Permuted and Inverse | Does the learned ordering matter? |
| `q` removed from predictor | Does degradation reliability add information after task features? |
| KL-only versus KL+InfoNCE | Does batch-dependent InfoNCE allocation help or confound? |
| Visual-only versus visual+audio | Does the failed audio utility signal remain a boundary? |
| Refresh cadence | Accuracy/overhead trade-off |
| Projection block versus larger parameter block | How local is the utility label? |

## 8. Robustness and stress tests

- Gaussian noise at seen and unseen severities;
- timestep dropout;
- contiguous masking;
- temporal shift;
- full modality missingness;
- shuffled auxiliary scores;
- teacher checkpoint degradation or weaker teacher;
- batch-size sensitivity;
- label-buffer distribution shift across epochs.

Report clean and corrupted results together. Do not call a score robust because it
only detects the corruption used to train it.

## 9. Statistics

- Development pilots: one seed, clearly labeled exploratory.
- Central final matrix: minimum three seeds; use five for the proposed method,
  Constant, and strongest baseline if compute allows.
- Report every seed, mean, sample standard deviation, and 95% confidence intervals.
- Use paired bootstrap over test samples for MAE differences and Fisher-transformed
  intervals for correlation; treat seed-level inference cautiously when `n=3`.
- Predeclare the primary metric pair as MAE (lower) and Corr (higher). Classification
  metrics are secondary and must still be reported.
- Never select a method because one metric on one seed improves.

## 10. Efficiency reporting

Record:

- wall-clock training time after warmup;
- exact-label generation time;
- predictor forward/backward time;
- peak allocated GPU memory;
- additional parameters;
- refresh cadence and label-buffer size;
- inference time and parameters used at test time.

The target is at most 15% added training time and no test-time overhead. If exact-label
refresh exceeds this, reduce cadence only after predictor accuracy at the new cadence
is measured; do not reuse stale gradients, because that route already failed.

## 11. Execution priority

### P0: finish attribution already in progress

Complete the existing per-sample-only C4 run and preserve its checkpoint, log, hash,
and test result. This closes the old method's component attribution.

### P1: new read-only feasibility

Collect richer cheap features together with exact utility at early, middle, and late
checkpoints for seed 1111, then repeat the locked analysis at seed 2222. No 25-epoch
new-method run is allowed before Gate B passes.

The seed-1111 final-checkpoint P6 sample-only predictor failed on 2026-10-03
(`all` AUROC `0.510`, Spearman `0.013`). Before expanding to other checkpoints,
run the single P7 repeated-meta stability audit on fresh samples. It tests whether
the failure came from random meta-batch context and whether an averaged target is
predictable. If P7 fails its locked stability and predictor checks, stop P1 and use
the audit route; do not proceed to P2.

P7 subsequently failed every repair check on 2026-10-06. P1 is closed, and P2/P3
below are retained only as the preregistered historical stop path.

### P2: short training pilot

Run Constant, gate-only uniform, and proposed allocation for five epochs at one seed.

### P3: central MOSEI matrix

Only after P2 passes, run three seeds for Constant, Old-q, gate-only, proposed,
Permuted, and Inverse.

### P4: breadth and submission package

Add MOSI/SIMS, second backbone, strong external baselines, robustness, efficiency,
figures, and release materials.

## 12. Result-table templates

### Main result

| Dataset | Backbone | Method | Seeds | MAE down | Corr up | Acc-2 up | Train overhead | Test overhead |
|---|---|---|---:|---:|---:|---:|---:|---:|
| MOSEI | MFON | Constant |  |  |  |  |  |  |
| MOSEI | MFON | Old-q |  |  |  |  |  |  |
| MOSEI | MFON | Proposed |  |  |  |  |  |  |

### Actionability

| Method | Active coverage | Selected harmful rate | Gate-only delta | Permuted delta | Inverse delta | Realized KL budget |
|---|---:|---:|---:|---:|---:|---:|
| Proposed |  |  |  |  |  |  |

### Predictor quality

| Dataset | Seed/checkpoint split | Feature group | AUROC | AUPRC | Spearman | ECE | Coverage |
|---|---|---|---:|---:|---:|---:|---:|
| MOSEI | early/mid -> late | all |  |  |  |  |  |
