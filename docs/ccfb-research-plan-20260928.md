# CCF-B research plan: from modality reliability to training utility

Date: 2026-09-28
Status: actionability-audit route active after the P6/P7 utility-predictor gates failed

## 0. Paper positioning

Application setting: **multimodal sentiment analysis of online opinion videos**.

Current working Chinese title:

**可靠性不等于训练效用：多模态情感辅助学习分数的作用性审计**

Working English title:

**Reliability Is Not Training Utility: An Actionability Audit of Sample-wise Auxiliary Weighting in Multimodal Sentiment Analysis**

The paper addresses one narrow question: when text is the dominant sentiment signal,
do sample-level reliability, quality, uncertainty, and consistency scores actually
identify useful auxiliary training actions? It does not claim that every score is
invalid or that the observed boundaries apply to all video tasks.

This is an upgrade of the current reliability manuscript, not a second paper. The
existing manuscript supplies the first audited score, per-sample loss interface,
fixed-budget allocator, and matched controls. P6 and P7 show that its local utility
cannot be treated as a stable sample attribute or cheaply predicted from the tested
features. The upgraded manuscript therefore expands the audit across score families,
datasets, and backbones and is submitted as the single CCF-B paper required for the
thesis.

## 1. Decision

Do not discard the completed paper, and do not keep extending its old reliability
score as if more seeds could repair the interpretation. Use the completed work as
the diagnosis and experimental foundation for a stronger question:

> **When is a sample-level multimodal score sufficiently stable and actionable to
> justify changing auxiliary training weights?**

The active direction is the broader audit paper, **Reliability Is Not Training
Utility**. The sparse-supervised task-utility predictor was the predeclared positive
method route; it stopped after both P6 and its single P7 repair failed. These failures
are retained as evidence and are not hidden or followed by unrestricted model search.

This upgrade spans two analysis layers of the thesis field **trustworthy auxiliary
learning for non-text modalities in text-dominant multimodal sentiment analysis**:
what a score measures and whether acting on it improves training. The current version
supplies the input-reliability case study, per-sample loss interface, fixed-budget
allocator, and actionability controls. The revised version adds a common audit
protocol, representative score families, and cross-setting evidence. Do not submit
the overlapping old and upgraded versions as separate papers. Their exact chapter
and evidence mapping is recorded in
`docs/thesis-integration-roadmap-20260928.md`.

## 2. What the existing work contributes

The existing work is useful in five concrete ways:

1. It provides a complete MFON reproduction and a fixed-budget per-sample auxiliary
   objective.
2. It separates degradation detection from teacher-target fidelity and downstream
   task utility.
3. It supplies matched constant, permuted, inverse, and oracle-style controls.
4. It gives negative evidence that the old reliability score is not a valid
   cross-sample allocation score.
5. It gives positive numerical evidence that visual parameter-space gradient utility
   predicts a reversible local task-loss direction, while also measuring why direct
   online use is too expensive.

These remain in the upgraded manuscript as its motivation, evaluation protocol, and
failure boundary. They are not enough by themselves for a positive allocation claim.

## 3. Stopped utility-predictor candidate and retained diagnostic

### 3.1 Exact but sparse utility label

For modality `m`, auxiliary sample `i`, current parameters `theta_m`, and a disjoint
training meta batch `M`, define

`u_i^m = <grad_theta_m L_task(M), grad_theta_m L_aux(i,m)>`.

A positive value predicts that a small update in the negative auxiliary-gradient
direction will reduce task loss on `M`. Existing audits support this local
interpretation for the visual projection block at one final checkpoint; they do not
yet establish training-time or cross-checkpoint validity.

During training, calculate exact labels on only one refresh batch every `K` batches.
Use disjoint training samples for meta and auxiliary batches. Never use validation or
test labels to train the score.

### 3.2 Cheap utility predictor tested in P6/P7

P6/P7 tested stopped-gradient predictors from forward-pass features already available
in MFON:

- old degradation reliability `q`;
- per-sample teacher KL;
- student-teacher cosine similarity;
- student and teacher representation norms;
- main-task residual `|prediction - label|` on the training sample;
- prediction magnitude/confidence;
- modality sequence length and energy;
- epoch progress and recent batch statistics.

The candidate was intended to output the probability that `u > 0`. Complete
meta/auxiliary or auxiliary-group units were held out. Both the single-context and
six-context-mean targets failed the predeclared gates, so this component is not part
of the proposed paper method.

### 3.3 Cancelled downstream allocation policy

The planned policy would have allocated visual KL only when a calibrated lower
confidence bound passed a locked threshold, with fixed mean coefficients on active
batches. Because P6/P7 failed, this policy was never authorized for short or full
training and is not claimed as a contribution.

The following controls remain part of the general actionability audit:

- `Constant`: uniform visual KL on every batch;
- `Old-q`: current degradation-score allocation;
- `Gate-only uniform`: identical active-batch and active-sample mask, uniform weight;
- `Permuted`: predicted scores assigned to the wrong samples;
- `Inverse`: rank order reversed;
- `Exact local diagnostic`: exact utility on audited batches only, never presented as
  a stable sample score.

These controls distinguish correct assignment from lower total regularization,
abstention, or extra compute.

### 3.4 Scientific value of the stopped candidate

The failed mechanism followed one testable chain:

`auxiliary update -> measured local task effect -> repeated-context stability ->
cheap prediction -> fixed-budget action`.

The chain broke at repeated-context stability and prediction, before allocation
training. This is evidence for the audit hierarchy and prevents a negative result
from being repackaged as an unverified method.

## 4. Required distinction from closest work

| Closest work family | Existing emphasis | Required distinction here |
|---|---|---|
| Meta-reweighting | Learns weights from validation gradients | Audit stability across independent meta contexts before calling a value sample-specific |
| Sample-level modality valuation | Estimates modality contribution | Separate inference contribution from auxiliary-update actionability |
| Gradient-conflict methods | Resolves task-level or batch-level conflicts | Test local validity, context stability, and final fixed-budget action separately |
| Gradient-guided distillation | Uses task gradients to guide teachers or losses | Include Permuted/Inverse and repeated-context controls |
| Quality-aware fusion | Uses quality at inference or in fusion | Audit whether the same score justifies training-time reweighting |

The audit is no longer distinctive if it only reports correlations on one model. Its
novelty depends on the four-level protocol, matched actionability controls, multiple
score families, reusable result schemas, and cross-setting evidence. Exact novelty
relative to every recent audit paper still requires final literature verification.

## 5. Go/no-go gates

### Gate A: utility-label validity

Run at least three training stages (early, middle, late), two seeds, and fresh
training pairs. The visual exact label must retain at least 85% sign agreement with a
reversible small-step measurement. If it fails, stop calling it utility and move to
the audit track.

### Gate B: predictor feasibility

On temporally later or checkpoint-held-out records, require all of:

- AUROC at least 0.70 for `u > 0`;
- AUPRC at least 0.10 above positive prevalence;
- Spearman correlation at least 0.25 with signed utility;
- harmful-update rate in the selected top quarter at least 30% lower than the
  unfiltered rate;
- stable calibration across two seeds.

These are internal engineering gates, not guaranteed paper results. Do not tune them
on the final test set.

**2026-10-03 P6 result.** The first seed-1111 final-checkpoint screen collected
2,048 labels from 64 disjoint meta/auxiliary pairs. All auxiliary-sample-only
predictors were near chance: `q_only` AUROC `0.516`, `q+KL+alignment` `0.515`, all
features `0.510`, and all features without `q` `0.482`. The all-feature Spearman
correlation was `0.013`, and its selected top quarter retained a `0.444` harmful
rate. Gate B therefore failed and no five-epoch training pilot is authorized.

The label is conditional on both an auxiliary sample and a randomly drawn meta
batch, while P6 exposed only auxiliary-sample features to the predictor. One final
repair screen is allowed before abandoning the positive method route: P7 evaluates
the same fresh auxiliary samples against six independent meta batches, measures
sign stability, uses mean cross-context utility as the target, and compares the
locked linear baseline with one fixed 16-unit MLP. It passes only if at least half
the samples have a 5/6 sign majority and the predeclared MLP thresholds above all
pass on complete held-out auxiliary groups. Failure ends utility-predictor tuning;
do not try additional architectures or thresholds on the same data.

**2026-10-06 P7 result and route decision.** P7 collected 768 fresh auxiliary
samples, each evaluated against six independent meta batches. A first random context
matched the cross-context mean sign only `0.641` of the time; mean pairwise sign
agreement was `0.496`; only `0.215` of samples had a 5/6 sign majority; and the
context reliability ratio was `0.171`. Linear and fixed-MLP predictors remained near
chance. The primary `mlp_all` model obtained AUROC `0.494`, Spearman `-0.005`, and a
top-quarter harmful rate of `0.562`; every repair check failed. The utility-predictor
route is therefore closed. No short or full allocation training is authorized from
this target.

### Gate C: short training pilot — not entered

The utility-predictor candidate did not enter this gate.

### Gate D: central full experiment — cancelled for the predictor

No predictor-based central training is permitted. The active audit track has a new
central matrix specified in `docs/ccfb-actionability-audit-plan-20261006.md`.

## 6. CCF-B evidence package

A defensible package should contain:

1. **Mechanistic finding:** degradation reliability, target fidelity, and update
   utility are empirically distinct.
2. **Evaluation contribution:** a four-level protocol separating degradation
   sensitivity, target fidelity, update stability, and allocation actionability.
3. **Actionability evidence:** correct assignment is compared with gate-only,
   Constant, Permuted, and Inverse controls, including negative results.
4. **Breadth:** MOSEI, MOSI, and SIMS; MFON plus one structurally different fusion
   backbone.
5. **Robustness:** clean, Gaussian corruption, timestep dropout, contiguous masking,
   temporal shift, and modality missingness.
6. **Statistics:** at least three seeds for the full matrix, confidence intervals,
   paired sample bootstrap where appropriate, and all failed or mixed metrics shown.
7. **Efficiency:** audit cost, training overhead of evaluated policies, GPU memory,
   and inference overhead where applicable.
8. **Reproducibility:** frozen configurations, checkpoint hashes, raw per-seed tables,
   code, and commands.

## 7. Active actionability-audit track

The P6/P7 gates failed, so expand the existing audit into a method-comparison
benchmark:

- three datasets;
- four to six representative models or score families;
- degradation detection, target fidelity, actual local update utility, and final
  allocation actionability;
- fixed-budget Constant, Permuted, and Inverse controls;
- public audit scripts and result schema.

The intended contribution is a general empirical result and evaluation protocol:
scores that detect corruption or estimate uncertainty must not be assumed to predict
whether an auxiliary training action helps. A single MFON model on MOSEI is
insufficient for this track; breadth, matched controls, and reusable tooling are the
novelty. The full active plan is in
`docs/ccfb-actionability-audit-plan-20261006.md`.

## 8. Venue strategy

The March 2026 CCF directory lists IEEE TMM, ACM TOMM, EMNLP, ICME, and ICASSP as
CCF-B venues in their respective categories. The directory itself is a recommendation
list and does not guarantee the quality or acceptance of an individual paper.

Recommended order:

1. **Primary: IEEE Transactions on Multimedia or ACM TOMM.** The journal format fits
   a combined method, audit, robustness, and efficiency package.
2. **Conference alternative: EMNLP main.** Use only if the story is centered on
   multimodal sentiment learning and language-dominant failure; a Findings paper does
   not satisfy a main-track CCF-B target.
3. **Engineering alternative: ICME or ICASSP full paper.** Suitable for a narrower
   method with strong signal-processing/multimedia experiments.

The current NCA manuscript is useful as a source draft, but NCA is listed as CCF-C in
the current CCF AI table. A CCF-B submission needs the new evidence package rather
than a title or venue change.

## 9. Schedule on one RTX 4090-class GPU

| Phase | Work | Expected time |
|---|---|---:|
| 1 | Freeze P6/P7 and consolidate the four-level result schema | 3--5 days |
| 2 | Implement and lock five score families on MFON/MOSEI | 1--2 weeks |
| 3 | Port read-only audits to MOSI/SIMS and assess a second backbone | 2--3 weeks |
| 4 | Run the frozen three-seed actionability matrix | 3--5 weeks |
| 5 | Second-backbone confirmation, robustness, ablations, efficiency | 3--5 weeks |
| 6 | Paper figures, writing, internal review, release package | 2--4 weeks |

Total expected research time is roughly 10--16 weeks if the second-backbone port is
feasible. Publication cannot be guaranteed; the audit requires breadth and reusable
evidence to be reviewable at CCF-B level.
