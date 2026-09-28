# CCF-B research plan: from modality reliability to training utility

Date: 2026-09-28  
Status: method and evidence plan; no positive result is assumed

## 1. Decision

Do not discard the completed paper, and do not keep extending its old reliability
score as if more seeds could repair the interpretation. Use the completed work as
the diagnosis and experimental foundation for a stronger question:

> **When should a sample-modality teacher loss be applied, and can its true task
> utility be predicted cheaply enough to guide training?**

The primary CCF-B direction is **sparse-supervised task-utility allocation**. The
backup direction is a broader audit paper, **Reliability Is Not Training Utility**.
The two directions share code, datasets, controls, and most measurements, so the
backup is not wasted work.

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

These become the motivation, evaluation protocol, and failure boundary of the new
paper. They are not enough by themselves for a positive allocation claim.

## 3. Proposed method

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

### 3.2 Cheap utility predictor

Train a small stopped-gradient predictor from forward-pass features already available
in MFON:

- old degradation reliability `q`;
- per-sample teacher KL;
- student-teacher cosine similarity;
- student and teacher representation norms;
- main-task residual `|prediction - label|` on the training sample;
- prediction magnitude/confidence;
- modality sequence length and energy;
- epoch progress and recent batch statistics.

The predictor outputs both the probability that `u > 0` and a normalized magnitude.
It is trained on a rolling buffer of sparse exact labels. Split the buffer by time or
checkpoint so that reported predictor quality is measured on later, unseen training
states rather than on records used to fit it.

### 3.3 Conservative abstention and fixed budget

Allocate visual KL only when a calibrated lower confidence bound for positive utility
passes a locked threshold. When no sample passes, abstain from visual KL for that
batch. On active batches, normalize weights to preserve the original mean visual KL
coefficient. Keep audio KL and both InfoNCE terms uniform in the first version because
the present audio utility signal failed its numerical checks.

Required controls:

- `Constant`: uniform visual KL on every batch;
- `Old-q`: current degradation-score allocation;
- `Gate-only uniform`: identical active-batch and active-sample mask, uniform weight;
- `Permuted`: predicted scores assigned to the wrong samples;
- `Inverse`: rank order reversed;
- `Exact sparse oracle`: exact utility on refresh batches only, as an upper-bound
  diagnostic rather than the final method.

These controls distinguish correct assignment from lower total regularization,
abstention, or extra compute.

### 3.4 Why this is not a module stack

The mechanism follows one causal chain:

`auxiliary update -> measured task effect -> learned cheap predictor -> conservative
fixed-budget action`.

The reliability head, gradient audit, predictor, and allocation are not independent
features added for score gains. Each answers one part of the same question. The paper
must remove any component whose role cannot be verified by this chain.

## 4. Novelty relative to closest work

| Closest work | Existing idea | Required distinction here |
|---|---|---|
| Meta-Weight-Net | Learns sample weights from meta gradients | Predict utility of a modality-specific teacher update, under fixed budget and abstention |
| SLMV / CMoB | Sample-level modality valuation | Value the *training action*, not general modality contribution |
| MMPareto / DTO-KD | Resolve task/auxiliary gradient conflict | Select sample-modality auxiliary targets and amortize sparse labels |
| G2D | Gradient-guided multimodal distillation | Test whether each teacher update helps on separate data; allow rejection |
| QMF | Quality-aware fusion | Act during training and evaluate allocation actionability, not only inference fusion |

The method is no longer novel if the paper drops sparse supervision, modality-specific
teacher utility, conservative abstention, or matched actionability controls.

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

### Gate C: short training pilot

Use one seed and five epochs. Continue only if the method is stable, adds at most 15%
training time at the chosen sparse-label cadence, and beats both Constant and
Gate-only uniform on the validation primary metric without a contradictory allocation
diagnostic.

### Gate D: central full experiment

Across three seeds, the method must improve the primary regression results over both
Constant and Old-q, and the correct assignment must beat Permuted/Inverse. If only
Gate-only uniform improves, rewrite the method as conservative auxiliary gating. If
all matched controls are tied, stop the method track.

## 6. CCF-B evidence package

A defensible package should contain:

1. **Mechanistic finding:** degradation reliability, target fidelity, and update
   utility are empirically distinct.
2. **Method:** sparse utility labeling, calibrated amortization, and abstaining
   fixed-budget allocation.
3. **Actionability evidence:** correct assignment beats gate-only, permuted, and
   inverse controls.
4. **Breadth:** MOSEI, MOSI, and SIMS; MFON plus one structurally different fusion
   backbone.
5. **Robustness:** clean, Gaussian corruption, timestep dropout, contiguous masking,
   temporal shift, and modality missingness.
6. **Statistics:** at least three seeds for the full matrix, confidence intervals,
   paired sample bootstrap where appropriate, and all failed or mixed metrics shown.
7. **Efficiency:** parameters, GPU memory, training-time overhead, refresh cadence,
   and zero inference overhead if the predictor is training-only.
8. **Reproducibility:** frozen configurations, checkpoint hashes, raw per-seed tables,
   code, and commands.

## 7. Backup audit track

If Gate A or B fails, expand the existing audit into a method-comparison benchmark:

- three datasets;
- four to six representative models or score families;
- degradation detection, target fidelity, actual local update utility, and final
  allocation actionability;
- fixed-budget Constant, Permuted, and Inverse controls;
- public audit scripts and result schema.

The contribution becomes a general empirical result: scores that detect corruption or
estimate uncertainty often fail to predict whether an auxiliary training action helps.
A single MFON model on MOSEI is insufficient for this track; breadth is the novelty.

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
| 1 | Utility collector and predictor data schema | 4--7 days |
| 2 | Multi-checkpoint, two-seed feasibility and predictor gate | 7--10 days |
| 3 | Five-epoch pilot and matched controls | 4--7 days |
| 4 | Three-dataset, three-seed central matrix | 3--5 weeks |
| 5 | Second backbone, robustness, ablations, efficiency | 3--5 weeks |
| 6 | Paper figures, writing, internal review, release package | 2--4 weeks |

Total expected research time is roughly 10--16 weeks if the primary direction passes
early gates. A failed gate should shorten the method attempt and trigger the audit
track. Publication cannot be guaranteed; this plan raises the work to a reviewable
CCF-B scope and prevents months of unfocused training.

