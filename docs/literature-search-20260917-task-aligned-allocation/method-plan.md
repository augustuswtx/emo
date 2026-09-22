# MFON redesign: task-aligned auxiliary allocation (proposal, no new results)

Date: 2026-09-17. Status: design only. The old P4 three-seed experiments and
negative C1/C5 audits remain unchanged. A positive result for this proposal
cannot be assumed or inferred from the cited papers.

## 1. Exact failure to repair

The existing reliability head is taught only that synthetic clean input should
rank above corrupted input. `q_i^m` then redistributes clean-sample KL and
InfoNCE across the current batch. This training objective does **not** teach
that high `q_i^m` identifies a teacher target or auxiliary gradient that helps
the sentiment task. In the three-seed C1 audit, visual `q` vs `-KL` is negative
and acoustic association weak. In seed-1111 C5, Learned is not better than
Constant in either high zero-ablation-sensitivity stratum. Those proxies have
different meanings; neither is an observed auxiliary-update utility.

## 2. Proposed new claim and score

New claim to test: **under a matched auxiliary budget, sample-modality pairs
whose KL update predicts a reduction in a separate main-task loss should
receive more KL weight.** Here “positive” describes a measured update utility,
not a monotonic corruption-quality score.

For a current training model `theta`, sample-modality KL loss `a_i^m(theta)`,
and a disjoint meta-training batch `M`, define a local parameter-space score

`u_i^m = < grad_theta L_task(M; theta), P_t grad_theta a_i^m(theta) >`,

where `P_t` approximates the current optimizer's update preconditioner. A
first-order Taylor expansion gives the contribution of an auxiliary update
`-eta * P_t grad a_i^m` to meta-task loss as approximately
`-eta * u_i^m`. Positive `u` is locally favorable. This is an inference from
the gradient formula and assumes a sufficiently small step; curvature,
optimizer state, label noise, and distribution shift can overturn it. A
representation-gradient cosine is a cheaper *proxy*, not the same guarantee
for actual parameters. The final paper must name which version is used.

Possible weight rule on a batch of size `B`, per modality:

`s_i^m = max(0, u_i^m / (scale_m + eps))`;
`w_i^m = B * delta_m * (eps + s_i^m) / sum_j(eps + s_j^m)`.

This preserves the existing mean KL coefficient `delta_m=0.3` when at least
one pair has positive utility. If *all* utilities are nonpositive, forcing an
exact budget may be harmful. Pre-register a batch-level abstention rule with
zero KL for that modality in such batches, and compare against a **gate-only
uniform** control with the identical active-batch mask and realized total
budget. This separates the effect of choosing samples from the effect of
turning KL off. Keep InfoNCE uniform at `0.001` in the first prototype because
its negatives are batch-dependent; add InfoNCE allocation only after a
separate utility definition and control.

## 3. Feasible implementation route

**Stage A, read-only feasibility audit before any new training.** Use existing
frozen checkpoints on inner-training batches only. For a small set of batches
and checkpoints, compute visual/audio KL per-sample parameter gradients on a
specified block (`proj_v` / `proj_a`, then the cross-modal Transformer if
feasible), and main-task gradients on a disjoint meta-training batch. Record
utility sign, score variance, relation to old `q`, length/energy confounds,
and compute time/peak GPU memory. On a small subset, compare predicted sign
with the *actual* meta-loss change after a reversible tiny update on a cloned
model/optimizer state. Do not write to a trained checkpoint. Stop if useful
scores collapse, signs are unstable, or runtime is prohibitive.

The first screening script is now prepared at
`MFON/audit_task_aligned_gradient.py`. It compares KL gradients on
`proj_v.weight`/`proj_a.weight` with task-MSE gradients from the preceding,
disjoint **training** batch, recording each dot product, cosine, old `q`, KL,
time, and peak GPU memory. It does not yet run a cloned optimizer step, and
its two adjacent batches are not a locked speaker-disjoint meta split; these
are explicit requirements before claiming validated utility. Start with
`--batch-size 4 --max-pairs 1 --max-samples 4` and a unique JSON filename.
The local environment lacks PyTorch. Server seed-1111 smoke and 32-pair
screening completed without a logged runtime error. The 32-pair screening
showed weak or negative old-score association with gradient alignment; see
`docs/research-execution-roadmap-20260917.md` for exact exploratory numbers.
A separate script,
`MFON/audit_task_aligned_step.py`, now performs reversible single-projection
SGD-style perturbations and compares first-order predicted change with actual
meta-batch task-loss change. A one-pair server smoke completed: visual changes
matched first-order directions 4/4, while the audio first-order signal was
far smaller than finite-step changes and central differences agreed only 2/4.
In an eight-pair expansion, visual first-order direction matched finite-step
task-loss direction for 29 of 31 valid probes; old degradation score `q`
did not reliably rank measured utility within each batch (four positive and
four negative batch-wise correlations). Audio remained too small and unstable
under this probe. These are same-checkpoint, sequential-training-batch
screening results, not independent confirmation or an Adam replay. The first
allocation prototype should vary **visual KL only** and keep audio KL and both
InfoNCE terms uniform, pending a low-cost score that reproduces the visual
parameter-gradient signal. The script does not replay Adam because the
original checkpoint does not include optimizer state.

A read-only feasibility tool, `MFON/audit_visual_representation_proxy.py`,
checked a cheaper alternative: the meta-task gradient at the visual fused
embedding dotted with the analytic per-sample visual KL gradient at that
embedding. It matched checkpoint hashes and sample IDs to the eight-pair
finite-step JSON before comparing signs. The server audit completed on 2026-09-21:
the proxy agreed with measured visual step directions in 19/31 valid cases.
The same records contain 20/31 helpful steps, so an always-helpful rule is
already more accurate overall. However, 10 of the 12 predicted-helpful steps
were actually helpful (precision 83.3%, recall 50%), with large pair-to-pair
variation: `[4/4,2/4,0/3,2/4,2/4,4/4,1/4,4/4]`. The audit segment took 7.3
seconds and peaked at about 1002 MiB of PyTorch-allocated GPU memory. Do
**not** use this proxy for allocation or launch a training run around it yet.
There is an exploratory conservative-selection signal, but no independent
evidence of its stability. Do not tune a threshold on these 31 records. Test
the existing zero threshold on fresh seeded-random training pairs outside the
first 256 previously screened train indices. Compare predicted-helpful
precision with the new sample's helpful-step prevalence and report coverage,
recall, per-pair variation, and compute cost. This is a failure of overall
accuracy for the cheap embedding-level candidate on one screened sample, not
evidence that parameter-space task utility is impossible.
The next read-only check is fixed before seeing new results: 24 meta/aux batch
pairs of size four, sample-order seed 271828, excluding train indices 0–255.
The existing reversible step and representation scripts now accept these
arguments and record them in their JSON. The sampled pairs still come from
the training split of a previously trained checkpoint, so this remains a
method-development check rather than held-out confirmation.
The reversible step half of this check has completed: among valid visual
probes, predicted and actual loss-change signs agreed in 91/95 cases, and
predicted and central-difference signs in 93/95. Audio remained unreliable
(43/96 predicted/actual sign matches). The representation proxy on these same
24 pairs agreed in only 49/95 cases, versus 50/95 helpful steps for an
always-helpful majority baseline. Its confusion matrix was TP 27, FP 23,
TN 22, FN 23: predicted-helpful precision 27/50 = 54%, barely above the
50/95 = 52.6% helpful prevalence. The first eight-pair pilot's 10/12
precision did not replicate. This supports measuring a local visual
parameter-space signal, not its cheap embedding-level approximation or
final-model benefit. Retire this embedding-level proxy; do not start full
training with it.

A second low-cost candidate is being implemented as a read-only audit:
normalize the visual meta-task gradient on `proj_v.weight`, perturb that
weight by `+h` and `-h`, and obtain every auxiliary sample's KL at both
points in two batch forwards. The central difference estimates the exact
parameter-space dot product without one backward pass per auxiliary sample.
Use the already fixed 24-pair order and a prespecified relative displacement
`h / ||proj_v.weight|| = 0.0003`. Compare its signs to the stored exact dots
and actual reversible steps, check restoration and compute cost. This is a
standard directional derivative approximation, not novelty by itself. Even
if it works in eval mode, stochastic training mode and Adam state remain
unsolved; do not claim final allocation effectiveness from this audit.
The server's same-24-pair audit has now completed: central directional
differences agreed with the exact visual projection dot in 93/96 records and
with measured one-block task-loss change in 91/95 valid records. This clears
the preregistered numerical sign checks for this development audit. Runtime
was 168.4 seconds versus about 160 seconds for the earlier step audit, so the
prespecified cost check failed; run-to-run load may affect the exact ratio.
Peak PyTorch-allocated memory was 1037.2 MiB. Two nonzero exact-dot sign
mismatches include one at +1.20e-4, so not every discrepancy can be dismissed
as a tie. The current code recomputes BERT text, audio, and classifier paths
for each `+h/-h` forward even though only `proj_v.weight` changes. A cached
visual-only audit can test whether eliminating those redundant paths preserves
the same KL and signs at lower cost, but it must be reported as a separate
implementation, not a retroactive pass of the first cost check. Full-training
efficiency and test benefit remain unestablished.
The cached variant now skips repeated BERT, audio, and classifier calls in the
two perturbed auxiliary forwards; it first checks the cached visual embedding
against the complete model's output on the same batch. It keeps the identical
24 batch pairs and fixed displacement, writing a separate JSON. This is an
engineering optimization of the audit, not a new independent experiment or
evidence of generalization.
After a server-only unit-test compatibility fix (`Linear(...).double()`), the
cached run completed on the same 24 pairs. Its sign results were unchanged:
93/96 versus exact dots and 91/95 versus measured steps. The first-pair
embedding parity maximum absolute difference was 0.0. Audit time fell from
168.4 to 133.5 seconds (1.26x), below the earlier 160-second step audit;
peak PyTorch-allocated GPU memory was 1040.4 MiB. The numeric and audit-cost
screens therefore pass on this single development run. A real training batch
still needs a separate cost and stochastic-mode check before implementation
or full training, because meta backward and batch-size scaling are unknown.
A read-only online-cost screen is prepared for three disjoint training pairs
at the actual auxiliary batch size 32 and a meta batch size of four. It times
the old P4 train-mode loss/backward against the *additional* eval-mode cached
directional score, synchronizing CUDA and excluding data loading. It does no
optimizer step and writes no checkpoint. The first pair is warmup; if the
median scoring cost on the remaining two pairs already exceeds one normal
forward/backward, reduce scoring cadence or abandon an every-batch gate before
any 25-epoch run. The proposed sign-only visual KL weights have exact mean
budget 0.3 on active batches and abstain when all signs are nonpositive; this
requires a matched gate-only uniform control in later training. This cost
screen cannot validate Adam, dropout, or final performance.
The server run completed. After discarding its warmup pair, normal P4
forward/backward took 0.863 and 0.759 seconds, while the additional cached
score took 1.234 and 1.139 seconds (1.43x and 1.50x the normal computation).
An every-batch implementation would therefore cost roughly 2.46 times the
measured forward/backward compute in these two pairs, before data loading and
optimizer steps. This fails the prespecified online-cost gate. Do not launch
25 epochs with per-batch scoring. Inspect weight/budget diagnostics and
profile score components; pursue a lower-cadence scoring policy with matched
uniform controls or a predictor trained on sparse task-utility labels only if
the engineering evidence warrants it.
The three pairs assigned positive visual utility to 15/32, 12/32, and 15/32
samples. All gates were active; both the original and proposed weight vectors
had mean 0.3. This checks mean weight only, not equal realized weighted KL
loss or gradient magnitude. Peak allocated GPU memory was 5473.7 MiB.
The instrumented three-pair rerun passed three server unit tests and completed.
After warmup, additional scoring took 1.639 and 1.841 seconds against 1.045
and 1.148 seconds for normal forward/backward (1.57x and 1.60x). The meta
gradient used 0.939 and 0.937 seconds, the cached auxiliary features 0.047
seconds each, and the two-sided visual difference 0.652 and 0.857 seconds.
Caching is not the bottleneck. Reusing one meta gradient for four or eight
auxiliary batches could amortize the meta part, but only if its sample-positive
decisions remain sufficiently stable. A fixed-checkpoint, 12-pair read-only
screen of this assumption was run on newly sampled training examples. On
non-refresh batches, four-batch reuse matched fresh positive decisions for
164/288 samples (56.9%, false-positive rate 44.3%); eight-batch reuse matched
174/320 (54.4%, false-positive rate 48.6%). Both failed the locked engineering
gate of at least 90% agreement and at most 10% false positives. Stop the
current online directional-difference/reuse training route. This result tests
one frozen checkpoint and a fresh-score reference, not optimizer-step utility
or all possible task-conditioned scoring methods. Prioritize the existing P4
per-sample-only component attribution before proposing another method. Any
future cadence reduction must be paired with the same cadence in uniform and
permuted controls and must check realized KL contribution, not only mean weight.

**Stage B, minimum method prototype.** Integrate a detached utility gate for
KL only; keep the original architecture, teacher encoders, 25-epoch schedule,
and main task objective. If per-sample full-parameter gradients are too costly,
seek another low-cost score or a small predictor trained on sparse
parameter-gradient labels from inner training data. The tested fused-embedding
dot product remains a screened candidate only for conservative selection,
pending fresh-sample confirmation. Freeze the scoring recipe before
formal runs. Never feed the
checkpoint-selection validation labels into gate learning. An off-the-shelf
MoE or new backbone is deferred because it would confound the allocation test.

**Stage C, independent evaluation.** Compare the new gate with matched
Constant, old degradation-q Learned, gate-only uniform, new-score Permuted,
new-score Inverse, and a simple teacher/task-error heuristic. Run the same
seeds, epochs, encoders, optimizer, and checkpoint-selection rule for every
arm. Only the allocation score should vary in the central matrix. Keep
reliability-head loss either identically on or identically off within that
matrix; first finish the existing per-sample-only C4 attribution.

## 4. Code hooks and compatibility

- `MFON/MOSEI/models/model.py:forward` currently constructs the complete
  budget *before* the training loop sees labels. A task-gradient gate requires
  refactoring that boundary or computing the allocation in `TVA_train.py`
  after `pred` and `label` are available. The same change will be needed for
  MOSI if replicated.
- `MFON/budgeted_auxiliary.py:fixed_budget_weights` already normalizes a
  detached score to the exact mean budget. Add a separate utility-score path
  and explicit all-nonpositive handling; retain old score behavior for
  controls. Do not alter existing checkpoint names or old runs.
- `MFON/MOSEI/train/TVA_train.py` currently uses Adam and `update_epochs`
  gradient accumulation. A plain gradient dot product may not match the
  eventual Adam update, so the audit should also measure realized one-step
  meta loss and report this approximation error.
- The remote Python is 3.8 and may have an older PyTorch. Do not depend on
  `torch.func`/`vmap` until server versions are checked. Start with a few
  explicit `autograd.grad` calls in the read-only audit and profile cost.
- The current model loader filters checkpoint keys and uses `strict=False`;
  audit scripts must report missing/unexpected keys, so an apparent utility
  failure is not caused by a partly loaded model.

## 5. Claim-to-evidence gates

| Claim | Minimum new evidence | Failure decision |
|---|---|---|
| Score measures useful auxiliary updates | On disjoint inner/meta data, score predicts signed one-step meta-loss change; positive rank association and >0.5 pairwise concordance across seeds/modalities, with length/energy diagnostics | Do not call it task utility; keep as exploratory proxy |
| Allocation, not just KL abstention, helps | New gate beats **gate-only uniform**, new-score Permuted, and new-score Inverse on MAE and Corr under matched realized KL budget | Do not claim positive sample assignment |
| Effect generalizes | Prespecified seeds and an independently locked confirmation setting; report every seed, paired MAE/Corr differences, classification metrics, and compute overhead | Narrow to a dataset-specific exploratory effect |
| Robust multimodal use | Actual prediction sensitivity and robustness under original-media visual/acoustic corruption, with text-only and strong multimodal baselines | Do not claim robust fusion |

MOSEI test aggregates and C5 strata have already informed this redesign. For
the *new method*, they are not a blind confirmation set. Form the inner/meta
split only from training data, keep the existing validation split for
checkpoint selection, and lock an independent confirmation dataset or
untouched group-held-out split *before* inspecting its results. If using a new
held-out portion of MOSEI train, all compared variants must be retrained on
the same reduced training partition. A second dataset requires matching
preprocessing and cannot borrow the old MOSEI checkpoint comparison.

## 6. Ordered execution and stop conditions

1. Finish read-only C5 seeds 1112–1113; they cannot be reinterpreted as
   evidence for the new gate.
2. Build the read-only gradient and tiny-update audit, first on seed 1111
   with at most a few batches, then 1112–1113 if the smoke is numerically and
   computationally sound. Check disk and GPU occupancy before any run.
3. Complete the existing C4 per-sample-only attribution before formal
   comparisons. If the task-utility target is predictive and affordable,
   implement the KL-only gate and gate-only uniform control. Run one seed as
   engineering smoke; inspect loss weights, finite gradients, realized budget,
   and runtime.
4. Lock score form, hyperparameters, seeds, and primary endpoints before
   additional formal training. Expand matched controls only if the pilot
   warrants the compute. Preserve all negative results.
5. Rewrite title, Abstract, and method around **task-aligned auxiliary
   allocation** only after actual results support it. The old degradation
   score remains the baseline that motivated the change.

Prior art requiring explicit comparison: Ren et al. 2018 (meta weighting),
Wu et al. 2022 (multimodal gradient curriculum), Wei et al. 2024 (sample-level
modality valuation), Jin et al. 2021 (modality-specific distillation), and
Jiang et al. 2026 (instance-wise gradient calibration). See `papers.md` for
primary links. No new performance result or publication outcome is claimed.
