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
The local environment lacks PyTorch, so only syntax has been checked; server
inference and gradient smoke remain pending.

**Stage B, minimum method prototype.** Integrate a detached utility gate for
KL only; keep the original architecture, teacher encoders, 25-epoch schedule,
and main task objective. If per-sample full-parameter gradients are too costly,
use a low-cost representation-gradient proxy as a clearly labeled candidate,
or a small predictor trained on sparse parameter-gradient labels from inner
training data. Freeze the scoring recipe before formal runs. Never feed the
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
