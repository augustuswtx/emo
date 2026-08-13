# DAMFON Experiment Log

## 2026-07-29 Fixed-Budget P0

The local fixed-budget implementation was uploaded to
`/home/jovyan/projects/MFON` and passed all 8 CPU contract tests:

```text
Ran 8 tests in 0.289s
OK
```

The first two-epoch server smoke attempt completed epoch 1 and produced:

```text
q_v=0.513745, q_a=0.558596
q_v_std=0.017907, q_a_std=0.023901
w_v=0.05, w_a=0.05
w_v_std=0.001743, w_a_std=0.002134
w_nce_v=0.05, w_nce_a=0.05
progress=0.1
```

Interpretation:

- The per-sample quality and weights are not constant within a batch.
- The mean auxiliary weights satisfy the expected first-epoch fixed budget.
- Forward/backward training reached validation and checkpoint saving.
- This is implementation evidence only, not evidence of accuracy or robustness gains.

The attempt then failed while writing the checkpoint:

```text
OSError: [Errno 28] No space left on device
```

Disk inspection showed a 50GB volume at 100% capacity; inode use was not the
problem. The partial 269MB smoke checkpoint is invalid. Obsolete pre-fix/debug
checkpoints and the pip cache were selected for cleanup while preserving:

- three repaired `baseline_pos_fixed` checkpoints;
- three `alw_norm_css_min05_pos_fixed` checkpoints;
- three `full_norm_dpg_css_min05_pos_fixed` checkpoints;
- unimodal encoder checkpoints;
- repaired `css_pos_fixed` evidence.

After cleanup, the same two-epoch smoke test completed successfully. Epoch 2
produced:

```text
q_v=0.531683, q_a=0.609833
q_v_std=0.018626, q_a_std=0.029877
w_v=0.1, w_a=0.1
w_v_std=0.003505, w_a_std=0.004901
w_nce_v=0.1, w_nce_a=0.1
loss_v=0.093673, loss_a=0.064907, loss_nce=0.56148
progress=0.2
```

The checkpoint was saved without a traceback at:

```text
/home/jovyan/projects/MFON/MOSI/save_models/all_model/MOSI/1111/budgeted_aux_align_smoke/TVA_fusion_model.pt
```

The two training epochs therefore pass the budget conservation, non-constant
allocation, finite-value, training stability, and checkpoint-write checks.
The saved checkpoint also loaded and completed test inference:

```text
Has0 Acc-2=0.8120, Has0 F1=0.8123
Non0 Acc-2=0.8277, Non0 F1=0.8286
Acc-5=0.4082, Acc-7=0.3688
MAE=0.9176, Corr=0.7510, Loss=1.417873002697358
```

These are two-epoch smoke metrics and must not be compared as formal results
against the 25-epoch repaired baseline.

Decision rule: do not start a 25-epoch or five-seed run solely because the smoke
test completes. First audit the candidate reliability proxy under controlled
degradation. The `norm` proxy is already rejected by the completed CSS audit.

### Align proxy pilot audit

The loaded two-epoch checkpoint was audited on the first five MOSI test batches
(`n=160`) using modality-specific Gaussian noise at severities
`0, 0.25, 0.5, 1.0`.

| Modality | Spearman(severity, q) | Clean/corrupt AUROC | Highest severity below clean |
|---|---:|---:|---:|
| Vision | -0.1353 | 0.5372 | 68.13% |
| Audio | -0.3482 | 0.6371 | 86.88% |

Clean-score confounds:

| Check | Correlation |
|---|---:|
| `q_v` vs vision length | -0.0508 |
| `q_v` vs vision energy | -0.4131 |
| `q_a` vs audio length | -0.0536 |
| `q_a` vs audio energy | -0.2624 |
| `q_v` vs absolute label | 0.0852 |
| `q_a` vs absolute label | -0.1117 |

Interpretation: `align` is weak for visual degradation and only moderately
responsive to audio degradation. It is not strongly length-confounded in this
pilot, but it is associated with feature energy. The structural explanation is
that both sides of the alignment score are computed from the same corrupted
input, so they can move together while retaining cosine similarity. Task metrics
were nearly unchanged across these pilot severities, so this run establishes
proxy direction only and does not establish robustness or actionability.

The complete MOSI test split (`n=686`) confirmed the pilot:

| Modality | Spearman(severity, q) | Clean/corrupt AUROC | Highest severity below clean |
|---|---:|---:|---:|
| Vision | -0.1894 | 0.5672 | 75.07% |
| Audio | -0.3575 | 0.6421 | 87.90% |

Full-split clean-score confounds:

| Check | Correlation |
|---|---:|
| `q_v` vs vision length | -0.0230 |
| `q_v` vs vision energy | -0.3252 |
| `q_a` vs audio length | -0.0758 |
| `q_a` vs audio energy | -0.2286 |
| `q_v` vs absolute label | -0.0891 |
| `q_a` vs absolute label | -0.1081 |

At severity 1.0, the clean-to-corrupt change in task metrics remained negligible:
vision MAE changed from 0.9176 to 0.9177 and audio MAE to 0.9194; Corr changed
from 0.7510 to 0.7509 and 0.7506, respectively. This does not establish useful
actionability.

Decision: reject `align` as the final reliability proxy and do not run its
25-epoch configuration. Retain it as a weak-proxy control. The next method
version must learn reliability from controlled degradation and explicitly test
energy confounding.

## 2026-07-29 Interventional Reliability P1 Smoke

The MOSI P1 path completed epoch 2 with the following logs:

```text
Budgeted auxiliary epoch 2:
q_v=0.477927, q_a=0.454722
q_v_std=0.214582, q_a_std=0.055506
w_v=0.1, w_a=0.1
w_v_std=0.045030, w_a_std=0.012189
w_nce_v=0.1, w_nce_a=0.1
loss_v=0.101275, loss_a=0.077796, loss_nce=0.552776
progress=0.2

Interventional reliability epoch 2:
loss_reliability=0.047002
loss_rank=0.046999
loss_invariance=0.000025
q_clean_v=0.663584, q_clean_a=0.477441
q_gap_v=0.185657, q_gap_a=0.022719
severity_v=0.326937, severity_a=0.342043
```

Interpretation:

- The four mean auxiliary weights exactly satisfy the epoch-2 budget of 0.1.
- Quality and weight standard deviations are non-zero, so allocation is
  sample-specific.
- Both clean-minus-corrupt gaps are positive, showing the learned heads assign
  lower scores to stronger corruption on the sampled training triplets.
- The visual gap is substantially larger than the acoustic gap; acoustic
  reliability remains a risk requiring held-out degradation audit.
- The near-zero scale-invariance loss is expected from the normalized head
  design and confirms that simple positive energy scaling is not being used as
  the main shortcut.

The P1 checkpoint loaded and completed clean test inference:

```text
Has0 Acc-2=0.8032, Has0 F1=0.7986
Non0 Acc-2=0.8262, Non0 F1=0.8228
Acc-5=0.2274, Acc-7=0.2201
MAE=1.1387, Corr=0.6365, Loss=1.8413170421436298
```

Compared with the earlier two-epoch fixed-budget smoke checkpoint, MAE increased
from 0.9176 to 1.1387 and Corr decreased from 0.7510 to 0.6365. Although
two-epoch checkpoints are not formal performance results, the matched smoke
setting makes this a meaningful engineering warning: applying sampled
corruption to main-task inputs at full strength from the first epoch may damage
early clean optimization.

Status: training, save, load, and inference paths pass. Clean preservation does
not pass the provisional smoke comparison. Run the held-out learned-score audit
before deciding whether the reliability head or the main-task corruption
schedule is responsible. Do not start 25-epoch training.

### P1 held-out reliability audit and P1.1 decision

The five-batch test audit used 160 held-out samples:

```text
vision: spearman(severity,q)=-0.951643, AUROC=0.980651,
        fraction(highest_below_clean)=1.000000
audio:  spearman(severity,q)=-0.049789, AUROC=0.501445,
        fraction(highest_below_clean)=0.587500
```

The visual head passes the pilot degradation gate decisively. The generic audio
head is indistinguishable from random separation and fails. Combined with the
clean-performance warning, this rules out a 25-epoch P1 run.

P1.1 keeps the visual head, replaces the audio head with temporal descriptors
(normalized level, adjacent-frame differences, difference energy, lag
correlation, and clipped kurtosis), and separates the two schedules. Reliability
ranking still observes full clean/mild/strong triplets from epoch 1. Main-task
corruption is blended from 0 to full strength over 10 epochs. The fixed budget
uses the reliability scores of the blended main-task inputs. Local validation:
23 CPU tests pass, Python compilation passes, and `git diff --check` passes.

The P1.1 two-epoch checkpoint completed clean inference on all 686 MOSI test
samples:

```text
Has0 Acc-2=0.8309, Has0 F1=0.8285
Non0 Acc-2=0.8537, Non0 F1=0.8521
Acc-5=0.4679, Acc-7=0.4038
MAE=0.7881, Corr=0.7584, Loss=1.1002073840566697
```

This removes the P1 clean-preservation warning in the matched two-epoch
engineering comparison, but it is not a formal performance result and must not
be compared as if it were a converged 25-epoch model.

The held-out five-batch P1.1 audit used 160 samples:

```text
vision: spearman=-0.952679, AUROC=0.980169,
        fraction(highest_below_clean)=1.000000
audio:  spearman=-0.554874, AUROC=0.860553,
        fraction(highest_below_clean)=0.956250
```

The visual result remains stable relative to P1. The audio-specific temporal
head improves AUROC from 0.501445 to 0.860553 and therefore passes the pilot
degradation gate. Remaining risks are non-trivial score correlations with
vision energy (-0.365255) and audio length (0.293830), plus nearly unchanged
sentiment predictions under the audited corruptions. Run the full 686-sample
audit next; actionability controls remain mandatory before formal training
claims.

The full P1.1 MOSI test audit used all 686 samples:

```text
vision: spearman=-0.944177, AUROC=0.970536,
        fraction(highest_below_clean)=0.998542
audio:  spearman=-0.555608, AUROC=0.853704,
        fraction(highest_below_clean)=0.928571
```

The pilot conclusions reproduce at full-test scale. Full-test confound
correlations are lower than the 160-sample estimates:

```text
corr(q_v, vision_length)=-0.044788
corr(q_v, vision_energy)=-0.139763
corr(q_a, audio_length)= 0.213261
corr(q_a, audio_energy)=-0.110135
```

P1.1 therefore passes the granularity, degradation-direction, discrimination,
and provisional confound gates for a seed-1111 converged pilot. The sentiment
metrics remain nearly unchanged under modality corruption (highest-severity
audio: MAE 0.7881 to 0.7884; Corr 0.7584 to 0.7582). This is not evidence of
actionability: it may reflect text dominance or weak inference-time dependence
on the corrupted modalities. Formal training must be followed by equal-budget
constant, permuted/reversed, and Oracle controls.

## 2026-07-30 P1.1 Converged Seed-1111 Pilot

The 25-epoch learned-allocation run completed with exact full auxiliary budgets:

```text
mean w_v/w_a/w_nce_v/w_nce_a=0.5
q_gap_v=0.175127, q_gap_a=0.145695
loss_rank=0.003032
task_corruption_progress=1.0
```

The best-validation checkpoint produced:

| Method, seed 1111 | Has0 Acc-2 | Has0 F1 | Non0 Acc-2 | Non0 F1 | Acc-5 | Acc-7 | MAE | Corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Repaired MFON | 0.8265 | 0.8253 | 0.8476 | 0.8471 | 0.5029 | 0.4388 | 0.7279 | 0.7959 |
| P1.1 learned allocation | 0.8353 | 0.8345 | 0.8552 | 0.8550 | 0.5117 | 0.4446 | 0.7171 | 0.7972 |

Absolute changes relative to the matched repaired baseline are +0.0088 Has0
Acc-2, +0.0092 Has0 F1, +0.0076 Non0 Acc-2, +0.0079 Non0 F1, +0.0088
Acc-5, +0.0058 Acc-7, -0.0108 MAE, and +0.0013 Corr. All reported metrics
move in the favorable direction, but this remains one seed and does not
establish a stable improvement. Audit the converged checkpoint and run
equal-budget controls before adding seeds.

The full 686-sample audit of the converged best checkpoint produced:

```text
vision: spearman=-0.965883, AUROC=0.998882,
        fraction(highest_below_clean)=0.998542
audio:  spearman=-0.815835, AUROC=0.945113,
        fraction(highest_below_clean)=0.998542
```

Converged-score confounds:

```text
corr(q_v, vision_length)= 0.087347
corr(q_v, vision_energy)= 0.007884
corr(q_a, audio_length)= 0.252213
corr(q_a, audio_energy)=-0.024787
```

Both heads improve relative to the two-epoch audit and decisively pass the
degradation gate. Energy confounding is negligible. Acoustic length correlation
remains moderate and must be reported and tested across datasets. Prediction
metrics again change very little under isolated feature corruption, so this
result validates reliability measurement but not allocation actionability. The
project now advances to equal-budget P2 controls.

## 2026-08-05 P2 Constant-Control Smoke

The two-epoch constant-allocation path completed and saved its checkpoint. At
epoch 2, `q_v=q_a=1.0`, all quality and allocation standard deviations were
exactly zero, and all four mean auxiliary weights were exactly 0.1. Reliability
supervision remained active (`q_gap_v=0.192871`, `q_gap_a=0.017517`), confirming
that the control changes only auxiliary allocation rather than disabling the
reliability heads. Checkpoint load/inference remains the final smoke gate.

The constant smoke checkpoint loaded successfully and completed inference:

```text
Has0 Acc-2=0.8178, Acc-7=0.3950
MAE=0.8414, Corr=0.7354
```

This completes the engineering smoke gate. Its two-epoch metrics are not a
formal actionability comparison; the matched 25-epoch constant run is required.

The 25-epoch constant control completed with exact uniform budgets and produced:

| Allocation, seed 1111 | Has0 Acc-2 | Has0 F1 | Non0 Acc-2 | Non0 F1 | Acc-5 | Acc-7 | MAE | Corr | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Learned | 0.8353 | 0.8345 | 0.8552 | 0.8550 | 0.5117 | 0.4446 | 0.7171 | 0.7972 | 0.9567 |
| Constant | 0.8309 | 0.8300 | 0.8491 | 0.8487 | 0.5087 | 0.4504 | 0.7221 | 0.7900 | 0.9845 |

Learned allocation improves both binary metric pairs, Acc-5, MAE, Corr, and
test loss relative to constant allocation; constant is higher on Acc-7 by
0.0058. This is favorable but incomplete single-seed actionability evidence.
Reversed and permuted controls remain required.

The two-epoch reversed-allocation smoke path also passed training, exact-budget,
save, load, and inference checks. It produced Has0 Acc-2 0.8134, Acc-7 0.4023,
MAE 0.8287, and Corr 0.7423. These smoke metrics are directionally worse than
the matched two-epoch learned run but are not used as formal evidence. The
25-epoch reversed control is the next matched comparison.

The matched 25-epoch reversed control produced:

| Allocation, seed 1111 | Has0 Acc-2 | Has0 F1 | Non0 Acc-2 | Non0 F1 | Acc-5 | Acc-7 | MAE | Corr | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Learned | 0.8353 | 0.8345 | 0.8552 | 0.8550 | 0.5117 | 0.4446 | 0.7171 | 0.7972 | 0.9567 |
| Reversed | 0.8338 | 0.8335 | 0.8521 | 0.8524 | 0.4985 | 0.4242 | 0.7393 | 0.7993 | 1.0266 |

Learned allocation is favorable on both binary metric pairs, Acc-5, Acc-7,
MAE, and test loss. Reversed is higher on Corr by 0.0021. The largest learned
advantages are +0.0204 Acc-7 and -0.0222 MAE. This supports score-order
actionability at one seed, while the permuted control remains necessary to test
sample-score correspondence.

The two-epoch permuted-allocation smoke path passed exact-budget training,
checkpoint save/load, and inference. It produced Has0 Acc-2 0.8178, Acc-7
0.3950, MAE 0.8435, and Corr 0.7356. These smoke values are not formal evidence;
the matched 25-epoch permuted run is required.

The matched 25-epoch permuted control produced:

| Allocation, seed 1111 | Has0 Acc-2 | Has0 F1 | Non0 Acc-2 | Non0 F1 | Acc-5 | Acc-7 | MAE | Corr | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Learned | 0.8353 | 0.8345 | 0.8552 | 0.8550 | 0.5117 | 0.4446 | 0.7171 | 0.7972 | 0.9567 |
| Permuted | 0.8294 | 0.8284 | 0.8476 | 0.8471 | 0.5219 | 0.4577 | 0.7190 | 0.7940 | 0.9812 |

Learned is favorable on both binary metric pairs, MAE, Corr, and test loss;
permuted is higher on Acc-5 and Acc-7. The single-seed actionability evidence is
therefore metric-dependent rather than uniform. The Oracle control and
multi-seed uncertainty are required before making a stable allocation claim.

## 2026-08-06 P2 Oracle-Control Smoke

The two-epoch Oracle-allocation training path completed and saved its
checkpoint. At epoch 2, mean Oracle qualities were 0.934053 (vision) and
0.935742 (audio), with non-zero standard deviations 0.074186 and 0.073381.
All four mean auxiliary weights were exactly 0.1; visual/audio weight standard
deviations were 0.007948/0.007847. Reliability supervision remained identical
to matched controls. The checkpoint loaded successfully and completed full-test
inference: Has0 Acc-2=0.8192, Has0 F1=0.8193, Non0 Acc-2=0.8415, Non0
F1=0.8422, Acc-5=0.4723, Acc-7=0.3892, MAE=0.8588, Corr=0.7383, and
Loss=1.3551. This completes the Oracle engineering smoke gate. These two-epoch
metrics are not formal evidence; the matched 25-epoch Oracle run is required.

The matched 25-epoch Oracle control completed and produced Has0 Acc-2=0.8338,
Has0 F1=0.8326, Non0 Acc-2=0.8537, Non0 F1=0.8531, Acc-5=0.5146,
Acc-7=0.4548, MAE=0.7164, Corr=0.7923, and Loss=0.9769. Relative to
Learned, Oracle is favorable on Acc-5, Acc-7, and MAE, whereas Learned is
favorable on both binary metric pairs, Corr, and test loss. The five-control
seed-1111 evidence is therefore metric-dependent and requires multi-seed
replication before a stable actionability claim.

## 2026-08-06 P2 Learned Multi-seed Replication

The matched seed-1112 Learned run completed, but did not reproduce the
seed-1111 improvement. It produced Has0 Acc-2=0.8163, Has0 F1=0.8168,
Non0 Acc-2=0.8247, Non0 F1=0.8257, Acc-5=0.4636, Acc-7=0.4111,
MAE=0.7802, Corr=0.7855, and Loss=1.0828. Relative to the repaired seed-1112
baseline, the deltas are -0.0102 Has0 Acc-2, -0.0090 Has0 F1, -0.0213 Non0
Acc-2, -0.0202 Non0 F1, -0.0349 Acc-5, -0.0379 Acc-7, +0.0533 MAE, and
-0.0086 Corr. This is a material cross-seed failure rather than evidence of
stable improvement. A full reliability audit is required before spending GPU
time on the seed-1112 Permuted control.

The full 686-sample seed-1112 reliability audit remained strong: vision
Spearman=-0.963491 and AUROC=0.994929; audio Spearman=-0.789313 and
AUROC=0.916853. Energy confounds were near zero (vision 0.009095, audio
-0.015360), while the recurring audio-length correlation was 0.238209. Thus,
the replication failure is not explained by a collapsed reliability head. The
current diagnosis is unstable downstream optimization from reliability-based
allocation. A matched seed-1112 Constant run is the next isolating control:
if it recovers baseline-level performance, learned redistribution is the
failure source; if it also degrades, the broader interventional auxiliary
training setup is implicated.

The matched seed-1112 Constant control produced Has0 Acc-2=0.8105, Has0
F1=0.8110, Non0 Acc-2=0.8186, Non0 F1=0.8196, Acc-5=0.4723,
Acc-7=0.4155, MAE=0.7695, Corr=0.7870, and Loss=1.0607. Constant improves
MAE, Corr, loss, Acc-5, and Acc-7 relative to seed-1112 Learned, but is worse
on both binary metric pairs; both variants remain materially below the repaired
baseline. Learned redistribution is therefore not the sole failure source.
Because both variants progressively blend synthetic corruption into the task
features, task-path corruption is now the primary isolating hypothesis.

A new `--reliability-task-corrupt-scale` control preserves ordered corruption
supervision for both reliability heads while independently scaling corruption
entering the sentiment task. Scale zero keeps task inputs clean. The default is
one, preserving every completed experiment. Twenty-seven local unit tests pass.
The next diagnostic is seed-1112 Constant with scale zero; Permuted and seed
1113 expansion remain paused until this gate is resolved.

The full seed-1112 P3 Constant clean-task diagnostic produced Has0 Acc-2=
0.8134, Has0 F1=0.8139, Non0 Acc-2=0.8216, Non0 F1=0.8227, Acc-5=0.4752,
Acc-7=0.4169, MAE=0.7708, Corr=0.7869, and Loss=1.0607. It is nearly
identical to P2 Constant: task-path corruption is not the primary failure
source. The negligible prediction response in the corruption audits is also
consistent with strong text dominance.

Code inspection identified a remaining mismatch: the legacy budget warmup
scales the mean auxiliary budget from 0.05 at epoch 1 to 0.5 at epoch 10,
whereas the repaired baseline uses 0.5 from the first epoch. A new
`--budget-warmup-mode allocation` keeps each mean budget fixed at its baseline
value while interpolating only the sample allocation from uniform to
reliability-proportional. The historical `scale` behavior remains the default.
Twenty-nine local tests pass. This P4 equal-budget schedule is the final planned
repair of the training-loss allocation branch; failure on seed 1112 triggers a
pivot away from that branch rather than further hyperparameter searching.

The full seed-1112 P4 Constant true-budget run recovered baseline-level
performance: Has0 Acc-2=0.8265, Has0 F1=0.8260, Non0 Acc-2=0.8476, Non0
F1=0.8476, Acc-5=0.4985, Acc-7=0.4359, MAE=0.7295, Corr=0.7974, and
Loss=0.9880. Relative to the repaired baseline, Has0 Acc-2 and Acc-5 match,
both Non0 metrics and Corr are slightly higher, MAE is 0.0026 worse, and Acc-7
is 0.0131 lower. The earlier seed-1112 collapse is therefore primarily
attributable to scaling down the mean auxiliary budget during warmup. The next
and decisive experiment is P4 Learned under the same true fixed budget and
clean task path.

The full seed-1112 P4 Learned true-budget run then produced Has0 Acc-2=0.8309,
Has0 F1=0.8301, Non0 Acc-2=0.8506, Non0 F1=0.8505, Acc-5=0.5073,
Acc-7=0.4402, MAE=0.7187, Corr=0.7992, and Loss=0.9724. Its epoch-25
allocation means remained exactly 0.5 for vision KL, audio KL, and both
InfoNCE components; non-zero allocation standard deviations (vision 0.030679,
audio 0.063716) confirm sample-specific redistribution. The task corruption
progress remained zero.

Under the matched fixed-budget control, Learned improves all nine reported
metrics over P4 Constant: +0.0044 Has0 Acc-2, +0.0041 Has0 F1, +0.0030 Non0
Acc-2, +0.0029 Non0 F1, +0.0088 Acc-5, +0.0043 Acc-7, -0.0108 MAE,
+0.0018 Corr, and -0.0156 Loss. Relative to the repaired baseline, Learned is
better on both binary metric pairs, Acc-5, MAE, and Corr, while Acc-7 is 0.0088
lower. This is the first fair seed-1112 evidence supporting reliability-aware
allocation; it does not yet establish a multi-seed or cross-dataset claim.

The frozen P4 Learned configuration was then replicated on seed 1113. It
produced Has0 Acc-2=0.8324, Has0 F1=0.8312, Non0 Acc-2=0.8521, Non0
F1=0.8516, Acc-5=0.4942, Acc-7=0.4402, MAE=0.7161, Corr=0.7934, and
Loss=0.9579. Relative to the repaired seed-1113 baseline, the deltas are
+0.0044 Has0 Acc-2, +0.0043 Has0 F1, +0.0030 Non0 Acc-2, +0.0029 Non0
F1, -0.0233 Acc-5, -0.0234 Acc-7, -0.0065 MAE, and +0.0004 Corr. Thus,
the binary/regression trend replicated, but fine-grained classification
degraded. A matched seed-1113 P4 Constant run is required before attributing
these changes to learned reliability allocation.

The matched seed-1113 P4 Constant control produced Has0 Acc-2=0.8309,
Has0 F1=0.8297, Non0 Acc-2=0.8506, Non0 F1=0.8500, Acc-5=0.5000,
Acc-7=0.4461, MAE=0.7183, Corr=0.7924, and Loss=0.9626. Learned improves
both binary metric pairs, MAE, Corr, and Loss over Constant, while Acc-5 and
Acc-7 are lower by 0.0058 and 0.0059. Across seeds 1112 and 1113, the
unweighted two-seed mean favors Learned on every reported metric except
Acc-7. The remaining gate is a final frozen P4 Learned/Constant comparison on
seed 1111 before computing the formal three-seed summary.

The final frozen P4 Learned run on seed 1111 produced Has0 Acc-2=0.8265,
Has0 F1=0.8258, Non0 Acc-2=0.8460, Non0 F1=0.8459, Acc-5=0.4956,
Acc-7=0.4286, MAE=0.7290, Corr=0.7929, and Loss=0.9821. Relative to the
repaired seed-1111 baseline, Has0 Acc-2 is unchanged and Has0 F1 is 0.0005
higher, while Non0 Acc-2/F1, Acc-5/7, MAE, and Corr are slightly worse. The
baseline improvement therefore did not replicate uniformly across all three
seeds. The matched seed-1111 P4 Constant control remains necessary to evaluate
the allocation claim independently of baseline variance.

The matched seed-1111 P4 Constant control produced Has0 Acc-2=0.8280,
Has0 F1=0.8273, Non0 Acc-2=0.8476, Non0 F1=0.8475, Acc-5=0.4985,
Acc-7=0.4315, MAE=0.7310, Corr=0.7913, and Loss=0.9921. This completes
the three-seed matched P4 comparison.

Using sample standard deviation across seeds 1111, 1112, and 1113:

| Method | Has0 Acc-2 | Has0 F1 | Non0 Acc-2 | Non0 F1 | Acc-5 | Acc-7 | MAE | Corr | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Repaired MFON | 0.8270±0.0009 | 0.8260±0.0008 | 0.8476±0.0016 | 0.8472±0.0014 | 0.5063±0.0099 | 0.4505±0.0125 | 0.7258±0.0028 | 0.7943±0.0015 | TBD |
| P4 Constant | 0.8285±0.0022 | 0.8277±0.0019 | 0.8486±0.0017 | 0.8484±0.0014 | 0.4990±0.0009 | 0.4378±0.0075 | 0.7263±0.0069 | 0.7937±0.0033 | 0.9809±0.0160 |
| P4 Learned | 0.8299±0.0031 | 0.8290±0.0029 | 0.8496±0.0032 | 0.8493±0.0030 | 0.4990±0.0072 | 0.4363±0.0067 | 0.7213±0.0068 | 0.7952±0.0035 | 0.9708±0.0122 |

Learned minus Constant mean deltas are +0.0015 Has0 Acc-2, +0.0014 Has0
F1, +0.0010 Non0 Acc-2/F1, approximately zero Acc-5, -0.0015 Acc-7,
-0.0050 MAE, +0.0015 Corr, and -0.0101 Loss. Learned minus repaired MFON
mean deltas are +0.0029 Has0 Acc-2, +0.0030 Has0 F1, +0.0020 Non0 Acc-2,
+0.0021 Non0 F1, -0.0073 Acc-5, -0.0141 Acc-7, -0.0045 MAE, and +0.0008
Corr. These results support a bounded claim about binary/regression performance
and learned allocation versus uniform allocation; they do not support universal
improvement over MFON. Three seeds alone do not establish statistical
significance.

The final seed-1113 P4 Learned checkpoint passed the full 686-sample quality
audit. Vision achieved Spearman(severity,q)=-0.958149, clean/corrupt
AUROC=0.989878, and 0.998542 of highest-severity samples below their clean
scores. Audio achieved Spearman=-0.828746, AUROC=0.949643, and 0.997085 below
clean. Clean-score confounds were vision length -0.087036, vision energy
0.153522, audio length 0.243383, audio energy 0.002750, vision absolute label
-0.060530, and audio absolute label -0.002204. The recurring audio-length
association remains a cross-dataset risk.

Task predictions changed negligibly under severe visual or acoustic Gaussian
corruption. This is evidence of strong text dominance in the MOSI model, not
evidence that reliability scores are ineffective. The current method uses
reliability for training-time auxiliary allocation, so the defensible claim is
about reliability estimation and supervision allocation; it must not be
presented as inference-time noise-adaptive fusion without an additional gating
mechanism and corresponding evidence.

The final P4 Learned audits were also completed on seeds 1111 and 1112.
Seed 1111 produced vision Spearman=-0.965883/AUROC=0.998882 and audio
Spearman=-0.815835/AUROC=0.945113. Seed 1112 produced vision
Spearman=-0.964805/AUROC=0.996831 and audio
Spearman=-0.814968/AUROC=0.931147. Across the three final checkpoints, the
sample mean±standard deviation is -0.962946±0.004189 vision Spearman,
0.995197±0.004719 vision AUROC, -0.819850±0.007717 audio Spearman, and
0.941968±0.009641 audio AUROC. Audio-length correlation is
0.241182±0.012280, confirming a repeatable confounding risk that must be
rechecked on MOSEI and SIMS.

## 2026-08-10 MOSEI P5 Port

The frozen P4 implementation was ported from MOSI to MOSEI without changing
MOSEI's original feature dimensions, learning rates, or base auxiliary weights.
The port adds the interventional reliability heads, clean task path, fixed-budget
allocation warmup, allocation controls, optimizer parameters, and reliability
logging. Four source-level cross-dataset consistency tests and Python syntax
compilation passed locally. After installation on the GPU server, all 33 tests
passed in 4.966 seconds and all changed files compiled successfully.

Server preflight found the 13GB MOSEI dataset but no unimodal encoders for seeds
1111--1113. Seventeen obsolete MOSI diagnostic checkpoints were removed while
retaining repaired baseline, P4 Constant, and P4 Learned checkpoints for all
three seeds. Available disk space increased from 1.2GB to 12GB. The next action
is seed-1111 MOSEI audio and vision encoder training, followed by a two-epoch P4
Learned fusion smoke test; no 25-epoch MOSEI fusion run is authorized yet.

## Current Evidence Summary

The paper is being reframed from "MFON + three add-on modules" to a quality-guided multimodal fusion and optimization framework:

- Working name: `Q-DAMFON`
- Core idea: use sample-level modality quality to control loss weighting, dynamic prompt interaction, and curriculum sampling.
- Full configuration completed on MOSI seeds 1111--1113 and is retained as a seed-sensitivity ablation:

```bash
--use-alw --q-type norm --warmup-epoch 10 \
--use-dpg \
--use-css --css-epoch 20 --css-min-ratio 0.5
```

Important caveat: neither current configuration has established broad, stable superiority over the repaired baseline. The Full configuration is seed-sensitive. The lighter `ALW norm + CSS min05` configuration has lower three-seed variability than Full and slightly higher mean Has0 metrics and Corr than baseline, but it remains lower on Acc-7 and higher on MAE.

| Seed | Method | Has0 Acc-2 | Has0 F1 | Non0 Acc-2 | Non0 F1 | Acc-5 | Acc-7 | MAE | Corr |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1111 | baseline_pos_fixed | 0.8265 | 0.8253 | 0.8476 | 0.8471 | 0.5029 | 0.4388 | 0.7279 | 0.7959 |
| 1111 | alw_norm_css_min05_pos_fixed | 0.8353 | 0.8357 | 0.8491 | 0.8499 | 0.5015 | 0.4359 | 0.7324 | 0.7939 |
| 1111 | full_norm_dpg_css_min05_pos_fixed | 0.8324 | 0.8317 | 0.8537 | 0.8536 | 0.5117 | 0.4534 | 0.7177 | 0.7992 |
| 1112 | baseline_pos_fixed | 0.8265 | 0.8258 | 0.8460 | 0.8459 | 0.4985 | 0.4490 | 0.7269 | 0.7941 |
| 1112 | alw_norm_css_min05_pos_fixed | 0.8236 | 0.8226 | 0.8430 | 0.8426 | 0.5087 | 0.4446 | 0.7279 | 0.7925 |
| 1112 | full_norm_dpg_css_min05_pos_fixed | 0.8178 | 0.8179 | 0.8308 | 0.8314 | 0.4927 | 0.4227 | 0.7640 | 0.7860 |
| 1113 | baseline_pos_fixed | 0.8280 | 0.8269 | 0.8491 | 0.8487 | 0.5175 | 0.4636 | 0.7226 | 0.7930 |
| 1113 | alw_norm_css_min05_pos_fixed | 0.8309 | 0.8302 | 0.8460 | 0.8458 | 0.5102 | 0.4388 | 0.7248 | 0.8000 |
| 1113 | full_norm_dpg_css_min05_pos_fixed | 0.8353 | 0.8351 | 0.8476 | 0.8478 | 0.5117 | 0.4417 | 0.7372 | 0.7967 |

Three-seed summary:

| Method | Has0 Acc-2 | Has0 F1 | Non0 Acc-2 | Non0 F1 | Acc-5 | Acc-7 | MAE | Corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_pos_fixed | 0.8270±0.0009 | 0.8260±0.0008 | 0.8476±0.0016 | 0.8472±0.0014 | 0.5063±0.0099 | 0.4505±0.0125 | 0.7258±0.0028 | 0.7943±0.0015 |
| alw_norm_css_min05_pos_fixed | 0.8299±0.0059 | 0.8295±0.0066 | 0.8460±0.0031 | 0.8461±0.0037 | 0.5068±0.0047 | 0.4398±0.0044 | 0.7284±0.0038 | 0.7955±0.0040 |
| full_norm_dpg_css_min05_pos_fixed | 0.8285±0.0094 | 0.8282±0.0091 | 0.8440±0.0119 | 0.8443±0.0115 | 0.5054±0.0110 | 0.4393±0.0155 | 0.7396±0.0232 | 0.7940±0.0070 |

Interpretation:

- Seed 1111: `ALW norm + CSS min05` improves Has0 and Non0 binary metrics; Full improves all main metrics.
- Seed 1112: Full underperforms baseline; the lighter configuration improves Acc-5 but is lower on the binary metrics.
- Seed 1113: the lighter configuration improves Has0 Acc-2/F1 and Corr over baseline and yields lower MAE than Full, but is lower on Non0 metrics and Acc-7 than baseline.
- Across three seeds, the lighter configuration is descriptively less variable than Full, especially on Acc-7, MAE, and Corr. It is therefore the current candidate for further validation, not yet a proven replacement for the baseline.
- Current writing boundary: do not claim stable superiority over MFON. The safe claim is that the quality-aware loss-and-curriculum configuration has a more stable preliminary profile than the Full configuration, while its gains remain metric-dependent.

Completed server gate:

```bash
cd /home/jovyan/projects/MFON

ps -ef | grep -E "run_experiment|python" | grep -v grep

find MOSI/save_models/all_model/MOSI/1113/alw_norm_css_min05_pos_fixed -type f -ls 2>/dev/null
```

Seed 1113 training completed and the checkpoint was saved:

```bash
MOSI/save_models/all_model/MOSI/1113/alw_norm_css_min05_pos_fixed/TVA_fusion_model.pt
```

Seed 1113 test result:

```bash
Has0 Acc-2=0.8309, Has0 F1=0.8302, Non0 Acc-2=0.8460, Non0 F1=0.8458,
Acc-5=0.5102, Acc-7=0.4388, MAE=0.7248, Corr=0.8000, Loss=0.9666029527653063
```

Detailed rewrite plan: see `paper-v2-quality-aware-draft.md`.

## Environment Checklist

- Python/PyTorch environment: ready in `.venv` (`torch 2.12.0`, MPS available)
- Dataset files under `MFON/data/{MOSI,MOSEI,SIMS}/`: missing; official Google Drive currently rate-limited
- Server project root: `/home/jovyan/projects/MFON`
- Server has MOSI seed 1111-1113 repaired baseline/full checkpoints and tests from the handoff.
- Seed 1113 unimodal encoders exist on the server.
- MOSI seed 1113 `alw_norm_css_min05_pos_fixed` has completed training and testing.

## Commands

Run from `MFON/` with `../.venv/bin/python` unless another PyTorch environment is active.

## Server Patch And CSS Rerun

Use this block in the school server Web Terminal. It is idempotent: already patched files are left unchanged, and backups are created once.

```bash
cd /home/jovyan/projects/MFON

for f in \
  MOSI/models/model.py MOSEI/models/model.py SIMS/models/model.py \
  MOSI/models/trans/position_embedding.py MOSEI/models/trans/position_embedding.py SIMS/models/trans/position_embedding.py \
  MOSI/train/TVA_train.py MOSEI/train/TVA_train.py SIMS/train/TVA_train.py
do
  [ -f "$f" ] && [ ! -f "$f.bak_damfon_fix" ] && cp "$f" "$f.bak_damfon_fix"
done

python - <<'PY'
from pathlib import Path

position_files = [
    Path("MOSI/models/trans/position_embedding.py"),
    Path("MOSEI/models/trans/position_embedding.py"),
    Path("SIMS/models/trans/position_embedding.py"),
]
old_pos = "        positions = make_positions(input, self.padding_idx, self.left_pad)\n"
new_pos = """        positions = torch.arange(
            self.padding_idx + 1,
            self.padding_idx + 1 + seq_len,
            device=input.device,
        ).long().unsqueeze(0).expand(bsz, seq_len)
"""

for p in position_files:
    s = p.read_text()
    if old_pos in s:
        p.write_text(s.replace(old_pos, new_pos))
        print(f"patched position embedding: {p}")
    elif "positions = torch.arange(" in s:
        print(f"already patched position embedding: {p}")
    else:
        raise SystemExit(f"position pattern not found: {p}")

model_files = [
    Path("MOSI/models/model.py"),
    Path("MOSEI/models/model.py"),
    Path("SIMS/models/model.py"),
]
for p in model_files:
    s = p.read_text()
    s2 = s.replace(
        "x_v_embed_froze = self.vision_encoder_froze(vision).squeeze()",
        "x_v_embed_froze = self.vision_encoder_froze(vision)",
    ).replace(
        "x_a_embed_froze = self.audio_encoder_froze(audio).squeeze()",
        "x_a_embed_froze = self.audio_encoder_froze(audio)",
    )
    if s2 != s:
        p.write_text(s2)
        print(f"patched frozen encoder squeeze: {p}")
    else:
        print(f"squeeze patch already ok: {p}")

train_files = [
    Path("MOSI/train/TVA_train.py"),
    Path("MOSEI/train/TVA_train.py"),
    Path("SIMS/train/TVA_train.py"),
]
for p in train_files:
    s = p.read_text()
    old = "                if not css_mask.any():\n                    continue\n"
    new = "                if css_mask.sum().item() < 2:\n                    continue\n"
    if old in s:
        p.write_text(s.replace(old, new))
        print(f"patched CSS batch guard: {p}")
    elif "css_mask.sum().item() < 2" in s:
        print(f"already patched CSS batch guard: {p}")
    else:
        raise SystemExit(f"CSS guard pattern not found: {p}")
PY

grep -R -n "positions = torch.arange\|make_positions(input" MOSI/models/trans/position_embedding.py MOSEI/models/trans/position_embedding.py SIMS/models/trans/position_embedding.py
grep -R -n "x_[va]_embed_froze = self\\..*squeeze()" MOSI/models/model.py MOSEI/models/model.py SIMS/models/model.py || true
grep -R -n "css_mask.sum().item() < 2\|not css_mask.any" MOSI/train/TVA_train.py MOSEI/train/TVA_train.py SIMS/train/TVA_train.py
```

After the verification lines show `torch.arange` and `css_mask.sum().item() < 2`, rerun CSS on MOSI seed 1111:

```bash
cd /home/jovyan/projects/MFON

CUDA_LAUNCH_BLOCKING=1 python run_experiment.py --dataset MOSI --stage train-fusion --seed 1111 \
  --use-css --css-epoch 20 --exp-name css_pos_fixed \
  2>&1 | tee mosi_css_pos_fixed_1111.log

python run_experiment.py --dataset MOSI --stage test-fusion --seed 1111 \
  --use-css --css-epoch 20 --exp-name css_pos_fixed \
  2>&1 | tee mosi_css_pos_fixed_test_1111.log
```

If the school web terminal disconnects easily, run the training in the background:

```bash
cd /home/jovyan/projects/MFON
nohup bash -lc 'CUDA_LAUNCH_BLOCKING=1 python run_experiment.py --dataset MOSI --stage train-fusion --seed 1111 --use-css --css-epoch 20 --exp-name css_pos_fixed' \
  > mosi_css_pos_fixed_1111.log 2>&1 &
tail -f mosi_css_pos_fixed_1111.log
```

Evidence status: the old CSS-only result is invalid for the final paper because it was produced before the position-embedding and single-sample CSS fixes. Only results from `css_pos_fixed` or later reruns should enter tables.

Checkpoint status on 2026-07-08: server output confirms
`MOSI/save_models/all_model/MOSI/1111/css_pos_fixed/TVA_fusion_model.pt` exists. Next action is to test this checkpoint before any rerun. Do not rerun CSS unless the train log shows a post-save error or the test command fails to load the checkpoint.

```bash
# Required first if pretrained unimodal encoders are not provided
../.venv/bin/python run_experiment.py --dataset MOSI --stage train-unimodal --seed 1111

# Baseline
../.venv/bin/python run_experiment.py --dataset MOSI --stage train-fusion --seed 1111
../.venv/bin/python run_experiment.py --dataset MOSI --stage test-fusion --seed 1111

# ALW only
../.venv/bin/python run_experiment.py --dataset MOSI --stage train-fusion --seed 1111 --use-alw --q-type align --warmup-epoch 10
../.venv/bin/python run_experiment.py --dataset MOSI --stage test-fusion --seed 1111

# DPG only
../.venv/bin/python run_experiment.py --dataset MOSI --stage train-fusion --seed 1111 --use-dpg
../.venv/bin/python run_experiment.py --dataset MOSI --stage test-fusion --seed 1111

# ALW + DPG
../.venv/bin/python run_experiment.py --dataset MOSI --stage train-fusion --seed 1111 --use-alw --use-dpg --q-type align --warmup-epoch 10
../.venv/bin/python run_experiment.py --dataset MOSI --stage test-fusion --seed 1111

# CSS only
../.venv/bin/python run_experiment.py --dataset MOSI --stage train-fusion --seed 1111 --use-css --css-epoch 20
../.venv/bin/python run_experiment.py --dataset MOSI --stage test-fusion --seed 1111

# Full DAMFON
../.venv/bin/python run_experiment.py --dataset MOSI --stage train-fusion --seed 1111 --use-alw --use-dpg --use-css --q-type align --warmup-epoch 10 --css-epoch 20
../.venv/bin/python run_experiment.py --dataset MOSI --stage test-fusion --seed 1111
```

## MOSI Results

Formal repaired MOSI results are maintained in the Current Evidence Summary at the top of this file. Do not use old CSS debug runs or pre-`pos_fixed` results in the paper.

The old CSS-only checkpoint `css_pos_fixed` on seed 1111 was tested and is weaker than the repaired baseline:

| Setting | Seed | Has0 Acc-2 | Has0 F1 | Non0 Acc-2 | Non0 F1 | Acc-7 | MAE | Corr | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| + CSS (`css_pos_fixed`) | 1111 | 0.7945 | 0.7928 | 0.8155 | 0.8148 | 0.3294 | 0.9157 | 0.7093 | fixed checkpoint tested; weaker than baseline |

## ALW Dynamics

Copy the printed `ALW epoch` logs here after each run.

```text
not collected in this local handoff
```

## Current Experiment Queue

1. Freeze P4 settings; do not tune further against the MOSI test split.
2. P4 Learned and Constant are complete on seeds 1112/1113.
3. The MOSI three-seed matched comparison is complete; freeze it and stop tuning on the MOSI test split.
4. Use validation results for any later candidate selection; reserve test results for the frozen method.
5. Extend the frozen method and matched baselines to MOSEI and SIMS, then add controlled-degradation robustness, ablation, efficiency, and significance evidence.

## 2026-08-12 MOSEI Seed-1111 Unimodal Encoder Milestone

The user confirmed that MOSEI seed-1111 acoustic encoder training completed and
that both expected checkpoints exist:

```text
MOSEI/save_models/uni_fea_encoder/MOSEI/1111/best_loss_audio_encoder.pt  28M
MOSEI/save_models/uni_fea_encoder/MOSEI/1111/best_loss_audio_decoder.pt  1.3M
```

The server reported both file modification times as Aug 11 00:17. No acoustic
training process should be started again.

After this checkpoint gate passed, the user started the frozen seed-1111 visual
encoder command with `nohup`. The shell reported background PID 839 and the log
target is `mosei_vision_encoder_1111.log`. This records only successful process
launch, not training completion. The next action is read-only monitoring of the
visual process and log. Do not start another visual encoder. Fusion smoke
testing remains blocked until both visual checkpoint files exist.

## 2026-08-13 MOSEI Seed-1111 Visual Encoder Completion

The user confirmed that both expected MOSEI seed-1111 visual checkpoints
exist:

```text
MOSEI/save_models/uni_fea_encoder/MOSEI/1111/best_loss_vision_encoder.pt  55M
MOSEI/save_models/uni_fea_encoder/MOSEI/1111/best_loss_vision_decoder.pt  1.3M
```

The server reported both file modification times as Aug 12 05:43. The audio
and visual unimodal checkpoint gates are therefore complete. No unimodal
training process should be started again. The next gate is to rerun the 33
server tests, then execute only the two-epoch P4 Learned fusion smoke. A formal
25-epoch fusion run remains blocked until the smoke checkpoint is saved and
successfully reloaded by `test-fusion` with the same P4 configuration.

## 2026-08-13 MOSEI P4 Learned Two-Epoch Smoke Training Gate

The user supplied the epoch-2 training log for
`p5_mosei_p4_learned_smoke`. The reported statistics were:

```text
q_v=0.719222, q_a=0.943149
q_v_std=0.063758, q_a_std=0.008467
w_v=0.300000, w_a=0.300000
w_v_std=0.005322, w_a_std=0.000540
w_nce_v=0.001000, w_nce_a=0.001000
loss_v=0.126483, loss_a=0.022530, loss_nce=0.006446
allocation_progress=0.2
loss_reliability=0.004354
loss_rank=0.004343, loss_invariance=0.000114
q_gap_v=0.197342, q_gap_a=0.243674
q_task_gap_v=0.0, q_task_gap_a=0.0
severity_v=0.336377, severity_a=0.331287
task_corruption_progress=0.0
```

The checkpoint was reported saved at:

```text
/home/jovyan/projects/MFON/MOSEI/save_models/all_model/MOSEI/1111/p5_mosei_p4_learned_smoke/TVA_fusion_model.pt
```

The training-side smoke gate passes: all four mean budgets exactly match the
MOSEI baseline, both learned allocation standard deviations are non-zero, both
clean-minus-corrupt reliability gaps are positive, the task path remains
clean, and a checkpoint was saved. The audio allocation dispersion is much
smaller than the visual dispersion but remains non-zero; this is a monitoring
item rather than a smoke failure. No effectiveness claim is made from a
two-epoch smoke. The only authorized next step is checkpoint reload and
`test-fusion` with the same P4 flags and experiment name. A 25-epoch run remains
blocked until that test completes successfully.
