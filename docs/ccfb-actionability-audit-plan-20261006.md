# CCF-B actionability-audit plan

Date: 2026-10-06
Status: active route after P6/P7 stopped the sample-only utility predictor

## 1. Paper decision

This remains the same small paper. The current reliability-aware allocation
manuscript becomes its first case study and experimental foundation; it is not
submitted separately. The revised paper studies a broader, reviewer-relevant gap:

> A multimodal score may detect degradation, uncertainty, or teacher disagreement,
> yet still fail to identify which sample-level auxiliary training action helps the
> downstream sentiment task. How should such scores be audited before they are used
> for reweighting?

Working title:

**Reliability Is Not Training Utility: An Actionability Audit of Sample-wise
Auxiliary Weighting in Multimodal Sentiment Analysis**

Application scope: online opinion-video sentiment analysis under text-dominant
fusion. The paper does not claim a universal impossibility result.

## 2. What P6/P7 establish

P6 used 2,048 records and found that auxiliary-sample forward features did not
predict a single-context visual update-utility label (`all` AUROC `0.510`, Spearman
`0.013`). P7 then removed the most plausible design objection by evaluating 768 fresh
samples against six independent meta contexts:

| Diagnostic | Result |
|---|---:|
| First-context versus mean-sign agreement | 0.641 |
| Mean pairwise context-sign agreement | 0.496 |
| Samples with at least 5/6 sign majority | 0.215 |
| Context reliability ratio | 0.171 |
| Fixed MLP AUROC | 0.494 |
| Fixed MLP Spearman | -0.005 |
| Fixed MLP selected harmful rate | 0.562 |

All predeclared repair checks failed. These results support a bounded conclusion:
at the audited final checkpoint and parameter block, local gradient utility is
primarily context-dependent and is not a predictable intrinsic sample property under
the tested features. They do not prove that every contextual or optimizer-aware
utility method is impossible.

## 3. Four-level audit protocol

Every score is evaluated at four distinct levels:

1. **Degradation sensitivity:** does the score respond to modeled and unseen input
   corruption?
2. **Target fidelity:** does the score rank teacher-target KL/consistency in the
   intended direction after controlling for length and energy?
3. **Update stability:** is the measured utility stable across batches, checkpoints,
   seeds, and reversible local-step checks?
4. **Allocation actionability:** under a fixed mean auxiliary budget, does correct
   score-to-sample assignment beat Constant, Gate-only, Permuted, and Inverse?

Passing an earlier level never counts as evidence for a later level. This separation
is the central evaluation contribution.

## 4. Score-family matrix

| Score family | Example available signal | What it claims to measure | Required audit |
|---|---|---|---|
| Learned degradation reliability | Existing P4 `q_v`, `q_a` | Input preservation under trained corruption | Levels 1--4 |
| Teacher disagreement | Per-sample KL and representation cosine | Auxiliary-target mismatch | Levels 2--4 |
| Task confidence/uncertainty | Prediction magnitude, entropy or MC variance | Main-task uncertainty | Levels 1--4 |
| Counterfactual modality sensitivity | Error change under masked/zeroed modality | Prediction dependence on a modality | Levels 1--4 plus OOD-mask caveat |
| Local gradient utility | Existing exact/directional dot products | Immediate parameter-space compatibility | Levels 3--4 |
| Simple combinations | Locked linear calibration on training/validation only | Whether signals complement each other | Complete held-group evaluation |

Do not add a score family only because it improves one inspected test result. Freeze
definitions before the central matrix.

## 5. Dataset and backbone scope

| Setting | Role |
|---|---|
| MOSEI + MFON | Main development, complete four-level audit, existing results |
| MOSI + MFON | English small-data replication |
| SIMS/CH-SIMS + MFON | Chinese transfer and unimodal-label analysis |
| One structurally different backbone | Tests whether conclusions are MFON-specific |

The second backbone is a required design choice, not yet implemented. Prefer MulT or
MISA only after checking that the same auxiliary teacher loss and score definitions
can be added without changing the central comparison. CH-SIMS v2.0 remains a locked
confirmation setting if data and raw media are available.

## 6. Minimum central comparisons

Read-only audits can cover all score families. Full training is reserved for three
representative policies selected without test-set tuning:

- uniform Constant;
- the original learned reliability allocation;
- one simple non-reliability score fixed before final testing;
- Permuted and Inverse versions of each nonconstant score;
- Gate-only uniform whenever a policy can abstain.

Every arm must preserve the same mean active auxiliary coefficient. Report realized
weighted loss and gradient norm so that apparent allocation gains cannot come from
changing total regularization.

## 7. Claim-evidence matrix

| Claim | Required evidence | Failure interpretation |
|---|---|---|
| Reliability, fidelity, and utility are distinct | Correlation, concordance, confound controls across seeds/datasets | Narrow the claim to settings that differ |
| Local utility is context-dependent | Repeated-meta stability at early/mid/late checkpoints and at least two seeds | P7 final-checkpoint result is only a case study |
| Existing score use is often non-actionable | Correct versus Constant/Permuted/Inverse under fixed budgets | Do not generalize if only one score/model fails |
| The audit protocol prevents unsupported claims | Apply it uniformly to all score families and publish raw schemas | Protocol contribution weakens without reusable code |
| Findings generalize beyond MFON | Three datasets and a second backbone | Reframe as an MFON case study and lower venue ambition |

## 8. Evidence package for a CCF-B attempt

- Three datasets and two backbones;
- at least five predeclared score families;
- three seeds for central allocation comparisons;
- seen and unseen corruption families, modality missingness, and clean performance;
- Constant, Gate-only, Permuted, and Inverse controls;
- per-seed values, confidence intervals, paired bootstrap where appropriate;
- compute, storage, and training/inference overhead;
- public audit code, frozen configurations, checkpoint hashes, and raw JSON/CSV;
- honest failure cases and scope boundaries.

This package can make an audit paper reviewable at a CCF-B multimedia venue, but no
venue outcome is guaranteed. A single negative MFON/MOSEI table is not sufficient.

## 9. Execution order

1. Freeze and archive P6/P7 outputs and hashes.
2. Consolidate existing C1/C2/C5/P6/P7 outputs into one versioned audit schema.
3. Implement the five score families on MFON/MOSEI validation data and lock them.
4. Port the read-only protocol to MOSI and SIMS.
5. Select and integrate the second backbone after a two-day compatibility audit.
6. Freeze the central training matrix; then run three seeds.
7. Run corruption, efficiency, and statistical analyses.
8. Rewrite the current manuscript around the four-level audit rather than appending
   P6/P7 as isolated negative experiments.

## 10. Stop and downgrade rules

- Do not resume sample-only utility-predictor tuning on the same checkpoint.
- If a second backbone cannot support comparable auxiliary actions, narrow the paper
  and reconsider the target venue rather than presenting incomparable results.
- If correct allocation never differs from Permuted/Inverse across score families,
  the result is valuable only with broad cross-setting evidence and a released
  benchmark/protocol.
- Do not describe validation/test analyses inspected during development as untouched
  confirmation.
