# Full reviewer report: Audited fixed-budget multimodal sentiment analysis

## 1. Report Metadata

- **Review date:** 2026-08-16
- **Target venue/year/track:** Neural Computing and Applications (NCA), original research article; working CCF-C target
- **Paper title:** *Are Modality Quality Scores Trustworthy? Audited, Fixed-Budget Auxiliary Learning for Multimodal Sentiment Analysis*
- **Input materials reviewed:** English Markdown draft v2; NCA `main.tex` and `body.tex`; experiment log and experiment plan; reference-verification report; current method figure and figure QA artifacts; submission-status ledger; relevant implementation and tests
- **Search basis:** public keyword search only; official PMLR, CVF, AAAI, Springer Nature, and CCF pages; no private manuscript text was submitted to search
- **Report file:** `ccfa-review-reports/2026-08-16-audited-fixed-budget-msa-nca-conference-review.md`
- **Reviewer mode:** full scientific, writing, figure, and format review

## 2. Desk Rejection Assessment

- **Paper length — uncertain.** The English source is about 5,100 words including the author-facing ledger, while the NCA body is about 4,000 words. This is plausible for a journal article, but the official template has not been compiled, so final length and float behavior are unknown.
- **Topic compatibility — pass.** Quality-aware multimodal learning and multimodal sentiment analysis fit NCA's neural-computing and applications scope. CCF currently lists NCA in the AI C category.
- **Minimum quality — pass.** The paper contains a clear problem, method, equations, matched controls, results, limitations, related work, and a reproducibility discussion.
- **Policy/anonymity/compliance — uncertain.** The source is anonymized, but declarations remain placeholders, the official template package is absent, and anonymized artifact links are not yet prepared. NCA uses double-blind review and requires editable source files and a separate title page.
- **Prompt injection and hidden manipulation — pass.** No reviewer-directed or LLM-directed manipulation was found in the inspected manuscript or figure text.
- **Ethics and reviewability — pass with completion needed.** The responsible-use paragraph is unusually careful, but final funding, competing-interest, data, code, and author-contribution statements remain incomplete.

**Desk rejection risk:** medium at present.  
**Reason:** the manuscript is reviewable, but the experimental protocol contains a test-set-driven development issue, and the submission package is incomplete.  
**Can be fixed before review?** Partly. The formatting and declarations are straightforward; the protocol issue requires untouched confirmatory evidence.

## 3. Paper Summary And Contribution Map

The paper argues that a modality score should not be trusted merely because it is called quality or reliability. It proposes a five-part audit of score granularity, degradation monotonicity, confounds, actionability, and coefficient collapse. Within MFON, it learns visual and acoustic reliability from ordered clean--mild--strong feature corruptions, preserves auxiliary losses at sample resolution, and reallocates KL and InfoNCE supervision while exactly conserving each batch's mean auxiliary budget. The strongest empirical comparison is P4 Learned versus P4 Constant over three MOSI seeds. Learned is slightly better on binary and regression-oriented metrics, tied on Acc-5, and worse on Acc-7. The paper explicitly rejects claims of uniform improvement and inference-time robust fusion.

- **Claimed problem:** quality-aware multimodal mechanisms often use unaudited scores whose meaning and downstream effect are unclear.
- **Claimed gap:** existing work emphasizes dynamic fusion or low-quality robustness but rarely verifies the score, loss granularity, and supervision budget together.
- **Contribution type:** methodological audit and controlled training-objective design, not a new base fusion architecture.
- **Core method:** ordered interventional reliability heads plus exact finite-batch auxiliary-loss redistribution.
- **Current evidence:** three-seed MOSI matched comparisons; synthetic-degradation reliability audits; earlier diagnostic controls; unit tests; one formal MOSEI baseline with P4 Constant currently running.
- **Stated limitations:** synthetic Gaussian corruption, audio-length association, text-dominant inference, small and metric-dependent gains, incomplete MOSEI, incomplete final-schedule controls, and no statistical-significance claim.

## 4. Search And Related-Work Basis

- **Queries used:** reliability/quality-aware multimodal sentiment; fixed-budget auxiliary loss; low-quality multimodal fusion; sample-level modality valuation and gradient balancing.
- **Sources searched:** PMLR, CVF Open Access, AAAI proceedings, Springer Nature, CCF, and arXiv discovery records when no final page was immediately available.
- **Closest works already handled:** QMF, Predictive Dynamic Fusion, sample-level modality valuation, SAM-LML, MFON, QA-MoE, EBMC, CPSC, and the leakage-safe quality-dependence diagnostic.
- **Unverified related-work risks:** rapidly appearing 2026 MSA and low-quality multimodal methods may change the strongest-baseline set before submission.
- **Source-quality screening:** primary proceedings/publisher pages were preferred; MDPI results were excluded.

Potential additions or sharper comparisons:

1. **Ma et al., “Calibrating Multimodal Learning,” ICML 2023 — searched.** It formalizes a multimodal confidence-calibration principle. The paper should distinguish score auditing from output calibration, not merely dynamic fusion.
2. **Wei et al., “Improving Multimodal Learning via Imbalanced Learning,” ICCV 2025 — searched.** It coordinates modality optimization through auxiliary regularizers and gradient-related signals. This is relevant to why reliability-proportional auxiliary allocation should outperform difficulty- or imbalance-aware allocation.
3. **MDF, AAAI 2026 — searched.** It is not a direct audit method, but it is a recent MSA architecture that may be needed for current benchmark context if the paper makes competitive-performance claims.
4. **Recent CVPR 2026 MSA models — searched at discovery level.** Their relevance is primarily as strong current task baselines; they are not substitutes for the paper's matched Constant/Learned control.

Primary pages used in this pass: [NCA submission guidelines](https://link.springer.com/journal/521/submission-guidelines), [CCF AI directory](https://www.ccf.org.cn/Academic_Evaluation/AI/), [QMF](https://proceedings.mlr.press/v202/zhang23ar.html), [Calibrating Multimodal Learning](https://proceedings.mlr.press/v202/ma23i.html), [Sample-level Modality Valuation](https://openaccess.thecvf.com/content/CVPR2024/html/Wei_Enhancing_Multimodal_Cooperation_via_Sample-level_Modality_Valuation_CVPR_2024_paper.html), and [ARL](https://openaccess.thecvf.com/content/ICCV2025/html/Wei_Improving_Multimodal_Learning_via_Imbalanced_Learning_ICCV_2025_paper.html).

## 5. Expected Review Outcome

- **Expected outcome if submitted today:** weak reject / major revision before submission.
- **Main accept signal:** a clear, honest, falsifiable audit-and-control story with exact budget conservation, matched Constant/Learned comparison, and unusually explicit negative results and limitations.
- **Main reject signal:** MOSI test samples were used during method development to replace the generic audio reliability head and guide later phase decisions, while the manuscript states that test results were not used for tuning. This makes MOSI exploratory rather than clean confirmatory evidence.
- **Confidence:** 4/5. The full draft, logs, code, tests, figure, and public related-work sources were available; a compiled NCA PDF, raw run artifacts, and completed MOSEI comparison were not.

## 6. Strengths And Weaknesses

### Strengths

1. **The paper identifies real implementation failure modes.** Batch aggregation, premature loss reduction, and unconstrained coefficient shrinkage are concrete and testable rather than rhetorical motivations.
2. **The budget control is mathematically clean.** The mean auxiliary coefficient is conserved for every finite batch and every epoch, not only in expectation.
3. **The strongest comparison is appropriately matched.** Constant and Learned retain reliability-head training and the same total auxiliary budget; the intended difference is sample assignment.
4. **The paper is honest about mixed outcomes.** Acc-7 degradation, the audio-length association, text dominance, synthetic corruption, and the absence of significance are visible in the abstract, results, and discussion.
5. **The method is audit-friendly.** The code detaches reliability when forming allocation weights, unit tests check mathematical contracts, and the logs expose score/weight means and dispersion.
6. **The new method figure materially improves reviewability.** It clearly separates the clean MFON task path from the training-only reliability and auxiliary-supervision path.

### Major weaknesses

#### W1. Test-set-driven development undermines the confirmatory status of MOSI

- **Evidence basis:** `docs/experiment-log.md` records that a five-batch MOSI test audit (`n=160`) produced visual AUROC 0.981 and audio AUROC 0.501; this result was used to retain the visual head and redesign the audio head with temporal descriptors. Later full-test audits also informed progression between phases. The manuscript nevertheless states in Section 6.2 that test results were not used for further tuning.
- **Reviewer deduction:** the MOSI test split influenced architecture and development decisions. The final MOSI metrics and reliability audit cannot be treated as an untouched confirmatory evaluation.
- **Required fix:** disclose the development history; designate MOSI as exploratory; freeze all decisions; use MOSEI or a newly isolated holdout as confirmatory evidence. Ideally repeat architecture selection and controls using training/validation data only, then evaluate once on an untouched test set.

#### W2. The central performance effect is small, mixed, and currently single-dataset

- **Evidence basis:** Learned-minus-Constant mean differences are approximately +0.001 to +0.0015 on binary metrics, -0.0050 MAE, +0.0015 Corr, -0.0101 Loss, tied Acc-5, and -0.0015 Acc-7. Only three seeds are reported and no statistical claim is made. MOSEI Constant/Learned is incomplete.
- **Reviewer deduction:** the paper shows a plausible allocation effect but not stable cross-dataset usefulness.
- **Required fix:** complete the frozen MOSEI comparison, preferably over multiple seeds; report per-seed paired deltas and uncertainty; keep the conclusion methodological if effects remain mixed.

#### W3. Reliability-proportional allocation lacks a decisive design justification

- **Evidence basis:** Section 5 allocates more auxiliary weight to samples with higher learned reliability. The manuscript does not establish why reliable/easier samples should receive more distillation and contrastive supervision than unreliable/harder samples.
- **Reviewer deduction:** exact budget conservation isolates allocation, but it does not establish that the chosen allocation direction is principled.
- **Required fix:** add gradient/allocation analysis and at least one matched difficulty-aware or inverse-reliability control under the final schedule. The existing earlier reversed/oracle pilots are not sufficient because their schedule differs.

#### W4. The reliability claim is narrow because the corruption family is synthetic and partly test-tuned

- **Evidence basis:** high AUROC is measured mainly under Gaussian corruption of pre-extracted features; real background audio, occlusion, dropped frames, misalignment, ASR error, and missing modalities are absent.
- **Reviewer deduction:** the heads detect the training intervention, but the evidence does not establish perceptual or deployment reliability.
- **Required fix:** add at least one realistic acoustic and visual corruption, preserve padding, and repeat confound checks on untouched MOSEI data.

#### W5. The final actionability suite is incomplete

- **Evidence basis:** final three-seed P4 includes only Constant and Learned; Permuted, Reversed, and Oracle belong to an earlier schedule.
- **Reviewer deduction:** the score--sample correspondence and direction are not fully isolated under the final protocol.
- **Required fix:** repeat at least Permuted and Reversed under the frozen final schedule, or narrow “actionability” to “Learned differs descriptively from uniform allocation.”

### Writing and presentation weaknesses

1. **Stale scope statements conflict.** The header and Section 6.1 say MOSEI encoder/formal fusion work is pending, while Section 7.6 correctly reports completed encoders and a completed formal baseline.
2. **`TBD` remains in the main MOSI result table.** Baseline loss should be reported, justified as unavailable, or shown as an em dash with a caption note.
3. **The main figure is informative but dense.** It is acceptable as a two-column `figure*`; font size should be verified in the compiled PDF, especially the right-side “No q → fusion edge” note and equations.
4. **The author-facing drafting ledger must not enter the submitted manuscript.** It is useful internally but should remain outside `body.tex`.
5. **The title is effective but the contribution can still look like an engineering checklist.** The introduction should explicitly state why the combination changes what can be inferred from a quality-aware experiment.

## 7. Potentially Missing Related Work

| Work | Status | Why relevant | Overlap | Needed comparison |
|---|---|---|---|---|
| *Calibrating Multimodal Learning* (ICML 2023) | searched | Audits multimodal confidence behavior | Reliability/calibration validity | Explain internal score auditing versus output-confidence calibration |
| *Improving Multimodal Learning via Imbalanced Learning* (ICCV 2025) | searched | Uses modality-related auxiliary regularization and optimization balancing | Training-time modality optimization | Explain why sample reliability allocation addresses a different imbalance and compare gradient effects |
| MDF (AAAI 2026) | searched | Recent MSA representation/fusion method | Task performance, not score auditing | Use only as current MSA context or baseline if making competitive claims |
| Recent 2026 MSA fusion models | unverified as a complete set | May define the current task-performance frontier | Architecture and task accuracy | Run a final closest-work and strongest-baseline pass after MOSEI stabilizes |

## 8. Claim-Evidence Audit

| Claim | Where stated | Evidence provided | Strength | Reviewer deduction | Required fix |
|---|---|---|---|---|---|
| Quality scores require multi-axis auditing | Abstract; Sections 1 and 4 | Five explicit audit axes and concrete prototype failures | adequate | Useful framing, but not yet validated beyond MFON/MOSI | Present as a scoped framework, not universal standard |
| The old norm proxy is invalid | Sections 1, 7.1 | Length correlation, severity correlation, clean-versus-strong comparison | strong for this proxy | Convincing negative result | Preserve exact scope and split provenance |
| P4 keeps the auxiliary budget exactly fixed | Sections 5.3–5.4 | Equation, implementation, unit tests, logged mean weights | strong | Best-supported technical claim | Add pseudocode/config table for reproduction |
| Learned allocation is preferable to Constant | Abstract; Sections 7.3, 8.1, 10 | Three-seed MOSI means; mixed metric deltas | weak-to-adequate | Descriptive evidence exists, but test-driven development and small effects limit inference | Confirm on untouched MOSEI and report paired seeds |
| Reliability heads detect synthetic degradation | Abstract; Section 7.2 | High Spearman/AUROC on MOSI synthetic corruptions | adequate for the intervention, weak for generalization | Audio head was selected using MOSI test audit | Re-evaluate frozen heads on untouched MOSEI and realistic corruptions |
| The method provides inference-time robust fusion | Explicitly rejected in Sections 7.5 and 8.2 | Predictions barely change under isolated corruption | strong negative evidence | Honest and scientifically useful boundary | Keep this limitation prominent |
| The method generalizes across datasets | Not claimed as complete | MOSEI baseline only; Constant running; Learned pending | absent | Cross-dataset inference is unavailable | Complete frozen matched comparison before adding claim |

## 9. Experiment / Benchmark / Reproducibility Audit

- **Baselines:** repaired MFON and equal-budget Constant are appropriate internal controls. Current strong external MSA baselines are discussed but not experimentally compared; this is acceptable only if the paper avoids competitiveness claims.
- **Ablations:** loss granularity, fixed budget, score controls, and early diagnostics exist, but the final schedule lacks Permuted/Reversed and an alternative allocation direction.
- **Datasets:** MOSI is complete but development-contaminated; MOSEI is the critical untouched confirmatory dataset; CH-SIMS is optional if the NCA claim remains narrow.
- **Metrics:** binary, fine-grained, regression, correlation, loss, AUROC, Spearman, and confounds give broad coverage. Metric multiplicity is high, so the paper should predeclare primary metrics rather than emphasize whichever move favorably.
- **Statistical rigor:** three seeds and mean±sample SD are descriptive. Paired per-seed deltas, bootstrap intervals, or more seeds would improve interpretation; no significance claim should be made without adequate analysis.
- **Robustness/failure cases:** synthetic degradation detection is strong, but realistic corruption and missing-modality tests are absent. Text dominance weakens any downstream robustness narrative.
- **Implementation details:** equations and contracts are clear; the differing visual/audio reliability heads need a compact architecture/hyperparameter table.
- **Artifacts:** code and tests exist locally, but an anonymized reproducible package, data provenance, environment lock, and exact run commands are not yet part of the submission.
- **Limitations:** unusually honest and specific. The missing item is explicit disclosure that MOSI test data influenced development.

## 10. Multi-Reviewer Panel

### Best-Justified Reviewer

- **Expertise:** multimodal learning methodology
- **Likely score:** 6/10, borderline positive
- **Confidence:** 4/5
- **Main positive signal:** exact budget control and audit discipline make a useful methodological point even without SOTA gains.
- **Main negative signal:** current empirical breadth is too small for broad relevance.
- **Evidence basis:** Sections 4–5, Table 2, explicit limitations.
- **Score-change condition:** untouched MOSEI replication with the same bounded trend would support 7/10.

### Critical Reviewer

- **Expertise:** experimental protocol and evaluation leakage
- **Likely score:** 3/10, reject
- **Confidence:** 5/5
- **Main positive signal:** development history is well logged.
- **Main negative signal:** MOSI test audits were used to redesign the audio head and guide phases, contradicting the manuscript's no-test-tuning statement.
- **Evidence basis:** experiment log P1/P1.1 decision trail versus Section 6.2.
- **Fatal concern:** confirmatory status of MOSI is invalid.
- **Score-change condition:** transparent relabeling plus untouched frozen MOSEI confirmation.

### Method / Soundness Reviewer

- **Expertise:** optimization and multimodal auxiliary learning
- **Likely score:** 5/10
- **Confidence:** 4/5
- **Main positive signal:** budget conservation and detached allocation weights are technically coherent.
- **Main negative signal:** the direction “more reliable → more auxiliary weight” is under-justified.
- **Evidence basis:** Section 5.4 and `budgeted_auxiliary.py`.
- **Score-change condition:** final-schedule inverse/difficulty control plus gradient analysis.

### Evidence / Experiment Reviewer

- **Expertise:** empirical ML evaluation
- **Likely score:** 4/10
- **Confidence:** 4/5
- **Main positive signal:** multiple metrics, three seeds, mixed results reported honestly.
- **Main negative signal:** small effects, no clean confirmatory dataset, incomplete final controls, and synthetic-only corruption.
- **Evidence basis:** Tables 1–2 and Sections 7.4–7.6.
- **Score-change condition:** frozen multi-seed MOSEI and realistic corruptions.

### Novelty / Positioning Reviewer

- **Expertise:** quality-aware and low-quality multimodal learning
- **Likely score:** 5/10
- **Confidence:** 4/5
- **Main positive signal:** the combined audit + exact finite-batch control is narrower and more defensible than claiming a new reliability head.
- **Main negative signal:** corruption ranking, quality weighting, sample valuation, and multimodal calibration have strong precedents.
- **Evidence basis:** Related Work and public QMF/calibration/SMV/ARL sources.
- **Score-change condition:** sharpen the falsifiable inference enabled by the package and show cross-dataset transfer.

### Writing / Clarity Reviewer

- **Expertise:** scientific communication
- **Likely score:** 7/10
- **Confidence:** 4/5
- **Main positive signal:** abstract, contribution list, limitations, and method figure make the bounded story recoverable.
- **Main negative signal:** stale MOSEI status and the `TBD` cell create avoidable inconsistency.
- **Evidence basis:** header, Sections 6.1/7.6, Table 2, Fig. 1.
- **Score-change condition:** consistency pass and compiled figure readability check.

### Ethics / Reproducibility Reviewer

- **Expertise:** responsible research and artifacts
- **Likely score:** 6/10
- **Confidence:** 4/5
- **Main positive signal:** responsible-use limitations and non-redistribution boundaries are explicit.
- **Main negative signal:** artifact and declaration statements remain promises; test-use disclosure is incomplete.
- **Evidence basis:** Section 9 and `main.tex` declarations.
- **Score-change condition:** anonymized artifact, provenance, completed declarations, and transparent split history.

### Domain Application Reviewer

- **Expertise:** multimodal sentiment analysis
- **Likely score:** 5/10
- **Confidence:** 4/5
- **Main positive signal:** the paper correctly recognizes text dominance and does not market itself as an image model.
- **Main negative signal:** Gaussian feature noise is a weak proxy for real acoustic/visual degradation.
- **Evidence basis:** Sections 7.5 and 8.3.
- **Score-change condition:** realistic background audio, occlusion/dropped-frame, or missing-modality tests.

### Evidence / Ablation Reviewer

- **Expertise:** controlled ablation design
- **Likely score:** 4/10
- **Confidence:** 4/5
- **Main positive signal:** Constant is a strong budget-matched control.
- **Main negative signal:** final-schedule correspondence and direction controls are missing.
- **Evidence basis:** Sections 5.5 and 7.4.
- **Score-change condition:** final Permuted/Reversed or a narrower actionability claim.

### Reproducibility Reviewer

- **Expertise:** ML artifact reproduction
- **Likely score:** 6/10
- **Confidence:** 4/5
- **Main positive signal:** staged commands, logs, tests, seeds, and checkpoint naming are unusually traceable.
- **Main negative signal:** the reproducible public package and compact hyperparameter table are absent.
- **Evidence basis:** repository instructions, experiment log, Section 9.
- **Score-change condition:** anonymized artifact with one-command test and documented data preparation.

### Novice Advocate Reviewer

- **Expertise:** reader accessibility
- **Likely score:** 7/10
- **Confidence:** 4/5
- **Main positive signal:** reliability versus task utility and training versus inference are clearly distinguished.
- **Main negative signal:** the dense method figure and many internal phase names can overwhelm a new reader.
- **Evidence basis:** Sections 3, 5, terminology ledger, Fig. 1.
- **Score-change condition:** remove “P4” from reader-facing prose where possible and add a compact algorithm/config table.

### Panel Synthesis

- **Agreement:** the methodological discipline, exact budget, and bounded writing are real strengths; evidence breadth and protocol cleanliness are the dominant weaknesses.
- **Disagreement:** sympathetic reviewers may value the paper as an audit/negative-result contribution, while strict empirical reviewers will treat the MOSI test-guided development as decisive.
- **Decisive positive axis:** whether the audit-and-control package yields a reproducible, transferable methodological lesson.
- **Decisive negative axis:** whether untouched confirmatory evidence exists after all design choices are frozen.
- **Unresolved evidence:** complete MOSEI Constant/Learned results, MOSEI reliability audit, realistic corruptions, and final-schedule allocation controls.
- **AC stance:** weak reject today; reconsider after transparent protocol correction and untouched cross-dataset confirmation.

## 11. Concerns Table

| ID | Severity | Concern | Evidence basis | Affected criterion | Fix class | Required action | Owner skill | Score-change condition |
|---|---|---|---|---|---|---|---|---|
| C1 | fatal | MOSI test data influenced audio-head and phase design | Experiment log P1/P1.1 versus Section 6.2 | soundness, evidence, integrity | method/soundness | Disclose; relabel MOSI exploratory; confirm frozen method on untouched MOSEI/new holdout | ccf-integrity-auditor + ccf-experiment-designer | Untouched confirmation removes fatal status |
| C2 | major | Main evidence is one small dataset with small mixed effects | Table 2; incomplete Section 7.6 | evidence, significance | experiment | Complete matched multi-seed MOSEI and paired analysis | ccf-experiment-designer | Consistent cross-dataset trend could add about +1 overall |
| C3 | major | Reliability-proportional direction lacks justification | Section 5.4 | soundness, novelty | method/soundness | Add gradient analysis and inverse/difficulty-aware matched control | ccf-experiment-designer | Clear mechanism/control raises soundness |
| C4 | major | Reliability validated mainly on Gaussian feature noise | Sections 7.2, 8.3 | evidence, domain validity | experiment | Add realistic audio and visual corruptions on untouched data | ccf-experiment-designer | External-validity evidence raises evidence score |
| C5 | major | Final-schedule actionability controls incomplete | Sections 5.5, 7.4 | evidence | experiment | Repeat Permuted/Reversed or narrow claim | ccf-experiment-designer | Removes correspondence/direction ambiguity |
| C6 | moderate | Three seeds and many metrics make tiny gains unstable | Table 2 | statistical rigor | experiment | Report paired seeds, primary metrics, intervals/more seeds | ccf-experiment-designer | More stable uncertainty supports descriptive claim |
| C7 | moderate | Novelty may collapse to careful engineering | Sections 2.2, 8.4 | novelty, significance | related-work | Compare calibration/imbalance work and state the new inference enabled | ccf-literature-searcher + ccf-paper-writer | Sharper delta raises novelty presentation |
| C8 | moderate | MOSEI status and `TBD` are internally inconsistent | Header, Sections 6.1/7.6, Table 2 | clarity, integrity | writing | Synchronize status and replace/justify missing loss | ccf-paper-writer | Eliminates avoidable reviewer distrust |
| C9 | moderate | NCA template, declarations, artifact, and compilation incomplete | `main.tex`, submission ledger | compliance, reproducibility | reproducibility | Compile official template; finish declarations and anonymized artifact | ccf-submission-checker | Required before submission |
| C10 | minor | Method figure may be dense after final scaling | Fig. 1 | clarity | writing | Check printed two-column readability and simplify small annotations | nature-figure | Figure remains legible in compiled PDF |

## 12. AC / Meta-Review

The reviewers would agree that the paper is more thoughtful than a routine weighting paper: it identifies concrete implementation failures, derives an exact finite-batch budget, supplies a strong uniform-allocation control, and writes limitations honestly. The likely disagreement is whether this methodological contribution is sufficient without strong performance gains. The decisive rejection axis is experimental protocol: the audio head and phase progression were informed by MOSI test audits, so the current MOSI tables cannot carry confirmatory weight despite the manuscript's contrary statement. The decisive acceptance axis is an untouched, frozen MOSEI replication showing that the audit and allocation conclusions persist without test-guided adaptation. Until then, the paper is promising but not submission-ready.

## 13. Quantitative Scores

## Scorecard

| Dimension | Score (1-5) | Confidence (1-5) | Evidence basis | Deduction / score-change condition |
|:---|:---:|:---:|:---|:---|
| Novelty | 3 | 4 | Sections 2.2–2.3, 5.4, 8.4; QMF/SMV/calibration/ARL context | Components are known; cross-dataset evidence and sharper audit inference could raise to 4 |
| Soundness | 2 | 5 | Section 6.2 versus experiment-log P1/P1.1; Section 5.4 | Test-guided development and under-justified allocation direction; untouched confirmation is required |
| Evidence | 2 | 5 | Tables 1–2; Sections 7.4–7.6 | One development-contaminated dataset, small effects, synthetic noise, incomplete controls; frozen MOSEI could raise to 3–4 |
| Significance | 3 | 4 | Abstract, Introduction, Discussion | Audit lesson is useful but current demonstrated impact is narrow |
| Clarity | 4 | 4 | Abstract, contributions, Fig. 1, Sections 8–9 | Strong bounded story; stale status and `TBD` prevent 5 |
| Reproducibility | 3 | 4 | Tests, logs, commands, Section 9 | Good internal traceability; no final anonymized artifact or clean split protocol |
| Ethics / Limitations | 4 | 4 | Section 9 and NCA declarations | Honest scope and responsible use; declarations and test-use disclosure incomplete |

**Overall:** 4/10  | **Scholarly Confidence:** 4/5

**Recommendation:** weak reject / major revision before submission  
**Verdict:** untouched frozen MOSEI confirmation plus transparent MOSI protocol disclosure could move the paper to 5–6; multi-seed confirmation, realistic corruptions, and final controls could move it toward 7. Failure to disclose test-guided development would lower the stance to clear reject.

- **Quality:** 3/5
- **Clarity:** 4/5
- **Significance:** 3/5
- **Originality:** 3/5
- **Soundness:** 2/5
- **Evidence:** 2/5
- **Reproducibility:** 3/5
- **Ethics / Limitations:** 4/5
- **Overall:** 4/10
- **Confidence:** 4/5

### Score-change conditions

| Change | Condition | Likely affected dimensions | Expected movement |
|---|---|---|---|
| Raise score | Frozen method replicates on untouched MOSEI with transparent split history | soundness, evidence, significance | about +1 overall |
| Raise score | Multi-seed MOSEI, realistic corruptions, and final Permuted/Reversed controls | evidence, soundness, novelty | additional +1 overall possible |
| Lower score | Continued claim that test data were not used despite logged test-driven design | integrity, soundness, evidence | -1 or fatal clear-reject stance |
| No quick change | Establishing broad model-agnostic reliability beyond MFON | novelty, significance | requires new backbone/dataset work |

## 14. Questions For Authors

1. Which exact MOSI split and samples were inspected when the audio reliability head was redesigned, and which later decisions used test-set reliability or sentiment metrics?
2. Has every P4 design choice been frozen before any MOSEI Constant/Learned result was inspected? If yes, can MOSEI be declared the untouched confirmatory study?
3. Why should higher-reliability samples receive more auxiliary supervision than low-reliability or difficult samples? What gradient or representation evidence supports this direction?
4. Can the final protocol report per-seed paired Learned-minus-Constant deltas and identify primary versus secondary metrics before MOSEI testing?
5. Can the authors release an anonymized configuration, environment, data-preparation description, and one-command test suite without redistributing restricted data or checkpoints?

## 15. Score Revision Criteria

**Raising the score would require:**

1. Correct the split-history claim and treat MOSI as exploratory where appropriate.
2. Complete frozen, untouched MOSEI baseline/Constant/Learned comparison; preferably multiple seeds.
3. Add realistic acoustic/visual corruption and repeat confound audits.
4. Add final-schedule correspondence/direction controls or narrow actionability.
5. Compile the official NCA template and complete the anonymized artifact/declarations.

**Lowering the score would be triggered by:**

- evidence that MOSEI choices were changed after examining formal test results;
- inconsistent or irreproducible seed-level numbers;
- reliability scores failing on untouched MOSEI or realistic corruptions while the abstract retains general language;
- failure to disclose MOSI test-guided architecture selection.

**Concerns unlikely to change before submission without new results:** model-agnostic generality, real-world robustness, and statistical confidence from only three seeds.

## 16. Action Plan And CCFA Handoffs

### Priority 0

- **Action:** audit every method decision against the split used and create a development/test provenance ledger.
- **Owner skill:** `ccf-integrity-auditor`
- **Input needed:** experiment log, commands, split labels, code history.
- **Expected output:** claim-support and data-split integrity report.
- **Handoff required:** yes before rewriting claims.

### Priority 1

- **Action:** finish MOSEI Constant and Learned without changing frozen settings; then expand seeds only if the predefined gate passes.
- **Owner skill:** `ccf-experiment-designer`
- **Input needed:** formal reload metrics and logs.
- **Expected output:** untouched confirmatory comparison and paired result table.
- **Handoff required:** no; experiment is already running.

### Priority 2

- **Action:** add realistic corruption, MOSEI confound audit, efficiency, and final Permuted/Reversed or a narrower claim.
- **Owner skill:** `ccf-experiment-designer`
- **Input needed:** frozen checkpoints and corruption definitions.
- **Expected output:** claim-aligned evidence package.
- **Handoff required:** yes before adding new experiment scope.

### Priority 3

- **Action:** revise the manuscript only after protocol classification and MOSEI results stabilize.
- **Owner skill:** `ccf-paper-writer`
- **Input needed:** integrity report and final tables.
- **Expected output:** synchronized English, Chinese, and NCA text with corrected claims.
- **Handoff required:** yes.

### Priority 4

- **Action:** run final numeric integrity, official-template compilation, anonymity, declarations, and artifact checks.
- **Owner skill:** `ccf-submission-checker`
- **Input needed:** final TeX, figures, bibliography, title page, artifact.
- **Expected output:** submission-readiness verdict.
- **Handoff required:** yes near submission.

**Checks run:** four-pass scientific review; public-safe related-work search; claim-evidence audit; experiment/protocol audit; code-path inspection; multi-reviewer panel; writing consistency scan; current method-figure inspection; NCA official-policy and CCF-category verification.  
**Checks skipped:** full LaTeX compilation because the official `sn-jnl` package/engine is unavailable; raw checkpoint reruns; statistical recomputation from raw per-seed files; plagiarism scan.  
**Unresolved risks:** completed MOSEI evidence, clean split provenance, realistic corruption behavior, final-schedule controls, artifact readiness, and final figure readability in the compiled NCA PDF.
