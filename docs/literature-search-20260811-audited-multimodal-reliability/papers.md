# Literature Search: Audited Reliability and Fixed-Budget Learning for Multimodal Sentiment Analysis

Date: 2026-08-11  
Search purpose: strengthen the Introduction, Related Work, dataset attribution, and baseline context of `small-paper-draft-v2-en.md`.  
Target venue/family: unspecified AI/NLP/ML venue; user-custom CCF writing format.  
Source-quality policy: primary or stable scholarly sources only; MDPI and untraceable sources excluded.

## Summary

- **Foundational progression:** dataset/task definition -> explicit tensor/recurrent fusion -> efficient factorized fusion -> cross-modal attention -> pretrained and representation-disentangled models.
- **Closest robustness progression:** feature reconstruction -> uncertainty/calibration -> supervised corruption -> incomplete-modality proxies -> reliability-aware routing and calibration.
- **Opportunity:** quality-score auditing and exact finite-batch auxiliary-budget control remain a defensible diagnostic/control framing, but generic reliability weighting and corruption ranking are already covered concepts.
- **Primary novelty risk:** without final-schedule actionability controls and cross-dataset evidence, the work can be read as a well-controlled MFON ablation rather than a general method.
- **Recommended writing action:** lead with falsifiable audit failures and budget fairness; present the reliability head as an implementation component, not the sole novelty claim.

## Paper Table

Scores are screening judgments on a 1--5 scale, based on verified publication records and the paper's role in the MSA literature. They are not acceptance predictions.

| # | Title | Year | Venue/source | Link | Type | Insight | Completeness | Numeric evidence | Overall | Relevance |
|---:|---|---:|---|---|---|---:|---:|---:|---|---|
| 1 | Multimodal Sentiment Intensity Analysis in Videos | 2016 | IEEE Intelligent Systems | [DOI](https://doi.org/10.1109/MIS.2016.94) | method + benchmark | 4 | 3 | 3 | A | CMU-MOSI task/data anchor |
| 2 | Tensor Fusion Network for Multimodal Sentiment Analysis | 2017 | EMNLP | [ACL](https://aclanthology.org/D17-1115/) | pure method | 4 | 4 | 4 | A | Explicit high-order interaction baseline |
| 3 | Memory Fusion Network for Multi-view Sequential Learning | 2018 | AAAI | [DOI](https://doi.org/10.1609/aaai.v32i1.12021) | pure method | 4 | 4 | 4 | A | Temporal multi-view fusion anchor |
| 4 | Efficient Low-rank Multimodal Fusion with Modality-Specific Factors | 2018 | ACL | [ACL](https://aclanthology.org/P18-1209/) | pure method | 4 | 4 | 4 | A | Efficient tensor-fusion baseline |
| 5 | Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph | 2018 | ACL | [ACL](https://aclanthology.org/P18-1208/) | method + benchmark | 5 | 5 | N/A benchmark | A | CMU-MOSEI source and scale anchor |
| 6 | Multimodal Transformer for Unaligned Multimodal Language Sequences | 2019 | ACL | [ACL](https://aclanthology.org/P19-1656/) | pure method | 5 | 5 | 5 | A | Cross-modal attention and unaligned fusion |
| 7 | MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis | 2020 | ACM MM | [DOI](https://doi.org/10.1145/3394171.3413678) | pure method | 4 | 4 | 4 | A | Shared/specific representation baseline |
| 8 | Integrating Multimodal Information in Large Pretrained Transformers | 2020 | ACL | [ACL](https://aclanthology.org/2020.acl-main.214/) | pure method | 4 | 4 | 4 | B | Pretrained-language-model fusion line |
| 9 | Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning | 2021 | AAAI | [DOI](https://doi.org/10.1609/aaai.v35i12.17289) | pure method | 4 | 4 | 4 | A | Modality-specific auxiliary supervision |
| 10 | Improving Multimodal Fusion with Hierarchical Mutual Information Maximization | 2021 | EMNLP | [ACL](https://aclanthology.org/2021.emnlp-main.723/) | pure method | 4 | 4 | 4 | A | Mutual-information auxiliary objective |
| 11 | Transformer-Based Feature Reconstruction Network for Robust MSA | 2021 | ACM MM | [DOI](https://doi.org/10.1145/3474085.3475585) | pure method | 4 | 4 | 4 | Risk | Close robustness baseline using feature reconstruction |
| 12 | UniMSE | 2022 | EMNLP | [ACL](https://aclanthology.org/2022.emnlp-main.534/) | pure method | 4 | 4 | 4 | B | Unified sentiment/emotion learning context |
| 13 | CubeMLP | 2022 | ACM MM | [DOI](https://doi.org/10.1145/3503161.3548025) | pure method | 3 | 4 | 4 | B | Alternative lightweight interaction architecture |
| 14 | Learning Language-Guided Adaptive Hyper-Modality Representation | 2023 | EMNLP | [ACL](https://aclanthology.org/2023.emnlp-main.49/) | pure method | 4 | 4 | 4 | A | Adaptive language-guided fusion line |
| 15 | TETFN | 2023 | Pattern Recognition | [DOI](https://doi.org/10.1016/j.patcog.2022.109259) | pure method | 3 | 4 | 4 | B | Text-enhanced transformer comparison |
| 16 | Multimodal Multi-Loss Fusion Network for Sentiment Analysis | 2024 | NAACL | [ACL](https://aclanthology.org/2024.naacl-long.197/) | pure method | 4 | 4 | 4 | A | Multi-loss optimization context |

## Clusters

### Cluster 1: Task, datasets, and classical fusion

- **Representative papers:** MOSI, TFN, MFN, LMF, CMU-MOSEI.
- **Already covered:** task formulation, high-order interaction, temporal fusion, efficient factorization, and large-scale benchmark construction.
- **Remaining gap:** these works do not establish that a learned sample-wise quality variable corresponds to controlled degradation.
- **Effect on this paper:** cite them to define the task and the progression of fusion, not as direct quality-aware competitors.

### Cluster 2: Representation and auxiliary-objective learning

- **Representative papers:** MulT, MAG, MISA, Self-MM, MMIM, UniMSE, CubeMLP, ALMT, TETFN, MMLF, MFON.
- **Already covered:** cross-modal attention, pretrained text conditioning, shared/specific representations, self-supervision, mutual-information objectives, and multi-loss learning.
- **Remaining gap:** auxiliary objectives can improve representations while leaving sample-wise reliability unmeasured or unaudited.
- **Effect on this paper:** MFON ownership must remain explicit; the current contribution concerns how its existing auxiliary losses are allocated and audited.

### Cluster 3: Robust and quality-aware multimodal learning

- **Representative papers:** TFR-Net, QMF, PDF, SAM-LML, P-RMF, QA-MoE, EBMC, CPSC, and the leakage-safe quality-dependence diagnostic.
- **Already covered:** reconstruction, uncertainty-aware fusion, calibration, corruption ordering, proxy modalities, expert routing, modality trust, and score permutation.
- **Remaining gap:** exact finite-batch control of training-time auxiliary supervision and a combined granularity/monotonicity/confound/actionability/collapse audit remain under-tested as a package.
- **Effect on this paper:** generic claims such as “first reliability weighting” or “first corruption ranking” are untenable. The paper must emphasize auditability, matched budgets, and bounded empirical conclusions.

## Opportunity Map

| Cluster | Status | Open gap | Possible direction | Evidence needed | Risk |
|---|---|---|---|---|---|
| Quality-aware fusion | Crowded but open | Scores may be non-actionable or confounded | Standardized quality-score audit | Multi-method, multi-dataset audits | High if evaluated only on MFON |
| Auxiliary optimization | Mechanism gap | Dynamic weights confound allocation with total supervision | Exact-budget allocation controls | Final-schedule constant/permuted/reversed/oracle comparisons | Medium-high |
| Robust MSA | Benchmark gap | Synthetic and real corruptions are not aligned across papers | Shared severity and unknown-corruption protocol | MOSEI/SIMS plus realistic corruption families | High cost |
| Reliability estimation | Theory/analysis gap | Reliability and task utility are often conflated | Separate measurement from allocation utility | Gradient analysis and utility-aware controls | Medium |
| Text-dominant fusion | Negative-result opportunity | Audio/visual degradation may barely affect predictions | Report dependence failure rather than hide it | Modality ablation and inference dependence metrics | Medium |

## Citation and Positioning Cautions

- Cite MFON whenever describing the base prompts, frozen teachers, KL losses, InfoNCE losses, or fusion decoder.
- Cite SAM-LML for corruption-based ordering; do not claim that ordered clean/corrupt supervision is new by itself.
- Cite QMF/PDF/QA-MoE when discussing quality-aware fusion; do not claim first use of reliability-aware weighting.
- Cite the score-permutation diagnostic when motivating actionability tests.
- Treat TFR-Net and recent robust methods as required baselines or explicit protocol-mismatch references once cross-dataset experiments are complete.
- The closest-work search should be refreshed after the final MOSEI results determine whether the paper is primarily diagnostic, methodological, or both.
