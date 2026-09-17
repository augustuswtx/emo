# Task-aligned auxiliary allocation for multimodal sentiment analysis

Date: 2026-09-17. Purpose: find a defensible redesign of MFON's training-time,
per-sample visual/acoustic auxiliary-loss allocation. This is a targeted,
non-exhaustive search of directly relevant top-field proceedings and journals;
"all papers" is not a finite or verifiable search claim. Titles, venue status,
and the one-sentence mechanisms below were checked against linked primary
records. Scores are provisional screening judgments based mainly on proceedings
abstracts, not a substitute for full-paper replication or a venue ranking.

Scores are insight/completeness/numeric-evidence on a 1–5 scale. Numeric-evidence
scores summarize the *reported evaluation scope*, not verified effect sizes.
`A` means close prior art or a necessary control, `B` means supporting work.

## Decision in brief

The strongest bridge is to replace degradation-response `q` as the allocation
criterion with a **task-aligned utility estimate** for the actual auxiliary
update. Retain the old `q` only as a separately audited input condition or
feature. The strongest novelty risk is that generic validation-gradient
reweighting, per-sample multimodal valuation, and gradient conflict handling
already exist. The contribution must be the particular *auxiliary-target
utility* question, a fixed-budget implementation, and evidence that the score
predicts the effect of an auxiliary update under matched controls.

## Verified paper matrix

| # | Paper (primary link) | Year / source | Type | Scores | Role and closest overlap |
|---:|---|---|---|---|---|
| 1 | [Learning to Reweight Examples for Robust Deep Learning](https://proceedings.mlr.press/v80/ren18a.html) | 2018 ICML | pure method | 5/4/4 A | Clean validation gradients set example weights; direct ancestor of task-utility weighting. |
| 2 | [Meta-Weight-Net](https://papers.nips.cc/paper/2019/hash/e58cc5ca94270acaceed13bc82dfedf7-Abstract.html) | 2019 NeurIPS | pure method | 4/4/4 A | Learns a weighting map from sample loss with unbiased meta-data; learnable-gate baseline. |
| 3 | [Adaptive Auxiliary Task Weighting for Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2019/hash/0e900ad84f63618452210ab8baae0218-Abstract.html) | 2019 NeurIPS | pure method | 4/4/3 A | Defines auxiliary usefulness through main-task loss reduction; task-level precursor. |
| 4 | [Auxiliary Task Reweighting for Minimum-data Learning](https://papers.nips.cc/paper/2020/hash/4f87658ef0de194413056248a00ce009-Abstract.html) | 2020 NeurIPS | pure method | 4/4/4 A | Learns relative auxiliary-task weights; compare scope with per-sample, per-modality allocation. |
| 5 | [Gradient Surgery for Multi-Task Learning](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) | 2020 NeurIPS | pure method | 5/4/4 A | Removes conflicting gradient components; diagnostic and gradient-control baseline. |
| 6 | [Scaling Multimodal Pre-Training via Cross-Modality Gradient Harmonization](https://proceedings.neurips.cc/paper_files/paper/2022/hash/eacad5b8e67850f2b8dd33d87691d097-Abstract-Conference.html) | 2022 NeurIPS | pure method | 5/4/4 A | Detects within-triplet contrastive gradient conflict and uses gradient curriculum; closest multimodal gradient precedent. |
| 7 | [Balanced Multimodal Learning via On-the-Fly Gradient Modulation](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.html) | 2022 CVPR | pure method | 4/4/4 A | Adjusts modality optimization imbalance; global modality-level comparator. |
| 8 | [Enhancing Multimodal Cooperation via Sample-level Modality Valuation](https://openaccess.thecvf.com/content/CVPR2024/papers/Wei_Enhancing_Multimodal_Cooperation_via_Sample-level_Modality_Valuation_CVPR_2024_paper.pdf) | 2024 CVPR | pure method | 5/4/4 A | Estimates sample-level modality contribution; very close terminology and evaluation risk. |
| 9 | [Classifier-guided Gradient Modulation for Enhanced Multimodal Learning](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f0c68d99827dc09ed28aa073455efcbe-Abstract-Conference.html) | 2024 NeurIPS | pure method | 4/4/4 A | Uses gradient magnitude and direction to address modality dominance; includes CMU-MOSI regression. |
| 10 | [Provable Dynamic Fusion for Low-Quality Multimodal Data](https://proceedings.mlr.press/v202/zhang23ar.html) | 2023 ICML | pure method | 5/4/4 A | QMF estimates uncertainty for inference-time dynamic fusion; different action, strong quality-aware baseline. |
| 11 | [Predictive Dynamic Fusion](https://proceedings.mlr.press/v235/cao24c.html) | 2024 ICML | pure method | 4/4/4 B | Calibrated multimodal confidence for dynamic fusion; avoid copying inference-time claims. |
| 12 | [Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis](https://openaccess.thecvf.com/content/CVPR2026/papers/He_Enhance-then-Balance_Modality_Collaboration_for_Robust_Multimodal_Sentiment_Analysis_CVPR_2026_paper.pdf) | 2026 CVPR | pure method | 4/4/4 A | Instance-aware trust distillation plus gradient balancing; unusually close MSA prior art. |
| 13 | [Multimodal Learning on Low-Quality Data with Conformal Predictive Self-Calibration](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Multimodal_Learning_on_Low-Quality_Data_with_Conformal_Predictive_Self-Calibration_CVPR_2026_paper.html) | 2026 CVPR | pure method | 4/4/4 A | Instance reliability also calibrates gradient flow; strongest risk for a generic gradient-gate claim. |
| 14 | [QA-MoE](https://aclanthology.org/2026.acl-long.1461/) | 2026 ACL | pure method | 4/4/4 A | Continuous reliability controls expert routing under degraded/missing MSA modalities; different action from auxiliary allocation. |
| 15 | [MSD: Saliency-aware Knowledge Distillation for Multimodal Understanding](https://aclanthology.org/2021.findings-emnlp.302/) | 2021 Findings EMNLP | pure method | 4/4/4 A | Weights modality-specific distillation via saliency; directly relevant to auxiliary target selection. |
| 16 | [Knowledge-Guided Dynamic Modality Attention Fusion](https://aclanthology.org/2024.findings-emnlp.865/) | 2024 Findings EMNLP | pure method | 3/4/4 B | KuDA adjusts dominant modality at inference; task-grounded fusion comparator. |
| 17 | [MissModal](https://aclanthology.org/2023.tacl-1.94/) | 2023 TACL | pure method | 4/4/4 B | Missing-modality representation alignment; baseline only if missing-modality robustness is claimed. |
| 18 | [Supervised Attention Mechanism for Low-quality Multimodal Data](https://aclanthology.org/2025.emnlp-main.1084/) | 2025 EMNLP | pure method | 4/4/4 A | Ranking supervision for attention weights under noisy/missing modalities; warns that ordinal ranks need a validated action. |
| 19 | [Analyzing Modality Robustness in Multimodal Sentiment Analysis](https://aclanthology.org/2022.naacl-main.50/) | 2022 NAACL | other: empirical analysis | 4/4/4 B | Reference protocol for perturbation and text dominance claims. |
| 20 | [Dataset Cartography](https://aclanthology.org/2020.emnlp-main.746/) | 2020 EMNLP | pure method | 5/4/4 B | Training dynamics separate easy, ambiguous, and hard examples; alternatives to raw confidence. |
| 21 | [Estimating Training Data Influence by Tracing Gradient Descent](https://papers.nips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html) | 2020 NeurIPS | pure method | 5/4/4 B | TracIn offers a checkpoint/gradient-based validation of sample influence; expensive audit comparator. |
| 22 | [Multimodal Multi-loss Fusion Network for Sentiment Analysis](https://aclanthology.org/2024.naacl-long.197/) | 2024 NAACL | pure method | 3/4/4 B | Relevant MSA multi-loss and encoder baseline; task-level comparison requires matching preprocessing. |

## Closest-work clusters and opportunity

1. **Validation-guided weighting** (1–4): already establishes that a useful
   weight can be learned from main-task feedback. Open here: fixed-budget,
   *per-sample auxiliary KL* for visual/acoustic teachers, with distinct
   allocation and reliability hypotheses. Novelty cannot be “meta weighting.”
2. **Gradient conflict and multimodal balancing** (5–9, 13): already uses
   gradient direction and sample/modal granularity. Open here: distinguish
   “auxiliary update helps the sentiment objective” from “modalities have
   balanced magnitudes,” and test score correspondence and direction with
   matched Constant/Permuted/Inverse controls.
3. **Reliability and fusion** (10–14, 16–18): already estimates modality quality
   for fusion, expert routing, or generic gradient modulation. Open here:
   measure *training-time supervision utility* and keep inference fixed.
4. **Target and data diagnostics** (15, 19–22): evidence is needed that a
   frozen teacher's target and its gradient are useful. Low KL, clean-input
   response, confidence, and sample difficulty are different properties.

## Scope and citation caution

- The proposed gradient-alignment score has a first-order **local update**
  interpretation. That does not imply global convergence or higher test Corr.
- A zero-ablation error change is a perturbation sensitivity proxy, not causal
  modality contribution. The CVPR 2024 modality-valuation paper must be
  compared before using “sample-level modality value” as novel terminology.
- A 2026 arXiv [decision-dependence diagnostic](https://arxiv.org/abs/2606.26473)
  is a close caution for permutation controls, but its official venue record
  was not independently located in this search; keep its status qualified.
- The CVPR/ACL 2026 works make “reliability-aware multimodal learning” a
  crowded central claim. The manuscript needs a specific, audited auxiliary
  allocation mechanism and a clear distinction from inference-time fusion.
