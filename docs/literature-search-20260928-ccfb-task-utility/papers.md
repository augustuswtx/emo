# Literature map: task-utility allocation for CCF-B multimodal learning

Date: 2026-09-28

## Executive conclusion

The literature does not support a paper whose novelty is merely dynamic sample
weighting, modality balancing, gradient modulation, or knowledge distillation. Those
areas already contain strong and recent methods. The viable opening is to distinguish
three quantities that existing MSA work often merges:

1. **degradation detectability**: whether a modality looks corrupted;
2. **teacher-target fidelity**: whether a student matches a frozen unimodal teacher;
3. **training utility**: whether applying that sample-modality auxiliary update helps
   the downstream task on separate data.

The completed MFON experiments already show that the first quantity does not imply
the third. They also show that a visual parameter-space utility measurement predicts
local update direction well but is too expensive to run every batch. This creates a
coherent method problem: learn a cheap predictor from sparse exact utility labels and
use conservative abstention under a fixed auxiliary budget.

## Cluster 1: sample and modality valuation

| Work | What it occupies | Consequence for this project |
|---|---|---|
| [SLMV, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wei_Enhancing_Multimodal_Cooperation_via_Sample-level_Modality_Valuation_CVPR_2024_paper.html) | Sample-level modality contribution and targeted enhancement | We cannot claim novelty from per-sample modality weights alone. |
| [CMoB, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/76a0f4215050c7d4e88c03cb8a11fee6-Abstract-Conference.html) | Causal sample-level modality valuation over training stages | We must compare against it and distinguish auxiliary-update utility from modality contribution. |
| [QMF, ICML 2023](https://proceedings.mlr.press/v202/zhang23ar.html) | Quality uncertainty for low-quality fusion | Our score acts on training-time auxiliary supervision, not only inference fusion. |
| [ARL, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wei_Improving_Multimodal_Learning_via_Imbalanced_Learning_ICCV_2025_paper.html) | Variance-driven asymmetric optimization | Equalizing modality learning is not automatically a desirable goal. |

## Cluster 2: gradient control in multimodal learning

| Work | What it occupies | Consequence for this project |
|---|---|---|
| [OGM-GE, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.html) | Online modality-level gradient modulation | Required older baseline. |
| [PMR, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Fan_PMR_Prototypical_Modal_Rebalance_for_Multimodal_Learning_CVPR_2023_paper.html) | Prototype-based modality rebalance | Required modality imbalance comparator. |
| [AGM, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Boosting_Multi-modal_Model_Performance_with_Adaptive_Gradient_Modulation_ICCV_2023_paper.html) | Adaptive gradient direction and magnitude | Generic gradient modulation is occupied. |
| [MMPareto, ICML 2024](https://proceedings.mlr.press/v235/wei24d.html) | Pareto reconciliation of multimodal and unimodal objectives | A direct strong baseline for task/auxiliary conflicts. |
| [CGGM, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f0c68d99827dc09ed28aa073455efcbe-Abstract-Conference.html) | Classifier-guided gradient modulation | The new method needs sample-modality evidence, not another global coefficient. |
| [AutAuT, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/61c2975281d60d3b1ce4cefc157d99df-Abstract-Conference.html) | Gradient-aligned auxiliary-task selection and weighting | Task-level gradient alignment is current, making sample-level actionability evidence essential. |

## Cluster 3: multimodal knowledge distillation

| Work | What it occupies | Consequence for this project |
|---|---|---|
| [MFON, COLING 2025](https://aclanthology.org/2025.coling-main.309/) | Intra-modal teacher distillation plus cross-modal contrastive learning | The inherited backbone and teacher losses. |
| [TelME, NAACL 2024](https://aclanthology.org/2024.naacl-long.5/) | Language-teacher distillation to nonverbal students | Relevant MSA/emotion KD baseline. |
| [G2D, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Rakib_G2D_Boosting_Multimodal_Learning_with_Gradient-Guided_Distillation_ICCV_2025_paper.html) | Gradient-guided multimodal KD and sequential modality prioritization | The closest multimodal KD novelty threat. |
| [DTO-KD, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e546ce4230c15bc422a500bb47b341ea-Abstract-Conference.html) | Dynamic task/KD gradient conflict and dominance optimization | Global task/KD trade-off is occupied; our claim must be sample-modality selection plus amortization. |
| [GRACE, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/59fa5d049b777d880fdcddba2b24738f-Abstract-Conference.html) | Lightweight gradient-based teacher compatibility | Motivates measuring teacher usefulness but operates at teacher selection scale. |

## Cluster 4: meta weighting and data valuation

| Work | What it occupies | Consequence for this project |
|---|---|---|
| [Learning to Reweight, ICML 2018](https://proceedings.mlr.press/v80/ren18a.html) | Validation-gradient example weighting | Direct theoretical ancestor; must be cited, not presented as new. |
| [Meta-Weight-Net, NeurIPS 2019](https://papers.nips.cc/paper/2019/hash/e58cc5ca94270acaceed13bc82dfedf7-Abstract.html) | Learned mapping from sample loss to meta-derived weight | Closest precedent for the amortized predictor. Our distinction is modality-specific teacher update, fixed budget, and abstention. |
| [TracIn, NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html) | Gradient-traced example influence | Supports the use of local parameter-space labels while warning that one checkpoint is insufficient. |
| [PCGrad, NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) | Projection of conflicting task gradients | Required generic conflict-control baseline. |

## Opportunity map

| Opportunity | Evidence already available | Missing evidence | Priority |
|---|---|---|---:|
| Reliability and utility are different | Three-seed negative cross-sample audit; C5 stratification; inverse/permuted control | Replication across models and datasets | High |
| Exact visual utility is locally meaningful | 91/95 predicted/actual sign agreement on fresh random pairs | Multiple checkpoints, seeds, Adam-aware or longer-horizon check | High |
| Exact scoring is too expensive | 1.57--1.60x extra scoring cost; stale reuse failed | Sparse-label cadence cost under real training | High |
| A cheap predictor can amortize utility | Not yet established | Cross-batch and cross-checkpoint predictor evaluation | Critical go/no-go |
| Conservative allocation improves task performance | Not yet established | Matched-budget training controls and multi-seed results | Critical go/no-go |
| Audio should be allocated dynamically | Current evidence is negative | A stable audio utility definition | Low; keep uniform initially |

## Benchmark candidates

- Datasets already supported: CMU-MOSEI, CMU-MOSI, CH-SIMS/SIMS.
- Primary backbone: MFON with frozen unimodal teachers.
- Transfer backbone: one structurally different fusion model, preferably MISA or
  MulT, equipped with the same frozen-teacher auxiliary objective for a controlled
  portability test.
- Strong method baselines: Constant auxiliary weight, old reliability allocation,
  SLMV, CMoB, MMPareto, G2D, ARL, DTO-KD/PCGrad-style objective control.
- Robustness comparators: QMF and P-RMF where protocol compatibility permits.

## Positioning risks

1. If the utility predictor works only at the final checkpoint, it is an audit tool,
   not a training method.
2. If the predictor improves performance but permuted allocation does too, the gain
   is likely regularization or gating rather than correct assignment.
3. If gate-only uniform matches the method, abstention rather than allocation caused
   the gain.
4. If only MFON/MOSEI works, the evidence is too narrow for the intended CCF-B claim.
5. If computation exceeds roughly 15% added training cost, the amortization claim is
   weak unless gains are unusually large.

