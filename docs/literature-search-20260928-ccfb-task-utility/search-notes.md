# CCF-B task-utility allocation literature search notes

Date: 2026-09-28

## Purpose

Identify a CCF-B-level research direction that reuses the completed MFON fixed-budget,
reliability, and actionability audits while avoiding a cosmetic combination of modules.
The search asks four questions:

1. How do strong multimodal methods value samples or modalities during training?
2. How do auxiliary-learning and distillation methods prevent harmful gradients?
3. What is already occupied by recent 2024--2026 work?
4. What evidence would distinguish a new allocation mechanism from gating, extra
   computation, or ordinary modality balancing?

This is a targeted, reproducible search, not a claim to have found every paper.

## Source policy

- Primary sources: official conference proceedings, ACL Anthology, PMLR, CVF Open
  Access, NeurIPS proceedings, ICLR proceedings, and the official CCF directory.
- Secondary search results were used only to locate primary records. An anonymous
  OpenReview submission is recorded only as a novelty risk and not treated as an
  accepted paper.
- The private manuscript and experiment logs were used only to define the research
  problem. Their text and unpublished results were not used as public search queries.
- No venue label is inferred from impact factor. CCF labels are checked against the
  seventh directory released in March 2026.

## Query families

- `sample-level modality valuation multimodal learning`
- `multimodal gradient modulation knowledge distillation modality imbalance`
- `task gradient auxiliary loss weighting sample reweighting`
- `cross-batch gradient knowledge distillation adaptive loss balancing`
- `multimodal sentiment reliability uncertainty auxiliary supervision`
- `teacher student compatibility gradient data valuation`
- Official-site constrained variants for CVF, PMLR, ACL Anthology, NeurIPS, ICLR,
  and CCF.

## Screening rules

Include papers that contribute at least one of the following:

- sample- or modality-level valuation;
- task/auxiliary gradient alignment or conflict handling;
- multimodal knowledge distillation or under-optimized modality learning;
- uncertainty/reliability-aware fusion with a usable control;
- influence/data valuation that clarifies the proposed label definition.

Exclude papers whose only overlap is a generic attention, fusion, or emotion model.

## Search conclusion

The broad ideas "sample-level modality weighting", "gradient-guided multimodal
distillation", and "dynamic task/KD balancing" are already occupied by SLMV,
CMoB, G2D, MMPareto, AutAuT, and DTO-KD. A CCF-B contribution therefore cannot be
"add a quality head and reweight samples" or "combine MFON with PCGrad".

The remaining defensible gap is narrower: learn whether a *particular
sample-modality auxiliary update* helps the sentiment objective, then amortize this
otherwise expensive utility measurement into a cheap, calibrated predictor. The
existing MFON audits supply unusually useful negative and positive evidence for this
gap: degradation reliability did not predict allocation validity, while visual
parameter-space utility predicted reversible local loss-change direction but was too
expensive to compute every batch. This motivates the method rather than merely
decorating it.

## Novelty threats to monitor

1. CMoB already performs causal sample-level modality valuation.
2. SLMV already values modalities per sample and applies targeted enhancement.
3. G2D already combines gradient guidance, distillation, and modality prioritization.
4. DTO-KD and MMPareto already handle task/distillation gradient conflict and
   dominance at objective level.
5. Meta-Weight-Net already learns an explicit sample-weight mapping from meta
   gradients.
6. The anonymous work *Cross-Batch Gradient Consistency for Adaptive Loss Balancing
   in Knowledge Distillation* is extremely close to using a separate batch gradient
   for KD weighting. Its final acceptance status was not verified from official
   proceedings in this search, so it is a risk, not an established accepted baseline.

The proposed work must therefore demonstrate all four distinctions: modality-specific
auxiliary-update utility, sparse utility supervision, calibrated abstention, and
fixed-budget actionability controls.

