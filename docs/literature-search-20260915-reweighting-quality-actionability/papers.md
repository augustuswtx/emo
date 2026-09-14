# Targeted literature update

## Why these papers matter

1. **Ren et al. (ICML 2018)** learn example weights from validation-gradient directions. This is the clearest contrast to a hand-designed reliability-proportional rule: it ties weighting to downstream validation utility rather than assuming that a proxy score is comparable across samples.
2. **Kumar et al. (NeurIPS 2010)** formalize self-paced, easy-first learning. It makes the allocation direction an explicit modeling choice and motivates the required inverse/difficulty-aware control.
3. **Mai et al. (EMNLP 2022)** bring curriculum learning into multimodal correlation learning and define difficulty through pair losses. It is a domain-specific precedent for choosing which multimodal pairs receive training emphasis.
4. **Zou et al. (Findings ACL 2026)** estimate reconstruction reliability under missing modalities and use it for inference-time gating. It is close in vocabulary but different in evidence and deployment: the present study uses a training-time Gaussian-degradation response to allocate auxiliary losses and does not route inference.

## Positioning consequence

The revised paper does not present reliability weighting, corruption ranking, or easy-first selection as individually novel. Its defensible contribution is the joint audit-and-control package: sample-granular loss construction, exact finite-batch budget conservation, explicit score actionability controls, and negative cross-corruption evidence. The revised text also states that within-sample ordinal supervision does not by itself establish cross-sample calibration.
