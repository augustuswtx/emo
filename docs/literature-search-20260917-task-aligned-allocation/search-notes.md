# Search notes: task-aligned auxiliary allocation

Date: 2026-09-17. Mode: standard search feeding method and experiment design.
The user's private draft text and unpublished MFON numerical results were not
used in external queries. Twenty-two included records were deduplicated by
title and linked to official proceedings or journal pages, except where a
primary proceedings PDF was more accessible than its HTML page.

## Public search strings used

- `Learning to Reweight Examples for Robust Deep Learning ICML`
- `Meta-Weight-Net sample weighting NeurIPS`
- `auxiliary task reweighting gradient main task NeurIPS`
- `Scaling Multimodal Pre-Training Cross-Modality Gradient Harmonization`
- `sample-level modality valuation CVPR`
- `quality aware multimodal fusion ICML ACL`
- `multimodal gradient modulation sentiment analysis CVPR NeurIPS`
- `modality-specific distillation saliency ACL`
- `multimodal sentiment analysis missing modality robustness ACL`
- `Dataset Cartography training dynamics EMNLP`
- `TracIn training data influence NeurIPS`

## Sources and screening

- Primary source families checked: PMLR/ICML, NeurIPS proceedings, CVF/CVPR,
  ACL Anthology/NAACL/EMNLP/TACL, and arXiv for one status-qualified diagnostic.
- Included 22 proceedings/journal papers spanning example weighting,
  auxiliary-task weighting, gradient conflict, sample modality valuation,
  quality-aware fusion, sentiment analysis, and data influence.
- Removed duplicate PDF/HTML hits and review-page copies of the same work.
- Excluded publisher-policy sources, untraceable summaries, and unrelated
  medical or tabular missing-modality papers from the scored table. No claim
  about excluded source quality is inferred from a search snippet.
- The one arXiv decision-dependence diagnostic appears in the caution notes,
  not in the scored table, because an official conference record was not
  independently verified.

## Evidence depth and unknowns

- All included title/year/venue fields are backed by primary records. Core
  mechanisms were checked against abstracts; selected PDFs were inspected for
  the especially close CVPR 2024/2026 and NeurIPS 2022 papers.
- The quality scores are triage labels. The full numerical tables, data splits,
  significance tests, implementation licenses, and code reproducibility of
  every paper were not independently audited.
- Whether a validation-gradient gate will improve MFON on unseen seeds or on
  a second dataset is unknown. No published result is transferred as an
  expected MFON gain.
- Direct integration of MoE or new encoders would confound the allocation
  question and require new matched controls; it was screened as lower priority.

## Handoff to experimental design

Test the narrow claim: at the same total auxiliary KL budget, a score tied to
main-task update utility assigns more weight to useful sample-modality pairs.
Do a read-only gradient/influence audit before any new 25-epoch run, then a
single-seed prototype against Constant, original learned-q, score-permuted,
and score-inverted controls. Keep meta-train data separate from
checkpoint-selection validation. Existing MOSEI test aggregates and C5 strata
have already informed this redesign, so do not describe that test set as an
untouched confirmation set for the *new* method; plan an independent locked
confirmation protocol before making a generalization claim.
