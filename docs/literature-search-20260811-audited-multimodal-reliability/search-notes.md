# Search Notes

## Safe Queries Used

- `multimodal sentiment analysis CMU-MOSI dataset`
- `CMU-MOSEI dataset interpretable dynamic fusion graph`
- `tensor fusion memory fusion low-rank multimodal fusion sentiment`
- `multimodal transformer unaligned MISA MAG-BERT Self-MM MMIM`
- `robust multimodal sentiment feature reconstruction quality-aware fusion`
- Exact-title and DOI lookups for all retained papers

No unpublished manuscript sentence, unpublished result, private path, or private experiment name was used as a search query.

## Sources Checked

- ACL Anthology
- Crossref DOI records
- PMLR proceedings
- CVF Open Access
- AAAI DOI records
- ACM DOI records
- DBLP for missing page metadata
- arXiv for the 2026 leakage-safe diagnostic preprint

## Excluded Sources

- MDPI sources were excluded by policy.
- Search snippets, untraceable PDFs, low-signal venues, and title-near but technically unrelated papers were not retained.
- A Journal of Physics paper with a title similar to MISA was rejected after its DOI metadata did not match the intended ACM MM paper.

## Unknowns

- The INTERSPEECH proceedings record for `moon2026quality` was not independently verified; the arXiv record is accessible and the BibTeX entry remains marked as accepted/arXiv.
- A final 2026 closest-work sweep should be repeated after MOSEI evidence stabilizes.
- Public baseline numbers were not imported because feature, split, and preprocessing comparability has not been established.

## Handoff Notes

- **For writing:** the manuscript now has 27 cited/available entries and a mechanism-grouped Related Work section.
- **For experiment design:** TFR-Net, QMF, PDF, SAM-LML, QA-MoE, EBMC, and CPSC are the strongest named robustness/quality-aware comparators, subject to protocol compatibility.
- **For review:** the main novelty risk is that the audit/control package may appear MFON-specific without final-schedule controls and cross-dataset evidence.
