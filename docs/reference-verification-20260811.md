# Reference Verification Report

Date: 2026-08-11  
Input: `docs/small-paper-references.bib`  
Verification sources: DOI/Crossref, ACL Anthology, PMLR, CVF Open Access, AAAI/ACM DOI records, DBLP, and arXiv.

## Summary

- Entries after expansion: **27**
- Verified against stable publication metadata: **26**
- Check suggested: **1**
- Corrected critical metadata issues: **1**
- Unverifiable: **0**

## Corrected

| Key | Field | Previous value | Corrected value | Evidence |
|---|---|---|---|---|
| `mai2025samlml` | Pages | 21366--21386 | 21377--21397 | Crossref record for DOI `10.18653/v1/2025.emnlp-main.1084` |

## Check Suggested

| Key | Issue | Current treatment |
|---|---|---|
| `moon2026quality` | The arXiv metadata is verified, but the final INTERSPEECH proceedings record was not independently located during this pass. | Retain the arXiv URL and `Accepted; arXiv:2606.26473` note; replace with final proceedings metadata when available. |

## Verified Source Groups

- ACL/EMNLP/NAACL/COLING entries: DOI records and/or ACL Anthology pages matched titles, authors, years, and pages.
- ICML entries (`zhang2023qmf`, `cao2024pdf`): official PMLR pages matched titles, authors, years, and pages.
- CVPR entries (`wei2024smv`, `he2026ebmc`, `jiang2026cpsc`): CVF Open Access metadata matched titles, authors, years, and pages.
- ACM MM, AAAI, IEEE, and Pattern Recognition additions: DOI records matched titles, authors, venues, years, and available page metadata.

## Remaining Submission Check

Run one final verification after choosing the target venue and after the 2026 proceedings cycle is complete. At that point, normalize venue abbreviations and capitalization to the selected bibliography style, replace the provisional INTERSPEECH/arXiv record if a final record exists, and remove any bibliography entry that is no longer cited.
