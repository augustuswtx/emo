# NCA submission-status ledger

Updated: 2026-08-16

## Current state

- Working venue: Neural Computing and Applications (CCF-C in the current CCF
  artificial-intelligence directory).
- Article type: original research article.
- Review model: double-blind.
- Format: generic Springer Nature `sn-jnl` working structure; latest official
  class and bibliography files still need to be downloaded.
- Abstract: 209 words, within the usual 150--250 word range.
- Keywords: five, within the requested 4--6 range.
- Compilation: not compiled because the local template and TeX engine are
  unavailable.
- Evidence designation: MOSI is exploratory/development evidence because
  test-split reliability diagnostics informed head design. The frozen MOSEI
  protocol is the confirmatory comparison.
- Active queue: the complete seed-1111 repaired MFON/Constant/Learned pilot is
  available; frozen replications on seeds 1112 and 1113 remain pending.
- Confirmatory outcome hierarchy, fixed before the P4 Constant formal test:
  MAE/Corr primary, Has0/Non0 accuracy/F1 secondary, and Acc-5/7, loss, and
  reliability/confound measures diagnostic.

## Evidence gates

- Complete the frozen CMU-MOSEI comparison without test-guided tuning; do not
  use the seed-1111 test direction to decide whether seeds 1112/1113 are run or
  reported.
- Preserve the explicit MOSI development-history disclosure in the Abstract,
  Experimental Protocol, Limitations, and Conclusion.
- Recheck the acoustic length confound on MOSEI.
- Add realistic acoustic and visual corruptions beyond Gaussian feature noise.
- Repeat at least Constant, Inverse/Difficulty-aware, and Permuted allocation
  controls under the final P4 schedule, or narrow the actionability claim.
- Report parameter count, training/inference time, and memory.
- Add the method, audit, and claim-aligned ablation figures/tables.

## Submission-package gates

- Download and compile with the latest official Springer Nature template.
- Prepare a separate title page containing author identities, affiliations,
  corresponding-author details, ORCIDs, acknowledgments, funding, and author
  contributions.
- Remove identities and identifying repository links from the manuscript,
  figures, supplementary files, data links, and PDF metadata.
- Complete Funding, Competing Interests, Data Availability, Code Availability,
  and Author Contributions statements.
- Verify that all editable source files, tables, figures, and supplementary
  material are included.
- Re-run citation and numeric-integrity checks after MOSEI integration.

## Do not claim yet

- State-of-the-art performance or comprehensive superiority.
- Statistical significance from three seeds.
- Real-world or inference-time robustness.
- General reliability estimation beyond the tested synthetic corruption
  family.
- Untouched-holdout confirmation from MOSI.
