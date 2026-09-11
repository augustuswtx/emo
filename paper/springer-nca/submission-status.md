# NCA submission-status ledger

Updated: 2026-09-11

## Current state

- Working venue: Neural Computing and Applications (CCF-C in the current CCF
  artificial-intelligence directory).
- Article type: original research article.
- Review model: double-blind.
- Format: generic Springer Nature `sn-jnl` working structure; latest official
  class and bibliography files still need to be downloaded.
- Abstract: 222 words, within the usual 150--250 word range.
- Keywords: five, within the requested 4--6 range.
- Compilation: not compiled because the local template and TeX engine are
  unavailable.
- Evidence designation: MOSI is exploratory/development evidence because
  test-split reliability diagnostics informed head design. The frozen MOSEI
  protocol is the confirmatory comparison.
- Evidence complete: all three prespecified MOSEI seeds for repaired MFON,
  P4 Constant, and P4 Learned; full-test Gaussian reliability audits; and four
  held-out corruption audits; plus a matched single-GPU efficiency audit. The
  extended implementation suite passes 37 tests.
- Confirmatory outcome hierarchy, fixed before the P4 Constant formal test:
  MAE/Corr primary, Has0/Non0 accuracy/F1 secondary, and Acc-5/7, loss, and
  reliability/confound measures diagnostic.

## Evidence gates

- Preserve the complete three-seed MOSEI table and the explicit finding that
  Learned improves the primary endpoints over Constant but not uniformly over
  repaired MFON.
- Preserve the explicit MOSI development-history disclosure in the Abstract,
  Experimental Protocol, Limitations, and Conclusion.
- Preserve the MOSEI acoustic-length confound and the negative cross-corruption
  results; do not describe the reliability heads as general quality estimators.
- Add original-media acoustic and visual corruptions beyond feature-level
  Gaussian/non-Gaussian stress tests if deployment robustness is claimed.
- Repeat at least Constant, Inverse/Difficulty-aware, and Permuted allocation
  controls under the final P4 schedule, or narrow the actionability claim.
- Preserve the measured parameter, latency, throughput, and memory table, while
  labeling it as a single-GPU microbenchmark rather than a speedup claim.
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
- Run citation and numeric-integrity checks after this integration and again
  after any final-schedule allocation controls.

## Do not claim yet

- State-of-the-art performance or comprehensive superiority.
- Statistical significance from three seeds.
- Real-world or inference-time robustness.
- General reliability estimation beyond the tested synthetic corruption
  family.
- Untouched-holdout confirmation from MOSI.
