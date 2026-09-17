# NCA submission-status ledger

Updated: 2026-09-17

## Current state

- Working venue: Neural Computing and Applications (CCF-C in the current CCF
  artificial-intelligence directory).
- Article type: original research article.
- Review model: double-blind.
- Format: official December 2024 Springer Nature `sn-jnl` class with numbered
  `sn-basic` citations; class and bibliography style are vendored locally.
- Major-revision state: reviewer score 5/10 (Major Revision). Language,
  terminology, literature positioning, paired visualization, implementation
  hooks, and experiment planning have been revised. C1 is complete with a
  negative cross-sample target-fidelity result. C2 has matched seed-1111
  Inverse and Permuted tests, but its majority-of-seeds criterion is open;
  C3--C4 remain open or conditional. C5 has a matched seed-1111 full-test
  zero-ablation sensitivity audit, with no Learned advantage in either
  high-sensitivity modality stratum; seeds 1112--1113 remain open. The scientific score should not be
  treated as improved yet.
- Abstract: 216 words, within the working 150--250-word target.
- Keywords: five, within the requested 4--6 range.
- Compilation: the last isolated Tectonic 0.17 preflight, before the C1 result
  integration, produced a 25-page A4 anonymous draft without errors, missing
  files, undefined citations/references, or overfull boxes. The current C1
  and seed-1111 C2 revision has passed source/package checks but still requires a fresh TeX
  compile and raster review on a host with a TeX engine.
- Packaging: the manuscript packager produces one flattened `main.tex`, the
  BibTeX database, official class/style files, and five English PDF figures.
  A separate anonymous code-artifact packager includes implementation, tests,
  audit programs, figure source values, and figure scripts while excluding
  datasets, pretrained files, checkpoints, logs, credentials, and git history.
- Anonymity: manuscript and code-artifact marker scans passed; PDF metadata has
  no author/title metadata, custom metadata, forms, or JavaScript.
- Evidence designation: MOSI is exploratory/development evidence because
  test-split reliability diagnostics informed head design. The frozen MOSEI
  protocol is the frozen-protocol cross-dataset replication.
- Existing evidence complete: all three fixed MOSEI seeds for repaired MFON,
  P4 Constant, and P4 Learned; full-test Gaussian reliability audits; and four
  held-out corruption audits; plus a matched single-GPU efficiency audit. The
  previously deployed extended implementation suite passes 37 tests. The five
  cross-sample-audit tests pass in the server PyTorch environment, and full
  validation audits are complete for all three frozen Learned checkpoints.
- Figure package complete: bilingual method overview, frozen MOSEI task
  comparison, Gaussian reliability/confound audit, held-out corruption audit,
  and efficiency audit. Fig. 2 now displays paired seed-level trajectories and
  per-seed favorable-direction deltas; each quantitative figure has standalone
  CSV source data and editable SVG/PDF exports.
- Frozen outcome hierarchy, fixed before the P4 Constant formal test:
  MAE/Corr primary, Has0/Non0 accuracy/F1 secondary, and Acc-5/7, loss, and
  reliability/confound measures diagnostic.

## Evidence gates

- Preserve the completed post-hoc C1 validation audit: visual score--KL
  fidelity is consistently reversed and acoustic association is weak. Do not
  infer clean-sample target fidelity from within-sample ordinal ranking or
  Gaussian AUROC, and do not add an unnecessary test-split C1 analysis.

- Preserve the complete three-seed MOSEI table and the explicit finding that
  Learned improves the primary endpoints over Constant but not uniformly over
  repaired MFON.
- Preserve the explicit MOSI development-history disclosure in the Abstract,
  Experimental Protocol, Limitations, and Conclusion.
- Preserve the MOSEI acoustic-length confound and the negative cross-corruption
  results; do not describe the reliability heads as general quality estimators.
- Add original-media acoustic and visual corruptions beyond feature-level
  Gaussian/non-Gaussian stress tests if deployment robustness is claimed.
- Report the completed seed-1111 final-schedule Constant, Learned,
  Inverse/Difficulty-aware, and Permuted controls as single-seed evidence;
  replicate the latter two on other seeds before claiming allocation direction
  or narrow the actionability claim.
- Preserve the measured parameter, latency, throughput, and memory table, while
  labeling it as a single-GPU microbenchmark rather than a speedup claim.
- Preserve the method, main-result, reliability, cross-corruption, and
  efficiency figures together with their claim-boundary captions and source
  CSV files.

## Submission-package gates

- Recompile the flattened package with a conventional `pdflatex`/BibTeX TeX
  Live environment or Editorial Manager as the final engine check.
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

## Major-revision artifacts

- Reviewer report: `ccfa-review-reports/2026-09-14-nca-anonymous-draft-pdf-full-review.md`
- Point-by-point internal response: `docs/major-revision-response-20260915.md`
- Executable C1--C5 experiment plan: `docs/major-revision-experiment-plan-20260915.md`
- Read-only C1 audit: `MFON/audit_cross_sample_validity.py`
- Anonymous artifact builder: `paper/springer-nca/build_anonymous_artifact.py`

## Do not claim yet

- State-of-the-art performance or comprehensive superiority.
- Statistical significance from three seeds.
- Real-world or inference-time robustness.
- General reliability estimation beyond the tested synthetic corruption
  family.
- Cross-sample auxiliary-target fidelity of the learned scores.
- Untouched-holdout confirmation from MOSI.
