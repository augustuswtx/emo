# TAC first-paper submission status

Updated: 2026-10-09. Target: submit the first of two thesis papers by 2026-10-31.
Working venue: IEEE Transactions on Affective Computing (TAC), CCF-B AI journal.
This is a working target, not a claim of acceptance or current readiness.

## Manuscript and evidence

- `main.tex` now states the conservative MOSEI/C1/C2 result and the first paper's
  audit question. `body.tex` has been synchronized from the more recent NCA
  revision; the old MOSI-only TAC draft is no longer the factual baseline.
- Existing three-seed MOSEI Repaired/Constant/Learned matrix, C1, three-seed
  corruption and efficiency summaries, and C2 seed-1111 controls are included.
- C4 Per-sample-only is unrun. C2 Permuted/Inverse remain single-seed, so the
  manuscript cannot claim that positive score assignment is validated.
- C1's negative-KL variable is a teacher--student agreement proxy, not direct
  ground truth for teacher correctness; the current text now states this limit.
- P6/P7 results are transcribed in the experiment log, but their four raw/summary
  JSON files have not yet been archived locally or incorporated into this body.
- C5 zero-ablation is a post-hoc single-seed exploratory result; it does not
  demonstrate an advantage in high-sensitivity strata.
- MOSI is development evidence; the MOSEI test has since been inspected in C2/C5
  and should not be relabeled as an untouched confirmation set.

## Checks completed locally

- The [IEEE Computer Society author guide](https://www.computer.org/publications/author-resources)
  gives 100--200 words for a regular-paper abstract and a 12-formatted-page
  regular-paper baseline for Transactions (including references/biographies);
  journal-specific submission limits may differ and still need verification.
- The abstract has been shortened to fit the general 200-word ceiling and has
  no mathematical notation or references. The body is still a long draft;
  actual formatted page count and any necessary compression remain open.
- All 31 citation keys used by the current TAC source exist in the shared BibTeX.
- All five referenced figure PDFs exist at their relative paths.
- `git diff --check` passed for the source edits. No local TeX engine is installed;
  the integrated IEEEtran manuscript has **not** been compiled, visually
  inspected, or page-counted.

## Required before submission

1. Archive and cross-check P6/P7 raw JSON and C2 source logs/checkpoint hashes.
2. Run C4 and, if feasible under the fixed protocol, C2 matched controls on
   seeds 1112/1113. Preserve all outcomes, including negative ones.
3. Make the paper's claim-evidence table explicit; update abstract, results and
   limitations after the final frozen experiment table is known.
4. Verify the current TAC author information: article type, submission page
   limit, template, review anonymity, declarations, supplemental policy, and
   overlength charges. The general IEEE Computer Society guide is not a
   substitute for the TAC-specific author page.
5. Compile the final source, inspect every page, resolve citations and floats,
   and check the PDF, figures and supplementary files for accurate metadata.
6. Obtain final author order, affiliations, corresponding author, funding,
   contributions, disclosure text and coauthor approval.

The scientific go/no-go and daily schedule are in
`../../docs/two-paper-thesis-and-october-2026-submission.md`. A submission in
October is the target; a publication or acceptance date is not under author
control.
