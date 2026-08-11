# TAC submission-status ledger

Updated: 2026-08-11

## Current state

- Backup venue: IEEE Transactions on Affective Computing (CCF-B in the current
  CCF artificial-intelligence directory). Neural Computing and Applications is
  now the working CCF-C primary target.
- Format: standard `IEEEtran` journal working format; TAC-specific submission
  requirements still need a final official-policy check.
- Manuscript: complete MOSI-based long-form draft with conservative claims.
- Compilation: not yet compiled locally because no TeX engine is installed.
- Authorship: author names, affiliations, corresponding author, funding, and
  acknowledgments are placeholders.

## Evidence gates before submission

- Complete the frozen CMU-MOSEI comparison without tuning P4 from test results.
- Recheck the acoustic length confound on MOSEI.
- Add realistic acoustic and visual corruptions beyond Gaussian feature noise.
- Repeat the key allocation controls under the final P4 schedule, or keep the
  actionability claim explicitly diagnostic.
- Report parameter count, training and inference time, and memory.
- Add a method figure, corruption/audit figure, and claim-aligned ablation table.
- Decide whether three seeds remain descriptive or run enough seeds for a
  justified inferential analysis.

## Manuscript checks before submission

- Replace all author and acknowledgment placeholders.
- Confirm TAC article type, anonymity policy, page limits, overlength charges,
  supplementary-material rules, and required AI-use disclosure from current
  official pages.
- Compile with the official template and inspect every equation, table, float,
  citation, font, and page break.
- Re-run reference verification, especially the currently provisional
  `moon2026quality` entry.
- Remove all statements about experiments being “currently running” and replace
  them with frozen outcomes or an explicit omission.
- Ensure the abstract, conclusion, tables, and supplementary material report the
  same numeric precision and claim boundaries.

## Do not claim yet

- State-of-the-art performance.
- Statistical significance from the current three seeds.
- Uniform improvement over repaired MFON.
- Real-world or inference-time robustness.
- General reliability estimation beyond the tested synthetic corruption family.
