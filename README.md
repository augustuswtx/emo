# EMO: Reliable Multimodal Sentiment Analysis

This repository preserves the reproducible research context for an MFON-based multimodal sentiment analysis project.

## Start Here

For the completed MOSEI matrix and the remaining experiment gates, read [`MOSEI-HANDOFF-LATEST.md`](MOSEI-HANDOFF-LATEST.md) first. Then read [`PROJECT-CONTEXT-LATEST.md`](PROJECT-CONTEXT-LATEST.md) for the complete method and MOSI evidence history.

## Contents

- `docs/small-paper-draft-v2-en.md`: current evidence-grounded English manuscript draft.
- `docs/small-paper-draft-v2-zh.md`: complete Chinese counterpart aligned with the frozen evidence and claim boundaries.
- `docs/small-paper-references.bib`: 27-entry bibliography used by the current draft.
- `docs/reference-verification-20260811.md`: field-level citation verification ledger and remaining caveats.
- `docs/literature-search-20260811-audited-multimodal-reliability/`: screened literature report, evidence table, and reproducible search notes.
- `docs/small-paper-draft-v1.md`: preserved Chinese historical draft from the earlier method stage.
- `paper/VENUE-STRATEGY.md`: current venue decision, with Neural Computing and Applications as the working CCF-C primary target.
- `paper/springer-nca/`: double-blind Springer Nature working draft and NCA submission-readiness ledger.
- `paper/ieee-tac/`: preserved higher-risk IEEE TAC backup draft.
- `paper/icassp2027/`: preserved compact ICASSP 2027 backup draft and venue-readiness checklist.
- `docs/`: experiment log, evidence plan, references, literature notes, and server handoffs.
- `MFON/`: current experiment source code and CPU contract tests.
- `PROJECT-CONTEXT-LATEST.md`: single-file handoff for a new Codex conversation.
- `MOSEI-HANDOFF-LATEST.md`: concise live handoff for the current MOSEI server experiments.

## Current Status

The MOSI three-seed study is complete and explicitly exploratory. The frozen MOSEI confirmation is also complete for repaired MFON, P4 Constant, and P4 Learned across seeds 1111--1113. Learned improves mean MAE by 0.0025 and correlation by 0.0007 over the equal-budget Constant control, while secondary metrics are mixed and repaired MFON is not uniformly surpassed. Full-test Gaussian and four held-out-corruption reliability audits are complete; the extended server suite passes 37 tests. See `MOSEI-HANDOFF-LATEST.md` for the frozen numbers, limitations, and the remaining efficiency/actionability gates.

## Data And Integrity

This public repository intentionally excludes datasets, pretrained weights, checkpoints, large logs, credentials, personal attachments, and generated presentation files. Reported numbers are limited to experiments actually run by the author. MFON is prior work (COLING 2025); this project does not claim ownership of the MFON base architecture.
