# EMO: Reliable Multimodal Sentiment Analysis

This repository preserves the reproducible research context for an MFON-based multimodal sentiment analysis project.

## Start Here

For the active MOSEI run, read [`MOSEI-HANDOFF-LATEST.md`](MOSEI-HANDOFF-LATEST.md) first. Then read [`PROJECT-CONTEXT-LATEST.md`](PROJECT-CONTEXT-LATEST.md) for the complete method and MOSI evidence history.

## Contents

- `docs/small-paper-draft-v2-en.md`: current evidence-grounded English manuscript draft.
- `docs/small-paper-draft-v2-zh.md`: complete Chinese counterpart aligned with the frozen evidence and claim boundaries.
- `docs/small-paper-references.bib`: 27-entry bibliography used by the current draft.
- `docs/reference-verification-20260811.md`: field-level citation verification ledger and remaining caveats.
- `docs/literature-search-20260811-audited-multimodal-reliability/`: screened literature report, evidence table, and reproducible search notes.
- `docs/small-paper-draft-v1.md`: preserved Chinese historical draft from the earlier method stage.
- `paper/VENUE-STRATEGY.md`: current venue decision, with IEEE Transactions on Affective Computing as the working primary target.
- `paper/ieee-tac/`: long-form IEEE TAC working draft and submission-readiness ledger.
- `paper/icassp2027/`: preserved compact ICASSP 2027 backup draft and venue-readiness checklist.
- `docs/`: experiment log, evidence plan, references, literature notes, and server handoffs.
- `MFON/`: current experiment source code and CPU contract tests.
- `PROJECT-CONTEXT-LATEST.md`: single-file handoff for a new Codex conversation.
- `MOSEI-HANDOFF-LATEST.md`: concise live handoff for the current MOSEI server experiments.

## Current Status

The MOSI three-seed P4 comparison and final reliability audits are complete and frozen. P4 has been ported to MOSEI and passed 33 server tests. MOSEI seed 1111 unimodal encoder preparation is in progress; no MOSEI fusion result has been claimed yet.

## Data And Integrity

This public repository intentionally excludes datasets, pretrained weights, checkpoints, large logs, credentials, personal attachments, and generated presentation files. Reported numbers are limited to experiments actually run by the author. MFON is prior work (COLING 2025); this project does not claim ownership of the MFON base architecture.
