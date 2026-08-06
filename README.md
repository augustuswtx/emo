# EMO: Reliable Multimodal Sentiment Analysis

This repository preserves the reproducible research context for an MFON-based multimodal sentiment analysis project.

## Start Here

Read [`PROJECT-CONTEXT-LATEST.md`](PROJECT-CONTEXT-LATEST.md) first. It records the current method, verified results, integrity boundaries, server paths, and exact next experiment.

## Contents

- `docs/`: experiment log, evidence plan, manuscript draft, references, literature notes, and server handoffs.
- `MFON/`: current experiment source code and CPU contract tests.
- `PROJECT-CONTEXT-LATEST.md`: single-file handoff for a new Codex conversation.

## Current Status

The P2 stage evaluates whether learned modality-reliability scores improve fixed-budget auxiliary supervision compared with constant, reversed, permuted, and Oracle controls. MOSI seed 1111 results for learned, constant, reversed, and permuted are recorded. Oracle is the next experiment.

## Data And Integrity

This public repository intentionally excludes datasets, pretrained weights, checkpoints, large logs, credentials, personal attachments, and generated presentation files. Reported numbers are limited to experiments actually run by the author. MFON is prior work (COLING 2025); this project does not claim ownership of the MFON base architecture.
