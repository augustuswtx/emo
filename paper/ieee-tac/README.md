# IEEE Transactions on Affective Computing working draft

As of 2026-10-09, TAC is the provisional CCF-B target for the first of two
thesis papers, with an October submission goal. The manuscript text now
incorporates the newer NCA revision's MOSEI and C1/C2 evidence, but still
requires source verification, additional experiments, compilation and author
metadata. This TAC draft is **not yet submission-ready**. See the
[October plan](../../docs/two-paper-thesis-and-october-2026-submission.md).

## Files

- `main.tex`: IEEEtran wrapper, title, abstract, keywords, and bibliography.
- `body.tex`: current evidence-grounded body synchronized from the NCA revision.
- `submission-status.md`: unresolved evidence, format, and authorship checks.

## Build

From this directory, a standard TeX installation with `IEEEtran` can build the
draft with:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The bibliography is intentionally shared with the long-form Markdown draft.
No TeX engine is currently available in this local workspace, so the generated
PDF and page count have not yet been verified.

## Scope boundary

The newer abstract/body state the completed MOSEI results, negative C1 audit,
and seed-1111 C2 controls conservatively. C4, extended C2, P6/P7 raw archives,
venue-specific formatting and author declarations remain open.
