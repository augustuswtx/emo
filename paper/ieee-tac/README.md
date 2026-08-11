# IEEE Transactions on Affective Computing working draft

This directory preserves the long-form IEEE Transactions on Affective
Computing (TAC) backup draft. Neural Computing and Applications is now the
working CCF-C primary target.

## Files

- `main.tex`: IEEEtran wrapper, title, abstract, keywords, and bibliography.
- `body.tex`: evidence-grounded manuscript body converted from
  `../../docs/small-paper-draft-v2-en.md`.
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

CMU-MOSI results are frozen. CMU-MOSEI is an active experiment and is described
only as future or ongoing validation. Do not add a MOSEI result until its
comparison gate is complete and the value is copied from the frozen experiment
record.
