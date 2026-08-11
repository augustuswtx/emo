# Neural Computing and Applications working draft

This directory contains the current primary CCF-C journal working draft for
Neural Computing and Applications (NCA).

## Files

- `main.tex`: double-blind Springer Nature wrapper, abstract, keywords, and
  declarations.
- `body.tex`: evidence-grounded manuscript body shared in content with the
  long-form English draft.
- `submission-status.md`: NCA-specific readiness ledger.

## Template and build status

NCA accepts manuscripts with mathematical content in LaTeX and recommends the
Springer Nature LaTeX template. Download the latest official template before
submission and place its `sn-jnl.cls` and required bibliography files in this
directory. A typical build is:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

This workspace currently has neither the official class package nor a TeX
engine, so the draft has not been compiled or visually inspected.

## Double-blind boundary

The review manuscript must not contain author names, affiliations,
acknowledgments, funding details, repository identities, or identifying links.
Prepare those items in a separate title page. The current placeholders are not
submission-ready and must be handled during final packaging.

CMU-MOSI results are frozen. CMU-MOSEI remains an active experiment and must
not be reported as complete until its experiment gates have passed.
