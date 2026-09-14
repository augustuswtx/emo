# Neural Computing and Applications working draft

This directory contains the current primary CCF-C journal working draft for
Neural Computing and Applications (NCA).

## Files

- `main.tex`: double-blind Springer Nature wrapper, abstract, keywords, and
  declarations.
- `body.tex`: evidence-grounded manuscript body shared in content with the
  long-form English draft.
- `sn-jnl.cls` and `sn-basic.bst`: official Springer Nature template files
  from the December 2024 author package.
- `build_submission_package.py`: creates a flat, anonymous editable-source
  archive without `\\input` dependencies.
- `title-page-template.tex`: separate identity-bearing title-page template;
  do not include it in the anonymous review archive.
- `submission-status.md`: NCA-specific readiness ledger.

## Template and build status

NCA accepts manuscripts with mathematical content in LaTeX and recommends the
Springer Nature LaTeX template. The official December 2024 `sn-jnl` class and
numbered `sn-basic` bibliography style are vendored here. Build the anonymous
editable-source package first:

```sh
python3 build_submission_package.py
```

This writes `build/anonymous-manuscript/` and
`build/nca-anonymous-manuscript.zip`. The flattened package can then be built
with the standard Springer Nature sequence:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

An isolated Tectonic 0.17 preflight produced the 23-page A4 PDF at
`../../output/pdf/nca_anonymous_draft.pdf`. The final log has no LaTeX errors,
undefined citations or references, missing files, or overfull boxes. All pages
were rasterized and visually checked; no clipping, overlap, or missing glyphs
were found. Tectonic was used only as a local preflight. Repeat the standard
`pdflatex`/BibTeX build in the submission package or Editorial Manager before
uploading.

## Double-blind boundary

The review manuscript must not contain author names, affiliations,
acknowledgments, funding details, repository identities, or identifying links.
Prepare those items in a separate title page. The current placeholders are not
submission-ready and must be handled during final packaging.

CMU-MOSI is development-stage exploratory evidence because test-split
reliability diagnostics informed reliability-head design. The P4 protocol was
then frozen for CMU-MOSEI. Its three-seed matched comparison, full-test Gaussian
audit, and four held-out-corruption stress audits are now complete. The remaining
empirical gate is, if resources permit, the final-schedule
inverse/difficulty-aware and permutation control. The uniform efficiency
microbenchmark is complete and is reported as a single-GPU cost audit rather
than a general speed comparison. The bilingual F1--F5 figure package is also
complete, with quantitative source CSV files and editable SVG/PDF exports.
