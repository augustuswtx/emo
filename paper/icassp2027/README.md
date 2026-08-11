# ICASSP 2027 backup draft

This folder contains a compact backup version of the longer evidence manuscript in `docs/small-paper-draft-v2-en.md`. It is no longer the primary venue route: because the project is not deadline-driven, IEEE Transactions on Affective Computing is the working primary target and the long manuscript should be strengthened before journal formatting.

## Backup venue

- Venue: 2027 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2027)
- Paper type: Regular conference paper
- Primary area: Multimedia Signal Processing
- Secondary area: Machine Learning for Signal Processing
- Submission deadline: 16 September 2026
- Length: at most four pages containing technical content; an optional fifth page may contain references only
- Review: single-anonymous; author names must be included

Official sources checked on 2026-08-11:

- ICASSP 2027 call: https://2027.ieeeicassp.org/call-for-papers/
- Publishing and page policy: https://2027.ieeeicassp.org/publishing-and-paper-presentation-options/
- Editorial and review policy: https://2027.ieeeicassp.org/about/editorial-policies/
- Current CCF graphics and multimedia directory: https://www.ccf.org.cn/Academic_Evaluation/CGAndMT/

## Files

- `main.tex`: compact ICASSP manuscript derived from the frozen MOSI evidence
- `../../docs/small-paper-references.bib`: shared verified bibliography
- `submission-status.md`: venue and readiness checklist

## Build

The official ICASSP 2027 paper kit was not linked from the conference author pages when this folder was created. Download the official 2027 kit when released and place `spconf.sty` and `IEEEbib.bst` in this directory, then run:

```sh
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Do not substitute an old paper kit for the final submission. The current machine does not have a TeX engine, so page count and font embedding remain unchecked.

## Evidence boundary

The draft reports only completed three-seed MOSI results and frozen reliability audits. MOSEI is explicitly marked as ongoing. Replace the author placeholders and revisit the acknowledgment/disclosure wording with all authors before submission.
