# Venue strategy

Updated: 2026-08-11

## Primary target: Neural Computing and Applications

Neural Computing and Applications (NCA) is the working primary target. The
current CCF artificial-intelligence directory lists NCA as a C-tier journal.
Its neural-computing and application orientation accommodates the present
MFON-based multimodal sentiment analysis study, and the journal has published
work on multimodal sentiment analysis. This route is more conservative than
the previous IEEE Transactions on Affective Computing target while preserving
the paper's core technical story.

The intended story remains: modality-quality signals should be audited before
they are trusted, and fixed-budget auxiliary allocation isolates sample
assignment from changes in the total amount of supervision. The paper is not
positioned as an image model, a universal performance improvement, or a
demonstrated inference-time robust-fusion system.

Before submission, the following evidence gates should be resolved:

1. Complete the frozen MOSEI comparison without changing P4 based on test
   outcomes.
2. Recheck the audio-length confound on MOSEI.
3. Add at least one realistic acoustic and one realistic visual corruption.
4. Repeat the key Constant, Permuted, and Reversed controls under the final P4
   schedule, or keep the actionability claim explicitly diagnostic.
5. Add parameter count, training time, inference time, and memory reporting.
6. Add a method figure, corruption/audit figure, and claim-aligned ablation
   table.

## Working format

NCA currently uses double-blind review. The review manuscript and all
supplementary materials must omit author-identifying information; author names,
affiliations, acknowledgments, funding, and contact details belong on a
separate title page. The journal accepts Word and manuscripts with mathematical
content in the Springer Nature LaTeX template. The usual abstract range is
150--250 words and 4--6 keywords are requested. The current 209-word abstract
and five keywords satisfy those working constraints.

The `paper/springer-nca/` directory uses the generic Springer Nature `sn-jnl`
structure as a working draft. Before submission, download the latest official
template package and recheck all policies; the local workspace does not yet
contain `sn-jnl.cls` or a TeX engine.

## Backups

- IEEE Transactions on Affective Computing (CCF-B): preserved as a higher-risk
  future target if the full cross-dataset and robustness evidence becomes much
  stronger.
- ICASSP: preserved as a compact conference backup, but its four-page technical
  content limit removes too much audit evidence.
- Knowledge-Based Systems and Neurocomputing: both appear in the current CCF-C
  artificial-intelligence list, but their framing is less directly aligned than
  NCA with the present neural auxiliary-learning study.

## Policy freshness

Official sources checked on 2026-08-11:

- CCF artificial-intelligence directory: https://www.ccf.org.cn/Academic_Evaluation/AI/
- NCA submission guidelines: https://link.springer.com/journal/521/submission-guidelines
- Springer Nature LaTeX author support: https://www.springernature.com/gp/authors/campaigns/latex-author-support

CCF classification, editorial policy, templates, declarations, and submission
requirements must be checked again immediately before submission.
