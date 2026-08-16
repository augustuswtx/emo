# F1 visual and scientific-boundary QA

Date: 2026-08-16

## Figure contract

- Core conclusion: reliability redistributes training-time auxiliary
  supervision under an exact batch budget; it does not gate the clean MFON
  inference path.
- Archetype: schematic-led composite.
- Backend: Python/matplotlib only.
- Final size: 183 x 112 mm, double-column placement.
- Empirical values: none.

## Checks

| Check | Result |
|---|---|
| Clean text/vision/audio path reaches MFON fusion and prediction | Pass |
| P4 task path uses clean features | Pass; directly labelled `clean only` |
| Ordered clean--mild--strong interventions are training-only | Pass; contained in dashed training band |
| Visual and acoustic reliability remain sample-wise | Pass; `q_i^v, q_i^a` directly labelled |
| KL and InfoNCE remain unreduced before weighting | Pass |
| Frozen unimodal teachers and their clean input are visible | Pass |
| Exact finite-batch mean budget is stated | Pass; equation shown |
| Reliability has no edge to MFON fusion | Pass; no edge and explicit callout |
| No inference-time dynamic reliability weighting is implied | Pass; explicit callout |
| MOSEI smoke or unfinished results appear | Pass; no empirical results appear |
| SVG text remains editable | Pass; SVG uses text elements (`svg.fonttype=none`) |
| PDF is vector and page size is correct | Pass; 518.4 x 316.8 pt (183 x 112 mm nominal) |
| PNG preview resolution | Pass; 2160 x 1320 at 300 dpi |
| English and Chinese labels fit without overlap | Pass after full-resolution visual review |
| Colour remains redundant with borders, direct labels, and line style | Pass |

## Rebuild command

```bash
python3 paper/figures/scripts/make_f1_method_overview.py
```
