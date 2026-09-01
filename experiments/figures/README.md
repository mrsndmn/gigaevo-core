# Figures

Summary cards for the vocabulary-factorization programme, 26–30 Aug 2026.
**These cover the methods that worked**; the tracks that were killed live in
`experiments/vocab-factorization.md` with their kill numbers.

| file | what it shows |
|---|---|
| `vocab-factorization-01-result.svg` | the headline: a 4.5× smaller table below native BPB, and the three rules |
| `vocab-factorization-02-curve.svg` | BPB and HellaSwag against compression, two panels on one shared axis |
| `vocab-factorization-03-case-folding.svg` | why folding case at tokenization beats merging token ids |

Every number is traceable to a result JSON under the artifacts root and to
`experiments/vocab-factorization.md`. The `.png` files are renders of the `.svg`
sources, kept so the cards preview where SVG is not rendered.

Two deliberate choices, both about not overstating:

- **Every compression claim carries its HellaSwag number.** This programme's own
  hardening gate established that corpus BPB does not imply downstream ability —
  one arm moved BPB by 0.0165 while losing 25.7 accuracy points. A compression
  figure quoted without the downstream figure is exactly the claim that gate
  exists to catch, so the cards never separate them.
- **Card 2 uses two stacked panels, not one plot with two y-axes.** BPB and
  accuracy are different measures; a dual axis would let the author choose where
  they cross, which is the thing the reader needs to judge for themselves.
