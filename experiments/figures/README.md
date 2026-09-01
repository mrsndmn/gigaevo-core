# Figures

Summary cards for the vocabulary-factorization programme, 26–30 Aug 2026.

| file | what it shows |
|---|---|
| `vocab-factorization-01-gates.svg` | the four pre-registered gates and their verdicts |
| `vocab-factorization-02-shrink-curve.svg` | clean BPB and HellaSwag against compression, on one shared axis |
| `vocab-factorization-03-specified-vs-searched.svg` | the shrink result against the killed tree grid |

Every number is traceable to a result JSON under the artifacts root and to
`experiments/vocab-factorization.md`. The `.png` files are renders of the `.svg`
sources, kept so the cards preview in contexts that do not render SVG.

Card 2 uses two stacked panels rather than one plot with two y-axes: BPB and
accuracy are different measures, and a dual axis lets the author choose the
crossing point. The point of the card is that the two measures *disagree*, which
a dual axis would make unfalsifiable.
