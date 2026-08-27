# Vocabulary factorization: smaller vocab, surface-robust LM

## Hypothesis

A large BPE vocabulary spends most of its entries on *surface variants* of the
same lexical item: `token` / ` token` / `Token` / `TOKEN`, and in Russian a
long tail of inflectional endings. If those axes are pulled out of the
vocabulary and modelled as separate low-cardinality prediction heads, then at a
matched vocabulary size the factorized tokenizer should

1. compress better (fewer tokens per byte, fewer pieces per word),
2. be nearly immune to whole-text case transforms, and
3. keep bits-per-byte competitive with a BPE twice its size.

Typos are a *separate* mechanism and are not expected to be helped by the
factorization: a corrupted stem is still an unknown string. The character
n-gram bag on the input embedding is the arm that targets those, and it is
ablated independently so the two claims never get conflated.

## Metric contract

Two rules make the comparison meaningful; both are enforced in code, not
assumed.

**Bits per byte, never per-token perplexity.** Per-token PPL falls mechanically
as a vocabulary shrinks and sequences lengthen, so it cannot compare arms.
`BPB = total_NLL_nats / ln2 / bytes_scored`, and the sum runs over *every*
head: an arm that moves case into a side head still pays to predict it
(`train.py:eval_bpb`).

**Byte-exact invertibility as a hard gate.** A scheme that casefolds without
predicting case back would win on BPB by discarding information. Every arm must
satisfy `decode(encode(x)) == x` on clean *and* perturbed text, including
`randcase`, which is what exercises the fallback path. Words whose casing is
not restorable from the lowercase form (`iPhone`, German `STRAßE`, Turkish
dotted i) are detected by verification rather than by rule and kept verbatim
(`surface.py:classify_case`).

## Setup

Everything is defined in `experiments/vocab_factorization/configs.py`
(`ARMS`, `EXPERIMENTS`); this file deliberately duplicates none of it.

- Corpus: ru+en Wikipedia, NFC-normalized once at build time, frozen bytes
  thereafter. Document-hash split. `data.py`.
- Tokenizer arms: byte-level BPE at 32k/16k/8k (the 16k and 8k rows are the
  null hypothesis — "just train a smaller BPE"), raw bytes, and the factorized
  arm at 16k/8k plus a no-suffix ablation. `tokenization.py`.
- Model: decoder-only, pre-LN, RoPE, one softmax head per feature, optionally
  conditioned on the stem being predicted so the factorization is
  `P(stem|h)·P(role|h,stem)·P(suffix|h,stem)` rather than an independence
  assumption. `model.py`.
- Protocol: **data** is held fixed across arms (same bytes, same epochs);
  **compute** is therefore not — the byte arm emits ~5x the tokens and costs
  ~5x. Each run records its own steps/tokens/wall-clock.
- Perturbation suites: keyboard-adjacent typos at 1/5/10%, and `lower`,
  `upper`, `title`, `randcase`. `perturb.py`.

Two-stage evaluation: a training-free screen (`screen.py`, seconds per arm)
ranks segmentation quality before anything touches the GPU; only then does the
training grid run (`train.py`).

Known confound, reported rather than hidden: sequence length is fixed at 1024
*tokens* for all arms, so a finer tokenizer sees fewer bytes of context. The
`context_bytes` column records how much.

## Results

Training-free screen (ru+en valid, 3.05 MB) — all arms lossless on all suites:

| arm | vocab | tok/byte | pieces/word | infl UPPER | infl typo5 |
|---|---|---|---|---|---|
| bpe32k | 32000 | 0.1751 | 1.542 | 2.769 | 1.192 |
| bpe16k | 16000 | 0.1961 | 1.804 | 2.706 | 1.160 |
| bpe8k | 8000 | 0.2231 | 2.146 | 2.595 | 1.125 |
| byte | 256 | 1.0000 | 13.568 | 1.000 | 1.000 |
| fact16k | 16000 | 0.1866 | 1.436 | 0.999 | 1.178 |
| fact8k | 8000 | 0.2160 | 1.770 | 0.999 | 1.135 |
| fact16k-nosuf | 16000 | 0.2136 | 1.820 | 0.999 | 1.113 |

Reading: at matched 16k vocabulary the factorized arm needs 20% fewer pieces
per word than BPE, and beats even the 32k BPE on pieces-per-word (1.436 vs
1.542) with half the entries. Uppercasing inflates BPE token counts by ~2.7x
and leaves the factorized arms untouched. Typo inflation is essentially
unchanged, exactly as predicted — the factorization does not address typos.

Training grid, 10 runs, 33M-param LM, identical 400 MB x 3 epochs per arm
(full tables via `python -m experiments.vocab_factorization.report`):

| experiment | BPB clean | d upper | d title | d typo5 | embed params |
|---|---|---|---|---|---|
| bpe32k | 1.2039 | 2.076 | 0.658 | 0.707 | 16,384,000 |
| bpe16k | 1.1919 | 1.977 | 0.662 | 0.714 | 8,192,000 |
| bpe8k | 1.1897 | 1.949 | 0.672 | 0.716 | 4,096,000 |
| **fact8k-nosuf** | **1.2068** | **0.833** | **0.335** | 0.750 | **4,104,192** |
| fact16k-nosuf | 1.2137 | 0.723 | 0.311 | 0.749 | 8,200,192 |
| fact16k | 1.4610 | 0.711 | 0.303 | 0.676 | 8,463,360 |
| fact8k | 1.3911 | 0.820 | 0.329 | 0.698 | 4,367,360 |
| fact16k-ngram | 1.4565 | 0.705 | 0.305 | 0.679 | 9,020,416 |
| bpe16k-ngram | 1.2143 | 2.095 | 0.716 | 0.706 | 8,749,056 |
| byte | 1.1110 | 1.900 | 0.553 | 0.771 | 131,072 |

**1. Vocabulary size barely moves BPB at this scale — the null hypothesis
wins.** bpe8k (1.1897) ≤ bpe16k (1.1919) ≤ bpe32k (1.2039). Shrinking 32k → 8k
*improves* BPB slightly while saving 12.3M parameters. No new mechanism is
needed to justify a smaller vocabulary at 33M params; the interesting question
is what else the smaller vocabulary can buy.

**2. Case+space factorization buys large robustness for ~1.4% BPB.**
`fact8k-nosuf` costs +0.017 BPB over bpe8k and cuts case damage 2.3x
(d upper 0.83 vs 1.95, d title 0.34 vs 0.67, d lower 0.12 vs 0.23). Against
the bpe32k reference it is a wash on BPB (1.2068 vs 1.2039, +0.24%) with a 4x
smaller embedding table and 2.5x less case damage.

**3. Tokenizer-level robustness is not model-level robustness.** The byte arm
has perfect token-count stability under case transforms (screen inflation
1.000) yet its model still pays d upper = 1.90, nearly as much as BPE. Only
*explicitly factoring case out of the prediction target* helps. The plan's
premise that bytes are the robustness ceiling was wrong for case.

**4. The morphology head is a clear net loss.** fact16k (1.4610) vs
fact16k-nosuf (1.2137): +0.247 BPB, and it buys no robustness — the case
deltas are unchanged. Diagnosis: the factorization itself is exact (chain
rule), so this is a conditioning deficiency, not an information-theoretic one.
Factorizing moved only 0.074 bits/byte out of the stem head but charged 0.285
bits/byte on the suffix head — a ~4x inefficiency. The reason is structural:
BPE predicts a word's ending on a *later position*, with a full transformer
forward pass over the stem token; the suffix head predicts it from the same
hidden state via a linear map on `h + E_stem[target]`. That function class is
far too weak.

**5. The character n-gram bag does nothing.** fact16k-ngram vs fact16k:
-0.005 BPB clean, d typo5 0.679 vs 0.676 — noise. On BPE it actively hurt
(1.2143 vs 1.1919). Diagnosis is as much about the metric as the mechanism:
d BPB under typos is dominated by the *output* side (the model must emit the
corrupted string), and an input-side embedding change cannot reduce that cost.
The experiment could not have detected a working input-side mechanism.

**6. Mixed-case noise defeats every arm** (d randcase ~1.8-2.1 everywhere,
bytes included), because words whose casing is not restorable fall to the
verbatim path by design.

**7. Typo deltas are within noise across all ten arms** (0.15-0.17 at 1%,
0.68-0.77 at 5%). Nothing tested here helps typos.

## Conclusions

The case/whitespace half of the hypothesis holds and is the practical result:
**a 4x smaller vocabulary at parity BPB with 2.5x less case damage**
(`fact8k-nosuf` vs `bpe32k`). It is cheap, exactly invertible, and the
mechanism is a role head with 8 values.

The morphology half does not hold as implemented, and the n-gram/typo half was
not testable with the chosen metric. Both are fixable and neither is refuted in
principle:

- *Suffix head*: give it real capacity (MLP over `[h; E_stem]`, or low-rank
  bilinear), or emit the suffix as its own sequence position so it gets a full
  forward pass like BPE's ending token does. The current linear head is the
  prime suspect and the cheapest thing to test.
- *Typo robustness*: stop measuring it as d BPB on corrupted text. Measure BPB
  of a **clean continuation given a corrupted context**, which isolates whether
  typos damage comprehension rather than charging the model for reproducing
  noise. Add a downstream task under perturbation.
- *Mixed case*: the verbatim fallback is the bottleneck. A per-character case
  mask head would cover `rAnDoM` at the cost of one more cheap head.

Scale caveat: everything here is one seed at 33M parameters on 400 MB. Embedding
tables are ~25-40% of these models, so vocabulary effects are inflated relative
to a 1B+ model; differences under ~0.02 BPB should not be trusted, and the
vocab-size ranking in finding 1 may well invert at larger scale.

Protocol caveat: the byte arm's 1.1110 is not a like-for-like win. Under the
fixed-data protocol it received 5x the tokens and 5x the compute of every other
arm, and it sees 1024 bytes of context against ~4,100-5,500 for the rest.

## Changelog

- 2026-08-26 — Built harness: corpus prep, surface factorization with verified
  invertibility, 7 tokenizer arms, perturbation suites, training-free screen,
  factorized-softmax LM, 9-run ablation grid. Screen results in.
- 2026-08-27 — Full training grid done (10 runs, ~3.5 GPU-hours on one A100).
  Case+space factorization confirmed: 4x smaller vocab at parity BPB with 2.5x
  less case damage. Suffix head and n-gram bag are negative results; added
  `fact8k-nosuf` after the first 9 runs showed it was the configuration the
  data pointed at. Figure: `results/vocab_factorization.png`.
