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

Training grid: pending (see Changelog).

## Conclusions

Pending the training grid. The screen already establishes that the compression
and case claims hold at the segmentation level; what the grid has to settle is
whether the factorized softmax gives that advantage back in BPB.

## Changelog

- 2026-08-26 — Built harness: corpus prep, surface factorization with verified
  invertibility, 7 tokenizer arms, perturbation suites, training-free screen,
  factorized-softmax LM, 9-run ablation grid. Screen results in.
- 2026-08-27 — Launched the full training grid on one A100.
