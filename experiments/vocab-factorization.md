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

## Where things live

- **This file** — hypothesis, method, findings, conclusions.
- **`experiments/vocab-factorization-tables.md`** — every measured number, in
  full, for all three studies. Generated; `results/` is gitignored, so that
  snapshot is what actually preserves the data. Regenerate with
  `python -m experiments.vocab_factorization.report --out
  experiments/vocab-factorization-tables.md`.
- **`experiments/vocab_factorization/configs.py`** — the grid itself
  (`ARMS`, `EXPERIMENTS`, `TRANSPLANTS`). The single source of truth.

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

## Study 2 — transplant onto a frozen Qwen3-1.7B

Directly attacks the scale caveat above. Qwen3-1.7B's transformer body is
frozen; only a new embedding table and heads train, on 241 MB of the same
corpus (~56M tokens, ~51 min/arm on one A100). Grid in `configs.TRANSPLANTS`,
runner `train_transplant.py`. Figure: `results/vocab_transplant.png`.

| arm | init | BPB step0 | BPB final | d upper | d title | d lower | d randcase | embed | total params |
|---|---|---|---|---|---|---|---|---|---|
| orig-frozen | native | 0.7125 | 0.7125 | 0.305 | 0.118 | 0.039 | 1.001 | 311.2M | 1.721B |
| orig-reinit | random | 4.9866 | 1.3318 | 2.770 | 0.534 | 0.118 | 1.680 | 311.2M | 1.721B |
| bpe8k-t-random | random | 4.3854 | 1.3130 | 1.517 | 0.486 | 0.089 | 1.532 | 16.4M | 1.426B |
| **bpe8k-t** | mean-of-pieces | 2.7078 | **0.7031** | 0.698 | 0.196 | 0.047 | 0.974 | 16.4M | 1.426B |
| fact8k-nosuf-t | mean-of-pieces | 4.1309 | 0.7390 | 0.460 | **0.073** | **0.015** | 1.796 | 16.4M | 1.426B |

**1. The transplant works, and beats the model it replaced.** `bpe8k-t` reaches
0.7031 against the untouched pretrained 0.7125 — 1.3% *better* — with a 19x
smaller embedding table, 295M fewer parameters (17% of the model), and one
GPU-hour of embedding-only training. A domain-fit 8k vocabulary is enough to
serve a frozen 1.7B body on this corpus.

**2. Initialisation dominates, and "on the fly" does not work.** `bpe8k-t`
(0.7031) vs `bpe8k-t-random` (1.3130) differ only in init: mean-of-pieces is
worth 0.61 BPB, more than any vocabulary effect measured in either study. But
zero-training transplant is not viable — the best step-0 BPB is 2.7078 against
a 0.7125 reference. The heuristic init is a starting point, not an answer.

**3. The control earned its place.** `orig-reinit` lands at 1.3318, nearly
double the pretrained 0.7125. Measured against the pretrained model alone every
transplant looks like a catastrophe; measured against the matched-init control,
`bpe8k-t-random` already *wins* with 19x fewer embedding parameters. The
pretrained embedding table is worth ~0.62 BPB that 53M tokens cannot rebuild —
and that has nothing to do with which vocabulary is used.

**4. Case factorization still buys robustness, at a higher price than at 33M.**
`fact8k-nosuf-t` vs `bpe8k-t`: +5.1% clean BPB (0.739 vs 0.703) for d upper
1.5x better, d title 2.7x better, d lower 3.1x better. Its d title (0.073) and
d lower (0.015) beat even the untouched pretrained model.

**5. Scale shrinks the headroom.** At 33M every baseline had d upper ≈ 2.0;
pretrained Qwen3-1.7B has d upper = 0.305. A large pretrained model has already
learned most of the case robustness the factorization supplies by construction.
The factorization still beats a *retrofitted* vocabulary, but no longer beats
the original model on ALL-UPPER — only on Title and lower.

**6. randcase regressed and is now the worst arm** (1.796 vs 0.974 for
`bpe8k-t`), the same verbatim-fallback limitation as study 1, and more costly
here because the rest of the field improved.

### Study 2 conclusions

For shrinking an existing model's vocabulary: transplant a domain-fit BPE with
mean-of-pieces init. 17% of parameters removed, BPB slightly *better*, about a
GPU-hour. That is the deployable result, and it needs none of the factorization.

The case factorization is a second-order effect at this scale: worth its 5% BPB
only if Title/lower robustness is specifically valuable, or if the body will be
unfrozen afterwards so the clean-BPB gap can close.

Caveats specific to study 2: one seed. Qwen3 has near-certainly trained on ru/en
Wikipedia, so `orig-frozen` is optimistic — but the body is shared across arms,
so relative comparisons hold. Loss curves were still descending slowly at 53-59M
tokens, so the gaps to the pretrained reference are upper bounds. **Only BPB was
measured**: a transplanted model's downstream task quality is untested and is
the obvious next check, since good BPB does not guarantee preserved
instruction-following. Finally, freezing the body is a strict constraint;
unfreezing LayerNorms or the top blocks is standard in transplant work and would
likely close the remaining gap.

## Study 3 — two tokenizer fixes and a metric that can see input-side effects

Round-to-round noise, measured by running `bpe8k` twice under an identical
config: **0.0016 BPB**. Differences below ~0.003 are not real.

| arm | BPB | tok/byte | context bytes | d upper | d title | d randcase |
|---|---|---|---|---|---|---|
| bpe8k | 1.1881 | 0.2374 | 4,313 | 1.937 | 0.670 | 2.051 |
| fact8k-nosuf (before) | 1.2074 | 0.2488 | 4,116 | 0.839 | 0.330 | 2.509 |
| fact8k-freqfix (fix 1) | 1.2060 | 0.2213 | 4,627 | 0.733 | 0.292 | 2.028 |
| **fact8k-v2** (fix 1+2) | 1.2056 | 0.2209 | 4,635 | **0.691** | 0.303 | 2.093 |

**Fix 1 (natural-frequency stem BPE) is real, but its payoff is not BPB.**
Tokens per byte fell 11.2% (0.2488 -> 0.2209) and clean BPB moved 0.0018 —
at the noise floor. What the fix actually bought: the factorized arm went from
a 4.8% token *deficit* against bpe8k to a 7.0% *surplus*, so it now sees 7.5%
more bytes of context per 1024-token window and is cheaper to run at the same
byte throughput. Case robustness also improved: d upper 0.839 -> 0.733.

**Fix 2 (per-run case classification) is a clean engineering win with no
measurable BPB benefit.** It removes the verbatim escape hatch entirely — every
piece of every mixed-case word now stays in the stem vocabulary, verified
byte-exact on all suites. Under `randcase` its absolute tokens/byte is 31%
better than fix 1 alone (0.276 vs 0.400). None of that reaches BPB: 1.2060 ->
1.2056, and d randcase is if anything slightly worse. The reason is that
`randcase` BPB is dominated by the entropy of the random case pattern itself,
which every representation has to pay; only the token count was ever fixable.

**Fix 4 (corrupted-context BPB) is the one that changed a conclusion.**

| arm | ctx clean | d typo1 | d typo5 | d typo10 | d randcase | d MISMATCH | typo10 / mismatch |
|---|---|---|---|---|---|---|---|
| bpe8k | 1.1754 | 0.0005 | 0.0024 | 0.0045 | 0.0081 | 0.0262 | 17.1% |
| fact8k-v2 | 1.1939 | 0.0008 | 0.0035 | 0.0066 | 0.0103 | 0.0269 | 24.6% |

Corrupting the *context* with 10% typos costs 0.0045 BPB, against roughly 1.3
for the same perturbation under the old whole-text metric — about **250x
smaller**. So ~99.6% of what this study previously reported as "typo
robustness" was the model being charged for reproducing noise, which no
input-side mechanism could ever have reduced. That fully explains why the
character n-gram bag scored 0.679 against its own control's 0.676.

The `mismatch` control makes this readable rather than merely small: replacing
the context with an unrelated document costs 0.0262-0.0269, so the metric does
have dynamic range and typo damage is a real 17-25% of the worst case. Caveat:
that worst case is itself only ~2% of clean BPB, i.e. a 33M model with 1024
tokens of context leans mostly on local statistics — the *ratio* is the
meaningful quantity here, not the absolute.

### Methodological correction

The training-free screen is a good robustness filter and a poor BPB predictor.
Two specific traps, both of which misled the round-2 predictions in this file:

- **tokens/byte does not forecast BPB.** Fix 1 delivered 11% fewer tokens and
  0.0018 BPB. Fewer-but-harder tokens is close to a wash per byte.
- **`inflation` is relative to each arm's own clean baseline, so it rewards a
  bad baseline.** `fact8k-nosuf` had the *best* randcase inflation (1.226)
  purely because its clean tokens/byte was the worst. Absolute tokens/byte
  under perturbation is the metric that ranks correctly; the screen JSON has
  always stored it as `tpb.<suite>`.

### Study 3 conclusions

Against bpe8k, the fixed factorized arm now costs +1.5% BPB (1.2056 vs 1.1881)
and returns 2.8x better ALL-UPPER robustness, 2.2x better Title Case, *and* 7%
shorter sequences. Before these fixes it cost the same +1.5% BPB while being
4.8% longer, so the trade is strictly better even though BPB did not move.

The honest summary of fixes 1 and 2 is that both worked exactly as designed at
the tokenizer level and neither improved bits-per-byte. That is worth stating
plainly: the remaining gap to plain BPE is not a segmentation problem, which
makes the joint (stem x case) softmax — idea 3, the one deliberately deferred —
the only remaining candidate that could close it.

## Study 4 — tree-structured vocabulary, stage 0: distill the host table

The program this gates: replace the flat embedding table with a decision tree
over embedding space — any string routes through learned splits, a token's
vector is composed from shift vectors along its root-to-leaf path, parameters
live at shared nodes instead of one row per token. Stage 0 is the cheapest
decisive probe, pre-registered before running: **fit structured approximations
of the frozen Qwen3-1.7B table offline** (no LM training anywhere), plug each
reconstruction into the frozen body as the tied embedding + output head, and
measure clean-BPB damage. Kill criterion: **> 0.05 BPB at ~x10 compression
kills the route**; the hope was < 0.01 at x20.

Grid in `configs.DISTILLS` (svd / vq / rvq / hierarchical-k-means tree /
tree + low-rank residual, at matched float budgets ~x4-x32, each optionally
fit under token-frequency weighting), runner `distill_embed.py`, full numbers
in tables §5, figure `results/distill_stage0.png`. Evaluation is the study-2
harness on the full valid split; the native table re-evaluated in this exact
harness scores 0.7801 (study 2's 0.7125 was a 2.2 MB cut of the same split —
deltas here are internal to stage 0).

How the stage-0 tree is built (`distill_embed._fit_tree`): recursive
*weighted k-means over embedding space* — no surface-feature rules anywhere
at this stage. Level 1 clusters the table rows (branching 64); level 2
clusters each cluster's *residuals* (row minus parent centroid); every node
stores its cluster's weighted residual mean, so the sum of node vectors
along a root-to-leaf path is exactly the leaf-cluster mean. Routing is
nearest-centroid, 12 bits per token, counted separately from floats. Two
structural facts fell out: (a) a depth-2 path-sum is *representationally
equivalent to flat VQ* with the same leaf count — the hierarchy constrains
how centroids are found and provides routing, but adds no expressivity,
which is why tree-only ≈ vq4k below; (b) embedding space is so unevenly
populated that the nominal 4,096 leaves collapse to 2,104 (unweighted) or
1,249 (frequency-weighted). `tree_res` adds the identity channel: a
(weighted) PCA of what the tree leaves behind, shared rank-r basis plus r
coefficients per token.

| arm | compress | rel err | d native BPB |
|---|---|---|---|
| rvq8x4k (best absolute) | x4.6 | 0.506 | **+1.371** |
| rvq4x4k | x9.3 | 0.615 | +1.791 |
| svd128-freq | x15.8 | 0.884 | +1.611 |
| tree64x2-r64-freq (best per float) | x24.8 | 0.901 | **+1.486** |
| svd512 | x3.9 | 0.576 | +3.082 |
| tree64x2-r64-freq-norm (control) | x24.5 | 1.395 | +6.861 |

**1. The kill criterion fired, with a 30-60x margin.** The best fit at any
compression costs +1.37 BPB; the criterion was 0.05. No method, objective, or
budget came within an order of magnitude, so this is not a tuning shortfall.

**2. The table is not low-rank — at all.** Rank 512 (a quarter of d_model)
leaves 57.6% of the variance unexplained, and SVD's BPB damage is *flat* in
rank (+3.1 to +3.3 from x3.9 to x31.6). The spectrum is nearly uniform: the
table's information is per-token idiosyncratic, with no dominant shared
subspace for a structural fit to exploit.

**3. The fit objective matters more than the structure.** Frequency-weighted
fitting roughly halves the damage at matched budget (svd128 3.25 -> 1.61;
tree-r64 2.44 -> 1.49) — the biggest single lever found — while swapping
structures at matched budget moves far less. The decision-tree constraint
itself is nearly free (tree-only at x70 ≈ flat VQ at x37) but helps nothing.

**4. The damage is directional, not a softmax-temperature artifact.** The
obvious objection — low-rank fits shrink row norms, maybe rescaling fixes
it — was run as a control and made things far *worse* (+1.49 -> +6.86,
reconstruction error above 1.0): restoring full magnitude to wrong directions
amplifies the error. Consistently, BPB damage is not monotone in MSE across
methods; what the body cares about is not Frobenius distance.

**5. Both sides of the model are catastrophic, the output side more so.**
Reconstructing only the input embedding costs +2.10, only the output head
+2.89, both +3.25 (svd128). There is no cheap half-measure.

**6. Capacity was never the obstacle — the offline objective is.** Study 2's
`bpe8k-t` trained a 16.4M-float table (x19 fewer floats than native) through
the frozen body and *beat* the native model (0.7031 vs 0.7125). A x19
compressed embedding function for this body demonstrably exists and is
reachable by one GPU-hour of training — it is just not reachable by
approximating the pretrained matrix. Matching the table is not matching the
function.

### Stage 0 conclusions

The distillation route — compress an existing model's embedding table
offline, plug it in, keep BPB — is **dead by its pre-registered criterion**,
and died for a structural reason, not a tuning one: a trained embedding table
has effectively full intrinsic parametric dimension under offline
reconstruction. That is itself a finding worth keeping: every one of the
311M floats is doing work that MSE — even frequency-weighted MSE — cannot
summarize.

What survives is the *trained* route, and study 2 already proves its
capacity side. If the tree-vocabulary program continues, stage 1 must train
the tree-composed parameterization through the frozen body (study-2 protocol,
~1 GPU-hour per arm) rather than distill it; the question becomes whether
routed, shared-node structure matches a flat table at equal floats when both
are *trained*, with open-vocabulary routing as the remaining upside. The
cheap 33M-scale alternative from study 3's conclusion — the joint
(stem x case) softmax — is unaffected by this result and remains the other
open thread.

### Stage 1 plan — train the tree, don't distill it (pre-registered, not run)

Stage 0's lesson dictates the design: the tree supplies *structure* (which
parameters a token touches); gradients through the frozen body supply the
values. Four commitments, fixed before any code runs:

1. **Trainable path composition.** A token's embedding is the sum of node
   shift vectors along its fixed root-to-leaf path plus a per-token low-rank
   residual; node vectors and residual factors train through the frozen
   Qwen3-1.7B body under the study-2 transplant protocol. The softmax stays
   *flat* over the vocabulary with the output embedding composed from the
   same tied tree — stage 0's in/out ablation showed the output side is the
   sensitive one, so prediction never goes hierarchical.
2. **Routing is fixed before training, from mean-of-pieces vectors.** Stage 0
   clustered a table stage 1 does not have; the mean-of-pieces vectors
   (study 2's init, worth 0.61 BPB there) are computed per token, clustered
   hierarchically to assign paths, then discarded. Optional surface-rule
   upper levels (script, case shape, leading space) keep the top of the tree
   interpretable. Learned routing (periodic re-clustering) is an ablation to
   run only if fixed routing is the measured bottleneck — it imports
   codebook-collapse failure modes.
3. **A multiplicative-composition ablation.** The additive path-sum is
   representationally flat-VQ-plus-residual (see method note above); one arm
   gives each node a FiLM pair (elementwise scale + shift applied down the
   path) so depth can buy expressivity. The additive arm stays the default
   so the baseline remains clean.
4. **Small vocabulary, matched control.** The comparison is against the
   trained flat table `bpe8k-t` (0.7031, 16.4M embedding floats) on the same
   bpe8k vocabulary, with tree arms at ~x2/x4 fewer embedding floats plus
   the FiLM and re-clustering ablations. Open-vocabulary routing over
   arbitrary strings is stage 2, contingent on stage 1 surviving.

Success criterion: clean BPB within ~0.005 of `bpe8k-t` at >=x2 fewer
embedding floats, robustness suites unchanged. Kill criterion: worse than
+0.05 BPB at x2. Estimated cost ~4-5 GPU-hours. The concrete grid lands in
`configs.py` (`TREE_TRANSPLANTS`) when implementation starts; per this
file's convention the code, not this section, will then be the source of
truth for hyperparameters.

## Changelog

- 2026-08-26 — Built harness: corpus prep, surface factorization with verified
  invertibility, 7 tokenizer arms, perturbation suites, training-free screen,
  factorized-softmax LM, 9-run ablation grid. Screen results in.
- 2026-08-27 — Full training grid done (10 runs, ~3.5 GPU-hours on one A100).
  Case+space factorization confirmed: 4x smaller vocab at parity BPB with 2.5x
  less case damage. Suffix head and n-gram bag are negative results; added
  `fact8k-nosuf` after the first 9 runs showed it was the configuration the
  data pointed at. Figure: `results/vocab_factorization.png`.
- 2026-08-27 — Post-hoc diagnosis of the suffix head (`dump_example.py`, plus
  the entropy probes recorded below) and a methods figure built from real
  tokenizer output: `results/vocab_factorization_method.png`.

### Why the suffix head failed — measured, not guessed

Three separable problems, only the first two of which are bugs:

1. The head is **2.6x worse than a count table**. A plug-in `stem -> P(suffix)`
   lookup scores 0.1114 bits/byte held-out; the trained head paid 0.2854 while
   having strictly more information (context *and* stem). Cause: the head
   computes `W·(h + E_stem[target])`, so the stem's contribution is a rank-512
   read-off of a *tied* embedding table shaped almost entirely by the stem
   loss. The head owns no parameters for the stem->suffix map.
2. It **fires at every position**. 24.2% of tokens carry a real suffix
   decision, yet ~35% of the head's budget goes to predicting "none here" at
   the other 76%. Gating it to word-final positions would recover ~0.038.
3. **The premise does not hold.** Splitting suffixes out made the stem head
   only 0.034 bits/byte easier (1.1519 -> 1.1181), which is less than any
   achievable suffix cost. Even an oracle count-quality head lands at 1.2871
   against 1.2137 for no suffix head at all. Fixing (1) and (2) turns a -0.251
   loss into roughly break-even; it never becomes a win.

The morphology *inductive bias* framing also fails its own test: clustering
stems into paradigm classes loses information rather than sharing it
(H(suffix|stem) = 1.51 bits, H(suffix|128 classes) = 2.77), and only 4.0k of
1.93M held-out suffix decisions involve a stem seen fewer than 50 times. At
this corpus scale there is no generalization gap for a morphological prior to
fill. It should only start paying in low-resource settings or on genuinely
unseen words.
- 2026-08-27 — Study 2: vocabulary transplant onto a frozen Qwen3-1.7B (5 arms,
  ~4.5 GPU-hours). An 8k BPE with mean-of-pieces init beats the untouched
  pretrained model on BPB (0.7031 vs 0.7125) with 295M fewer parameters. Init,
  not vocabulary, is the dominant factor; zero-training transplant does not
  work. Case factorization costs 5% BPB at this scale for 1.5-3x better case
  robustness. Figure: `results/vocab_transplant.png`.
- 2026-08-28 — Study 3: fixes 1, 2, 4. Natural-frequency stem BPE (-11% tokens),
  per-run case classification (verbatim fallback removed), and a
  corrupted-context BPB metric with a mismatch control. Neither tokenizer fix
  moved BPB (noise floor measured at 0.0016), but together they turned a 4.8%
  token deficit vs bpe8k into a 7% surplus and improved d upper 0.839 -> 0.691.
  The metric showed ~99.6% of the old typo-robustness number was emission cost.
- 2026-08-28 — Study 4 stage 0: offline distillation of the Qwen3-1.7B
  embedding table (19 fits + native + controls, ~1.5 GPU-hours of eval on a
  contended shared GPU; two OOM crashes fixed by reserving peak memory up
  front). Killed by its pre-registered criterion with a 30-60x margin: best
  fit +1.37 BPB vs the 0.05 line. The table has effectively full intrinsic
  dimension under offline reconstruction; the trained route (study 2) is the
  one that survives. Figure: `results/distill_stage0.png`.
- 2026-08-28 — Stage 1 pre-registered (not run): train the tree-composed
  parameterization through the frozen body against the `bpe8k-t` flat-table
  control at matched vocabulary; fixed mean-of-pieces routing, flat softmax,
  FiLM and re-clustering ablations. Success: parity BPB at >=x2 fewer
  embedding floats; kill: +0.05 at x2. Recorded stage-0 tree-construction
  method note (path-sum = leaf mean = flat-VQ equivalence; leaf collapse
  2,104 / 1,249 of nominal 4,096).
