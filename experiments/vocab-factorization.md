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

### Stage 1 plan v2 — feature-routed trees, trained through the body (pre-registered, not run)

Supersedes the same-day v1 registration (k-means routing) before any code was
written; the success/kill gates are unchanged. Three changes and their
reasons: routing moves from k-means-in-embedding-space to **interpretable
token features** (boosted vector-leaf trees); optimization and accounting are
restricted to the **active vocabulary**; and the plan now includes the
evolution fitness ladder and the progressive-shrink track. This section is
written to be self-contained for an implementing session that has not seen
the discussion.

**Unchanged commitments (from v1, with stage-0 reasons).** The tree supplies
structure, gradients through the frozen Qwen3-1.7B body supply the values
(stage 0: offline fits of any structure cost +1.4 BPB minimum; matching the
matrix is not matching the function). The softmax stays *flat* over the
vocabulary, output embedding composed from the same tied structure (stage
0's in/out ablation: the output side is the more sensitive one). Additive
composition is the default arm; a FiLM arm (per-node elementwise scale +
shift) is the ablation that lets depth buy expressivity, since an additive
path-sum is representationally flat-VQ-plus-residual (method note above).
Training protocol is study 2's (`TransplantTrainSpec`, ~1 GPU-hour/arm,
loop in `train_transplant.py`; keep `distill_embed._reserve`'s
grab-peak-memory-once pattern — the GPU is contended and two stage-0
launches died to the allocation race).

**Routing v2: vector-leaf boosted trees over token features.** A GBDT whose
leaves hold vectors *is* the additive-codes family with named rule paths:
each shallow tree (depth ~6-8, ensemble of many) routes a token by feature
tests and contributes its leaf vector; the sum is the embedding. Structure
is fit once, offline, to mean-of-pieces vectors (targets define the
*partition only* — study 2 measured mean-of-pieces worth 0.61 BPB as an
init, and stage 0 says fitted *values* must not be trusted, so the leaf
vectors are re-learned through the body). Implementation: sklearn 1.9 is in
the env and its multi-output `DecisionTreeRegressor` + a manual residual
boosting loop suffices; `pip install catboost` (MultiRMSE) is the
alternative — neither catboost nor xgboost is currently installed. One
optional arm does **functional gradient boosting**: each round fits trees to
the loss gradient w.r.t. token embeddings collected through the frozen body
— boosting on the true objective instead of matrix MSE, which is exactly the
correction stage 0 demands.

**Feature set v0 (later the evolution genome).** Per token id, computed once:
- *surface*: script class (latin/cyrillic/digit/punct/other), case shape
  (lower/Title/UPPER/mixed), byte length, leading-space flag, digit/punct
  content flags;
- *decomposition*: number of host-tokenizer pieces of the surface string,
  bucketed first/last piece id (machinery exists in
  `transplant.mean_of_pieces_init` / `tokenization.piece_string`);
- *char n-grams*: hashed 3-5-gram presence buckets (reuse
  `model.build_ngram_table`'s FNV-1a hashing);
- *corpus n-gram statistics* from the cached stream
  `data/tokens/qwen/train.230mb.npz`: log unigram frequency, left- and
  right-neighbor entropy, top-neighbor bigram PMI, document frequency /
  burstiness.
Study 1 killed char n-grams as an *additive input embedding* under an
emission-dominated metric; here they are *routing features* for parameter
sharing — a different role, and that metric objection does not transfer.

**Active-vocabulary convention (fairness).** Measured in stage 0: only
72,304 of 151,936 host tokens occur in the 230 MB train stream; the rest is
mostly other-language mass this ru+en corpus never exercises. Convention,
applied to *every* arm including the native reference: (a) *active* = ids
occurring in the train stream or the full valid split; (b) only active
tokens are parameterized and optimized; (c) inactive tokens are **masked out
of the softmax in both training and evaluation** — without the shared mask
they leak into every number through the normalizer, so per-arm deltas would
be incomparable; (d) compression ratios are quoted against the *active*
flat table, not 311M. The native reference
must be re-evaluated under the mask (expect slightly below the unmasked
0.7801; that masked number becomes the stage-1 reference). Deployment of a
masked-vocab model is out of scope for stage 1. Arms on the corpus-trained
bpe8k vocabulary satisfy the convention trivially.

*Superseded by measurement (W0 below): the active set is **74,957**, not the
72,304 estimated here, because the perturbation suites emit ids the train
stream and clean valid split do not; the flat table is therefore 153.5M
floats, and the masked native reference is **0.7796**. `active_vocab.py` is
the implementation and the source of these numbers.*

**Grid sketch** (lands in `configs.py` as `TREE_TRANSPLANTS`; code becomes
the source of truth for hyperparameters once written):
- `bpe8k-t` control re-run under the active-vocab convention (its 0.7031
  predates the mask);
- tree arms on the bpe8k vocabulary at ~x2 and ~x4 fewer embedding floats
  than the 16.4M flat control;
- the FiLM arm and the functional-gradient-boosting arm at the x2 budget;
- one bridge arm on the *active host vocabulary* (72k): tree fit offline to
  the native active rows as init, then trained — directly tests whether
  training rescues the distilled init that stage 0 showed fails zero-shot.

**Gates (unchanged).** Success: clean BPB within ~0.005 of the flat control
at >=x2 fewer embedding floats, robustness suites unchanged. Kill: worse
than +0.05 BPB at x2. Estimated ~5-7 GPU-hours for the grid.

**Fitness ladder (built during stage 1, used by the evolution phase
after).** L0, seconds: frequency-weighted explained variance of
mean-of-pieces vectors under the candidate partition. L1, ~1 min: step-0
clean BPB on a small eval slice under the active-vocab mask (study 2 showed
step-0 BPB discriminates inits). L2, ~13 min: one 33M from-scratch run. L3,
~1 h: full transplant. The evolution phase — GigaEvo mutating the
feature-extractor *program* (`token_features.py` is the genome) — runs on
L0/L1 with periodic promotion of elites to L2/L3; this study was misled by
cheap proxies twice (study 3's methodological correction), so no
evolutionary conclusion is accepted from L0/L1 alone.

**Progressive-shrink track (independent; composes with the tree).** Start
from the intact native table; rule 0 is the active-vocab mask itself (an
instant 152k -> 72k shrink); subsequent interpretable rules (merge
case/space variants of one lexeme into stem + role bit — the fact
factorization applied to a live vocabulary; re-piece the frequency tail)
each followed by a short healing train. Output: the study's first BPB vs
vocab-size curve on a pretrained body, and a warm-started small vocabulary
for the tree parameterization. One-shot replacement is what stage 0 killed;
a curriculum of small healed edits is the change the evidence permits.

**Suggested module layout.** `token_features.py` (pure feature extractor,
unit-testable), `tree_embed.py` (boosted-partition fit + a `TreeEmbedding`
module with fixed routing and trainable leaf/residual parameters),
`TREE_TRANSPLANTS` in `configs.py`, reusing `train_transplant.py`'s loop
with two hooks: the softmax mask and the embedding-module swap.

## Priorities — what runs next, and in what order

Four studies have left five open tracks and no ordering among them. This
section fixes one. Decisions that were not obvious are recorded as ADRs
(`docs/adr/`); the terminology is in `CONTEXT.md`; the full derivation,
including the alternatives rejected, is
`run/deep-interview/deep-interview-vocab-factorization-priorities.md`.

**Ranking objective: information gained per GPU-hour toward a claim that
survives scrutiny** (ADR 0001). Compute is not the binding constraint — the
grid runs as parallel MLS jobs over a week-plus — so the ordering is by what a
result *changes*, not by what it costs. A cheap probe that can return a
decisive negative outranks an expensive one that can only return a positive,
which is the same principle that made stage 0 worth running.

| # | Track | Cost | Gate |
|---|---|---|---|
| W0 | *prerequisites* — MLS job scaffolding, checkpoint saving, **P0** = active-vocab mask + masked native reference | ~2 GPU-h | P0's masked BPB becomes stage 1's reference (expect slightly below the unmasked 0.7801) |
| 1 | **transplant-hardening** — downstream-task eval + an unfrozen-body arm | ~1.3 GPU-h marginal | `bpe8k-t` within a few points of `orig-frozen` on loglikelihood MC → study 2's headline stands; else it is rewritten from "beats the model it replaced" to "matches it on corpus BPB" |
| 2 | **stage1-tree** — the pre-registered v2 grid above | 5–7 GPU-h + 2 modules | unchanged: parity BPB at ≥x2 fewer embedding floats; kill at +0.05 at x2 |
| 3 | **ladder-calibration** — rank-correlate L0/L1 against L2 | ~4 GPU-h | Spearman ρ ≳ 0.7 → the evolution phase is unblocked; else it is killed (ADR 0002) |
| 4 | **progressive-shrink** — the curriculum past rule 0 | ~3 GPU-h | continue while healed BPB stays within +0.01 of the masked reference; stop at the vocabulary size where it does not |
| 5 | **evolution-phase** — GigaEvo over `token_features.py` | ~0 GPU-h now | pre-registration only: problem dir, `metrics.yaml`, initial program, and an explicit choice of MAP-Elites descriptors. Launch blocked on #3 |
| — | **stem×case-softmax** | — | **deferred** (ADR 0003) |

Waves, since execution is parallel: **W0** blocking → **W1** transplant-hardening
‖ stage-1 module implementation → **W2** stage-1 grid ‖ ladder-calibration ‖
progressive-shrink → **W3** evolution pre-registration.

**Why hardening outranks the flagship.** It is the cheapest thing available that
can falsify something already written in this file. Study 2's headline — a
transplant that beats the model it replaced with 295M fewer parameters — rests
entirely on BPB measured on the same corpus the new embeddings were trained on,
and the file already concedes that "good BPB does not guarantee preserved
instruction-following". It costs almost nothing because it budgets **no
dedicated runs**: stage 1 v2 already schedules a `bpe8k-t` re-run under the
active-vocab convention, `orig-frozen` needs no training at all
(`train=False`), and `train_transplant.py` now saves checkpoints — so the
evaluation rides along on runs that are happening anyway. Round 1 of study 2
saved nothing, which is why this could not simply be measured on the existing
models: they no longer exist.

**Why the evolution phase is gated rather than scheduled.** Its ladder has two
defects that only show up against this study's own findings. L0 scores
frequency-weighted explained variance — a Frobenius-family fit — which is
precisely what stage 0 finding 4 showed does not predict BPB. And the
trustworthy rungs cannot carry a search: a completed GigaEvo run in this repo
(`problems/tabular_regression/elite_intra_extra_20260524.py`) records ~800
unique evaluations, which at L2's ~13 min is ~173 GPU-h — 25x this programme's
total spend to date, spent driving selection on an unvalidated fitness
function. That is the failure mode the methodological correction above
documents twice. ~4 GPU-h of rank-correlation buys the right to spend it, or
not to. The calibration candidates are feature-group ablations **plus three or
four deliberately-bad anchors** (random routing, single-feature routing,
length-only): without a known-bad low end, ρ is measured over a restricted
range and reads optimistically — the same range-restriction trap as `inflation`
rewarding a bad baseline.

**Rule 0 is not part of progressive-shrink.** It is a prerequisite of stage 1 —
the masked native reference every stage-1 arm is measured against — so it runs
in W0 regardless. It is also free on BPB by construction: masking removes
competitors from the normalizer and can only help. The curve's gate therefore
lives on rules 1 and beyond, where parameters are actually deleted.

### W0 — done (2026-08-28)

All three prerequisites have landed; W1 is unblocked.

**MLS job scaffolding.** `jobs.py` holds the job registry, derived from
`configs.py` rather than duplicating it — adding an arm there makes it
launchable here. `run_train_jobs.py` and `run_eval.py` submit it, one
`a100.1gpu` job per arm (the allocation offers 42 of them, so the wave
schedule's parallelism is real). Two environment facts were measured, not
assumed: worker containers mount the workspace at `/workspace-SR004.nfs2/`
rather than `/mnt/virtual_ai0001053-00054_SR004-nfs2/`, and `mls` is importable
only from the `arkhip` env while the entrypoints need `gigaevo-core`'s torch.
`configs.py` is therefore kept standard-library-only, so the launcher reads the
grid under one interpreter and the job runs it under another. Launchers refuse
a dirty tree, stage the tracked tree to `<artifacts>/code/<commit>`, skip
finished and already-queued work, and refuse to fan out over a token cache that
does not exist yet — `get_tokens` is a read-through cache, so N jobs missing the
same entry would race to write it.

**P0 — the masked native reference is 0.7796** (`results/transplant_masked/`),
against 0.7801 unmasked in the same harness. Every suite moves the same way and
by the same negligible amount:

| suite | masked | unmasked | Δ |
|---|---|---|---|
| clean | **0.7796** | 0.7801 | −0.0005 |
| typo1 | 0.9358 | 0.9365 | −0.0007 |
| typo5 | 1.3905 | 1.3921 | −0.0016 |
| lower | 0.8243 | 0.8249 | −0.0006 |
| upper | 1.0702 | 1.0710 | −0.0009 |
| title | 0.8885 | 0.8891 | −0.0006 |
| randcase | 1.8151 | 1.8165 | −0.0014 |

Every delta is at or under the 0.0016 noise floor, so **the mask costs the
native model nothing measurable** — as predicted, since a pretrained model
already puts almost no mass on the other-language tail the mask removes. Its
value is comparability, not the number: an arm with a bad init *can* put real
mass there, and without the shared mask that would land in its BPB as a
normalizer artifact rather than as anything about its embedding.

**The active set is 74,957, not 72,304.** The train stream reproduces stage 0's
72,304 exactly and the clean valid split adds 173. The remaining 2,480 come
from the six perturbation suites, and taking them is not optional: the `upper`
suite alone emits 1,323 ids the pre-registered set lacks, covering 3.4% of its
tokens, and under the mask each is a target of probability zero. Scoring it
against the pre-registered set would not have shifted `upper` BPB — it would
have made it unbounded. So the convention is "every text the harness scores",
`assert_targets_active` fails loudly rather than quietly if that is ever
violated again, and compression is quoted against 74,957 x 2048 = **153.5M
floats** in place of 148.1M. On the corpus-trained arms the convention is
near-vacuous as the plan expected: bpe8k keeps 7,949 of 8,000 ids,
fact8k-nosuf 7,951.

**Checkpointing** (landed 0626e5ab) now has a reader: `eval_transplant.py`
reloads a saved embedding and head, re-scores it, and records the delta against
the run that saved it. Its round trip is covered on CPU in
`tests/experiments/test_vocab_factorization_w0.py` — with a stub body, so the
mask and checkpoint contracts are tested without a GPU or a 1.7B download. It
has not yet run against a real checkpoint, because none exists: producing one
means training an arm, and the local card could not spare the ~40 GB while
neighbouring processes held 46 of its 80. That is the constraint W0 exists to
remove, and the first W1 job removes it.

**End-to-end validation.** `orig-frozen` was submitted through the scaffolding
as `lm-mpi-job-5d8813f0-f297-45cd-b989-6e244519a5a2` and reproduced the local
P0 run **digit for digit on all seven suites** — 0.7796 / 0.9358 / 1.3905 /
0.8243 / 1.0702 / 0.8885 / 1.8151 — from a staged checkout of `c5aee2e1`, in
377 s against roughly forty minutes on the contended local card. Staging, both
path translations, the env prefix, the shared HF cache and the cached active
set are therefore all confirmed working inside a real container rather than
argued to. A re-launch now skips the arm on its result marker.

### W1 strand A — transplant-hardening (2026-08-29)

The downstream evaluator exists and its reference is measured.
`hellaswag.py` scores loglikelihood MC over the four endings, summing **every**
head so a factorized arm cannot move its case cost into a side head, and
normalising `acc_norm` by continuation **bytes** rather than tokens — the
endings are the same strings for every arm, so a byte denominator is identical
across arms while a token denominator would make the comparison partly about
tokenization.

**Native reference: `acc` 0.4610 / `acc_norm` 0.6035** over the full 10,042-example
validation split. That number is itself the harness's correctness check:
Qwen3-1.7B's published zero-shot HellaSwag is ≈0.60, and a hand-written scorer
that lands on the published value is very unlikely to be wrong in a way that
matters.

**The mask is deliberately lifted for this benchmark.** It exists to make
per-arm BPB on *this* ru+en corpus comparable; HellaSwag is a different
distribution, and scoring its tokens under the mask would make them impossible
rather than unlikely. What the mask would have hidden is reported instead: the
fraction of scored tokens outside each arm's active set. For the native arm
that is **712 of 1,182,755 (0.06%)** — the corpus-derived active vocabulary
covers the benchmark almost exactly, so the mask is not what any arm's score
will turn on. For the 8k arms it is the number to read first.

**The unfrozen-body arm** (`bpe8k-t-unfrozen`) isolates whether the residual
BPB gap is the embedding's fault or the body's refusal to move. It is the only
arm that trains the body, at a quarter of `bpe8k-t`'s micro-batch: AdamW over
1.7B parameters adds ~13.6 GB of moments and ~6.8 GB of fp32 gradients, while
activation cost is unchanged because gradients already traversed the whole body
to reach layer 0. It also broke a checkpoint assumption — a saved arm's body
was documented as recoverable from HF, which stops being true the moment one
trains it, so `_save_checkpoint` now stores the body for unfrozen arms only and
`load_checkpoint` refuses an unfrozen checkpoint that lacks one rather than
silently pairing a pristine body with an embedding trained against a different
one.

Jobs are registered as the `hellaswag` group. `orig-frozen` is launchable now
(`--native`, no checkpoint, since `train=False` means it never writes one); the
other five carry a `requires` on their training checkpoint and are skipped with
"waiting on an earlier wave" until it exists — which is what makes the wave
schedule enforce itself rather than rely on launch order.

### W1 strand B — the stage-1 modules (2026-08-29)

Both modules exist and the grid is launchable. Gates unchanged.

`token_features.py` extracts the pre-registered feature set v0 — **345 columns**
on the bpe8k arm (surface 17, decomposition 66, ngram 256, corpus 6), in ~7 s.
Its interface matters more than its contents because it becomes the evolution
genome: groups are first class (`select` / `without` / `columns`), so the
feature-group ablations and the deliberately-bad anchors calibration needs are
one call rather than a re-extraction, every column is named, and extraction is
deterministic with no hidden state.

`tree_embed.py` fits the partition offline and relearns the values through the
body. Two decisions are worth recording:

- **Split search runs on projected targets.** sklearn's multi-output splitter is
  O(features x samples x outputs) per level, so 2048-wide targets take hours.
  Targets are projected to 32 dimensions first. This is not stage 0's mistake in
  miniature: there, discarded detail became frozen *values* that were then
  wrong; here it can only make the partition slightly less well chosen, and the
  vectors are relearned regardless.
- **Composition accumulates one tree at a time.** The obvious `leaves[routing]`
  gathers `(V, n_trees, d)` in one allocation — 59 GB for the host bridge arm.
  Per-tree accumulation keeps the peak at a couple of `(V, d)` tensors, because
  addition's backward needs neither operand. The FiLM path cannot have that (a
  product's backward keeps both), which is precisely why FiLM is an 8k-vocabulary
  arm and there is no FiLM arm on the 152k host.

**Measured on the real bpe8k arm**: `tree8k-x2` fits 3,968 leaves over 32 trees
= **8.13M floats against the 16.3M active flat table, exactly x2.00**, at 224
route bits per token (bits reported beside the floats, never inside the ratio,
per stage 0's convention). `param_counts` reports `output_heads: 0` — the tie is
structural, both sides being the *same* module, so the leaves are counted once.

**Untrained, that routed table scores 2.1348 clean BPB** against the flat arm's
0.70. That is not a result, it is stage 0's finding reappearing in the new
parameterization: leaf means of mean-of-pieces vectors are fitted *values*, and
fitted values do not transfer. The entire stage-1 claim rests on what training
through the frozen body does to that number, which is what the gate measures.

Grid in `configs.TREE_TRANSPLANTS`, launchable as the `tree` job group: additive
at x2 and x4, the FiLM arm, the functional-gradient-boosting arm (which fits the
partition to the loss gradient w.r.t. embedding rows collected through the body,
rather than to matrix residuals — the correction stage 0 demands, in its
single-pass form), and the active-host-vocabulary bridge arm. The flat control
is `TRANSPLANTS['bpe8k-t']`, which the `transplant` group already re-runs under
the mask; no arm here budgets a second copy of it.

**What was considered and rejected.** An open-vocabulary gate on stage 1 —
held-out active ids scored against a flat control — was proposed on the grounds
that the pre-registered gates are parity-only and so cannot detect the one
thing a flat table structurally cannot do. It was **rejected**: stage 1's gates
stand as pre-registered. The consequence is recorded rather than hidden — a
passing stage 1 supports "a routed table matches a flat one at half the floats"
and not a capability claim.

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
- 2026-08-28 — Priorities fixed across the five open tracks (no runs yet).
  Ranking objective recorded as ADR 0001; the evolution phase gated behind a
  ladder-calibration track as ADR 0002; `stem×case-softmax` deferred and the
  33M setup demoted from a venue for claims to ladder rung L2 as ADR 0003.
  `train_transplant.py` now saves the trained embedding and heads — study 2's
  models were unrecoverable, which is why hardening had to be sequenced around
  runs that were already scheduled rather than measured directly. Glossary in
  `CONTEXT.md`; derivation in `run/deep-interview/`.
- 2026-08-28 — Stage 1 plan revised to v2 before implementation (gates
  unchanged): routing by interpretable token features (surface + char
  n-gram + corpus n-gram statistics) via boosted vector-leaf trees instead
  of k-means in embedding space; optimization, softmax, and accounting
  restricted to the active vocabulary (72,304/151,936 ids seen in the
  corpus — masking shared by every arm including the native reference);
  added the functional-gradient-boosting arm, the active-host-vocab bridge
  arm, the L0-L3 fitness ladder for the later GigaEvo feature-evolution
  phase, and the progressive-shrink track. Written self-contained for a
  fresh implementing session.
- 2026-08-28 — **W0 done**, W1 unblocked. MLS job scaffolding (`jobs.py`,
  `run_train_jobs.py`, `run_eval.py`) fans the grid out one arm per
  `a100.1gpu`; `active_vocab.py` + a stem-softmax mask in `TransplantLM`
  implement the active-vocabulary convention for every arm; `eval_transplant.py`
  gives the checkpoints a reader. **P0: the masked native reference is 0.7796**
  (unmasked 0.7801 in the same harness) — every suite moves by ≤0.0016, so the
  mask costs the native model nothing measurable and buys comparability only.
  The active set is **74,957**, not the pre-registered 72,304: the train stream
  reproduces 72,304 exactly, but the perturbation suites emit 2,653 further ids
  (`upper` alone 1,323, over 3.4% of its tokens) that the mask would otherwise
  score at probability zero. Compression is quoted against 153.5M floats.
  Mask and checkpoint contracts covered on CPU in `tests/experiments/`.
