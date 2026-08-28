# CONTEXT — domain glossary

Canonical names for the concepts this project reasons about. Definitions only:
what a term *is*, never how it is implemented. Implementation lives in code;
findings live in `experiments/*.md`.

## Research-programme terms

### Track
An independently rankable line of experimental work: it can be started,
paused, killed, or completed without any other track finishing first. A track
is larger than a run and smaller than a study — `progressive-shrink` is a
track, one transplant arm is a run, "Study 2" is a study. Priority is assigned
to tracks, not to runs.
_Avoid_: workstream, thread, phase, epic.

### Ranking objective
The single quantity that orders tracks against each other. For the
vocabulary-factorization programme it is **information gained per GPU-hour
toward a claim that survives scrutiny** — so a cheap probe that can return a
decisive negative outranks an expensive one that can only return a positive.
_Avoid_: priority (unqualified), importance, value.

### Decisive probe
An experiment whose *cheapest* version can already falsify the thing it is
testing, chosen so that a negative result ends a line of work rather than
inviting a bigger rerun. Study 4 stage 0 is the archetype: no LM training
anywhere, and it killed the distillation route outright.
_Avoid_: pilot, smoke test, quick experiment.

### Kill gate
A quantitative threshold, written down *before* the experiment runs, at which
the line of work is abandoned. Distinct from a success gate, which is the
threshold for continuing; a result may land between the two, which is a
"reconsider", not a pass.
_Avoid_: kill criterion / success criterion used interchangeably, threshold.

### Arm
One configuration inside a comparison — the unit that is scheduled, trained,
and reported as a row. Arms within a study share a protocol so that the only
thing differing between two rows is the property under test. Arms are the unit
of *execution*; tracks are the unit of *prioritization*.
_Avoid_: variant, config, condition, experiment (which is the whole comparison).

### Fitness ladder
An ordered set of evaluation *rungs* for the same candidate, from seconds to
hours, used so that cheap rungs filter what expensive rungs confirm. A ladder
is only valid once its cheap rungs are shown to **rank-correlate** with the
expensive one on a common set of candidates; an uncalibrated ladder is a
guess about which proxies matter, and this programme has twice been misled by
exactly that (study 3's methodological correction).
_Avoid_: proxy metric, screen (a screen is one rung), cascade.

### Wave
A set of tracks that may execute concurrently because none of them depends on
another's result. Waves are separated by barriers: everything in wave N must
report before wave N+1 starts. The unit in which a schedule is expressed once
compute is parallel rather than serial.
_Avoid_: stage (reserved for a study's internal phases, e.g. "stage 0"), sprint,
batch.

## Vocabulary / embedding terms

### Healing train
A short training run whose only purpose is to let a model recover from an edit
made to its vocabulary or parameterization, rather than to learn anything new.
It is what distinguishes a *curriculum* of small vocabulary edits from a
one-shot replacement — the latter is what study 4 stage 0 killed.
_Avoid_: fine-tune, recovery run, adaptation.

### Open-vocabulary routing
The property of an embedding function that it can assign a vector to a string
having **no entry of its own** — the vector is composed from structure the
string shares with known strings, rather than looked up. A flat embedding table
cannot do this by construction; a feature-routed one can. This is the only
capability in the programme that no flat-table baseline can match, and it is
therefore what a routed parameterization must be judged on, separately from
parameter count.
_Avoid_: OOV handling (too narrow — this covers seen-but-untabled strings too),
subword fallback, generalization.

### Transplant
Replacing a pretrained model's vocabulary — its input embedding and output
head — while holding the transformer body frozen, then training only the new
parameters. The body is shared across arms by construction, which is what
makes transplant comparisons internally valid even when the absolute numbers
are optimistic. Since study 2 this is the programme's primary protocol; the
33M from-scratch setup is now a *rung*, not a venue for claims.
_Avoid_: retrofit, vocabulary swap, re-embedding.

### Active vocabulary
The token ids that actually occur in the corpus under study (train stream or
full valid split), as opposed to the host tokenizer's full id space. Measured
at 72,304 of Qwen3-1.7B's 151,936 on this ru+en corpus. Load-bearing because
inactive ids leak into every result through the softmax normalizer unless every
arm — including the untouched reference — masks the same set.
_Avoid_: used vocab, effective vocab, corpus vocab.
