"""HellaSwag loglikelihood multiple choice for transplanted models.

Study 2's headline is that a transplant *beats* the model it replaced: bpe8k-t
0.7031 clean BPB against orig-frozen's 0.7125, with 295M fewer parameters. That
number is measured on the same corpus the new embeddings were trained on, and
the plan concedes in as many words that "good BPB does not guarantee preserved
instruction-following". This module is the cheapest available thing that can
falsify it: a task the transplant was never trained for, scored the same way
for every arm.

**Gate (pre-registered).** ``bpe8k-t`` within a few points of ``orig-frozen``
on HellaSwag → study 2's headline stands. Otherwise it gets rewritten from
"beats the model it replaced" to "matches it on corpus BPB".

Scoring is the standard harness protocol: for each of the four endings, sum the
log-probability of the continuation tokens given the context, and take the
argmax. Two numbers come out, because the choice between them is not neutral --
``acc`` uses the raw sum and therefore favours short endings, while ``acc_norm``
divides by the continuation's **byte** length. Bytes rather than tokens is the
load-bearing detail: the endings are the same strings for every arm, so a byte
denominator is identical across arms, whereas a token denominator would hand a
coarser tokenizer a different normalizer and quietly make the comparison about
tokenization rather than about the embedding.

**The active-vocabulary mask is deliberately lifted here.** The mask exists so
that per-arm BPB on *this* ru+en corpus is comparable -- it removes ids the
corpus never exercises from a shared normalizer. HellaSwag is English from a
different distribution, so some of its tokens fall outside every arm's active
set; scoring them under the mask would make them impossible rather than
unlikely, and the benchmark would measure the mask instead of the model. So the
mask comes off, and the thing it would have hidden is reported instead: the
fraction of scored tokens outside each arm's active set is a first-class field
in the result. An arm that cannot represent the benchmark's text is not a
scoring inconvenience, it is a finding about the transplant.

**A progressive-shrink arm is scored through its codec.** Its model does not
index host ids at all: rule 0 deletes the rows this corpus never exercised, rule
1 relabels case and space into stem + role, rule 2 re-pieces the frequency tail.
Handing such a model host ids would measure nothing, so the codec is rebuilt from
the checkpoint's ``stage`` + ``curriculum`` — by the same function
``train_transplant`` builds it with, never a second copy of the logic — and
applied to every encoded document. Two consequences that are easy to get quietly
wrong, and are therefore the two things most heavily commented below: the
continuation boundary has to be **recomputed** after the rewrite (rule 2 turns
one token into several), and a benchmark token whose row was deleted has to be
spelled out of the rows that remain rather than dropped.

Run::

    python -m experiments.vocab_factorization.hellaswag --native
    python -m experiments.vocab_factorization.hellaswag --ckpt <artifacts>/ckpt/bpe8k-t.pt
    python -m experiments.vocab_factorization.hellaswag --ckpt <artifacts>/ckpt/shrink1-fold.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from experiments.vocab_factorization import active_vocab, shrink, transplant
from experiments.vocab_factorization.build_arms import get_arm
from experiments.vocab_factorization.configs import (
    HOST_ARM,
    HOST_MODEL,
    SHRINKS_BY_NAME,
    TransplantTrainSpec,
)
from experiments.vocab_factorization.train_transplant import _reserve, build_shrink_plan

DATASET = "hellaswag"
SPLIT = "validation"
_BRACKETED = re.compile(r"\[.*?\]")


def preprocess(text: str) -> str:
    """The harness's canonical HellaSwag cleanup.

    Reproduced rather than invented: HellaSwag's raw text carries WikiHow
    markup ("[title]", "[step]", "[header]") that no model is meant to predict,
    and every published number applies exactly this normalization. Deviating
    here would make our accuracies incomparable to anyone else's, which is most
    of the point of using a standard benchmark.
    """
    text = text.strip().replace(" [title]", ". ")
    return _BRACKETED.sub("", text).replace("  ", " ")


def load_docs(limit: int = 0) -> list[dict]:
    """(query, four choices, gold) per validation example."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=SPLIT)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    docs = []
    for r in ds:
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        docs.append(
            {
                "query": preprocess(r["activity_label"] + ": " + ctx),
                "choices": [preprocess(e) for e in r["endings"]],
                "gold": int(r["label"]),
            }
        )
    return docs


class HostCodec:
    """Host ids straight into the model's heads. The six transplant arms' path.

    Their tables are indexed by the arm's own ids, so encoding *is* the whole
    job: the arrays are the encoding and the continuation boundary needs no
    translation. Kept as a class only so ``encode_docs`` does not have to know
    which kind of arm it is walking.
    """

    shrink = False

    def __init__(self, arm, heads: dict[str, int]) -> None:
        self.arm, self.heads = arm, heads

    def encode(self, text: str):
        return self.arm.encode(text)

    def source_len(self, enc) -> int:
        return int(enc.stem.size)

    def document(self, enc, src_ctx_len: int, cont_bytes: int) -> dict:
        return {
            "arrays": {n: getattr(enc, n).astype(np.int64) for n in self.heads},
            "ctx_len": src_ctx_len,
            "cont_bytes": cont_bytes,
        }

    def check(self, arrays: dict, raw: bytes) -> None:
        """Nothing to certify: these ids are the arm's own, unrewritten."""


class ShrinkCodec:
    """A shrink arm's rewritten vocabulary, applied to a benchmark document.

    The plan is rebuilt by ``train_transplant.build_shrink_plan`` — the function
    the training run itself used — so what is scored here is the vocabulary the
    checkpoint was trained against and not a lookalike.

    Three things this has to get right that the host path does not:

    * **the encoding.** A ``fold`` arm's stream is not ``arm.encode`` at all: it
      is ``fold_encode`` over the lowercased text, with the case read back off
      the original. The two curricula share stage numbers and produce different
      vocabularies at each, so the curriculum name decides which one runs.
    * **the continuation boundary.** See ``document``.
    * **tokens whose row was deleted.** See ``_pieces``.
    """

    shrink = True

    def __init__(self, plan, arm, folding: bool) -> None:
        self.plan, self.arm, self.folding = plan, arm, folding
        self.segmenter = shrink.Segmenter(plan.vocab)

    def encode(self, text: str):
        if self.folding:
            return shrink.fold_encode(text, self.arm.tok, self.plan.host)
        return self.arm.encode(text)

    def source_ids(self, enc) -> np.ndarray:
        return np.asarray(enc.ids if self.folding else enc.stem, dtype=np.int64)

    def source_len(self, enc) -> int:
        return int(enc.ids.size if self.folding else enc.stem.size)

    def document(self, enc, src_ctx_len: int, cont_bytes: int) -> dict:
        """Arrays, plus the continuation boundary **recomputed** in their index space.

        This is the subtle one. ``src_ctx_len`` counts *source* tokens, and under
        a re-piecing codec one source token becomes several — 14.07% more of them
        at ``shrink3-tail1k``, and not spread evenly, because the tokens that
        re-piece are the rare ones. Carrying the boundary across the rewrite
        would therefore land it somewhere inside the context, and every choice
        would be scored on a sum that includes context NLL: a number that looks
        entirely reasonable and is not the benchmark.

        The boundary is instead defined as "what the same codec does to the same
        prefix", which cannot drift from the array it indexes because it is the
        array's own prefix. It costs one extra codec pass over the context.
        """
        n = self.source_len(enc)
        cut = min(src_ctx_len, n)
        covered = self.plan.covers(self.source_ids(enc))
        return {
            "arrays": self._apply(enc, covered, 0, n),
            "ctx_len": int(self._apply(enc, covered, 0, cut)["stem"].size),
            "cont_bytes": cont_bytes,
            # The coverage diagnostic's raw material, counted here because it is
            # the only place that still knows which *source* tokens these arrays
            # came from. See ``codec_coverage``.
            "source_cont_tokens": n - cut,
            "source_cont_repieced": int((~covered[cut:]).sum()),
        }

    def check(self, arrays: dict, raw: bytes) -> None:
        """Spell the document back out and compare it to its own bytes.

        ``verify_fold`` certifies any codec-applied stream, not only a folded
        one: it re-spells every (stem, role) pair and diffs against the text. Run
        per document rather than argued about, because three separate things here
        can produce a plausible wrong stream — the fold encoder's case decision,
        the re-piecing, and the fallback splice below — and all three would leave
        ``acc_norm`` looking perfectly ordinary. ``cont_bytes`` is the accuracy's
        denominator, so a stream that does not spell exactly these bytes is
        normalizing by a length it does not have.
        """
        shrink.verify_fold(self.plan, arrays, raw)

    def _slice(self, enc, lo: int, hi: int) -> dict[str, np.ndarray]:
        """The codec over a run of source tokens it covers."""
        if self.folding:
            # Position-independent: every rung either relabels or re-pieces one
            # token at a time, so a slice's image is the image's slice.
            return self.plan.fold_stream(
                shrink.FoldEncoding(enc.ids[lo:hi], enc.case[lo:hi], 0)
            )
        return self.plan.apply_stream(np.asarray(enc.stem[lo:hi], dtype=np.int64))

    def _pieces(self, enc, i: int) -> list[tuple[int, int]]:
        """A token whose row was deleted, spelled out of the rows that remain.

        HellaSwag is English from a different distribution than this ru+en
        corpus, and 2,072 of its 3,166,583 host tokens (0.065%, 207 distinct
        surfaces — " mascara", " tablespoons", " walmart") never occur in it. A
        transplant arm scores those unmasked, because the row still exists; a
        shrink arm cannot, because rule 0 deleted it. Re-piecing is the arm's own
        answer to a surface it has no row for — it is literally rule 2 — so it
        measures the model rather than the workaround, and it preserves the byte
        stream that ``cont_bytes`` normalizes by.
        """
        stem = int(enc.ids[i]) if self.folding else int(enc.stem[i])
        raw = (
            shrink.fold_surface(self.plan.host, stem, int(enc.case[i]))
            if self.folding
            else self.plan.host.stems[stem]
        )
        pieces = self.segmenter(raw)
        if pieces is None:
            # Only reachable before a ``repiece_tail`` rung, where the byte floor
            # is still incomplete. Silence here would drop bytes the accuracy is
            # normalized by, so it is a stop rather than a warning.
            raise ValueError(
                f"{self.plan.host.stems[stem]!r} (host id {stem}) is outside this "
                f"arm's vocabulary and cannot be spelled out of its "
                f"{len(self.plan.vocab)} remaining stems either — the benchmark "
                "is not representable by this arm at all"
            )
        return pieces

    def _apply(
        self, enc, covered: np.ndarray, lo: int, hi: int
    ) -> dict[str, np.ndarray]:
        """The codec over ``[lo, hi)``, splicing the uncovered tokens' spellings in.

        Covered tokens go through the vectorized codec in runs; only the handful
        the codec has no image for cost a Python detour.
        """
        role_head = self.plan.vocab.n_roles > 1
        stems: list[np.ndarray] = []
        roles: list[np.ndarray] = []
        i = lo
        while i < hi:
            j = i
            while j < hi and covered[j]:
                j += 1
            if j > i:
                part = self._slice(enc, i, j)
                stems.append(part["stem"])
                if role_head:
                    roles.append(part["role"])
                i = j
            while i < hi and not covered[i]:
                pieces = self._pieces(enc, i)
                stems.append(np.fromiter((p[0] for p in pieces), np.int64, len(pieces)))
                if role_head:
                    roles.append(
                        np.fromiter((p[1] for p in pieces), np.int64, len(pieces))
                    )
                i += 1
        out = {"stem": _concat(stems)}
        if role_head:
            out["role"] = _concat(roles)
        return out


def _concat(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def build_codec(meta: dict, arm, args, tspec: TransplantTrainSpec):
    """What turns a document into the arrays this checkpoint's model indexes."""
    if not meta.get("stage", 0):
        return HostCodec(arm, meta["heads"])

    spec = _shrink_spec(meta)
    plan, fold, _ = build_shrink_plan(
        spec, arm, meta["host_vocab"], args, tspec, tspec.train_bytes
    )
    # The codec is derived, so nothing checks it against the model unless this
    # does. A plan built over a different budget or a drifted curriculum yields a
    # different vocabulary, and its ids would index the checkpoint's table
    # happily and mean something else. Head cardinalities are the cheap, total
    # statement that the two agree.
    if plan.heads() != meta["heads"]:
        raise ValueError(
            f"rebuilt codec for {meta['experiment']} has heads {plan.heads()} but "
            f"the checkpoint's table is {meta['heads']} — the plan does not "
            "describe this model, so scoring it would measure a different "
            "vocabulary"
        )
    return ShrinkCodec(plan, arm, fold is not None)


def _shrink_spec(meta: dict):
    """The arm's row in ``configs.SHRINKS``, cross-checked against the checkpoint.

    ``override_min_count`` is *not* in the checkpoint, and three arms
    (``shrink2-tail200``/``300``/``500``) differ from ``shrink2-tail100`` in
    nothing else — so configs stays the source of truth for what an arm is, and
    the ``stage``/``curriculum`` the checkpoint does carry are read back as the
    check that the row still describes the model on disk.
    """
    spec = SHRINKS_BY_NAME.get(meta["experiment"])
    if spec is None:
        raise KeyError(
            f"{meta['experiment']} is a stage-{meta['stage']} checkpoint with no "
            "row in configs.SHRINKS, so its curriculum's thresholds cannot be "
            "recovered — add the row rather than guessing them here"
        )
    got = (meta["stage"], meta.get("curriculum", "default"))
    if (spec.stage, spec.curriculum) != got:
        raise ValueError(
            f"configs.SHRINKS says {spec.name} is stage {spec.stage} of the "
            f"{spec.curriculum!r} curriculum; the checkpoint says {got}. The two "
            "curricula share stage numbers and produce different vocabularies at "
            "each, so this is not a naming quibble"
        )
    return spec


def encode_docs(codec, docs: list[dict]) -> list[dict]:
    """Tokenize each (context, ending) pair the way this arm's model reads text.

    The continuation boundary starts as ``len(encode(context))`` -- the context
    and the concatenation are encoded separately and the split is taken by
    length. For a BPE-family tokenizer the context's encoding is not guaranteed
    to be a strict prefix of the concatenation's, so a token straddling the join
    can shift the boundary by one. That is the same approximation every published
    harness makes, it is identical in kind for every arm here, and the
    alternative (scoring the context tokens too) would swamp the signal with
    context. What the codec must not do is *add* to that approximation, which is
    why ``ShrinkCodec.document`` recomputes rather than carries it.
    """
    out = []
    for d in docs:
        src_ctx_len = codec.source_len(codec.encode(d["query"]))
        seqs = []
        for choice in d["choices"]:
            text = d["query"] + " " + choice
            enc = codec.encode(text)
            seq = codec.document(enc, src_ctx_len, len((" " + choice).encode("utf-8")))
            codec.check(seq["arrays"], text.encode("utf-8"))
            seqs.append(seq)
        out.append({"gold": d["gold"], "seqs": seqs})
    return out


def _pad_batch(seqs: list[dict], heads: dict[str, int], device):
    """Right-pad to the batch's longest sequence.

    Padded target positions carry -100 so ``_ce_tokens``'s ``ignore_index``
    zeroes them; padded *input* positions carry 0, whose hidden state is
    computed and then discarded. Left-padding would be wrong here without an
    attention mask -- the frozen body is called with ``inputs_embeds`` only, so
    a pad on the left would sit inside the causal context of real tokens.
    """
    lens = [len(s["arrays"]["stem"]) for s in seqs]
    width = max(lens) - 1  # inputs are seq[:-1], targets seq[1:]
    inputs = {n: torch.zeros(len(seqs), width, dtype=torch.long) for n in heads}
    targets = {n: torch.full((len(seqs), width), -100, dtype=torch.long) for n in heads}
    keep = torch.zeros(len(seqs), width, dtype=torch.bool)
    for i, s in enumerate(seqs):
        n = lens[i] - 1
        for name in heads:
            a = torch.from_numpy(s["arrays"][name])
            inputs[name][i, :n] = a[:-1]
            targets[name][i, :n] = a[1:]
        # Target position j predicts token j+1, so the continuation starts at
        # j = ctx_len - 1.
        keep[i, max(0, s["ctx_len"] - 1) : n] = True
    return (
        {k: v.to(device) for k, v in inputs.items()},
        {k: v.to(device) for k, v in targets.items()},
        keep.to(device),
    )


@torch.no_grad()
def score(model, encoded: list[dict], heads: dict[str, int], device, batch: int = 8):
    """Summed continuation NLL per choice, over every head.

    Every head is summed, exactly as ``train.eval_bpb`` does. Scoring only the
    stem head would let a factorized arm move its case and morphology cost into
    a side head and appear to predict the continuation better than it does.
    """
    nll = np.zeros((len(encoded), 4), dtype=np.float64)
    flat = [(i, c) for i in range(len(encoded)) for c in range(4)]
    device = torch.device(device)
    for start in range(0, len(flat), batch):
        chunk = flat[start : start + batch]
        seqs = [encoded[i]["seqs"][c] for i, c in chunk]
        inputs, targets, keep = _pad_batch(seqs, heads, device)
        # Autocast only where the BPB harness uses it. On CPU (the tests) it is
        # off, so the arithmetic being checked is the arithmetic, not bf16.
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            per_head = model.token_nll(inputs, targets)
        total = sum(per_head.values())
        sums = (total * keep).sum(dim=1)
        for (i, c), v in zip(chunk, sums.tolist(), strict=True):
            nll[i, c] = v
    return nll


def accuracy(nll: np.ndarray, encoded: list[dict]) -> dict:
    gold = np.array([e["gold"] for e in encoded])
    cont_bytes = np.array(
        [[s["cont_bytes"] for s in e["seqs"]] for e in encoded], dtype=np.float64
    )
    return {
        "acc": float((nll.argmin(1) == gold).mean()),
        "acc_norm": float(((nll / cont_bytes).argmin(1) == gold).mean()),
        "n": len(encoded),
    }


def coverage(encoded: list[dict], ids: np.ndarray) -> dict:
    """How much of the benchmark each arm's active vocabulary can even represent.

    Reported instead of enforced -- see the module docstring. A high number here
    does not invalidate the accuracy above it (the mask is off when scoring), it
    says the arm's corpus-derived vocabulary does not cover the task, which is
    exactly the kind of thing a corpus-BPB headline cannot tell you.
    """
    total = outside = 0
    for e in encoded:
        for s in e["seqs"]:
            stem = s["arrays"]["stem"][s["ctx_len"] :]
            total += stem.size
            outside += int((~np.isin(stem, ids)).sum())
    return {
        "continuation_tokens": total,
        "tokens_outside_active": outside,
        "fraction_outside_active": outside / max(1, total),
    }


def codec_coverage(encoded: list[dict]) -> dict:
    """The same question asked of a vocabulary that has no mask to lift.

    "Fraction outside the active set" is meaningless for a shrink arm: the mask
    is subsumed by the codec, because rule 0 *deleted* the inactive rows instead
    of masking them, and every id the codec accepts is active by construction. So
    reporting that number would report 0.0000 forever while measuring nothing.

    The honest analogue is one rung earlier: how much of the benchmark this
    vocabulary has no row for at all. Those tokens are still scored -- spelled
    out of the rows that survive, see ``ShrinkCodec._pieces`` -- so like the
    unmasked fraction above this is reported rather than enforced, and like it,
    it is a finding about the arm rather than a scoring inconvenience. Counted in
    *source* tokens (host ids, or folded ids for a ``fold`` arm) because that is
    the space the question is asked in; ``continuation_tokens`` is what actually
    got scored, and the gap between them is the re-piecing.
    """
    source = sum(s["source_cont_tokens"] for e in encoded for s in e["seqs"])
    repieced = sum(s["source_cont_repieced"] for e in encoded for s in e["seqs"])
    scored = sum(
        int(s["arrays"]["stem"].size) - s["ctx_len"] for e in encoded for s in e["seqs"]
    )
    return {
        "mode": "codec",
        "continuation_tokens": scored,
        "source_continuation_tokens": source,
        "tokens_outside_codec": repieced,
        "fraction_outside_codec": repieced / max(1, source),
        "note": (
            "a shrink arm's inactive rows are deleted, not masked, so "
            "'outside the active vocabulary' is vacuous here; this counts "
            "benchmark tokens the codec has no row for, which are scored by "
            "being re-pieced out of the rows that remain"
        ),
    }


def build_native(device):
    """The untouched pretrained model, with no checkpoint involved.

    ``orig-frozen`` is ``train=False``, so it never writes one -- and it is the
    arm every other arm is compared against. Reconstructing it from the host
    table is therefore not a convenience path, it is the only way the reference
    exists at all.
    """
    body, host_emb, _tok, d_model = transplant.load_host(HOST_MODEL)
    heads = {"stem": host_emb.shape[0]}
    model = transplant.TransplantLM(body, heads, d_model, tie_embeddings=True).to(
        device
    )
    transplant.native_init(model, host_emb)
    del host_emb
    return (
        model,
        heads,
        # ``heads`` is carried in the metadata as well so this reconstruction
        # describes itself exactly as a checkpoint does, and ``build_codec`` can
        # read one shape for both.
        {
            "experiment": "orig-frozen",
            "arm": HOST_ARM,
            "init": "native",
            "steps": 0,
            "heads": heads,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", type=Path)
    g.add_argument("--native", action="store_true", help="score the untouched host")
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--arms-dir", type=Path, default=Path("data/arms"))
    ap.add_argument("--tokens-dir", type=Path, default=Path("data/tokens"))
    ap.add_argument("--out", type=Path, default=Path("results/hellaswag"))
    ap.add_argument("--limit", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--reserve-gb", type=float, default=0.0)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    _reserve(args.reserve_gb)
    t0 = time.time()

    tspec = TransplantTrainSpec()
    if args.native:
        model, heads, meta = build_native(device)
    else:
        from experiments.vocab_factorization.eval_transplant import load_checkpoint

        model, meta = load_checkpoint(args.ckpt, device)
        heads = meta["heads"]
    name = meta["experiment"]
    arm = get_arm(meta["arm"], args.arms_dir, args.corpus)
    print(f"[{name}] arm={meta['arm']} heads={heads}", flush=True)

    codec = build_codec(meta, arm, args, tspec)
    if codec.shrink:
        print(
            f"[{name}] codec rebuilt: stage {meta['stage']} of the "
            f"{meta.get('curriculum', 'default')!r} curriculum, "
            f"{[s.rule for s in codec.plan.steps]} -> {codec.plan.heads()}"
            f"{', folded encoding' if codec.folding else ''} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    docs = load_docs(args.limit)
    encoded = encode_docs(codec, docs)
    print(f"[{name}] {len(encoded)} docs encoded ({time.time() - t0:.0f}s)", flush=True)

    if codec.shrink:
        cov = codec_coverage(encoded)
        print(
            f"[{name}] outside the codec: {cov['tokens_outside_codec']}/"
            f"{cov['source_continuation_tokens']} continuation source tokens "
            f"({cov['fraction_outside_codec']:.2%}), re-pieced into "
            f"{cov['continuation_tokens']} scored tokens",
            flush=True,
        )
    else:
        ids = active_vocab.load_or_build(
            meta["arm"],
            arm.heads["stem"],
            args.corpus,
            args.arms_dir,
            args.tokens_dir,
            tspec.train_bytes,
            workers=args.workers,
        )
        cov = coverage(encoded, ids)
        print(
            f"[{name}] outside active vocab: {cov['tokens_outside_active']}/"
            f"{cov['continuation_tokens']} continuation tokens "
            f"({cov['fraction_outside_active']:.2%})",
            flush=True,
        )

    model.set_active_mask(None)  # explicit: the benchmark is scored unmasked
    nll = score(model, encoded, heads, device, args.batch)
    res = accuracy(nll, encoded)

    results = {
        "experiment": name,
        "arm": meta["arm"],
        "init": meta["init"],
        "steps": meta["steps"],
        "benchmark": f"{DATASET}/{SPLIT}",
        "masked": False,
        "checkpoint": None if args.native else str(args.ckpt),
        **res,
        "active_vocab_coverage": cov,
        "eval_seconds": round(time.time() - t0, 1),
    }
    if codec.shrink:
        # Emitted only for the arms it describes, so the six published
        # transplant artifacts stay byte-identical on a re-run.
        results["shrink"] = {
            "stage": meta["stage"],
            "curriculum": meta.get("curriculum", "default"),
            "heads": heads,
            "folded_encoding": codec.folding,
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{name}.json").write_text(json.dumps(results, indent=2))
    print(
        f"[{name}] acc={res['acc']:.4f} acc_norm={res['acc_norm']:.4f} "
        f"n={res['n']} ({time.time() - t0:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
