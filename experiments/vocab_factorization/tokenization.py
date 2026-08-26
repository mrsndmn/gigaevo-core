"""Tokenizer arms for the vocabulary-factorization study.

Every arm implements the same contract:

    encode(text) -> Encoding     # parallel int arrays, one row per position
    decode(Encoding) -> str      # MUST satisfy decode(encode(x)) == x
    heads                        # {head_name: cardinality} for the LM heads

``heads`` is what makes bits-per-byte comparable across arms: the model puts a
softmax on every head, and the BPB sum includes every head's NLL. An arm that
moves information out of the stem vocabulary into a feature head does not get
that information for free -- it still has to predict it.

Arms
----
``BPEArm``          byte-level BPE at a given vocab size (the baselines,
                    including the "just train a smaller BPE" null hypothesis)
``ByteArm``         raw UTF-8 bytes (robustness ceiling, compression floor)
``FactorizedArm``   stem vocabulary + (space, case) role head + suffix head
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from experiments.vocab_factorization.surface import (
    CASE_LOWER,
    CASE_TITLE,
    CASE_UPPER,
    KIND_WORD,
    rebuild_word,
    segment,
)

# ---------------------------------------------------------------------------
# Role head: word-start (with its space/case combo), word-continuation, or raw.
# ---------------------------------------------------------------------------
# 6 word-start combos = space(2) x case(3, OTHER never reaches this path)
ROLE_CONT = 6
ROLE_RAW = 7
N_ROLE = 8
_CASE_TO_SLOT = {CASE_LOWER: 0, CASE_TITLE: 1, CASE_UPPER: 2}
_SLOT_TO_CASE = {0: CASE_LOWER, 1: CASE_TITLE, 2: CASE_UPPER}


def role_of(space: int, case: int) -> int:
    return space * 3 + _CASE_TO_SLOT[case]


def role_to_space_case(role: int) -> tuple[int, int]:
    return role // 3, _SLOT_TO_CASE[role % 3]


SUFFIX_NONE = 0  # id 0 of the suffix head means "no suffix"


@dataclass(slots=True)
class Encoding:
    """Parallel arrays; all the same length T."""

    stem: np.ndarray  # int32, the piece id
    role: np.ndarray  # int8
    suffix: np.ndarray  # int16

    def __len__(self) -> int:
        return int(self.stem.shape[0])


# ---------------------------------------------------------------------------
# Shared BPE plumbing
# ---------------------------------------------------------------------------


def _new_bpe(use_regex: bool):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    tok = Tokenizer(models.BPE(unk_token=None, fuse_unk=False))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=use_regex
    )
    tok.decoder = decoders.ByteLevel()
    return tok


def _train_bpe(tok, corpus_iter: Iterable[str], vocab_size: int):
    from tokenizers import pre_tokenizers, trainers

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=False,
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus_iter, trainer=trainer)
    return tok


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


class BPEArm:
    """Byte-level BPE. One head, the standard setup."""

    kind = "bpe"

    def __init__(self, tok, name: str):
        self.tok = tok
        self.name = name
        self.vocab_size = tok.get_vocab_size()

    @classmethod
    def train(cls, corpus: str, vocab_size: int, name: str | None = None) -> BPEArm:
        tok = _new_bpe(use_regex=True)
        _train_bpe(tok, _lines(corpus), vocab_size)
        return cls(tok, name or f"bpe{vocab_size // 1000}k")

    @property
    def heads(self) -> dict[str, int]:
        return {"stem": self.vocab_size}

    def encode(self, text: str) -> Encoding:
        ids = self.tok.encode(text, add_special_tokens=False).ids
        n = len(ids)
        return Encoding(
            stem=np.asarray(ids, dtype=np.int32),
            role=np.zeros(n, dtype=np.int8),
            suffix=np.zeros(n, dtype=np.int16),
        )

    def decode(self, enc: Encoding) -> str:
        return self.tok.decode(enc.stem.tolist(), skip_special_tokens=False)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(path / "bpe.json"))
        (path / "arm.json").write_text(json.dumps({"kind": "bpe", "name": self.name}))

    @classmethod
    def load(cls, path: Path) -> BPEArm:
        from tokenizers import Tokenizer

        meta = json.loads((path / "arm.json").read_text())
        return cls(Tokenizer.from_file(str(path / "bpe.json")), meta["name"])


class ByteArm:
    """Raw UTF-8 bytes."""

    kind = "byte"
    name = "byte"
    vocab_size = 256

    @property
    def heads(self) -> dict[str, int]:
        return {"stem": 256}

    def encode(self, text: str) -> Encoding:
        b = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
        n = b.shape[0]
        return Encoding(
            stem=b, role=np.zeros(n, dtype=np.int8), suffix=np.zeros(n, dtype=np.int16)
        )

    def decode(self, enc: Encoding) -> str:
        return bytes(enc.stem.astype(np.uint8).tolist()).decode(
            "utf-8", errors="strict"
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "arm.json").write_text(json.dumps({"kind": "byte", "name": "byte"}))

    @classmethod
    def load(cls, path: Path) -> ByteArm:
        return cls()


class FactorizedArm:
    """Stem vocabulary + role head (space x case) + suffix head.

    Encoding of one word:

    * feature extraction happens in ``surface.segment`` (lossless, verified)
    * the lowercased core is split ``stem + suffix`` if some inventory suffix
      leaves a stem that the BPE encodes as a *single* piece; otherwise the
      whole core is BPE'd and no suffix is emitted
    * ``role`` marks the first piece with (space, case); later pieces get
      ``ROLE_CONT``; non-word material gets ``ROLE_RAW``

    The suffix rides on the *last* piece of the word, so multi-piece stems can
    still carry one.
    """

    kind = "factorized"

    def __init__(self, tok, suffixes: list[str], name: str, use_suffix: bool = True):
        self.tok = tok
        self.suffixes = suffixes
        self.name = name
        self.use_suffix = use_suffix
        self.vocab_size = tok.get_vocab_size()
        self._suffix_id = {s: i + 1 for i, s in enumerate(suffixes)}
        self._by_len: dict[int, set[str]] = {}
        for s in suffixes:
            self._by_len.setdefault(len(s), set()).add(s)
        self._lens = sorted(self._by_len, reverse=True)
        self._vocab = tok.get_vocab()
        self._single: dict[str, int] = {}  # core -> id, for cores that are 1 piece

    @property
    def heads(self) -> dict[str, int]:
        h = {"stem": self.vocab_size, "role": N_ROLE}
        if self.use_suffix:
            h["suffix"] = len(self.suffixes) + 1
        return h

    # -- construction -------------------------------------------------------

    @classmethod
    def train(
        cls,
        corpus: str,
        vocab_size: int,
        n_suffix: int = 256,
        use_suffix: bool = True,
        min_distinct_stems: int = 50,
        name: str | None = None,
    ) -> FactorizedArm:
        word_freq, raw_freq = count_spans(corpus)

        suffixes = (
            _build_suffix_inventory(word_freq, n_suffix, min_distinct_stems)
            if use_suffix
            else []
        )
        suffix_set = set(suffixes)

        # Stem BPE is trained on suffix-stripped cores plus raw spans, each as
        # its own line so merges never cross a span boundary.
        # Feed the trainer the type inventory with capped multiplicity: the
        # cap keeps the iterator small while preserving the rough frequency
        # ordering that decides which merges are learned.
        def lines() -> Iterable[str]:
            for w, f in word_freq.items():
                stem = _strip_suffix(w, suffix_set) if use_suffix else w
                for _ in range(min(f, 32)):
                    yield stem
            for r, f in raw_freq.items():
                for _ in range(min(f, 32)):
                    yield r

        tok = _new_bpe(use_regex=False)
        _train_bpe(tok, lines(), vocab_size)
        suffix_tag = "" if use_suffix else "-nosuf"
        return cls(
            tok,
            suffixes,
            name or f"fact{vocab_size // 1000}k{suffix_tag}",
            use_suffix=use_suffix,
        )

    # -- encode / decode ----------------------------------------------------

    def _pieces(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False).ids

    def _single_piece(self, s: str) -> int | None:
        hit = self._single.get(s)
        if hit is not None:
            return hit
        vid = self._vocab.get(_to_bytelevel(s))
        if vid is not None:
            self._single[s] = vid
        return vid

    def _split_word(self, core: str) -> tuple[list[int], int]:
        """Return (piece ids, suffix id) for a lowercased word core."""
        if self.use_suffix:
            for ln in self._lens:
                if len(core) - ln < 3:
                    continue
                suf = core[-ln:]
                if suf in self._by_len[ln]:
                    stem = core[:-ln]
                    vid = self._single_piece(stem)
                    if vid is not None:
                        return [vid], self._suffix_id[suf]
        return self._pieces(core), SUFFIX_NONE

    def encode(self, text: str) -> Encoding:
        stem: list[int] = []
        role: list[int] = []
        suffix: list[int] = []
        for sp in segment(text):
            if sp.kind == KIND_WORD:
                pieces, suf = self._split_word(sp.core)
                if not pieces:  # empty core cannot happen, but stay total
                    continue
                stem.extend(pieces)
                role.append(role_of(sp.space, sp.case))
                role.extend([ROLE_CONT] * (len(pieces) - 1))
                suffix.extend([SUFFIX_NONE] * (len(pieces) - 1))
                suffix.append(suf)
            else:
                pieces = self._pieces(sp.text)
                stem.extend(pieces)
                role.extend([ROLE_RAW] * len(pieces))
                suffix.extend([SUFFIX_NONE] * len(pieces))
        return Encoding(
            stem=np.asarray(stem, dtype=np.int32),
            role=np.asarray(role, dtype=np.int8),
            suffix=np.asarray(suffix, dtype=np.int16),
        )

    def decode(self, enc: Encoding) -> str:
        out: list[str] = []
        buf: list[int] = []
        cur_space = 0
        cur_case = CASE_LOWER
        cur_suffix = SUFFIX_NONE
        in_word = False

        def flush() -> None:
            nonlocal in_word, cur_suffix
            if not in_word:
                return
            core = self.tok.decode(buf, skip_special_tokens=False)
            if cur_suffix != SUFFIX_NONE:
                core += self.suffixes[cur_suffix - 1]
            out.append(rebuild_word(cur_space, cur_case, core))
            buf.clear()
            cur_suffix = SUFFIX_NONE
            in_word = False

        raw_buf: list[int] = []

        def flush_raw() -> None:
            if raw_buf:
                out.append(self.tok.decode(raw_buf, skip_special_tokens=False))
                raw_buf.clear()

        for tid, r, sf in zip(
            enc.stem.tolist(), enc.role.tolist(), enc.suffix.tolist(), strict=True
        ):
            if r == ROLE_RAW:
                flush()
                raw_buf.append(tid)
                continue
            flush_raw()
            if r == ROLE_CONT:
                if not in_word:  # defensive: stray CONT starts a lower word
                    in_word, cur_space, cur_case = True, 0, CASE_LOWER
                buf.append(tid)
            else:
                flush()
                cur_space, cur_case = role_to_space_case(r)
                in_word = True
                buf.append(tid)
            if sf != SUFFIX_NONE:
                cur_suffix = sf
        flush()
        flush_raw()
        return "".join(out)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(path / "bpe.json"))
        (path / "arm.json").write_text(
            json.dumps(
                {
                    "kind": "factorized",
                    "name": self.name,
                    "suffixes": self.suffixes,
                    "use_suffix": self.use_suffix,
                },
                ensure_ascii=False,
            )
        )

    @classmethod
    def load(cls, path: Path) -> FactorizedArm:
        from tokenizers import Tokenizer

        meta = json.loads((path / "arm.json").read_text())
        return cls(
            Tokenizer.from_file(str(path / "bpe.json")),
            meta["suffixes"],
            meta["name"],
            use_suffix=meta.get("use_suffix", True),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BYTE_TO_UNICODE: dict[int, str] | None = None


def _bytelevel_map() -> dict[int, str]:
    """Reproduce the ByteLevel byte->char alphabet used by the tokenizers lib."""
    global _BYTE_TO_UNICODE
    if _BYTE_TO_UNICODE is None:
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("\xa1"), ord("\xac") + 1))
            + list(range(ord("\xae"), ord("\xff") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        _BYTE_TO_UNICODE = {b: chr(c) for b, c in zip(bs, cs, strict=True)}
    return _BYTE_TO_UNICODE


def _to_bytelevel(s: str) -> str:
    m = _bytelevel_map()
    return "".join(m[b] for b in s.encode("utf-8"))


def iter_chunks(text: str, size: int = 1 << 20) -> Iterable[str]:
    """Yield ~``size``-char chunks split on newline boundaries.

    Segmenting a 300 MB string in one call would materialize tens of millions
    of Span objects; every corpus-scale pass goes through here instead.
    """
    start = 0
    n = len(text)
    while start < n:
        end = text.find("\n", start + size)
        if end == -1:
            end = n
        yield text[start : end + 1]
        start = end + 1


def count_spans(corpus: str) -> tuple[Counter, Counter]:
    """(word core frequencies, raw span frequencies) over a corpus."""
    word_freq: Counter[str] = Counter()
    raw_freq: Counter[str] = Counter()
    for chunk in iter_chunks(corpus):
        for s in segment(chunk):
            if s.kind == KIND_WORD:
                word_freq[s.core] += 1
            else:
                raw_freq[s.text] += 1
    return word_freq, raw_freq


def _lines(corpus: str, chunk: int = 8192) -> Iterable[str]:
    """Feed the trainer reasonably sized chunks split on newlines."""
    start = 0
    n = len(corpus)
    while start < n:
        end = corpus.find("\n", start + chunk)
        if end == -1:
            end = n
        yield corpus[start:end]
        start = end + 1


def _strip_suffix(word: str, suffix_set: set[str]) -> str:
    for ln in range(5, 0, -1):
        if len(word) - ln >= 3 and word[-ln:] in suffix_set:
            return word[:-ln]
    return word


def _build_suffix_inventory(
    word_freq: Counter[str], n_suffix: int, min_distinct_stems: int
) -> list[str]:
    """Pick orthographic endings that many distinct stems share.

    Frequency alone would pick up accidental endings; requiring a suffix to
    attach to many *distinct* stems is what makes the inventory morphological
    rather than a list of common letter pairs.
    """
    tok_count: Counter[str] = Counter()
    stems: dict[str, set[str]] = {}
    for w, f in word_freq.items():
        for ln in range(1, 6):
            if len(w) - ln < 3:
                break
            suf = w[-ln:]
            tok_count[suf] += f
            stems.setdefault(suf, set()).add(w[:-ln])
    scored = [
        (tok_count[s] * (len(stems[s]) ** 0.5), s)
        for s in tok_count
        if len(stems[s]) >= min_distinct_stems
    ]
    scored.sort(reverse=True)
    return [s for _, s in scored[:n_suffix]]


def load_arm(path: Path):
    kind = json.loads((path / "arm.json").read_text())["kind"]
    return {"bpe": BPEArm, "byte": ByteArm, "factorized": FactorizedArm}[kind].load(
        path
    )
