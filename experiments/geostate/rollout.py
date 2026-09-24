"""Constrained rollouts of a V1 model in the exact environment.

At every step the model may only emit one of the legal moves of the current
state — `line`/`circle` over two distinct existing points, `meet` over two
distinct existing curves, or `answer` over any existing handle — decoded
greedily under a token trie built from those strings. Both arms are decoded
the same way, so a reference to a missing object is impossible in either and
what remains measurable is *which* legal object the model picks (H1). After
a move the environment appends the new objects' state block (tokens or text)
and decoding continues; a degenerate move or an exhausted move budget ends
the episode without an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .client import write_json
from .configs import ARM_TOKENS, LENGTH_BUCKETS, TOL, V1_GATE, ToleranceSpec, V1Config
from .encoder import pad_features
from .features import object_features
from .generator import Problem, goal_holds
from .sequence import prompt_text, state_block
from .state import CIRCLE, LINE, POINT, GeometricState, Move, MoveError

END = -1


class Trie:
    def __init__(self):
        self.root: dict = {}

    def add(self, ids: tuple[int, ...]) -> None:
        node = self.root
        for i in ids:
            node = node.setdefault(i, {})
        node[END] = {}

    def node(self, prefix: list[int]) -> dict | None:
        node = self.root
        for i in prefix:
            node = node.get(i)
            if node is None:
                return None
        return node


class MoveVocabulary:
    """Tokenisation of candidate move strings, cached across steps and problems."""

    def __init__(self, tok):
        self.tok = tok
        self.cache: dict[str, tuple[int, ...]] = {}

    def ids(self, strings: list[str]) -> list[tuple[int, ...]]:
        missing = [s for s in strings if s not in self.cache]
        if missing:
            encoded = self.tok(missing, add_special_tokens=False)["input_ids"]
            for s, ids in zip(missing, encoded, strict=True):
                self.cache[s] = tuple(int(i) for i in ids)
        return [self.cache[s] for s in strings]


def legal_moves(state: GeometricState) -> list[str]:
    points = [o.handle for o in state.objects if o.kind == POINT]
    curves = [o.handle for o in state.objects if o.kind in (LINE, CIRCLE)]
    out: list[str] = []
    for a in points:
        for b in points:
            if a != b:
                out.append(f"line({a}, {b})\n")
                out.append(f"circle({a}, {b})\n")
    for a in curves:
        for b in curves:
            if a != b:
                out.append(f"meet({a}, {b})\n")
    out.extend(f"answer {o.handle}\n" for o in state.objects)
    return out


def build_trie(vocab: MoveVocabulary, moves: list[str]) -> Trie:
    trie = Trie()
    for ids in vocab.ids(moves):
        trie.add(ids)
    return trie


def move_budget(cfg: V1Config, n_reference: int) -> int:
    return min(
        cfg.move_budget_cap,
        int(cfg.move_budget_factor * n_reference) + cfg.move_budget_extra,
    )


@dataclass
class Episode:
    problem_id: str
    bucket: str
    n_moves_reference: int
    success: bool = False
    target_reached: bool = False
    answered: bool = False
    selection_error: bool = False
    error_kind: str | None = None
    n_moves: int = 0
    tokens_processed: int = 0
    tokens_generated: int = 0
    moves: list[str] = field(default_factory=list)
    answer: str | None = None
    wall_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _features_for(state: GeometricState, handles: list[str]):
    items = [
        (torch.from_numpy(s), torch.from_numpy(p), torch.from_numpy(r))
        for s, p, r in (object_features(state, h) for h in handles)
    ]
    return pad_features(items)


def answer_moves(state: GeometricState) -> list[str]:
    return [f"answer {o.handle}\n" for o in state.objects]


@torch.no_grad()
def rollout(
    model,
    problem: Problem,
    cfg: V1Config,
    vocab: MoveVocabulary,
    tol: ToleranceSpec = TOL,
) -> Episode:
    from transformers import DynamicCache

    started = time.monotonic()
    tok = model.tok
    state = problem.state(tol)
    ep = Episode(problem.id, problem.bucket, problem.n_moves)
    cache = DynamicCache()
    arm = model.arm

    def feed(ids: list[int], handles: list[str] | None) -> torch.Tensor:
        st = None
        if arm == ARM_TOKENS and handles:
            st = model.state_embeddings(*_features_for(state, handles))
        embeds = model.embed(torch.tensor([ids], device=model.device), st)
        out = model.lm(inputs_embeds=embeds, past_key_values=cache, use_cache=True)
        return out.logits[0, -1].float()

    given = [o.handle for o in state.objects]
    ids = tok(prompt_text(problem), add_special_tokens=False)["input_ids"]
    block = tok(state_block(list(state.objects), arm), add_special_tokens=False)[
        "input_ids"
    ]
    ep.tokens_processed += len(ids) + len(block)
    logits = feed(list(ids) + list(block), given)
    budget = move_budget(cfg, problem.n_moves)

    must_answer = False  # budget spent: the model may still name an object
    while True:
        candidates = answer_moves(state) if must_answer else legal_moves(state)
        trie = build_trie(vocab, candidates)
        prefix: list[int] = []
        pending: list[int] = []  # generated but not yet fed to the model
        for _ in range(cfg.max_move_tokens):
            node = trie.node(prefix)
            if node is None:
                break
            allowed = [i for i in node if i != END]
            if not allowed:
                break
            masked = torch.full_like(logits, float("-inf"))
            idx = torch.tensor(allowed, device=logits.device)
            masked[idx] = logits[idx]
            nxt = int(torch.argmax(masked).item())
            prefix.append(nxt)
            ep.tokens_generated += 1
            ep.tokens_processed += 1
            node = trie.node(prefix)
            if node is not None and END in node and len(node) == 1:
                pending = [nxt]  # the move is complete; fed with the next block
                break
            logits = feed([nxt], None)
        text = tok.decode(prefix).strip()
        if text.startswith("answer "):
            handle = text.split(" ", 1)[1].strip()
            ep.answered = True
            ep.answer = handle
            try:
                obj = state.get(handle)
            except MoveError:
                ep.error_kind = "undefined_reference"
                break
            ep.success = goal_holds(problem, state, obj, tol)
            break
        try:
            op, rest = text.split("(", 1)
            a, b = [x.strip() for x in rest.rstrip(")").split(",")]
            produced = state.apply(Move(op.strip(), (a, b)))
        except MoveError as exc:
            ep.error_kind = exc.kind
            ep.moves.append(text)
            break
        except ValueError:
            ep.error_kind = "syntax"
            break
        ep.moves.append(text)
        ep.n_moves += 1
        block_text = state_block(produced, arm)
        block = tok(block_text, add_special_tokens=False)["input_ids"]
        ep.tokens_processed += len(block)
        if ep.n_moves >= budget:
            # one last turn, answers only; a wrong or missing answer is still
            # recorded under "budget" so the failure mode stays visible
            must_answer = True
            ep.error_kind = "budget"
        # the move's last token and the new state block enter the cache together
        logits = feed(pending + list(block), [o.handle for o in produced])
    ep.target_reached = any(goal_holds(problem, state, o, tol) for o in state.objects)
    ep.selection_error = bool(ep.answered and ep.target_reached and not ep.success)
    ep.wall_s = time.monotonic() - started
    return ep


def _rate(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def metrics_from(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    buckets = [n for n, _, _ in LENGTH_BUCKETS] + [
        "all",
        f">={V1_GATE.min_moves} moves",
    ]
    for bucket in buckets:
        if bucket == "all":
            sel = episodes
        elif bucket.startswith(">="):
            sel = [e for e in episodes if e["n_moves_reference"] >= V1_GATE.min_moves]
        else:
            sel = [e for e in episodes if e["bucket"] == bucket]
        if not sel:
            continue
        out[bucket] = {
            "n": len(sel),
            "success": _rate([float(e["success"]) for e in sel]),
            "target_reached": _rate([float(e["target_reached"]) for e in sel]),
            "selection_error": _rate([float(e["selection_error"]) for e in sel]),
            "answered": _rate([float(e["answered"]) for e in sel]),
            "degenerate": _rate([float(e["error_kind"] == "degenerate") for e in sel]),
            "budget": _rate([float(e["error_kind"] == "budget") for e in sel]),
            "tokens_processed": _rate([float(e["tokens_processed"]) for e in sel]),
            "tokens_generated": _rate([float(e["tokens_generated"]) for e in sel]),
            "moves": _rate([float(e["n_moves"]) for e in sel]),
        }
    return out


def evaluate(
    model, problems: list[Problem], cfg: V1Config, out: Path, tol: ToleranceSpec = TOL
) -> dict[str, Any]:
    """Roll out every problem, resuming from `out/eval.jsonl`; write metrics."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "eval.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from an interrupted run
            done[rec["problem_id"]] = rec
    vocab = MoveVocabulary(model.tok)
    model.eval()
    with path.open("a") as fh:
        for i, problem in enumerate(problems, start=1):
            if problem.id in done:
                continue
            ep = rollout(model, problem, cfg, vocab, tol).to_dict()
            done[problem.id] = ep
            fh.write(json.dumps(ep) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(problems):
                print(f"  eval {i}/{len(problems)}", flush=True)
    episodes = [done[p.id] for p in problems if p.id in done]
    metrics = {
        "arm": model.arm,
        "n": len(episodes),
        "n_expected": len(problems),
        "complete": len(episodes) == len(problems),
        "buckets": metrics_from(episodes),
    }
    write_json(out / "metrics.json", metrics)
    return metrics
