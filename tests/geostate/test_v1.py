"""V1: invariant features, sequences, constrained rollouts, training step, report."""

from __future__ import annotations

import json
import math
import random

import numpy as np
import pytest

from experiments.geostate import cga, features, sequence
from experiments.geostate.configs import ARM_TABLE, ARM_TOKENS, V1, V1_GATE
from experiments.geostate.generator import generate
from experiments.geostate.state import GeometricState, Move

torch = pytest.importorskip("torch")
torch.set_num_threads(4)


def _similarity(rng: random.Random) -> cga.Similarity:
    return cga.Similarity(
        tx=rng.uniform(-6, 6),
        ty=rng.uniform(-6, 6),
        theta=rng.uniform(0, 2 * math.pi),
        scale=math.exp(rng.uniform(math.log(0.5), math.log(2.0))),
    )


@pytest.mark.parametrize("seed", range(5))
def test_features_are_invariant_under_similarities(seed):
    rng = random.Random(seed)
    problem = generate(seed=30 + seed, per_bucket=2)[-1]
    _, state = sequence.build_sequence(problem, ARM_TOKENS)
    moved = state.transform(_similarity(rng))
    for h in state.handles():
        s1, p1 = features.object_features(state, h)
        s2, p2 = features.object_features(moved, h)
        assert np.abs(s1 - s2).max() < 1e-9
        assert np.abs(p1 - p2).max() < 1e-9
        assert p1.shape[1] == features.PAIR_DIM and s1.shape == (features.SELF_DIM,)


def test_features_change_under_a_reflection_only_where_orientation_matters():
    st = GeometricState.from_given({"A": (0.0, 0.0), "B": (4.0, 0.0), "C": (1.0, 3.0)})
    st.apply(Move("line", ("A", "B")))
    refl = st.transform(cga.Similarity(reflect=True))
    _, p1 = features.object_features(st, "h1")
    _, p2 = features.object_features(refl, "h1")
    # the signed point-line distance of C flips sign under a reflection
    assert p1[2, 6] == pytest.approx(-p2[2, 6])
    assert abs(p1[2, 6]) == pytest.approx(3.0 / st.scale)


def test_sequences_share_targets_and_differ_only_in_state_blocks():
    problem = generate(seed=40, per_bucket=2)[2]
    seq_t, state_t = sequence.build_sequence(problem, ARM_TOKENS)
    seq_b, state_b = sequence.build_sequence(problem, ARM_TABLE)
    loss_t = [s.text for s in seq_t.segments if s.loss]
    loss_b = [s.text for s in seq_b.segments if s.loss]
    assert loss_t == loss_b
    assert loss_t[-1] == f"answer {problem.answer}\n"
    assert len(loss_t) == problem.n_moves + 1
    assert seq_t.text.count(sequence.ST) == len(state_t.objects)
    assert sequence.ST not in seq_b.text
    assert seq_t.state_handles == state_t.handles() == state_b.handles()
    # environment blocks carry no loss
    assert all(not s.loss for s in seq_t.segments if sequence.STATE_OPEN in s.text)
    assert "point (" in seq_b.text and "d=" in seq_b.text


def test_legal_moves_are_type_aware_and_distinct():
    from experiments.geostate.rollout import legal_moves

    st = GeometricState.from_given({"A": (0.0, 0.0), "B": (4.0, 0.0), "C": (1.0, 3.0)})
    st.apply(Move("line", ("A", "B")))
    st.apply(Move("circle", ("A", "B")))
    moves = legal_moves(st)
    assert "line(A, B)\n" in moves and "line(A, A)\n" not in moves
    assert "meet(h1, h2)\n" in moves and "meet(h1, h1)\n" not in moves
    assert "meet(A, h1)\n" not in moves and "line(h1, A)\n" not in moves
    assert "answer h2\n" in moves and "answer A\n" in moves
    assert len(moves) == 3 * 2 * 2 + 2 + 5


def _tiny_model(arm: str):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from experiments.geostate.v1_model import V1Model, load_tokenizer

    try:
        tok = load_tokenizer(V1)
    except OSError:
        pytest.skip("Qwen3-1.7B tokenizer not in the local HF cache")
    cfg = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=len(tok),
        max_position_embeddings=4096,
    )
    torch.manual_seed(0)
    lm = Qwen3ForCausalLM(cfg)
    return V1Model(V1, arm, device="cpu", dtype=torch.float32, lm=lm, tokenizer=tok)


def test_encoder_output_is_invariant_and_batched():
    from experiments.geostate.encoder import InvariantEncoder, pad_features

    problem = generate(seed=41, per_bucket=1)[1]
    _, state = sequence.build_sequence(problem, ARM_TOKENS)
    moved = state.transform(_similarity(random.Random(3)))
    enc = InvariantEncoder(16, 8)
    items = [
        tuple(torch.from_numpy(x) for x in features.object_features(state, h))
        for h in state.handles()
    ]
    items_m = [
        tuple(torch.from_numpy(x) for x in features.object_features(moved, h))
        for h in state.handles()
    ]
    out = enc(*pad_features(items))
    out_m = enc(*pad_features(items_m))
    assert out.shape == (len(items), 8)
    assert torch.allclose(out, out_m, atol=1e-5)
    assert torch.norm(out, dim=1).mean() < 1.0  # starts small next to token embeddings


def test_tiny_model_trains_both_arms_and_rolls_out(tmp_path):
    from experiments.geostate.rollout import MoveVocabulary, rollout
    from experiments.geostate.train_v1 import TrajectoryDataset, collate

    problems = generate(seed=42, per_bucket=2)
    for arm in (ARM_TOKENS, ARM_TABLE):
        model = _tiny_model(arm)
        ds = TrajectoryDataset(problems, arm, model.tok, V1)
        assert len(ds) == len(problems) and ds.dropped == 0
        batch = collate([ds[i] for i in range(3)], model.pad_id)
        n_st = int((batch["input_ids"] == model.st_id).sum())
        assert (n_st > 0) == (arm == ARM_TOKENS)
        assert n_st == batch["self_feats"].shape[0]
        # loss only where labels are set, i.e. on move lines
        assert (batch["labels"] != -100).sum() > 0
        loss = model(**batch)
        loss.backward()
        assert math.isfinite(float(loss))
        grads = [p.grad for g in model.param_groups() for p in g["params"]]
        assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)
        # adapters round-trip
        path = model.save_adapters(tmp_path / f"{arm}.pt")
        other = _tiny_model(arm)
        other.load_adapters(path)
        with pytest.raises(ValueError, match="arm"):
            _tiny_model(ARM_TABLE if arm == ARM_TOKENS else ARM_TOKENS).load_adapters(
                path
            )
        # a constrained rollout always ends with a legal outcome
        model.eval()
        ep = rollout(model, problems[0], V1, MoveVocabulary(model.tok))
        assert ep.error_kind in (None, "degenerate", "budget")
        assert ep.answered == (ep.error_kind is None)
        assert ep.tokens_generated > 0 and ep.tokens_processed > ep.tokens_generated
        for move in ep.moves:
            assert move.split("(")[0] in ("line", "circle", "meet")


def test_state_token_count_mismatch_is_an_error():
    model = _tiny_model(ARM_TOKENS)
    ids = torch.tensor([[1, model.st_id, 2, model.st_id]])
    with pytest.raises(ValueError, match="state tokens"):
        model.embed(ids, torch.zeros(1, model.lm.get_input_embeddings().embedding_dim))


def _fake_run(root, arm, seed, success_by_bucket):
    d = root / f"{arm}-s{seed}"
    d.mkdir(parents=True)
    (d / "train.json").write_text(json.dumps({"arm": arm, "seed": seed}))
    with (d / "eval.jsonl").open("w") as fh:
        for i in range(40):
            bucket = ["1-3", "4-9", "10+"][i % 3]
            n_ref = {"1-3": 2, "4-9": 6, "10+": 12}[bucket]
            ok = (i * 7 + seed) % 10 < success_by_bucket[bucket] * 10
            fh.write(
                json.dumps(
                    {
                        "problem_id": f"eval-{i}",
                        "bucket": bucket,
                        "n_moves_reference": n_ref,
                        "success": ok,
                        "target_reached": ok,
                        "answered": True,
                        "selection_error": False,
                        "error_kind": None if ok else "budget",
                        "n_moves": n_ref,
                        "tokens_processed": 300 if arm == ARM_TOKENS else 500,
                        "tokens_generated": 40,
                    }
                )
                + "\n"
            )


def test_report_pairs_problems_and_reads_the_gate(tmp_path, monkeypatch):
    from experiments.geostate import v1_report

    for seed in (1, 2):
        _fake_run(tmp_path, ARM_TOKENS, seed, {"1-3": 0.9, "4-9": 0.8, "10+": 0.7})
        _fake_run(tmp_path, ARM_TABLE, seed, {"1-3": 0.9, "4-9": 0.5, "10+": 0.3})
    payload, text = v1_report.report(tmp_path)
    assert sorted(payload["runs"]) == ["table/s1", "table/s2", "tokens/s1", "tokens/s2"]
    assert payload["summary"]["tokens/all"]["tokens_processed"] == 300
    gate = payload["gate"]
    assert gate["delta_pts"] > 10 and gate["ci_pts"][0] > 0
    assert gate["verdict"] == "withheld"  # 27 paired problems < n_eval
    monkeypatch.setattr(
        v1_report,
        "V1_GATE",
        type(V1_GATE)(min_moves=5, n_eval=10, go_delta_pts=10.0, kill_delta_pts=3.0),
    )
    payload, text = v1_report.report(tmp_path)
    assert payload["gate"]["verdict"] == "GO" and "GO" in text
    assert "| tokens | 2 | >=5 moves |" in text


def test_v1_launcher_payload():
    from experiments.geostate import run_v1_jobs as L

    plan = L.payload("abc", L.run_dir(V1, ARM_TOKENS, 1), V1, ARM_TOKENS, 1)
    job = plan["job"]
    assert job["description"] == "[gigaevo/geostate] v1 tokens s1"
    env = job["environment"]["variables"]
    assert (
        env["HF_HOME"].startswith("/workspace-SR004.nfs2/")
        and env["HF_HUB_OFFLINE"] == "1"
    )
    assert (
        "--arm tokens --seed 1" in job["script"] and "--train --eval" in job["script"]
    )
    assert "/workspace-SR004.nfs2/" in job["script"] and "--data-root" in job["script"]
    assert L.run_dir(V1, ARM_TOKENS, 1).name.startswith("tokens-s1-")
    assert L.run_dir(V1, ARM_TOKENS, 1) != L.run_dir(V1, ARM_TABLE, 1)
