"""Generated problems: determinism, buckets, and the goal being false before the last move."""

from __future__ import annotations

from experiments.geostate.configs import GENERATOR, LENGTH_BUCKETS, TOL
from experiments.geostate.generator import (
    Problem,
    count_macros,
    generate,
    load_problems,
    save_problems,
)
from experiments.geostate.macros import object_from_params
from experiments.geostate.state import same_geometry


def test_generation_is_deterministic():
    a = generate(seed=7, per_bucket=3)
    b = generate(seed=7, per_bucket=3)
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b]
    c = generate(seed=8, per_bucket=3)
    assert [p.to_dict() for p in a] != [p.to_dict() for p in c]


def test_every_bucket_is_filled_and_capped():
    problems = generate(seed=1, per_bucket=4)
    counts = {name: 0 for name, _, _ in LENGTH_BUCKETS}
    for p in problems:
        counts[p.bucket] += 1
        assert p.n_moves == len(p.trajectory)
        assert p.n_moves <= GENERATOR.max_moves
        assert count_macros(p.expr) <= GENERATOR.max_macros
        assert p.description and not p.description.endswith(".")
    assert all(v == 4 for v in counts.values()), counts


def test_goal_false_before_last_move_and_true_after():
    """AC-7: replaying the trajectory reaches the target exactly at the last move."""
    for p in generate(seed=2, per_bucket=4):
        st = p.state()
        target = object_from_params(p.target_kind, p.target_params)
        for move in p.trajectory[:-1]:
            st.apply(move)
            assert not any(
                same_geometry(o, target, st.scale, TOL.coincidence) for o in st.objects
            )
        produced = st.apply(p.trajectory[-1])
        assert any(
            same_geometry(o, target, st.scale, TOL.coincidence) for o in produced
        )
        assert st.get(p.answer).kind == p.target_kind
        assert same_geometry(st.get(p.answer), target, st.scale, TOL.coincidence)
        for label in p.given:
            assert (
                not st.coincides(p.answer, label) if p.target_kind == "point" else True
            )


def test_problems_round_trip_through_jsonl(tmp_path):
    problems = generate(seed=5, per_bucket=2)
    path = save_problems(problems, tmp_path / "problems.jsonl")
    loaded = load_problems(path)
    assert [p.to_dict() for p in loaded] == [p.to_dict() for p in problems]
    assert isinstance(loaded[0], Problem)
    assert loaded[0].state().handles() == list(problems[0].given)
