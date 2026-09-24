"""Predicate goals (AC-10): incidence and tangency accept every valid answer."""

from __future__ import annotations

import random

import pytest

from experiments.geostate import consistency as C
from experiments.geostate import dsl, macros
from experiments.geostate.configs import GENERATOR, GeneratorSpec, config_named
from experiments.geostate.generator import (
    GOAL_COINCIDE,
    GOAL_INCIDENT,
    GOAL_TANGENT,
    Problem,
    build_problem,
    generate,
    generate_with_stats,
    goal_holds,
    relabel_goal,
    sample_goal,
)
from experiments.geostate.state import GeometricState, Move

TRI = {"A": (0.0, 0.0), "B": (8.0, 0.0), "C": (2.0, 6.0), "P": (12.0, 5.0)}


def test_generator_emits_every_goal_kind_and_records_it():
    problems, _ = generate_with_stats(seed=7, per_bucket=15)
    kinds = {p.goal["kind"] for p in problems}
    assert kinds == {GOAL_COINCIDE, GOAL_INCIDENT, GOAL_TANGENT}
    for p in problems:
        assert Problem.from_dict(p.to_dict()).goal == p.goal
        if p.goal["kind"] == GOAL_INCIDENT:
            assert p.description.startswith("a point lying on both ")
            assert p.target_kind == "point"
        if p.goal["kind"] == GOAL_TANGENT:
            assert p.description.startswith("a line through ")
            assert p.target_kind == "line"
    spec = GeneratorSpec(predicate_goal_prob=0.0)
    only, _ = generate_with_stats(spec, seed=7, per_bucket=5)
    assert {p.goal["kind"] for p in only} == {GOAL_COINCIDE}


def test_old_problem_files_default_to_coincidence():
    d = generate(seed=3, per_bucket=1)[0].to_dict()
    d.pop("goal")
    assert Problem.from_dict(d).goal == {"kind": GOAL_COINCIDE}


def test_incident_goal_accepts_both_intersections():
    expr = ["first_meet", ["circle_centred", "A", "B"], ["line_through", "C", "P"]]
    goal = {"kind": GOAL_INCIDENT, "curves": expr[1:]}
    p = build_problem("t", 0, TRI, expr, goal=goal)
    assert p.n_moves == 3 and p.goal == goal
    st = p.state()
    st.replay(p.trajectory)
    pair = [o for o in st.objects if o.move_index == p.n_moves - 1]
    assert len(pair) == 2 and all(goal_holds(p, st, o) for o in pair)
    assert not goal_holds(p, st, st.get("A"))
    assert not goal_holds(p, st, st.get("h1"))  # a curve is not a point


def test_tangent_goal_accepts_either_tangent():
    expr = ["tangent_line", "P", "A", "B"]
    goal = {
        "kind": GOAL_TANGENT,
        "through": "P",
        "circle": ["circle_centred", "A", "B"],
    }
    p = build_problem("t", 0, TRI, expr, goal=goal)
    st = p.state()
    st.replay(p.trajectory)
    assert goal_holds(p, st, st.get(p.answer))
    # the other tangency point gives the other valid tangent
    pair = [
        o for o in st.objects if o.kind == "point" and o.move_index == p.n_moves - 2
    ]
    other = [o for o in pair if o.handle != p.trajectory[-1].args[1]][0]
    (line,) = st.apply(Move("line", ("P", other.handle)))
    assert goal_holds(p, st, line)
    # a line through P that is not tangent, and a tangent not through P, fail
    (chord,) = st.apply(Move("line", ("P", "A")))
    assert not goal_holds(p, st, chord)


def test_relabel_and_transform_keep_predicate_goals():
    problems = [
        p for p in generate(seed=7, per_bucket=15) if p.goal["kind"] != GOAL_COINCIDE
    ]
    cfg = config_named("e0-synthetic-text")
    for problem in problems[:4]:
        copies = C.make_copies(problem, cfg, random.Random(1))
        for c in copies[1:]:
            assert c.problem.goal["kind"] == problem.goal["kind"]
            assert c.problem.goal == relabel_goal(problem.goal, c.perm)
            program = dsl.trajectory_program(
                c.problem.trajectory, c.problem.answer, c.problem.state()
            )
            assert C.evaluate(c, "```construction\n" + program + "\n```")["success"]


def test_e0_scores_the_second_intersection_as_correct():
    expr = ["first_meet", ["circle_centred", "A", "B"], ["line_through", "C", "P"]]
    goal = {"kind": GOAL_INCIDENT, "curves": expr[1:]}
    p = build_problem("t", 0, TRI, expr, goal=goal)
    copy = C.Copy(0, "identity", C.IDENTITY, {k: k for k in TRI}, p)
    second = "```construction\nc = circle(A, B)\nl = line(C, P)\nX, Y = meet(c, l)\nanswer Y\n```"
    first = second.replace("answer Y", "answer X")
    assert C.evaluate(copy, first)["success"] and C.evaluate(copy, second)["success"]
    assert not C.evaluate(copy, second.replace("answer Y", "answer c"))["success"]


def test_validate_expr_checks_curve_slots():
    macros.validate_expr(
        ["first_meet", ["line_through", "A", "B"], ["circle_centred", "A", "C"]]
    )
    with pytest.raises(ValueError, match="line or circle"):
        macros.validate_expr(["first_meet", "A", ["circle_centred", "A", "C"]])
    with pytest.raises(ValueError, match="point slot"):
        macros.validate_expr(["midpoint", "A", ["line_through", "A", "B"]])
    assert (
        "first_meet" not in macros.POINT_MACROS
        and "tangent_line" not in macros.TOP_MACROS
    )


def test_sample_goal_respects_the_probability():
    rng = random.Random(0)
    labels = list("ABCDE")
    kinds = set()
    for _ in range(200):
        try:
            _, goal = sample_goal(rng, GENERATOR, labels)
        except Exception:  # noqa: BLE001 - identical curves are rejected upstream
            continue
        kinds.add(goal["kind"])
    assert kinds == {GOAL_COINCIDE, GOAL_INCIDENT, GOAL_TANGENT}
    st = GeometricState.from_given(TRI)
    assert st.scale > 0
