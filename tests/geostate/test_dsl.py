"""Parsing and executing model-written construction programs."""

from __future__ import annotations

import pytest

from experiments.geostate import dsl
from experiments.geostate.generator import generate
from experiments.geostate.state import (
    ERR_DEGENERATE,
    ERR_TYPE,
    ERR_UNDEFINED,
    GeometricState,
)

TRI = {"A": (0.0, 0.0), "B": (8.0, 0.0), "C": (2.0, 6.0)}

PROGRAM = """
Let me construct the perpendicular bisector.
```construction
c1 = circle(A, B)   # radius AB
c2 = circle(B, A)
X, Y = meet(c1, c2)
bis = line(X, Y)
answer bis
```
"""


def test_parse_takes_last_fenced_block_and_skips_comments():
    text = "```\nignored = line(A, B)\n```\n" + PROGRAM
    program = dsl.parse(text)
    assert [s.text() for s in program.statements] == [
        "c1 = circle(A, B)",
        "c2 = circle(B, A)",
        "X, Y = meet(c1, c2)",
        "bis = line(X, Y)",
    ]
    assert program.answer == "bis"
    assert program.text().endswith("answer bis")


def test_execute_binds_names_and_returns_answer():
    ex = dsl.execute(dsl.parse(PROGRAM), GeometricState.from_given(TRI))
    assert ex.ok and ex.error is None
    assert ex.answer is not None and ex.answer.kind == "line"
    assert ex.n_moves == 4
    assert ex.bindings["X"] == "h3" and ex.bindings["bis"] == "h5"
    nx, ny, d = ex.answer.params()
    assert abs(nx * 4.0 + ny * 0.0 - d) < 1e-9  # through the midpoint of AB


@pytest.mark.parametrize(
    "text, kind, needle",
    [
        ("l = line(A, Q)\nanswer l", ERR_UNDEFINED, "unknown object 'Q'"),
        ("p = meet(A, B)\nanswer p", ERR_TYPE, "expected a line or circle"),
        ("c = circle(A, A)\nanswer c", ERR_DEGENERATE, "zero radius"),
        ("l = line(A, B)", dsl.ERR_NO_ANSWER, "no answer"),
        ("l = line(A, B)\nanswer m", ERR_UNDEFINED, "answer names unknown"),
        ("l := line(A, B)\nanswer l", dsl.ERR_SYNTAX, "cannot parse"),
        ("just prose, no program", dsl.ERR_SYNTAX, "cannot parse"),
        ("", dsl.ERR_EMPTY, "empty"),
    ],
)
def test_error_kinds(text, kind, needle):
    ex = dsl.run_text(text, GeometricState.from_given(TRI))
    assert not ex.ok
    assert ex.error_kind == kind
    assert needle in (ex.error or "")


def test_execution_stops_at_first_error_and_keeps_state():
    ex = dsl.run_text(
        "l = line(A, B)\nm = line(A, Z)\nk = line(B, C)\nanswer k",
        GeometricState.from_given(TRI),
    )
    assert ex.error_kind == ERR_UNDEFINED
    assert ex.n_moves == 1
    assert ex.failed_statement is not None and ex.failed_statement.line_no == 2
    assert "h1" in ex.state and "h2" not in ex.state


def test_reference_programs_reproduce_targets():
    for p in generate(seed=11, per_bucket=2):
        text = dsl.trajectory_program(p.trajectory, p.answer, p.state())
        ex = dsl.run_text(text, p.state())
        assert ex.ok, ex.error
        assert ex.answer is not None and ex.answer.handle == p.answer
        assert ex.answer.params() == pytest.approx(p.target_params)


def test_single_name_for_a_pair_binds_the_first():
    ex = dsl.run_text(
        "c1 = circle(A, B)\nc2 = circle(B, A)\nX = meet(c1, c2)\nanswer X",
        GeometricState.from_given(TRI),
    )
    assert ex.ok and ex.answer is not None and ex.answer.handle == "h3"
