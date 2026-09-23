"""Moves, ordering, predicates and the equivariance of the transition (AC-5, AC-6)."""

from __future__ import annotations

import math
import random

import pytest

from experiments.geostate import cga
from experiments.geostate.configs import TOL
from experiments.geostate.state import (
    CIRCLE,
    ERR_DEGENERATE,
    ERR_TYPE,
    ERR_UNDEFINED,
    LINE,
    POINT,
    GeometricState,
    Move,
    MoveError,
    incident,
    relabel_move,
    same_geometry,
    tangent,
)

TRIANGLE = {"A": (0.0, 0.0), "B": (8.0, 0.0), "C": (2.0, 6.0)}


def test_given_points_and_handles():
    st = GeometricState.from_given(TRIANGLE)
    assert st.handles() == ["A", "B", "C"]
    assert st.scale == pytest.approx(math.hypot(6.0, 6.0))
    (line,) = st.apply(Move("line", ("A", "B")))
    assert line.handle == "h1" and line.kind == LINE
    (circle,) = st.apply(Move("circle", ("A", "B")))
    assert circle.handle == "h2" and circle.kind == CIRCLE
    assert circle.params() == pytest.approx((0.0, 0.0, 8.0))
    pts = st.apply(Move("meet", ("h2", "h1")))
    assert [p.handle for p in pts] == ["h3", "h4"]
    assert st.n_moves == 3
    assert "h3" in st and "zzz" not in st


def test_errors_are_typed():
    st = GeometricState.from_given(TRIANGLE)
    with pytest.raises(MoveError) as e:
        st.apply(Move("line", ("A", "Q")))
    assert e.value.kind == ERR_UNDEFINED
    with pytest.raises(MoveError) as e:
        st.apply(Move("meet", ("A", "B")))
    assert e.value.kind == ERR_TYPE
    with pytest.raises(MoveError) as e:
        st.apply(Move("circle", ("A", "A")))
    assert e.value.kind == ERR_DEGENERATE
    st.apply(Move("line", ("A", "B")))
    with pytest.raises(MoveError) as e:  # a line with itself
        st.apply(Move("meet", ("h1", "h1")))
    assert e.value.kind == ERR_DEGENERATE
    with pytest.raises(MoveError) as e:
        st.apply(Move("frobnicate", ("A", "B")))
    assert e.value.kind == ERR_TYPE
    assert st.n_moves == 1  # failed moves do not count


def test_degenerate_meets():
    st = GeometricState.from_given(
        {"A": (0.0, 0.0), "B": (8.0, 0.0), "C": (0.0, 1.0), "D": (8.0, 1.0)}
    )
    st.apply(Move("line", ("A", "B")))  # h1
    st.apply(Move("line", ("C", "D")))  # h2, parallel
    with pytest.raises(MoveError, match="parallel"):
        st.apply(Move("meet", ("h1", "h2")))
    st.apply(Move("circle", ("A", "C")))  # h3: radius 1 around A
    st.apply(Move("circle", ("B", "D")))  # h4: radius 1 around B, disjoint
    with pytest.raises(MoveError, match="do not intersect"):
        st.apply(Move("meet", ("h3", "h4")))
    st.apply(
        Move("line", ("C", "D"))
    )  # h5: the same line as h2 -> tangent to both circles
    with pytest.raises(MoveError, match="tangent"):
        st.apply(Move("meet", ("h3", "h5")))


def test_circle_circle_order_is_left_of_centre_line():
    st = GeometricState.from_given({"A": (0.0, 0.0), "B": (2.0, 0.0)})
    st.apply(Move("circle", ("A", "B")))  # h1 centre A r=2
    st.apply(Move("circle", ("B", "A")))  # h2 centre B r=2
    first, second = st.apply(Move("meet", ("h1", "h2")))
    assert first.xy() == pytest.approx((1.0, math.sqrt(3.0)))  # left of A->B is +y
    assert second.xy() == pytest.approx((1.0, -math.sqrt(3.0)))
    # swapping the arguments swaps the order
    st2 = GeometricState.from_given({"A": (0.0, 0.0), "B": (2.0, 0.0)})
    st2.apply(Move("circle", ("A", "B")))
    st2.apply(Move("circle", ("B", "A")))
    first2, _ = st2.apply(Move("meet", ("h2", "h1")))
    assert first2.xy() == pytest.approx((1.0, -math.sqrt(3.0)))


def test_circle_line_order_follows_line_direction():
    st = GeometricState.from_given({"A": (-5.0, 0.0), "B": (5.0, 0.0), "O": (0.0, 1.0)})
    st.apply(Move("line", ("A", "B")))  # h1 runs -x -> +x
    st.apply(Move("circle", ("O", "A")))  # h2
    first, second = st.apply(Move("meet", ("h2", "h1")))
    assert first.xy() == pytest.approx((-5.0, 0.0))
    assert second.xy() == pytest.approx((5.0, 0.0))
    st.apply(Move("line", ("B", "A")))  # h5 runs +x -> -x
    first, second = st.apply(Move("meet", ("h5", "h2")))
    assert first.xy() == pytest.approx((5.0, 0.0))


def _random_state(rng: random.Random) -> GeometricState:
    while True:
        pts = {}
        for label in "ABCD":
            pts[label] = (float(rng.randint(-10, 10)), float(rng.randint(-10, 10)))
        try:
            st = GeometricState.from_given(pts)
        except ValueError:
            continue
        try:
            st.apply(Move("line", ("A", "B")))
            st.apply(Move("circle", ("C", "A")))
            st.apply(Move("circle", ("A", "D")))
            st.apply(Move("line", ("C", "D")))
        except MoveError:
            continue
        return st


def _random_similarity(rng: random.Random) -> cga.Similarity:
    return cga.Similarity(
        tx=rng.uniform(-6, 6),
        ty=rng.uniform(-6, 6),
        theta=rng.uniform(0, 2 * math.pi),
        scale=math.exp(rng.uniform(math.log(0.5), math.log(2.0))),
    )


@pytest.mark.parametrize("seed", range(20))
def test_transition_is_equivariant(seed):
    """T(g.S, a) == g.T(S, a) for every primitive, including pair order (AC-5)."""
    rng = random.Random(seed)
    st = _random_state(rng)
    g = _random_similarity(rng)
    moves = [
        Move("line", ("A", "C")),
        Move("circle", ("B", "C")),
        Move("meet", ("h2", "h3")),  # circle & circle
        Move("meet", ("h2", "h1")),  # circle & line
        Move("meet", ("h1", "h4")),  # line & line
    ]
    for move in moves:
        try:
            expected = [(k, mv) for k, mv in st.compute(move)]
        except MoveError:
            continue
        transformed = st.transform(g)
        got = transformed.compute(move)
        assert len(got) == len(expected)
        for (k1, mv1), (k2, mv2) in zip(expected, got, strict=True):
            assert k1 == k2
            # g applied after the move must equal the move after g
            after = cga.apply(g.versor(), mv1)
            if k1 == POINT:
                assert cga.to_xy(after) == pytest.approx(cga.to_xy(mv2), abs=1e-7)
            elif k1 == CIRCLE:
                assert cga.circle_params(after) == pytest.approx(
                    cga.circle_params(mv2), abs=1e-7
                )
            else:
                a = cga.line_params(after)
                b = cga.line_params(mv2)
                sign = 1.0 if a[0] * b[0] + a[1] * b[1] >= 0 else -1.0
                assert (a[0], a[1], a[2]) == pytest.approx(
                    (sign * b[0], sign * b[1], sign * b[2]), abs=1e-7
                )
        st.apply(move)


def test_transform_keeps_handles_and_scales_the_scale():
    st = GeometricState.from_given(TRIANGLE)
    st.apply(Move("line", ("A", "B")))
    g = cga.Similarity(tx=1.0, ty=1.0, theta=0.3, scale=2.0)
    t = st.transform(g)
    assert t.handles() == st.handles()
    assert t.scale == pytest.approx(2.0 * st.scale)
    assert t.get("h1").kind == LINE
    assert cga.line_direction(t.get("h1").mv) == pytest.approx(g.linear(1.0, 0.0))


def test_relabel_equivariance(seed=3):
    """Permuting labels and mapping the moves gives the same geometry (AC-6)."""
    rng = random.Random(seed)
    st = _random_state(rng)
    moves = [
        Move("line", ("A", "C")),
        Move("circle", ("B", "D")),
        Move("meet", ("h2", "h5")),
    ]
    perm = {"A": "C", "B": "A", "C": "D", "D": "B"}
    original = GeometricState.from_given({o.handle: o.xy() for o in st.given})
    relabelled = original.relabel(perm)
    assert sorted(relabelled.handles()) == sorted(perm.values())
    for move in st.objects[4:]:
        pass
    for move in [
        Move("line", ("A", "B")),
        Move("circle", ("C", "A")),
        Move("circle", ("A", "D")),
        Move("line", ("C", "D")),
    ] + moves:
        try:
            a = original.apply(move)
        except MoveError:
            continue
        b = relabelled.apply(relabel_move(move, perm))
        for x, y in zip(a, b, strict=True):
            assert x.handle == y.handle
            assert same_geometry(x, y, original.scale, 1e-9)


def test_predicates_are_scale_invariant():
    for s in (1.0, 0.01, 100.0):
        st = GeometricState.from_given(
            {k: (x * s, y * s) for k, (x, y) in TRIANGLE.items()}
        )
        st.apply(Move("line", ("A", "B")))
        st.apply(Move("circle", ("A", "B")))
        assert incident(st.get("B"), st.get("h1"), st.scale, TOL.incidence)
        assert incident(st.get("B"), st.get("h2"), st.scale, TOL.incidence)
        assert not incident(st.get("C"), st.get("h1"), st.scale, TOL.incidence)
        st.apply(
            Move("circle", ("B", "A"))
        )  # h3: same radius, centres 8 apart -> not tangent
        assert not tangent(st.get("h2"), st.get("h3"), st.scale, TOL.tangency)
        # a line at distance r from the centre is tangent
        r = 8.0 * s
        st2 = GeometricState.from_given(
            {"A": (0.0, 0.0), "B": (r, 0.0), "P": (-r, r), "Q": (r, r)}
        )
        st2.apply(Move("circle", ("A", "B")))
        st2.apply(Move("line", ("P", "Q")))
        assert tangent(st2.get("h1"), st2.get("h2"), st2.scale, TOL.tangency)
        assert st.coincides("A", "A") and not st.coincides("A", "B")


def test_table_lists_labels_and_handles():
    st = GeometricState.from_given(TRIANGLE)
    st.apply(Move("line", ("A", "B")))
    text = st.table()
    assert text.splitlines()[0].startswith("A: point (0.000, 0.000)")
    assert "h1: line" in text


def test_copy_is_independent():
    st = GeometricState.from_given(TRIANGLE)
    clone = st.copy()
    clone.apply(Move("line", ("A", "B")))
    assert st.n_moves == 0 and "h1" not in st and "h1" in clone
