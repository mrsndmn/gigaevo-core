"""Every macro's expansion produces the analytic target."""

from __future__ import annotations

import math

import pytest

from experiments.geostate import macros
from experiments.geostate.state import CIRCLE, LINE, POINT, GeometricState, MoveError

TRI = {"A": (1.0, 1.0), "B": (9.0, 2.0), "C": (3.0, 8.0)}


def _build(given, expr):
    st = GeometricState.from_given(given)
    b = macros.Builder(st)
    h = macros.expand(b, expr)
    return st, b, st.get(h)


def _dist_to_line(x, y, p, q):
    (px, py), (qx, qy) = p, q
    return abs((qx - px) * (py - y) - (px - x) * (qy - py)) / math.hypot(
        qx - px, qy - py
    )


def test_midpoint_and_perp_bisector():
    st, b, m = _build(TRI, ["midpoint", "A", "B"])
    assert m.xy() == pytest.approx((5.0, 1.5))
    assert len(b.moves) == 6
    st, b, line = _build(TRI, ["perp_bisector", "A", "B"])
    assert line.kind == LINE and len(b.moves) == 4
    nx, ny, d = line.params()
    assert abs(nx * 5.0 + ny * 1.5 - d) < 1e-9  # passes through the midpoint
    assert abs(nx * 1.0 - ny * 8.0) < 1e-9  # the normal is parallel to AB


def test_circumcenter_incenter_centroid():
    _, b, o = _build(TRI, ["circumcenter", "A", "B", "C"])
    dists = [math.dist(o.xy(), p) for p in TRI.values()]
    assert max(dists) - min(dists) < 1e-9
    assert len(b.moves) == 9
    _, b, c = _build(TRI, ["circumcircle", "A", "B", "C"])
    assert c.kind == CIRCLE and c.params()[2] == pytest.approx(dists[0])
    _, b, i = _build(TRI, ["incenter", "A", "B", "C"])
    x, y = i.xy()
    d = [
        _dist_to_line(x, y, TRI["A"], TRI["B"]),
        _dist_to_line(x, y, TRI["B"], TRI["C"]),
        _dist_to_line(x, y, TRI["C"], TRI["A"]),
    ]
    assert max(d) - min(d) < 1e-9
    _, b, g = _build(TRI, ["centroid", "A", "B", "C"])
    assert g.xy() == pytest.approx((13.0 / 3.0, 11.0 / 3.0))


def test_reflect_foot_perp_parallel():
    given = {"P": (2.0, 5.0), "A": (0.0, 0.0), "Q": (10.0, 0.0)}
    _, b, r = _build(given, ["reflect", "P", "A", "Q"])
    assert r.xy() == pytest.approx((2.0, -5.0))
    assert len(b.moves) == 3
    _, b, f = _build(given, ["foot", "P", "A", "Q"])
    assert f.xy() == pytest.approx((2.0, 0.0))
    _, b, l1 = _build(given, ["perp_through", "P", "A", "Q"])
    nx, ny, d = l1.params()
    assert abs(abs(nx) - 1.0) < 1e-9 and abs(d / nx - 2.0) < 1e-9  # the line x = 2
    _, b, l2 = _build(given, ["parallel_through", "P", "A", "Q"])
    nx, ny, d = l2.params()
    assert abs(abs(ny) - 1.0) < 1e-9 and abs(d / ny - 5.0) < 1e-9  # the line y = 5
    # perpendicular through a point ON the line takes the other construction
    on = {"P": (4.0, 0.0), "A": (0.0, 0.0), "Q": (10.0, 0.0)}
    _, b, l3 = _build(on, ["perp_through", "P", "A", "Q"])
    nx, ny, d = l3.params()
    assert abs(abs(nx) - 1.0) < 1e-9 and abs(d / nx - 4.0) < 1e-9
    with pytest.raises(MoveError):
        _build(on, ["parallel_through", "P", "A", "Q"])


def test_angle_bisector_theorem_and_line_meet():
    _, b, d = _build(TRI, ["angle_bisector_foot", "A", "B", "C"])
    x, y = d.xy()
    # D on BC with BD/DC = AB/AC
    assert _dist_to_line(x, y, TRI["B"], TRI["C"]) < 1e-9
    ratio = math.dist((x, y), TRI["B"]) / math.dist((x, y), TRI["C"])
    expected = math.dist(TRI["A"], TRI["B"]) / math.dist(TRI["A"], TRI["C"])
    assert ratio == pytest.approx(expected)
    _, b, bis = _build(TRI, ["angle_bisector", "A", "B", "C"])
    nx, ny, dd = bis.params()
    assert abs(nx * 1.0 + ny * 1.0 - dd) < 1e-9  # through A
    given = {"A": (0.0, 0.0), "B": (4.0, 4.0), "C": (0.0, 4.0), "D": (4.0, 0.0)}
    _, b, m = _build(given, ["line_meet", "A", "B", "C", "D"])
    assert m.xy() == pytest.approx((2.0, 2.0)) and len(b.moves) == 3


def test_rotations():
    given = {"P": (3.0, 1.0), "O": (1.0, 1.0)}
    _, b, r = _build(given, ["rotate90", "P", "O"])
    assert r.xy() == pytest.approx((1.0, 3.0))
    _, b, r = _build(given, ["rotate270", "P", "O"])
    assert r.xy() == pytest.approx((1.0, -1.0))
    _, b, r = _build(given, ["rotate180", "P", "O"])
    assert r.xy() == pytest.approx((-1.0, 1.0))


def test_builder_reuses_existing_geometry():
    st = GeometricState.from_given(TRI)
    b = macros.Builder(st)
    macros.expand(b, ["midpoint", "A", "B"])
    n = len(b.moves)
    macros.expand(b, ["midpoint", "B", "A"])  # same objects, other argument order
    assert len(b.moves) == n
    macros.expand(b, ["perp_bisector", "A", "B"])
    assert len(b.moves) == n


def test_nested_expressions_describe_and_relabel():
    expr = ["midpoint", "A", ["circumcenter", "A", "B", "C"]]
    macros.validate_expr(expr)
    assert (
        macros.describe(expr)
        == "the midpoint of A and (the circumcentre of the triangle with vertices A, B and C)"
    )
    assert macros.relabel_expr(expr, {"A": "B", "B": "A"}) == [
        "midpoint",
        "B",
        ["circumcenter", "B", "A", "C"],
    ]
    assert macros.base_labels(expr) == ["A", "A", "B", "C"]
    assert macros.result_kind(expr) == POINT
    with pytest.raises(ValueError):
        macros.validate_expr(["midpoint", "A", ["perp_bisector", "A", "B"]])
    with pytest.raises(ValueError):
        macros.validate_expr(["midpoint", "A"])
    with pytest.raises(ValueError):
        macros.validate_expr(["nope", "A", "B"])
    _, _, m = _build(TRI, expr)
    _, _, o = _build(TRI, ["circumcenter", "A", "B", "C"])
    ox, oy = o.xy()
    assert m.xy() == pytest.approx(((1.0 + ox) / 2, (1.0 + oy) / 2))


def test_param_transforms_round_trip():
    from experiments.geostate import cga

    g = cga.Similarity(tx=2.0, ty=-1.0, theta=0.9, scale=1.7)
    for kind, params in (
        (POINT, (1.0, 2.0)),
        (CIRCLE, (1.0, 2.0, 3.0)),
        (LINE, (0.6, 0.8, 2.0)),
    ):
        there = macros.transform_params(kind, params, g)
        back = macros.inverse_params(kind, there, g)
        assert macros.distance_between(kind, back, params) < 1e-9


def test_strict_expand_rejects_a_macro_that_adds_no_move():
    # the outer intersection restates the inner one: same lines, same point
    given = {"A": (0.0, 0.0), "B": (8.0, 0.0), "C": (2.0, 6.0), "E": (4.0, 5.0)}
    expr = ["line_meet", ["line_meet", "C", "E", "A", "B"], "E", "A", "B"]
    st = GeometricState.from_given(given)
    assert macros.expand(macros.Builder(st), expr) == "h3"  # lenient: fine
    with pytest.raises(MoveError, match="adds no move"):
        macros.expand(
            macros.Builder(GeometricState.from_given(given)), expr, strict=True
        )
