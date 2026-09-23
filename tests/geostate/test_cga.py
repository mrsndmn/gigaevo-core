"""The Cl(3,1) kernel against analytic plane geometry."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from experiments.geostate import cga


def test_points_are_null_and_inner_product_is_half_squared_distance():
    P = cga.point(1.0, 2.0)
    Q = cga.point(-3.0, 0.5)
    assert abs(cga.scalar(cga.gp(P, P))) < 1e-12
    d2 = (1 + 3) ** 2 + (2 - 0.5) ** 2
    assert cga.scalar(cga.gp(P, Q)) == pytest.approx(-0.5 * d2)
    assert cga.to_xy(3.0 * P) == pytest.approx((1.0, 2.0))


def test_line_through_direction_and_incidence():
    P, Q = cga.point(1.0, 2.0), cga.point(-3.0, 0.5)
    L = cga.line_through(P, Q)
    tx, ty = cga.line_direction(L)
    n = math.hypot(4.0, 1.5)
    assert (tx, ty) == pytest.approx((-4.0 / n, -1.5 / n))
    assert abs(cga.e0_coeff(L)) < 1e-12  # a line has no e_0 part
    M = cga.point(-1.0, 1.25)  # the midpoint
    assert abs(cga.scalar(cga.gp(M, L))) < 1e-12
    # the reverse line has the opposite direction but the same point set
    R = cga.line_through(Q, P)
    assert cga.line_direction(R) == pytest.approx((-tx, -ty))
    assert abs(cga.scalar(cga.gp(M, R))) < 1e-12


def test_circle_params_and_incidence():
    S = cga.circle(1.0, -2.0, 3.0)
    assert cga.circle_params(S) == pytest.approx((1.0, -2.0, 3.0))
    X = cga.point(4.0, -2.0)
    assert abs(cga.scalar(cga.gp(X, S))) < 1e-12
    T = cga.circle_through(cga.point(0.0, 0.0), cga.point(3.0, 4.0))
    assert cga.circle_params(T) == pytest.approx((0.0, 0.0, 5.0))


def test_meet_circle_circle_matches_analytic():
    A = cga.circle(0.0, 0.0, 2.0)
    B = cga.circle(2.0, 0.0, 2.0)
    T, s = cga.meet_pair(A, B)
    assert s > 0
    P, Q = cga.pair_points(T, s)
    got = sorted([cga.to_xy(P), cga.to_xy(Q)])
    assert got[0] == pytest.approx((1.0, -math.sqrt(3.0)))
    assert got[1] == pytest.approx((1.0, math.sqrt(3.0)))


def test_meet_line_circle_and_no_intersection_and_tangent():
    S = cga.circle(0.0, 0.0, 2.0)
    L = cga.line_through(cga.point(-5.0, 0.0), cga.point(5.0, 0.0))
    T, s = cga.meet_pair(S, L)
    P, Q = cga.pair_points(T, s)
    assert sorted([cga.to_xy(P), cga.to_xy(Q)]) == pytest.approx(
        [(-2.0, 0.0), (2.0, 0.0)]
    )
    far = cga.line_through(cga.point(-5.0, 3.0), cga.point(5.0, 3.0))
    _, s_far = cga.meet_pair(S, far)
    assert s_far < 0
    tangent = cga.line_through(cga.point(-5.0, 2.0), cga.point(5.0, 2.0))
    _, s_tan = cga.meet_pair(S, tangent)
    assert abs(s_tan) < 1e-9


def test_meet_line_line_and_parallel():
    L1 = cga.line_through(cga.point(-5.0, 0.0), cga.point(5.0, 0.0))
    L2 = cga.line_through(cga.point(1.0, -1.0), cga.point(1.0, 1.0))
    T, _ = cga.meet_pair(L1, L2)
    assert cga.flat_point_xy(T) == pytest.approx((1.0, 0.0))
    L3 = cga.line_through(cga.point(0.0, 1.0), cga.point(5.0, 1.0))
    T, _ = cga.meet_pair(L1, L3)
    assert cga.flat_point_xy(T) is None


def test_versors_move_points_as_the_analytic_map():
    P = cga.point(1.0, 2.0)
    assert cga.to_xy(cga.apply(cga.translator(3.0, -1.0), P)) == pytest.approx(
        (4.0, 1.0)
    )
    assert cga.to_xy(
        cga.apply(cga.rotor(math.pi / 2), cga.point(1.0, 0.0))
    ) == pytest.approx((0.0, 1.0))
    assert cga.to_xy(cga.apply(cga.dilator(3.0), P)) == pytest.approx((3.0, 6.0))
    assert cga.to_xy(cga.apply(cga.reflector_x_axis(), P, odd=True)) == pytest.approx(
        (1.0, -2.0)
    )


@pytest.mark.parametrize("seed", range(5))
def test_similarity_versor_agrees_with_xy_and_inverse(seed):
    rng = random.Random(seed)
    g = cga.Similarity(
        tx=rng.uniform(-5, 5),
        ty=rng.uniform(-5, 5),
        theta=rng.uniform(0, 2 * math.pi),
        scale=math.exp(rng.uniform(-1, 1)),
        reflect=bool(seed % 2),
    )
    x, y = rng.uniform(-9, 9), rng.uniform(-9, 9)
    via_versor = cga.to_xy(g.transform(cga.point(x, y)))
    assert via_versor == pytest.approx(g.xy(x, y))
    assert g.inverse_xy(*g.xy(x, y)) == pytest.approx((x, y))
    # circles stay circles with scaled radius; lines stay lines
    S = g.transform(cga.circle(x, y, 2.0))
    assert cga.circle_params(S)[2] == pytest.approx(2.0 * g.scale)
    L = g.transform(cga.line_through(cga.point(0.0, 0.0), cga.point(1.0, 1.0)))
    assert abs(cga.e0_coeff(L)) < 1e-9 * np.abs(L).max()
    assert cga.Similarity.from_dict(g.to_dict()) == g
