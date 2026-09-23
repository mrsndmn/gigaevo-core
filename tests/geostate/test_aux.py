"""The auxiliary construction set reproduces NoReGeo's golden points (AC-8)."""

from __future__ import annotations

import pytest

from experiments.geostate import aux
from experiments.geostate.configs import AUX_STEMS
from experiments.noregeo_synth.configs import DATA_DIR, TASK_FILES


def _data_present() -> bool:
    return all((DATA_DIR / f).exists() for f in TASK_FILES)


def test_recipes_parse_the_question_parameters():
    row = {
        "points": {"P": [5, -5]},
        "question": "Point P=(5, -5) reflected across the X-axis. What are the coordinates of P'?",
    }
    expr, given = aux._expr_and_given("reflection", row)
    assert expr == ["reflect", "P", "O", "U"] and given["U"] == (1.0, 0.0)
    row["question"] = "Point P=(5, -5) reflected across the Y-axis."
    _, given = aux._expr_and_given("reflection", row)
    assert given["U"] == (0.0, 1.0)
    row = {
        "points": {"P": [0, 3]},
        "question": "Point P=(0, 3) rotated by 270° about the origin. What is P'?",
    }
    expr, given = aux._expr_and_given("rotation_point", row)
    assert expr == ["rotate270", "P", "O"] and given["O"] == (0.0, 0.0)
    row["question"] = "rotated by 45° about the origin"
    with pytest.raises(ValueError):
        aux._expr_and_given("rotation_point", row)
    row = {
        "points": {"A": [-3, 2], "B": [12, 5], "C": [1, -19]},
        "question": "Triangle ...; find the intersection of the angle bisector at B.",
    }
    expr, _ = aux._expr_and_given("bisector", row)
    assert expr == ["angle_bisector_foot", "B", "A", "C"]
    row = {"points": [[9, -4], [-3, 8], [-1, 8]], "question": "..."}
    expr, given = aux._expr_and_given("inner_circle_center", row)
    assert expr == ["incenter", "A", "B", "C"] and list(given) == ["A", "B", "C"]


@pytest.mark.skipif(
    not _data_present(), reason="NoReGeo task files not cached in data/noregeo"
)
def test_kernel_reproduces_golden_points_on_the_seven_stems():
    ok, skipped, mismatched = aux.check_all()
    assert mismatched == [], [s.reason for s in mismatched]
    assert len(ok) + len(skipped) == 100 * len(AUX_STEMS)
    assert len(ok) >= 680, [s.reason for s in skipped]
    allowed = (
        "parallel",
        "tangent",
        "zero radius",
        "too close",
        "target existed",
        "given point",
    )
    for s in skipped:
        assert any(a in s.reason for a in allowed), s
    for p in ok:
        assert p.target_kind == "point"
        assert p.n_moves == len(p.trajectory) >= 3


@pytest.mark.skipif(
    not _data_present(), reason="NoReGeo task files not cached in data/noregeo"
)
def test_sample_is_deterministic_and_per_stem():
    a, _ = aux.load_aux_problems(per_stem=3, seed=1)
    b, _ = aux.load_aux_problems(per_stem=3, seed=1)
    assert [p.id for p in a] == [p.id for p in b]
    counts: dict[str, int] = {}
    for p in a:
        counts[p.meta["stem"]] = counts.get(p.meta["stem"], 0) + 1
    assert counts == {stem: 3 for stem in AUX_STEMS}
