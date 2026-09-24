"""Prompts for the E0 consistency probe.

The system prompt is the language reference the model constructs in. It has
to state the intersection-pair ordering rule (ADR 0005), because a model that
does not know which of the two `meet` outputs is which cannot select the right
one — and selection errors are what H1 measures.
"""

from __future__ import annotations

import base64
from pathlib import Path

SYSTEM = """You solve straightedge-and-compass construction problems in a small formal language.

Objects are points, lines and circles. The given points are named by letters. Every object you construct must get a fresh name of your choice.

The only operations are:
  NAME = line(P, Q)      the line through points P and Q (P and Q must differ). Its direction runs from P to Q.
  NAME = circle(O, P)    the circle centred at O that passes through P (O and P must differ).
  X, Y = meet(U, V)      the intersection points of two lines/circles U and V.
      two lines            -> exactly one point; write  X = meet(U, V)
      otherwise            -> exactly two points X and Y, always in this order:
        circle & circle    X is the point on the LEFT of the directed line from the centre of U to the centre of V; Y is on the right.
        circle & line      X is the point that comes FIRST along the line's direction (from its first defining point towards its second); Y comes second.
                           Example: for L = line(A, B) and C = circle(A, B), meet(C, L) gives X on the far side of A from B (the point symmetric to B about A) and Y = B itself, because walking along L from A towards B you reach X before B.
      meet fails if the objects are parallel, identical, tangent, or do not intersect.
      Decide which of X and Y you need from the coordinates. When the task asks for *a* point or *a* line with a property, any object with that property is accepted; when it names one specific object, only that object is.

Every name you use must have been defined earlier (or be a given point). Finish with the line
  answer NAME
naming the object that solves the task.

You may think first. Then output the program in one fenced block, exactly like:
```construction
h1 = circle(A, B)
h2 = circle(B, A)
h3, h4 = meet(h1, h2)
h5 = line(h3, h4)
answer h5
```
"""


def format_coord(v: float) -> str:
    text = f"{v:.2f}"
    text = text.rstrip("0").rstrip(".") if "." in text else text
    return "0" if text in ("-0", "") else text


def given_points_text(given: dict[str, tuple[float, float]]) -> str:
    return ", ".join(
        f"{label} = ({format_coord(x)}, {format_coord(y)})"
        for label, (x, y) in given.items()
    )


def user_text(given: dict[str, tuple[float, float]], description: str) -> str:
    return (
        f"Given points: {given_points_text(given)}.\n"
        f"Construct {description}.\n"
        "Use only line, circle and meet, then give the answer."
    )


def user_figure(labels: list[str], description: str) -> str:
    names = ", ".join(labels)
    return (
        f"The figure shows the given points {names} on a coordinate grid; read their "
        "positions from the figure.\n"
        f"Construct {description}.\n"
        "Use only line, circle and meet, then give the answer."
    )


def data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def messages_text(
    given: dict[str, tuple[float, float]], description: str
) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_text(given, description)},
    ]


def messages_figure(labels: list[str], description: str, image: Path) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_figure(labels, description)},
                {"type": "image_url", "image_url": {"url": data_url(image)}},
            ],
        },
    ]
