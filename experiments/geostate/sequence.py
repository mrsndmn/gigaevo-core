"""Training and rollout sequences for the two V1 arms.

Both arms see the same text: a short language reference, the given points
with coordinates, the construction to make, then an alternation of *moves*
(written by the model, the only positions carrying loss) and *state blocks*
(written by the environment after each move, no loss). The arms differ in
what a state block contains for each new object:

    tokens arm:  h3:<st>                      (one placeholder token; its input
                                               embedding is the encoder's output)
    table arm:   h3: point (1.23, -4.56)      (the object's parameters as text)

Handles are assigned by the environment (`h1, h2, ...`), so a move is just
`line(A, B)` and the model never invents names. The state block that follows
lists the handles the move produced, which is how a name binds to a token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .configs import ARM_TABLE, ARM_TOKENS
from .generator import Problem
from .prompts import format_coord, given_points_text
from .state import GeometricState, Move, Obj

ARMS_V1 = (ARM_TOKENS, ARM_TABLE)

ST = "<st>"  # placeholder token; one per object
STATE_OPEN = "[state]"
STATE_CLOSE = "[/state]"

SYSTEM_V1 = """Straightedge-and-compass construction. Objects: points, lines, circles.
Moves, one per line: line(P, Q) | circle(O, P) | meet(U, V) | answer NAME
meet gives one point for two lines, else two points X, Y: for two circles X is
left of the directed line from the first centre to the second; for a circle
and a line X comes first along the line's direction (first defining point to
second). After every move the environment lists the new objects' handles.
"""


def object_row(obj: Obj, arm: str) -> str:
    if arm == ARM_TOKENS:
        return f"{obj.handle}:{ST}"
    p = obj.params()
    if obj.kind == "point":
        return f"{obj.handle}: point ({format_coord(p[0])}, {format_coord(p[1])})"
    if obj.kind == "line":
        return (
            f"{obj.handle}: line n=({format_coord(p[0])}, {format_coord(p[1])}) "
            f"d={format_coord(p[2])}"
        )
    return (
        f"{obj.handle}: circle c=({format_coord(p[0])}, {format_coord(p[1])}) "
        f"r={format_coord(p[2])}"
    )


def state_block(objs: list[Obj], arm: str) -> str:
    return (
        f"{STATE_OPEN} "
        + " ".join(object_row(o, arm) for o in objs)
        + f" {STATE_CLOSE}\n"
    )


def prompt_text(problem: Problem) -> str:
    return (
        SYSTEM_V1
        + f"Given points: {given_points_text(problem.given)}.\n"
        + f"Construct {problem.description}.\n"
    )


@dataclass
class Segment:
    text: str
    loss: bool  # model-written (move) or environment-written (state/prompt)
    handles: list[str] = field(default_factory=list)  # state tokens, in order


@dataclass
class Sequence:
    segments: list[Segment]
    n_moves: int

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segments)

    @property
    def state_handles(self) -> list[str]:
        out: list[str] = []
        for s in self.segments:
            out.extend(s.handles)
        return out


def build_sequence(problem: Problem, arm: str) -> tuple[Sequence, GeometricState]:
    """The full teacher-forced sequence of a trajectory, plus the final state."""
    state = problem.state()
    segments = [Segment(prompt_text(problem), False)]
    given = list(state.objects)
    segments.append(Segment(state_block(given, arm), False, [o.handle for o in given]))
    for move in problem.trajectory:
        segments.append(Segment(f"{move.text()}\n", True))
        produced = state.apply(move)
        segments.append(
            Segment(state_block(produced, arm), False, [o.handle for o in produced])
        )
    segments.append(Segment(f"answer {problem.answer}\n", True))
    return Sequence(segments, len(problem.trajectory)), state


def move_text(move: Move) -> str:
    return f"{move.text()}\n"
