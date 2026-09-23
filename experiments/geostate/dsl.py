"""The construction language a model writes, and its executor.

One statement per line:

    NAME = line(P, Q)
    NAME = circle(O, P)
    X, Y = meet(U, V)        (X = meet(U, V) for two lines)
    answer NAME

Names are the model's own; given points are pre-bound by their labels. The
executor resolves every name against the state, so a reference to an object
that does not exist fails with `undefined_reference` — the failure class
GeoBuildBench found most common — instead of being silently invented. Nothing
here is evaluated as Python: a statement either matches the grammar or is a
syntax error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .state import (
    ERR_DEGENERATE,
    ERR_TYPE,
    ERR_UNDEFINED,
    GeometricState,
    Move,
    MoveError,
    Obj,
)

ERR_SYNTAX = "syntax"
ERR_NO_ANSWER = "no_answer"
ERR_EMPTY = "empty"
ERROR_KINDS = (
    ERR_SYNTAX,
    ERR_UNDEFINED,
    ERR_TYPE,
    ERR_DEGENERATE,
    ERR_NO_ANSWER,
    ERR_EMPTY,
)

_NAME = r"[A-Za-z_][A-Za-z0-9_']*"
STATEMENT_RE = re.compile(
    rf"^\s*({_NAME})\s*(?:,\s*({_NAME}))?\s*=\s*(line|circle|meet)\s*\(\s*({_NAME})\s*,\s*({_NAME})\s*\)\s*$"
)
ANSWER_RE = re.compile(rf"^\s*answer\s*[:=]?\s*({_NAME})\s*$", re.IGNORECASE)
FENCE_RE = re.compile(r"```[A-Za-z_-]*[ \t]*\n(.*?)```", re.DOTALL)


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class Statement:
    names: tuple[str, ...]
    op: str
    args: tuple[str, str]
    line_no: int

    def text(self) -> str:
        return f"{', '.join(self.names)} = {self.op}({self.args[0]}, {self.args[1]})"


@dataclass
class Program:
    statements: list[Statement]
    answer: str | None
    source: str

    def text(self) -> str:
        lines = [s.text() for s in self.statements]
        if self.answer is not None:
            lines.append(f"answer {self.answer}")
        return "\n".join(lines)


def extract_block(text: str) -> str:
    """The last fenced block if there is one, else the whole text."""
    blocks = FENCE_RE.findall(text)
    return blocks[-1] if blocks else text


def parse(text: str) -> Program:
    """Parse a program. Lines that are blank or comments are skipped."""
    source = extract_block(text)
    statements: list[Statement] = []
    answer: str | None = None
    for line_no, raw in enumerate(source.splitlines(), start=1):
        stripped = raw.split("#", 1)[0].strip()
        if not stripped:
            continue
        m = ANSWER_RE.match(stripped)
        if m:
            answer = m.group(1)
            continue
        m = STATEMENT_RE.match(stripped)
        if not m:
            raise ParseError(f"line {line_no}: cannot parse {stripped!r}")
        names = (m.group(1),) if m.group(2) is None else (m.group(1), m.group(2))
        statements.append(
            Statement(names, m.group(3), (m.group(4), m.group(5)), line_no)
        )
    if not statements and answer is None:
        raise ParseError("empty program")
    return Program(statements, answer, source)


@dataclass
class Execution:
    state: GeometricState
    answer: Obj | None = None
    error: str | None = None
    error_kind: str | None = None
    n_moves: int = 0
    bindings: dict[str, str] = field(default_factory=dict)
    failed_statement: Statement | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.answer is not None


def execute(program: Program, state: GeometricState) -> Execution:
    """Run a program on a copy of the state; stops at the first error."""
    st = state.copy()
    ex = Execution(state=st)
    bindings = {o.handle: o.handle for o in st.objects}
    for stmt in program.statements:
        try:
            args = []
            for a in stmt.args:
                if a not in bindings:
                    raise MoveError(ERR_UNDEFINED, f"unknown object {a!r}")
                args.append(bindings[a])
            produced = st.apply(Move(stmt.op, (args[0], args[1])))
        except MoveError as exc:
            ex.error = f"line {stmt.line_no}: {exc}"
            ex.error_kind = exc.kind
            ex.failed_statement = stmt
            ex.bindings = bindings
            return ex
        ex.n_moves += 1
        for name, obj in zip(stmt.names, produced, strict=False):
            bindings[name] = obj.handle
    ex.bindings = bindings
    if program.answer is None:
        ex.error = "no answer statement"
        ex.error_kind = ERR_NO_ANSWER
        return ex
    if program.answer not in bindings:
        ex.error = f"answer names unknown object {program.answer!r}"
        ex.error_kind = ERR_UNDEFINED
        return ex
    ex.answer = st.get(bindings[program.answer])
    return ex


def run_text(text: str, state: GeometricState) -> Execution:
    """Parse + execute; a parse failure is an execution with a syntax error."""
    try:
        program = parse(text)
    except ParseError as exc:
        ex = Execution(state=state.copy())
        ex.error = str(exc)
        ex.error_kind = ERR_EMPTY if "empty" in str(exc) else ERR_SYNTAX
        return ex
    return execute(program, state)


def trajectory_program(moves: list[Move], answer: str, state: GeometricState) -> str:
    """The reference program for a recorded trajectory (handles as names)."""
    st = state.copy()
    lines = []
    for move in moves:
        produced = st.apply(move)
        names = ", ".join(o.handle for o in produced)
        lines.append(f"{names} = {move.text()}")
    lines.append(f"answer {answer}")
    return "\n".join(lines)
