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

Sugar that models write unprompted is desugared, not rejected, because it is
unambiguous and rejecting it would measure our parser, not the model:
nested calls (`meet(l, line(A, B))`) are hoisted into temporaries in
evaluation order, `answer meet(l1, l2)` / `answer = X` / `answer: X` are
accepted, and `meet(...)[0]` / `[1]` select one point of a pair.
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

OPS = ("line", "circle", "meet")
_NAME = r"[A-Za-z_][A-Za-z0-9_']*"
NAME_RE = re.compile(rf"^{_NAME}$")
FENCE_RE = re.compile(r"```[A-Za-z_-]*[ \t]*\n(.*?)```", re.DOTALL)
# `answer` must be the whole word: `answer_point = meet(...)` is an assignment
ANSWER_RE = re.compile(
    r"^\s*answer(?![A-Za-z0-9_'])\s*([:=])?\s*(.+?)\s*$", re.IGNORECASE
)
ASSIGN_RE = re.compile(rf"^\s*({_NAME})\s*(?:,\s*({_NAME}))?\s*=\s*(.+?)\s*$")
# a prefix no model-written name can match (names are [A-Za-z_][A-Za-z0-9_']*)
TEMP_PREFIX = "$t"
MAX_DEPTH = 32


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class Statement:
    names: tuple[str, ...]
    op: str
    args: tuple[str, str]
    line_no: int
    select: int | None = None  # `meet(...)[k]`: bind the k-th point only

    def text(self) -> str:
        suffix = "" if self.select is None else f"[{self.select}]"
        return f"{', '.join(self.names)} = {self.op}({self.args[0]}, {self.args[1]}){suffix}"


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


# ---------------------------------------------------------------------------
# Expressions: NAME | op(arg, arg)[k]
# ---------------------------------------------------------------------------


class _Expr:
    def __init__(self, text: str, line_no: int):
        self.text = text
        self.pos = 0
        self.line_no = line_no

    def _skip(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _fail(self, what: str) -> ParseError:
        return ParseError(
            f"line {self.line_no}: cannot parse {self.text.strip()!r} ({what})"
        )

    def _name(self) -> str:
        self._skip()
        m = re.match(_NAME, self.text[self.pos :])
        if not m:
            raise self._fail("expected a name")
        self.pos += m.end()
        return m.group(0)

    def _expect(self, ch: str) -> None:
        self._skip()
        if self.pos >= len(self.text) or self.text[self.pos] != ch:
            raise self._fail(f"expected {ch!r}")
        self.pos += 1

    def parse(
        self,
        hoisted: list[tuple[str, str, tuple[str, str], int | None]],
        depth: int = 0,
    ) -> str:
        """Parse one expression; nested calls are appended to `hoisted` as
        (temp name, op, args, select) in evaluation order. Returns the name
        the expression's value is bound to."""
        if depth > MAX_DEPTH:
            raise self._fail("nesting too deep")
        name = self._name()
        self._skip()
        if self.pos < len(self.text) and self.text[self.pos] == "(":
            if name not in OPS:
                raise self._fail(f"unknown primitive {name!r}")
            self.pos += 1
            a = self.parse(hoisted, depth + 1)
            self._expect(",")
            b = self.parse(hoisted, depth + 1)
            self._expect(")")
            select = None
            self._skip()
            if self.text[self.pos : self.pos + 1] == "[":
                m = re.match(r"\[\s*([01])\s*\]", self.text[self.pos :])
                if not m:
                    raise self._fail("index must be [0] or [1]")
                select = int(m.group(1))
                self.pos += m.end()
            temp = f"{TEMP_PREFIX}{len(hoisted) + 1}"
            hoisted.append((temp, name, (a, b), select))
            return temp
        return name

    def done(self) -> bool:
        self._skip()
        return self.pos == len(self.text)


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
            expr = _Expr(m.group(2), line_no)
            hoisted: list = []
            name = expr.parse(hoisted)
            if not expr.done():
                raise ParseError(f"line {line_no}: cannot parse {stripped!r}")
            for temp, op, args, select in hoisted:
                statements.append(Statement((temp,), op, args, line_no, select))
            if m.group(1) or hoisted:
                # `answer = X` / `answer = meet(...)` also binds the name
                # `answer`, so a later `answer answer` (models write it) resolves
                statements.append(
                    Statement(("answer",), "alias", (name, name), line_no)
                )
            answer = name
            continue
        m = ASSIGN_RE.match(stripped)
        if not m:
            raise ParseError(f"line {line_no}: cannot parse {stripped!r}")
        names = (m.group(1),) if m.group(2) is None else (m.group(1), m.group(2))
        expr = _Expr(m.group(3), line_no)
        hoisted = []
        value = expr.parse(hoisted)
        if not expr.done():
            raise ParseError(f"line {line_no}: cannot parse {stripped!r}")
        if len(names) == 2 and hoisted and hoisted[-1][3] is not None:
            raise ParseError(
                f"line {line_no}: two names but [{hoisted[-1][3]}] selects one point"
            )
        if not hoisted:
            # `X = Y`: a plain alias
            statements.append(Statement(names, "alias", (value, value), line_no))
            continue
        # the outermost call binds the statement's names; inner calls keep temps
        *inner, (_, op, args, select) = hoisted
        for temp, iop, iargs, isel in inner:
            statements.append(Statement((temp,), iop, iargs, line_no, isel))
        statements.append(Statement(names, op, args, line_no, select))
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
            if stmt.op == "alias":
                for name in stmt.names:
                    bindings[name] = args[0]
                continue
            produced = st.apply(Move(stmt.op, (args[0], args[1])))
        except MoveError as exc:
            ex.error = f"line {stmt.line_no}: {exc}"
            ex.error_kind = exc.kind
            ex.failed_statement = stmt
            ex.bindings = bindings
            return ex
        ex.n_moves += 1
        if stmt.select is not None:
            if stmt.select >= len(produced):
                ex.error = f"line {stmt.line_no}: [{stmt.select}] on a single point"
                ex.error_kind = ERR_TYPE
                ex.failed_statement = stmt
                ex.bindings = bindings
                return ex
            produced = [produced[stmt.select]]
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
    except RecursionError:
        ex = Execution(state=state.copy())
        ex.error = "nesting too deep"
        ex.error_kind = ERR_SYNTAX
        return ex
    except ParseError as exc:
        ex = Execution(state=state.copy())
        ex.error = str(exc)
        ex.error_kind = ERR_EMPTY if "empty" in str(exc) else ERR_SYNTAX
        return ex
    return execute(program, state)


def scorer_hash() -> str:
    """Identifies the parser/executor that scored a record."""
    import hashlib
    import inspect
    import sys

    module = sys.modules[__name__]
    return hashlib.sha256(inspect.getsource(module).encode()).hexdigest()[:12]


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
