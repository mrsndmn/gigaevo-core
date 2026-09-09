"""Static check on model-written plotting code.

This is a *guard rail*, not a sandbox: it rejects the obvious ways generated
code could touch the system or break the incremental-canvas contract (clearing
the axes, opening a second figure, saving files itself). Actual isolation comes
from running the code in a separate `python -I` subprocess with a timeout.
"""

from __future__ import annotations

import ast

ALLOWED_IMPORTS = {"math", "numpy", "matplotlib", "matplotlib.pyplot"}

BANNED_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "__import__",
}

# Attributes that either escape the process or destroy the shared canvas.
BANNED_ATTRS = {
    "cla",
    "clf",
    "clear",
    "close",
    "show",
    "savefig",
    "figure",
    "subplots",
    "subplot",
    "remove",
    "system",
    "popen",
    "spawn",
    "fork",
    "unlink",
    "rmtree",
}


def check(code: str) -> list[str]:
    """Return a list of human-readable violations; empty means the code is ok."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]

    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in {
                    m.split(".")[0] for m in ALLOWED_IMPORTS
                }:
                    problems.append(f"import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in {m.split(".")[0] for m in ALLOWED_IMPORTS}:
                problems.append(f"import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"use of builtin {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                problems.append(f"dunder attribute {node.attr!r} is not allowed")
            elif node.attr in BANNED_ATTRS:
                problems.append(
                    f"{node.attr!r} is not allowed: it breaks the shared "
                    "incremental canvas or escapes the process"
                )
    return problems
