"""Static check on model-written plotting code.

This is a *guard rail*, not a sandbox. It rejects the obvious ways generated
code could touch the system or break the incremental-canvas contract (clearing
the axes, deleting artists, resizing the figure, zooming the view in, opening a
second figure, saving files itself). It is a name-level check, so it cannot
stop a determined escape — `np.save`-style exfiltration is blocked by name, but
reflection through an unlisted alias is not. Actual isolation comes from
running the code in a separate `python -I` subprocess with a timeout, and the
canvas contract is additionally enforced *dynamically* by the per-step pixel
and ink checks in `verify.py`.
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
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "__import__",
    "__builtins__",
}

# Attributes that escape the process, destroy the shared canvas, or silently
# break the "step k's image = step k-1's image + new content" contract.
BANNED_ATTRS = {
    # canvas destruction
    "cla",
    "clf",
    "clear",
    "close",
    "show",
    "figure",
    "subplots",
    "subplot",
    "remove",
    "set_visible",
    "set_data",
    "set_xdata",
    "set_ydata",
    # the frame must not change shape: differing PNG sizes make consecutive
    # steps incomparable, which used to turn the incrementality check off
    "set_size_inches",
    "set_dpi",
    "set_figwidth",
    "set_figheight",
    # the view may only GROW, and only through the preamble's expand_view()
    "set_xlim",
    "set_ylim",
    "set_aspect",
    "autoscale",
    "axis",
    # filesystem / process
    "savefig",
    "save",
    "savez",
    "savetxt",
    "tofile",
    "write",
    "writelines",
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

    allowed_roots = {m.split(".")[0] for m in ALLOWED_IMPORTS}
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed_roots:
                    problems.append(f"import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in allowed_roots:
                problems.append(f"import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Delete):
            problems.append("`del` is not allowed: a step may only add to the canvas")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"use of {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                problems.append(f"dunder attribute {node.attr!r} is not allowed")
            elif node.attr in BANNED_ATTRS:
                problems.append(
                    f"{node.attr!r} is not allowed: it breaks the shared "
                    "incremental canvas or escapes the process"
                )
    return problems
