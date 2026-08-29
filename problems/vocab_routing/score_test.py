"""Held-out reporting for an evolved routing program.

Delegates to ``validate.score_on_test``, which forces the full-transplant rung.
Kept separate and never called during search: a number that has been selected on
is not a held-out number, and this study's whole methodology rests on that line.
"""

from __future__ import annotations

from validate import score_on_test  # noqa: F401
