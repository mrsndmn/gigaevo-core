"""Prompt + response-schema construction for the synthesis run."""

from __future__ import annotations

import json
from typing import Any

from .configs import GenConfig
from .dataset import named_points

SYSTEM = (
    "You are a geometry teacher who explains a solution as a sequence of "
    "drawings. Each step of your reasoning is paired with Python/matplotlib "
    "code that adds exactly what that step introduces to a shared, persistent "
    "canvas. You are precise: the arithmetic in the text and the geometry in "
    "the code always agree, and the code always runs."
)

CONTRACT = """\
## The drawing environment

The following code has ALREADY been executed. Your step snippets run after it,
in the same namespace, in order:

```python
{preamble}
```

## Rules for the code of every step

1. A step's `code` is a FRAGMENT, not a program: no `def main()`, no imports
   beyond `math` / `numpy` / `matplotlib`, no `if __name__`.
2. All steps share ONE figure and ONE axes. Code may only ADD to the canvas.
   Never call `ax.clear()`, `ax.cla()`, `plt.clf()`, `plt.close()`,
   `plt.show()`, `plt.figure()`, `plt.subplots()`, `fig.savefig()` or
   `<artist>.remove()`. The figure is saved after every step, so the image of
   step k must show everything from steps 1..k-1 PLUS what step k adds.
3. Every step must draw something visible: a point, a segment, a line, a
   circle, an arc, a shaded polygon, an annotation with a computed number.
   A step whose code changes nothing on the canvas is a broken step.
4. Do not set the axes title — the title is set automatically from your
   step `title`. Legends, labels and annotations are welcome.
5. Use `P` for the given points instead of retyping their coordinates
   (`P` is {points_hint}). Use `mark(name, xy)` for new points you construct
   and `expand_view(...)` whenever a construction leaves the current view box —
   after adding a circle of centre `O` and radius `r`, call
   `expand_view(O + r, O - r)` so the whole circle stays visible.
6. Store numbers you compute in `state` (e.g. `state["area"] = 17.5`) and
   reuse them in later steps instead of recomputing.
7. Keep the code short (typically 3-12 lines) and deterministic: no randomness,
   no file or network access, no `input()`.
8. The LAST step must close the loop between the prose and the picture:
   - compute the answer *in code* (from the constructions of the earlier
     steps — do not paste a literal you worked out in your head),
   - draw it: `mark("O", O)` for a point answer, an annotation or shading
     carrying the number for a numeric answer, a text label for a choice,
   - assign it: `state["answer"] = {answer_assignment}`
   Both are checked. `state["answer"]` is compared against the golden answer,
   and for point answers the canvas is scanned for a marked point at the golden
   coordinates. A solution whose text is right but whose drawing lands
   somewhere else is rejected, so verify your signs and directions.

## Rules for the reasoning

- {min_steps}-{max_steps} steps, each a genuine advance of the solution, in the
  order a person would actually solve it: set up what is given, construct the
  auxiliary objects, compute, then read off the answer.
- `reasoning` is 1-4 sentences of plain prose with the actual numbers in it
  (e.g. "Midpoint of BC is (-6, -7), and BC is vertical, so its perpendicular
  bisector is the horizontal line y = -7."). No markdown headers, no code.
- The last step must state and draw the final answer.

## Example of the style (a different problem — do not copy the content)

step title: "Perpendicular bisector of AB"
step reasoning: "The midpoint of AB is (0.5, 3.0) and AB has direction
(7, -2), so the perpendicular bisector runs through (0.5, 3.0) along (2, 7)."
step code:
```python
mid_ab = (P["A"] + P["B"]) / 2
d = P["B"] - P["A"]
n = np.array([-d[1], d[0]], dtype=float)
n = n / np.linalg.norm(n)
t = np.linspace(-12, 12, 2)
pts = mid_ab[None, :] + t[:, None] * n[None, :]
ax.plot(pts[:, 0], pts[:, 1], color="tab:green", lw=1.6, ls="--",
        label="perp. bisector of AB")
mark("M_AB", mid_ab, color="tab:green")
state["mid_ab"] = mid_ab
ax.legend(loc="best", fontsize=8)
```
"""


def _answer_schema(row: dict[str, Any]) -> dict[str, Any]:
    schema_type = row.get("schema_type")
    if schema_type == "classification":
        choices = row.get("choices") or []
        node: dict[str, Any] = {"type": "string"}
        if choices:
            node["enum"] = list(choices)
        return node
    if schema_type == "point":
        return {
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["x", "y"],
            "additionalProperties": False,
        }
    return {"type": "number"}


def response_schema(row: dict[str, Any], cfg: GenConfig) -> dict[str, Any]:
    return {
        "name": "noregeo_visual_solution",
        "schema": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "One sentence: the route to the answer.",
                },
                "steps": {
                    "type": "array",
                    "minItems": cfg.min_steps,
                    "maxItems": cfg.max_steps,
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "reasoning": {"type": "string"},
                            "code": {"type": "string"},
                        },
                        "required": ["title", "reasoning", "code"],
                        "additionalProperties": False,
                    },
                },
                "final_answer": _answer_schema(row),
            },
            "required": ["plan", "steps", "final_answer"],
            "additionalProperties": False,
        },
    }


def _answer_format_hint(row: dict[str, Any]) -> str:
    schema_type = row.get("schema_type")
    if schema_type == "classification":
        choices = row.get("choices") or []
        return f"`final_answer` is one of: {choices}."
    if schema_type == "point":
        return '`final_answer` is an object {"x": <float>, "y": <float>}.'
    return "`final_answer` is a single number."


def _answer_assignment(row: dict[str, Any]) -> str:
    schema_type = row.get("schema_type")
    if schema_type == "classification":
        choices = row.get("choices") or []
        opts = " | ".join(f'"{c}"' for c in choices) or '"<your choice>"'
        return opts
    if schema_type == "point":
        return "np.array([x, y], dtype=float)  # the answer point"
    return "float(<the computed value>)"


def build_messages(
    row: dict[str, Any], cfg: GenConfig, preamble: str
) -> list[dict[str, str]]:
    pts = named_points(row)
    points_hint = (
        f"keyed by {sorted(pts)}"
        if pts
        else "EMPTY for this task, so take the coordinates from the problem text"
    )
    contract = CONTRACT.format(
        preamble=preamble.rstrip(),
        points_hint=points_hint,
        min_steps=cfg.min_steps,
        max_steps=cfg.max_steps,
        answer_assignment=_answer_assignment(row),
    )

    problem = [
        "## Problem",
        "",
        f"Category: {row.get('category')} (difficulty {row.get('difficulty')})",
        f"Answer schema: {row.get('schema_type')}",
        "",
        row.get("question", ""),
    ]
    if row.get("choices"):
        problem.append(f"Options: {row['choices']}")
    problem.append("")
    problem.append(_answer_format_hint(row))
    if cfg.reveal_answer:
        problem += [
            "",
            f"The verified correct answer is: {json.dumps(row.get('answer'))}. "
            "Your steps must genuinely derive it — do not skip to it, and do "
            "not mention that it was given to you.",
        ]

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n".join(problem) + "\n\n" + contract},
    ]


def repair_messages(
    base: list[dict[str, str]],
    previous: dict[str, Any],
    failure: str,
) -> list[dict[str, str]]:
    """Feed an execution failure back for one repair round."""
    return base + [
        {"role": "assistant", "content": json.dumps(previous, ensure_ascii=False)},
        {
            "role": "user",
            "content": (
                "Your solution did not survive execution:\n\n"
                f"{failure}\n\n"
                "Return the FULL corrected JSON object again (same structure, "
                "same rules). Fix the broken step and any later step that "
                "depended on it; keep the parts that were fine."
            ),
        },
    ]
