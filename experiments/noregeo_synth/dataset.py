"""Fetch and index the NoReGeo benchmark tasks.

The HF auto-converted parquet for `FusionBrainLab/NoReGeo` is an imagefolder
(`image` + `label` only) — the question/answer text lives in 27 JSONL files in
the repo tree, which this module downloads (once) into `data/noregeo/`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import urllib.request

from .configs import DATA_DIR, HF_REPO, HF_RESOLVE, HF_REVISION, TASK_FILES


@dataclass(frozen=True)
class Task:
    """One benchmark row plus where it came from."""

    stem: str  # e.g. "outer_circle_center" (task file name minus _tasks.jsonl)
    file: str  # repo-relative path of the JSONL
    row: dict[str, Any]

    @property
    def qid(self) -> int:
        return int(self.row["qid"])

    @property
    def task_id(self) -> str:
        return f"{self.stem}:{self.qid}"


def _download(path: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = HF_RESOLVE.format(repo=HF_REPO, rev=HF_REVISION, path=path)
    req = urllib.request.Request(url, headers={"User-Agent": "noregeo-synth/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def ensure_task_files(files: tuple[str, ...] = TASK_FILES) -> list[Path]:
    return [_download(f, DATA_DIR / f) for f in files]


def fetch_image(repo_path: str) -> Path:
    """Download one of the benchmark's own reference images."""
    rel = repo_path.lstrip("/")
    return _download(rel, DATA_DIR / rel)


def load_tasks(files: tuple[str, ...] = TASK_FILES) -> list[Task]:
    tasks: list[Task] = []
    for f in files:
        local = _download(f, DATA_DIR / f)
        stem = Path(f).name.removesuffix("_tasks.jsonl")
        for line in local.read_text().splitlines():
            if line.strip():
                tasks.append(Task(stem=stem, file=f, row=json.loads(line)))
    return tasks


def select(tasks: list[Task], spec: str) -> Task:
    """Resolve a "<stem>:<qid>" or bare "<qid>" selector to a single task."""
    if ":" in spec:
        stem, _, qid = spec.partition(":")
        hits = [t for t in tasks if t.stem == stem and t.qid == int(qid)]
    else:
        hits = [t for t in tasks if t.qid == int(spec)]
    if not hits:
        raise KeyError(f"no NoReGeo task matches {spec!r}")
    if len(hits) > 1:
        ids = ", ".join(t.task_id for t in hits)
        raise KeyError(
            f"{spec!r} is ambiguous (qid is unique only within a file): {ids}"
        )
    return hits[0]


# ---------------------------------------------------------------------------
# Geometry extracted from a row
# ---------------------------------------------------------------------------


def named_points(row: dict[str, Any]) -> dict[str, list]:
    """Normalise the `points` field to {name: point-or-list-of-points}.

    Three shapes occur in the benchmark: a dict ({"A": [x, y]}, and also
    {"polygon": [[x, y], ...]}), a bare list of points (named A, B, C... in
    order), or null (area-comparison tasks, where the coordinates only appear
    in the question text).
    """
    pts = row.get("points")
    if pts is None:
        return {}
    if isinstance(pts, dict):
        return {str(k): v for k, v in pts.items()}
    names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return {names[i % 26]: p for i, p in enumerate(pts)}


def _flatten_pairs(obj: Any, out: list[tuple[float, float]]) -> None:
    if (
        isinstance(obj, (list, tuple))
        and len(obj) == 2
        and all(isinstance(v, (int, float)) for v in obj)
    ):
        out.append((float(obj[0]), float(obj[1])))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _flatten_pairs(item, out)
    elif isinstance(obj, dict):
        for item in obj.values():
            _flatten_pairs(item, out)


def coordinate_pairs(row: dict[str, Any]) -> list[tuple[float, float]]:
    """Every (x, y) pair we can see for the task, for the initial view box.

    Uses `points` when present and additionally the golden answer (so a
    circumcenter far outside the triangle still fits on the canvas). Falls back
    to parsing bracketed pairs out of the question text for rows where
    `points` is null.
    """
    pairs: list[tuple[float, float]] = []
    _flatten_pairs(row.get("points"), pairs)
    ans = row.get("answer")
    if isinstance(ans, dict):
        _flatten_pairs(ans, pairs)
    if not pairs:
        import re

        num = r"-?\d+(?:\.\d+)?"
        for m in re.finditer(
            rf"[\[(]\s*({num})\s*,\s*({num})\s*[\])]", str(row.get("question", ""))
        ):
            pairs.append((float(m.group(1)), float(m.group(2))))
    return pairs
