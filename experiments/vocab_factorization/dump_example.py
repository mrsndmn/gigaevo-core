"""Dump a real worked example of both tokenizers, for the methods figure.

Everything the diagram shows comes from here, so the figure cannot drift away
from what the code actually does.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.vocab_factorization.surface import CASE_NAMES, segment
from experiments.vocab_factorization.tokenization import (
    ROLE_CONT,
    ROLE_RAW,
    load_arm,
    role_to_space_case,
)

DEFAULT_TEXT = "В городах России живут миллионы людей."


def role_label(r: int) -> str:
    if r == ROLE_CONT:
        return "CONT"
    if r == ROLE_RAW:
        return "RAW"
    space, case = role_to_space_case(r)
    return ("␣" if space else "") + CASE_NAMES[case]


def describe(arm, text: str) -> list[dict]:
    enc = arm.encode(text)
    rows = []
    for i in range(len(enc)):
        sid = int(enc.stem[i])
        piece = arm.tok.decode([sid]) if hasattr(arm, "tok") else chr(sid)
        row = {"piece": piece, "id": sid}
        if "role" in arm.heads:
            row["role"] = role_label(int(enc.role[i]))
        if "suffix" in arm.heads:
            sf = int(enc.suffix[i])
            row["suffix"] = arm.suffixes[sf - 1] if sf else ""
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--arms-dir", type=Path, default=Path("data/arms"))
    ap.add_argument("--out", type=Path, default=Path("results/example.json"))
    args = ap.parse_args()

    fa = load_arm(args.arms_dir / "fact16k")
    fn = load_arm(args.arms_dir / "fact8k-nosuf")
    bp = load_arm(args.arms_dir / "bpe16k")

    spans = [
        {
            "text": s.text,
            "kind": s.kind,
            "space": s.space,
            "case": CASE_NAMES[s.case],
            "core": s.core,
        }
        for s in segment(args.text)
    ]

    out = {"text": args.text, "spans": spans, "streams": {}}
    for label, t in (("clean", args.text), ("upper", args.text.upper())):
        out["streams"][label] = {
            "text": t,
            "bpe16k": describe(bp, t),
            "fact16k": describe(fa, t),
            "fact8k-nosuf": describe(fn, t),
        }
        # Roundtrip is the study's hard gate; assert it on the figure's own
        # example so the diagram can claim it truthfully.
        for arm in (bp, fa, fn):
            assert arm.decode(arm.encode(t)) == t, f"{arm.name} lost {label}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    print(f"TEXT: {args.text}")
    for s in spans:
        print(
            f"  {s['text']!r:14s} {s['kind']:5s} space={s['space']} "
            f"case={s['case']:6s} core={s['core']!r}"
        )
    for label, st in out["streams"].items():
        print(f"\n--- {label}: {st['text']}")
        for name in ("bpe16k", "fact16k", "fact8k-nosuf"):
            rows = st[name]
            print(f"  {name:14s} {len(rows):2d} tokens")
            for r in rows:
                extra = ""
                if "role" in r:
                    extra = f"  role={r['role']:8s}"
                if r.get("suffix"):
                    extra += f"  +suffix={r['suffix']!r}"
                print(f"      {r['piece']!r:16s}{extra}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
