"""Hourly TG-PDF run report for the 4 evo runs launched 2026-05-24 22:58.

Pipeline:
  1. gigaevo trajectory  -> per-gen JSON for each run
  2. gigaevo top -n 1    -> best program per run
  3. Redis dbsize        -> key activity indicator
  4. gigaevo plot comparison (paper styling) -> tabular pair + spherical pair PDF plots
  5. Assemble LaTeX report, tectonic-compile to PDF
  6. POST to Telegram via httpx through Squid HTTPS_PROXY
"""

from __future__ import annotations

import concurrent.futures
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import httpx
from gigaevo.utils.redis_conn import redis_connect_kwargs

GIGAEVO = "/home/jovyan/.mlspace/envs/evo/bin/gigaevo"
TECTONIC = "/home/user/conda/bin/tectonic"
REPORT_ROOT = Path("/home/jovyan/gigaevo/output/hourly_reports_20260524")

RUNS = [
    {
        "prefix": "tabular_regression_optuna",
        "db": 14,
        "label": "TR-Optuna",
        "group": "tabular",
    },
    {
        "prefix": "tabular_regression",
        "db": 15,
        "label": "TR-NoOptuna",
        "group": "tabular",
    },
    {
        "prefix": "spherical_codes_improver",
        "db": 0,
        "label": "SC-Improver",
        "group": "spherical",
    },
    {
        "prefix": "spherical_codes_baseline",
        "db": 1,
        "label": "SC-Baseline",
        "group": "spherical",
    },
]


def run_cli(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def query_run(run: dict) -> dict:
    spec = f"{run['prefix']}@{run['db']}:{run['label']}"
    try:
        traj_proc = run_cli(
            [GIGAEVO, "-r", spec, "trajectory", "--tail", "500"], timeout=120
        )
        traj = json.loads(traj_proc.stdout) if traj_proc.stdout.strip() else []
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        traj = []
    try:
        top_proc = run_cli([GIGAEVO, "-r", spec, "top", "-n", "1"], timeout=120)
        top = json.loads(top_proc.stdout) if top_proc.stdout.strip() else []
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        top = []

    try:
        import redis

        r = redis.Redis(**redis_connect_kwargs(db=run["db"]), socket_timeout=5)
        dbsize = r.dbsize()
    except Exception as e:
        dbsize = f"err:{type(e).__name__}"

    return {
        "label": run["label"],
        "prefix": run["prefix"],
        "db": run["db"],
        "group": run["group"],
        "redis_keys": dbsize,
        "gen_count": len(traj),
        "last_gen": traj[-1]["Gen"] if traj else None,
        "best_fitness": (
            top[0]["Fitness"] if top else (traj[-1]["Best"] if traj else None)
        ),
        "mean_fitness": traj[-1]["Mean"] if traj else None,
        "best_program_id": top[0]["ID"] if top else None,
        "best_program_state": top[0]["State"] if top else None,
    }


def make_plot(
    group: str, out_dir: Path
) -> tuple[Path | None, subprocess.CompletedProcess | None]:
    pair = [r for r in RUNS if r["group"] == group]
    args = [GIGAEVO]
    for r in pair:
        args += ["-r", f"{r['prefix']}@{r['db']}:{r['label']}"]
    args += [
        "plot",
        "comparison",
        "-o",
        str(out_dir),
        "--paper",
        "--smoothing",
        "ema",
        "--window",
        "5",
        "--annotate-frontier",
    ]
    try:
        proc = run_cli(args, timeout=300)
    except subprocess.TimeoutExpired as e:
        print(
            f"[plot {group}] subprocess timeout after {e.timeout}s — degrading to no-plot",
            file=sys.stderr,
        )
        return None, None
    pdf = out_dir / "evolution_runs_comparison.pdf"
    return (pdf if pdf.exists() else None), proc


def fmt_num(x, fmt=".5g"):
    if x is None:
        return "n/a"
    try:
        return format(float(x), fmt)
    except (TypeError, ValueError):
        return str(x)


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )


def write_tex(report_dir: Path, summary: list[dict], plots: dict[str, Path]) -> Path:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    rows = []
    for s in summary:
        rows.append(
            " & ".join(
                [
                    latex_escape(s["label"]),
                    latex_escape(s["prefix"]),
                    str(s["db"]),
                    str(s["last_gen"]) if s["last_gen"] is not None else "-",
                    fmt_num(s["best_fitness"]),
                    fmt_num(s["mean_fitness"]),
                    str(s["redis_keys"]),
                ]
            )
            + r" \\"
        )
    rows_tex = "\n".join(rows)

    def plot_block(group: str, header: str) -> str:
        p = plots.get(group)
        if p:
            return (
                rf"\section*{{{header}}}" + "\n"
                rf"\begin{{center}}\includegraphics[width=0.95\linewidth]{{{p}}}\end{{center}}"
            )
        return (
            rf"\section*{{{header}}}"
            + "\n"
            + r"\emph{Plot unavailable --- no trajectory data yet for both runs.}"
        )

    tex = (
        r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{graphicx,booktabs,xcolor,parskip}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\title{GigaEvo --- Hourly Run Report}
\author{Auto-generated via tools/hourly\_run\_report.py}
\date{"""
        + now
        + r"""}
\begin{document}
\maketitle

\section*{Run Lineup}

All 4 runs launched 2026-05-24 22:58 (UTC+3) after Redis flush, post mutation-prompt
CONTEXT-leak fix (commit \texttt{06ab4719}). Shared settings:
\texttt{pipeline=intra\_extra\_memory}, \texttt{num\_parents=2}, \texttt{max\_mutants=800},
\texttt{primary\_resolution=50}, \texttt{island\_max\_size=50}.

\begin{center}
\begin{tabular}{lllrrrr}
\toprule
Label & Prefix & DB & Last Gen & Best Fitness & Mean (last gen) & Redis Keys \\
\midrule
"""
        + rows_tex
        + r"""
\bottomrule
\end{tabular}
\end{center}

"""
        + plot_block(
            "tabular", "Tabular Regression: Optuna vs No-Optuna (Qwen3-235B-Thinking)"
        )
        + r"""

"""
        + plot_block(
            "spherical", "Spherical Codes: Baseline vs Improver (gemini-3-flash)"
        )
        + r"""

\section*{Treatments}
\begin{itemize}
  \item \textbf{Tabular pair} --- both run Qwen3-235B-A22B-Thinking-2507 via LiteLLM proxy on the
        \texttt{tabular\_regression} problem family. TR-Optuna enables the optuna-driven HP-tuning
        stage (\texttt{enable\_optuna\_stage=true}); TR-NoOptuna does not.
  \item \textbf{Spherical pair} --- both run \texttt{gemini-3-flash} via OpenRouter on the
        \texttt{spherical\_codes} family. SC-Baseline is the bare problem; SC-Improver wraps the
        same setup with the improver scaffold.
\end{itemize}

\end{document}
"""
    )
    p = report_dir / "report.tex"
    p.write_text(tex)
    return p


def compile_tex(tex_path: Path) -> tuple[Path | None, subprocess.CompletedProcess]:
    proc = subprocess.run(
        [TECTONIC, "--keep-logs", "--chatter", "minimal", str(tex_path)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=tex_path.parent,
    )
    pdf = tex_path.with_suffix(".pdf")
    return (pdf if pdf.exists() else None), proc


def send_telegram(pdf_path: Path, caption: str) -> tuple[int, str]:
    bot = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    proxy = os.environ.get("HTTPS_PROXY")
    url = f"https://api.telegram.org/bot{bot}/sendDocument"
    with httpx.Client(proxy=proxy, timeout=120.0) as c:
        with open(pdf_path, "rb") as f:
            files = {"document": (pdf_path.name, f, "application/pdf")}
            data = {"chat_id": chat_id, "caption": caption[:1000]}
            r = c.post(url, data=data, files=files)
    return r.status_code, r.text[:500]


def main() -> int:
    if not (
        os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID")
    ):
        print("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing", file=sys.stderr)
        return 2

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_dir = REPORT_ROOT / ts
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{ts}] Querying runs (parallel)...", file=sys.stderr)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(RUNS)) as ex:
        summary = list(ex.map(query_run, RUNS))
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    plots: dict[str, Path] = {}
    for group in ("tabular", "spherical"):
        plot_dir = report_dir / f"plots_{group}"
        plot_dir.mkdir(exist_ok=True)
        path, proc = make_plot(group, plot_dir)
        if proc is not None:
            (report_dir / f"plot_{group}.log").write_text(
                f"args:\n{' '.join(proc.args)}\n\nrc={proc.returncode}\n\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )
        else:
            (report_dir / f"plot_{group}.log").write_text(
                "subprocess timed out — see stderr above\n"
            )
        if path:
            plots[group] = path
            print(f"[plot {group}] OK -> {path}", file=sys.stderr)
        else:
            rc = proc.returncode if proc is not None else "TIMEOUT"
            err = proc.stderr[:200] if proc is not None else ""
            print(f"[plot {group}] FAILED rc={rc}: {err}", file=sys.stderr)

    tex_path = write_tex(report_dir, summary, plots)
    pdf, tx = compile_tex(tex_path)
    (report_dir / "tectonic.log").write_text(
        f"rc={tx.returncode}\n\nstdout:\n{tx.stdout}\n\nstderr:\n{tx.stderr}"
    )
    if not pdf:
        print(f"[tectonic] FAILED rc={tx.returncode}", file=sys.stderr)
        print(tx.stderr[:1000], file=sys.stderr)
        return 3

    cap_lines = [f"GigaEvo hourly report {ts}"]
    for s in summary:
        cap_lines.append(
            f"  {s['label']}: gen={s['last_gen']} best={fmt_num(s['best_fitness'])} keys={s['redis_keys']}"
        )
    caption = "\n".join(cap_lines)
    status, body = send_telegram(pdf, caption)
    print(f"[TG] status={status}", file=sys.stderr)
    (report_dir / "telegram.log").write_text(f"status={status}\nbody={body}")
    if status != 200:
        return 4

    latest = REPORT_ROOT / "latest.pdf"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    try:
        latest.symlink_to(pdf)
    except OSError:
        shutil.copy(pdf, latest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
