"""Compare two WebArena multiworker runs (TSA vs dense SGLang) by pass rate.

Both inputs are ``scores.jsonl`` files in the format produced by
``mp.orchestrator``: one JSON object per line with at least the keys
``task_id``, ``score`` (float, where 1.0 means pass), ``error``,
``inference_backend``, ``model``, and ``openai_api_base``.

Outputs an executive markdown report and a per-task CSV, including:

* Overall pass rate per backend on the intersection of completed task_ids.
* Per-site pass rate (the site for a task is read from the WebArena task
  config file ``test.raw.json``).
* A 2×2 agreement matrix (TSA pass × dense pass).
* Bootstrap 95% CI on the pass-rate delta (TSA − dense).
* Header records capturing model/endpoint provenance from both runs.

The tool is strict about input integrity: if the two runs do not cover the
same set of task_ids, it raises so the comparison cannot accidentally be
made on a subset of the benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Row:
    task_id: int
    score: float | None
    error: str | None
    inference_backend: str | None
    model: str | None
    openai_api_base: str | None
    eval_api_base: str | None
    duration_seconds: float | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Row":
        return cls(
            task_id=int(d["task_id"]),
            score=(float(d["score"]) if d.get("score") is not None else None),
            error=d.get("error"),
            inference_backend=d.get("inference_backend"),
            model=d.get("model"),
            openai_api_base=d.get("openai_api_base"),
            eval_api_base=d.get("eval_api_base"),
            duration_seconds=(
                float(d["duration_seconds"]) if d.get("duration_seconds") is not None else None
            ),
        )


def _load_scores(path: Path) -> dict[int, Row]:
    """Read a scores.jsonl file; later rows for the same task_id win (re-runs)."""
    rows: dict[int, Row] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = Row.from_dict(obj)
            rows[row.task_id] = row
    return rows


def _load_task_sites(tasks_json: Path | None) -> dict[int, str]:
    """Map task_id → primary site (first entry of `sites` list).

    Returns an empty dict when the tasks file is unavailable; the report
    simply omits per-site breakdowns in that case.
    """
    if tasks_json is None or not tasks_json.exists():
        return {}
    out: dict[int, str] = {}
    with tasks_json.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        return {}
    for entry in data:
        try:
            tid = int(entry["task_id"])
        except Exception:
            continue
        sites = entry.get("sites", []) or []
        if sites:
            out[tid] = str(sites[0])
    return out


def _pass_rate(rows: dict[int, Row], task_ids: set[int]) -> tuple[float, int, int]:
    """Return (rate, numerator, denominator) over the given task_ids."""
    n = 0
    passes = 0
    for tid in task_ids:
        row = rows.get(tid)
        if row is None:
            continue
        n += 1
        if row.error is None and row.score is not None and row.score >= 1.0:
            passes += 1
    return (passes / n if n else 0.0, passes, n)


def _bootstrap_ci(
    rows_a: dict[int, Row],
    rows_b: dict[int, Row],
    task_ids: list[int],
    n_iter: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap 95% CI on (pass_rate_a − pass_rate_b) over the task_ids list."""
    if not task_ids:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(task_ids)
    deltas: list[float] = []
    for _ in range(n_iter):
        sample = [task_ids[rng.randrange(n)] for _ in range(n)]
        a = sum(
            1
            for t in sample
            if (r := rows_a.get(t)) is not None
            and r.error is None
            and r.score is not None
            and r.score >= 1.0
        ) / n
        b = sum(
            1
            for t in sample
            if (r := rows_b.get(t)) is not None
            and r.error is None
            and r.score is not None
            and r.score >= 1.0
        ) / n
        deltas.append(a - b)
    deltas.sort()
    return (deltas[int(0.025 * n_iter)], deltas[int(0.975 * n_iter)])


def _provenance(rows: dict[int, Row]) -> dict[str, str]:
    """Most-common provenance values across the rows."""
    pick = lambda field: Counter(  # noqa: E731 — small helper
        getattr(r, field) for r in rows.values() if getattr(r, field) is not None
    ).most_common(1)
    return {
        "inference_backend": pick("inference_backend")[0][0] if pick("inference_backend") else "?",
        "model": pick("model")[0][0] if pick("model") else "?",
        "openai_api_base": pick("openai_api_base")[0][0] if pick("openai_api_base") else "?",
        "eval_api_base": pick("eval_api_base")[0][0] if pick("eval_api_base") else "?",
    }


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    sep = ["---"] * len(headers)
    out = ["| " + " | ".join(map(str, headers)) + " |",
           "| " + " | ".join(sep) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare TSA vs dense pass rate on WebArena")
    p.add_argument("--tsa", required=True, type=Path, help="scores.jsonl from the TSA run")
    p.add_argument("--dense", required=True, type=Path, help="scores.jsonl from the dense run")
    p.add_argument("--tasks", type=Path, default=None,
                   help="config_files/test.raw.json (for per-site breakdowns)")
    p.add_argument("--out", type=Path, default=Path("comparison_report.md"))
    p.add_argument("--csv", type=Path, default=Path("comparison.csv"))
    p.add_argument("--strict-task-ids", action="store_true", default=True,
                   help="Require both runs to cover the same task_id set (default: on)")
    p.add_argument("--no-strict-task-ids", action="store_false", dest="strict_task_ids")
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    args = p.parse_args(argv)

    tsa_rows = _load_scores(args.tsa)
    dense_rows = _load_scores(args.dense)
    task_sites = _load_task_sites(args.tasks)

    tsa_ids = set(tsa_rows.keys())
    dense_ids = set(dense_rows.keys())
    common = tsa_ids & dense_ids
    only_tsa = tsa_ids - dense_ids
    only_dense = dense_ids - tsa_ids

    if args.strict_task_ids and (only_tsa or only_dense):
        print(
            f"FATAL: task_id sets differ. TSA-only={sorted(only_tsa)[:10]}... "
            f"({len(only_tsa)} extras), dense-only={sorted(only_dense)[:10]}... "
            f"({len(only_dense)} extras). Pass --no-strict-task-ids to compare "
            "the intersection anyway (NOT recommended).",
            file=sys.stderr,
        )
        return 2

    common_list = sorted(common)
    tsa_prov = _provenance(tsa_rows)
    dense_prov = _provenance(dense_rows)

    tsa_rate, tsa_pass, tsa_n = _pass_rate(tsa_rows, common)
    dense_rate, dense_pass, dense_n = _pass_rate(dense_rows, common)
    ci_lo, ci_hi = _bootstrap_ci(tsa_rows, dense_rows, common_list, args.bootstrap_iters)

    # Per-site breakdown.
    per_site_rows: list[list[Any]] = []
    if task_sites:
        per_site: dict[str, list[int]] = defaultdict(list)
        for tid in common_list:
            site = task_sites.get(tid, "?")
            per_site[site].append(tid)
        for site, ids in sorted(per_site.items()):
            tr, tp, tn = _pass_rate(tsa_rows, set(ids))
            dr, dp, dn = _pass_rate(dense_rows, set(ids))
            per_site_rows.append([site, len(ids),
                                  f"{tp}/{tn} ({100*tr:.1f}%)",
                                  f"{dp}/{dn} ({100*dr:.1f}%)",
                                  f"{100*(tr-dr):+.1f}pp"])

    # 2×2 agreement matrix.
    def _passed(r: Row | None) -> bool:
        return r is not None and r.error is None and r.score is not None and r.score >= 1.0

    both = sum(1 for t in common_list if _passed(tsa_rows.get(t)) and _passed(dense_rows.get(t)))
    only_t = sum(1 for t in common_list if _passed(tsa_rows.get(t)) and not _passed(dense_rows.get(t)))
    only_d = sum(1 for t in common_list if not _passed(tsa_rows.get(t)) and _passed(dense_rows.get(t)))
    neither = sum(1 for t in common_list if not _passed(tsa_rows.get(t)) and not _passed(dense_rows.get(t)))

    # Errors and incompletes.
    tsa_errors = sum(1 for t in common_list if (r := tsa_rows.get(t)) and r.error)
    dense_errors = sum(1 for t in common_list if (r := dense_rows.get(t)) and r.error)

    # Write per-task CSV.
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "task_id", "site",
            "tsa_score", "tsa_error", "tsa_duration_s",
            "dense_score", "dense_error", "dense_duration_s",
            "tsa_pass", "dense_pass", "agreement",
        ])
        for tid in common_list:
            t = tsa_rows.get(tid)
            d = dense_rows.get(tid)
            tp = _passed(t)
            dp = _passed(d)
            label = ("both" if tp and dp else
                     "tsa_only" if tp and not dp else
                     "dense_only" if not tp and dp else
                     "neither")
            w.writerow([
                tid, task_sites.get(tid, ""),
                t.score if t else "", (t.error or "") if t else "",
                t.duration_seconds if t else "",
                d.score if d else "", (d.error or "") if d else "",
                d.duration_seconds if d else "",
                int(tp), int(dp), label,
            ])

    # Write markdown report.
    md: list[str] = []
    md.append("# TSA vs Dense — WebArena pass-rate comparison")
    md.append("")
    md.append("## Run provenance")
    md.append(_md_table(
        ["", "TSA run", "Dense run"],
        [
            ["scores.jsonl", str(args.tsa), str(args.dense)],
            ["task_ids covered", len(tsa_ids), len(dense_ids)],
            ["task_ids in common", len(common), len(common)],
            ["inference_backend", tsa_prov["inference_backend"], dense_prov["inference_backend"]],
            ["model", tsa_prov["model"], dense_prov["model"]],
            ["openai_api_base", tsa_prov["openai_api_base"], dense_prov["openai_api_base"]],
            ["eval_api_base (judge)", tsa_prov["eval_api_base"], dense_prov["eval_api_base"]],
            ["non-judge errors", tsa_errors, dense_errors],
        ],
    ))
    md.append("")
    md.append("## Overall pass rate")
    md.append(_md_table(
        ["backend", "passes", "denominator", "rate"],
        [
            ["TSA",   tsa_pass,   tsa_n,   f"{100*tsa_rate:.2f}%"],
            ["Dense", dense_pass, dense_n, f"{100*dense_rate:.2f}%"],
        ],
    ))
    md.append("")
    md.append(f"**Delta (TSA − Dense)**: {100*(tsa_rate-dense_rate):+.2f} pp  "
              f"(bootstrap 95% CI: [{100*ci_lo:+.2f}, {100*ci_hi:+.2f}] pp, "
              f"{args.bootstrap_iters} iters)")
    md.append("")
    if per_site_rows:
        md.append("## Per-site pass rate")
        md.append(_md_table(
            ["site", "n", "TSA", "Dense", "delta"],
            per_site_rows,
        ))
        md.append("")
    md.append("## Agreement matrix (TSA pass × Dense pass)")
    md.append(_md_table(
        ["", "Dense passed", "Dense failed", "Row total"],
        [
            ["TSA passed", both, only_t, both + only_t],
            ["TSA failed", only_d, neither, only_d + neither],
            ["Column total", both + only_d, only_t + neither, len(common_list)],
        ],
    ))
    md.append("")
    md.append("## Files")
    md.append(f"- Per-task CSV: `{args.csv}`")
    md.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md))
    print(f"wrote {args.out} and {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
