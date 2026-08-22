"""Run the whole pipeline over a folder of REAL files and report what breaks.

    python tests/test_corpus.py                 # sample 40 files
    python tests/test_corpus.py --limit 150
    python tests/test_corpus.py --root "D:/some/folder" --max-mb 40
    python tests/test_corpus.py --only glide    # substring filter on the path

Why this exists. Every serious bug in this tool was found by pointing it at one
real file: a title row above the table, a GA4 comment block, a sheet that was the
same records as another, a phone column reported as a mean. The synthetic
fixtures in `test_edges.py` were written *after* each discovery and they only
protect against the shapes already seen. This walks a real archive — 803 files
and 880 MB in the folder it was written against — so the next unseen shape is
found by a command rather than by a colleague.

**No data is committed and none is printed.** The report carries file names,
shapes, timings and error types only: never a cell value, never a column of
someone's answers. The corpus lives outside the repository and the runner is
excluded from CI, because CI has no access to it and should not.

Outcomes, in order of seriousness:

  ERROR  — an exception. Always a bug in the tool: a file that cannot be read
           should degrade to a thin page, never a traceback.
  EMPTY  — zero charts and *nothing said about why*. This is the one to look
           at: it is where a silent misread hides, like the
           8,750-rows-two-columns GA4 read. There should be none of these.
  NOTTABLE — nothing to chart, and the tool says why on the page: a form
           definition, a cover sheet, a pivot layout, a list with no numbers,
           a workbook Excel cannot open. A good outcome.
  SLOW   — slower than 20s + 2s per MB. Scaled by size, so a 22 MB workbook
           taking half a minute is fine and a 5 MB one taking four is not.
  OK     — rows, columns and at least one chart.

Compare two runs to see whether a change helped: each run writes a JSON summary
next to the log, and `--baseline <file>` diffs against it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import charts, insights, metrics, profiling as pf, report  # noqa: E402

DEFAULT_ROOTS = [
    Path(r"C:\Users\SolomonOladimeji\OneDrive - Health Watch Foundation\Download"),
    Path.home() / "Downloads",
]
SUFFIXES = {".csv", ".xlsx", ".xlsm", ".tsv"}
if pf._CALAMINE:
    SUFFIXES.add(".xls")            # readable only with the faster engine
# Scaled by size, because "slow" means slow *for this file*. A flat 25s
# flagged the two 22 MB, 600-column ICL exports — where pandas spends the time
# converting 13 million cells and nothing is wrong — while a 5 MB file taking
# 261s sat just as far over the line. Size-aware, the second one still shows.
SLOW_BASE, SLOW_PER_MB = 20.0, 2.0


def slow_after(mb: float) -> float:
    return SLOW_BASE + SLOW_PER_MB * max(mb, 0.0)


def find_files(root: Path, max_mb: float, only: str | None) -> list:
    out = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in SUFFIXES or not p.is_file():
            continue
        if p.name.startswith(("~$", ".")):          # Excel lock files
            continue
        if only and only.lower() not in str(p).lower():
            continue
        try:
            if p.stat().st_size > max_mb * 1024 * 1024:
                continue
        except OSError:
            continue
        out.append(p)
    return out


def sample(files: list, limit: int, seed: int) -> list:
    """Spread the sample across folders rather than taking the first N.

    An archive is organised by month, so the first N files are all from one
    month and one project — which is the opposite of the coverage this is for.
    """
    by_dir: dict = {}
    for f in files:
        by_dir.setdefault(f.parent, []).append(f)
    rng = random.Random(seed)
    for group in by_dir.values():
        rng.shuffle(group)
    picked, dirs = [], sorted(by_dir, key=lambda d: str(d))
    while len(picked) < limit and any(by_dir[d] for d in dirs):
        for d in dirs:
            if by_dir[d] and len(picked) < limit:
                picked.append(by_dir[d].pop())
    return picked


def exercise(path: Path) -> dict:
    """One file, all the way through. Never raises."""
    started = time.perf_counter()
    row = {"file": str(path), "name": path.name,
           "mb": round(path.stat().st_size / 1048576, 2),
           "rows": 0, "cols": 0, "charts": 0, "findings": 0,
           "measures": 0, "sheets": "", "status": "OK", "detail": ""}
    try:
        data = path.read_bytes()
        sheets = pf.sheet_names(data, path.name)
        sheet = pf.ALL_SHEETS if len(sheets) > 1 else (sheets[0] if sheets else None)
        prof = pf.build(data, path.name, sheet=sheet, row_cap=200_000)
        row.update(rows=prof.rows, cols=len(prof.columns),
                   sheets=prof.sheet or "")
        row["measures"] = len([m for m in metrics.measures(prof, limit=4)
                               if m.col != metrics.RECORDS.col])
        found = insights.analyse(prof)
        row["findings"] = len(found)
        broken = [f.text for f in found if "could not run" in f.text]
        panels = charts.dashboard(prof, limit=6)
        row["charts"] = len(panels)
        # One export, to catch a chart that renders on screen but not to file.
        for q in panels:
            if q.chart is not None:
                if not report.chart_svg(q):
                    row.update(status="ERROR",
                               detail=f"chart did not export: {q.title[:50]}")
                break
        if broken:
            row.update(status="ERROR", detail="rule crashed: " + broken[0][:80])
        elif row["charts"] == 0:
            # The page must say why it is empty. Either profiling recognised
            # the shape, or charts.why_nothing names what was missing. Only a
            # file that gets neither is worth investigating: that is where a
            # silent misread hides.
            said = prof.shape_note or charts.why_nothing(prof)
            if said:
                row.update(status="NOTTABLE", detail=said[:100])
            else:
                row.update(status="EMPTY",
                           detail="nothing to chart, and nothing said about why")
        elif prof.shape_note:
            row.update(status="NOTTABLE", detail=prof.shape_note[:90])
    except ValueError as exc:
        # A ValueError from build is a deliberate refusal with a sentence for
        # the reader — an .xls, a workbook openpyxl cannot parse. That is the
        # designed outcome, not a crash.
        row.update(status="NOTTABLE", detail=str(exc)[:110])
    except Exception as exc:                        # noqa: BLE001 - that is the point
        row.update(status="ERROR",
                   detail=f"{type(exc).__name__}: {str(exc)[:120]}")
        row["traceback"] = traceback.format_exc(limit=6)
    row["seconds"] = round(time.perf_counter() - started, 1)
    limit = slow_after(row["mb"])
    if row["status"] == "OK" and row["seconds"] > limit:
        row.update(status="SLOW",
                   detail=f"{row['seconds']}s for {row['mb']} MB "
                          f"(over {limit:.0f}s)")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--max-mb", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    roots = [Path(args.root)] if args.root else DEFAULT_ROOTS
    root = next((r for r in roots if r.exists()), None)
    if root is None:
        print("No corpus folder found. Pass --root, or set one of:")
        for r in roots:
            print(f"   {r}")
        print("\nThis runner needs real files and is deliberately not part of "
              "the committed test suite.")
        return 0

    files = find_files(root, args.max_mb, args.only)
    picked = sample(files, args.limit, args.seed)
    print(f"corpus: {root}")
    print(f"{len(files)} candidate files under {args.max_mb} MB; "
          f"running {len(picked)}\n")
    print(f"{'':4} {'status':6} {'rows':>8} {'col':>4} {'ch':>3} {'fnd':>4} "
          f"{'sec':>5}  file")

    rows = []
    for i, path in enumerate(picked, 1):
        r = exercise(path)
        rows.append(r)
        print(f"{i:>4} {r['status']:6} {r['rows']:>8,} {r['cols']:>4} "
              f"{r['charts']:>3} {r['findings']:>4} {r['seconds']:>5} "
              f" {r['name'][:58]}")
        if r["detail"]:
            print(f"{'':22}{r['detail'][:96]}")

    counts: dict = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\n" + " · ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    total_time = sum(r["seconds"] for r in rows)
    print(f"{total_time:.0f}s total, "
          f"{total_time / max(len(rows), 1):.1f}s per file")

    errors = [r for r in rows if r["status"] == "ERROR"]
    if errors:
        print(f"\n{len(errors)} ERROR(S) — these are tool bugs:")
        for r in errors:
            print(f"   {r['name'][:60]}")
            print(f"      {r['detail'][:150]}")
    empties = [r for r in rows if r["status"] == "EMPTY"]
    if empties:
        print(f"\n{len(empties)} EMPTY — check each: a notes sheet is fine, a "
              "silent misread is not:")
        for r in empties:
            print(f"   {r['name'][:60]} — {r['detail']}")

    out = Path(args.out) if args.out else (
        ROOT / "samples" / "_corpus_last_run.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    slim = [{k: v for k, v in r.items() if k != "traceback"} for r in rows]
    out.write_text(json.dumps(slim, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")

    if args.baseline:
        base = {r["file"]: r for r in json.loads(
            Path(args.baseline).read_text(encoding="utf-8"))}
        changed = []
        for r in rows:
            was = base.get(r["file"])
            if was and was["status"] != r["status"]:
                changed.append((r["name"], was["status"], r["status"]))
        print(f"\nagainst baseline: {len(changed)} status change(s)")
        for name, a, b in changed:
            print(f"   {name[:52]}  {a} -> {b}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
