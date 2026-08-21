"""Awkward uploads must degrade, never crash.

Every case here is a file somebody will actually drop in: three rows typed by
hand, one column, headers with no data, a monitoring sheet that is all text, a
crosstab pasted from a report, and a 100k-row export. A traceback on the screen
is worse than a thin page, because the user concludes the tool is broken and
goes back to Excel for good.

    python tests/test_edges.py
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import charts, insights, profiling as pf, report  # noqa: E402

fails = []


def run(label: str, frame: pd.DataFrame, as_csv: bool = False) -> None:
    """Push one frame through the whole pipeline and report what came out."""
    buf = io.BytesIO()
    if as_csv:
        data = frame.to_csv(index=False).encode("utf-8")
        name = "hand_typed.csv"
    else:
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            frame.to_excel(xw, index=False, sheet_name="Sheet1")
        data, name = buf.getvalue(), "upload.xlsx"
    t0 = time.perf_counter()
    try:
        prof = pf.build(data, name)
        found = insights.analyse(prof)
        panels = [p for p in charts.auto_panels(prof) if p]
        extras = [charts.missing_bar(prof), charts.corr_grid(prof),
                  charts.type_table(prof), charts.time_trend(prof)]
        panels += [p for p in extras if p]
        html = report.html_report(prof, found, panels, title=label)
        xl = report.excel_book(prof, found, panels)
        broke = [f.text for f in found if "could not run" in f.text]
        dt = time.perf_counter() - t0
        ok = not broke and len(html) > 2000 and len(xl) > 2000
        print(f"  {'PASS' if ok else 'FAIL'}  {label:34} "
              f"{prof.rows:>7,} rows {len(prof.columns):>3} cols  "
              f"{len(found):>2} findings {len(panels):>2} panels  {dt:5.2f}s")
        if broke:
            print("        broken rules: " + "; ".join(broke)[:200])
        if not ok:
            fails.append(label)
    except Exception as exc:
        import traceback
        print(f"  FAIL  {label:34} raised {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=4)
        fails.append(label)


def main() -> int:
    rng = np.random.default_rng(11)
    print("\n--- awkward uploads ---")

    run("three rows typed by hand",
        pd.DataFrame({"Name": ["Aisha", "Chidi", "Tunde"],
                      "Score": [4, 9, 7]}), as_csv=True)

    run("one column only", pd.DataFrame({"state": ["Kano", "Lagos"] * 30}))

    run("headers but no data rows",
        pd.DataFrame({"a": pd.Series(dtype="float64"),
                      "b": pd.Series(dtype="object")}))

    run("all free text",
        pd.DataFrame({"note": [f"A sentence of some length number {i}, "
                               "written by a field officer." for i in range(80)]}))

    run("all numeric, no dates",
        pd.DataFrame(rng.normal(50, 12, (400, 6)),
                     columns=[f"indicator_{i}" for i in range(6)]))

    run("every value missing in one column",
        pd.DataFrame({"lga": ["A", "B"] * 50, "target": [np.nan] * 100}))

    run("one value repeated everywhere",
        pd.DataFrame({"programme": ["GLIDE"] * 200, "year": [2026] * 200}))

    run("crosstab pasted from a report",
        pd.DataFrame({"Unnamed: 0": ["Kano", "Lagos", "Total"],
                      "2024": [12, 18, 30], "2025": [15, 21, 36]}))

    run("dates as text in three formats",
        pd.DataFrame({"visit": (["2026-01-04"] * 30 + ["05/02/2026"] * 30
                                + ["2026-03-06"] * 30),
                      "n": rng.integers(1, 9, 90)}))

    run("duplicated header names",
        pd.DataFrame(np.arange(60).reshape(20, 3), columns=["x", "x", "y"]))

    run("commas and quotes in a CSV",
        pd.DataFrame({"facility": ['Clinic "A", Kano', "Ward B, Lagos"] * 25,
                      "beds": [12, 30] * 25}), as_csv=True)

    run("place names from another country",
        pd.DataFrame({"state": ["Accra", "Kumasi", "Tamale"] * 30,
                      "cases": rng.integers(1, 40, 90)}))

    run("one state, LGAs spelled loosely",
        pd.DataFrame({"state": ["Kano"] * 80,
                      "lga": rng.choice(["Dala", "Fagge", "Nassarawa"], 80),
                      "n": rng.integers(1, 9, 80)}))

    print("\n--- maps ---")
    # A map must refuse to draw rather than draw something misleading: two
    # shaded polygons and thirty-five grey ones is not a map of anything.
    ghana = pd.DataFrame({"state": ["Accra", "Kumasi"] * 40, "n": range(80)})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        ghana.to_excel(xw, index=False)
    prof = pf.build(buf.getvalue(), "gh.xlsx")
    drawn = charts.state_map(prof, "state")
    ok = drawn is None
    print(f"  {'PASS' if ok else 'FAIL'}  no map when nothing matches a Nigerian "
          "boundary")
    if not ok:
        fails.append("map drawn from unmatchable place names")

    print("\n--- size ---")
    n = 100_000
    big = pd.DataFrame({
        "submission_date": pd.Timestamp("2026-01-01") + pd.to_timedelta(
            rng.integers(0, 200, n), unit="D"),
        "state": rng.choice(["Kano", "Lagos", "Borno", "Oyo", "FCT"], n),
        "lga": rng.choice([f"LGA {i}" for i in range(25)], n),
        "sex": rng.choice(["Female", "Male"], n),
        "age": rng.integers(15, 80, n),
        "wait_minutes": rng.normal(45, 15, n).round(1),
        "visits": rng.poisson(3, n),
        "satisfied": rng.choice(["Yes", "No"], n),
    })
    run("100,000 rows x 8 cols", big)

    print("\nFAILURES: " + (", ".join(fails) if fails else "none"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
