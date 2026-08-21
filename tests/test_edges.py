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


def kobo_shaped() -> "pd.DataFrame":
    """A Kobo export the way the CBPS household survey arrives.

    Every one of these shapes broke something when the tool met real survey
    data: whole questions as column names, GPS columns, a select_all family
    whose parent column also exists, and yes/no attitude questions that are the
    only real measures in the file.
    """
    rng = np.random.default_rng(5)
    n = 400
    q = "Do you feel that the PHC facility is adequately equipped:"
    parent = "What factors influence your choice of PHC facility (select all that apply):"
    d = {
        "_id": range(1, n + 1),
        "Date": pd.Timestamp("2026-03-01") + pd.to_timedelta(
            rng.integers(0, 120, n), unit="D"),
        "State": rng.choice(["Kano", "Lagos", "Borno", "Enugu"], n),
        "LGA": rng.choice([f"LGA {i}" for i in range(12)], n),
        "_Geo-Point_latitude": rng.uniform(4, 13, n),
        "_Geo-Point_longitude": rng.uniform(3, 14, n),
        "_Geo-Point_altitude": rng.uniform(20, 500, n),
        "_Geo-Point_precision": rng.uniform(3, 20, n),
        "__version__": "v3",
        "deviceid": rng.choice([f"dev{i}" for i in range(6)], n),
        q: rng.choice(["Yes", "No"], n, p=[0.6, 0.4]),
        "Do you experience long wait times or delays when accessing care?":
            rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "What is the number of people living in the household":
            rng.integers(1, 12, n),
        "Sex of Household Head": rng.choice(["Male", "Female"], n),
        parent: ["Proximity to home Cost of services"] * n,
    }
    for opt in ["Proximity to home", "Cost of services", "Quality of care",
                "Availability of specific services"]:
        d[f"{parent}/{opt}"] = (rng.random(n) < 0.5).astype(int)
    return pd.DataFrame(d)


def kobo_checks() -> None:
    """The behaviours a real Kobo export taught us, on synthetic data."""
    from app import metrics
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        kobo_shaped().to_excel(xw, index=False, sheet_name="HHS")
    prof = pf.build(buf.getvalue(), "kobo.xlsx")

    def ck(label, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + label
              + (f"  {detail}" if detail and not cond else ""))
        if not cond:
            fails.append(label)

    geo = [c.name for c in prof.by_role(pf.GEO)]
    ck("question text containing 'facility' is not geography",
       not any(len(g) > 40 for g in geo), str(geo))
    ck("State and LGA are still geography",
       {"State", "LGA"} <= set(geo), str(geo))

    ms = metrics.measures(prof, limit=6)
    cols = [m.col for m in ms]
    ck("GPS columns are never measures",
       not any("Geo-Point" in c for c in cols), str(cols)[:90])
    ck("household size is a measure",
       any("number of people" in c for c in cols))
    ck("yes/no questions become % yes measures",
       any(m.agg == "share" for m in ms),
       str([(m.label[:20], m.agg) for m in ms]))
    ck("Male/Female does not become a % measure",
       not any(m.col == "Sex of Household Head" and m.agg == "share"
               for m in ms))
    share = next((m for m in ms if m.agg == "share"), None)
    if share:
        v = metrics.total(prof.df, share, prof)
        ck("a share reads as a percentage, not a fraction", 1 < v < 100,
           f"{v:.3f}")

    dims = [d.name for d in metrics.dimensions(prof)]
    ck("Kobo plumbing is not offered as a dimension",
       "__version__" not in dims and "deviceid" not in dims, str(dims)[:90])

    ck("a select-all family is folded even when its parent column exists",
       len(prof.kobo) == 1 and len(next(iter(prof.kobo.values()))) == 4,
       f"{len(prof.kobo)} families")

    titles = [q.title for q in charts.dashboard(prof, limit=8)]
    ck("chart titles are short enough to read",
       all(len(t) < 95 for t in titles),
       max(titles, key=len)[:110] if titles else "none")


def tracker_shaped() -> bytes:
    """A hand-maintained workbook: title rows, spacer column, a sheet per week.

    The shape of the campaign tracker that read as ONE column and 71 rows
    before header detection existed.
    """
    rng = np.random.default_rng(9)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        for wk in range(1, 5):
            n = 20
            body = pd.DataFrame({
                "SN": [1] + [None] * (n - 1),
                "Content": [None] * n,
                "Links to posts": [f"https://x.test/{wk}/{i}" for i in range(n)],
                "Author": rng.choice(["NHW", "Ugo", "Anto"], n),
                "Channel ": rng.choice(["Twitter", "Facebook"], n),
                "No of Views": rng.integers(50, 25000, n),
                "No of Retweets": rng.integers(0, 40, n),
                "No of Likes": rng.integers(0, 3000, n),
            })
            # Four junk rows above the header, as a person would leave them.
            head = pd.DataFrame([["Campaign Tracker"] + [None] * 7,
                                 [None] * 8, [None] * 8, [None] * 8],
                                columns=range(8))
            head.to_excel(xw, sheet_name=f"Week {wk}", index=False, header=False)
            body.to_excel(xw, sheet_name=f"Week {wk}", index=False, startrow=4)
        # A pre-aggregated summary that must NOT be stacked with the raw rows.
        pd.DataFrame({"Week": [f"Week {i}" for i in range(1, 5)],
                      "No of Views": [100, 200, 300, 400]}
                     ).to_excel(xw, sheet_name="Summary", index=False, startrow=4)
    return buf.getvalue()


def tracker_checks() -> None:
    from app import metrics

    def ck(label, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + label
              + (f"  {detail}" if detail and not cond else ""))
        if not cond:
            fails.append(label)

    data = tracker_shaped()
    one = pf.build(data, "tracker.xlsx", sheet="Week 1")
    ck("a header below title rows is found",
       "No of Views" in one.columns and one.rows == 20,
       f"{one.rows} rows, cols {list(one.columns)[:4]}")

    allp = pf.build(data, "tracker.xlsx", sheet=pf.ALL_SHEETS)
    ck("sheets sharing a layout are stacked", allp.rows == 80, f"{allp.rows} rows")
    ck("the sheet name becomes a column", "Sheet" in allp.columns)
    ck("a pre-aggregated summary sheet is left out",
       "Summary" not in allp.sheets_used and any(
           n == "Summary" for n, _ in allp.sheets_skipped),
       f"used {allp.sheets_used}")
    aggs = {m.col: m.agg for m in metrics.measures(allp, limit=6)}
    ck("engagement counts are totalled, not averaged",
       aggs.get("No of Views") == "sum" and aggs.get("No of Likes") == "sum",
       str(aggs))
    ck("the sheet column is offered as a dimension",
       "Sheet" in [d.name for d in metrics.dimensions(allp)])

    order = metrics.ordinal_order(allp.df["Sheet"])
    ck("Week 1..4 is recognised as an ordered run, not text",
       order == ["Week 1", "Week 2", "Week 3", "Week 4"], str(order))
    ck("a run with one stray label still counts",
       metrics.ordinal_order(pd.Series(
           ["Week 1", "Week 2", "Week 3", "Week 11", "Videos"]))
       == ["Week 1", "Week 2", "Week 3", "Week 11"])
    ck("unordered labels are not faked into a sequence",
       metrics.ordinal_order(pd.Series(["Kano", "Lagos", "Borno"])) == [])
    kinds = {q.kind for q in charts.dashboard(allp, limit=8)}
    ck("a file with no date column still gets a trend",
       "sequence_trend" in kinds, str(kinds))

    # A report workbook: many sheets, all different, plus an embedded Total row.
    buf = io.BytesIO()
    rng2 = np.random.default_rng(4)
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        pd.DataFrame({"Metric": ["Sent", "Opened"], "Value": [10, 4]}
                     ).to_excel(xw, sheet_name="Mail_Summary", index=False,
                                startrow=3)
        pd.DataFrame({"Metric": ["Old", "New"], "Value": [100, 20]}
                     ).to_excel(xw, sheet_name="Traffic", index=False, startrow=3)
        big = pd.DataFrame({"Page title": ["Total"] + [f"Article {i}" for i in range(60)],
                            "Views": [12345] + list(rng2.integers(5, 900, 60)),
                            "Active users": [9000] + list(rng2.integers(2, 400, 60))})
        big.to_excel(xw, sheet_name="GA4_Raw", index=False, startrow=3)
    rep = pf.build(buf.getvalue(), "report.xlsx", sheet=pf.ALL_SHEETS)
    ck("unrelated sheets are not stacked into a meaningless union",
       rep.rows > 50 and not rep.sheets_used,
       f"{rep.rows} rows, used {rep.sheets_used}")
    ck("it falls back to the sheet with the most data, and says why",
       "GA4_Raw" in rep.sheet_note, rep.sheet_note[:70])
    ck("an embedded Total row is removed",
       rep.total_rows_dropped == 1 and rep.rows == 60,
       f"{rep.rows} rows, {rep.total_rows_dropped} dropped")
    # The right check is that the subtotal's own value is gone, not that the
    # remaining sum is smaller — sixty random articles can legitimately add up
    # to more than one subtotal row.
    # A long list of named things with metrics — a GA4 export, an article list.
    buf2 = io.BytesIO()
    rng3 = np.random.default_rng(12)
    n = 300
    with pd.ExcelWriter(buf2, engine="xlsxwriter") as xw:
        pd.DataFrame({
            "Page title": [f"An article about health topic {i}" for i in range(n)],
            "Views": rng3.integers(1, 9000, n),
            "Active users": rng3.integers(1, 4000, n),
            "Combined_pageviews": rng3.integers(1, 500, n),
        }).to_excel(xw, sheet_name="GA4", index=False, startrow=3)
    ga4 = pf.build(buf2.getvalue(), "ga4.xlsx")
    ck("a long tail of labels still produces charts",
       any(q.kind == "top_items" for q in charts.dashboard(ga4, limit=6)),
       "300 titles and no dimension gave an empty dashboard")
    aggs2 = {m.col: m.agg for m in metrics.measures(ga4, limit=5)}
    ck("pageviews are totalled, not averaged because of the letters a-g-e",
       aggs2.get("Combined_pageviews") == "sum", str(aggs2))
    tiny = pf.build(pd.DataFrame({
        "Episode": ["Ep 1", "Ep 2", "Ep 3", "Ep 4"],
        "Views": [1954, 4650, 2100, 2629],
        "Subscribers_Gained": [8, 17, 9, 10]}).to_csv(index=False).encode(),
        "tiny.csv")
    ck("a four-row sheet still has measures",
       any(m.col == "Views" for m in metrics.measures(tiny)),
       "the floor of 5 filled values excluded everything")

    ck("the subtotal's value is no longer in the data",
       12345 not in set(rep.df["Views"].tolist())
       and 9000 not in set(rep.df["Active users"].tolist()),
       "the Total row's figures are still present")


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

    print("\n--- kobo-shaped survey (what the real CBPS file taught us) ---")
    kobo_checks()

    print("\n--- hand-maintained tracker (title rows, one sheet per week) ---")
    tracker_checks()

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
