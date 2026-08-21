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
import warnings
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

    # A pipeline workbook: raw, raw+flags (same records, two extra columns), a
    # long narrow log, and an empty cleaned sheet.
    buf3 = io.BytesIO()
    rng4 = np.random.default_rng(21)
    n2 = 200
    raw_df = pd.DataFrame({
        "_id": range(1, n2 + 1),
        "imei": rng4.choice([f"3566{i:011d}" for i in range(8)], n2),
        "phonenumber": rng4.choice([8031234567 + i for i in range(40)], n2),
        "State": rng4.choice(["Bauchi", "Kebbi"], n2),
        "Consent": rng4.choice(["Yes", "No"], n2, p=[0.99, 0.01]),
    })
    flagged = raw_df.copy()
    flagged["dq_issue"] = rng4.choice(["", "state mismatch"], n2)
    flagged["dq_note"] = ""
    with pd.ExcelWriter(buf3, engine="xlsxwriter") as xw:
        raw_df.to_excel(xw, sheet_name="Kobo_Raw_Data", index=False)
        flagged.to_excel(xw, sheet_name="Raw+DqIssues", index=False)
        pd.DataFrame({"ts": range(9000), "level": ["INFO"] * 9000,
                      "msg": ["step"] * 9000}
                     ).to_excel(xw, sheet_name="Pipeline_Log", index=False)
        raw_df.head(0).to_excel(xw, sheet_name="Cleaned Data", index=False)
    pipe = pf.build(buf3.getvalue(), "pipeline.xlsx", sheet=pf.ALL_SHEETS)
    ck("a long narrow log does not win over a wide data sheet",
       pipe.rows < 1000, f"{pipe.rows} rows — the 9,000-row log was chosen")
    ck("raw and raw+flags are not stacked into double the records",
       pipe.rows <= n2, f"{pipe.rows} rows for {n2} records")
    joined2, used2, skipped2 = pf.combine_sheets(buf3.getvalue(), "p.xlsx")
    same = [why for sh, why in skipped2 if "same records" in why]
    ck("the same-records overlap is named, not just skipped",
       bool(same) or joined2 is None, str(skipped2)[:80])
    one = pf.build(buf3.getvalue(), "p.xlsx", sheet="Raw+DqIssues")
    roles = {c.name: c.role for c in one.columns.values()}
    ck("a phone column is an identifier even when values repeat",
       roles.get("phonenumber") == "identifier", str(roles.get("phonenumber")))
    ck("an imei is not offered as a dimension",
       "imei" not in [d.name for d in metrics.dimensions(one)])
    ck("a survey with no numbers still charts its questions",
       any(q.kind == "categorical" for q in charts.dashboard(one, limit=6)),
       str({q.kind for q in charts.dashboard(one, limit=6)}))

    # A MetricsForNews / Parse.ly style CSV: the columns declare their own job.
    n3 = 40
    mfn_df = pd.DataFrame({
        'url': [f'https://x.test/a{i}/' for i in range(n3)],
        'unique_id': [f'/a{i}/' for i in range(n3)],
        'headline': [f'Headline number {i}' for i in range(n3)],
        'pubdate': pd.date_range('2026-01-01', periods=n3, freq='D'),
        'Focus Area': ['HSS' if i % 2 else 'MNCH' for i in range(n3)],
        'tag: Topic': [f'Topic {i % 3}' for i in range(n3)],
        'metric: Total pageviews': [100 + i * 7 for i in range(n3)],
        'metric: Total attention hours': [2.5 + i * 0.1 for i in range(n3)],
        'metric: Average reading minutes': [1.2 + i * 0.01 for i in range(n3)],
    })
    mfn = pf.build(mfn_df.to_csv(index=False).encode(), 'mfn.csv')
    ms3 = metrics.measures(mfn, limit=4)
    titles3 = [m.title() for m in ms3]
    ck('a metric: prefix is stripped from titles',
       not any('metric:' in t for t in titles3), str(titles3))
    ck('a title does not stutter Total total',
       not any('total total' in t.lower() or 'average average' in t.lower()
               for t in titles3), str(titles3))
    ck('a Total X column is summed even behind a prefix',
       {m.col: m.agg for m in ms3}.get('metric: Total attention hours') == 'sum',
       str({m.col: m.agg for m in ms3}))
    ck('headline beats a slug as the label column',
       metrics.label_column(mfn) == 'headline', str(metrics.label_column(mfn)))

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

    print("")
    print("--- files that were never datasets ---")
    # The corpus turned these up repeatedly: a document written into a grid. No
    # numbers, no repeated categories, so nothing to chart. The tool has to say
    # why rather than show a blank page, so the note is what is asserted here.
    prose = pd.DataFrame({
        "Section": [f"Section {i}: what this covers" for i in range(1, 11)],
        "Details": [f"A paragraph of prose unique to row {i}." for i in range(10)],
        "Data Sources": [f"Source list {i}, cited in full." for i in range(10)],
    })
    for label, as_book in [("a written outline in a grid is explained", False),
                           ("the same outline as a spreadsheet", True)]:
        if as_book:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
                prose.to_excel(xw, index=False)
            prof = pf.build(buf.getvalue(), "outline.xlsx")
        else:
            prof = pf.build(prose.to_csv(index=False).encode(), "outline.csv")
        note, drawn = prof.shape_note, len(charts.dashboard(prof, limit=6))
        ok = bool(note) and "free text" in note and drawn == 0
        print(f"  {'PASS' if ok else 'FAIL'}  {label}"
              f"  (charts={drawn}, note={'yes' if note else 'NONE'})")
        if not ok:
            fails.append(label)

    # ...and the same rule must stay silent on a real dataset, or every upload
    # gets told it is not a dataset.
    real = pd.DataFrame({"state": rng.choice(["Kano", "Lagos"], 60),
                         "cases": rng.integers(1, 40, 60)})
    quiet = pf.build(real.to_csv(index=False).encode(), "real.csv")
    ok = not quiet.shape_note
    print(f"  {'PASS' if ok else 'FAIL'}  a genuine dataset gets no shape note")
    if not ok:
        fails.append("shape note fired on a genuine dataset: "
                     + quiet.shape_note[:60])

    print("")
    print("--- one day of timestamps, and a mostly-empty export ---")
    # A GPS track and an hourly traffic export both cover a single day, so the
    # span in whole days is zero. That read as "no time axis" and the file lost
    # its trend chart entirely - the one chart it could support.
    intraday = pd.DataFrame({
        "timestamp": pd.Timestamp("2026-03-09 06:00") + pd.to_timedelta(
            rng.integers(0, 8 * 3600, 900), unit="s"),
        "speed_kmh": rng.normal(28, 9, 900).round(1),
        "accuracy_m": rng.integers(3, 40, 900),
    })
    prof = pf.build(intraday.to_csv(index=False).encode(), "track.csv")
    panels = charts.dashboard(prof, limit=6)
    trends = [q for q in panels if q.kind == "metric_trend" and q.chart is not None]
    per = [q.note for q in trends]
    ok = bool(trends) and any("per hour" in n or "per minute" in n for n in per)
    print(f"  {'PASS' if ok else 'FAIL'}  eight hours of pings still get a "
          f"trend  ({len(trends)} trend(s): {per[:1]})")
    if not ok:
        fails.append("no trend for a single day of timestamps")

    # Same data with an offset. to_period drops the zone, and once the axis
    # shows hours that decides whether 14:00 means Lagos or UTC.
    tz = intraday.copy()
    tz["timestamp"] = tz["timestamp"].dt.tz_localize("Africa/Lagos")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ptz = pf.build(tz.to_csv(index=False).encode(), "track_tz.csv")
        charts.dashboard(ptz, limit=6)
    noisy = [str(w.message) for w in caught if "timezone" in str(w.message)]
    ok = not noisy
    print(f"  {'PASS' if ok else 'FAIL'}  a tz-aware timestamp column is "
          f"handled explicitly, not warned about")
    if not ok:
        fails.append("timezone dropped with a warning: " + noisy[0][:60])

    # A contacts export carries the whole column list and almost no data. The
    # emptiness IS the finding, so the page has to say it.
    wide = {"ContactID": [f"C{i:05d}" for i in range(400)],
            "FirstName": [f"Name {i}" for i in range(400)]}
    for i in range(1, 12):
        wide[f"CustomValue{i}"] = [None] * 400
    prof = pf.build(pd.DataFrame(wide).to_csv(index=False).encode(), "book.csv")
    note = prof.shape_note
    ok = "empty" in note and "11 of 13" in note
    print(f"  {'PASS' if ok else 'FAIL'}  a mostly-empty export says how many "
          f"columns are blank  (note={note[:44]!r})")
    if not ok:
        fails.append("no note for a mostly-empty export")

    # The header row must survive a sparse table. Requiring the rows beneath it
    # to be as wide as the header itself demoted the header to data, so the
    # first record supplied the column names and was then lost.
    sparse = pd.DataFrame({"ContactID": [f"C{i:04d}" for i in range(50)],
                           "FirstName": [f"Name {i}" for i in range(50)]})
    for i in range(1, 12):
        sparse[f"Custom{i}"] = None
    prof = pf.build(sparse.to_csv(index=False).encode(), "sparse.csv")
    ok = prof.rows == 50 and "ContactID" in prof.columns
    print(f"  {'PASS' if ok else 'FAIL'}  a wide header over a sparse table is "
          f"kept  ({prof.rows} rows, first col "
          f"{list(prof.columns)[0] if prof.columns else 'none'!r})")
    if not ok:
        fails.append("header row lost on a sparse table")

    # Duplicate headings get a numeric suffix - which has to be checked against
    # the header too. Two "Notes" columns beside a real "Notes 2" produced two
    # columns both called "Notes 2", and every per-column check downstream then
    # raised on a DataFrame where it expected a Series.
    body = "Notes,Notes,Notes 2,n" + chr(10) + "a,b,c,1" + chr(10) + "d,e,f,2"
    prof = pf.build(body.encode(), "clash.csv")
    labels = list(prof.df.columns)
    ok = len(set(labels)) == len(labels) == 4 and prof.rows == 2
    print(f"  {'PASS' if ok else 'FAIL'}  a repeated heading beside its own "
          f"suffix stays unique  ({labels})")
    if not ok:
        fails.append("duplicate column labels collided")

    # A rating grid: the response options 1..5 are the columns, and the real
    # heading sits above them in a merged cell. The header search picks the
    # first data row because it has more filled cells, so the column names come
    # out as bare numbers. The tool cannot melt that shape - it has to say so.
    grid = [",Sub-domain,Statement,Rating format,,,,",
            ",Stakeholder involvement,P1. The national plan,1,2,3,4,5",
            ",Costing,P2. The plan is costed,1,2,3,4,5",
            ",Alignment,P3. Partners align to it,1,2,3,4,5",
            ",Review,P4. It is reviewed yearly,1,2,3,4,5"]
    prof = pf.build(chr(10).join(grid).encode(), "grid.csv")
    labels = [str(c) for c in prof.columns]
    ok = (prof.rows == 4 and "Sub-domain" in labels
          and bool(prof.shape_note)
          and len(charts.dashboard(prof, limit=6)) == 0)
    print(f"  {'PASS' if ok else 'FAIL'}  a rating grid keeps its header row "
          f"and is explained  ({prof.rows} rows, {labels[:2]})")
    if not ok:
        fails.append("rating grid misread or unexplained")

    # With no header row at all, the first record supplies the names. Nothing
    # can fix that, so the tool has to name what it sees.
    head = [",".join(["Adamawa", "NE", "GF", "21", "9.5", "11.5", "7", "8"]),
            ",".join(["Bauchi", "NE", "GF", "18", "8.5", "10.0", "6", "9"]),
            ",".join(["Benue", "NC", "GF", "25", "10.5", "12.5", "8", "7"]),
            ",".join(["Kogi", "NC", "DP", "19", "7.5", "9.5", "5", "6"]),
            ",".join(["Kwara", "NC", "DP", "22", "9.0", "11.0", "7", "8"])]
    prof = pf.build(chr(10).join(head).encode(), "headerless.csv")
    ok = "bare numbers" in prof.shape_note
    print(f"  {'PASS' if ok else 'FAIL'}  a file with no header row is named as "
          f"such  (note={prof.shape_note[:34]!r})")
    if not ok:
        fails.append("headerless file not recognised")

    # ...but a year-per-column table is a legitimate wide shape that charts.
    years = pd.DataFrame({"state": ["Kano", "Lagos", "Oyo"] * 10})
    for y in range(2019, 2026):
        years[str(y)] = rng.integers(1, 500, 30)
    prof = pf.build(years.to_csv(index=False).encode(), "years.csv")
    ok = not prof.shape_note and len(charts.dashboard(prof, limit=4)) > 0
    print(f"  {'PASS' if ok else 'FAIL'}  a year-per-column table is left alone"
          f"  (note={prof.shape_note[:30]!r})")
    if not ok:
        fails.append("bare-number check fired on year columns")

    # A two-row header: "Pre-Test" merged across three columns, then the three
    # sub-headings beneath. Reading only the upper row left three columns
    # called Unnamed; reading only the lower one gave "#Correct" twice with no
    # way to tell which test it belonged to.
    two = ["S/N,Name,Pre-Test,,,Post-Test,,",
           ",,#Correct,#Incorrect,Score,#Correct,#Incorrect,Score",
           "1,Binta K,12,14,0.46,23,3,0.88",
           "2,Hassana A,10,16,0.38,21,5,0.81",
           "3,Halima M,9,17,0.35,21,5,0.81",
           "4,Fatima B,11,15,0.42,22,4,0.85",
           "5,Aisha S,13,13,0.50,24,2,0.92"]
    prof = pf.build(chr(10).join(two).encode(), "twohead.csv")
    labels = [str(c) for c in prof.columns]
    ok = (prof.rows == 5 and "Pre-Test #Correct" in labels
          and "Post-Test Score" in labels and "Name" in labels)
    print(f"  {'PASS' if ok else 'FAIL'}  a merged two-row header is folded "
          f"into one  ({prof.rows} rows, {labels[2:4]})")
    if not ok:
        fails.append("two-row header not merged")

    # A wide table whose row-label column has no heading. Dropping it as
    # "Unnamed" left years as columns and nothing to name them against.
    wide = [",2015,2016,2017,2018",
            "Total population,186083442,192038112,198183332,204525199",
            "Total funding,41200000,43800000,44100000,46900000",
            "Gap,144883442,148238112,154083332,157625199"]
    prof = pf.build(chr(10).join(wide).encode(), "wide.csv")
    ok = prof.rows == 3 and len(prof.columns) == 5
    print(f"  {'PASS' if ok else 'FAIL'}  a blank-headed label column is kept"
          f"  ({prof.rows} rows x {len(prof.columns)} cols)")
    if not ok:
        fails.append("row-label column dropped from a wide table")

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
