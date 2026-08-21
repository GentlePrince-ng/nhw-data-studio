"""End-to-end check against samples/sample_phc_survey.xlsx.

Not a unit-test suite: it runs the pipeline the way the app does — read, type,
analyse, chart, export — and asserts that the defects planted in the sample are
the ones reported. If a rule silently stops firing, this is what catches it.

    python tests/test_pipeline.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import charts, geo, insights, profiling as pf, report  # noqa: E402

SAMPLE = ROOT / "samples" / "sample_phc_survey.xlsx"
fails = []


def check(label: str, cond: bool, detail: str = "") -> None:
    # Detail only on failure: printed on a pass it reads as if the check failed
    # ("PASS phone column is not a measurement  got identifier").
    print(("  PASS  " + label) if cond
          else ("  FAIL  " + label + (f"  {detail}" if detail else "")))
    if not cond:
        fails.append(label)


def main() -> int:
    if not SAMPLE.exists():
        print("Run samples/make_sample.py first.")
        return 1
    data = SAMPLE.read_bytes()

    t0 = time.perf_counter()
    prof = pf.build(data, SAMPLE.name, "survey_data")
    t_read = time.perf_counter() - t0

    print(f"\nread + typed {prof.rows:,} rows x {len(prof.columns)} cols "
          f"in {t_read:.2f}s")
    print("--- typing ---")
    roles = {n: c.role for n, c in prof.columns.items()}
    for want, cols in {
            pf.DATETIME: ["submission_date"],
            pf.GEO: ["state", "lga"],
            pf.NUMERIC: ["respondent_age", "wait_minutes", "staff_on_duty",
                         "visits_last_year"],
            pf.CATEGORICAL: ["facility_type", "enumerator_notes"],
            pf.BINARY: ["respondent_sex", "satisfied_with_visit"],
            pf.IDENTIFIER: ["_id"],
            pf.CONSTANT: ["form_version"],
            pf.EMPTY: ["audit_url"]}.items():
        for col in cols:
            check(f"{col} reads as {want}", roles.get(col) == want,
                  f"got {roles.get(col)}")
    check("best date column found", prof.best_date() == "submission_date")
    check("best geo column is state", prof.best_geo() == "state")
    check("sentinel 99 caught in age",
          99 in prof.columns["respondent_age"].sentinel_hits)
    check("age mean excludes the 99s",
          prof.columns["respondent_age"].stats["mean"] < 40,
          f"mean {prof.columns['respondent_age'].stats['mean']:.1f}")
    # Regressions, each one a bug this suite caught once:
    check("phone column is not a measurement",
          roles.get("callback_number") == pf.IDENTIFIER,
          f"got {roles.get('callback_number')} — a mean of phone numbers")
    check("continuous readings are not identifiers",
          prof.columns["wait_minutes"].role == pf.NUMERIC)
    check("kobo option columns stay off the front page",
          not any("/" in c.name for c in prof.headline_cats(limit=8)))
    check("kobo select_multiple folded",
          "barriers_to_care" in prof.kobo
          and len(prof.kobo["barriers_to_care"]) == 6,
          str(list(prof.kobo)))

    t0 = time.perf_counter()
    found = insights.analyse(prof)
    t_ins = time.perf_counter() - t0
    print(f"\n--- {len(found)} findings in {t_ins:.2f}s ---")
    for f in found[:14]:
        print(f"  [{f.severity:5}] {f.kind:14} {f.text[:120]}")

    blob = " ".join(f.text.lower() for f in found)
    check("duplicate rows reported", "duplicate" in blob)
    check("case/spacing split reported", "case or spacing" in blob)
    check("sentinel coding reported", "refused" in blob or "don't-know" in blob)
    check("sparse column reported", "empty" in blob)
    check("collection gap reported", "not continuous" in blob)
    check("state coverage reported", "states" in blob)
    check("wait/facility difference found",
          "wait minutes varies by facility type" in blob
          or "facility type" in blob)
    check("no rule crashed",
          not any("could not run" in f.text for f in found),
          [f.text for f in found if "could not run" in f.text])
    check("headline paragraph builds", len(insights.headline(prof, found)) > 80)

    t0 = time.perf_counter()
    panels = charts.auto_panels(prof)
    t_ch = time.perf_counter() - t0
    print(f"\n--- {len(panels)} panels in {t_ch:.2f}s ---")
    for p in panels:
        print(f"  {p.title[:34]:34} {p.kind:11} "
              f"{'chart' if p.chart is not None else '     '} "
              f"{len(p.table) if p.table is not None else 0} rows")
    check("panels built", len(panels) >= 6)
    check("every panel has a chart or a table",
          all(p.chart is not None or p.table is not None for p in panels))
    # Every panel table must be an aggregate, not a row dump. The threshold is
    # relative rather than a flat 200: a 265-row table of LGAs with data is a
    # legitimate summary of 10,180 records, and a flat cap called it a failure.
    dumps = [(q.title, len(q.table)) for q in panels
             if q.table is not None and len(q.table) > max(50, 0.1 * prof.rows)]
    check("no panel ships raw rows to the browser", not dumps, f"{dumps}")
    # The spec is what actually reaches the browser, so that is what gets
    # measured. A chart that inlines the whole column is slow on screen and
    # silently unexportable.
    heavy = [(q.title, n) for q in panels
             if q.chart is not None
             and (n := q.chart.to_json().count('{"')) > 6000]
    check("no chart spec carries a whole column", not heavy, f"{heavy}")

    # The boundary files are not in the repository, so the map assertions only
    # apply where they have been built. A fresh clone must pass this suite.
    maps = [q for q in panels if q.kind == "map"]
    if not geo.available("state"):
        print("  SKIP  map checks — no boundary files built "
              "(static/build_boundaries.py)")
        check("no map panels without boundary files", not maps)
    else:
        mp = maps[0] if maps else None
        check("state map is on the front page", mp is not None)
        if mp:
            check("map matched the state column", "12 of 37" in mp.note, mp.note)
            check("map reports no-data places", "no records, not zero" in mp.note)
            check("map reports combined spellings", "spellings combined" in mp.note)
            check("map renders to SVG", report.chart_svg(mp).count("<path") > 30)
        check("LGA map drawn where a bar cannot cope", len(maps) == 2,
              f"{len(maps)} maps")

    # ── the dashboard: measures, not columns ────────────────────────────
    from app import metrics
    ms = metrics.measures(prof)
    dims = metrics.dimensions(prof)
    print()
    print("--- measures and dimensions ---")
    for m in ms:
        print(f"  {m.title():34} {m.why}")
    print("  dimensions: " + ", ".join(d.label for d in dims))

    check("Records is the first measure", ms and ms[0].col == "__records__")
    aggs = {m.col: m.agg for m in ms}
    check("an age column is averaged, never summed",
          aggs.get("respondent_age") == "mean", str(aggs.get("respondent_age")))
    check("a visits column is totalled",
          aggs.get("visits_last_year") == "sum", str(aggs.get("visits_last_year")))
    check("wait minutes is averaged", aggs.get("wait_minutes") == "mean")
    check("staff on duty is averaged, not summed across visits",
          aggs.get("staff_on_duty") == "mean", str(aggs.get("staff_on_duty")))
    check("an aggregation override wins over the guess",
          {m.col: m.agg for m in
           metrics.measures(prof, overrides={"respondent_age": "sum"})
           }.get("respondent_age") == "sum")

    dim_names = [d.name for d in dims]
    check("free text is not offered as a dimension",
          "enumerator_notes" not in dim_names, str(dim_names))
    check("the state column is a dimension", "state" in dim_names)

    cmp = metrics.compare_periods(prof.df, ms[0], prof.best_date(), prof)
    check("period-over-period comparison computed", bool(cmp), str(cmp)[:80])
    if cmp:
        check("comparison windows are equal length and non-empty",
              cmp["now"] > 0 and cmp["then"] > 0 and cmp["window"] > 0)

    dash = charts.dashboard(prof, limit=10)
    print()
    print(f"--- dashboard: {len(dash)} panels ---")
    for q in dash:
        print(f"  {q.kind:14} {q.title[:60]}")
    kinds = {q.kind for q in dash}
    check("dashboard leads with measure charts, not distributions",
          {"metric_trend", "measure_by"} <= kinds, str(kinds))
    check("dashboard answers what moved", "change_by" in kinds, str(kinds))
    check("no distribution charts on the dashboard",
          "numeric" not in kinds, str(kinds))
    check("every dashboard panel exports",
          not [q.title for q in dash
               if q.chart is not None and not report.chart_svg(q)])
    check("every dashboard panel states its aggregation",
          all(q.note for q in dash))

    # Kano arrives as Kano / KANO / "kano " in the sample.
    folded = metrics.by_dimension(prof.df, ms[0], "state", prof)
    cats = list(folded["Category"])
    check("case-variant labels are folded in breakdowns",
          len(cats) == len(set(c.lower() for c in cats))
          and folded.attrs.get("merged", 0) == 2,
          f"{cats[:4]} merged={folded.attrs.get('merged')}")
    check("folding adds the split counts together",
          int(folded.loc[folded["Category"].str.lower() == "kano", "Total"].iloc[0])
          > 2000, "Kano should be ~2,261, not ~1,921")
    check("a breakdown with no variation is left off the dashboard",
          not [q for q in dash
               if q.kind == "measure_by" and "respondent age by" in q.title.lower()],
          "average age by state was 34 everywhere")

    extra = [charts.corr_grid(prof), charts.missing_bar(prof),
             charts.cross_heat(prof, "facility_type", "satisfied_with_visit"),
             charts.scatter(prof, "wait_minutes", "staff_on_duty"),
             charts.group_bar(prof, "wait_minutes", "facility_type"),
             charts.cat_trend(prof, "submission_date", "state"),
             charts.multi_bar(prof, "barriers_to_care")]
    check("secondary charts build", all(p is not None for p in extra))

    t0 = time.perf_counter()
    html = report.html_report(prof, found, panels + [p for p in extra if p],
                             title="Sample PHC survey")
    t_html = time.perf_counter() - t0
    check("HTML report built", len(html) > 40_000, f"{len(html) / 1024:.0f} KB")
    # Not ">= 8": EVERY panel that has a chart must export. A loose floor let
    # the histograms go missing from every report for a day — they tripped
    # Altair's row ceiling and the exporter swallowed it.
    charted = [q for q in panels + [p for p in extra if p] if q.chart is not None]
    failed = [q.title for q in charted if not report.chart_svg(q)]
    check("every chart exports to SVG", not failed, f"failed: {failed}")
    check("one SVG per charted panel",
          html.count(b"<svg") == len(charted),
          f"{html.count(b'<svg')} svg for {len(charted)} charts")
    # An xmlns="http://www.w3.org/2000/svg" is a namespace, not a request; what
    # must not appear is anything the browser would go and fetch.
    check("no external requests in the report",
          not any(t in html for t in (b'src="http', b'href="http', b"@import",
                                      b"<script", b"url(http")))
    out = ROOT / "samples" / "sample_report.html"
    out.write_bytes(html)

    xl = report.excel_book(prof, found, panels + [p for p in extra if p])
    (ROOT / "samples" / "sample_tables.xlsx").write_bytes(xl)
    check("Excel workbook built", len(xl) > 10_000, f"{len(xl) / 1024:.0f} KB")

    total = t_read + t_ins + t_ch + t_html
    print(f"\ntotal read->report {total:.2f}s for {prof.rows:,} rows "
          f"(html {t_html:.2f}s)")
    check("under 30s end to end", total < 30, f"{total:.1f}s")
    print(f"\nwrote {out.name} and sample_tables.xlsx to samples/")
    print("FAILURES: " + (", ".join(fails) if fails else "none"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
