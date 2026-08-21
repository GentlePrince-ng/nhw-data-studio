"""NHW Data Studio — upload a spreadsheet, get a read on it.

Run: streamlit run app/studio_app.py

The design brief was five minutes from file to something you can send someone,
which sets the order of everything here: the findings come before the charts,
because the first question is what to look at; the charts are chosen rather than
exhaustive; and the download is on the same screen as the analysis rather than
behind a menu.

Nothing is written anywhere. The upload lives in this browser session and goes
when the tab closes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import charts, gate, geo, insights, narrative, profiling as pf, report, theme  # noqa: E402

ROW_CAP = 300_000        # far above the 10k target; guards a rogue 5m-row export
PAGES = ["Overview", "Findings", "Data health", "Explore", "Written brief"]

st.set_page_config(page_title="NHW Data Studio", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")
theme.chrome()


# ═══ cached compute ═════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, max_entries=4)
def _sheets(key: str, data: bytes, name: str):
    return pf.sheet_names(data, name)


@st.cache_data(show_spinner="Reading and typing the columns…", max_entries=4)
def _profile(key: str, data: bytes, name: str, sheet: str | None):
    return pf.build(data, name, sheet, row_cap=ROW_CAP)


@st.cache_data(show_spinner="Looking for what stands out…", max_entries=8,
               hash_funcs={pf.Profile: lambda p: (p.source, p.sheet, p.rows,
                                                  tuple(p.columns))})
def _findings(prof, sig: str):
    return insights.analyse(prof)


@st.cache_data(show_spinner="Applying filters…", max_entries=6)
def _subset(key: str, data: bytes, name: str, sheet: str | None, sig: str,
            mask_index) -> pf.Profile:
    """Re-type the filtered rows so every statistic reflects the filter.

    Filtering the frame but keeping the parent's column statistics would print a
    national mean over a one-state selection, which is the kind of error that
    survives review because the number looks plausible.
    """
    base = _profile(key, data, name, sheet)
    sub = base.df.loc[list(mask_index)]
    df, cols = pf.classify(sub.reset_index(drop=True))
    out = pf.Profile(df=df, columns=cols, source=base.source, sheet=base.sheet,
                     truncated=base.truncated)
    out.kobo = pf.kobo_groups(cols)
    out.dropped = [c.name for c in cols.values()
                   if c.role in (pf.EMPTY, pf.CONSTANT)]
    return out


# ═══ sidebar: file, sheet, filters ══════════════════════════════════════════
def sidebar_source():
    st.sidebar.markdown("#### Your data")
    up = st.sidebar.file_uploader(
        "Excel or CSV", type=["xlsx", "xlsm", "csv", "tsv", "txt"],
        label_visibility="collapsed",
        help="Nothing is uploaded to a server or stored. The file stays in "
             "this browser session.")
    if up is None:
        return None
    data = up.getvalue()
    key = pf.file_key(up.name, data)
    sheet = None
    names = _sheets(key, data, up.name)
    if len(names) > 1:
        sheet = st.sidebar.selectbox("Sheet", names, help=f"{len(names)} sheets "
                                    "in this workbook")
    elif names:
        sheet = names[0]
    return key, data, up.name, sheet


def sidebar_filters(prof: pf.Profile):
    """Date range plus up to two category filters, applied everywhere.

    Returns (signature, mask). An empty signature means nothing was narrowed and
    the caller should use the unfiltered profile as-is.
    """
    st.sidebar.markdown("#### Filter")
    mask = pd.Series(True, index=prof.df.index)
    bits = []

    date = prof.best_date()
    if date:
        s = pd.to_datetime(prof.df[date], errors="coerce")
        lo, hi = s.min(), s.max()
        if pd.notna(lo) and pd.notna(hi) and lo.date() < hi.date():
            picked = st.sidebar.date_input(
                prof.label(date), value=(lo.date(), hi.date()),
                min_value=lo.date(), max_value=hi.date())
            if isinstance(picked, tuple) and len(picked) == 2:
                a, b = pd.Timestamp(picked[0]), pd.Timestamp(picked[1]) + \
                    pd.Timedelta(days=1)
                if a > lo or b < hi:
                    mask &= s.between(a, b)
                    bits.append(f"{prof.label(date)} {picked[0]}..{picked[1]}")

    cands = [c for c in prof.headline_cats(limit=8) if 1 < c.unique <= 40]
    for c in cands[:2]:
        opts = [str(v) for v in prof.df[c.name].value_counts().index[:40]]
        chosen = st.sidebar.multiselect(c.label, opts,
                                        help=f"{c.unique} values in this column")
        if chosen:
            mask &= prof.df[c.name].astype(str).isin(chosen)
            bits.append(f"{c.label}={'|'.join(chosen)}")

    if not bits:
        return "", mask
    kept = int(mask.sum())
    st.sidebar.caption(f"**{kept:,}** of {prof.rows:,} records "
                       f"({100 * kept / max(prof.rows, 1):.0f}%)")
    if kept == 0:
        st.sidebar.error("No records match — showing everything instead.")
        return "", pd.Series(True, index=prof.df.index)
    return "; ".join(bits), mask


# ═══ shared chrome ══════════════════════════════════════════════════════════
def kpi_row(prof: pf.Profile, found: list):
    cols = st.columns(4)
    theme.kpi_card(cols[0], "Records", f"{prof.rows:,}",
                   note=f"{len(prof.columns)} columns")
    date = prof.best_date()
    if date:
        stt = prof.columns[date].stats
        theme.kpi_card(cols[1], "Covers",
                       f"{stt.get('span_days', 0):,} days",
                       note=f"{stt['start']:%b %Y} – {stt['end']:%b %Y}")
    else:
        analysable = len([c for c in prof.columns.values()
                          if c.role in pf.ANALYSABLE])
        theme.kpi_card(cols[1], "Analysable columns", f"{analysable}",
                       note="no date column found")
    complete = 100 * (1 - sum(c.missing for c in prof.columns.values())
                      / max(prof.rows * len(prof.columns), 1))
    theme.kpi_card(cols[2], "Cells filled", f"{complete:.0f}%",
                   note=f"{sum(1 for c in prof.columns.values() if c.missing)} "
                        "columns have blanks")
    flags = sum(1 for f in found if f.severity == insights.FLAG)
    theme.kpi_card(cols[3], "Needs attention", f"{flags}",
                   note="issues to settle before quoting figures",
                   colour=theme.BEHIND if flags else theme.GREEN_DEEP)


def kpi_export(prof: pf.Profile, found: list):
    """The same four figures, for the HTML report."""
    date = prof.best_date()
    span = (f"{prof.columns[date].stats.get('span_days', 0):,} days"
            if date else "—")
    complete = 100 * (1 - sum(c.missing for c in prof.columns.values())
                      / max(prof.rows * len(prof.columns), 1))
    flags = sum(1 for f in found if f.severity == insights.FLAG)
    return [("Records", f"{prof.rows:,}", f"{len(prof.columns)} columns"),
            ("Covers", span, "date range in the data"),
            ("Cells filled", f"{complete:.0f}%", "across every column"),
            ("Needs attention", f"{flags}", "flagged issues")]


def maps_note(prof: pf.Profile):
    """Say why there is no map, when there is a place column but no boundaries.

    The boundary files are not in the repository, so a fresh clone has a place
    column and no map and no explanation. Silence there reads as a bug in the
    tool rather than a missing build step.
    """
    if geo.available("state") or not prof.by_role(pf.GEO):
        return
    st.caption(
        "There is a place column here but no map: the Nigeria boundary files "
        "are not in this repository. Build them once with "
        "`pip install geopandas shapely` then "
        "`python static/build_boundaries.py path/to/lga_boundaries.geojson` — "
        "see that file for where to get a source layer. Everything else on "
        "this page works without them.")


def render_grid(panels: list):
    """Two panels per row, except maps, which get a row to themselves.

    A map carries an explicit pixel width (its projection needs one), so it
    cannot shrink to fit a half-width column the way a bar chart does — dropped
    into the grid it pushed the whole page into horizontal scrolling on any
    screen under about 1,600px. It is also the panel that most rewards the extra
    width.
    """
    pending = []

    def flush():
        nonlocal pending
        if not pending:
            return
        cols = st.columns(2)
        for col, panel in zip(cols, pending):
            with col:
                panel.render(st)
        pending = []

    for p in panels:
        if p is None:
            continue
        if p.kind == "map":
            flush()
            p.render(st)
        else:
            pending.append(p)
            if len(pending) == 2:
                flush()
    flush()


def finding_list(found: list, limit: int | None = None, tables: bool = True):
    shown = found[:limit] if limit else found
    for f in shown:
        kind = {insights.FLAG: "bad", insights.WATCH: "warn"}.get(f.severity, "")
        theme.insight(f"**{f.kind}.** {f.text}", kind)
        if tables and f.table is not None and len(f.table):
            with st.expander(f"Figures behind this — {f.kind.lower()}", False):
                st.dataframe(f.table, width="stretch", hide_index=True)


def download_row(prof, found, panels, label: str, key: str):
    """Prepare-then-download: building the SVGs takes a second or two, and doing
    it on every rerun would tax every page view for a file most users never take."""
    theme.section("Take it with you",
                  "self-contained HTML · opens offline · prints to PDF")
    c1, c2, c3 = st.columns([1.1, 1.1, 2])
    slug = Path(prof.source).stem[:40].replace(" ", "_")
    if c1.button("Build HTML report", key=f"bh{key}", width="stretch"):
        st.session_state[f"html{key}"] = report.html_report(
            prof, found, panels, title=label, kpis=kpi_export(prof, found))
    if f"html{key}" in st.session_state:
        c1.download_button("⬇ HTML report", st.session_state[f"html{key}"],
                           file_name=f"{slug}_{key}.html", mime="text/html",
                           key=f"dh{key}", width="stretch")
    if c2.button("Build Excel workbook", key=f"bx{key}", width="stretch"):
        st.session_state[f"xl{key}"] = report.excel_book(prof, found, panels)
    if f"xl{key}" in st.session_state:
        c2.download_button(
            "⬇ Excel workbook", st.session_state[f"xl{key}"],
            file_name=f"{slug}_{key}.xlsx", key=f"dx{key}", width="stretch",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    note = ("Charts are embedded as images — no internet needed to open it."
            if report.SVG_OK else
            "Chart images unavailable here (vl-convert not installed): the "
            "report will carry the figures as tables.")
    c3.caption(f"{note} The workbook holds every table behind every chart, plus "
               "the column map and a provenance sheet.")


# ═══ pages ══════════════════════════════════════════════════════════════════
def page_overview(prof, found, filt):
    theme.band("Overview", f"{prof.source}" + (f" · {prof.sheet}" if prof.sheet else "")
               + (f" · filtered: {filt}" if filt else ""))
    kpi_row(prof, found)
    theme.section("The short version", "computed from your file, not written")
    theme.insight(insights.headline(prof, found))
    theme.section("What stands out", f"top 6 of {len(found)} findings — the rest "
                                     "are on the Findings page")
    finding_list(found, limit=6, tables=False)

    panels = charts.auto_panels(prof, limit=8)
    theme.section("Charts chosen for this data",
                  f"{len(panels)} of the columns are worth a picture")
    render_grid(panels)
    maps_note(prof)
    download_row(prof, found, panels, "Overview", "overview")


def page_findings(prof, found, filt):
    theme.band("Findings", f"{len(found)} computed observations, most "
                           "consequential first")
    kinds = sorted({f.kind for f in found})
    pick = st.multiselect("Show", kinds, default=kinds,
                          label_visibility="collapsed")
    sev = st.radio("Severity", ["Everything", "Needs attention only"],
                   horizontal=True, label_visibility="collapsed")
    shown = [f for f in found if f.kind in pick
             and (sev == "Everything" or f.severity != insights.INFO)]
    st.caption(f"{len(shown)} of {len(found)} findings"
               + (f" · filtered: {filt}" if filt else ""))
    finding_list(shown)
    download_row(prof, shown, [], "Findings", "findings")


def page_health(prof, found, filt):
    theme.band("Data health", "what to fix, and what to say about it in the "
                              "methods note")
    panels = []
    mb = charts.missing_bar(prof)
    if mb:
        panels.append(mb)
    quality = [f for f in found if f.kind in (insights.QUALITY, insights.COVERAGE)]
    theme.section("Issues found", f"{len(quality)} checks fired")
    if quality:
        finding_list(quality)
    else:
        theme.insight("No completeness, duplication, coding or coverage problem "
                      "cleared the reporting threshold. That is not a "
                      "certificate — it means these particular checks found "
                      "nothing.")
    for p in panels:
        p.render(st)

    tt = charts.type_table(prof)
    theme.section(tt.title, tt.note)
    st.dataframe(tt.table, width="stretch", hide_index=True, height=380)
    with st.expander("Duplicate rows, in full", False):
        dup = prof.df[prof.df.duplicated(keep=False)]
        if len(dup):
            st.caption(f"{len(dup):,} rows are part of a duplicate group.")
            st.dataframe(dup.head(500), width="stretch")
        else:
            st.caption("No exact duplicate rows.")

    theme.section("Cleaned copy", "typed columns, blanks normalised, "
                                  "nothing dropped")
    st.download_button(
        "⬇ Download as CSV", prof.df.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(prof.source).stem[:40]}_typed.csv", mime="text/csv")
    download_row(prof, quality, panels + [tt], "Data health", "health")


def page_explore(prof, found, filt):
    theme.band("Explore", "pick the columns; the chart type follows the data")
    usable = [c.name for c in prof.by_role(*pf.ANALYSABLE)]
    if not usable:
        st.warning("No analysable columns in this sheet.")
        return
    labels = {c: f"{prof.label(c)}  ({prof.columns[c].role})" for c in usable}

    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("Look at", usable, format_func=labels.get)
    others = ["(nothing)"] + [c for c in usable if c != x]
    y = c2.selectbox("Broken down by", others,
                     format_func=lambda c: labels.get(c, c))
    agg = c3.selectbox("Summarise with", ["count", "mean", "median", "sum"])

    rx = prof.columns[x].role
    ry = prof.columns[y].role if y != "(nothing)" else None
    panels = []

    # A place column earns a map as well as a bar: the map answers where, the
    # bar answers how many, and a reader of a coverage question needs both.
    def _map(place, value=None):
        return charts.state_map(prof, place, value,
                                "mean" if (value and agg == "count") else agg,
                                level=charts.geo_level(place))

    if ry is None:
        if rx == pf.NUMERIC:
            panels.append(charts.num_hist(prof, x))
        elif rx == pf.DATETIME:
            panels.append(charts.time_trend(prof, x))
        else:
            if rx == pf.GEO:
                panels.append(_map(x))
            panels.append(charts.cat_bar(prof, x, n=25))
    elif rx == pf.NUMERIC and ry == pf.NUMERIC:
        panels.append(charts.scatter(prof, x, y))
    elif rx == pf.NUMERIC and ry in (pf.CATEGORICAL, pf.GEO, pf.BINARY):
        if ry == pf.GEO:
            panels.append(_map(y, x))
        panels.append(charts.group_bar(prof, x, y,
                                       "mean" if agg == "count" else agg))
    elif ry == pf.NUMERIC and rx in (pf.CATEGORICAL, pf.GEO, pf.BINARY):
        if rx == pf.GEO:
            panels.append(_map(x, y))
        panels.append(charts.group_bar(prof, y, x,
                                       "mean" if agg == "count" else agg))
    elif rx == pf.DATETIME and ry in (pf.CATEGORICAL, pf.GEO, pf.BINARY):
        panels.append(charts.cat_trend(prof, x, y))
    elif ry == pf.DATETIME and rx in (pf.CATEGORICAL, pf.GEO, pf.BINARY):
        panels.append(charts.cat_trend(prof, y, x))
    elif rx == pf.DATETIME and ry == pf.NUMERIC:
        panels.append(charts.time_trend(prof, x))
        panels.append(charts.num_hist(prof, y))
    else:
        panels.append(charts.cross_heat(prof, x, y))

    for p in [p for p in panels if p]:
        p.render(st, show_table=True)

    maps_note(prof)
    cg = charts.corr_grid(prof)
    if cg:
        theme.section(cg.title, cg.note)
        theme.chart(cg.chart)
        panels.append(cg)
    download_row(prof, [], [p for p in panels if p], "Explore", "explore")


def page_brief(prof, found, filt):
    theme.band("Written brief", "an optional draft, written from the findings "
                                "on the other pages")
    # Streamlit does not export secrets.toml into the environment, and the SDK
    # only reads the environment — so a key sitting in secrets.toml looks to the
    # app like no key at all until it is copied across here.
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            # .strip(): a key pasted with a trailing newline or a stray space
            # goes into the auth header verbatim and comes back 401, which looks
            # exactly like a revoked key.
            os.environ.setdefault(
                "ANTHROPIC_API_KEY", str(st.secrets["ANTHROPIC_API_KEY"]).strip())
    except Exception:
        pass
    ok, why = narrative.available()
    st.caption("Nigeria Health Watch · draft for editing, not for sending as is.")
    theme.insight(
        "**What would be sent.** The column map, the finding sentences, and the "
        "small summary tables — the aggregates already on your screen. **No row "
        "of your data leaves this machine.** Figures are computed here and the "
        "model is instructed to use no number it was not given; check them "
        "against the Findings page before the brief goes anywhere.")
    if not ok:
        st.warning(f"{why} Everything else in the app works without it — the "
                   "Findings page is the same analysis in list form.")
        return
    good, problem = _key_ok(os.environ.get("ANTHROPIC_API_KEY", "")[-8:])
    if not good:
        st.error(problem)
        st.caption("Everything else in the app works without a key — the "
                   "Findings page is this same analysis in list form.")
        return

    effort = st.radio("Depth", ["medium", "high"], horizontal=True,
                      help="Higher takes longer and costs more.")
    payload = narrative.build_payload(
        prof, found, [(f.kind, f.table) for f in found[:6] if f.table is not None])
    sig = narrative.digest(payload)
    if st.button("Write the brief", type="primary"):
        with st.spinner("Claude is drafting from your findings…"):
            text, note = narrative.write_brief(payload, effort)
        st.session_state[f"brief{sig}"] = (text, note)
    got = st.session_state.get(f"brief{sig}")
    if got:
        text, note = got
        if not text:
            st.error(note)
            return
        st.markdown(text)
        st.caption(note)
        st.download_button("⬇ Download as markdown", text.encode("utf-8"),
                           file_name=f"{Path(prof.source).stem[:40]}_brief.md",
                           mime="text/markdown")
    with st.expander("Exactly what gets sent", False):
        st.code(payload[:20000], language="markdown")


@st.cache_data(show_spinner="Checking the API key…", ttl=900)
def _key_ok(key_tail: str):
    """Cached per key, so replacing the secret re-checks and a good key is
    verified once rather than on every rerun."""
    return narrative.verify_key()


# ═══ welcome ════════════════════════════════════════════════════════════════
def welcome():
    theme.band("NHW Data Studio",
               "Upload a spreadsheet. Get the read, the charts and a shareable "
               "report — without writing any code.")
    a, b = st.columns([2, 1.15])
    with a:
        theme.section("What it does")
        st.markdown("""
Point it at any Excel or CSV export — Kobo, DHIS2, a partner's tracker, a
monitoring sheet somebody keeps by hand — and it will

- **work out what every column is** (dates, categories, numbers, IDs, free
  text) and tell you what it decided, so nothing is a black box;
- **say what stands out**, ranked: the data-quality problems that change how
  every other figure reads come first, then coverage, concentration, trend and
  association;
- **draw only the charts worth drawing** — a sheet with sixty columns has about
  six charts in it, and this picks those;
- **hand you a report** — one self-contained HTML file that opens with no
  internet and prints to PDF, plus an Excel workbook holding every table behind
  every chart and a provenance sheet.

It is built for the exports NHW actually gets. Kobo `select_multiple` questions
are folded back into one question instead of forty stray yes/no columns.
Don't-know codes (98, 99, 999) are found and kept out of the averages. Values
that differ only by case or a trailing space — *Kano*, *kano&nbsp;* — are
reported, because they quietly split one category into three.
        """)
        theme.section("What it will not do")
        st.markdown("""
It will not tell you a number is *right*. Every figure it prints carries the
denominator it was computed from, and where the data cannot support a claim it
says so instead of drawing a prettier chart. Associations are reported as
associations. Nothing is written to disk or sent anywhere: the file lives in
this browser session and goes when you close the tab.
        """)
    with b:
        with st.container(border=True):
            st.markdown("#### Start")
            st.markdown("**← Drop a file in the sidebar.**\n\nComfortable to a "
                        "few hundred thousand rows; the 10,000-row exports this "
                        "was built for are instant.")
            st.markdown("Accepts `.xlsx`, `.xlsm`, `.csv`, `.tsv`. Multi-sheet "
                        "workbooks let you pick the sheet.")
        with st.container(border=True):
            st.markdown("#### Maps")
            st.caption(geo.ATTRIBUTION + " Nigeria state and LGA boundaries, "
                       "shown where your data has a place column.")
        with st.container(border=True):
            st.markdown("#### Five pages")
            for name, desc in [
                    ("Overview", "the read, the KPIs, the chosen charts"),
                    ("Findings", "every observation, filterable"),
                    ("Data health", "what to fix before quoting anything"),
                    ("Explore", "your own cuts, chart type chosen for you"),
                    ("Written brief", "optional Claude draft from the findings")]:
                st.markdown(f"**{name}** — {desc}")


# ═══ main ═══════════════════════════════════════════════════════════════════
def main():
    # Access first: nothing about the data, not even the uploader, before the
    # gate is satisfied. See app/gate.py for why it fails closed.
    if not gate.require():
        return
    src = sidebar_source()
    if src is None:
        welcome()
        st.sidebar.caption("Nothing loaded yet.")
        return
    key, data, name, sheet = src

    try:
        base = _profile(key, data, name, sheet)
    except Exception as exc:
        st.error(f"Could not read that file: {exc}")
        st.caption("If it is a Kobo or DHIS2 export, try the raw download "
                   "rather than a re-saved copy.")
        return
    if base.rows == 0:
        st.warning("That sheet has no data rows.")
        return

    # Nav above the filters: a multiselect that has been opened swallows clicks
    # in the space below it, so page links underneath a filter are unreliable to
    # hit — and the page you want is the more frequent choice anyway.
    st.sidebar.markdown("#### Page")
    page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")
    st.sidebar.divider()

    filt, mask = sidebar_filters(base)
    prof = base if not filt else _subset(
        key, data, name, sheet, filt, tuple(base.df.index[mask.to_numpy()]))
    found = _findings(prof, filt)
    st.sidebar.divider()
    st.sidebar.caption(
        f"**{prof.rows:,}** records · **{len(prof.columns)}** columns"
        + (f" · {base.truncated:,} rows past the cap were not read"
           if base.truncated else ""))
    st.sidebar.caption("Nothing is stored. Close the tab and it is gone.")

    {"Overview": page_overview, "Findings": page_findings,
     "Data health": page_health, "Explore": page_explore,
     "Written brief": page_brief}[page](prof, found, filt)


main()
