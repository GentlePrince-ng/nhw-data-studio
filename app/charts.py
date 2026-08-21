"""Pick the right chart for a column, and build it once.

Every visual in the app is a `Panel`: a title, a caption stating the denominator,
an Altair chart, and the aggregated frame behind it. Pages render panels; the
report exporter re-renders the same panels to SVG. Because both read one object,
what downloads is what was on screen — there is no second charting path that can
drift from the first.

Aggregation always happens here, before Altair sees anything. A 10,000-row sheet
becomes a 12-row bar frame, so chart size is set by the number of *categories*,
never the number of records, and the page stays instant at any file size. Only
the scatter sends rows to the browser, and it is sampled with the sample size
stated on the chart.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import altair as alt
import numpy as np
import pandas as pd

from . import geo
from . import profiling as pf
from . import theme

TOP_N = 12               # categories shown before the rest becomes "Other"
SCATTER_MAX = 4000       # points sent to the browser


@dataclass
class Panel:
    title: str
    chart: object = None
    table: object = None
    note: str = ""
    kind: str = ""
    cols: list = field(default_factory=list)

    def render(self, st, show_table: bool = False) -> None:
        theme.section(self.title, self.note)
        if self.chart is not None:
            # Maps take a different render path — see theme.geo_chart for why
            # Streamlit's Altair handling cannot carry GeoJSON.
            (theme.geo_chart if self.kind == "map" else theme.chart)(self.chart)
        if self.table is not None and (show_table or self.chart is None):
            # Column labels forced to strings: a crosstab of a numeric-looking
            # category gives a frame with mixed-type column names, which
            # Streamlit warns about on every render and converts anyway.
            t = self.table.copy()
            t.columns = [str(c) for c in t.columns]
            st.dataframe(t, width="stretch", hide_index=True)


def _top_n(vc: pd.Series, n: int = TOP_N) -> pd.Series:
    """Keep the n biggest categories, fold the rest into one honest bucket."""
    if len(vc) <= n:
        return vc
    head = vc.iloc[:n]
    rest = vc.iloc[n:]
    label = f"Other ({len(rest)} categories)"
    return pd.concat([head, pd.Series({label: int(rest.sum())})])


# ═══ one column at a time ═══════════════════════════════════════════════════
def cat_bar(p: pf.Profile, col: str, n: int = TOP_N) -> Panel:
    c = p.columns[col]
    vc = _top_n(p.df[col].value_counts(), n)
    frame = pd.DataFrame({"Category": vc.index.astype(str),
                          "Records": vc.to_numpy()})
    frame["Share"] = (100 * frame["Records"] / max(c.filled, 1)).round(1)
    note = f"{c.filled:,} answered"
    if c.missing:
        note += f" · {c.missing_pct:.0f}% blank"
    if c.unique > n:
        note += f" · {c.unique} categories, top {n} shown"
    return Panel(c.short or c.label, theme.bar(frame, "Records", "Category"),
                 frame, note + (f" · {c.label}" if c.short != c.label else ""),
                 "categorical", [col])


def num_hist(p: pf.Profile, col: str, bins: int = 28) -> Panel:
    """Distribution, with sentinel codes stripped and said so."""
    c = p.columns[col]
    s = pd.to_numeric(p.df[col], errors="coerce").dropna()
    note_bits = [f"{len(s):,} values"]
    if c.sentinel_hits:
        before = len(s)
        s = s[~s.isin(list(c.sentinel_hits))]
        note_bits.append(f"{before - len(s):,} coded "
                         f"{'/'.join(map(str, sorted(c.sentinel_hits)))} excluded")
    if s.empty:
        return Panel(c.label, None, None, "no usable values", "numeric", [col])
    st_ = c.stats
    note_bits.append(f"mean {st_.get('mean', 0):,.1f} · median "
                     f"{st_.get('median', 0):,.1f}")
    # Binned here in numpy, not by Vega. Handing Vega the raw column meant
    # 10,000 values in the chart spec: it tripped Altair's 5,000-row ceiling, and
    # because the exporter swallows chart errors the histograms were quietly
    # missing from every downloaded report while looking fine on screen. Twenty-
    # eight pre-counted bins render identically and always export.
    counts, edges = np.histogram(s.to_numpy(dtype="float64"), bins=bins)
    binned = pd.DataFrame({"start": edges[:-1], "end": edges[1:],
                           "Records": counts})
    ch = alt.Chart(binned).mark_bar(
        color=theme.GREEN_DEEP, cornerRadiusEnd=3, opacity=0.9,
    ).encode(
        x=alt.X("start:Q", bin="binned", title=c.label,
                scale=alt.Scale(zero=False)),
        x2="end:Q",
        y=alt.Y("Records:Q", title="Records"),
        tooltip=[alt.Tooltip("start:Q", title="from", format=",.4~f"),
                 alt.Tooltip("end:Q", title="to", format=",.4~f"),
                 alt.Tooltip("Records:Q", title="records", format=",")],
    ).properties(height=200)
    quant = s.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    table = quant.rename("Value").rename_axis("Statistic").reset_index()
    table["Value"] = table["Value"].round(2)
    return Panel(c.label, ch, table, " · ".join(note_bits), "numeric", [col])


def time_trend(p: pf.Profile, col: str | None = None,
               freq: str | None = None) -> Panel | None:
    """Records per period. The grain follows the span, not a fixed guess."""
    col = col or p.best_date()
    if not col:
        return None
    s = pd.to_datetime(p.df[col], errors="coerce").dropna()
    if len(s) < 3:
        return None
    span = (s.max() - s.min()).days
    if freq is None:
        freq = "D" if span <= 62 else ("W" if span <= 400 else "MS")
    label = {"D": "day", "W": "week", "MS": "month"}[freq]
    g = s.dt.to_period({"D": "D", "W": "W", "MS": "M"}[freq]).value_counts().sort_index()
    frame = pd.DataFrame({"Period": g.index.to_timestamp(),
                          "Records": g.to_numpy()})
    ch = theme.trend(frame, "Period", "Records", height=210, y_title="Records",
                     x_format={"D": "%d %b", "W": "%d %b", "MS": "%b %y"}[freq])
    note = (f"{len(s):,} dated records · per {label} · "
            f"{theme.day(s.min())} to {theme.day(s.max())}")
    out = frame.copy()
    out["Period"] = out["Period"].dt.strftime("%Y-%m-%d")
    return Panel(f"Records over time — {p.label(col)}", ch, out, note,
                 "trend", [col])


def multi_bar(p: pf.Profile, parent: str) -> Panel | None:
    """One folded Kobo select_multiple question."""
    members = p.kobo.get(parent, [])
    if not members:
        return None
    counts = pf.multi_counts(p.df, members)
    if counts.empty:
        return None
    frame = counts.rename(columns={"option": "Option", "n": "Selected"})
    frame["Share of rows"] = (100 * frame["Selected"] / max(p.rows, 1)).round(1)
    return Panel(pf.clean_label(parent),
                 theme.bar(frame.head(TOP_N), "Selected", "Option"), frame,
                 f"multi-select · {len(members)} options · shares sum past 100%",
                 "multi", [parent])


# ═══ two columns ════════════════════════════════════════════════════════════
def group_bar(p: pf.Profile, num: str, cat: str, agg: str = "mean") -> Panel:
    """A numeric summarised by a category — the workhorse comparison."""
    c = p.columns[num]
    s = pd.to_numeric(p.df[num], errors="coerce")
    if c.sentinel_hits:
        s = s.where(~s.isin(list(c.sentinel_hits)))
    d = pd.DataFrame({"g": p.df[cat].astype(str), "v": s}).dropna()
    g = d.groupby("g")["v"].agg([agg, "count"]).reset_index()
    g.columns = ["Category", agg.title(), "Records"]
    g = g.sort_values(agg.title(), ascending=False).head(TOP_N)
    g[agg.title()] = g[agg.title()].round(2)
    thin = int((g["Records"] < 20).sum())
    note = f"{agg} of {c.label.lower()} by {p.label(cat).lower()}"
    if thin:
        note += f" · {thin} group(s) under 20 records — read with care"
    return Panel(f"{c.short or c.label} by {p.short(cat)}",
                 theme.bar(g, agg.title(), "Category", fmt=",.2f"), g, note,
                 "group", [num, cat])


def cross_heat(p: pf.Profile, a: str, b: str, pct: bool = True) -> Panel:
    """Crosstab as a shaded grid, row-percentaged so groups stay comparable.

    Counts alone make the biggest group look like the strongest pattern. Row
    percentages answer the question the reader has — within this group, how does
    it split — and the count stays in the tooltip.
    """
    ct = pd.crosstab(p.df[a].astype(str), p.df[b].astype(str))
    if ct.empty:
        return Panel(f"{p.label(a)} by {p.label(b)}", None, None, "no overlap")
    ct = ct.loc[ct.sum(1).sort_values(ascending=False).index[:14],
                ct.sum(0).sort_values(ascending=False).index[:14]]
    long = ct.reset_index().melt(id_vars=ct.index.name or "index",
                                 var_name="col", value_name="n")
    long.columns = ["row", "col", "n"]
    totals = long.groupby("row")["n"].transform("sum")
    long["share"] = (100 * long["n"] / totals.replace(0, np.nan)).round(1)
    field = "share" if pct else "n"
    ch = alt.Chart(long).mark_rect(stroke=theme.SURFACE, strokeWidth=1).encode(
        x=alt.X("col:N", title=p.label(b), axis=alt.Axis(labelAngle=-35)),
        y=alt.Y("row:N", title=p.label(a)),
        color=alt.Color(f"{field}:Q",
                        title="% of row" if pct else "records",
                        scale=alt.Scale(range=theme.RAMP), legend=alt.Legend(
                            orient="right", gradientLength=120)),
        tooltip=[alt.Tooltip("row:N", title=p.label(a)),
                 alt.Tooltip("col:N", title=p.label(b)),
                 alt.Tooltip("n:Q", title="records", format=","),
                 alt.Tooltip("share:Q", title="% of row", format=".1f")],
    ).properties(height=alt.Step(26))
    out = ct.reset_index()
    return Panel(f"{p.short(a)} by {p.short(b)}", ch, out,
                 f"{int(ct.to_numpy().sum()):,} records with both answered · "
                 "shaded by % of row", "cross", [a, b])


def scatter(p: pf.Profile, x: str, y: str, colour: str | None = None) -> Panel:
    """Two numerics, sampled, with the correlation stated rather than implied."""
    cols = [x, y] + ([colour] if colour else [])
    d = p.df[cols].copy()
    for c in (x, y):
        d[c] = pd.to_numeric(d[c], errors="coerce")
        hits = p.columns[c].sentinel_hits
        if hits:
            d[c] = d[c].where(~d[c].isin(list(hits)))
    d = d.dropna(subset=[x, y])
    n = len(d)
    r = d[x].corr(d[y]) if n > 2 else np.nan
    if n > SCATTER_MAX:
        d = d.sample(SCATTER_MAX, random_state=7)
    enc = dict(x=alt.X(f"{x}:Q", title=p.label(x), scale=alt.Scale(zero=False)),
               y=alt.Y(f"{y}:Q", title=p.label(y), scale=alt.Scale(zero=False)))
    tips = [alt.Tooltip(f"{x}:Q", title=p.label(x), format=",.2f"),
            alt.Tooltip(f"{y}:Q", title=p.label(y), format=",.2f")]
    if colour:
        d[colour] = d[colour].astype(str)
        enc["color"] = alt.Color(f"{colour}:N", title=p.label(colour),
                                 scale=alt.Scale(range=theme.SERIES))
        tips.append(alt.Tooltip(f"{colour}:N", title=p.label(colour)))
    pts = alt.Chart(d).mark_circle(size=42, opacity=0.5,
                                   color=theme.GREEN_DEEP).encode(**enc,
                                                                  tooltip=tips)
    ch = pts + pts.transform_regression(x, y).mark_line(
        color=theme.INK2, strokeWidth=1.5, strokeDash=[4, 3])
    note = f"{n:,} records with both values"
    if n > SCATTER_MAX:
        note += f" · {SCATTER_MAX:,} plotted at random"
    if pd.notna(r):
        note += f" · correlation {r:+.2f} (not causation)"
    return Panel(f"{p.short(y)} against {p.short(x)}",
                 ch.properties(height=300), None, note, "scatter", [x, y])


def cat_trend(p: pf.Profile, date: str, cat: str, n: int = 5) -> Panel | None:
    """Top categories over time — how the mix moved, not just the total."""
    s = pd.to_datetime(p.df[date], errors="coerce")
    d = pd.DataFrame({"d": s, "g": p.df[cat].astype(str)}).dropna()
    if d.empty:
        return None
    keep = d["g"].value_counts().head(n).index
    d = d[d["g"].isin(keep)]
    span = (d["d"].max() - d["d"].min()).days
    per = "W" if span <= 400 else "M"
    d["Period"] = d["d"].dt.to_period(per).dt.to_timestamp()
    g = d.groupby(["Period", "g"], observed=True).size().reset_index(name="Records")
    ch = alt.Chart(g).mark_line(strokeWidth=2).encode(
        x=alt.X("Period:T", title=None,
                axis=alt.Axis(format="%d %b" if per == "W" else "%b %y")),
        y=alt.Y("Records:Q", title="Records"),
        color=alt.Color("g:N", title=None, scale=alt.Scale(range=theme.SERIES)),
        tooltip=[alt.Tooltip("Period:T", title="period"),
                 alt.Tooltip("g:N", title=p.label(cat)),
                 alt.Tooltip("Records:Q", format=",")],
    ).properties(height=230)
    out = g.copy()
    out["Period"] = out["Period"].dt.strftime("%Y-%m-%d")
    out = out.rename(columns={"g": p.label(cat)})
    return Panel(f"{p.short(cat)} over time", ch, out,
                 f"top {len(keep)} of {p.df[cat].nunique()} categories · "
                 f"per {'week' if per == 'W' else 'month'}", "cat_trend",
                 [date, cat])


# ═══ maps ══════════════════════════════════════════════════════════════════
def geo_level(name) -> str:
    """Which boundary layer a place column belongs to.

    Decided from the column name, not the values: an LGA column and a state
    column both hold Nigerian place names, and joining LGA values against state
    boundaries returns a nearly empty map with no obvious reason why.
    """
    low = str(name).lower()
    if any(w in low for w in ("lga", "local government", "council")):
        return "lga"
    if any(w in low for w in ("ward", "community", "facility", "settlement")):
        return "lga"          # finer than LGA: the LGA layer is the closest fit
    return "state"


def state_map(p: pf.Profile, place: str | None = None, value: str | None = None,
              agg: str = "count", level: str = "state") -> Panel | None:
    """A shaded map of Nigeria, if the place column can actually be matched.

    Returns None rather than an empty map when fewer than four places join — a
    map of three shaded states and thirty-four grey ones tells the reader less
    than the bar chart does, and takes four times the space to do it.
    """
    place = place or (p.best_geo() if level == "state" else None)
    if not place or not geo.available(level):
        return None
    state_col = None
    if level == "lga":
        state_col = next((c.name for c in p.by_role(pf.GEO)
                          if "state" in c.name.lower()), None)
    j = geo.summarise(p.df, place, value, agg, state_col=state_col, level=level)
    if not j.features or j.matched < 4:
        return None

    counting = agg == "count" or not value
    what = "Records" if counting else f"{agg.title()} of {p.short(value)}"
    fmt = ",.0f" if counting else ",.2f"
    unit = "states and the FCT" if level == "state" else "LGAs"
    bits = [f"{j.matched} of {len(j.features)} {unit} have data"]
    if j.unused:
        bits.append(f"{len(j.unused)} shown grey — no records, not zero")
    if j.unmatched:
        shown = ", ".join(map(str, j.unmatched[:4]))
        bits.append(f"{len(j.unmatched)} value(s) matched no boundary ({shown}"
                    f"{'…' if len(j.unmatched) > 4 else ''}) and are not mapped")
    if j.merged:
        bits.append(f"{len(j.merged)} place(s) had spellings combined to match "
                    "(the bar chart still shows them apart)")
    if level == "lga" and not state_col:
        bits.append("no state column, so LGA names were matched unqualified — "
                    "five names belong to two states each")
    table = j.table.rename(columns={"Value": what})
    return Panel(f"{what} by {p.label(place) if level == 'state' else 'LGA'}",
                 geo.choropleth(j.features, "", what, fmt,
                                outline=level == "lga"), table,
                 " · ".join(bits), "map", [place] + ([value] if value else []))


# ═══ whole-sheet views ══════════════════════════════════════════════════════
def missing_bar(p: pf.Profile, limit: int = 20) -> Panel | None:
    """Completeness, worst first — the only chart that should be read before
    any other number on the page."""
    rows = [{"Column": c.label, "Missing %": round(c.missing_pct, 1),
             "Filled": c.filled}
            for c in p.columns.values() if c.missing]
    if not rows:
        return None
    frame = pd.DataFrame(rows).sort_values("Missing %", ascending=False)
    shown = frame.head(limit)
    ch = theme.bar(shown, "Missing %", "Column", colour=theme.CLOSE, fmt=".1f")
    return Panel("Missing values by column", ch, frame,
                 f"{len(frame)} of {len(p.columns)} columns have blanks"
                 + (f" · worst {limit} shown" if len(frame) > limit else ""),
                 "missing")


def type_table(p: pf.Profile) -> Panel:
    rows = []
    for c in p.columns.values():
        rows.append({"Column": c.name, "Reads as": c.role,
                     "Filled": c.filled, "Missing %": round(c.missing_pct, 1),
                     "Distinct": c.unique,
                     "Example": c.sample[0] if c.sample else ""})
    frame = pd.DataFrame(rows)
    return Panel("Every column, as the tool read it", None, frame,
                 f"{len(frame)} columns · change nothing here — this is what "
                 "the analysis used", "types")


def corr_grid(p: pf.Profile, limit: int = 10) -> Panel | None:
    """Correlation matrix over the most usable numerics."""
    nums = [c.name for c in p.headline_nums(limit=limit)]
    if len(nums) < 3:
        return None
    sub = p.df[nums].apply(pd.to_numeric, errors="coerce")
    for c in nums:
        hits = p.columns[c].sentinel_hits
        if hits:
            sub[c] = sub[c].where(~sub[c].isin(list(hits)))
    corr = sub.corr().round(2)
    long = corr.reset_index().melt(id_vars="index", var_name="b",
                                   value_name="r").dropna()
    long.columns = ["a", "b", "r"]
    lbl = {n: p.label(n) for n in nums}
    long["a"] = long["a"].map(lbl)
    long["b"] = long["b"].map(lbl)
    ch = alt.Chart(long).mark_rect(stroke=theme.SURFACE, strokeWidth=1).encode(
        x=alt.X("b:N", title=None, axis=alt.Axis(labelAngle=-35)),
        y=alt.Y("a:N", title=None),
        # Diverging: sign is the point, so a two-ended scale is the honest one.
        color=alt.Color("r:Q", title="r",
                        scale=alt.Scale(domain=[-1, 0, 1],
                                        range=["#B42318", "#F5F5F4", theme.GREEN_DEEP]),
                        legend=alt.Legend(orient="right", gradientLength=120)),
        tooltip=[alt.Tooltip("a:N"), alt.Tooltip("b:N"),
                 alt.Tooltip("r:Q", format="+.2f")],
    ).properties(height=alt.Step(26))
    return Panel("How the numeric columns move together", ch,
                 corr.reset_index().rename(columns={"index": ""}),
                 f"{len(nums)} columns · Pearson r · sentinel codes excluded",
                 "corr")


def auto_panels(p: pf.Profile, limit: int = 8) -> list:
    """The front page, assembled: time first, then the strongest categories.

    Chosen rather than exhaustive. A sheet with 60 columns has 60 possible bar
    charts and about six worth looking at, and a page that draws all 60 is the
    reason people go back to Excel.
    """
    panels = []
    trend_panel = time_trend(p)
    if trend_panel:
        panels.append(trend_panel)
    place = p.best_geo()
    if place:
        # Map first where it can be drawn, then the bar: the map answers "where"
        # at a glance, the bar answers "how many" precisely, and they are worth
        # different things. Neither replaces the other.
        mp = state_map(p, place, level=geo_level(place))
        if mp:
            panels.append(mp)
        panels.append(cat_bar(p, place))

    # A finer place column — LGA, ward — gets a map and only a map. Above about
    # 25 categories a bar chart is a wall of labels nobody reads, and the top-12
    # fold hides most of the country; a choropleth shows all 774 at once. This is
    # the one case where the map is not a companion to a bar but a replacement.
    fine = next((c for c in p.by_role(pf.GEO)
                 if geo_level(c.name) == "lga" and c.unique > 25
                 and c.name != place), None)
    if fine:
        fp = state_map(p, fine.name, level="lga")
        if fp:
            panels.append(fp)
    for c in p.headline_cats(limit=limit):
        if place and c.name == place:
            continue
        panels.append(cat_bar(p, c.name))
        if len(panels) >= limit:
            break
    for parent in list(p.kobo)[:3]:
        mp = multi_bar(p, parent)
        if mp:
            panels.append(mp)
    for c in p.headline_nums(limit=3):
        panels.append(num_hist(p, c.name))
    return panels


# ═══ measure × dimension: the charts a dashboard is actually made of ════════
# Everything above draws one column at a time — a distribution, a count of
# categories. Useful once, to see what is in the file. What a reader wants is a
# measure sliced by something: wait time by facility type, doses by month, the
# change by state. These take a metrics.Measure and always state which
# aggregation was used, because "average" and "total" answer different questions
# and a bar chart shows no sign of which one it is.

def measure_trend(p: pf.Profile, m, date: str | None = None,
                  freq: str | None = None) -> Panel | None:
    """A measure over time — the front-page chart of any dashboard."""
    from . import metrics
    date = date or p.best_date()
    if not date:
        return None
    frame, grain = metrics.over_time(p.df, m, date, p, freq)
    if len(frame) < 2:
        return None
    fmt = {"D": "%d %b", "W": "%d %b", "MS": "%b %y"}[
        freq or metrics.grain_for(
            (pd.to_datetime(p.df[date], errors="coerce").max()
             - pd.to_datetime(p.df[date], errors="coerce").min()).days)]
    ch = theme.trend(frame, "Period", m.verb, height=230, y_title=m.verb,
                     x_format=fmt)
    out = frame.copy()
    out["Period"] = pd.to_datetime(out["Period"]).dt.strftime("%Y-%m-%d")
    return Panel(f"{m.title()} over time", ch, out,
                 f"per {grain} · {m.why}"
                 + (f" · {m.full}" if m.full and m.full != m.label else ""),
                 "metric_trend", [m.col, date])


def measure_by(p: pf.Profile, m, dim: str, top: int = 12,
               min_spread: float = 0.0) -> Panel | None:
    """A measure broken down by a dimension, with the group sizes attached.

    `min_spread` suppresses a breakdown that says nothing: if every group lands
    within a few percent of the mean, the chart is noise wearing the clothes of
    a finding. The front page passes a threshold; Explore passes none, because
    there the reader asked for that specific cut and deserves it either way.
    """
    from . import metrics
    frame = metrics.by_dimension(p.df, m, dim, p, top)
    if frame.empty:
        return None
    if min_spread and len(frame) > 2             and metrics.spread(frame, m.verb) < min_spread:
        return None
    thin = int((frame["Records"] < 20).sum())
    note = m.why
    if m.full and m.full != m.label:
        note += f" · {m.full}"
    merged = frame.attrs.get("merged", 0)
    if merged:
        note += f" · {merged} spelling(s) merged into their commonest form"
    if thin:
        note += f" · {thin} group(s) under 20 records"
    if len(frame) == top:
        note += f" · top {top}"
    return Panel(f"{m.title()} by {p.short(dim)}",
                 theme.bar(frame, m.verb, "Category", fmt=m.fmt()), frame,
                 note, "measure_by", [m.col, dim])


def change_by(p: pf.Profile, m, dim: str, date: str | None = None) -> Panel | None:
    """What moved. A diverging bar of the change per dimension value.

    The one chart that answers "the number changed, why" — which is the question
    a breakdown of the *level* cannot answer, however many bars it has.
    """
    from . import metrics
    date = date or p.best_date()
    if not date:
        return None
    frame = metrics.change_by_dimension(p.df, m, dim, date, p)
    if frame.empty or frame["Change"].abs().sum() == 0:
        return None
    d = frame.copy()
    d["Direction"] = np.where(d["Change"] >= 0, "up", "down")
    ch = alt.Chart(d).mark_bar(cornerRadiusEnd=3, height=13).encode(
        x=alt.X("Change:Q", title=f"change in {m.verb.lower()}"),
        y=alt.Y("Category:N", sort=alt.EncodingSortField(
            field="Change", op="sum", order="descending"), title=None),
        color=alt.Color("Direction:N", legend=None,
                        scale=alt.Scale(domain=["up", "down"],
                                        range=[theme.GREEN_DEEP, theme.BEHIND])),
        tooltip=[alt.Tooltip("Category:N", title=""),
                 alt.Tooltip("Before:Q", format=m.fmt()),
                 alt.Tooltip("After:Q", format=m.fmt()),
                 alt.Tooltip("Change:Q", format="+" + m.fmt()),
                 alt.Tooltip("Records:Q", title="records now", format=",")],
    ).properties(height=alt.Step(24))
    win = metrics.compare_periods(p.df, m, date, p).get("window", 30)
    return Panel(f"What moved — {m.title()} by {p.short(dim)}",
                 ch, frame,
                 f"last {win} days against the {win} before · green up, red down",
                 "change_by", [m.col, dim, date])


def dashboard(p: pf.Profile, limit: int = 10) -> list:
    """The front page: metrics over time, then broken down, then what moved.

    Deliberately measure-led rather than column-led. The old front page drew
    every interesting column once; this draws the two or three measures that
    matter, each answered three ways — level, split, and change.
    """
    from . import metrics
    ms = metrics.measures(p, limit=3)
    dims = metrics.dimensions(p, limit=3)
    date = p.best_date()
    place = p.best_geo()
    panels = []

    label_col = metrics.label_column(p) if len(dims) < 3 else None
    seq_dim = next((d.name for d in dims
                    if len(metrics.ordinal_order(p.df[d.name])) >= 3), None)
    for m in ms[:3]:
        if date:
            panels.append(measure_trend(p, m, date))
        elif seq_dim:
            # No date anywhere, but an ordered run of labels is a time axis.
            panels.append(sequence_trend(p, m, seq_dim))
        for dim in dims[:2]:
            # The sequence line already shows this dimension; a bar of the same
            # numbers beside it is padding, not a second view.
            if dim.name == seq_dim and not date:
                continue
            panels.append(measure_by(p, m, dim.name, min_spread=0.12))
        if date and dims:
            panels.append(change_by(p, m, dims[0].name, date))
        # A long tail of named things: rank them. This is the only chart a GA4
        # export or an article list can support, and it is also the one anybody
        # would ask for first.
        if label_col and m.col != metrics.RECORDS.col:
            panels.append(top_items(p, m, label_col))
        if place and m.col != metrics.RECORDS.col:
            panels.append(state_map(p, place, m.col,
                                    "mean" if m.agg == "mean" else "sum",
                                    level=geo_level(place)))
        if len([x for x in panels if x]) >= limit:
            break
    out = [x for x in panels if x]
    # Same idea on the dashboard: no measures worth charting means the questions
    # are the story.
    if len([m for m in ms if m.col != metrics.RECORDS.col]) <= 2:
        for d in dims[:4]:
            if len(out) >= limit:
                break
            out.append(cat_bar(p, d.name, n=12))
    return out[:limit]

def sequence_trend(p: pf.Profile, m, dim: str) -> Panel | None:
    """A measure across an ordered run of labels — Week 1..11, Wave 1..4.

    The trend chart for a file with no date column, which is most
    hand-maintained trackers. Without it a weekly campaign has no time axis at
    all, and the only view of eleven weeks is a bar chart sorted by size.
    """
    from . import metrics
    order = metrics.ordinal_order(p.df[dim])
    if len(order) < 3:
        return None
    frame = metrics.by_dimension(p.df, m, dim, p, top=200)
    if frame.empty:
        return None
    frame = frame[frame["Category"].isin(order)]
    if len(frame) < 3:
        return None
    ch = alt.Chart(frame).mark_line(
        strokeWidth=2, point=alt.OverlayMarkDef(size=45, filled=True),
        color=theme.GREEN_DEEP,
    ).encode(
        x=alt.X("Category:N", sort=order, title=None,
                axis=alt.Axis(labelAngle=-35)),
        y=alt.Y(f"{m.verb}:Q", title=m.verb, scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("Category:N", title=p.short(dim)),
                 alt.Tooltip(f"{m.verb}:Q", format=m.fmt()),
                 alt.Tooltip("Records:Q", title="records", format=",")],
    ).properties(height=230)
    return Panel(f"{m.title()} across {p.short(dim)}", ch, frame,
                 f"{len(frame)} in sequence · {m.why}", "sequence_trend",
                 [m.col, dim])

def top_items(p: pf.Profile, m, label: str, top: int = 15) -> Panel | None:
    """A ranked bar of the individual things that make up a measure.

    The right chart for a long tail: articles, pages, videos, facilities. A
    dimension has a handful of values and answers "which group"; a label has
    thousands and answers "which one" — and "which one" is the question an
    editorial or campaign report is written to answer.
    """
    from . import metrics
    frame = metrics.rank_items(p.df, m, label, p, top=top)
    if frame.empty or len(frame) < 3:
        return None
    total = metrics.total(p.df, m, p)
    shown = float(pd.to_numeric(frame[m.verb], errors="coerce").sum())
    note = f"top {len(frame)} of {p.df[label].nunique():,}"
    if m.agg == "sum" and total:
        note += f" · they are {shown / total:.0%} of the {m.title().lower()}"
    d = frame.copy()
    d["Item"] = d["Item"].str.slice(0, 58)
    return Panel(f"{m.title()} — top {len(frame)} by {p.short(label)}",
                 theme.bar(d, m.verb, "Item", fmt=m.fmt()), frame, note,
                 "top_items", [m.col, label])
