"""Measures and dimensions — the model this tool was missing.

Typing a column is not the same as knowing what to *do* with it, and the gap
between those two is the difference between a profiling report and a dashboard.
Every auto-dashboard tool worth using makes the same move first: split the
columns into **measures** (things you aggregate — visits, wait time, doses) and
**dimensions** (things you slice by — state, facility type, month), then build
metrics from measure × dimension × period. Without that step you can only draw
one column at a time, which is how you end up with a wall of distributions and a
KPI row reporting how many rows the file has.

Two judgements live here and both are guesses that must be visible to the reader:

* **The default aggregation.** Summing an age column is meaningless; averaging a
  doses-administered column hides the total. The name is the best signal
  available — `*_count`, `visits`, `doses`, `cases` sum; `rate`, `pct`, `age`,
  `minutes`, `score` average — and the chosen aggregation is printed on every
  figure, because a wrong guess that says what it did is recoverable and a wrong
  guess that stays quiet is not.
* **What counts as a comparison period.** Period-over-period is where the
  meaning is, but a half-finished month against a full one always reads as
  collapse. Windows are equal-length and end at the last *complete* period.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import profiling as pf

# Name fragments that imply a total rather than a level. Deliberately about
# counting things, not measuring them.
SUM_WORDS = ("count", "total", "number", "num", "visits", "attendance",
             "attended", "doses", "cases", "deaths", "births", "tests",
             "referrals", "sessions", "reached", "trained", "screened",
             "distributed", "stock", "quantity", "qty", "amount", "cost",
             "budget", "spend", "revenue", "population", "beneficiaries",
             # Engagement metrics are counts of events, and a campaign report
             # wants their totals. A tracker came out with retweets summed but
             # likes and views averaged — same kind of number, different
             # treatment, purely because of where the integer cut-off fell.
             "views", "pageviews", "pageview", "sessions", "impressions",
             "reach", "likes", "retweets", "shares", "emails", "articles",
             "comments", "clicks", "engagement", "engagements", "followers",
             "subscribers", "plays", "watch", "downloads", "opens", "sends",
             "bookmarks", "quotes", "saves", "replies", "mentions", "posts")

# A name that opens by counting something is a total, whatever follows.
COUNT_PREFIXES = ("no of", "number of", "count of", "total", "sum of", "# of",
                  "#", "no. of", "nos of")
# Fragments that imply a level or a rate, where a total is nonsense.
MEAN_WORDS = ("rate", "ratio", "pct", "percent", "percentage", "share",
              "proportion", "average", "mean", "median", "score", "index",
              "age", "minutes", "hours", "days", "duration", "wait", "time",
              "weight", "height", "bmi", "temperature", "distance", "km",
              "coverage", "density", "price", "salary", "per",
              # Per-record attributes of a facility or person: a total across
              # rows counts the same nurse once per visit.
              "staff", "duty", "present", "available", "beds", "level",
              "strength", "size")


@dataclass
class Measure:
    """A number worth aggregating, and how to aggregate it."""
    col: str
    label: str                    # short, for titles
    agg: str                      # "sum", "mean" or "share"
    why: str                      # how the aggregation was chosen, for the UI
    integerish: bool = False
    full: str = ""                # the complete column name, for captions

    # A survey's real measures are usually shares, not sums: "do you experience
    # long wait times" has no total worth reporting, but the percentage saying
    # yes is the number that goes in the report. Pandas cannot aggregate
    # "share", so it runs as a mean and is scaled at the end.
    @property
    def pagg(self) -> str:
        return "mean" if self.agg in ("mean", "share") else "sum"

    @property
    def scale(self) -> float:
        return 100.0 if self.agg == "share" else 1.0

    @property
    def verb(self) -> str:
        return {"sum": "Total", "mean": "Average", "share": "% yes"}[self.agg]

    # Column names often already say how they are aggregated — "Total
    # pageviews", "Average reading minutes". Prepending the verb anyway produced
    # "Total total pageviews" and "Average average reading minutes".
    SELF_DESCRIBING = ("total ", "sum ", "average ", "avg ", "mean ", "median ",
                       "percent", "percentage", "share of", "no of",
                       "number of", "count of", "rate of", "ratio of")

    def title(self) -> str:
        if self.agg == "share":
            return f"{self.label} — % yes"
        low = self.label.strip().lower()
        if low.startswith(self.SELF_DESCRIBING):
            return self.label[:1].upper() + self.label[1:]
        return f"{self.verb} {self.label.lower()}"

    def fmt(self) -> str:
        if self.agg == "share":
            return ",.1f"
        return ",.0f" if (self.agg == "sum" or self.integerish) else ",.1f"


RECORDS = Measure("__records__", "Records", "sum",
                  "one per row in the file", integerish=True)

YES = {"yes", "y", "true", "1", "1.0"}
NO = {"no", "n", "false", "0", "0.0"}


def _yes_no(prof: pf.Profile, col: str) -> bool:
    """True when a column is a yes/no question, so a share is meaningful.

    Deliberately strict. A two-valued column of Male/Female is binary too, but
    "% Male" is not a metric anybody asked for — the share only means something
    when one of the two answers is affirmative.
    """
    vals = {str(v).strip().lower()
            for v in prof.df[col].dropna().unique()[:10]}
    return bool(vals) and vals <= (YES | NO) and bool(vals & YES)


def share_measures(prof: pf.Profile, limit: int = 12) -> list:
    """One measure per yes/no question: the percentage answering yes."""
    whole, options = [], []
    for c in prof.by_role(pf.BINARY):
        if pf.is_system(c.name) or not _yes_no(prof, c.name):
            continue
        if c.filled < max(20, 0.1 * c.n):
            continue
        m = Measure(c.name, c.short or c.label, "share",
                    "share of answers that were yes", full=c.label)
        # An option inside a select-all-that-apply family is a real share, but
        # six of them from one question would crowd out five other questions.
        # Distinct questions first, options after.
        if "/" in c.name and c.name.rsplit("/", 1)[0] in prof.kobo:
            options.append(m)
        else:
            whole.append(m)
    return (whole + options)[:limit]


def _agg_for(name: str, stats: dict) -> tuple[str, str]:
    """(aggregation, why). Name first, then the shape of the values."""
    low = re.sub(r"[^a-z# ]+", " ", pf.strip_prefix(name).lower()).strip()
    low = re.sub(r"\s+", " ", low)
    tokens = set(low.split())
    if low.startswith(COUNT_PREFIXES):
        return "sum", "totalled: the name opens by counting something"
    # Whole tokens only. Substring matching read "Combined_pageviews" as a rate
    # because "pageviews" contains "age", so the headline metric of an editorial
    # report was averaged instead of totalled — 146 rather than 15,142.
    if tokens & set(MEAN_WORDS):
        return "mean", "averaged: the name reads as a rate, level or duration"
    if tokens & set(SUM_WORDS):
        return "sum", "totalled: the name reads as a count or an amount"
    # No signal in the name. Small non-negative integers behave like counts;
    # anything else is safer averaged.
    if stats.get("integerish") and stats.get("min", -1) >= 0 \
            and stats.get("max", 1e9) <= 1000:
        return "sum", "totalled: small whole numbers, which usually count things"
    return "mean", "averaged: no signal in the name, so a total may not be meaningful"


def measures(prof: pf.Profile, limit: int = 6, overrides: dict | None = None
             ) -> list:
    """The measures worth putting on a dashboard, best first.

    `Records` is always first: a count of rows is the one measure every dataset
    has, and for survey data it is usually the measure people actually mean.

    `overrides` maps a column to "sum" or "mean" and wins over the guess. Every
    name-based rule here will be wrong for somebody's column — "staff on duty"
    is a level, "doses given" is a total, and no word list settles it — so the
    guess is a starting point the reader can correct, not a verdict.
    """
    overrides = overrides or {}
    out = [RECORDS]
    for c in prof.headline_nums(limit=limit * 2):
        agg, why = _agg_for(c.name, c.stats)
        if c.name in overrides and overrides[c.name] != agg:
            agg, why = overrides[c.name], "set by you on this page"
        if pf.is_system(c.name):
            continue
        out.append(Measure(c.name, c.short or c.label, agg, why,
                           bool(c.stats.get("integerish")), full=c.label))
        if len(out) > limit:
            break
    # A perception survey has few things to total and many things to take a
    # share of; without these it produced a two-measure dashboard from 88
    # columns, 22 of which were yes/no questions.
    out += share_measures(prof, limit=max(2, limit))
    return out[: limit * 3]


def _sliceable(prof: pf.Profile, c) -> bool:
    """A dimension has to be a label, not a sentence.

    `enumerator_notes` held five canned sentences, so it typed as categorical
    and — being evenly spread — ranked as one of the *best* dimensions. Nobody
    wants "average wait time by 'Facility busy at time of visit.'". Long values
    are prose; prose is not a dimension.
    """
    if not (2 <= c.unique <= 25):
        return False
    vals = prof.df[c.name].dropna().astype(str)
    if vals.empty:
        return False
    return float(vals.str.len().mean()) <= 28


def year_column(prof: pf.Profile) -> str | None:
    """A numeric column of years, which is a time axis rather than a number.

    Long-format indicator tables — Region / Indicator / Year / Value — are the
    commonest shape in health data and they carry no date column at all. The
    year sat in a numeric column, so it was neither a dimension nor a date, and
    an 800-row indicator table produced no charts whatsoever.
    """
    for name, c in prof.columns.items():
        if c.role != pf.NUMERIC or pf.is_system(name):
            continue
        vals = pd.to_numeric(prof.df[name], errors="coerce").dropna()
        if vals.empty or not 3 <= c.unique <= 40:
            continue
        if (vals % 1 == 0).all() and vals.between(1900, 2100).all():
            return name
    return None


def dimensions(prof: pf.Profile, limit: int = 6) -> list:
    """Columns worth slicing by — few enough levels to compare at a glance."""
    dims = [c for c in prof.headline_cats(limit=limit * 4)
            if _sliceable(prof, c) and not pf.is_system(c.name)]
    year = year_column(prof)
    if year and year not in [d.name for d in dims]:
        dims.insert(0, prof.columns[year])
    place = prof.best_geo()
    if place and place not in [d.name for d in dims]:
        col = prof.columns.get(place)
        if col and 2 <= col.unique <= 25:
            dims.insert(0, col)
    return dims[:limit]


def clean_labels(values: pd.Series) -> tuple:
    """Fold spellings that differ only by case or padding onto the commonest one.

    A bar chart showing Kano, KANO and "kano " as three states is the thing that
    makes an auto-generated chart look broken — and the split understates the
    real total, so it is wrong as well as ugly. The map already merged them; the
    bars did not, and the two disagreeing on the same page is worse than either.

    Returns (cleaned series, how many spellings were merged) so the caller can
    say what it did instead of quietly changing the data.
    """
    raw = values.astype(str).str.strip()
    key = raw.str.lower()
    if key.nunique() == raw.nunique():
        return raw, 0
    # The commonest original spelling wins, so "Kano" beats "KANO" on merit.
    best = (pd.DataFrame({"k": key, "r": raw}).groupby(["k", "r"]).size()
            .reset_index(name="n").sort_values("n", ascending=False)
            .drop_duplicates("k").set_index("k")["r"])
    return key.map(best).fillna(raw), int(raw.nunique() - key.nunique())


def spread(frame: pd.DataFrame, col: str) -> float:
    """Relative spread of a breakdown: (max - min) / |mean|.

    Used to decide whether a breakdown says anything. Average age by state came
    out 34, 34, 35, 34, 34 — a chart that costs a slot on the page and teaches
    the reader that this section is noise.
    """
    v = pd.to_numeric(frame[col], errors="coerce").dropna()
    if len(v) < 2 or not float(abs(v.mean())):
        return 0.0
    return float((v.max() - v.min()) / abs(v.mean()))


SEQ_WORDS = ("week", "wk", "month", "wave", "round", "quarter", "q", "phase",
             "term", "period", "cycle", "year", "day", "visit", "batch")


def ordinal_order(values) -> list:
    """Put Week 1, Week 2 … Week 11 in order, or return [] if they are not a run.

    A weekly tracker keeps its time axis in the sheet names, so after combining
    there is a column of "Week 1".."Week 11" and no date anywhere in the file.
    Sorted as text that reads Week 1, Week 10, Week 11, Week 2 — and plotted as
    a bar chart sorted by size it stops being a time axis at all. Recognising
    the run is what turns it back into a trend.
    """
    labels = [str(v).strip() for v in pd.unique(pd.Series(values).dropna())]
    if len(labels) < 3:
        return []
    parsed, stray = [], 0
    for lab in labels:
        m = re.match(r"^\s*([A-Za-z]{1,10})?\s*[-_ ]?(\d{1,4})(?:\.0)?\s*$", lab)
        word = (m.group(1) or "").lower() if m else ""
        if not m or (word and word not in SEQ_WORDS):
            stray += 1
            continue
        parsed.append((int(m.group(2)), lab))
    # A minority of labels outside the run is normal and must not cancel it: the
    # campaign tracker has Week 1..11 and a "Videos" sheet, and requiring every
    # label to fit meant eleven weeks of a campaign had no time axis because of
    # one extra tab. The strays keep their bars; they are just not on the line.
    if len(parsed) < 3 or len(parsed) < 0.6 * len(labels):
        return []
    parsed.sort()
    return [lab for _, lab in parsed]


LABEL_WORDS = {"headline", "title", "name", "label", "article", "story",
               "question", "item", "video", "episode", "post", "page",
               "facility", "product", "topic", "subject"}


def label_column(prof: pf.Profile) -> str | None:
    """The column that names the *thing* each row is about.

    Page titles, article headlines, video titles, facility names. Too many
    distinct values to be a dimension — 9,295 page titles will never be a bar
    chart of categories — but they are exactly what a "top 15 by views" chart
    ranks, and without one a GA4 export of 9,295 rows and 127,836 views produced
    no charts whatsoever.

    Prefers the first column of the sheet when it qualifies, because that is
    where a person puts the name of the thing.
    """
    best, best_score = None, -1.0
    for i, (name, c) in enumerate(prof.columns.items()):
        if c.role not in (pf.IDENTIFIER, pf.TEXT, pf.CATEGORICAL, pf.GEO):
            continue
        if pf.is_system(name) or c.filled < max(3, 0.3 * c.n):
            continue
        vals = prof.df[name].dropna().astype(str)
        if vals.empty:
            continue
        avg_len = float(vals.str.len().mean())
        # A label is a name, not a URL and not a paragraph.
        if avg_len > 120 or vals.str.startswith(("http://", "https://")).mean() > 0.5:
            continue
        distinct = c.unique / max(c.filled, 1)
        if distinct < 0.3 and c.unique <= 25:  # that is a dimension, not a label
            continue
        # A column called headline or title beats one called unique_id, even
        # when both are near-unique and early in the sheet: "top 15 articles"
        # should list headlines, not URL slugs.
        named = bool(set(re.split(r"[^a-z]+", str(name).lower())) & LABEL_WORDS)
        score = (distinct + (0.4 if i < 3 else 0) - min(avg_len, 100) / 400
                 + (1.0 if named else 0))
        if score > best_score:
            best, best_score = name, score
    return best


def rank_items(df: pd.DataFrame, m: Measure, label: str, prof: pf.Profile,
               top: int = 15, worst: bool = False) -> pd.DataFrame:
    """The top (or bottom) rows by a measure, one row per label."""
    d = pd.DataFrame({"Item": df[label].astype(str).str.strip(),
                      "_v": apply(df, m, prof)}).dropna()
    if d.empty:
        return pd.DataFrame()
    g = d.groupby("Item", observed=True)["_v"].agg([m.pagg, "size"])
    g.columns = [m.verb, "Records"]
    g[m.verb] = (g[m.verb] * m.scale).round(2)
    g = g.sort_values(m.verb, ascending=bool(worst)).reset_index()
    return g.head(top)


def _series(df: pd.DataFrame, m: Measure) -> pd.Series:
    """The measure as a clean numeric column, sentinels removed."""
    if m.col == RECORDS.col:
        return pd.Series(np.ones(len(df)), index=df.index)
    if m.agg == "share":
        low = df[m.col].astype(str).str.strip().str.lower()
        return low.map(lambda v: 1.0 if v in YES else (0.0 if v in NO else np.nan))
    return pd.to_numeric(df[m.col], errors="coerce")


def apply(df: pd.DataFrame, m: Measure, prof: pf.Profile) -> pd.Series:
    s = _series(df, m)
    if m.col != RECORDS.col:
        hits = prof.columns[m.col].sentinel_hits
        if hits:
            s = s.where(~s.isin(list(hits)))
    return s


def total(df: pd.DataFrame, m: Measure, prof: pf.Profile) -> float:
    s = apply(df, m, prof).dropna()
    if s.empty:
        return float("nan")
    return float(s.sum() if m.agg == "sum" else s.mean()) * m.scale


def by_dimension(df: pd.DataFrame, m: Measure, dim: str,
                 prof: pf.Profile, top: int = 12) -> pd.DataFrame:
    """One row per dimension value: the measure, and how many records it rests on.

    The record count travels with every aggregate because a mean over eleven
    responses and a mean over eleven hundred look identical on a bar chart and
    mean entirely different things.
    """
    labels, merged = clean_labels(df[dim])
    d = pd.DataFrame({"Category": labels,
                      "_v": apply(df, m, prof)}).dropna(subset=["_v"])
    if d.empty:
        return pd.DataFrame(columns=["Category", m.verb, "Records"])
    g = d.groupby("Category", observed=True)["_v"].agg([m.pagg, "size"])
    g.columns = [m.verb, "Records"]
    g[m.verb] = g[m.verb] * m.scale
    g = g.sort_values(m.verb, ascending=False).reset_index()
    g[m.verb] = g[m.verb].round(2)
    g.attrs["merged"] = merged
    return g.head(top)


# ═══ time ═══════════════════════════════════════════════════════════════════
GRAINS = {"min": ("minute", "min"), "h": ("hour", "h"),
          "D": ("day", "D"), "W": ("week", "W"), "MS": ("month", "M")}


def grain_for(span_days: int, span_seconds: int | None = None) -> str:
    """The period to bucket by. Sub-day spans get sub-day buckets.

    Passing only span_days keeps the old behaviour, so callers that never see
    intraday data are unaffected.
    """
    if span_days < 1 and span_seconds is not None:
        if span_seconds <= 4 * 3600:
            return "min"
        return "h"
    if span_days <= 3:
        return "h"
    if span_days <= 62:
        return "D"
    return "W" if span_days <= 400 else "MS"


def over_time(df: pd.DataFrame, m: Measure, date: str, prof: pf.Profile,
              freq: str | None = None) -> tuple:
    """(frame, grain label). The measure aggregated per period."""
    s = pd.to_datetime(df[date], errors="coerce")
    # Kobo and GA4 exports carry a UTC offset. Bucketing a tz-aware column
    # dropped the zone with a warning nobody saw; now that an hourly grain
    # exists, that decided whether "14:00" meant Lagos or UTC. Keep the wall
    # clock the export was written in, which is the one the reader recognises.
    if getattr(s.dtype, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    d = pd.DataFrame({"_d": s, "_v": apply(df, m, prof)}).dropna(subset=["_d"])
    d = d.dropna(subset=["_v"])
    if d.empty:
        return pd.DataFrame(columns=["Period", m.verb]), "period"
    gap = d["_d"].max() - d["_d"].min()
    freq = freq or grain_for(gap.days, int(gap.total_seconds()))
    label, code = GRAINS[freq]
    d["Period"] = d["_d"].dt.to_period(code).dt.to_timestamp()
    g = d.groupby("Period")["_v"].agg(m.pagg).reset_index()
    g.columns = ["Period", m.verb]
    g[m.verb] = g[m.verb] * m.scale
    g[m.verb] = g[m.verb].round(2)
    return g, label


def compare_periods(df: pd.DataFrame, m: Measure, date: str, prof: pf.Profile
                    ) -> dict:
    """Latest complete window against the one before it, equal length.

    Returns now/then/pct plus the window labels, or an empty dict when the data
    does not span two windows — in which case the UI must show no delta rather
    than a made-up one.
    """
    s = pd.to_datetime(df[date], errors="coerce")
    d = pd.DataFrame({"_d": s, "_v": apply(df, m, prof)}).dropna()
    if d.empty:
        return {}
    end = d["_d"].max()
    span = (end - d["_d"].min()).days
    # Window length: a month for long series, a week for short ones. Enough to
    # be stable, short enough to be current.
    win = 7 if span <= 120 else 30 if span <= 800 else 90
    if span < win * 2:
        return {}
    a_start, a_end = end - pd.Timedelta(days=win), end
    b_start, b_end = end - pd.Timedelta(days=win * 2), a_start
    cur = d[(d["_d"] > a_start) & (d["_d"] <= a_end)]["_v"]
    prev = d[(d["_d"] > b_start) & (d["_d"] <= b_end)]["_v"]
    if cur.empty or prev.empty:
        return {}
    now = float(cur.sum() if m.agg == "sum" else cur.mean()) * m.scale
    then = float(prev.sum() if m.agg == "sum" else prev.mean()) * m.scale
    return {"now": now, "then": then, "window": win,
            "label": f"previous {win} days",
            "detail": (f"{a_start:%d %b} – {a_end:%d %b} against "
                       f"{b_start:%d %b} – {b_start + pd.Timedelta(days=win):%d %b}")}


def change_by_dimension(df: pd.DataFrame, m: Measure, dim: str, date: str,
                        prof: pf.Profile, top: int = 10) -> pd.DataFrame:
    """Which dimension values drove the change between the last two windows.

    This is the question a dashboard exists to answer — the total moved, so
    where did it move? A breakdown of the level tells you who is biggest; a
    breakdown of the *change* tells you what happened.
    """
    s = pd.to_datetime(df[date], errors="coerce")
    labels, _ = clean_labels(df[dim])
    d = pd.DataFrame({"_d": s, "_g": labels,
                      "_v": apply(df, m, prof)}).dropna()
    if d.empty:
        return pd.DataFrame()
    end = d["_d"].max()
    span = (end - d["_d"].min()).days
    win = 7 if span <= 120 else 30 if span <= 800 else 90
    if span < win * 2:
        return pd.DataFrame()
    cur = d[d["_d"] > end - pd.Timedelta(days=win)]
    prev = d[(d["_d"] > end - pd.Timedelta(days=win * 2))
             & (d["_d"] <= end - pd.Timedelta(days=win))]
    a = cur.groupby("_g")["_v"].agg(m.pagg) * m.scale
    b = prev.groupby("_g")["_v"].agg(m.pagg) * m.scale
    out = pd.concat([b.rename("Before"), a.rename("After")], axis=1).fillna(0)
    out["Change"] = (out["After"] - out["Before"]).round(2)
    out = out.reset_index().rename(columns={"_g": "Category"})
    out["Records"] = out["Category"].map(cur.groupby("_g").size()).fillna(0).astype(int)
    out = out.reindex(out["Change"].abs().sort_values(ascending=False).index)
    return out.head(top).reset_index(drop=True)
