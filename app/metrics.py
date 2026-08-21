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
             "budget", "spend", "revenue", "population", "beneficiaries")
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
    label: str
    agg: str                      # "sum" or "mean"
    why: str                      # how the aggregation was chosen, for the UI
    integerish: bool = False

    @property
    def verb(self) -> str:
        return "Total" if self.agg == "sum" else "Average"

    def title(self) -> str:
        return f"{self.verb} {self.label.lower()}"

    def fmt(self) -> str:
        return ",.0f" if (self.agg == "sum" or self.integerish) else ",.1f"


RECORDS = Measure("__records__", "Records", "sum",
                  "one per row in the file", integerish=True)


def _agg_for(name: str, stats: dict) -> tuple[str, str]:
    """(aggregation, why). Name first, then the shape of the values."""
    low = re.sub(r"[^a-z ]+", " ", str(name).lower())
    tokens = set(low.split())
    if tokens & set(MEAN_WORDS) or any(w in low for w in MEAN_WORDS):
        return "mean", "averaged: the name reads as a rate, level or duration"
    if tokens & set(SUM_WORDS) or any(w in low for w in SUM_WORDS):
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
        out.append(Measure(c.name, c.label, agg, why,
                           bool(c.stats.get("integerish"))))
        if len(out) > limit:
            break
    return out


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


def dimensions(prof: pf.Profile, limit: int = 6) -> list:
    """Columns worth slicing by — few enough levels to compare at a glance."""
    dims = [c for c in prof.headline_cats(limit=limit * 3)
            if _sliceable(prof, c)]
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


def _series(df: pd.DataFrame, m: Measure) -> pd.Series:
    """The measure as a clean numeric column, sentinels removed."""
    if m.col == RECORDS.col:
        return pd.Series(np.ones(len(df)), index=df.index)
    s = pd.to_numeric(df[m.col], errors="coerce")
    return s


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
    return float(s.sum() if m.agg == "sum" else s.mean())


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
    g = d.groupby("Category", observed=True)["_v"].agg([m.agg, "size"])
    g.columns = [m.verb, "Records"]
    g = g.sort_values(m.verb, ascending=False).reset_index()
    g[m.verb] = g[m.verb].round(2)
    g.attrs["merged"] = merged
    return g.head(top)


# ═══ time ═══════════════════════════════════════════════════════════════════
GRAINS = {"D": ("day", "D"), "W": ("week", "W"), "MS": ("month", "M")}


def grain_for(span_days: int) -> str:
    if span_days <= 62:
        return "D"
    return "W" if span_days <= 400 else "MS"


def over_time(df: pd.DataFrame, m: Measure, date: str, prof: pf.Profile,
              freq: str | None = None) -> tuple:
    """(frame, grain label). The measure aggregated per period."""
    s = pd.to_datetime(df[date], errors="coerce")
    d = pd.DataFrame({"_d": s, "_v": apply(df, m, prof)}).dropna(subset=["_d"])
    d = d.dropna(subset=["_v"])
    if d.empty:
        return pd.DataFrame(columns=["Period", m.verb]), "period"
    span = (d["_d"].max() - d["_d"].min()).days
    freq = freq or grain_for(span)
    label, code = GRAINS[freq]
    d["Period"] = d["_d"].dt.to_period(code).dt.to_timestamp()
    g = d.groupby("Period")["_v"].agg(m.agg).reset_index()
    g.columns = ["Period", m.verb]
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
    now = float(cur.sum() if m.agg == "sum" else cur.mean())
    then = float(prev.sum() if m.agg == "sum" else prev.mean())
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
    a = cur.groupby("_g")["_v"].agg(m.agg)
    b = prev.groupby("_g")["_v"].agg(m.agg)
    out = pd.concat([b.rename("Before"), a.rename("After")], axis=1).fillna(0)
    out["Change"] = (out["After"] - out["Before"]).round(2)
    out = out.reset_index().rename(columns={"_g": "Category"})
    out["Records"] = out["Category"].map(cur.groupby("_g").size()).fillna(0).astype(int)
    out = out.reindex(out["Change"].abs().sort_values(ascending=False).index)
    return out.head(top).reset_index(drop=True)
