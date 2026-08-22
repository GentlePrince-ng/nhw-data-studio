"""Turn a Profile into ranked, plain-English findings.

This is the part that makes the tool worth opening: anyone can draw a bar chart
of a column, but the question an analyst actually arrives with is *what should I
be looking at*. Every finding here is computed from the data — nothing is
generated text — so the same upload always produces the same set, and each one
carries the numbers a reader would ask for next.

Ranking is by `weight`, and the weights are deliberate: a finding that would
change how you interpret every other number (a fifth of rows duplicated, a
sentinel code inflating a mean, a state missing entirely) outranks a finding
that is merely interesting (two columns correlate). Analysts read the top five
and stop, so the order is the product.

Cost control matters because the promise is "under five minutes". Pairwise tests
are capped at a bounded number of columns rather than run across everything, so
a 60-column sheet does not become 1,770 chi-square tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import profiling as pf

# Kinds double as filter chips on the page.
SHAPE = "Shape"
QUALITY = "Data quality"
CONCENTRATION = "Concentration"
TREND = "Trend"
RELATIONSHIP = "Relationship"
DISTRIBUTION = "Distribution"
COVERAGE = "Coverage"

FLAG, WATCH, INFO = "flag", "watch", "info"

MAX_PAIR_COLS = 12       # ceiling on columns entering pairwise tests
MIN_GROUP = 20           # a group smaller than this is not worth comparing


@dataclass
class Finding:
    text: str
    kind: str
    severity: str = INFO
    weight: float = 1.0
    cols: list = field(default_factory=list)
    table: object = None       # optional supporting frame, shown on demand


def _fmt(x, dp: int = 1) -> str:
    """Numbers the way a report writes them, not the way numpy prints them."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    x = float(x)
    if abs(x - round(x)) < 1e-9 and abs(x) < 1e15:
        return f"{int(round(x)):,}"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:,.{dp}f}"


def _pct(part: float, whole: float, dp: int = 0) -> str:
    return "—" if not whole else f"{100.0 * part / whole:.{dp}f}%"


def _day(ts) -> str:
    """A date as a person writes it. Never "%-d": that directive is a glibc
    extension and raises on Windows, so it would pass on Streamlit Cloud and
    fail on the laptop the analysis is actually run from."""
    try:
        if ts is None or pd.isna(ts):
            return "—"
        return pd.Timestamp(ts).strftime("%d %b %Y").lstrip("0")
    except (ValueError, TypeError):
        return str(ts)


# ═══ individual rule families ═══════════════════════════════════════════════
def _shape(p: pf.Profile) -> list:
    out = []
    usable = len([c for c in p.columns.values() if c.role in pf.ANALYSABLE])
    txt = (f"{_fmt(p.rows)} records and {len(p.columns)} columns, "
           f"of which {usable} carry analysable values.")
    if p.dropped:
        txt += f" {len(p.dropped)} column(s) are empty or hold one repeated value."
    out.append(Finding(txt, SHAPE, INFO, weight=9.0))

    date = p.best_date()
    if date:
        st = p.columns[date].stats
        span = st.get("span_days", 0)
        out.append(Finding(
            f"Records span {_day(st['start'])} to {_day(st['end'])} "
            f"({_fmt(span)} days) on {p.label(date)}.",
            SHAPE, INFO, weight=8.5, cols=[date]))
    if p.truncated:
        out.append(Finding(
            f"Only the first {_fmt(p.rows)} rows were read; {_fmt(p.truncated)} "
            "further rows in the file were skipped by the row cap.",
            SHAPE, FLAG, weight=9.5))
    return out


def _quality(p: pf.Profile) -> list:
    """Missingness, duplication, and the label-hygiene problems that skew counts."""
    out = []
    df = p.df

    sparse = sorted((c for c in p.by_role(*pf.ANALYSABLE) if c.missing_pct >= 40),
                    key=lambda c: -c.missing_pct)
    if sparse:
        worst = sparse[0]
        out.append(Finding(
            f"{len(sparse)} column(s) are at least 40% empty — worst is "
            f"{worst.label} at {worst.missing_pct:.0f}% missing "
            f"({_fmt(worst.filled)} of {_fmt(worst.n)} filled). Percentages on "
            "these describe who answered, not the whole sample.",
            QUALITY, FLAG if len(sparse) > 3 else WATCH, weight=7.6,
            cols=[c.name for c in sparse[:8]],
            table=pd.DataFrame([{"Column": c.label, "Filled": c.filled,
                                 "Missing %": round(c.missing_pct, 1)}
                                for c in sparse])))

    dups = int(df.duplicated().sum())
    if dups:
        out.append(Finding(
            f"{_fmt(dups)} rows ({_pct(dups, len(df), 1)}) are exact duplicates of "
            "another row across every column.",
            QUALITY, FLAG if dups / max(len(df), 1) > 0.02 else WATCH,
            weight=8.2))

    for c in p.by_role(pf.IDENTIFIER):
        if c.filled < 20:
            continue
        d = int(df[c.name].dropna().duplicated().sum())
        if d:
            out.append(Finding(
                f"{c.label} looks like an identifier but repeats: {_fmt(d)} "
                f"duplicate value(s) across {_fmt(c.filled)} filled rows.",
                QUALITY, FLAG, weight=8.4, cols=[c.name]))
            break

    for c in p.by_role(pf.NUMERIC):
        if not c.sentinel_hits:
            continue
        codes = ", ".join(str(k) for k in sorted(c.sentinel_hits))
        hits = sum(c.sentinel_hits.values())
        out.append(Finding(
            f"{c.label} contains {_fmt(hits)} values coded {codes} "
            f"({_pct(hits, c.filled)} of answers) — read as "
            "don't-know/refused and excluded from the mean shown here. A "
            "spreadsheet average over this column would be wrong.",
            QUALITY, FLAG, weight=8.8, cols=[c.name]))

    # Label hygiene: "Kano", "kano ", "KANO" count as three states otherwise.
    for c in p.by_role(pf.CATEGORICAL, pf.GEO):
        raw = df[c.name].dropna().astype(str)
        if raw.empty:
            continue
        norm = raw.str.strip().str.lower()
        if norm.nunique() < raw.nunique():
            collapsed = raw.nunique() - norm.nunique()
            examples = (pd.DataFrame({"raw": raw, "norm": norm})
                        .groupby("norm")["raw"].unique())
            examples = [v for v in examples if len(v) > 1][:3]
            shown = "; ".join(" / ".join(map(str, v[:3])) for v in examples)
            out.append(Finding(
                f"{c.label} has {collapsed} value(s) that differ only by case or "
                f"spacing — {shown}. These split one category into several and "
                "understate its count.",
                QUALITY, FLAG, weight=8.0, cols=[c.name]))
    return out


def _concentration(p: pf.Profile) -> list:
    """Where the records actually sit — the first thing anyone asks."""
    out = []
    for c in p.headline_cats(limit=5):
        vc = p.df[c.name].value_counts()
        if vc.empty:
            continue
        top, top_n = str(vc.index[0]), int(vc.iloc[0])
        share = top_n / max(c.filled, 1)
        text = (f"{top} is the largest {c.label.lower()} group with "
                f"{_fmt(top_n)} records ({_pct(top_n, c.filled)} of those "
                f"answering)")
        if len(vc) > 1:
            second_n = int(vc.iloc[1])
            ratio = top_n / max(second_n, 1)
            if ratio >= 1.5:
                text += (f", {ratio:.1f}x the next group "
                         f"({vc.index[1]}, {_fmt(second_n)})")
        out.append(Finding(
            text + ".", CONCENTRATION,
            WATCH if share > 0.6 else INFO,
            weight=7.0 + min(share, 0.9),
            cols=[c.name],
            table=vc.rename("Records").rename_axis(c.label).reset_index()))

        tail = int((vc < max(3, 0.005 * c.filled)).sum())
        if tail >= 3 and len(vc) > 6:
            out.append(Finding(
                f"{c.label} has a long tail: {tail} of {len(vc)} categories hold "
                "under 0.5% of records each. Group them before charting.",
                CONCENTRATION, INFO, weight=4.0, cols=[c.name]))
    return out


def _trend(p: pf.Profile) -> list:
    """Direction over time, on whichever date column is most usable."""
    out = []
    date = p.best_date()
    if not date:
        return out
    s = pd.to_datetime(p.df[date], errors="coerce").dropna()
    # Same as metrics.over_time: bucketing a tz-aware column drops the zone
    # with a warning nobody reads. Keep the wall clock the export was written
    # in, explicitly.
    if getattr(s.dtype, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    if len(s) < 12:
        return out
    span = (s.max() - s.min()).days
    freq, unit = ("W", "week") if span <= 400 else ("MS", "month")
    counts = s.dt.to_period("W" if freq == "W" else "M").value_counts().sort_index()
    counts = counts[counts.index.notna()]
    if len(counts) < 3:
        return out

    peak_i = counts.idxmax()
    out.append(Finding(
        f"Activity peaked in the {unit} of {_day(peak_i.start_time)} with "
        f"{_fmt(int(counts.max()))} records; the median {unit} carries "
        f"{_fmt(float(counts.median()))}.",
        TREND, INFO, weight=6.4, cols=[date],
        table=counts.rename("Records").rename_axis(unit.title()).reset_index()))

    # Compare the last three complete periods with the three before them; the
    # final period is dropped because a part-week always reads as a collapse.
    body = counts.iloc[:-1] if len(counts) >= 5 else counts
    if len(body) >= 4:
        k = max(2, min(3, len(body) // 2))
        recent, prior = body.iloc[-k:].mean(), body.iloc[-2 * k:-k].mean()
        if prior > 0:
            chg = 100.0 * (recent - prior) / prior
            if abs(chg) >= 15:
                word = "up" if chg > 0 else "down"
                out.append(Finding(
                    f"The last {k} complete {unit}s average "
                    f"{_fmt(recent)} records — {word} {abs(chg):.0f}% on the "
                    f"{k} {unit}s before ({_fmt(prior)}).",
                    TREND, WATCH if chg <= -30 else INFO,
                    weight=7.2 + min(abs(chg) / 100, 0.8), cols=[date]))

    gaps = counts.index.to_timestamp().to_series().diff().dt.days.dropna()
    step = 7 if freq == "W" else 28
    if (gaps > step * 2.5).any():
        n_gaps = int((gaps > step * 2.5).sum())
        out.append(Finding(
            f"Collection is not continuous: {n_gaps} gap(s) of more than two "
            f"{unit}s with no records at all. Treat any per-{unit} average as "
            "covering active periods only.",
            TREND, WATCH, weight=6.0, cols=[date]))
    return out


def _distribution(p: pf.Profile) -> list:
    out = []
    for c in p.headline_nums(limit=6):
        st = c.stats
        if not st:
            continue
        mean, med = st.get("mean", 0), st.get("median", 0)
        bits = [f"{c.label} averages {_fmt(mean)} (median {_fmt(med)}, "
                f"range {_fmt(st.get('min'))} to {_fmt(st.get('max'))})"]
        sev, w = INFO, 5.0
        if med and abs(mean - med) / max(abs(med), 1e-9) > 0.5:
            bits.append("the mean sits well above the median, so a few large "
                        "values are pulling it — quote the median")
            sev, w = WATCH, 6.2
        if st.get("outliers"):
            bits.append(f"{_fmt(st['outliers'])} values fall outside the usual "
                        f"range (beyond {_fmt(st['out_hi'])})")
            w = max(w, 6.0)
        if st.get("negatives"):
            bits.append(f"{_fmt(st['negatives'])} negative value(s)")
            sev, w = FLAG, 7.4
        if st.get("zeros", 0) / max(st.get("n", 1), 1) > 0.4:
            bits.append(f"{_pct(st['zeros'], st['n'])} of values are zero")
            sev = WATCH if sev == INFO else sev
        out.append(Finding("; ".join(bits) + ".", DISTRIBUTION, sev,
                           weight=w, cols=[c.name]))
    return out


def _cramers_v(ct: pd.DataFrame) -> float:
    """Bias-corrected Cramér's V. Uncorrected V inflates badly on sparse tables."""
    from scipy.stats import chi2_contingency
    if ct.shape[0] < 2 or ct.shape[1] < 2 or ct.to_numpy().sum() == 0:
        return 0.0
    chi2, p_val, _, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
    rc = r - (r - 1) ** 2 / max(n - 1, 1)
    kc = k - (k - 1) ** 2 / max(n - 1, 1)
    denom = min(kc - 1, rc - 1)
    v = float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0
    return v if p_val < 0.05 else 0.0


def _relationships(p: pf.Profile) -> list:
    """Pairs worth a second look: numeric correlations and category associations."""
    out = []
    df = p.df

    nums = [c.name for c in p.headline_nums(limit=MAX_PAIR_COLS)]
    if len(nums) >= 2:
        sub = df[nums].apply(pd.to_numeric, errors="coerce")
        for c in nums:                                    # sentinels would fake it
            hits = p.columns[c].sentinel_hits
            if hits:
                sub[c] = sub[c].where(~sub[c].isin(list(hits)))
        corr = sub.corr(numeric_only=True)
        seen = set()
        pairs = []
        for a in corr.columns:
            for b in corr.columns:
                if a == b or (b, a) in seen:
                    continue
                seen.add((a, b))
                r = corr.loc[a, b]
                if pd.notna(r) and abs(r) >= 0.5:
                    pairs.append((abs(r), r, a, b))
        for _, r, a, b in sorted(pairs, reverse=True)[:4]:
            direction = "rises with" if r > 0 else "falls as"
            out.append(Finding(
                f"{p.label(a)} {direction} {p.label(b)} "
                f"(correlation {r:+.2f}). Correlation only — not cause.",
                RELATIONSHIP, INFO, weight=5.5 + abs(r), cols=[a, b]))

    cats = [c for c in p.headline_cats(limit=MAX_PAIR_COLS) if 2 <= c.unique <= 12]
    scored = []
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            ct = pd.crosstab(df[a.name], df[b.name])
            if ct.to_numpy().sum() < MIN_GROUP * 2:
                continue
            v = _cramers_v(ct)
            if v >= 0.25:
                scored.append((v, a, b, ct))
    for v, a, b, ct in sorted(scored, key=lambda t: -t[0])[:3]:
        strength = "strongly" if v >= 0.45 else "moderately"
        out.append(Finding(
            f"{a.label} and {b.label} are {strength} associated "
            f"(Cramér's V {v:.2f}, p<0.05) — the split of {b.label.lower()} "
            f"differs by {a.label.lower()}, so report them together.",
            RELATIONSHIP, INFO, weight=6.0 + v, cols=[a.name, b.name],
            table=ct.reset_index()))

    # A numeric that differs sharply between groups is usually the real story.
    for num in p.headline_nums(limit=3):
        for cat in [c for c in p.headline_cats(limit=4) if 2 <= c.unique <= 10]:
            g = df.groupby(cat.name, observed=True)[num.name].agg(["mean", "count"])
            g = g[g["count"] >= MIN_GROUP].dropna()
            if len(g) < 2:
                continue
            hi, lo = g["mean"].idxmax(), g["mean"].idxmin()
            hv, lv = g["mean"].max(), g["mean"].min()
            if lv and abs(hv - lv) / max(abs(lv), 1e-9) >= 0.4:
                out.append(Finding(
                    f"{num.label} varies by {cat.label.lower()}: "
                    f"{hi} averages {_fmt(hv)} against {lo} at {_fmt(lv)} "
                    f"({(hv / lv if lv else 0):.1f}x, groups of "
                    f"{_fmt(int(g.loc[hi, 'count']))} and "
                    f"{_fmt(int(g.loc[lo, 'count']))}).",
                    RELATIONSHIP, INFO, weight=6.6,
                    cols=[num.name, cat.name],
                    table=g.rename(columns={"mean": f"Mean {num.label}",
                                            "count": "Records"}).reset_index()))
                break
    return out


def _coverage(p: pf.Profile) -> list:
    """Geographic reach — and, for Nigeria data, which states are missing."""
    out = []
    geo_col = p.best_geo()
    if not geo_col:
        return out
    vc = p.df[geo_col].value_counts()
    if vc.empty:
        return out
    present = {str(v).strip().lower() for v in vc.index}
    hits = present & pf.STATES
    if len(hits) >= 5:
        missing = sorted(pf.STATES - present -
                         {"abuja", "federal capital territory"})
        out.append(Finding(
            f"{len(hits)} of Nigeria's 37 states and the FCT appear in {p.label(geo_col)}."
            + (f" Absent: {', '.join(s.title() for s in missing[:8])}"
               + (f" and {len(missing) - 8} more" if len(missing) > 8 else "") + "."
               if missing else " Every state is represented."),
            COVERAGE, WATCH if len(missing) > 10 else INFO, weight=7.8,
            cols=[geo_col]))
    # Values that match no boundary at all. These are the ones that vanish from
    # a map without explanation and get dropped from any geographic total — a
    # single misspelling can quietly remove a whole state from the analysis.
    try:
        from . import geo
        if geo.available("state"):
            known = {geo.canon_state(n) for n in geo.boundary_names("state")}
            bad = sorted({str(v) for v in vc.index
                          if geo.canon_state(v) not in known})
            if bad and len(bad) < len(vc):
                out.append(Finding(
                    f"{len(bad)} value(s) in {p.label(geo_col)} match no Nigerian "
                    f"state or LGA boundary: {', '.join(bad[:6])}"
                    f"{'…' if len(bad) > 6 else ''}. They are excluded from every "
                    "map and from any geographic total.",
                    COVERAGE, FLAG, weight=8.1, cols=[geo_col]))
    except Exception:
        pass

    thin = vc[vc < max(3, 0.002 * p.rows)]
    if len(thin):
        out.append(Finding(
            f"{len(thin)} {p.label(geo_col).lower()} value(s) have fewer than "
            f"{max(3, int(0.002 * p.rows))} records "
            f"({', '.join(map(str, thin.index[:5]))}"
            f"{'...' if len(thin) > 5 else ''}). Any rate computed for these "
            "rests on a handful of rows.",
            COVERAGE, WATCH, weight=7.0, cols=[geo_col],
            table=thin.rename("Records").rename_axis(p.label(geo_col)).reset_index()))
    return out


def _kobo(p: pf.Profile) -> list:
    out = []
    for parent, members in list(p.kobo.items())[:4]:
        counts = pf.multi_counts(p.df, members)
        if counts.empty:
            continue
        top = counts.iloc[0]
        out.append(Finding(
            f"{pf.clean_label(parent)} is a multi-select question with "
            f"{len(members)} options, exported as one column each. Most selected: "
            f"{top['option']} ({_fmt(int(top['n']))} records, "
            f"{_pct(int(top['n']), p.rows)} of all rows). Percentages here sum "
            "past 100 because respondents pick several.",
            SHAPE, INFO, weight=6.8, cols=list(members[:1]),
            table=counts.rename(columns={"option": "Option", "n": "Selected"})))
    return out


# ═══ entry point ════════════════════════════════════════════════════════════
def analyse(p: pf.Profile) -> list:
    """Every rule, ranked. Order is the deliverable — the top five get read."""
    found = []
    for rule in (_shape, _quality, _concentration, _trend, _distribution,
                 _relationships, _coverage, _kobo):
        try:
            found.extend(rule(p))
        except Exception as exc:                  # one bad column must not
            found.append(Finding(                 # cost the whole page
                f"[{rule.__name__.strip('_')} checks could not run on this "
                f"sheet: {type(exc).__name__}]", QUALITY, INFO, weight=0.1))
    sev_rank = {FLAG: 2.0, WATCH: 1.0, INFO: 0.0}
    found.sort(key=lambda f: -(f.weight + sev_rank[f.severity]))
    return found


def headline(p: pf.Profile, found: list) -> str:
    """One paragraph a programme lead can paste into an email."""
    date = p.best_date()
    geo_col = p.best_geo()
    bits = [f"This sheet holds {_fmt(p.rows)} records across "
            f"{len(p.columns)} columns"]
    if date:
        st = p.columns[date].stats
        bits.append(f"collected between {_day(st['start'])} and {_day(st['end'])}")
    if geo_col:
        bits.append(f"covering {p.df[geo_col].nunique()} {p.label(geo_col).lower()} values")
    para = ", ".join(bits) + ". "
    flags = [f for f in found if f.severity == FLAG]
    if flags:
        para += (f"{len(flags)} issue(s) need attention before the numbers are "
                 f"quoted — starting with: {flags[0].text} ")
    lead = next((f for f in found if f.kind in (CONCENTRATION, TREND)), None)
    if lead:
        para += lead.text
    return para.strip()
