"""Nigeria choropleths, and the name-matching that decides whether they are true.

Drawing the map is the easy half. The half that decides whether the map is worth
looking at is joining "KANO ", "Kano State", "kano" and "Akwa-Ibom" onto boundary
records that say "Kano" and "Akwa Ibom" — and, crucially, **saying out loud what
did not match**. A choropleth with six silently unshaded states looks finished; a
reader takes the pale ones for low values rather than for absent data, which is
the single most misleading thing a map can do. So every map here returns its
unmatched values and prints them under the chart.

Two boundary files are vendored in `static/`, built once by
`static/build_boundaries.py` (see that file for provenance and why it is a build
step). The app needs nothing but `json` to read them.

LGA joins are keyed `state|lga` whenever the data has a state column, because
five LGA names belong to two states each. Without a state column the join falls
back to LGA name alone and the ambiguity is reported rather than hidden.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import altair as alt
import pandas as pd

from . import theme

STATIC = Path(__file__).resolve().parents[1] / "static"
STATES_FILE = STATIC / "nigeria_states.geojson"
LGAS_FILE = STATIC / "nigeria_lgas.geojson"

# Written forms that occur in real NHW datasets, mapped to the boundary spelling.
STATE_ALIASES = {
    "fct": "federal capital territory",
    "abuja": "federal capital territory",
    "fctabuja": "federal capital territory",
    "abujafct": "federal capital territory",
    "fedcapitalterritory": "federal capital territory",
    "nassarawa": "nasarawa",
    "nassawara": "nasarawa",
    "akwaibom": "akwa ibom",
    "crossriver": "cross river",
    "crossrivers": "cross river",
    "river": "rivers",
    "niger state": "niger",
    "kadunna": "kaduna",
    "borno state": "borno",
    "zamfarra": "zamfara",
    "bauchii": "bauchi",
    "ondo state": "ondo",
    "lagos state": "lagos",
}


def canon_state(value) -> str:
    """Normalise a written state name to the boundary file's spelling.

    Strips case, padding, a trailing "state", and any punctuation or hyphen, then
    applies the alias table. `Akwa-Ibom State ` and `AKWA IBOM` both come out as
    `akwa ibom`, which is what makes the join work at all.
    """
    s = str(value).strip().lower()
    s = re.sub(r"\bstate\b", " ", s)
    s = re.sub(r"[^a-z ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in STATE_ALIASES:
        return STATE_ALIASES[s]
    squashed = s.replace(" ", "")
    if squashed in STATE_ALIASES:
        return STATE_ALIASES[squashed]
    return s


def canon_lga(value) -> str:
    """Same treatment for LGA names, minus the state-specific aliases.

    LGA spellings vary more than state spellings and there is no authoritative
    alias list, so this only handles the mechanical differences: case, padding,
    punctuation, and the `/` and `-` that Kobo and DHIS2 disagree about.
    """
    s = str(value).strip().lower()
    s = re.sub(r"\b(lga|local government( area)?)\b", " ", s)
    s = re.sub(r"[^a-z ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=2)
def _load(level: str) -> dict:
    """Read a vendored boundary file once per process.

    lru_cache rather than st.cache_data: this is a 120 KB constant on disk, the
    same for every user and every upload, so per-session caching would be the
    wrong shape and would pickle it repeatedly.
    """
    path = STATES_FILE if level == "state" else LGAS_FILE
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def available(level: str = "state") -> bool:
    return bool(_load(level).get("features"))


def boundary_names(level: str = "state") -> list:
    gj = _load(level)
    key = "state" if level == "state" else "lga"
    return sorted({f["properties"][key] for f in gj.get("features", [])})


# ═══ the join ═══════════════════════════════════════════════════════════════
@dataclass
class Join:
    """The result of putting a column of place names onto the boundaries.

    `unmatched`, `unused` and `merged` are the parts a reader needs and a map
    cannot show: values that landed nowhere, boundaries with no data, and
    spellings that were folded together to make the join work at all.
    """
    features: list = field(default_factory=list)
    table: object = None
    unmatched: list = field(default_factory=list)
    unused: list = field(default_factory=list)
    merged: list = field(default_factory=list)

    @property
    def matched(self) -> int:
        return len(self.features) - len(self.unused)


def summarise(df: pd.DataFrame, place_col: str, value_col: str | None = None,
              agg: str = "count", state_col: str | None = None,
              level: str = "state") -> Join:
    """Aggregate a frame to one row per place and join it onto the boundaries."""
    gj = _load(level)
    if not gj.get("features"):
        return Join(table=pd.DataFrame())

    work = pd.DataFrame({"place": df[place_col].astype(str)})
    if level == "lga" and state_col:
        work["state"] = df[state_col].astype(str).map(canon_state)
        work["key"] = work["state"] + "|" + work["place"].map(canon_lga)
    elif level == "lga":
        work["key"] = work["place"].map(canon_lga)
    else:
        work["key"] = work["place"].map(canon_state)
    work = work[work["key"].str.strip().str.len() > 0]

    if value_col and agg != "count":
        work["value"] = pd.to_numeric(df[value_col], errors="coerce")
        grouped = work.dropna(subset=["value"]).groupby("key").agg(
            value=("value", agg), records=("value", "size"))
    else:
        grouped = work.groupby("key").agg(value=("place", "size"))
        grouped["records"] = grouped["value"]
    grouped = grouped.reset_index()
    # The label shown in the tooltip: whatever the data itself called the place.
    raw_by_key = work.groupby("key")["place"].unique()
    grouped["label"] = grouped["key"].map(lambda k: raw_by_key[k][0])
    # Canonicalising is what makes the join possible, and it also quietly fixes
    # a data-entry problem: KANO, Kano and "kano " become one shaded state. The
    # bar chart on the same page still shows them split, so the map has to say
    # that it combined them or the two charts look like they disagree.
    # Compare RAW spellings, not casefolded ones: KANO / Kano / "kano " differ
    # only by case, so a casefolded comparison finds one value and the note
    # never fires — which is exactly the case worth reporting.
    merged = [(k, sorted({str(x) for x in v})) for k, v in raw_by_key.items()
              if len({str(x) for x in v}) > 1]

    lookup = dict(zip(grouped["key"], zip(grouped["value"], grouped["records"])))
    feats, hit_keys, canon_name = [], set(), {}
    for f in gj["features"]:
        props = dict(f["properties"])
        if level == "state":
            bkey = canon_state(props["state"])
        elif state_col:
            bkey = canon_state(props["state"]) + "|" + canon_lga(props["lga"])
        else:
            bkey = canon_lga(props["lga"])
        got = lookup.get(bkey)
        if got:
            hit_keys.add(bkey)
            # Prefer the boundary file's spelling in the table. Where three
            # spellings were combined, printing whichever the data happened to
            # use first ("KANO") beside a map keyed on "Kano" invites the reader
            # to think two different places are being discussed.
            canon_name[bkey] = props.get("lga") or props.get("state")
        props["value"] = float(got[0]) if got else None
        props["records"] = int(got[1]) if got else 0
        props["name"] = props.get("lga") or props.get("state")
        feats.append({"type": "Feature", "geometry": f["geometry"],
                      "properties": props})

    unmatched = sorted(set(grouped["label"][~grouped["key"].isin(hit_keys)]))
    unused = sorted(f["properties"]["name"] for f in feats
                    if f["properties"]["records"] == 0)
    grouped["label"] = [canon_name.get(k, lbl)
                        for k, lbl in zip(grouped["key"], grouped["label"])]
    grouped["value"] = grouped["value"].round(2)
    table = grouped.rename(columns={"label": "Place", "value": "Value",
                                    "records": "Records"})
    # With agg="count" the value *is* the record count, so a Records column
    # beside it would be the same numbers under a second heading.
    keep = ["Place", "Value"] if (agg == "count" or not value_col) \
        else ["Place", "Value", "Records"]
    # Unmatched places stay in the table — dropping them would make the table
    # disagree with the dataset — but they are marked, because a reader adding
    # the column up otherwise gets a total the map does not account for. The
    # column appears only when there is something to mark.
    if unmatched:
        table["On map"] = ["yes" if k in hit_keys else "no"
                           for k in grouped["key"]]
        keep = keep + ["On map"]
    table = table[keep].sort_values("Value", ascending=False).reset_index(drop=True)
    return Join(feats, table, unmatched, unused, merged)


# ═══ the chart ══════════════════════════════════════════════════════════════
# The ramp's palest step (#EAF5EC) is within a hair of any light grey, so a
# state holding 296 records looked the same as a state holding none. Data is
# drawn on the ramp from its second step up, and no-data gets a neutral grey
# that is deliberately not a green at all — the one distinction on this chart
# that has to survive being printed, projected, or photographed off a screen.
DATA_RAMP = theme.RAMP[1:]
NO_DATA_FILL = "#E4E7E5"
NO_DATA_STROKE = "#C9CFCB"


def _fmt_num(v, fmt: str) -> str:
    try:
        return format(float(v), fmt.lstrip(","))if fmt.startswith(",") \
            else format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


NO_DATA_LABEL = "no data"


def class_bands(feats: list, fmt: str, k: int = 5) -> list:
    """Sort the places into k classes and write the class onto each feature.

    A classed choropleth rather than a continuous ramp, for three reasons. The
    reader can actually read a value off a discrete swatch, which they cannot do
    from a gradient. **"no data" becomes a legend entry** instead of a caption
    footnote, which is the difference between a reader knowing a state is
    unsurveyed and assuming it is low. And the class is a plain string, so Vega
    never has to colour a null — the continuous version either dropped those
    polygons entirely or painted them the palest green in the scale, both of
    which say "nearly zero" about data that does not exist.

    Returns the legend domain in order, with `no data` last if it occurs.
    """
    vals = sorted(f["properties"]["value"] for f in feats
                  if f["properties"].get("value") is not None)
    labels: list = []
    if not vals:
        for f in feats:
            f["properties"]["band"] = NO_DATA_LABEL
        return [NO_DATA_LABEL]

    uniq = sorted(set(vals))
    if len(uniq) <= k:
        # Few enough distinct values to label each one exactly.
        for f in feats:
            v = f["properties"].get("value")
            f["properties"]["band"] = (NO_DATA_LABEL if v is None
                                       else _fmt_num(v, fmt))
        labels = [_fmt_num(v, fmt) for v in uniq]
    else:
        cats = pd.qcut(pd.Series(vals), q=min(k, len(uniq)),
                       duplicates="drop")
        edges = list(cats.cat.categories)
        bounds = [edges[0].left] + [iv.right for iv in edges]
        bounds[0] = min(vals)                      # qcut opens below the minimum
        # Counts are integers, so "6 – 19" followed by "19 – 29" leaves the
        # reader unable to place a 19. Whole-number classes start one above the
        # previous class rather than sharing an edge.
        whole = all(float(v).is_integer() for v in vals)
        labels = []
        for i in range(len(bounds) - 1):
            lo = bounds[i] + 1 if (whole and i) else bounds[i]
            labels.append(f"{_fmt_num(lo, fmt)} – {_fmt_num(bounds[i + 1], fmt)}")

        def band_of(v):
            for i in range(len(bounds) - 1):
                if v <= bounds[i + 1] or i == len(bounds) - 2:
                    return labels[i]
            return labels[-1]

        for f in feats:
            v = f["properties"].get("value")
            f["properties"]["band"] = (NO_DATA_LABEL if v is None
                                       else band_of(float(v)))
    if any(f["properties"]["band"] == NO_DATA_LABEL for f in feats):
        labels = labels + [NO_DATA_LABEL]
    return labels


def choropleth(feats: list, title: str, value_title: str,
               fmt: str = ",.0f", height: int = 400, outline: bool = False):
    """A shaded map of Nigeria on the brand sequential ramp.

    Values live in each feature's own properties rather than being joined in
    Vega with `transform_lookup`. That is deliberate: the inline-data path
    renders identically in the browser and in vl-convert, so the map in the
    downloaded HTML report is the map that was on screen. A lookup transform
    against a second inline dataset does not survive the SVG conversion as
    reliably.

    Places with no data are drawn in a neutral grey with the boundary still
    visible, never left white — a missing state must read as missing, not as
    zero. The count of them is printed in the caption by the caller.

    `outline` overlays the state borders, which the LGA map needs: 774 small
    polygons with no state lines is a mosaic nobody can locate a place in.
    """
    # A FeatureCollection with format property="features" — not a bare list of
    # features. The bare list renders in the browser but throws inside
    # vl-convert ("Cannot convert undefined or null to object"), so the map
    # would have been silently missing from every downloaded report.
    domain = class_bands(feats, fmt)
    n_data = len([d for d in domain if d != NO_DATA_LABEL])
    ramp = theme.RAMP[1:] if n_data >= 4 else theme.RAMP[2:]
    colours = [ramp[min(int(i * len(ramp) / max(n_data, 1)), len(ramp) - 1)]
               for i in range(n_data)]
    if NO_DATA_LABEL in domain:
        colours = colours + [NO_DATA_FILL]

    data = alt.InlineData(values={"type": "FeatureCollection", "features": feats},
                          format=alt.DataFormat(property="features"))
    layers = [alt.Chart(data).mark_geoshape(
        stroke=theme.SURFACE, strokeWidth=0.6,
    ).encode(
        # A nominal class, not a quantitative value. Vega then never has to
        # colour a null: "no data" is just another category with its own swatch.
        color=alt.Color("properties.band:N", title=value_title,
                        scale=alt.Scale(domain=domain, range=colours),
                        legend=alt.Legend(orient="right", direction="vertical",
                                          symbolType="square", symbolSize=110,
                                          labelLimit=180)),
        tooltip=[alt.Tooltip("properties.name:N", title=""),
                 alt.Tooltip("properties.value:Q", title=value_title, format=fmt),
                 alt.Tooltip("properties.records:Q", title="records", format=",")],
    )]
    if outline and _load("state").get("features"):
        layers.append(alt.Chart(alt.InlineData(
            values=_load("state"), format=alt.DataFormat(property="features"))
        ).mark_geoshape(fill=None, stroke=theme.INK2, strokeWidth=0.7,
                        strokeOpacity=0.55))
    # An explicit width matched to Nigeria's aspect ratio (~1.3:1 in mercator at
    # this latitude). Without one the chart falls back to the theme's 400px
    # default, the projection fits to *that* rather than to the height, and the
    # map comes out two-thirds the size it was asked for.
    return alt.layer(*layers).properties(
        width=round(height * 1.3), height=height,
        title=title).project(type="mercator")
