"""Read any spreadsheet and work out what every column actually *is*.

Everything downstream — which charts get drawn, which insights fire, what the
health page flags — reads the column roles this module assigns. Nothing else in
the app is allowed to guess a column's type.

Two things here are specific to how NHW data arrives:

1. **Kobo select_multiple groups.** Kobo exports one binary column per option,
   named `parent/option`. Left alone that reads as 40 unrelated yes/no columns
   and the dashboard drowns. `kobo_groups()` folds them back into one logical
   question, which is what the GLIDE and ICL forms needed by hand every time.
2. **Sentinel numbers.** 98/99/999/9999 standing in for "don't know / refused"
   is normal in the surveys NHW runs, and a mean that swallows them is wrong
   rather than merely noisy. They are detected, reported, and excluded from
   descriptives.
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ── roles ───────────────────────────────────────────────────────────────────
NUMERIC = "numeric"
DATETIME = "datetime"
CATEGORICAL = "categorical"
BINARY = "binary"
TEXT = "text"
IDENTIFIER = "identifier"
GEO = "geo"
CONSTANT = "constant"
EMPTY = "empty"

ANALYSABLE = (NUMERIC, DATETIME, CATEGORICAL, BINARY, GEO)

MAX_CAT_LEVELS = 40          # above this, a low-signal object column is text/id
SENTINELS = {98, 99, 998, 999, 9998, 9999, -1, -99, -999, 8888, 7777}
NA_STRINGS = ["", " ", "NA", "N/A", "n/a", "na", "None", "none", "NULL", "null",
              "-", "--", "nan", "NaN", "#N/A", "#VALUE!", "#DIV/0!", "."]

STATES = {
    "abia", "adamawa", "akwa ibom", "anambra", "bauchi", "bayelsa", "benue",
    "borno", "cross river", "delta", "ebonyi", "edo", "ekiti", "enugu", "fct",
    "abuja", "federal capital territory", "gombe", "imo", "jigawa", "kaduna",
    "kano", "katsina", "kebbi", "kogi", "kwara", "lagos", "nasarawa", "niger",
    "ogun", "ondo", "osun", "oyo", "plateau", "rivers", "sokoto", "taraba",
    "yobe", "zamfara",
}
GEO_WORDS = ("state", "lga", "ward", "region", "zone", "district", "province",
             "country", "facility", "community", "location", "senatorial")
ID_WORDS = ("_id", "id", "uuid", "index", "submission", "instance", "serial",
            "code", "phone", "msisdn", "email", "nin", "hash", "_uri")
YES_NO = {"yes", "no", "y", "n", "true", "false", "1", "0", "1.0", "0.0",
          "oui", "non"}


SYSTEM_COLS = ("_id", "_uuid", "_index", "_status", "_submitted_by", "_tags",
               "_notes", "_validation_status", "_submission_time", "__version__",
               "deviceid", "username", "start", "end", "today", "instanceid",
               "meta", "audit", "formhub", "simserial", "subscriberid")


def is_system(name) -> bool:
    """Kobo/ODK plumbing: real columns, but never the subject of a chart."""
    low = str(name).strip().lower()
    return low in SYSTEM_COLS or low.startswith(("_geo", "meta/", "formhub/"))


COORD_WORDS = ("latitude", "longitude", "lat", "lon", "lng", "altitude",
               "precision", "geopoint", "geo point", "gps", "accuracy",
               "easting", "northing", "elevation")


def _is_coordinate(name) -> bool:
    """A GPS field, not a measurement. Never a measure, never a correlation."""
    low = re.sub(r"[^a-z ]+", " ", str(name).lower())
    tokens = set(low.split())
    return bool(tokens & set(COORD_WORDS)) or any(w in low for w in
                                                  ("geopoint", "geo point"))


@dataclass
class Column:
    """One column, with the numbers every page needs already computed."""
    name: str
    role: str
    dtype: str
    n: int
    missing: int
    unique: int
    label: str = ""                      # cleaned, human-readable
    short: str = ""                      # title-length version of the label
    sample: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    sentinel_hits: dict = field(default_factory=dict)

    @property
    def missing_pct(self) -> float:
        return 100.0 * self.missing / self.n if self.n else 0.0

    @property
    def filled(self) -> int:
        return self.n - self.missing


@dataclass
class Profile:
    """A whole sheet: the frame, its columns, and helpers for picking columns."""
    df: pd.DataFrame
    columns: dict
    source: str = ""
    sheet: str = ""
    kobo: dict = field(default_factory=dict)
    dropped: list = field(default_factory=list)
    truncated: int = 0                   # rows beyond the read cap, if any

    def by_role(self, *roles: str) -> list:
        return [c for c in self.columns.values() if c.role in roles]

    def names(self, *roles: str) -> list:
        return [c.name for c in self.by_role(*roles)]

    @property
    def rows(self) -> int:
        return len(self.df)

    def label(self, col: str) -> str:
        c = self.columns.get(col)
        return c.label if c and c.label else str(col)

    def short(self, col: str) -> str:
        """The title-length label. Falls back to the full one."""
        c = self.columns.get(col)
        return (c.short or c.label) if c else str(col)

    def best_date(self):
        """The date column a trend should use: most complete, then most spread."""
        dates = [c for c in self.by_role(DATETIME) if c.stats.get("span_days", 0) > 0]
        if not dates:
            return None
        return max(dates, key=lambda c: (c.filled, c.stats.get("span_days", 0))).name

    def best_geo(self):
        """A state column beats an LGA column: fewer levels, reads better."""
        geo = [c for c in self.by_role(GEO) if 1 < c.unique <= 60]
        if not geo:
            return None
        return min(geo, key=lambda c: (0 if "state" in c.name.lower() else 1,
                                       c.unique)).name

    def headline_cats(self, limit: int = 6) -> list:
        """Categoricals worth putting on the front page, most useful first.

        Useful means: few enough levels to read as a bar chart, mostly filled,
        and not dominated by a single value (a 99%-one-answer column tells the
        reader nothing).
        """
        out = []
        for c in self.by_role(CATEGORICAL, GEO, BINARY):
            if c.unique < 2 or c.unique > 25 or c.filled < max(5, 0.25 * c.n):
                continue
            # A folded Kobo option is already covered by its question's panel;
            # left in, six of them crowd the front page and say the same thing.
            if "/" in c.name and c.name.rsplit("/", 1)[0] in self.kobo:
                continue
            share = c.stats.get("top_n", 0) / max(c.filled, 1)
            if share > 0.97:
                continue
            out.append((share, -c.filled, c))
        out.sort(key=lambda t: (t[0], t[1]))
        return [c for _, _, c in out[:limit]]

    def headline_nums(self, limit: int = 4) -> list:
        """Numerics worth a KPI tile: filled, varying, not a sentinel dump.

        Coordinates are excluded outright. Kobo ships every submission with
        latitude, longitude, altitude and GPS precision, all of which are
        numeric, well filled and varying — so they sailed onto the dashboard as
        "Average geo-point latitude", which is not a number anybody wants a mean
        of. They are location, not measurement.
        """
        out = [c for c in self.by_role(NUMERIC)
               if c.filled >= max(5, 0.2 * c.n) and c.stats.get("std", 0) > 0
               and not _is_coordinate(c.name)]
        out.sort(key=lambda c: -c.filled)
        return out[:limit]


# ═══ loading ════════════════════════════════════════════════════════════════
def file_key(name: str, data: bytes) -> str:
    """Cache key for an upload — content hash, so a re-upload is free."""
    return name + ":" + hashlib.sha1(data).hexdigest()[:16]


def sheet_names(data: bytes, name: str) -> list:
    if name.lower().endswith((".csv", ".txt", ".tsv")):
        return []
    with pd.ExcelFile(io.BytesIO(data)) as xl:
        return list(xl.sheet_names)


def read_any(data: bytes, name: str, sheet=None, nrows=None) -> pd.DataFrame:
    """Read CSV or Excel into a frame, tolerating the usual export mess."""
    low = name.lower()
    if low.endswith(".xls"):
        raise ValueError("Old .xls format — open it in Excel and Save As .xlsx")
    if low.endswith((".csv", ".txt", ".tsv")):
        sep = "\t" if low.endswith(".tsv") else None
        last = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(data), sep=sep, engine="python",
                                   encoding=enc, na_values=NA_STRINGS,
                                   nrows=nrows, on_bad_lines="skip")
            except (UnicodeDecodeError, LookupError) as exc:
                last = exc
        raise ValueError("Could not decode this CSV — try saving it as .xlsx") from last
    frame = pd.read_excel(io.BytesIO(data), sheet_name=sheet if sheet else 0,
                          na_values=NA_STRINGS, nrows=nrows)
    if isinstance(frame, dict):
        frame = next(iter(frame.values()))
    return frame


# ═══ typing ═════════════════════════════════════════════════════════════════
def clean_label(name) -> str:
    """`hh/water_source_main` becomes `Water source main`. Kobo names are noise."""
    s = str(name).split("/")[-1]
    s = re.sub(r"[_\.]+", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    if s.isupper() and len(s) > 4:
        s = s.capitalize()
    return (s[:1].upper() + s[1:]) if s else str(name)


def _texty(s: pd.Series) -> bool:
    """True for a column of strings under either pandas generation.

    pandas 3.0 reads text as StringDtype ("str"), not object, so the obvious
    `s.dtype == object` test is False for every text column in the file. That
    single check silently turned off date parsing, geography detection and blank
    normalisation — the sample workbook profiled with no date column at all
    until this helper replaced it.
    """
    dt = s.dtype
    if dt == object:
        return True
    if isinstance(dt, pd.CategoricalDtype):
        return False
    try:
        return bool(pd.api.types.is_string_dtype(dt))
    except (TypeError, ValueError):
        return False


# Question stems that carry no information in a chart title. A Kobo column name
# is the entire question, so "Do you experience long wait times or delays when
# accessing care at the PHC facility?" has to become "Experience long wait times"
# or every title on the page is a paragraph.
STEMS = (
    "in your opinion,", "based on your recent experiences,", "based on your",
    "please tell us", "we would like to know", "kindly tell us",
    "how satisfied are you with the", "how satisfied are you",
    "how would you rate the", "how would you rate",
    "how often do you", "how has the", "how have these", "how do you think",
    "how many", "how ", "what is the", "what are the", "what other",
    "what specific", "what factors", "what role do you think",
    "what improvements do you think are needed to", "what ", "which ",
    "do you feel that your", "do you feel that the", "do you feel that",
    "do you believe that the", "do you believe that", "do you receive",
    "do you experience", "do you face any", "do you consent to", "do you or your",
    "do you ", "are there ", "are you ", "have you ", "has ", "is there any",
    "is there ", "can you describe", "can you give", "can you share", "can you ",
    "when you or your family members need", "when ",
)
TAILS = ("(select all that apply):", "(select all that apply)", "(please specify)",
         "- please specify", ":", "?")


def short_label(name, limit: int = 46) -> str:
    """A chart-title-length version of a column name.

    Kept separate from `label`, which stays complete: the short form goes in
    titles and axis labels, the full question stays in the caption and the
    tooltip, so nothing is lost — the reader can still see exactly which
    question a bar came from.
    """
    s = clean_label(name)
    low = s.lower()
    for t in TAILS:
        if low.endswith(t):
            s = s[: len(s) - len(t)].strip(" -–—:")
            low = s.lower()
    for stem in STEMS:
        if low.startswith(stem):
            s = s[len(stem):].strip()
            break
    s = re.sub(r"\s+", " ", s).strip(" -–—:,")
    if len(s) > limit:
        cut = s[:limit].rsplit(" ", 1)[0]
        s = (cut or s[:limit]).rstrip(" ,;:") + "…"
    return (s[:1].upper() + s[1:]) if s else clean_label(name)


def _looks_like_id(name, s: pd.Series, unique: int, filled: int) -> bool:
    """An identifier labels a row; a measurement describes it.

    Near-uniqueness alone is not enough to tell them apart, and treating it as
    enough was wrong in the common case: 400 rows of continuous readings are 400
    distinct values, so six real indicator columns were all filed as IDs and
    vanished from the analysis. A float is never an identifier. An unnamed
    integer column qualifies only if its values form a near-contiguous run —
    what a serial or a row number looks like, and what a measurement does not.
    """
    low = str(name).lower()
    named = any(low == w or low.endswith(w) or low.startswith(w.strip("_"))
                for w in ID_WORDS)
    if filled <= 20 or unique / filled < 0.95:
        return False
    if not pd.api.types.is_numeric_dtype(s):
        return True                      # near-unique text: a code or free text
    nums = pd.to_numeric(s, errors="coerce").dropna()
    if nums.empty or not bool(np.all(np.mod(nums.to_numpy(dtype="float64"), 1) == 0)):
        return False                     # fractional values: a measurement
    span = float(nums.max() - nums.min()) + 1
    sequential = span <= 1.6 * unique          # a serial or a row number
    # Fixed-width long integers are codes, not quantities: phone numbers, NINs,
    # account numbers. Excel strips the leading zero from 08031234567 and hands
    # back an 8.0e9 integer, and the tool would otherwise print the mean of a
    # column of phone numbers. Fixed width is what separates these from money —
    # a naira column runs 4 digits to 8, a phone column is 10 every time.
    widths = nums.abs().astype("int64").astype(str).str.len()
    fixed_width_code = bool(widths.nunique() == 1 and widths.iloc[0] >= 8)
    return named or sequential or fixed_width_code


NOT_PLACE = {"type", "types", "category", "kind", "level", "class", "code",
             "count", "number", "num", "status", "name"}


def _is_geo(name, values: pd.Series) -> bool:
    """Whole-token matching, so `facility_type` is a category and `facility` a place.

    Substring matching read facility_type as geography, which then sorted it into
    the map-shaped part of the app and out of the crosstabs where it belonged.

    The name rule applies only to **short** names. A Kobo export uses the whole
    question as the column name, and fifteen columns of a real household survey
    typed as geography because the question happened to contain "facility" or
    "community" — "Do you feel that the PHC facility is adequately equipped" is
    not a place. A column name that is a sentence is a question; only a label
    can be a place name, so anything long has to earn `geo` from its values.
    """
    label = str(name).strip()
    tokens = set(re.split(r"[^a-z0-9]+", label.lower())) - {""}
    short = len(label) <= 40 and len(tokens) <= 5
    if short and tokens & set(GEO_WORDS) and not tokens & NOT_PLACE:
        return True
    if _texty(values) and len(values) > 5:
        vals = {str(v).strip().lower() for v in values.head(400)}
        hit = len(vals & STATES)
        return hit >= 3 and hit / max(len(vals), 1) > 0.5
    return False


def _try_dates(s: pd.Series):
    """Parse to datetime only when most non-null values actually parse."""
    sample = s.dropna().astype(str).head(300)
    if sample.empty:
        return None
    first = sample.iloc[0]
    if not any(ch in first for ch in "-/:") and not re.match(r"^\d{4}", first):
        return None
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except Exception:
        return None
    if parsed.notna().mean() < 0.85:
        return None
    try:
        full = pd.to_datetime(s, errors="coerce", format="mixed")
    except Exception:
        return None
    return full if full.notna().mean() >= 0.5 else None


def _maybe_dates(s: pd.Series, name):
    """Only attempt date parsing on strings, and trust date-ish names more."""
    if not _texty(s):
        return None
    low = str(name).lower()
    hinted = any(w in low for w in ("date", "time", "day", "month", "when",
                                    "start", "end", "submit", "dob", "visit"))
    parsed = _try_dates(s)
    if parsed is None:
        return None
    # Unhinted columns must parse near-perfectly to be believed as dates.
    return parsed if hinted or parsed.notna().mean() > 0.95 else None


def numeric_stats(s: pd.Series):
    """Descriptives, plus which sentinel codes are present and how often.

    Sentinels are excluded from mean/median/std. A survey where 12% answered
    "don't know" as 99 has a real mean and a fake one; this returns the real one
    and tells the caller the fake one was avoided.
    """
    clean = pd.to_numeric(s, errors="coerce").dropna()
    if clean.empty:
        return {}, {}
    hits = {}
    if float(clean.max()) >= 98 or float(clean.min()) < 0:
        for sv in SENTINELS:
            c = int((clean == sv).sum())
            # A sentinel only counts when it is a spike, not the natural range.
            if c and c / len(clean) > 0.005:
                hits[sv] = c
    real = clean[~clean.isin(list(hits))] if hits else clean
    if real.empty:
        real, hits = clean, {}
    q1, q3 = float(real.quantile(0.25)), float(real.quantile(0.75))
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    vals = real.to_numpy(dtype="float64", na_value=np.nan)
    stats = {
        "min": float(real.min()), "max": float(real.max()),
        "mean": float(real.mean()), "median": float(real.median()),
        "std": float(real.std()) if len(real) > 1 else 0.0,
        "sum": float(real.sum()), "q1": q1, "q3": q3, "n": int(len(real)),
        "zeros": int((real == 0).sum()),
        "negatives": int((real < 0).sum()),
        "outliers": int(((real < lo) | (real > hi)).sum()) if iqr > 0 else 0,
        "out_lo": lo, "out_hi": hi,
        "integerish": bool(np.all(np.mod(vals[~np.isnan(vals)], 1) == 0)),
    }
    return stats, hits


def classify(df: pd.DataFrame):
    """Assign a role to every column, coercing dtypes where the role implies one.

    Order matters: empty, constant, datetime, binary, numeric, geo, id, then
    categorical or free text. Returns the retyped frame and the column map.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    n = len(df)
    cols = {}

    for name in list(df.columns):
        s = df[name]
        if _texty(s):
            s = s.replace(NA_STRINGS, np.nan)
            df[name] = s
        missing = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))
        filled = n - missing
        stats, sent = {}, {}
        parsed = None

        if filled == 0:
            role = EMPTY
        elif unique == 1:
            role = CONSTANT
            stats = {"value": str(s.dropna().iloc[0])[:80]}
        elif pd.api.types.is_datetime64_any_dtype(s):
            role = DATETIME
        elif (parsed := _maybe_dates(s, name)) is not None:
            df[name] = parsed
            s = parsed
            role = DATETIME
        elif unique <= 3 and s.dropna().astype(str).str.strip().str.lower().isin(YES_NO).all():
            role = BINARY
        elif pd.api.types.is_numeric_dtype(s):
            if _looks_like_id(name, s, unique, filled):
                role = IDENTIFIER
            elif unique == 2:
                role = BINARY
            else:
                role = NUMERIC
                stats, sent = numeric_stats(s)
        elif _is_geo(name, s.dropna()):
            role = GEO
        elif _looks_like_id(name, s, unique, filled):
            role = IDENTIFIER
        elif unique == 2:
            role = BINARY
        else:
            as_num = pd.to_numeric(s, errors="coerce")
            if filled and as_num.notna().sum() / filled > 0.9:
                df[name] = as_num
                s = as_num
                role = NUMERIC
                stats, sent = numeric_stats(s)
            elif unique <= MAX_CAT_LEVELS or (filled > 200 and unique / filled < 0.1):
                role = CATEGORICAL
            else:
                avg_len = s.dropna().astype(str).str.len().mean()
                role = TEXT if avg_len and avg_len > 25 else CATEGORICAL

        if role == DATETIME:
            good = pd.to_datetime(df[name], errors="coerce").dropna()
            if not good.empty:
                stats = {"start": good.min(), "end": good.max(),
                         "span_days": int((good.max() - good.min()).days)}
        if role in (CATEGORICAL, BINARY, GEO):
            vc = s.value_counts(dropna=True)
            stats["top"] = str(vc.index[0]) if len(vc) else ""
            stats["top_n"] = int(vc.iloc[0]) if len(vc) else 0
            stats["levels"] = int(len(vc))
            if role != BINARY and 1 < unique <= MAX_CAT_LEVELS:
                df[name] = s.astype("category")   # 10k rows x 40 cats: cheap

        cols[name] = Column(
            name=name, role=role, dtype=str(df[name].dtype), n=n,
            missing=missing, unique=unique, label=clean_label(name),
            short=short_label(name),
            sample=[str(v)[:60] for v in s.dropna().head(4).tolist()],
            stats=stats, sentinel_hits=sent,
        )
    return df, cols


# ═══ Kobo select_multiple folding ═══════════════════════════════════════════
def kobo_groups(cols: dict) -> dict:
    """Find `parent/option` binary families — one Kobo select_multiple question.

    Three or more binary siblings are required, which stops ordinary
    slash-named columns being folded together by accident.
    """
    fams = {}
    for name, c in cols.items():
        if "/" not in name:
            continue
        parent = name.rsplit("/", 1)[0]
        if c.role in (BINARY, EMPTY, CONSTANT) or c.unique <= 3:
            fams.setdefault(parent, []).append(name)
    # The parent column is allowed to exist. Kobo sometimes ships the joined
    # answer as its own column *and* one binary column per option; requiring the
    # parent to be absent meant a real household survey with seven such families
    # detected none of them, and the option columns scattered across the page as
    # unrelated yes/no fields.
    return {p: v for p, v in fams.items() if len(v) >= 3}


def multi_counts(df: pd.DataFrame, members: list) -> pd.DataFrame:
    """Selected-count per option for one folded question, as a tidy frame."""
    rows = []
    for m in members:
        s = df[m]
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype(object)
        if s.dtype == object:
            sel = s.astype(str).str.strip().str.lower().isin(
                {"1", "yes", "true", "1.0", "y"})
        else:
            sel = pd.to_numeric(s, errors="coerce").fillna(0) > 0
        rows.append({"option": clean_label(m), "n": int(sel.sum())})
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    return out[out["n"] > 0].reset_index(drop=True)


def build(data: bytes, name: str, sheet=None, row_cap: int | None = None) -> Profile:
    """Upload bytes to a fully typed Profile. The one entry point the app calls."""
    raw = read_any(data, name, sheet)
    # Blank rows go; wholly blank *columns* stay, and are marked EMPTY. A column
    # nobody filled in is a finding about the form, and dropping it here would
    # make it invisible on the column map where someone can act on it.
    raw = raw.dropna(axis=0, how="all")
    keep = [c for c in raw.columns if not re.match(r"^Unnamed: \d+$", str(c))]
    raw = raw[keep]
    truncated = 0
    if row_cap and len(raw) > row_cap:
        truncated = len(raw) - row_cap
        raw = raw.head(row_cap)
    df, cols = classify(raw)
    prof = Profile(df=df, columns=cols, source=name, sheet=sheet or "",
                   truncated=truncated)
    prof.kobo = kobo_groups(cols)
    prof.dropped = [c.name for c in cols.values() if c.role in (EMPTY, CONSTANT)]
    return prof
