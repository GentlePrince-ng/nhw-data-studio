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
               "meta", "audit", "formhub", "simserial", "subscriberid",
               "imei", "phonenumber", "phone number", "devicephonenum",
               "_attachments", "_geolocation", "_submission__id", "_xform_id",
               "_tags", "_uri", "_attachments_count")


def is_system(name) -> bool:
    """Kobo/ODK plumbing: real columns, but never the subject of a chart."""
    low = str(name).strip().lower()
    return low in SYSTEM_COLS or low.startswith(("_geo", "meta/", "formhub/"))


COORD_WORDS = ("latitude", "longitude", "lat", "lon", "lng", "altitude",
               "precision", "geopoint", "geo point", "gps", "accuracy",
               "easting", "northing", "elevation")


# Time parts held as numbers: a year is a label for a period, not a quantity.
# "Average year 2021" is not a metric, and summing them is worse.
TIME_PART_WORDS = ("year", "yr", "month", "mth", "week", "wk", "day", "quarter",
                   "qtr", "fy", "epoch", "timestamp", "hour", "minute")


# Tokens that carry no meaning of their own, so "week number" is still a week.
FILLER = {"of", "the", "in", "a", "number", "no", "num", "nos", "name", "id",
          "index", "period", "label", "value", "fiscal", "financial",
          "calendar", "reporting", "survey", "data"}


def _is_time_part(name) -> bool:
    """True only when the name is *entirely* a time label.

    A containment test was too greedy: `visits_last_year` contains "year", so a
    genuine count of visits stopped being a measure. The whole name has to be
    the time label — "Year", "Fiscal year", "week number" — for the column to
    be a period marker rather than a quantity.
    """
    toks = set(re.sub(r"[^a-z ]+", " ", str(name).lower()).split()) - FILLER
    return bool(toks) and toks <= set(TIME_PART_WORDS)


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
    sheets_used: list = field(default_factory=list)
    sheets_skipped: list = field(default_factory=list)
    total_rows_dropped: int = 0
    sheet_note: str = ""
    source_notes: list = field(default_factory=list)
    shape_note: str = ""

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
        dates = [c for c in self.by_role(DATETIME)
                 if c.stats.get("span_days", 0) > 0
                 or c.stats.get("span_seconds", 0) >= 900]
        if not dates:
            return None
        return max(dates, key=lambda c: (c.filled,
                                         c.stats.get("span_seconds", 0))).name

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
        # min(3, n) rather than a flat floor of 5: a four-row sheet of podcast
        # episodes had every numeric column excluded by the floor, so the page
        # showed a record count and nothing else.
        floor = min(max(3, int(0.2 * self.rows)), max(self.rows, 1))
        out = [c for c in self.by_role(NUMERIC)
               if c.filled >= floor and c.stats.get("std", 0) > 0
               and not _is_coordinate(c.name)
               and not _is_time_part(c.name)]
        out.sort(key=lambda c: -c.filled)
        return out[:limit]


# ═══ loading ════════════════════════════════════════════════════════════════
def file_key(name: str, data: bytes) -> str:
    """Cache key for an upload — content hash, so a re-upload is free."""
    return name + ":" + hashlib.sha1(data).hexdigest()[:16]


UNREADABLE = ("Excel could not open this workbook — the file's formatting "
              "section is invalid, which some export tools produce. Open it in "
              "Excel and use Save As to write a fresh .xlsx, then upload that.")


def sheet_names(data: bytes, name: str) -> list:
    """The sheet list, or [] when the workbook cannot be opened at all.

    A STATcompiler export raised deep inside openpyxl's stylesheet parser. The
    sidebar asks for the sheet list before anything else, so an exception here
    took out the whole page — and a file the tool cannot open should still get
    a sentence saying so.
    """
    if name.lower().endswith((".csv", ".txt", ".tsv")):
        return []
    try:
        with pd.ExcelFile(io.BytesIO(data)) as xl:
            return list(xl.sheet_names)
    except Exception:                    # noqa: BLE001 - any parser failure
        return []


def split_comments(data: bytes):
    """Peel off a leading block of `#` comment lines, keeping them as notes.

    GA4, Search Console, YouTube Analytics and Matomo all export a CSV that
    opens with a metadata block:

        # ----------------------------------------
        # Pages and screens: Page path and screen class
        # Account: Nigeria Health Watch
        # Start date: 20230101

    Left in place it is not merely noise. pandas sniffs the delimiter from the
    first line, finds no comma in `# ------`, splits on spaces instead — and an
    8,801-row eight-column export read as 8,750 rows and TWO columns. Only the
    leading run is removed, so a `#` inside real data is untouched, and the lines
    are kept because they carry the property and the date range: exactly the
    provenance a chart should be able to cite.
    """
    notes, idx, n = [], 0, len(data)
    while idx < n:
        end = data.find(b"\n", idx)
        if end == -1:
            end = n
        line = data[idx:end].strip()
        if not line.startswith(b"#"):
            break
        text = line.lstrip(b"#").strip().decode("utf-8", "replace")
        if text and set(text) != {"-"}:
            notes.append(text)
        idx = end + 1
    return data[idx:], notes


def sniff_sep(body: bytes) -> str:
    """Pick the delimiter by counting candidates across the first few lines.

    `sep=None` asks pandas to sniff, and it sniffs the first line only — the
    wrong line whenever a file opens with anything unusual. Counting across
    several lines and preferring a consistent count is cheaper and harder to
    fool: a comma inside prose varies line to line, a delimiter does not.
    """
    head = body[:16000].decode("utf-8", "replace").splitlines()
    lines = [ln for ln in head if ln.strip()][:8]
    if not lines:
        return ","
    best, best_score = ",", -1.0
    for cand in (",", "\t", ";", "|"):
        counts = [ln.count(cand) for ln in lines]
        if min(counts) < 1:
            continue
        consistency = 1.0 - (max(counts) - min(counts)) / max(max(counts), 1)
        score = min(counts) * (0.5 + consistency)
        if score > best_score:
            best, best_score = cand, score
    return best


def find_header(raw: pd.DataFrame, look: int = 15) -> int:
    """Which row holds the column names.

    A hand-maintained tracker almost never starts with its header: there is a
    title, some blank rows, maybe a note, and then the table. Read with the
    default `header=0` and a campaign tracker came out as **one column and 71
    rows** — the title became the only column name and every real heading became
    data. Nothing downstream can recover from that.

    The rule is **type contrast**, not width. A header is text sitting on top of
    numbers; a row of data matches the type of its own column. Counting both
    ways, as a share of the columns that can be judged, is what separates them.
    Width was the main term once, and it meant any full row of data beat a
    header with merged or empty cells above it — a post-test analysis came out
    headed with a trainee's name, a funding table with 186083442.0. Width is
    now only a tie-breaker.

    Four smaller rules, each from one real file: a year is a name, so a table
    headed 2015..2021 is a header row; a row with nothing beneath it is not
    scored at all, because every signal collapses to zero at the bottom of the
    sampled block and the last row then wins by default; a row holding an email
    or a URL is never a header, while *length* must not be a signal because a
    Kobo heading is a whole question; and where two adjacent rows both look like
    headers the lower one wins, so `_merge_header` can fold the merged group
    name above it back in.

    A well-formed export wins on row 0 immediately and is unaffected.
    """
    if raw.empty:
        return 0
    block = raw.head(look)
    filled = block.notna().sum(axis=1)
    best, best_score, scored = 0, -1.0, {}
    for r in range(min(look, len(block))):
        # Only judge a row that has rows beneath it to be judged against. At
        # the bottom of the sampled block there is nothing below, so the
        # type-contrast terms all collapse to zero and the last row scored
        # highest by default — which is exactly what it did.
        if len(block) - (r + 1) < 3 and r > 0:
            break
        vals = block.iloc[r].dropna()
        n = len(vals)
        if n < 2:
            continue
        labels = [str(v).strip() for v in vals]
        # A year is a name, not a measurement. A funding table headed
        # 2015..2021 was rejected as "all numeric", so the first row of figures
        # became the column names and every heading read like 186083442.0.
        wordy = sum(1 for v in labels
                    if not str(v).replace(".", "", 1).lstrip("-").isdigit()
                    or 1900 <= _as_year(v) <= 2100)
        # An all-numeric row is data, never a header. Without this, a sheet
        # whose real header repeats a label ("x", "x", "y") had that row
        # rejected and the first row of numbers promoted instead, quietly
        # eating a record.
        if wordy == 0:
            continue
        # Rows below must carry data — but only *some* data. Requiring them to
        # be as wide as the header rejected the header of any sparse export: a
        # contacts file with 13 columns and 2 filled per row lost its header
        # row, which then became the first record's names and cost a record.
        below = filled.iloc[r + 1: r + 6]
        if below.empty or int(below.max()) < 2:
            continue
        # Repeated labels are a smell, not a disqualification: real trackers do
        # ship two columns called "Notes".
        # Type contrast, column by column. A header is text sitting on top of
        # numbers; a row of data matches the type of its own column. Counting
        # both ways is what separates them, and it does not care how wide the
        # row is — which width did, so a header split across two merged rows
        # always lost to the first full row of data beneath it.
        switch = datalike = 0
        for pos in range(block.shape[1]):
            cell = block.iat[r, pos]
            if pd.isna(cell):
                continue
            under = block.iloc[r + 1: r + 8, pos].dropna()
            if len(under) < 2:
                continue
            numbers_below = sum(_numberish(v) for v in under) >= 0.6 * len(under)
            if not numbers_below:
                continue
            if _numberish(cell):
                datalike += 1        # a number over numbers: this is a record
            else:
                switch += 1          # a name over numbers: this is a heading
        # An email address or a link is a value, never a column name. On an
        # all-text table — a Zoom attendee report — the type contrast above is
        # silent, so the widest row of attendees won and every column came out
        # prefixed with somebody's address. Length is deliberately not a signal
        # here: a Kobo header is a full question, sixty characters and more.
        if any(re.match(r"\S+@\S+\.\S+$", v) or v.lower().startswith(
                ("http://", "https://", "www.")) for v in labels):
            continue
        dupes = len(labels) - len(set(labels))
        # A quarter-point, not a whole one. At -1 a single repeated heading was
        # enough for the first row of data to outscore the real header on a
        # narrow table, because a text-heavy data row scores nearly as well.
        # As a *share* of the columns that could be judged, not a raw count: a
        # count rewarded rows far down the sheet where the data thins out and
        # only two columns can be judged at all, and one of those outscored the
        # real header. Capped weight, so one judgeable column cannot decide it.
        judged = switch + datalike
        contrast = ((switch - datalike) / judged * min(judged, 3)) if judged else 0
        # Width is a tie-breaker now, not the main term. When it led, any row
        # of data outscored a header with merged or empty cells above it.
        score = (n * 0.1 + wordy * 0.5 + contrast
                 - dupes * 0.25 - r * 0.01)         # early row wins ties
        scored[r] = (score, contrast)
        if score > best_score:
            best, best_score = r, score
    # Two-row headers: "Pre-Test" merged across three columns, then "#Correct /
    # #Incorrect / Score" beneath. Both rows score as headers and the upper one
    # can win by a hair, which loses the sub-headings and leaves three columns
    # called Unnamed. Take the lower row — _merge_header folds the group name
    # back in, giving "Pre-Test #Correct".
    while best + 1 in scored:
        nxt, nxt_contrast = scored[best + 1]
        if nxt_contrast <= 0 or nxt < best_score - 1.0:
            break
        best = best + 1
    return best


def read_any(data: bytes, name: str, sheet=None, nrows=None) -> pd.DataFrame:
    """Read CSV or Excel into a frame, tolerating the usual export mess."""
    low = name.lower()
    if low.endswith(".xls"):
        raise ValueError("Old .xls format — open it in Excel and Save As .xlsx")
    if not low.endswith((".csv", ".txt", ".tsv")) and not sheet_names(data, name):
        raise ValueError(UNREADABLE)
    if low.endswith((".csv", ".txt", ".tsv")):
        body, notes = split_comments(data)
        sep = "\t" if low.endswith(".tsv") else sniff_sep(body)
        last = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                frame = _apply_header(pd.read_csv(
                    io.BytesIO(body), sep=sep, engine="python", encoding=enc,
                    na_values=NA_STRINGS, nrows=nrows, header=None,
                    on_bad_lines="skip"))
                frame.attrs["notes"] = notes
                return frame
            except (UnicodeDecodeError, LookupError) as exc:
                last = exc
        raise ValueError("Could not decode this CSV — try saving it as .xlsx") from last
    raw = pd.read_excel(io.BytesIO(data), sheet_name=sheet if sheet else 0,
                        na_values=NA_STRINGS, header=None, nrows=nrows)
    if isinstance(raw, dict):
        raw = next(iter(raw.values()))
    return _apply_header(raw)


def _merge_header(raw: pd.DataFrame, h: int) -> list:
    """The header row, with a merged group header above it folded in.

    Excel headers are routinely two rows: "Pre-Test" merged across three
    columns, then "#Correct / #Incorrect / Score" beneath. Reading only the
    lower row gave "#Correct" and "#Correct 2" — indistinguishable — and left
    the S/N and Name columns nameless, so they were dropped as Unnamed and the
    file lost the one column that said whose score it was.

    A merged cell reads as a value followed by blanks, so the row above is
    forward-filled. A one-cell title is not a group header and is ignored,
    otherwise every column would be prefixed with the report's name.
    """
    row = list(raw.iloc[h])
    if h == 0:
        return row
    above = raw.iloc[h - 1]
    if above.notna().sum() < 2 or above.notna().sum() >= raw.iloc[h].notna().sum():
        return row
    spread = above.ffill()
    out = []
    for i, v in enumerate(row):
        top = spread.iloc[i] if i < len(spread) else None
        if pd.isna(v) or not str(v).strip() or str(v).strip().lower() == "nan":
            out.append(top)                       # blank below a name: use it
        elif pd.notna(top) and str(top).strip() and str(top).strip() != str(v).strip():
            out.append(f"{str(top).strip()} {str(v).strip()}")
        else:
            out.append(v)
    return out


def _apply_header(raw: pd.DataFrame) -> pd.DataFrame:
    """Promote the detected header row and drop everything above it."""
    if raw.empty:
        return raw
    h = find_header(raw)
    frame = raw.iloc[h + 1:].reset_index(drop=True)
    row = _merge_header(raw, h)
    names, seen = [], {}
    for i, v in enumerate(row):
        label = str(v).strip() if pd.notna(v) else f"Unnamed: {i}"
        if not label or label.lower() == "nan":
            label = f"Unnamed: {i}"
        # Duplicate headings get a suffix rather than being silently dropped:
        # a tracker with two "Notes" columns still has two columns of notes.
        if label in seen:
            # The suffix has to be checked too. A form export with two "Notes"
            # columns *and* a real "Notes 2" produced two columns both called
            # "Notes 2", and from then on frame[col] returned a DataFrame:
            # every per-column check downstream raised on .dtype.
            base, n = label, seen[label]
            while label in seen:
                n += 1
                label = f"{base} {n}"
            seen[base] = n
        seen[label] = seen.get(label, 1)
        names.append(label)
    frame.columns = names
    # Reading with header=None makes pandas infer every column as object,
    # because the title row sits in the same column as the numbers. Restoring
    # the types here matters more than it looks: `_looks_like_id` asks whether a
    # column is numeric, so an all-object frame had "No of Views" — near-unique
    # integers — filed as an identifier and dropped from the measures entirely.
    frame = frame.infer_objects()
    for col in frame.columns:
        # _texty, not `== object`: pandas 3 gives strings the `str` dtype, so an
        # object-only test silently skipped every column and left the numbers as
        # text — which is how a Views column of four-digit counts reached the
        # date parser at all.
        if _texty(frame[col]):
            asnum = pd.to_numeric(frame[col], errors="coerce")
            filled = frame[col].notna().sum()
            if filled and asnum.notna().sum() / filled > 0.95:
                frame[col] = asnum
    return frame


# ═══ typing ═════════════════════════════════════════════════════════════════
# Export conventions that label a column's job in its name. MetricsForNews and
# Parse.ly ship "metric: Total pageviews" and "tag: Topic"; the prefix is useful
# to know and useless to read, so it is stripped for display and for the
# aggregation rules — "metric: Total attention hours" was being averaged because
# the rule that looks for a leading "total" saw "metric" there instead.
FIELD_PREFIXES = ("metric", "metrics", "tag", "tags", "dim", "dimension",
                  "measure", "field", "attr", "attribute", "custom")


def strip_prefix(name) -> str:
    """`metric: Total pageviews` becomes `Total pageviews`."""
    raw = str(name).strip()
    if ":" in raw:
        head, rest = raw.split(":", 1)
        if head.strip().lower() in FIELD_PREFIXES and rest.strip():
            return rest.strip()
    return raw


def prefix_of(name) -> str:
    """The convention prefix, lowercased, or "" — "metric", "tag"."""
    raw = str(name).strip()
    if ":" in raw:
        head = raw.split(":", 1)[0].strip().lower()
        if head in FIELD_PREFIXES:
            return head
    return ""


def clean_label(name) -> str:
    """`hh/water_source_main` becomes `Water source main`. Kobo names are noise."""
    s = strip_prefix(str(name)).split("/")[-1]
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
    s = clean_label(strip_prefix(name))
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
    # A column called phonenumber is never a measurement, however many values it
    # repeats. Requiring near-uniqueness first meant a phone column with a few
    # duplicates escaped this test entirely and the dashboard reported "average
    # phonenumber 7,799,717,270".
    if named and unique > 1:
        return True
    # Fixed-width long integers are codes whether or not they repeat: an NIN, an
    # account number, a meter number.
    if pd.api.types.is_numeric_dtype(s) and unique > 20:
        nums = pd.to_numeric(s, errors="coerce").dropna()
        if len(nums) and bool(np.all(np.mod(
                nums.to_numpy(dtype="float64"), 1) == 0)):
            widths = nums.abs().astype("int64").astype(str).str.len()
            if widths.nunique() == 1 and widths.iloc[0] >= 8:
                return True
    if filled <= 20 or unique / filled < 0.95:
        return False
    if not pd.api.types.is_numeric_dtype(s):
        # Unless it is numbers wearing an object dtype, which is what an
        # unheadered read produces.
        asnum = pd.to_numeric(s, errors="coerce")
        if not (filled and asnum.notna().sum() / filled > 0.95):
            return True                  # near-unique text: a code or free text
        s = asnum
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
    # A column of bare four-digit numbers is only a date if those numbers are
    # plausible years. Without this, Views of [1954, 4650, 2100, 2629] parsed as
    # 1954-01-01, 4650-01-01 … and the headline measure of a podcast sheet
    # became a datetime column.
    bare = sample.str.fullmatch(r"\s*\d{4}(\.0)?\s*")
    if bare.mean() > 0.5:
        years = pd.to_numeric(sample.str.extract(r"(\d{4})")[0], errors="coerce")
        if not ((years >= 1900) & (years <= 2100)).all():
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
        # A sentinel has to be an OUTLIER as well as a spike. 98/99/999 are
        # ordinary values in a web metric — a page with 99 entries is not
        # "don't know" — so a candidate must also sit outside the bulk of the
        # distribution. On a survey age column 99 clears the 97.5th percentile
        # of everything else; on a pageviews column it is nowhere near it.
        for sv in SENTINELS:
            c = int((clean == sv).sum())
            if not c or c / len(clean) <= 0.005:
                continue
            rest = clean[clean != sv]
            if rest.empty:
                continue
            if sv > 0 and sv <= float(rest.quantile(0.975)):
                continue
            if sv < 0 and sv >= float(rest.quantile(0.025)):
                continue
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
                gap = good.max() - good.min()
                stats = {"start": good.min(), "end": good.max(),
                         "span_days": int(gap.days),
                         # A day of GPS pings or a day of hourly traffic has a
                         # span of 0 *days*, which read as "no time axis" and
                         # cost the file its trend chart entirely.
                         "span_seconds": int(gap.total_seconds())}
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


TOTAL_LABELS = {"total", "totals", "grand total", "sum", "subtotal",
                "overall", "all", "average", "mean", "grand totals"}


def drop_total_rows(frame: pd.DataFrame):
    """Remove subtotal rows that live inside the data, and count them.

    A YouTube export in a real report workbook opens with a row labelled
    "Total" carrying the sum of every video below it. Left in, every total the
    tool reports is exactly double, every average is wrong, and "Total" appears
    as a category in its own right. Dropped silently it would be worse — so the
    count comes back and the page says what went.

    Only a row whose *first* filled text cell is a total label counts, so an
    article headline containing the word "total" is safe.
    """
    if frame.empty:
        return frame, 0
    text_cols = [c for c in frame.columns
                 if frame[c].dtype == object or pd.api.types.is_string_dtype(frame[c])]
    if not text_cols:
        return frame, 0
    first = frame[text_cols[0]].astype(str).str.strip().str.lower()
    mask = first.isin(TOTAL_LABELS).to_numpy()
    if not mask.any():
        return frame, 0
    return frame.loc[~mask].reset_index(drop=True), int(mask.sum())


ALL_SHEETS = "__all_sheets__"


def _row_key(frame: pd.DataFrame):
    """A set of row identities for a sheet, or None if it has no usable key.

    Used to spot two sheets holding the same records. Prefers an explicit id
    column; falls back to any near-unique column, which is what an export
    without an id still has.
    """
    for name in frame.columns:
        low = str(name).strip().lower()
        if low in ("_id", "_uuid", "id", "uuid", "_index", "submission id"):
            vals = frame[name].dropna()
            if len(vals):
                return set(vals.astype(str))
    for name in frame.columns:
        vals = frame[name].dropna()
        if len(vals) > 10 and vals.nunique() / len(vals) > 0.98:
            return set(vals.astype(str))
    return None


def combine_sheets(data: bytes, name: str, min_overlap: float = 0.7):
    """Stack every sheet that shares the workbook's dominant layout.

    A hand-maintained tracker is usually one sheet per week, and the analysis
    everybody wants is all of them together — the campaign tracker this was
    written for has eleven weekly sheets, a Videos sheet and a Summary. Reading
    one sheet at a time makes the tool show a twelfth of the campaign.

    The dominant layout is the most common set of column names. A sheet joins if
    it shares at least `min_overlap` of that set, which keeps a Summary sheet of
    pre-aggregated totals out of a pile of raw rows — including it would double
    count everything. Which sheets joined, and which did not, is returned so the
    page can say so rather than quietly leaving data out.

    The sheet name becomes a column. For a weekly tracker with no date column
    that is the only time axis in the file, and it is what makes a trend
    possible at all.
    """
    frames, skipped = {}, []
    with pd.ExcelFile(io.BytesIO(data)) as xl:
        names = list(xl.sheet_names)
        for sh in names:
            try:
                raw = pd.read_excel(xl, sheet_name=sh, na_values=NA_STRINGS,
                                    header=None)
            except Exception:
                skipped.append((sh, "could not be read"))
                continue
            f = _apply_header(raw).dropna(axis=0, how="all")
            real = [c for c in f.columns if not re.match(r"^Unnamed: \d+$", str(c))]
            if len(f) == 0 or len(real) < 2:
                skipped.append((sh, "no data rows"))
                continue
            frames[sh] = f[real]

    if not frames:
        return None, [], skipped
    # The dominant layout is the one holding the most ROWS, not the one on the
    # most sheets. Counting sheets picked a two-column "Metric | Value" shape
    # shared by three summary tabs of a fourteen-sheet report workbook, stacked
    # them into 25 rows, and skipped the eleven sheets with the actual data —
    # including a 9,299-row GA4 export.
    from collections import defaultdict
    import math
    rows_of, sheets_of = defaultdict(int), defaultdict(list)
    for sh, f in frames.items():
        key = frozenset(f.columns)
        rows_of[key] += len(f)
        sheets_of[key].append(sh)

    def table_score(n_cols: int, n_rows: int) -> float:
        """How much a sheet looks like the dataset rather than a log.

        Width counts far more than length. Neither rows nor cells work: a
        9,000-row three-column pipeline log beats a 200-row seven-column data
        table on both. Columns times log(rows) prefers the wide table while
        still favouring the fuller of two similar sheets.
        """
        return max(n_cols, 1) * math.log1p(max(n_rows, 1))

    modal_key = max(rows_of, key=lambda k: (table_score(len(k), rows_of[k]),
                                            len(sheets_of[k])))
    modal = set(modal_key)
    # One sheet is not a combination. A report workbook of unrelated tables has
    # nothing to stack, and pretending otherwise hides the data behind a
    # meaningless union.
    if len(sheets_of[modal_key]) < 2:
        biggest = max(frames, key=lambda sh: table_score(
            len(frames[sh].columns), len(frames[sh])))
        return None, [], [("__fallback__", biggest)]
    keep, out, seen_keys = [], [], []
    # Candidate sheets, widest first: when two sheets hold the same records, the
    # one with more columns is the annotated version and the one to keep.
    ordered = sorted(frames.items(), key=lambda kv: -len(kv[1].columns))
    for sh, f in ordered:
        overlap = len(modal & set(f.columns)) / max(len(modal), 1)
        if overlap < min_overlap:
            skipped.append((sh, f"different layout ({overlap:.0%} of the columns "
                                "in common)"))
            continue
        ids = _row_key(f)
        dup_of = None
        for other_sh, other_ids in seen_keys:
            if ids is not None and other_ids is not None:
                shared = len(ids & other_ids) / max(len(ids), 1)
                if shared > 0.5:
                    dup_of = (other_sh, shared)
                    break
        if dup_of:
            # Stacking a raw sheet with its own annotated copy silently doubles
            # every record — the trap a layout test cannot see, because the two
            # layouts are 98% identical by design.
            skipped.append((sh, f"the same records as {dup_of[0]} "
                                f"({dup_of[1]:.0%} of its ids appear there), so "
                                "stacking them would count everything twice"))
            continue
        if ids is not None:
            seen_keys.append((sh, ids))
        g = f.copy()
        g.insert(0, "Sheet", sh)
        keep.append(sh)
        out.append(g)
    if not out:
        return None, [], skipped
    combined = pd.concat(out, ignore_index=True, sort=False)
    return combined, keep, skipped


XLSFORM_SHEETS = {"survey", "choices", "settings"}


def looks_like_xlsform(sheets: list, columns) -> bool:
    """An XLSForm is a questionnaire definition, not data.

    NHW has hundreds of these beside their exports, and stacking `survey` with
    `choices` produced an 11,339-row "dataset" of form logic — rows about
    questions rather than answers. Recognising it is kinder than charting it.
    """
    have = {str(s).strip().lower() for s in (sheets or [])}
    if len(XLSFORM_SHEETS & have) >= 2:
        return True
    # `columns or []` is a truth test, and a pandas Index raises on one — which
    # sent the whole diagnosis into its except branch, so the files that most
    # needed an explanation silently got none.
    cols = {str(c).strip().lower().split("::")[0]
            for c in (list(columns) if columns is not None else [])}
    return {"type", "name"} <= cols and bool(cols & {"label", "hint",
                                                     "relevant", "appearance"})


def _numberish(v) -> bool:
    """Would this cell read as a number? Used to tell a heading from a value."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return True
    try:
        float(str(v).replace(",", "").strip())
        return True
    except (TypeError, ValueError):
        return False


def _as_year(label) -> int:
    """The label as a year if it plausibly is one, else 0.

    A wide table with 2019..2025 as columns is a legitimate shape that charts
    fine, so it must not be caught by the bare-number check.
    """
    try:
        return int(float(str(label).strip().split(" ")[0]))
    except (TypeError, ValueError):
        return 0


def describe_shape(prof, sheets: list) -> str:
    """A sentence explaining why a file has nothing to analyse, or "".

    An empty page is the worst outcome: the reader cannot tell a broken tool
    from a file that was never a table. Every case here came out of running the
    corpus — a pivot-style sheet with values scattered across cells, a form
    definition, a report preamble with no table under it.
    """
    usable = [c for c in prof.columns.values() if c.role in ANALYSABLE]
    if looks_like_xlsform(sheets, prof.df.columns):
        return ("This looks like an **XLSForm questionnaire definition** — the "
                "survey/choices/settings sheets that define a form, not the "
                "answers people gave. There is nothing to analyse here; upload "
                "the data export from Kobo instead.")
    if len(prof.columns) == 0 or prof.rows == 0:
        return ("No table was found in this sheet. It may be a pivot-style "
                "layout with values scattered across cells, or a cover page. "
                "The tool needs one row per record and one column per field.")
    # Column names that are bare numbers are cell values that got promoted:
    # a rating grid where the response options 1..5 are the columns and the
    # real heading row sits above them, merged. The tool cannot melt that, and
    # saying "pivot-style" is far better than charting "Average 3.0".
    numeric_names = [c for c in prof.columns.values()
                     if re.fullmatch(r"[-+]?\d+(\.\d+)?( \d+)?",
                                     str(c.name).strip())
                     and not (1900 <= _as_year(c.name) <= 2100)]
    if len(numeric_names) >= 3 and len(numeric_names) >= 0.4 * len(prof.columns):
        return (f"{len(numeric_names)} of {len(prof.columns)} column names are "
                "bare numbers, which means a row of values was read as the "
                "heading row. This is usually a rating grid or a pivot-style "
                "layout, where the answer options are the columns and the real "
                "heading sits above them in a merged cell. Pick a different "
                "sheet, or reshape it to one row per record before uploading.")
    blank = [c for c in prof.columns.values() if c.role == EMPTY]
    if len(blank) >= max(4, 0.6 * len(prof.columns)):
        return (f"**{len(blank)} of {len(prof.columns)} columns are completely "
                "empty.** That is the finding here: the export carries the "
                "full column list but almost none of it was filled in. The "
                "column map lists which ones.")
    if not usable:
        return ("This sheet has no analysable columns — every column is empty, "
                "holds a single repeated value, or is free text. It may be a "
                "cover page, a notes sheet, or a report preamble.")
    if len(prof.columns) <= 2 and prof.rows < 15:
        return ("This looks like a preamble rather than a data table: a couple "
                "of columns and a handful of rows. If the export has the table "
                "on another sheet, pick it in the sidebar.")
    # Countable means the charts would actually use it: enough levels to
    # compare, and enough rows filled in. A stray 13-row block in the corner of
    # an attendee sheet is not a dimension, and counting it as one kept the
    # page silent about a file with nothing to aggregate.
    counted = [c for c in usable
               if c.role in (CATEGORICAL, BINARY, GEO) and c.filled
               and c.unique < max(2, c.filled * 0.9)
               and c.unique <= MAX_CAT_LEVELS
               and c.filled >= max(5, 0.25 * prof.rows)]
    numbers = [c for c in usable if c.role == NUMERIC]
    if (prof.rows < 30 and not counted
            and not [c for c in usable if c.role in (NUMERIC, DATETIME)]):
        # A written document laid out in a grid: an outline, a theory of change,
        # a gaps-and-questions table. Every cell is distinct prose, so there is
        # nothing to count and nothing to total. Charting it would produce a row
        # of bars all of height one.
        return ("Every column here holds distinct free text, with no numbers "
                "and no repeated categories — this reads as a written document "
                "laid out in a grid (an outline, a plan, a set of notes) rather "
                "than a dataset. There is nothing to count or total. The "
                "**Brief** can still summarise it, and the table below shows it "
                "in full.")
    if not counted and not numbers:
        # A list where every row is its own thing: a collector list, a
        # schedule, an attendee sheet. A date column does not rescue it — a
        # trend through seven distinct rows is not a trend.
        return (f"{prof.rows:,} rows, and no column with numbers to total or "
                "repeated categories to count — this is a list rather than a "
                "dataset. The table below shows it in full, and the **Brief** "
                "can still read it.")
    return ""


def build(data: bytes, name: str, sheet=None, row_cap: int | None = None) -> Profile:
    """Upload bytes to a fully typed Profile. The one entry point the app calls."""
    joined, used, skipped = (None, [], [])
    note = ""
    if sheet == ALL_SHEETS:
        joined, used, skipped = combine_sheets(data, name)
        if joined is None and skipped and skipped[0][0] == "__fallback__":
            # Nothing to combine: show the sheet with the most data and say so,
            # rather than a union of whatever happened to share two columns.
            sheet = skipped[0][1]
            note = (f"The sheets have different layouts, so there is nothing to "
                    f"combine. Showing **{sheet}**, the one with the most data — "
                    "pick another in the sidebar.")
            skipped = []
    raw = joined if joined is not None else read_any(data, name, sheet)
    raw, totals = drop_total_rows(raw)
    # Blank rows go; wholly blank *columns* stay, and are marked EMPTY. A column
    # nobody filled in is a finding about the form, and dropping it here would
    # make it invisible on the column map where someone can act on it.
    raw = raw.dropna(axis=0, how="all")
    # Only drop an unnamed column if it is also empty. A wide table's row-label
    # column often has a blank heading, and dropping it took away the one
    # dimension the file had — leaving years as columns and nothing to name.
    keep = [c for c in raw.columns
            if not re.match(r"^Unnamed: \d+$", str(c))
            or raw[c].notna().any()]
    raw = raw[keep]
    truncated = 0
    if row_cap and len(raw) > row_cap:
        truncated = len(raw) - row_cap
        raw = raw.head(row_cap)
    df, cols = classify(raw)
    prof = Profile(df=df, columns=cols, source=name,
                   sheet=(f"{len(used)} sheets combined" if used else (sheet or "")),
                   truncated=truncated)
    prof.sheets_used, prof.sheets_skipped = used, skipped
    prof.total_rows_dropped, prof.sheet_note = totals, note
    # Metadata the export carried at the top of the file: property, date range.
    prof.source_notes = list(getattr(raw, "attrs", {}).get("notes", []))
    try:
        prof.shape_note = describe_shape(prof, sheet_names(data, name))
    except Exception:
        prof.shape_note = ""
    prof.kobo = kobo_groups(cols)
    prof.dropped = [c.name for c in cols.values() if c.role in (EMPTY, CONSTANT)]
    return prof
