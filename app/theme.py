"""Nigeria Health Watch brand system: tokens, fonts, CSS, and chart helpers.

Deliberately the same visual system as the GLIDE dashboard
(github.com/GentlePrince-ng/glide-dashboard) — same brand greens, same Poppins
embedding, same white chart surface — so NHW's Streamlit apps read as siblings.

Three rules hold the palette together:

1. **Green is chrome, never a data series.** NHW's brand is green-dominant, so a
   green line beside a green header reads as branding rather than data. The one
   exception is the sequential ramp, where a single hue *is* the correct encoding
   for magnitude.
2. **Brand #30B349 never carries a mark or text.** It measures 2.74:1 on white.
   The deeper brand green #166335 (7.32:1) does that work; #30B349 stays identity.
3. **Series order is load-bearing.** The order below passes all six checks of the
   dataviz validator on a white surface — worst adjacent pair ΔE 13.0 deuteranopia,
   30.6 normal vision. Blue next to violet, or red next to rose, fails badly.

Status colours are reserved for target attainment and are never reused as a
series. All four clear AA text contrast on white, which matters because the focus
cards print them as ink on a pale ground rather than white-on-colour — the earlier
version put white text on #A8A818 at 2.2:1, which was unreadable.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

STATIC = Path(__file__).resolve().parents[1] / "static"
LOGO = STATIC / "nhw-logo.png"
FONT_STACK = 'Poppins, system-ui, -apple-system, "Segoe UI", sans-serif'

# ── brand (chrome and identity only) ────────────────────────────────────────
GREEN_BRAND = "#30B349"   # brand primary one — identity, gradients, never a mark
GREEN_DEEP = "#166335"    # marks, headings, on-target
GREEN_ACCENT = "#8CC43D"  # brand primary two — sparing
GREEN_PALE = "#F1F8F3"    # card ground
GREEN_BAND = "#78C084"

# Kept as aliases: earlier pages import these names.
GREEN_MID = GREEN_BRAND

# ── surface and ink ─────────────────────────────────────────────────────────
SURFACE = "#FFFFFF"
PANEL = "#F7FAF8"
GRID = "#E2E8E3"
AXIS = "#B4C4B8"
BORDER = "#E2E8E3"
INK = "#080A09"
INK2 = "#2A332D"
MUTED = "#5A665E"

# ── target attainment (reserved; never a series) ────────────────────────────
MET = "#166335"      # at or above target
CLOSE = "#946200"    # within 20%
BEHIND = "#B42318"   # further behind than that
NO_TARGET = "#6E7A72"

# Back-compat names used by the older pages.
ON_TARGET, NEAR_TARGET, UNDER_TARGET = MET, CLOSE, BEHIND

# ── sequential ramp — one hue, light to dark, for magnitude ─────────────────
RAMP = ["#EAF5EC", "#C2E3CB", "#8ACCA0", "#47A868", "#166335"]

# ── categorical series — green-free; this order is validated ────────────────
S_MAIN = "#2563EB"    # GA4 legacy property
S_MFN = "#B45309"     # MetricsForNews, frozen
S_APEX = "#7C3AED"    # GA4 new-site property
S_VIDEO = "#DC2626"   # YouTube
S_AUDIO = "#0891B2"   # BeyondWords
S_MAIL = "#BE185D"    # Mailchimp
SERIES = [S_MAIN, S_MFN, S_APEX, S_VIDEO, S_AUDIO, S_MAIL]

SOURCE_COLOURS = {"ga4_main": S_MAIN, "ga4_apex": S_APEX, "mfn": S_MFN,
                  "beyondwords": S_AUDIO, "youtube": S_VIDEO, "mailchimp": S_MAIL}

SOURCE_LABELS = {"ga4_main": "Legacy site (GA4)", "ga4_apex": "New site (GA4)",
                 "mfn": "MetricsForNews", "beyondwords": "Audio",
                 "youtube": "YouTube", "mailchimp": "Newsletter"}

POPPINS_WEIGHTS = {"Regular": 400, "Medium": 500, "SemiBold": 600, "Bold": 700}


# ═══ fonts and chrome ══════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _font_css() -> str:
    """Embed Poppins as base64 @font-face.

    The obvious route — [[theme.fontFaces]] pointing at static/ — needs
    server.enableStaticServing, and any [server] section makes Streamlit
    Community Cloud restart onto port 8502 while its proxy serves only 8501: the
    app runs but is unreachable. Inlining ~31 KB of woff2 avoids the config
    entirely and behaves identically locally and hosted.

    @font-face definitions ONLY. `theme.font = "Poppins"` in config.toml is what
    applies the family. Never set font-family on a selector broad enough to reach
    Streamlit's Material Symbols spans — their glyphs are ligatures, so an
    expander caret renders as the literal text "arrow_right" over its own label.
    """
    faces = []
    for name, weight in POPPINS_WEIGHTS.items():
        p = STATIC / f"Poppins-{name}.woff2"
        if not p.exists():
            continue
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        faces.append("@font-face{font-family:'Poppins';font-style:normal;"
                     f"font-weight:{weight};font-display:swap;"
                     f"src:url(data:font/woff2;base64,{b64}) format('woff2');}}")
    return "".join(faces)


_CSS = """
<style>
__FONTS__
:root {
  --nhw-brand: __BRAND__; --nhw-deep: __DEEP__; --nhw-accent: __ACCENT__;
  --nhw-ink: __INK__; --nhw-ink2: __INK2__; --nhw-muted: __MUTED__;
  --nhw-border: __BORDER__; --nhw-panel: __PANEL__; --nhw-pale: __PALE__;
  --nhw-met: __MET__; --nhw-close: __CLOSE__; --nhw-behind: __BEHIND__;
}

/* ── headings: tighter, deeper, less default-Streamlit ─────────────────── */
h1 { color: var(--nhw-deep); letter-spacing: -0.02em; font-weight: 700;
     font-size: 1.55rem !important; padding-bottom: 0.2rem !important; }
h2, h3 { color: var(--nhw-deep); letter-spacing: -0.012em; font-weight: 600; }
h4 { color: var(--nhw-ink2); font-weight: 600; letter-spacing: -0.008em; }
a { text-decoration: none; }
a:hover { text-decoration: underline; }

/* Give the page some air without touching Streamlit's DOM structure. */
.block-container { padding-top: 2.2rem; max-width: 1500px; }

/* ── masthead ──────────────────────────────────────────────────────────── */
.nhw-band {
  background: linear-gradient(100deg, __DEEP__ 0%, #1F8B45 52%, __BRAND__ 100%);
  color: #fff; padding: 0.95rem 1.2rem 1rem; border-radius: 8px;
  margin: 0 0 1.15rem; display: flex; align-items: baseline;
  gap: 0.85rem; flex-wrap: wrap;
}
.nhw-band .t { font-size: 1.12rem; font-weight: 650; letter-spacing: -0.015em; }
.nhw-band .s { font-size: 0.8rem; opacity: 0.93; font-weight: 400; }

/* ── section rule: an eyebrow label on a hairline ──────────────────────── */
.nhw-sec { display: flex; align-items: center; gap: 0.7rem;
           margin: 1.6rem 0 0.75rem; }
.nhw-sec .lbl { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.09em;
                text-transform: uppercase; color: var(--nhw-deep);
                white-space: nowrap; }
.nhw-sec .rule { flex: 1; height: 1px; background: var(--nhw-border); }
.nhw-sec .note { font-size: 0.75rem; color: var(--nhw-muted); white-space: nowrap; }

/* ── KPI card ──────────────────────────────────────────────────────────── */
.kpi { display: flex; flex-direction: column; gap: 0.15rem; }
.kpi .l { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.055em;
          text-transform: uppercase; color: var(--nhw-muted); line-height: 1.3; }
.kpi .v { font-size: 1.72rem; font-weight: 650; line-height: 1.12;
          color: var(--nhw-ink); font-variant-numeric: tabular-nums;
          letter-spacing: -0.025em; }
.kpi .v.sm { font-size: 1.3rem; }
.kpi .n { font-size: 0.72rem; color: var(--nhw-muted); line-height: 1.35; }
/* Two-part breakdown at the foot of a KPI card. Every card in the row gets one,
   whether or not it is the split that was asked for — Streamlit columns size to
   their own content, so a strip on three of six would leave the other three
   visibly short. Each therefore carries its own honest pair rather than padding. */
.kpi-split { display: flex; gap: 0.55rem; border-top: 1px solid var(--nhw-border);
             margin-top: 0.4rem; padding-top: 0.35rem; }
.kpi-split > span { flex: 1 1 0; min-width: 0; display: flex;
                    flex-direction: column; gap: 0.02rem; }
/* min-height on both text rows so the strip is one fixed height whether a label
   wraps to two lines or not, and whether it carries a note or not. Without it the
   six cards ended at three different heights — "Africa editions" wrapping while
   "Nigeria" did not was enough to knock the row out of line. */
.kpi-split .sl { font-size: 0.61rem; font-weight: 600; letter-spacing: 0.045em;
                 text-transform: uppercase; color: var(--nhw-muted);
                 line-height: 1.25; min-height: 1.53rem; }
.kpi-split .sv { font-size: 0.98rem; font-weight: 650; color: var(--nhw-ink2);
                 font-variant-numeric: tabular-nums; line-height: 1.2;
                 letter-spacing: -0.015em; }
.kpi-split .sn { font-size: 0.6rem; color: var(--nhw-muted); line-height: 1.25;
                 min-height: 1.5rem; }

.chip { display: inline-flex; align-items: center; gap: 0.22rem;
        font-size: 0.72rem; font-weight: 600; padding: 0.08rem 0.4rem;
        border-radius: 4px; font-variant-numeric: tabular-nums;
        width: fit-content; }
.chip.up   { color: __MET__;    background: #EAF5EC; }
.chip.down { color: __BEHIND__; background: #FDECEA; }
.chip.flat { color: var(--nhw-muted); background: var(--nhw-panel); }
/* What the change is measured against. Lighter and unbolded so the chip still
   reads as one number at a glance — the comparison is context, not the value —
   but present, because an arrow without a period is not a fact. */
.chip .vs  { font-weight: 400; font-size: 0.66rem; opacity: 0.78;
             margin-left: 0.1rem; letter-spacing: 0.005em; }

/* ── focus-area cards: ink on pale, status as a rail + progress bar ─────
   The previous version set white text on the status colour, which at #A8A818
   measured 2.2:1. Colour now sits in the rail and the bar, where it carries
   the same information at full strength and the number stays readable. */
/* Adaptive: as many columns as fit at 190px, so the cards stay legible on a
   laptop and use the width on a wide screen. A fixed five-across was tried and
   reverted — it looked tidy at 1600px and cramped everywhere else. */
.fa-grid { display: grid; gap: 0.6rem;
           grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); }
.fa { border: 1px solid var(--nhw-border); border-left: 4px solid var(--nhw-muted);
      border-radius: 7px; padding: 0.6rem 0.75rem 0.7rem; background: #fff;
      display: flex; flex-direction: column; gap: 0.28rem; }
.fa .n { font-size: 0.76rem; font-weight: 600; color: var(--nhw-ink2);
         line-height: 1.25; }
.fa .row { display: flex; align-items: baseline; gap: 0.35rem; }
.fa .v { font-size: 1.5rem; font-weight: 650; line-height: 1;
         color: var(--nhw-ink); font-variant-numeric: tabular-nums;
         letter-spacing: -0.02em; }
.fa .of { font-size: 0.76rem; color: var(--nhw-muted);
          font-variant-numeric: tabular-nums; }
.fa .track { height: 5px; border-radius: 3px; background: #EDF1EE;
             overflow: hidden; margin-top: 0.1rem; }
.fa .fill { height: 100%; border-radius: 3px; }
.fa .t { font-size: 0.7rem; color: var(--nhw-muted);
         font-variant-numeric: tabular-nums; }
.fa.none { background: var(--nhw-panel); }
.fa.none .v { color: var(--nhw-ink2); }

/* ── tally strip: shows an identity, so a gap reads as accounted for ───── */
.tally { display: flex; align-items: baseline; gap: 0.45rem; flex-wrap: wrap;
         border: 1px solid var(--nhw-border); border-radius: 7px;
         background: var(--nhw-panel); padding: 0.5rem 0.8rem;
         margin: 0.55rem 0 0.3rem; }
.tally .n { font-size: 1.05rem; font-weight: 650; color: var(--nhw-ink);
            font-variant-numeric: tabular-nums; letter-spacing: -0.015em; }
.tally .n.gap { color: __CLOSE__; }
.tally .l { font-size: 0.78rem; color: var(--nhw-muted); }
.tally .op { font-size: 0.95rem; color: var(--nhw-muted); font-weight: 500;
             padding: 0 0.15rem; }

/* ── insight callouts ──────────────────────────────────────────────────── */
.ins { border-left: 3px solid var(--nhw-brand); background: var(--nhw-panel);
       padding: 0.5rem 0.7rem; margin: 0.35rem 0; font-size: 0.875rem;
       line-height: 1.5; border-radius: 0 5px 5px 0; color: var(--nhw-ink2); }
.ins.good { border-left-color: __MET__;   background: #F2F9F4; }
.ins.warn { border-left-color: __CLOSE__; background: #FDF8EE; }
.ins.bad  { border-left-color: __BEHIND__; background: #FDF2F0; }
.ins b, .ins strong { color: var(--nhw-ink); font-weight: 650; }

/* ── preset groups inside the period picker ────────────────────────────
   The picker holds three labelled groups of buttons, and at Streamlit's
   default block gap they sat further apart than the groups were tall. The
   gap is narrowed only inside the expander, so the rest of the page keeps
   its breathing room. Both selectors are Streamlit internals: if either
   testid changes the spacing simply reverts to default, which is why this
   is a rule about gaps and not about position. */
/* The period picker's group labels ("Quick periods", "Fiscal years", "Calendar
   years"), written as `###### …` so Streamlit emits the element and this only
   restyles it into an eyebrow. h6 is unused elsewhere in the app.

   Do NOT narrow the gap on [data-testid="stVerticalBlock"] to tighten these
   further. Tried, and it collapsed each label's wrapper to height 0 so the
   button row was laid out on top and every label lost its bottom half —
   measured 18px of label at y=107 against a button at y=116. The spacing here
   comes from removing the empty st.write("") between groups instead, which is
   most of what was there and costs nothing. */
[data-testid="stExpanderDetails"] h6 {
  font-size: 0.68rem !important; font-weight: 600; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--nhw-muted) !important;
  margin: 0; padding: 0; line-height: 1.3; }

/* ── period chip in the picker header ──────────────────────────────────── */
.per { display: flex; align-items: baseline; gap: 0.55rem; flex-wrap: wrap; }
.per .d { font-size: 1rem; font-weight: 650; color: var(--nhw-deep);
          font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }
.per .m { font-size: 0.78rem; color: var(--nhw-muted); }

/* Streamlit's own metric, where it is still used, gets aligned digits. */
div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums;
                                   letter-spacing: -0.02em; }
div[data-testid="stMetricLabel"] p { font-size: 0.72rem !important;
  text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;
  color: __MUTED__ !important; }

/* Dataframes read as data, not as a widget. */
div[data-testid="stDataFrame"] { border-radius: 7px; }
</style>
"""


def inject() -> None:
    """Fonts and CSS. Safe to call more than once."""
    css = (_CSS.replace("__FONTS__", _font_css())
           .replace("__BRAND__", GREEN_BRAND).replace("__DEEP__", GREEN_DEEP)
           .replace("__ACCENT__", GREEN_ACCENT).replace("__PALE__", GREEN_PALE)
           .replace("__INK2__", INK2).replace("__INK__", INK)
           .replace("__MUTED__", MUTED).replace("__BORDER__", BORDER)
           .replace("__PANEL__", PANEL).replace("__MET__", MET)
           .replace("__CLOSE__", CLOSE).replace("__BEHIND__", BEHIND))
    st.markdown(css, unsafe_allow_html=True)
    _register_chart_theme()


def chrome() -> None:
    """Everything brand-level, once, before any page body renders."""
    inject()
    if LOGO.exists():
        # Per the brand manual's clear-space rule: alone, unboxed, at its supplied
        # colours. st.logo reads from disk, so no static serving is needed.
        st.logo(str(LOGO), size="large",
                link="https://www.nigeriahealthwatch.com")


# ═══ Altair theme — every chart inherits it, so none needs restyling ═══════
def _chart_config() -> dict:
    return {"config": {
        "background": SURFACE,
        "font": FONT_STACK,
        "padding": {"left": 2, "right": 8, "top": 4, "bottom": 2},
        "view": {"stroke": None, "continuousWidth": 400, "continuousHeight": 220},
        "title": {"font": FONT_STACK, "fontSize": 13, "fontWeight": 600,
                  "color": INK2, "anchor": "start", "offset": 10},
        "axis": {"labelFont": FONT_STACK, "labelFontSize": 11.5,
                 "labelColor": MUTED, "titleFont": FONT_STACK,
                 "titleFontSize": 11.5, "titleColor": MUTED,
                 "titleFontWeight": 500, "titlePadding": 8,
                 "domainColor": AXIS, "tickColor": AXIS, "tickSize": 4,
                 "gridColor": GRID, "gridWidth": 1, "labelPadding": 5},
        "axisX": {"grid": False, "domain": True},
        "axisY": {"grid": True, "domain": False, "ticks": False,
                  "labelColor": INK2},
        "legend": {"labelFont": FONT_STACK, "labelFontSize": 11.5,
                   "labelColor": INK2, "titleFont": FONT_STACK,
                   "titleFontSize": 11, "titleColor": MUTED,
                   "titleFontWeight": 500, "orient": "top",
                   "direction": "horizontal", "symbolType": "square",
                   "symbolSize": 90, "offset": 6, "padding": 0},
        "range": {"category": SERIES, "heatmap": RAMP, "ramp": RAMP},
        "bar": {"cornerRadiusEnd": 4},
        "line": {"strokeWidth": 2},
        "point": {"size": 60, "filled": True},
        "text": {"font": FONT_STACK, "fontSize": 11.5, "color": INK2},
    }}


def _register_chart_theme() -> None:
    """Altair 5.5 renamed themes -> theme. Support both, and never let a theme
    failure take the page down — a chart in default styling still carries the
    data."""
    try:
        if hasattr(alt, "theme"):
            if "nhw" not in alt.theme.names():
                alt.theme.register("nhw", enable=True)(_chart_config)
            alt.theme.enable("nhw")
        else:                                       # Altair < 5.5
            alt.themes.register("nhw", _chart_config)
            alt.themes.enable("nhw")
    except Exception:
        pass


# ═══ layout pieces ═════════════════════════════════════════════════════════
def day(d) -> str:
    """A date as a person writes it: "6 Jan 2025".

    DuckDB DATE columns arrive as pandas Timestamps, so interpolating one into an
    f-string prints "2025-01-06 00:00:00" — a midnight that means nothing, on a
    column that has no time component at all. Every date shown as text goes
    through here.

    Uses lstrip("0") rather than "%-d": that directive is a glibc extension and
    raises on Windows, so it would work on Streamlit Cloud and fail locally.
    """
    try:
        if d is None or pd.isna(d):
            return "—"
        return pd.Timestamp(d).strftime("%d %b %Y").lstrip("0")
    except (TypeError, ValueError):
        return str(d)


def band(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="nhw-band"><span class="t">{title}</span>'
                f'<span class="s">{subtitle}</span></div>', unsafe_allow_html=True)


def section(label: str, note: str = "") -> None:
    """An eyebrow label on a hairline rule. Replaces st.divider + st.subheader,
    which together ate about 70px of vertical space per section."""
    tail = f'<span class="note">{note}</span>' if note else ""
    st.markdown(f'<div class="nhw-sec"><span class="lbl">{label}</span>'
                f'<span class="rule"></span>{tail}</div>', unsafe_allow_html=True)


def delta_chip(now: float | None, then: float | None, unit: str = "%",
               higher_is_better: bool = True, against: str = "",
               detail: str = "") -> str:
    """A signed change against the comparison period, or an honest blank.

    Returns HTML. When `then` is missing or zero there is no comparison to make
    and this returns a neutral chip saying so — inventing 'up 100%' from a zero
    base is the kind of figure that ends up in a funder report.

    `against` names what the change is measured against and is printed on the
    chip. "▼ 17%" alone is not a fact: down against last month and down against
    last year are different claims, and a reader who has to scroll to the period
    picker to find out which will assume whichever suits them. `detail` carries
    the exact dates into the hover, so the chip stays short without the precise
    window becoming unavailable.
    """
    vs = (f'<span class="vs">vs {against}</span>' if against else "")
    tip = f' title="{detail}"' if detail else ""
    if now is None or then is None or pd.isna(now) or pd.isna(then) or not then:
        return f'<span class="chip flat"{tip}>no prior period</span>'
    pct = (float(now) - float(then)) / abs(float(then)) * 100
    if abs(pct) < 0.5:
        return f'<span class="chip flat"{tip}>level{vs}</span>'
    up = pct > 0
    good = up if higher_is_better else not up
    arrow = "▲" if up else "▼"
    return (f'<span class="chip {"up" if good else "down"}"{tip}>{arrow} '
            f'{abs(pct):.0f}{unit}{vs}</span>')


def kpi_card(col, label: str, value: str, delta: str = "", note: str = "",
             small: bool = False, spark_df=None, spark_y: str = "",
             spark_x: str = "", colour: str = GREEN_DEEP,
             split: list[tuple[str, str, str]] | None = None) -> None:
    """One bordered card: eyebrow label, big value, optional change chip, note,
    optional sparkline, optional two-part breakdown at the foot.

    The border container wraps the chart too, which an HTML card could not do —
    a Streamlit chart cannot live inside a markdown div. The split is emitted
    after the chart for the same reason, so the card reads headline, then trend,
    then breakdown.

    `split` is a list of (label, value, note) — two entries in practice.
    """
    with col:
        with st.container(border=True):
            bits = [f'<div class="kpi"><span class="l">{label}</span>'
                    f'<span class="v{" sm" if small else ""}">{value}</span>']
            if delta:
                bits.append(delta)
            if note:
                bits.append(f'<span class="n">{note}</span>')
            bits.append("</div>")
            st.markdown("".join(bits), unsafe_allow_html=True)
            if spark_df is not None and len(spark_df) > 1:
                chart(spark(spark_df, spark_x, spark_y, colour))
            if split:
                # The note row is always emitted, empty or not, so its reserved
                # height applies to every card in the row.
                cells = "".join(
                    f'<span><span class="sl">{sl}</span>'
                    f'<span class="sv">{sv}</span>'
                    f'<span class="sn">{sn or "&nbsp;"}</span></span>'
                    for sl, sv, sn in split)
                st.markdown(f'<div class="kpi-split">{cells}</div>',
                            unsafe_allow_html=True)


_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)


def tally(parts: list[tuple[int, str, bool]], total: int, total_label: str) -> None:
    """Render `a + b = total` so a gap between two figures reads as accounted for
    rather than as a contradiction.

    The focus-area cards sum to fewer articles than the published count, because
    an untagged article belongs to no focus area and so appears on no card. That
    is correct, and it looked like a bug for exactly as long as the page declined
    to do the subtraction out loud.

    parts: (value, label, is_gap) — is_gap tints the number amber.
    """
    bits = []
    for i, (v, label, gap) in enumerate(parts):
        if i:
            bits.append('<span class="op">+</span>')
        bits.append(f'<span class="n{" gap" if gap else ""}">{v:,}</span>'
                    f'<span class="l">{label}</span>')
    bits.append('<span class="op">=</span>')
    bits.append(f'<span class="n">{total:,}</span>'
                f'<span class="l">{total_label}</span>')
    st.markdown(f'<div class="tally">{"".join(bits)}</div>',
                unsafe_allow_html=True)


def insight(text: str, kind: str = "") -> None:
    """Markdown emphasis is converted by hand.

    Streamlit does not parse markdown inside raw HTML, so **like this** rendered
    as literal asterisks. The insight strings are written in markdown because
    that is what every other caption on the page uses, so the conversion happens
    here rather than making every rule author HTML."""
    st.markdown(f'<div class="ins {kind}">{_BOLD.sub(r"<b>\1</b>", text)}</div>',
                unsafe_allow_html=True)


def target_status(published: float, target) -> tuple[str, str]:
    """(colour, word). Green at or above target, amber within 20%, red below
    that, grey where no target was ever set."""
    if target is None or target != target or target <= 0:
        return NO_TARGET, "no target"
    pct = published / target
    if pct >= 1:
        return MET, "met"
    if pct >= 0.8:
        return CLOSE, "close"
    return BEHIND, "behind"


# Kept for callers that only want the colour.
def target_colour(published: float, target) -> str:
    return target_status(published, target)[0]


def focus_grid(rows, days_left: int | None = None,
               full_year: bool = True) -> None:
    """rows: (name, published, target_exact|None, target_whole|None).

    Every area renders, including the ones with no target and no articles — the
    absence is itself information, and the Looker page showed them.

    Two figures for the target, deliberately. `target_whole` is what the card
    prints, because an article is indivisible and "of 3.2" is not a thing anyone
    can publish. `target_exact` drives the percentage and the on-target colour, so
    the arithmetic stays right underneath a rounded label.

    `full_year` picks the wording. Over a whole reporting year the useful sentence
    is the pace still needed ("9 to go, 1.4/month"), because there is a year-end
    to hit. Over a shorter window it is whether output is keeping up with the
    months covered — "year end" would be meaningless there.
    """
    cells = []
    for row in rows:
        name, pub, tgt, tgt_whole = (row if len(row) == 4 else (*row, None))
        # "Configured" and "greater than zero" are different questions, and
        # conflating them sent WASH — whose 6 a year floors to 0 a month, so its
        # whole allocation sits in April — into the "no target set" branch for
        # every other month of the year.
        configured = tgt is not None and tgt == tgt
        whole = int(round(tgt_whole if (tgt_whole is not None and tgt_whole == tgt_whole)
                          else (tgt if configured else 0)))

        if configured and whole <= 0:
            # Either the months covered carry a zero target (an annual figure
            # under 12 floors to nothing outside April), or under one article is
            # expected. Neither supports a percentage, and colouring it red would
            # invent a shortfall out of a rounding boundary.
            note = ("none expected in this period" if tgt < 0.05
                    else "under 1 expected in this period")
            cells.append(
                f'<div class="fa none" style="border-left-color:{NO_TARGET}">'
                f'<span class="n">{name}</span>'
                f'<span class="row"><span class="v">{int(pub):,}</span></span>'
                f'<span class="t">{note}</span></div>')
            continue

        colour, word = target_status(pub, tgt)
        if configured:
            pct = pub / tgt * 100
            gap = whole - int(pub)
            if full_year:
                if gap <= 0:
                    tline = f"{pct:.0f}% · target {whole:,} met"
                elif days_left and days_left > 0:
                    per_month = gap / max(days_left / 30.44, 0.5)
                    tline = f"{pct:.0f}% · {gap} to go, {per_month:.1f}/month"
                else:
                    tline = f"{pct:.0f}% · {gap} short at year end"
            elif gap <= 0:
                tline = f"{pct:.0f}% · ahead of the {whole:,} expected"
            elif not pub:
                # "3 behind 3 expected" would say the same number twice.
                tline = f"none of the {whole:,} expected"
            else:
                tline = f"{pct:.0f}% · {gap} behind the {whole:,} expected"
            bar_html = (f'<div class="track"><div class="fill" style="width:'
                        f'{min(pct, 100):.0f}%;background:{colour}"></div></div>')
            cells.append(
                f'<div class="fa" style="border-left-color:{colour}">'
                f'<span class="n">{name}</span>'
                f'<span class="row"><span class="v">{int(pub):,}</span>'
                f'<span class="of">of {whole:,}</span></span>'
                f'{bar_html}<span class="t">{tline}</span></div>')
        else:
            cells.append(
                f'<div class="fa none" style="border-left-color:{NO_TARGET}">'
                f'<span class="n">{name}</span>'
                f'<span class="row"><span class="v">{int(pub):,}</span></span>'
                f'<span class="t">no target set</span></div>')
    st.markdown(f'<div class="fa-grid">{"".join(cells)}</div>',
                unsafe_allow_html=True)


# ═══ charts ════════════════════════════════════════════════════════════════
def chart(ch, **kw) -> None:
    """Render an Altair chart with the NHW theme intact.

    `theme=None` is the whole point of this wrapper. Streamlit's default,
    `theme="streamlit"`, merges its own Vega config over the chart's — which
    silently overrode the explicit `height` on every rect mark here, compressing
    a four-row calendar into about 40px and stacking its row labels on top of one
    another. With it off, charts render exactly as specified.
    """
    st.altair_chart(ch, width="stretch", theme=None, **kw)


def geo_chart(ch) -> None:
    """Render a map, bypassing Streamlit's Altair data handling.

    `st.altair_chart` hands the spec to Streamlit's marshaller, which lifts the
    top-level `data`/`datasets` out and re-serialises them as Arrow tables. A
    GeoJSON FeatureCollection does not survive that: the map arrived in the
    browser with its geometry gone, drew nothing, and captioned its legend
    "NaN" — while the same chart exported to SVG perfectly, because vl-convert
    never goes through Streamlit.

    So the dataset is inlined into each layer by hand and the top-level keys are
    removed. With nothing at the top level to lift, the marshaller passes the
    spec through untouched and Vega reads the GeoJSON as it is written.
    """
    spec = ch.to_dict()
    datasets = spec.pop("datasets", None) or {}
    top = spec.pop("data", None)

    def resolve(ref):
        if isinstance(ref, dict) and ref.get("name") in datasets:
            out = {"values": datasets[ref["name"]]}
            if "format" in ref:
                out["format"] = ref["format"]
            return out
        return ref

    layers = spec.get("layer")
    if layers:
        for layer in layers:
            layer["data"] = resolve(layer.get("data") or top)
    else:                        # a single-view map: wrap it so data can live
        view = {k: spec.pop(k) for k in ("mark", "encoding", "transform")
                if k in spec}
        view["data"] = resolve(top)
        spec["layer"] = [view]
    # width="content": use_container_width is deprecated after 2025-12-31 and
    # was filling the deployment logs with the warning on every map render.
    st.vega_lite_chart(spec, width="content", theme=None)


def spark(df, x: str, y: str, colour: str = GREEN_DEEP, height: int = 34):
    """A bare trend strip for a KPI card. No axes, no labels — the number above
    it is the value; this only carries shape."""
    base = alt.Chart(df).mark_area(
        line={"color": colour, "strokeWidth": 1.6}, opacity=0.16, color=colour,
    ).encode(
        x=alt.X(f"{x}:T", axis=None),
        y=alt.Y(f"{y}:Q", axis=None, scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip(f"{x}:T", title="date"),
                 alt.Tooltip(f"{y}:Q", format=",", title="value")],
    )
    return base.properties(height=height).configure_view(stroke=None)


def bar(df, x: str, y: str, colour: str = GREEN_DEEP, height: int | None = None,
        fmt: str = ",", labels: bool = True):
    """Horizontal magnitude bars: thin marks, rounded data-ends, values printed
    at the end of each bar so the axis is a reference rather than a lookup."""
    df = df.copy()
    base = alt.Chart(df).encode(
        y=alt.Y(f"{y}:N", sort="-x", title=None),
        tooltip=[alt.Tooltip(f"{y}:N", title=""),
                 alt.Tooltip(f"{x}:Q", format=fmt, title="")],
    )
    # Headroom for the end-of-bar label. Without it the longest bar's own value
    # is clipped by the plot edge — the one number a reader most wants to read.
    # (Divergence from the Observatory copy of this theme, which never draws a
    # bar wide enough to hit it.)
    top = pd.to_numeric(df[x], errors="coerce").max()
    pad = alt.Scale(domain=[0, float(top) * 1.13]) if labels and pd.notna(top) \
        and top > 0 else alt.Undefined
    marks = base.mark_bar(cornerRadiusEnd=4, height=13, color=colour).encode(
        x=alt.X(f"{x}:Q", title=None, scale=pad,
                axis=alt.Axis(grid=True, tickCount=4)))
    out = marks
    if labels and len(df) <= 25:
        out = marks + base.mark_text(
            align="left", dx=5, fontSize=11.5, color=INK2, fontWeight=500,
        ).encode(x=alt.X(f"{x}:Q", title=None),
                 text=alt.Text(f"{x}:Q", format=fmt))
    # Step, for the same reason as calendar(): one bar band is 26px whatever the
    # container does, so ten categories are never crushed into six bars' worth of
    # space.
    return out.properties(height=height or alt.Step(26))


def trend(df, x: str, y: str, colour: str = GREEN_DEEP, height: int = 240,
          series: str | None = None, y_title: str | None = None,
          x_format: str = "%b %y"):
    """Change over time. One series gets an area with a 2px line and no legend —
    the section title names it. Two or more get lines plus a legend.

    `x_format` has to follow the grain of the data: at weekly grain the monthly
    default prints "Feb 26" five times in a row, which reads as a broken axis
    rather than as five distinct weeks.
    """
    enc = dict(
        x=alt.X(f"{x}:T", title=None, axis=alt.Axis(format=x_format)),
        y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(zero=True)),
    )
    if series:
        doms = sorted(df[series].dropna().unique().tolist())
        d = df.copy()
        d["_lbl"] = d[series].map(lambda s: SOURCE_LABELS.get(s, s))
        labels = [SOURCE_LABELS.get(s, s) for s in doms]
        ch = alt.Chart(d).mark_line(strokeWidth=2, point=False).encode(
            **enc,
            color=alt.Color("_lbl:N", title=None,
                            scale=alt.Scale(domain=labels,
                                            range=[SOURCE_COLOURS.get(s, SERIES[i % 6])
                                                   for i, s in enumerate(doms)])),
            tooltip=[alt.Tooltip(f"{x}:T", title="period"),
                     alt.Tooltip("_lbl:N", title="source"),
                     alt.Tooltip(f"{y}:Q", format=",", title="value")])
    else:
        ch = alt.Chart(df).mark_area(
            line={"color": colour, "strokeWidth": 2}, opacity=0.15, color=colour,
        ).encode(**enc, tooltip=[alt.Tooltip(f"{x}:T", title="period"),
                                 alt.Tooltip(f"{y}:Q", format=",", title="value")])
    return ch.properties(height=height)


def calendar(df, date_col: str = "metric_date", value_col: str = "pageviews",
             height: int | None = None, title: str = "", grain: str = "day"):
    """A calendar grid, shaded by magnitude.

    `grain="day"`  — one row per month, one column per day of the month.
    `grain="month"`— one row per year, one column per month. Use this above about
                     18 months: 36 day-rows either need 650px or get crushed to
                     3px each, and a calendar you cannot read is worse than a
                     line chart.

    A weekday-aligned contribution grid was tried first and abandoned: rect marks
    on a temporal axis have no band width, so every month collapsed into a stripe.
    Month rows against day-of-month columns is a true band scale on both axes, so
    the cells are always square-ish whether the range is a fortnight or four
    years, and rows stay comparable — the same day of two months sits in the same
    column. Weekday is in the tooltip, which is where the weekend dip can still
    be read.

    Empty days render as empty cells rather than closing the gap. That is the
    point of the chart: 11 May – 11 June 2026 shows as a blank run, and a hole in
    the data otherwise looks exactly like a fall in readership.
    """
    d = df[[date_col, value_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.groupby(date_col, as_index=False)[value_col].sum()
    if d.empty:
        return None
    full = pd.date_range(d[date_col].min(), d[date_col].max(), freq="D")
    d = d.set_index(date_col).reindex(full).rename_axis(date_col).reset_index()
    d[value_col] = d[value_col].fillna(0)

    if grain == "month":
        d["col"] = d[date_col].dt.month
        d["label"] = d[date_col].dt.strftime("%Y")
        d = d.groupby(["label", "col"], as_index=False)[value_col].sum()
        col_axis = alt.Axis(grid=False, labelAngle=0, labelFontSize=10,
                            ticks=False, domain=False,
                            labelExpr="['J','F','M','A','M','J','J','A','S','O',"
                                      "'N','D'][datum.value - 1]")
        tip = [alt.Tooltip("label:N", title="year"),
               alt.Tooltip("col:O", title="month"),
               alt.Tooltip(f"{value_col}:Q", format=",", title=value_col)]
        rows_per_px = 22
    else:
        d["col"] = d[date_col].dt.day
        d["label"] = d[date_col].dt.strftime("%b %Y")
        # 31 labels crowd at narrow widths; every other one locates a day.
        col_axis = alt.Axis(grid=False, labelAngle=0, labelFontSize=10,
                            ticks=False, domain=False,
                            labelExpr="datum.value % 2 == 1 ? datum.label : ''")
        tip = [alt.Tooltip(f"{date_col}:T", title="date", format="%a %d %b %Y"),
               alt.Tooltip(f"{value_col}:Q", format=",", title=value_col)]
        rows_per_px = 18

    labels = d["label"].drop_duplicates().tolist()   # already chronological
    # Step sizing, not a pixel height. Streamlit renders stretched charts with
    # Vega's `autosize: fit`, which squeezes a numeric height into whatever the
    # element ends up being — four month-rows arrived at 9px each and their
    # labels stacked on top of one another. `Step` makes the row height the input
    # and the total height the output, which fit cannot override.
    h = height or alt.Step(rows_per_px)
    return alt.Chart(d).mark_rect(cornerRadius=1.5, stroke=SURFACE,
                                  strokeWidth=1).encode(
        x=alt.X("col:O", title=None, axis=col_axis),
        # labelOverlap=False is load-bearing for the same reason.
        y=alt.Y("label:N", title=None, sort=labels,
                axis=alt.Axis(domain=False, ticks=False, labelFontSize=10.5,
                              grid=False, labelOverlap=False)),
        color=alt.Color(f"{value_col}:Q", title=None,
                        scale=alt.Scale(range=RAMP, type="sqrt"),
                        legend=alt.Legend(orient="top", gradientLength=150,
                                          format=",.0f", tickCount=4,
                                          labelFontSize=10)),
        tooltip=tip,
    ).properties(height=h, title=title)


def stacked_bar(df, x: str, y: str, series: str, height: int = 240,
                y_title: str | None = None):
    """Composition over time. A 1px surface-coloured stroke separates segments —
    without it, two adjacent fills of similar lightness read as one block."""
    doms = sorted(df[series].dropna().unique().tolist())
    d = df.copy()
    d["_lbl"] = d[series].map(lambda s: SOURCE_LABELS.get(s, s))
    labels = [SOURCE_LABELS.get(s, s) for s in doms]
    return alt.Chart(d).mark_bar(
        cornerRadiusEnd=3, stroke=SURFACE, strokeWidth=1,
    ).encode(
        # timeUnit makes this a band scale, so each month's bar fills its slot.
        # Without it a temporal x gives every bar a 1px width and the chart reads
        # as a row of hairlines whatever the range.
        x=alt.X(f"{x}:T", timeUnit="yearmonth", title=None,
                axis=alt.Axis(format="%b %y", labelAngle=0)),
        y=alt.Y(f"{y}:Q", title=y_title, stack=True),
        color=alt.Color("_lbl:N", title=None,
                        scale=alt.Scale(domain=labels,
                                        range=[SOURCE_COLOURS.get(s, SERIES[i % 6])
                                               for i, s in enumerate(doms)])),
        tooltip=[alt.Tooltip(f"{x}:T", title="period"),
                 alt.Tooltip("_lbl:N", title="source"),
                 alt.Tooltip(f"{y}:Q", format=",", title="pageviews")],
    ).properties(height=height)


def series_scale(domain: list[str]) -> alt.Scale:
    return alt.Scale(domain=domain,
                     range=[SOURCE_COLOURS.get(d, SERIES[i % len(SERIES)])
                            for i, d in enumerate(domain)])
