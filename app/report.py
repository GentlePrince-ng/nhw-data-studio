"""Take what is on screen off the screen: HTML, Excel, PNG.

The download is the deliverable. Someone runs this at 09:40 for a 10:00 meeting,
and what they need is a file that opens on a laptop with no network, no Python
and no Streamlit, and that still says where the numbers came from.

So charts are rasterised server-side into inline SVG (vl-convert, offline) rather
than embedded as Vega specs pointing at a CDN, and the fonts are embedded as
base64. The HTML is one file with no external requests. If vl-convert is ever
missing the report still builds, with the aggregated table in place of each chart
and a line saying why — a report with tables beats a failed download.

Every export carries the same provenance block: source file, sheet, row count,
and the timestamp. A chart in a WhatsApp group with no denominator is how a
figure ends up being quoted wrongly for a year.
"""
from __future__ import annotations

import html
import io

import pandas as pd

from . import theme

try:
    import vl_convert as vlc
except ImportError:                              # exports degrade, never fail
    vlc = None

SVG_OK = vlc is not None


def _sized(chart, width: int):
    """Pin an explicit width before rasterising.

    On screen Streamlit stretches a chart to its container, so no width is set
    in the spec; rendered to a standalone file the same spec falls back to
    Altair's 400px default and the report reads as a column of thumbnails beside
    full-width text. Charts whose height is an alt.Step are left alone in that
    axis — the step is what keeps their bars readable.
    """
    try:
        return chart.properties(width=width)
    except Exception:
        return chart


def chart_svg(panel, width: int = 760) -> str:
    """One panel as inline SVG, or an empty string if it cannot be rendered."""
    if vlc is None or panel.chart is None:
        return ""
    try:
        # A map is sized by its projection, not by the page. Forcing a width on
        # it leaves 270px of blank canvas inside the SVG — the geometry still
        # fits to the height, so the map sits off to one side of its own image.
        # Spec building is inside the try because it can raise too: Altair
        # enforces its row ceiling at to_json() time, which is how a chart with
        # too much data in it went missing from reports without a word.
        spec = (panel.chart if panel.kind == "map"
                else _sized(panel.chart, width)).to_json()
        return vlc.vegalite_to_svg(spec)
    except Exception:
        try:                                  # a faceted or layered edge case
            return vlc.vegalite_to_svg(panel.chart.to_json())
        except Exception:
            return ""


def chart_png(panel, scale: float = 2.0, width: int = 760) -> bytes:
    """One panel as PNG bytes for slides. Empty bytes if unavailable."""
    if vlc is None or panel.chart is None:
        return b""
    try:
        return vlc.vegalite_to_png(_sized(panel.chart, width).to_json(),
                                   scale=scale)
    except Exception:
        return b""


def _table_html(df, limit: int = 60) -> str:
    if df is None or not len(df):
        return ""
    shown = df.head(limit)
    cells = []
    cells.append("<tr>" + "".join(
        f"<th>{html.escape(str(c))}</th>" for c in shown.columns) + "</tr>")
    for _, row in shown.iterrows():
        tds = []
        for v in row:
            txt = "" if pd.isna(v) else (f"{v:,.2f}".rstrip("0").rstrip(".")
                                         if isinstance(v, float) else f"{v:,}"
                                         if isinstance(v, int) else str(v))
            tds.append(f"<td>{html.escape(txt)}</td>")
        cells.append("<tr>" + "".join(tds) + "</tr>")
    more = (f'<p class="more">{len(df) - limit:,} further rows in the Excel '
            "download</p>" if len(df) > limit else "")
    return f'<div class="tw"><table>{"".join(cells)}</table></div>{more}'


_CSS = """
__FONTS__
:root{--deep:__DEEP__;--brand:__BRAND__;--ink:__INK__;--ink2:__INK2__;
 --muted:__MUTED__;--border:__BORDER__;--panel:__PANEL__;--pale:__PALE__;
 --flag:__BEHIND__;--watch:__CLOSE__;}
*{box-sizing:border-box;}
body{margin:0;background:#fff;color:var(--ink);
 font:15px/1.55 Poppins,system-ui,-apple-system,"Segoe UI",sans-serif;}
.wrap{max-width:900px;margin:0 auto;padding:0 28px 72px;}
.band{background:linear-gradient(100deg,__DEEP__ 0%,#1F8B45 52%,__BRAND__ 100%);
 color:#fff;padding:26px 28px;margin-bottom:26px;}
.band h1{margin:0;font-size:1.5rem;letter-spacing:-.02em;font-weight:700;}
.band p{margin:6px 0 0;opacity:.9;font-size:.86rem;}
/* Flex, not float. The caption on a map runs to a couple of hundred characters
   — matched places, greyed places, unmatched values, combined spellings — and a
   floated span clipped the end of it, losing exactly the caveats the caption
   exists to carry. Flex-wrap drops a long note onto its own line instead. */
h2{color:var(--deep);font-size:1.02rem;letter-spacing:-.01em;margin:34px 0 4px;
 padding-bottom:7px;border-bottom:1px solid var(--border);
 display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 14px;}
h2 .note{color:var(--muted);font-size:.74rem;font-weight:500;line-height:1.45;
 text-transform:none;letter-spacing:0;margin-left:auto;max-width:100%;}
.lede{background:var(--pale);border-left:3px solid var(--brand);
 padding:14px 16px;font-size:.95rem;color:var(--ink2);}
ol.f{padding-left:0;list-style:none;margin:0;}
ol.f li{padding:11px 0 11px 14px;border-bottom:1px solid var(--border);
 border-left:3px solid transparent;font-size:.92rem;color:var(--ink2);}
ol.f li.flag{border-left-color:var(--flag);}
ol.f li.watch{border-left-color:var(--watch);}
ol.f li .k{display:inline-block;font-size:.66rem;text-transform:uppercase;
 letter-spacing:.06em;color:var(--muted);font-weight:600;margin-right:8px;}
.kpis{display:flex;flex-wrap:wrap;gap:10px;margin-top:14px;}
.kpi{flex:1 1 150px;border:1px solid var(--border);border-radius:8px;
 padding:12px 14px;background:#fff;}
.kpi .l{display:block;font-size:.66rem;text-transform:uppercase;
 letter-spacing:.06em;color:var(--muted);font-weight:600;}
.kpi .v{display:block;font-size:1.5rem;font-weight:700;color:var(--deep);
 letter-spacing:-.02em;line-height:1.25;}
.kpi .n{display:block;font-size:.72rem;color:var(--muted);}
figure{margin:14px 0 0;}
figure svg{max-width:100%;height:auto;}
.tw{overflow-x:auto;margin-top:12px;}
table{border-collapse:collapse;width:100%;font-size:.83rem;}
th{text-align:left;font-weight:600;color:var(--muted);font-size:.7rem;
 text-transform:uppercase;letter-spacing:.05em;padding:7px 10px;
 border-bottom:1px solid var(--border);white-space:nowrap;}
td{padding:6px 10px;border-bottom:1px solid #F1F4F2;color:var(--ink2);}
tr:nth-child(even) td{background:var(--panel);}
.more{font-size:.75rem;color:var(--muted);margin:6px 0 0;}
.prov{margin-top:40px;padding-top:14px;border-top:1px solid var(--border);
 font-size:.76rem;color:var(--muted);}
.prov b{color:var(--ink2);font-weight:600;}
@media print{.band{-webkit-print-color-adjust:exact;print-color-adjust:exact;}
 h2{break-after:avoid;}figure{break-inside:avoid;}}
"""


def _css() -> str:
    return (_CSS.replace("__FONTS__", theme._font_css())
            .replace("__DEEP__", theme.GREEN_DEEP)
            .replace("__BRAND__", theme.GREEN_BRAND)
            .replace("__INK2__", theme.INK2).replace("__INK__", theme.INK)
            .replace("__MUTED__", theme.MUTED).replace("__BORDER__", theme.BORDER)
            .replace("__PANEL__", theme.PANEL).replace("__PALE__", theme.GREEN_PALE)
            .replace("__BEHIND__", theme.BEHIND).replace("__CLOSE__", theme.CLOSE))


def html_report(prof, findings, panels, title: str = "",
                kpis=None, tables: bool = True) -> bytes:
    """A single self-contained HTML file: findings, then every panel, then
    provenance. Opens offline; prints to PDF from the browser."""
    stamp = pd.Timestamp.now().strftime("%d %B %Y, %H:%M").lstrip("0")
    name = title or f"{prof.source}"
    head = [f'<!doctype html><html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            f"<title>{html.escape(name)} — NHW Data Studio</title>",
            f"<style>{_css()}</style></head><body>",
            f'<div class="band"><h1>{html.escape(name)}</h1>',
            f'<p>Automated read of {prof.rows:,} records and '
            f'{len(prof.columns)} columns · Nigeria Health Watch Data Studio · '
            f'{stamp}</p></div><div class="wrap">']
    body = []

    if kpis:
        body.append('<div class="kpis">')
        for label, value, note in kpis:
            body.append(f'<div class="kpi"><span class="l">'
                        f'{html.escape(label)}</span><span class="v">'
                        f'{html.escape(str(value))}</span><span class="n">'
                        f'{html.escape(note)}</span></div>')
        body.append("</div>")

    if findings:
        from . import insights as ins
        body.append(f'<h2>What stands out<span class="note">'
                    f'{len(findings)} findings, most consequential first</span></h2>')
        body.append(f'<p class="lede">{html.escape(ins.headline(prof, findings))}</p>')
        body.append('<ol class="f">')
        for f in findings:
            body.append(f'<li class="{f.severity}"><span class="k">'
                        f'{html.escape(f.kind)}</span>{html.escape(f.text)}</li>')
        body.append("</ol>")

    for p in panels:
        note = f'<span class="note">{html.escape(p.note)}</span>' if p.note else ""
        body.append(f"<h2>{html.escape(p.title)}{note}</h2>")
        svg = chart_svg(p)
        if svg:
            body.append(f"<figure>{svg}</figure>")
        elif p.chart is not None:
            body.append('<p class="more">Chart image unavailable in this '
                        "environment — the figures behind it are below.</p>")
        # A table under its own chart says the same thing twice and doubles the
        # length of the report. It appears only where there is no chart to read
        # — the column map, or a panel whose image could not be rendered — and
        # the Excel workbook carries every table regardless.
        if tables and p.table is not None and not svg:
            body.append(_table_html(p.table))

    sheet = f" · sheet <b>{html.escape(prof.sheet)}</b>" if prof.sheet else ""
    trunc = (f" · <b>{prof.truncated:,} rows beyond the read cap were not "
             "included</b>" if prof.truncated else "")
    body.append(
        f'<p class="prov">Source file <b>{html.escape(prof.source)}</b>{sheet} · '
        f'<b>{prof.rows:,}</b> records read · generated {stamp} by NHW Data '
        f'Studio{trunc}.<br>Figures are computed directly from the uploaded '
        "file. Percentages are stated against the number of records that "
        "answered the question concerned, which is printed under each chart, "
        "not against the full sample. Correlations are associations, not "
        "causes.</p>")
    return ("".join(head + body) + "</div></body></html>").encode("utf-8")


def _safe_sheet(name: str, used: set) -> str:
    """Excel sheet names: 31 chars, no []:*?/\\, and unique."""
    base = "".join(ch for ch in str(name) if ch not in "[]:*?/\\")[:28].strip()
    base = base or "Sheet"
    out, i = base, 2
    while out.lower() in used:
        out = f"{base[:26]}_{i}"
        i += 1
    used.add(out.lower())
    return out


def excel_book(prof, findings, panels) -> bytes:
    """Every table behind the dashboard, one sheet each, plus the column map."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        used = set()
        pd.DataFrame([{"Severity": f.severity, "Type": f.kind,
                       "Finding": f.text} for f in findings]).to_excel(
            xw, sheet_name=_safe_sheet("Findings", used), index=False)
        pd.DataFrame([{"Column": c.name, "Reads as": c.role, "Dtype": c.dtype,
                       "Filled": c.filled, "Missing": c.missing,
                       "Missing %": round(c.missing_pct, 1),
                       "Distinct": c.unique,
                       "Example": c.sample[0] if c.sample else "",
                       "Sentinel codes": ", ".join(map(str, c.sentinel_hits))
                       } for c in prof.columns.values()]).to_excel(
            xw, sheet_name=_safe_sheet("Column map", used), index=False)
        for p in panels:
            if p.table is None or not len(p.table):
                continue
            p.table.to_excel(xw, sheet_name=_safe_sheet(p.title, used),
                             index=False)
        meta = pd.DataFrame([
            {"Item": "Source file", "Value": prof.source},
            {"Item": "Sheet", "Value": prof.sheet or "(first)"},
            {"Item": "Records read", "Value": prof.rows},
            {"Item": "Rows skipped by cap", "Value": prof.truncated},
            {"Item": "Columns", "Value": len(prof.columns)},
            {"Item": "Generated", "Value": pd.Timestamp.now().strftime(
                "%Y-%m-%d %H:%M")},
            {"Item": "Tool", "Value": "NHW Data Studio"},
        ])
        meta.to_excel(xw, sheet_name=_safe_sheet("Provenance", used), index=False)
    return buf.getvalue()
