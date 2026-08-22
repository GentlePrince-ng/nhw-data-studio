# NHW Data Studio

Upload a spreadsheet, get the read on it, take a report away. No code, no setup
per dataset, no waiting for an analyst to be free.

```bash
pip install -r requirements.txt
streamlit run app/studio_app.py
```

Then drag any `.xlsx` / `.xlsm` / `.csv` / `.tsv` into the sidebar.

The Nigeria state and LGA maps work out of the box — boundaries are included
under CC BY 4.0, see [The maps](#the-maps).

To see it work before pointing it at real data:

```bash
python samples/make_sample.py     # a 10,180-row Kobo-style survey with planted defects
python tests/test_pipeline.py     # read -> type -> analyse -> chart -> export, asserted
python tests/test_edges.py        # every awkward upload we have met, + 100,000 rows
```

Both exit non-zero on failure, so `python tests/test_edges.py && echo ok` is the
check — not reading the output, which is how two red suites once got reported as
green.

On the sample that runs in about **six seconds** end to end — including drawing
both maps and building the downloadable report. The largest real workbooks in
the archive — 66,000 rows, or 576 columns across 21 sheets — take under ten.

## What it does

**Types every column** — dates, categories, numbers, binaries, identifiers, free
text, geography, empty, constant — and shows you its decisions on the Data
health page. Nothing about the analysis is hidden.

**Builds metrics, not a profile.** Columns are split into **measures** (things
you aggregate — visits, wait time, doses) and **dimensions** (things you slice by
— state, facility type, month). The top row carries those measures with
period-over-period change and a sparkline, not the record count and the
percentage of cells filled. Whether a column is totalled or averaged is a guess
from its name, it is printed on every figure, and you can change it on the page.

**Answers "what moved", not just "what is biggest".** Each measure gets three
views: the level over time, the split by dimension, and a diverging bar of the
*change* per dimension value between the last two equal windows. A breakdown of
the level tells you who is largest; only a breakdown of the change tells you
what happened.

**Ranks data-quality problems, but keeps them out of the way.** They are
collapsed into one strip on the front page and laid out in full on Data health.
Every finding is computed from the file and carries its own numbers.

**Draws the charts worth drawing.** A sixty-column sheet has about six charts in
it. The front page picks those: a trend if there is a usable date, geography if
there is a place column, then the categories with real variation.

**Maps Nigeria properly.** A state column gets a choropleth beside its bar
chart; an LGA or ward column gets a choropleth *instead* of one, because 774
categories is not a bar chart. See below — the map is the part with the most
ways to mislead, so it is the part with the most caveats printed on it.

**Hands you a report.** One self-contained HTML file — charts embedded as SVG,
fonts embedded, no external requests, opens on a laptop with no internet, prints
to PDF from the browser — plus an Excel workbook with every table behind every
chart, the column map, and a provenance sheet.

## Built for the exports NHW actually gets

| What arrives | What the tool does |
|---|---|
| Kobo `select_multiple` as 40 `parent/option` binary columns | Folds them back into one question; shares stated as % of rows with a note that they sum past 100 |
| `98` / `99` / `999` for don't-know / refused | Detected, reported as a flag, and **excluded from means** — with the excluded count printed under the chart |
| `Kano`, `kano `, `KANO` | Reported as one category split three ways, with the variants named |
| Re-synced tablets producing duplicate submissions | Exact-duplicate rows counted; a repeating identifier column called out separately |
| A column the field team stopped filling in | Flagged with the filled count, so a percentage is not read against the whole sample |
| Collection that stopped for a fortnight | Gaps in the date coverage reported, so per-week averages are not read as continuous |
| A phone number column Excel turned into a number | Read as an identifier, kept out of the statistics |
| A title and blank rows above the table | The header row is found rather than assumed — without this a campaign tracker read as **one column and 71 rows** |
| One sheet per week, plus a Summary | Sheets sharing a layout are stacked into one dataset with a `Sheet` column; a pre-aggregated summary is left out and said so, because stacking it double counts |
| `Week 1 … Week 11` and no date column anywhere | Recognised as an ordered run and plotted as a trend, instead of a bar chart sorted by size |
| Whole survey questions as column names | Titles use a short form; the full question stays in the caption |
| GPS latitude/longitude/altitude columns | Never treated as measures — "average geo-point latitude" is not a metric |
| 22 yes/no attitude questions and nothing to total | Each becomes a **% yes** measure, which is what a perception survey actually reports |

Percentages are always stated against the number of records that answered the
question concerned, and that denominator is printed under the chart.

## The maps

37 states and the FCT, and all 774 LGAs, are committed in `static/` — so the
maps work on a fresh clone with nothing to set up.

> Boundary data from **geoBoundaries** (gbOpen), **CC BY 4.0** — Runfola, D.
> et al. (2020) *geoBoundaries: A global database of political administrative
> boundaries.* PLoS ONE 15(4): e0231866.

Provenance, the build command, and the checks the build had to pass are in
[`static/BOUNDARIES.md`](static/BOUNDARIES.md). Two points worth repeating here:
geoBoundaries' LGA layer does not name its parent state, so each LGA's state
comes from a point-in-polygon join against the state layer; and that join was
verified against the official LGA count of **every** state — Kano 44, Katsina 34,
FCT 6 — all 37 correct, 774 total.

To rebuild from a different source (the GRID3 operational layer, for instance,
which is what Nigerian health programmes use):

```bash
pip install geopandas shapely
python static/build_boundaries.py path/to/lga_boundaries.geojson
# or drop the layer into static/source/ (git-ignored) and run it bare
```

`geopandas` and `shapely` are build-time only and deliberately absent from
`requirements.txt` — the app reads the GeoJSON with the standard library, and if
the files are ever missing it runs normally and says under the charts why there
is no map.

Joining a spreadsheet to a boundary file is where a map goes wrong, so:

- **Names are canonicalised before matching** — case, padding, punctuation, a
  trailing "State", plus an alias table (`FCT`/`Abuja`, `Nassarawa`,
  `Akwa-Ibom`, `Crossriver`…). `KANO`, `Kano` and `kano ` become one shaded
  state, and the caption says so, because the bar chart on the same page still
  shows them apart and two charts that appear to disagree need explaining.
- **Anything that matches no boundary is named, not dropped** — under the map,
  in the table (an `On map` column appears), and as a flagged finding. A single
  misspelling otherwise removes a whole state from every geographic total in
  silence.
- **"No data" is a legend entry, not a pale shade.** Classes are quantiles with
  discrete swatches; unsurveyed places are neutral grey and labelled *no data*.
  A continuous ramp paints an unsurveyed state the same near-white as its
  lowest class, and every reader takes that for "almost none".
- **LGA joins are keyed `state|lga`** when a state column exists, because five
  LGA names belong to two states each (Surulere, Ifelodun, Irepodun, Bassa,
  Obi). Without a state column the join is unqualified and says so.
- **A map that cannot be drawn honestly is not drawn.** Fewer than four places
  matched, and you get the bar chart alone.

## Pages

- **Overview** — measure KPIs with period-over-period change, a paragraph you can paste into an email, then the charts: level, split, and what moved. Data-quality flags collapse into one strip; single-column distributions collapse into another. The downloads are at the foot.
- **Findings** — everything, filterable by type and severity, each with the figures behind it.
- **Data health** — the issues, completeness by column, the full column map, duplicate rows, and a typed CSV copy of your data.
- **Explore** — pick a column and something to break it down by; the chart type follows the pair of roles (two numerics → scatter with the correlation stated; numeric by category → group means with small groups flagged; date by category → lines; two categories → row-percentaged crosstab).
- **Written brief** — optional. Sends the *aggregates* — column map, finding sentences, small summary tables — to Claude and asks for a draft brief in four sections. No row of data leaves the machine, and the model is instructed to use no number it was not given. Needs `pip install "anthropic>=1"` and a key in `ANTHROPIC_API_KEY` or `.streamlit/secrets.toml` (which the app copies into the environment itself, because Streamlit does not). Everything else works without it, and the page says which of the two is missing.

## Deploying it

Streamlit Community Cloud has **no private apps** any more — that moved to
Snowflake — so a deployed URL is reachable by anyone who has it. Since this tool
takes uploads of health data, it ships with an access gate that **fails closed**:

- Set `ACCESS_CODE` in the app's secrets (share.streamlit.io → *Settings* →
  *Secrets*) and everyone needs that code once per browser session. Change the
  code to revoke access from everyone at once. `ACCESS_CODES = "a,b,c"` accepts
  several.
- With no code configured the app admits **nobody** and prints the fix for the
  owner. That is deliberate: open-by-default with a warning banner puts a real
  dataset one forgotten setting away from a public page.
- Running it on your own machine, set `NHW_STUDIO_OPEN=1` and there is no gate.

A shared code says *someone was given this*, not *who*. If per-person access
matters, the hashed-roster pattern from the NHW dashboard belongs here instead.

### The two secrets

Both live in *Settings* → *Secrets* on share.streamlit.io, in one TOML block:

```toml
ACCESS_CODE = "choose-something-not-guessable"
ANTHROPIC_API_KEY = "sk-ant-..."          # optional: only the Written brief page
```

`ACCESS_CODE` is required — without it nobody gets in. `ANTHROPIC_API_KEY` is
optional: leave it out and every page works except the Written brief, which says
so on itself. Streamlit does not export secrets into the environment and the
Anthropic SDK only reads the environment, so `page_brief` copies it across at
runtime; that indirection is why a key in `secrets.toml` works at all.

Prefer a key created inside an **organisation workspace** with a spend limit over
a personal one. A key in a workspace can be revoked by an admin, bills to the
organisation, and its usage is attributable to this tool rather than mixed in
with everything else you have running.

## Testing it against real files

`tests/test_pipeline.py` and `tests/test_edges.py` are the committed suites and
run anywhere. `tests/test_corpus.py` is different: it points the whole pipeline
at a folder of **real** exports and reports what breaks.

```bash
python tests/test_corpus.py                        # 40 files, sampled across folders
python tests/test_corpus.py --limit 220 --max-mb 20
python tests/test_corpus.py --only glide           # substring filter on the path
python tests/test_corpus.py --baseline samples/_corpus_last_run.json
```

It is **local only and deliberately not part of CI** — CI has no access to the
archive and should not have. No data is committed and none is printed: the
report carries file names, shapes, timings and error types, never a cell value.
The run log it writes is git-ignored, because file names alone can identify
people.

Outcomes, worst first:

| | meaning |
|---|---|
| `ERROR` | an exception. Always a tool bug: an unreadable file should degrade to a thin page, never a traceback. |
| `EMPTY` | zero charts **and nothing said about why**. Where a silent misread hides. There should be none. |
| `NOTTABLE` | nothing to chart, and the page says why — a form definition, a cover sheet, a list with no numbers. A good outcome. |
| `SLOW` | slower than 20s + 2s per MB. Scaled by size, so a 22 MB workbook taking half a minute is fine and a 5 MB one taking four minutes is not. |
| `OK` | rows, columns, and at least one chart. |

Why it exists: every serious bug in this tool was found by pointing it at one
real file. A title row above the table. A GA4 comment block that made the
delimiter a space. A sheet that was the same records as another. A phone number
reported as an average. A header row that lost to the first row of data, so a
trainee's name became a column heading. The fixtures in `test_edges.py` were
each written *after* the discovery, and they only protect against what has
already been seen — this is how the next shape gets found by a command instead
of by a colleague.

Last full run, 400 files across the archive: **288 charted, 110 explained, 0
empty, 0 errors, 0 slow — 1.7s a file.** The two slowest are 22 MB ICL exports
with 600 columns, at just under half a minute each; pandas spends that
converting 13 million cells and there is nothing left to shave.

Reading is the part that used to be slow, not the analysis: fifteen of 355 files
took over 25 seconds and one took 261. `python-calamine` reads the same
workbooks 6-9x faster than openpyxl with identical results, the sheet list is
cached rather than parsed three times per upload, and a column with neither a
heading nor more than a trace of data is dropped before anything is named — one
tracker sheet had a single pasted row of 16,377 links, which read as 16,377
columns. calamine is optional: without it everything still works, and `.xls`
uploads are not offered because openpyxl cannot read them.

## Size

Built for the 10,000-row exports this was commissioned for, comfortable to a few
hundred thousand. Aggregation happens in pandas before Altair sees anything, so
chart cost tracks the number of *categories*, not the number of records; the
scatter is the only chart that ships rows to the browser and it samples with the
sample size printed on it. Reads are capped at 300,000 rows (`ROW_CAP` in
`studio_app.py`) and anything skipped is stated on the page and in the report
rather than silently dropped.

## What it will not do

It will not tell you a number is right. It reports associations as associations,
prints the denominator with every figure, and says where the data cannot support
a claim instead of drawing a prettier chart. Nothing is written to disk or sent
anywhere — the upload lives in the browser session and goes when the tab closes.

## Layout

```
app/
  studio_app.py   entry point, sidebar, the five pages
  profiling.py    read + assign a role to every column   (Kobo folding, sentinels)
  insights.py     the ranked findings engine
  charts.py       Panel: title, note, chart, table — one object per visual
  geo.py          the state/LGA name join, and the choropleth
  report.py       HTML / Excel / PNG export
  narrative.py    optional Claude brief (aggregates only)
  theme.py        NHW brand system, shared with the Impact Observatory and GLIDE
static/           Poppins, the logo, the two boundary files + build_boundaries.py
samples/          generator + a sample workbook, report and workbook
tests/            end-to-end assertions over the sample, plus awkward uploads
                  (test_corpus.py is the local-only harness, see below)
```

`theme.py` is the same brand system as the Impact Observatory and GLIDE
dashboards, so the three read as siblings. Three things diverge from that copy
and say so in their docstrings: `bar()` pads the axis so an end-of-bar value
label is not clipped, `trend()` takes the axis date format as an argument
because at weekly grain the monthly default prints "Feb 26" five times in a row,
and `geo_chart()` is new — maps cannot go through `st.altair_chart` at all,
because Streamlit lifts the top-level data out of the spec and re-serialises it
as Arrow, which leaves a GeoJSON FeatureCollection with no geometry and a legend
labelled NaN.

**Never add a `[server]` section to `.streamlit/config.toml`.** Streamlit
Community Cloud restarts the app onto port 8502 while its proxy still serves
8501: the app runs and is unreachable.
