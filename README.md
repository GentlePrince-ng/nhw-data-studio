# NHW Data Studio

Upload a spreadsheet, get the read on it, take a report away. No code, no setup
per dataset, no waiting for an analyst to be free.

```bash
pip install -r requirements.txt
streamlit run app/studio_app.py
```

Then drag any `.xlsx` / `.xlsm` / `.csv` / `.tsv` into the sidebar.

Maps need one extra step, because the Nigeria boundary files are not in this
repository — see [The maps](#the-maps). Everything else works without them.

To see it work before pointing it at real data:

```bash
python samples/make_sample.py     # a 10,180-row Kobo-style survey with planted defects
python tests/test_pipeline.py     # read -> type -> analyse -> chart -> export, asserted
python tests/test_edges.py        # 14 awkward uploads, plus 100,000 rows
```

On the sample that runs in about **six seconds** end to end — including drawing
both maps and building the downloadable report.

## What it does

**Types every column** — dates, categories, numbers, binaries, identifiers, free
text, geography, empty, constant — and shows you its decisions on the Data
health page. Nothing about the analysis is hidden.

**Ranks what stands out.** Data-quality problems that change how every other
figure reads come first, then coverage, concentration, trend, distribution and
association. Every finding is computed from the file and carries its own
numbers. The top five are the product; nobody reads the twentieth.

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

Percentages are always stated against the number of records that answered the
question concerned, and that denominator is printed under the chart.

## The maps

**The boundary files are not in this repository.** They are derived from a
national administrative layer whose onward licensing is not this project's to
decide, so you build them once from your own source:

```bash
pip install geopandas shapely
python static/build_boundaries.py path/to/lga_boundaries.geojson
# or drop the layer into static/source/ (git-ignored) and run it bare
```

That writes `static/nigeria_states.geojson` (37 polygons, 121 KB) and
`static/nigeria_lgas.geojson` (774 polygons, 635 KB) — dissolved, simplified and
rounded to 3 decimal places. The source needs one polygon per LGA plus an LGA
name and a state name; the script recognises the field names used by GRID3,
geoBoundaries, HDX/OCHA and GADM, and prints what it found. See its docstring
for where to get a layer, and prefer the operational (GRID3) one: global
datasets disagree with NPHCDA on LGA counts.

**Without the files the app runs normally and draws no maps**, saying so under
the charts rather than leaving you to wonder. `geopandas` and `shapely` are
build-time only and deliberately absent from `requirements.txt`; the app itself
reads the GeoJSON with the standard library.

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

- **Overview** — four KPIs, a paragraph you can paste into an email, the top six findings, the chosen charts, the downloads.
- **Findings** — everything, filterable by type and severity, each with the figures behind it.
- **Data health** — the issues, completeness by column, the full column map, duplicate rows, and a typed CSV copy of your data.
- **Explore** — pick a column and something to break it down by; the chart type follows the pair of roles (two numerics → scatter with the correlation stated; numeric by category → group means with small groups flagged; date by category → lines; two categories → row-percentaged crosstab).
- **Written brief** — optional. Sends the *aggregates* — column map, finding sentences, small summary tables — to Claude and asks for a draft brief in four sections. No row of data leaves the machine, and the model is instructed to use no number it was not given. Needs `pip install anthropic` and `ANTHROPIC_API_KEY`; everything else works without it.

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
