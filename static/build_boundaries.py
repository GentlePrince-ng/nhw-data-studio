"""Build the two boundary files the maps need. Run once; the app never runs this.

    pip install geopandas shapely
    python static/build_boundaries.py path/to/lga_boundaries.geojson

or drop the source layer into static/source/ (git-ignored) and run it bare.

**The built files are not in this repository.** They are derived from a national
administrative boundary layer whose onward licensing is not this project's to
decide, so you supply the source and build them locally. Without them the app
runs normally and simply draws no maps.

The source needs one polygon per LGA and two attributes: an LGA name and a state
name (`lganame`/`statename` by default — see NAME_FIELDS). Any of these work:

  * GRID3 Nigeria operational LGA boundaries (grid3.org) — what this was built
    against, and the layer Nigerian health programmes actually use.
  * geoBoundaries ADM2 for NGA (geoboundaries.org), field names `shapeName` and
    `shapeGroup` — adjust NAME_FIELDS.
  * HDX / OCHA Nigeria admin boundaries, fields `ADM2_EN` / `ADM1_EN`.

Prefer the operational layer over a global dataset: global sets disagree with
NPHCDA on LGA counts, and a map that shows 768 LGAs to a room that works with
774 spends the meeting on the map instead of the finding.

Output:

    static/nigeria_states.geojson   37 polygons (36 states + FCT)
    static/nigeria_lgas.geojson     774 polygons, state-qualified keys

Why a build step rather than reading the source at runtime: the source is 8 MB,
geopandas and shapely are heavy dependencies to put on Streamlit Community
Cloud, and the app only ever needs `json.load`. Simplification and coordinate
rounding happen here, once, so the app ships small files and no geo stack.

Two decisions worth knowing:

* **Coordinates are rounded to 3 decimal places** (~110 m at this latitude).
  That is invisible on a national choropleth and it is the single biggest size
  saving — bigger than simplification — because most of a GeoJSON is digits.
* **LGA keys are `state|lga`**, because five LGA names are used by two states
  each (Surulere, Ifelodun, Irepodun, Bassa, Obi). An unqualified join on LGA
  name silently puts Lagos figures in Oyo.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).resolve().parent

# Where to look for the source layer, in order: the command line, then
# NHW_BOUNDARY_SOURCE, then anything dropped into static/source/ (which git
# ignores). No path is hardcoded — one was, and a hardcoded absolute path is
# both a private detail in a public repository and useless on anyone else's
# machine.
DROP_DIR = HERE / "source"
DROP_PATTERNS = ("*.geojson", "*.json", "*.shp", "*.gpkg")

# (lga field, state field) candidates, tried in order against the source.
NAME_FIELDS = [("lganame", "statename"), ("LGA", "STATE"),
               ("ADM2_EN", "ADM1_EN"), ("shapeName", "shapeGroup"),
               ("NAME_2", "NAME_1")]

STATE_TOLERANCE = 0.004      # degrees; state outlines survive this easily
LGA_TOLERANCE = 0.004        # ~440 m; a fifth off the vertex count with
                             # no empty or invalid polygon in the 774
PRECISION = 3


def _round_geom(obj, nd: int = PRECISION):
    """Round every coordinate in a nested GeoJSON coordinate structure."""
    if isinstance(obj, (int, float)):
        return round(float(obj), nd)
    return [_round_geom(x, nd) for x in obj]


def _write(gdf, path: Path, props: list) -> None:
    feats = []
    for _, row in gdf.iterrows():
        geom = json.loads(gpd.GeoSeries([row.geometry]).to_json())
        g = geom["features"][0]["geometry"]
        g["coordinates"] = _round_geom(g["coordinates"])
        feats.append({"type": "Feature", "geometry": g,
                      "properties": {k: row[k] for k in props}})
    out = {"type": "FeatureCollection", "features": feats}
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"  {path.name:26} {len(feats):>4} features  "
          f"{path.stat().st_size / 1024:>7.0f} KB")


def _resolve_source() -> Path | None:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    env = os.environ.get("NHW_BOUNDARY_SOURCE")
    if env:
        return Path(env)
    for pattern in DROP_PATTERNS:
        found = sorted(DROP_DIR.glob(pattern))
        if found:
            return found[0]
    return None


def main() -> int:
    source = _resolve_source()
    if source is None or not source.exists():
        print(__doc__)
        print("\nNo source layer found"
              + (f" at:\n  {source}" if source else "")
              + "\nPass one as an argument, set NHW_BOUNDARY_SOURCE, or drop it "
              f"into:\n  {DROP_DIR}")
        return 1
    gdf = gpd.read_file(source)
    print(f"read {len(gdf)} features from {source.name}, crs {gdf.crs}")
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)

    lga_f, state_f = next(
        ((a, b) for a, b in NAME_FIELDS if a in gdf.columns and b in gdf.columns),
        (None, None))
    if lga_f is None:
        print(f"None of the expected name fields are in this layer. It has:\n"
              f"  {', '.join(map(str, gdf.columns))}\n"
              "Add the pair to NAME_FIELDS and re-run.")
        return 1
    print(f"using {lga_f!r} for LGA and {state_f!r} for state")
    gdf = gdf.rename(columns={lga_f: "lganame", state_f: "statename"})

    # "Fct" in the GRID3 layer; the FCT is what everyone else writes.
    gdf["statename"] = gdf["statename"].astype(str).str.strip().replace(
        {"Fct": "Federal Capital Territory", "FCT": "Federal Capital Territory",
         "Abuja": "Federal Capital Territory"})
    gdf["lganame"] = gdf["lganame"].astype(str).str.strip()

    states = gdf[["statename", "geometry"]].dissolve(by="statename").reset_index()
    states["geometry"] = states.geometry.simplify(STATE_TOLERANCE,
                                                  preserve_topology=True)
    states["state"] = states["statename"]
    _write(states, HERE / "nigeria_states.geojson", ["state"])

    lgas = gdf[["statename", "lganame", "geometry"]].copy()
    lgas["geometry"] = lgas.geometry.simplify(LGA_TOLERANCE,
                                              preserve_topology=True)
    lgas["state"] = lgas["statename"]
    lgas["lga"] = lgas["lganame"]
    lgas["key"] = (lgas["state"].str.lower() + "|" + lgas["lga"].str.lower())
    _write(lgas, HERE / "nigeria_lgas.geojson", ["state", "lga", "key"])

    dupes = lgas["lga"].str.lower().duplicated(keep=False).sum()
    print(f"\n{len(states)} states, {len(lgas)} LGAs. "
          f"{dupes} LGA rows share a name with an LGA in another state — which "
          "is why the LGA layer is keyed state|lga.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
