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


def _find_adm1(lga_path: Path) -> Path | None:
    """Look for a state layer next to the LGA one, by the usual naming.

    Spatial formats before plain `.json`, and anything called `meta*` skipped:
    geoBoundaries' API metadata sits beside the download as `meta_ADM1.json`,
    and it matched the ADM1 pattern before it matched nothing readable.
    """
    for ext in (".geojson", ".shp", ".gpkg", ".json"):
        for pattern in ("*ADM1*", "*adm1*", "*state*", "*STATE*"):
            for cand in sorted(lga_path.parent.glob(pattern + ext)):
                if cand == lga_path or cand.name.lower().startswith("meta"):
                    continue
                return cand
    return None


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


def _read(path: Path):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(4326)


def _lga_field(gdf) -> str | None:
    for a, _ in NAME_FIELDS:
        if a in gdf.columns:
            return a
    return None


def _state_field(gdf) -> str | None:
    """A field naming the parent state — not merely present, but plausible.

    geoBoundaries ADM2 carries `shapeGroup`, which pairs with `shapeName` in
    NAME_FIELDS but holds the country code "NGA" for every row. Taking it as the
    state name yielded one state containing all 774 LGAs, which then dissolved
    into a single blob shaped like Nigeria. A parent field has to have several
    distinct values and fewer of them than there are rows.
    """
    for _, b in NAME_FIELDS:
        if b not in gdf.columns:
            continue
        n = gdf[b].nunique(dropna=True)
        if 2 <= n < len(gdf):
            return b
    return None


def _attach_states(lgas, adm1_path: Path):
    """Give every LGA its state by point-in-polygon against the ADM1 layer.

    Needed for sources like geoBoundaries whose ADM2 layer names no parent. The
    join is on each LGA's representative point, which is guaranteed to lie
    inside its own polygon (a centroid is not, for a crescent-shaped LGA), and
    anything still unplaced falls back to the nearest state so no LGA is lost.
    """
    adm1 = _read(adm1_path)
    field = _lga_field(adm1)
    if field is None:
        print(f"  cannot find a name field in {adm1_path.name}")
        return None
    adm1 = adm1[[field, "geometry"]].rename(columns={field: "statename"})
    print(f"  ADM1 layer {adm1_path.name}: {len(adm1)} states, "
          f"name field {field!r}")

    pts = lgas.copy()
    pts["geometry"] = lgas.geometry.representative_point()
    hit = gpd.sjoin(pts, adm1, how="left", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    out = lgas.copy()
    out["statename"] = hit["statename"].to_numpy()

    missing = out["statename"].isna()
    if missing.any():
        near = gpd.sjoin_nearest(pts.loc[missing], adm1, how="left")
        near = near[~near.index.duplicated(keep="first")]
        out.loc[missing, "statename"] = near["statename"].to_numpy()
        print(f"  {int(missing.sum())} LGA(s) fell outside every state polygon "
              "and were assigned to the nearest one")
    return out


def main() -> int:
    source = _resolve_source()
    if source is None or not source.exists():
        print(__doc__)
        print("\nNo source layer found"
              + (f" at:\n  {source}" if source else "")
              + "\nPass one as an argument, set NHW_BOUNDARY_SOURCE, or drop it "
              f"into:\n  {DROP_DIR}")
        return 1
    gdf = _read(source)
    print(f"read {len(gdf)} features from {source.name}, crs 4326")

    lga_f = _lga_field(gdf)
    if lga_f is None:
        print("No recognised LGA name field. This layer has:\n"
              f"  {', '.join(map(str, gdf.columns))}\n"
              "Add the pair to NAME_FIELDS and re-run.")
        return 1
    state_f = _state_field(gdf)
    gdf = gdf.rename(columns={lga_f: "lganame"})
    print(f"using {lga_f!r} for LGA")

    adm1_path = Path(sys.argv[2]) if len(sys.argv) > 2 else _find_adm1(source)
    states_src = None
    if state_f:
        print(f"using {state_f!r} for state")
        gdf = gdf.rename(columns={state_f: "statename"})
    else:
        if adm1_path is None or not adm1_path.exists():
            print("This layer names no parent state (geoBoundaries ADM2 does "
                  "not), so a state layer is required. Pass it as a second "
                  f"argument or put it beside the first in:\n  {DROP_DIR}")
            return 1
        print(f"no parent state in this layer — joining to {adm1_path.name}")
        gdf = _attach_states(gdf, adm1_path)
        if gdf is None:
            return 1
        states_src = adm1_path

    gdf["statename"] = gdf["statename"].astype(str).str.strip().replace(
        {"Fct": "Federal Capital Territory", "FCT": "Federal Capital Territory",
         "Abuja": "Federal Capital Territory"})
    gdf["lganame"] = gdf["lganame"].astype(str).str.strip()

    # Prefer the published state layer over dissolving the LGAs: dissolve leaves
    # slivers where LGA edges disagree, and the published outline is the one
    # readers recognise.
    if states_src is not None:
        states = _read(states_src)
        f = _lga_field(states)
        states = states[[f, "geometry"]].rename(columns={f: "statename"})
        states["statename"] = states["statename"].astype(str).str.strip()
    else:
        states = gdf[["statename", "geometry"]].dissolve(
            by="statename").reset_index()
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
