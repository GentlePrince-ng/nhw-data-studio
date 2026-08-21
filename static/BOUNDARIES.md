# Boundary data — source, licence, attribution

`nigeria_states.geojson` and `nigeria_lgas.geojson` in this directory are derived
from **geoBoundaries**, released under **CC BY 4.0**, which permits
redistribution with attribution. They are committed to this repository so the
maps work on a fresh clone and on the deployed app.

## Attribution (required by CC BY 4.0)

> Boundary data from geoBoundaries (gbOpen), CC BY 4.0.
> Runfola, D. et al. (2020) *geoBoundaries: A global database of political
> administrative boundaries.* PLoS ONE 15(4): e0231866.
> <https://doi.org/10.1371/journal.pone.0231866>

This line also appears in the app (welcome page) and in the provenance block of
every downloaded HTML report, because attribution has to travel with the work
rather than sit only in the repository.

## What was downloaded

| | |
|---|---|
| Source | geoBoundaries API, `gbOpen` release, `NGA` |
| Layers | `ADM1` (37 features, 2.2 MB) and `ADM2` (774 features, 9.0 MB) |
| Vintage | boundaries as represented in 2022 |
| Retrieved | 21 August 2026 |
| Licence | Creative Commons Attribution 4.0 International |

The originals are not committed — they live in the git-ignored
`static/source/geoboundaries/` and are re-downloadable from
`https://www.geoboundaries.org/api/current/gbOpen/NGA/ADM1/` (and `ADM2/`),
whose JSON response carries a `gjDownloadURL`.

## How these files were built

```bash
python static/build_boundaries.py static/source/geoboundaries/NGA_ADM2.geojson
```

The script found the ADM1 layer beside it automatically. Two things it had to do
that are worth knowing:

**geoBoundaries ADM2 does not name its parent state.** Its `shapeGroup` field
holds the country code `NGA` for every one of the 774 rows, so each LGA's state
comes from a point-in-polygon join against ADM1, on each LGA's representative
point (a centroid can fall outside a crescent-shaped LGA; a representative point
cannot). Nothing needed the nearest-state fallback.

**The join was validated against the official LGA count for every state** —
Kano 44, Katsina 34, Osun 30, FCT 6, and so on for all 37. Every state came out
with exactly its official number, and the total is 774. That check is the reason
to trust the state labels; re-run it if you rebuild from a different source.

The state outlines come from the published ADM1 layer rather than from dissolving
the LGAs, which leaves slivers where LGA edges disagree.

## Earlier version

These files were first built from the GRID3 / eHA operational LGA layer, which is
the layer Nigerian health programmes use and is preferable on accuracy grounds.
It was replaced for licensing reasons only: its onward redistribution terms were
not clear enough to publish in a public repository, and geoBoundaries' are
explicit. If you need the operational boundaries, point the build script at them
locally — the field names are already recognised, and nothing else changes.
