# v.in.ghcn — Claude Code context

## What this module does

GRASS GIS addon that pulls NOAA GHCN-Daily (GHCNd) climate station data into a GRASS location as a vector point map and SQLite time series table. No authentication required — data fetched directly from NCEI over HTTPS.

**Capabilities:**
- **Station locations**: vector point map from the GHCNd station inventory, filtered to the current GRASS region (or explicit station IDs), with columns `station_id`, `latitude`, `longitude`, `elevation`, `state`, `name`, `gsn_flag`, `hcncrn_flag`, `wmo_id`
- **Element filtering**: only stations with at least one requested element on inventory record are included
- **Record length filtering**: `min_years` option filters by years of record within the requested date range
- **Time series** (default, skip with `-l`): daily records → SQLite table `{output}_timeseries` (columns: `cat`, `station_id`, `datetime`, `element`, `value`, `q_flag`); indexed on `(cat, element, datetime)`
- **Quality flags**: `strict` (default, QC-passed = blank q_flag only) or `all`
- **Two-pass domain enclosure**: when `domain=` is given, Pass 1 expands the bbox until inventory-based hull encloses the domain centroid; Pass 2 downloads data then expands further if per-decade data hull gaps remain

**Elements supported:** PRCP (precipitation, mm), TMAX (max temp, °C), TMIN (min temp, °C), SNOW (snowfall, mm), SNWD (snow depth, mm), plus TOBS, AWND, EVAP, WDMV and others.

**Unit conversion:** GHCN raw values for PRCP, TMAX, TMIN, TOBS, AWND, EVAP, WDMV are stored in tenths of the standard unit — the module divides by 10 on import. SNOW and SNWD are already in mm.

**Monthly frequency:** parameter accepted but not yet implemented (`gs.fatal` on selection).

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output` | vector name | required | Output vector map of station locations |
| `stations` | string | — | Comma-separated GHCN station IDs; bypasses bbox search |
| `frequency` | string | `daily` | `daily` or `monthly` |
| `elements` | string | `PRCP` | Comma-separated element codes |
| `start_date` | string | — | `YYYY-MM-DD`; omit for record start |
| `end_date` | string | today | `YYYY-MM-DD` |
| `min_years` | integer | — | Minimum years of record within requested period |
| `padding` | double | `0` | Fixed bbox expansion in degrees (applied before adaptive expansion) |
| `min_stations` | integer | — | Minimum stations per element; bbox expanded until satisfied |
| `domain` | vector name | — | Domain polygon; bbox expanded until stations enclose it (two passes) |
| `max_distance` | double | `10.0` | Maximum total bbox expansion in degrees; shared across both passes |
| `max_iterations` | integer | `40` | Maximum expansion steps (0.5° each); shared across both passes |
| `q_flags` | string | `strict` | `strict` (blank q_flag only) or `all` |
| `-l` | flag | — | Import station locations only, skip time series download |

## Two-pass domain enclosure

When `domain=` is given, the module runs two passes to ensure the station convex hull encloses the basin centroid with actual data coverage:

**Pass 1 (inventory-based, fast):**  
Expands bbox in 0.5° steps until:
- `min_stations` met for every element (if specified), AND
- basin centroid inside the convex hull of stations with ≥`min_years` for EACH element (per `per_element_sids` from `filter_stations()`)

**Pass 2 (data-based, after download):**  
Checks SQLite per decade (10-year windows): for each element, queries which cats have actual records in that decade, builds hull, tests centroid. Expands and downloads additional stations if gaps remain. Uses same `max_distance`/`max_iterations` budget as Pass 1.

Hull test uses `scipy.spatial.Delaunay.find_simplex()`. Fewer than 3 qualifying stations for any element immediately fails the hull check.

Temporal hull gaps that cannot be closed by expansion (budget exhausted) produce warnings but do not abort — early decades often have no surrounding stations regardless of bbox size.

## Python dependencies

- `requests`
- `pandas`
- `geopandas`
- `shapely`
- `scipy` (only when `domain=` is given)

## Data sources (all HTTPS, no auth)

- `https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt` — full station inventory (fixed-width)
- `https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt` — element × station × year range inventory
- `https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{station_id}.csv.gz` — per-station daily CSV, gzip-compressed

## Key implementation details

- `fetch_station_inventory()`: fixed-width parse with `colspecs` — column positions are specific to the GHCNd format and must not be changed
- `fetch_element_inventory()`: whitespace-delimited, 6 columns
- `filter_stations()`: bbox filter → element filter → min_years filter (in that order). Returns **3-tuple** `(df, per_element_counts, per_element_sids)`. `per_element_sids` maps each element to the set of `station_id`s with ≥`min_years` for THAT element — used for convex hull checks. The `fatal=False` early-return path also returns a 3-tuple `(empty_df, {}, {})`.
- `get_mapset_db()`: returns path to mapset SQLite db (`{GISDBASE}/{LOCATION}/{MAPSET}/sqlite/sqlite.db`), creating the directory if needed. Used by both fetch functions instead of duplicating the path logic.
- `fetch_and_write_timeseries()`: per-station gzip fetch, CSV parse, date filter, unit conversion, batch `executemany` insert. `append=False` (default) drops and recreates the table; `append=True` inserts into the existing table. Pass 2 must use `append=True` to avoid destroying Pass 1 data.
- `fetch_and_write_monthly_timeseries()`: same `append=` contract.
- `get_sample_centroid_ll()`: extracts basin centroid in lon/lat via `v.out.ascii input= format=point type=centroid`, then `m.proj flags=od`. Falls back to bbox centre if centroid features are absent.
- `_hull_criterion(filtered_df, elem_inv_df, per_element_sids, centroid_ll, start_date, end_date)`: returns `(ok: bool, inv_gaps: dict)`. Dispatches to `inventory_decade_hull_gaps()` when `start_date` is set, or `basin_inside_hull()` when not (because `inventory_decade_hull_gaps` degenerates to a single current-decade check without `start_date`). Called in both the Pass 1 expansion loop and the no-expansion else branch.
- `basin_inside_hull()`: aggregate hull check — centroid inside Delaunay hull of per-element qualifying stations. Used by `_hull_criterion` when `start_date` is absent. Returns False if any element has < 3 stations.
- `check_data_decade_hull()`: Pass 2 per-decade check using actual SQLite records. Returns `{element: [decade_start_years_with_gaps]}`.
- `report_temporal_hull_coverage()`: inventory-based per-decade warning (Pass 1 diagnostic, not blocking).
- **Deferred vector write**: cats are pre-assigned in memory (sorted alphabetically by `station_id`, `cat = 1..N`) before any data download. `geodataframe_to_grass()` is called once after all expansion rounds. `get_cat_map()` has been removed (was dead code after deferred vector write was introduced).
- `geodataframe_to_grass()`: `mkstemp` placeholder removed before `gdf.to_file()` — fiona must create the GeoPackage itself.

## Compatibility fixes (shared with v.in.waterdata)

1. **Shebang**: `#!/usr/bin/python3` (hard path). `#!/usr/bin/env python3` picks up Anaconda's Python when conda is active, pulling in NumPy 1.x compiled packages that conflict with system NumPy 2.x.
2. **PROJ_DATA**: set unconditionally at module level: `if os.path.exists('/usr/share/proj/proj.db'): os.environ['PROJ_DATA'] = '/usr/share/proj'`. Anaconda sets PROJ_DATA to its own older proj.db, causing `g.proj`/`m.proj` to segfault. Do NOT guard with "if not already set".
3. **fiona + GeoPackage**: `mkstemp` placeholder file is removed before `gdf.to_file()` — fiona must create the GeoPackage itself.
4. **pandas 3.x Arrow strings**: all non-geometry columns cast to `object` dtype before `to_file()`.
5. **m.proj in projected CRS**: `get_geographic_bbox()` uses `gs.read_command('m.proj', coordinates=..., flags='od', quiet=True)` and splits on `|`. Flag `-o` = current location → WGS84; `-d` = decimal degrees. The old subprocess approach with `-i` was wrong (`-i` is the inverse direction: WGS84 → location CRS).

## Repo and status

- GitHub: `https://github.com/awickert/v.in.ghcn`
- Local repo: `/home/awickert/dataanalysis/v.in.ghcn`
- **Not yet submitted to GRASS addons** (as of June 2026, currently in testing)
- No GRASS addons copy yet (contrast with v.in.waterdata which is at `~/.grass8/addons/scripts/v.in.waterdata`)


## Broader context

`v.in.ghcn` is the second module in the planned GIS-native hydrological initialize layer. For GSFLOW-GRASS specifically, PRCP + TMAX + TMIN are the primary daily forcing inputs for PRMS, making this module directly relevant to the GSFLOW-GRASS revival — a user can set the GRASS region to their watershed and pull all nearby climate stations automatically.

See `v.in.waterdata` for the companion discharge/basin module. Both modules share the same compatibility fix patterns and SQLite-based time series storage approach.
