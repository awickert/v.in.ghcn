#!/usr/bin/python3
############################################################################
#
# MODULE:       v.in.ghcn
#
# AUTHOR(S):    Andrew Wickert
#
# PURPOSE:      Import NOAA GHCN station locations and time series into GRASS GIS
#
# COPYRIGHT:    (c) 2026 Andrew Wickert
#
#               This program is free software under the GNU General Public
#               License (>=v2). Read the file COPYING that comes with GRASS
#               for details.
#
#############################################################################

#%module
#% description: Import NOAA GHCN station locations and climate time series
#% keyword: vector
#% keyword: import
#% keyword: hydrology
#% keyword: GHCN
#% keyword: precipitation
#% keyword: rain gauge
#% keyword: climate
#%end

#%option G_OPT_V_OUTPUT
#%  key: output
#%  label: Output vector map of station locations
#%  required: yes
#%end

#%option
#%  key: stations
#%  type: string
#%  label: Comma-separated GHCN station IDs (if omitted, searches within current region)
#%  required: no
#%end

#%option
#%  key: frequency
#%  type: string
#%  label: Temporal frequency of GHCN product
#%  options: daily,monthly
#%  answer: daily
#%  required: no
#%end

#%option
#%  key: elements
#%  type: string
#%  label: Comma-separated element codes to import
#%  description: PRCP=precipitation(mm), SNOW=snowfall(mm), SNWD=snow depth(mm), TMAX=max temp(C), TMIN=min temp(C)
#%  answer: PRCP
#%  required: no
#%end

#%option
#%  key: start_date
#%  type: string
#%  label: Start date (YYYY-MM-DD); omit for station record start
#%  required: no
#%end

#%option
#%  key: end_date
#%  type: string
#%  label: End date (YYYY-MM-DD); omit for today
#%  required: no
#%end

#%option
#%  key: min_years
#%  type: integer
#%  label: Minimum years of record within requested period
#%  required: no
#%end

#%option
#%  key: padding
#%  type: double
#%  label: Expand region bounding box by this many degrees in each direction
#%  required: no
#%  answer: 0
#%end

#%option
#%  key: min_stations
#%  type: integer
#%  label: Minimum number of stations; bbox is expanded until satisfied
#%  required: no
#%end

#%option G_OPT_V_INPUT
#%  key: domain
#%  label: Domain polygon; bbox is expanded until stations enclose the polygon (inventory check), then verified against downloaded data
#%  required: no
#%end

#%option
#%  key: max_distance
#%  type: double
#%  label: Maximum bbox expansion in each direction (degrees); shared across inventory and data passes
#%  answer: 10.0
#%  required: no
#%end

#%option
#%  key: max_iterations
#%  type: integer
#%  label: Maximum number of expansion steps (each 0.5 degrees); shared across inventory and data passes
#%  answer: 40
#%  required: no
#%end

#%option
#%  key: q_flags
#%  type: string
#%  label: Quality flag filter
#%  options: strict,all
#%  answer: strict
#%  description: strict=QC-passed records only; all=include all records regardless of QC
#%  required: no
#%end

#%flag
#%  key: l
#%  description: Import station locations only, skip time series
#%end

import csv
import gzip
import importlib
import io
import os
import sqlite3
import tempfile
import atexit
from datetime import date

import grass.script as gs

if os.path.exists('/usr/share/proj/proj.db'):
    os.environ['PROJ_DATA'] = '/usr/share/proj'

TMPFILES = []

_GHCND_BASE  = 'https://www.ncei.noaa.gov/pub/data/ghcn/daily'
_GHCNM_PRCP_BASE = ('https://www.ncei.noaa.gov/data/'
                    'global-historical-climatology-network-monthly/'
                    'v4/precipitation/access')

# Elements whose raw values are in tenths of the standard unit
_TENTHS_ELEMENTS = {'PRCP', 'TMAX', 'TMIN', 'TOBS', 'AWND', 'EVAP', 'WDMV'}


def cleanup():
    for f in TMPFILES:
        try:
            os.remove(f)
        except OSError:
            pass


def require_package(import_name, pip_name=None):
    try:
        return importlib.import_module(import_name)
    except ImportError:
        gs.fatal(
            "Python package '{}' is required but not installed.\n"
            "Install with: pip install {}".format(import_name, pip_name or import_name)
        )


def get_geographic_bbox():
    """Return (west, south, east, north) in decimal degrees for the current region."""
    proj = gs.parse_command('g.proj', flags='g')
    region = gs.region()

    if proj.get('proj') in ('ll', 'longlat'):
        return region['w'], region['s'], region['e'], region['n']

    def _to_ll(x, y):
        out = gs.read_command('m.proj', coordinates='{},{}'.format(x, y),
                              flags='od', quiet=True)
        lon, lat = out.strip().split('|')[:2]
        return float(lon), float(lat)

    sw_lon, sw_lat = _to_ll(region['w'], region['s'])
    ne_lon, ne_lat = _to_ll(region['e'], region['n'])
    return sw_lon, sw_lat, ne_lon, ne_lat


def geodataframe_to_grass(gdf, output):
    """Write a GeoDataFrame to a temp GeoPackage and import into GRASS."""
    fd, tmp = tempfile.mkstemp(suffix='.gpkg')
    os.close(fd)
    os.remove(tmp)
    TMPFILES.append(tmp)

    for col in [c for c in gdf.columns if c != 'geometry']:
        try:
            gdf[col] = gdf[col].astype(object)
        except (TypeError, ValueError):
            gdf[col] = gdf[col].astype(str).astype(object)

    gdf.to_file(tmp, driver='GPKG')
    gs.run_command('v.import', input=tmp, output=output,
                   overwrite=gs.overwrite(), quiet=True)


def fetch_station_inventory():
    """Download GHCNd station inventory; return DataFrame."""
    import requests
    import pandas as pd

    url = '{}/ghcnd-stations.txt'.format(_GHCND_BASE)
    gs.message("Downloading station inventory...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    colspecs = [(0, 11), (12, 20), (21, 30), (31, 37), (38, 40),
                (41, 71), (72, 75), (76, 79), (80, 85)]
    names = ['station_id', 'latitude', 'longitude', 'elevation', 'state',
             'name', 'gsn_flag', 'hcncrn_flag', 'wmo_id']

    df = pd.read_fwf(io.StringIO(r.text), colspecs=colspecs, names=names, header=None)
    df['name'] = df['name'].str.strip()
    df['state'] = df['state'].str.strip()
    return df


def fetch_element_inventory():
    """Download GHCNd element inventory (station × element × year range)."""
    import requests
    import pandas as pd

    url = '{}/ghcnd-inventory.txt'.format(_GHCND_BASE)
    gs.message("Downloading element inventory...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(
        io.StringIO(r.text), sep=r'\s+', header=None,
        names=['station_id', 'latitude', 'longitude', 'element', 'firstyear', 'lastyear']
    )
    return df


def filter_stations(station_df, elem_inv_df, bbox, station_ids, elements, min_years,
                    start_date, end_date, fatal=True):
    """Filter station inventory to bbox/IDs, requested elements, and min_years.

    Returns (df, per_element_counts) where per_element_counts maps each requested
    element to the number of stations that carry it with sufficient years of record.
    A station qualifies for the overall df if it has ANY element with enough years;
    per_element_counts reflects how many stations have EACH element specifically.

    When fatal=False an empty DataFrame is returned (with empty counts) instead of
    calling gs.fatal for the bbox-empty case; used by the adaptive expansion loop.
    """
    if station_ids:
        df = station_df[station_df['station_id'].isin(station_ids)].copy()
        if df.empty:
            gs.fatal("None of the specified station IDs were found in the inventory.")
    else:
        west, south, east, north = bbox
        df = station_df[
            (station_df['latitude'] >= south) &
            (station_df['latitude'] <= north) &
            (station_df['longitude'] >= west) &
            (station_df['longitude'] <= east)
        ].copy()
        if df.empty:
            if not fatal:
                return df, {}, {}
            gs.fatal("No stations found within the current region.")

    # Keep only stations that have at least one requested element on record
    elem_filter = elem_inv_df[elem_inv_df['element'].isin(elements)]
    has_element = set(elem_filter['station_id'].unique())
    df = df[df['station_id'].isin(has_element)]
    if df.empty:
        gs.fatal("No stations have any of the requested elements: {}.".format(
            ', '.join(elements)))

    sids_in_df = set(df['station_id'])

    # Compute per-element station IDs (with data in the requested period) and counts.
    # per_element_sids maps each element to the set of station_ids that have sufficient
    # coverage for THAT element within the date range — used for convex hull checks.
    per_element_sids   = {}
    per_element_counts = {}
    if min_years:
        end_yr = int(end_date[:4]) if end_date else date.today().year
        ef = elem_inv_df[elem_inv_df['element'].isin(elements)].copy()
        if start_date:
            start_yr = int(start_date[:4])
            ef = ef[ef['lastyear'] >= start_yr]
            ef['eff_first'] = ef['firstyear'].clip(lower=start_yr)
        else:
            ef['eff_first'] = ef['firstyear']
        ef['eff_last'] = ef['lastyear'].clip(upper=end_yr)
        ef['record_years'] = ef['eff_last'] - ef['eff_first'] + 1
        ef_sufficient = ef[ef['record_years'] >= min_years]

        for elem in elements:
            sids = (set(ef_sufficient[ef_sufficient['element'] == elem]['station_id'])
                    & sids_in_df)
            per_element_sids[elem]   = sids
            per_element_counts[elem] = len(sids)

        sufficient = set(ef_sufficient['station_id'].unique())
        df = df[df['station_id'].isin(sufficient)]
        if df.empty:
            gs.fatal("No stations pass the min_years={} filter.".format(min_years))
    else:
        for elem in elements:
            sids = (set(elem_inv_df[elem_inv_df['element'] == elem]['station_id'])
                    & sids_in_df)
            per_element_sids[elem]   = sids
            per_element_counts[elem] = len(sids)

    gs.message("Found {} station(s) — per element: {}".format(
        len(df),
        ', '.join('{}={}'.format(e, per_element_counts.get(e, 0)) for e in elements)
    ))
    return df.reset_index(drop=True), per_element_counts, per_element_sids


def _year_ranges(years):
    """Convert a sorted list of integers to a compact range string like '1890-1920, 1945'."""
    if not years:
        return ''
    ranges = []
    start = prev = years[0]
    for y in years[1:]:
        if y == prev + 1:
            prev = y
        else:
            ranges.append((start, prev))
            start = prev = y
    ranges.append((start, prev))
    return ', '.join(str(a) if a == b else '{}-{}'.format(a, b) for a, b in ranges)


def report_temporal_coverage(df, elem_inv_df, elements, start_date, end_date,
                              sparse_threshold=4):
    """Report per-year station counts per element; warn about sparse periods.

    sparse_threshold: years with fewer than this many active stations generate
    a warning (default 4 matches v.interp.timeseries npoints default).
    """
    sids = set(df['station_id'])
    ef = elem_inv_df[
        elem_inv_df['element'].isin(elements) &
        elem_inv_df['station_id'].isin(sids)
    ].copy()

    end_yr = int(end_date[:4]) if end_date else date.today().year
    if start_date:
        start_yr = int(start_date[:4])
    else:
        start_yr = int(ef['firstyear'].min()) if not ef.empty else end_yr

    years = list(range(start_yr, end_yr + 1))
    if not years:
        return

    gs.message("Temporal coverage ({}-{}, active stations per element):".format(
        start_yr, end_yr))

    for elem in elements:
        ef_e = ef[ef['element'] == elem]
        if ef_e.empty:
            gs.warning("  {}: no stations in inventory for this element.".format(elem))
            continue

        counts = [
            int(((ef_e['firstyear'] <= yr) & (ef_e['lastyear'] >= yr)).sum())
            for yr in years
        ]

        min_n  = min(counts)
        max_n  = max(counts)
        mean_n = sum(counts) / len(counts)

        # Year with minimum coverage
        min_yr = years[counts.index(min_n)]

        gs.message("  {}: min={} ({}), mean={:.1f}, max={}".format(
            elem, min_n, min_yr, mean_n, max_n))

        sparse = sorted(yr for yr, n in zip(years, counts) if n < sparse_threshold)
        if sparse:
            gs.warning(
                "  {}: {} year(s) with < {} station(s): {}".format(
                    elem, len(sparse), sparse_threshold, _year_ranges(sparse)))


def inventory_decade_hull_gaps(filtered_df, elem_inv_df, per_element_sids,
                               centroid_ll, start_date, end_date):
    """Return per-element inventory hull gaps across decades.

    For each decade in [start_date, end_date], checks whether the basin centroid
    falls inside the convex hull of stations that (a) passed per-element min_years
    filtering (per_element_sids) and (b) have inventory overlap with that decade
    (firstyear <= dec+9 AND lastyear >= dec).

    Returns dict: element -> list of decade-start years where hull fails.
    Empty dict means full inventory-level temporal coverage.

    Used as a Pass 1 expansion criterion so the bbox is enlarged until the
    inventory predicts surrounding coverage in every decade, before any data
    is downloaded.
    """
    import numpy as np
    from scipy.spatial import Delaunay

    end_yr   = int(end_date[:4]) if end_date   else date.today().year
    start_yr = int(start_date[:4]) if start_date else end_yr

    lon_c, lat_c = centroid_ll
    pt = np.array([[lon_c, lat_c]])

    decade_starts = list(range((start_yr // 10) * 10, end_yr + 1, 10))
    gaps = {}

    for elem, sids in per_element_sids.items():
        ef_elem = elem_inv_df[
            (elem_inv_df['element'] == elem) &
            (elem_inv_df['station_id'].isin(sids))
        ]
        elem_gaps = []
        for dec in decade_starts:
            active = ef_elem[
                (ef_elem['firstyear'] <= dec + 9) &
                (ef_elem['lastyear']  >= dec)
            ]
            coords = filtered_df[
                filtered_df['station_id'].isin(set(active['station_id']))
            ][['longitude', 'latitude']].values.astype(np.float64)

            if len(coords) < 3:
                elem_gaps.append(dec)
                continue
            try:
                hull = Delaunay(coords)
            except Exception:
                elem_gaps.append(dec)
                continue
            if hull.find_simplex(pt)[0] < 0:
                elem_gaps.append(dec)

        if elem_gaps:
            gaps[elem] = elem_gaps

    return gaps


def report_inventory_hull_gaps(gaps):
    """Print inventory hull gap dict returned by inventory_decade_hull_gaps()."""
    if not gaps:
        gs.message("Inventory temporal hull: basin enclosed in all decades.")
        return
    for elem, bad_decades in sorted(gaps.items()):
        gs.warning(
            "Inventory hull gap — {}: basin centroid outside active-station hull "
            "for decade(s) starting: {}. Spatial interpolation may extrapolate "
            "for these periods.".format(elem, _year_ranges(bad_decades)))


def get_cat_map(output):
    """Return dict mapping station_id → cat from the imported vector map."""
    raw = gs.read_command('v.db.select', map=output, columns='cat,station_id', flags='c')
    cat_map = {}
    for line in raw.strip().splitlines():
        parts = line.split('|')
        if len(parts) == 2:
            cat_map[parts[1].strip()] = int(parts[0].strip())
    return cat_map


def get_mapset_db():
    """Return path to the current mapset's SQLite database, creating the directory if needed."""
    gisenv = gs.gisenv()
    db_dir = os.path.join(
        gisenv['GISDBASE'], gisenv['LOCATION_NAME'], gisenv['MAPSET'], 'sqlite'
    )
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, 'sqlite.db')


def fetch_and_write_timeseries(station_ids, cat_map, elements, start_date, end_date,
                               q_flags, table_name, append=False):
    """Download per-station GHCNd CSVs and write records to mapset SQLite database.

    When append=True the existing table is kept and new rows are inserted.
    When append=False (default) the table is dropped and recreated first.
    """
    import requests

    db_path = get_mapset_db()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    if not append:
        cur.execute('DROP TABLE IF EXISTS "{}"'.format(table_name))
        cur.execute('''
            CREATE TABLE "{}" (
                cat        INTEGER,
                station_id TEXT,
                datetime   TEXT,
                element    TEXT,
                value      REAL,
                q_flag     TEXT
            )
        '''.format(table_name))
        cur.execute(
            'CREATE INDEX "{}_idx" ON "{}" (cat, element, datetime)'.format(
                table_name, table_name)
        )
        conn.commit()

    total_rows = 0
    for station_id in station_ids:
        cat = cat_map.get(station_id)
        if cat is None:
            gs.warning("No cat found for station {}; skipping.".format(station_id))
            continue

        url = '{}/by_station/{}.csv.gz'.format(_GHCND_BASE, station_id)
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            content = gzip.decompress(r.content).decode('utf-8', errors='replace')
        except Exception as e:
            gs.warning("Could not fetch {}: {}".format(station_id, e))
            continue

        rows = []
        for rec in csv.reader(content.splitlines()):
            if len(rec) < 6:
                continue
            _sid, raw_date, element, raw_value = rec[0], rec[1], rec[2], rec[3]
            q_flag = rec[5]

            if element not in elements:
                continue
            if q_flags == 'strict' and q_flag.strip() != '':
                continue

            try:
                dt = '{}-{}-{}'.format(raw_date[:4], raw_date[4:6], raw_date[6:8])
            except Exception:
                continue

            if start_date and dt < start_date:
                continue
            if end_date and dt > end_date:
                continue

            try:
                val = float(raw_value)
                if element in _TENTHS_ELEMENTS:
                    val = val / 10.0
            except (ValueError, TypeError):
                val = None

            rows.append((cat, station_id, dt, element, val,
                         q_flag.strip() or None))

        if rows:
            cur.executemany(
                'INSERT INTO "{}" VALUES (?, ?, ?, ?, ?, ?)'.format(table_name), rows
            )
            conn.commit()
            total_rows += len(rows)
        gs.message("  {} → {:,} records".format(station_id, len(rows)))

    conn.close()
    return total_rows


def fetch_and_write_monthly_timeseries(station_ids, cat_map, start_date, end_date,
                                       q_flags, table_name, append=False):
    """Download per-station GHCNm precipitation CSVs and write to mapset SQLite.

    When append=True the existing table is kept and new rows are inserted.
    When append=False (default) the table is dropped and recreated first.
    """
    import requests

    db_path = get_mapset_db()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    if not append:
        cur.execute('DROP TABLE IF EXISTS "{}"'.format(table_name))
        cur.execute('''
            CREATE TABLE "{}" (
                cat        INTEGER,
                station_id TEXT,
                datetime   TEXT,
                element    TEXT,
                value      REAL,
                q_flag     TEXT
            )
        '''.format(table_name))
        cur.execute(
            'CREATE INDEX "{}_idx" ON "{}" (cat, element, datetime)'.format(
                table_name, table_name)
        )
        conn.commit()

    # Convert YYYY-MM-DD date bounds to integer YYYYMM for fast comparison
    start_ym = int(start_date[:4] + start_date[5:7]) if start_date else None
    end_ym   = int(end_date[:4]   + end_date[5:7])   if end_date   else None

    total_rows = 0
    for station_id in station_ids:
        cat = cat_map.get(station_id)
        if cat is None:
            gs.warning("No cat found for station {}; skipping.".format(station_id))
            continue

        url = '{}/{}.csv'.format(_GHCNM_PRCP_BASE, station_id)
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                gs.verbose(
                    "  {} not in GHCNm precipitation dataset; skipping.".format(station_id)
                )
                continue
            r.raise_for_status()
        except Exception as e:
            gs.warning("Could not fetch {}: {}".format(station_id, e))
            continue

        rows = []
        for rec in csv.reader(r.text.splitlines()):
            if len(rec) < 9:
                continue
            # cols: 0=station_id 1=name 2=lat 3=lon 4=elev
            #       5=YYYYMM 6=value(tenths mm) 7=dm_flag 8=qc_flag 9=ds_flag
            try:
                yyyymm = int(rec[5].strip())
            except ValueError:
                continue

            if start_ym and yyyymm < start_ym:
                continue
            if end_ym and yyyymm > end_ym:
                continue

            qc_flag = rec[8].strip()
            if q_flags == 'strict' and qc_flag:
                continue

            try:
                val = float(rec[6].strip())
                if val == -9999:
                    continue
                val = val / 10.0   # tenths of mm → mm
            except ValueError:
                continue

            year  = yyyymm // 100
            month = yyyymm  % 100
            dt = '{:04d}-{:02d}-01'.format(year, month)

            rows.append((cat, station_id, dt, 'PRCP', val, qc_flag or None))

        if rows:
            cur.executemany(
                'INSERT INTO "{}" VALUES (?, ?, ?, ?, ?, ?)'.format(table_name), rows
            )
            conn.commit()
            total_rows += len(rows)
        gs.message("  {} → {:,} records".format(station_id, len(rows)))

    conn.close()
    return total_rows


def get_sample_centroid_ll(sample_map):
    """Return (lon, lat) in decimal degrees for the centroid of a polygon map.

    Tries the explicit centroid feature first; falls back to bbox centre so
    the function works even if centroid features were not built into the map.
    """
    try:
        out = gs.read_command('v.out.ascii', input=sample_map, format='point',
                              type='centroid', separator='pipe', quiet=True)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if lines:
            x, y = lines[0].split('|')[:2]
            proj_out = gs.read_command('m.proj', coordinates='{},{}'.format(x, y),
                                       flags='od', quiet=True)
            lon, lat = proj_out.strip().split('|')[:2]
            return float(lon), float(lat)
    except Exception:
        pass

    # Fallback: geographic centre of the map's bounding box
    info = gs.parse_command('v.info', map=sample_map, flags='g')
    cx = (float(info['west']) + float(info['east'])) / 2
    cy = (float(info['south']) + float(info['north'])) / 2
    proj_out = gs.read_command('m.proj', coordinates='{},{}'.format(cx, cy),
                               flags='od', quiet=True)
    lon, lat = proj_out.strip().split('|')[:2]
    gs.message("sample= centroid: using bbox centre lon={:.4f} lat={:.4f}".format(
        float(lon), float(lat)))
    return float(lon), float(lat)



def basin_inside_hull(filtered_df, per_element_sids, centroid_ll):
    """Return True if centroid_ll is inside the convex hull of qualifying stations.

    Used when no start_date is given, so there are no meaningful decades to
    iterate over.  Checks aggregate coverage: stations with sufficient data
    anywhere in their record (per_element_sids) form a hull around the centroid
    for every requested element.  Returns False for any element with < 3 stations.
    """
    import numpy as np
    from scipy.spatial import Delaunay

    lon_c, lat_c = centroid_ll
    pt = np.array([[lon_c, lat_c]])

    for elem, sids in per_element_sids.items():
        coords = filtered_df[filtered_df['station_id'].isin(sids)][
            ['longitude', 'latitude']].values.astype(np.float64)
        if len(coords) < 3:
            return False
        try:
            hull = Delaunay(coords)
        except Exception:
            return False
        if hull.find_simplex(pt)[0] < 0:
            return False
    return True


def check_data_decade_hull(cur, table_name, elements, cat_to_xy_ll, centroid_ll,
                            start_date, end_date):
    """Check per-decade hull coverage using actual downloaded data in SQLite.

    For each decade in [start_date, end_date], queries which cats have records
    for each element in that decade, builds a convex hull from their lon/lat
    coordinates, and tests containment of centroid_ll.

    Returns dict: element -> list of decade-start years where hull fails.
    Empty dict means full coverage.
    """
    import numpy as np
    from scipy.spatial import Delaunay

    end_yr   = int(end_date[:4]) if end_date   else date.today().year
    start_yr = int(start_date[:4]) if start_date else end_yr
    decade_starts = list(range((start_yr // 10) * 10, end_yr + 1, 10))

    lon_c, lat_c = centroid_ll
    pt = np.array([[lon_c, lat_c]])

    gaps = {}
    for elem in elements:
        elem_gaps = []
        for dec in decade_starts:
            cur.execute(
                'SELECT DISTINCT cat FROM "{}" WHERE element=? '
                'AND datetime >= ? AND datetime <= ?'.format(table_name),
                (elem,
                 '{:04d}-01-01'.format(dec),
                 '{:04d}-12-31'.format(dec + 9))
            )
            active_cats = {row[0] for row in cur.fetchall()}
            coords = np.array([cat_to_xy_ll[c] for c in active_cats
                                if c in cat_to_xy_ll], dtype=np.float64)
            if len(coords) < 3:
                elem_gaps.append(dec)
                continue
            try:
                hull = Delaunay(coords)
            except Exception:
                elem_gaps.append(dec)
                continue
            if hull.find_simplex(pt)[0] < 0:
                elem_gaps.append(dec)
        if elem_gaps:
            gaps[elem] = elem_gaps
    return gaps


def main():
    options, flags = gs.parser()
    atexit.register(cleanup)

    output = options['output']
    stations_str = options['stations']
    frequency = options['frequency']
    elements_str = options['elements']
    start_date = options['start_date'] or None
    end_date = options['end_date'] or None
    min_years      = int(options['min_years'])      if options['min_years']      else None
    padding        = float(options['padding'])      if options['padding']        else 0.0
    min_stations   = int(options['min_stations'])   if options['min_stations']   else None
    sample_map     = options['domain']              if options['domain']         else None
    max_distance   = float(options['max_distance']) if options['max_distance']   else 10.0
    max_iterations = int(options['max_iterations']) if options['max_iterations'] else 40
    q_flags        = options['q_flags']
    flag_locations = flags['l']

    if frequency == 'monthly':
        non_prcp = [e for e in elements_str.split(',') if e.strip().upper() != 'PRCP']
        if non_prcp:
            gs.warning(
                "Monthly frequency only supports PRCP. "
                "Elements '{}' will be skipped.".format(', '.join(non_prcp))
            )

    require_package('requests')
    require_package('pandas')
    require_package('geopandas')
    require_package('shapely')

    elements = [e.strip().upper() for e in elements_str.split(',')]
    if frequency == 'monthly':
        elements = ['PRCP']   # GHCNm precipitation only; used for element inventory filter
    station_ids = [s.strip() for s in stations_str.split(',')] if stations_str else None
    bbox = None if station_ids else get_geographic_bbox()

    if not flag_locations and not end_date:
        end_date = date.today().isoformat()

    if min_years and start_date:
        _start_yr = int(start_date[:4])
        _end_yr   = int(end_date[:4]) if end_date else date.today().year
        _range_yrs = _end_yr - _start_yr + 1
        if min_years > _range_yrs:
            gs.fatal(
                "min_years={} exceeds the requested date range of {} year(s) "
                "({}-{}): no station can satisfy this filter. "
                "Lower min_years or widen the date range.".format(
                    min_years, _range_yrs, _start_yr, _end_yr)
            )

    # Apply fixed padding to bbox
    if bbox and padding > 0.0:
        w, s, e, n = bbox
        bbox = (w - padding, s - padding, e + padding, n + padding)
        gs.message("Padding bbox by {:.3g}°: W={:.4f} S={:.4f} E={:.4f} N={:.4f}".format(
            padding, *bbox))

    station_df = fetch_station_inventory()
    elem_inv_df = fetch_element_inventory()

    # Get basin centroid in lon/lat for hull containment check (if sample= given).
    centroid_ll = get_sample_centroid_ll(sample_map) if sample_map else None
    if centroid_ll:
        require_package('scipy')
        gs.message("Basin centroid for hull check: lon={:.4f} lat={:.4f}".format(*centroid_ll))

    # ── Pass 1: inventory-based expansion ─────────────────────────────────────
    # Expand bbox until min_stations and/or inventory hull criteria are met,
    # subject to shared max_distance / max_iterations budget.
    _STEP = 0.5   # degrees per expansion step
    need_expansion = (min_stations or centroid_ll) and not station_ids
    w, s, e, n = bbox if bbox else (None, None, None, None)
    expansion  = 0.0
    iterations = 0

    inv_gaps = {}   # populated inside the loop when centroid_ll is set

    if need_expansion:
        while True:
            filtered, elem_counts, elem_sids = filter_stations(
                station_df, elem_inv_df, (w, s, e, n),
                None, elements, min_years, start_date, end_date, fatal=False)

            counts_ok = (not min_stations or
                         (not filtered.empty and
                          min(elem_counts.get(el, 0) for el in elements) >= min_stations))

            # Hull criterion depends on whether a date range is specified:
            #   start_date given → per-decade inventory hull (temporal check)
            #   no start_date    → aggregate hull (basin_inside_hull), because
            #     inventory_decade_hull_gaps degenerates to a single current-decade
            #     check when start_date is None, which is not meaningful.
            if centroid_ll and not filtered.empty:
                if start_date:
                    inv_gaps = inventory_decade_hull_gaps(
                        filtered, elem_inv_df, elem_sids,
                        centroid_ll, start_date, end_date)
                    temporal_hull_ok = not inv_gaps
                else:
                    temporal_hull_ok = basin_inside_hull(
                        filtered, elem_sids, centroid_ll)
                    inv_gaps = {} if temporal_hull_ok else {'_': ['aggregate hull']}
            else:
                inv_gaps = {}
                temporal_hull_ok = not centroid_ll or filtered.empty

            if counts_ok and temporal_hull_ok:
                break

            if expansion >= max_distance or iterations >= max_iterations:
                reasons = []
                if not counts_ok:
                    reasons.append("min_stations={} not met ({})".format(
                        min_stations,
                        ', '.join('{}={}'.format(el, elem_counts.get(el, 0))
                                  for el in elements)))
                if not temporal_hull_ok:
                    reasons.append(
                        "inventory hull gap in decade(s): {}".format(
                            ', '.join(
                                '{} {}'.format(el, _year_ranges(bad))
                                for el, bad in sorted(inv_gaps.items()))))
                gs.warning(
                    "Pass 1: could not satisfy all criteria within "
                    "{:.1f}° / {:d} steps: {}. "
                    "Proceeding with best available stations; Pass 2 will "
                    "verify data coverage.".format(
                        max_distance, max_iterations, '; '.join(reasons)))
                break

            w -= _STEP; s -= _STEP; e += _STEP; n += _STEP
            expansion  += _STEP
            iterations += 1
            reasons = []
            if not counts_ok:
                reasons.append("min_stations={} not met".format(min_stations))
            if not temporal_hull_ok:
                if start_date:
                    bad_summary = ', '.join(
                        '{} {}'.format(el, _year_ranges(bad))
                        for el, bad in sorted(inv_gaps.items()))
                    reasons.append("inventory hull gap: {}".format(bad_summary))
                else:
                    reasons.append("basin outside aggregate station hull")
            gs.message(
                "  Pass 1: {} — expanding by {:.1f}° (total: {:.1f}°, step {:d})".format(
                    ', '.join(reasons), _STEP, expansion, iterations))

        if expansion > 0.0:
            gs.message(
                "Pass 1 complete: {:.1f}° / {:d} step(s), {:d} stations.".format(
                    expansion, iterations, len(filtered)))
    else:
        filtered, elem_counts, elem_sids = filter_stations(
            station_df, elem_inv_df, bbox, station_ids,
            elements, min_years, start_date, end_date)
        if centroid_ll and not filtered.empty and start_date:
            inv_gaps = inventory_decade_hull_gaps(
                filtered, elem_inv_df, elem_sids,
                centroid_ll, start_date, end_date)

    report_temporal_coverage(filtered, elem_inv_df, elements, start_date, end_date)

    if centroid_ll:
        report_inventory_hull_gaps(inv_gaps)

    if flag_locations:
        # Write vector and return without downloading time series.
        import geopandas as gpd
        from shapely.geometry import Point
        gdf = gpd.GeoDataFrame(
            filtered,
            geometry=[Point(lon, lat)
                      for lon, lat in zip(filtered['longitude'], filtered['latitude'])],
            crs='EPSG:4326')
        geodataframe_to_grass(gdf, output)
        gs.message("Station locations imported to '{}'.".format(output))
        return

    # ── Pre-assign cats before downloading so Pass 2 can add stations ──────────
    # Cats are assigned in alphabetical station_id order and held in memory until
    # the vector map is written after Pass 2.  This decouples cat assignment from
    # the vector write so new stations can be appended without re-importing.
    import pandas as pd

    def build_cat_map(df):
        """Assign cat 1..N in sorted station_id order; return sid->cat dict."""
        return {sid: i + 1
                for i, sid in enumerate(sorted(df['station_id'].tolist()))}

    cat_map = build_cat_map(filtered)
    # lon/lat coords keyed by cat, for data-based hull checks
    cat_to_xy_ll = {cat_map[row['station_id']]: (row['longitude'], row['latitude'])
                    for _, row in filtered.iterrows()}

    table_name = '{}_timeseries'.format(output)

    # ── Download data for Pass 1 stations ──────────────────────────────────────
    downloaded_sids = set(filtered['station_id'])
    gs.message("Pass 1: fetching time series for {:d} station(s)...".format(
        len(downloaded_sids)))
    if frequency == 'monthly':
        total_rows = fetch_and_write_monthly_timeseries(
            sorted(downloaded_sids), cat_map, start_date, end_date,
            q_flags, table_name)
    else:
        total_rows = fetch_and_write_timeseries(
            sorted(downloaded_sids), cat_map, set(elements),
            start_date, end_date, q_flags, table_name)

    # ── Pass 2: data-based expansion ───────────────────────────────────────────
    # Now that actual records are in SQLite, check per-decade hull coverage and
    # expand to fetch additional stations if gaps remain and budget allows.
    if centroid_ll and need_expansion and not station_ids:
        require_package('scipy')
        db_path = get_mapset_db()
        conn2 = sqlite3.connect(db_path)
        cur2  = conn2.cursor()

        pass2_iter = 0
        while True:
            gaps = check_data_decade_hull(
                cur2, table_name, elements, cat_to_xy_ll,
                centroid_ll, start_date, end_date)
            if not gaps:
                gs.message("Pass 2: data-based hull check passed.")
                break

            if expansion >= max_distance or iterations >= max_iterations:
                for elem, bad in sorted(gaps.items()):
                    gs.warning(
                        "Pass 2: {}: data hull gap in decades {} — "
                        "budget exhausted (max_distance={:.1f}°, "
                        "max_iterations={:d}).".format(
                            elem, _year_ranges(bad),
                            max_distance, max_iterations))
                break

            w -= _STEP; s -= _STEP; e += _STEP; n += _STEP
            expansion  += _STEP
            iterations += 1
            pass2_iter += 1

            # Find stations in expanded bbox not yet downloaded
            expanded_all, _, _ = filter_stations(
                station_df, elem_inv_df, (w, s, e, n),
                None, elements, min_years, start_date, end_date, fatal=False)
            new_stations = expanded_all[
                ~expanded_all['station_id'].isin(downloaded_sids)]

            if new_stations.empty:
                gs.message(
                    "  Pass 2 step {:d}: no new stations in expanded bbox "
                    "({:.1f}° total); stopping.".format(pass2_iter, expansion))
                break

            # Assign cats to new stations continuing from current max
            next_cat = max(cat_map.values()) + 1
            new_map  = {sid: next_cat + i
                        for i, sid in enumerate(sorted(new_stations['station_id']))}
            cat_map.update(new_map)
            for _, row in new_stations.iterrows():
                cat_to_xy_ll[cat_map[row['station_id']]] = (
                    row['longitude'], row['latitude'])

            new_sids = sorted(new_stations['station_id'])
            gs.message(
                "  Pass 2 step {:d}: expanding {:.1f}° — downloading "
                "{:d} new station(s).".format(pass2_iter, expansion, len(new_sids)))

            if frequency == 'monthly':
                total_rows += fetch_and_write_monthly_timeseries(
                    new_sids, cat_map, start_date, end_date, q_flags, table_name,
                    append=True)
            else:
                total_rows += fetch_and_write_timeseries(
                    new_sids, cat_map, set(elements),
                    start_date, end_date, q_flags, table_name, append=True)

            downloaded_sids.update(new_sids)
            filtered = pd.concat(
                [filtered, new_stations], ignore_index=True)

        conn2.close()

    # ── Write vector map (once, after all expansion) ───────────────────────────
    import geopandas as gpd
    from shapely.geometry import Point

    # Sort to match cat assignment order (alphabetical station_id)
    filtered_sorted = filtered.sort_values('station_id').reset_index(drop=True)
    geometry = [Point(lon, lat)
                for lon, lat in zip(filtered_sorted['longitude'],
                                    filtered_sorted['latitude'])]
    gdf = gpd.GeoDataFrame(filtered_sorted, geometry=geometry, crs='EPSG:4326')
    geodataframe_to_grass(gdf, output)
    gs.message("Station locations imported to '{}' ({:d} stations).".format(
        output, len(filtered_sorted)))

    gs.message("Time series stored: table '{}', {:,} records.".format(
        table_name, total_rows))
    gs.message(
        "Query example:\n"
        "  db.select sql=\"SELECT datetime, value FROM {t} "
        "WHERE cat=1 AND element='PRCP' LIMIT 10\"".format(t=table_name))


if __name__ == '__main__':
    main()
