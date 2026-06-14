"""
Pytest tests for pure-Python utility functions in v.in.ghcn.

These tests cover logic that has no GRASS dependency and can run without a
GRASS installation or location.  A minimal mock of grass.script is injected
before the module is imported so that gs.message / gs.fatal / etc. are stubs.

Targeted regressions:
  - filter_stations() must return a 3-tuple even on the fatal=False empty path
  - basin_inside_hull() used for no-start_date case; inventory_decade_hull_gaps()
    for the dated case — dispatch depends on whether start_date is given
  - inventory_decade_hull_gaps() degenerates when start_date is absent (design
    validation: shows why basin_inside_hull() is still needed)
  - check_data_decade_hull() correctly queries SQLite per decade
  - _year_ranges() compact formatting
"""

import importlib.util
import sqlite3
import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Inject a grass.script mock so the module can be imported without GRASS.
# gs.fatal is made to raise RuntimeError so we can assert on fatal conditions.
# ---------------------------------------------------------------------------

_gs_mock = MagicMock()
_gs_mock.fatal.side_effect = RuntimeError("gs.fatal")
_gs_mock.warning = MagicMock()
_gs_mock.message = MagicMock()

_grass_mock = ModuleType("grass")
_grass_mock.script = _gs_mock

sys.modules.setdefault("grass", _grass_mock)
sys.modules.setdefault("grass.script", _gs_mock)

_spec = importlib.util.spec_from_file_location(
    "v_in_ghcn",
    str(__file__).replace("testsuite/test_hull_functions.py", "v.in.ghcn.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

filter_stations              = _mod.filter_stations
basin_inside_hull            = _mod.basin_inside_hull
inventory_decade_hull_gaps   = _mod.inventory_decade_hull_gaps
check_data_decade_hull       = _mod.check_data_decade_hull
_year_ranges                 = _mod._year_ranges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _station_df(*rows):
    """Build a minimal station DataFrame.  Each row: (sid, lat, lon)."""
    return pd.DataFrame(
        [{"station_id": sid, "latitude": lat, "longitude": lon,
          "elevation": 300.0, "state": "MN", "name": sid,
          "gsn_flag": "", "hcncrn_flag": "", "wmo_id": ""}
         for sid, lat, lon in rows]
    )


def _elem_inv(*rows):
    """Build element inventory DataFrame.  Each row: (sid, element, fy, ly)."""
    return pd.DataFrame(
        [{"station_id": sid, "latitude": 44.0, "longitude": -93.0,
          "element": elem, "firstyear": fy, "lastyear": ly}
         for sid, elem, fy, ly in rows]
    )


def _centroid():
    """Basin centroid at (lon=-93.0, lat=44.0) — centre of our test hulls."""
    return (-93.0, 44.0)


def _surrounding_stations():
    """Four stations that form a hull enclosing the centroid (44N, 93W)."""
    return _station_df(
        ("SID_N", 46.0, -93.0),   # north
        ("SID_S", 42.0, -93.0),   # south
        ("SID_E", 44.0, -90.0),   # east
        ("SID_W", 44.0, -96.0),   # west
    )


def _surrounding_sids():
    return {"SID_N", "SID_S", "SID_E", "SID_W"}


# ---------------------------------------------------------------------------
# _year_ranges
# ---------------------------------------------------------------------------

class TestYearRanges:
    def test_single(self):
        assert _year_ranges([1950]) == "1950"

    def test_contiguous(self):
        assert _year_ranges([1890, 1891, 1892]) == "1890-1892"

    def test_gap(self):
        assert _year_ranges([1890, 1891, 1900]) == "1890-1891, 1900"

    def test_multiple_ranges(self):
        assert _year_ranges([1890, 1891, 1900, 1910, 1911]) == "1890-1891, 1900, 1910-1911"

    def test_empty(self):
        assert _year_ranges([]) == ""


# ---------------------------------------------------------------------------
# filter_stations
# ---------------------------------------------------------------------------

class TestFilterStationsReturnType:
    """filter_stations must always return a 3-tuple (df, counts, sids).

    Regression: the fatal=False empty-df early-return path previously returned
    a 2-tuple (df, {}), causing a ValueError when callers unpacked three values.
    """

    def _make_inv(self):
        return _elem_inv(("FAR_SID", "PRCP", 1950, 2020))

    def test_empty_bbox_returns_three_tuple(self):
        """No stations inside bbox → 3-tuple with two empty dicts."""
        station_df = _station_df(("FAR", 10.0, 10.0))
        elem_inv   = self._make_inv()
        bbox = (-95.0, 43.0, -90.0, 46.0)   # FAR is not inside

        result = filter_stations(
            station_df, elem_inv, bbox,
            station_ids=None,
            elements=["PRCP"],
            min_years=None,
            start_date=None,
            end_date=None,
            fatal=False,
        )

        assert len(result) == 3, "expected 3-tuple, got {}".format(len(result))
        df, counts, sids = result
        assert df.empty
        assert counts == {}
        assert sids == {}

    def test_normal_path_returns_three_tuple(self):
        """Stations found → 3-tuple with populated counts and sids."""
        station_df = _surrounding_stations()
        elem_inv = _elem_inv(
            ("SID_N", "PRCP", 1950, 2020),
            ("SID_S", "PRCP", 1950, 2020),
            ("SID_E", "PRCP", 1950, 2020),
            ("SID_W", "PRCP", 1950, 2020),
        )
        bbox = (-98.0, 40.0, -88.0, 48.0)

        df, counts, sids = filter_stations(
            station_df, elem_inv, bbox,
            station_ids=None,
            elements=["PRCP"],
            min_years=None,
            start_date=None,
            end_date=None,
        )

        assert len(df) == 4
        assert counts["PRCP"] == 4
        assert sids["PRCP"] == _surrounding_sids()

    def test_min_years_filter_updates_sids(self):
        """Stations with insufficient record years are excluded from sids."""
        station_df = _surrounding_stations()
        elem_inv = _elem_inv(
            ("SID_N", "PRCP", 2015, 2020),   # 6 years — too short
            ("SID_S", "PRCP", 1950, 2020),
            ("SID_E", "PRCP", 1950, 2020),
            ("SID_W", "PRCP", 1950, 2020),
        )
        bbox = (-98.0, 40.0, -88.0, 48.0)

        df, counts, sids = filter_stations(
            station_df, elem_inv, bbox,
            station_ids=None,
            elements=["PRCP"],
            min_years=10,
            start_date="1950-01-01",
            end_date="2020-12-31",
        )

        # SID_N passes the overall filter (has PRCP on inventory) but NOT
        # min_years, so it is excluded from per_element_sids for PRCP.
        assert "SID_N" not in sids["PRCP"]
        assert counts["PRCP"] == 3


# ---------------------------------------------------------------------------
# basin_inside_hull
# ---------------------------------------------------------------------------

class TestBasinInsideHull:
    """basin_inside_hull: aggregate check, used when no start_date is given."""

    def test_centroid_inside_hull(self):
        df   = _surrounding_stations()
        sids = {"PRCP": _surrounding_sids()}
        assert basin_inside_hull(df, sids, _centroid()) is True

    def test_centroid_outside_hull(self):
        """All stations on the north side — centroid (44N) is below the hull."""
        df = _station_df(
            ("SID_A", 46.0, -95.0),
            ("SID_B", 46.0, -93.0),
            ("SID_C", 46.0, -91.0),
        )
        sids = {"PRCP": {"SID_A", "SID_B", "SID_C"}}
        assert basin_inside_hull(df, sids, _centroid()) is False

    def test_fewer_than_three_stations_returns_false(self):
        df   = _surrounding_stations()
        sids = {"PRCP": {"SID_N", "SID_S"}}   # only 2
        assert basin_inside_hull(df, sids, _centroid()) is False

    def test_one_element_fails_returns_false(self):
        """PRCP hull passes but TMAX fails → overall False."""
        df = _surrounding_stations()
        # Add a TMAX station set that doesn't enclose the centroid
        df_tmax = _station_df(
            ("TMAX_A", 46.0, -95.0),
            ("TMAX_B", 46.0, -93.0),
            ("TMAX_C", 46.0, -91.0),
        )
        df_all = pd.concat([df, df_tmax], ignore_index=True)
        sids = {
            "PRCP": _surrounding_sids(),
            "TMAX": {"TMAX_A", "TMAX_B", "TMAX_C"},
        }
        assert basin_inside_hull(df_all, sids, _centroid()) is False


# ---------------------------------------------------------------------------
# inventory_decade_hull_gaps
# ---------------------------------------------------------------------------

class TestInventoryDecadeHullGaps:
    """inventory_decade_hull_gaps: temporal per-decade check using firstyear/lastyear."""

    def _surrounding_inv_all_decades(self, start_yr=1890, end_yr=2020):
        """All four surrounding stations active from start_yr to end_yr."""
        rows = []
        for sid in ("SID_N", "SID_S", "SID_E", "SID_W"):
            rows.append((sid, "PRCP", start_yr, end_yr))
        return _elem_inv(*rows)

    def test_full_coverage_returns_empty(self):
        df      = _surrounding_stations()
        inv     = self._surrounding_inv_all_decades()
        sids    = {"PRCP": _surrounding_sids()}
        gaps = inventory_decade_hull_gaps(
            df, inv, sids, _centroid(), "1890-01-01", "2020-12-31")
        assert gaps == {}

    def test_early_decade_gap(self):
        """Stations only active from 1920 onwards → 1890-1919 decades fail."""
        df   = _surrounding_stations()
        inv  = self._surrounding_inv_all_decades(start_yr=1920)
        sids = {"PRCP": _surrounding_sids()}
        gaps = inventory_decade_hull_gaps(
            df, inv, sids, _centroid(), "1890-01-01", "2020-12-31")

        assert "PRCP" in gaps
        assert 1890 in gaps["PRCP"]
        assert 1910 in gaps["PRCP"]
        assert 1920 not in gaps["PRCP"]

    def test_per_element_gap(self):
        """PRCP covered; TMAX has early gap → only TMAX in gaps dict."""
        df = _surrounding_stations()
        inv = _elem_inv(
            ("SID_N", "PRCP", 1890, 2020),
            ("SID_S", "PRCP", 1890, 2020),
            ("SID_E", "PRCP", 1890, 2020),
            ("SID_W", "PRCP", 1890, 2020),
            ("SID_N", "TMAX", 1950, 2020),   # TMAX only from 1950
            ("SID_S", "TMAX", 1950, 2020),
            ("SID_E", "TMAX", 1950, 2020),
            ("SID_W", "TMAX", 1950, 2020),
        )
        sids = {"PRCP": _surrounding_sids(), "TMAX": _surrounding_sids()}
        gaps = inventory_decade_hull_gaps(
            df, inv, sids, _centroid(), "1890-01-01", "2020-12-31")

        assert "PRCP" not in gaps
        assert "TMAX" in gaps
        assert 1890 in gaps["TMAX"]
        assert 1950 not in gaps["TMAX"]

    def test_degeneration_without_start_date(self):
        """When start_date is None, start_yr falls back to end_yr (today's year).

        This means only the current decade is checked — a degenerate result that
        validates why basin_inside_hull() must be used instead when start_date
        is absent.  Stations active only 1890–1950 will appear to pass because
        the current-decade check is vacuous (no active stations → < 3 coords →
        the function appends a gap, but only for the current decade).
        """
        df  = _surrounding_stations()
        inv = self._surrounding_inv_all_decades(start_yr=1890, end_yr=1950)
        sids = {"PRCP": _surrounding_sids()}

        # With no start_date, inventory_decade_hull_gaps checks only the
        # current decade (≈2020).  Since our stations ended in 1950 they are
        # absent from the current decade → gap is reported, but only for 2020,
        # not for the historically covered 1890–1950 range.
        gaps = inventory_decade_hull_gaps(
            df, inv, sids, _centroid(), start_date=None, end_date="2024-12-31")

        assert "PRCP" in gaps, (
            "Expected a gap in the current decade (stations ended 1950), "
            "confirming degenerate single-decade behaviour without start_date")
        # Only the current decade is checked — historical gaps are invisible
        assert len(gaps["PRCP"]) == 1, (
            "Without start_date only one decade is checked; "
            "got {}".format(gaps["PRCP"]))


# ---------------------------------------------------------------------------
# check_data_decade_hull  (SQLite-based, no GRASS required)
# ---------------------------------------------------------------------------

class TestCheckDataDecadeHull:
    """check_data_decade_hull: per-decade hull check on actual downloaded records."""

    def _make_db_with_records(self, records):
        """Create an in-memory SQLite db and return (conn, cursor, table_name)."""
        conn  = sqlite3.connect(":memory:")
        cur   = conn.cursor()
        table = "ghcn_timeseries"
        cur.execute(
            'CREATE TABLE "{}" '
            '(cat INTEGER, station_id TEXT, datetime TEXT, '
            ' element TEXT, value REAL, q_flag TEXT)'.format(table)
        )
        cur.executemany(
            'INSERT INTO "{}" VALUES (?, ?, ?, ?, ?, ?)'.format(table), records
        )
        conn.commit()
        return conn, cur, table

    def _cat_to_xy(self):
        """cat → (lon, lat) matching _surrounding_stations()."""
        return {
            1: (-93.0, 46.0),   # SID_E  (sorted: E=1, N=2, S=3, W=4)
            2: (-93.0, 46.0),   # placeholder — override below
            3: (-93.0, 42.0),
            4: (-96.0, 44.0),
        }

    def _full_cat_to_xy(self):
        """Alphabetical order: SID_E=1, SID_N=2, SID_S=3, SID_W=4."""
        return {
            1: (-90.0, 44.0),   # SID_E
            2: (-93.0, 46.0),   # SID_N
            3: (-93.0, 42.0),   # SID_S
            4: (-96.0, 44.0),   # SID_W
        }

    def _records_for_decade(self, cat_to_xy, decade, elements=("PRCP",)):
        """One record per cat per element in the given decade."""
        rows = []
        for cat in cat_to_xy:
            for elem in elements:
                rows.append(
                    (cat, "SID_{}".format(cat),
                     "{}-06-15".format(decade), elem, 5.0, None)
                )
        return rows

    def test_full_coverage_no_gaps(self):
        cat_to_xy = self._full_cat_to_xy()
        records   = self._records_for_decade(cat_to_xy, 1960)
        conn, cur, table = self._make_db_with_records(records)

        gaps = check_data_decade_hull(
            cur, table, ["PRCP"], cat_to_xy, _centroid(),
            "1960-01-01", "1969-12-31")
        conn.close()
        assert gaps == {}

    def test_missing_decade_data(self):
        """No records in 1890s → gap reported for that decade."""
        cat_to_xy = self._full_cat_to_xy()
        # Records only in 1950s
        records   = self._records_for_decade(cat_to_xy, 1950)
        conn, cur, table = self._make_db_with_records(records)

        gaps = check_data_decade_hull(
            cur, table, ["PRCP"], cat_to_xy, _centroid(),
            "1890-01-01", "1959-12-31")
        conn.close()

        assert "PRCP" in gaps
        assert 1890 in gaps["PRCP"]
        assert 1950 not in gaps["PRCP"]

    def test_fewer_than_three_cats_with_data(self):
        """Only 2 cats have records in a decade → hull fails (< 3 points)."""
        cat_to_xy = self._full_cat_to_xy()
        records = [
            (1, "SID_E", "1890-06-15", "PRCP", 5.0, None),
            (2, "SID_N", "1890-06-15", "PRCP", 5.0, None),
        ]
        conn, cur, table = self._make_db_with_records(records)

        gaps = check_data_decade_hull(
            cur, table, ["PRCP"], cat_to_xy, _centroid(),
            "1890-01-01", "1899-12-31")
        conn.close()

        assert "PRCP" in gaps
        assert 1890 in gaps["PRCP"]
