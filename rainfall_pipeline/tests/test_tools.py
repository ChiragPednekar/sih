"""Tests for the data-preparation helpers in ``tools/``.

These cover the static-field builder, which produces ``elevation`` and
``coastal_distance`` -- the two columns the pipeline expects to already exist
but has no way of deriving on its own.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools.build_static_fields import (
    build_static_fields,
    compute_coastal_distance,
    read_grid,
)


@pytest.fixture
def grid_file(tmp_path: Path) -> Path:
    """A small lat/lon grid written as parquet, with a duplicate row."""
    frame = pd.DataFrame(
        {
            "lat": [18.0, 18.0, 19.0, 19.0, 18.0],
            "lon": [72.0, 73.0, 72.0, 73.0, 72.0],
            "humidity": [1, 2, 3, 4, 5],
        }
    )
    path = tmp_path / "grid.parquet"
    frame.to_parquet(path, index=False)
    return path


@pytest.fixture
def coastline_file(tmp_path: Path) -> Path:
    """A straight north-south coastline at 72.0 E."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString

    layer = gpd.GeoDataFrame(
        {"name": ["coast"]},
        geometry=[LineString([(72.0, 10.0), (72.0, 25.0)])],
        crs="EPSG:4326",
    )
    path = tmp_path / "coast.geojson"
    layer.to_file(path, driver="GeoJSON")
    return path


def test_read_grid_deduplicates_cells(grid_file: Path) -> None:
    """Repeated dates must not produce repeated static rows."""
    cells = read_grid(grid_file)
    assert len(cells) == 4
    assert list(cells.columns) == ["lat", "lon"]


def test_read_grid_rejects_a_table_without_coordinates(tmp_path: Path) -> None:
    """A file with no lat/lon must say so, naming what it did find."""
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"date": ["2020-06-01"], "rain": [1.0]}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="must carry lat and lon|no .*column"):
        read_grid(path)


def test_read_grid_reports_a_missing_file(tmp_path: Path) -> None:
    """A missing grid file must raise a path-carrying error, not a bare KeyError."""
    with pytest.raises(FileNotFoundError, match="No grid file"):
        read_grid(tmp_path / "absent.parquet")


def test_coastal_distance_is_zero_on_the_coast(grid_file, coastline_file) -> None:
    """A cell sitting on the coastline must be at distance zero."""
    cells = read_grid(grid_file)
    distance = compute_coastal_distance(cells, coastline_file)
    on_coast = cells["lon"] == 72.0
    assert distance[on_coast].abs().max() == pytest.approx(0.0, abs=1e-6)


def test_coastal_distance_grows_with_separation(grid_file, coastline_file) -> None:
    """A cell one degree inland must be roughly one degree of longitude away."""
    cells = read_grid(grid_file)
    distance = compute_coastal_distance(cells, coastline_file)
    inland = distance[cells["lon"] == 73.0]
    # ~111 km per degree, reduced by the latitude correction near 18-19 N.
    assert inland.min() > 90.0
    assert inland.max() < 115.0


def test_coastal_distance_accepts_polygon_coastlines(grid_file, tmp_path) -> None:
    """A landmass polygon must be treated as its boundary, not its interior."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    path = tmp_path / "land.geojson"
    gpd.GeoDataFrame(
        {"name": ["land"]}, geometry=[box(72.0, 10.0, 80.0, 25.0)], crs="EPSG:4326"
    ).to_file(path, driver="GeoJSON")

    cells = read_grid(grid_file)
    distance = compute_coastal_distance(cells, path)
    # Every test cell lies on or inside the polygon, so distance-to-interior
    # would be 0 everywhere. Distance-to-boundary must not be.
    assert distance.max() > 50.0


def test_build_writes_a_mergeable_table(grid_file, coastline_file, tmp_path) -> None:
    """The output must join back onto the grid on (lat, lon)."""
    out = tmp_path / "static.parquet"
    build_static_fields(grid_file, out, coastline_path=coastline_file)

    written = pd.read_parquet(out)
    assert set(written.columns) == {"lat", "lon", "coastal_distance"}
    merged = pd.read_parquet(grid_file).merge(written, on=["lat", "lon"], how="left")
    assert merged["coastal_distance"].notna().all(), "every grid row must find a match"


def test_build_requires_at_least_one_source(grid_file, tmp_path) -> None:
    """Asking for nothing must be an error, not an empty file."""
    with pytest.raises(ValueError, match="at least one"):
        build_static_fields(grid_file, tmp_path / "out.parquet")
