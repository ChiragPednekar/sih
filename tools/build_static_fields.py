"""Derive the static per-cell fields the feature table needs.

``elevation`` and ``coastal_distance`` do not vary with the forecast date, so
they are built once against your grid and merged into the ERA5 table rather
than being recomputed every run. Roadmap phase 1 hides this work inside "data
acquisition"; this script is that work.

Two fields, two very different requirements:

* ``coastal_distance`` needs a coastline vector layer only, so it runs on the
  geopandas/shapely stack the pipeline already depends on.
* ``elevation`` needs a DEM raster, and therefore ``rasterio``, which is *not*
  a pipeline dependency. Ask for elevation without it installed and the script
  says so and exits rather than writing a column of nulls that would silently
  disable every orographic feature downstream.

Usage::

    python -m tools.build_static_fields \\
        --grid data_store/era5.parquet \\
        --coastline coastline.geojson \\
        --dem srtm_india.tif \\
        --out data_store/static_fields.parquet

The grid may be any file carrying ``lat``/``lon`` columns -- the script takes
the unique coordinate pairs from it, so pointing it at your ERA5 table
guarantees the static fields land on exactly the cells the pipeline uses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

#: Metres per degree of latitude, used to convert the projected distance the
#: geometry library returns into kilometres.
_KM_PER_DEGREE = 111.32


def read_grid(path: Path) -> pd.DataFrame:
    """Read the unique ``(lat, lon)`` cells out of a table.

    Args:
        path: A ``.parquet`` or ``.csv`` file with ``lat`` and ``lon`` columns.

    Returns:
        A frame of unique ``lat``/``lon`` pairs, sorted.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file has no ``lat``/``lon`` columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No grid file at '{path}'.")

    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    missing = {"lat", "lon"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"'{path}' has no {sorted(missing)} column(s); the grid file must "
            f"carry lat and lon. Found: {sorted(frame.columns)[:12]}"
        )
    cells = (
        frame[["lat", "lon"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["lat", "lon"])
        .reset_index(drop=True)
    )
    if cells.empty:
        raise ValueError(f"'{path}' produced no usable (lat, lon) pairs.")
    return cells


def compute_coastal_distance(cells: pd.DataFrame, coastline_path: Path) -> pd.Series:
    """Great-circle distance from every cell to the nearest coastline, in km.

    Distance is measured in degrees against the coastline geometry and then
    scaled, with a ``cos(lat)`` correction on the longitudinal component. Over
    India that is accurate to a few percent, which is far finer than the 75 km
    threshold the coastal regime rule actually uses.

    Args:
        cells: Frame with ``lat`` and ``lon`` columns.
        coastline_path: Any vector file geopandas can read (GeoJSON,
            shapefile, GeoPackage) holding coastline lines or polygons.

    Returns:
        Distance in km, aligned to ``cells.index``.

    Raises:
        FileNotFoundError: If the coastline file does not exist.
        ValueError: If the coastline file holds no geometry.
    """
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.ops import unary_union

    coastline_path = Path(coastline_path)
    if not coastline_path.exists():
        raise FileNotFoundError(f"No coastline file at '{coastline_path}'.")

    coast = gpd.read_file(coastline_path)
    if coast.empty:
        raise ValueError(f"'{coastline_path}' contains no geometry.")
    if coast.crs is not None and coast.crs.to_epsg() != 4326:
        coast = coast.to_crs("EPSG:4326")

    # Polygons describe land, not the coast; their boundary is the coastline.
    geometry = unary_union(
        [g.boundary if g.geom_type in {"Polygon", "MultiPolygon"} else g for g in coast.geometry]
    )

    latitudes = cells["lat"].to_numpy(dtype="float64")
    distances_deg = np.array(
        [Point(lon, lat).distance(geometry) for lat, lon in zip(latitudes, cells["lon"])],
        dtype="float64",
    )
    # Scale by latitude so a degree of longitude is not treated as a degree of
    # latitude at 20 N, where it is about 6% shorter.
    scale = _KM_PER_DEGREE * np.sqrt((1.0 + np.cos(np.radians(latitudes)) ** 2) / 2.0)
    return pd.Series(distances_deg * scale, index=cells.index, name="coastal_distance")


def compute_elevation(cells: pd.DataFrame, dem_path: Path) -> pd.Series:
    """Sample a DEM raster at every grid-cell centre, in metres.

    Args:
        cells: Frame with ``lat`` and ``lon`` columns.
        dem_path: A DEM raster (GeoTIFF or anything rasterio reads).

    Returns:
        Elevation in metres, aligned to ``cells.index``. Sea-level nodata is
        returned as 0.0; cells outside the raster come back as NaN so they are
        visible rather than silently flattened.

    Raises:
        FileNotFoundError: If the DEM does not exist.
        SystemExit: If ``rasterio`` is not installed.
    """
    try:
        import rasterio
    except ImportError:
        sys.exit(
            "Reading a DEM needs rasterio, which is not a pipeline dependency.\n"
            "  pip install rasterio\n"
            "Or omit --dem and supply the elevation column yourself. This exits "
            "rather than writing nulls, because a null elevation column would "
            "silently disable every orographic feature in the model."
        )

    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"No DEM at '{dem_path}'.")

    with rasterio.open(dem_path) as src:
        nodata = src.nodata
        samples = np.array(
            [v[0] for v in src.sample(list(zip(cells["lon"], cells["lat"])))],
            dtype="float64",
        )

    if nodata is not None:
        # SRTM marks ocean as nodata; that is sea level, not missing.
        samples = np.where(samples == nodata, 0.0, samples)
    samples = np.where(samples < -500.0, np.nan, samples)
    return pd.Series(samples, index=cells.index, name="elevation")


def build_static_fields(
    grid_path: Path,
    out_path: Path,
    *,
    coastline_path: Optional[Path] = None,
    dem_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build the static field table and write it to ``out_path``.

    Args:
        grid_path: Table carrying the ``lat``/``lon`` grid.
        out_path: Where to write the resulting parquet file.
        coastline_path: Coastline vector layer for ``coastal_distance``.
        dem_path: DEM raster for ``elevation``.

    Returns:
        The frame that was written.

    Raises:
        ValueError: If neither a coastline nor a DEM was supplied.
    """
    if coastline_path is None and dem_path is None:
        raise ValueError("Supply at least one of --coastline or --dem.")

    cells = read_grid(grid_path)
    print(f"Grid: {len(cells):,} unique cells from {grid_path}")

    if coastline_path is not None:
        cells["coastal_distance"] = compute_coastal_distance(cells, coastline_path)
        print(
            f"  coastal_distance : {cells['coastal_distance'].min():.1f} to "
            f"{cells['coastal_distance'].max():.1f} km"
        )
    if dem_path is not None:
        cells["elevation"] = compute_elevation(cells, dem_path)
        finite = cells["elevation"].dropna()
        print(
            f"  elevation        : {finite.min():.0f} to {finite.max():.0f} m"
            + (f" ({cells['elevation'].isna().sum()} cells outside the DEM)"
               if cells["elevation"].isna().any() else "")
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")
    print(
        "\nMerge it into your ERA5 table on (lat, lon) before training:\n"
        "  era5 = era5.merge(static, on=['lat', 'lon'], how='left')"
    )
    return cells


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Derive elevation and coastal_distance for a lat/lon grid.",
    )
    parser.add_argument("--grid", required=True, type=Path,
                        help="Table with lat/lon columns (usually your era5.parquet).")
    parser.add_argument("--out", required=True, type=Path,
                        help="Destination parquet file.")
    parser.add_argument("--coastline", type=Path, default=None,
                        help="Coastline vector layer, for coastal_distance.")
    parser.add_argument("--dem", type=Path, default=None,
                        help="DEM raster, for elevation. Needs rasterio.")
    args = parser.parse_args(argv)

    try:
        build_static_fields(
            args.grid, args.out, coastline_path=args.coastline, dem_path=args.dem
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
