"""Loader stubs for the four raw inputs.

Every loader reads a local CSV or Parquet file, validates it against the data
contract in :mod:`rainfall_pipeline.data.schema`, filters it to the requested
date range and bounding box, and returns a tidy DataFrame.

IMPORTANT
---------
None of these functions fabricate data. If the file is missing or empty they
raise :class:`MissingDataError` with an explanation of exactly what the file
must contain. ``_dummy_dataframe`` at the bottom exists **only** so the unit
tests can exercise the plumbing -- it is never called from a loader.
"""

from __future__ import annotations

from datetime import date as _date
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from ..config.regions import INDIA, BBox
from . import schema as sch

DateLike = Union[str, _date, pd.Timestamp, None]
PathLike = Union[str, Path]


class MissingDataError(FileNotFoundError):
    """Raised when an expected input file is absent or empty.

    The message always spells out the columns the file needs, so a user hitting
    this error knows exactly what to prepare.
    """


def _expected_format_message(
    *,
    source: str,
    path: PathLike,
    required: Sequence[str],
    optional: Sequence[str] = (),
    reason: str,
) -> str:
    """Build the standard 'here is what I need' error message.

    Args:
        source: Friendly name of the data source.
        path: Path that was attempted.
        required: Columns the file must contain.
        optional: Columns that are used if present.
        reason: Why the load failed ("does not exist" / "is empty").

    Returns:
        A multi-line, human-readable error message.
    """
    msg = [
        f"{source} data not found: the file '{path}' {reason}.",
        "",
        "No synthetic data will be generated. Please create this file first.",
        "",
        "Accepted formats: .parquet (preferred) or .csv",
        "One row per (date, lat, lon).",
        "",
        "Required columns:",
        sch.describe_columns(list(required)),
    ]
    if optional:
        msg += ["", "Optional columns (used if present):", sch.describe_columns(list(optional))]
    msg += [
        "",
        "See rainfall_pipeline/data/README.md for full details and an example header row.",
    ]
    return "\n".join(msg)


def _read_table(path: PathLike) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame based on its suffix.

    Args:
        path: File to read.

    Returns:
        The raw DataFrame, before any schema validation.

    Raises:
        ValueError: If the file extension is not supported.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(p)
    raise ValueError(
        f"Unsupported file type '{suffix}' for '{p}'. Use .parquet or .csv."
    )


def _load_validated(
    path: PathLike,
    *,
    source: str,
    required: Sequence[str],
    optional: Sequence[str],
    start_date: DateLike,
    end_date: DateLike,
    bbox: Optional[BBox],
) -> pd.DataFrame:
    """Shared read -> validate -> filter path used by all tabular loaders.

    Args:
        path: File to read.
        source: Friendly source name for error messages.
        required: Columns that must be present.
        optional: Columns kept if present.
        start_date: Inclusive lower date bound, or None for no bound.
        end_date: Inclusive upper date bound, or None for no bound.
        bbox: Spatial filter, or None for no spatial filter.

    Returns:
        A validated, filtered, type-normalised DataFrame.

    Raises:
        MissingDataError: If the file is missing or empty.
        SchemaError: If required columns are absent.
    """
    p = Path(path)
    if not p.exists():
        raise MissingDataError(
            _expected_format_message(
                source=source, path=p, required=required, optional=optional,
                reason="does not exist",
            )
        )
    if p.stat().st_size == 0:
        raise MissingDataError(
            _expected_format_message(
                source=source, path=p, required=required, optional=optional,
                reason="is empty (0 bytes)",
            )
        )

    df = _read_table(p)
    if df.empty:
        raise MissingDataError(
            _expected_format_message(
                source=source, path=p, required=required, optional=optional,
                reason="contains no rows",
            )
        )

    sch.validate_columns(df, required, source=source, path=str(p))

    keep = [c for c in list(required) + list(optional) if c in df.columns]
    df = sch.coerce_types(df[keep])
    df = filter_domain(df, start_date=start_date, end_date=end_date, bbox=bbox)

    if df.empty:
        raise MissingDataError(
            f"{source} file at '{p}' was read successfully but no rows fall inside "
            f"the requested window (start_date={start_date}, end_date={end_date}, "
            f"bbox={bbox}). Widen the filters or check the file's coordinate "
            f"conventions (lat in degrees north, lon in degrees east, 0-360 "
            f"longitudes must be converted to -180..180 first)."
        )
    return df.sort_values(sch.KEY_COLUMNS).reset_index(drop=True)


def filter_domain(
    df: pd.DataFrame,
    *,
    start_date: DateLike = None,
    end_date: DateLike = None,
    bbox: Optional[BBox] = None,
) -> pd.DataFrame:
    """Filter a schema-conformant frame to a date window and bounding box.

    Args:
        df: Frame containing ``date``, ``lat`` and ``lon``.
        start_date: Inclusive lower bound, or None.
        end_date: Inclusive upper bound, or None.
        bbox: Spatial bounds, or None.

    Returns:
        The filtered frame (a copy).
    """
    out = df
    if start_date is not None:
        out = out[out["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        out = out[out["date"] <= pd.Timestamp(end_date)]
    if bbox is not None:
        out = out[
            out["lat"].between(bbox.lat_min, bbox.lat_max)
            & out["lon"].between(bbox.lon_min, bbox.lon_max)
        ]
    return out.copy()


# ---------------------------------------------------------------------------
# Public loaders
#
# >>> PLUG IN YOUR REAL DATA HERE <<<
# Each of these reads a flat file you prepare. If your source is NetCDF/GRIB
# rather than a flat table, convert it once with xarray -- see
# ``flatten_gridded_dataset`` below -- and write the result to Parquet.
# ---------------------------------------------------------------------------

def load_era5(
    path: PathLike,
    start_date: DateLike = None,
    end_date: DateLike = None,
    bbox: Optional[BBox] = INDIA,
) -> pd.DataFrame:
    """Load the atmospheric predictor fields.

    Args:
        path: Path to the ERA5 (or equivalent reanalysis) table.
        start_date: Inclusive lower date bound.
        end_date: Inclusive upper date bound.
        bbox: Spatial filter. Defaults to the whole-India domain.

    Returns:
        DataFrame with ``date, lat, lon`` plus every atmospheric column, and
        ``elevation``/``coastal_distance``/``district`` if the file carries them.

    Raises:
        MissingDataError: If the file is missing or empty.
        SchemaError: If required columns are absent.
    """
    return _load_validated(
        path,
        source="ERA5 atmospheric predictor",
        required=sch.ERA5_REQUIRED,
        optional=sch.ERA5_OPTIONAL,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
    )


def load_observed_rainfall(
    path: PathLike,
    start_date: DateLike = None,
    end_date: DateLike = None,
    bbox: Optional[BBox] = INDIA,
) -> pd.DataFrame:
    """Load the ground-truth rainfall observations.

    Args:
        path: Path to the gridded observed rainfall table.
        start_date: Inclusive lower date bound.
        end_date: Inclusive upper date bound.
        bbox: Spatial filter.

    Returns:
        DataFrame with ``date, lat, lon, observed_mm`` (plus ``district`` if
        present).

    Raises:
        MissingDataError: If the file is missing or empty.
        SchemaError: If required columns are absent.
    """
    return _load_validated(
        path,
        source="Observed rainfall",
        required=sch.OBSERVED_REQUIRED,
        optional=sch.OBSERVED_OPTIONAL,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
    )


def load_raw_nwp_forecast(
    path: PathLike,
    start_date: DateLike = None,
    end_date: DateLike = None,
    bbox: Optional[BBox] = INDIA,
) -> pd.DataFrame:
    """Load the raw NWP / AI model forecast that will be bias-corrected.

    Args:
        path: Path to the raw forecast table.
        start_date: Inclusive lower date bound.
        end_date: Inclusive upper date bound.
        bbox: Spatial filter.

    Returns:
        DataFrame with ``date, lat, lon, raw_forecast_mm`` and ``lead_time`` if
        the file provides it.

    Raises:
        MissingDataError: If the file is missing or empty.
        SchemaError: If required columns are absent.
    """
    return _load_validated(
        path,
        source="Raw NWP forecast",
        required=sch.NWP_REQUIRED,
        optional=sch.NWP_OPTIONAL,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
    )


def load_district_boundaries(path: PathLike):
    """Load district polygons as a GeoDataFrame.

    Args:
        path: Path to a shapefile (``.shp``), GeoPackage, or GeoJSON of Indian
            district boundaries.

    Returns:
        A ``geopandas.GeoDataFrame`` reprojected to EPSG:4326 with a
        ``district`` column holding the district name.

    Raises:
        MissingDataError: If the file is absent or has no features.
        SchemaError: If no usable district-name column can be found.
    """
    # Imported lazily: geopandas pulls in GDAL, and the rest of the pipeline
    # should stay usable even if the geo stack is not installed.
    import geopandas as gpd

    from ..config.regions import DISTRICT_NAME_FIELD, GEO_CRS

    p = Path(path)
    if not p.exists():
        raise MissingDataError(
            "\n".join(
                [
                    f"District boundary file not found: '{p}' does not exist.",
                    "",
                    ">>> PLUG IN YOUR REAL SHAPEFILE HERE <<<",
                    "",
                    "Provide a polygon layer of Indian districts in any format",
                    "GeoPandas/Fiona can read: .shp (with its .dbf/.shx/.prj",
                    "siblings), .gpkg, or .geojson.",
                    "",
                    "Requirements:",
                    "  - geometry: Polygon or MultiPolygon, one feature per district",
                    "  - a district-name attribute. The pipeline looks for a column",
                    f"    called '{DISTRICT_NAME_FIELD}'; if your file names it",
                    "    something else (DISTRICT, dtname, NAME_2, ...) either rename",
                    "    it or set the RAINFALL_DISTRICT_NAME_FIELD environment variable.",
                    f"  - any CRS; it is reprojected to {GEO_CRS} on load.",
                ]
            )
        )

    gdf = gpd.read_file(p)
    if len(gdf) == 0:
        raise MissingDataError(
            f"District boundary file '{p}' was read but contains no features."
        )

    name_col = _resolve_district_name_column(gdf, DISTRICT_NAME_FIELD)
    gdf = gdf.rename(columns={name_col: sch.DISTRICT_COLUMN})
    gdf[sch.DISTRICT_COLUMN] = gdf[sch.DISTRICT_COLUMN].astype("string").str.strip()

    if gdf.crs is None:
        # A missing .prj is common in ad-hoc shapefiles. Assume lat/lon rather
        # than failing, but make the assumption visible.
        gdf = gdf.set_crs(GEO_CRS)
    else:
        gdf = gdf.to_crs(GEO_CRS)

    return gdf[[sch.DISTRICT_COLUMN, "geometry"]]


def _resolve_district_name_column(gdf, preferred: str) -> str:
    """Find the column in ``gdf`` that holds the district name.

    Args:
        gdf: The GeoDataFrame just read from disk.
        preferred: Column name configured by the user.

    Returns:
        The name of the column to use.

    Raises:
        SchemaError: If no plausible candidate exists.
    """
    lookup = {str(c).lower(): str(c) for c in gdf.columns}
    candidates = [preferred, "district", "DISTRICT", "dtname", "NAME_2", "distname"]
    for cand in candidates:
        if cand in gdf.columns:
            return cand
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    raise sch.SchemaError(
        f"Could not find a district-name column in the boundary file. "
        f"Looked for {candidates}; the file has {list(gdf.columns)}. "
        f"Rename your column or set RAINFALL_DISTRICT_NAME_FIELD."
    )


def flatten_gridded_dataset(
    dataset,
    variable_map: dict[str, str],
    *,
    time_dim: str = "time",
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
) -> pd.DataFrame:
    """Convert an :mod:`xarray` Dataset into a schema-shaped flat table.

    Provided as a one-off conversion helper for users whose ERA5/IMD data
    arrives as NetCDF or GRIB. It does no I/O of its own -- open the file with
    ``xarray.open_dataset`` yourself, pass the Dataset here, and write the
    result to Parquet.

    Args:
        dataset: An ``xarray.Dataset`` with time/lat/lon dimensions.
        variable_map: Mapping of ``{dataset_variable_name: schema_column_name}``,
            e.g. ``{"msl": "pressure_msl", "u": "wind_u_850"}``.
        time_dim: Name of the time dimension in the Dataset.
        lat_dim: Name of the latitude dimension.
        lon_dim: Name of the longitude dimension.

    Returns:
        A DataFrame with ``date, lat, lon`` plus one column per mapped variable.

    Raises:
        KeyError: If a variable named in ``variable_map`` is absent.
    """
    missing = [v for v in variable_map if v not in dataset.variables]
    if missing:
        raise KeyError(
            f"Variables {missing} are not in the dataset. "
            f"Available: {list(dataset.data_vars)}"
        )

    subset = dataset[list(variable_map)]
    df = subset.to_dataframe().reset_index()
    df = df.rename(
        columns={time_dim: "date", lat_dim: "lat", lon_dim: "lon", **variable_map}
    )
    # ERA5 longitudes run 0..360; the pipeline uses -180..180.
    df["lon"] = ((df["lon"] + 180.0) % 360.0) - 180.0
    return sch.coerce_types(df)


# ---------------------------------------------------------------------------
# TEST-ONLY dummy data.
#
# This is NOT a fallback and is deliberately not referenced by any loader.
# It exists so the unit tests can verify that the pipeline's shape is correct
# (columns line up, models fit, the API responds). The numbers below are
# arbitrary and physically meaningless -- never draw a conclusion from them,
# never tune a hyperparameter on them, never quote a metric computed on them.
# ---------------------------------------------------------------------------

def _dummy_dataframe(n_rows: int = 8) -> pd.DataFrame:
    """Return a handful of hand-written rows matching the common schema.

    FOR UNIT TESTS ONLY. See the module note above.

    Args:
        n_rows: How many of the hand-written rows to return (1-8).

    Returns:
        A DataFrame with every column in :data:`schema.COMMON_SCHEMA`.
    """
    rows = [
        # date,        lat,  lon,   district,     msl,    u850, v850, u200, v200, olr,  rh,   cape,  vort,    elev,  coast, raw,  obs
        ("2020-07-01", 19.0, 73.0, "Pune",       1002.0,  12.0,  4.0, -18.0, 2.0, 190.0, 88.0, 1400.0, 5.0e-5,  560.0,  60.0,  40.0,  72.0),
        ("2020-07-01", 22.0, 78.0, "Bhopal",     1004.0,   9.0,  2.0, -14.0, 1.0, 210.0, 80.0, 1100.0, 3.0e-5,  480.0, 520.0,  25.0,  31.0),
        ("2020-07-02", 19.0, 73.0, "Pune",       1000.0,  14.0,  5.0, -20.0, 3.0, 180.0, 91.0, 1700.0, 6.5e-5,  560.0,  60.0,  70.0, 130.0),
        ("2020-07-02", 22.0, 78.0, "Bhopal",     1006.0,   3.0,  1.0, -10.0, 0.5, 245.0, 62.0,  600.0, 1.0e-5,  480.0, 520.0,  10.0,   4.0),
        ("2020-07-03", 25.0, 85.0, "Patna",       998.0,  11.0,  6.0, -16.0, 2.5, 195.0, 86.0, 1300.0, 7.5e-5,   65.0, 480.0,  55.0,  95.0),
        ("2020-07-03", 11.0, 76.5, "Wayanad",    1007.0,  13.0,  3.0, -22.0, 1.5, 200.0, 93.0, 1500.0, 2.0e-5, 1100.0,  45.0,  85.0, 160.0),
        ("2020-07-04", 25.0, 85.0, "Patna",      1005.0,   1.5,  0.5,  -8.0, 0.2, 250.0, 58.0,  400.0, 0.5e-5,   65.0, 480.0,   6.0,   1.0),
        ("2020-07-04", 11.0, 76.5, "Wayanad",    1005.0,  15.0,  4.5, -21.0, 2.0, 185.0, 95.0, 1800.0, 3.5e-5, 1100.0,  45.0, 120.0, 215.0),
    ]
    df = pd.DataFrame(rows[:n_rows], columns=sch.COMMON_SCHEMA)
    return sch.coerce_types(df)
