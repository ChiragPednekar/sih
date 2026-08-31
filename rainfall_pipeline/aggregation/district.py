"""Stage 5 -- district-level product.

Grid-cell predictions are spatially joined to district polygons and reduced to
one row per district per date:

* **mean corrected rainfall** -- the district's areal average. A mean is the
  right summary for rainfall totals; using the max here would make every
  district look extreme because of one convective cell.
* **max heavy-rain probability** -- deliberately *not* a mean. A warning product
  should trigger on the worst cell in the district: if one taluka has a 70%
  chance of 115 mm, the district needs a warning even if the average is 15%.
* **warning level** -- ``none``/``watch``/``warning``/``severe`` from the
  probability cut-points in :mod:`rainfall_pipeline.config.thresholds`.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..config.regions import GEO_CRS, BBox
from ..config.thresholds import (
    RAIN_THRESHOLDS,
    WARNING_NONE,
    WARNING_SEVERE,
    WARNING_WARNING,
    WARNING_WATCH,
    WARNING_RULES,
    WarningLevelConfig,
)
from ..data import schema as sch

#: Column holding the corrected forecast in the grid prediction frame.
CORRECTED_COLUMN = "corrected_forecast_mm"

#: Prefix for probability columns, e.g. ``prob_heavy``.
PROB_PREFIX = "prob_"


class AggregationError(ValueError):
    """Raised when grid predictions cannot be aggregated to districts."""


def probability_column(threshold_name: str) -> str:
    """Return the frame column name for a threshold's probability.

    Args:
        threshold_name: e.g. ``"heavy"``.

    Returns:
        e.g. ``"prob_heavy"``.
    """
    return f"{PROB_PREFIX}{threshold_name}"


def assign_districts(
    grid_df: pd.DataFrame,
    district_gdf,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
):
    """Spatially join grid-cell centres to district polygons.

    Uses a point-in-polygon join on the cell centre. This is the right choice
    for a 0.25 deg grid over India: cells are ~25 km across and most districts
    are considerably larger, so area-weighted allocation would add complexity
    for little benefit. If you move to a much coarser grid, replace this with an
    area-weighted overlay instead.

    Args:
        grid_df: Grid-cell frame with latitude and longitude columns.
        district_gdf: District polygons from
            :func:`~rainfall_pipeline.data.loaders.load_district_boundaries`.
        lat_col: Latitude column name.
        lon_col: Longitude column name.

    Returns:
        A ``geopandas.GeoDataFrame`` of the input rows with a ``district``
        column. Cells falling outside every polygon get ``<NA>``.

    Raises:
        AggregationError: If the coordinate columns are missing.
    """
    import geopandas as gpd

    missing = [c for c in (lat_col, lon_col) if c not in grid_df.columns]
    if missing:
        raise AggregationError(f"Grid predictions are missing column(s) {missing}.")

    points = gpd.GeoDataFrame(
        grid_df.copy(),
        geometry=gpd.points_from_xy(grid_df[lon_col], grid_df[lat_col]),
        crs=GEO_CRS,
    )
    polys = district_gdf.to_crs(GEO_CRS)[[sch.DISTRICT_COLUMN, "geometry"]]

    # Drop any pre-existing district column so the join result is unambiguous.
    if sch.DISTRICT_COLUMN in points.columns:
        points = points.drop(columns=[sch.DISTRICT_COLUMN])

    joined = gpd.sjoin(points, polys, how="left", predicate="within")
    return joined.drop(columns=[c for c in ("index_right",) if c in joined.columns])


def assign_region(
    df: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    subregions: Optional[Dict[str, BBox]] = None,
) -> pd.Series:
    """Label each grid cell with a named sub-region.

    Districts are the operational unit, but there are hundreds of them and a
    per-district table is unreadable in a report. Sub-regions give the coarser
    cut that actually answers "where does this help?" -- the west coast, the
    core monsoon zone, the northeast -- which is the level at which a monsoon
    post-processing scheme succeeds or fails.

    The boxes overlap in places (they are coarse rectangles, not a partition),
    so the first match in :data:`~rainfall_pipeline.config.regions.SUBREGIONS`
    order wins. Cells matching nothing get ``"other"`` rather than NaN, so no
    row silently drops out of the breakdown.

    Args:
        df: Frame with latitude and longitude columns.
        lat_col: Latitude column name.
        lon_col: Longitude column name.
        subregions: ``{name: BBox}``. Defaults to the configured set.

    Returns:
        A string Series of region names aligned to ``df.index``.

    Raises:
        AggregationError: If the coordinate columns are missing.
    """
    from ..config.regions import SUBREGIONS

    missing = [c for c in (lat_col, lon_col) if c not in df.columns]
    if missing:
        raise AggregationError(f"assign_region needs column(s) {missing}.")

    boxes = SUBREGIONS if subregions is None else subregions
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")

    out = pd.Series("other", index=df.index, dtype="object")
    assigned = pd.Series(False, index=df.index)
    for name, box in boxes.items():
        hit = (
            lat.between(box.lat_min, box.lat_max)
            & lon.between(box.lon_min, box.lon_max)
            & ~assigned
        )
        out = out.mask(hit, name)
        assigned = assigned | hit
    return out.rename("region")


def classify_warning_level(
    probabilities: Dict[str, float],
    config: WarningLevelConfig = WARNING_RULES,
) -> str:
    """Map threshold probabilities to a single warning level.

    Args:
        probabilities: ``{threshold_name: probability}``. ``heavy`` and
            ``very_heavy`` are consulted; missing keys are treated as 0.
        config: Probability cut-points.

    Returns:
        One of ``none``, ``watch``, ``warning``, ``severe``.
    """
    heavy = float(probabilities.get("heavy", 0.0) or 0.0)
    very_heavy = float(probabilities.get("very_heavy", 0.0) or 0.0)

    if very_heavy >= config.severe_very_heavy_prob:
        return WARNING_SEVERE
    if heavy >= config.warning_heavy_prob:
        return WARNING_WARNING
    if heavy >= config.watch_heavy_prob:
        return WARNING_WATCH
    return WARNING_NONE


def aggregate_to_district(
    grid_predictions_df: pd.DataFrame,
    district_gdf=None,
    *,
    threshold_names: Optional[Iterable[str]] = None,
    config: WarningLevelConfig = WARNING_RULES,
    return_geometry: bool = False,
):
    """Reduce grid-cell predictions to one row per (date, district).

    Args:
        grid_predictions_df: One row per ``(date, lat, lon)`` with at least
            ``corrected_forecast_mm`` and the ``prob_*`` columns. May already
            carry a ``district`` column, in which case ``district_gdf`` is
            optional.
        district_gdf: District polygons. Required if ``grid_predictions_df``
            has no usable ``district`` column, or if ``return_geometry`` is set.
        threshold_names: Which thresholds to aggregate. Defaults to the IMD
            categories.
        config: Warning-level cut-points.
        return_geometry: If True, return a GeoDataFrame with the district
            polygon attached, ready to serve as GeoJSON.

    Returns:
        A district-level frame with ``date``, ``district``, ``n_grid_cells``,
        ``mean_corrected_mm``, ``max_corrected_mm``, ``mean_raw_mm`` (if the raw
        forecast was present), ``max_prob_*`` per threshold and
        ``warning_level``.

    Raises:
        AggregationError: If districts cannot be assigned, or if required
            columns are missing.
    """
    names = list(threshold_names) if threshold_names is not None else list(RAIN_THRESHOLDS)
    df = grid_predictions_df

    needs_join = (
        sch.DISTRICT_COLUMN not in df.columns
        or df[sch.DISTRICT_COLUMN].isna().all()
        or return_geometry
    )
    if needs_join:
        if district_gdf is None:
            raise AggregationError(
                "Grid predictions have no usable 'district' column and no "
                "district_gdf was supplied. Either add a 'district' column to "
                "your input data or pass the district polygons from "
                "load_district_boundaries()."
            )
        df = assign_districts(df, district_gdf)

    if CORRECTED_COLUMN not in df.columns:
        raise AggregationError(
            f"Grid predictions are missing '{CORRECTED_COLUMN}'. Run the bias "
            f"correction stage before aggregating."
        )

    work = pd.DataFrame(df.drop(columns=["geometry"], errors="ignore"))
    work = work[work[sch.DISTRICT_COLUMN].notna()]
    if work.empty:
        raise AggregationError(
            "No grid cells fell inside any district polygon. Check that the "
            "grid coordinates and the shapefile use the same CRS and the same "
            "longitude convention (-180..180)."
        )

    group_keys = ["date", sch.DISTRICT_COLUMN] if "date" in work.columns else [sch.DISTRICT_COLUMN]

    agg: Dict[str, tuple] = {
        "n_grid_cells": (CORRECTED_COLUMN, "size"),
        "mean_corrected_mm": (CORRECTED_COLUMN, "mean"),
        "max_corrected_mm": (CORRECTED_COLUMN, "max"),
    }
    if sch.FORECAST_COLUMN in work.columns:
        agg["mean_raw_mm"] = (sch.FORECAST_COLUMN, "mean")
    if sch.OBSERVED_COLUMN in work.columns:
        agg["mean_observed_mm"] = (sch.OBSERVED_COLUMN, "mean")
    for name in names:
        col = probability_column(name)
        if col in work.columns:
            # Max, not mean: a warning must fire on the worst cell in the
            # district (see module docstring).
            agg[f"max_{col}"] = (col, "max")

    out = work.groupby(group_keys, as_index=False, observed=True).agg(**agg)

    prob_lookup = {name: f"max_{probability_column(name)}" for name in names}
    available = {n: c for n, c in prob_lookup.items() if c in out.columns}
    if available:
        out["warning_level"] = [
            classify_warning_level({n: row[c] for n, c in available.items()}, config)
            for _, row in out.iterrows()
        ]
    else:
        out["warning_level"] = WARNING_NONE

    if return_geometry:
        import geopandas as gpd

        polys = district_gdf.to_crs(GEO_CRS)[[sch.DISTRICT_COLUMN, "geometry"]]
        out = gpd.GeoDataFrame(
            out.merge(polys, on=sch.DISTRICT_COLUMN, how="left"),
            geometry="geometry",
            crs=GEO_CRS,
        )
    return out


def list_districts(district_gdf=None, grid_df: Optional[pd.DataFrame] = None) -> List[str]:
    """List the district names available from the shapefile or the data.

    Args:
        district_gdf: District polygons, or None.
        grid_df: A frame carrying a ``district`` column, or None.

    Returns:
        Sorted unique district names. Empty if neither source is available.
    """
    names: set[str] = set()
    if district_gdf is not None and sch.DISTRICT_COLUMN in district_gdf.columns:
        names |= {str(v) for v in district_gdf[sch.DISTRICT_COLUMN].dropna().unique()}
    if grid_df is not None and sch.DISTRICT_COLUMN in grid_df.columns:
        names |= {str(v) for v in grid_df[sch.DISTRICT_COLUMN].dropna().unique()}
    return sorted(names)


# ---------------------------------------------------------------------------
# Shapefile stub
# ---------------------------------------------------------------------------

DISTRICT_SHAPEFILE_INSTRUCTIONS = """\
>>> YOU NEED TO SUPPLY A DISTRICT SHAPEFILE <<<

No Indian district boundary file is bundled with this repository, and nothing
is downloaded at runtime. Obtain one yourself and place it where the pipeline
expects it (default: data_store/districts.geojson, override with the
RAINFALL_DISTRICTS_PATH environment variable).

Requirements:
  * Format: .shp (with .dbf/.shx/.prj siblings), .gpkg, or .geojson
  * Geometry: Polygon / MultiPolygon, one feature per district
  * Attribute: a district-name column. The loader looks for 'district' and
    falls back to DISTRICT / dtname / NAME_2 / distname; set
    RAINFALL_DISTRICT_NAME_FIELD if yours is named otherwise.
  * CRS: any; it is reprojected to EPSG:4326 on load.

Commonly used sources include Survey of India / data.gov.in administrative
boundaries and the GADM level-2 dataset. Check the licence terms of whichever
you choose before using it in a submission.

Until this file exists, district aggregation will raise a clear error and the
API's /districts endpoint returns an empty list.
"""
