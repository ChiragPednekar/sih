"""The fixed data contract every stage of the pipeline codes against.

One row per ``(date, lat, lon)``. Each loader in :mod:`rainfall_pipeline.data.loaders`
returns a subset of these columns; :func:`rainfall_pipeline.features.engineering.build_feature_table`
joins them into the full schema.

Keeping the contract in one module means that when the real data arrives, the
only thing that has to change is the *reading* code in ``loaders.py`` -- nothing
downstream needs to know where the numbers came from.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

#: Keys that uniquely identify a row.
KEY_COLUMNS: List[str] = ["date", "lat", "lon"]

#: Atmospheric predictors (ERA5 or equivalent reanalysis/analysis fields).
ATMOSPHERIC_COLUMNS: List[str] = [
    "pressure_msl",
    "wind_u_850",
    "wind_v_850",
    "wind_u_200",
    "wind_v_200",
    "olr",
    "humidity",
    "cape",
    "vorticity",
]

#: Static/geographic predictors. Constant in time for a given (lat, lon).
STATIC_COLUMNS: List[str] = ["elevation", "coastal_distance"]

#: The forecast being corrected.
FORECAST_COLUMN = "raw_forecast_mm"

#: Ground truth.
OBSERVED_COLUMN = "observed_mm"

#: District identifier attached to each grid cell.
DISTRICT_COLUMN = "district"

#: The full common schema, in canonical order.
COMMON_SCHEMA: List[str] = (
    KEY_COLUMNS
    + [DISTRICT_COLUMN]
    + ATMOSPHERIC_COLUMNS
    + STATIC_COLUMNS
    + [FORECAST_COLUMN, OBSERVED_COLUMN]
)

#: Expected pandas dtype for each column. ``date`` is handled separately because
#: it is parsed to datetime64 rather than cast.
COLUMN_DTYPES: Dict[str, str] = {
    "lat": "float64",
    "lon": "float64",
    DISTRICT_COLUMN: "string",
    **{c: "float64" for c in ATMOSPHERIC_COLUMNS},
    **{c: "float64" for c in STATIC_COLUMNS},
    FORECAST_COLUMN: "float64",
    OBSERVED_COLUMN: "float64",
}

#: Human-readable units/description for each column. Surfaced in loader error
#: messages and in ``data/README.md`` so there is a single source of truth.
COLUMN_DESCRIPTIONS: Dict[str, str] = {
    "date": "Valid date of the forecast/observation. Any pandas-parseable date (YYYY-MM-DD preferred).",
    "lat": "Latitude of the grid cell centre, decimal degrees north.",
    "lon": "Longitude of the grid cell centre, decimal degrees east.",
    DISTRICT_COLUMN: "District name the grid cell falls in. May be left blank if you let the aggregation stage assign it from the shapefile.",
    "pressure_msl": "Mean sea level pressure, hPa.",
    "wind_u_850": "Zonal (east-west) wind at 850 hPa, m/s.",
    "wind_v_850": "Meridional (north-south) wind at 850 hPa, m/s.",
    "wind_u_200": "Zonal wind at 200 hPa, m/s.",
    "wind_v_200": "Meridional wind at 200 hPa, m/s.",
    "olr": "Outgoing longwave radiation, W/m^2.",
    "humidity": "Relative humidity, percent (0-100). Use a single representative level, e.g. 700 hPa.",
    "cape": "Convective available potential energy, J/kg.",
    "vorticity": "Relative vorticity at 850 hPa, 1/s (values are typically order 1e-5).",
    "elevation": "Terrain elevation of the grid cell, metres above sea level.",
    "coastal_distance": "Great-circle distance from the grid cell to the nearest coastline, km. 0 on the coast.",
    FORECAST_COLUMN: "Raw NWP (or AI model) 24 h accumulated rainfall forecast, mm.",
    OBSERVED_COLUMN: "Observed 24 h accumulated rainfall, mm. This is the ground truth.",
}


# ---------------------------------------------------------------------------
# Per-loader required columns
# ---------------------------------------------------------------------------

ERA5_REQUIRED: List[str] = KEY_COLUMNS + ATMOSPHERIC_COLUMNS
OBSERVED_REQUIRED: List[str] = KEY_COLUMNS + [OBSERVED_COLUMN]
NWP_REQUIRED: List[str] = KEY_COLUMNS + [FORECAST_COLUMN]

#: Static fields are optional in the ERA5 file -- if they are absent there, they
#: must be present in one of the other inputs, or supplied separately.
ERA5_OPTIONAL: List[str] = STATIC_COLUMNS + [DISTRICT_COLUMN]
OBSERVED_OPTIONAL: List[str] = [DISTRICT_COLUMN]
NWP_OPTIONAL: List[str] = ["lead_time"]


class SchemaError(ValueError):
    """Raised when a loaded file does not satisfy the data contract."""


def describe_columns(columns: Sequence[str]) -> str:
    """Render a human-readable ``name -- description`` block for ``columns``.

    Args:
        columns: Column names to describe.

    Returns:
        A newline-joined block suitable for embedding in an error message.
    """
    lines = []
    for col in columns:
        desc = COLUMN_DESCRIPTIONS.get(col, "(no description available)")
        lines.append(f"  - {col:<18} {desc}")
    return "\n".join(lines)


def validate_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    *,
    source: str,
    path: str,
) -> None:
    """Check that ``df`` contains every column in ``required``.

    Args:
        df: The DataFrame that was read from disk.
        required: Columns that must be present.
        source: Friendly name of the data source, used in the error message.
        path: Path the data was read from, used in the error message.

    Raises:
        SchemaError: If any required column is missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{source} file at '{path}' is missing required column(s): "
            f"{', '.join(missing)}.\n"
            f"The file must contain, at minimum:\n{describe_columns(required)}\n"
            f"Found columns: {', '.join(map(str, df.columns))}"
        )


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes of any schema columns present in ``df``.

    ``date`` is parsed to ``datetime64[ns]`` and normalised to midnight so that
    joins across sources line up. Unknown columns are left untouched.

    Args:
        df: DataFrame to normalise (not modified in place).

    Returns:
        A new DataFrame with normalised dtypes.
    """
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    for col, dtype in COLUMN_DTYPES.items():
        if col in out.columns:
            try:
                out[col] = out[col].astype(dtype)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise SchemaError(
                    f"Column '{col}' could not be cast to {dtype}: {exc}"
                ) from exc
    return out
