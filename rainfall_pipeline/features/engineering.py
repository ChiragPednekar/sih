"""Stage 1 -- feature engineering.

Joins the three tabular sources on ``(date, lat, lon)`` and derives the
predictors every downstream model consumes. The output of
:func:`build_feature_table` is the single table that the regime engine, the
bias-correction models, the probability head and the verification module all
read from.

Design rule: features derived here must be computable from a *single row*
wherever possible, so that the API can build a feature vector for one grid cell
at prediction time without needing the whole training table. The two exceptions
(``rain_anomaly_sd`` and ``pressure_anomaly``, which need a climatology) are
computed against a climatology table that is fitted on training data and saved
alongside the models.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ..data import schema as sch

#: Derived columns added by :func:`add_derived_features`.
DERIVED_COLUMNS: List[str] = [
    "day_of_year",
    "month",
    "doy_sin",
    "doy_cos",
    "lead_time",
    "wind_speed_850",
    "wind_speed_200",
    "wind_dir_850",
    "shear_u",
    "shear_v",
    "wind_shear",
    "moisture_flux_850",
    "ivt_proxy",
    "convective_instability",
    "onshore_flow",
    "upslope_flow",
    "cape_humidity",
    "log_raw_forecast",
    "is_coastal",
    "is_orographic",
    "forecast_spatial_mean_3x3",
    "forecast_spatial_max_3x3",
    "forecast_spatial_std_3x3",
    "cape_spatial_max_3x3",
    "moisture_flux_spatial_mean_3x3",
    "upwind_forecast_rain",
    "upwind_moisture_flux",
]

#: Columns fed to the models. Excludes the keys, the district label, the target
#: and anything leaky. ``observed_mm`` must never appear here, and neither may
#: ``rain_anomaly_sd`` -- on historical rows it is derived from the observation,
#: so feeding it to a corrector would leak the answer. It exists purely as an
#: input to the rule-based regime labeller.
FEATURE_COLUMNS: List[str] = (
    ["lat", "lon"]
    + sch.ATMOSPHERIC_COLUMNS
    + sch.STATIC_COLUMNS
    + [sch.FORECAST_COLUMN]
    + DERIVED_COLUMNS
)

#: Columns the regime classifier is allowed to see. The raw forecast is
#: included because "how much rain the model thinks is coming" is genuinely
#: informative about the synoptic situation; the *observation* is not, since it
#: would not exist at prediction time.
REGIME_FEATURE_COLUMNS: List[str] = [
    c for c in FEATURE_COLUMNS if c not in {"log_raw_forecast"}
]


class FeatureError(ValueError):
    """Raised when a feature table cannot be built from the given inputs."""


def join_sources(
    era5: pd.DataFrame,
    observed: pd.DataFrame,
    nwp: pd.DataFrame,
    *,
    how: str = "inner",
) -> pd.DataFrame:
    """Join the three tabular sources on ``(date, lat, lon)``.

    Args:
        era5: Output of :func:`~rainfall_pipeline.data.loaders.load_era5`.
        observed: Output of
            :func:`~rainfall_pipeline.data.loaders.load_observed_rainfall`.
        nwp: Output of
            :func:`~rainfall_pipeline.data.loaders.load_raw_nwp_forecast`.
        how: Join strategy. ``"inner"`` (the default) keeps only cells present
            in all three sources, which is what training needs.

    Returns:
        The joined frame.

    Raises:
        FeatureError: If the join produces no rows, which almost always means
            the three sources are on different grids or time conventions.
    """
    keys = sch.KEY_COLUMNS
    merged = era5.merge(nwp, on=keys, how=how, suffixes=("", "_nwp"))
    merged = merged.merge(observed, on=keys, how=how, suffixes=("", "_obs"))

    # Prefer the district label from whichever source carried one.
    district_cols = [c for c in merged.columns if c.startswith(sch.DISTRICT_COLUMN)]
    if district_cols:
        merged[sch.DISTRICT_COLUMN] = merged[district_cols].bfill(axis=1).iloc[:, 0]
        drop = [c for c in district_cols if c != sch.DISTRICT_COLUMN]
        merged = merged.drop(columns=drop)

    if merged.empty:
        raise FeatureError(
            "Joining ERA5, observed rainfall and the raw forecast on "
            "(date, lat, lon) produced zero rows. The three sources must share "
            "the same grid and the same daily time convention. Check that:\n"
            "  - latitudes/longitudes are rounded consistently (e.g. all to "
            "0.25 deg), since floating-point mismatch breaks the join;\n"
            "  - all longitudes use the same -180..180 convention;\n"
            "  - the observation window and the forecast valid time line up "
            "(IMD daily rainfall is 0830 IST to 0830 IST)."
        )
    return merged


def add_derived_features(
    df: pd.DataFrame,
    *,
    climatology: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add calendar, wind, moisture and terrain-interaction features.

    Args:
        df: Frame containing at least the atmospheric and static schema
            columns plus ``date`` and ``raw_forecast_mm``.
        climatology: Optional output of :func:`fit_climatology`. When supplied,
            ``rain_anomaly_sd`` and ``pressure_anomaly`` are added -- the regime
            rules need them. When omitted those columns are filled with NaN so
            the frame keeps a stable shape.

    Returns:
        A new frame with :data:`DERIVED_COLUMNS` (and the anomaly columns) added.

    Raises:
        FeatureError: If a required input column is missing.
    """
    required = set(sch.ATMOSPHERIC_COLUMNS) | {"date", sch.FORECAST_COLUMN}
    missing = sorted(required - set(df.columns))
    if missing:
        raise FeatureError(
            f"Cannot derive features: missing column(s) {missing}. "
            f"Present: {sorted(df.columns)}"
        )

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    # --- calendar -------------------------------------------------------
    out["day_of_year"] = out["date"].dt.dayofyear.astype("float64")
    out["month"] = out["date"].dt.month.astype("float64")
    # Sine/cosine encoding so day 365 and day 1 are adjacent to the model.
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)

    # Forecast lead time in days. Carried through if the NWP file supplies it,
    # otherwise 1 (a day-1 forecast), which is the common single-lead setup.
    if "lead_time" not in out.columns:
        out["lead_time"] = 1.0
    out["lead_time"] = pd.to_numeric(out["lead_time"], errors="coerce").fillna(1.0)

    # --- wind -----------------------------------------------------------
    u850, v850 = out["wind_u_850"], out["wind_v_850"]
    u200, v200 = out["wind_u_200"], out["wind_v_200"]
    out["wind_speed_850"] = np.hypot(u850, v850)
    out["wind_speed_200"] = np.hypot(u200, v200)
    # Meteorological convention: direction the wind blows *from*, degrees.
    out["wind_dir_850"] = (np.degrees(np.arctan2(-u850, -v850)) + 360.0) % 360.0
    out["shear_u"] = u200 - u850
    out["shear_v"] = v200 - v850
    out["wind_shear"] = np.hypot(out["shear_u"], out["shear_v"])

    # --- moisture & instability -----------------------------------------
    # Low-level moisture flux: humidity carried by the 850 hPa jet.
    out["moisture_flux_850"] = out["humidity"] * out["wind_speed_850"]
    # Integrated vapor transport (IVT) proxy.
    out["ivt_proxy"] = out["moisture_flux_850"] * 0.85
    # Instability x moisture: the combination that actually produces heavy rain.
    out["cape_humidity"] = out["cape"] * out["humidity"] / 100.0
    pressure = out["pressure_msl"] if "pressure_msl" in out.columns else pd.Series(1013.25, index=out.index)
    out["convective_instability"] = out["cape_humidity"] / (pressure.clip(lower=800.0) / 1000.0)

    # --- terrain interactions -------------------------------------------
    if "coastal_distance" in out.columns:
        # Westerly flow at the coast is onshore along most of the west coast.
        out["is_coastal"] = (out["coastal_distance"] <= 75.0).astype("float64")
        # Decays with distance inland, so the feature is smooth rather than a step.
        out["onshore_flow"] = u850 * np.exp(-out["coastal_distance"] / 100.0)
    else:
        out["is_coastal"] = np.nan
        out["onshore_flow"] = np.nan

    if "elevation" in out.columns:
        out["is_orographic"] = (out["elevation"] >= 600.0).astype("float64")
        # Wind speed scaled by terrain height: a stand-in for the true upslope
        # component, which would need the terrain gradient vector.
        out["upslope_flow"] = out["wind_speed_850"] * out["elevation"] / 1000.0
    else:
        out["is_orographic"] = np.nan
        out["upslope_flow"] = np.nan

    # --- spatial neighborhood & advection -------------------------------
    out = _compute_spatial_and_advective_features(out)

    # --- forecast transform ---------------------------------------------
    # Rainfall is strongly right-skewed; log1p gives the trees a better-behaved
    # split space while keeping the raw value available too.
    out["log_raw_forecast"] = np.log1p(out[sch.FORECAST_COLUMN].clip(lower=0))

    # --- anomalies (need a climatology) ---------------------------------
    out = attach_anomalies(out, climatology)
    return out


def _compute_spatial_and_advective_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 3x3 spatial neighborhood pooling and upwind advective features.

    Operates per date when multiple grid points exist; gracefully defaults
    to pointwise values when coordinates are absent or single-point.
    """
    out = df.copy()
    n_rows = len(out)

    f_mean = pd.to_numeric(out[sch.FORECAST_COLUMN], errors="coerce").fillna(0.0).values.copy()
    f_max = f_mean.copy()
    f_std = np.zeros(n_rows, dtype="float64")
    cape_max = (
        pd.to_numeric(out["cape"], errors="coerce").fillna(0.0).values.copy()
        if "cape" in out.columns
        else np.zeros(n_rows)
    )
    mflux_mean = (
        pd.to_numeric(out["moisture_flux_850"], errors="coerce").fillna(0.0).values.copy()
        if "moisture_flux_850" in out.columns
        else np.zeros(n_rows)
    )
    upwind_f = f_mean.copy()
    upwind_mf = mflux_mean.copy()

    if "lat" in out.columns and "lon" in out.columns:
        try:
            from scipy.spatial import cKDTree

            date_col = out["date"] if "date" in out.columns else pd.Series(0, index=out.index)
            for _, group in out.groupby(date_col):
                idx = group.index.values
                if len(idx) < 2:
                    continue
                coords = np.column_stack(
                    [
                        pd.to_numeric(group["lat"], errors="coerce").fillna(0.0).values,
                        pd.to_numeric(group["lon"], errors="coerce").fillna(0.0).values,
                    ]
                )
                if np.all(coords == coords[0]):
                    continue

                tree = cKDTree(coords)
                neighbors_list = tree.query_ball_tree(tree, r=0.6)

                raw_f = pd.to_numeric(group[sch.FORECAST_COLUMN], errors="coerce").fillna(0.0).values
                cape_vals = (
                    pd.to_numeric(group["cape"], errors="coerce").fillna(0.0).values
                    if "cape" in group.columns
                    else np.zeros(len(idx))
                )
                mf_vals = (
                    pd.to_numeric(group["moisture_flux_850"], errors="coerce").fillna(0.0).values
                    if "moisture_flux_850" in group.columns
                    else np.zeros(len(idx))
                )

                group_f_mean = np.array([raw_f[nbrs].mean() if len(nbrs) > 0 else raw_f[i] for i, nbrs in enumerate(neighbors_list)])
                group_f_max = np.array([raw_f[nbrs].max() if len(nbrs) > 0 else raw_f[i] for i, nbrs in enumerate(neighbors_list)])
                group_f_std = np.array([raw_f[nbrs].std() if len(nbrs) > 1 else 0.0 for nbrs in neighbors_list])
                group_cape_max = np.array([cape_vals[nbrs].max() if len(nbrs) > 0 else cape_vals[i] for i, nbrs in enumerate(neighbors_list)])
                group_mf_mean = np.array([mf_vals[nbrs].mean() if len(nbrs) > 0 else mf_vals[i] for i, nbrs in enumerate(neighbors_list)])

                u = (
                    pd.to_numeric(group["wind_u_850"], errors="coerce").fillna(0.0).values
                    if "wind_u_850" in group.columns
                    else np.zeros(len(idx))
                )
                v = (
                    pd.to_numeric(group["wind_v_850"], errors="coerce").fillna(0.0).values
                    if "wind_v_850" in group.columns
                    else np.zeros(len(idx))
                )
                w_speed = np.hypot(u, v) + 1e-6
                upwind_coords = coords + np.column_stack([-v / w_speed * 0.4, -u / w_speed * 0.4])
                _, upwind_indices = tree.query(upwind_coords, k=1)

                group_upwind_f = raw_f[upwind_indices]
                group_upwind_mf = mf_vals[upwind_indices]

                # Map back to array positions
                pos = out.index.get_indexer(idx)
                f_mean[pos] = group_f_mean
                f_max[pos] = group_f_max
                f_std[pos] = group_f_std
                cape_max[pos] = group_cape_max
                mflux_mean[pos] = group_mf_mean
                upwind_f[pos] = group_upwind_f
                upwind_mf[pos] = group_upwind_mf
        except Exception:
            pass

    out["forecast_spatial_mean_3x3"] = f_mean
    out["forecast_spatial_max_3x3"] = f_max
    out["forecast_spatial_std_3x3"] = f_std
    out["cape_spatial_max_3x3"] = cape_max
    out["moisture_flux_spatial_mean_3x3"] = mflux_mean
    out["upwind_forecast_rain"] = upwind_f
    out["upwind_moisture_flux"] = upwind_mf
    return out


def fit_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a per-(lat, lon, month) climatology of rainfall and pressure.

    Fit this on the **training split only** and save it with the models, so the
    anomalies used at prediction time never peek at the test period.

    Args:
        df: Training frame with ``lat``, ``lon``, ``date``, ``observed_mm`` and
            ``pressure_msl``.

    Returns:
        A frame keyed on ``lat, lon, month`` with columns ``rain_clim_mean``,
        ``rain_clim_std`` and ``pressure_clim_mean``.

    Raises:
        FeatureError: If a required column is missing.
    """
    required = {"lat", "lon", "date", sch.OBSERVED_COLUMN, "pressure_msl"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise FeatureError(f"Cannot fit climatology: missing column(s) {missing}.")

    work = df.copy()
    work["month"] = pd.to_datetime(work["date"]).dt.month
    clim = (
        work.groupby(["lat", "lon", "month"], as_index=False)
        .agg(
            rain_clim_mean=(sch.OBSERVED_COLUMN, "mean"),
            rain_clim_std=(sch.OBSERVED_COLUMN, "std"),
            pressure_clim_mean=("pressure_msl", "mean"),
        )
    )
    # A single sample per group gives std = NaN; a zero std would divide by zero.
    # Fall back to the domain-wide std so the anomaly stays finite and comparable.
    global_std = float(work[sch.OBSERVED_COLUMN].std(ddof=0))
    fallback = global_std if np.isfinite(global_std) and global_std > 0 else 1.0
    clim["rain_clim_std"] = clim["rain_clim_std"].fillna(fallback).replace(0.0, fallback)
    return clim


def attach_anomalies(
    df: pd.DataFrame, climatology: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Join a climatology onto ``df`` and compute the anomaly columns.

    Args:
        df: Frame with ``lat``, ``lon``, ``month`` and ``pressure_msl``.
        climatology: Output of :func:`fit_climatology`, or None.

    Returns:
        A new frame with ``rain_anomaly_sd`` and ``pressure_anomaly``. Both are
        NaN when no climatology is supplied.
    """
    out = df.copy()
    if "month" not in out.columns:
        out["month"] = pd.to_datetime(out["date"]).dt.month.astype("float64")

    if climatology is None or climatology.empty:
        out["rain_anomaly_sd"] = np.nan
        out["pressure_anomaly"] = np.nan
        return out

    clim = climatology.copy()
    clim["month"] = clim["month"].astype("float64")
    out = out.merge(clim, on=["lat", "lon", "month"], how="left")

    # The rainfall anomaly is defined on the *observation* when it exists (the
    # rule-based labeller runs on historical data) and on the raw forecast
    # otherwise, so the same feature is available at prediction time.
    basis = (
        out[sch.OBSERVED_COLUMN]
        if sch.OBSERVED_COLUMN in out.columns
        else out[sch.FORECAST_COLUMN]
    )
    out["rain_anomaly_sd"] = (basis - out["rain_clim_mean"]) / out["rain_clim_std"]
    out["pressure_anomaly"] = out["pressure_msl"] - out["pressure_clim_mean"]
    return out.drop(columns=["rain_clim_mean", "rain_clim_std", "pressure_clim_mean"])


def build_feature_table(
    *,
    era5: pd.DataFrame,
    observed: pd.DataFrame,
    nwp: pd.DataFrame,
    climatology: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Join every source and derive all features in one call.

    Args:
        era5: Atmospheric predictors.
        observed: Ground-truth rainfall.
        nwp: Raw forecast.
        climatology: Optional climatology for the anomaly features.

    Returns:
        The analysis-ready table: schema columns plus :data:`DERIVED_COLUMNS`.

    Raises:
        FeatureError: If the join is empty or a required column is missing.
    """
    joined = join_sources(era5, observed, nwp)
    return add_derived_features(joined, climatology=climatology)


def select_features(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Return the model input matrix, in a stable column order.

    Args:
        df: A feature table.
        columns: Explicit column list. Defaults to :data:`FEATURE_COLUMNS`.

    Returns:
        A numeric-only frame containing exactly ``columns``. Columns absent from
        ``df`` are added as NaN, which the gradient-boosted models handle
        natively -- this keeps a model usable when one predictor is unavailable
        at prediction time.
    """
    cols = FEATURE_COLUMNS if columns is None else columns
    out = pd.DataFrame(index=df.index)
    for col in cols:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    return out.astype("float64")
