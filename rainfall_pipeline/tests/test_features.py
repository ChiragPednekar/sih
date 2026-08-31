"""Tests for Stage 1 -- feature engineering.

The leakage tests here are the important ones: if ``observed_mm`` or a quantity
derived from it ever reaches the model feature matrix, every downstream number
becomes fiction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rainfall_pipeline.data import schema as sch
from rainfall_pipeline.features.engineering import (
    DERIVED_COLUMNS,
    FEATURE_COLUMNS,
    REGIME_FEATURE_COLUMNS,
    FeatureError,
    add_derived_features,
    attach_anomalies,
    build_feature_table,
    fit_climatology,
    join_sources,
    select_features,
)


def test_derived_columns_are_all_added(dummy_features: pd.DataFrame) -> None:
    """Every column the module advertises must actually appear."""
    for col in DERIVED_COLUMNS:
        assert col in dummy_features.columns, col


def test_feature_matrix_excludes_the_target(dummy_features: pd.DataFrame) -> None:
    """The observation must never be a model input."""
    assert sch.OBSERVED_COLUMN not in FEATURE_COLUMNS
    assert sch.OBSERVED_COLUMN not in REGIME_FEATURE_COLUMNS
    X = select_features(dummy_features)
    assert sch.OBSERVED_COLUMN not in X.columns


def test_feature_matrix_excludes_the_observation_derived_anomaly() -> None:
    """``rain_anomaly_sd`` is computed from the observation, so it must not leak."""
    assert "rain_anomaly_sd" not in FEATURE_COLUMNS
    assert "rain_anomaly_sd" not in REGIME_FEATURE_COLUMNS


def test_select_features_is_numeric_and_ordered(dummy_features: pd.DataFrame) -> None:
    """The model matrix must be float-typed and in the declared column order."""
    X = select_features(dummy_features)
    assert list(X.columns) == FEATURE_COLUMNS
    assert (X.dtypes == "float64").all()


def test_select_features_tolerates_a_missing_predictor(dummy_features: pd.DataFrame) -> None:
    """A predictor absent at prediction time becomes NaN, not a crash."""
    without_cape = dummy_features.drop(columns=["cape"])
    X = select_features(without_cape)
    assert X["cape"].isna().all()
    assert list(X.columns) == FEATURE_COLUMNS


def test_wind_shear_is_the_vector_difference(dummy_features: pd.DataFrame) -> None:
    """Shear must be the 200-850 hPa vector difference magnitude."""
    row = dummy_features.iloc[0]
    expected = float(
        np.hypot(row["wind_u_200"] - row["wind_u_850"], row["wind_v_200"] - row["wind_v_850"])
    )
    assert row["wind_shear"] == pytest.approx(expected)


def test_calendar_features_round_trip(dummy_features: pd.DataFrame) -> None:
    """Day-of-year and month must match the date column."""
    dates = pd.to_datetime(dummy_features["date"])
    assert (dummy_features["day_of_year"] == dates.dt.dayofyear).all()
    assert (dummy_features["month"] == dates.dt.month).all()
    # The sine/cosine encoding must lie on the unit circle.
    radius = np.hypot(dummy_features["doy_sin"], dummy_features["doy_cos"])
    assert np.allclose(radius, 1.0)


def test_missing_required_column_raises(dummy_raw: pd.DataFrame) -> None:
    """Deriving features without the atmospheric inputs must fail loudly."""
    with pytest.raises(FeatureError, match="missing column"):
        add_derived_features(dummy_raw.drop(columns=["cape", "olr"]))


def test_anomalies_are_nan_without_a_climatology(dummy_raw: pd.DataFrame) -> None:
    """No climatology means no anomaly -- never a silently-zero anomaly."""
    features = add_derived_features(dummy_raw, climatology=None)
    assert features["rain_anomaly_sd"].isna().all()
    assert features["pressure_anomaly"].isna().all()


def test_climatology_std_never_divides_by_zero(dummy_raw: pd.DataFrame) -> None:
    """A single-sample group must not produce an infinite anomaly."""
    clim = fit_climatology(dummy_raw)
    assert (clim["rain_clim_std"] > 0).all()
    features = attach_anomalies(add_derived_features(dummy_raw), clim)
    assert np.isfinite(features["rain_anomaly_sd"]).all()


def test_join_sources_produces_the_full_schema(source_files) -> None:
    """Joining the three sources must reconstruct the common schema."""
    from rainfall_pipeline.data.loaders import (
        load_era5,
        load_observed_rainfall,
        load_raw_nwp_forecast,
    )

    table = build_feature_table(
        era5=load_era5(source_files["era5"]),
        observed=load_observed_rainfall(source_files["observed"]),
        nwp=load_raw_nwp_forecast(source_files["nwp"]),
    )
    for col in sch.COMMON_SCHEMA:
        assert col in table.columns, col
    assert len(table) == 8


def test_join_on_mismatched_grids_raises_with_guidance(dummy_raw: pd.DataFrame) -> None:
    """A grid mismatch is the most likely real-world failure; it must explain itself."""
    era5 = dummy_raw[sch.KEY_COLUMNS + sch.ATMOSPHERIC_COLUMNS]
    observed = dummy_raw[sch.KEY_COLUMNS + [sch.OBSERVED_COLUMN]].copy()
    nwp = dummy_raw[sch.KEY_COLUMNS + [sch.FORECAST_COLUMN]].copy()
    # Shift the forecast grid so nothing lines up.
    nwp["lat"] = nwp["lat"] + 0.13
    with pytest.raises(FeatureError, match="zero rows"):
        join_sources(era5, observed, nwp)


def test_lead_time_defaults_to_one_day(dummy_raw: pd.DataFrame) -> None:
    """When the forecast file carries no lead time, day-1 is assumed."""
    features = add_derived_features(dummy_raw)
    assert (features["lead_time"] == 1.0).all()


def test_lead_time_is_carried_through_when_supplied(dummy_raw: pd.DataFrame) -> None:
    """An explicit lead time in the input must survive feature engineering."""
    with_lead = dummy_raw.copy()
    with_lead["lead_time"] = 3.0
    features = add_derived_features(with_lead)
    assert (features["lead_time"] == 3.0).all()
