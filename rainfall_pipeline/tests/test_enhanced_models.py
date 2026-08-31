"""Tests for enhanced features, losses, ensemble blending, and monotonic probability heads."""

import numpy as np
import pandas as pd
import pytest

from rainfall_pipeline.config.thresholds import (
    HEAVY_MM,
    REGIME_ACTIVE,
    REGIME_BREAK,
    REGIME_COASTAL,
    REGIME_DEPRESSION_LOW,
    REGIME_LABELS,
    REGIME_OROGRAPHIC,
    VERY_HEAVY_MM,
)
from rainfall_pipeline.data import schema as sch
from rainfall_pipeline.features.engineering import (
    DERIVED_COLUMNS,
    FEATURE_COLUMNS,
    add_derived_features,
    select_features,
)
from rainfall_pipeline.models.baselines import GlobalBiasCorrector, residual_target
from rainfall_pipeline.models.bias_correction import RegimeBiasCorrector
from rainfall_pipeline.models.heavy_rain_probability import (
    HeavyRainProbabilityModel,
    attach_corrected_forecast,
)
from rainfall_pipeline.models.regime_classifier import RegimeClassifier


@pytest.fixture
def sample_spatial_grid():
    """Create a synthetic 4x4 spatial grid on 2 dates."""
    dates = pd.date_range("2022-07-01", periods=2, freq="D")
    lats = [18.0, 18.25, 18.5, 18.75]
    lons = [72.0, 72.25, 72.5, 72.75]
    rows = []
    for d in dates:
        for lat in lats:
            for lon in lons:
                rows.append(
                    {
                        "date": d,
                        "lat": lat,
                        "lon": lon,
                        "pressure_msl": 1008.0 - (lat - 18.0),
                        "wind_u_850": 10.0,
                        "wind_v_850": 5.0,
                        "wind_u_200": -5.0,
                        "wind_v_200": 2.0,
                        "olr": 180.0,
                        "humidity": 85.0,
                        "cape": 1200.0,
                        "vorticity": 2.5e-5,
                        "elevation": 150.0 + (lon - 72.0) * 100,
                        "coastal_distance": (lon - 72.0) * 80,
                        "raw_forecast_mm": 45.0 + (lat - 18.0) * 20.0,
                        "observed_mm": 50.0 + (lat - 18.0) * 22.0,
                    }
                )
    return pd.DataFrame(rows)


def test_spatial_and_advective_features(sample_spatial_grid):
    df_feat = add_derived_features(sample_spatial_grid)
    for col in [
        "forecast_spatial_mean_3x3",
        "forecast_spatial_max_3x3",
        "forecast_spatial_std_3x3",
        "cape_spatial_max_3x3",
        "moisture_flux_spatial_mean_3x3",
        "upwind_forecast_rain",
        "upwind_moisture_flux",
        "ivt_proxy",
        "convective_instability",
    ]:
        assert col in df_feat.columns
        assert col in DERIVED_COLUMNS
        assert col in FEATURE_COLUMNS
        assert df_feat[col].notna().all()


def test_single_row_feature_fallback():
    single_row = pd.DataFrame(
        [
            {
                "date": "2022-07-01",
                "lat": 19.0,
                "lon": 73.0,
                "pressure_msl": 1005.0,
                "wind_u_850": 8.0,
                "wind_v_850": 2.0,
                "wind_u_200": -4.0,
                "wind_v_200": 1.0,
                "olr": 190.0,
                "humidity": 80.0,
                "cape": 1500.0,
                "vorticity": 1e-5,
                "elevation": 100.0,
                "coastal_distance": 20.0,
                "raw_forecast_mm": 25.0,
                "observed_mm": 30.0,
            }
        ]
    )
    df_feat = add_derived_features(single_row)
    assert df_feat["forecast_spatial_mean_3x3"].iloc[0] == 25.0
    assert df_feat["forecast_spatial_max_3x3"].iloc[0] == 25.0
    assert df_feat["forecast_spatial_std_3x3"].iloc[0] == 0.0
    assert df_feat["ivt_proxy"].iloc[0] > 0


def test_tweedie_and_ensemble_bias_corrector(sample_spatial_grid):
    df_feat = add_derived_features(sample_spatial_grid)
    
    # Test Tweedie loss
    tweedie_corrector = GlobalBiasCorrector(backend="xgboost", loss="tweedie")
    tweedie_corrector.fit(df_feat)
    pred_bias = tweedie_corrector.predict_bias(df_feat)
    assert len(pred_bias) == len(df_feat)
    assert pred_bias.notna().all()
    
    # Test Ensemble backend
    ensemble_corrector = GlobalBiasCorrector(backend="ensemble")
    ensemble_corrector.fit(df_feat)
    ens_bias = ensemble_corrector.predict_bias(df_feat)
    assert len(ens_bias) == len(df_feat)
    assert ens_bias.notna().all()


def test_regime_bias_corrector_with_ensemble(sample_spatial_grid):
    df_feat = add_derived_features(sample_spatial_grid)
    regimes = pd.Series([REGIME_COASTAL, REGIME_OROGRAPHIC] * (len(df_feat) // 2), index=df_feat.index)
    
    corrector = RegimeBiasCorrector(backend="ensemble", loss="huber", min_rows_per_regime=5)
    corrector.fit(df_feat, regimes)
    
    corrected = corrector.predict(df_feat, regimes)
    assert (corrected >= 0).all()
    assert len(corrected) == len(df_feat)


def test_regime_classifier_ensemble(sample_spatial_grid):
    df_feat = add_derived_features(sample_spatial_grid)
    labels = pd.Series([REGIME_ACTIVE, REGIME_BREAK] * (len(df_feat) // 2), index=df_feat.index)
    
    clf = RegimeClassifier(backend="ensemble")
    clf.fit(df_feat, labels)
    
    probs = clf.predict_proba(df_feat)
    assert (probs.sum(axis=1).round(3) == 1.0).all()
    preds = clf.predict(df_feat)
    assert set(preds.unique()).issubset(set(REGIME_LABELS))


def test_monotonic_heavy_rain_probability(sample_spatial_grid):
    df_feat = add_derived_features(sample_spatial_grid)
    corrected = df_feat["raw_forecast_mm"] + 5.0
    prob_df = attach_corrected_forecast(df_feat, corrected)
    
    model = HeavyRainProbabilityModel(backend="xgboost")
    model.fit(prob_df)
    
    probs = model.predict_proba(prob_df, enforce_monotonicity=True)
    # Check that heavy >= very_heavy >= extremely_heavy for every row
    assert (probs["heavy"] >= probs["very_heavy"] - 1e-6).all()
    assert (probs["very_heavy"] >= probs["extremely_heavy"] - 1e-6).all()
