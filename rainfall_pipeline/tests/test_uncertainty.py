"""Tests for the prediction-interval model (Stage 3b).

A range is only worth showing if it behaves like a range: ordered bounds, a
band that actually contains the point forecast most of the time, and an honest
account of how often it contains the observation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rainfall_pipeline.models.baselines import NotFittedError
from rainfall_pipeline.models.uncertainty import (
    PredictionIntervalModel,
    interval_coverage,
)

from .conftest import FAST_PARAMS

#: LightGBM needs its leaf constraints relaxed to fit anything on 8 rows.
TINY = {**FAST_PARAMS, "min_child_samples": 1, "num_leaves": 3, "verbose": -1}


@pytest.fixture
def fitted(dummy_features, dummy_regimes) -> PredictionIntervalModel:
    """An interval model fitted on the dummy rows."""
    return PredictionIntervalModel(params=TINY, min_rows_per_regime=1).fit(
        dummy_features, dummy_regimes
    )


def test_bounds_are_ordered(fitted, dummy_features, dummy_regimes) -> None:
    """The low bound must never exceed the high bound.

    The two quantiles are fitted independently and can cross where data is
    thin; an inverted interval would render as a negative-width range.
    """
    band = fitted.predict_interval(dummy_features, dummy_regimes)
    assert (band["corrected_low"] <= band["corrected_high"]).all()


def test_bounds_are_never_negative(fitted, dummy_features, dummy_regimes) -> None:
    """Negative rainfall is unphysical at both ends of the band."""
    band = fitted.predict_interval(dummy_features, dummy_regimes)
    assert (band["corrected_low"] >= 0).all()
    assert (band["corrected_high"] >= 0).all()


def test_nominal_coverage_matches_the_quantiles() -> None:
    """The advertised width must follow from the quantiles actually fitted."""
    assert PredictionIntervalModel(quantiles=(0.1, 0.9)).nominal_coverage == pytest.approx(0.8)
    assert PredictionIntervalModel(quantiles=(0.25, 0.75)).nominal_coverage == pytest.approx(0.5)


def test_invalid_quantiles_are_rejected() -> None:
    """A reversed or out-of-range pair must raise, not silently invert."""
    for bad in [(0.9, 0.1), (0.0, 0.9), (0.1, 1.0), (0.5, 0.5)]:
        with pytest.raises(ValueError, match="quantiles must satisfy"):
            PredictionIntervalModel(quantiles=bad)


def test_predicting_before_fitting_raises(dummy_features, dummy_regimes) -> None:
    """An unfitted model must refuse rather than return zeros."""
    model = PredictionIntervalModel()
    with pytest.raises(NotFittedError, match="not fitted"):
        model.predict_interval(dummy_features, dummy_regimes)


def test_thin_regimes_fall_back(dummy_features, dummy_regimes) -> None:
    """A regime with too little data must reuse the global pair, not fail."""
    model = PredictionIntervalModel(params=TINY, min_rows_per_regime=10**6).fit(
        dummy_features, dummy_regimes
    )
    assert not model.models, "this test needs every regime to be too thin"
    band = model.predict_interval(dummy_features, dummy_regimes)
    assert band["corrected_low"].notna().all()


def test_roundtrip_preserves_predictions(fitted, dummy_features, dummy_regimes, tmp_path) -> None:
    """A saved and reloaded model must predict identically."""
    path = fitted.save(tmp_path / "intervals.joblib")
    reloaded = PredictionIntervalModel.load(path)

    pd.testing.assert_frame_equal(
        fitted.predict_interval(dummy_features, dummy_regimes),
        reloaded.predict_interval(dummy_features, dummy_regimes),
    )
    assert reloaded.quantiles == fitted.quantiles


def test_loading_a_missing_artifact_raises(tmp_path) -> None:
    """A missing file must name the path it looked for."""
    with pytest.raises(FileNotFoundError, match="prediction-interval artifact"):
        PredictionIntervalModel.load(tmp_path / "absent.joblib")


# ---------------------------------------------------------------------------
# Coverage measurement
# ---------------------------------------------------------------------------

def test_coverage_counts_observations_inside_the_band() -> None:
    """Coverage must be the plain fraction of observations inside the band."""
    observed = pd.Series([5.0, 15.0, 25.0, 35.0])
    low = pd.Series([0.0, 10.0, 30.0, 30.0])
    high = pd.Series([10.0, 20.0, 40.0, 32.0])
    # Inside, inside, outside (25 < 30), outside (35 > 32).
    result = interval_coverage(observed, low, high)
    assert result["coverage"] == pytest.approx(0.5)
    assert result["n"] == 4.0


def test_coverage_ignores_rows_it_cannot_judge() -> None:
    """Rows with no observation must not be counted as misses."""
    observed = pd.Series([5.0, np.nan, 25.0])
    low = pd.Series([0.0, 0.0, 20.0])
    high = pd.Series([10.0, 10.0, 30.0])
    result = interval_coverage(observed, low, high)
    assert result["n"] == 2.0
    assert result["coverage"] == pytest.approx(1.0)


def test_coverage_is_nan_when_nothing_is_judgeable() -> None:
    """No usable row must give NaN, never a flattering 0 or 1."""
    result = interval_coverage(
        pd.Series([np.nan, np.nan]), pd.Series([0.0, 0.0]), pd.Series([1.0, 1.0])
    )
    assert np.isnan(result["coverage"])
    assert result["n"] == 0.0


def test_coverage_reports_band_width() -> None:
    """Width belongs beside coverage: a band can be honest by being useless."""
    result = interval_coverage(
        pd.Series([5.0, 5.0]), pd.Series([0.0, 0.0]), pd.Series([10.0, 20.0])
    )
    assert result["coverage"] == pytest.approx(1.0)
    assert result["mean_width_mm"] == pytest.approx(15.0)


def test_a_single_row_regime_falls_back_rather_than_crashing(
    dummy_features, dummy_regimes
) -> None:
    """A regime with one training row must route to the fallback.

    The underlying estimator cannot fit a single sample, so without a floor a
    thin regime turns a configuration choice into a crash at training time.
    """
    model = PredictionIntervalModel(params=TINY, min_rows_per_regime=1).fit(
        dummy_features, dummy_regimes
    )
    for regime, count in model.training_counts.items():
        if count < 2:
            assert regime not in model.models, (
                f"regime '{regime}' has {count} row(s) but got its own model"
            )
    band = model.predict_interval(dummy_features, dummy_regimes)
    assert band["corrected_low"].notna().all()
