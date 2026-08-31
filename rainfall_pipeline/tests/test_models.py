"""Tests for Stages 2-5 -- regime engine, correction, probability, aggregation.

These are plumbing tests. They confirm that models fit, predict, save, load and
route correctly on 8 hand-written rows. They deliberately assert nothing about
accuracy: on this data no accuracy statement would mean anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rainfall_pipeline.aggregation.district import (
    CORRECTED_COLUMN,
    AggregationError,
    aggregate_to_district,
    classify_warning_level,
    list_districts,
    probability_column,
)
from rainfall_pipeline.config.thresholds import (
    EXTREMELY_HEAVY_MM,
    HEAVY_MM,
    REGIME_LABELS,
    VERY_HEAVY_MM,
    WARNING_NONE,
    WARNING_SEVERE,
    WARNING_WARNING,
    WARNING_WATCH,
)
from rainfall_pipeline.data import schema as sch
from rainfall_pipeline.models.baselines import (
    GlobalBiasCorrector,
    NotFittedError,
    QuantileMapping,
    RawForecastBaseline,
    RegimeQuantileMapping,
    residual_target,
)
from rainfall_pipeline.models.bias_correction import RegimeBiasCorrector, correct_forecast
from rainfall_pipeline.models.heavy_rain_probability import (
    HeavyRainProbabilityModel,
    attach_corrected_forecast,
    binary_target,
    predict_heavy_rain_probability,
)
from rainfall_pipeline.models.regime_classifier import (
    ModelNotTrainedError,
    RegimeClassifier,
    label_regimes,
    regime_label_summary,
)

from .conftest import FAST_PARAMS


# ---------------------------------------------------------------------------
# Stage 2 -- regime engine
# ---------------------------------------------------------------------------

def test_rule_labels_cover_every_row(dummy_regimes: pd.Series) -> None:
    """Every row must get a label from the canonical set -- no NaNs, no strays."""
    assert len(dummy_regimes) == 8
    assert dummy_regimes.notna().all()
    assert set(dummy_regimes) <= set(REGIME_LABELS)


def test_regime_summary_lists_all_five_regimes(dummy_regimes: pd.Series) -> None:
    """The summary must report zero counts, not omit unseen regimes."""
    summary = regime_label_summary(dummy_regimes)
    assert set(summary) == set(REGIME_LABELS)
    assert sum(summary.values()) == len(dummy_regimes)


def test_rule_labels_respect_the_depression_override(dummy_features: pd.DataFrame) -> None:
    """A strong vorticity anomaly must dominate whatever else the row looks like."""
    from rainfall_pipeline.config.thresholds import REGIME_DEPRESSION_LOW, REGIME_RULES

    forced = dummy_features.copy()
    forced["vorticity"] = REGIME_RULES.depression_vorticity * 2
    assert (label_regimes(forced) == REGIME_DEPRESSION_LOW).all()


def test_classifier_probabilities_span_all_regimes(dummy_features, dummy_regimes) -> None:
    """The probability frame must have all five columns and sum to one per row."""
    clf = RegimeClassifier(params=FAST_PARAMS).fit(dummy_features, dummy_regimes)
    probs = clf.predict_proba(dummy_features)
    assert list(probs.columns) == REGIME_LABELS
    assert np.allclose(probs.sum(axis=1), 1.0)
    # Regimes absent from training get exactly zero, not a spurious small value.
    unseen = set(REGIME_LABELS) - set(dummy_regimes)
    for regime in unseen:
        assert (probs[regime] == 0.0).all()


def test_classifier_refuses_a_single_class(dummy_features: pd.DataFrame) -> None:
    """One regime in training means no classifier; say so instead of pretending."""
    single = pd.Series([REGIME_LABELS[0]] * len(dummy_features), index=dummy_features.index)
    with pytest.raises(ValueError, match="Only one regime"):
        RegimeClassifier(params=FAST_PARAMS).fit(dummy_features, single)


def test_unfitted_classifier_raises_with_instructions(dummy_features: pd.DataFrame) -> None:
    """Predicting before training must point at the training script."""
    with pytest.raises(ModelNotTrainedError, match="train_regime_classifier"):
        RegimeClassifier().predict(dummy_features)


def test_shap_explanation_is_well_formed(dummy_features, dummy_regimes) -> None:
    """The explanation must name real features and rank by absolute contribution."""
    clf = RegimeClassifier(params=FAST_PARAMS).fit(dummy_features, dummy_regimes)
    explanation = clf.explain_row(dummy_features, top_n=4)
    assert explanation.regime in REGIME_LABELS
    assert len(explanation.top_features) == 4
    names = [f for f, _ in explanation.top_features]
    assert set(names) <= set(clf.feature_columns)
    magnitudes = [abs(v) for _, v in explanation.top_features]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert set(explanation.to_dict()) == {"regime", "probabilities", "top_features", "base_value"}


def test_classifier_roundtrips_through_disk(dummy_features, dummy_regimes, tmp_path) -> None:
    """A saved and reloaded classifier must give identical predictions."""
    clf = RegimeClassifier(params=FAST_PARAMS).fit(dummy_features, dummy_regimes)
    path = clf.save(tmp_path / "regime.joblib")
    reloaded = RegimeClassifier.load(path)
    pd.testing.assert_frame_equal(
        clf.predict_proba(dummy_features), reloaded.predict_proba(dummy_features)
    )


# ---------------------------------------------------------------------------
# Stage 3 -- bias correction and baselines
# ---------------------------------------------------------------------------

def test_residual_target_is_observed_minus_raw(dummy_features: pd.DataFrame) -> None:
    """The target must be the residual, not the rainfall value."""
    target = residual_target(dummy_features)
    expected = dummy_features[sch.OBSERVED_COLUMN] - dummy_features[sch.FORECAST_COLUMN]
    pd.testing.assert_series_equal(target, expected.rename("bias"))


def test_baseline_a_passes_the_forecast_through(dummy_features: pd.DataFrame) -> None:
    """Baseline A must return the raw forecast unchanged (bar the zero clip)."""
    out = RawForecastBaseline().predict(dummy_features)
    assert np.allclose(out, dummy_features[sch.FORECAST_COLUMN].clip(lower=0))


@pytest.mark.parametrize("corrector_cls", [GlobalBiasCorrector, QuantileMapping])
def test_correctors_never_return_negative_rainfall(corrector_cls, dummy_features) -> None:
    """Negative rainfall is unphysical; every corrector must clip at zero."""
    model = corrector_cls(params=FAST_PARAMS) if corrector_cls is GlobalBiasCorrector else corrector_cls()
    model.fit(dummy_features)
    assert (model.predict(dummy_features) >= 0).all()


def test_quantile_mapping_maps_onto_the_observed_range(dummy_features) -> None:
    """The mapped output must live inside the observed distribution's range."""
    qm = QuantileMapping().fit(dummy_features)
    mapped = qm.predict(dummy_features)
    observed = dummy_features[sch.OBSERVED_COLUMN]
    assert mapped.min() >= observed.min() - 1e-9
    assert mapped.max() <= observed.max() + 1e-9


def test_quantile_mapping_needs_two_rows(dummy_features) -> None:
    """A one-row CDF is not a CDF; refuse rather than emit nonsense."""
    with pytest.raises(ValueError, match="at least 2 rows"):
        QuantileMapping().fit(dummy_features.iloc[:1])


def test_regime_corrector_routes_to_the_right_model(dummy_features, dummy_regimes) -> None:
    """Each regime with enough data must get its own model; the rest fall back."""
    model = RegimeBiasCorrector(params=FAST_PARAMS, min_rows_per_regime=2).fit(
        dummy_features, dummy_regimes
    )
    counts = dummy_regimes.value_counts()
    for regime, n in counts.items():
        if n >= 2:
            assert regime in model.models
            assert model.model_for(regime) is model.models[regime]
        else:
            assert model.model_for(regime) is model.fallback
    # An unseen regime must route to the fallback rather than raising.
    assert model.model_for("Coastal") is not None


def test_corrected_forecast_is_raw_plus_predicted_bias(dummy_features, dummy_regimes) -> None:
    """The published relationship must actually hold."""
    model = RegimeBiasCorrector(params=FAST_PARAMS, min_rows_per_regime=2).fit(
        dummy_features, dummy_regimes
    )
    bias = model.predict_bias(dummy_features, dummy_regimes)
    corrected = model.predict(dummy_features, dummy_regimes)
    expected = (dummy_features[sch.FORECAST_COLUMN] + bias).clip(lower=0)
    assert np.allclose(corrected, expected)


def test_correct_forecast_handles_one_row(dummy_features, dummy_regimes) -> None:
    """The single-row API used by the service must return a plain float."""
    model = RegimeBiasCorrector(params=FAST_PARAMS, min_rows_per_regime=2).fit(
        dummy_features, dummy_regimes
    )
    value = correct_forecast(dummy_features.iloc[[0]], dummy_regimes.iloc[0], model)
    assert isinstance(value, float)
    assert value >= 0

    with pytest.raises(ValueError, match="exactly 1 row"):
        correct_forecast(dummy_features, dummy_regimes.iloc[0], model)


def test_regime_corrector_roundtrips_through_disk(dummy_features, dummy_regimes, tmp_path) -> None:
    """Saving and reloading must preserve every per-regime model."""
    model = RegimeBiasCorrector(params=FAST_PARAMS, min_rows_per_regime=2).fit(
        dummy_features, dummy_regimes
    )
    reloaded = RegimeBiasCorrector.load(model.save(tmp_path / "bias.joblib"))
    assert set(reloaded.models) == set(model.models)
    assert np.allclose(
        model.predict(dummy_features, dummy_regimes),
        reloaded.predict(dummy_features, dummy_regimes),
    )


def test_unfitted_corrector_raises_with_instructions(dummy_features, dummy_regimes) -> None:
    """Predicting before training must point at the training script."""
    with pytest.raises(NotFittedError, match="train_bias_correction"):
        RegimeBiasCorrector().predict(dummy_features, dummy_regimes)


def test_regime_quantile_mapping_covers_every_row(dummy_features, dummy_regimes) -> None:
    """Thin regimes must fall back rather than leaving rows unpredicted."""
    model = RegimeQuantileMapping().fit(dummy_features, dummy_regimes)
    out = model.predict(dummy_features, dummy_regimes)
    assert out.notna().all()


# ---------------------------------------------------------------------------
# Stage 4 -- heavy-rain probability
# ---------------------------------------------------------------------------

def test_imd_thresholds_are_the_published_values() -> None:
    """These are IMD's category boundaries and must not drift."""
    assert HEAVY_MM == 64.5
    assert VERY_HEAVY_MM == 115.6
    assert EXTREMELY_HEAVY_MM == 204.4


def test_binary_target_matches_the_threshold(dummy_features: pd.DataFrame) -> None:
    """The target must be a strict exceedance of the observation."""
    target = binary_target(dummy_features, HEAVY_MM)
    expected = dummy_features[sch.OBSERVED_COLUMN] > HEAVY_MM
    assert (target.astype(bool) == expected).all()


def test_probabilities_are_in_range_for_every_threshold(dummy_features, dummy_regimes) -> None:
    """Every calibrated probability must be a probability."""
    corrector = RegimeBiasCorrector(params=FAST_PARAMS, min_rows_per_regime=2).fit(
        dummy_features, dummy_regimes
    )
    prepared = attach_corrected_forecast(
        dummy_features, corrector.predict(dummy_features, dummy_regimes)
    )
    model = HeavyRainProbabilityModel(params=FAST_PARAMS).fit(prepared)
    probs = model.predict_proba(prepared)
    assert set(probs.columns) == {"heavy", "very_heavy", "extremely_heavy"}
    assert ((probs >= 0) & (probs <= 1)).all().all()


def test_degenerate_threshold_falls_back_to_the_base_rate(dummy_features, dummy_regimes) -> None:
    """A threshold never exceeded in training must be flagged, not fabricated."""
    corrector = RegimeBiasCorrector(params=FAST_PARAMS, min_rows_per_regime=2).fit(
        dummy_features, dummy_regimes
    )
    prepared = attach_corrected_forecast(
        dummy_features, corrector.predict(dummy_features, dummy_regimes)
    )
    model = HeavyRainProbabilityModel(params=FAST_PARAMS).fit(prepared)
    # No dummy row observes more than 215 mm... except one, so force the case.
    dry = prepared.copy()
    dry[sch.OBSERVED_COLUMN] = 1.0
    dry_model = HeavyRainProbabilityModel(params=FAST_PARAMS).fit(dry)
    assert set(dry_model.degenerate_thresholds()) == {"heavy", "very_heavy", "extremely_heavy"}
    assert (dry_model.predict_proba(dry) == 0.0).all().all()
    assert isinstance(model.degenerate_thresholds(), list)


def test_single_row_probability_api(dummy_features, dummy_regimes) -> None:
    """The service's single-row helper must return one float per threshold."""
    corrector = RegimeBiasCorrector(params=FAST_PARAMS, min_rows_per_regime=2).fit(
        dummy_features, dummy_regimes
    )
    prepared = attach_corrected_forecast(
        dummy_features, corrector.predict(dummy_features, dummy_regimes)
    )
    model = HeavyRainProbabilityModel(params=FAST_PARAMS).fit(prepared)
    out = predict_heavy_rain_probability(dummy_features.iloc[[0]], 90.0, model)
    assert set(out) == {"heavy", "very_heavy", "extremely_heavy"}
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())


def test_probability_model_roundtrips_through_disk(dummy_features, dummy_regimes, tmp_path) -> None:
    """Saving and reloading must preserve the classifiers and calibrators."""
    corrector = RegimeBiasCorrector(params=FAST_PARAMS, min_rows_per_regime=2).fit(
        dummy_features, dummy_regimes
    )
    prepared = attach_corrected_forecast(
        dummy_features, corrector.predict(dummy_features, dummy_regimes)
    )
    model = HeavyRainProbabilityModel(params=FAST_PARAMS).fit(prepared)
    reloaded = HeavyRainProbabilityModel.load(model.save(tmp_path / "prob.joblib"))
    pd.testing.assert_frame_equal(
        model.predict_proba(prepared), reloaded.predict_proba(prepared)
    )


# ---------------------------------------------------------------------------
# Stage 5 -- district aggregation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "probs, expected",
    [
        ({"heavy": 0.05, "very_heavy": 0.0}, WARNING_NONE),
        ({"heavy": 0.35, "very_heavy": 0.1}, WARNING_WATCH),
        ({"heavy": 0.60, "very_heavy": 0.2}, WARNING_WARNING),
        ({"heavy": 0.90, "very_heavy": 0.6}, WARNING_SEVERE),
        ({}, WARNING_NONE),
    ],
)
def test_warning_levels_follow_the_configured_cutpoints(probs, expected) -> None:
    """The warning ladder must match the documented probability cut-points."""
    assert classify_warning_level(probs) == expected


def test_aggregation_uses_mean_rainfall_and_max_probability(
    dummy_features, district_polygons
) -> None:
    """Rainfall averages over the district; probability takes the worst cell."""
    grid = dummy_features.drop(columns=[sch.DISTRICT_COLUMN]).copy()
    grid[CORRECTED_COLUMN] = grid[sch.FORECAST_COLUMN] * 1.2
    grid[probability_column("heavy")] = [0.1, 0.2, 0.9, 0.05, 0.4, 0.7, 0.0, 0.8]
    grid[probability_column("very_heavy")] = 0.05

    out = aggregate_to_district(
        grid, district_polygons, threshold_names=["heavy", "very_heavy"]
    )
    assert {"date", sch.DISTRICT_COLUMN, "mean_corrected_mm", "warning_level"} <= set(out.columns)

    # Each dummy district has exactly one cell per date, so the district value
    # must equal that cell's value.
    merged = out.merge(
        grid.assign(district=dummy_features[sch.DISTRICT_COLUMN]),
        on=["date", sch.DISTRICT_COLUMN],
    )
    assert np.allclose(merged["mean_corrected_mm"], merged[CORRECTED_COLUMN])
    assert np.allclose(merged["max_prob_heavy"], merged[probability_column("heavy")])


def test_aggregation_without_districts_explains_itself(dummy_features) -> None:
    """No district labels and no shapefile must produce a useful error."""
    grid = dummy_features.drop(columns=[sch.DISTRICT_COLUMN]).copy()
    grid[CORRECTED_COLUMN] = grid[sch.FORECAST_COLUMN]
    with pytest.raises(AggregationError, match="load_district_boundaries"):
        aggregate_to_district(grid, None)


def test_aggregation_requires_a_corrected_forecast(dummy_features, district_polygons) -> None:
    """Aggregating before correction must point at the correction stage."""
    grid = dummy_features.drop(columns=[sch.DISTRICT_COLUMN]).copy()
    with pytest.raises(AggregationError, match="bias correction"):
        aggregate_to_district(grid, district_polygons)


def test_aggregation_can_return_geometry(dummy_features, district_polygons) -> None:
    """The GeoJSON-ready variant must carry a geometry for every district."""
    grid = dummy_features.drop(columns=[sch.DISTRICT_COLUMN]).copy()
    grid[CORRECTED_COLUMN] = grid[sch.FORECAST_COLUMN]
    out = aggregate_to_district(grid, district_polygons, return_geometry=True)
    assert out.geometry.notna().all()


def test_assign_region_labels_every_row(dummy_features) -> None:
    """Every cell must get a region; unmatched cells become 'other', not NaN."""
    from rainfall_pipeline.aggregation.district import assign_region

    regions = assign_region(dummy_features)
    assert regions.notna().all()
    assert len(regions) == len(dummy_features)


def test_assign_region_places_known_points_correctly() -> None:
    """Coordinates in each configured box must land in that box's region."""
    from rainfall_pipeline.aggregation.district import assign_region

    points = pd.DataFrame(
        {"lat": [12.0, 22.0, 26.0, 30.0, 45.0], "lon": [75.0, 80.0, 92.0, 76.0, 10.0]}
    )
    assert assign_region(points).tolist() == [
        "west_coast",
        "central_india",
        "northeast",
        "north_india",
        "other",
    ]


def test_assign_region_requires_coordinates() -> None:
    """Without lat/lon there is no region; say so rather than guessing."""
    from rainfall_pipeline.aggregation.district import assign_region

    with pytest.raises(AggregationError, match="assign_region needs"):
        assign_region(pd.DataFrame({"x": [1]}))


def test_list_districts_merges_both_sources(dummy_features, district_polygons) -> None:
    """District names come from the shapefile, the data, or both."""
    assert list_districts(district_polygons) == ["Bhopal", "Patna", "Pune", "Wayanad"]
    assert list_districts(None, dummy_features) == ["Bhopal", "Patna", "Pune", "Wayanad"]
    assert list_districts(None, None) == []


# ---------------------------------------------------------------------------
# Soft (probability-weighted) regime routing
# ---------------------------------------------------------------------------

def _fitted_corrector(dummy_features, dummy_regimes):
    """A corrector fitted on the dummy rows, with a model for every regime present."""
    return RegimeBiasCorrector(params=FAST_PARAMS, min_rows_per_regime=1).fit(
        dummy_features, dummy_regimes
    )


def _one_hot(regimes: pd.Series, index) -> pd.DataFrame:
    """Build a degenerate probability frame that puts all mass on one regime."""
    frame = pd.DataFrame(0.0, index=index, columns=REGIME_LABELS)
    for row, regime in zip(index, regimes):
        frame.loc[row, str(regime)] = 1.0
    return frame


def test_soft_routing_matches_hard_routing_when_certain(dummy_features, dummy_regimes) -> None:
    """A one-hot distribution must reproduce hard routing exactly.

    This is the property that makes soft routing safe to turn on by default: it
    is a strict generalisation, not a different model.
    """
    corrector = _fitted_corrector(dummy_features, dummy_regimes)
    probs = _one_hot(dummy_regimes, dummy_features.index)

    hard = corrector.predict_bias(dummy_features, dummy_regimes)
    soft = corrector.predict_bias_soft(dummy_features, probs)
    pd.testing.assert_series_equal(hard, soft, check_names=False, rtol=1e-9)


def test_soft_routing_blends_between_regimes(dummy_features, dummy_regimes) -> None:
    """A 50/50 split must land between the two regimes' own predictions."""
    corrector = _fitted_corrector(dummy_features, dummy_regimes)
    available = sorted(corrector.models)
    if len(available) < 2:
        pytest.skip("dummy rows produced fewer than two per-regime models")
    first, second = available[0], available[1]

    only_first = corrector.predict_bias(
        dummy_features, pd.Series(first, index=dummy_features.index)
    )
    only_second = corrector.predict_bias(
        dummy_features, pd.Series(second, index=dummy_features.index)
    )

    probs = pd.DataFrame(0.0, index=dummy_features.index, columns=REGIME_LABELS)
    probs[first] = 0.5
    probs[second] = 0.5
    blended = corrector.predict_bias_soft(dummy_features, probs)

    expected = 0.5 * only_first + 0.5 * only_second
    np.testing.assert_allclose(blended.values, expected.values, rtol=1e-9, atol=1e-9)


def test_soft_routing_normalises_unnormalised_input(dummy_features, dummy_regimes) -> None:
    """Weights that do not sum to 1 must be renormalised, not trusted blindly."""
    corrector = _fitted_corrector(dummy_features, dummy_regimes)
    regime = sorted(corrector.models)[0]

    unit = pd.DataFrame(0.0, index=dummy_features.index, columns=REGIME_LABELS)
    unit[regime] = 1.0
    doubled = unit * 2.0

    pd.testing.assert_series_equal(
        corrector.predict_bias_soft(dummy_features, unit),
        corrector.predict_bias_soft(dummy_features, doubled),
        rtol=1e-9,
    )


def test_soft_routing_sends_all_zero_rows_to_the_fallback(
    dummy_features, dummy_regimes
) -> None:
    """A row carrying no regime information must use the global model."""
    corrector = _fitted_corrector(dummy_features, dummy_regimes)
    zeros = pd.DataFrame(0.0, index=dummy_features.index, columns=REGIME_LABELS)

    blended = corrector.predict_bias_soft(dummy_features, zeros)
    fallback = corrector.fallback.predict_bias(dummy_features)
    np.testing.assert_allclose(blended.values, fallback.values, rtol=1e-9, atol=1e-9)


def test_soft_routing_pools_unmodelled_regimes_onto_the_fallback(
    dummy_features, dummy_regimes
) -> None:
    """Probability mass on a regime with no model must not be silently dropped."""
    corrector = RegimeBiasCorrector(params=FAST_PARAMS, min_rows_per_regime=10**6).fit(
        dummy_features, dummy_regimes
    )
    assert not corrector.models, "this test needs every regime to be too thin"

    probs = pd.DataFrame(1.0 / len(REGIME_LABELS), index=dummy_features.index, columns=REGIME_LABELS)
    blended = corrector.predict_bias_soft(dummy_features, probs)
    fallback = corrector.fallback.predict_bias(dummy_features)
    np.testing.assert_allclose(blended.values, fallback.values, rtol=1e-9, atol=1e-9)


def test_soft_routing_rejects_misaligned_probabilities(dummy_features, dummy_regimes) -> None:
    """A probability frame of the wrong length must raise, not broadcast."""
    corrector = _fitted_corrector(dummy_features, dummy_regimes)
    probs = pd.DataFrame(0.2, index=range(3), columns=REGIME_LABELS)
    with pytest.raises(ValueError, match="rows but df has"):
        corrector.predict_bias_soft(dummy_features, probs)


def test_predict_soft_clips_at_zero(dummy_features, dummy_regimes) -> None:
    """Soft routing must respect the same physical floor as hard routing."""
    corrector = _fitted_corrector(dummy_features, dummy_regimes)
    probs = _one_hot(dummy_regimes, dummy_features.index)
    assert (corrector.predict_soft(dummy_features, probs) >= 0).all()


# ---------------------------------------------------------------------------
# Explaining the correction
# ---------------------------------------------------------------------------

def test_corrector_explains_its_own_correction(dummy_features, dummy_regimes) -> None:
    """The corrector must explain the rainfall change it applied."""
    corrector = _fitted_corrector(dummy_features, dummy_regimes)
    regime = str(dummy_regimes.iloc[0])

    detail = corrector.explain_row(dummy_features.iloc[[0]], regime, top_n=3)
    assert len(detail.top_features) <= 3
    assert detail.corrected_mm == pytest.approx(
        max(detail.raw_mm + detail.predicted_bias_mm, 0.0), abs=1e-6
    )
    # Contributions must be ordered by absolute influence.
    magnitudes = [abs(v) for _, v in detail.top_features]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_bias_explanation_matches_the_prediction(dummy_features, dummy_regimes) -> None:
    """The explained bias must be the bias the model actually predicted."""
    corrector = _fitted_corrector(dummy_features, dummy_regimes)
    row = dummy_features.iloc[[0]]
    regime = str(dummy_regimes.iloc[0])

    detail = corrector.explain_row(row, regime)
    predicted = float(corrector.predict_bias(row, pd.Series([regime], index=row.index)).iloc[0])
    assert detail.predicted_bias_mm == pytest.approx(predicted, abs=1e-6)


def test_bias_explanation_admits_when_it_used_the_fallback(
    dummy_features, dummy_regimes
) -> None:
    """A regime with no model of its own must be reported as the fallback.

    Claiming a specialised correction that did not happen would make the
    dashboard's explanation a fiction.
    """
    corrector = RegimeBiasCorrector(params=FAST_PARAMS, min_rows_per_regime=10**6).fit(
        dummy_features, dummy_regimes
    )
    detail = corrector.explain_row(dummy_features.iloc[[0]], str(dummy_regimes.iloc[0]))
    assert detail.regime == "__fallback__"


def test_explain_row_rejects_an_empty_frame(dummy_features, dummy_regimes) -> None:
    """Explaining nothing must raise rather than return an empty explanation."""
    corrector = _fitted_corrector(dummy_features, dummy_regimes)
    with pytest.raises(ValueError, match="at least one row"):
        corrector.explain_row(dummy_features.iloc[0:0], REGIME_LABELS[0])
