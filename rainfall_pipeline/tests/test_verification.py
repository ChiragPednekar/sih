"""Tests for Stage 6 -- metrics, splitting and the report.

Metrics are tested against hand-computed values so a regression in the formulas
is caught immediately. The split tests exist because a random split would
silently invalidate every number the project produces.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rainfall_pipeline.verification import metrics as mx
from rainfall_pipeline.verification.report import (
    LEAD_TIME_COLUMN,
    VerificationInputs,
    brier_score,
    brier_skill_score,
    build_verification_report,
    reliability_curve,
    render_html,
    render_markdown,
    summary_table,
    write_report,
)
from rainfall_pipeline.verification.splits import SplitError, chronological_split


# ---------------------------------------------------------------------------
# Continuous metrics
# ---------------------------------------------------------------------------

def test_continuous_metrics_against_hand_computed_values() -> None:
    """RMSE, bias and MAE must match values computed by hand."""
    observed = [0.0, 10.0, 20.0]
    predicted = [1.0, 8.0, 26.0]
    # errors: +1, -2, +6  ->  mean +5/3, MAE 3, RMSE sqrt((1+4+36)/3)
    assert mx.mean_bias(observed, predicted) == pytest.approx(5 / 3)
    assert mx.mean_absolute_error(observed, predicted) == pytest.approx(3.0)
    assert mx.rmse(observed, predicted) == pytest.approx(np.sqrt(41 / 3))


def test_metrics_ignore_non_finite_pairs() -> None:
    """A missing observation must be dropped, not treated as zero."""
    assert mx.rmse([1.0, np.nan, 3.0], [1.0, 99.0, 3.0]) == pytest.approx(0.0)


def test_metrics_return_nan_rather_than_guessing() -> None:
    """With no usable pairs the answer is 'undefined', not a number."""
    assert np.isnan(mx.rmse([], []))
    assert np.isnan(mx.correlation([1.0, 1.0], [2.0, 3.0]))


def test_mismatched_lengths_raise() -> None:
    """Silently truncating misaligned series would corrupt every score."""
    with pytest.raises(ValueError, match="same shape"):
        mx.rmse([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# Categorical metrics
# ---------------------------------------------------------------------------

def test_contingency_table_counts() -> None:
    """The 2x2 table must classify each pair into exactly one cell."""
    observed = [100.0, 100.0, 10.0, 10.0]
    predicted = [100.0, 10.0, 100.0, 10.0]
    table = mx.contingency_table(observed, predicted, 64.5)
    assert (table.hits, table.misses, table.false_alarms, table.correct_negatives) == (1, 1, 1, 1)
    assert table.total == 4


def test_categorical_scores_against_hand_computed_values() -> None:
    """POD, FAR, CSI and ETS must match the textbook definitions."""
    table = mx.ContingencyTable(hits=6, false_alarms=2, misses=3, correct_negatives=89)
    assert mx.pod(table) == pytest.approx(6 / 9)
    assert mx.far(table) == pytest.approx(2 / 8)
    assert mx.csi(table) == pytest.approx(6 / 11)
    assert mx.frequency_bias(table) == pytest.approx(8 / 9)
    hits_random = 9 * 8 / 100
    assert mx.ets(table) == pytest.approx((6 - hits_random) / (11 - hits_random))


def test_perfect_and_useless_forecasts_bracket_the_scores() -> None:
    """A perfect forecast scores 1; one that always misses scores 0."""
    observed = [100.0, 100.0, 10.0, 10.0]
    perfect = mx.categorical_scores(observed, observed, 64.5)
    assert perfect["pod"] == 1.0 and perfect["far"] == 0.0 and perfect["csi"] == 1.0

    always_dry = mx.categorical_scores(observed, [0.0] * 4, 64.5)
    assert always_dry["pod"] == 0.0 and always_dry["csi"] == 0.0


def test_ets_is_undefined_for_an_empty_sample() -> None:
    """No data means no skill score, not a default value."""
    assert np.isnan(mx.ets(mx.ContingencyTable(0, 0, 0, 0)))


# ---------------------------------------------------------------------------
# FSS
# ---------------------------------------------------------------------------

def test_fss_is_one_for_an_identical_field() -> None:
    """A forecast identical to the observation must score a perfect FSS."""
    field = np.zeros((5, 5))
    field[2, 2] = 100.0
    assert mx.fss_from_fields(field, field, 64.5, 1) == pytest.approx(1.0)


def test_fss_penalises_a_displaced_feature_less_as_the_window_grows() -> None:
    """This is the whole point of FSS: a near-miss should not score as a total miss."""
    observed = np.zeros((7, 7))
    observed[3, 3] = 100.0
    predicted = np.zeros((7, 7))
    predicted[3, 4] = 100.0  # displaced by one grid cell

    point_scale = mx.fss_from_fields(observed, predicted, 64.5, 1)
    neighbourhood = mx.fss_from_fields(observed, predicted, 64.5, 3)
    assert point_scale == pytest.approx(0.0)
    assert neighbourhood > point_scale


def test_fss_is_undefined_when_nothing_exceeds_the_threshold() -> None:
    """A dry day must not inflate the seasonal mean with a free 1.0."""
    dry = np.zeros((4, 4))
    assert np.isnan(mx.fss_from_fields(dry, dry, 64.5, 3))


def test_fss_rejects_mismatched_fields() -> None:
    """Comparing differently shaped grids is a bug, not a score."""
    with pytest.raises(ValueError, match="Field shapes differ"):
        mx.fss_from_fields(np.zeros((3, 3)), np.zeros((4, 4)), 1.0, 1)


def test_to_field_pivots_onto_a_regular_grid(dummy_features: pd.DataFrame) -> None:
    """Gaps in the grid must become NaN, not shift the remaining cells."""
    day = dummy_features[dummy_features["date"] == pd.Timestamp("2020-07-01")]
    field = mx.to_field(day, "observed_mm")
    assert field.ndim == 2
    assert np.isfinite(field).sum() == len(day)


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

def test_intensity_buckets_partition_the_observations(dummy_features: pd.DataFrame) -> None:
    """Every observed value must land in exactly one bucket."""
    buckets = mx.bucket_by_intensity(dummy_features, "observed_mm")
    assert buckets.notna().all()
    assert set(buckets) <= {
        "no_rain", "light", "moderate", "heavy", "very_heavy", "extremely_heavy"
    }


def test_evaluate_by_group_covers_every_group(dummy_features, dummy_regimes) -> None:
    """Stratified results must include one entry per group present."""
    df = dummy_features.copy()
    df["regime"] = dummy_regimes.values
    df["pred"] = df["raw_forecast_mm"]
    out = mx.evaluate_by_group(df, "observed_mm", "pred", "regime")
    assert set(out) == set(dummy_regimes.unique())


def test_evaluate_by_group_rejects_an_unknown_column(dummy_features) -> None:
    """Stratifying by a column that does not exist must fail loudly."""
    df = dummy_features.assign(pred=dummy_features["raw_forecast_mm"])
    with pytest.raises(KeyError, match="not present"):
        mx.evaluate_by_group(df, "observed_mm", "pred", "no_such_column")


# ---------------------------------------------------------------------------
# Chronological splitting
# ---------------------------------------------------------------------------

def test_split_is_strictly_chronological(dummy_features: pd.DataFrame) -> None:
    """Every training date must precede every test date -- no interleaving."""
    split = chronological_split(dummy_features)
    assert split.train["date"].max() < split.validation["date"].min()
    assert split.validation["date"].max() < split.test["date"].min()


def test_split_keeps_a_whole_day_on_one_side(dummy_features: pd.DataFrame) -> None:
    """All grid cells for a date must land in the same split."""
    split = chronological_split(dummy_features)
    dates = [set(part["date"]) for part in (split.train, split.validation, split.test)]
    assert dates[0].isdisjoint(dates[1])
    assert dates[1].isdisjoint(dates[2])
    assert dates[0].isdisjoint(dates[2])


def test_explicit_boundaries_are_honoured(dummy_features: pd.DataFrame) -> None:
    """Configured split dates must override the fraction fallback exactly."""
    split = chronological_split(
        dummy_features, train_end="2020-07-01", val_end="2020-07-02"
    )
    assert split.train["date"].max() == pd.Timestamp("2020-07-01")
    assert set(split.test["date"].dt.strftime("%Y-%m-%d")) == {"2020-07-03", "2020-07-04"}


def test_split_refuses_too_few_dates(dummy_features: pd.DataFrame) -> None:
    """Two dates cannot make a three-way split; say so rather than guess."""
    two_days = dummy_features[dummy_features["date"] <= pd.Timestamp("2020-07-02")]
    with pytest.raises(SplitError, match="at least 3"):
        chronological_split(two_days)


def test_split_refuses_boundaries_that_empty_a_side(dummy_features: pd.DataFrame) -> None:
    """A boundary past the end of the data leaves no test set; that must fail."""
    with pytest.raises(SplitError, match="empty"):
        chronological_split(dummy_features, train_end="2030-01-01", val_end="2030-01-02")


def test_split_has_no_random_state() -> None:
    """A shuffle option must not exist -- its presence would invite misuse."""
    import inspect

    signature = inspect.signature(chronological_split)
    assert "random_state" not in signature.parameters
    assert "shuffle" not in signature.parameters


# ---------------------------------------------------------------------------
# Probabilistic scores and the report
# ---------------------------------------------------------------------------

def test_brier_score_against_hand_computed_values() -> None:
    """The Brier score is the mean squared probability error."""
    observed = pd.Series([1.0, 0.0])
    probability = pd.Series([0.8, 0.3])
    assert brier_score(observed, probability) == pytest.approx((0.04 + 0.09) / 2)


def test_brier_skill_score_is_zero_for_a_climatological_forecast() -> None:
    """Forecasting the base rate every day must score exactly no skill."""
    observed = pd.Series([1.0, 0.0, 0.0, 0.0])
    climatology = pd.Series([0.25] * 4)
    assert brier_skill_score(observed, climatology) == pytest.approx(0.0)


def test_reliability_curve_bins_are_well_formed() -> None:
    """Each bin must report its own count, mean forecast and observed frequency."""
    observed = pd.Series([1.0, 1.0, 0.0, 0.0])
    probability = pd.Series([0.95, 0.85, 0.05, 0.15])
    curve = reliability_curve(observed, probability, n_bins=10)
    assert sum(row["n"] for row in curve) == 4
    for row in curve:
        assert row["bin_lower"] <= row["mean_probability"] <= row["bin_upper"]
        assert 0.0 <= row["observed_frequency"] <= 1.0


def _make_report(dummy_features: pd.DataFrame, dummy_regimes: pd.Series) -> dict:
    """Build a report over all five model slots from the dummy rows."""
    test_df = dummy_features.copy()
    test_df["regime"] = dummy_regimes.values
    raw = test_df["raw_forecast_mm"]
    predictions = {
        "A_raw_nwp": raw,
        "B_global_ml": raw * 1.1,
        "C_quantile_mapping": raw * 1.2,
        "D_regime_residual": raw * 1.3,
        "E_regime_residual_probability": raw * 1.3,
    }
    probabilities = {
        "E_regime_residual_probability": {
            "heavy": pd.Series(np.linspace(0.05, 0.95, len(test_df)), index=test_df.index)
        }
    }
    return build_verification_report(
        VerificationInputs(
            test_df=test_df,
            predictions=predictions,
            probabilities=probabilities,
            split_summary={"train": {"n_rows": 4}},
        )
    )


def test_report_contains_all_five_models(dummy_features, dummy_regimes) -> None:
    """The brief requires A through E side by side; the report must deliver that."""
    report = _make_report(dummy_features, dummy_regimes)
    assert set(report["models"]) == {
        "A_raw_nwp",
        "B_global_ml",
        "C_quantile_mapping",
        "D_regime_residual",
        "E_regime_residual_probability",
    }
    table = summary_table(report)
    assert len(table) == 5


def test_report_breaks_results_down_by_regime_and_intensity(dummy_features, dummy_regimes) -> None:
    """A single overall number hides where a model actually helps."""
    report = _make_report(dummy_features, dummy_regimes)
    assert "regime" in report["stratified_by"]
    assert "intensity_bucket" in report["stratified_by"]
    entry = report["models"]["D_regime_residual"]
    assert set(entry["by"]["regime"]) == set(dummy_regimes.unique())


def test_report_notes_that_d_and_e_share_a_rainfall_field(dummy_features, dummy_regimes) -> None:
    """Two identical rows must be explained, not left to be misread as a result."""
    report = _make_report(dummy_features, dummy_regimes)
    assert any("identical by construction" in note for note in report["notes"])


def test_report_records_models_it_could_not_evaluate(dummy_features, dummy_regimes) -> None:
    """A missing model is an omission to disclose, not a row to invent."""
    test_df = dummy_features.copy()
    report = build_verification_report(
        VerificationInputs(
            test_df=test_df, predictions={"A_raw_nwp": test_df["raw_forecast_mm"]}
        )
    )
    assert any("Not evaluated" in note for note in report["notes"])
    assert "B_global_ml" not in report["models"]


def test_report_rejects_an_empty_prediction_set(dummy_features) -> None:
    """Verifying nothing must raise rather than emit an empty 'result'."""
    with pytest.raises(ValueError, match="nothing to verify"):
        build_verification_report(VerificationInputs(test_df=dummy_features, predictions={}))


def test_report_writes_json_markdown_and_html(dummy_features, dummy_regimes, tmp_path) -> None:
    """All three output formats must be produced and be valid."""
    report = _make_report(dummy_features, dummy_regimes)
    paths = write_report(
        report,
        json_path=tmp_path / "r.json",
        markdown_path=tmp_path / "r.md",
        html_path=tmp_path / "r.html",
    )
    loaded = json.loads(paths["json"].read_text())
    assert set(loaded["models"]) == set(report["models"])

    markdown = paths["markdown"].read_text()
    assert "# Verification report" in markdown
    assert "A_raw_nwp" in markdown and "E_regime_residual_probability" in markdown

    html = paths["html"].read_text()
    assert html.startswith("<!doctype html>")


def test_json_output_has_no_nan_literals(dummy_features, dummy_regimes, tmp_path) -> None:
    """``NaN`` is not valid JSON; undefined metrics must serialise as null."""
    report = _make_report(dummy_features, dummy_regimes)
    path = write_report(
        report,
        json_path=tmp_path / "r.json",
        markdown_path=tmp_path / "r.md",
        html_path=tmp_path / "r.html",
    )["json"]
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)  # strict parse: would raise on NaN


def test_rendered_output_contains_no_improvement_claim(dummy_features, dummy_regimes) -> None:
    """No percentage-improvement claim may ever be hardcoded into a report."""
    report = _make_report(dummy_features, dummy_regimes)
    for text in (render_markdown(report), render_html(report)):
        lowered = text.lower()
        assert "% improvement" not in lowered
        assert "improves by" not in lowered
        assert "better than" not in lowered


# ---------------------------------------------------------------------------
# Lead-time stratification
# ---------------------------------------------------------------------------

def _report_with_leads(dummy_features: pd.DataFrame, leads) -> dict:
    """Build a report over a copy of the dummy rows carrying given lead times."""
    test_df = dummy_features.copy()
    test_df["lead_time"] = list(leads)
    predictions = {"A_raw_nwp": test_df["raw_forecast_mm"].to_numpy()}
    return build_verification_report(
        VerificationInputs(test_df=test_df, predictions=predictions)
    )


def test_report_stratifies_by_lead_time_when_leads_vary(dummy_features) -> None:
    """A forecast set spanning several leads must be scored per lead.

    Skill decays with lead time; a single pooled number hides whether the
    correction still helps at day 3, which is exactly what an operational
    reader needs to know.
    """
    leads = [1, 1, 2, 2, 3, 3, 1, 2][: len(dummy_features)]
    report = _report_with_leads(dummy_features, leads)

    assert LEAD_TIME_COLUMN in report["stratified_by"]
    groups = report["models"]["A_raw_nwp"]["by"][LEAD_TIME_COLUMN]
    assert set(groups) == {"1", "2", "3"}
    for scores in groups.values():
        assert "rmse" in scores["continuous"]


def test_report_omits_lead_time_when_there_is_only_one(dummy_features) -> None:
    """A single lead makes a one-row table that says nothing; skip it."""
    report = _report_with_leads(dummy_features, [1] * len(dummy_features))
    assert LEAD_TIME_COLUMN not in report["stratified_by"]


def test_report_omits_lead_time_when_the_column_is_absent(dummy_features) -> None:
    """A dataset with no lead information must not grow a phantom column."""
    test_df = dummy_features.drop(columns=["lead_time"], errors="ignore")
    report = build_verification_report(
        VerificationInputs(
            test_df=test_df, predictions={"A_raw_nwp": test_df["raw_forecast_mm"].to_numpy()}
        )
    )
    assert LEAD_TIME_COLUMN not in report["stratified_by"]


def test_lead_time_buckets_are_whole_steps(dummy_features) -> None:
    """Continuous leads must be bucketed, not grouped on raw floats."""
    leads = [1.0, 1.2, 2.0, 2.4, 3.0, 2.6, 1.4, 3.1][: len(dummy_features)]
    report = _report_with_leads(dummy_features, leads)
    groups = report["models"]["A_raw_nwp"]["by"][LEAD_TIME_COLUMN]
    assert set(groups) <= {"1", "2", "3"}, "leads must round to whole forecast steps"


def test_lead_time_appears_in_the_markdown(dummy_features) -> None:
    """The rendered report must show the lead-time breakdown, not just carry it."""
    leads = [1, 2, 3, 1, 2, 3, 1, 2][: len(dummy_features)]
    markdown = render_markdown(_report_with_leads(dummy_features, leads))
    assert f"## RMSE by {LEAD_TIME_COLUMN}" in markdown
