"""Pydantic request/response models for the FastAPI service."""

from __future__ import annotations

from datetime import date as _date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    """Health and readiness of the service.

    Attributes:
        status: ``"ok"`` when the process is up. Always ``"ok"`` if you got a
            response at all -- readiness is described by the other fields.
        models_loaded: Whether a usable set of trained artifacts was found.
        data_connected: Whether an analysis table is available to serve from.
        districts_available: Whether district boundaries or district labels
            could be resolved.
        missing: Human-readable list of what is still needed.
        artifact_dir: Where the service looked for model artifacts.
    """

    status: str = "ok"
    models_loaded: bool
    data_connected: bool
    districts_available: bool
    missing: List[str] = Field(default_factory=list)
    artifact_dir: str


class FeatureContribution(BaseModel):
    """One feature's SHAP contribution to a regime prediction.

    Attributes:
        feature: Feature name.
        shap_value: Signed contribution to the predicted class's log-odds.
    """

    feature: str
    shap_value: float


class RegimeExplanationResponse(BaseModel):
    """Why the regime engine chose the regime it did.

    Attributes:
        top_features: Largest absolute contributions, most influential first.
        base_value: The model's expected output before feature contributions.
    """

    top_features: List[FeatureContribution] = Field(default_factory=list)
    base_value: Optional[float] = None


class BiasContribution(BaseModel):
    """One feature's SHAP contribution to the applied correction.

    Attributes:
        feature: Feature name.
        shap_value: Signed contribution **in mm of bias**. Positive means this
            feature pushed the correction upward.
    """

    feature: str
    shap_value: float


class BiasExplanationResponse(BaseModel):
    """Why the corrector changed the rainfall amount that it did.

    This is the panel the dashboard labels "Why did AI correct this?". It
    explains the *correction*, which is a different question from why the
    regime engine picked a regime -- both are returned so neither stands in for
    the other.

    Attributes:
        regime: Which regime's model produced the correction, or
            ``"__fallback__"`` if the regime had too few training rows to earn
            its own model.
        raw_mm: The uncorrected forecast at the explained grid cell.
        predicted_bias_mm: The signed correction applied, in mm.
        corrected_mm: The result after correction.
        top_features: Largest absolute contributions, most influential first.
        base_value: Expected bias before any feature contributions.
    """

    regime: str
    raw_mm: Optional[float] = None
    predicted_bias_mm: Optional[float] = None
    corrected_mm: Optional[float] = None
    top_features: List[BiasContribution] = Field(default_factory=list)
    base_value: Optional[float] = None


class RegimeBlendComponent(BaseModel):
    """One regime's share of a blended regime assignment.

    Attributes:
        regime: Regime label.
        probability: The classifier's probability for it over the district.
    """

    regime: str
    probability: float


class PredictionResponse(BaseModel):
    """The full pipeline output for one district on one date.

    Attributes:
        date: Valid date of the forecast.
        district: District name.
        regime: Most likely monsoon regime over the district (the argmax).
        regime_label: Display label. When a second regime carries meaningful
            probability this reads ``"Coastal + Active"``; otherwise it equals
            ``regime``.
        regime_blend: The regimes making up ``regime_label``, largest first.
        regime_probability: Probability distribution over all five regimes.
        routing: ``"soft"`` when the correction blended every regime's model by
            probability, ``"hard"`` when it was routed to the argmax alone.
        raw_forecast_mm: District-mean uncorrected NWP rainfall.
        corrected_forecast_mm: District-mean corrected rainfall.
        heavy_rain_probability: ``{threshold_name: calibrated probability}``,
            taken as the maximum over the district's grid cells.
        warning_level: ``none`` / ``watch`` / ``warning`` / ``severe``.
        n_grid_cells: How many grid cells the district aggregate covers.
        explanation: SHAP explanation for the regime call.
        bias_explanation: SHAP explanation for the rainfall correction itself.
        peak_cell_mm: The wettest single grid cell, which drives the warning
            and which the district mean hides.
        centroid_lat: Mean latitude of the district's grid cells.
        centroid_lon: Mean longitude of the district's grid cells.
        regime_confidence: Probability of the leading regime.
        interval: The range the corrected amount is expected to fall in, with
            its measured coverage. Null when no interval model is loaded.
        anomaly: The forecast against what is normal for this place and month.
        threshold_probabilities: Every threshold as a list, including a custom
            one when the caller asked for it.
        observed_mm: District-mean observation, when the date is historical and
            an observation exists. Null for a genuine forecast.
    """

    date: _date
    district: str
    regime: str
    regime_label: str
    regime_blend: List[RegimeBlendComponent] = Field(default_factory=list)
    regime_probability: Dict[str, float]
    routing: str = "soft"
    raw_forecast_mm: Optional[float] = None
    corrected_forecast_mm: float
    heavy_rain_probability: Dict[str, float]
    warning_level: str
    n_grid_cells: int
    explanation: Optional[RegimeExplanationResponse] = None
    bias_explanation: Optional[BiasExplanationResponse] = None
    observed_mm: Optional[float] = None
    peak_cell_mm: Optional[float] = None
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None
    regime_confidence: float = 0.0
    interval: Optional[ForecastInterval] = None
    anomaly: Optional[RainfallAnomaly] = None
    threshold_probabilities: List[ThresholdProbability] = Field(default_factory=list)


class DistrictListResponse(BaseModel):
    """Districts the service can serve predictions for.

    Attributes:
        count: Number of districts.
        source: Where the names came from -- ``"shapefile"``, ``"data"``, or
            ``"none"``.
        districts: Sorted district names.
    """

    count: int
    source: str
    districts: List[str] = Field(default_factory=list)


class NotReadyResponse(BaseModel):
    """Returned instead of a prediction when the system is not trained yet.

    Attributes:
        detail: What is missing and what to do about it.
        missing: Machine-readable list of missing prerequisites.
        next_steps: Ordered commands to run.
    """

    detail: str
    missing: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)


class VerificationReportResponse(BaseModel):
    """The saved verification report, or an explanation of its absence.

    Attributes:
        available: Whether a report has been generated.
        detail: Explanation when ``available`` is False.
        report: The report JSON when available.
    """

    available: bool
    detail: Optional[str] = None
    report: Optional[dict] = None


class GridPanelStats(BaseModel):
    """Summary of one map panel, used to drive a shared legend.

    Attributes:
        name: Panel key -- ``raw``, ``corrected``, ``difference`` or
            ``observed``.
        label: Human-readable panel title.
        min_value: Smallest finite value on the panel.
        max_value: Largest finite value on the panel.
        n_finite: How many cells carry a value.
    """

    name: str
    label: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    n_finite: int = 0


class GridResponse(BaseModel):
    """Gridded fields for one date, ready to draw as map panels.

    Every panel shares one date, one extent and one set of cells, so the
    dashboard can render them with a common legend -- which is the only way a
    raw-vs-corrected comparison means anything.

    Attributes:
        date: Valid date of the forecast.
        n_cells: Number of grid cells returned.
        lats: Sorted unique latitudes.
        lons: Sorted unique longitudes.
        bbox: ``[lat_min, lat_max, lon_min, lon_max]``.
        regimes: Per-cell regime label, row-major over ``lats`` x ``lons``.
        panels: ``{panel_name: [value_or_null, ...]}`` row-major over
            ``lats`` x ``lons``.
        panel_stats: Per-panel ranges for the legend.
        observed_available: Whether the observed panel carries real values --
            false for a genuine forecast date, where there is nothing to
            compare against yet.
        synthetic: True when the connected dataset is the fabricated demo set,
            so the dashboard can label the maps as an illustrative mock-up.
    """

    date: _date
    n_cells: int
    lats: List[float] = Field(default_factory=list)
    lons: List[float] = Field(default_factory=list)
    bbox: List[float] = Field(default_factory=list)
    regimes: List[Optional[str]] = Field(default_factory=list)
    panels: Dict[str, List[Optional[float]]] = Field(default_factory=dict)
    panel_stats: List[GridPanelStats] = Field(default_factory=list)
    observed_available: bool = False
    synthetic: bool = False


class DateRangeResponse(BaseModel):
    """The dates the connected dataset can serve.

    Attributes:
        available: Whether any data is connected.
        start: First date with data.
        end: Last date with data.
        n_dates: How many distinct dates are available.
        synthetic: True when the connected dataset is the fabricated demo set.
    """

    available: bool
    start: Optional[_date] = None
    end: Optional[_date] = None
    n_dates: int = 0
    synthetic: bool = False


# ---------------------------------------------------------------------------
# Uncertainty, anomaly and custom thresholds  (spec 4.7, 4.8, section 2)
# ---------------------------------------------------------------------------

class ForecastInterval(BaseModel):
    """The range a corrected forecast is expected to fall in.

    Attributes:
        low_mm: Lower bound of the band.
        high_mm: Upper bound of the band.
        nominal_coverage: The fraction of observations the band is *meant* to
            contain, e.g. 0.8 for a 10th-to-90th-percentile band.
        measured_coverage: The fraction it *actually* contained on the held-out
            test set, or null if that was never measured. Read this before
            trusting the band -- a nominal 80% band that caught 55% of
            observations is a false reassurance, not a range.
        calibrated: True when measured coverage is within 10 points of nominal.
    """

    low_mm: float
    high_mm: float
    nominal_coverage: Optional[float] = None
    measured_coverage: Optional[float] = None
    calibrated: Optional[bool] = None


class RainfallAnomaly(BaseModel):
    """The forecast set against what is normal for this place and month.

    Attributes:
        climatology_mm: The historical mean for this district and month.
        anomaly_mm: ``corrected - climatology``.
        anomaly_pct: The same as a percentage of climatology, or null when the
            climatological mean is effectively zero and the ratio would explode.
        month: Which month the climatology was taken from.
    """

    climatology_mm: float
    anomaly_mm: float
    anomaly_pct: Optional[float] = None
    month: int


class ThresholdProbability(BaseModel):
    """Probability of exceeding one rainfall threshold.

    Attributes:
        name: Threshold key, or ``"custom"`` for a user-supplied amount.
        threshold_mm: The amount being exceeded.
        probability: Calibrated probability of exceedance.
        interpolated: True when this is a custom threshold estimated between
            two trained ones rather than a model fitted at this exact amount.
    """

    name: str
    threshold_mm: float
    probability: float
    interpolated: bool = False


# ---------------------------------------------------------------------------
# Regime timeline / forecast time machine  (spec 4.3, 4.4)
# ---------------------------------------------------------------------------

class TimelineStep(BaseModel):
    """One step in the forecast evolution for a district.

    Attributes:
        date: The valid date of this step.
        offset_days: Days relative to the anchor date; negative is the past.
        label: Human label for the step, e.g. ``"-1 day"`` or ``"now"``.
        regime: Leading regime.
        regime_label: Display label, compound when two regimes are named.
        regime_confidence: Probability of the leading regime.
        regime_probability: Full distribution over the regimes.
        raw_forecast_mm: District-mean uncorrected rainfall.
        corrected_forecast_mm: District-mean corrected rainfall.
        observed_mm: District-mean observation where one exists.
        heavy_rain_probability: Calibrated probability per threshold.
        warning_level: Warning level at this step.
        regime_changed: True when the leading regime differs from the previous
            step -- the "regime transition detected" state.
    """

    date: _date
    offset_days: int
    label: str
    regime: str
    regime_label: str
    regime_confidence: float
    regime_probability: Dict[str, float] = Field(default_factory=dict)
    raw_forecast_mm: Optional[float] = None
    corrected_forecast_mm: Optional[float] = None
    observed_mm: Optional[float] = None
    heavy_rain_probability: Dict[str, float] = Field(default_factory=dict)
    warning_level: str = "none"
    regime_changed: bool = False


class TimelineResponse(BaseModel):
    """Forecast evolution across a window of days for one district.

    Attributes:
        district: District name.
        anchor_date: The date offsets are measured from.
        step_hours: Spacing between steps. The pipeline's accumulation window
            is 24 hours, so this is 24 -- sub-daily steps are not available
            from daily data and are not faked.
        steps: The timeline, earliest first.
        transitions: Human-readable descriptions of each regime change.
    """

    district: str
    anchor_date: _date
    step_hours: int = 24
    steps: List[TimelineStep] = Field(default_factory=list)
    transitions: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# District risk matrix and extreme-rain watch  (spec 4.5, 4.6)
# ---------------------------------------------------------------------------

class DistrictRisk(BaseModel):
    """One district's risk summary for a date.

    Attributes:
        district: District name.
        regime: Leading regime.
        regime_label: Display label for the regime.
        raw_forecast_mm: District-mean uncorrected rainfall.
        corrected_forecast_mm: District-mean corrected rainfall.
        peak_cell_mm: The wettest single grid cell in the district -- the value
            that drives the warning, which a district mean hides.
        observed_mm: District-mean observation where one exists.
        heavy_rain_probability: Calibrated probability per threshold.
        warning_level: ``none`` / ``watch`` / ``warning`` / ``severe``.
        n_grid_cells: How many cells the aggregate covers.
        centroid_lat: Mean latitude of the district's grid cells.
        centroid_lon: Mean longitude of the district's grid cells.
    """

    district: str
    regime: str
    regime_label: str
    raw_forecast_mm: Optional[float] = None
    corrected_forecast_mm: float
    peak_cell_mm: Optional[float] = None
    observed_mm: Optional[float] = None
    heavy_rain_probability: Dict[str, float] = Field(default_factory=dict)
    warning_level: str = "none"
    n_grid_cells: int = 0
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None


class RiskMatrixResponse(BaseModel):
    """Risk across every district the service can serve, for one date.

    Attributes:
        date: The valid date.
        thresholds_mm: The thresholds the probabilities refer to.
        districts: One row per district, ordered by descending risk.
        counts_by_warning: How many districts sit at each warning level.
        synthetic: True when serving the fabricated demo dataset.
    """

    date: _date
    thresholds_mm: Dict[str, float] = Field(default_factory=dict)
    districts: List[DistrictRisk] = Field(default_factory=list)
    counts_by_warning: Dict[str, int] = Field(default_factory=dict)
    synthetic: bool = False


class WatchResponse(BaseModel):
    """The highest-risk districts for a date, as an early-warning view.

    Attributes:
        date: The valid date.
        districts: Highest-risk first, filtered to those carrying real risk.
        n_screened: How many districts were examined to produce the list.
        counts_by_warning: How many districts sit at each warning level, over
            *every* district screened -- not just the ones listed. The listed
            rows are truncated by ``limit``, so counting them would understate
            the picture.
        quiet: True when nothing crossed the watch threshold -- an explicit
            "all clear" rather than an empty table the reader has to interpret.
        synthetic: True when serving the fabricated demo dataset.
    """

    date: _date
    districts: List[DistrictRisk] = Field(default_factory=list)
    n_screened: int = 0
    counts_by_warning: Dict[str, int] = Field(default_factory=dict)
    quiet: bool = True
    synthetic: bool = False


# ---------------------------------------------------------------------------
# Historical event replay  (spec 4.9)
# ---------------------------------------------------------------------------

class ReplayEvent(BaseModel):
    """One historical heavy-rainfall event available for replay.

    Attributes:
        date: The date of the event.
        district: The worst-affected district on that date.
        observed_mm: The district-mean observation.
        peak_observed_mm: The wettest single cell that day.
        raw_forecast_mm: What the uncorrected model forecast.
        corrected_forecast_mm: What the system forecast.
        regime_label: The regime the system detected.
        raw_error_mm: ``|raw - observed|``.
        corrected_error_mm: ``|corrected - observed|``.
        improved: True when the correction moved the forecast closer.
        in_training_period: True if this date fell inside the training split,
            in which case the system has seen it before and the replay is not
            evidence of anything. Reported rather than filtered, so the caller
            can decide.
    """

    date: _date
    district: str
    observed_mm: float
    peak_observed_mm: Optional[float] = None
    raw_forecast_mm: Optional[float] = None
    corrected_forecast_mm: Optional[float] = None
    regime_label: Optional[str] = None
    raw_error_mm: Optional[float] = None
    corrected_error_mm: Optional[float] = None
    improved: Optional[bool] = None
    in_training_period: bool = False


class ReplayListResponse(BaseModel):
    """The biggest observed rainfall events in the connected dataset.

    Attributes:
        events: Largest observed rainfall first.
        n_candidates: How many district-days were screened.
        test_period_start: First date of the held-out test split, when known.
            Events on or after it were never trained on.
        synthetic: True when serving the fabricated demo dataset.
    """

    events: List[ReplayEvent] = Field(default_factory=list)
    n_candidates: int = 0
    test_period_start: Optional[_date] = None
    synthetic: bool = False


# ---------------------------------------------------------------------------
# What-if scenario simulator  (spec 4.11)
# ---------------------------------------------------------------------------

class ScenarioAdjustment(BaseModel):
    """One knob that was turned in a what-if scenario.

    Attributes:
        control: The control name the caller used.
        columns: The feature columns it moved.
        delta: The change applied, in each column's own units.
        applied: False when none of the columns exist in the connected data,
            so the caller is told the knob did nothing rather than reading an
            unchanged result as "no effect".
    """

    control: str
    columns: List[str] = Field(default_factory=list)
    delta: float = 0.0
    applied: bool = True


class ScenarioResponse(BaseModel):
    """A hypothetical re-run of the pipeline under adjusted conditions.

    This is the model's response to altered inputs. It is **not** a forecast:
    nothing here says the atmosphere will do this, only what the model would
    have said if it had.

    Attributes:
        district: District name.
        date: The valid date the scenario is built on.
        adjustments: What was changed.
        baseline: The unmodified result.
        scenario: The result under the adjusted conditions.
        delta_corrected_mm: Change in corrected rainfall.
        delta_probability: Change in each threshold probability.
        regime_changed: True when the adjustment flipped the detected regime.
        disclaimer: Text the caller must display alongside the numbers.
    """

    district: str
    date: _date
    adjustments: List[ScenarioAdjustment] = Field(default_factory=list)
    baseline: Dict[str, Any] = Field(default_factory=dict)
    scenario: Dict[str, Any] = Field(default_factory=dict)
    delta_corrected_mm: float = 0.0
    delta_probability: Dict[str, float] = Field(default_factory=dict)
    regime_changed: bool = False
    disclaimer: str = (
        "A model scenario, not a forecast. This shows how the model responds to "
        "altered inputs; it says nothing about what the atmosphere will do."
    )


# ---------------------------------------------------------------------------
# Top drivers across the domain  (spec 4.12)
# ---------------------------------------------------------------------------

class DomainDriver(BaseModel):
    """One factor's average influence across the whole forecast area.

    Attributes:
        feature: Feature column name.
        mean_abs_contribution_mm: Mean absolute influence on the correction, in
            mm -- how much this factor moves the forecast on a typical cell.
        mean_signed_contribution_mm: Mean signed influence: positive means it
            is pushing rainfall up across the domain today.
        direction: ``"up"``, ``"down"`` or ``"mixed"``.
    """

    feature: str
    mean_abs_contribution_mm: float
    mean_signed_contribution_mm: float
    direction: str


class DriversResponse(BaseModel):
    """What is driving the corrections across the domain on one date.

    Attributes:
        date: The valid date.
        n_cells_sampled: How many grid cells the averages were taken over.
        sampled: True when a subset was used rather than every cell, because
            explaining a full grid is expensive.
        drivers: Strongest average influence first.
        dominant_regime: The most common regime across the sampled cells.
        regime_share: Fraction of sampled cells in each regime.
    """

    date: _date
    n_cells_sampled: int = 0
    sampled: bool = False
    drivers: List[DomainDriver] = Field(default_factory=list)
    dominant_regime: Optional[str] = None
    regime_share: Dict[str, float] = Field(default_factory=dict)
