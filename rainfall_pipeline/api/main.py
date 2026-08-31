"""Stage 7 -- the FastAPI service.

Start it with:

    uvicorn rainfall_pipeline.api.main:app --reload

The service is designed to come up cleanly in every state:

* no data and no models -- it starts, ``/health`` says what is missing, and
  ``/predict`` returns 503 with instructions rather than a stack trace;
* data but no models -- ``/districts`` works, ``/predict`` still returns 503;
* fully trained -- everything works.

All model artifacts are loaded once at startup into a module-level
:class:`ModelBundle`, never per request.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date as _date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..aggregation.district import (
    CORRECTED_COLUMN,
    aggregate_to_district,
    list_districts,
    probability_column,
)
from ..config.regions import ARTIFACT_DIR, DATA_DIR, DISTRICTS_PATH, STORE_DIR
from ..config.thresholds import (
    RAIN_THRESHOLDS,
    WARNING_LEVELS,
    WARNING_NONE,
    REGIME_BLEND_MIN_PROBABILITY,
    REGIME_LABELS,
)
from ..data import schema as sch
from ..data.loaders import MissingDataError, load_district_boundaries
from ..data.store import StoreEmptyError, read_table
from ..features.engineering import add_derived_features
from ..models.bias_correction import BIAS_MODEL_FILENAME, RegimeBiasCorrector
from ..models.heavy_rain_probability import (
    PROBABILITY_MODEL_FILENAME,
    HeavyRainProbabilityModel,
    attach_corrected_forecast,
)
from ..models.regime_classifier import REGIME_MODEL_FILENAME, RegimeClassifier
from ..models.uncertainty import INTERVAL_MODEL_FILENAME, PredictionIntervalModel
from ..training.common import ANALYSIS_TABLE, CLIMATOLOGY_FILENAME, read_manifest
from ..verification.report import load_report
from .schemas import (
    BiasContribution,
    DistrictRisk,
    ForecastInterval,
    BiasExplanationResponse,
    DateRangeResponse,
    DistrictListResponse,
    FeatureContribution,
    GridPanelStats,
    GridResponse,
    NotReadyResponse,
    PredictionResponse,
    RegimeBlendComponent,
    RainfallAnomaly,
    RegimeExplanationResponse,
    RiskMatrixResponse,
    ServiceStatus,
    DomainDriver,
    DriversResponse,
    ReplayEvent,
    ReplayListResponse,
    ScenarioAdjustment,
    ScenarioResponse,
    ThresholdProbability,
    TimelineResponse,
    TimelineStep,
    VerificationReportResponse,
    WatchResponse,
)

LOGGER = logging.getLogger("rainfall_pipeline.api")

#: The commands a user must run, in order, to make the service fully functional.
NEXT_STEPS: List[str] = [
    "Add your data files (see rainfall_pipeline/data/README.md).",
    "python -m rainfall_pipeline.training.run_full_training_pipeline",
    "Restart the API so it picks up the new artifacts.",
]


@dataclass
class ModelBundle:
    """Every artifact the service needs, loaded once at startup.

    Attributes:
        classifier: The regime classifier, or None.
        corrector: The regime-specific bias corrector, or None.
        probability: The calibrated probability head, or None.
        intervals: The prediction-interval model, or None. Optional -- the
            service forecasts without it, just without a range.
        climatology: Climatology table for the anomaly features, or None.
        districts: District polygons GeoDataFrame, or None.
        analysis_table: The cached analysis table, or None.
        manifest: The training manifest.
        load_errors: Non-fatal problems hit while loading, for ``/health``.
    """

    classifier: Optional[RegimeClassifier] = None
    corrector: Optional[RegimeBiasCorrector] = None
    probability: Optional[HeavyRainProbabilityModel] = None
    intervals: Optional[PredictionIntervalModel] = None
    climatology: Optional[pd.DataFrame] = None
    districts: Any = None
    analysis_table: Optional[pd.DataFrame] = None
    manifest: Dict[str, Any] = field(default_factory=dict)
    load_errors: List[str] = field(default_factory=list)

    @property
    def models_loaded(self) -> bool:
        """True when the minimum set of models for ``/predict`` is present."""
        return self.classifier is not None and self.corrector is not None

    @property
    def data_connected(self) -> bool:
        """True when there is an analysis table to serve from."""
        return self.analysis_table is not None and not self.analysis_table.empty

    def missing(self) -> List[str]:
        """List what is still needed for a full prediction.

        Returns:
            Human-readable descriptions of missing prerequisites.
        """
        out: List[str] = []
        if self.analysis_table is None or self.analysis_table.empty:
            out.append(
                "analysis data (no cached analysis table; add your data files and "
                "run the training pipeline)"
            )
        if self.classifier is None:
            out.append(f"regime classifier ({REGIME_MODEL_FILENAME})")
        if self.corrector is None:
            out.append(f"bias-correction models ({BIAS_MODEL_FILENAME})")
        if self.probability is None:
            out.append(f"heavy-rain probability models ({PROBABILITY_MODEL_FILENAME})")
        if self.districts is None:
            out.append(f"district boundaries ({DISTRICTS_PATH})")
        return out + self.load_errors


def load_bundle(artifact_dir: Optional[Path] = None) -> ModelBundle:
    """Load every available artifact, tolerating whatever is absent.

    Nothing here raises: a missing artifact is recorded, not fatal, so the
    service always starts and can explain itself.

    Args:
        artifact_dir: Where to look. Defaults to the configured directory.

    Returns:
        The populated bundle.
    """
    art = Path(artifact_dir or ARTIFACT_DIR)
    bundle = ModelBundle(manifest=read_manifest(art))

    def _try(label: str, fn):
        try:
            return fn()
        except FileNotFoundError:
            LOGGER.info("%s not found; the API will report it as missing.", label)
        except Exception as exc:  # noqa: BLE001 - startup must never crash
            LOGGER.warning("Failed to load %s: %s", label, exc)
            bundle.load_errors.append(f"{label} failed to load: {exc}")
        return None

    bundle.classifier = _try(
        "regime classifier", lambda: RegimeClassifier.load(art / REGIME_MODEL_FILENAME)
    )
    bundle.corrector = _try(
        "bias-correction models", lambda: RegimeBiasCorrector.load(art / BIAS_MODEL_FILENAME)
    )
    bundle.probability = _try(
        "heavy-rain probability models",
        lambda: HeavyRainProbabilityModel.load(art / PROBABILITY_MODEL_FILENAME),
    )
    # Intervals are an enhancement, not a prerequisite: an older artifact
    # directory predating them still serves point forecasts.
    bundle.intervals = _try(
        "prediction intervals",
        lambda: PredictionIntervalModel.load(art / INTERVAL_MODEL_FILENAME),
    )

    clim_path = art / CLIMATOLOGY_FILENAME
    if clim_path.exists():
        bundle.climatology = _try("climatology", lambda: pd.read_parquet(clim_path))

    bundle.districts = _try(
        "district boundaries", lambda: load_district_boundaries(DISTRICTS_PATH)
    )

    try:
        bundle.analysis_table = read_table(ANALYSIS_TABLE)
        LOGGER.info("Loaded analysis table: %d rows.", len(bundle.analysis_table))
    except (StoreEmptyError, MissingDataError) as exc:
        LOGGER.info("No cached analysis table: %s", str(exc).splitlines()[0])

    LOGGER.info(
        "Startup complete. models_loaded=%s data_connected=%s",
        bundle.models_loaded, bundle.data_connected,
    )
    return bundle


#: Populated during startup by the lifespan handler.
BUNDLE = ModelBundle()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once, before the first request is served."""
    global BUNDLE
    logging.basicConfig(level=logging.INFO)
    BUNDLE = load_bundle()
    yield


app = FastAPI(
    title="Regime-aware monsoon rainfall post-processing",
    description=(
        "Detects the prevailing monsoon regime, applies a regime-specific bias "
        "correction to a raw NWP rainfall forecast, estimates calibrated "
        "heavy-rain probability and aggregates to district level."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _not_ready(detail: str) -> JSONResponse:
    """Build the standard 503 response for an untrained/unconnected system.

    Args:
        detail: What went wrong, in one sentence.

    Returns:
        A 503 JSON response carrying the missing prerequisites and next steps.
    """
    payload = NotReadyResponse(
        detail=detail, missing=BUNDLE.missing(), next_steps=NEXT_STEPS
    )
    return JSONResponse(status_code=503, content=payload.model_dump())


@app.get("/health", response_model=ServiceStatus, tags=["meta"])
def health() -> ServiceStatus:
    """Report what the service has loaded and what it is still missing.

    Returns:
        The service's readiness status.
    """
    return ServiceStatus(
        models_loaded=BUNDLE.models_loaded,
        data_connected=BUNDLE.data_connected,
        districts_available=bool(_district_names()),
        missing=BUNDLE.missing(),
        artifact_dir=str(ARTIFACT_DIR),
    )


def _district_names() -> List[str]:
    """Return the district names available from either source."""
    return list_districts(BUNDLE.districts, BUNDLE.analysis_table)


@app.get("/districts", response_model=DistrictListResponse, tags=["reference"])
def districts() -> DistrictListResponse:
    """List the districts the service can serve predictions for.

    Names come from the district shapefile when one is configured, and
    otherwise from the ``district`` column of the connected data.

    Returns:
        The district list and where it came from.
    """
    names = _district_names()
    if BUNDLE.districts is not None:
        source = "shapefile"
    elif BUNDLE.analysis_table is not None and sch.DISTRICT_COLUMN in BUNDLE.analysis_table.columns:
        source = "data"
    else:
        source = "none"
    return DistrictListResponse(count=len(names), source=source, districts=names)


@app.get("/verification-report", response_model=VerificationReportResponse, tags=["verification"])
def verification_report() -> VerificationReportResponse:
    """Return the saved verification report.

    Returns:
        The report, or an explanation that none has been generated yet.
    """
    report = load_report()
    if report is None:
        return VerificationReportResponse(
            available=False,
            detail=(
                "No verification report has been generated. Connect your data and "
                "run: python -m rainfall_pipeline.training.run_full_training_pipeline"
            ),
        )
    return VerificationReportResponse(available=True, report=report)


#: Directories checked for the marker the synthetic generator leaves behind.
_SYNTHETIC_MARKER = "SYNTHETIC.marker"


def is_synthetic_dataset() -> bool:
    """Return True when the service is serving the fabricated demo dataset.

    The dashboard has to be able to stamp "Illustrative mock-up" across maps
    built from fake data. Detection is by marker file rather than by directory
    name so that renaming ``sample_data/`` cannot silently turn fake output
    into something that looks measured.

    Returns:
        True if a synthetic marker is present anywhere the service reads from.
    """
    candidates = [STORE_DIR, STORE_DIR.parent, ARTIFACT_DIR, DATA_DIR]
    return any((d / _SYNTHETIC_MARKER).exists() for d in candidates)


def blend_regimes(
    probabilities: pd.Series,
    *,
    min_probability: float = REGIME_BLEND_MIN_PROBABILITY,
) -> tuple[str, List[RegimeBlendComponent]]:
    """Turn a regime distribution into a display label and its components.

    A single argmax label misrepresents the common case: a Konkan cell in an
    active spell is genuinely both Coastal and Active, and calling it one of
    those hides half the story. When the runner-up carries at least
    ``min_probability`` the label names both.

    Args:
        probabilities: Regime probabilities, indexed by regime label.
        min_probability: Probability the second regime must reach to be named.

    Returns:
        ``(label, components)`` where ``label`` is e.g. ``"Coastal + Active"``
        and ``components`` lists the named regimes, largest first.
    """
    ordered = probabilities.sort_values(ascending=False)
    if ordered.empty:
        return "unknown", []

    chosen = [(str(ordered.index[0]), float(ordered.iloc[0]))]
    if len(ordered) > 1 and float(ordered.iloc[1]) >= min_probability:
        chosen.append((str(ordered.index[1]), float(ordered.iloc[1])))

    label = " + ".join(name for name, _ in chosen)
    return label, [
        RegimeBlendComponent(regime=name, probability=prob) for name, prob in chosen
    ]


def run_grid_forecast(
    rows: pd.DataFrame,
    *,
    soft_routing: bool = True,
) -> Dict[str, Any]:
    """Run the whole pipeline over a set of grid rows.

    Every view in the service -- single district, timeline, risk matrix,
    watchlist, scenario -- needs the same chain: features, regime, correction,
    interval, probabilities. Sharing one implementation is what keeps the risk
    matrix and ``/predict`` from quietly disagreeing about the same district.

    Args:
        rows: Raw analysis-table rows for one or more dates.
        soft_routing: Blend correctors by regime probability rather than
            routing to the single most likely regime.

    Returns:
        ``{"features", "regime_probs", "cell_regimes", "corrected", "interval",
        "grid", "threshold_names", "routing"}``. ``grid`` is the frame the
        district aggregation consumes.

    Raises:
        RuntimeError: If the classifier or corrector is not loaded.
    """
    if BUNDLE.classifier is None or BUNDLE.corrector is None:
        raise RuntimeError("Models are not loaded.")

    features = add_derived_features(rows.reset_index(drop=True), climatology=BUNDLE.climatology)
    regime_probs = BUNDLE.classifier.predict_proba(features)
    cell_regimes = regime_probs.idxmax(axis=1)

    routing = "soft" if soft_routing else "hard"
    if soft_routing:
        try:
            corrected = BUNDLE.corrector.predict_soft(features, regime_probs)
        except Exception as exc:  # noqa: BLE001 - never fail a request on this
            LOGGER.warning("Soft routing failed, falling back to hard: %s", exc)
            corrected = BUNDLE.corrector.predict(features, cell_regimes)
            routing = "hard"
    else:
        corrected = BUNDLE.corrector.predict(features, cell_regimes)

    interval = None
    if BUNDLE.intervals is not None:
        try:
            interval = BUNDLE.intervals.predict_interval(features, cell_regimes)
        except Exception as exc:  # noqa: BLE001 - the range is an enhancement
            LOGGER.warning("Prediction interval failed: %s", exc)

    grid = features.copy()
    grid[CORRECTED_COLUMN] = corrected.values

    threshold_names: List[str] = []
    if BUNDLE.probability is not None:
        probs = BUNDLE.probability.predict_proba(attach_corrected_forecast(features, corrected))
        for name in probs.columns:
            grid[probability_column(str(name))] = probs[name].values
            threshold_names.append(str(name))

    return {
        "features": features,
        "regime_probs": regime_probs,
        "cell_regimes": cell_regimes,
        "corrected": corrected,
        "interval": interval,
        "grid": grid,
        "threshold_names": threshold_names,
        "routing": routing,
    }


def interval_for(result: Dict[str, Any], index: Any = None) -> Optional[ForecastInterval]:
    """Build the response interval, carrying its measured coverage.

    The nominal band width is a property of the model; whether observations
    actually land inside it was measured on held-out data at training time and
    recorded in the manifest. Both are returned, because only the second one
    tells the reader whether to believe the range.

    Args:
        result: Output of :func:`run_grid_forecast`.
        index: Rows to average the band over. Defaults to every row.

    Returns:
        A :class:`ForecastInterval`, or None when no interval model is loaded.
    """
    band = result.get("interval")
    if band is None or BUNDLE.intervals is None:
        return None
    subset = band if index is None else band.loc[index]
    if subset.empty:
        return None

    meta = (BUNDLE.manifest or {}).get("prediction_intervals") or {}
    nominal = meta.get("nominal_coverage", BUNDLE.intervals.nominal_coverage)
    measured = (meta.get("test_coverage") or {}).get("coverage")
    calibrated = (
        abs(float(measured) - float(nominal)) <= 0.1
        if isinstance(measured, (int, float)) and isinstance(nominal, (int, float))
        else None
    )
    return ForecastInterval(
        low_mm=float(subset["corrected_low"].mean()),
        high_mm=float(subset["corrected_high"].mean()),
        nominal_coverage=float(nominal) if nominal is not None else None,
        measured_coverage=float(measured) if isinstance(measured, (int, float)) else None,
        calibrated=calibrated,
    )


def anomaly_for(features: pd.DataFrame, corrected_mm: float) -> Optional[RainfallAnomaly]:
    """Compare a corrected forecast against the local climatology.

    Args:
        features: The district's feature rows, carrying ``rain_clim_mean`` when
            a climatology was attached.
        corrected_mm: The district-mean corrected forecast.

    Returns:
        A :class:`RainfallAnomaly`, or None when no climatology is available.
    """
    # The feature pipeline drops the climatology columns once it has turned
    # them into anomalies, so the join is redone here rather than widening the
    # model's feature table for a presentational number.
    if BUNDLE.climatology is None or BUNDLE.climatology.empty:
        return None
    if not {"lat", "lon"}.issubset(features.columns):
        return None

    months = pd.to_datetime(features["date"]).dt.month
    month = int(months.mode().iloc[0])

    clim = BUNDLE.climatology.copy()
    clim["month"] = pd.to_numeric(clim["month"], errors="coerce")
    joined = features[["lat", "lon"]].assign(month=months.astype("float64")).merge(
        clim[["lat", "lon", "month", "rain_clim_mean"]].assign(
            month=lambda f: f["month"].astype("float64")
        ),
        on=["lat", "lon", "month"],
        how="left",
    )
    values = pd.to_numeric(joined["rain_clim_mean"], errors="coerce").dropna()
    if values.empty:
        return None

    baseline = float(values.mean())
    anomaly = corrected_mm - baseline
    # A ratio against a near-zero climatology explodes to a meaningless number,
    # so it is withheld rather than reported as "+40,000%".
    pct = (anomaly / baseline) * 100.0 if baseline >= 0.1 else None
    return RainfallAnomaly(
        climatology_mm=baseline,
        anomaly_mm=anomaly,
        anomaly_pct=pct,
        month=month,
    )


def threshold_probabilities_for(
    record: pd.Series,
    threshold_names: List[str],
    aggregated_columns: Any,
    custom_threshold: Optional[float] = None,
) -> List[ThresholdProbability]:
    """Assemble the per-threshold probability list, including a custom one.

    A custom threshold is estimated by interpolating between the trained ones
    on a log-rainfall axis, and is flagged as interpolated. Fitting a fresh
    classifier per request is not an option, and presenting an interpolation as
    a trained probability would be a lie of omission.

    Args:
        record: The aggregated district row.
        threshold_names: Trained threshold keys.
        aggregated_columns: Columns present on the aggregate.
        custom_threshold: Optional user-supplied amount in mm.

    Returns:
        One entry per threshold, trained ones first.
    """
    out: List[ThresholdProbability] = []
    points: List[tuple] = []
    for name in threshold_names:
        column = f"max_{probability_column(name)}"
        if column not in aggregated_columns:
            continue
        mm = float(RAIN_THRESHOLDS.get(name, float("nan")))
        prob = float(record[column])
        out.append(ThresholdProbability(name=name, threshold_mm=mm, probability=prob))
        if mm == mm:  # not NaN
            points.append((mm, prob))

    if custom_threshold is None or not points:
        return out

    points.sort()
    xs = np.log1p([p[0] for p in points])
    ys = [p[1] for p in points]
    target = float(np.log1p(custom_threshold))
    # np.interp clamps outside the fitted range, which is the honest behaviour
    # here: below the lowest trained threshold the true probability is higher
    # than any we modelled, and we must not invent it.
    estimate = float(np.interp(target, xs, ys))
    exact = next((p for p in points if abs(p[0] - custom_threshold) < 1e-6), None)
    out.append(
        ThresholdProbability(
            name="custom",
            threshold_mm=float(custom_threshold),
            probability=exact[1] if exact else estimate,
            interpolated=exact is None,
        )
    )
    return out


def _rows_for(date_value: _date, district: str) -> pd.DataFrame:
    """Fetch the grid rows for one district on one date.

    Args:
        date_value: The valid date.
        district: District name.

    Returns:
        The matching rows, with a ``district`` column guaranteed present.

    Raises:
        HTTPException: 404 if no rows match, 503 if districts cannot be resolved.
    """
    table = BUNDLE.analysis_table
    assert table is not None  # guarded by the caller

    rows = table[table["date"] == pd.Timestamp(date_value)]
    if rows.empty:
        available = table["date"]
        raise HTTPException(
            status_code=404,
            detail=(
                f"No data for {date_value}. The connected dataset covers "
                f"{available.min().date()} to {available.max().date()}."
            ),
        )

    if sch.DISTRICT_COLUMN not in rows.columns or rows[sch.DISTRICT_COLUMN].isna().all():
        if BUNDLE.districts is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The connected data has no 'district' column and no district "
                    "shapefile is configured, so grid cells cannot be attributed "
                    f"to a district. Add a shapefile at {DISTRICTS_PATH} or set "
                    "RAINFALL_DISTRICTS_PATH."
                ),
            )
        from ..aggregation.district import assign_districts

        rows = pd.DataFrame(assign_districts(rows, BUNDLE.districts).drop(columns=["geometry"]))

    matched = rows[rows[sch.DISTRICT_COLUMN].astype("string").str.casefold() == district.casefold()]
    if matched.empty:
        known = sorted({str(v) for v in rows[sch.DISTRICT_COLUMN].dropna().unique()})[:20]
        raise HTTPException(
            status_code=404,
            detail=(
                f"No grid cells for district '{district}' on {date_value}. "
                f"Districts with data on that date include: {', '.join(known) or 'none'}. "
                f"Call /districts for the full list."
            ),
        )
    return matched.reset_index(drop=True)


@app.get(
    "/predict",
    response_model=PredictionResponse,
    responses={503: {"model": NotReadyResponse}},
    tags=["prediction"],
)
def predict(
    date: _date = Query(..., description="Valid date of the forecast (YYYY-MM-DD)."),
    district: str = Query(..., description="District name; see /districts."),
    explain: bool = Query(True, description="Include the SHAP explanations."),
    soft_routing: bool = Query(
        True,
        description=(
            "Blend every regime's corrector by probability. Set false to route "
            "to the single most likely regime instead."
        ),
    ),
    threshold: Optional[float] = Query(
        None,
        gt=0,
        le=2000,
        description=(
            "Optional custom rainfall threshold in mm. Its probability is "
            "interpolated between the trained thresholds and flagged as such."
        ),
    ),
):
    """Run the full pipeline for one district on one date.

    Feature lookup -> regime -> regime-specific correction -> calibrated
    heavy-rain probability -> district aggregation.

    Args:
        date: The valid date.
        district: District name.
        explain: Whether to compute the SHAP explanation, which costs extra time.

    Returns:
        A :class:`PredictionResponse`, or a 503 ``NotReadyResponse`` if the
        system has not been trained yet.

    Raises:
        HTTPException: 404 if the date or district has no data.
    """
    if not BUNDLE.data_connected:
        return _not_ready(
            "No data is connected yet, so no prediction can be made. "
            "Add your data files and run the training pipeline."
        )
    if not BUNDLE.models_loaded:
        return _not_ready(
            "The models have not been trained yet, so no prediction can be made."
        )

    rows = _rows_for(date, district)
    result = run_grid_forecast(rows, soft_routing=soft_routing)
    features = result["features"]
    regime_probs = result["regime_probs"]
    cell_regimes = result["cell_regimes"]
    corrected = result["corrected"]
    routing = result["routing"]

    # --- regime ---------------------------------------------------------
    # One regime for the whole district: average the per-cell distributions,
    # which is more stable than taking the modal cell.
    district_probs = regime_probs.mean(axis=0)
    regime = str(district_probs.idxmax())
    regime_label, regime_blend = blend_regimes(district_probs)

    explanation: Optional[RegimeExplanationResponse] = None
    if explain:
        try:
            detail = BUNDLE.classifier.explain_row(features)
            explanation = RegimeExplanationResponse(
                top_features=[
                    FeatureContribution(feature=f, shap_value=v) for f, v in detail.top_features
                ],
                base_value=detail.base_value,
            )
        except Exception as exc:  # noqa: BLE001 - explanation is best-effort
            LOGGER.warning("SHAP explanation failed: %s", exc)

    bias_explanation: Optional[BiasExplanationResponse] = None
    if explain:
        try:
            # Explain the wettest cell: on a district of many cells that is the
            # one driving the warning, and so the one the user is asking about.
            order = pd.to_numeric(corrected, errors="coerce").fillna(-1.0)
            wettest = features.loc[[order.idxmax()]]
            detail = BUNDLE.corrector.explain_row(wettest, str(cell_regimes.loc[order.idxmax()]))
            bias_explanation = BiasExplanationResponse(
                regime=detail.regime,
                raw_mm=detail.raw_mm,
                predicted_bias_mm=detail.predicted_bias_mm,
                corrected_mm=detail.corrected_mm,
                top_features=[
                    BiasContribution(feature=f, shap_value=v) for f, v in detail.top_features
                ],
                base_value=detail.base_value,
            )
        except Exception as exc:  # noqa: BLE001 - explanation is best-effort
            LOGGER.warning("Bias SHAP explanation failed: %s", exc)

    grid = result["grid"]
    threshold_names = result["threshold_names"]

    # --- district aggregation ---------------------------------------------
    aggregated = aggregate_to_district(
        grid, BUNDLE.districts, threshold_names=threshold_names or None
    )
    record = aggregated.iloc[0]

    heavy_rain_probability = {
        name: float(record[f"max_{probability_column(name)}"])
        for name in threshold_names
        if f"max_{probability_column(name)}" in aggregated.columns
    }

    return PredictionResponse(
        date=date,
        district=str(record[sch.DISTRICT_COLUMN]),
        regime=regime,
        regime_label=regime_label,
        regime_blend=regime_blend,
        regime_probability={name: float(district_probs.get(name, 0.0)) for name in REGIME_LABELS},
        routing=routing,
        raw_forecast_mm=float(record["mean_raw_mm"]) if "mean_raw_mm" in aggregated.columns else None,
        corrected_forecast_mm=float(record["mean_corrected_mm"]),
        heavy_rain_probability=heavy_rain_probability,
        warning_level=str(record["warning_level"]),
        n_grid_cells=int(record["n_grid_cells"]),
        explanation=explanation,
        bias_explanation=bias_explanation,
        observed_mm=(
            float(record["mean_observed_mm"])
            if "mean_observed_mm" in aggregated.columns and pd.notna(record["mean_observed_mm"])
            else None
        ),
        peak_cell_mm=float(pd.to_numeric(corrected, errors="coerce").max()),
        # Mean position of the district's grid cells -- a real coordinate for
        # the panel, not a lookup from a hard-coded gazetteer.
        centroid_lat=(
            float(pd.to_numeric(rows["lat"], errors="coerce").mean())
            if "lat" in rows.columns else None
        ),
        centroid_lon=(
            float(pd.to_numeric(rows["lon"], errors="coerce").mean())
            if "lon" in rows.columns else None
        ),
        regime_confidence=float(district_probs.max()),
        interval=interval_for(result),
        anomaly=anomaly_for(features, float(record["mean_corrected_mm"])),
        threshold_probabilities=threshold_probabilities_for(
            record, threshold_names, aggregated.columns, threshold
        ),
    )


def _district_risk_rows(
    date_value: _date,
    *,
    soft_routing: bool = True,
) -> tuple[List[DistrictRisk], List[str]]:
    """Score every district on one date in a single pass.

    Looping ``/predict`` over 700 districts would re-run the classifier 700
    times over overlapping data. The grid is scored once and aggregated to all
    districts together instead.

    Args:
        date_value: The valid date.
        soft_routing: Blend correctors by regime probability.

    Returns:
        ``(rows, threshold_names)`` with rows ordered most severe first.

    Raises:
        HTTPException: 404 if the date has no data.
    """
    table = BUNDLE.analysis_table
    assert table is not None
    rows = table[table["date"] == pd.Timestamp(date_value)]
    if rows.empty:
        available = table["date"]
        raise HTTPException(
            status_code=404,
            detail=(
                f"No data for {date_value}. The connected dataset covers "
                f"{available.min().date()} to {available.max().date()}."
            ),
        )

    result = run_grid_forecast(rows, soft_routing=soft_routing)
    grid = result["grid"]
    features = result["features"]
    threshold_names = result["threshold_names"]

    # Per-cell regime, so each district can report the regime that dominates it.
    grid = grid.copy()
    grid["__regime"] = result["cell_regimes"].values

    aggregated = aggregate_to_district(
        grid, BUNDLE.districts, threshold_names=threshold_names or None
    )
    if sch.DISTRICT_COLUMN not in grid.columns:
        from ..aggregation.district import assign_districts

        grid = pd.DataFrame(assign_districts(grid, BUNDLE.districts).drop(columns=["geometry"]))

    peak_by_district = grid.groupby(sch.DISTRICT_COLUMN)[CORRECTED_COLUMN].max()
    # Mean cell position per district: a real coordinate for the table and the
    # CSV export, rather than a lookup from a hard-coded gazetteer.
    lat_by_district = (
        grid.groupby(sch.DISTRICT_COLUMN)["lat"].mean() if "lat" in grid.columns else None
    )
    lon_by_district = (
        grid.groupby(sch.DISTRICT_COLUMN)["lon"].mean() if "lon" in grid.columns else None
    )
    regime_by_district = (
        grid.groupby(sch.DISTRICT_COLUMN)["__regime"]
        .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else "")
    )
    probs_by_district = {
        name: grid.groupby(sch.DISTRICT_COLUMN)[probability_column(name)].mean()
        for name in threshold_names
        if probability_column(name) in grid.columns
    }

    out: List[DistrictRisk] = []
    for _, record in aggregated.iterrows():
        name = str(record[sch.DISTRICT_COLUMN])
        regime = str(regime_by_district.get(name, ""))
        # A district-level blend needs the district's own distribution, which
        # the per-cell modal label approximates well enough for a table.
        probabilities = {
            key: float(record[f"max_{probability_column(key)}"])
            for key in threshold_names
            if f"max_{probability_column(key)}" in aggregated.columns
        }
        out.append(
            DistrictRisk(
                district=name,
                regime=regime,
                regime_label=regime,
                raw_forecast_mm=(
                    float(record["mean_raw_mm"]) if "mean_raw_mm" in aggregated.columns else None
                ),
                corrected_forecast_mm=float(record["mean_corrected_mm"]),
                peak_cell_mm=float(peak_by_district.get(name, float("nan")))
                if name in peak_by_district.index
                else None,
                observed_mm=(
                    float(record["mean_observed_mm"])
                    if "mean_observed_mm" in aggregated.columns
                    and pd.notna(record["mean_observed_mm"])
                    else None
                ),
                heavy_rain_probability=probabilities,
                warning_level=str(record["warning_level"]),
                n_grid_cells=int(record["n_grid_cells"]),
                centroid_lat=(
                    float(lat_by_district[name])
                    if lat_by_district is not None and name in lat_by_district.index
                    else None
                ),
                centroid_lon=(
                    float(lon_by_district[name])
                    if lon_by_district is not None and name in lon_by_district.index
                    else None
                ),
            )
        )

    severity = {name: i for i, name in enumerate(WARNING_LEVELS)}
    out.sort(
        key=lambda r: (
            severity.get(r.warning_level, 0),
            max(r.heavy_rain_probability.values(), default=0.0),
            r.corrected_forecast_mm,
        ),
        reverse=True,
    )
    return out, threshold_names


@app.get(
    "/risk-matrix",
    response_model=RiskMatrixResponse,
    responses={503: {"model": NotReadyResponse}},
    tags=["prediction"],
)
def risk_matrix(
    date: _date = Query(..., description="Valid date of the forecast (YYYY-MM-DD)."),
    soft_routing: bool = Query(True, description="Blend correctors by regime probability."),
):
    """Score every district on one date, ordered most severe first.

    Args:
        date: The valid date.
        soft_routing: Whether to blend correctors by regime probability.

    Returns:
        A :class:`RiskMatrixResponse`, or 503 when untrained.
    """
    if not BUNDLE.data_connected:
        return _not_ready("No data is connected yet, so no risk matrix can be built.")
    if not BUNDLE.models_loaded:
        return _not_ready("The models have not been trained yet.")

    rows, threshold_names = _district_risk_rows(date, soft_routing=soft_routing)
    counts: Dict[str, int] = {level: 0 for level in WARNING_LEVELS}
    for row in rows:
        counts[row.warning_level] = counts.get(row.warning_level, 0) + 1

    return RiskMatrixResponse(
        date=date,
        thresholds_mm={n: float(RAIN_THRESHOLDS[n]) for n in threshold_names if n in RAIN_THRESHOLDS},
        districts=rows,
        counts_by_warning=counts,
        synthetic=is_synthetic_dataset(),
    )


@app.get(
    "/watch",
    response_model=WatchResponse,
    responses={503: {"model": NotReadyResponse}},
    tags=["prediction"],
)
def watch(
    date: _date = Query(..., description="Valid date of the forecast (YYYY-MM-DD)."),
    limit: int = Query(15, ge=1, le=200, description="Maximum districts to return."),
    min_probability: float = Query(
        0.05,
        ge=0.0,
        le=1.0,
        description="Smallest heavy-rain probability worth listing.",
    ),
):
    """The highest-risk districts for a date, as an early-warning view.

    Args:
        date: The valid date.
        limit: Maximum districts to return.
        min_probability: Districts below this carry no listed risk.

    Returns:
        A :class:`WatchResponse`. When nothing crosses the threshold it says so
        explicitly rather than returning a bare empty list.
    """
    if not BUNDLE.data_connected:
        return _not_ready("No data is connected yet, so no watchlist can be built.")
    if not BUNDLE.models_loaded:
        return _not_ready("The models have not been trained yet.")

    rows, _ = _district_risk_rows(date)
    flagged = [
        row
        for row in rows
        if row.warning_level != WARNING_NONE
        or max(row.heavy_rain_probability.values(), default=0.0) >= min_probability
    ]
    # Counted over every district screened, not the truncated list, so the
    # summary still says "45 clear" when only 15 rows are shown.
    counts: Dict[str, int] = {level: 0 for level in WARNING_LEVELS}
    for row in rows:
        counts[row.warning_level] = counts.get(row.warning_level, 0) + 1

    return WatchResponse(
        date=date,
        districts=flagged[:limit],
        n_screened=len(rows),
        counts_by_warning=counts,
        quiet=not flagged,
        synthetic=is_synthetic_dataset(),
    )


@app.get("/dates", response_model=DateRangeResponse, tags=["reference"])
def dates() -> DateRangeResponse:
    """Report the date range the connected dataset can serve.

    Returns:
        A :class:`DateRangeResponse`. ``available`` is False when no data is
        connected, rather than raising.
    """
    table = BUNDLE.analysis_table
    if table is None or table.empty or "date" not in table.columns:
        return DateRangeResponse(available=False, synthetic=is_synthetic_dataset())
    values = pd.to_datetime(table["date"], errors="coerce").dropna()
    if values.empty:
        return DateRangeResponse(available=False, synthetic=is_synthetic_dataset())
    return DateRangeResponse(
        available=True,
        start=values.min().date(),
        end=values.max().date(),
        n_dates=int(values.nunique()),
        synthetic=is_synthetic_dataset(),
    )


@app.get(
    "/grid",
    response_model=GridResponse,
    responses={503: {"model": NotReadyResponse}},
    tags=["prediction"],
)
def grid(
    date: _date = Query(..., description="Valid date of the forecast (YYYY-MM-DD)."),
    max_cells: int = Query(20000, ge=1, le=200000, description="Safety cap on cells returned."),
    soft_routing: bool = Query(True, description="Blend correctors by regime probability."),
):
    """Return raw, corrected, difference and observed fields for one date.

    This backs the three-panel map comparison. Every panel is built from the
    same cells on the same date and returned together, so the dashboard can
    draw them on one shared legend -- panels drawn on independent scales make
    any correction look dramatic and mean nothing.

    Args:
        date: The valid date.
        max_cells: Refuse rather than serialise an unbounded grid.
        soft_routing: Whether to blend correctors by regime probability.

    Returns:
        A :class:`GridResponse`, or a 503 ``NotReadyResponse`` when untrained.

    Raises:
        HTTPException: 404 if the date has no data, 413 if the grid is too big.
    """
    if not BUNDLE.data_connected:
        return _not_ready("No data is connected yet, so no grid can be built.")
    if not BUNDLE.models_loaded:
        return _not_ready("The models have not been trained yet, so no grid can be built.")

    table = BUNDLE.analysis_table
    assert table is not None
    rows = table[table["date"] == pd.Timestamp(date)]
    if rows.empty:
        available = table["date"]
        raise HTTPException(
            status_code=404,
            detail=(
                f"No data for {date}. The connected dataset covers "
                f"{available.min().date()} to {available.max().date()}."
            ),
        )
    if len(rows) > max_cells:
        raise HTTPException(
            status_code=413,
            detail=(
                f"{len(rows):,} grid cells on {date} exceeds max_cells={max_cells:,}. "
                f"Raise max_cells or narrow the domain."
            ),
        )

    features = add_derived_features(rows.reset_index(drop=True), climatology=BUNDLE.climatology)
    regime_probs = BUNDLE.classifier.predict_proba(features)
    cell_regimes = regime_probs.idxmax(axis=1)

    if soft_routing:
        try:
            corrected = BUNDLE.corrector.predict_soft(features, regime_probs)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Soft routing failed on grid, falling back to hard: %s", exc)
            corrected = BUNDLE.corrector.predict(features, cell_regimes)
    else:
        corrected = BUNDLE.corrector.predict(features, cell_regimes)

    raw = pd.to_numeric(features[sch.FORECAST_COLUMN], errors="coerce")
    observed = (
        pd.to_numeric(features[sch.OBSERVED_COLUMN], errors="coerce")
        if sch.OBSERVED_COLUMN in features.columns
        else pd.Series(float("nan"), index=features.index)
    )

    # Lay the cells out on a dense lat/lon lattice so the client can index
    # row-major without needing to know anything about the grid.
    lats = sorted(features["lat"].dropna().unique().tolist())
    lons = sorted(features["lon"].dropna().unique().tolist())
    lat_pos = {v: i for i, v in enumerate(lats)}
    lon_pos = {v: i for i, v in enumerate(lons)}
    n_cells = len(lats) * len(lons)

    fields = {
        "raw": raw,
        "corrected": corrected,
        "difference": corrected - raw,
        "observed": observed,
    }
    panels: Dict[str, List[Optional[float]]] = {
        name: [None] * n_cells for name in fields
    }
    regimes: List[Optional[str]] = [None] * n_cells

    flat_index = [
        lat_pos[lat] * len(lons) + lon_pos[lon]
        for lat, lon in zip(features["lat"], features["lon"])
    ]
    for name, series in fields.items():
        target = panels[name]
        for slot, value in zip(flat_index, series):
            target[slot] = None if pd.isna(value) else float(value)
    for slot, value in zip(flat_index, cell_regimes):
        regimes[slot] = None if pd.isna(value) else str(value)

    labels = {
        "raw": "Raw NWP forecast",
        "corrected": "AI-corrected forecast",
        "difference": "Correction applied (corrected - raw)",
        "observed": "Observed rainfall",
    }
    stats = []
    for name, series in fields.items():
        finite = series[series.notna()]
        stats.append(
            GridPanelStats(
                name=name,
                label=labels[name],
                min_value=float(finite.min()) if not finite.empty else None,
                max_value=float(finite.max()) if not finite.empty else None,
                n_finite=int(finite.shape[0]),
            )
        )

    return GridResponse(
        date=date,
        n_cells=n_cells,
        lats=[float(v) for v in lats],
        lons=[float(v) for v in lons],
        bbox=[float(min(lats)), float(max(lats)), float(min(lons)), float(max(lons))]
        if lats and lons
        else [],
        regimes=regimes,
        panels=panels,
        panel_stats=stats,
        observed_available=bool(observed.notna().any()),
        synthetic=is_synthetic_dataset(),
    )


@app.get(
    "/timeline",
    response_model=TimelineResponse,
    responses={503: {"model": NotReadyResponse}},
    tags=["prediction"],
)
def timeline(
    date: _date = Query(..., description="Anchor date (YYYY-MM-DD)."),
    district: str = Query(..., description="District name; see /districts."),
    back: int = Query(1, ge=0, le=7, description="Days of history to include."),
    forward: int = Query(2, ge=0, le=7, description="Days ahead to include."),
    soft_routing: bool = Query(True, description="Blend correctors by regime probability."),
):
    """Regime and rainfall evolution around a date, for one district.

    This backs both the regime-transition timeline and the forecast time
    machine. Steps are whole days because the pipeline's accumulation window is
    24 hours: a 6-hourly slider over daily data would be fabricated detail, so
    the response states its own step size rather than implying finer.

    Args:
        date: Anchor date; offsets are measured from it.
        district: District name.
        back: Days of history to include.
        forward: Days ahead to include.
        soft_routing: Whether to blend correctors by regime probability.

    Returns:
        A :class:`TimelineResponse` covering whichever of the requested days
        exist in the connected data.

    Raises:
        HTTPException: 404 if no day in the window has data for this district.
    """
    if not BUNDLE.data_connected:
        return _not_ready("No data is connected yet, so no timeline can be built.")
    if not BUNDLE.models_loaded:
        return _not_ready("The models have not been trained yet.")

    steps: List[TimelineStep] = []
    previous_regime: Optional[str] = None
    transitions: List[str] = []

    for offset in range(-back, forward + 1):
        step_date = date + timedelta(days=offset)
        try:
            rows = _rows_for(step_date, district)
        except HTTPException:
            # A gap at the edge of the dataset is normal; skip it rather than
            # failing the whole window.
            continue

        result = run_grid_forecast(rows, soft_routing=soft_routing)
        district_probs = result["regime_probs"].mean(axis=0)
        regime = str(district_probs.idxmax())
        label, _ = blend_regimes(district_probs)

        aggregated = aggregate_to_district(
            result["grid"], BUNDLE.districts,
            threshold_names=result["threshold_names"] or None,
        )
        record = aggregated.iloc[0]
        probabilities = {
            name: float(record[f"max_{probability_column(name)}"])
            for name in result["threshold_names"]
            if f"max_{probability_column(name)}" in aggregated.columns
        }

        # A transition is judged on the label the user is shown. The leading
        # regime can hold steady while a second one appears or falls away --
        # "Orographic" becoming "Orographic + Coastal" is a real change in the
        # weather, and comparing argmax alone would miss it entirely.
        changed = previous_regime is not None and label != previous_regime
        if changed:
            transitions.append(
                f"{previous_regime} to {label} on {step_date.isoformat()}"
            )

        steps.append(
            TimelineStep(
                date=step_date,
                offset_days=offset,
                label="now" if offset == 0 else f"{offset:+d} day{'s' if abs(offset) != 1 else ''}",
                regime=regime,
                regime_label=label,
                regime_confidence=float(district_probs.max()),
                regime_probability={n: float(district_probs.get(n, 0.0)) for n in REGIME_LABELS},
                raw_forecast_mm=(
                    float(record["mean_raw_mm"]) if "mean_raw_mm" in aggregated.columns else None
                ),
                corrected_forecast_mm=float(record["mean_corrected_mm"]),
                observed_mm=(
                    float(record["mean_observed_mm"])
                    if "mean_observed_mm" in aggregated.columns
                    and pd.notna(record["mean_observed_mm"])
                    else None
                ),
                heavy_rain_probability=probabilities,
                warning_level=str(record["warning_level"]),
                regime_changed=changed,
            )
        )
        previous_regime = label

    if not steps:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No data for '{district}' anywhere between "
                f"{(date - timedelta(days=back)).isoformat()} and "
                f"{(date + timedelta(days=forward)).isoformat()}."
            ),
        )

    return TimelineResponse(
        district=str(district),
        anchor_date=date,
        step_hours=24,
        steps=steps,
        transitions=transitions,
    )


#: Scenario controls the caller may turn, and the feature columns each moves.
#: Deliberately small: every knob here is something a forecaster could describe
#: in words, and nothing exposes a raw model feature as a user input.
SCENARIO_CONTROLS: Dict[str, Dict[str, Any]] = {
    "humidity": {"columns": ["humidity"], "unit": "percent points"},
    "wind": {"columns": ["wind_u_850", "wind_v_850"], "unit": "m/s"},
    "pressure": {"columns": ["pressure_msl"], "unit": "hPa"},
    "instability": {"columns": ["cape"], "unit": "J/kg"},
}


@app.get(
    "/what-if",
    response_model=ScenarioResponse,
    responses={503: {"model": NotReadyResponse}},
    tags=["prediction"],
)
def what_if(
    date: _date = Query(..., description="Valid date of the forecast (YYYY-MM-DD)."),
    district: str = Query(..., description="District name; see /districts."),
    humidity: float = Query(0.0, ge=-50, le=50, description="Change in humidity."),
    wind: float = Query(0.0, ge=-20, le=20, description="Change in 850 hPa wind, m/s."),
    pressure: float = Query(0.0, ge=-30, le=30, description="Change in sea-level pressure, hPa."),
    instability: float = Query(0.0, ge=-3000, le=3000, description="Change in CAPE, J/kg."),
    soft_routing: bool = Query(True, description="Blend correctors by regime probability."),
):
    """Re-run the pipeline with atmospheric inputs nudged.

    The derived features are rebuilt from the adjusted columns, so raising the
    wind also moves wind speed, moisture flux and onshore flow -- adjusting the
    raw column alone would leave the model reading a physically incoherent row.

    This is a model probe, not a forecast. The response carries a disclaimer
    the caller is expected to display.

    Args:
        date: The valid date to build the scenario on.
        district: District name.
        humidity: Change in humidity.
        wind: Change applied to both 850 hPa wind components, m/s.
        pressure: Change in sea-level pressure, hPa.
        instability: Change in CAPE, J/kg.
        soft_routing: Whether to blend correctors by regime probability.

    Returns:
        A :class:`ScenarioResponse` comparing baseline against scenario.
    """
    if not BUNDLE.data_connected:
        return _not_ready("No data is connected yet, so no scenario can be run.")
    if not BUNDLE.models_loaded:
        return _not_ready("The models have not been trained yet.")

    requested = {
        "humidity": humidity, "wind": wind, "pressure": pressure, "instability": instability,
    }
    rows = _rows_for(date, district)

    def summarise(frame: pd.DataFrame) -> Dict[str, Any]:
        """Run the pipeline over ``frame`` and reduce it to a district summary."""
        result = run_grid_forecast(frame, soft_routing=soft_routing)
        district_probs = result["regime_probs"].mean(axis=0)
        label, _ = blend_regimes(district_probs)
        aggregated = aggregate_to_district(
            result["grid"], BUNDLE.districts,
            threshold_names=result["threshold_names"] or None,
        )
        record = aggregated.iloc[0]
        return {
            "regime": str(district_probs.idxmax()),
            "regime_label": label,
            "regime_confidence": float(district_probs.max()),
            "corrected_forecast_mm": float(record["mean_corrected_mm"]),
            "warning_level": str(record["warning_level"]),
            "heavy_rain_probability": {
                name: float(record[f"max_{probability_column(name)}"])
                for name in result["threshold_names"]
                if f"max_{probability_column(name)}" in aggregated.columns
            },
        }

    baseline = summarise(rows)

    adjusted = rows.copy()
    adjustments: List[ScenarioAdjustment] = []
    for control, delta in requested.items():
        spec = SCENARIO_CONTROLS[control]
        present = [c for c in spec["columns"] if c in adjusted.columns]
        if delta:
            for column in present:
                adjusted[column] = pd.to_numeric(adjusted[column], errors="coerce") + delta
        adjustments.append(
            ScenarioAdjustment(
                control=control,
                columns=present,
                delta=float(delta),
                # A knob whose columns are absent from the connected data did
                # nothing; saying so stops an unchanged result being read as
                # "this factor has no effect".
                applied=bool(present) if delta else True,
            )
        )

    scenario = summarise(adjusted)
    thresholds = set(baseline["heavy_rain_probability"]) | set(scenario["heavy_rain_probability"])

    return ScenarioResponse(
        district=str(district),
        date=date,
        adjustments=adjustments,
        baseline=baseline,
        scenario=scenario,
        delta_corrected_mm=scenario["corrected_forecast_mm"] - baseline["corrected_forecast_mm"],
        delta_probability={
            name: scenario["heavy_rain_probability"].get(name, 0.0)
            - baseline["heavy_rain_probability"].get(name, 0.0)
            for name in sorted(thresholds)
        },
        regime_changed=scenario["regime"] != baseline["regime"],
    )


@app.get(
    "/drivers",
    response_model=DriversResponse,
    responses={503: {"model": NotReadyResponse}},
    tags=["prediction"],
)
def drivers(
    date: _date = Query(..., description="Valid date of the forecast (YYYY-MM-DD)."),
    top_n: int = Query(6, ge=1, le=20, description="How many drivers to return."),
    max_cells: int = Query(400, ge=20, le=5000, description="Cells to sample for the average."),
):
    """What is driving the corrections across the whole domain on one date.

    Explaining every cell of a full grid is expensive, so a random sample is
    used and the response says so. The sample is drawn with a fixed seed, so
    the same date gives the same answer twice.

    Args:
        date: The valid date.
        top_n: How many drivers to return.
        max_cells: Cells to sample.

    Returns:
        A :class:`DriversResponse` ordered by average absolute influence.

    Raises:
        HTTPException: 404 if the date has no data.
    """
    if not BUNDLE.data_connected:
        return _not_ready("No data is connected yet, so drivers cannot be computed.")
    if not BUNDLE.models_loaded:
        return _not_ready("The models have not been trained yet.")

    table = BUNDLE.analysis_table
    assert table is not None
    rows = table[table["date"] == pd.Timestamp(date)]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No data for {date}.")

    sampled = len(rows) > max_cells
    if sampled:
        rows = rows.sample(max_cells, random_state=42)

    result = run_grid_forecast(rows)
    features = result["features"]
    cell_regimes = result["cell_regimes"]

    counts = cell_regimes.value_counts(normalize=True)
    regime_share = {str(k): float(v) for k, v in counts.items()}
    dominant = str(counts.idxmax()) if not counts.empty else None

    # Average SHAP over the cells, grouped by the regime model that produced
    # them -- each regime has its own corrector, so a single global explainer
    # would be explaining a model that does not exist.
    totals: Dict[str, List[float]] = {}
    for regime, idx in features.groupby(cell_regimes.values).groups.items():
        try:
            model = BUNDLE.corrector.model_for(str(regime))
            values = model.shap_values(features.loc[idx])
        except Exception as exc:  # noqa: BLE001 - a regime that cannot explain is skipped
            LOGGER.warning("Driver SHAP failed for regime %s: %s", regime, exc)
            continue
        for position, column in enumerate(model.feature_columns):
            totals.setdefault(column, []).extend(values[:, position].tolist())

    if not totals:
        return DriversResponse(
            date=date, n_cells_sampled=len(features), sampled=sampled,
            dominant_regime=dominant, regime_share=regime_share,
        )

    ranked = sorted(
        (
            (column, float(np.mean(np.abs(vals))), float(np.mean(vals)))
            for column, vals in totals.items()
        ),
        key=lambda item: -item[1],
    )[:top_n]

    return DriversResponse(
        date=date,
        n_cells_sampled=len(features),
        sampled=sampled,
        drivers=[
            DomainDriver(
                feature=column,
                mean_abs_contribution_mm=abs_mean,
                mean_signed_contribution_mm=signed_mean,
                # "Mixed" when the signed average is small next to the absolute
                # one: the factor matters everywhere but pulls both ways.
                direction=(
                    "up" if signed_mean > 0.2 * abs_mean
                    else "down" if signed_mean < -0.2 * abs_mean
                    else "mixed"
                ),
            )
            for column, abs_mean, signed_mean in ranked
        ],
        dominant_regime=dominant,
        regime_share=regime_share,
    )


@app.get(
    "/events",
    response_model=ReplayListResponse,
    responses={503: {"model": NotReadyResponse}},
    tags=["verification"],
)
def events(
    limit: int = Query(10, ge=1, le=50, description="How many events to return."),
    min_observed_mm: float = Query(
        64.5, ge=0, description="Smallest district-mean observation that counts as an event."
    ),
    unseen_only: bool = Query(
        True,
        description=(
            "Restrict to dates after the training period. Events the models "
            "trained on prove nothing about generalisation."
        ),
    ),
):
    """The biggest observed rainfall events available for replay.

    Each event carries what the raw model forecast, what the system corrected
    it to, and what actually fell -- the whole chain on a real day.

    Events inside the training period are marked rather than silently mixed in:
    replaying a day the model was fitted on demonstrates nothing, and the
    caller needs to be able to tell the difference.

    Args:
        limit: How many events to return.
        min_observed_mm: Smallest district-mean observation that counts.
        unseen_only: Restrict to dates the models never trained on.

    Returns:
        A :class:`ReplayListResponse`, largest observed rainfall first.
    """
    if not BUNDLE.data_connected:
        return _not_ready("No data is connected yet, so no events can be replayed.")
    if not BUNDLE.models_loaded:
        return _not_ready("The models have not been trained yet.")

    table = BUNDLE.analysis_table
    assert table is not None
    if sch.OBSERVED_COLUMN not in table.columns:
        return ReplayListResponse(synthetic=is_synthetic_dataset())

    split = (BUNDLE.manifest or {}).get("split") or {}
    test_start_raw = ((split.get("test") or {}).get("start")) or None
    test_start = pd.to_datetime(test_start_raw).date() if test_start_raw else None

    work = table.copy()
    if sch.DISTRICT_COLUMN not in work.columns or work[sch.DISTRICT_COLUMN].isna().all():
        if BUNDLE.districts is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The connected data has no 'district' column and no district "
                    "boundaries are configured, so events cannot be attributed."
                ),
            )
        from ..aggregation.district import assign_districts

        work = pd.DataFrame(assign_districts(work, BUNDLE.districts).drop(columns=["geometry"]))

    observed = pd.to_numeric(work[sch.OBSERVED_COLUMN], errors="coerce")
    work = work.assign(__observed=observed).dropna(subset=["__observed"])

    per_day = (
        work.groupby(["date", sch.DISTRICT_COLUMN])
        .agg(mean_observed=("__observed", "mean"), peak_observed=("__observed", "max"))
        .reset_index()
    )
    per_day = per_day[per_day["mean_observed"] >= min_observed_mm]
    n_candidates = int(len(per_day))

    if test_start is not None and unseen_only:
        per_day = per_day[per_day["date"] >= pd.Timestamp(test_start)]

    per_day = per_day.sort_values("mean_observed", ascending=False).head(limit)

    out: List[ReplayEvent] = []
    for _, candidate in per_day.iterrows():
        event_date = candidate["date"].date()
        district = str(candidate[sch.DISTRICT_COLUMN])
        try:
            rows = _rows_for(event_date, district)
            result = run_grid_forecast(rows)
            district_probs = result["regime_probs"].mean(axis=0)
            label, _ = blend_regimes(district_probs)
            aggregated = aggregate_to_district(
                result["grid"], BUNDLE.districts,
                threshold_names=result["threshold_names"] or None,
            )
            record = aggregated.iloc[0]
            raw = float(record["mean_raw_mm"]) if "mean_raw_mm" in aggregated.columns else None
            corrected = float(record["mean_corrected_mm"])
        except Exception as exc:  # noqa: BLE001 - a bad day must not kill the list
            LOGGER.warning("Event replay failed for %s on %s: %s", district, event_date, exc)
            continue

        truth = float(candidate["mean_observed"])
        raw_error = abs(raw - truth) if raw is not None else None
        corrected_error = abs(corrected - truth)
        out.append(
            ReplayEvent(
                date=event_date,
                district=district,
                observed_mm=truth,
                peak_observed_mm=float(candidate["peak_observed"]),
                raw_forecast_mm=raw,
                corrected_forecast_mm=corrected,
                regime_label=label,
                raw_error_mm=raw_error,
                corrected_error_mm=corrected_error,
                improved=(corrected_error < raw_error) if raw_error is not None else None,
                in_training_period=bool(test_start is not None and event_date < test_start),
            )
        )

    return ReplayListResponse(
        events=out,
        n_candidates=n_candidates,
        test_period_start=test_start,
        synthetic=is_synthetic_dataset(),
    )


@app.get("/api", tags=["meta"])
def root() -> Dict[str, Any]:
    """Describe the service and its endpoints.

    Returns:
        A short index of what is available.
    """
    return {
        "service": "Regime-aware monsoon rainfall post-processing",
        "endpoints": {
            "GET /health": "Readiness: what is loaded, what is missing.",
            "GET /predict?date=YYYY-MM-DD&district=<name>": "Full pipeline for one district.",
            "GET /grid?date=YYYY-MM-DD": "Gridded raw / corrected / difference / observed fields.",
            "GET /timeline?date=&district=": "Regime and rainfall evolution around a date.",
            "GET /risk-matrix?date=": "Every district scored and ranked for one date.",
            "GET /watch?date=": "Highest-risk districts, as an early-warning view.",
            "GET /events": "Biggest observed rainfall events, for replay.",
            "GET /what-if?date=&district=": "Re-run the pipeline with inputs nudged.",
            "GET /drivers?date=": "What is driving corrections across the domain.",
            "GET /districts": "District names available.",
            "GET /dates": "Date range the connected dataset covers.",
            "GET /verification-report": "Saved verification report JSON.",
            "GET /dashboard": "Human-facing dashboard.",
            "GET /docs": "Interactive OpenAPI documentation.",
        },
        "models_loaded": BUNDLE.models_loaded,
        "data_connected": BUNDLE.data_connected,
        "synthetic_dataset": is_synthetic_dataset(),
    }


# ---------------------------------------------------------------------------
# Dashboard
#
# Mounted last so it cannot shadow an API route. The dashboard is static files
# only -- it holds no rainfall logic of its own and renders whatever the API
# returns, so there is no second place for a number to be invented.
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def dashboard_root() -> FileResponse:
    """Serve the dashboard at the site root.

    Returns:
        The dashboard's ``index.html``.

    Raises:
        HTTPException: 404 if the static bundle is missing.
    """
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Dashboard assets are not installed.")
    return FileResponse(index)


@app.get("/dashboard", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the dashboard at its named path.

    Returns:
        The dashboard's ``index.html``.

    Raises:
        HTTPException: 404 if the static bundle is missing.
    """
    return dashboard_root()


class _NoCacheStaticFiles(StaticFiles):
    """Static files that are always revalidated before use.

    Browsers hold on to ``app.css`` and ``app.js`` aggressively. On a dashboard
    that is actively being edited -- and demonstrated -- that means a reload can
    silently show yesterday's layout over today's numbers, which is a
    confusing failure to diagnose. ETags still do the real work, so a
    revalidated hit costs one 304 and no body.
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Any:
        """Attach a no-cache header to every file served.

        Returns:
            The response, with revalidation forced.
        """
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response


if STATIC_DIR.exists():
    app.mount("/", _NoCacheStaticFiles(directory=STATIC_DIR, html=True), name="static")


