"""Tests for Stage 7 -- the FastAPI service.

Two states are covered: a cold service with no data and no models (which must
still start and explain itself), and a fully trained service (which must serve
a complete prediction).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from rainfall_pipeline.api import main as api_main
from rainfall_pipeline.api.main import ModelBundle, app
from rainfall_pipeline.config.thresholds import REGIME_LABELS, WARNING_LEVELS
from rainfall_pipeline.models.bias_correction import RegimeBiasCorrector
from rainfall_pipeline.models.heavy_rain_probability import (
    HeavyRainProbabilityModel,
    attach_corrected_forecast,
)
from rainfall_pipeline.models.regime_classifier import RegimeClassifier

from .conftest import FAST_PARAMS


@pytest.fixture
def cold_client(monkeypatch) -> TestClient:
    """A client whose service found no data and no trained models."""
    monkeypatch.setattr(api_main, "load_bundle", lambda *a, **k: ModelBundle())
    with TestClient(app) as client:
        yield client


@pytest.fixture
def trained_bundle(dummy_features, dummy_regimes, district_polygons) -> ModelBundle:
    """A bundle with every artifact fitted on the dummy rows.

    The models here are meaningless -- 8 rows train nothing real. The fixture
    exists to exercise the request path end to end.
    """
    classifier = RegimeClassifier(params=FAST_PARAMS).fit(dummy_features, dummy_regimes)
    corrector = RegimeBiasCorrector(params=FAST_PARAMS, min_rows_per_regime=2).fit(
        dummy_features, dummy_regimes
    )
    prepared = attach_corrected_forecast(
        dummy_features, corrector.predict(dummy_features, dummy_regimes)
    )
    probability = HeavyRainProbabilityModel(params=FAST_PARAMS).fit(prepared)

    return ModelBundle(
        classifier=classifier,
        corrector=corrector,
        probability=probability,
        climatology=None,
        districts=district_polygons,
        analysis_table=dummy_features,
        manifest={"pipeline_complete": True},
    )


@pytest.fixture
def trained_client(monkeypatch, trained_bundle) -> TestClient:
    """A client whose service has a full set of artifacts loaded."""
    monkeypatch.setattr(api_main, "load_bundle", lambda *a, **k: trained_bundle)
    with TestClient(app) as client:
        yield client


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------

def test_service_starts_without_data_or_models(cold_client: TestClient) -> None:
    """A brand-new checkout must start, not crash."""
    response = cold_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["models_loaded"] is False
    assert body["data_connected"] is False
    assert body["missing"], "health must say what is missing"


def test_predict_without_models_returns_503_with_instructions(cold_client: TestClient) -> None:
    """An untrained service must explain itself rather than 500."""
    response = cold_client.get("/predict", params={"date": "2020-07-01", "district": "Pune"})
    assert response.status_code == 503
    body = response.json()
    assert "no prediction can be made" in body["detail"]
    assert any("run_full_training_pipeline" in step for step in body["next_steps"])


def test_districts_is_empty_but_valid_without_data(cold_client: TestClient) -> None:
    """The district list must be an empty list, never an error."""
    body = cold_client.get("/districts").json()
    assert body == {"count": 0, "source": "none", "districts": []}


def test_verification_report_absent_is_reported_cleanly(cold_client, monkeypatch) -> None:
    """No report yet must be a 200 with a clear explanation."""
    monkeypatch.setattr(api_main, "load_report", lambda *a, **k: None)
    body = cold_client.get("/verification-report").json()
    assert body["available"] is False
    assert "run_full_training_pipeline" in body["detail"]
    assert body["report"] is None


def test_api_index_lists_the_endpoints(cold_client: TestClient) -> None:
    """The JSON index must document what the service offers."""
    body = cold_client.get("/api").json()
    assert "GET /predict?date=YYYY-MM-DD&district=<name>" in body["endpoints"]
    assert "GET /grid?date=YYYY-MM-DD" in body["endpoints"]


def test_root_serves_the_dashboard(cold_client: TestClient) -> None:
    """``/`` must serve the dashboard, not JSON.

    The dashboard has to come up even when nothing is trained -- it is the
    surface that explains what is missing, so it cannot depend on the models
    being present.
    """
    response = cold_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Forecasting Lab" in response.text
    for view in ("overview", "historical", "risk", "scenario", "settings"):
        assert f'data-view="{view}"' in response.text, f"missing nav entry for {view}"


def test_dashboard_assets_are_served(cold_client: TestClient) -> None:
    """The dashboard's stylesheet and script must be reachable."""
    for asset in ("/app.css", "/app.js"):
        response = cold_client.get(asset)
        assert response.status_code == 200, asset
        assert response.text.strip()


# ---------------------------------------------------------------------------
# Trained service
# ---------------------------------------------------------------------------

def test_health_reports_ready(trained_client: TestClient) -> None:
    """With everything loaded, health must say so."""
    body = trained_client.get("/health").json()
    assert body["models_loaded"] is True
    assert body["data_connected"] is True
    assert body["districts_available"] is True


def test_districts_come_from_the_shapefile(trained_client: TestClient) -> None:
    """When boundaries are configured they are the authoritative source."""
    body = trained_client.get("/districts").json()
    assert body["source"] == "shapefile"
    assert "Pune" in body["districts"]


def test_predict_returns_the_full_contract(trained_client: TestClient) -> None:
    """The response must carry every field the brief specifies."""
    response = trained_client.get(
        "/predict", params={"date": "2020-07-02", "district": "Pune"}
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["date"] == "2020-07-02"
    assert body["district"] == "Pune"
    assert body["regime"] in REGIME_LABELS
    assert set(body["regime_probability"]) == set(REGIME_LABELS)
    assert sum(body["regime_probability"].values()) == pytest.approx(1.0)
    assert body["corrected_forecast_mm"] >= 0
    assert body["raw_forecast_mm"] is not None
    assert set(body["heavy_rain_probability"]) == {"heavy", "very_heavy", "extremely_heavy"}
    assert all(0.0 <= v <= 1.0 for v in body["heavy_rain_probability"].values())
    assert body["warning_level"] in WARNING_LEVELS
    assert body["n_grid_cells"] >= 1


def test_predict_includes_a_shap_explanation(trained_client: TestClient) -> None:
    """The regime call must be explainable, not a black box."""
    body = trained_client.get(
        "/predict", params={"date": "2020-07-02", "district": "Pune"}
    ).json()
    assert body["explanation"] is not None
    assert body["explanation"]["top_features"]
    for item in body["explanation"]["top_features"]:
        assert set(item) == {"feature", "shap_value"}


def test_explanation_can_be_switched_off(trained_client: TestClient) -> None:
    """SHAP costs time; callers must be able to skip it."""
    body = trained_client.get(
        "/predict", params={"date": "2020-07-02", "district": "Pune", "explain": "false"}
    ).json()
    assert body["explanation"] is None


def test_district_name_matching_is_case_insensitive(trained_client: TestClient) -> None:
    """A user typing 'pune' must not get a 404."""
    response = trained_client.get(
        "/predict", params={"date": "2020-07-02", "district": "pUnE"}
    )
    assert response.status_code == 200
    assert response.json()["district"] == "Pune"


def test_unknown_date_reports_the_available_range(trained_client: TestClient) -> None:
    """A 404 must tell the caller what period is actually covered."""
    response = trained_client.get(
        "/predict", params={"date": "1999-01-01", "district": "Pune"}
    )
    assert response.status_code == 404
    assert "2020-07-01" in response.json()["detail"]


def test_unknown_district_suggests_the_endpoint(trained_client: TestClient) -> None:
    """A 404 for a bad district must point at /districts."""
    response = trained_client.get(
        "/predict", params={"date": "2020-07-02", "district": "Atlantis"}
    )
    assert response.status_code == 404
    assert "/districts" in response.json()["detail"]


def test_malformed_date_is_rejected(trained_client: TestClient) -> None:
    """Pydantic must reject a non-date before it reaches the pipeline."""
    assert trained_client.get(
        "/predict", params={"date": "not-a-date", "district": "Pune"}
    ).status_code == 422


def test_bundle_load_tolerates_a_missing_artifact_dir(tmp_path: Path) -> None:
    """Loading from an empty directory must return an empty bundle, not raise."""
    bundle = api_main.load_bundle(tmp_path)
    assert bundle.models_loaded is False
    assert bundle.missing()


# ---------------------------------------------------------------------------
# Soft routing, compound regimes and the correction explanation
# ---------------------------------------------------------------------------

def test_predict_reports_a_regime_blend(trained_client: TestClient) -> None:
    """The response must carry a display label and the regimes behind it."""
    body = trained_client.get(
        "/predict", params={"date": "2020-07-01", "district": "Pune"}
    ).json()

    assert body["regime_label"], "a display label is required"
    assert body["regime_blend"], "the blend must name at least the leading regime"
    assert body["regime_blend"][0]["regime"] == body["regime"]
    # Probabilities must descend, so the label reads leading-regime-first.
    probs = [c["probability"] for c in body["regime_blend"]]
    assert probs == sorted(probs, reverse=True)
    # The label must be built from exactly the blend components.
    assert body["regime_label"] == " + ".join(c["regime"] for c in body["regime_blend"])


def test_predict_defaults_to_soft_routing(trained_client: TestClient) -> None:
    """Soft routing is the default and must be reported as such."""
    body = trained_client.get(
        "/predict", params={"date": "2020-07-01", "district": "Pune"}
    ).json()
    assert body["routing"] == "soft"


def test_predict_can_be_forced_to_hard_routing(trained_client: TestClient) -> None:
    """Hard routing stays available for comparison against the old behaviour."""
    body = trained_client.get(
        "/predict",
        params={"date": "2020-07-01", "district": "Pune", "soft_routing": "false"},
    ).json()
    assert body["routing"] == "hard"


def test_predict_explains_the_correction(trained_client: TestClient) -> None:
    """The correction, not just the regime, must come with an explanation."""
    body = trained_client.get(
        "/predict", params={"date": "2020-07-01", "district": "Pune"}
    ).json()

    detail = body["bias_explanation"]
    assert detail is not None, "the 'why did AI correct this' panel needs this field"
    assert detail["top_features"], "an explanation with no contributions is useless"
    # The explanation must describe an actual correction, not a restatement.
    assert detail["raw_mm"] is not None
    assert detail["corrected_mm"] is not None
    assert detail["corrected_mm"] >= 0.0


def test_regime_and_bias_explanations_are_distinct(trained_client: TestClient) -> None:
    """The two panels must answer different questions.

    A regime explanation is in log-odds of a class; a bias explanation is in mm
    of rainfall. Returning one for both would quietly mislead the reader.
    """
    body = trained_client.get(
        "/predict", params={"date": "2020-07-01", "district": "Pune"}
    ).json()
    assert body["explanation"] is not None
    assert body["bias_explanation"] is not None
    assert "predicted_bias_mm" in body["bias_explanation"]
    assert "predicted_bias_mm" not in body["explanation"]


# ---------------------------------------------------------------------------
# Gridded map endpoint
# ---------------------------------------------------------------------------

def test_grid_returns_aligned_panels(trained_client: TestClient) -> None:
    """Every panel must cover the same cells so one legend can serve them all."""
    body = trained_client.get("/grid", params={"date": "2020-07-01"}).json()

    expected = len(body["lats"]) * len(body["lons"])
    assert body["n_cells"] == expected
    assert set(body["panels"]) == {"raw", "corrected", "difference", "observed"}
    for name, values in body["panels"].items():
        assert len(values) == expected, f"panel '{name}' is not aligned to the lattice"
    assert len(body["regimes"]) == expected


def test_grid_difference_is_corrected_minus_raw(trained_client: TestClient) -> None:
    """The correction panel must be exactly what it claims to be."""
    body = trained_client.get("/grid", params={"date": "2020-07-01"}).json()

    checked = 0
    for raw, corrected, diff in zip(
        body["panels"]["raw"], body["panels"]["corrected"], body["panels"]["difference"]
    ):
        if raw is None or corrected is None or diff is None:
            continue
        assert diff == pytest.approx(corrected - raw, abs=1e-6)
        checked += 1
    assert checked, "no cell carried all three values, so nothing was verified"


def test_grid_without_models_returns_503(cold_client: TestClient) -> None:
    """An untrained service must refuse the grid the same way it refuses /predict."""
    response = cold_client.get("/grid", params={"date": "2020-07-01"})
    assert response.status_code == 503
    assert "no grid can be built" in response.json()["detail"]


def test_grid_rejects_an_unknown_date(trained_client: TestClient) -> None:
    """A date outside the dataset must 404 with the range that does exist."""
    response = trained_client.get("/grid", params={"date": "1999-01-01"})
    assert response.status_code == 404
    assert "covers" in response.json()["detail"]


def test_grid_enforces_the_cell_cap(trained_client: TestClient) -> None:
    """The cap must refuse rather than serialise an unbounded grid."""
    response = trained_client.get("/grid", params={"date": "2020-07-01", "max_cells": 1})
    assert response.status_code == 413
    assert "max_cells" in response.json()["detail"]


def test_dates_reports_the_available_range(trained_client: TestClient) -> None:
    """The dashboard needs to know which dates it may ask for."""
    body = trained_client.get("/dates").json()
    assert body["available"] is True
    assert body["start"] <= body["end"]
    assert body["n_dates"] >= 1


def test_dates_is_safe_without_data(cold_client: TestClient) -> None:
    """With no data the range endpoint must report absence, not raise."""
    body = cold_client.get("/dates").json()
    assert body["available"] is False
    assert body["start"] is None


# ---------------------------------------------------------------------------
# Regime blending helper
# ---------------------------------------------------------------------------

def test_blend_names_one_regime_when_the_leader_is_clear() -> None:
    """A confident classification must not be dressed up as a mixture."""
    probs = pd.Series({"Active": 0.90, "Break": 0.05, "Coastal": 0.05})
    label, blend = api_main.blend_regimes(probs)
    assert label == "Active"
    assert len(blend) == 1


def test_blend_names_two_regimes_when_the_runner_up_is_real() -> None:
    """A genuine mixture must be reported as one.

    This is the Konkan case from the brief: a coastal cell in an active spell
    is meaningfully both, and an argmax label hides half of that.
    """
    probs = pd.Series({"Coastal": 0.55, "Active": 0.35, "Break": 0.10})
    label, blend = api_main.blend_regimes(probs)
    assert label == "Coastal + Active"
    assert [c.regime for c in blend] == ["Coastal", "Active"]
    assert blend[0].probability > blend[1].probability


def test_blend_respects_the_threshold() -> None:
    """The cut-point must actually govern whether a second regime is named."""
    probs = pd.Series({"Coastal": 0.70, "Active": 0.30})
    assert api_main.blend_regimes(probs, min_probability=0.25)[0] == "Coastal + Active"
    assert api_main.blend_regimes(probs, min_probability=0.35)[0] == "Coastal"


def test_blend_never_names_more_than_two() -> None:
    """Three regimes over the cut-point must still yield a readable label."""
    probs = pd.Series({"Active": 0.34, "Coastal": 0.33, "Orographic": 0.33})
    label, blend = api_main.blend_regimes(probs)
    assert len(blend) == 2
    assert label.count("+") == 1


def test_blend_names_one_regime_when_every_option_is_weak() -> None:
    """A flat, uninformative distribution must not manufacture a mixture."""
    probs = pd.Series({name: 1.0 / len(REGIME_LABELS) for name in REGIME_LABELS})
    label, blend = api_main.blend_regimes(probs)
    assert len(blend) == 1
    assert "+" not in label


def test_blend_handles_an_empty_distribution() -> None:
    """An empty distribution must degrade, not raise."""
    label, blend = api_main.blend_regimes(pd.Series(dtype="float64"))
    assert label == "unknown"
    assert blend == []


# ---------------------------------------------------------------------------
# "How to read this" mode
# ---------------------------------------------------------------------------

def test_dashboard_ships_the_guidance_containers(cold_client: TestClient) -> None:
    """Every panel that needs an explainer must have somewhere to put it."""
    html = cold_client.get("/").text
    for panel in (
        "howto-intro",
        "howto-regime",
        "howto-rainfall",
        "howto-probability",
        "howto-warning",
        "howto-maps",
        "howto-bias",
        "howto-regime-shap",
    ):
        assert f'id="{panel}"' in html, f"missing guidance container for {panel}"
    assert 'id="howto-toggle"' in html


def test_guidance_containers_have_matching_content(cold_client: TestClient) -> None:
    """Every container in the page must be filled by the script, and vice versa.

    A container with no entry renders as an empty blue box; an entry with no
    container renders nowhere at all. Both fail silently in a browser, so they
    are checked here instead.
    """
    import re

    html = cold_client.get("/").text
    script = cold_client.get("/app.js").text

    in_page = set(re.findall(r'id="(howto-[a-z-]+)"', html)) - {"howto-toggle"}
    in_script = set(re.findall(r'"(howto-[a-z-]+)":\s*\{', script))
    assert in_page == in_script, (
        f"only in page: {sorted(in_page - in_script)}; "
        f"only in script: {sorted(in_script - in_page)}"
    )


def test_cards_put_data_before_guidance(cold_client: TestClient) -> None:
    """Turning guidance on must not push the numbers out of view.

    Where a card exists to show a value, the explainer trails it. The map and
    accuracy panels are the other way round -- there you need to know how to
    read the thing before you look at it.
    """
    html = cold_client.get("/").text
    for value_id, guidance_id in (
        ("regime-label", "howto-regime"),
        ("raw-mm", "howto-rainfall"),
        ("probability-bars", "howto-probability"),
        ("warning-level", "howto-warning"),
    ):
        assert html.index(f'id="{value_id}"') < html.index(f'id="{guidance_id}"'), (
            f"{guidance_id} must come after {value_id}"
        )

    # ...and the reverse for the map panel, which you need to know how to read
    # before you look at it.
    assert html.index('id="howto-maps"') < html.index('id="map-grid"')


def test_guidance_is_hidden_until_switched_on(cold_client: TestClient) -> None:
    """The mode is opt-in; the default view must stay uncluttered."""
    css = cold_client.get("/app.css").text
    assert ".howto { display: none; }" in css
    assert "body.howto-on .howto {" in css


# ---------------------------------------------------------------------------
# Plain-language presentation
# ---------------------------------------------------------------------------

def test_dashboard_keeps_jargon_out_of_the_default_view(cold_client: TestClient) -> None:
    """Verification acronyms must not appear in the page's own markup.

    The scorecard uses them, but it is built by the script and sits behind a
    disclosure. Anything hard-coded into the HTML is on screen the moment the
    page loads, and none of it should need a meteorology background to read.
    """
    import re

    html = cold_client.get("/").text
    body = html[html.index("<body>"):]
    # Strip tags: element ids and class names are internal plumbing, and
    # `card--regime` is not something a reader ever sees.
    visible = re.sub(r"<[^>]+>", " ", body)

    # Acronyms are matched case-sensitively -- "FAR" is jargon, "how far ahead"
    # is English, and only the capitalisation tells them apart.
    for acronym in ("RMSE", "POD", "FAR", "CSI", "ETS", "FSS", "BSS", "SHAP"):
        assert not re.search(rf"\b{acronym}\b", visible), (
            f"'{acronym}' is visible in the default view"
        )
    assert not re.search(r"\bregimes?\b", visible, re.IGNORECASE), (
        "'regime' is visible in the default view"
    )


def test_every_feature_in_the_pipeline_has_a_plain_label(cold_client: TestClient) -> None:
    """No raw column name may reach the explanation panels.

    An unmapped feature falls back to its underscored column name, which is how
    `cape_humidity` ends up on screen. The map has to stay in step with the
    feature list as the pipeline grows.
    """
    import re

    from rainfall_pipeline.features.engineering import FEATURE_COLUMNS, REGIME_FEATURE_COLUMNS

    script = cold_client.get("/app.js").text
    block = script[script.index("const FEATURE_LABELS"):script.index("const REGIME_INFO")]
    labelled = set(re.findall(r"^\s{2}(\w+):", block, re.MULTILINE))

    missing = (set(FEATURE_COLUMNS) | set(REGIME_FEATURE_COLUMNS)) - labelled
    assert not missing, f"features with no plain-language label: {sorted(missing)}"


def test_every_regime_has_a_plain_description(cold_client: TestClient) -> None:
    """A regime with no description shows its internal label to the user."""
    from rainfall_pipeline.config.thresholds import REGIME_LABELS

    script = cold_client.get("/app.js").text
    block = script[script.index("const REGIME_INFO"):script.index("const MODEL_INFO")]
    for regime in REGIME_LABELS:
        assert f'"{regime}"' in block or f"  {regime}: {{" in block, (
            f"regime '{regime}' has no plain-language description"
        )


def test_every_reported_model_has_a_plain_name(cold_client: TestClient) -> None:
    """A model missing from the vocabulary shows its raw id in the ranking."""
    from rainfall_pipeline.verification.report import MODEL_DESCRIPTIONS

    script = cold_client.get("/app.js").text
    block = script[script.index("const MODEL_INFO"):script.index("const OUR_MODEL")]
    for model_id in MODEL_DESCRIPTIONS:
        assert f"{model_id}:" in block, f"model '{model_id}' has no plain-language name"


def test_verification_report_is_still_reachable(cold_client: TestClient) -> None:
    """The scorecard is no longer rendered on screen, but it must not be lost.

    The Model Comparison view was removed from the dashboard. The numbers still
    have to be obtainable, so the endpoint stays and the page keeps a control
    that downloads it.
    """
    html = cold_client.get("/").text
    assert 'id="export-report"' in html, "no way to obtain the report from the page"
    assert cold_client.get("/verification-report").status_code == 200


def test_static_assets_are_never_served_stale(cold_client: TestClient) -> None:
    """A cached stylesheet showing yesterday's layout is a confusing failure."""
    for asset in ("/app.css", "/app.js"):
        cache = cold_client.get(asset).headers.get("cache-control", "")
        assert "no-cache" in cache, f"{asset} may be cached: {cache!r}"


# ---------------------------------------------------------------------------
# Uncertainty, anomaly and custom thresholds
# ---------------------------------------------------------------------------

def test_predict_reports_regime_confidence(trained_client: TestClient) -> None:
    """A single number without a confidence invites unearned certainty."""
    body = trained_client.get(
        "/predict", params={"date": "2020-07-01", "district": "Pune"}
    ).json()
    assert 0.0 <= body["regime_confidence"] <= 1.0
    assert body["regime_confidence"] == pytest.approx(
        max(body["regime_probability"].values()), abs=1e-6
    )


def test_predict_reports_the_peak_cell(trained_client: TestClient) -> None:
    """The wettest cell drives the warning and must not be hidden by the mean."""
    body = trained_client.get(
        "/predict", params={"date": "2020-07-01", "district": "Pune"}
    ).json()
    assert body["peak_cell_mm"] is not None
    assert body["peak_cell_mm"] >= body["corrected_forecast_mm"] - 1e-6


def test_predict_survives_without_an_interval_model(trained_client: TestClient) -> None:
    """The range is an enhancement; its absence must not break a forecast."""
    body = trained_client.get(
        "/predict", params={"date": "2020-07-01", "district": "Pune"}
    ).json()
    # The trained fixture has no interval artifact, so this must be null
    # rather than an error or a fabricated band.
    assert body["interval"] is None
    assert body["corrected_forecast_mm"] is not None


def test_custom_threshold_is_returned_and_flagged(trained_client: TestClient) -> None:
    """A user's own limit must be answered, and marked as an estimate."""
    body = trained_client.get(
        "/predict",
        params={"date": "2020-07-01", "district": "Pune", "threshold": 90},
    ).json()

    custom = [t for t in body["threshold_probabilities"] if t["name"] == "custom"]
    assert custom, "the requested threshold must appear in the response"
    assert custom[0]["threshold_mm"] == pytest.approx(90.0)
    assert custom[0]["interpolated"] is True, (
        "an interpolated estimate must never be presented as a fitted model"
    )
    assert 0.0 <= custom[0]["probability"] <= 1.0


def test_custom_threshold_matching_a_trained_one_is_not_flagged(
    trained_client: TestClient,
) -> None:
    """Asking for exactly 64.5 mm should give the trained answer, not an estimate."""
    body = trained_client.get(
        "/predict",
        params={"date": "2020-07-01", "district": "Pune", "threshold": 64.5},
    ).json()
    custom = [t for t in body["threshold_probabilities"] if t["name"] == "custom"][0]
    heavy = [t for t in body["threshold_probabilities"] if t["name"] == "heavy"][0]
    assert custom["interpolated"] is False
    assert custom["probability"] == pytest.approx(heavy["probability"])


def test_no_threshold_means_no_custom_row(trained_client: TestClient) -> None:
    """Without a request there must be no phantom custom entry."""
    body = trained_client.get(
        "/predict", params={"date": "2020-07-01", "district": "Pune"}
    ).json()
    assert not [t for t in body["threshold_probabilities"] if t["name"] == "custom"]


def test_threshold_rejects_nonsense(trained_client: TestClient) -> None:
    """A negative or absurd threshold must be refused by validation."""
    for bad in (-5, 0, 5000):
        response = trained_client.get(
            "/predict",
            params={"date": "2020-07-01", "district": "Pune", "threshold": bad},
        )
        assert response.status_code == 422, f"threshold={bad} should be rejected"


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

def test_timeline_returns_ordered_steps(trained_client: TestClient) -> None:
    """Steps must run earliest to latest with the anchor at offset zero."""
    body = trained_client.get(
        "/timeline", params={"date": "2020-07-01", "district": "Pune", "back": 1, "forward": 1}
    ).json()

    offsets = [s["offset_days"] for s in body["steps"]]
    assert offsets == sorted(offsets)
    assert body["anchor_date"] == "2020-07-01"


def test_timeline_declares_its_step_size(trained_client: TestClient) -> None:
    """Daily data must not imply sub-daily resolution it does not have."""
    body = trained_client.get(
        "/timeline", params={"date": "2020-07-01", "district": "Pune"}
    ).json()
    assert body["step_hours"] == 24


def test_timeline_flags_regime_changes(trained_client: TestClient) -> None:
    """A transition must be recorded on the step where it happens."""
    body = trained_client.get(
        "/timeline", params={"date": "2020-07-01", "district": "Pune", "back": 2, "forward": 2}
    ).json()

    changed = [s for s in body["steps"] if s["regime_changed"]]
    assert len(changed) == len(body["transitions"]), (
        "every flagged step must have a matching transition description"
    )
    # The first step has nothing to compare against.
    if body["steps"]:
        assert body["steps"][0]["regime_changed"] is False


def test_timeline_skips_missing_days_rather_than_failing(trained_client: TestClient) -> None:
    """A window running past the data must return what exists, not 404."""
    body = trained_client.get(
        "/timeline", params={"date": "2020-07-01", "district": "Pune", "back": 7, "forward": 7}
    ).json()
    assert body["steps"], "days inside the dataset must still be returned"


def test_timeline_without_models_returns_503(cold_client: TestClient) -> None:
    """An untrained service must explain itself here too."""
    response = cold_client.get(
        "/timeline", params={"date": "2020-07-01", "district": "Pune"}
    )
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Risk matrix, watchlist, drivers
# ---------------------------------------------------------------------------

def test_risk_matrix_ranks_by_severity(trained_client: TestClient) -> None:
    """The most severe district must come first, or the table is unreadable."""
    body = trained_client.get("/risk-matrix", params={"date": "2020-07-01"}).json()
    assert body["districts"], "the matrix must cover the districts with data"

    order = {name: i for i, name in enumerate(WARNING_LEVELS)}
    severities = [order[d["warning_level"]] for d in body["districts"]]
    assert severities == sorted(severities, reverse=True)


def test_risk_matrix_counts_match_the_rows(trained_client: TestClient) -> None:
    """The summary must total the same districts the table lists."""
    body = trained_client.get("/risk-matrix", params={"date": "2020-07-01"}).json()
    assert sum(body["counts_by_warning"].values()) == len(body["districts"])


def test_risk_matrix_reports_the_peak_cell(trained_client: TestClient) -> None:
    """A district average hides the one cell that floods; both are needed."""
    body = trained_client.get("/risk-matrix", params={"date": "2020-07-01"}).json()
    for district in body["districts"]:
        if district["peak_cell_mm"] is not None:
            assert district["peak_cell_mm"] >= district["corrected_forecast_mm"] - 1e-6


def test_watch_counts_cover_every_district_not_just_listed(
    trained_client: TestClient,
) -> None:
    """The summary must count all districts screened, not the truncated list.

    Otherwise a watchlist capped at 15 rows reports "15 districts" on a day
    when 700 were checked and 45 were clear.
    """
    body = trained_client.get(
        "/watch", params={"date": "2020-07-01", "limit": 1}
    ).json()
    assert sum(body["counts_by_warning"].values()) == body["n_screened"]
    assert len(body["districts"]) <= 1


def test_watch_says_all_clear_explicitly(trained_client: TestClient) -> None:
    """An empty list must be distinguishable from a failed lookup."""
    body = trained_client.get(
        "/watch", params={"date": "2020-07-01", "min_probability": 1.0}
    ).json()
    if not body["districts"]:
        assert body["quiet"] is True
        assert body["n_screened"] > 0


def test_drivers_are_ranked_by_influence(trained_client: TestClient) -> None:
    """Drivers must descend by average absolute influence."""
    body = trained_client.get("/drivers", params={"date": "2020-07-01"}).json()
    magnitudes = [d["mean_abs_contribution_mm"] for d in body["drivers"]]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for driver in body["drivers"]:
        assert driver["direction"] in {"up", "down", "mixed"}


def test_drivers_regime_share_sums_to_one(trained_client: TestClient) -> None:
    """The share of the domain in each regime must be a distribution."""
    body = trained_client.get("/drivers", params={"date": "2020-07-01"}).json()
    if body["regime_share"]:
        assert sum(body["regime_share"].values()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Event replay
# ---------------------------------------------------------------------------

def test_events_are_ordered_by_size(trained_client: TestClient) -> None:
    """The biggest rainfall day must lead the list."""
    body = trained_client.get(
        "/events", params={"limit": 5, "min_observed_mm": 0, "unseen_only": False}
    ).json()
    observed = [e["observed_mm"] for e in body["events"]]
    assert observed == sorted(observed, reverse=True)


def test_events_mark_days_the_model_trained_on(trained_client: TestClient) -> None:
    """Replaying a training day proves nothing and must be labelled as such."""
    body = trained_client.get(
        "/events", params={"limit": 5, "min_observed_mm": 0, "unseen_only": False}
    ).json()
    for event in body["events"]:
        assert isinstance(event["in_training_period"], bool)


def test_events_error_arithmetic_is_consistent(trained_client: TestClient) -> None:
    """The reported errors and the 'improved' verdict must agree."""
    body = trained_client.get(
        "/events", params={"limit": 5, "min_observed_mm": 0, "unseen_only": False}
    ).json()
    for event in body["events"]:
        if event["raw_error_mm"] is None:
            continue
        assert event["raw_error_mm"] == pytest.approx(
            abs(event["raw_forecast_mm"] - event["observed_mm"]), abs=1e-6
        )
        assert event["corrected_error_mm"] == pytest.approx(
            abs(event["corrected_forecast_mm"] - event["observed_mm"]), abs=1e-6
        )
        assert event["improved"] == (event["corrected_error_mm"] < event["raw_error_mm"])


# ---------------------------------------------------------------------------
# What-if scenarios
# ---------------------------------------------------------------------------

def test_scenario_with_no_change_reproduces_the_baseline(
    trained_client: TestClient,
) -> None:
    """Zero adjustment must be a no-op, or every delta is noise."""
    body = trained_client.get(
        "/what-if", params={"date": "2020-07-01", "district": "Pune"}
    ).json()
    assert body["delta_corrected_mm"] == pytest.approx(0.0, abs=1e-9)
    assert body["regime_changed"] is False
    for delta in body["delta_probability"].values():
        assert delta == pytest.approx(0.0, abs=1e-9)


def test_scenario_responds_to_an_adjustment(trained_client: TestClient) -> None:
    """A large nudge must move something, or the controls are decorative."""
    body = trained_client.get(
        "/what-if",
        params={"date": "2020-07-01", "district": "Pune", "humidity": 25, "pressure": -15},
    ).json()
    moved = (
        abs(body["delta_corrected_mm"]) > 1e-9
        or any(abs(d) > 1e-9 for d in body["delta_probability"].values())
        or body["regime_changed"]
    )
    assert moved, "adjusting humidity and pressure must change the model's answer"


def test_scenario_always_carries_its_disclaimer(trained_client: TestClient) -> None:
    """A scenario presented as a forecast would be a serious misuse."""
    body = trained_client.get(
        "/what-if", params={"date": "2020-07-01", "district": "Pune", "wind": 5}
    ).json()
    assert "not a forecast" in body["disclaimer"]


def test_scenario_reports_controls_it_could_not_apply(trained_client: TestClient) -> None:
    """A knob with no matching column must say so, not read as 'no effect'."""
    body = trained_client.get(
        "/what-if", params={"date": "2020-07-01", "district": "Pune", "instability": 500}
    ).json()
    entries = {a["control"]: a for a in body["adjustments"]}
    assert set(entries) == {"humidity", "wind", "pressure", "instability"}
    for adjustment in body["adjustments"]:
        if adjustment["delta"] and not adjustment["columns"]:
            assert adjustment["applied"] is False


# ---------------------------------------------------------------------------
# Dashboard coverage of the new views
# ---------------------------------------------------------------------------

def test_dashboard_has_a_panel_for_every_new_feature(cold_client: TestClient) -> None:
    """Each backend capability must have somewhere on screen to appear.

    An endpoint with no panel is a feature nobody can find.
    """
    html = cold_client.get("/").text
    for element, feature in (
        ("timeline", "regime timeline / time machine"),
        ("watch-rows", "extreme rain watch"),
        ("risk-summary", "district risk matrix"),
        ("drivers", "top drivers"),
        ("events", "historical event replay"),
        ("scenario-sliders", "what-if simulator"),
        ("corrected-range", "prediction interval"),
        ("anomaly-line", "rainfall anomaly"),
        ("threshold", "custom threshold input"),
    ):
        assert f'id="{element}"' in html, f"no panel for {feature}"


def test_dashboard_exposes_no_raw_meteorological_inputs(cold_client: TestClient) -> None:
    """The brief is explicit: users enter district, date and horizon only.

    Humidity, wind, pressure and CAPE are backend concerns. The what-if sliders
    are built by script and are deliberately not part of the entry form.
    """
    import re

    html = cold_client.get("/").text
    form = html[html.index('class="controls"'):html.index("</section>", html.index('class="controls"'))]
    for banned in ("humidity", "wind", "pressure", "cape", "elevation", "vorticity"):
        assert not re.search(banned, form, re.IGNORECASE), (
            f"'{banned}' is exposed as a user input in the query form"
        )


def test_every_scenario_control_has_a_backend_counterpart(cold_client: TestClient) -> None:
    """A slider the API does not accept would silently do nothing."""
    import re

    from rainfall_pipeline.api.main import SCENARIO_CONTROLS as BACKEND_CONTROLS

    script = cold_client.get("/app.js").text
    block = script[script.index("const SCENARIO_CONTROLS = ["):script.index("function renderScenarioControls")]
    frontend = set(re.findall(r'key:\s*"(\w+)"', block))
    assert frontend == set(BACKEND_CONTROLS), (
        f"only in dashboard: {sorted(frontend - set(BACKEND_CONTROLS))}; "
        f"only in API: {sorted(set(BACKEND_CONTROLS) - frontend)}"
    )
