"""End-to-end wiring test for the training scripts.

Runs the full pipeline over the 8 dummy rows to prove every stage hands off to
the next correctly and that the report is produced. It asserts on structure
only -- 8 fabricated rows cannot support any claim about skill, and this test
makes none.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from rainfall_pipeline.training.run_full_training_pipeline import run_pipeline


def _args(source_files, tmp_path: Path) -> argparse.Namespace:
    """Build the argument namespace the pipeline expects."""
    return argparse.Namespace(
        era5_path=str(source_files["era5"]),
        observed_path=str(source_files["observed"]),
        nwp_path=str(source_files["nwp"]),
        start_date=None,
        end_date=None,
        train_end=None,
        val_end=None,
        rebuild=True,
        backend="xgboost",
        artifact_dir=str(tmp_path / "artifacts"),
        use_rule_labels=True,
        min_rows_per_regime=1,
        probability_backend="lightgbm",
        calibration="isotonic",
    )


@pytest.fixture
def pipeline_run(source_files, tmp_path, monkeypatch):
    """Run the full pipeline into a temporary directory and return the report."""
    import rainfall_pipeline.data.store as store

    store_dir = tmp_path / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(store, "STORE_DIR", store_dir)
    monkeypatch.setattr(store, "ensure_dirs", lambda: None)

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    report = run_pipeline(_args(source_files, tmp_path))
    return report, artifacts


def test_pipeline_runs_every_stage(pipeline_run) -> None:
    """All five model slots must be scored on the held-out test split."""
    report, _ = pipeline_run
    assert set(report["models"]) == {
        "A_raw_nwp",
        "B_global_ml",
        "C_quantile_mapping",
        "C_regime_quantile_mapping",
        "D_regime_residual",
        "E_regime_residual_probability",
    }
    assert report["n_test_rows"] > 0


def test_pipeline_writes_every_artifact(pipeline_run) -> None:
    """Each stage must leave behind a loadable artifact."""
    _, artifacts = pipeline_run
    for name in (
        "verification_report.json",
        "verification_report.md",
        "verification_report.html",
        "climatology.parquet",
        "regime_classifier.joblib",
        "baseline_b_global_ml.joblib",
        "baseline_c_quantile_mapping.joblib",
        "bias_correction.joblib",
        "heavy_rain_probability.joblib",
        "training_manifest.json",
    ):
        assert (artifacts / name).exists(), name


def test_manifest_records_the_split_and_regime_counts(pipeline_run) -> None:
    """The manifest is the provenance record; it must describe the run."""
    _, artifacts = pipeline_run
    manifest = json.loads((artifacts / "training_manifest.json").read_text())
    assert manifest["pipeline_complete"] is True
    assert set(manifest["split"]) == {"train", "validation", "test"}
    assert "label_counts" in manifest["regime_classifier"]
    assert "regime_training_counts" in manifest["bias_correction"]


def test_pipeline_test_period_follows_the_training_period(pipeline_run) -> None:
    """The held-out period must come after training -- never interleaved."""
    report, _ = pipeline_run
    assert report["split"]["train"]["end"] < report["test_period"]["start"]


def test_report_records_probability_caveats(pipeline_run) -> None:
    """Degenerate thresholds and calibration shortcuts must be disclosed."""
    report, _ = pipeline_run
    assert report["notes"], "the report must carry its caveats"
    assert any("classifier" in note for note in report["notes"])


def test_pipeline_writes_the_report_into_the_artifact_dir(pipeline_run, tmp_path) -> None:
    """--artifact-dir must fully contain a run, so a test cannot clobber the
    report belonging to the real trained system."""
    _, artifacts = pipeline_run
    from rainfall_pipeline.config.regions import VERIFICATION_REPORT_PATH

    assert (artifacts / "verification_report.json").exists()
    assert artifacts != VERIFICATION_REPORT_PATH.parent
