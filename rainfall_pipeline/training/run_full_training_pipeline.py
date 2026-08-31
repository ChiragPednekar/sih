"""Run every training stage end to end, then verify all five models.

    python -m rainfall_pipeline.training.run_full_training_pipeline

Stages, in order:

1. Load and cache the analysis table.
2. Split chronologically (never randomly).
3. Fit the climatology on the training split only.
4. Train the regime classifier on rule-based labels.
5. Train Baseline B, Baseline C and Model D.
6. Train the calibrated heavy-rain probability head (Model E's second half).
7. Score A, B, C, D and E on the held-out test split and write the report.

Every number in the report comes from step 7 running on real held-out data.
Nothing in this script asserts, assumes or hardcodes a level of skill.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

from ..aggregation.district import assign_districts, assign_region
from ..config.regions import DISTRICTS_PATH
from ..data import schema as sch
from ..data.loaders import load_district_boundaries
from ..models.baselines import RawForecastBaseline
from ..models.heavy_rain_probability import attach_corrected_forecast
from ..models.uncertainty import (
    INTERVAL_MODEL_FILENAME,
    PredictionIntervalModel,
    interval_coverage,
)
from ..models.regime_classifier import label_regimes
from ..verification.report import VerificationInputs, build_verification_report, write_report
from .common import (
    LOGGER,
    add_common_arguments,
    apply_climatology,
    artifact_dir,
    build_climatology,
    configure_logging,
    load_analysis_table,
    make_split,
    update_manifest,
)
from .train_bias_correction import train_all_correctors
from .train_heavy_rain_models import train_probability_head
from .train_regime_classifier import evaluate_agreement, train_regime_classifier


def _attach_districts(test_df):
    """Attach a district label to the test rows for the report breakdown.

    Uses whatever is available: a ``district`` column already in the data, or a
    point-in-polygon join against the configured shapefile. If neither exists
    the frame is returned unchanged and the district breakdown is simply absent
    from the report -- an omission, never a fabricated label.

    Args:
        test_df: The held-out test rows.

    Returns:
        The frame, with a ``district`` column when one could be resolved.
    """
    if sch.DISTRICT_COLUMN in test_df.columns and test_df[sch.DISTRICT_COLUMN].notna().any():
        return test_df
    try:
        districts = load_district_boundaries(DISTRICTS_PATH)
    except Exception as exc:  # noqa: BLE001 - the shapefile is optional
        LOGGER.info(
            "No district boundaries available (%s); the verification report "
            "will omit the per-district breakdown.",
            type(exc).__name__,
        )
        return test_df

    joined = assign_districts(test_df, districts)
    out = test_df.copy()
    out[sch.DISTRICT_COLUMN] = joined[sch.DISTRICT_COLUMN].values
    matched = int(out[sch.DISTRICT_COLUMN].notna().sum())
    LOGGER.info(
        "Attached districts to %d of %d test rows (%.1f%%).",
        matched, len(out), 100.0 * matched / max(len(out), 1),
    )
    return out


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the full training and verification pipeline.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The verification report dict.

    Raises:
        MissingDataError: If the raw data files are absent.
        SplitError: If the data cannot be split chronologically.
    """
    out_dir = artifact_dir(args.artifact_dir)

    # --- 1-2. data and split -------------------------------------------
    table = load_analysis_table(args)
    LOGGER.info("Analysis table: %d rows, %d columns", len(table), table.shape[1])
    split = make_split(table, args)
    LOGGER.info("Chronological split: %s", split.summary())

    # --- 3. climatology (training rows only) ----------------------------
    clim = build_climatology(split.train, out_dir)
    train_df = apply_climatology(split.train, clim)
    val_df = apply_climatology(split.validation, clim) if not split.validation.empty else None
    test_df = apply_climatology(split.test, clim)

    # --- 4. regime engine -----------------------------------------------
    LOGGER.info("=== Stage 2: regime classifier ===")
    classifier, regime_meta = train_regime_classifier(
        train_df, backend=args.backend, out_dir=out_dir
    )
    regime_meta["train_agreement"] = evaluate_agreement(classifier, train_df)
    if val_df is not None:
        regime_meta["validation_agreement"] = evaluate_agreement(classifier, val_df)

    train_regimes = label_regimes(train_df) if args.use_rule_labels else classifier.predict(train_df)
    # Routing at prediction time must use the classifier: the rules need the
    # observation, which does not exist for a real forecast.
    test_regimes = classifier.predict(test_df)
    test_df = test_df.copy()
    test_df["regime"] = test_regimes.values
    # Stratification labels. Without these the report collapses to one overall
    # number, which hides the thing that actually matters: whether the scheme
    # helps where the rain is, or only where it already was easy.
    test_df["region"] = assign_region(test_df).values
    test_df = _attach_districts(test_df)

    # --- 5. correction models -------------------------------------------
    LOGGER.info("=== Stage 3: bias correction and baselines ===")
    corr = train_all_correctors(
        train_df,
        train_regimes,
        backend=args.backend,
        loss=getattr(args, "loss", "mse"),
        min_rows_per_regime=args.min_rows_per_regime,
        out_dir=out_dir,
    )
    models = corr["models"]

    # --- 5b. prediction intervals ----------------------------------------
    LOGGER.info("=== Stage 3b: prediction intervals ===")
    intervals = PredictionIntervalModel(min_rows_per_regime=args.min_rows_per_regime)
    interval_meta: Dict[str, Any] = {}
    try:
        intervals.fit(train_df, train_regimes)
        interval_path = intervals.save(out_dir / INTERVAL_MODEL_FILENAME)
        interval_meta = {
            "quantiles": list(intervals.quantiles),
            "nominal_coverage": intervals.nominal_coverage,
            "regime_training_counts": intervals.training_counts,
            "regimes_with_own_model": sorted(intervals.models),
            "artifact": str(interval_path),
        }
        LOGGER.info(
            "Fitted %.0f%% prediction intervals (own models for: %s)",
            intervals.nominal_coverage * 100,
            sorted(intervals.models) or "none",
        )
    except Exception as exc:  # noqa: BLE001 - intervals are an add-on, not a gate
        # A failure here must not cost the run its point forecasts, but it must
        # be recorded rather than leaving a silently absent artifact.
        LOGGER.warning("Prediction intervals could not be fitted: %s", exc)
        interval_meta = {"error": str(exc)}
        intervals = None  # type: ignore[assignment]

    # --- 6. probability head ---------------------------------------------
    LOGGER.info("=== Stage 4: heavy-rain probability ===")
    corrected_train = models["model_d"].predict(train_df, train_regimes)
    prob_train = attach_corrected_forecast(train_df, corrected_train)

    prob_calib = None
    if val_df is not None:
        val_regimes = classifier.predict(val_df)
        prob_calib = attach_corrected_forecast(
            val_df, models["model_d"].predict(val_df, val_regimes)
        )

    prob_model, prob_meta = train_probability_head(
        prob_train,
        prob_calib,
        backend=args.probability_backend,
        method=args.calibration,
        out_dir=out_dir,
    )

    # --- 7. verification on the held-out test split -----------------------
    LOGGER.info("=== Stage 6: verification (A-E on held-out test data) ===")
    corrected_test = models["model_d"].predict(test_df, test_regimes)
    predictions = {
        "A_raw_nwp": RawForecastBaseline().predict(test_df),
        "B_global_ml": models["baseline_b"].predict(test_df),
        "C_quantile_mapping": models["baseline_c"].predict(test_df),
        "C_regime_quantile_mapping": models["baseline_c_per_regime"].predict(
            test_df, test_regimes
        ),
        "D_regime_residual": corrected_test,
        # Model E's rainfall field is Model D's; what makes it E is the
        # calibrated probability head scored alongside it below.
        "E_regime_residual_probability": corrected_test,
    }

    prob_test = attach_corrected_forecast(test_df, corrected_test)
    test_probs = prob_model.predict_proba(prob_test)
    probabilities = {
        "E_regime_residual_probability": {c: test_probs[c] for c in test_probs.columns}
    }

    # Whether the interval is honest is an empirical question, answered here
    # on held-out data. A nominal 80% band that contains 55% of observations is
    # not a range, it is a false reassurance -- so this is measured and
    # reported rather than assumed.
    if intervals is not None:
        try:
            test_interval = intervals.predict_interval(test_df, test_regimes)
            coverage = interval_coverage(
                test_df[sch.OBSERVED_COLUMN],
                test_interval["corrected_low"],
                test_interval["corrected_high"],
            )
            interval_meta["test_coverage"] = coverage
            LOGGER.info(
                "Prediction intervals: nominal %.0f%%, actual coverage %.1f%% "
                "on held-out data (mean width %.1f mm)",
                intervals.nominal_coverage * 100,
                coverage["coverage"] * 100,
                coverage["mean_width_mm"],
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Interval coverage could not be measured: %s", exc)
            interval_meta["test_coverage"] = {"error": str(exc)}

    notes = [
        "Regime routing at test time uses the learned classifier, not the "
        "rule labeller, because the rules require the observation.",
        f"Regimes with their own correction model: "
        f"{sorted(models['model_d'].models) or 'none'}; all others routed to the "
        f"global fallback.",
    ]
    if prob_meta["degenerate_thresholds"]:
        notes.append(
            "Threshold(s) "
            + ", ".join(prob_meta["degenerate_thresholds"])
            + " had no exceedance variation in training; those probabilities are "
            "the constant training base rate, not a fitted model."
        )
    cov = (interval_meta or {}).get("test_coverage") or {}
    if isinstance(cov, dict) and isinstance(cov.get("coverage"), float):
        nominal = interval_meta.get("nominal_coverage", 0.0)
        notes.append(
            f"Prediction intervals: nominal {nominal:.0%}, actual coverage "
            f"{cov['coverage']:.1%} on the test set (mean width "
            f"{cov['mean_width_mm']:.1f} mm)."
        )
        if abs(cov["coverage"] - nominal) > 0.1:
            notes.append(
                "The interval's actual coverage is more than 10 points from its "
                "nominal level, so the range is not calibrated and should be "
                "presented as indicative only."
            )

    if not prob_meta["calibration_fitted_on_holdout"]:
        notes.append(
            "Probability calibration was fitted on the training predictions "
            "(no validation split available), so the reliability table is "
            "optimistic."
        )

    report = build_verification_report(
        VerificationInputs(
            test_df=test_df,
            predictions=predictions,
            probabilities=probabilities,
            split_summary=split.summary(),
            notes=notes,
        )
    )
    # Written into the artifact directory rather than the global default, so a
    # run with --artifact-dir (including the test suite's temporary directory)
    # never overwrites the report for the real, trained system.
    paths = write_report(
        report,
        json_path=out_dir / "verification_report.json",
        markdown_path=out_dir / "verification_report.md",
        html_path=out_dir / "verification_report.html",
    )
    LOGGER.info("Wrote verification report: %s", {k: str(v) for k, v in paths.items()})

    update_manifest(
        out_dir,
        {
            "regime_classifier": regime_meta,
            "bias_correction": corr["meta"],
            "heavy_rain_probability": prob_meta,
            "prediction_intervals": interval_meta,
            "split": split.summary(),
            "verification_report": {k: str(v) for k, v in paths.items()},
            "pipeline_complete": True,
        },
    )
    return report


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list, or None to read ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = add_common_arguments(
        argparse.ArgumentParser(description="Run the full training + verification pipeline.")
    )
    parser.add_argument("--use-rule-labels", action="store_true")
    parser.add_argument("--min-rows-per-regime", type=int, default=200)
    parser.add_argument(
        "--probability-backend", default="lightgbm", choices=["lightgbm", "xgboost"]
    )
    parser.add_argument(
        "--calibration", default="isotonic", choices=["isotonic", "sigmoid", "none"]
    )
    args = parser.parse_args(argv)
    configure_logging()

    report = run_pipeline(args)
    print()
    print(f"Verification complete over {report['n_test_rows']:,} held-out test rows "
          f"({report['test_period']['start']} to {report['test_period']['end']}).")
    print("See artifacts/verification_report.md for the full comparison table.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
