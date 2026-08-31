"""Train the calibrated heavy-rain probability head (Stage 4).

Run after the bias-correction models:

    python -m rainfall_pipeline.training.train_heavy_rain_models

The classifiers are fitted on the training split and the isotonic/Platt
calibrators on the chronologically later validation split, so the calibration
map is never fitted on data the classifier has already seen.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ..config.thresholds import RAIN_THRESHOLDS
from ..models.bias_correction import BIAS_MODEL_FILENAME, RegimeBiasCorrector
from ..models.heavy_rain_probability import (
    PROBABILITY_MODEL_FILENAME,
    HeavyRainProbabilityModel,
    attach_corrected_forecast,
    binary_target,
)
from ..models.regime_classifier import REGIME_MODEL_FILENAME, RegimeClassifier, label_regimes
from .common import (
    LOGGER,
    add_common_arguments,
    apply_climatology,
    artifact_dir,
    configure_logging,
    load_analysis_table,
    load_climatology,
    make_split,
    update_manifest,
)


def train_probability_head(
    train_df: pd.DataFrame,
    calib_df: Optional[pd.DataFrame],
    *,
    backend: str = "lightgbm",
    method: str = "isotonic",
    params: Optional[Dict[str, Any]] = None,
    out_dir: Optional[Path] = None,
) -> tuple[HeavyRainProbabilityModel, Dict[str, Any]]:
    """Fit one calibrated classifier per threshold.

    Args:
        train_df: Training rows with the corrected forecast attached.
        calib_df: Chronologically later calibration rows, also with the
            corrected forecast attached. May be None.
        backend: ``"lightgbm"`` or ``"xgboost"``.
        method: ``"isotonic"``, ``"sigmoid"`` or ``"none"``.
        params: Extra estimator parameters.
        out_dir: Where to save the artifact. Not saved if None.

    Returns:
        ``(fitted_model, metadata)``.
    """
    base_rates = {
        name: float(binary_target(train_df, mm).mean(skipna=True))
        for name, mm in RAIN_THRESHOLDS.items()
    }
    LOGGER.info("Training-split exceedance base rates: %s", base_rates)

    model = HeavyRainProbabilityModel(
        backend=backend, method=method, params=params
    ).fit(train_df, calib_df)

    degenerate = model.degenerate_thresholds()
    if degenerate:
        LOGGER.warning(
            "Threshold(s) %s had no variation in the training split, so those "
            "models predict a constant base rate rather than a real "
            "probability. Extend the training period if you need them.",
            ", ".join(degenerate),
        )

    meta: Dict[str, Any] = {
        "backend": backend,
        "calibration_method": method,
        "thresholds_mm": dict(RAIN_THRESHOLDS),
        "training_base_rates": base_rates,
        "degenerate_thresholds": degenerate,
        "n_training_rows": int(len(train_df)),
        "n_calibration_rows": int(len(calib_df)) if calib_df is not None else 0,
        "calibration_fitted_on_holdout": calib_df is not None and not calib_df.empty,
    }
    if out_dir is not None:
        path = model.save(Path(out_dir) / PROBABILITY_MODEL_FILENAME)
        meta["artifact"] = str(path)
        LOGGER.info("Wrote heavy-rain probability models to %s", path)
    return model, meta


def prepare_with_corrections(
    df: pd.DataFrame,
    corrector: RegimeBiasCorrector,
    classifier: Optional[RegimeClassifier],
) -> pd.DataFrame:
    """Attach regime labels and the corrected forecast to ``df``.

    Args:
        df: A feature table with derived features.
        corrector: The fitted regime-specific corrector.
        classifier: The fitted regime classifier, or None to use rule labels.

    Returns:
        The frame with ``regime`` and ``corrected_forecast_mm`` added.
    """
    regimes = classifier.predict(df) if classifier is not None else label_regimes(df)
    out = df.copy()
    out["regime"] = regimes.values
    return attach_corrected_forecast(out, corrector.predict(df, regimes))


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list, or None to read ``sys.argv``.

    Returns:
        Process exit code, 1 if a prerequisite artifact is missing.
    """
    parser = add_common_arguments(
        argparse.ArgumentParser(description="Train the calibrated heavy-rain probability head.")
    )
    parser.add_argument(
        "--probability-backend", default="lightgbm", choices=["lightgbm", "xgboost"]
    )
    parser.add_argument(
        "--calibration",
        default="isotonic",
        choices=["isotonic", "sigmoid", "none"],
        help="Probability calibration method.",
    )
    args = parser.parse_args(argv)
    configure_logging()

    out_dir = artifact_dir(args.artifact_dir)
    bias_path = out_dir / BIAS_MODEL_FILENAME
    if not bias_path.exists():
        LOGGER.error(
            "No bias-correction artifact at %s. Run "
            "training/train_bias_correction.py first.",
            bias_path,
        )
        return 1

    corrector = RegimeBiasCorrector.load(bias_path)
    clf_path = out_dir / REGIME_MODEL_FILENAME
    classifier = RegimeClassifier.load(clf_path) if clf_path.exists() else None

    table = load_analysis_table(args)
    split = make_split(table, args)
    clim = load_climatology(out_dir)

    train_df = prepare_with_corrections(
        apply_climatology(split.train, clim), corrector, classifier
    )
    calib_df = None
    if not split.validation.empty:
        calib_df = prepare_with_corrections(
            apply_climatology(split.validation, clim), corrector, classifier
        )
    else:
        LOGGER.warning(
            "Validation split is empty, so calibration will be fitted on the "
            "training predictions. Those probabilities will look better "
            "calibrated than they are -- widen the data period."
        )

    _, meta = train_probability_head(
        train_df,
        calib_df,
        backend=args.probability_backend,
        method=args.calibration,
        out_dir=out_dir,
    )
    update_manifest(out_dir, {"heavy_rain_probability": meta, "split": split.summary()})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
