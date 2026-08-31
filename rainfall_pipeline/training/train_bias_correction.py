"""Train the bias-correction models and the two ML/statistical baselines (Stage 3).

Run after the regime classifier:

    python -m rainfall_pipeline.training.train_bias_correction

Fits, and saves:

* Baseline B -- one global regime-blind residual model
* Baseline C -- global empirical quantile mapping, plus a per-regime variant
* Model D -- one residual model per regime

Baseline A needs no fitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ..models.baselines import (
    GlobalBiasCorrector,
    QuantileMapping,
    RegimeQuantileMapping,
    residual_target,
)
from ..models.bias_correction import BIAS_MODEL_FILENAME, RegimeBiasCorrector
from ..models.regime_classifier import REGIME_MODEL_FILENAME, RegimeClassifier, label_regimes
from .common import (
    LOGGER,
    add_common_arguments,
    apply_climatology,
    artifact_dir,
    build_climatology,
    configure_logging,
    load_analysis_table,
    load_climatology,
    make_split,
    update_manifest,
)

#: Artifact filenames for the baselines.
GLOBAL_MODEL_FILENAME = "baseline_b_global_ml.joblib"
QUANTILE_MAP_FILENAME = "baseline_c_quantile_mapping.joblib"
REGIME_QUANTILE_MAP_FILENAME = "baseline_c_regime_quantile_mapping.joblib"


def resolve_regimes(
    df: pd.DataFrame, classifier: Optional[RegimeClassifier], *, use_rules: bool
) -> pd.Series:
    """Get a regime label per row, from the rules or from the classifier.

    Which one to use is a real modelling choice. Training the correctors on
    *rule* labels gives each model a clean, physically-defined subset. Routing
    at prediction time necessarily uses the *classifier*, since the rules need
    the observation. Training on rule labels and predicting with classifier
    labels means the corrector sees a slightly different mix in production than
    it did in training -- which is exactly what
    ``train_regime_classifier.evaluate_agreement`` quantifies.

    Args:
        df: Feature table.
        classifier: A fitted classifier, or None.
        use_rules: If True, use the rule labeller.

    Returns:
        Regime label per row.

    Raises:
        ValueError: If ``use_rules`` is False and no classifier was supplied.
    """
    if use_rules:
        return label_regimes(df)
    if classifier is None:
        raise ValueError(
            "No regime classifier available. Run "
            "training/train_regime_classifier.py first, or pass --use-rule-labels."
        )
    return classifier.predict(df)


def train_all_correctors(
    train_df: pd.DataFrame,
    regimes: pd.Series,
    *,
    backend: str = "xgboost",
    loss: str = "mse",
    params: Optional[Dict[str, Any]] = None,
    min_rows_per_regime: int = 200,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fit Baseline B, Baseline C (global + per regime) and Model D.

    Args:
        train_df: Training split with derived features.
        regimes: Regime label per training row.
        backend: ``"xgboost"``, ``"catboost"``, ``"lightgbm"``, or ``"ensemble"``.
        loss: Loss objective (``"mse"``, ``"tweedie"``, ``"huber"``).
        params: Extra estimator parameters.
        min_rows_per_regime: Threshold below which a regime uses the fallback.
        out_dir: Where to save artifacts. Not saved if None.

    Returns:
        ``{"models": {...fitted objects...}, "meta": {...}}``.
    """
    target = residual_target(train_df)
    LOGGER.info(
        "Residual target over %d training rows: mean %.3f mm, sd %.3f mm",
        len(target), float(target.mean()), float(target.std()),
    )

    LOGGER.info("Fitting Baseline B (global ML residual model with backend=%s, loss=%s)...", backend, loss)
    global_ml = GlobalBiasCorrector(backend=backend, loss=loss, params=params).fit(train_df, target)

    LOGGER.info("Fitting Baseline C (global quantile mapping)...")
    qm = QuantileMapping().fit(train_df)

    LOGGER.info("Fitting per-regime quantile mapping (ablation)...")
    regime_qm = RegimeQuantileMapping().fit(train_df, regimes)

    LOGGER.info("Fitting Model D (regime-specific residual models with backend=%s, loss=%s)...", backend, loss)
    regime_ml = RegimeBiasCorrector(
        backend=backend, loss=loss, params=params, min_rows_per_regime=min_rows_per_regime
    ).fit(train_df, regimes, target)
    LOGGER.info(
        "Per-regime training counts: %s (own model for: %s)",
        regime_ml.training_counts, sorted(regime_ml.models),
    )

    meta: Dict[str, Any] = {
        "backend": backend,
        "loss": loss,
        "n_training_rows": int(len(train_df)),
        "min_rows_per_regime": min_rows_per_regime,
        "regime_training_counts": regime_ml.training_counts,
        "regimes_with_own_model": sorted(regime_ml.models),
        "regimes_using_fallback": sorted(
            set(regime_ml.training_counts) - set(regime_ml.models)
        ),
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        meta["artifacts"] = {
            "baseline_b": str(global_ml.save(out_dir / GLOBAL_MODEL_FILENAME)),
            "baseline_c": str(qm.save(out_dir / QUANTILE_MAP_FILENAME)),
            "model_d": str(regime_ml.save(out_dir / BIAS_MODEL_FILENAME)),
        }
        import joblib

        path = out_dir / REGIME_QUANTILE_MAP_FILENAME
        joblib.dump(regime_qm, path)
        meta["artifacts"]["baseline_c_per_regime"] = str(path)
        LOGGER.info("Wrote correction artifacts to %s", out_dir)

    return {
        "models": {
            "baseline_b": global_ml,
            "baseline_c": qm,
            "baseline_c_per_regime": regime_qm,
            "model_d": regime_ml,
        },
        "meta": meta,
    }


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list, or None to read ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = add_common_arguments(
        argparse.ArgumentParser(description="Train regime-specific bias correction and baselines.")
    )
    parser.add_argument(
        "--loss",
        default="mse",
        choices=["mse", "tweedie", "huber"],
        help="Loss objective for residual models (mse, tweedie, huber).",
    )
    parser.add_argument(
        "--use-rule-labels",
        action="store_true",
        help="Train on rule-based regime labels instead of classifier predictions.",
    )
    parser.add_argument(
        "--min-rows-per-regime",
        type=int,
        default=200,
        help="Regimes with fewer training rows fall back to the global model.",
    )
    args = parser.parse_args(argv)
    configure_logging()

    out_dir = artifact_dir(args.artifact_dir)
    table = load_analysis_table(args)
    split = make_split(table, args)
    LOGGER.info("Split: %s", split.summary())

    clim = load_climatology(out_dir)
    if clim is None:
        LOGGER.info("No saved climatology found; fitting one on the training split.")
        clim = build_climatology(split.train, out_dir)
    train_df = apply_climatology(split.train, clim)

    classifier: Optional[RegimeClassifier] = None
    clf_path = out_dir / REGIME_MODEL_FILENAME
    if clf_path.exists():
        classifier = RegimeClassifier.load(clf_path)
    elif not args.use_rule_labels:
        LOGGER.warning(
            "No regime classifier at %s; falling back to rule labels for training.",
            clf_path,
        )

    regimes = resolve_regimes(
        train_df, classifier, use_rules=args.use_rule_labels or classifier is None
    )

    result = train_all_correctors(
        train_df,
        regimes,
        backend=args.backend,
        loss=args.loss,
        min_rows_per_regime=args.min_rows_per_regime,
        out_dir=out_dir,
    )
    update_manifest(out_dir, {"bias_correction": result["meta"], "split": split.summary()})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
