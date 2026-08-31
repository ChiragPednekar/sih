"""Train the regime classifier (Stage 2).

Run after adding your data files:

    python -m rainfall_pipeline.training.train_regime_classifier

Against real data this fits an XGBoost/CatBoost multi-class model on the
rule-based regime labels and writes ``regime_classifier.joblib``. Against the
tiny dummy frame used in the tests it only proves the code path executes -- the
resulting model is meaningless and must never be used for anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ..models.regime_classifier import (
    REGIME_MODEL_FILENAME,
    RegimeClassifier,
    label_regimes,
    regime_label_summary,
)
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


def train_regime_classifier(
    train_df: pd.DataFrame,
    *,
    backend: str = "xgboost",
    params: Optional[Dict[str, Any]] = None,
    out_dir: Optional[Path] = None,
) -> tuple[RegimeClassifier, Dict[str, Any]]:
    """Label the training rows by rule and fit a classifier on those labels.

    Args:
        train_df: Training split, with anomaly features already attached.
        backend: ``"xgboost"`` or ``"catboost"``.
        params: Extra estimator parameters.
        out_dir: Where to save the artifact. Not saved if None.

    Returns:
        ``(fitted_classifier, metadata)`` where metadata records the label
        distribution and the artifact path.

    Raises:
        ValueError: If the rules produce fewer than two distinct regimes.
    """
    labels = label_regimes(train_df)
    counts = regime_label_summary(labels)
    LOGGER.info("Rule-based regime distribution (training split): %s", counts)

    empty = [name for name, n in counts.items() if n == 0]
    if empty:
        LOGGER.warning(
            "No training rows for regime(s): %s. Those regimes will route to the "
            "global fallback in the correction stage. If this is unexpected, "
            "revisit RegimeRuleConfig in config/thresholds.py.",
            ", ".join(empty),
        )

    clf = RegimeClassifier(backend=backend, params=params).fit(train_df, labels)

    meta: Dict[str, Any] = {
        "backend": backend,
        "label_counts": counts,
        "n_training_rows": int(len(train_df)),
        "feature_columns": clf.feature_columns,
    }
    if out_dir is not None:
        path = clf.save(Path(out_dir) / REGIME_MODEL_FILENAME)
        meta["artifact"] = str(path)
        LOGGER.info("Wrote regime classifier to %s", path)
    return clf, meta


def evaluate_agreement(clf: RegimeClassifier, df: pd.DataFrame) -> Dict[str, float]:
    """Measure how well the classifier reproduces the rule labels.

    This is not a skill score against reality -- the rules are themselves a
    simplification. It only answers "did the model successfully distil the
    rule?". A low value means the atmospheric features do not carry enough
    information to reconstruct the rule, which is worth knowing before the
    router is trusted.

    Args:
        clf: A fitted classifier.
        df: Rows to check, with anomaly features attached.

    Returns:
        ``{"accuracy": ..., "macro_f1": ...}``.
    """
    from sklearn.metrics import accuracy_score, f1_score

    truth = label_regimes(df)
    pred = clf.predict(df)
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
    }


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list, or None to read ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = add_common_arguments(
        argparse.ArgumentParser(description="Train the monsoon regime classifier.")
    )
    args = parser.parse_args(argv)
    configure_logging()

    out_dir = artifact_dir(args.artifact_dir)
    table = load_analysis_table(args)
    split = make_split(table, args)
    LOGGER.info("Split: %s", split.summary())

    clim = build_climatology(split.train, out_dir)
    train_df = apply_climatology(split.train, clim)
    val_df = apply_climatology(split.validation, clim) if not split.validation.empty else None

    clf, meta = train_regime_classifier(
        train_df, backend=args.backend, out_dir=out_dir
    )
    meta["train_agreement"] = evaluate_agreement(clf, train_df)
    if val_df is not None and not val_df.empty:
        meta["validation_agreement"] = evaluate_agreement(clf, val_df)
        LOGGER.info("Validation rule-agreement: %s", meta["validation_agreement"])

    update_manifest(out_dir, {"regime_classifier": meta, "split": split.summary()})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
