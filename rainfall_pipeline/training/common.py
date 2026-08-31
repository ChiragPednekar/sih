"""Shared helpers for the training scripts.

Keeps data loading, splitting and artifact naming in one place so the four
scripts in this package stay short and cannot drift apart in how they prepare
data.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ..config.regions import ARTIFACT_DIR, ensure_dirs
from ..config.thresholds import SPLIT
from ..data.store import build_analysis_table, read_table, table_exists
from ..features.engineering import add_derived_features, fit_climatology
from ..verification.splits import Split, chronological_split, resolve_split_bounds

LOGGER = logging.getLogger("rainfall_pipeline.training")

#: Name of the cached analysis table in the store.
ANALYSIS_TABLE = "analysis_table"

#: Filename of the climatology used for the anomaly features.
CLIMATOLOGY_FILENAME = "climatology.parquet"

#: Filename of the manifest recording what was trained, when, and on what.
MANIFEST_FILENAME = "training_manifest.json"


def configure_logging(verbose: bool = True) -> None:
    """Set up console logging for a training run.

    Args:
        verbose: Whether to log at DEBUG rather than INFO.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the data-path and split arguments every training script accepts.

    Args:
        parser: The parser to extend.

    Returns:
        The same parser, for chaining.
    """
    parser.add_argument("--era5-path", default=None, help="Override the ERA5 file path.")
    parser.add_argument("--observed-path", default=None, help="Override the observed rainfall file path.")
    parser.add_argument("--nwp-path", default=None, help="Override the raw NWP forecast file path.")
    parser.add_argument("--start-date", default=None, help="Inclusive lower date bound (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Inclusive upper date bound (YYYY-MM-DD).")
    parser.add_argument("--train-end", default=None, help="Last date of the training split (YYYY-MM-DD).")
    parser.add_argument("--val-end", default=None, help="Last date of the validation split (YYYY-MM-DD).")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-read the raw files even if a cached analysis table exists.",
    )
    parser.add_argument(
        "--backend",
        default="xgboost",
        choices=["xgboost", "catboost", "lightgbm", "ensemble"],
        help="Gradient boosting library for the regime and correction models.",
    )
    parser.add_argument(
        "--loss",
        default="mse",
        choices=["mse", "tweedie", "huber"],
        help="Loss objective for residual models (mse, tweedie, huber).",
    )
    parser.add_argument("--artifact-dir", default=None, help="Where to write model artifacts.")
    return parser


def artifact_dir(override: Optional[str] = None) -> Path:
    """Return the artifact directory, creating it if necessary.

    Args:
        override: An explicit directory, or None to use the configured default.

    Returns:
        The directory path.
    """
    path = Path(override) if override else ARTIFACT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_analysis_table(args: argparse.Namespace) -> pd.DataFrame:
    """Load the analysis table, from cache when possible.

    >>> PLUG IN YOUR REAL DATA HERE <<<
    This is where every training script first touches your data. If the files
    are not present, the loaders raise with an explanation of what is needed.

    Args:
        args: Parsed command-line arguments from :func:`add_common_arguments`.

    Returns:
        The analysis-ready feature table.

    Raises:
        MissingDataError: If the raw files are absent and no cache exists.
    """
    if not args.rebuild and table_exists(ANALYSIS_TABLE):
        LOGGER.info("Reading cached analysis table from the store.")
        return read_table(ANALYSIS_TABLE, start_date=args.start_date, end_date=args.end_date)

    LOGGER.info("Building the analysis table from the raw files.")
    return build_analysis_table(
        era5_path=args.era5_path,
        observed_path=args.observed_path,
        nwp_path=args.nwp_path,
        start_date=args.start_date,
        end_date=args.end_date,
    )


def make_split(df: pd.DataFrame, args: argparse.Namespace) -> Split:
    """Split the table chronologically, honouring CLI and config settings.

    Precedence: command-line flags, then
    :data:`~rainfall_pipeline.config.thresholds.SPLIT`, then the fraction
    fallback.

    Args:
        df: The analysis table.
        args: Parsed command-line arguments.

    Returns:
        The chronological split.

    Raises:
        SplitError: If the boundaries leave a split empty.
    """
    cfg_train, cfg_val = resolve_split_bounds(SPLIT.train_end, SPLIT.val_end)
    return chronological_split(
        df,
        train_end=args.train_end or cfg_train,
        val_end=args.val_end or cfg_val,
    )


def build_climatology(train_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Fit the climatology on training rows only and persist it.

    Fitting on the training split alone is what stops the anomaly features from
    peeking at the validation and test periods.

    Args:
        train_df: The training split.
        out_dir: Artifact directory.

    Returns:
        The climatology table.
    """
    clim = fit_climatology(train_df)
    path = out_dir / CLIMATOLOGY_FILENAME
    clim.to_parquet(path, index=False)
    LOGGER.info("Wrote climatology (%d rows) to %s", len(clim), path)
    return clim


def load_climatology(out_dir: Path) -> Optional[pd.DataFrame]:
    """Load a previously fitted climatology.

    Args:
        out_dir: Artifact directory.

    Returns:
        The climatology table, or None if it has not been fitted yet.
    """
    path = Path(out_dir) / CLIMATOLOGY_FILENAME
    return pd.read_parquet(path) if path.exists() else None


def apply_climatology(df: pd.DataFrame, climatology: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Recompute derived features with a climatology attached.

    Args:
        df: A feature table.
        climatology: Output of :func:`build_climatology`, or None.

    Returns:
        The frame with anomaly features filled in.
    """
    return add_derived_features(df, climatology=climatology)


def update_manifest(out_dir: Path, entries: Dict[str, Any]) -> Path:
    """Merge ``entries`` into the training manifest.

    The manifest is what the API reads to decide whether a usable set of
    artifacts exists, and what the report cites for provenance.

    Args:
        out_dir: Artifact directory.
        entries: Keys to add or overwrite.

    Returns:
        The manifest path.
    """
    ensure_dirs()
    path = Path(out_dir) / MANIFEST_FILENAME
    manifest: Dict[str, Any] = {}
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LOGGER.warning("Manifest at %s is corrupt; starting a fresh one.", path)
    manifest.update(entries)
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def read_manifest(out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Read the training manifest.

    Args:
        out_dir: Artifact directory. Defaults to the configured one.

    Returns:
        The manifest dict, or an empty dict if none exists.
    """
    path = Path(out_dir or ARTIFACT_DIR) / MANIFEST_FILENAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
