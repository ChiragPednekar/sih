"""Local persistence for loaded data, so re-runs do not re-parse raw files.

Two backends are available and both write inside ``STORE_DIR``:

* **Parquet** (default) -- one file per table, fastest for the whole-table reads
  the training scripts do.
* **SQLite** -- useful when you want to query subsets by date/district without
  loading everything, and for the API's point lookups.

The store is a cache, not a source of truth. Deleting ``STORE_DIR`` only costs
you the time to re-read the raw files.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ..config.regions import STORE_DIR, ensure_dirs
from . import schema as sch

PathLike = Union[str, Path]

#: Default SQLite database file.
SQLITE_PATH = STORE_DIR / "rainfall.sqlite"


class StoreEmptyError(FileNotFoundError):
    """Raised when a table is requested from the store but has never been written."""


def _parquet_path(table: str) -> Path:
    """Return the Parquet file path for ``table``."""
    return STORE_DIR / f"{table}.parquet"


def write_table(df: pd.DataFrame, table: str, *, backend: str = "parquet") -> Path:
    """Persist ``df`` under the name ``table``.

    Args:
        df: Frame to persist. Written as-is; validate before calling.
        table: Logical table name, e.g. ``"analysis_table"``.
        backend: ``"parquet"`` or ``"sqlite"``.

    Returns:
        The path written to.

    Raises:
        ValueError: If ``backend`` is unknown.
    """
    ensure_dirs()
    if backend == "parquet":
        path = _parquet_path(table)
        df.to_parquet(path, index=False)
        return path
    if backend == "sqlite":
        with sqlite3.connect(SQLITE_PATH) as conn:
            out = df.copy()
            if "date" in out.columns:
                # SQLite has no date type; store ISO strings and parse on read.
                out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
            out.to_sql(table, conn, if_exists="replace", index=False)
        return SQLITE_PATH
    raise ValueError(f"Unknown storage backend '{backend}'. Use 'parquet' or 'sqlite'.")


def read_table(
    table: str,
    *,
    backend: str = "parquet",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    district: Optional[str] = None,
) -> pd.DataFrame:
    """Read a persisted table back, optionally filtered.

    Args:
        table: Logical table name.
        backend: ``"parquet"`` or ``"sqlite"``.
        start_date: Inclusive lower date bound.
        end_date: Inclusive upper date bound.
        district: Restrict to a single district.

    Returns:
        The stored frame, with ``date`` parsed back to datetime64.

    Raises:
        StoreEmptyError: If the table has not been written yet.
        ValueError: If ``backend`` is unknown.
    """
    if backend == "parquet":
        path = _parquet_path(table)
        if not path.exists():
            raise StoreEmptyError(
                f"No cached table '{table}' at '{path}'. Run the ingestion step "
                f"(training/run_full_training_pipeline.py, or "
                f"rainfall_pipeline.data.store.build_analysis_table) after adding "
                f"your raw data files."
            )
        df = pd.read_parquet(path)
    elif backend == "sqlite":
        if not SQLITE_PATH.exists():
            raise StoreEmptyError(f"No SQLite store at '{SQLITE_PATH}'.")
        with sqlite3.connect(SQLITE_PATH) as conn:
            try:
                df = pd.read_sql(f"SELECT * FROM {table}", conn)  # noqa: S608 - table name is internal
            except pd.errors.DatabaseError as exc:
                raise StoreEmptyError(
                    f"Table '{table}' does not exist in '{SQLITE_PATH}'."
                ) from exc
    else:
        raise ValueError(f"Unknown storage backend '{backend}'.")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if start_date is not None:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["date"] <= pd.Timestamp(end_date)]
    if district is not None and sch.DISTRICT_COLUMN in df.columns:
        df = df[df[sch.DISTRICT_COLUMN] == district]
    return df.reset_index(drop=True)


def table_exists(table: str, *, backend: str = "parquet") -> bool:
    """Return True if ``table`` has been written to the store.

    Args:
        table: Logical table name.
        backend: ``"parquet"`` or ``"sqlite"``.

    Returns:
        Whether the table can be read.
    """
    try:
        read_table(table, backend=backend)
        return True
    except (StoreEmptyError, ValueError):
        return False


def build_analysis_table(
    *,
    era5_path: Optional[PathLike] = None,
    observed_path: Optional[PathLike] = None,
    nwp_path: Optional[PathLike] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    bbox=None,
    persist: bool = True,
    backend: str = "parquet",
) -> pd.DataFrame:
    """Load all three tabular sources, join them, and cache the result.

    This is the single entry point the training scripts and the API use to get
    analysis-ready data. It performs no imputation beyond the inner join: a
    ``(date, lat, lon)`` must exist in all three sources to survive.

    >>> PLUG IN YOUR REAL DATA HERE <<<
    Pass explicit paths, or set RAINFALL_ERA5_PATH / RAINFALL_OBSERVED_PATH /
    RAINFALL_NWP_PATH, or drop the files at the defaults in ``config.regions``.

    Args:
        era5_path: Override for the ERA5 file path.
        observed_path: Override for the observed rainfall file path.
        nwp_path: Override for the raw forecast file path.
        start_date: Inclusive lower date bound.
        end_date: Inclusive upper date bound.
        bbox: Spatial filter; defaults to the whole-India domain.
        persist: Whether to write the joined table to the store.
        backend: Storage backend used when ``persist`` is True.

    Returns:
        The joined, feature-engineered analysis table.

    Raises:
        MissingDataError: If any of the three inputs is missing or empty.
    """
    from ..config.regions import ERA5_PATH, INDIA, NWP_PATH, OBSERVED_PATH
    from ..features.engineering import build_feature_table
    from .loaders import load_era5, load_observed_rainfall, load_raw_nwp_forecast

    bbox = INDIA if bbox is None else bbox
    era5 = load_era5(era5_path or ERA5_PATH, start_date, end_date, bbox)
    observed = load_observed_rainfall(observed_path or OBSERVED_PATH, start_date, end_date, bbox)
    nwp = load_raw_nwp_forecast(nwp_path or NWP_PATH, start_date, end_date, bbox)

    table = build_feature_table(era5=era5, observed=observed, nwp=nwp)
    if persist:
        write_table(table, "analysis_table", backend=backend)
    return table
