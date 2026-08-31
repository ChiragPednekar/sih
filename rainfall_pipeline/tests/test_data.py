"""Tests for the data contract and the loader stubs.

The most important behaviour under test: a missing or malformed file must raise
a clear error, never silently produce data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rainfall_pipeline.data import schema as sch
from rainfall_pipeline.data.loaders import (
    MissingDataError,
    _dummy_dataframe,
    load_era5,
    load_observed_rainfall,
    load_raw_nwp_forecast,
)
from rainfall_pipeline.data.store import StoreEmptyError, read_table, write_table


def test_dummy_dataframe_matches_common_schema() -> None:
    """The dummy rows must satisfy the schema the whole pipeline codes against."""
    df = _dummy_dataframe()
    assert list(df.columns) == sch.COMMON_SCHEMA
    assert len(df) == 8
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


@pytest.mark.parametrize(
    "loader", [load_era5, load_observed_rainfall, load_raw_nwp_forecast]
)
def test_missing_file_raises_with_expected_format(loader, tmp_path: Path) -> None:
    """A missing file must explain what to prepare, not fabricate data."""
    with pytest.raises(MissingDataError) as exc:
        loader(tmp_path / "nope.parquet")
    message = str(exc.value)
    assert "does not exist" in message
    assert "No synthetic data will be generated" in message
    assert "Required columns" in message
    assert "date" in message and "lat" in message and "lon" in message


def test_empty_file_raises(tmp_path: Path) -> None:
    """A zero-byte file is treated as missing data, not as an empty result."""
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(MissingDataError, match="is empty"):
        load_era5(path)


def test_missing_column_names_the_column(tmp_path: Path) -> None:
    """A file missing a required column must say which one."""
    path = tmp_path / "partial.csv"
    _dummy_dataframe().drop(columns=["cape"]).to_csv(path, index=False)
    with pytest.raises(sch.SchemaError, match="cape"):
        load_era5(path)


def test_loader_filters_by_date_and_bbox(source_files) -> None:
    """Date and bounding-box filters must actually restrict the rows returned."""
    from rainfall_pipeline.config.regions import BBox

    full = load_era5(source_files["era5"])
    assert len(full) == 8

    one_day = load_era5(source_files["era5"], start_date="2020-07-02", end_date="2020-07-02")
    assert set(one_day["date"].dt.strftime("%Y-%m-%d")) == {"2020-07-02"}

    narrow = load_era5(source_files["era5"], bbox=BBox(18.0, 20.0, 72.0, 74.0))
    assert narrow["lat"].between(18.0, 20.0).all()


def test_loader_raises_when_filters_exclude_everything(source_files) -> None:
    """An empty filter result must explain itself rather than returning nothing."""
    with pytest.raises(MissingDataError, match="no rows fall inside"):
        load_era5(source_files["era5"], start_date="2030-01-01")


def test_store_roundtrip(tmp_path: Path, monkeypatch, dummy_raw) -> None:
    """A table written to the store must come back with the same shape."""
    import rainfall_pipeline.data.store as store

    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    monkeypatch.setattr(store, "ensure_dirs", lambda: None)
    write_table(dummy_raw, "unit_test_table")
    back = read_table("unit_test_table")
    assert len(back) == len(dummy_raw)
    assert pd.api.types.is_datetime64_any_dtype(back["date"])


def test_store_missing_table_raises(tmp_path: Path, monkeypatch) -> None:
    """Reading a table that was never written must point at the ingestion step."""
    import rainfall_pipeline.data.store as store

    monkeypatch.setattr(store, "STORE_DIR", tmp_path)
    with pytest.raises(StoreEmptyError, match="run_full_training_pipeline"):
        read_table("never_written")
