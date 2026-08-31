"""Shared pytest fixtures.

Every fixture here is built from
:func:`rainfall_pipeline.data.loaders._dummy_dataframe` -- 8 hand-written rows
whose only purpose is to prove the pipeline is wired together correctly.

Nothing in this test suite asserts a level of skill, accuracy or improvement,
because no such assertion would be meaningful on 8 fabricated rows. The tests
check shapes, contracts, error messages and the absence of leakage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rainfall_pipeline.data import schema as sch
from rainfall_pipeline.data.loaders import _dummy_dataframe
from rainfall_pipeline.features.engineering import add_derived_features, fit_climatology
from rainfall_pipeline.models.regime_classifier import label_regimes

#: Small model sizes so the suite stays fast; these are test settings, not
#: recommended hyperparameters.
FAST_PARAMS = {"n_estimators": 10, "max_depth": 3}


@pytest.fixture
def dummy_raw() -> pd.DataFrame:
    """The raw 8-row dummy table in the common schema."""
    return _dummy_dataframe()


@pytest.fixture
def dummy_features(dummy_raw: pd.DataFrame) -> pd.DataFrame:
    """The dummy table with derived features and anomalies attached."""
    return add_derived_features(dummy_raw, climatology=fit_climatology(dummy_raw))


@pytest.fixture
def dummy_regimes(dummy_features: pd.DataFrame) -> pd.Series:
    """Rule-based regime labels for the dummy rows."""
    return label_regimes(dummy_features)


@pytest.fixture
def source_files(tmp_path: Path, dummy_raw: pd.DataFrame) -> dict[str, Path]:
    """Write the dummy rows out as the three separate source files.

    Args:
        tmp_path: pytest's per-test temporary directory.
        dummy_raw: The dummy table.

    Returns:
        ``{"era5": path, "observed": path, "nwp": path}``.
    """
    era5 = tmp_path / "era5.parquet"
    observed = tmp_path / "observed.parquet"
    nwp = tmp_path / "nwp.parquet"

    era5_cols = sch.KEY_COLUMNS + sch.ATMOSPHERIC_COLUMNS + sch.STATIC_COLUMNS + [sch.DISTRICT_COLUMN]
    dummy_raw[era5_cols].to_parquet(era5, index=False)
    dummy_raw[sch.KEY_COLUMNS + [sch.OBSERVED_COLUMN]].to_parquet(observed, index=False)
    dummy_raw[sch.KEY_COLUMNS + [sch.FORECAST_COLUMN]].to_parquet(nwp, index=False)
    return {"era5": era5, "observed": observed, "nwp": nwp}


@pytest.fixture
def district_polygons():
    """Four square polygons covering the dummy grid cells.

    These are rectangles drawn around the dummy coordinates so the spatial join
    has something to hit. They are not real district boundaries.
    """
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {
            sch.DISTRICT_COLUMN: pd.Series(
                ["Pune", "Bhopal", "Patna", "Wayanad"], dtype="string"
            ),
            "geometry": [
                box(72.5, 18.5, 73.5, 19.5),
                box(77.5, 21.5, 78.5, 22.5),
                box(84.5, 24.5, 85.5, 25.5),
                box(76.0, 10.5, 77.0, 11.5),
            ],
        },
        crs="EPSG:4326",
    )
