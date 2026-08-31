"""Geographic configuration: bounding boxes, grid settings and file locations.

Nothing here downloads or fetches anything. The paths below are simply where
the pipeline *expects* to find the files you will add later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

#: Repository root (the directory containing the ``rainfall_pipeline`` package).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Where you will drop the raw input files. Override with the RAINFALL_DATA_DIR
#: environment variable if your data lives elsewhere.
DATA_DIR = Path(os.environ.get("RAINFALL_DATA_DIR", PROJECT_ROOT / "data_store"))

#: Where the cached/aligned analysis table and the trained model artifacts go.
STORE_DIR = Path(os.environ.get("RAINFALL_STORE_DIR", DATA_DIR / "store"))
ARTIFACT_DIR = Path(os.environ.get("RAINFALL_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))

# ---------------------------------------------------------------------------
# >>> PLUG IN YOUR REAL DATA HERE <<<
# Expected default locations for the four raw inputs. See data/README.md for
# the exact column contract each file must satisfy. Any of these may also be
# passed explicitly to the loader functions instead of relying on the default.
# ---------------------------------------------------------------------------
ERA5_PATH = Path(os.environ.get("RAINFALL_ERA5_PATH", DATA_DIR / "era5.parquet"))
OBSERVED_PATH = Path(os.environ.get("RAINFALL_OBSERVED_PATH", DATA_DIR / "observed_rainfall.parquet"))
NWP_PATH = Path(os.environ.get("RAINFALL_NWP_PATH", DATA_DIR / "raw_nwp_forecast.parquet"))
DISTRICTS_PATH = Path(os.environ.get("RAINFALL_DISTRICTS_PATH", DATA_DIR / "districts.geojson"))

#: Cached analysis-ready feature table written by ``data.store``.
ANALYSIS_TABLE_PATH = STORE_DIR / "analysis_table.parquet"

#: Verification report written by ``verification.report``.
VERIFICATION_REPORT_PATH = ARTIFACT_DIR / "verification_report.json"
VERIFICATION_MARKDOWN_PATH = ARTIFACT_DIR / "verification_report.md"
VERIFICATION_HTML_PATH = ARTIFACT_DIR / "verification_report.html"


@dataclass(frozen=True)
class BBox:
    """Geographic bounding box in degrees.

    Attributes:
        lat_min: Southern edge.
        lat_max: Northern edge.
        lon_min: Western edge.
        lon_max: Eastern edge.
    """

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return ``(lat_min, lat_max, lon_min, lon_max)``."""
        return (self.lat_min, self.lat_max, self.lon_min, self.lon_max)

    def contains(self, lat: float, lon: float) -> bool:
        """Return True if ``(lat, lon)`` falls inside the box (edges included)."""
        return (
            self.lat_min <= lat <= self.lat_max
            and self.lon_min <= lon <= self.lon_max
        )


#: Whole-of-India domain, used as the default filter for every loader.
INDIA = BBox(lat_min=6.0, lat_max=38.0, lon_min=66.0, lon_max=98.0)

#: Named sub-regions, useful for breaking verification results down by region.
#: These are coarse rectangles, not official boundaries -- for anything that
#: must be authoritative, use the district shapefile instead.
SUBREGIONS: Dict[str, BBox] = {
    "west_coast": BBox(8.0, 21.0, 72.0, 76.5),
    "central_india": BBox(18.0, 26.0, 74.0, 86.0),
    "northeast": BBox(22.0, 29.5, 88.0, 97.0),
    "north_india": BBox(26.0, 35.0, 73.0, 82.0),
    "peninsular_india": BBox(8.0, 18.0, 74.0, 84.0),
}

#: Nominal grid spacing in degrees. Used only for documenting the FSS
#: neighbourhood sizes and for the district spatial join tolerance -- update it
#: to match whatever grid the real data actually uses.
GRID_SPACING_DEG: float = 0.25

#: Column name used throughout for the district identifier.
DISTRICT_COLUMN = "district"

#: Column in the district GeoDataFrame holding the district name. Change this
#: to whatever your shapefile actually uses (commonly ``DISTRICT``, ``dtname``
#: or ``NAME_2`` depending on the source).
DISTRICT_NAME_FIELD = os.environ.get("RAINFALL_DISTRICT_NAME_FIELD", "district")

#: CRS every geometry is normalised to before the spatial join.
GEO_CRS = "EPSG:4326"


def ensure_dirs() -> None:
    """Create the store and artifact directories if they do not exist."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
