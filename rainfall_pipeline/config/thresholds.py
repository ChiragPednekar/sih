"""Central configuration for all thresholds used across the pipeline.

Everything that is a tunable meteorological or modelling constant lives here so
it can be adjusted in one place once real data is connected and the official
IMD / IITM definitions are substituted for the simplified placeholders below.

NOTE: none of these values have been validated against real data yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# IMD daily rainfall categories (mm/day).
# These are the published IMD category boundaries and are NOT placeholders.
# ---------------------------------------------------------------------------
HEAVY_MM: float = 64.5
VERY_HEAVY_MM: float = 115.6
EXTREMELY_HEAVY_MM: float = 204.4

#: Ordered mapping of threshold name -> mm/day. The heavy-rain probability head
#: trains one binary classifier per entry.
RAIN_THRESHOLDS: Dict[str, float] = {
    "heavy": HEAVY_MM,
    "very_heavy": VERY_HEAVY_MM,
    "extremely_heavy": EXTREMELY_HEAVY_MM,
}

# ---------------------------------------------------------------------------
# Regime labels.
# ---------------------------------------------------------------------------
REGIME_ACTIVE = "Active"
REGIME_BREAK = "Break"
REGIME_DEPRESSION_LOW = "Depression-Low"
REGIME_COASTAL = "Coastal"
REGIME_OROGRAPHIC = "Orographic"

#: Canonical ordering. The classifier's integer class indices follow this list,
#: so do not reorder it without retraining.
REGIME_LABELS: List[str] = [
    REGIME_ACTIVE,
    REGIME_BREAK,
    REGIME_DEPRESSION_LOW,
    REGIME_COASTAL,
    REGIME_OROGRAPHIC,
]

REGIME_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(REGIME_LABELS)}
INDEX_TO_REGIME: Dict[int, str] = {i: name for name, i in REGIME_TO_INDEX.items()}


# ---------------------------------------------------------------------------
# Rule-based regime labelling thresholds.
#
# PLACEHOLDER VALUES. These are simplified stand-ins for the published
# active/break monsoon criteria (e.g. Rajeevan et al. 2010 uses standardised
# rainfall anomalies over a core monsoon zone; the Webster-Yang / low-level
# westerly indices use 850 hPa zonal wind). Replace the numbers below with the
# official IMD / IITM definitions once the real reference data is available --
# every consumer reads them from this dataclass, so nothing else needs editing.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegimeRuleConfig:
    """Thresholds for the rule-based regime labeller.

    Attributes:
        core_zone_bbox: (lat_min, lat_max, lon_min, lon_max) of the reference
            box over which the rainfall/wind indices are averaged. Defaults to a
            rough Core Monsoon Zone footprint over central India.
        active_rain_anomaly_sd: Standardised daily rainfall anomaly over the
            reference box above which a day is considered "active".
        break_rain_anomaly_sd: Standardised anomaly below which a day is
            considered a "break".
        active_zonal_wind_850: 850 hPa zonal wind (m/s) over the reference box
            above which the low-level westerly jet is treated as strong.
        break_zonal_wind_850: 850 hPa zonal wind (m/s) below which the westerly
            flow is treated as weak/broken.
        depression_vorticity: Relative vorticity (1/s) above which a grid cell
            is flagged as belonging to a synoptic low/depression.
        depression_pressure_anomaly_hpa: Mean-sea-level pressure departure (hPa)
            below the local climatological mean that also flags a low.
        coastal_distance_km: Grid cells nearer than this to the coast are
            candidates for the Coastal regime.
        coastal_wind_speed: 850 hPa wind speed (m/s) above which onshore flow is
            treated as strong enough to drive a coastal-convergence regime.
        orographic_elevation_m: Elevation (m) above which a grid cell is a
            candidate for the Orographic regime.
        orographic_wind_speed: 850 hPa wind speed (m/s) above which upslope flow
            is treated as strong enough to force orographic rainfall.
        monsoon_months: Months treated as the monsoon season.
    """

    core_zone_bbox: Tuple[float, float, float, float] = (18.0, 28.0, 65.0, 88.0)

    active_rain_anomaly_sd: float = 1.0
    break_rain_anomaly_sd: float = -1.0

    active_zonal_wind_850: float = 8.0
    break_zonal_wind_850: float = 2.0

    depression_vorticity: float = 4e-5
    depression_pressure_anomaly_hpa: float = -3.0

    coastal_distance_km: float = 75.0
    coastal_wind_speed: float = 8.0

    orographic_elevation_m: float = 600.0
    orographic_wind_speed: float = 6.0

    monsoon_months: Tuple[int, ...] = (6, 7, 8, 9)


REGIME_RULES = RegimeRuleConfig()

#: A grid cell is rarely one pure regime -- the Western Ghats in an active spell
#: are genuinely part Coastal and part Orographic. When the classifier's second
#: regime carries at least this much probability, the product reports both
#: rather than hiding the mixture behind an argmax.
#:
#: PLACEHOLDER: 0.25 is a display choice, not a calibrated one. It does not
#: affect the numbers -- soft routing already blends by the full distribution --
#: only how the regime is named to the user.
REGIME_BLEND_MIN_PROBABILITY: float = 0.25


# ---------------------------------------------------------------------------
# Warning levels for the district product.
#
# PLACEHOLDER VALUES: probability cut-points have not been tuned against any
# observed base rate. Revisit once the calibrated probabilities exist.
# ---------------------------------------------------------------------------
WARNING_NONE = "none"
WARNING_WATCH = "watch"
WARNING_WARNING = "warning"
WARNING_SEVERE = "severe"

WARNING_LEVELS: List[str] = [
    WARNING_NONE,
    WARNING_WATCH,
    WARNING_WARNING,
    WARNING_SEVERE,
]


@dataclass(frozen=True)
class WarningLevelConfig:
    """Probability cut-points that map heavy-rain probability -> warning level.

    ``watch``/``warning`` are driven by the ``heavy`` (64.5 mm) probability;
    ``severe`` is driven by the ``very_heavy`` (115.6 mm) probability.
    """

    watch_heavy_prob: float = 0.30
    warning_heavy_prob: float = 0.50
    severe_very_heavy_prob: float = 0.50


WARNING_RULES = WarningLevelConfig()


# ---------------------------------------------------------------------------
# Verification settings.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VerificationConfig:
    """Settings for the verification module.

    Attributes:
        fss_neighborhood_sizes: Neighbourhood window sizes (in grid cells, as
            the side length of a square window) for the Fraction Skill Score.
            1 is the grid-scale (no smoothing) reference; 3 and 5 give roughly
            0.75 deg and 1.25 deg windows on a 0.25 deg grid, which is the range
            over which convective-scale rainfall is usually verified. Adjust to
            match the real grid spacing once data is connected.
        intensity_buckets: (label, lower_mm_inclusive, upper_mm_exclusive) used
            to stratify metrics by observed rainfall intensity.
    """

    fss_neighborhood_sizes: Tuple[int, ...] = (1, 3, 5)
    intensity_buckets: Tuple[Tuple[str, float, float], ...] = (
        ("no_rain", 0.0, 2.5),
        ("light", 2.5, 15.6),
        ("moderate", 15.6, 64.5),
        ("heavy", 64.5, 115.6),
        ("very_heavy", 115.6, 204.4),
        ("extremely_heavy", 204.4, float("inf")),
    )


VERIFICATION = VerificationConfig()


# ---------------------------------------------------------------------------
# Chronological split.
#
# PLACEHOLDER: set these to real dates once the data period is known. The
# splitter refuses to run with all three set to None so a random split can never
# happen by accident.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitConfig:
    """Chronological train/validation/test boundaries (inclusive of train).

    A time series must never be split randomly. ``train_end`` and ``val_end``
    are ISO date strings: rows with ``date <= train_end`` are training data,
    ``train_end < date <= val_end`` is validation/calibration, and anything
    after ``val_end`` is the held-out test set.
    """

    train_end: str | None = None
    val_end: str | None = None


SPLIT = SplitConfig()
