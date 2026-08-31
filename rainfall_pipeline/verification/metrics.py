"""Stage 6 -- verification metrics.

Continuous accuracy (RMSE, bias, correlation), categorical skill from 2x2
contingency tables (POD, FAR, CSI, ETS) and a neighbourhood-based Fraction Skill
Score.

A note on why all four categorical scores appear together: POD and FAR trade off
against each other -- you can drive POD to 1 by forecasting rain everywhere, at
the cost of FAR. CSI combines them but is sensitive to the event base rate, so
it is not comparable between a common threshold (64.5 mm) and a rare one
(204.4 mm). ETS corrects CSI for the hits a random forecast would get by chance,
which is what makes cross-threshold comparison meaningful. Read them as a set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class ContingencyTable:
    """A 2x2 contingency table for a single threshold.

    Attributes:
        hits: Forecast yes, observed yes.
        false_alarms: Forecast yes, observed no.
        misses: Forecast no, observed yes.
        correct_negatives: Forecast no, observed no.
    """

    hits: int
    false_alarms: int
    misses: int
    correct_negatives: int

    @property
    def total(self) -> int:
        """Total number of forecast-observation pairs."""
        return self.hits + self.false_alarms + self.misses + self.correct_negatives

    def to_dict(self) -> Dict[str, int]:
        """Return the counts as a plain dict."""
        return asdict(self)


def _finite_pair(
    observed: Sequence[float], predicted: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the observed/predicted pairs where both values are finite.

    Args:
        observed: Observed values.
        predicted: Predicted values.

    Returns:
        Two aligned float arrays with non-finite pairs removed.

    Raises:
        ValueError: If the two inputs have different lengths.
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(
            f"observed and predicted must be the same shape, got {obs.shape} and {pred.shape}."
        )
    mask = np.isfinite(obs) & np.isfinite(pred)
    return obs[mask], pred[mask]


def rmse(observed: Sequence[float], predicted: Sequence[float]) -> float:
    """Root mean squared error, in the units of the inputs (mm).

    Args:
        observed: Observed rainfall.
        predicted: Predicted rainfall.

    Returns:
        The RMSE, or NaN if no finite pairs exist.
    """
    obs, pred = _finite_pair(observed, predicted)
    if obs.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def mean_bias(observed: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean error ``predicted - observed``.

    Positive means the forecast is systematically too wet.

    Args:
        observed: Observed rainfall.
        predicted: Predicted rainfall.

    Returns:
        The mean bias in mm, or NaN if no finite pairs exist.
    """
    obs, pred = _finite_pair(observed, predicted)
    if obs.size == 0:
        return float("nan")
    return float(np.mean(pred - obs))


def mean_absolute_error(observed: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute error in mm.

    Args:
        observed: Observed rainfall.
        predicted: Predicted rainfall.

    Returns:
        The MAE, or NaN if no finite pairs exist.
    """
    obs, pred = _finite_pair(observed, predicted)
    if obs.size == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - obs)))


def correlation(observed: Sequence[float], predicted: Sequence[float]) -> float:
    """Pearson correlation between predicted and observed.

    Args:
        observed: Observed rainfall.
        predicted: Predicted rainfall.

    Returns:
        The correlation coefficient, or NaN if it is undefined (fewer than two
        pairs, or either series is constant).
    """
    obs, pred = _finite_pair(observed, predicted)
    if obs.size < 2 or np.std(obs) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(np.corrcoef(obs, pred)[0, 1])


def contingency_table(
    observed: Sequence[float], predicted: Sequence[float], threshold: float
) -> ContingencyTable:
    """Build a 2x2 contingency table at ``threshold``.

    Args:
        observed: Observed rainfall in mm.
        predicted: Predicted rainfall in mm (or a probability, if ``threshold``
            is a probability cut-point).
        threshold: Exceedance threshold; an event is ``value > threshold``.

    Returns:
        The populated :class:`ContingencyTable`.
    """
    obs, pred = _finite_pair(observed, predicted)
    obs_event = obs > threshold
    pred_event = pred > threshold
    return ContingencyTable(
        hits=int(np.sum(obs_event & pred_event)),
        false_alarms=int(np.sum(~obs_event & pred_event)),
        misses=int(np.sum(obs_event & ~pred_event)),
        correct_negatives=int(np.sum(~obs_event & ~pred_event)),
    )


def pod(table: ContingencyTable) -> float:
    """Probability of detection (hit rate), ``hits / (hits + misses)``.

    Args:
        table: A contingency table.

    Returns:
        POD in ``[0, 1]``, or NaN if the event never occurred.
    """
    denom = table.hits + table.misses
    return float(table.hits / denom) if denom else float("nan")


def far(table: ContingencyTable) -> float:
    """False alarm ratio, ``false_alarms / (hits + false_alarms)``.

    Args:
        table: A contingency table.

    Returns:
        FAR in ``[0, 1]``, or NaN if the event was never forecast.
    """
    denom = table.hits + table.false_alarms
    return float(table.false_alarms / denom) if denom else float("nan")


def csi(table: ContingencyTable) -> float:
    """Critical success index (threat score).

    ``hits / (hits + misses + false_alarms)`` -- correct negatives are excluded,
    which matters for rare events where they would otherwise dominate.

    Args:
        table: A contingency table.

    Returns:
        CSI in ``[0, 1]``, or NaN if the event was neither observed nor forecast.
    """
    denom = table.hits + table.misses + table.false_alarms
    return float(table.hits / denom) if denom else float("nan")


def ets(table: ContingencyTable) -> float:
    """Equitable threat score (Gilbert skill score).

    CSI adjusted for the hits a random forecast with the same event frequency
    would achieve, which makes it comparable across thresholds with different
    base rates.

    Args:
        table: A contingency table.

    Returns:
        ETS in ``[-1/3, 1]``, or NaN if undefined.
    """
    total = table.total
    if total == 0:
        return float("nan")
    hits_random = (table.hits + table.misses) * (table.hits + table.false_alarms) / total
    denom = table.hits + table.misses + table.false_alarms - hits_random
    if denom == 0:
        return float("nan")
    return float((table.hits - hits_random) / denom)


def frequency_bias(table: ContingencyTable) -> float:
    """Frequency bias, ``(hits + false_alarms) / (hits + misses)``.

    1 means the event is forecast as often as it occurs; >1 over-forecast.

    Args:
        table: A contingency table.

    Returns:
        The frequency bias, or NaN if the event never occurred.
    """
    denom = table.hits + table.misses
    if denom == 0:
        return float("nan")
    return float((table.hits + table.false_alarms) / denom)


def categorical_scores(
    observed: Sequence[float], predicted: Sequence[float], threshold: float
) -> Dict[str, float]:
    """Compute every categorical score at one threshold.

    Args:
        observed: Observed rainfall in mm.
        predicted: Predicted rainfall in mm.
        threshold: Exceedance threshold in mm.

    Returns:
        ``{"pod", "far", "csi", "ets", "frequency_bias", "hits", "false_alarms",
        "misses", "correct_negatives"}``.
    """
    table = contingency_table(observed, predicted, threshold)
    return {
        "pod": pod(table),
        "far": far(table),
        "csi": csi(table),
        "ets": ets(table),
        "frequency_bias": frequency_bias(table),
        **{k: float(v) for k, v in table.to_dict().items()},
    }


def continuous_scores(
    observed: Sequence[float], predicted: Sequence[float]
) -> Dict[str, float]:
    """Compute every continuous score.

    Args:
        observed: Observed rainfall in mm.
        predicted: Predicted rainfall in mm.

    Returns:
        ``{"n", "rmse", "bias", "mae", "correlation"}``.
    """
    obs, _ = _finite_pair(observed, predicted)
    return {
        "n": float(obs.size),
        "rmse": rmse(observed, predicted),
        "bias": mean_bias(observed, predicted),
        "mae": mean_absolute_error(observed, predicted),
        "correlation": correlation(observed, predicted),
    }


# ---------------------------------------------------------------------------
# Fraction Skill Score
# ---------------------------------------------------------------------------

def _neighborhood_fraction(field: np.ndarray, window: int) -> np.ndarray:
    """Fraction of grid points exceeding the threshold within each window.

    Args:
        field: 2-D binary field (1 = event), possibly containing NaN for cells
            outside the domain.
        window: Side length of the square neighbourhood, in grid cells.

    Returns:
        A 2-D array of fractions with the same shape as ``field``.
    """
    from scipy.ndimage import uniform_filter

    valid = np.isfinite(field).astype(float)
    filled = np.where(np.isfinite(field), field, 0.0)
    # Normalise by the count of *valid* cells in the window, so cells near the
    # domain edge or over data gaps are not diluted toward zero.
    numer = uniform_filter(filled, size=window, mode="constant", cval=0.0)
    denom = uniform_filter(valid, size=window, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denom > 0, numer / denom, np.nan)
    return out


def fss_from_fields(
    observed_field: np.ndarray,
    predicted_field: np.ndarray,
    threshold: float,
    window: int,
) -> float:
    """Fraction Skill Score for a single 2-D field pair.

    FSS compares the *fraction* of exceedances in a neighbourhood rather than
    demanding a point-for-point match, which is the right way to score a
    high-resolution rainfall forecast: a convective cell displaced by one grid
    box is a near-miss, not a total failure, and point verification punishes it
    twice (once as a miss, once as a false alarm).

    Args:
        observed_field: 2-D observed rainfall (mm), NaN outside the domain.
        predicted_field: 2-D predicted rainfall (mm), same shape.
        threshold: Exceedance threshold in mm.
        window: Neighbourhood side length in grid cells. 1 reduces to a
            point-wise Brier-type skill score.

    Returns:
        FSS in ``[0, 1]`` (1 is perfect), or NaN when neither field has any
        exceedance -- in which case the score is undefined rather than perfect.

    Raises:
        ValueError: If the two fields have different shapes.
    """
    if observed_field.shape != predicted_field.shape:
        raise ValueError(
            f"Field shapes differ: {observed_field.shape} vs {predicted_field.shape}."
        )

    obs_bin = np.where(np.isfinite(observed_field), observed_field > threshold, np.nan).astype(float)
    pred_bin = np.where(np.isfinite(predicted_field), predicted_field > threshold, np.nan).astype(float)

    obs_frac = _neighborhood_fraction(obs_bin, window)
    pred_frac = _neighborhood_fraction(pred_bin, window)

    mask = np.isfinite(obs_frac) & np.isfinite(pred_frac)
    if not mask.any():
        return float("nan")

    o, p = obs_frac[mask], pred_frac[mask]
    mse = float(np.mean((p - o) ** 2))
    reference = float(np.mean(p**2) + np.mean(o**2))
    if reference == 0:
        # Neither forecast nor observation has any exceedance anywhere. FSS is
        # undefined here; returning 1.0 would flatter every model equally on
        # dry days and inflate the seasonal average.
        return float("nan")
    return float(1.0 - mse / reference)


def to_field(
    df: pd.DataFrame, value_column: str, *, lat_col: str = "lat", lon_col: str = "lon"
) -> np.ndarray:
    """Pivot a single date's rows into a 2-D lat/lon field.

    Missing grid cells become NaN, which the FSS neighbourhood handles.

    Args:
        df: Rows for one date.
        value_column: Column to pivot.
        lat_col: Latitude column.
        lon_col: Longitude column.

    Returns:
        A 2-D array with latitude increasing along axis 0 and longitude along
        axis 1.
    """
    grid = df.pivot_table(index=lat_col, columns=lon_col, values=value_column, aggfunc="mean")
    return grid.sort_index().sort_index(axis=1).to_numpy(dtype=float)


def fss(
    df: pd.DataFrame,
    observed_column: str,
    predicted_column: str,
    threshold: float,
    window: int,
    *,
    date_col: str = "date",
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> float:
    """Fraction Skill Score averaged over every date in ``df``.

    Args:
        df: Long-format frame with date, lat, lon and the two value columns.
        observed_column: Observed rainfall column.
        predicted_column: Predicted rainfall column.
        threshold: Exceedance threshold in mm.
        window: Neighbourhood side length in grid cells.
        date_col: Date column name.
        lat_col: Latitude column name.
        lon_col: Longitude column name.

    Returns:
        The mean FSS over dates where it is defined, or NaN if it is defined on
        no date.
    """
    scores: List[float] = []
    for _, day in df.groupby(date_col):
        obs_field = to_field(day, observed_column, lat_col=lat_col, lon_col=lon_col)
        pred_field = to_field(day, predicted_column, lat_col=lat_col, lon_col=lon_col)
        score = fss_from_fields(obs_field, pred_field, threshold, window)
        if np.isfinite(score):
            scores.append(score)
    return float(np.mean(scores)) if scores else float("nan")


# ---------------------------------------------------------------------------
# Stratified evaluation
# ---------------------------------------------------------------------------

def evaluate(
    df: pd.DataFrame,
    observed_column: str,
    predicted_column: str,
    *,
    thresholds: Optional[Dict[str, float]] = None,
    fss_windows: Optional[Iterable[int]] = None,
    compute_fss: bool = True,
) -> Dict[str, object]:
    """Compute the full metric set for one prediction column.

    Args:
        df: Frame with the observation, the prediction and (for FSS) date/lat/lon.
        observed_column: Observed rainfall column.
        predicted_column: Predicted rainfall column.
        thresholds: ``{name: mm}``. Defaults to the IMD categories.
        fss_windows: Neighbourhood sizes. Defaults to the configured set.
        compute_fss: Set False to skip FSS, which is by far the most expensive
            metric here.

    Returns:
        ``{"continuous": {...}, "categorical": {name: {...}}, "fss": {name:
        {window: score}}}``.
    """
    from ..config.thresholds import RAIN_THRESHOLDS, VERIFICATION

    thresholds = dict(thresholds or RAIN_THRESHOLDS)
    windows = list(fss_windows if fss_windows is not None else VERIFICATION.fss_neighborhood_sizes)

    obs = df[observed_column]
    pred = df[predicted_column]

    result: Dict[str, object] = {
        "continuous": continuous_scores(obs, pred),
        "categorical": {
            name: categorical_scores(obs, pred, mm) for name, mm in thresholds.items()
        },
    }

    if compute_fss and {"date", "lat", "lon"}.issubset(df.columns):
        result["fss"] = {
            name: {
                f"window_{w}": fss(df, observed_column, predicted_column, mm, w)
                for w in windows
            }
            for name, mm in thresholds.items()
        }
    else:
        result["fss"] = {}
    return result


def evaluate_by_group(
    df: pd.DataFrame,
    observed_column: str,
    predicted_column: str,
    group_column: str,
    *,
    thresholds: Optional[Dict[str, float]] = None,
    compute_fss: bool = False,
) -> Dict[str, Dict[str, object]]:
    """Compute the metric set separately for each value of ``group_column``.

    Args:
        df: Frame to evaluate.
        observed_column: Observed rainfall column.
        predicted_column: Predicted rainfall column.
        group_column: Column to stratify by (regime, district, region, ...).
        thresholds: ``{name: mm}``.
        compute_fss: Whether to compute FSS per group. Off by default -- a
            regime or district subset is not a complete field, so its
            neighbourhood fractions are computed over a fragmentary grid.

    Returns:
        ``{group_value: metrics}``.

    Raises:
        KeyError: If ``group_column`` is absent.
    """
    if group_column not in df.columns:
        raise KeyError(f"Cannot stratify by '{group_column}': column not present.")
    out: Dict[str, Dict[str, object]] = {}
    for value, group in df.groupby(group_column, observed=True):
        out[str(value)] = evaluate(
            group,
            observed_column,
            predicted_column,
            thresholds=thresholds,
            compute_fss=compute_fss,
        )
    return out


def bucket_by_intensity(
    df: pd.DataFrame, observed_column: str, buckets: Optional[Sequence[tuple]] = None
) -> pd.Series:
    """Label each row with an observed-rainfall intensity bucket.

    Args:
        df: Frame containing the observation column.
        observed_column: Observed rainfall column.
        buckets: ``[(label, lower_inclusive, upper_exclusive), ...]``. Defaults
            to the configured set.

    Returns:
        A string Series of bucket labels aligned to ``df.index``.
    """
    from ..config.thresholds import VERIFICATION

    buckets = list(buckets if buckets is not None else VERIFICATION.intensity_buckets)
    values = pd.to_numeric(df[observed_column], errors="coerce")
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for label, low, high in buckets:
        out = out.mask(values.between(low, high, inclusive="left"), label)
    return out.rename("intensity_bucket")
