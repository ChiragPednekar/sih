"""Stage 3b -- prediction intervals for the corrected rainfall.

A single corrected number invites a confidence it has not earned. This module
fits, per regime, a pair of quantile regressors on the same residual target the
point corrector uses, so every forecast can be reported as a range rather than
a claim::

    corrected_low  = raw + bias_at_10th_percentile
    corrected      = raw + bias_at_50th_percentile   (the point corrector)
    corrected_high = raw + bias_at_90th_percentile

The quantiles are fitted directly with LightGBM's pinball loss rather than
being derived from residual spread, because the residual distribution of
rainfall bias is strongly skewed and heteroscedastic -- a symmetric interval
around the point forecast would be too wide on dry days and far too narrow on
the heavy ones that matter.

Two honest limitations, both surfaced by the API rather than hidden:

* These are *quantiles of the training residual distribution*, not calibrated
  predictive intervals. Whether 80% of observations actually land inside the
  80% band is an empirical question, and :func:`interval_coverage` answers it
  on held-out data. Read that number before trusting the range.
* Regimes too thin to have earned their own point model share the global
  fallback here too, so their intervals are correspondingly generic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data import schema as sch
from ..features.engineering import FEATURE_COLUMNS, select_features
from .baselines import MIN_RAINFALL_MM, NotFittedError, residual_target

#: Artifact filename under ``ARTIFACT_DIR``.
INTERVAL_MODEL_FILENAME = "prediction_intervals.joblib"

#: Lower and upper quantiles of the bias distribution. 0.1/0.9 gives a nominal
#: 80% band -- wide enough to be honest, narrow enough to be useful.
DEFAULT_QUANTILES: Tuple[float, float] = (0.1, 0.9)

#: Regimes with fewer than this many rows reuse the global fallback interval.
DEFAULT_MIN_ROWS_PER_REGIME = 200

#: Hard floor, independent of configuration. The underlying estimator cannot
#: fit a single sample, so a lower ``min_rows_per_regime`` would turn a thin
#: regime into a crash instead of a fallback.
ABSOLUTE_MIN_ROWS = 2


class PredictionIntervalModel:
    """Per-regime quantile regressors over the residual target.

    Attributes:
        quantiles: ``(low, high)`` probabilities being fitted.
        feature_columns: Features used, in order.
        models: ``{regime: {quantile: booster}}`` for regimes with enough data.
        fallback: ``{quantile: booster}`` used for thin or unseen regimes.
        training_counts: ``{regime: n_training_rows}``, kept for the report.
    """

    name = "prediction_intervals"

    def __init__(
        self,
        *,
        quantiles: Tuple[float, float] = DEFAULT_QUANTILES,
        feature_columns: Optional[Sequence[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        min_rows_per_regime: int = DEFAULT_MIN_ROWS_PER_REGIME,
    ) -> None:
        """Initialise an unfitted interval model.

        Args:
            quantiles: ``(low, high)``, each strictly between 0 and 1.
            feature_columns: Feature order. Defaults to the shared list.
            params: Extra LightGBM parameters merged over the defaults.
            min_rows_per_regime: Minimum rows before a regime gets its own
                pair. Values below :data:`ABSOLUTE_MIN_ROWS` are raised to
                it, since the estimator cannot fit fewer.

        Raises:
            ValueError: If the quantiles are not a valid increasing pair.
        """
        low, high = float(quantiles[0]), float(quantiles[1])
        if not 0.0 < low < high < 1.0:
            raise ValueError(
                f"quantiles must satisfy 0 < low < high < 1, got ({low}, {high})."
            )
        self.quantiles: Tuple[float, float] = (low, high)
        self.feature_columns: List[str] = list(
            feature_columns if feature_columns is not None else FEATURE_COLUMNS
        )
        # Untuned defaults; tune on a real validation split, never on dummy rows.
        self.params: Dict[str, Any] = {
            "n_estimators": 300,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
        }
        if params:
            self.params.update(params)
        self.min_rows_per_regime = int(min_rows_per_regime)
        self.models: Dict[str, Dict[float, Any]] = {}
        self.fallback: Optional[Dict[float, Any]] = None
        self.training_counts: Dict[str, int] = {}

    def _build(self, alpha: float) -> Any:
        """Instantiate one quantile regressor.

        Args:
            alpha: The quantile to fit.

        Returns:
            An unfitted LightGBM regressor using the pinball loss.
        """
        from lightgbm import LGBMRegressor

        return LGBMRegressor(objective="quantile", alpha=alpha, **self.params)

    def _fit_pair(self, X: pd.DataFrame, y: pd.Series) -> Dict[float, Any]:
        """Fit both quantiles on one slice of data.

        Args:
            X: Feature frame.
            y: Residual target.

        Returns:
            ``{quantile: fitted booster}``.
        """
        return {q: self._build(q).fit(X.values, y.values) for q in self.quantiles}

    def fit(
        self,
        df: pd.DataFrame,
        regimes: pd.Series,
        target: Optional[pd.Series] = None,
    ) -> "PredictionIntervalModel":
        """Fit a quantile pair per regime, plus a global fallback pair.

        Args:
            df: Training feature table.
            regimes: Regime label per row, aligned to ``df.index``.
            target: Bias target. Defaults to :func:`residual_target` of ``df``.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If ``regimes`` is misaligned, or nothing is fittable.
        """
        if len(regimes) != len(df):
            raise ValueError(
                f"regimes has {len(regimes)} entries but df has {len(df)} rows."
            )
        y = residual_target(df) if target is None else target
        X = select_features(df, self.feature_columns)
        mask = y.notna()
        if not mask.any():
            raise ValueError("No rows with a finite bias target; cannot fit intervals.")

        regimes = pd.Series(np.asarray(regimes), index=df.index)
        self.fallback = self._fit_pair(X[mask], y[mask])

        self.models = {}
        self.training_counts = {}
        for regime, idx in df[mask].groupby(regimes[mask].values).groups.items():
            regime = str(regime)
            self.training_counts[regime] = len(idx)
            if len(idx) < max(self.min_rows_per_regime, ABSOLUTE_MIN_ROWS):
                continue
            self.models[regime] = self._fit_pair(X.loc[idx], y.loc[idx])
        return self

    def _check_fitted(self) -> None:
        """Raise if nothing has been fitted."""
        if self.fallback is None:
            raise NotFittedError(
                "PredictionIntervalModel is not fitted. Run "
                "training/train_bias_correction.py after connecting real data."
            )

    def models_for(self, regime: str) -> Dict[float, Any]:
        """Return the quantile pair handling ``regime``.

        Args:
            regime: A regime label.

        Returns:
            The regime's own pair if it has one, otherwise the global fallback.
        """
        self._check_fitted()
        return self.models.get(str(regime), self.fallback)  # type: ignore[return-value]

    def predict_bias_interval(
        self, df: pd.DataFrame, regimes: pd.Series
    ) -> pd.DataFrame:
        """Predict the low and high bias quantiles per row.

        Args:
            df: Feature table.
            regimes: Regime label per row.

        Returns:
            Frame with ``bias_low`` and ``bias_high`` columns, aligned to ``df``.

        Raises:
            NotFittedError: If nothing has been fitted.
        """
        self._check_fitted()
        regimes = pd.Series(np.asarray(regimes), index=df.index)
        X = select_features(df, self.feature_columns)

        low_q, high_q = self.quantiles
        out = pd.DataFrame(
            {"bias_low": np.nan, "bias_high": np.nan}, index=df.index, dtype="float64"
        )
        for regime, idx in df.groupby(regimes.values).groups.items():
            pair = self.models_for(str(regime))
            out.loc[idx, "bias_low"] = pair[low_q].predict(X.loc[idx].values)
            out.loc[idx, "bias_high"] = pair[high_q].predict(X.loc[idx].values)

        # Quantile regressors are fitted independently and can cross on a row
        # where the data is thin. Order them rather than emitting a negative
        # width, which would render as an inverted interval.
        crossed = out["bias_low"] > out["bias_high"]
        if crossed.any():
            out.loc[crossed, ["bias_low", "bias_high"]] = out.loc[
                crossed, ["bias_high", "bias_low"]
            ].values
        return out

    def predict_interval(self, df: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
        """Return the corrected-rainfall interval per row.

        Args:
            df: Feature table containing ``raw_forecast_mm``.
            regimes: Regime label per row.

        Returns:
            Frame with ``corrected_low`` and ``corrected_high`` in mm, each
            clipped at zero.
        """
        bias = self.predict_bias_interval(df, regimes)
        raw = pd.to_numeric(df[sch.FORECAST_COLUMN], errors="coerce")
        return pd.DataFrame(
            {
                "corrected_low": (raw + bias["bias_low"]).clip(lower=MIN_RAINFALL_MM),
                "corrected_high": (raw + bias["bias_high"]).clip(lower=MIN_RAINFALL_MM),
            },
            index=df.index,
        )

    @property
    def nominal_coverage(self) -> float:
        """The fraction of observations the band is *meant* to contain."""
        return float(self.quantiles[1] - self.quantiles[0])

    def save(self, path: Path) -> Path:
        """Serialise every quantile model to a single ``.joblib`` file.

        Args:
            path: Destination file.

        Returns:
            The path written.

        Raises:
            NotFittedError: If nothing has been fitted.
        """
        import joblib

        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "quantiles": self.quantiles,
                "feature_columns": self.feature_columns,
                "params": self.params,
                "min_rows_per_regime": self.min_rows_per_regime,
                "training_counts": self.training_counts,
                "models": self.models,
                "fallback": self.fallback,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "PredictionIntervalModel":
        """Load an interval model written by :meth:`save`.

        Args:
            path: The ``.joblib`` file.

        Returns:
            A fitted :class:`PredictionIntervalModel`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No prediction-interval artifact at '{path}'.")
        blob = joblib.load(path)
        obj = cls(
            quantiles=tuple(blob["quantiles"]),  # type: ignore[arg-type]
            feature_columns=blob["feature_columns"],
            params=blob["params"],
            min_rows_per_regime=blob["min_rows_per_regime"],
        )
        obj.training_counts = blob.get("training_counts", {})
        obj.models = blob["models"]
        obj.fallback = blob["fallback"]
        return obj


def interval_coverage(
    observed: pd.Series, low: pd.Series, high: pd.Series
) -> Dict[str, float]:
    """Measure how often the observation actually lands inside the band.

    This is the number that decides whether a range may be shown to anyone. A
    nominal 80% band that contains 55% of observations is not a range, it is a
    false reassurance.

    Args:
        observed: Observed rainfall.
        low: Lower bound per row.
        high: Upper bound per row.

    Returns:
        ``{"coverage", "mean_width_mm", "median_width_mm", "n"}``. Coverage is
        NaN when no row has both an observation and a finite band.
    """
    obs = pd.to_numeric(observed, errors="coerce")
    lo = pd.to_numeric(low, errors="coerce")
    hi = pd.to_numeric(high, errors="coerce")
    mask = obs.notna() & lo.notna() & hi.notna()
    if not mask.any():
        return {
            "coverage": float("nan"),
            "mean_width_mm": float("nan"),
            "median_width_mm": float("nan"),
            "n": 0.0,
        }
    inside = (obs[mask] >= lo[mask]) & (obs[mask] <= hi[mask])
    width = (hi[mask] - lo[mask]).abs()
    return {
        "coverage": float(inside.mean()),
        "mean_width_mm": float(width.mean()),
        "median_width_mm": float(width.median()),
        "n": float(mask.sum()),
    }
