"""Stage 3 -- regime-specific residual bias correction (Model D).

One residual regressor per regime. Each model is trained only on the rows the
rule-based labeller assigned to its regime, so it can specialise on that
regime's error structure rather than averaging over five different physical
situations the way Baseline B has to.

The target is always the residual ``bias = observed_mm - raw_forecast_mm`` and
the final product is ``corrected = raw_forecast_mm + predicted_bias``, clipped
at zero.

Regimes with too few training rows fall back to a globally fitted model rather
than fitting an unstable one -- with a five-way split of a monsoon season, the
rarer regimes can end up thin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config.thresholds import REGIME_LABELS
from ..data import schema as sch
from ..features.engineering import FEATURE_COLUMNS
from .baselines import (
    MIN_RAINFALL_MM,
    BiasExplanation,
    GlobalBiasCorrector,
    NotFittedError,
    residual_target,
)

#: Artifact filename under ``ARTIFACT_DIR``.
BIAS_MODEL_FILENAME = "bias_correction.joblib"

#: Regimes with fewer than this many training rows reuse the global fallback.
#: PLACEHOLDER: revisit once the real per-regime row counts are known.
DEFAULT_MIN_ROWS_PER_REGIME = 200


class RegimeBiasCorrector:
    """Model D -- a dictionary of per-regime residual regressors.

    Attributes:
        backend: ``"xgboost"``, ``"catboost"``, ``"lightgbm"``, or ``"ensemble"``.
        loss: Loss objective (``"mse"``, ``"tweedie"``, ``"huber"``).
        feature_columns: Features used, in order. Contains no regime column;
            the regime is expressed by *which* model is used, not by a feature.
        models: ``{regime: GlobalBiasCorrector}`` for regimes with enough data.
        fallback: A globally fitted corrector used for thin or unseen regimes.
        training_counts: ``{regime: n_training_rows}``, kept for the report.
    """

    name = "D_regime_residual"

    def __init__(
        self,
        *,
        backend: str = "xgboost",
        loss: str = "mse",
        feature_columns: Optional[Sequence[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        min_rows_per_regime: int = DEFAULT_MIN_ROWS_PER_REGIME,
    ) -> None:
        """Initialise an unfitted regime-specific corrector.

        Args:
            backend: ``"xgboost"``, ``"catboost"``, ``"lightgbm"``, or ``"ensemble"``.
            loss: Loss objective (``"mse"``, ``"tweedie"``, ``"huber"``).
            feature_columns: Feature order. Defaults to
                :data:`~rainfall_pipeline.features.engineering.FEATURE_COLUMNS`.
            params: Extra estimator parameters merged over the defaults.
            min_rows_per_regime: Minimum rows before a regime gets its own model.
        """
        self.backend = backend
        self.loss = loss
        self.feature_columns: List[str] = list(
            feature_columns if feature_columns is not None else FEATURE_COLUMNS
        )
        self.params = dict(params) if params else None
        self.min_rows_per_regime = int(min_rows_per_regime)
        self.models: Dict[str, GlobalBiasCorrector] = {}
        self.fallback: Optional[GlobalBiasCorrector] = None
        self.training_counts: Dict[str, int] = {}

    def _new_model(self) -> GlobalBiasCorrector:
        """Build one unfitted per-regime regressor."""
        return GlobalBiasCorrector(
            backend=self.backend,
            loss=self.loss,
            feature_columns=self.feature_columns,
            params=self.params,
        )

    def fit(
        self,
        df: pd.DataFrame,
        regimes: pd.Series,
        target: Optional[pd.Series] = None,
    ) -> "RegimeBiasCorrector":
        """Fit one residual model per regime, plus the global fallback.

        Args:
            df: Training feature table.
            regimes: Regime label per row, aligned to ``df.index``.
            target: Bias target. Defaults to :func:`residual_target` of ``df``.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If ``regimes`` is not aligned to ``df``, or if there are
                no usable training rows at all.
        """
        if len(regimes) != len(df):
            raise ValueError(
                f"regimes has {len(regimes)} entries but df has {len(df)} rows."
            )
        y = residual_target(df) if target is None else target
        regimes = pd.Series(np.asarray(regimes), index=df.index)

        # The fallback is fitted on everything, so it is exactly Baseline B's
        # estimator. Sharing the construction keeps the comparison honest.
        self.fallback = self._new_model().fit(df, y)

        self.models = {}
        self.training_counts = {name: 0 for name in REGIME_LABELS}
        for regime, idx in df.groupby(regimes.values).groups.items():
            regime = str(regime)
            self.training_counts[regime] = len(idx)
            if len(idx) < self.min_rows_per_regime:
                # Too thin to specialise; this regime will route to the fallback.
                continue
            self.models[regime] = self._new_model().fit(df.loc[idx], y.loc[idx])
        return self

    def _check_fitted(self) -> None:
        """Raise if no models have been fitted."""
        if self.fallback is None:
            raise NotFittedError(
                "RegimeBiasCorrector is not fitted. Run "
                "training/train_bias_correction.py after connecting real data."
            )

    def model_for(self, regime: str) -> GlobalBiasCorrector:
        """Return the model that handles ``regime``.

        Args:
            regime: A regime label.

        Returns:
            The regime's own model if it has one, otherwise the global fallback.

        Raises:
            NotFittedError: If nothing has been fitted.
        """
        self._check_fitted()
        return self.models.get(str(regime), self.fallback)  # type: ignore[return-value]

    def predict_bias(self, df: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        """Predict the bias for every row, routing by regime.

        Args:
            df: Feature table.
            regimes: Regime label per row.

        Returns:
            Predicted ``observed - raw`` per row, in mm.

        Raises:
            NotFittedError: If nothing has been fitted.
        """
        self._check_fitted()
        regimes = pd.Series(np.asarray(regimes), index=df.index)
        out = pd.Series(np.nan, index=df.index, name="predicted_bias", dtype="float64")
        for regime, idx in df.groupby(regimes.values).groups.items():
            out.loc[idx] = self.model_for(str(regime)).predict_bias(df.loc[idx]).values
        return out

    def predict(self, df: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        """Return the corrected forecast, routing by regime.

        Args:
            df: Feature table containing ``raw_forecast_mm``.
            regimes: Regime label per row.

        Returns:
            ``raw_forecast_mm + predicted_bias``, clipped at zero.
        """
        corrected = df[sch.FORECAST_COLUMN] + self.predict_bias(df, regimes)
        return corrected.clip(lower=MIN_RAINFALL_MM).rename("corrected_mm")

    # -- soft (probability-weighted) routing ------------------------------

    def predict_bias_soft(
        self,
        df: pd.DataFrame,
        regime_probs: pd.DataFrame,
        *,
        min_weight: float = 1e-4,
    ) -> pd.Series:
        """Predict the bias by blending every regime's model by its probability.

        Hard routing sends each row to ``argmax`` and discards the rest of the
        classifier's distribution. That is a poor fit for the monsoon, where a
        Western Ghats cell on an active day is genuinely part Coastal, part
        Orographic and part Active -- the argmax throws away two thirds of what
        the classifier knows, and it makes the correction discontinuous at the
        point where the leading regime flips.

        This blends instead::

            predicted_bias = sum_r  P(regime = r) * bias_model_r(row)

        Probability mass belonging to regimes that never got their own model
        (too few training rows) is pooled onto the global fallback, so the
        weights always sum to one.

        Args:
            df: Feature table.
            regime_probs: One column per regime label, rows aligned to ``df``.
                Rows are renormalised to sum to 1; an all-zero row falls back
                entirely to the global model.
            min_weight: Regimes whose maximum weight across all rows is below
                this are skipped, which keeps the cost near hard routing when
                the classifier is confident.

        Returns:
            Predicted ``observed - raw`` per row, in mm.

        Raises:
            NotFittedError: If nothing has been fitted.
            ValueError: If ``regime_probs`` is not aligned to ``df``.

        Note:
            This evaluates up to one model per regime over the whole frame,
            where hard routing evaluates one model per row. For a single API
            request the difference is irrelevant; over a full verification set
            it is roughly ``n_regimes`` times the work.
        """
        self._check_fitted()
        if len(regime_probs) != len(df):
            raise ValueError(
                f"regime_probs has {len(regime_probs)} rows but df has {len(df)}."
            )

        probs = regime_probs.copy()
        probs.index = df.index
        probs = probs.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)

        row_total = probs.sum(axis=1)
        # An all-zero row carries no information; hand it to the fallback whole.
        probs = probs.div(row_total.where(row_total > 0, 1.0), axis=0)
        fallback_weight = (1.0 - probs.sum(axis=1)).clip(lower=0.0)

        out = pd.Series(0.0, index=df.index, name="predicted_bias", dtype="float64")
        for regime in probs.columns:
            weight = probs[regime]
            if float(weight.max()) < min_weight:
                continue
            model = self.models.get(str(regime))
            if model is None:
                # No specialised model for this regime -- pool onto the fallback.
                fallback_weight = fallback_weight + weight
                continue
            out = out + weight * model.predict_bias(df).values

        if float(fallback_weight.max()) >= min_weight:
            out = out + fallback_weight * self.fallback.predict_bias(df).values  # type: ignore[union-attr]
        return out

    def predict_soft(
        self,
        df: pd.DataFrame,
        regime_probs: pd.DataFrame,
        *,
        min_weight: float = 1e-4,
    ) -> pd.Series:
        """Return the corrected forecast using probability-weighted routing.

        Args:
            df: Feature table containing ``raw_forecast_mm``.
            regime_probs: One column per regime label, rows aligned to ``df``.
            min_weight: See :meth:`predict_bias_soft`.

        Returns:
            ``raw_forecast_mm + predicted_bias``, clipped at zero.
        """
        bias = self.predict_bias_soft(df, regime_probs, min_weight=min_weight)
        corrected = df[sch.FORECAST_COLUMN] + bias
        return corrected.clip(lower=MIN_RAINFALL_MM).rename("corrected_mm")

    # -- explainability ---------------------------------------------------

    def explain_row(
        self,
        df: pd.DataFrame,
        regime: str,
        top_n: int = 5,
    ) -> BiasExplanation:
        """Explain the correction the routed model applied to the first row.

        Args:
            df: A feature table; only the first row is explained.
            regime: The regime whose model should do the explaining.
            top_n: How many contributing features to return.

        Returns:
            A :class:`~rainfall_pipeline.models.baselines.BiasExplanation`. The
            ``regime`` field reports ``"__fallback__"`` when the requested
            regime has no model of its own, so the dashboard never implies a
            specialised correction that did not happen.

        Raises:
            ValueError: If ``df`` is empty.
            NotFittedError: If nothing has been fitted.
        """
        self._check_fitted()
        label = str(regime)
        model = self.models.get(label)
        if model is None:
            model, label = self.fallback, "__fallback__"  # type: ignore[assignment]
        return model.explain_row(df, top_n=top_n, regime=label)  # type: ignore[union-attr]

    def save(self, path: Path) -> Path:
        """Serialise every per-regime model to a single ``.joblib`` file.

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
                "backend": self.backend,
                "feature_columns": self.feature_columns,
                "params": self.params,
                "min_rows_per_regime": self.min_rows_per_regime,
                "training_counts": self.training_counts,
                "models": {k: v.model for k, v in self.models.items()},
                "fallback": self.fallback.model if self.fallback else None,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "RegimeBiasCorrector":
        """Load a corrector written by :meth:`save`.

        Args:
            path: The ``.joblib`` file.

        Returns:
            A fitted :class:`RegimeBiasCorrector`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No bias-correction artifact at '{path}'.")
        blob = joblib.load(path)
        obj = cls(
            backend=blob["backend"],
            feature_columns=blob["feature_columns"],
            params=blob["params"],
            min_rows_per_regime=blob["min_rows_per_regime"],
        )
        obj.training_counts = blob.get("training_counts", {})
        for regime, raw_model in blob["models"].items():
            wrapper = obj._new_model()
            wrapper.model = raw_model
            obj.models[regime] = wrapper
        if blob["fallback"] is not None:
            wrapper = obj._new_model()
            wrapper.model = blob["fallback"]
            obj.fallback = wrapper
        return obj


def correct_forecast(
    feature_row: pd.DataFrame,
    regime_label: str,
    corrector: RegimeBiasCorrector,
) -> float:
    """Correct a single forecast, routing to the regime's model automatically.

    Args:
        feature_row: A one-row feature table containing ``raw_forecast_mm``.
        regime_label: The regime assigned to this row by the regime engine.
        corrector: A fitted :class:`RegimeBiasCorrector`.

    Returns:
        The corrected rainfall forecast in mm.

    Raises:
        ValueError: If ``feature_row`` does not have exactly one row.
        NotFittedError: If the corrector is unfitted.
    """
    if len(feature_row) != 1:
        raise ValueError(f"correct_forecast expects exactly 1 row, got {len(feature_row)}.")
    regimes = pd.Series([regime_label], index=feature_row.index)
    return float(corrector.predict(feature_row, regimes).iloc[0])
