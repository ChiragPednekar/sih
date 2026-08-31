"""Baselines A, B and C -- the comparators the full system must beat.

* **Baseline A** (:class:`RawForecastBaseline`) -- the uncorrected NWP forecast.
  Trivial by construction, but it is the number that matters operationally: any
  correction scheme that cannot beat it is not worth deploying.
* **Baseline B** (:class:`GlobalBiasCorrector`) -- a single regime-blind
  gradient-boosted residual model. This isolates the value of *regime awareness*
  specifically: B and the regime-specific models see the same features, so the
  only difference between them is the routing.
* **Baseline C** (:class:`QuantileMapping`) -- classical empirical quantile
  mapping, no ML at all. Fitted globally and, for comparison, per regime.

Every corrector exposes the same three methods -- ``fit``, ``predict``,
``save``/``load`` -- so the verification module can treat them interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..data import schema as sch
from ..features.engineering import FEATURE_COLUMNS, select_features

#: Minimum corrected rainfall. Negative rainfall is unphysical, so every
#: corrector clips at zero as its final step.
MIN_RAINFALL_MM = 0.0


class NotFittedError(RuntimeError):
    """Raised when ``predict`` is called before ``fit``."""


def residual_target(df: pd.DataFrame) -> pd.Series:
    """Compute the residual training target ``observed - raw_forecast``.

    Training on the residual rather than on rainfall itself means the model only
    has to learn the *error structure* of the NWP, which is a much smaller and
    better-conditioned problem than learning rainfall from scratch.

    Args:
        df: Frame containing ``observed_mm`` and ``raw_forecast_mm``.

    Returns:
        The bias, in mm, per row.

    Raises:
        KeyError: If either column is absent.
    """
    for col in (sch.OBSERVED_COLUMN, sch.FORECAST_COLUMN):
        if col not in df.columns:
            raise KeyError(f"residual_target needs column '{col}'.")
    return (df[sch.OBSERVED_COLUMN] - df[sch.FORECAST_COLUMN]).rename("bias")


# ---------------------------------------------------------------------------
# Baseline A
# ---------------------------------------------------------------------------

class RawForecastBaseline:
    """Baseline A -- pass the raw forecast through unchanged.

    Implements the corrector interface so it can sit alongside the others in the
    verification loop without special-casing.
    """

    name = "A_raw_nwp"

    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> "RawForecastBaseline":
        """No-op. Present for interface compatibility.

        Args:
            df: Ignored.
            target: Ignored.

        Returns:
            ``self``.
        """
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return the raw forecast, clipped at zero.

        Args:
            df: Frame containing ``raw_forecast_mm``.

        Returns:
            The uncorrected forecast in mm.
        """
        return df[sch.FORECAST_COLUMN].clip(lower=MIN_RAINFALL_MM).rename("corrected_mm")


# ---------------------------------------------------------------------------
# Baseline B
# ---------------------------------------------------------------------------

@dataclass
class BiasExplanation:
    """SHAP explanation for a single bias-correction prediction.

    This answers the question the dashboard actually asks -- "why did the model
    change 85 mm into 118 mm?" -- which is a *corrector* decision. The regime
    classifier's own explanation answers a different question (why this regime),
    so the two are reported side by side rather than one standing in for the
    other.

    Attributes:
        regime: The regime whose model produced this correction, or
            ``"__fallback__"`` when the row was routed to the global fallback.
        raw_mm: The uncorrected forecast for the row.
        predicted_bias_mm: The signed correction the model applied, in mm.
        corrected_mm: ``raw_mm + predicted_bias_mm``, clipped at zero.
        top_features: ``[(feature_name, shap_value), ...]`` ordered by absolute
            contribution, largest first. Values are in **mm of bias** -- they
            sum with ``base_value`` to ``predicted_bias_mm``.
        base_value: The model's expected bias before any feature contributions.
    """

    regime: str
    raw_mm: float
    predicted_bias_mm: float
    corrected_mm: float
    top_features: List[tuple[str, float]]
    base_value: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "regime": self.regime,
            "raw_mm": float(self.raw_mm),
            "predicted_bias_mm": float(self.predicted_bias_mm),
            "corrected_mm": float(self.corrected_mm),
            "top_features": [
                {"feature": f, "shap_value": float(v)} for f, v in self.top_features
            ],
            "base_value": float(self.base_value),
        }


class GlobalBiasCorrector:
    """Baseline B -- one regime-blind residual model for the whole domain.

    Attributes:
        backend: ``"xgboost"``, ``"catboost"``, ``"lightgbm"``, or ``"ensemble"``.
        loss: Loss objective (``"mse"``, ``"tweedie"``, ``"huber"``).
        feature_columns: Features used, in order.
        model: The fitted regressor, or None.
    """

    name = "B_global_ml"

    def __init__(
        self,
        *,
        backend: str = "xgboost",
        loss: str = "mse",
        feature_columns: Optional[Sequence[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise an unfitted global corrector.

        Args:
            backend: ``"xgboost"``, ``"catboost"``, ``"lightgbm"``, or ``"ensemble"``.
            loss: Loss objective (``"mse"``, ``"tweedie"``, ``"huber"``).
            feature_columns: Feature order. Defaults to
                :data:`~rainfall_pipeline.features.engineering.FEATURE_COLUMNS`.
                Note this list deliberately excludes any regime column -- that
                is the whole point of Baseline B.
            params: Extra estimator parameters merged over the defaults.
        """
        valid_backends = {"xgboost", "catboost", "lightgbm", "ensemble"}
        if backend not in valid_backends:
            raise ValueError(f"Unknown backend '{backend}'. Valid: {sorted(valid_backends)}")
        self.backend = backend
        self.loss = loss
        self.feature_columns: List[str] = list(
            feature_columns if feature_columns is not None else FEATURE_COLUMNS
        )
        # Untuned defaults; tune on a real validation split, never on dummy rows.
        self.params: Dict[str, Any] = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        if params:
            self.params.update(params)
        self.model: Any = None
        self.ensemble_models: Optional[Dict[str, Any]] = None
        self._explainer: Any = None

    def _build_single(self, backend: str) -> Any:
        """Instantiate a single regressor for the given backend."""
        if backend == "xgboost":
            from xgboost import XGBRegressor

            obj = "reg:pseudohubererror" if self.loss in {"huber", "tweedie"} else "reg:squarederror"
            return XGBRegressor(
                objective=obj, tree_method="hist", **self.params
            )
        if backend == "lightgbm":
            from lightgbm import LGBMRegressor

            obj = "huber" if self.loss in {"huber", "tweedie"} else "regression"
            return LGBMRegressor(
                objective=obj,
                n_estimators=self.params.get("n_estimators", 400),
                max_depth=self.params.get("max_depth", 6),
                learning_rate=self.params.get("learning_rate", 0.05),
                subsample=self.params.get("subsample", 0.8),
                colsample_bytree=self.params.get("colsample_bytree", 0.8),
                random_state=self.params.get("random_state", 42),
                verbose=-1,
            )
        from catboost import CatBoostRegressor

        loss_fn = "Huber:delta=1.5" if self.loss in {"huber", "tweedie"} else "RMSE"
        return CatBoostRegressor(
            iterations=self.params.get("n_estimators", 400),
            depth=self.params.get("max_depth", 6),
            learning_rate=self.params.get("learning_rate", 0.05),
            random_seed=self.params.get("random_state", 42),
            loss_function=loss_fn,
            verbose=False,
        )

    def _build(self) -> Any:
        """Instantiate the underlying regressor or ensemble dictionary."""
        if self.backend == "ensemble":
            return {
                "xgboost": self._build_single("xgboost"),
                "lightgbm": self._build_single("lightgbm"),
                "catboost": self._build_single("catboost"),
            }
        return self._build_single(self.backend)

    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> "GlobalBiasCorrector":
        """Fit the residual model.

        Args:
            df: Training feature table.
            target: Bias target. Defaults to :func:`residual_target` of ``df``.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If there are no usable training rows.
        """
        y = residual_target(df) if target is None else target
        X = select_features(df, self.feature_columns)
        mask = y.notna()
        if not mask.any():
            raise ValueError("No rows with a finite bias target; cannot fit.")

        built = self._build()
        if self.backend == "ensemble":
            self.ensemble_models = {}
            for name, m in built.items():
                m.fit(X[mask].values, y[mask].values)
                self.ensemble_models[name] = m
            self.model = self.ensemble_models["xgboost"]
        else:
            self.model = built
            self.model.fit(X[mask].values, y[mask].values)
            self.ensemble_models = None
        self._explainer = None
        return self

    def predict_bias(self, df: pd.DataFrame) -> pd.Series:
        """Predict the bias (mm) without applying it.

        Args:
            df: Feature table.

        Returns:
            Predicted ``observed - raw`` per row.

        Raises:
            NotFittedError: If the model is unfitted.
        """
        if self.model is None:
            raise NotFittedError(
                "GlobalBiasCorrector is not fitted. Run "
                "training/train_bias_correction.py after connecting real data."
            )
        X = select_features(df, self.feature_columns)
        if self.ensemble_models is not None:
            preds = [np.asarray(m.predict(X.values), dtype="float64") for m in self.ensemble_models.values()]
            avg_pred = np.mean(preds, axis=0)
            return pd.Series(avg_pred, index=df.index, name="predicted_bias")
        return pd.Series(self.model.predict(X.values), index=df.index, name="predicted_bias")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return the corrected forecast ``raw + predicted_bias``, clipped at 0.

        Args:
            df: Feature table containing ``raw_forecast_mm``.

        Returns:
            The corrected forecast in mm.
        """
        corrected = df[sch.FORECAST_COLUMN] + self.predict_bias(df)
        return corrected.clip(lower=MIN_RAINFALL_MM).rename("corrected_mm")

    # -- explainability --------------------------------------------------

    def _get_explainer(self) -> Any:
        """Return a cached SHAP TreeExplainer for the fitted model.

        Raises:
            NotFittedError: If the model is unfitted.
        """
        if self.model is None:
            raise NotFittedError("Cannot explain an unfitted GlobalBiasCorrector.")
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def shap_values(self, df: pd.DataFrame) -> np.ndarray:
        """Compute per-feature SHAP contributions to the predicted bias.

        Args:
            df: Feature table.

        Returns:
            Array of shape ``(n_rows, n_features)``, in mm of bias.

        Raises:
            NotFittedError: If the model is unfitted.
        """
        explainer = self._get_explainer()
        X = select_features(df, self.feature_columns)
        return np.asarray(explainer.shap_values(X.values), dtype="float64")

    def explain_row(
        self,
        df: pd.DataFrame,
        top_n: int = 5,
        *,
        regime: str = "__global__",
    ) -> BiasExplanation:
        """Explain the correction applied to the first row of ``df``.

        Args:
            df: A feature table; only the first row is explained.
            top_n: How many contributing features to return.
            regime: Label recorded on the explanation, so the caller can say
                which regime's model produced it.

        Returns:
            A :class:`BiasExplanation` for that row.

        Raises:
            ValueError: If ``df`` is empty.
            NotFittedError: If the model is unfitted.
        """
        if df.empty:
            raise ValueError("explain_row needs at least one row.")
        row = df.iloc[[0]]

        contributions = self.shap_values(row)[0]
        order = np.argsort(-np.abs(contributions))[:top_n]
        top = [(self.feature_columns[i], float(contributions[i])) for i in order]

        raw = float(pd.to_numeric(row[sch.FORECAST_COLUMN], errors="coerce").iloc[0])
        bias = float(self.predict_bias(row).iloc[0])
        expected = np.atleast_1d(self._get_explainer().expected_value)
        return BiasExplanation(
            regime=str(regime),
            raw_mm=raw,
            predicted_bias_mm=bias,
            corrected_mm=max(raw + bias, MIN_RAINFALL_MM),
            top_features=top,
            base_value=float(expected[0]),
        )

    def save(self, path: Path) -> Path:
        """Serialise to ``path``.

        Args:
            path: Destination ``.joblib`` file.

        Returns:
            The path written.

        Raises:
            NotFittedError: If the model is unfitted.
        """
        import joblib

        if self.model is None:
            raise NotFittedError("Cannot save an unfitted GlobalBiasCorrector.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "backend": self.backend,
                "feature_columns": self.feature_columns,
                "params": self.params,
                "model": self.model,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "GlobalBiasCorrector":
        """Load a corrector written by :meth:`save`.

        Args:
            path: The ``.joblib`` file.

        Returns:
            A fitted :class:`GlobalBiasCorrector`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No global bias-correction artifact at '{path}'.")
        blob = joblib.load(path)
        obj = cls(
            backend=blob["backend"],
            feature_columns=blob["feature_columns"],
            params=blob["params"],
        )
        obj.model = blob["model"]
        return obj


# ---------------------------------------------------------------------------
# Baseline C
# ---------------------------------------------------------------------------

class QuantileMapping:
    """Baseline C -- empirical quantile mapping, no machine learning.

    Fits the empirical CDFs of the raw forecast and of the observations, then at
    prediction time replaces a forecast value with the observed value at the
    same non-exceedance probability. This corrects the *distribution* of the
    forecast but, unlike a residual model, cannot correct individual events --
    which is exactly the contrast the verification report is meant to expose.

    Attributes:
        n_quantiles: Number of quantile knots used to represent each CDF.
        extrapolate: How to handle forecasts beyond the fitted range;
            ``"constant"`` holds the end-point correction, ``"linear"``
            extrapolates the end-point ratio.
    """

    name = "C_quantile_mapping"

    def __init__(self, *, n_quantiles: int = 100, extrapolate: str = "constant") -> None:
        """Initialise an unfitted quantile mapper.

        Args:
            n_quantiles: Number of evenly spaced probability knots.
            extrapolate: ``"constant"`` or ``"linear"``.
        """
        if extrapolate not in {"constant", "linear"}:
            raise ValueError("extrapolate must be 'constant' or 'linear'.")
        self.n_quantiles = int(n_quantiles)
        self.extrapolate = extrapolate
        self._probs: Optional[np.ndarray] = None
        self._forecast_q: Optional[np.ndarray] = None
        self._observed_q: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> "QuantileMapping":
        """Fit the forecast and observation CDFs.

        Args:
            df: Training frame with ``raw_forecast_mm`` and ``observed_mm``.
            target: Unused; present for interface compatibility.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If fewer than two usable rows are available.
        """
        forecast = pd.to_numeric(df[sch.FORECAST_COLUMN], errors="coerce")
        observed = pd.to_numeric(df[sch.OBSERVED_COLUMN], errors="coerce")
        mask = forecast.notna() & observed.notna()
        if mask.sum() < 2:
            raise ValueError(
                "Quantile mapping needs at least 2 rows with both a forecast and "
                "an observation."
            )
        self._probs = np.linspace(0.0, 1.0, self.n_quantiles)
        self._forecast_q = np.quantile(forecast[mask].to_numpy(), self._probs)
        self._observed_q = np.quantile(observed[mask].to_numpy(), self._probs)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Map raw forecasts onto the observed distribution.

        Args:
            df: Frame containing ``raw_forecast_mm``.

        Returns:
            The quantile-mapped forecast in mm.

        Raises:
            NotFittedError: If the mapper is unfitted.
        """
        if self._forecast_q is None or self._observed_q is None:
            raise NotFittedError("QuantileMapping is not fitted.")
        values = pd.to_numeric(df[sch.FORECAST_COLUMN], errors="coerce").to_numpy(dtype=float)

        # np.interp already clamps to the end points, which is the "constant"
        # behaviour; "linear" then re-applies the end-point ratio beyond the
        # fitted range so extreme forecasts are not flattened to the maximum
        # observed value.
        mapped = np.interp(values, self._forecast_q, self._observed_q)
        if self.extrapolate == "linear":
            hi_f, hi_o = self._forecast_q[-1], self._observed_q[-1]
            above = values > hi_f
            if hi_f > 0:
                mapped = np.where(above, hi_o * (values / hi_f), mapped)
            else:
                mapped = np.where(above, hi_o + (values - hi_f), mapped)
        out = pd.Series(mapped, index=df.index, name="corrected_mm")
        return out.clip(lower=MIN_RAINFALL_MM)

    def save(self, path: Path) -> Path:
        """Serialise the fitted CDFs to ``path``.

        Args:
            path: Destination ``.joblib`` file.

        Returns:
            The path written.

        Raises:
            NotFittedError: If the mapper is unfitted.
        """
        import joblib

        if self._forecast_q is None:
            raise NotFittedError("Cannot save an unfitted QuantileMapping.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "n_quantiles": self.n_quantiles,
                "extrapolate": self.extrapolate,
                "probs": self._probs,
                "forecast_q": self._forecast_q,
                "observed_q": self._observed_q,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "QuantileMapping":
        """Load a mapper written by :meth:`save`.

        Args:
            path: The ``.joblib`` file.

        Returns:
            A fitted :class:`QuantileMapping`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No quantile-mapping artifact at '{path}'.")
        blob = joblib.load(path)
        obj = cls(n_quantiles=blob["n_quantiles"], extrapolate=blob["extrapolate"])
        obj._probs = blob["probs"]
        obj._forecast_q = blob["forecast_q"]
        obj._observed_q = blob["observed_q"]
        return obj


class RegimeQuantileMapping:
    """Quantile mapping fitted separately per regime.

    Not one of the five mandated rows in the verification report, but it is the
    cheap ablation that separates "regime awareness helps" from "machine
    learning helps" -- comparing it against Baseline C isolates the first, and
    against Model D the second.

    Attributes:
        models: ``{regime: QuantileMapping}`` for each regime seen in training.
        fallback: A globally fitted mapper used for regimes with too little data.
    """

    name = "C_regime_quantile_mapping"

    def __init__(self, *, n_quantiles: int = 100, min_rows: int = 100) -> None:
        """Initialise an unfitted per-regime mapper.

        Args:
            n_quantiles: Quantile knots per regime.
            min_rows: Regimes with fewer training rows than this fall back to
                the global mapper rather than fitting an unstable CDF. Floored
                at 2, since an empirical CDF needs at least two points.
        """
        self.n_quantiles = int(n_quantiles)
        self.min_rows = max(int(min_rows), 2)
        self.models: Dict[str, QuantileMapping] = {}
        self.fallback: Optional[QuantileMapping] = None

    def fit(self, df: pd.DataFrame, regimes: pd.Series) -> "RegimeQuantileMapping":
        """Fit one mapper per regime plus a global fallback.

        Args:
            df: Training frame.
            regimes: Regime label per row.

        Returns:
            ``self``, fitted.
        """
        self.fallback = QuantileMapping(n_quantiles=self.n_quantiles).fit(df)
        self.models = {}
        for regime, group in df.groupby(regimes.values):
            if len(group) < self.min_rows:
                continue
            self.models[str(regime)] = QuantileMapping(n_quantiles=self.n_quantiles).fit(group)
        return self

    def predict(self, df: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        """Map each row through its regime's CDF.

        Args:
            df: Feature table.
            regimes: Regime label per row.

        Returns:
            The corrected forecast in mm.

        Raises:
            NotFittedError: If the mapper is unfitted.
        """
        if self.fallback is None:
            raise NotFittedError("RegimeQuantileMapping is not fitted.")
        out = pd.Series(np.nan, index=df.index, name="corrected_mm", dtype="float64")
        for regime, idx in df.groupby(regimes.values).groups.items():
            model = self.models.get(str(regime), self.fallback)
            out.loc[idx] = model.predict(df.loc[idx]).values
        return out
