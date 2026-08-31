"""Stage 4 -- calibrated heavy-rainfall probability (the head that makes Model E).

One binary classifier per IMD threshold (heavy / very heavy / extremely heavy),
each taking the *corrected* forecast plus the full feature vector and predicting
``observed_mm > threshold``.

Calibration matters more than discrimination here. A warning product's
probability is read as a probability -- "30% chance of heavy rain" has to mean
that it rains heavily on about 30% of such days, or the number is worse than
useless for decision-making. Gradient-boosted classifiers on heavily imbalanced
targets are usually badly miscalibrated, so each classifier's raw score is
passed through an isotonic regression (or Platt sigmoid) fitted on a *held-out*
calibration split.

The calibration split must be chronologically separate from the training split.
``sklearn.calibration.CalibratedClassifierCV`` with its default k-fold CV would
shuffle across time and leak, so the fit is done explicitly here instead; pass
``method="sklearn_isotonic"`` if you specifically want the sklearn wrapper with
a ``prefit`` estimator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config.thresholds import RAIN_THRESHOLDS
from ..data import schema as sch
from ..features.engineering import FEATURE_COLUMNS, select_features
from .baselines import NotFittedError

#: Artifact filename under ``ARTIFACT_DIR``.
PROBABILITY_MODEL_FILENAME = "heavy_rain_probability.joblib"

#: The corrected forecast is appended to the feature matrix under this name.
CORRECTED_FEATURE = "corrected_forecast_mm"

#: Features used by the probability head: everything the correctors see, plus
#: the corrected forecast itself.
PROBABILITY_FEATURE_COLUMNS: List[str] = FEATURE_COLUMNS + [CORRECTED_FEATURE]


def binary_target(df: pd.DataFrame, threshold_mm: float) -> pd.Series:
    """Build the binary exceedance target ``observed_mm > threshold_mm``.

    Args:
        df: Frame containing ``observed_mm``.
        threshold_mm: Rainfall threshold in mm.

    Returns:
        An int Series of 0/1, with NaN observations dropped to NaN.

    Raises:
        KeyError: If ``observed_mm`` is absent.
    """
    if sch.OBSERVED_COLUMN not in df.columns:
        raise KeyError(f"binary_target needs column '{sch.OBSERVED_COLUMN}'.")
    obs = pd.to_numeric(df[sch.OBSERVED_COLUMN], errors="coerce")
    return (obs > threshold_mm).where(obs.notna()).rename(f"exceeds_{threshold_mm}")


def attach_corrected_forecast(
    df: pd.DataFrame, corrected: pd.Series
) -> pd.DataFrame:
    """Attach the corrected forecast to a feature table under a stable name.

    Args:
        df: Feature table.
        corrected: Corrected forecast in mm, aligned to ``df.index``.

    Returns:
        A copy of ``df`` with :data:`CORRECTED_FEATURE` added.
    """
    out = df.copy()
    out[CORRECTED_FEATURE] = pd.to_numeric(
        pd.Series(np.asarray(corrected), index=df.index), errors="coerce"
    )
    return out


class _CalibratedBinaryModel:
    """A single threshold's classifier plus its calibrator.

    Attributes:
        threshold_mm: The rainfall threshold this model predicts exceedance of.
        backend: ``"lightgbm"`` or ``"xgboost"``.
        method: ``"isotonic"``, ``"sigmoid"`` or ``"sklearn_isotonic"``.
        base_rate: Fraction of positive rows in the training split, kept for the
            report and used as the constant prediction in the degenerate case.
    """

    def __init__(
        self,
        threshold_mm: float,
        *,
        backend: str = "lightgbm",
        method: str = "isotonic",
        feature_columns: Optional[Sequence[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise an unfitted threshold model.

        Args:
            threshold_mm: Exceedance threshold in mm.
            backend: ``"lightgbm"`` or ``"xgboost"``.
            method: Calibration method.
            feature_columns: Feature order. Defaults to
                :data:`PROBABILITY_FEATURE_COLUMNS`.
            params: Extra estimator parameters merged over the defaults.

        Raises:
            ValueError: If ``backend`` or ``method`` is unknown.
        """
        if backend not in {"lightgbm", "xgboost"}:
            raise ValueError(f"Unknown backend '{backend}'.")
        if method not in {"isotonic", "sigmoid", "sklearn_isotonic", "none"}:
            raise ValueError(f"Unknown calibration method '{method}'.")
        self.threshold_mm = float(threshold_mm)
        self.backend = backend
        self.method = method
        self.feature_columns: List[str] = list(
            feature_columns if feature_columns is not None else PROBABILITY_FEATURE_COLUMNS
        )
        # Untuned defaults. Heavy-rain exceedance is a rare-event problem, so
        # the tree size is kept modest to limit overfitting on few positives.
        self.params: Dict[str, Any] = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        if params:
            self.params.update(params)
        self.model: Any = None
        self.calibrator: Any = None
        self.base_rate: float = float("nan")
        self._degenerate: bool = False

    def _build(self) -> Any:
        """Instantiate the underlying binary classifier."""
        if self.backend == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                objective="binary",
                class_weight="balanced" if self.params.get("balanced", False) else None,
                n_estimators=self.params["n_estimators"],
                learning_rate=self.params["learning_rate"],
                num_leaves=self.params.get("num_leaves", 31),
                max_depth=self.params.get("max_depth", 6),
                subsample=self.params.get("subsample", 0.8),
                colsample_bytree=self.params.get("colsample_bytree", 0.8),
                random_state=self.params.get("random_state", 42),
                verbose=-1,
            )
        from xgboost import XGBClassifier

        pos_weight = self.params.get("scale_pos_weight", 1.0)
        return XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            scale_pos_weight=pos_weight,
            n_estimators=self.params["n_estimators"],
            learning_rate=self.params["learning_rate"],
            max_depth=self.params.get("max_depth", 6),
            subsample=self.params.get("subsample", 0.8),
            colsample_bytree=self.params.get("colsample_bytree", 0.8),
            random_state=self.params.get("random_state", 42),
        )

    def fit(
        self,
        train_df: pd.DataFrame,
        calib_df: Optional[pd.DataFrame] = None,
    ) -> "_CalibratedBinaryModel":
        """Fit the classifier on ``train_df`` and the calibrator on ``calib_df``.

        Args:
            train_df: Training rows, with ``observed_mm`` and the corrected
                forecast already attached.
            calib_df: Chronologically later calibration rows. If None, the
                calibrator is fitted on the training predictions -- workable but
                optimistic, so prefer a real held-out split.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If ``train_df`` has no usable rows.
        """
        y = binary_target(train_df, self.threshold_mm)
        mask = y.notna()
        if not mask.any():
            raise ValueError(
                f"No rows with a usable observation for threshold "
                f"{self.threshold_mm} mm."
            )
        y = y[mask].astype(int)
        X = select_features(train_df[mask], self.feature_columns)
        self.base_rate = float(y.mean())

        if y.nunique() < 2:
            # The threshold was never exceeded (or always was) in training. A
            # classifier cannot be fitted; predict the constant base rate and
            # flag it clearly rather than crashing the whole training run --
            # 204.4 mm exceedances are genuinely rare and a short training
            # period may contain none.
            self._degenerate = True
            self.model = None
            self.calibrator = None
            return self

        self._degenerate = False
        self.model = self._build()
        self.model.fit(X.values, y.values)
        self._fit_calibrator(calib_df if calib_df is not None else train_df)
        return self

    def _fit_calibrator(self, df: pd.DataFrame) -> None:
        """Fit the probability calibrator on ``df``.

        Args:
            df: Calibration rows, ideally chronologically after training.
        """
        if self.method == "none":
            self.calibrator = None
            return

        y = binary_target(df, self.threshold_mm)
        mask = y.notna()
        y = y[mask].astype(int)
        if y.nunique() < 2 or len(y) < 10:
            # Not enough signal to calibrate against; leave the raw scores
            # rather than fitting a degenerate mapping.
            self.calibrator = None
            return

        raw = self._raw_scores(df[mask])
        if self.method == "sigmoid":
            from sklearn.linear_model import LogisticRegression

            cal = LogisticRegression(solver="lbfgs")
            cal.fit(raw.reshape(-1, 1), y.values)
        else:
            # "isotonic" and "sklearn_isotonic" both end up as an isotonic map
            # on the raw score; the difference is only which class does it.
            from sklearn.isotonic import IsotonicRegression

            cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            cal.fit(raw, y.values)
        self.calibrator = cal

    def _raw_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Return the uncalibrated positive-class probability for each row."""
        X = select_features(df, self.feature_columns)
        return np.asarray(self.model.predict_proba(X.values))[:, 1]

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Predict the calibrated exceedance probability.

        Args:
            df: Feature table with the corrected forecast attached.

        Returns:
            Probabilities in ``[0, 1]``, aligned to ``df.index``.

        Raises:
            NotFittedError: If the model was never fitted at all.
        """
        if self._degenerate:
            if np.isnan(self.base_rate):
                raise NotFittedError(
                    f"Threshold model for {self.threshold_mm} mm is not fitted."
                )
            return pd.Series(self.base_rate, index=df.index, dtype="float64")
        if self.model is None:
            raise NotFittedError(
                f"Threshold model for {self.threshold_mm} mm is not fitted. Run "
                f"training/train_heavy_rain_models.py after connecting real data."
            )

        raw = self._raw_scores(df)
        if self.calibrator is None:
            calibrated = raw
        elif self.method == "sigmoid":
            calibrated = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            calibrated = self.calibrator.predict(raw)
        return pd.Series(np.clip(calibrated, 0.0, 1.0), index=df.index, dtype="float64")


class HeavyRainProbabilityModel:
    """One calibrated binary classifier per IMD rainfall threshold.

    Attributes:
        thresholds: ``{name: mm}`` this model covers.
        models: ``{name: _CalibratedBinaryModel}`` after fitting.
    """

    name = "E_heavy_rain_probability"

    def __init__(
        self,
        *,
        thresholds: Optional[Dict[str, float]] = None,
        backend: str = "lightgbm",
        method: str = "isotonic",
        feature_columns: Optional[Sequence[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise an unfitted probability head.

        Args:
            thresholds: ``{name: mm}``. Defaults to the IMD categories.
            backend: ``"lightgbm"`` or ``"xgboost"``.
            method: Calibration method.
            feature_columns: Feature order.
            params: Extra estimator parameters.
        """
        self.thresholds: Dict[str, float] = dict(thresholds or RAIN_THRESHOLDS)
        self.backend = backend
        self.method = method
        self.feature_columns = list(
            feature_columns if feature_columns is not None else PROBABILITY_FEATURE_COLUMNS
        )
        self.params = dict(params) if params else None
        self.models: Dict[str, _CalibratedBinaryModel] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        calib_df: Optional[pd.DataFrame] = None,
    ) -> "HeavyRainProbabilityModel":
        """Fit one classifier + calibrator per threshold.

        Args:
            train_df: Training rows with the corrected forecast attached (see
                :func:`attach_corrected_forecast`).
            calib_df: Chronologically later calibration rows, also with the
                corrected forecast attached.

        Returns:
            ``self``, fitted.
        """
        self.models = {}
        for name, mm in self.thresholds.items():
            self.models[name] = _CalibratedBinaryModel(
                mm,
                backend=self.backend,
                method=self.method,
                feature_columns=self.feature_columns,
                params=self.params,
            ).fit(train_df, calib_df)
        return self

    def predict_proba(
        self, df: pd.DataFrame, *, enforce_monotonicity: bool = True
    ) -> pd.DataFrame:
        """Predict calibrated probabilities for every threshold.

        Args:
            df: Feature table with the corrected forecast attached.
            enforce_monotonicity: If True, guarantees P(higher threshold) <= P(lower threshold).

        Returns:
            A frame with one column per threshold name.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
        """
        if not self.models:
            raise NotFittedError(
                "HeavyRainProbabilityModel is not fitted. Run "
                "training/train_heavy_rain_models.py after connecting real data."
            )
        probs = pd.DataFrame(
            {name: model.predict_proba(df) for name, model in self.models.items()},
            index=df.index,
        )
        if enforce_monotonicity and len(self.thresholds) > 1:
            sorted_items = sorted(self.thresholds.items(), key=lambda x: x[1])
            prev_col = None
            for name, _ in sorted_items:
                if prev_col is not None and name in probs.columns and prev_col in probs.columns:
                    probs[name] = np.minimum(probs[name], probs[prev_col])
                prev_col = name
        return probs

    def degenerate_thresholds(self) -> List[str]:
        """Return thresholds that had no positive (or no negative) training rows.

        These fall back to predicting the constant training base rate, which the
        verification report should call out rather than treating as a real model.

        Returns:
            Threshold names that could not be fitted properly.
        """
        return [n for n, m in self.models.items() if m._degenerate]

    def save(self, path: Path) -> Path:
        """Serialise every threshold model to a single ``.joblib`` file.

        Args:
            path: Destination file.

        Returns:
            The path written.

        Raises:
            NotFittedError: If unfitted.
        """
        import joblib

        if not self.models:
            raise NotFittedError("Cannot save an unfitted HeavyRainProbabilityModel.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "thresholds": self.thresholds,
                "backend": self.backend,
                "method": self.method,
                "feature_columns": self.feature_columns,
                "params": self.params,
                "models": self.models,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "HeavyRainProbabilityModel":
        """Load a head written by :meth:`save`.

        Args:
            path: The ``.joblib`` file.

        Returns:
            A fitted :class:`HeavyRainProbabilityModel`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No heavy-rain probability artifact at '{path}'.")
        blob = joblib.load(path)
        obj = cls(
            thresholds=blob["thresholds"],
            backend=blob["backend"],
            method=blob["method"],
            feature_columns=blob["feature_columns"],
            params=blob["params"],
        )
        obj.models = blob["models"]
        return obj


def predict_heavy_rain_probability(
    feature_row: pd.DataFrame,
    corrected_forecast: float,
    model: HeavyRainProbabilityModel,
) -> Dict[str, float]:
    """Predict calibrated exceedance probabilities for a single grid cell.

    Args:
        feature_row: A one-row feature table.
        corrected_forecast: The corrected rainfall forecast in mm for that row.
        model: A fitted :class:`HeavyRainProbabilityModel`.

    Returns:
        ``{threshold_name: calibrated_probability}``.

    Raises:
        ValueError: If ``feature_row`` does not have exactly one row.
        NotFittedError: If the model is unfitted.
    """
    if len(feature_row) != 1:
        raise ValueError(
            f"predict_heavy_rain_probability expects exactly 1 row, got {len(feature_row)}."
        )
    with_corrected = attach_corrected_forecast(
        feature_row, pd.Series([corrected_forecast], index=feature_row.index)
    )
    probs = model.predict_proba(with_corrected).iloc[0]
    return {str(k): float(v) for k, v in probs.items()}
