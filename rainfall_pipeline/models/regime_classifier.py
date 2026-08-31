"""Stage 2 -- the regime engine.

Two pieces:

1. :func:`label_regimes` -- a deterministic, rule-based labeller that assigns
   one of the five monsoon regimes to every ``(date, lat, lon)`` row using
   simplified meteorological criteria. Its output is the *training target*.
2. :class:`RegimeClassifier` -- a gradient-boosted multi-class classifier that
   learns to reproduce those labels from atmospheric features alone, so that at
   prediction time a regime can be assigned without needing the observation or
   the climatology-based anomaly.

Why learn a classifier for something a rule already decides? Because the rule
needs ``rain_anomaly_sd``, which on historical rows is computed from the
observation. The classifier distils the rule into something evaluable from
forecast-time information only, and its probability output gives the router a
soft regime assignment instead of a hard one.

All thresholds live in :mod:`rainfall_pipeline.config.thresholds` and are
PLACEHOLDERS until replaced with the official IMD/IITM definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config.thresholds import (
    INDEX_TO_REGIME,
    REGIME_ACTIVE,
    REGIME_BREAK,
    REGIME_COASTAL,
    REGIME_DEPRESSION_LOW,
    REGIME_LABELS,
    REGIME_OROGRAPHIC,
    REGIME_RULES,
    REGIME_TO_INDEX,
    RegimeRuleConfig,
)
from ..features.engineering import REGIME_FEATURE_COLUMNS, select_features

#: Column the rule-based labeller writes into.
REGIME_COLUMN = "regime"

#: Artifact filename under ``ARTIFACT_DIR``.
REGIME_MODEL_FILENAME = "regime_classifier.joblib"


class ModelNotTrainedError(RuntimeError):
    """Raised when a prediction is requested before a model has been fitted."""


# ---------------------------------------------------------------------------
# 1. Rule-based labelling
# ---------------------------------------------------------------------------

def compute_core_zone_indices(
    df: pd.DataFrame, config: RegimeRuleConfig = REGIME_RULES
) -> pd.DataFrame:
    """Average the active/break indices over the reference box, per date.

    The published active/break definitions are *large-scale, day-level*
    statements about the monsoon: a day is active or in a break over the core
    monsoon zone as a whole, not at a single grid point. So the two indices are
    reduced to one value per date before being compared against thresholds.

    Args:
        df: Feature table containing ``date``, ``lat``, ``lon``,
            ``rain_anomaly_sd`` and ``wind_u_850``.
        config: Rule thresholds; only ``core_zone_bbox`` is used here.

    Returns:
        A frame indexed by ``date`` with ``core_rain_anomaly`` and
        ``core_u850`` columns.
    """
    lat_min, lat_max, lon_min, lon_max = config.core_zone_bbox
    in_box = (
        df["lat"].between(lat_min, lat_max) & df["lon"].between(lon_min, lon_max)
    )
    zone = df[in_box]
    if zone.empty:
        # No grid cells fall in the reference box (a small regional dataset, or
        # a bbox that needs adjusting). Fall back to the whole domain rather
        # than producing all-NaN indices, and make it obvious in the column.
        zone = df

    anomaly = (
        zone["rain_anomaly_sd"]
        if "rain_anomaly_sd" in zone.columns
        else pd.Series(np.nan, index=zone.index)
    )
    work = pd.DataFrame(
        {"date": zone["date"], "rain_anomaly_sd": anomaly, "wind_u_850": zone["wind_u_850"]}
    )
    return (
        work.groupby("date", as_index=False)
        .agg(core_rain_anomaly=("rain_anomaly_sd", "mean"), core_u850=("wind_u_850", "mean"))
        .set_index("date")
    )


def label_regimes(
    df: pd.DataFrame, config: RegimeRuleConfig = REGIME_RULES
) -> pd.Series:
    """Assign a rule-based regime label to every row.

    Precedence, highest first. A cell can satisfy several criteria at once (the
    Western Ghats are both coastal and orographic), so the order below decides
    which mechanism is treated as dominant. Change the order here if your
    meteorological reading differs -- nothing downstream depends on it.

    1. **Depression-Low** -- strong positive relative vorticity *or* a
       markedly negative pressure anomaly. A synoptic low dominates whatever
       else is going on locally.
    2. **Orographic** -- high terrain with strong low-level flow. Forced ascent
       over the barrier is the controlling mechanism.
    3. **Coastal** -- near the coast with strong onshore-capable flow.
    4. **Active** -- day-level: core-zone rainfall anomaly above threshold and
       a strong low-level westerly jet.
    5. **Break** -- day-level: core-zone anomaly below threshold and weak
       westerlies.

    Anything matching none of the above falls back to whichever of Active/Break
    the day-level rainfall anomaly is closer to, so every row gets a label.

    Args:
        df: Feature table. Needs ``date``, ``lat``, ``lon``, ``vorticity``,
            ``pressure_anomaly``, ``elevation``, ``coastal_distance``,
            ``wind_speed_850``, ``wind_u_850`` and ``rain_anomaly_sd``.
        config: Rule thresholds.

    Returns:
        A string Series aligned to ``df.index`` with values drawn from
        :data:`~rainfall_pipeline.config.thresholds.REGIME_LABELS`.

    Raises:
        KeyError: If a required column is missing.
    """
    required = [
        "date", "lat", "lon", "vorticity", "elevation", "coastal_distance",
        "wind_speed_850", "wind_u_850",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"label_regimes needs column(s) {missing}. Run "
            f"features.engineering.add_derived_features first."
        )

    indices = compute_core_zone_indices(df, config)
    core_rain = df["date"].map(indices["core_rain_anomaly"])
    core_u850 = df["date"].map(indices["core_u850"])

    pressure_anomaly = (
        df["pressure_anomaly"]
        if "pressure_anomaly" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    is_depression = (df["vorticity"] >= config.depression_vorticity) | (
        pressure_anomaly <= config.depression_pressure_anomaly_hpa
    )
    is_orographic = (df["elevation"] >= config.orographic_elevation_m) & (
        df["wind_speed_850"] >= config.orographic_wind_speed
    )
    is_coastal = (df["coastal_distance"] <= config.coastal_distance_km) & (
        df["wind_speed_850"] >= config.coastal_wind_speed
    )
    is_active = (core_rain >= config.active_rain_anomaly_sd) & (
        core_u850 >= config.active_zonal_wind_850
    )
    is_break = (core_rain <= config.break_rain_anomaly_sd) & (
        core_u850 <= config.break_zonal_wind_850
    )

    # Fallback for days that are neither clearly active nor clearly in a break:
    # side with whichever end of the anomaly scale is nearer. NaN anomalies
    # (no climatology available) fall to Break, the drier default.
    fallback = np.where(core_rain.fillna(-np.inf) >= 0.0, REGIME_ACTIVE, REGIME_BREAK)

    labels = pd.Series(fallback, index=df.index, dtype=object)
    labels = labels.mask(is_break, REGIME_BREAK)
    labels = labels.mask(is_active, REGIME_ACTIVE)
    labels = labels.mask(is_coastal, REGIME_COASTAL)
    labels = labels.mask(is_orographic, REGIME_OROGRAPHIC)
    labels = labels.mask(is_depression, REGIME_DEPRESSION_LOW)
    return labels.astype("object").rename(REGIME_COLUMN)


def regime_label_summary(labels: pd.Series) -> Dict[str, int]:
    """Count rows per regime, including regimes with zero rows.

    Args:
        labels: Output of :func:`label_regimes`.

    Returns:
        ``{regime_name: row_count}`` covering all five regimes.
    """
    counts = labels.value_counts().to_dict()
    return {name: int(counts.get(name, 0)) for name in REGIME_LABELS}


# ---------------------------------------------------------------------------
# 2. Learned classifier
# ---------------------------------------------------------------------------

@dataclass
class RegimeExplanation:
    """SHAP explanation for a single regime prediction.

    Attributes:
        regime: The predicted regime label.
        probabilities: ``{regime: probability}`` over all five regimes.
        top_features: ``[(feature_name, shap_value), ...]`` ordered by absolute
            contribution, largest first, for the predicted class.
        base_value: The model's expected output for the predicted class before
            any feature contributions.
    """

    regime: str
    probabilities: Dict[str, float]
    top_features: List[tuple[str, float]]
    base_value: float

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "regime": self.regime,
            "probabilities": self.probabilities,
            "top_features": [
                {"feature": f, "shap_value": float(v)} for f, v in self.top_features
            ],
            "base_value": float(self.base_value),
        }


class RegimeClassifier:
    """Multi-class regime classifier over atmospheric features.

    Wraps XGBoost (default) or CatBoost behind a small, stable interface so the
    rest of the pipeline never touches the underlying library directly.

    Attributes:
        backend: ``"xgboost"`` or ``"catboost"``.
        feature_columns: Columns the model was fitted on, in order.
        model: The fitted estimator, or None before :meth:`fit`.
    """

    def __init__(
        self,
        *,
        backend: str = "xgboost",
        feature_columns: Optional[Sequence[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise an unfitted classifier.

        Args:
            backend: ``"xgboost"``, ``"catboost"``, ``"lightgbm"``, or ``"ensemble"``.
            feature_columns: Feature order. Defaults to
                :data:`~rainfall_pipeline.features.engineering.REGIME_FEATURE_COLUMNS`.
            params: Extra estimator parameters, merged over the defaults.
        """
        valid_backends = {"xgboost", "catboost", "lightgbm", "ensemble"}
        if backend not in valid_backends:
            raise ValueError(f"Unknown backend '{backend}'. Valid: {sorted(valid_backends)}")
        self.backend = backend
        self.feature_columns: List[str] = list(
            feature_columns if feature_columns is not None else REGIME_FEATURE_COLUMNS
        )
        # Conservative, untuned defaults. Do NOT tune these on the dummy test
        # rows -- tune on a real validation split once data is connected.
        self.params: Dict[str, Any] = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "random_state": 42,
        }
        if params:
            self.params.update(params)
        self.model: Any = None
        self.ensemble_models: Optional[Dict[str, Any]] = None
        self._explainer: Any = None

    def _build_single(self, backend: str, n_classes: int) -> Any:
        """Instantiate a single estimator for the given backend."""
        if backend == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                tree_method="hist",
                **self.params,
            )
        if backend == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                objective="multiclass",
                num_class=n_classes,
                n_estimators=self.params.get("n_estimators", 300),
                max_depth=self.params.get("max_depth", 6),
                learning_rate=self.params.get("learning_rate", 0.05),
                subsample=self.params.get("subsample", 0.8),
                random_state=self.params.get("random_state", 42),
                verbose=-1,
            )
        from catboost import CatBoostClassifier

        cat_params = {
            "iterations": self.params.get("n_estimators", 300),
            "depth": self.params.get("max_depth", 6),
            "learning_rate": self.params.get("learning_rate", 0.05),
            "random_seed": self.params.get("random_state", 42),
            "loss_function": "MultiClass",
            "verbose": False,
        }
        return CatBoostClassifier(**cat_params)

    def _build(self, n_classes: int) -> Any:
        """Instantiate the underlying estimator or ensemble dictionary."""
        if self.backend == "ensemble":
            return {
                "xgboost": self._build_single("xgboost", n_classes),
                "lightgbm": self._build_single("lightgbm", n_classes),
                "catboost": self._build_single("catboost", n_classes),
            }
        return self._build_single(self.backend, n_classes)

    def fit(self, df: pd.DataFrame, labels: pd.Series) -> "RegimeClassifier":
        """Fit the classifier on rule-based labels.

        Args:
            df: Feature table.
            labels: Regime label per row, as produced by :func:`label_regimes`.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If fewer than two distinct regimes are present, which
                means the rule thresholds need adjusting for this dataset.
        """
        X = select_features(df, self.feature_columns)
        y = labels.map(REGIME_TO_INDEX)
        if y.isna().any():
            bad = sorted(set(labels[y.isna()]))
            raise ValueError(f"Unknown regime label(s) in training target: {bad}")
        y = y.astype(int)

        present = sorted(y.unique())
        if len(present) < 2:
            only = INDEX_TO_REGIME[present[0]] if present else "none"
            raise ValueError(
                f"Only one regime ('{only}') is present in the training data, so a "
                f"classifier cannot be fitted. This usually means the rule "
                f"thresholds in config/thresholds.py::RegimeRuleConfig do not "
                f"discriminate on your data -- inspect "
                f"regime_label_summary(label_regimes(df)) and adjust them."
            )

        # XGBoost requires contiguous class indices 0..k-1. Keep a mapping back
        # to the canonical regime index so probabilities stay comparable across
        # models trained on different subsets.
        self._present_classes = present
        remap = {orig: i for i, orig in enumerate(present)}
        y_fit = y.map(remap).astype(int)

        built = self._build(len(present))
        if self.backend == "ensemble":
            self.ensemble_models = {}
            for name, m in built.items():
                m.fit(X.values, y_fit.values)
                self.ensemble_models[name] = m
            self.model = self.ensemble_models["xgboost"]
        else:
            self.model = built
            self.model.fit(X.values, y_fit.values)
            self.ensemble_models = None
        self._explainer = None
        return self

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if self.model is None:
            raise ModelNotTrainedError(
                "RegimeClassifier has not been fitted. Run "
                "training/train_regime_classifier.py after connecting real data."
            )

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict the regime probability distribution for each row.

        Args:
            df: Feature table.

        Returns:
            A frame with one column per regime in
            :data:`~rainfall_pipeline.config.thresholds.REGIME_LABELS` order.
            Regimes absent from the training data get probability 0.

        Raises:
            ModelNotTrainedError: If :meth:`fit` has not been called.
        """
        self._check_fitted()
        X = select_features(df, self.feature_columns)
        if self.ensemble_models is not None:
            raw_list = [np.asarray(m.predict_proba(X.values)) for m in self.ensemble_models.values()]
            raw = np.mean(raw_list, axis=0)
        else:
            raw = np.asarray(self.model.predict_proba(X.values))
        out = pd.DataFrame(0.0, index=df.index, columns=REGIME_LABELS)
        for col_idx, orig_idx in enumerate(self._present_classes):
            out[INDEX_TO_REGIME[orig_idx]] = raw[:, col_idx]
        return out

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict the most likely regime for each row.

        Args:
            df: Feature table.

        Returns:
            A string Series of regime labels.
        """
        return self.predict_proba(df).idxmax(axis=1).rename(REGIME_COLUMN)

    # -- explainability --------------------------------------------------

    def _get_explainer(self) -> Any:
        """Return a cached SHAP TreeExplainer for the fitted model."""
        self._check_fitted()
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def shap_values(self, df: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for every row and class.

        Args:
            df: Feature table.

        Returns:
            Array of shape ``(n_rows, n_features, n_present_classes)``.

        Raises:
            ModelNotTrainedError: If the model is unfitted.
        """
        explainer = self._get_explainer()
        X = select_features(df, self.feature_columns)
        values = explainer.shap_values(X.values)
        arr = np.asarray(values)
        if arr.ndim == 2:
            # Binary/degenerate case: promote to a single-class third axis.
            arr = arr[:, :, np.newaxis]
        elif arr.ndim == 3 and arr.shape[0] == len(self._present_classes):
            # Older SHAP returns (n_classes, n_rows, n_features).
            arr = np.transpose(arr, (1, 2, 0))
        return arr

    def explain_row(self, df: pd.DataFrame, top_n: int = 5) -> RegimeExplanation:
        """Explain the prediction for the first row of ``df``.

        Args:
            df: A feature table; only the first row is explained.
            top_n: How many contributing features to return.

        Returns:
            A :class:`RegimeExplanation` for that row.

        Raises:
            ValueError: If ``df`` is empty.
            ModelNotTrainedError: If the model is unfitted.
        """
        if df.empty:
            raise ValueError("explain_row needs at least one row.")
        row = df.iloc[[0]]
        probs = self.predict_proba(row).iloc[0]
        regime = str(probs.idxmax())

        arr = self.shap_values(row)
        try:
            class_pos = self._present_classes.index(REGIME_TO_INDEX[regime])
        except ValueError:  # predicted class not in the fitted set - impossible
            class_pos = 0
        contributions = arr[0, :, min(class_pos, arr.shape[2] - 1)]

        order = np.argsort(-np.abs(contributions))[:top_n]
        top = [(self.feature_columns[i], float(contributions[i])) for i in order]

        expected = self._get_explainer().expected_value
        base = float(np.atleast_1d(expected)[min(class_pos, np.atleast_1d(expected).size - 1)])
        return RegimeExplanation(
            regime=regime,
            probabilities={k: float(v) for k, v in probs.items()},
            top_features=top,
            base_value=base,
        )

    # -- persistence -----------------------------------------------------

    def save(self, path: Path) -> Path:
        """Serialise the fitted model to ``path``.

        Args:
            path: Destination ``.joblib`` file.

        Returns:
            The path written.

        Raises:
            ModelNotTrainedError: If the model is unfitted.
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
                "model": self.model,
                "present_classes": self._present_classes,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "RegimeClassifier":
        """Load a classifier previously written by :meth:`save`.

        Args:
            path: The ``.joblib`` file.

        Returns:
            A fitted :class:`RegimeClassifier`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No regime classifier artifact at '{path}'.")
        blob = joblib.load(path)
        obj = cls(
            backend=blob["backend"],
            feature_columns=blob["feature_columns"],
            params=blob["params"],
        )
        obj.model = blob["model"]
        obj._present_classes = blob["present_classes"]
        return obj


# ---------------------------------------------------------------------------
# Convenience API used by the FastAPI service
# ---------------------------------------------------------------------------

def predict_regime(
    feature_row: pd.DataFrame,
    classifier: RegimeClassifier,
    *,
    top_n: int = 5,
) -> RegimeExplanation:
    """Predict the regime for a single feature row, with a SHAP explanation.

    Args:
        feature_row: A one-row feature table (as produced by
            :func:`~rainfall_pipeline.features.engineering.add_derived_features`).
        classifier: A fitted :class:`RegimeClassifier`.
        top_n: How many contributing features to return.

    Returns:
        The regime label, the full probability distribution and the top
        contributing features.

    Raises:
        ModelNotTrainedError: If the classifier is unfitted.
    """
    return classifier.explain_row(feature_row, top_n=top_n)
