"""
XGBoost price regression model for the PowerPrice Futures Signals platform.

Predicts the expected electricity spot price (EUR/MWh) ``horizon_hours``
ahead.  Walk-forward (TimeSeriesSplit) cross-validation is used to avoid
look-ahead bias.  Reported metrics: MAE, RMSE, R², MAPE.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from app.core.config import settings
from app.core.logging import get_logger
from app.features.engineering import FeatureEngineer

logger = get_logger(__name__)

_MODEL_FILE = "price_regression.joblib"
_SCALER_FILE = "price_regression_scaler.joblib"
_META_FILE = "price_regression_meta.json"


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error; skips rows where y_true == 0."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


class PriceRegressionModel:
    """Predicts expected electricity price (EUR/MWh) for the next N hours.

    The target variable is ``price_eur_mwh`` shifted ``horizon_hours``
    into the future relative to each feature row.  This means the model
    answers: "given the current state of the grid and market, what price
    should I expect in ``horizon_hours`` hours?"

    Parameters
    ----------
    model_dir:
        Directory for persisting model artefacts.
    horizon_hours:
        Prediction horizon in hours (default 4).
    """

    def __init__(self, model_dir: str, horizon_hours: int = 4) -> None:
        self.model_dir = model_dir
        self.horizon_hours = horizon_hours

        self.model: Optional[XGBRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.metrics: Dict = {}

        self._fe = FeatureEngineer()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> Dict:
        """Train the regression model with walk-forward cross-validation.

        Parameters
        ----------
        df:
            Raw HourlyPrice DataFrame sorted by timestamp ascending.

        Returns
        -------
        dict
            Keys: ``mae``, ``rmse``, ``r2``, ``mape``, ``n_train``,
            ``n_folds``, ``training_time_s``.
        """
        t0 = time.perf_counter()
        logger.info(
            "Training PriceRegressionModel (horizon=%dh)", self.horizon_hours
        )

        X, _ = self._fe.get_feature_matrix(df)

        # Build forward-shifted target aligned to X's index
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        price_series = df_sorted["price_eur_mwh"]
        y_full = price_series.shift(-self.horizon_hours)  # future price
        y_full.index = range(len(y_full))

        # Align to valid rows
        y = y_full.loc[X.index]

        # Drop rows where future price is NaN (tail of dataset)
        valid = y.notna()
        X = X.loc[valid].reset_index(drop=True)
        y = y.loc[valid].reset_index(drop=True)

        if len(X) == 0:
            raise ValueError("No usable rows after target alignment and NaN removal.")

        self.feature_names = list(X.columns)
        n_total = len(X)

        logger.info("Regression dataset: %d rows", n_total)

        xgb_params = dict(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )

        # Walk-forward cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        fold_metrics: List[Dict] = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            model = XGBRegressor(**xgb_params)
            model.fit(
                X_tr_s, y_tr,
                eval_set=[(X_val_s, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_val_s)
            y_val_np = y_val.values

            fold_metrics.append({
                "mae": mean_absolute_error(y_val_np, y_pred),
                "rmse": float(np.sqrt(mean_squared_error(y_val_np, y_pred))),
                "r2": r2_score(y_val_np, y_pred),
                "mape": _mape(y_val_np, y_pred),
            })
            logger.debug(
                "Fold %d: MAE=%.2f RMSE=%.2f R2=%.4f",
                fold_idx,
                fold_metrics[-1]["mae"],
                fold_metrics[-1]["rmse"],
                fold_metrics[-1]["r2"],
            )

        if not fold_metrics:
            raise ValueError("No CV folds completed.")

        avg_metrics = {
            k: float(np.mean([m[k] for m in fold_metrics if not np.isnan(m[k])]))
            for k in fold_metrics[0]
        }

        # Final model on full dataset
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = XGBRegressor(**xgb_params)
        self.model.fit(X_scaled, y, verbose=False)

        elapsed = time.perf_counter() - t0
        self.metrics = {
            **avg_metrics,
            "n_train": n_total,
            "n_folds": len(fold_metrics),
            "training_time_s": round(elapsed, 2),
        }

        logger.info(
            "PriceRegressionModel trained: MAE=%.2f RMSE=%.2f R2=%.4f (%.1fs)",
            self.metrics["mae"],
            self.metrics["rmse"],
            self.metrics["r2"],
            elapsed,
        )
        return self.metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> float:
        """Return predicted price in EUR/MWh (scalar).

        Parameters
        ----------
        features:
            DataFrame row(s) produced by
            :meth:`FeatureEngineer.build_features`.

        Returns
        -------
        float
            Predicted price in EUR/MWh for ``horizon_hours`` ahead.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not loaded. Call load() or train() first.")

        X = self._align_features(features)
        X_s = self.scaler.transform(X)
        price = float(self.model.predict(X_s)[0])
        return price

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> str:
        """Persist model, scaler, and metadata to ``model_dir``.

        Returns
        -------
        str
            Absolute path to the saved model file.
        """
        if self.model is None:
            raise RuntimeError("No trained model to save.")

        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(self.model_dir, _MODEL_FILE)
        scaler_path = os.path.join(self.model_dir, _SCALER_FILE)
        meta_path = os.path.join(self.model_dir, _META_FILE)

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        meta = {
            "horizon_hours": self.horizon_hours,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "saved_at": pd.Timestamp.utcnow().isoformat(),
        }
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        logger.info("PriceRegressionModel saved to %s", model_path)
        return model_path

    def load(self) -> None:
        """Load model artefacts from ``model_dir``."""
        model_path = os.path.join(self.model_dir, _MODEL_FILE)
        scaler_path = os.path.join(self.model_dir, _SCALER_FILE)
        meta_path = os.path.join(self.model_dir, _META_FILE)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                meta = json.load(fh)
            self.feature_names = meta.get("feature_names")
            self.metrics = meta.get("metrics", {})
            saved_horizon = meta.get("horizon_hours")
            if saved_horizon is not None:
                self.horizon_hours = saved_horizon

        logger.info("PriceRegressionModel loaded from %s", model_path)

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> Dict[str, float]:
        """Return ``{feature_name: importance_score}`` sorted descending.

        Uses XGBoost's gain-based importance, which reflects the average
        reduction in loss from splitting on each feature.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        booster = self.model.get_booster()
        scores = booster.get_score(importance_type="gain")
        names = self.feature_names or list(scores.keys())

        if all(k.startswith("f") and k[1:].isdigit() for k in scores):
            importance = {
                names[int(k[1:])]: float(v)
                for k, v in scores.items()
                if int(k[1:]) < len(names)
            }
        else:
            importance = {k: float(v) for k, v in scores.items()}

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Select and order columns to match training feature set."""
        if self.feature_names is None:
            raise RuntimeError("feature_names not set; model may not be trained.")

        features = features.copy()
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0.0

        return features[self.feature_names].fillna(0.0)
