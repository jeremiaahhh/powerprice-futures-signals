"""
XGBoost-based binary classifier for electricity price rebound prediction.

Trained exclusively on rows where the spot price is at or below a
near-negative threshold.  Predicts whether the price will recover by at
least ``min_rebound_eur`` EUR/MWh within the next ``horizon_hours`` hours.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier

from app.core.config import settings
from app.core.logging import get_logger
from app.features.engineering import FeatureEngineer

logger = get_logger(__name__)

_MODEL_FILE = "rebound_classifier.joblib"
_SCALER_FILE = "rebound_scaler.joblib"
_META_FILE = "rebound_meta.json"

# Train on rows where price is at or below this value to capture the
# near-negative regime that precedes rebounds.
_NEAR_NEGATIVE_THRESHOLD_EUR = 5.0


class ReboundClassifier:
    """Predicts probability of price rebound after a negative/near-zero price.

    Only rows where ``price_eur_mwh <= near_negative_threshold`` contribute
    to training.  This focuses the classifier on the specific regime where
    Futures buy signals are most actionable.

    Parameters
    ----------
    model_dir:
        Directory for persisting model artefacts.
    horizon_hours:
        Look-ahead window in hours for the rebound check.
    min_rebound_eur:
        Minimum price recovery (EUR/MWh) required to label a row as a
        positive rebound event.
    """

    def __init__(
        self,
        model_dir: str,
        horizon_hours: int = 6,
        min_rebound_eur: float = 10.0,
    ) -> None:
        self.model_dir = model_dir
        self.horizon_hours = horizon_hours
        self.min_rebound_eur = min_rebound_eur

        self.model: Optional[XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.metrics: Dict = {}

        self._fe = FeatureEngineer()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> Dict:
        """Train the rebound classifier on near-negative price rows.

        Parameters
        ----------
        df:
            Raw HourlyPrice DataFrame sorted by timestamp.

        Returns
        -------
        dict
            Keys: ``accuracy``, ``auc_roc``, ``precision``, ``recall``,
            ``f1``, ``n_train``, ``n_positive``, ``class_balance``,
            ``n_folds``, ``training_time_s``.
        """
        t0 = time.perf_counter()
        logger.info(
            "Training ReboundClassifier (horizon=%dh, min_rebound=%.2f EUR/MWh)",
            self.horizon_hours,
            self.min_rebound_eur,
        )

        # Build full feature matrix first (requires the full df for lags)
        X_full, _ = self._fe.get_feature_matrix(df)

        # Align df to the valid rows retained by get_feature_matrix
        df_aligned = df.sort_values("timestamp").reset_index(drop=True)
        df_aligned = df_aligned.loc[X_full.index].reset_index(drop=True)
        X_full = X_full.reset_index(drop=True)

        # Filter to near-negative regime
        near_neg_mask = (
            df_aligned["price_eur_mwh"] <= _NEAR_NEGATIVE_THRESHOLD_EUR
        )
        X = X_full.loc[near_neg_mask].reset_index(drop=True)
        df_filtered = df_aligned.loc[near_neg_mask].reset_index(drop=True)

        if len(X) == 0:
            raise ValueError(
                "No near-negative rows found in training data. "
                f"Threshold = {_NEAR_NEGATIVE_THRESHOLD_EUR} EUR/MWh."
            )

        # Build rebound labels on the filtered subset
        y = self._fe.get_rebound_labels(
            df_filtered,
            horizon_hours=self.horizon_hours,
            min_rebound_eur=self.min_rebound_eur,
        ).reset_index(drop=True)

        self.feature_names = list(X.columns)
        n_pos = int(y.sum())
        n_total = len(y)
        n_neg = n_total - n_pos
        class_balance = n_pos / n_total if n_total > 0 else 0.0

        logger.info(
            "Rebound dataset: %d rows, %d positive (%.1f%%), %d negative",
            n_total, n_pos, class_balance * 100, n_neg,
        )

        if n_pos == 0:
            raise ValueError(
                "No positive rebound examples in training data. "
                "Adjust horizon_hours or min_rebound_eur."
            )

        spw = n_neg / n_pos if n_pos > 0 else 1.0

        xgb_params = dict(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1.0,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )

        # Walk-forward cross-validation – use at most 5 folds but fewer if
        # the dataset is small.
        n_splits = min(5, max(2, n_total // 50))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics: List[Dict] = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if y_val.sum() == 0 or y_tr.sum() == 0:
                logger.warning(
                    "Fold %d skipped: insufficient positive examples.", fold_idx
                )
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            model = XGBClassifier(**xgb_params)
            model.fit(
                X_tr_s, y_tr,
                eval_set=[(X_val_s, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_val_s)
            y_proba = model.predict_proba(X_val_s)[:, 1]

            fold_metrics.append({
                "accuracy": accuracy_score(y_val, y_pred),
                "auc_roc": roc_auc_score(y_val, y_proba),
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred, zero_division=0),
                "f1": f1_score(y_val, y_pred, zero_division=0),
            })
            logger.debug(
                "Fold %d: AUC=%.4f F1=%.4f", fold_idx,
                fold_metrics[-1]["auc_roc"], fold_metrics[-1]["f1"],
            )

        if not fold_metrics:
            raise ValueError("All CV folds were skipped (no usable folds).")

        avg_metrics = {
            k: float(np.mean([m[k] for m in fold_metrics]))
            for k in fold_metrics[0]
        }

        # Final model on full near-negative dataset
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = XGBClassifier(**xgb_params)
        self.model.fit(X_scaled, y, verbose=False)

        elapsed = time.perf_counter() - t0
        self.metrics = {
            **avg_metrics,
            "n_train": n_total,
            "n_positive": n_pos,
            "class_balance": round(class_balance, 4),
            "n_folds": len(fold_metrics),
            "training_time_s": round(elapsed, 2),
        }

        logger.info(
            "ReboundClassifier trained: AUC=%.4f F1=%.4f (%.1fs)",
            self.metrics["auc_roc"], self.metrics["f1"], elapsed,
        )
        return self.metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, features: pd.DataFrame) -> float:
        """Return probability of a price rebound (scalar 0 – 1).

        Parameters
        ----------
        features:
            Single-row (or small batch) DataFrame produced by
            :meth:`FeatureEngineer.build_features`.

        Returns
        -------
        float
            Probability in [0, 1].  Values close to 1 indicate a high
            likelihood of recovery >= ``min_rebound_eur`` within
            ``horizon_hours``.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not loaded. Call load() or train() first.")

        X = self._align_features(features)
        X_s = self.scaler.transform(X)
        prob = float(self.model.predict_proba(X_s)[0, 1])
        return prob

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
            "min_rebound_eur": self.min_rebound_eur,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "saved_at": pd.Timestamp.utcnow().isoformat(),
        }
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        logger.info("ReboundClassifier saved to %s", model_path)
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
            saved_min_reb = meta.get("min_rebound_eur")
            if saved_min_reb is not None:
                self.min_rebound_eur = saved_min_reb

        logger.info("ReboundClassifier loaded from %s", model_path)

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> Dict[str, float]:
        """Return ``{feature_name: importance_score}`` sorted descending."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        scores = self.model.get_booster().get_fscore()
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
