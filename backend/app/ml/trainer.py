"""
Model training orchestrator for the PowerPrice Futures Signals platform.

Loads historical data from PostgreSQL and co-ordinates training of all
three ML models (NegativePriceClassifier, ReboundClassifier,
PriceRegressionModel).  Designed to be called from a Celery periodic task
or a management CLI command.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sqlalchemy import create_engine, text

from app.core.config import settings
from app.core.logging import get_logger
from app.ml.negative_price_classifier import NegativePriceClassifier
from app.ml.rebound_classifier import ReboundClassifier
from app.ml.price_regression import PriceRegressionModel

logger = get_logger(__name__)

_METRICS_FILE = "training_metrics.json"
_LAST_TRAINED_FILE = "last_trained.json"

# SQL query to load HourlyPrice rows for a given date range
_LOAD_QUERY = text("""
    SELECT
        id,
        timestamp,
        source,
        price_eur_mwh,
        intraday_price_eur_mwh,
        load_mw,
        wind_onshore_mw,
        wind_offshore_mw,
        solar_mw,
        residual_load_mw,
        net_export_mw,
        temperature_c,
        wind_speed_ms,
        solar_radiation_wm2,
        cloud_cover_pct,
        is_holiday,
        is_weekend,
        hour,
        month
    FROM hourly_prices
    WHERE timestamp >= :start_ts
      AND price_eur_mwh IS NOT NULL
    ORDER BY timestamp ASC
""")


class ModelTrainer:
    """Orchestrates training of all ML models.

    Parameters
    ----------
    model_dir:
        Directory where trained model artefacts are stored.
    db_url:
        Synchronous SQLAlchemy database URL (e.g. ``postgresql+psycopg2://...``).
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        db_url: Optional[str] = None,
    ) -> None:
        self.model_dir = model_dir or settings.model_dir
        self.db_url = db_url or settings.database_url_sync
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_training_data(
        self,
        days_back: int = 365,
        before_ts: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load HourlyPrice rows from PostgreSQL.

        Parameters
        ----------
        days_back:
            Number of calendar days of history to load (default 365).
        before_ts:
            If provided, load data only before this timestamp (exclusive).
            Used for out-of-sample backtesting to prevent look-ahead bias.
        """
        cutoff = before_ts or datetime.now(tz=timezone.utc)
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        start_ts = cutoff - timedelta(days=days_back)
        logger.info(
            "Loading training data %s → %s (days_back=%d)",
            start_ts.isoformat(),
            cutoff.isoformat(),
            days_back,
        )

        import psycopg2
        dsn = (
            self.db_url
            .replace("postgresql+psycopg2://", "postgresql://")
            .replace("postgresql+asyncpg://", "postgresql://")
        )
        conn = psycopg2.connect(dsn)
        try:
            base_query = _LOAD_QUERY.text.replace(":start_ts", "%(start_ts)s")
            if before_ts is not None:
                base_query = base_query.replace(
                    "ORDER BY timestamp ASC",
                    "AND timestamp < %(before_ts)s\n    ORDER BY timestamp ASC",
                )
                params: dict = {"start_ts": start_ts, "before_ts": cutoff}
            else:
                params = {"start_ts": start_ts}
            with conn.cursor() as cur:
                cur.execute(base_query, params)
                cols = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=cols)
        finally:
            conn.close()

        if df.empty:
            logger.warning(
                "No training data found in the last %d days.", days_back
            )
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            "Loaded %d rows spanning %s – %s",
            len(df),
            df["timestamp"].min(),
            df["timestamp"].max(),
        )
        return df

    # ------------------------------------------------------------------
    # Training orchestration
    # ------------------------------------------------------------------

    def train_all(self) -> Dict:
        """Train all three models and persist artefacts.

        Loads data, engineers features, trains NegativePriceClassifier,
        ReboundClassifier, and PriceRegressionModel in sequence, saves
        each model, and writes a combined metrics file.

        Returns
        -------
        dict
            Combined metrics for all three models, keyed by model name.
        """
        t0 = time.perf_counter()
        logger.info("Starting full model training run.")

        df = self.load_training_data(days_back=365)
        if df.empty:
            raise ValueError("Cannot train models: no training data available.")

        combined_metrics: Dict = {}

        # ---- Negative price classifier ------------------------------
        neg_clf = NegativePriceClassifier(
            model_dir=self.model_dir, horizon_hours=4
        )
        try:
            neg_metrics = neg_clf.train(df)
            neg_clf.save()
            combined_metrics["negative_price_classifier"] = neg_metrics
            logger.info(
                "NegativePriceClassifier: AUC=%.4f F1=%.4f",
                neg_metrics.get("auc_roc", 0),
                neg_metrics.get("f1", 0),
            )
        except Exception as exc:
            logger.error("NegativePriceClassifier training failed: %s", exc)
            combined_metrics["negative_price_classifier"] = {"error": str(exc)}

        # ---- Rebound classifier ------------------------------------
        reb_clf = ReboundClassifier(
            model_dir=self.model_dir, horizon_hours=6, min_rebound_eur=10.0
        )
        try:
            reb_metrics = reb_clf.train(df)
            reb_clf.save()
            combined_metrics["rebound_classifier"] = reb_metrics
            logger.info(
                "ReboundClassifier: AUC=%.4f F1=%.4f",
                reb_metrics.get("auc_roc", 0),
                reb_metrics.get("f1", 0),
            )
        except Exception as exc:
            logger.error("ReboundClassifier training failed: %s", exc)
            combined_metrics["rebound_classifier"] = {"error": str(exc)}

        # ---- Price regression model --------------------------------
        reg_model = PriceRegressionModel(
            model_dir=self.model_dir, horizon_hours=4
        )
        try:
            reg_metrics = reg_model.train(df)
            reg_model.save()
            combined_metrics["price_regression"] = reg_metrics
            logger.info(
                "PriceRegressionModel: MAE=%.2f RMSE=%.2f R2=%.4f",
                reg_metrics.get("mae", 0),
                reg_metrics.get("rmse", 0),
                reg_metrics.get("r2", 0),
            )
        except Exception as exc:
            logger.error("PriceRegressionModel training failed: %s", exc)
            combined_metrics["price_regression"] = {"error": str(exc)}

        # ---- Persist combined metrics + timestamp ------------------
        elapsed = time.perf_counter() - t0
        combined_metrics["total_training_time_s"] = round(elapsed, 2)
        combined_metrics["trained_at"] = datetime.now(tz=timezone.utc).isoformat()

        metrics_path = os.path.join(self.model_dir, _METRICS_FILE)
        with open(metrics_path, "w") as fh:
            json.dump(combined_metrics, fh, indent=2)

        last_trained_path = os.path.join(self.model_dir, _LAST_TRAINED_FILE)
        with open(last_trained_path, "w") as fh:
            json.dump(
                {"trained_at": combined_metrics["trained_at"]}, fh, indent=2
            )

        logger.info(
            "Full training run complete in %.1fs. Metrics saved to %s.",
            elapsed,
            metrics_path,
        )
        return combined_metrics

    # ------------------------------------------------------------------
    # Conditional retraining
    # ------------------------------------------------------------------

    def retrain_if_needed(self) -> bool:
        """Retrain all models if artefacts are stale or absent.

        The staleness threshold is ``settings.retrain_interval_hours``.

        Returns
        -------
        bool
            ``True`` if models were retrained, ``False`` if still fresh.
        """
        last_trained_path = os.path.join(self.model_dir, _LAST_TRAINED_FILE)

        if not os.path.exists(last_trained_path):
            logger.info("No last-trained record found. Triggering training.")
            self.train_all()
            return True

        with open(last_trained_path) as fh:
            meta = json.load(fh)

        trained_at_str = meta.get("trained_at")
        if not trained_at_str:
            logger.warning(
                "last_trained.json has no 'trained_at' key. Retraining."
            )
            self.train_all()
            return True

        trained_at = datetime.fromisoformat(trained_at_str)
        # Ensure timezone-awareness for comparison
        if trained_at.tzinfo is None:
            trained_at = trained_at.replace(tzinfo=timezone.utc)

        age_hours = (
            datetime.now(tz=timezone.utc) - trained_at
        ).total_seconds() / 3600

        if age_hours >= settings.retrain_interval_hours:
            logger.info(
                "Models are %.1fh old (threshold %dh). Retraining.",
                age_hours,
                settings.retrain_interval_hours,
            )
            self.train_all()
            return True

        logger.info(
            "Models are %.1fh old (threshold %dh). No retraining needed.",
            age_hours,
            settings.retrain_interval_hours,
        )
        return False
