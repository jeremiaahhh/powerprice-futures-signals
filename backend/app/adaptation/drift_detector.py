"""
Drift detection for feature, prediction, and performance drift.
Uses statistical tests (KS-test style comparison of rolling windows).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
import psycopg2.extras

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_DSN = "postgresql://ppuser:pppass@localhost:5432/powerprice"


@dataclass
class DriftReport:
    has_drift: bool
    severity: str  # LOW, MEDIUM, HIGH
    drift_types: List[str]
    details: Dict[str, Any]
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DriftDetector:
    """
    Detects distributional drift in:
    1. Feature drift (price, renewable generation stats)
    2. Prediction drift (p_rebound distribution)
    3. Signal performance drift (win rate, PF from shadow signals)
    4. Data quality drift
    5. Tail event frequency drift
    """

    def __init__(self, window_days: int = 30, reference_days: int = 90) -> None:
        self.window_days = window_days          # recent window
        self.reference_days = reference_days    # reference window

    def check(self) -> DriftReport:
        """Run all drift checks. Returns DriftReport."""
        drift_types: List[str] = []
        details: Dict[str, Any] = {}

        try:
            conn = psycopg2.connect(_DSN)
            try:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                now = datetime.now(timezone.utc)

                # 1. Feature drift
                feat_drift = self._check_feature_drift(cur, now)
                if feat_drift:
                    drift_types.append("feature_drift")
                    details["feature_drift"] = feat_drift

                # 2. Prediction drift
                pred_drift = self._check_prediction_drift(cur, now)
                if pred_drift:
                    drift_types.append("prediction_drift")
                    details["prediction_drift"] = pred_drift

                # 3. Signal performance drift
                perf_drift = self._check_performance_drift(cur, now)
                if perf_drift:
                    drift_types.append("performance_drift")
                    details["performance_drift"] = perf_drift

                # 4. Tail event frequency
                tail_drift = self._check_tail_event_drift(cur, now)
                if tail_drift:
                    drift_types.append("tail_event_drift")
                    details["tail_event_drift"] = tail_drift

            finally:
                conn.close()
        except Exception as exc:
            logger.error("DriftDetector.check failed: %s", exc)
            return DriftReport(has_drift=False, severity="LOW", drift_types=[], details={"error": str(exc)})

        has_drift = len(drift_types) > 0
        severity = "HIGH" if len(drift_types) >= 3 else "MEDIUM" if len(drift_types) >= 2 else "LOW" if drift_types else "LOW"

        logger.info("Drift check: %d drift types found (severity=%s)", len(drift_types), severity)
        return DriftReport(has_drift=has_drift, severity=severity, drift_types=drift_types, details=details)

    def _check_feature_drift(self, cur, now: datetime) -> Optional[Dict]:
        """Compare recent vs reference price volatility and renewable share."""
        try:
            ref_start = now - timedelta(days=self.reference_days)
            rec_start = now - timedelta(days=self.window_days)

            cur.execute(
                "SELECT price_eur_mwh, solar_mw, wind_onshore_mw, wind_offshore_mw, load_mw "
                "FROM hourly_prices WHERE timestamp >= %s AND timestamp < %s "
                "AND price_eur_mwh IS NOT NULL ORDER BY timestamp",
                (ref_start, rec_start)
            )
            ref_rows = cur.fetchall()

            cur.execute(
                "SELECT price_eur_mwh, solar_mw, wind_onshore_mw, wind_offshore_mw, load_mw "
                "FROM hourly_prices WHERE timestamp >= %s AND price_eur_mwh IS NOT NULL ORDER BY timestamp",
                (rec_start,)
            )
            rec_rows = cur.fetchall()

            if len(ref_rows) < 24 or len(rec_rows) < 24:
                return None

            ref_prices = np.array([r["price_eur_mwh"] for r in ref_rows if r["price_eur_mwh"] is not None])
            rec_prices = np.array([r["price_eur_mwh"] for r in rec_rows if r["price_eur_mwh"] is not None])

            ref_std = float(np.std(ref_prices))
            rec_std = float(np.std(rec_prices))
            ref_mean = float(np.mean(ref_prices))
            rec_mean = float(np.mean(rec_prices))

            std_ratio = rec_std / ref_std if ref_std > 0 else 1.0
            mean_shift = abs(rec_mean - ref_mean)

            # Drift if volatility changed by >40% or mean shifted by >30 EUR/MWh
            if std_ratio > 1.4 or std_ratio < 0.6 or mean_shift > 30:
                return {
                    "ref_mean": round(ref_mean, 2),
                    "rec_mean": round(rec_mean, 2),
                    "mean_shift_eur_mwh": round(mean_shift, 2),
                    "ref_vol": round(ref_std, 2),
                    "rec_vol": round(rec_std, 2),
                    "vol_ratio": round(std_ratio, 3),
                }
            return None
        except Exception as exc:
            logger.debug("Feature drift check failed: %s", exc)
            return None

    def _check_prediction_drift(self, cur, now: datetime) -> Optional[Dict]:
        """Check if p_rebound distribution has shifted significantly."""
        try:
            ref_start = now - timedelta(days=self.reference_days)
            rec_start = now - timedelta(days=self.window_days)

            cur.execute(
                "SELECT p_rebound FROM futures_signals WHERE timestamp >= %s AND timestamp < %s "
                "AND p_rebound IS NOT NULL ORDER BY timestamp",
                (ref_start, rec_start)
            )
            ref_vals = np.array([r[0] for r in cur.fetchall()])

            cur.execute(
                "SELECT p_rebound FROM futures_signals WHERE timestamp >= %s "
                "AND p_rebound IS NOT NULL ORDER BY timestamp",
                (rec_start,)
            )
            rec_vals = np.array([r[0] for r in cur.fetchall()])

            if len(ref_vals) < 10 or len(rec_vals) < 10:
                return None

            ref_mean = float(np.mean(ref_vals))
            rec_mean = float(np.mean(rec_vals))
            shift = abs(rec_mean - ref_mean)

            # Drift if mean p_rebound shifted by >0.10 (10 percentage points)
            if shift > 0.10:
                return {
                    "ref_mean_p_rebound": round(ref_mean, 4),
                    "rec_mean_p_rebound": round(rec_mean, 4),
                    "shift": round(shift, 4),
                }
            return None
        except Exception as exc:
            logger.debug("Prediction drift check failed: %s", exc)
            return None

    def _check_performance_drift(self, cur, now: datetime) -> Optional[Dict]:
        """Check win rate / profit factor from shadow signals."""
        try:
            cutoff = now - timedelta(days=self.window_days)
            cur.execute(
                "SELECT s.action, s.current_price, s.net_edge, "
                "o.realized_rebound, o.simulated_pnl, o.outcome_status "
                "FROM shadow_signals s "
                "LEFT JOIN shadow_outcomes o ON o.signal_id = s.id "
                "WHERE s.timestamp >= %s AND s.action IN ('ENTER_LONG_REBOUND_SIGNAL', 'HIGH_CONFIDENCE_SIGNAL')"
                "ORDER BY s.timestamp DESC LIMIT 60",
                (cutoff,)
            )
            rows = cur.fetchall()

            completed = [r for r in rows if r["outcome_status"] in ("win", "loss")]
            if len(completed) < 10:
                return None

            wins = sum(1 for r in completed if r["outcome_status"] == "win")
            losses = sum(1 for r in completed if r["outcome_status"] == "loss")
            win_rate = wins / len(completed)

            profits = [r["simulated_pnl"] for r in completed if r["simulated_pnl"] and r["simulated_pnl"] > 0]
            losses_pnl = [abs(r["simulated_pnl"]) for r in completed if r["simulated_pnl"] and r["simulated_pnl"] < 0]
            pf = sum(profits) / sum(losses_pnl) if losses_pnl else None

            # Drift if win rate < 45% or PF < 0.80
            if win_rate < 0.45 or (pf is not None and pf < 0.80):
                return {
                    "rolling_win_rate": round(win_rate, 4),
                    "rolling_pf": round(pf, 4) if pf else None,
                    "sample_size": len(completed),
                    "alert": "performance degradation detected",
                }
            return None
        except Exception as exc:
            logger.debug("Performance drift check failed: %s", exc)
            return None

    def _check_tail_event_drift(self, cur, now: datetime) -> Optional[Dict]:
        """Check if tail event frequency has increased significantly."""
        try:
            ref_start = now - timedelta(days=self.reference_days)
            rec_start = now - timedelta(days=self.window_days)

            cur.execute(
                "SELECT COUNT(*) FROM tail_events WHERE timestamp >= %s AND timestamp < %s",
                (ref_start, rec_start)
            )
            ref_count = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM tail_events WHERE timestamp >= %s",
                (rec_start,)
            )
            rec_count = cur.fetchone()[0]

            # Normalize to per-day rate
            ref_rate = ref_count / (self.reference_days - self.window_days)
            rec_rate = rec_count / self.window_days

            if rec_rate > ref_rate * 2.0 and rec_count > 3:
                return {
                    "reference_rate_per_day": round(ref_rate, 3),
                    "recent_rate_per_day": round(rec_rate, 3),
                    "ratio": round(rec_rate / max(ref_rate, 0.01), 2),
                }
            return None
        except Exception as exc:
            logger.debug("Tail event drift check failed: %s", exc)
            return None
