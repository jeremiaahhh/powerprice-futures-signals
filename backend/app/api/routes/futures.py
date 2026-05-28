"""
Futures cost-model and signal routes.

GET  /futures/cost-model    – return current CostModelConfig
POST /futures/cost-model    – update CostModelConfig
GET  /futures/signal        – generate and return the current Signal
GET  /futures/explain-signal – return Signal with detailed feature_explanation
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import get_db
from app.db.models import HourlyPrice, FuturesSignal
from app.api.schemas import (
    CostModelConfig,
    CostBreakdown,
    SignalAction,
    SignalResponse,
)

logger = get_logger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level cost model instance (shared across requests)
# ---------------------------------------------------------------------------

_CONFIG_PERSIST_PATH = os.path.join(
    os.environ.get("MODEL_DIR", settings.model_dir), "cost_model_config.json"
)

_cost_model_config: CostModelConfig = CostModelConfig(
    avg_spread_eur_mwh=settings.futures_avg_spread_eur_mwh,
    slippage_eur_mwh=settings.futures_slippage_eur_mwh,
    overnight_fee_annual_pct=settings.futures_overnight_fee_annual_pct,
    weekend_fee_multiplier=settings.futures_weekend_fee_multiplier,
    broker_markup_eur_mwh=settings.futures_broker_markup_eur_mwh,
    safety_buffer_eur_mwh=settings.futures_safety_buffer_eur_mwh,
    min_edge_threshold=settings.futures_min_edge_threshold,
    holding_hours=4,
)


def _load_persisted_config() -> None:
    """Load persisted cost model config from disk, if available."""
    global _cost_model_config
    try:
        if os.path.exists(_CONFIG_PERSIST_PATH):
            with open(_CONFIG_PERSIST_PATH, "r") as fh:
                data = json.load(fh)
            _cost_model_config = CostModelConfig(**data)
            logger.info("Loaded persisted cost model config from %s", _CONFIG_PERSIST_PATH)
    except Exception as exc:
        logger.warning("Failed to load persisted config: %s", exc)


def _persist_config(config: CostModelConfig) -> None:
    """Persist cost model config to disk."""
    try:
        os.makedirs(os.path.dirname(_CONFIG_PERSIST_PATH), exist_ok=True)
        with open(_CONFIG_PERSIST_PATH, "w") as fh:
            json.dump(config.model_dump(), fh, indent=2)
    except Exception as exc:
        logger.warning("Failed to persist cost model config: %s", exc)


# Attempt to load on module import
_load_persisted_config()


# ---------------------------------------------------------------------------
# Cost calculation helpers
# ---------------------------------------------------------------------------


def _compute_cost_breakdown(
    config: CostModelConfig,
    is_weekend: bool = False,
) -> CostBreakdown:
    """Calculate itemised Futures costs for a trade."""
    spread_cost = config.avg_spread_eur_mwh
    slippage_cost = config.slippage_eur_mwh

    # Overnight fee: annual_pct / 365 / 24 * holding_hours
    # For weekends the broker typically charges 3x the daily overnight fee
    daily_fee_pct = config.overnight_fee_annual_pct / 365.0
    hourly_fee_pct = daily_fee_pct / 24.0
    base_overnight = hourly_fee_pct * config.holding_hours * 100.0  # expressed per 100 EUR/MWh notional

    if is_weekend:
        overnight_fee = base_overnight * config.weekend_fee_multiplier
    else:
        overnight_fee = base_overnight

    # Cap overnight fee to a reasonable range
    overnight_fee = min(overnight_fee, 5.0)

    broker_markup = config.broker_markup_eur_mwh
    safety_buffer = config.safety_buffer_eur_mwh

    total = spread_cost + slippage_cost + overnight_fee + broker_markup + safety_buffer

    return CostBreakdown(
        spread_cost_eur_mwh=round(spread_cost, 4),
        slippage_cost_eur_mwh=round(slippage_cost, 4),
        overnight_fee_eur_mwh=round(overnight_fee, 4),
        broker_markup_eur_mwh=round(broker_markup, 4),
        safety_buffer_eur_mwh=round(safety_buffer, 4),
        total_eur_mwh=round(total, 4),
        holding_hours_assumed=config.holding_hours,
        is_weekend=is_weekend,
    )


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


async def _fetch_latest_price_row(db: AsyncSession) -> Optional[HourlyPrice]:
    stmt = (
        select(HourlyPrice)
        .where(HourlyPrice.price_eur_mwh.isnot(None))
        .order_by(HourlyPrice.timestamp.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def _fetch_recent_df(db: AsyncSession, hours: int = 72) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    stmt = (
        select(HourlyPrice)
        .where(HourlyPrice.timestamp >= cutoff)
        .order_by(HourlyPrice.timestamp.asc())
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    data = []
    for r in rows:
        data.append({
            "timestamp": r.timestamp,
            "price_eur_mwh": r.price_eur_mwh,
            "load_mw": r.load_mw,
            "wind_onshore_mw": r.wind_onshore_mw,
            "wind_offshore_mw": r.wind_offshore_mw,
            "solar_mw": r.solar_mw,
            "residual_load_mw": r.residual_load_mw,
            "net_export_mw": r.net_export_mw,
            "temperature_c": r.temperature_c,
            "wind_speed_ms": r.wind_speed_ms,
            "solar_radiation_wm2": r.solar_radiation_wm2,
            "cloud_cover_pct": r.cloud_cover_pct,
            "battery_charge_mw": r.battery_charge_mw if hasattr(r, "battery_charge_mw") else None,
            "battery_discharge_mw": r.battery_discharge_mw if hasattr(r, "battery_discharge_mw") else None,
            "battery_net_mw": r.battery_net_mw if hasattr(r, "battery_net_mw") else None,
            "is_holiday": int(r.is_holiday) if r.is_holiday is not None else 0,
            "is_weekend": int(r.is_weekend) if r.is_weekend is not None else 0,
            "hour": r.hour,
            "month": r.month,
        })

    return pd.DataFrame(data)


def _build_feature_row(df: pd.DataFrame):
    """Return (X_latest, feature_names) or (None, []) if insufficient data."""
    if df.empty:
        return None, []
    try:
        from app.features.engineering import FeatureEngineer
        fe = FeatureEngineer()
        features_df = fe.build_features(df)
        available_cols = [c for c in fe.FEATURE_COLUMNS if c in features_df.columns]
        valid_mask = features_df[available_cols].notna().all(axis=1)
        valid_features = features_df.loc[valid_mask, available_cols]
        if valid_features.empty:
            return None, available_cols
        return valid_features.tail(1), available_cols
    except Exception as exc:
        logger.error("Feature engineering failed: %s", exc)
        return None, []


def _predict_proba(model, X) -> Optional[float]:
    """Call predict_proba on model; handles both scalar and array return values."""
    if model is None or X is None or X.empty:
        return None
    try:
        result = model.predict_proba(X)
        # Our NegativePriceClassifier / ReboundClassifier return a scalar float
        if isinstance(result, (int, float)):
            return float(result)
        arr = np.array(result)
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 1:
            return float(arr[-1])
        return float(arr[-1, 1])
    except Exception as exc:
        logger.warning("predict_proba failed: %s", exc)
        return None


def _predict_value(model, X) -> Optional[float]:
    """Call predict on regression model; handles scalar and array return."""
    if model is None or X is None or X.empty:
        return None
    try:
        result = model.predict(X)
        if isinstance(result, (int, float)):
            return float(result)
        arr = np.array(result)
        return float(arr.flat[-1])
    except Exception as exc:
        logger.warning("predict failed: %s", exc)
        return None


def _get_feature_explanation(
    model, X: pd.DataFrame, feature_names: List[str], top_n: int = 10
) -> Optional[Dict[str, Any]]:
    """Build a feature explanation dict using model feature importances."""
    if model is None or X is None or X.empty:
        return None
    try:
        fi = getattr(model, "feature_importances_", None)
        if fi is None and hasattr(model, "model"):
            fi = getattr(model.model, "feature_importances_", None)
        if fi is None:
            return None

        fi_arr = np.array(fi)
        if len(fi_arr) != len(feature_names):
            return None

        # Get the actual feature values for the current row
        row_values = X.iloc[-1].to_dict()

        # Build signed contributions (importance * sign of feature value relative to mean)
        pairs = sorted(
            zip(feature_names, fi_arr.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]

        explanation: Dict[str, Any] = {}
        for feat_name, importance in pairs:
            raw_val = row_values.get(feat_name, 0.0)
            if raw_val is None:
                raw_val = 0.0
            explanation[feat_name] = {
                "importance": round(float(importance), 4),
                "value": round(float(raw_val), 4),
            }

        return explanation
    except Exception as exc:
        logger.warning("Feature explanation failed: %s", exc)
        return None


def _check_gaps_and_volatility(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Guard against data gaps and extreme price volatility before entry.

    Returns (is_ok, reason).
    - Blocks if any of the last 6 hourly rows has a gap > 2 h to the previous row
      (missing data = unknown price action = unreliable signal).
    - Blocks if the 24 h rolling price std > 100 EUR/MWh (implied spread would be
      extreme; our cost model significantly underestimates real costs in this regime).
    """
    if df.empty or "timestamp" not in df.columns or "price_eur_mwh" not in df.columns:
        return True, "ok"

    recent = df.sort_values("timestamp").tail(24)
    if len(recent) < 2:
        return True, "ok"

    # Convert timestamps to UTC-aware, then compute gaps in hours
    ts_col = pd.to_datetime(recent["timestamp"], utc=True)
    gaps_h = ts_col.diff().dt.total_seconds().dropna() / 3600.0
    max_gap = float(gaps_h.tail(6).max()) if len(gaps_h) >= 1 else 0.0
    if max_gap > 2.0:
        return False, f"Data gap of {max_gap:.1f}h detected in last 6 rows — entry suppressed"

    # 24h price volatility check
    prices_24h = recent["price_eur_mwh"].dropna()
    if len(prices_24h) >= 6:
        vol = float(prices_24h.std())
        if vol > 100.0:
            return False, (
                f"24h price volatility {vol:.1f} EUR/MWh exceeds 100 EUR/MWh — "
                "real spread would be extreme; entry suppressed"
            )

    return True, "ok"


def _check_data_quality(latest_row: Optional[HourlyPrice]) -> tuple[bool, str]:
    """Return (is_ok, reason) for data quality."""
    if latest_row is None:
        return False, "No market data in database"

    ts = latest_row.timestamp
    if ts is not None and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    age_minutes = (now - ts).total_seconds() / 60.0 if ts else 9999.0

    if age_minutes > settings.max_data_age_minutes:
        return False, f"Market data is stale ({age_minutes:.0f} minutes old)"

    if latest_row.price_eur_mwh is None:
        return False, "Latest row has no price_eur_mwh"

    return True, "ok"


def _check_risk(
    current_price: float,
    p_rebound: Optional[float],
    net_edge: Optional[float],
    open_positions: int,
) -> tuple[bool, List[str]]:
    """Return (is_blocked, warnings)."""
    warnings: List[str] = []
    blocked = False

    if open_positions >= settings.max_open_positions:
        warnings.append(f"Max open positions reached ({open_positions}/{settings.max_open_positions})")
        blocked = True

    if current_price < -100:
        warnings.append("Extreme negative price — heightened tail risk")

    if p_rebound is not None and p_rebound < settings.min_confidence_threshold:
        warnings.append(
            f"Rebound confidence {p_rebound:.2f} below threshold {settings.min_confidence_threshold}"
        )

    return blocked, warnings


async def _count_open_positions(db: AsyncSession) -> int:
    """Count open paper positions."""
    from app.db.models import PaperPosition
    stmt = select(PaperPosition).where(PaperPosition.status == "open")
    result = await db.execute(stmt)
    return len(result.scalars().all())


async def _generate_signal(
    db: AsyncSession,
    include_explanation: bool = False,
) -> SignalResponse:
    """Core signal generation logic — three-tier: NO_TRADE / WATCH / ENTER / HIGH_CONFIDENCE."""
    global _cost_model_config
    now = datetime.now(timezone.utc)
    config = _cost_model_config

    # --- 1. Data quality check ---
    latest_row = await _fetch_latest_price_row(db)
    data_ok, dq_reason = _check_data_quality(latest_row)

    if not data_ok:
        return SignalResponse(
            action=SignalAction.DATA_QUALITY_BLOCKED,
            confidence=0.0,
            timestamp=now,
            generated_at=now,
            reason=dq_reason,
            risk_warnings=[dq_reason],
        )

    current_price = float(latest_row.price_eur_mwh)
    signal_ts = latest_row.timestamp
    if signal_ts.tzinfo is None:
        signal_ts = signal_ts.replace(tzinfo=timezone.utc)

    # --- 2. Early exit: positive price → no trade ---
    if current_price >= 0:
        return SignalResponse(
            action=SignalAction.NO_TRADE,
            confidence=1.0,
            timestamp=signal_ts,
            generated_at=now,
            current_price=current_price,
            reason=f"Current price {current_price:.2f} EUR/MWh is non-negative. No Futures rebound trade warranted.",
        )

    # --- 3. Load features and run ML models ---
    df = await _fetch_recent_df(db, hours=72)

    # Gap + extreme-volatility guard
    gap_ok, gap_reason = _check_gaps_and_volatility(df)
    if not gap_ok:
        logger.warning("Gap/volatility guard blocked signal: %s", gap_reason)
        return SignalResponse(
            action=SignalAction.DATA_QUALITY_BLOCKED,
            confidence=0.0,
            timestamp=signal_ts,
            generated_at=now,
            current_price=current_price,
            reason=gap_reason,
            risk_warnings=[gap_reason],
        )

    X_latest, feature_names = _build_feature_row(df)

    # Try loading models
    neg_clf = None
    reb_clf = None
    price_model = None

    try:
        from app.ml.negative_price_classifier import NegativePriceClassifier
        neg_clf = NegativePriceClassifier(model_dir=settings.model_dir)
        neg_clf.load()
    except Exception:
        pass

    try:
        from app.ml.rebound_classifier import ReboundClassifier
        reb_clf = ReboundClassifier(model_dir=settings.model_dir)
        reb_clf.load()
    except Exception:
        pass

    try:
        from app.ml.price_regression import PriceRegressionModel
        price_model = PriceRegressionModel(model_dir=settings.model_dir)
        price_model.load()
    except Exception:
        pass

    p_negative = _predict_proba(neg_clf, X_latest)
    p_rebound = _predict_proba(reb_clf, X_latest)
    predicted_price = _predict_value(price_model, X_latest)

    # Heuristic fallbacks when models are not available
    if p_negative is None:
        p_negative = 1.0 if current_price < 0 else 0.0
    if p_rebound is None:
        p_rebound = min(0.9, max(0.0, abs(current_price) / 100.0))
    if predicted_price is None:
        predicted_price = max(0.0, current_price + 30.0)

    # --- 4. Regime classification ---
    regime_result = None
    regime_net_edge_enter = config.min_edge_threshold
    regime_net_edge_hc = settings.futures_high_confidence_threshold

    try:
        from app.regime import RegimeClassifier
        if X_latest is not None and not X_latest.empty:
            clf = RegimeClassifier()
            regime_result = clf.classify(X_latest.iloc[0])
            thr = regime_result.signal_thresholds
            regime_net_edge_enter = thr.get("net_edge_enter", config.min_edge_threshold)
            regime_net_edge_hc = thr.get("net_edge_hc", settings.futures_high_confidence_threshold)
    except Exception as exc:
        logger.warning("Regime classification failed (using defaults): %s", exc)

    # STRESS regime blocks all entries with EXTREME_VOLATILITY_BLOCKED
    if regime_result is not None and regime_net_edge_enter >= 900.0:
        reason = f"STRESS regime detected — {regime_result.description}. All entries blocked."
        return SignalResponse(
            action=SignalAction.EXTREME_VOLATILITY_BLOCKED,
            confidence=0.0,
            timestamp=signal_ts,
            generated_at=now,
            current_price=current_price,
            p_negative=round(p_negative, 4),
            p_rebound=round(p_rebound, 4),
            reason=reason,
            risk_warnings=[reason],
        )

    # --- Tail Risk Gate ---
    tail_risk_score = 0.0
    gap_risk_score = 0.0
    negative_price_streak = 0
    if current_price < 0:
        try:
            from app.risk.tail_risk_engine import TailRiskEngine
            _tail_engine = TailRiskEngine(
                max_negative_price=settings.max_negative_price_eur_mwh,
                max_streak_hours=settings.max_negative_streak_hours,
                max_gap_size=settings.max_gap_size_eur_mwh,
                max_tail_risk_score=settings.max_tail_risk_score,
            )
            _tail = _tail_engine.assess(df, current_price)
            tail_risk_score = _tail.tail_risk_score
            gap_risk_score = _tail.gap_risk_score
            negative_price_streak = _tail.negative_price_streak
            if _tail.is_blocked:
                _blocked_action = (
                    SignalAction.GAP_RISK_BLOCKED
                    if _tail.block_reason == "GAP_RISK_BLOCKED"
                    else SignalAction.TAIL_RISK_BLOCKED
                )
                return SignalResponse(
                    action=_blocked_action,
                    confidence=round(tail_risk_score, 4),
                    timestamp=signal_ts,
                    generated_at=datetime.now(timezone.utc),
                    current_price=round(current_price, 2),
                    predicted_price=None,
                    p_negative=round(p_negative, 4) if p_negative is not None else None,
                    p_rebound=round(p_rebound, 4) if p_rebound is not None else None,
                    expected_rebound_eur_mwh=None,
                    gross_edge=None,
                    estimated_futures_costs=None,
                    net_edge=None,
                    cost_breakdown=None,
                    stop_loss=None,
                    take_profit=None,
                    max_holding_hours=None,
                    reason=_tail.block_detail,
                    risk_warnings=[
                        f"tail_risk_score={tail_risk_score:.2f}",
                        f"gap_risk_score={gap_risk_score:.2f}",
                        f"negative_streak={negative_price_streak}h",
                    ],
                    feature_explanation=None,
                )
        except ImportError:
            logger.debug("TailRiskEngine not available — tail risk gate skipped")
        except Exception as _tail_exc:
            logger.warning("Tail risk gate failed (non-blocking): %s", _tail_exc)

    # --- 5. Edge calculation ---
    is_weekend = signal_ts.weekday() >= 5
    cost_breakdown = _compute_cost_breakdown(config, is_weekend=is_weekend)
    estimated_futures_costs = cost_breakdown.total_eur_mwh

    expected_rebound = max(0.0, predicted_price - current_price)
    gross_edge = expected_rebound
    net_edge = gross_edge - estimated_futures_costs

    stop_loss = current_price - 20.0
    take_profit = current_price + (expected_rebound * 0.8)
    max_holding_hours = config.holding_hours

    # --- 6. Risk check ---
    open_positions = await _count_open_positions(db)
    risk_blocked, risk_warnings = _check_risk(
        current_price, p_rebound, net_edge, open_positions
    )

    if negative_price_streak > 1:
        risk_warnings.append(f"negative_streak={negative_price_streak}h")
    if tail_risk_score > 0.30:
        risk_warnings.append(f"tail_risk_score={tail_risk_score:.2f}")

    if risk_blocked:
        return SignalResponse(
            action=SignalAction.RISK_BLOCKED,
            confidence=p_rebound,
            timestamp=signal_ts,
            generated_at=now,
            current_price=current_price,
            predicted_price=predicted_price,
            p_negative=p_negative,
            p_rebound=p_rebound,
            expected_rebound_eur_mwh=round(expected_rebound, 2),
            gross_edge=round(gross_edge, 2),
            estimated_futures_costs=round(estimated_futures_costs, 2),
            net_edge=round(net_edge, 2),
            cost_breakdown=cost_breakdown,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            max_holding_hours=max_holding_hours,
            reason="Risk engine blocked the signal: " + "; ".join(risk_warnings),
            risk_warnings=risk_warnings,
        )

    # --- 7. Three-tier signal logic ---
    # p_negative is NOT used for entry decisions — kept for monitoring/visualization only.
    # Entry gate: price < 0 (already guaranteed) + p_rebound + net_edge.
    # HIGH_CONFIDENCE additionally requires directional confirmation from engineered features.
    confidence = p_rebound

    # Extract directional confirmation features (available after FeatureEngineer)
    residual_load_rising = False
    solar_falling = False
    evening_demand_spike = False
    if X_latest is not None and not X_latest.empty:
        row = X_latest.iloc[0]
        rl_ramp = row.get("residual_load_ramp_1h", np.nan)
        sol_ramp = row.get("solar_ramp_1h", np.nan)
        ev_spike = row.get("evening_demand_spike", np.nan)
        if not (rl_ramp is None or (isinstance(rl_ramp, float) and np.isnan(rl_ramp))):
            residual_load_rising = float(rl_ramp) > 0
        if not (sol_ramp is None or (isinstance(sol_ramp, float) and np.isnan(sol_ramp))):
            solar_falling = float(sol_ramp) < 0
        if not (ev_spike is None or (isinstance(ev_spike, float) and np.isnan(ev_spike))):
            evening_demand_spike = bool(float(ev_spike) > 0.5)

    directional_confirmed = residual_load_rising and (solar_falling or evening_demand_spike)

    # Battery guard for HIGH_CONFIDENCE: batteries near-full suppress rebound
    battery_saturation_ok     = True
    expected_absorption_ok    = True
    battery_guard_note        = ""
    try:
        from app.services.battery_service import BatteryService
        _batt_svc = BatteryService()
        batt_df = await _batt_svc.get_battery_features(df)
        if not batt_df.empty:
            brow = batt_df.iloc[-1]
            batt_sat = float(brow.get("battery_saturation_proxy", 0.5) or 0.5)
            exp_abs  = float(brow.get("expected_battery_absorption", 0.0) or 0.0)
            if batt_sat >= 0.85:
                battery_saturation_ok = False
                battery_guard_note = f"battery_saturation={batt_sat:.0%} (near full — rebound may be dampened)"
            elif exp_abs >= 8_000.0:
                expected_absorption_ok = False
                battery_guard_note = f"expected_absorption={exp_abs:.0f} MW (high — supply may persist)"
    except Exception as _batt_exc:
        logger.debug("Battery feature check skipped: %s", _batt_exc)

    battery_confirmed = battery_saturation_ok and expected_absorption_ok

    regime_label = regime_result.regime.value if regime_result is not None else "UNKNOWN"

    if (
        p_rebound >= settings.futures_p_rebound_entry
        and net_edge >= regime_net_edge_hc
        and directional_confirmed
        and battery_confirmed
    ):
        action = SignalAction.HIGH_CONFIDENCE_SIGNAL
        reason = (
            f"HIGH CONFIDENCE: price {current_price:.2f} EUR/MWh, "
            f"p_rebound={p_rebound:.0%}, net_edge={net_edge:.1f} EUR/MWh (>={regime_net_edge_hc:.0f}). "
            f"Directional confirmation: residual_load_rising={residual_load_rising}, "
            f"solar_falling={solar_falling}, evening_demand_spike={evening_demand_spike}. "
            f"Battery: saturation_ok={battery_saturation_ok}, absorption_ok={expected_absorption_ok}. "
            f"Regime: {regime_label}."
        )
    elif (
        p_rebound >= settings.futures_p_rebound_entry
        and net_edge >= regime_net_edge_enter
    ):
        action = SignalAction.ENTER_LONG_REBOUND_SIGNAL
        batt_note = f" Battery guard: {battery_guard_note}." if battery_guard_note else ""
        reason = (
            f"Price {current_price:.2f} EUR/MWh, p_rebound={p_rebound:.0%}, "
            f"net_edge={net_edge:.1f} EUR/MWh (>={regime_net_edge_enter:.0f}). "
            f"Regime: {regime_label}.{batt_note}"
        )
    elif net_edge >= settings.futures_watch_threshold:
        action = SignalAction.WATCH_LONG_REBOUND
        reason = (
            f"Price {current_price:.2f} EUR/MWh — WATCH zone: net_edge={net_edge:.1f} "
            f"EUR/MWh in [{settings.futures_watch_threshold:.0f}, {regime_net_edge_enter:.0f}) range "
            f"or p_rebound={p_rebound:.0%} below threshold. Regime: {regime_label}."
        )
    else:
        action = SignalAction.NO_TRADE
        reason = (
            f"Insufficient edge: price={current_price:.2f}, p_rebound={p_rebound:.2f}, "
            f"net_edge={net_edge:.2f} EUR/MWh (below watch threshold {settings.futures_watch_threshold:.0f}). "
            f"Regime: {regime_label}."
        )

    # --- 8. Feature explanation (optional) ---
    feature_explanation: Optional[Dict[str, Any]] = None
    if include_explanation:
        for model in [reb_clf, price_model, neg_clf]:
            if model is not None and X_latest is not None:
                feature_explanation = _get_feature_explanation(model, X_latest, feature_names)
                if feature_explanation:
                    break

    # --- 9. Persist signal to DB ---
    try:
        signal_record = FuturesSignal(
            timestamp=signal_ts,
            action=action.value,
            confidence=confidence,
            current_price=current_price,
            predicted_price=predicted_price,
            p_negative=p_negative,
            p_rebound=p_rebound,
            expected_rebound_eur_mwh=round(expected_rebound, 2),
            gross_edge=round(gross_edge, 2),
            estimated_futures_costs=round(estimated_futures_costs, 2),
            net_edge=round(net_edge, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            max_holding_hours=max_holding_hours,
            reason=reason,
            risk_warnings=risk_warnings,
            feature_explanation=feature_explanation,
        )
        db.add(signal_record)
        await db.flush()
    except Exception as exc:
        logger.warning("Failed to persist signal: %s", exc)

    return SignalResponse(
        action=action,
        confidence=confidence,
        timestamp=signal_ts,
        generated_at=now,
        current_price=current_price,
        predicted_price=round(predicted_price, 2),
        p_negative=round(p_negative, 4),
        p_rebound=round(p_rebound, 4),
        expected_rebound_eur_mwh=round(expected_rebound, 2),
        gross_edge=round(gross_edge, 2),
        estimated_futures_costs=round(estimated_futures_costs, 2),
        net_edge=round(net_edge, 2),
        cost_breakdown=cost_breakdown,
        stop_loss=round(stop_loss, 2),
        take_profit=round(take_profit, 2),
        max_holding_hours=max_holding_hours,
        reason=reason,
        risk_warnings=risk_warnings,
        feature_explanation=feature_explanation,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/cost-model", response_model=CostModelConfig, summary="Get current Futures cost model config")
async def get_cost_model() -> CostModelConfig:
    """Return the currently active Futures cost model configuration."""
    return _cost_model_config


@router.post("/cost-model", response_model=CostModelConfig, summary="Update Futures cost model config")
async def update_cost_model(config: CostModelConfig) -> CostModelConfig:
    """
    Update the Futures cost model configuration.

    Validates all parameters and persists the updated config to disk so that
    it survives service restarts.
    """
    global _cost_model_config

    # Additional business-logic validations
    if config.avg_spread_eur_mwh > settings.futures_max_spread_eur_mwh:
        raise HTTPException(
            status_code=422,
            detail=f"avg_spread_eur_mwh {config.avg_spread_eur_mwh} exceeds maximum "
                   f"allowed value of {settings.futures_max_spread_eur_mwh}",
        )
    if config.avg_spread_eur_mwh < settings.futures_min_spread_eur_mwh:
        raise HTTPException(
            status_code=422,
            detail=f"avg_spread_eur_mwh {config.avg_spread_eur_mwh} is below minimum "
                   f"allowed value of {settings.futures_min_spread_eur_mwh}",
        )

    _cost_model_config = config
    _persist_config(config)
    logger.info("Cost model config updated: %s", config.model_dump())
    return _cost_model_config


@router.get("/signal", response_model=SignalResponse, summary="Generate current Futures signal")
async def get_signal(db: AsyncSession = Depends(get_db)) -> SignalResponse:
    """
    Generate and return the current Futures trading signal.

    Fetches the latest market data, runs all ML models, evaluates the signal
    engine, and returns a structured signal with edge calculation and risk levels.

    SIGNAL ONLY - no live order execution.
    """
    return await _generate_signal(db, include_explanation=False)


@router.get("/signal/history", response_model=List[SignalResponse], summary="Recent signal history")
async def get_signal_history(
    limit: int = Query(default=20, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> List[SignalResponse]:
    """Return the last N persisted Futures signals from the database."""
    stmt = (
        select(FuturesSignal)
        .order_by(FuturesSignal.timestamp.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    out = []
    for r in rows:
        out.append(SignalResponse(
            action=r.action,
            confidence=r.confidence or 0.0,
            timestamp=r.timestamp,
            generated_at=r.created_at or r.timestamp,
            current_price=r.current_price,
            predicted_price=r.predicted_price,
            p_negative=r.p_negative,
            p_rebound=r.p_rebound,
            expected_rebound_eur_mwh=r.expected_rebound_eur_mwh,
            gross_edge=r.gross_edge,
            estimated_futures_costs=r.estimated_futures_costs,
            net_edge=r.net_edge,
            stop_loss=r.stop_loss,
            take_profit=r.take_profit,
            max_holding_hours=r.max_holding_hours,
            reason=r.reason or "",
            risk_warnings=r.risk_warnings or [],
        ))
    return out


@router.get("/explain-signal", response_model=SignalResponse, summary="Signal with feature explanation")
async def get_signal_with_explanation(db: AsyncSession = Depends(get_db)) -> SignalResponse:
    """
    Generate the current Futures signal with a detailed feature explanation.

    Returns the same signal as GET /futures/signal but enriches the response with
    the top model features and their contributions, enabling interpretability.
    """
    return await _generate_signal(db, include_explanation=True)
