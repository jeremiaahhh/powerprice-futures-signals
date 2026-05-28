from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, Text, JSON, ForeignKey,
    Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from app.db.base import Base


class HourlyPrice(Base):
    __tablename__ = "hourly_prices"
    __table_args__ = (
        UniqueConstraint("timestamp", name="uq_hourly_prices_timestamp"),
        Index("ix_hourly_prices_timestamp_source", "timestamp", "source"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    source = Column(String(50), nullable=False, default="smard")
    price_eur_mwh = Column(Float, nullable=True)
    intraday_price_eur_mwh = Column(Float, nullable=True)
    load_mw = Column(Float, nullable=True)
    wind_onshore_mw = Column(Float, nullable=True)
    wind_offshore_mw = Column(Float, nullable=True)
    solar_mw = Column(Float, nullable=True)
    residual_load_mw = Column(Float, nullable=True)
    net_export_mw = Column(Float, nullable=True)
    battery_charge_mw = Column(Float, nullable=True)
    battery_discharge_mw = Column(Float, nullable=True)
    battery_net_mw = Column(Float, nullable=True)
    temperature_c = Column(Float, nullable=True)
    wind_speed_ms = Column(Float, nullable=True)
    solar_radiation_wm2 = Column(Float, nullable=True)
    cloud_cover_pct = Column(Float, nullable=True)
    is_holiday = Column(Boolean, nullable=False, default=False)
    is_weekend = Column(Boolean, nullable=False, default=False)
    hour = Column(Integer, nullable=True)
    month = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class MLPrediction(Base):
    __tablename__ = "ml_predictions"
    __table_args__ = (Index("ix_ml_predictions_timestamp", "timestamp"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=True)
    p_negative = Column(Float, nullable=True)
    p_rebound = Column(Float, nullable=True)
    predicted_price_eur_mwh = Column(Float, nullable=True)
    horizon_hours = Column(Integer, nullable=True)
    features_used = Column(JSON, nullable=True)
    feature_importance = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class FuturesSignal(Base):
    __tablename__ = "futures_signals"
    __table_args__ = (Index("ix_futures_signals_timestamp", "timestamp"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    action = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=True)
    current_price = Column(Float, nullable=True)
    predicted_price = Column(Float, nullable=True)
    p_negative = Column(Float, nullable=True)
    p_rebound = Column(Float, nullable=True)
    expected_rebound_eur_mwh = Column(Float, nullable=True)
    gross_edge = Column(Float, nullable=True)
    estimated_futures_costs = Column(Float, nullable=True)
    net_edge = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    max_holding_hours = Column(Integer, nullable=True)
    reason = Column(Text, nullable=True)
    risk_warnings = Column(JSON, nullable=True)
    feature_explanation = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class PaperPosition(Base):
    __tablename__ = "paper_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(Integer, ForeignKey("futures_signals.id"), nullable=True)
    status = Column(String(20), nullable=False, default="open")  # open, closed
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    entry_timestamp = Column(DateTime(timezone=True), nullable=False)
    exit_timestamp = Column(DateTime(timezone=True), nullable=True)
    notional_size_mwh = Column(Float, nullable=False, default=1.0)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    max_holding_hours = Column(Integer, nullable=True)
    pnl_eur = Column(Float, nullable=True)
    futures_costs_eur = Column(Float, nullable=True)
    net_pnl_eur = Column(Float, nullable=True)
    exit_reason = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), nullable=False)
    strategy = Column(String(50), nullable=False)  # naive, ml_rebound
    start_date = Column(DateTime(timezone=True), nullable=True)
    end_date = Column(DateTime(timezone=True), nullable=True)
    total_return_pct = Column(Float, nullable=True)
    annualized_return_pct = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    max_drawdown_pct = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    win_rate_pct = Column(Float, nullable=True)
    avg_trade_eur_mwh = Column(Float, nullable=True)
    worst_trade_eur_mwh = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    trades_per_month = Column(Float, nullable=True)
    equity_curve = Column(JSON, nullable=True)
    monthly_performance = Column(JSON, nullable=True)
    parameters = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class DataQualityLog(Base):
    __tablename__ = "data_quality_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    checked_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    source = Column(String(50), nullable=False)
    latest_timestamp = Column(DateTime(timezone=True), nullable=True)
    age_minutes = Column(Float, nullable=True)
    missing_fields = Column(JSON, nullable=True)
    is_fresh = Column(Boolean, nullable=False, default=False)
    issues = Column(JSON, nullable=True)


class ShadowSignal(Base):
    """
    Tracks live signal predictions for realized-outcome comparison.
    Populated every time a signal is generated; realized prices filled in
    by the hourly scheduler once data is available.
    """
    __tablename__ = "shadow_signals"
    __table_args__ = (Index("ix_shadow_signals_timestamp", "timestamp"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    action = Column(String(60), nullable=False)
    regime = Column(String(50), nullable=True)
    current_price = Column(Float, nullable=True)
    predicted_price = Column(Float, nullable=True)
    p_rebound = Column(Float, nullable=True)
    p_negative = Column(Float, nullable=True)
    net_edge = Column(Float, nullable=True)
    residual_load_rising = Column(Boolean, nullable=True)
    solar_falling = Column(Boolean, nullable=True)
    evening_demand = Column(Boolean, nullable=True)
    realized_price_1h = Column(Float, nullable=True)
    realized_price_4h = Column(Float, nullable=True)
    realized_price_6h = Column(Float, nullable=True)
    realized_rebound = Column(Float, nullable=True)   # max(0, realized_6h - current)
    prediction_error = Column(Float, nullable=True)   # predicted_price - realized_price_6h
    guards_triggered = Column(JSON, nullable=True)    # list of guard names that fired
    tail_risk_score = Column(Float, nullable=True)
    sent_to_telegram = Column(Boolean, nullable=True, default=False)
    reason_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class RegimeSnapshot(Base):
    """Periodic snapshots of market regime for trend analysis."""
    __tablename__ = "regime_snapshots"
    __table_args__ = (Index("ix_regime_snapshots_timestamp", "timestamp"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    regime = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=True)
    renewable_share = Column(Float, nullable=True)
    price_volatility_24h = Column(Float, nullable=True)
    hours_negative_24h = Column(Float, nullable=True)
    solar_mw = Column(Float, nullable=True)
    wind_mw = Column(Float, nullable=True)
    oversupply_index = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class BatteryCapacityPoint(Base):
    """Installed battery storage capacity snapshots (from MaStR or interpolation)."""
    __tablename__ = "battery_capacity_points"
    __table_args__ = (Index("ix_battery_capacity_ts", "ts"),)

    id                    = Column(Integer, primary_key=True, autoincrement=True)
    ts                    = Column(DateTime(timezone=True), nullable=False)
    region                = Column(String(10), nullable=False, default="DE")
    installed_power_mw    = Column(Float, nullable=True)
    installed_capacity_mwh = Column(Float, nullable=True)
    source                = Column(String(50), nullable=False, default="mastr_milestone")
    data_quality_score    = Column(Float, nullable=True, default=0.70)
    created_at            = Column(DateTime(timezone=True), default=datetime.utcnow)


class BatteryFlowPoint(Base):
    """Hourly battery charge/discharge flow data (real or proxy)."""
    __tablename__ = "battery_flow_points"
    __table_args__ = (
        UniqueConstraint("ts", "region", name="uq_battery_flow_ts_region"),
        Index("ix_battery_flow_ts", "ts"),
    )

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    ts                  = Column(DateTime(timezone=True), nullable=False)
    region              = Column(String(10), nullable=False, default="DE")
    charging_mw         = Column(Float, nullable=True)
    discharging_mw      = Column(Float, nullable=True)
    net_battery_flow_mw = Column(Float, nullable=True)
    source              = Column(String(50), nullable=False, default="proxy")
    is_proxy            = Column(Boolean, nullable=False, default=True)
    data_quality_score  = Column(Float, nullable=True, default=0.35)
    created_at          = Column(DateTime(timezone=True), default=datetime.utcnow)


class StorageProxyPoint(Base):
    """Hourly storage proxy feature snapshots for analytics."""
    __tablename__ = "storage_proxy_points"
    __table_args__ = (
        UniqueConstraint("ts", name="uq_storage_proxy_ts"),
        Index("ix_storage_proxy_ts", "ts"),
    )

    id                          = Column(Integer, primary_key=True, autoincrement=True)
    ts                          = Column(DateTime(timezone=True), nullable=False)
    region                      = Column(String(10), nullable=False, default="DE")
    pv_surplus_index            = Column(Float, nullable=True)
    storage_charge_pressure     = Column(Float, nullable=True)
    storage_discharge_pressure  = Column(Float, nullable=True)
    midday_compression_index    = Column(Float, nullable=True)
    evening_arbitrage_index     = Column(Float, nullable=True)
    battery_saturation_proxy    = Column(Float, nullable=True)
    created_at                  = Column(DateTime(timezone=True), default=datetime.utcnow)


class TailEvent(Base):
    """Records extreme tail risk events for false-positive analysis and risk monitoring."""
    __tablename__ = "tail_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    event_type = Column(String(50), nullable=False)
    current_price = Column(Float, nullable=True)
    tail_risk_score = Column(Float, nullable=True)
    gap_risk_score = Column(Float, nullable=True)
    negative_price_streak = Column(Integer, nullable=True)
    max_price_gap_1h = Column(Float, nullable=True)
    volatility_24h = Column(Float, nullable=True)
    oversupply_stress_index = Column(Float, nullable=True)
    block_reason = Column(String(100), nullable=True)
    block_detail = Column(Text, nullable=True)
    would_have_entered = Column(Boolean, nullable=True)
    realized_outcome_6h = Column(Float, nullable=True)
    regime = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class BlockedTrade(Base):
    """Records every blocked signal for false-positive/false-negative analysis."""
    __tablename__ = "blocked_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    block_reason = Column(String(50), nullable=False)
    current_price = Column(Float, nullable=True)
    p_rebound = Column(Float, nullable=True)
    net_edge = Column(Float, nullable=True)
    tail_risk_score = Column(Float, nullable=True)
    gap_risk_score = Column(Float, nullable=True)
    negative_price_streak = Column(Integer, nullable=True)
    regime = Column(String(50), nullable=True)
    block_detail = Column(Text, nullable=True)
    price_6h_later = Column(Float, nullable=True)
    would_have_won = Column(Boolean, nullable=True)
    missed_pnl = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ShadowOutcome(Base):
    """Realized outcome for a shadow signal, filled in after 1h/2h/4h/6h."""
    __tablename__ = "shadow_outcomes"
    __table_args__ = (Index("ix_shadow_outcomes_signal_id", "signal_id"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(Integer, ForeignKey("shadow_signals.id"), nullable=False)
    evaluated_at = Column(DateTime(timezone=True), nullable=False)
    realized_price_1h = Column(Float, nullable=True)
    realized_price_2h = Column(Float, nullable=True)
    realized_price_4h = Column(Float, nullable=True)
    realized_rebound = Column(Float, nullable=True)
    simulated_pnl = Column(Float, nullable=True)
    would_hit_stop = Column(Boolean, nullable=True)
    would_hit_take_profit = Column(Boolean, nullable=True)
    outcome_status = Column(String(30), nullable=True)  # win, loss, partial, pending
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class NotificationEvent(Base):
    """Log of all outbound notifications for deduplication and auditing."""
    __tablename__ = "notification_events"
    __table_args__ = (Index("ix_notification_events_created_at", "created_at"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    channel = Column(String(50), nullable=False)      # telegram, webhook, etc.
    event_type = Column(String(50), nullable=False)   # signal, error, summary, retrain, drift
    signal_id = Column(Integer, nullable=True)
    fingerprint = Column(String(100), nullable=True)
    payload = Column(JSON, nullable=True)
    status = Column(String(20), nullable=False, default="sent")  # sent, failed, skipped
    error_message = Column(Text, nullable=True)


class DaemonHealthLog(Base):
    """Periodic daemon health snapshots for long-term trend analysis."""
    __tablename__ = "daemon_health_logs"
    __table_args__ = (Index("ix_daemon_health_created_at", "created_at"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    cycle_count = Column(Integer, nullable=True)
    consecutive_errors = Column(Integer, nullable=True, default=0)
    last_signal = Column(String(60), nullable=True)
    signal_mode = Column(String(20), nullable=True)
    rolling_pf = Column(Float, nullable=True)
    rolling_win_rate = Column(Float, nullable=True)
    telegram_sent_today = Column(Integer, nullable=True, default=0)
    notes = Column(Text, nullable=True)
