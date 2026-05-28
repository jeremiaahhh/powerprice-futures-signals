"""
Pydantic v2 schemas for PowerPrice Futures Signals API.

All monetary values are in EUR/MWh unless stated otherwise.
Probabilities are in [0.0, 1.0].
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SignalAction(str, Enum):
    """Possible actions produced by the signal engine."""

    NO_TRADE = "NO_TRADE"
    """Conditions do not warrant a position at this time."""

    WATCH_LONG_REBOUND = "WATCH_LONG_REBOUND"
    """Price is negative / very low but edge is below the cost threshold; monitor closely."""

    ENTER_LONG_REBOUND_SIGNAL = "ENTER_LONG_REBOUND_SIGNAL"
    """Price is negative / deeply discounted and net edge exceeds Futures costs; signal to enter long."""

    EXIT_TAKE_PROFIT_SIGNAL = "EXIT_TAKE_PROFIT_SIGNAL"
    """An open paper position has reached its take-profit target; signal to close."""

    EXIT_STOP_LOSS_SIGNAL = "EXIT_STOP_LOSS_SIGNAL"
    """An open paper position has hit the stop-loss level; signal to close."""

    HIGH_CONFIDENCE_SIGNAL = "HIGH_CONFIDENCE_SIGNAL"
    """All entry conditions met AND residual load rising, solar falling or evening demand spike; net_edge >= 35."""

    EXTREME_VOLATILITY_BLOCKED = "EXTREME_VOLATILITY_BLOCKED"
    """Signal blocked because 24h price volatility exceeds the extreme threshold (STRESS regime)."""

    TAIL_RISK_BLOCKED = "TAIL_RISK_BLOCKED"
    """Trade blocked by tail risk engine: price below floor, streak too long, or extreme gap."""

    GAP_RISK_BLOCKED = "GAP_RISK_BLOCKED"
    """Trade blocked due to extreme intra-hour price gap (gap risk exposure)."""

    RISK_BLOCKED = "RISK_BLOCKED"
    """Signal would have fired but risk guardrails prevented it (e.g. too many open positions)."""

    DATA_QUALITY_BLOCKED = "DATA_QUALITY_BLOCKED"
    """Signal engine could not generate a signal due to stale or missing market data."""


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class BacktestStrategy(str, Enum):
    NAIVE = "naive"
    ML_REBOUND = "ml_rebound"


# ---------------------------------------------------------------------------
# Futures Cost Model
# ---------------------------------------------------------------------------


class CostModelConfig(BaseModel):
    """
    User-supplied or system-default Futures cost assumptions.
    All fields in EUR/MWh or % per annum where noted.
    """

    avg_spread_eur_mwh: float = Field(
        default=5.0,
        ge=0.0,
        description="Bid-ask spread at entry + exit, EUR/MWh",
    )
    slippage_eur_mwh: float = Field(
        default=3.0,
        ge=0.0,
        description="Expected market-impact / slippage on entry and exit, EUR/MWh",
    )
    overnight_fee_annual_pct: float = Field(
        default=8.0,
        ge=0.0,
        description="Broker overnight / financing fee, % per annum on notional",
    )
    weekend_fee_multiplier: float = Field(
        default=1.5,
        ge=1.0,
        description="Multiplier applied to the overnight fee over weekends (broker typically charges 3 days)",
    )
    broker_markup_eur_mwh: float = Field(
        default=1.0,
        ge=0.0,
        description="Fixed broker markup per MWh on top of spread",
    )
    safety_buffer_eur_mwh: float = Field(
        default=5.0,
        ge=0.0,
        description="Conservative buffer added to total cost estimate to account for unmodelled friction",
    )
    min_edge_threshold: float = Field(
        default=30.0,
        ge=0.0,
        description="Minimum net edge (EUR/MWh) required before issuing ENTER signal",
    )
    holding_hours: int = Field(
        default=4,
        ge=1,
        le=168,
        description="Assumed holding period in hours for overnight-fee calculation",
    )

    @field_validator("avg_spread_eur_mwh", "slippage_eur_mwh", "broker_markup_eur_mwh", "safety_buffer_eur_mwh")
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Cost parameters must be non-negative")
        return v


class CostBreakdown(BaseModel):
    """Itemised Futures cost estimate for a trade."""

    spread_cost_eur_mwh: float = Field(description="Entry + exit spread, EUR/MWh")
    slippage_cost_eur_mwh: float = Field(description="Market impact / slippage, EUR/MWh")
    overnight_fee_eur_mwh: float = Field(description="Overnight financing cost for the expected holding period, EUR/MWh")
    broker_markup_eur_mwh: float = Field(description="Fixed broker markup, EUR/MWh")
    safety_buffer_eur_mwh: float = Field(description="Conservative buffer, EUR/MWh")
    total_eur_mwh: float = Field(description="Sum of all cost components, EUR/MWh")
    holding_hours_assumed: int = Field(description="Number of hours used for overnight fee calculation")
    is_weekend: bool = Field(default=False, description="Whether the weekend fee multiplier was applied")


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


class SignalResponse(BaseModel):
    """
    Full signal payload returned by GET /futures/signal.
    This is purely informational — no live execution occurs.
    """

    action: SignalAction = Field(description="Recommended action for this time window")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence score for the primary predicted class, [0, 1]",
    )
    timestamp: datetime = Field(description="UTC timestamp for which the signal applies")
    generated_at: datetime = Field(description="UTC timestamp at which the signal was generated")

    # Prices
    current_price: Optional[float] = Field(
        default=None,
        description="Latest available Day-Ahead spot price, EUR/MWh",
    )
    predicted_price: Optional[float] = Field(
        default=None,
        description="Model point-estimate of price at the end of the holding horizon, EUR/MWh",
    )

    # Probabilities
    p_negative: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Probability that the current price is negative (< 0 EUR/MWh)",
    )
    p_rebound: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Probability that price rebounds above the Futures break-even within the holding horizon",
    )

    # Edge calculation
    expected_rebound_eur_mwh: Optional[float] = Field(
        default=None,
        description="Expected gross price move from entry to rebound target, EUR/MWh",
    )
    gross_edge: Optional[float] = Field(
        default=None,
        description="Raw directional edge before subtracting Futures transaction costs, EUR/MWh",
    )
    estimated_futures_costs: Optional[float] = Field(
        default=None,
        description="Total estimated Futures costs (spread + slippage + overnight + markup + buffer), EUR/MWh",
    )
    net_edge: Optional[float] = Field(
        default=None,
        description="gross_edge minus estimated_futures_costs; positive means trade has positive expected value, EUR/MWh",
    )
    cost_breakdown: Optional[CostBreakdown] = Field(
        default=None,
        description="Itemised cost components",
    )

    # Trade levels
    stop_loss: Optional[float] = Field(
        default=None,
        description="Suggested stop-loss price level, EUR/MWh",
    )
    take_profit: Optional[float] = Field(
        default=None,
        description="Suggested take-profit price level, EUR/MWh",
    )
    max_holding_hours: Optional[int] = Field(
        default=None,
        description="Maximum recommended holding period before time-decay erodes the edge, hours",
    )

    # Narrative & diagnostics
    reason: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of why this action was chosen",
    )
    risk_warnings: List[str] = Field(
        default_factory=list,
        description="List of active risk flags (e.g. 'high volatility regime', 'approaching stop-loss')",
    )
    feature_explanation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Top features driving the model prediction with their signed contributions",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "action": "ENTER_LONG_REBOUND_SIGNAL",
            "confidence": 0.78,
            "timestamp": "2024-03-15T02:00:00Z",
            "generated_at": "2024-03-15T02:01:05Z",
            "current_price": -12.5,
            "predicted_price": 35.0,
            "p_negative": 0.91,
            "p_rebound": 0.74,
            "expected_rebound_eur_mwh": 47.5,
            "gross_edge": 47.5,
            "estimated_futures_costs": 16.0,
            "net_edge": 31.5,
            "stop_loss": -30.0,
            "take_profit": 30.0,
            "max_holding_hours": 6,
            "reason": "Price is deeply negative (-12.5 EUR/MWh). ML model assigns 91% probability of negative price regime and 74% probability of rebound within 6 hours. Net edge of 31.5 EUR/MWh exceeds minimum threshold of 10.0 EUR/MWh.",
            "risk_warnings": ["extreme negative price — heightened tail risk"],
            "feature_explanation": {
                "wind_onshore_mw": 0.31,
                "residual_load_mw": -0.25,
                "hour_of_day": 0.18,
            },
        }
    }}


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


class HourlyForecastPoint(BaseModel):
    """Single hour in a multi-horizon price forecast."""

    timestamp: datetime = Field(description="UTC hour start")
    predicted_price_eur_mwh: float = Field(description="Point estimate, EUR/MWh")
    lower_bound_eur_mwh: Optional[float] = Field(default=None, description="Lower confidence bound, EUR/MWh")
    upper_bound_eur_mwh: Optional[float] = Field(default=None, description="Upper confidence bound, EUR/MWh")
    p_negative: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    p_rebound: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ForecastResponse(BaseModel):
    """Multi-horizon forecast for the next N hours."""

    generated_at: datetime
    model_name: str
    model_version: Optional[str] = None
    horizon_hours: int = Field(ge=1, le=168)
    forecast: List[HourlyForecastPoint]
    feature_importance: Optional[Dict[str, float]] = Field(
        default=None,
        description="Global feature importances from the underlying model, summing to 1.0",
    )


# ---------------------------------------------------------------------------
# Market Data
# ---------------------------------------------------------------------------


class DataPoint(BaseModel):
    """A single hourly market data observation."""

    timestamp: datetime
    source: str = Field(description="Data source identifier (smard, entsoe, openmeteo, etc.)")
    price_eur_mwh: Optional[float] = None
    intraday_price_eur_mwh: Optional[float] = None
    load_mw: Optional[float] = None
    wind_onshore_mw: Optional[float] = None
    wind_offshore_mw: Optional[float] = None
    solar_mw: Optional[float] = None
    residual_load_mw: Optional[float] = None
    net_export_mw: Optional[float] = None
    temperature_c: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    solar_radiation_wm2: Optional[float] = None
    cloud_cover_pct: Optional[float] = None
    is_holiday: bool = False
    is_weekend: bool = False
    hour: Optional[int] = Field(default=None, ge=0, le=23)
    month: Optional[int] = Field(default=None, ge=1, le=12)


class DataPointPage(BaseModel):
    """Paginated list of historical data points."""

    total: int
    page: int
    page_size: int
    data: List[DataPoint]


class DataQualityResponse(BaseModel):
    """Result of a data-freshness and completeness check."""

    checked_at: datetime
    source: str
    latest_timestamp: Optional[datetime] = None
    age_minutes: Optional[float] = Field(default=None, description="How many minutes ago the latest row is dated")
    is_fresh: bool = Field(description="True if data is within the acceptable staleness window")
    missing_fields: List[str] = Field(
        default_factory=list,
        description="Column names that are NULL in the latest row",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Human-readable descriptions of data quality problems",
    )
    row_count_last_24h: Optional[int] = Field(default=None, description="Number of rows ingested in the last 24 hours")


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


class BacktestParams(BaseModel):
    """Parameters for a backtest run."""

    strategy: BacktestStrategy = Field(default=BacktestStrategy.ML_REBOUND)
    start_date: datetime
    end_date: datetime
    notional_size_mwh: float = Field(default=1.0, gt=0.0)
    cost_model: CostModelConfig = Field(default_factory=CostModelConfig)
    min_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    max_holding_hours: int = Field(default=6, ge=1, le=168)
    stop_loss_eur_mwh: float = Field(default=20.0, gt=0.0)
    take_profit_eur_mwh: float = Field(default=30.0, gt=0.0)
    use_walk_forward: bool = Field(default=False, description="Retrain model at each fold boundary")
    walk_forward_window_days: int = Field(default=90, ge=30)

    @model_validator(mode="after")
    def end_after_start(self) -> "BacktestParams":
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self


class MonthlyPerformance(BaseModel):
    """Aggregated performance metrics for a single calendar month."""

    period: str = Field(description="YYYY-MM format")
    trades: int
    win_rate_pct: float
    pnl_eur: float
    return_pct: float


class BacktestResult(BaseModel):
    """Full backtest result payload."""

    run_id: str
    strategy: BacktestStrategy
    start_date: datetime
    end_date: datetime
    parameters: BacktestParams

    # Summary statistics
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown_pct: float
    profit_factor: Optional[float] = Field(
        default=None,
        description="Gross profit divided by gross loss; > 1.0 is profitable",
    )
    win_rate_pct: float
    avg_trade_eur_mwh: float
    best_trade_eur_mwh: Optional[float] = None
    worst_trade_eur_mwh: Optional[float] = None
    total_trades: int
    winning_trades: int
    losing_trades: int
    trades_per_month: float
    avg_holding_hours: Optional[float] = None

    # Time-series
    equity_curve: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of {timestamp, equity, drawdown} dicts for charting",
    )
    monthly_performance: List[MonthlyPerformance] = Field(default_factory=list)

    created_at: datetime


class BacktestComparison(BaseModel):
    """Side-by-side comparison of two or more backtest strategies."""

    generated_at: datetime
    results: List[BacktestResult]
    summary_table: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tabular comparison with key metrics for each strategy as rows",
    )


# ---------------------------------------------------------------------------
# Paper Trading
# ---------------------------------------------------------------------------


class PaperTradeRequest(BaseModel):
    """Open a new paper position from a signal."""

    signal_id: Optional[int] = Field(default=None, description="ID of the FuturesSignal record that triggered this trade")
    entry_price: float = Field(description="Entry price for the paper position, EUR/MWh")
    notional_size_mwh: float = Field(default=1.0, gt=0.0, description="Contract size in MWh")
    stop_loss: Optional[float] = Field(default=None, description="Stop-loss price level, EUR/MWh")
    take_profit: Optional[float] = Field(default=None, description="Take-profit price level, EUR/MWh")
    max_holding_hours: Optional[int] = Field(default=None, ge=1, le=168)


class PaperPositionResponse(BaseModel):
    """A paper trading position record."""

    id: int
    signal_id: Optional[int] = None
    status: PositionStatus
    entry_price: float
    exit_price: Optional[float] = None
    entry_timestamp: datetime
    exit_timestamp: Optional[datetime] = None
    notional_size_mwh: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_holding_hours: Optional[int] = None
    pnl_eur: Optional[float] = None
    futures_costs_eur: Optional[float] = None
    net_pnl_eur: Optional[float] = None
    exit_reason: Optional[str] = None
    holding_hours: Optional[float] = Field(
        default=None,
        description="Elapsed hours since entry (for open positions) or total holding time (for closed positions)",
    )
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ClosePositionRequest(BaseModel):
    """Request body for manually closing a paper position."""

    exit_price: float = Field(description="Price at which to close the position, EUR/MWh")
    exit_reason: str = Field(default="manual", description="Reason for closing (manual, stop_loss, take_profit, time_exit)")


class PaperTradeResponse(BaseModel):
    """Response after opening or closing a paper position."""

    success: bool
    position: PaperPositionResponse
    message: str


class PaperStatusResponse(BaseModel):
    """Summary of all paper trading activity."""

    generated_at: datetime
    open_positions: int
    total_closed_positions: int
    total_net_pnl_eur: float
    total_gross_pnl_eur: float
    total_futures_costs_eur: float
    win_rate_pct: Optional[float] = None
    profit_factor: Optional[float] = None
    best_trade_net_pnl_eur: Optional[float] = None
    worst_trade_net_pnl_eur: Optional[float] = None
    avg_holding_hours: Optional[float] = None
    positions: List[PaperPositionResponse] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """
    Request body for the manual data-ingest trigger endpoint (POST /data/ingest).
    Allows operators to kick off a back-fill or on-demand refresh.
    """

    source: str = Field(
        default="smard",
        description="Data source to ingest from: smard | entsoe | openmeteo | all",
    )
    start_date: Optional[datetime] = Field(
        default=None,
        description="Start of the ingestion window (UTC). If None, defaults to the last available timestamp.",
    )
    end_date: Optional[datetime] = Field(
        default=None,
        description="End of the ingestion window (UTC). If None, defaults to now.",
    )
    force_overwrite: bool = Field(
        default=False,
        description="If True, overwrite existing rows that fall within the date range.",
    )

    @field_validator("source")
    @classmethod
    def valid_source(cls, v: str) -> str:
        allowed = {"smard", "entsoe", "openmeteo", "all"}
        if v.lower() not in allowed:
            raise ValueError(f"source must be one of {allowed}")
        return v.lower()

    @model_validator(mode="after")
    def end_after_start_if_both(self) -> "IngestRequest":
        if self.start_date and self.end_date and self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self


class IngestResponse(BaseModel):
    """Result of a data ingestion task."""

    task_id: Optional[str] = Field(default=None, description="Celery task ID for async tracking")
    source: str
    rows_inserted: int
    rows_updated: int
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class ComponentHealth(BaseModel):
    """Health status of a single service component."""

    name: str
    status: str = Field(description="ok | degraded | down")
    latency_ms: Optional[float] = None
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """Full system health check response."""

    status: str = Field(description="ok | degraded | down — reflects worst component state")
    version: str
    environment: str
    checked_at: datetime
    components: List[ComponentHealth] = Field(default_factory=list)
    signal_only_mode: bool = Field(
        description="When True, the platform never sends live orders; it is purely advisory."
    )
