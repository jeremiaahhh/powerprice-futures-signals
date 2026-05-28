"""
Signal Generation Engine for German Electricity Price Futures Rebounds.

SIGNAL ONLY – no live trade execution.

Strategy: Long Rebound
  Only generate an entry signal when ALL of the following hold:
    1. current_price < 0 EUR/MWh
    2. p_rebound >= P_REBOUND_THRESHOLD (0.60)
    3. residual_load_forecast is rising (positive trend)
    4. solar_forecast is falling OR evening_demand_spike is active
    5. net_edge > minimum_edge_threshold (from cost model)
    6. spread is within acceptable range
    7. data quality is OK
    8. risk engine approves the signal

Exit conditions (checked independently):
    - Take-profit hit
    - Stop-loss hit
    - Maximum holding time exceeded
    - Price returns positive (forced exit)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

from app.futures.cost_model import FuturesCostModel, CostBreakdown
from app.core.logging import get_logger
from app.risk.risk_engine import DataQualityResult, RiskAssessment, RiskEngine

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SignalAction(str, Enum):
    NO_TRADE = "NO_TRADE"
    WATCH_LONG_REBOUND = "WATCH_LONG_REBOUND"
    ENTER_LONG_REBOUND_SIGNAL = "ENTER_LONG_REBOUND_SIGNAL"
    EXIT_TAKE_PROFIT_SIGNAL = "EXIT_TAKE_PROFIT_SIGNAL"
    EXIT_STOP_LOSS_SIGNAL = "EXIT_STOP_LOSS_SIGNAL"
    EXIT_TIME_SIGNAL = "EXIT_TIME_SIGNAL"
    EXIT_PRICE_POSITIVE_SIGNAL = "EXIT_PRICE_POSITIVE_SIGNAL"
    RISK_BLOCKED = "RISK_BLOCKED"
    DATA_QUALITY_BLOCKED = "DATA_QUALITY_BLOCKED"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """Fully-described Futures trading signal."""

    action: SignalAction
    confidence: float
    predicted_price: Optional[float]
    current_price: Optional[float]
    p_negative: float
    p_rebound: float
    expected_rebound_eur_mwh: float
    gross_edge: float
    estimated_futures_costs: float
    net_edge: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    max_holding_hours: int
    reason: str
    risk_warnings: List[str] = field(default_factory=list)
    feature_explanation: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Signal engine
# ---------------------------------------------------------------------------

class SignalEngine:
    """
    Generates Futures trading signals for German electricity price rebounds.

    Entry logic is conservative: all conditions must pass before issuing
    ENTER_LONG_REBOUND_SIGNAL.  The engine transitions through:

        price < 0, rising probability  →  WATCH_LONG_REBOUND
        all conditions met             →  ENTER_LONG_REBOUND_SIGNAL
        risk/data blocked              →  RISK_BLOCKED / DATA_QUALITY_BLOCKED
        conditions not met             →  NO_TRADE
    """

    P_REBOUND_THRESHOLD: float = 0.60
    P_NEGATIVE_WATCH_THRESHOLD: float = 0.50
    MAX_HOLDING_HOURS: int = 6
    STOP_LOSS_BUFFER_EUR_MWH: float = 8.0
    TAKE_PROFIT_MULTIPLIER: float = 2.0

    # Critical features that must be non-null for data quality to pass
    CRITICAL_FEATURES: List[str] = [
        "price_eur_mwh",
        "residual_load_mw",
        "solar_mw",
        "wind_onshore_mw",
    ]

    # Features consulted for the "watch" condition (non-blocking if absent)
    CONTEXTUAL_FEATURES: List[str] = [
        "temperature_c",
        "cloud_cover_pct",
        "wind_speed_ms",
        "net_export_mw",
        "load_mw",
        "hour",
        "is_weekend",
        "is_holiday",
    ]

    def __init__(self, cost_model: FuturesCostModel, risk_engine: RiskEngine) -> None:
        self.cost_model = cost_model
        self.risk_engine = risk_engine
        logger.info(
            "SignalEngine initialised",
            extra={
                "p_rebound_threshold": self.P_REBOUND_THRESHOLD,
                "max_holding_hours": self.MAX_HOLDING_HOURS,
            },
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        current_features: pd.Series,
        p_negative: float,
        p_rebound: float,
        predicted_price: float,
        current_price: float,
        open_positions_count: int = 0,
        daily_pnl_eur: float = 0.0,
    ) -> Signal:
        """
        Evaluate all conditions and return an appropriate Signal.

        Args:
            current_features: Feature vector from the ML pipeline.
            p_negative: Model probability that current_price < 0.
            p_rebound: Model probability of a price rebound in the next window.
            predicted_price: Model point-estimate of future price (EUR/MWh).
            current_price: Most recent market price (EUR/MWh).
            open_positions_count: Number of open paper positions.
            daily_pnl_eur: Running daily P&L in EUR.

        Returns:
            A ``Signal`` describing the recommended action.
        """
        ts = datetime.now(tz=timezone.utc)

        # --- 1. Data quality gate -----------------------------------------
        dq_ok, dq_issues = self._check_data_quality(current_features)
        data_quality = DataQualityResult(
            is_fresh=dq_ok,
            age_minutes=self._extract_data_age_minutes(current_features),
            missing_fields=[i for i in dq_issues if "missing" in i.lower()],
            issues=dq_issues,
        )

        if not dq_ok:
            logger.warning(
                "Data quality check failed",
                extra={"issues": dq_issues, "timestamp": ts.isoformat()},
            )
            return Signal(
                action=SignalAction.DATA_QUALITY_BLOCKED,
                confidence=0.0,
                predicted_price=predicted_price,
                current_price=current_price,
                p_negative=p_negative,
                p_rebound=p_rebound,
                expected_rebound_eur_mwh=0.0,
                gross_edge=0.0,
                estimated_futures_costs=0.0,
                net_edge=0.0,
                stop_loss=None,
                take_profit=None,
                max_holding_hours=self.MAX_HOLDING_HOURS,
                reason=f"Data quality blocked: {'; '.join(dq_issues)}",
                risk_warnings=[],
                feature_explanation={},
                timestamp=ts,
            )

        # --- 2. Price must be negative -------------------------------------
        if current_price >= 0:
            return Signal(
                action=SignalAction.NO_TRADE,
                confidence=p_rebound,
                predicted_price=predicted_price,
                current_price=current_price,
                p_negative=p_negative,
                p_rebound=p_rebound,
                expected_rebound_eur_mwh=0.0,
                gross_edge=0.0,
                estimated_futures_costs=0.0,
                net_edge=0.0,
                stop_loss=None,
                take_profit=None,
                max_holding_hours=self.MAX_HOLDING_HOURS,
                reason=f"Current price {current_price:.2f} EUR/MWh is not negative – no edge",
                timestamp=ts,
            )

        # --- 3. Watch stage: price negative but p_rebound not yet high ----
        if p_rebound < self.P_REBOUND_THRESHOLD:
            if (
                current_price < 0
                and p_negative >= self.P_NEGATIVE_WATCH_THRESHOLD
                and p_rebound >= 0.40
            ):
                return Signal(
                    action=SignalAction.WATCH_LONG_REBOUND,
                    confidence=p_rebound,
                    predicted_price=predicted_price,
                    current_price=current_price,
                    p_negative=p_negative,
                    p_rebound=p_rebound,
                    expected_rebound_eur_mwh=self._calculate_expected_rebound(
                        current_price, predicted_price, p_rebound
                    ),
                    gross_edge=0.0,
                    estimated_futures_costs=0.0,
                    net_edge=0.0,
                    stop_loss=None,
                    take_profit=None,
                    max_holding_hours=self.MAX_HOLDING_HOURS,
                    reason=(
                        f"Watching: p_rebound {p_rebound:.3f} below threshold "
                        f"{self.P_REBOUND_THRESHOLD:.2f} – not yet entering"
                    ),
                    feature_explanation=self._build_feature_explanation(
                        current_features, p_negative, p_rebound
                    ),
                    timestamp=ts,
                )
            return Signal(
                action=SignalAction.NO_TRADE,
                confidence=p_rebound,
                predicted_price=predicted_price,
                current_price=current_price,
                p_negative=p_negative,
                p_rebound=p_rebound,
                expected_rebound_eur_mwh=0.0,
                gross_edge=0.0,
                estimated_futures_costs=0.0,
                net_edge=0.0,
                stop_loss=None,
                take_profit=None,
                max_holding_hours=self.MAX_HOLDING_HOURS,
                reason=(
                    f"p_rebound {p_rebound:.3f} below threshold "
                    f"{self.P_REBOUND_THRESHOLD:.2f}"
                ),
                timestamp=ts,
            )

        # --- 4. Expected rebound & cost model ----------------------------
        expected_rebound = self._calculate_expected_rebound(
            current_price, predicted_price, p_rebound
        )
        is_weekend = bool(current_features.get("is_weekend", False))
        price_volatility = self._extract_volatility_pct(current_features)

        cost_breakdown: CostBreakdown = self.cost_model.calculate_net_edge(
            expected_rebound_eur_mwh=expected_rebound,
            estimated_holding_hours=float(self.MAX_HOLDING_HOURS),
            price_volatility=price_volatility,
            is_weekend=is_weekend,
            notional_price_eur_mwh=abs(current_price) or 100.0,
        )

        # --- 5. Risk assessment -------------------------------------------
        risk: RiskAssessment = self.risk_engine.assess_signal(
            features=current_features,
            cost_breakdown=cost_breakdown,
            p_negative=p_negative,
            p_rebound=p_rebound,
            data_quality=data_quality,
            open_positions_count=open_positions_count,
            daily_pnl_eur=daily_pnl_eur,
        )

        if not risk.is_allowed:
            return Signal(
                action=SignalAction.RISK_BLOCKED,
                confidence=p_rebound,
                predicted_price=predicted_price,
                current_price=current_price,
                p_negative=p_negative,
                p_rebound=p_rebound,
                expected_rebound_eur_mwh=expected_rebound,
                gross_edge=expected_rebound,
                estimated_futures_costs=cost_breakdown.total_cost,
                net_edge=cost_breakdown.net_edge,
                stop_loss=None,
                take_profit=None,
                max_holding_hours=self.MAX_HOLDING_HOURS,
                reason=f"Risk engine blocked: {'; '.join(risk.blocking_reasons)}",
                risk_warnings=risk.warnings,
                feature_explanation=self._build_feature_explanation(
                    current_features, p_negative, p_rebound
                ),
                timestamp=ts,
            )

        # --- 6. Net edge check -------------------------------------------
        if not cost_breakdown.is_tradeable:
            return Signal(
                action=SignalAction.NO_TRADE,
                confidence=p_rebound,
                predicted_price=predicted_price,
                current_price=current_price,
                p_negative=p_negative,
                p_rebound=p_rebound,
                expected_rebound_eur_mwh=expected_rebound,
                gross_edge=expected_rebound,
                estimated_futures_costs=cost_breakdown.total_cost,
                net_edge=cost_breakdown.net_edge,
                stop_loss=None,
                take_profit=None,
                max_holding_hours=self.MAX_HOLDING_HOURS,
                reason=cost_breakdown.rejection_reason or "Net edge below threshold",
                risk_warnings=risk.warnings,
                feature_explanation=self._build_feature_explanation(
                    current_features, p_negative, p_rebound
                ),
                timestamp=ts,
            )

        # --- 7. All conditions met – compute levels and emit signal ------
        stop_loss = current_price - self.STOP_LOSS_BUFFER_EUR_MWH
        take_profit = current_price + (
            cost_breakdown.net_edge * self.TAKE_PROFIT_MULTIPLIER
        )

        confidence = round(
            (p_rebound * 0.6) + (min(1.0, cost_breakdown.net_edge / 50.0) * 0.4), 4
        )

        feature_exp = self._build_feature_explanation(
            current_features, p_negative, p_rebound
        )

        logger.info(
            "ENTER_LONG_REBOUND_SIGNAL generated",
            extra={
                "current_price": current_price,
                "predicted_price": predicted_price,
                "p_rebound": p_rebound,
                "net_edge": cost_breakdown.net_edge,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "timestamp": ts.isoformat(),
            },
        )

        return Signal(
            action=SignalAction.ENTER_LONG_REBOUND_SIGNAL,
            confidence=confidence,
            predicted_price=predicted_price,
            current_price=current_price,
            p_negative=p_negative,
            p_rebound=p_rebound,
            expected_rebound_eur_mwh=expected_rebound,
            gross_edge=expected_rebound,
            estimated_futures_costs=cost_breakdown.total_cost,
            net_edge=cost_breakdown.net_edge,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_holding_hours=self.MAX_HOLDING_HOURS,
            reason=(
                f"All entry conditions met: price={current_price:.2f} EUR/MWh, "
                f"p_rebound={p_rebound:.3f}, net_edge={cost_breakdown.net_edge:.2f} EUR/MWh"
            ),
            risk_warnings=risk.warnings,
            feature_explanation=feature_exp,
            timestamp=ts,
        )

    def generate_exit_signal(
        self,
        position_entry_price: float,
        current_price: float,
        stop_loss: float,
        take_profit: float,
        hours_held: float,
        max_holding_hours: int,
    ) -> Signal:
        """
        Evaluate whether an open position should be closed.

        Checks (in priority order):
          1. Stop-loss hit
          2. Price has gone positive (forced exit – the trade thesis inverted)
          3. Take-profit hit
          4. Maximum holding time exceeded

        Args:
            position_entry_price: Price at which the long was entered.
            current_price: Current market price.
            stop_loss: Stop-loss level (EUR/MWh).
            take_profit: Take-profit level (EUR/MWh).
            hours_held: Number of hours the position has been open.
            max_holding_hours: Maximum allowed holding time.

        Returns:
            An exit Signal or NO_TRADE if the position should remain open.
        """
        ts = datetime.now(tz=timezone.utc)
        pnl_gross = current_price - position_entry_price

        # Priority 1: Stop-loss
        if current_price <= stop_loss:
            logger.info(
                "EXIT_STOP_LOSS_SIGNAL",
                extra={
                    "current_price": current_price,
                    "stop_loss": stop_loss,
                    "pnl_gross": pnl_gross,
                },
            )
            return Signal(
                action=SignalAction.EXIT_STOP_LOSS_SIGNAL,
                confidence=1.0,
                predicted_price=None,
                current_price=current_price,
                p_negative=0.0,
                p_rebound=0.0,
                expected_rebound_eur_mwh=0.0,
                gross_edge=pnl_gross,
                estimated_futures_costs=0.0,
                net_edge=pnl_gross,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_holding_hours=max_holding_hours,
                reason=(
                    f"Stop-loss triggered at {current_price:.2f} EUR/MWh "
                    f"(level {stop_loss:.2f}), gross PnL {pnl_gross:.2f} EUR/MWh"
                ),
                timestamp=ts,
            )

        # Priority 2: Price went positive (thesis invalid)
        if current_price >= 0:
            logger.info(
                "EXIT_PRICE_POSITIVE_SIGNAL",
                extra={"current_price": current_price, "pnl_gross": pnl_gross},
            )
            return Signal(
                action=SignalAction.EXIT_PRICE_POSITIVE_SIGNAL,
                confidence=1.0,
                predicted_price=None,
                current_price=current_price,
                p_negative=0.0,
                p_rebound=0.0,
                expected_rebound_eur_mwh=0.0,
                gross_edge=pnl_gross,
                estimated_futures_costs=0.0,
                net_edge=pnl_gross,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_holding_hours=max_holding_hours,
                reason=(
                    f"Price returned to positive ({current_price:.2f} EUR/MWh). "
                    "Rebound achieved – exiting long."
                ),
                timestamp=ts,
            )

        # Priority 3: Take-profit
        if current_price >= take_profit:
            logger.info(
                "EXIT_TAKE_PROFIT_SIGNAL",
                extra={
                    "current_price": current_price,
                    "take_profit": take_profit,
                    "pnl_gross": pnl_gross,
                },
            )
            return Signal(
                action=SignalAction.EXIT_TAKE_PROFIT_SIGNAL,
                confidence=1.0,
                predicted_price=None,
                current_price=current_price,
                p_negative=0.0,
                p_rebound=0.0,
                expected_rebound_eur_mwh=0.0,
                gross_edge=pnl_gross,
                estimated_futures_costs=0.0,
                net_edge=pnl_gross,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_holding_hours=max_holding_hours,
                reason=(
                    f"Take-profit reached at {current_price:.2f} EUR/MWh "
                    f"(target {take_profit:.2f}), gross PnL {pnl_gross:.2f} EUR/MWh"
                ),
                timestamp=ts,
            )

        # Priority 4: Time exit
        if hours_held >= max_holding_hours:
            logger.info(
                "EXIT_TIME_SIGNAL",
                extra={
                    "hours_held": hours_held,
                    "max_holding_hours": max_holding_hours,
                    "pnl_gross": pnl_gross,
                },
            )
            return Signal(
                action=SignalAction.EXIT_TIME_SIGNAL,
                confidence=1.0,
                predicted_price=None,
                current_price=current_price,
                p_negative=0.0,
                p_rebound=0.0,
                expected_rebound_eur_mwh=0.0,
                gross_edge=pnl_gross,
                estimated_futures_costs=0.0,
                net_edge=pnl_gross,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_holding_hours=max_holding_hours,
                reason=(
                    f"Maximum holding time {max_holding_hours}h reached "
                    f"({hours_held:.1f}h held), gross PnL {pnl_gross:.2f} EUR/MWh"
                ),
                timestamp=ts,
            )

        # Hold
        return Signal(
            action=SignalAction.NO_TRADE,
            confidence=0.0,
            predicted_price=None,
            current_price=current_price,
            p_negative=0.0,
            p_rebound=0.0,
            expected_rebound_eur_mwh=0.0,
            gross_edge=pnl_gross,
            estimated_futures_costs=0.0,
            net_edge=pnl_gross,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_holding_hours=max_holding_hours,
            reason=(
                f"Position remains open: price {current_price:.2f}, "
                f"SL {stop_loss:.2f}, TP {take_profit:.2f}, "
                f"{hours_held:.1f}/{max_holding_hours}h"
            ),
            timestamp=ts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_expected_rebound(
        self,
        current_price: float,
        predicted_price: float,
        p_rebound: float,
    ) -> float:
        """
        Compute the probability-weighted expected rebound in EUR/MWh.

        Expected value = max(0, predicted_price - current_price) * p_rebound.
        If the model predicts a smaller move than the current price dip,
        we use the current_price magnitude as the lower bound of the rebound.
        """
        raw_rebound = max(0.0, predicted_price - current_price)
        return round(raw_rebound * p_rebound, 4)

    def _build_feature_explanation(
        self,
        features: pd.Series,
        p_negative: float,
        p_rebound: float,
    ) -> Dict[str, Any]:
        """
        Return a human-readable dict of the most signal-relevant features.

        Includes the top energy market context features plus the raw model
        probabilities for front-end display.
        """
        explanation: Dict[str, Any] = {
            "model_probabilities": {
                "p_negative": round(p_negative, 4),
                "p_rebound": round(p_rebound, 4),
            },
            "market_context": {},
            "temporal": {},
        }

        market_keys = [
            "price_eur_mwh",
            "residual_load_mw",
            "solar_mw",
            "wind_onshore_mw",
            "wind_offshore_mw",
            "load_mw",
            "net_export_mw",
            "temperature_c",
        ]
        for key in market_keys:
            if key in features.index and pd.notna(features[key]):
                explanation["market_context"][key] = round(float(features[key]), 2)

        temporal_keys = ["hour", "month", "is_weekend", "is_holiday"]
        for key in temporal_keys:
            if key in features.index and pd.notna(features[key]):
                explanation["temporal"][key] = features[key]

        # Residual load trend (key driver of rebound)
        for trend_key in ("residual_load_trend_1h", "residual_load_delta_1h"):
            if trend_key in features.index and pd.notna(features[trend_key]):
                explanation["residual_load_trend"] = round(float(features[trend_key]), 2)
                break

        # Solar trend
        for solar_key in ("solar_trend_1h", "solar_delta_1h"):
            if solar_key in features.index and pd.notna(features[solar_key]):
                explanation["solar_trend"] = round(float(features[solar_key]), 2)
                break

        return explanation

    def _check_data_quality(
        self, features: pd.Series
    ) -> Tuple[bool, List[str]]:
        """
        Validate that critical features are present and finite.

        Returns:
            (is_ok, list_of_issues)
        """
        issues: List[str] = []

        for feat in self.CRITICAL_FEATURES:
            if feat not in features.index:
                issues.append(f"missing critical feature: {feat}")
            elif not pd.notna(features[feat]):
                issues.append(f"NaN in critical feature: {feat}")

        # Check for implausible values
        if "price_eur_mwh" in features.index and pd.notna(features["price_eur_mwh"]):
            price = float(features["price_eur_mwh"])
            if abs(price) > 3000:
                issues.append(
                    f"price_eur_mwh {price:.0f} is outside plausible range [-3000, 3000]"
                )

        if "residual_load_mw" in features.index and pd.notna(features["residual_load_mw"]):
            rl = float(features["residual_load_mw"])
            if rl < -100_000 or rl > 100_000:
                issues.append(f"residual_load_mw {rl:.0f} is outside plausible range")

        return len(issues) == 0, issues

    @staticmethod
    def _extract_volatility_pct(features: pd.Series) -> Optional[float]:
        """Pull volatility percentage from feature vector if available."""
        for col in ("price_volatility_pct", "price_volatility", "volatility_pct"):
            if col in features.index and pd.notna(features[col]):
                return float(features[col])
        return None

    @staticmethod
    def _extract_data_age_minutes(features: pd.Series) -> float:
        """Extract data age from feature vector, defaulting to 0 if absent."""
        for col in ("data_age_minutes", "age_minutes"):
            if col in features.index and pd.notna(features[col]):
                return float(features[col])
        return 0.0
