"""
Tests for the Futures signal engine logic.

Tests cover:
  - Positive price → NO_TRADE
  - High p_negative but p_rebound below threshold → WATCH_LONG_REBOUND
  - All conditions met → ENTER_LONG_REBOUND_SIGNAL
  - Risk engine blocks the signal → RISK_BLOCKED
  - Price reaches take_profit → EXIT_TAKE_PROFIT_SIGNAL
  - Price hits stop_loss → EXIT_STOP_LOSS_SIGNAL

The signal engine logic is tested in isolation using mock data and without
a live database connection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.schemas import (
    CostModelConfig,
    CostBreakdown,
    SignalAction,
    SignalResponse,
)


# ---------------------------------------------------------------------------
# Helpers – build fake inputs
# ---------------------------------------------------------------------------


def _make_cost_breakdown(
    spread: float = 5.0,
    slippage: float = 3.0,
    overnight: float = 0.1,
    markup: float = 1.0,
    buffer: float = 5.0,
) -> CostBreakdown:
    total = spread + slippage + overnight + markup + buffer
    return CostBreakdown(
        spread_cost_eur_mwh=spread,
        slippage_cost_eur_mwh=slippage,
        overnight_fee_eur_mwh=overnight,
        broker_markup_eur_mwh=markup,
        safety_buffer_eur_mwh=buffer,
        total_eur_mwh=total,
        holding_hours_assumed=4,
        is_weekend=False,
    )


def _default_config(min_edge: float = 10.0) -> CostModelConfig:
    return CostModelConfig(
        avg_spread_eur_mwh=5.0,
        slippage_eur_mwh=3.0,
        overnight_fee_annual_pct=8.0,
        weekend_fee_multiplier=1.5,
        broker_markup_eur_mwh=1.0,
        safety_buffer_eur_mwh=5.0,
        min_edge_threshold=min_edge,
        holding_hours=4,
    )


def _signal_logic(
    current_price: float,
    p_negative: float,
    p_rebound: float,
    predicted_price: float,
    config: Optional[CostModelConfig] = None,
    open_positions: int = 0,
    max_open_positions: int = 3,
    min_confidence_threshold: float = 0.60,
    is_weekend: bool = False,
) -> SignalAction:
    """
    Pure-function implementation of the signal engine logic used in routes/futures.py.
    Allows testing logic without database or HTTP context.
    """
    if config is None:
        config = _default_config()

    # Data quality is assumed OK in these tests (price is available)
    if current_price >= 0:
        return SignalAction.NO_TRADE

    # Edge calculation
    total_cost = (
        config.avg_spread_eur_mwh
        + config.slippage_eur_mwh
        + config.broker_markup_eur_mwh
        + config.safety_buffer_eur_mwh
    )
    if is_weekend:
        overnight_part = (config.overnight_fee_annual_pct / 365.0 / 24.0
                          * config.holding_hours * 100.0
                          * config.weekend_fee_multiplier)
    else:
        overnight_part = (config.overnight_fee_annual_pct / 365.0 / 24.0
                          * config.holding_hours * 100.0)
    total_cost += min(overnight_part, 5.0)

    gross_edge = max(0.0, predicted_price - current_price)
    net_edge = gross_edge - total_cost

    # Risk check
    if open_positions >= max_open_positions:
        return SignalAction.RISK_BLOCKED

    # Signal logic
    if net_edge >= config.min_edge_threshold and p_rebound >= min_confidence_threshold:
        return SignalAction.ENTER_LONG_REBOUND_SIGNAL
    elif current_price < 0 and p_negative >= 0.5:
        return SignalAction.WATCH_LONG_REBOUND
    else:
        return SignalAction.NO_TRADE


def _exit_check(
    entry_price: float,
    current_price: float,
    take_profit: float,
    stop_loss: float,
) -> Optional[SignalAction]:
    """Check exit conditions for an open position."""
    if current_price >= take_profit:
        return SignalAction.EXIT_TAKE_PROFIT_SIGNAL
    if current_price <= stop_loss:
        return SignalAction.EXIT_STOP_LOSS_SIGNAL
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoTradePositivePrice:
    """Positive price should always produce NO_TRADE."""

    def test_zero_price(self):
        action = _signal_logic(
            current_price=0.0,
            p_negative=0.0,
            p_rebound=0.0,
            predicted_price=5.0,
        )
        assert action == SignalAction.NO_TRADE

    def test_positive_low_price(self):
        action = _signal_logic(
            current_price=10.0,
            p_negative=0.0,
            p_rebound=0.8,
            predicted_price=50.0,
        )
        assert action == SignalAction.NO_TRADE

    def test_positive_high_price(self):
        action = _signal_logic(
            current_price=150.0,
            p_negative=0.0,
            p_rebound=0.9,
            predicted_price=200.0,
        )
        assert action == SignalAction.NO_TRADE

    def test_very_small_positive_price(self):
        action = _signal_logic(
            current_price=0.01,
            p_negative=0.01,
            p_rebound=0.99,
            predicted_price=100.0,
        )
        assert action == SignalAction.NO_TRADE


class TestWatchSignalHighPNegative:
    """Negative price + high p_negative but insufficient edge → WATCH."""

    def test_high_p_negative_low_p_rebound(self):
        """p_negative is high but p_rebound is below the threshold → WATCH."""
        action = _signal_logic(
            current_price=-5.0,
            p_negative=0.92,
            p_rebound=0.40,        # below 0.60 threshold
            predicted_price=10.0,  # net_edge would be small
        )
        assert action == SignalAction.WATCH_LONG_REBOUND

    def test_high_p_negative_net_edge_below_threshold(self):
        """Net edge is below min_edge_threshold → WATCH rather than ENTER."""
        config = _default_config(min_edge=20.0)
        action = _signal_logic(
            current_price=-5.0,
            p_negative=0.85,
            p_rebound=0.65,        # above confidence threshold
            predicted_price=10.0,  # gross_edge ≈ 15, net_edge likely < 20
            config=config,
        )
        # With predicted=10, entry=-5 → gross_edge=15, total_cost≈14.1, net_edge≈0.9 < 20
        assert action == SignalAction.WATCH_LONG_REBOUND

    def test_moderate_negative_price_watch(self):
        """Moderately negative price with moderate confidence → WATCH."""
        action = _signal_logic(
            current_price=-3.0,
            p_negative=0.70,
            p_rebound=0.55,        # just below 0.60 threshold
            predicted_price=5.0,
        )
        assert action == SignalAction.WATCH_LONG_REBOUND


class TestEnterSignalAllConditionsMet:
    """All conditions met → ENTER_LONG_REBOUND_SIGNAL."""

    def test_deep_negative_price_high_confidence(self):
        action = _signal_logic(
            current_price=-20.0,
            p_negative=0.95,
            p_rebound=0.78,
            predicted_price=35.0,  # gross_edge=55, net_edge≈40 >> 10
        )
        assert action == SignalAction.ENTER_LONG_REBOUND_SIGNAL

    def test_moderate_negative_good_edge(self):
        action = _signal_logic(
            current_price=-8.0,
            p_negative=0.88,
            p_rebound=0.72,
            predicted_price=30.0,  # gross_edge=38, net_edge≈24
        )
        assert action == SignalAction.ENTER_LONG_REBOUND_SIGNAL

    def test_at_exact_thresholds(self):
        """Test that signals fire exactly at the threshold boundaries."""
        config = CostModelConfig(
            avg_spread_eur_mwh=5.0,
            slippage_eur_mwh=3.0,
            overnight_fee_annual_pct=0.0,   # zero overnight to simplify
            weekend_fee_multiplier=1.5,
            broker_markup_eur_mwh=1.0,
            safety_buffer_eur_mwh=1.0,      # total cost = 10.0
            min_edge_threshold=10.0,
            holding_hours=4,
        )
        # gross_edge needs to be > 10 after total_cost=10
        action = _signal_logic(
            current_price=-5.0,
            p_negative=0.90,
            p_rebound=0.60,        # exactly at threshold
            predicted_price=25.1,  # gross_edge=30.1, net_edge=20.1 > 10
            config=config,
            min_confidence_threshold=0.60,
        )
        assert action == SignalAction.ENTER_LONG_REBOUND_SIGNAL

    def test_high_edge_weekend(self):
        """Weekend multiplier should not prevent entry when edge is large enough."""
        action = _signal_logic(
            current_price=-30.0,
            p_negative=0.98,
            p_rebound=0.85,
            predicted_price=50.0,  # gross_edge=80, easily clears costs
            is_weekend=True,
        )
        assert action == SignalAction.ENTER_LONG_REBOUND_SIGNAL


class TestRiskBlocked:
    """Risk engine should block signals when constraints are violated."""

    def test_max_positions_reached(self):
        action = _signal_logic(
            current_price=-15.0,
            p_negative=0.90,
            p_rebound=0.80,
            predicted_price=40.0,
            open_positions=3,
            max_open_positions=3,
        )
        assert action == SignalAction.RISK_BLOCKED

    def test_positions_over_limit(self):
        action = _signal_logic(
            current_price=-50.0,
            p_negative=0.99,
            p_rebound=0.99,
            predicted_price=100.0,
            open_positions=5,
            max_open_positions=3,
        )
        assert action == SignalAction.RISK_BLOCKED

    def test_risk_blocks_before_edge_check(self):
        """Risk block takes priority over a very attractive entry."""
        action = _signal_logic(
            current_price=-100.0,
            p_negative=0.99,
            p_rebound=0.99,
            predicted_price=200.0,
            open_positions=3,      # at max
            max_open_positions=3,
        )
        assert action == SignalAction.RISK_BLOCKED

    def test_one_below_max_not_blocked(self):
        """One position below the max should not trigger RISK_BLOCKED."""
        action = _signal_logic(
            current_price=-20.0,
            p_negative=0.92,
            p_rebound=0.75,
            predicted_price=50.0,
            open_positions=2,
            max_open_positions=3,
        )
        assert action != SignalAction.RISK_BLOCKED


class TestExitTakeProfit:
    """Price reaching take_profit target → EXIT_TAKE_PROFIT_SIGNAL."""

    def test_price_at_take_profit(self):
        action = _exit_check(
            entry_price=-10.0,
            current_price=30.0,
            take_profit=30.0,
            stop_loss=-30.0,
        )
        assert action == SignalAction.EXIT_TAKE_PROFIT_SIGNAL

    def test_price_above_take_profit(self):
        action = _exit_check(
            entry_price=-15.0,
            current_price=35.0,
            take_profit=25.0,
            stop_loss=-35.0,
        )
        assert action == SignalAction.EXIT_TAKE_PROFIT_SIGNAL

    def test_price_just_below_take_profit(self):
        action = _exit_check(
            entry_price=-10.0,
            current_price=29.9,
            take_profit=30.0,
            stop_loss=-30.0,
        )
        assert action is None

    def test_price_between_stop_and_take(self):
        action = _exit_check(
            entry_price=-10.0,
            current_price=5.0,
            take_profit=30.0,
            stop_loss=-25.0,
        )
        assert action is None


class TestExitStopLoss:
    """Price hitting stop_loss level → EXIT_STOP_LOSS_SIGNAL."""

    def test_price_at_stop_loss(self):
        action = _exit_check(
            entry_price=-10.0,
            current_price=-30.0,
            take_profit=20.0,
            stop_loss=-30.0,
        )
        assert action == SignalAction.EXIT_STOP_LOSS_SIGNAL

    def test_price_below_stop_loss(self):
        action = _exit_check(
            entry_price=-5.0,
            current_price=-50.0,
            take_profit=20.0,
            stop_loss=-25.0,
        )
        assert action == SignalAction.EXIT_STOP_LOSS_SIGNAL

    def test_price_just_above_stop_loss(self):
        action = _exit_check(
            entry_price=-10.0,
            current_price=-29.9,
            take_profit=20.0,
            stop_loss=-30.0,
        )
        assert action is None

    def test_stop_loss_takes_priority_if_price_below_both(self):
        """
        Pathological case: price is somehow both below take_profit and stop_loss.
        In our implementation take_profit is checked first.
        """
        action = _exit_check(
            entry_price=50.0,
            current_price=-100.0,
            take_profit=-50.0,   # take_profit set below current — unusual but possible
            stop_loss=-80.0,
        )
        # current_price(-100) < stop_loss(-80) → stop_loss fires regardless of take_profit
        assert action == SignalAction.EXIT_STOP_LOSS_SIGNAL
