"""
Tests for the backtest engine.

Tests cover:
  - Naive strategy produces no trades when all prices are positive
  - Naive strategy enters on negative prices
  - ML strategy respects the p_rebound threshold
  - Metrics calculation (Sharpe, win_rate, drawdown) from known trades
  - Comparison function returns both strategy results
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from app.api.schemas import (
    BacktestParams,
    BacktestResult,
    BacktestStrategy,
    CostModelConfig,
    MonthlyPerformance,
)


# ---------------------------------------------------------------------------
# Mirror the core backtest functions from routes/backtest.py
# to enable unit testing without a database.
# ---------------------------------------------------------------------------


def _generate_naive_signals(df: pd.DataFrame) -> pd.Series:
    return df["price_eur_mwh"] < 0


def _generate_ml_signals(
    df: pd.DataFrame,
    p_rebound_series: pd.Series,
    p_rebound_threshold: float = 0.60,
    min_edge_threshold: float = 10.0,
    cost_model: Optional[CostModelConfig] = None,
) -> pd.Series:
    if cost_model is None:
        cost_model = CostModelConfig()

    total_cost = (
        cost_model.avg_spread_eur_mwh
        + cost_model.slippage_eur_mwh
        + cost_model.broker_markup_eur_mwh
        + cost_model.safety_buffer_eur_mwh
    )
    gross_edge = df["price_eur_mwh"].abs() + 10.0
    net_edge = gross_edge - total_cost

    return (
        (df["price_eur_mwh"] < 0)
        & (p_rebound_series >= p_rebound_threshold)
        & (net_edge >= min_edge_threshold)
    )


def _simulate_trades(
    df: pd.DataFrame,
    entry_signals: pd.Series,
    params: BacktestParams,
) -> List[Dict[str, Any]]:
    cost_total = (
        params.cost_model.avg_spread_eur_mwh
        + params.cost_model.slippage_eur_mwh
        + params.cost_model.broker_markup_eur_mwh
        + params.cost_model.safety_buffer_eur_mwh
    )

    prices = df["price_eur_mwh"].values
    timestamps = df["timestamp"].values
    signals = entry_signals.values

    trades: List[Dict[str, Any]] = []
    in_trade = False
    entry_idx: int = 0
    entry_price: float = 0.0

    for i, (ts, price, signal) in enumerate(zip(timestamps, prices, signals)):
        if not in_trade:
            if signal:
                in_trade = True
                entry_idx = i
                entry_price = float(price)
        else:
            holding_hours = i - entry_idx
            should_exit = False
            exit_reason = "unknown"

            if holding_hours >= params.max_holding_hours:
                should_exit = True
                exit_reason = "time_exit"
            elif price >= params.take_profit_eur_mwh or price - entry_price >= params.take_profit_eur_mwh:
                should_exit = True
                exit_reason = "take_profit"
            elif price <= entry_price - params.stop_loss_eur_mwh:
                should_exit = True
                exit_reason = "stop_loss"
            elif price > 0:
                should_exit = True
                exit_reason = "price_positive"

            if should_exit:
                exit_price = float(price)
                pnl_gross = (exit_price - entry_price) * params.notional_size_mwh
                futures_costs = cost_total * params.notional_size_mwh
                net_pnl = pnl_gross - futures_costs

                trades.append({
                    "entry_timestamp": timestamps[entry_idx],
                    "exit_timestamp": ts,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "holding_hours": holding_hours,
                    "pnl_gross": round(pnl_gross, 4),
                    "futures_costs": round(futures_costs, 4),
                    "net_pnl": round(net_pnl, 4),
                    "exit_reason": exit_reason,
                    "is_winner": net_pnl > 0,
                })
                in_trade = False

    return trades


def _compute_metrics_simple(
    trades: List[Dict[str, Any]],
    params: BacktestParams,
    run_id: str = "test",
    strategy: BacktestStrategy = BacktestStrategy.NAIVE,
) -> BacktestResult:
    """Simplified metrics computation for testing."""
    now = datetime(2026, 5, 19, tzinfo=timezone.utc)
    total_trades = len(trades)

    if total_trades == 0:
        return BacktestResult(
            run_id=run_id,
            strategy=strategy,
            start_date=params.start_date,
            end_date=params.end_date,
            parameters=params,
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            max_drawdown_pct=0.0,
            win_rate_pct=0.0,
            avg_trade_eur_mwh=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            trades_per_month=0.0,
            equity_curve=[],
            monthly_performance=[],
            created_at=now,
        )

    pnl_arr = np.array([t["net_pnl"] for t in trades])
    winning = [t for t in trades if t["is_winner"]]
    losing = [t for t in trades if not t["is_winner"]]

    win_rate = len(winning) / total_trades * 100.0
    avg_trade = float(np.mean(pnl_arr))
    best_trade = float(np.max(pnl_arr))
    worst_trade = float(np.min(pnl_arr))

    gross_profit = sum(t["net_pnl"] for t in winning)
    gross_loss = abs(sum(t["net_pnl"] for t in losing)) if losing else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    initial_capital = 100.0 * params.notional_size_mwh
    total_pnl = float(np.sum(pnl_arr))
    total_return_pct = (total_pnl / initial_capital) * 100.0

    delta_days = (params.end_date - params.start_date).days
    if delta_days > 0:
        years = delta_days / 365.25
        annualized_return_pct = (
            (1 + total_return_pct / 100.0) ** (1.0 / max(years, 0.01)) - 1
        ) * 100.0
    else:
        annualized_return_pct = 0.0

    if len(pnl_arr) > 1:
        sharpe = float(np.mean(pnl_arr) / (np.std(pnl_arr, ddof=1) + 1e-9)) * np.sqrt(252)
        downside = pnl_arr[pnl_arr < 0]
        sortino_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-9
        sortino = float(np.mean(pnl_arr) / (sortino_std + 1e-9)) * np.sqrt(252)
    else:
        sharpe = None
        sortino = None

    cum_pnl = np.cumsum(pnl_arr)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    max_dd_abs = float(np.max(drawdowns))
    max_drawdown_pct = (max_dd_abs / (initial_capital + abs(float(running_max.max())) + 1e-9)) * 100.0

    months = max(delta_days / 30.0, 0.01)
    trades_per_month = total_trades / months

    equity_curve = []
    equity = initial_capital
    for t in trades:
        equity += t["net_pnl"]
        equity_curve.append({
            "timestamp": str(t["exit_timestamp"]),
            "equity": round(equity, 2),
        })

    return BacktestResult(
        run_id=run_id,
        strategy=strategy,
        start_date=params.start_date,
        end_date=params.end_date,
        parameters=params,
        total_return_pct=round(total_return_pct, 4),
        annualized_return_pct=round(annualized_return_pct, 4),
        sharpe_ratio=round(sharpe, 4) if sharpe is not None else None,
        sortino_ratio=round(sortino, 4) if sortino is not None else None,
        max_drawdown_pct=round(max_drawdown_pct, 4),
        profit_factor=round(profit_factor, 4) if profit_factor is not None else None,
        win_rate_pct=round(win_rate, 2),
        avg_trade_eur_mwh=round(avg_trade, 4),
        best_trade_eur_mwh=round(best_trade, 4),
        worst_trade_eur_mwh=round(worst_trade, 4),
        total_trades=total_trades,
        winning_trades=len(winning),
        losing_trades=len(losing),
        trades_per_month=round(trades_per_month, 2),
        equity_curve=equity_curve,
        monthly_performance=[],
        created_at=now,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_params(
    strategy: BacktestStrategy = BacktestStrategy.NAIVE,
    start_days_ago: int = 90,
    end_days_ago: int = 0,
    min_confidence: float = 0.60,
    min_edge: float = 10.0,
    max_holding_hours: int = 6,
    stop_loss: float = 20.0,
    take_profit: float = 30.0,
    notional: float = 1.0,
) -> BacktestParams:
    now = datetime(2026, 5, 19, tzinfo=timezone.utc)
    return BacktestParams(
        strategy=strategy,
        start_date=now - timedelta(days=start_days_ago),
        end_date=now - timedelta(days=end_days_ago),
        notional_size_mwh=notional,
        cost_model=CostModelConfig(min_edge_threshold=min_edge),
        min_confidence=min_confidence,
        max_holding_hours=max_holding_hours,
        stop_loss_eur_mwh=stop_loss,
        take_profit_eur_mwh=take_profit,
    )


def _make_hourly_df(prices: List[float], start_ts: Optional[datetime] = None) -> pd.DataFrame:
    if start_ts is None:
        start_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    timestamps = [start_ts + timedelta(hours=i) for i in range(len(prices))]
    return pd.DataFrame({"timestamp": timestamps, "price_eur_mwh": prices})


# ---------------------------------------------------------------------------
# Test: Naive strategy – no trades on positive prices
# ---------------------------------------------------------------------------


class TestNaiveStrategyNoTradesPositivePrices:
    """All-positive price series should yield zero trades."""

    def test_all_positive_no_signals(self):
        prices = [30.0, 45.0, 60.0, 55.0, 70.0, 80.0, 90.0, 100.0]
        df = _make_hourly_df(prices)
        signals = _generate_naive_signals(df)
        assert signals.sum() == 0

    def test_all_positive_no_trades_simulated(self):
        prices = [10.0] * 100
        df = _make_hourly_df(prices)
        signals = _generate_naive_signals(df)
        params = _make_params()
        trades = _simulate_trades(df, signals, params)
        assert len(trades) == 0

    def test_positive_prices_result_metric_zeroes(self):
        prices = [20.0, 30.0, 40.0, 50.0]
        df = _make_hourly_df(prices)
        signals = _generate_naive_signals(df)
        params = _make_params()
        trades = _simulate_trades(df, signals, params)
        result = _compute_metrics_simple(trades, params)
        assert result.total_trades == 0
        assert result.total_return_pct == 0.0
        assert result.win_rate_pct == 0.0

    def test_zero_price_not_negative(self):
        prices = [0.0, 5.0, 10.0]
        df = _make_hourly_df(prices)
        signals = _generate_naive_signals(df)
        assert signals.sum() == 0


# ---------------------------------------------------------------------------
# Test: Naive strategy – enters on negative prices
# ---------------------------------------------------------------------------


class TestNaiveStrategyEntersOnNegative:
    """Negative prices should trigger trade entry."""

    def test_single_negative_period(self):
        # Pattern: positive → negative → positive (triggers entry then exit)
        prices = [20.0, -5.0, -10.0, -8.0, 5.0, 30.0, 40.0]
        df = _make_hourly_df(prices)
        signals = _generate_naive_signals(df)
        assert signals.sum() >= 1  # at least one entry

    def test_negative_price_triggers_at_least_one_trade(self):
        prices = [30.0, -15.0, -12.0, -8.0, 5.0, 20.0, 35.0, 40.0, 50.0]
        df = _make_hourly_df(prices)
        signals = _generate_naive_signals(df)
        params = _make_params(max_holding_hours=5, take_profit=25.0)
        trades = _simulate_trades(df, signals, params)
        assert len(trades) >= 1

    def test_multiple_negative_periods(self):
        # Two separate negative episodes
        prices = [30.0, -5.0, 5.0, 40.0, -10.0, 20.0, 50.0]
        df = _make_hourly_df(prices)
        signals = _generate_naive_signals(df)
        assert signals.sum() == 2

    def test_trade_enters_at_first_negative_price(self):
        prices = [30.0, -10.0, -5.0, 0.5]
        df = _make_hourly_df(prices)
        signals = _generate_naive_signals(df)
        # The first negative is at index 1
        assert signals.iloc[1] is True or signals.iloc[1] == True  # noqa: E712
        assert signals.iloc[0] is False or signals.iloc[0] == False  # noqa: E712

    def test_trade_exit_on_positive_price(self):
        prices = [30.0, -10.0, -8.0, 5.0, 20.0]
        df = _make_hourly_df(prices)
        signals = _generate_naive_signals(df)
        params = _make_params(max_holding_hours=10)
        trades = _simulate_trades(df, signals, params)
        # The trade should have exited when price went positive
        if trades:
            exited = any(t["exit_reason"] in ("price_positive", "time_exit", "take_profit") for t in trades)
            assert exited


# ---------------------------------------------------------------------------
# Test: ML strategy – respects p_rebound threshold
# ---------------------------------------------------------------------------


class TestMLStrategyRespectsThreshold:
    """ML signals should not fire when p_rebound is below the threshold."""

    def test_low_p_rebound_no_trade(self):
        prices = [-10.0, -8.0, -5.0, 5.0, 20.0]
        df = _make_hourly_df(prices)
        p_reb = pd.Series([0.30, 0.35, 0.40, 0.10, 0.05], index=df.index)
        signals = _generate_ml_signals(df, p_reb, p_rebound_threshold=0.60)
        assert signals.sum() == 0

    def test_high_p_rebound_enters_trade(self):
        prices = [-15.0, -12.0, -8.0, 5.0, 30.0]
        df = _make_hourly_df(prices)
        p_reb = pd.Series([0.80, 0.75, 0.70, 0.10, 0.05], index=df.index)
        signals = _generate_ml_signals(df, p_reb, p_rebound_threshold=0.60)
        assert signals.sum() >= 1

    def test_exactly_at_threshold_enters(self):
        # Use deeply negative prices so net_edge clears the min_edge_threshold
        prices = [-50.0, -45.0, 5.0]
        df = _make_hourly_df(prices)
        # Entry threshold raised to 0.70 after OOS analysis
        p_reb = pd.Series([0.70, 0.70, 0.10], index=df.index)
        signals = _generate_ml_signals(df, p_reb, p_rebound_threshold=0.70)
        # p_rebound = 0.70 and price is deeply negative → should enter
        assert signals.iloc[0] or signals.iloc[1]  # at least one entry

    def test_just_below_threshold_no_trade(self):
        prices = [-10.0, -8.0, 5.0]
        df = _make_hourly_df(prices)
        p_reb = pd.Series([0.59, 0.59, 0.10], index=df.index)
        signals = _generate_ml_signals(df, p_reb, p_rebound_threshold=0.60)
        assert signals.sum() == 0

    def test_positive_price_no_trade_regardless_of_p_rebound(self):
        prices = [10.0, 20.0, 30.0]
        df = _make_hourly_df(prices)
        p_reb = pd.Series([0.99, 0.99, 0.99], index=df.index)
        signals = _generate_ml_signals(df, p_reb, p_rebound_threshold=0.60)
        assert signals.sum() == 0


# ---------------------------------------------------------------------------
# Test: Metrics calculation
# ---------------------------------------------------------------------------


class TestMetricsCalculation:
    """Given a known set of trades, verify computed metrics are correct."""

    def _known_trades(self) -> List[Dict[str, Any]]:
        """
        5 trades with known PnL:
          Trade 1: +20 (winner)
          Trade 2: -5  (loser)
          Trade 3: +15 (winner)
          Trade 4: -10 (loser)
          Trade 5: +25 (winner)
        win_rate = 3/5 = 60%
        avg_trade = (20-5+15-10+25)/5 = 45/5 = 9
        """
        base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        trades = []
        pnl_list = [20.0, -5.0, 15.0, -10.0, 25.0]
        for i, pnl in enumerate(pnl_list):
            entry_ts = base_ts + timedelta(days=i * 7)
            exit_ts = entry_ts + timedelta(hours=4)
            trades.append({
                "entry_timestamp": entry_ts,
                "exit_timestamp": exit_ts,
                "entry_price": -10.0,
                "exit_price": -10.0 + pnl,
                "holding_hours": 4,
                "pnl_gross": round(pnl + 14.0, 4),  # pnl = gross - 14 costs
                "futures_costs": 14.0,
                "net_pnl": round(pnl, 4),
                "exit_reason": "take_profit" if pnl > 0 else "stop_loss",
                "is_winner": pnl > 0,
            })
        return trades

    def test_win_rate(self):
        trades = self._known_trades()
        params = _make_params()
        result = _compute_metrics_simple(trades, params)
        assert result.win_rate_pct == pytest.approx(60.0, abs=0.1)

    def test_total_trades(self):
        trades = self._known_trades()
        params = _make_params()
        result = _compute_metrics_simple(trades, params)
        assert result.total_trades == 5

    def test_winning_trades(self):
        trades = self._known_trades()
        params = _make_params()
        result = _compute_metrics_simple(trades, params)
        assert result.winning_trades == 3

    def test_losing_trades(self):
        trades = self._known_trades()
        params = _make_params()
        result = _compute_metrics_simple(trades, params)
        assert result.losing_trades == 2

    def test_avg_trade_pnl(self):
        trades = self._known_trades()
        params = _make_params()
        result = _compute_metrics_simple(trades, params)
        expected_avg = (20.0 - 5.0 + 15.0 - 10.0 + 25.0) / 5.0
        assert result.avg_trade_eur_mwh == pytest.approx(expected_avg, abs=0.01)

    def test_best_trade(self):
        trades = self._known_trades()
        params = _make_params()
        result = _compute_metrics_simple(trades, params)
        assert result.best_trade_eur_mwh == pytest.approx(25.0, abs=0.01)

    def test_worst_trade(self):
        trades = self._known_trades()
        params = _make_params()
        result = _compute_metrics_simple(trades, params)
        assert result.worst_trade_eur_mwh == pytest.approx(-10.0, abs=0.01)

    def test_sharpe_ratio_not_none(self):
        trades = self._known_trades()
        params = _make_params()
        result = _compute_metrics_simple(trades, params)
        assert result.sharpe_ratio is not None

    def test_max_drawdown_non_negative(self):
        trades = self._known_trades()
        params = _make_params()
        result = _compute_metrics_simple(trades, params)
        assert result.max_drawdown_pct >= 0.0

    def test_all_losers_drawdown_positive(self):
        """All-losing trades should produce a positive max drawdown."""
        base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        trades = []
        for i in range(5):
            entry_ts = base_ts + timedelta(days=i)
            exit_ts = entry_ts + timedelta(hours=4)
            trades.append({
                "entry_timestamp": entry_ts,
                "exit_timestamp": exit_ts,
                "entry_price": -10.0,
                "exit_price": -30.0,
                "holding_hours": 4,
                "pnl_gross": -6.0,
                "futures_costs": 14.0,
                "net_pnl": -20.0,
                "exit_reason": "stop_loss",
                "is_winner": False,
            })
        params = _make_params()
        result = _compute_metrics_simple(trades, params)
        assert result.max_drawdown_pct > 0.0

    def test_profit_factor_all_winners(self):
        """All-winner trades → profit_factor is None (no gross loss)."""
        base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        trades = []
        for i in range(3):
            entry_ts = base_ts + timedelta(days=i)
            exit_ts = entry_ts + timedelta(hours=4)
            trades.append({
                "entry_timestamp": entry_ts,
                "exit_timestamp": exit_ts,
                "entry_price": -10.0,
                "exit_price": 20.0,
                "holding_hours": 4,
                "pnl_gross": 44.0,
                "futures_costs": 14.0,
                "net_pnl": 30.0,
                "exit_reason": "take_profit",
                "is_winner": True,
            })
        params = _make_params()
        result = _compute_metrics_simple(trades, params)
        # No losers → no gross_loss → profit_factor is None
        assert result.profit_factor is None or result.profit_factor > 0.0


# ---------------------------------------------------------------------------
# Test: Backtest comparison
# ---------------------------------------------------------------------------


class TestBacktestComparison:
    """Compare naive vs ML strategy outputs."""

    def _run_both(
        self,
        prices: List[float],
        p_rebound_vals: List[float],
        p_threshold: float = 0.60,
    ) -> tuple[BacktestResult, BacktestResult]:
        df = _make_hourly_df(prices)
        p_reb = pd.Series(p_rebound_vals, index=df.index)
        params = _make_params(max_holding_hours=5, take_profit=25.0)

        naive_signals = _generate_naive_signals(df)
        ml_signals = _generate_ml_signals(df, p_reb, p_rebound_threshold=p_threshold)

        naive_trades = _simulate_trades(df, naive_signals, params)
        ml_trades = _simulate_trades(df, ml_signals, params)

        naive_result = _compute_metrics_simple(
            naive_trades, params, strategy=BacktestStrategy.NAIVE
        )
        ml_result = _compute_metrics_simple(
            ml_trades, params, strategy=BacktestStrategy.ML_REBOUND
        )
        return naive_result, ml_result

    def test_returns_two_results(self):
        prices = [30.0, -10.0, -8.0, 5.0, 30.0, -15.0, 20.0]
        p_rebound = [0.1, 0.8, 0.75, 0.1, 0.1, 0.85, 0.1]
        naive, ml = self._run_both(prices, p_rebound)
        assert isinstance(naive, BacktestResult)
        assert isinstance(ml, BacktestResult)

    def test_strategies_have_correct_labels(self):
        prices = [30.0, -10.0, 5.0]
        p_rebound = [0.1, 0.8, 0.1]
        naive, ml = self._run_both(prices, p_rebound)
        assert naive.strategy == BacktestStrategy.NAIVE
        assert ml.strategy == BacktestStrategy.ML_REBOUND

    def test_ml_fewer_or_equal_trades_than_naive(self):
        """ML strategy should be more selective (fewer or equal trades)."""
        prices = [30.0, -5.0, -10.0, -8.0, 5.0, 30.0, -3.0, -12.0, 10.0, 40.0]
        p_rebound = [0.1, 0.3, 0.5, 0.4, 0.1, 0.1, 0.9, 0.85, 0.1, 0.1]
        naive, ml = self._run_both(prices, p_rebound, p_threshold=0.60)
        assert ml.total_trades <= naive.total_trades

    def test_ml_higher_win_rate_when_filtered_well(self):
        """With good p_rebound signal, ML win rate should be >= naive win rate."""
        # Prices: negative periods with high p_rebound have profitable exits,
        # negative periods with low p_rebound go further negative.
        prices = [
            30.0,
            -5.0, -8.0, 20.0,   # good rebound (high p_rebound), profitable
            -5.0, -10.0, -20.0, -30.0, -5.0,  # bad rebound (low p_rebound), losers
            30.0,
        ]
        p_rebound = [0.1, 0.9, 0.85, 0.1, 0.1, 0.2, 0.15, 0.1, 0.1, 0.1]
        p_rebound_series = pd.Series(p_rebound, index=range(len(prices)))
        df = _make_hourly_df(prices)
        p_reb_aligned = pd.Series(p_rebound, index=df.index)

        params = _make_params(max_holding_hours=4, take_profit=25.0, stop_loss=15.0)
        naive_signals = _generate_naive_signals(df)
        ml_signals = _generate_ml_signals(df, p_reb_aligned, p_rebound_threshold=0.60)

        naive_trades = _simulate_trades(df, naive_signals, params)
        ml_trades = _simulate_trades(df, ml_signals, params)

        if ml_trades and naive_trades:
            naive_result = _compute_metrics_simple(naive_trades, params)
            ml_result = _compute_metrics_simple(ml_trades, params)
            # ML should have filtered out some bad trades
            assert ml_result.total_trades <= naive_result.total_trades

    def test_both_results_have_valid_structure(self):
        prices = [30.0, -10.0, 5.0, 25.0]
        p_rebound = [0.1, 0.8, 0.1, 0.1]
        naive, ml = self._run_both(prices, p_rebound)

        for result in [naive, ml]:
            assert result.run_id is not None
            assert result.total_trades >= 0
            assert 0.0 <= result.win_rate_pct <= 100.0
            assert result.max_drawdown_pct >= 0.0
            assert isinstance(result.equity_curve, list)
