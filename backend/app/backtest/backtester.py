"""
Backtesting Engine for German Electricity Price Futures Strategies.

Implements two strategies:
  - NaiveStrategy : Enter long whenever price < 0, exit after 4 h or price > 0.
  - MLReboundStrategy: Enter only when ML signals approve, apply cost model.

Both strategies produce a list of Trade objects.  The Backtester then
calculates a comprehensive set of performance metrics including:
  - Sharpe ratio (annualised)
  - Sortino ratio (annualised)
  - Max drawdown
  - Profit factor
  - Win rate
  - Equity curve / drawdown curve
  - Monthly performance breakdown
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.futures.cost_model import FuturesCostModel
from app.core.logging import get_logger

logger = get_logger(__name__)

# Annual trading hours (continuous market)
_HOURS_PER_YEAR = 8760.0
_RISK_FREE_HOURLY = 0.02 / _HOURS_PER_YEAR  # 2 % p.a. expressed per hour


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Record of a single completed Futures trade."""

    trade_id: str
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: float
    exit_price: float
    pnl_gross: float          # (exit_price - entry_price) * notional_mwh
    pnl_net: float            # pnl_gross - futures_costs
    futures_costs: float
    holding_hours: float
    exit_reason: str          # take_profit | stop_loss | time_exit | price_positive
    notional_mwh: float = 1.0


# ---------------------------------------------------------------------------
# Metrics container
# ---------------------------------------------------------------------------

@dataclass
class BacktestMetrics:
    """Full performance report for a single backtest run."""

    strategy: str
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    win_rate_pct: float
    avg_trade_eur_mwh: float
    worst_trade_eur_mwh: float
    best_trade_eur_mwh: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    trades_per_month: float
    equity_curve: List[Dict[str, Any]]    # [{timestamp, equity}]
    drawdown_curve: List[Dict[str, Any]]  # [{timestamp, drawdown_pct}]
    monthly_performance: Dict[str, float] # {"YYYY-MM": net_pnl}
    trades: List[Dict[str, Any]]          # serialised Trade records


# ---------------------------------------------------------------------------
# Naive strategy
# ---------------------------------------------------------------------------

class NaiveStrategy:
    """
    Naive baseline: go long whenever price < 0, exit after 4 h or when
    price returns to ≥ 0.  Futures costs are applied to every trade.
    """

    HOLDING_HOURS: int = 4
    NOTIONAL_MWH: float = 1.0

    def generate_trades(
        self,
        df: pd.DataFrame,
        cost_model: FuturesCostModel,
    ) -> List[Trade]:
        """
        Iterate through hourly price data and simulate the naive strategy.

        Args:
            df: DataFrame with at least a ``price_eur_mwh`` column and a
                DatetimeIndex (or ``timestamp`` column).
            cost_model: Used to deduct realistic Futures costs from each trade.

        Returns:
            List of completed Trade records.
        """
        df = _ensure_timestamp_index(df)
        trades: List[Trade] = []

        in_position = False
        entry_price: float = 0.0
        entry_ts: datetime = datetime.now(tz=timezone.utc)
        entry_idx: int = 0

        for i, (ts, row) in enumerate(df.iterrows()):
            price = row.get("price_eur_mwh")
            if price is None or not np.isfinite(price):
                continue

            if not in_position:
                if price < 0:
                    in_position = True
                    entry_price = float(price)
                    entry_ts = _to_utc_datetime(ts)
                    entry_idx = i
            else:
                hours_held = (i - entry_idx)  # 1 row = 1 hour
                exit_trigger = (
                    price >= 0 or hours_held >= self.HOLDING_HOURS
                )
                if exit_trigger:
                    exit_price = float(price)
                    exit_ts = _to_utc_datetime(ts)
                    is_weekend = bool(row.get("is_weekend", False))

                    cost_breakdown = cost_model.calculate_net_edge(
                        expected_rebound_eur_mwh=max(0.0, exit_price - entry_price),
                        estimated_holding_hours=float(hours_held),
                        is_weekend=is_weekend,
                        notional_price_eur_mwh=abs(entry_price) or 100.0,
                    )

                    pnl_gross = (exit_price - entry_price) * self.NOTIONAL_MWH
                    futures_costs = cost_breakdown.total_cost * self.NOTIONAL_MWH
                    pnl_net = pnl_gross - futures_costs

                    exit_reason = "price_positive" if price >= 0 else "time_exit"

                    trades.append(
                        Trade(
                            trade_id=str(uuid.uuid4()),
                            entry_timestamp=entry_ts,
                            exit_timestamp=exit_ts,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            pnl_gross=round(pnl_gross, 4),
                            pnl_net=round(pnl_net, 4),
                            futures_costs=round(futures_costs, 4),
                            holding_hours=float(hours_held),
                            exit_reason=exit_reason,
                            notional_mwh=self.NOTIONAL_MWH,
                        )
                    )
                    in_position = False

        logger.info(
            "NaiveStrategy complete",
            extra={"total_trades": len(trades)},
        )
        return trades


# ---------------------------------------------------------------------------
# ML rebound strategy
# ---------------------------------------------------------------------------

class MLReboundStrategy:
    """
    ML-driven rebound strategy.

    Entry conditions:
      - current price < 0
      - p_rebound >= p_rebound_threshold
      - net_edge >= min_edge_threshold (cost-model check)

    Exit conditions (checked each hour):
      - Take-profit hit
      - Stop-loss hit
      - Maximum holding time exceeded
      - Price ≥ 0 (rebound achieved)
    """

    STOP_LOSS_BUFFER_EUR_MWH: float = 8.0
    TAKE_PROFIT_MULTIPLIER: float = 2.0
    NOTIONAL_MWH: float = 1.0

    def __init__(
        self,
        cost_model: FuturesCostModel,
        p_rebound_threshold: float = 0.60,
        min_edge_threshold: float = 30.0,
        max_holding_hours: int = 6,
    ) -> None:
        self.cost_model = cost_model
        self.p_rebound_threshold = p_rebound_threshold
        self.min_edge_threshold = min_edge_threshold
        self.max_holding_hours = max_holding_hours

    def generate_trades(
        self,
        df: pd.DataFrame,
        predictions_df: pd.DataFrame,
    ) -> List[Trade]:
        """
        Simulate the ML rebound strategy over historical data.

        Args:
            df: Market data with ``price_eur_mwh`` and a DatetimeIndex.
            predictions_df: ML predictions aligned to the same index, with
                columns ``p_negative``, ``p_rebound``, ``predicted_price``.

        Returns:
            List of completed Trade records.
        """
        df = _ensure_timestamp_index(df)
        predictions_df = _ensure_timestamp_index(predictions_df)

        # Align predictions to market data index (forward-fill gaps)
        preds = predictions_df.reindex(df.index, method="ffill")

        trades: List[Trade] = []

        in_position = False
        entry_price: float = 0.0
        entry_ts: datetime = datetime.now(tz=timezone.utc)
        entry_idx: int = 0
        stop_loss: float = 0.0
        take_profit: float = 0.0

        rows = list(df.iterrows())

        for i, (ts, row) in enumerate(rows):
            price = row.get("price_eur_mwh")
            if price is None or not np.isfinite(float(price)):
                continue
            price = float(price)

            if not in_position:
                # --- Entry gate ------------------------------------------
                if price >= 0:
                    continue

                pred_row = preds.iloc[i] if i < len(preds) else None
                if pred_row is None:
                    continue

                p_rebound = float(pred_row.get("p_rebound", 0.0))
                predicted_price = float(pred_row.get("predicted_price", price))

                if p_rebound < self.p_rebound_threshold:
                    continue

                raw_rebound = max(0.0, predicted_price - price)
                expected_rebound = raw_rebound * p_rebound

                is_weekend = bool(row.get("is_weekend", False))
                cost_breakdown = self.cost_model.calculate_net_edge(
                    expected_rebound_eur_mwh=expected_rebound,
                    estimated_holding_hours=float(self.max_holding_hours),
                    is_weekend=is_weekend,
                    notional_price_eur_mwh=abs(price) or 100.0,
                )

                if cost_breakdown.net_edge < self.min_edge_threshold:
                    continue

                # Enter
                in_position = True
                entry_price = price
                entry_ts = _to_utc_datetime(ts)
                entry_idx = i
                stop_loss = entry_price - self.STOP_LOSS_BUFFER_EUR_MWH
                take_profit = entry_price + (
                    cost_breakdown.net_edge * self.TAKE_PROFIT_MULTIPLIER
                )

            else:
                # --- Exit gate -------------------------------------------
                hours_held = i - entry_idx

                exit_reason: Optional[str] = None
                exit_price: float = price

                if price <= stop_loss:
                    exit_reason = "stop_loss"
                elif price >= 0:
                    exit_reason = "price_positive"
                elif price >= take_profit:
                    exit_reason = "take_profit"
                elif hours_held >= self.max_holding_hours:
                    exit_reason = "time_exit"

                if exit_reason is not None:
                    exit_ts = _to_utc_datetime(ts)
                    is_weekend = bool(row.get("is_weekend", False))

                    cost_breakdown = self.cost_model.calculate_net_edge(
                        expected_rebound_eur_mwh=max(0.0, exit_price - entry_price),
                        estimated_holding_hours=float(hours_held),
                        is_weekend=is_weekend,
                        notional_price_eur_mwh=abs(entry_price) or 100.0,
                    )

                    pnl_gross = (exit_price - entry_price) * self.NOTIONAL_MWH
                    futures_costs = cost_breakdown.total_cost * self.NOTIONAL_MWH
                    pnl_net = pnl_gross - futures_costs

                    trades.append(
                        Trade(
                            trade_id=str(uuid.uuid4()),
                            entry_timestamp=entry_ts,
                            exit_timestamp=exit_ts,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            pnl_gross=round(pnl_gross, 4),
                            pnl_net=round(pnl_net, 4),
                            futures_costs=round(futures_costs, 4),
                            holding_hours=float(hours_held),
                            exit_reason=exit_reason,
                            notional_mwh=self.NOTIONAL_MWH,
                        )
                    )
                    in_position = False

        logger.info(
            "MLReboundStrategy complete",
            extra={
                "total_trades": len(trades),
                "p_rebound_threshold": self.p_rebound_threshold,
                "min_edge_threshold": self.min_edge_threshold,
            },
        )
        return trades


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Orchestrates both strategies and computes full performance metrics.

    Usage::

        bt = Backtester(cost_model)
        naive_metrics = bt.run_naive(df)
        ml_metrics    = bt.run_ml_rebound(df, predictions_df)
        comparison    = bt.compare(df, predictions_df)
    """

    def __init__(self, cost_model: FuturesCostModel) -> None:
        self.cost_model = cost_model

    def run_naive(self, df: pd.DataFrame) -> BacktestMetrics:
        """Run the naive strategy and return metrics."""
        strategy = NaiveStrategy()
        trades = strategy.generate_trades(df, self.cost_model)
        return self._calculate_metrics("naive", trades, df)

    def run_ml_rebound(
        self,
        df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        p_rebound_threshold: float = 0.60,
        min_edge_threshold: float = 30.0,
        max_holding_hours: int = 6,
    ) -> BacktestMetrics:
        """Run the ML rebound strategy and return metrics."""
        strategy = MLReboundStrategy(
            cost_model=self.cost_model,
            p_rebound_threshold=p_rebound_threshold,
            min_edge_threshold=min_edge_threshold,
            max_holding_hours=max_holding_hours,
        )
        trades = strategy.generate_trades(df, predictions_df)
        return self._calculate_metrics("ml_rebound", trades, df)

    def compare(
        self,
        df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        p_rebound_threshold: float = 0.60,
        min_edge_threshold: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Run both strategies and return a comparison dict.

        Returns:
            Dict with keys ``naive``, ``ml_rebound``, and ``improvement``.
        """
        naive = self.run_naive(df)
        ml = self.run_ml_rebound(df, predictions_df, p_rebound_threshold, min_edge_threshold)
        improvement = self._calc_improvement(naive, ml)

        logger.info(
            "Backtest comparison complete",
            extra={
                "naive_sharpe": naive.sharpe_ratio,
                "ml_sharpe": ml.sharpe_ratio,
                "naive_total_return": naive.total_return_pct,
                "ml_total_return": ml.total_return_pct,
            },
        )
        return {"naive": naive, "ml_rebound": ml, "improvement": improvement}

    # ------------------------------------------------------------------
    # Metrics calculation
    # ------------------------------------------------------------------

    def _calculate_metrics(
        self,
        strategy: str,
        trades: List[Trade],
        df: pd.DataFrame,
    ) -> BacktestMetrics:
        """Compute the full set of performance metrics from a trade list."""
        df = _ensure_timestamp_index(df)

        if not trades:
            return self._empty_metrics(strategy)

        net_pnls = np.array([t.pnl_net for t in trades])
        winning = [t for t in trades if t.pnl_net > 0]
        losing = [t for t in trades if t.pnl_net <= 0]

        total_pnl = float(net_pnls.sum())
        win_rate = len(winning) / len(trades) * 100.0
        avg_trade = float(net_pnls.mean())
        worst_trade = float(net_pnls.min())
        best_trade = float(net_pnls.max())

        # Equity curve aligned to hourly market data index
        equity_curve_arr, ts_index = self._build_equity_curve(trades, df)
        total_return_pct = (
            (equity_curve_arr[-1] / equity_curve_arr[0] - 1.0) * 100.0
            if equity_curve_arr[0] != 0
            else 0.0
        )

        # Annualised return
        date_range_hours = max(1.0, (ts_index[-1] - ts_index[0]).total_seconds() / 3600.0)
        date_range_years = date_range_hours / _HOURS_PER_YEAR
        annualised_return_pct = (
            ((1.0 + total_return_pct / 100.0) ** (1.0 / max(date_range_years, 1e-6)) - 1.0)
            * 100.0
        )

        # Sharpe / Sortino on per-trade net P&L
        sharpe = self._calc_sharpe(net_pnls)
        sortino = self._calc_sortino(net_pnls)

        # Max drawdown
        max_dd = self._calc_max_drawdown(equity_curve_arr)

        # Drawdown curve
        dd_curve = self._build_drawdown_curve(equity_curve_arr, ts_index)

        # Profit factor
        pf = self._calc_profit_factor(trades)

        # Trades per month
        months = max(1.0, date_range_hours / (24.0 * 30.0))
        trades_per_month = len(trades) / months

        # Monthly performance
        monthly_perf = self._calc_monthly_performance(trades)

        # Equity curve serialised
        equity_serialised = [
            {"timestamp": ts.isoformat(), "equity": round(float(eq), 4)}
            for ts, eq in zip(ts_index, equity_curve_arr)
        ]

        # Serialise trades
        trades_serialised = [
            {
                "trade_id": t.trade_id,
                "entry_timestamp": t.entry_timestamp.isoformat(),
                "exit_timestamp": t.exit_timestamp.isoformat(),
                "entry_price": round(t.entry_price, 4),
                "exit_price": round(t.exit_price, 4),
                "pnl_gross": round(t.pnl_gross, 4),
                "pnl_net": round(t.pnl_net, 4),
                "futures_costs": round(t.futures_costs, 4),
                "holding_hours": round(t.holding_hours, 2),
                "exit_reason": t.exit_reason,
            }
            for t in trades
        ]

        metrics = BacktestMetrics(
            strategy=strategy,
            total_return_pct=round(total_return_pct, 4),
            annualized_return_pct=round(annualised_return_pct, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            max_drawdown_pct=round(max_dd, 4),
            profit_factor=round(pf, 4),
            win_rate_pct=round(win_rate, 4),
            avg_trade_eur_mwh=round(avg_trade, 4),
            worst_trade_eur_mwh=round(worst_trade, 4),
            best_trade_eur_mwh=round(best_trade, 4),
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            trades_per_month=round(trades_per_month, 2),
            equity_curve=equity_serialised,
            drawdown_curve=dd_curve,
            monthly_performance=monthly_perf,
            trades=trades_serialised,
        )

        logger.info(
            "Backtest metrics calculated",
            extra={
                "strategy": strategy,
                "total_trades": len(trades),
                "sharpe": sharpe,
                "sortino": sortino,
                "max_drawdown_pct": max_dd,
                "win_rate_pct": win_rate,
                "total_return_pct": total_return_pct,
            },
        )
        return metrics

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    def _calc_sharpe(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02,
    ) -> float:
        """
        Annualised Sharpe ratio from a series of per-trade returns.

        Args:
            returns: Array of net P&L values (EUR/MWh or EUR) per trade.
            risk_free_rate: Annual risk-free rate (default 2 %).

        Returns:
            Annualised Sharpe ratio, or 0.0 if std dev is zero.
        """
        if len(returns) < 2:
            return 0.0
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        if std_r == 0:
            return 0.0
        # Approximate annualisation: assume ~50 trades/year as scaling
        trades_per_year = max(1, len(returns))
        annualisation = np.sqrt(trades_per_year)
        rf_per_trade = risk_free_rate / trades_per_year
        return float((mean_r - rf_per_trade) / std_r * annualisation)

    def _calc_sortino(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02,
    ) -> float:
        """
        Annualised Sortino ratio (uses downside deviation only).

        Args:
            returns: Array of net P&L values per trade.
            risk_free_rate: Annual risk-free rate (default 2 %).

        Returns:
            Annualised Sortino ratio, or 0.0 if downside std dev is zero.
        """
        if len(returns) < 2:
            return 0.0
        trades_per_year = max(1, len(returns))
        rf_per_trade = risk_free_rate / trades_per_year
        excess = returns - rf_per_trade
        downside = excess[excess < 0]
        if len(downside) == 0:
            return float(np.inf)  # no losing trades
        downside_std = np.std(downside, ddof=1)
        if downside_std == 0:
            return 0.0
        mean_excess = float(np.mean(excess))
        annualisation = np.sqrt(trades_per_year)
        return float(mean_excess / downside_std * annualisation)

    def _calc_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """
        Maximum peak-to-trough drawdown as a percentage.

        Args:
            equity_curve: Array of cumulative equity values.

        Returns:
            Max drawdown as a positive percentage (e.g. 12.5 for −12.5 %).
        """
        if len(equity_curve) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / np.where(peak != 0, peak, 1.0)
        max_dd = float(abs(drawdown.min()) * 100.0)
        return max_dd

    def _calc_profit_factor(self, trades: List[Trade]) -> float:
        """
        Ratio of total gross profit to total gross loss.

        Returns:
            Profit factor ≥ 0.  Returns ``inf`` when there are no losses.
        """
        gross_profit = sum(t.pnl_net for t in trades if t.pnl_net > 0)
        gross_loss = abs(sum(t.pnl_net for t in trades if t.pnl_net < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 1.0
        return round(gross_profit / gross_loss, 4)

    # ------------------------------------------------------------------
    # Curve builders
    # ------------------------------------------------------------------

    def _build_equity_curve(
        self,
        trades: List[Trade],
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, List[datetime]]:
        """
        Build a cumulative equity curve aligned to the market data index.

        Starts at 0.0 (relative P&L).  Each trade's net P&L is added at the
        exit timestamp.

        Returns:
            (equity_array, list_of_datetimes)
        """
        ts_index = [_to_utc_datetime(ts) for ts in df.index]
        equity = np.zeros(len(ts_index))

        # Map trades to the nearest index position by exit timestamp
        trade_map: dict[int, float] = {}
        ts_series = pd.DatetimeIndex(ts_index)

        for trade in trades:
            pos = ts_series.searchsorted(trade.exit_timestamp, side="right") - 1
            pos = int(np.clip(pos, 0, len(ts_index) - 1))
            trade_map[pos] = trade_map.get(pos, 0.0) + trade.pnl_net

        # Build cumulative equity
        for i in range(len(equity)):
            equity[i] = trade_map.get(i, 0.0)
        equity = np.cumsum(equity)

        return equity, ts_index

    def _build_drawdown_curve(
        self,
        equity_curve: np.ndarray,
        ts_index: List[datetime],
    ) -> List[Dict[str, Any]]:
        """Build the drawdown-over-time curve."""
        if len(equity_curve) == 0:
            return []
        peak = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - peak) / np.where(peak != 0, np.abs(peak), 1.0) * 100.0
        return [
            {"timestamp": ts.isoformat(), "drawdown_pct": round(float(d), 4)}
            for ts, d in zip(ts_index, dd)
        ]

    def _calc_monthly_performance(
        self, trades: List[Trade]
    ) -> Dict[str, float]:
        """Aggregate net P&L by calendar month (YYYY-MM)."""
        monthly: Dict[str, float] = {}
        for trade in trades:
            key = trade.exit_timestamp.strftime("%Y-%m")
            monthly[key] = round(monthly.get(key, 0.0) + trade.pnl_net, 4)
        return monthly

    def _calc_improvement(
        self, naive: BacktestMetrics, ml: BacktestMetrics
    ) -> Dict[str, float]:
        """Compute the relative improvement of ML over Naive."""
        def pct_change(a: float, b: float) -> float:
            if a == 0:
                return 0.0
            return round((b - a) / abs(a) * 100.0, 2)

        return {
            "total_return_pct_delta": round(ml.total_return_pct - naive.total_return_pct, 4),
            "sharpe_ratio_delta": round(ml.sharpe_ratio - naive.sharpe_ratio, 4),
            "sortino_ratio_delta": round(ml.sortino_ratio - naive.sortino_ratio, 4),
            "max_drawdown_improvement_pct": round(
                naive.max_drawdown_pct - ml.max_drawdown_pct, 4
            ),
            "win_rate_delta_pct": round(ml.win_rate_pct - naive.win_rate_pct, 4),
            "profit_factor_delta": round(ml.profit_factor - naive.profit_factor, 4),
            "trades_per_month_delta": round(
                ml.trades_per_month - naive.trades_per_month, 2
            ),
            "avg_trade_improvement_pct": pct_change(
                naive.avg_trade_eur_mwh, ml.avg_trade_eur_mwh
            ),
        }

    @staticmethod
    def _empty_metrics(strategy: str) -> BacktestMetrics:
        """Return zeroed-out metrics when there are no trades."""
        return BacktestMetrics(
            strategy=strategy,
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            profit_factor=0.0,
            win_rate_pct=0.0,
            avg_trade_eur_mwh=0.0,
            worst_trade_eur_mwh=0.0,
            best_trade_eur_mwh=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            trades_per_month=0.0,
            equity_curve=[],
            drawdown_curve=[],
            monthly_performance={},
            trades=[],
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ensure_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has a DatetimeIndex.

    Accepts either a DatetimeIndex already or a ``timestamp`` column.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    if "timestamp" in df.columns:
        return df.set_index(pd.DatetimeIndex(pd.to_datetime(df["timestamp"])))
    raise ValueError(
        "DataFrame must have a DatetimeIndex or a 'timestamp' column."
    )


def _to_utc_datetime(ts: Any) -> datetime:
    """Convert various timestamp types to a UTC-aware datetime."""
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
    dt = pd.Timestamp(ts).to_pydatetime()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
