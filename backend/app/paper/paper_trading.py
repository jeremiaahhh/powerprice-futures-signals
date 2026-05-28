"""
Paper Trading Service for Futures Signal Simulation.

IMPORTANT: NO real orders are placed.  This module simulates paper-trading
based on Futures signals and maintains a virtual journal of positions and P&L.

Lifecycle:
  1. Signal pipeline calls ``process_signal(signal)`` whenever a new
     ENTER_LONG_REBOUND_SIGNAL is emitted.
  2. ``check_open_positions(current_price)`` is called each hour to evaluate
     TP / SL / time-exit conditions on all open positions.
  3. All state is persisted to the database via SQLAlchemy AsyncSession.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.futures.cost_model import FuturesCostModel
from app.core.logging import get_logger
from app.db.models import PaperPosition
from app.signals.signal_engine import Signal, SignalAction

logger = get_logger(__name__)

# Default notional size for paper trades
DEFAULT_NOTIONAL_MWH: float = 1.0


# ---------------------------------------------------------------------------
# Paper trading service
# ---------------------------------------------------------------------------

class PaperTradingService:
    """
    Simulates paper trading based on Futures signals.

    All operations are async and persist state to the provided database
    session.  The service tracks:
      - Open positions (status="open")
      - Closed positions (status="closed") with full P&L attribution
      - Running daily P&L (reset each UTC day)
    """

    def __init__(
        self,
        cost_model: FuturesCostModel,
        db: AsyncSession,
        notional_mwh: float = DEFAULT_NOTIONAL_MWH,
    ) -> None:
        self.cost_model = cost_model
        self.db = db
        self.notional_mwh = notional_mwh
        self.is_running: bool = False
        self._daily_pnl_eur: float = 0.0
        self._daily_pnl_date: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Service control
    # ------------------------------------------------------------------

    async def start(self) -> dict:
        """Mark the service as active.  Returns status dict."""
        self.is_running = True
        logger.info("PaperTradingService started")
        return {
            "status": "started",
            "is_running": True,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

    async def stop(self) -> dict:
        """Mark the service as inactive.  Open positions are NOT closed."""
        self.is_running = False
        logger.info("PaperTradingService stopped")
        return {
            "status": "stopped",
            "is_running": False,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

    async def get_status(self) -> dict:
        """Return a snapshot of the service state."""
        open_positions = await self.get_open_positions()
        pnl_summary = await self.get_pnl_summary()
        return {
            "is_running": self.is_running,
            "open_positions_count": len(open_positions),
            "daily_pnl_eur": round(self._daily_pnl_eur, 4),
            "pnl_summary": pnl_summary,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    async def process_signal(self, signal: Signal) -> Optional[PaperPosition]:
        """
        Open a new paper position when the signal is ENTER_LONG_REBOUND_SIGNAL.

        Silently ignores all other signal actions.

        Args:
            signal: The signal emitted by the SignalEngine.

        Returns:
            The newly created PaperPosition, or None if no position was opened.
        """
        if not self.is_running:
            logger.debug("PaperTradingService is not running – signal ignored")
            return None

        if signal.action != SignalAction.ENTER_LONG_REBOUND_SIGNAL:
            logger.debug(
                "Signal action does not trigger entry",
                extra={"action": signal.action},
            )
            return None

        if signal.current_price is None:
            logger.warning("Signal has no current_price – cannot open position")
            return None

        current_time = signal.timestamp or datetime.now(tz=timezone.utc)

        position = PaperPosition(
            status="open",
            entry_price=signal.current_price,
            entry_timestamp=current_time,
            notional_size_mwh=self.notional_mwh,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            max_holding_hours=signal.max_holding_hours,
        )
        self.db.add(position)
        await self.db.flush()  # populate id without committing

        logger.info(
            "Paper position opened",
            extra={
                "position_id": position.id,
                "entry_price": position.entry_price,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "max_holding_hours": position.max_holding_hours,
            },
        )
        return position

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    async def check_open_positions(self, current_price: float) -> List[dict]:
        """
        Evaluate all open positions against current market price.

        Closes any position that has hit its stop-loss, take-profit, time limit,
        or where the price has turned positive.

        Args:
            current_price: The current market price in EUR/MWh.

        Returns:
            List of dicts describing each closed position (with P&L).
        """
        current_time = datetime.now(tz=timezone.utc)
        open_positions = await self.get_open_positions()
        closed: List[dict] = []

        for position in open_positions:
            exit_reason = self._evaluate_exit(position, current_price, current_time)
            if exit_reason is None:
                continue

            closed_pos = await self._close_position(
                position=position,
                exit_price=current_price,
                exit_reason=exit_reason,
                current_time=current_time,
            )

            pnl_gross, futures_costs, net_pnl = self._calculate_pnl(
                entry_price=closed_pos.entry_price,
                exit_price=current_price,
                notional_mwh=closed_pos.notional_size_mwh,
                holding_hours=closed_pos.pnl_eur or 0.0,  # repurposed field
                is_weekend=_is_weekend(current_time),
            )

            # Update daily P&L tracker
            self._update_daily_pnl(net_pnl, current_time)

            closed.append(
                {
                    "position_id": closed_pos.id,
                    "entry_price": closed_pos.entry_price,
                    "exit_price": current_price,
                    "exit_reason": exit_reason,
                    "pnl_gross_eur": round(pnl_gross, 4),
                    "futures_costs_eur": round(futures_costs, 4),
                    "net_pnl_eur": round(net_pnl, 4),
                    "exit_timestamp": current_time.isoformat(),
                }
            )

        if closed:
            logger.info(
                "Positions closed during check",
                extra={"closed_count": len(closed)},
            )

        return closed

    async def get_open_positions(self) -> List[PaperPosition]:
        """Fetch all open paper positions from the database."""
        result = await self.db.execute(
            select(PaperPosition).where(PaperPosition.status == "open")
        )
        return list(result.scalars().all())

    async def get_trade_journal(self, limit: int = 50) -> List[PaperPosition]:
        """
        Fetch the most recent closed positions (trade journal).

        Args:
            limit: Maximum number of records to return (default 50).
        """
        result = await self.db.execute(
            select(PaperPosition)
            .where(PaperPosition.status == "closed")
            .order_by(PaperPosition.exit_timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_pnl_summary(self) -> dict:
        """
        Compute aggregate P&L statistics across all closed positions.

        Returns a dict with total net P&L, win rate, trade count, and best/worst.
        """
        result = await self.db.execute(
            select(PaperPosition).where(PaperPosition.status == "closed")
        )
        closed_positions: List[PaperPosition] = list(result.scalars().all())

        if not closed_positions:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_pct": 0.0,
                "total_net_pnl_eur": 0.0,
                "avg_net_pnl_eur": 0.0,
                "best_trade_eur": 0.0,
                "worst_trade_eur": 0.0,
                "total_futures_costs_eur": 0.0,
                "daily_pnl_eur": round(self._daily_pnl_eur, 4),
            }

        net_pnls = [p.net_pnl_eur for p in closed_positions if p.net_pnl_eur is not None]
        total_costs = sum(
            p.futures_costs_eur for p in closed_positions if p.futures_costs_eur is not None
        )
        winning = [p for p in net_pnls if p > 0]
        losing = [p for p in net_pnls if p <= 0]

        return {
            "total_trades": len(closed_positions),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate_pct": round(
                len(winning) / max(len(net_pnls), 1) * 100.0, 2
            ),
            "total_net_pnl_eur": round(sum(net_pnls), 4),
            "avg_net_pnl_eur": round(
                sum(net_pnls) / max(len(net_pnls), 1), 4
            ),
            "best_trade_eur": round(max(net_pnls, default=0.0), 4),
            "worst_trade_eur": round(min(net_pnls, default=0.0), 4),
            "total_futures_costs_eur": round(total_costs, 4),
            "daily_pnl_eur": round(self._daily_pnl_eur, 4),
        }

    # ------------------------------------------------------------------
    # P&L calculation
    # ------------------------------------------------------------------

    def _calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        notional_mwh: float,
        holding_hours: float,
        is_weekend: bool = False,
    ) -> Tuple[float, float, float]:
        """
        Compute gross P&L, Futures cost deductions, and net P&L.

        For a long Futures position:
            pnl_gross = (exit_price - entry_price) * notional_mwh
            futures_costs = cost_model.total_cost * notional_mwh
            net_pnl   = pnl_gross - futures_costs

        Args:
            entry_price: Position entry price in EUR/MWh.
            exit_price: Position exit price in EUR/MWh.
            notional_mwh: Size of the position in MWh.
            holding_hours: Duration of the trade in hours.
            is_weekend: Whether the position was held over a weekend.

        Returns:
            Tuple (pnl_gross_eur, futures_costs_eur, net_pnl_eur).
        """
        pnl_gross = (exit_price - entry_price) * notional_mwh

        expected_rebound = max(0.0, exit_price - entry_price)
        cost_breakdown = self.cost_model.calculate_net_edge(
            expected_rebound_eur_mwh=expected_rebound,
            estimated_holding_hours=max(holding_hours, 1.0),
            is_weekend=is_weekend,
            notional_price_eur_mwh=abs(entry_price) or 100.0,
        )
        futures_costs = cost_breakdown.total_cost * notional_mwh
        net_pnl = pnl_gross - futures_costs

        return (
            round(pnl_gross, 4),
            round(futures_costs, 4),
            round(net_pnl, 4),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _close_position(
        self,
        position: PaperPosition,
        exit_price: float,
        exit_reason: str,
        current_time: datetime,
    ) -> PaperPosition:
        """
        Close a paper position: compute P&L and persist to the database.

        Args:
            position: The open PaperPosition ORM object.
            exit_price: Exit price in EUR/MWh.
            exit_reason: Human-readable exit trigger (stop_loss, take_profit, etc.).
            current_time: Current UTC datetime.

        Returns:
            The updated (closed) PaperPosition object.
        """
        holding_hours = (
            (current_time - position.entry_timestamp).total_seconds() / 3600.0
        )
        is_weekend = _is_weekend(current_time)

        pnl_gross, futures_costs, net_pnl = self._calculate_pnl(
            entry_price=position.entry_price,
            exit_price=exit_price,
            notional_mwh=position.notional_size_mwh,
            holding_hours=holding_hours,
            is_weekend=is_weekend,
        )

        position.status = "closed"
        position.exit_price = exit_price
        position.exit_timestamp = current_time
        position.pnl_eur = pnl_gross
        position.futures_costs_eur = futures_costs
        position.net_pnl_eur = net_pnl
        position.exit_reason = exit_reason
        position.updated_at = current_time

        await self.db.flush()

        logger.info(
            "Paper position closed",
            extra={
                "position_id": position.id,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "pnl_gross": pnl_gross,
                "futures_costs": futures_costs,
                "net_pnl": net_pnl,
                "holding_hours": round(holding_hours, 2),
            },
        )
        return position

    def _evaluate_exit(
        self,
        position: PaperPosition,
        current_price: float,
        current_time: datetime,
    ) -> Optional[str]:
        """
        Determine if a position should be closed and why.

        Evaluation order (same priority as SignalEngine):
          1. Stop-loss
          2. Price positive (rebound achieved)
          3. Take-profit
          4. Max holding time

        Returns:
            Exit reason string, or None if the position should remain open.
        """
        # 1. Stop-loss
        if position.stop_loss is not None and current_price <= position.stop_loss:
            return "stop_loss"

        # 2. Price turned positive
        if current_price >= 0:
            return "price_positive"

        # 3. Take-profit
        if position.take_profit is not None and current_price >= position.take_profit:
            return "take_profit"

        # 4. Time exit
        if position.max_holding_hours is not None:
            holding_hours = (
                current_time - position.entry_timestamp
            ).total_seconds() / 3600.0
            if holding_hours >= position.max_holding_hours:
                return "time_exit"

        return None

    def _update_daily_pnl(self, net_pnl: float, current_time: datetime) -> None:
        """Reset daily P&L counter on UTC day roll; accumulate otherwise."""
        today = current_time.date()
        if self._daily_pnl_date is None or self._daily_pnl_date != today:
            self._daily_pnl_eur = 0.0
            self._daily_pnl_date = today
        self._daily_pnl_eur += net_pnl


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _is_weekend(dt: datetime) -> bool:
    """Return True if *dt* falls on a Saturday (5) or Sunday (6)."""
    return dt.weekday() >= 5
