"""
Paper trading routes.

POST /paper/start      – start a paper trading session
POST /paper/stop       – stop a paper trading session
GET  /paper/status     – running status, total PnL, position count
GET  /paper/positions  – open positions
GET  /paper/trades     – trade journal (last 50)
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import get_db
from app.db.models import PaperPosition, FuturesSignal
from app.api.schemas import (
    CostModelConfig,
    PaperPositionResponse,
    PaperStatusResponse,
    PaperTradeRequest,
    PaperTradeResponse,
    ClosePositionRequest,
    PositionStatus,
    SignalAction,
)

logger = get_logger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level session state (in-process; use Redis for multi-process setups)
# ---------------------------------------------------------------------------

_session_active: bool = False
_session_started_at: Optional[datetime] = None
_session_stopped_at: Optional[datetime] = None
_session_lock = threading.Lock()


def _is_session_active() -> bool:
    with _session_lock:
        return _session_active


def _set_session(active: bool) -> None:
    global _session_active, _session_started_at, _session_stopped_at
    with _session_lock:
        _session_active = active
        if active:
            _session_started_at = datetime.now(timezone.utc)
            _session_stopped_at = None
        else:
            _session_stopped_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _position_to_response(pos: PaperPosition) -> PaperPositionResponse:
    now = datetime.now(timezone.utc)

    entry_ts = pos.entry_timestamp
    if entry_ts is not None and entry_ts.tzinfo is None:
        entry_ts = entry_ts.replace(tzinfo=timezone.utc)

    exit_ts = pos.exit_timestamp
    if exit_ts is not None and exit_ts.tzinfo is None:
        exit_ts = exit_ts.replace(tzinfo=timezone.utc)

    updated_ts = pos.updated_at or pos.created_at or now
    if updated_ts.tzinfo is None:
        updated_ts = updated_ts.replace(tzinfo=timezone.utc)

    created_ts = pos.created_at or now
    if created_ts.tzinfo is None:
        created_ts = created_ts.replace(tzinfo=timezone.utc)

    if pos.status == "open" and entry_ts is not None:
        holding_hours = (now - entry_ts).total_seconds() / 3600.0
    elif exit_ts is not None and entry_ts is not None:
        holding_hours = (exit_ts - entry_ts).total_seconds() / 3600.0
    else:
        holding_hours = None

    return PaperPositionResponse(
        id=pos.id,
        signal_id=pos.signal_id,
        status=PositionStatus(pos.status),
        entry_price=pos.entry_price,
        exit_price=pos.exit_price,
        entry_timestamp=entry_ts,
        exit_timestamp=exit_ts,
        notional_size_mwh=pos.notional_size_mwh,
        stop_loss=pos.stop_loss,
        take_profit=pos.take_profit,
        max_holding_hours=pos.max_holding_hours,
        pnl_eur=pos.pnl_eur,
        futures_costs_eur=pos.futures_costs_eur,
        net_pnl_eur=pos.net_pnl_eur,
        exit_reason=pos.exit_reason,
        holding_hours=round(holding_hours, 2) if holding_hours is not None else None,
        created_at=created_ts,
        updated_at=updated_ts,
    )


def _calculate_pnl(
    entry_price: float,
    exit_price: float,
    notional_size_mwh: float,
    cost_model: Optional[CostModelConfig] = None,
) -> tuple[float, float, float]:
    """Return (pnl_gross, futures_costs, net_pnl)."""
    if cost_model is None:
        cost_model = CostModelConfig()

    pnl_gross = (exit_price - entry_price) * notional_size_mwh

    # Total estimated Futures costs
    total_cost_per_mwh = (
        cost_model.avg_spread_eur_mwh
        + cost_model.slippage_eur_mwh
        + cost_model.broker_markup_eur_mwh
        + cost_model.safety_buffer_eur_mwh
    )
    futures_costs = total_cost_per_mwh * notional_size_mwh
    net_pnl = pnl_gross - futures_costs

    return round(pnl_gross, 4), round(futures_costs, 4), round(net_pnl, 4)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/start", summary="Start paper trading session")
async def start_paper_trading() -> Dict[str, Any]:
    """
    Start the paper trading session.

    When active, the system will track signals and simulate position entries/exits
    without placing any live orders.
    """
    if _is_session_active():
        raise HTTPException(status_code=400, detail="Paper trading session is already running.")

    _set_session(True)
    logger.info("Paper trading session started")

    return {
        "status": "started",
        "started_at": _session_started_at.isoformat(),
        "message": "Paper trading session is now active. SIGNAL ONLY — no live orders.",
    }


@router.post("/stop", summary="Stop paper trading session")
async def stop_paper_trading(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """
    Stop the paper trading session.

    Open positions are left open in the database for record-keeping; they are
    not automatically closed. Use POST /paper/positions/{id}/close to close them.
    """
    if not _is_session_active():
        raise HTTPException(status_code=400, detail="No active paper trading session.")

    _set_session(False)
    logger.info("Paper trading session stopped")

    # Count open positions
    stmt = select(func.count()).select_from(PaperPosition).where(PaperPosition.status == "open")
    result = await db.execute(stmt)
    open_count = result.scalar_one()

    return {
        "status": "stopped",
        "stopped_at": _session_stopped_at.isoformat(),
        "open_positions_remaining": open_count,
        "message": f"Session stopped. {open_count} open position(s) remain in the database.",
    }


@router.get(
    "/status",
    response_model=PaperStatusResponse,
    summary="Paper trading session status and PnL summary",
)
async def get_paper_status(db: AsyncSession = Depends(get_db)) -> PaperStatusResponse:
    """
    Return the current paper trading session status and aggregate PnL metrics.
    """
    now = datetime.now(timezone.utc)

    # Fetch all positions
    open_stmt = select(PaperPosition).where(PaperPosition.status == "open")
    closed_stmt = select(PaperPosition).where(PaperPosition.status == "closed")

    open_result = await db.execute(open_stmt)
    closed_result = await db.execute(closed_stmt)

    open_positions = open_result.scalars().all()
    closed_positions = closed_result.scalars().all()

    all_closed = list(closed_positions)

    total_net_pnl = sum(p.net_pnl_eur or 0.0 for p in all_closed)
    total_gross_pnl = sum(p.pnl_eur or 0.0 for p in all_closed)
    total_futures_costs = sum(p.futures_costs_eur or 0.0 for p in all_closed)

    winners = [p for p in all_closed if (p.net_pnl_eur or 0.0) > 0]
    losers = [p for p in all_closed if (p.net_pnl_eur or 0.0) <= 0]

    win_rate = (len(winners) / len(all_closed) * 100.0) if all_closed else None

    gross_profit = sum(p.net_pnl_eur or 0.0 for p in winners)
    gross_loss = abs(sum(p.net_pnl_eur or 0.0 for p in losers)) if losers else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    best_trade = max((p.net_pnl_eur for p in all_closed if p.net_pnl_eur is not None), default=None)
    worst_trade = min((p.net_pnl_eur for p in all_closed if p.net_pnl_eur is not None), default=None)

    # Average holding hours for closed positions
    holding_hours_list = []
    for p in all_closed:
        if p.entry_timestamp and p.exit_timestamp:
            entry = p.entry_timestamp
            exit_ = p.exit_timestamp
            if entry.tzinfo is None:
                entry = entry.replace(tzinfo=timezone.utc)
            if exit_.tzinfo is None:
                exit_ = exit_.replace(tzinfo=timezone.utc)
            holding_hours_list.append((exit_ - entry).total_seconds() / 3600.0)
    avg_holding_hours = sum(holding_hours_list) / len(holding_hours_list) if holding_hours_list else None

    open_responses = [_position_to_response(p) for p in open_positions]

    return PaperStatusResponse(
        generated_at=now,
        open_positions=len(open_positions),
        total_closed_positions=len(all_closed),
        total_net_pnl_eur=round(total_net_pnl, 2),
        total_gross_pnl_eur=round(total_gross_pnl, 2),
        total_futures_costs_eur=round(total_futures_costs, 2),
        win_rate_pct=round(win_rate, 2) if win_rate is not None else None,
        profit_factor=round(profit_factor, 4) if profit_factor is not None else None,
        best_trade_net_pnl_eur=round(best_trade, 2) if best_trade is not None else None,
        worst_trade_net_pnl_eur=round(worst_trade, 2) if worst_trade is not None else None,
        avg_holding_hours=round(avg_holding_hours, 2) if avg_holding_hours is not None else None,
        positions=open_responses,
    )


@router.get(
    "/positions",
    response_model=List[PaperPositionResponse],
    summary="List open paper positions",
)
async def get_open_positions(
    db: AsyncSession = Depends(get_db),
) -> List[PaperPositionResponse]:
    """Return all currently open paper positions."""
    stmt = (
        select(PaperPosition)
        .where(PaperPosition.status == "open")
        .order_by(PaperPosition.entry_timestamp.desc())
    )
    result = await db.execute(stmt)
    positions = result.scalars().all()
    return [_position_to_response(p) for p in positions]


@router.post(
    "/positions",
    response_model=PaperTradeResponse,
    summary="Open a new paper position",
)
async def open_paper_position(
    request: PaperTradeRequest,
    db: AsyncSession = Depends(get_db),
) -> PaperTradeResponse:
    """
    Manually open a new paper trading position.

    In production the signal engine triggers this automatically; this endpoint
    allows manual overrides for testing purposes.
    """
    now = datetime.now(timezone.utc)

    # Check max open positions
    count_stmt = select(func.count()).select_from(PaperPosition).where(
        PaperPosition.status == "open"
    )
    count_result = await db.execute(count_stmt)
    open_count = count_result.scalar_one()

    if open_count >= settings.max_open_positions:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum open positions ({settings.max_open_positions}) reached.",
        )

    position = PaperPosition(
        signal_id=request.signal_id,
        status="open",
        entry_price=request.entry_price,
        entry_timestamp=now,
        notional_size_mwh=request.notional_size_mwh,
        stop_loss=request.stop_loss,
        take_profit=request.take_profit,
        max_holding_hours=request.max_holding_hours,
    )
    db.add(position)
    await db.flush()
    await db.refresh(position)

    logger.info(
        "Paper position opened: id=%d entry=%.2f size=%.1f",
        position.id,
        position.entry_price,
        position.notional_size_mwh,
    )

    return PaperTradeResponse(
        success=True,
        position=_position_to_response(position),
        message=f"Paper position #{position.id} opened at {request.entry_price} EUR/MWh.",
    )


@router.post(
    "/positions/{position_id}/close",
    response_model=PaperTradeResponse,
    summary="Close a paper position",
)
async def close_paper_position(
    position_id: int,
    request: ClosePositionRequest,
    db: AsyncSession = Depends(get_db),
) -> PaperTradeResponse:
    """Close a specific paper position at the given exit price."""
    stmt = select(PaperPosition).where(PaperPosition.id == position_id)
    result = await db.execute(stmt)
    position = result.scalar_one_or_none()

    if position is None:
        raise HTTPException(status_code=404, detail=f"Position #{position_id} not found.")

    if position.status != "open":
        raise HTTPException(
            status_code=400,
            detail=f"Position #{position_id} is already {position.status}.",
        )

    now = datetime.now(timezone.utc)
    pnl_gross, futures_costs, net_pnl = _calculate_pnl(
        position.entry_price,
        request.exit_price,
        position.notional_size_mwh,
    )

    position.status = "closed"
    position.exit_price = request.exit_price
    position.exit_timestamp = now
    position.pnl_eur = pnl_gross
    position.futures_costs_eur = futures_costs
    position.net_pnl_eur = net_pnl
    position.exit_reason = request.exit_reason
    position.updated_at = now

    db.add(position)
    await db.flush()

    logger.info(
        "Paper position closed: id=%d exit=%.2f pnl=%.2f net=%.2f reason=%s",
        position.id,
        request.exit_price,
        pnl_gross,
        net_pnl,
        request.exit_reason,
    )

    return PaperTradeResponse(
        success=True,
        position=_position_to_response(position),
        message=(
            f"Position #{position_id} closed at {request.exit_price} EUR/MWh. "
            f"Net PnL: {net_pnl:.2f} EUR."
        ),
    )


@router.get(
    "/trades",
    response_model=List[PaperPositionResponse],
    summary="Trade journal (last N closed trades)",
)
async def get_trade_journal(
    limit: int = Query(default=50, ge=1, le=500, description="Number of trades to return"),
    db: AsyncSession = Depends(get_db),
) -> List[PaperPositionResponse]:
    """
    Return the last N closed paper trades (trade journal).

    Ordered by exit timestamp descending (most recent first).
    """
    stmt = (
        select(PaperPosition)
        .where(PaperPosition.status == "closed")
        .order_by(PaperPosition.exit_timestamp.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    trades = result.scalars().all()
    return [_position_to_response(t) for t in trades]
