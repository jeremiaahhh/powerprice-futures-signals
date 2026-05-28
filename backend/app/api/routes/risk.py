"""
Risk intelligence API endpoints.

GET /risk/tail         – Current tail risk assessment
GET /risk/gap          – Gap detection for recent prices
GET /risk/volatility   – Volatility regime classification
GET /risk/blocked-trades – Recently blocked trade signals
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.session import get_db
from app.db.models import HourlyPrice

logger = get_logger(__name__)
router = APIRouter()


async def _load_df(db: AsyncSession, hours: int = 48) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    stmt = (
        select(HourlyPrice)
        .where(HourlyPrice.timestamp >= cutoff, HourlyPrice.price_eur_mwh.isnot(None))
        .order_by(HourlyPrice.timestamp.asc())
    )
    rows = (await db.execute(stmt)).scalars().all()
    if not rows:
        return pd.DataFrame()
    data = []
    for r in rows:
        def sf(v):
            try: return float(v) if v is not None else None
            except: return None
        data.append({
            "timestamp": r.timestamp,
            "price_eur_mwh": sf(r.price_eur_mwh),
            "load_mw": sf(r.load_mw),
            "wind_onshore_mw": sf(r.wind_onshore_mw),
            "wind_offshore_mw": sf(r.wind_offshore_mw),
            "solar_mw": sf(r.solar_mw),
            "residual_load_mw": sf(r.residual_load_mw),
        })
    return pd.DataFrame(data)


@router.get("/tail", summary="Current tail risk assessment")
async def get_tail_risk(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    try:
        df = await _load_df(db, hours=48)
        if df.empty:
            raise HTTPException(status_code=503, detail="No market data available")
        from app.risk.tail_risk_engine import TailRiskEngine
        from app.core.config import settings
        engine = TailRiskEngine(
            max_negative_price=settings.max_negative_price_eur_mwh,
            max_streak_hours=settings.max_negative_streak_hours,
            max_gap_size=settings.max_gap_size_eur_mwh,
            max_tail_risk_score=settings.max_tail_risk_score,
        )
        current_price = float(df["price_eur_mwh"].iloc[-1])
        result = engine.assess(df, current_price)
        return {
            "tail_risk_score": result.tail_risk_score,
            "gap_risk_score": result.gap_risk_score,
            "oversupply_stress_index": result.oversupply_stress_index,
            "rebound_failure_probability": result.rebound_failure_probability,
            "negative_price_streak": result.negative_price_streak,
            "max_price_gap_1h": result.max_price_gap_1h,
            "volatility_24h": result.volatility_24h,
            "is_blocked": result.is_blocked,
            "block_reason": result.block_reason,
            "block_detail": result.block_detail,
            "components": result.components,
            "current_price": round(current_price, 2),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("GET /risk/tail failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/gap", summary="Gap detection for recent prices")
async def get_gap_assessment(
    window_hours: int = Query(default=12, ge=1, le=72),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    try:
        df = await _load_df(db, hours=max(window_hours + 2, 24))
        if df.empty:
            raise HTTPException(status_code=503, detail="No market data available")
        from app.risk.gap_detector import GapDetector
        from app.core.config import settings
        detector = GapDetector(threshold_eur_mwh=settings.max_gap_size_eur_mwh)
        result = detector.detect(df, window_hours=window_hours)
        return {
            "max_gap_1h": result.max_gap_1h,
            "gap_score": result.gap_score,
            "has_extreme_gap": result.has_extreme_gap,
            "gap_timestamps": result.gap_timestamps,
            "threshold_eur_mwh": settings.max_gap_size_eur_mwh,
            "window_hours": window_hours,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("GET /risk/gap failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/volatility", summary="Volatility regime classification")
async def get_volatility(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    try:
        df = await _load_df(db, hours=72)
        if df.empty:
            raise HTTPException(status_code=503, detail="No market data available")
        from app.risk.volatility_guard import VolatilityGuard
        from app.core.config import settings
        guard = VolatilityGuard(
            extreme_threshold=settings.tail_risk_extreme_vol_threshold,
            spike_multiplier=settings.tail_risk_vol_spike_multiplier,
        )
        result = guard.assess(df)
        return {
            "vol_1h": result.vol_1h,
            "vol_6h": result.vol_6h,
            "vol_24h": result.vol_24h,
            "vol_spike_ratio": result.vol_spike_ratio,
            "regime": result.regime,
            "is_blocked": result.is_blocked,
            "detail": result.detail,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("GET /risk/volatility failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/blocked-trades", summary="Recently blocked trade signals")
async def get_blocked_trades(
    hours: int = Query(default=168, ge=1, le=720),
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    try:
        from app.db.models import FuturesSignal
        _BLOCKED_ACTIONS = [
            "TAIL_RISK_BLOCKED", "GAP_RISK_BLOCKED",
            "EXTREME_VOLATILITY_BLOCKED", "DATA_QUALITY_BLOCKED", "RISK_BLOCKED",
        ]
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        stmt = (
            select(FuturesSignal)
            .where(
                FuturesSignal.action.in_(_BLOCKED_ACTIONS),
                FuturesSignal.timestamp >= cutoff,
            )
            .order_by(FuturesSignal.timestamp.desc())
            .limit(100)
        )
        rows = (await db.execute(stmt)).scalars().all()
        return [
            {
                "timestamp": str(r.timestamp)[:19],
                "action": r.action,
                "current_price": r.current_price,
                "p_rebound": r.p_rebound,
                "confidence": r.confidence,
                "reason": r.reason,
            }
            for r in rows
        ]
    except Exception as exc:
        logger.exception("GET /risk/blocked-trades failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
