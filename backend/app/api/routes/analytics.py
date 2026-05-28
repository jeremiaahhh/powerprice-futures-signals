from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.session import get_db
from app.db.models import FuturesSignal

logger = get_logger(__name__)
router = APIRouter()

_DSN = "postgresql://ppuser:pppass@localhost:5432/powerprice"

_ENTER_ACTIONS = {"ENTER_LONG_REBOUND_SIGNAL", "HIGH_CONFIDENCE_SIGNAL"}
_WIN_THRESHOLD = 14.0  # EUR/MWh


# ---------------------------------------------------------------------------
# Sync worker (runs in thread executor)
# ---------------------------------------------------------------------------


def _compute_loss_clusters() -> Dict[str, Any]:
    import psycopg2
    import psycopg2.extras
    from datetime import date

    conn = psycopg2.connect(_DSN)
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        cutoff_2y = datetime.now(timezone.utc) - timedelta(days=730)

        # Load hourly prices for the last 2 years
        cur.execute(
            """
            SELECT timestamp, price_eur_mwh, solar_mw,
                   wind_onshore_mw, wind_offshore_mw,
                   is_holiday, is_weekend
            FROM hourly_prices
            WHERE timestamp >= %s
              AND price_eur_mwh IS NOT NULL
            ORDER BY timestamp ASC
            """,
            (cutoff_2y,),
        )
        price_rows = cur.fetchall()

        if not price_rows:
            return {}

        # Build a timestamp -> row lookup
        ts_map: Dict[datetime, dict] = {}
        for row in price_rows:
            ts = row["timestamp"]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            ts_map[ts] = dict(row)

        # Load entry signals
        cur.execute(
            """
            SELECT timestamp, action, current_price
            FROM futures_signals
            WHERE action = ANY(%s)
              AND timestamp >= %s
            ORDER BY timestamp ASC
            """,
            (list(_ENTER_ACTIONS), cutoff_2y),
        )
        signals = cur.fetchall()

        if not signals:
            return {}

        total_signals = len(signals)
        wins: List[float] = []
        losses: List[Dict[str, Any]] = []

        for sig in signals:
            entry_ts = sig["timestamp"]
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.replace(tzinfo=timezone.utc)
            entry_price = sig["current_price"]
            if entry_price is None:
                continue

            ts_6h = entry_ts + timedelta(hours=6)
            price_row_6h = ts_map.get(ts_6h)
            if price_row_6h is None:
                continue

            price_6h = price_row_6h["price_eur_mwh"]
            if price_6h is None:
                continue

            realized_rebound = price_6h - entry_price

            # Compute vol proxy from surrounding prices
            nearby = [
                ts_map[entry_ts + timedelta(hours=h)]["price_eur_mwh"]
                for h in range(-12, 13)
                if (entry_ts + timedelta(hours=h)) in ts_map
                and ts_map[entry_ts + timedelta(hours=h)]["price_eur_mwh"] is not None
            ]
            price_vol = float(np.std(nearby)) if len(nearby) >= 3 else 0.0

            entry_row = ts_map.get(entry_ts) or {}
            solar = entry_row.get("solar_mw") or 0.0
            wind_on = entry_row.get("wind_onshore_mw") or 0.0
            wind_off = entry_row.get("wind_offshore_mw") or 0.0

            if realized_rebound > _WIN_THRESHOLD:
                wins.append(realized_rebound)
            else:
                losses.append({
                    "realized_rebound": realized_rebound,
                    "hour": entry_ts.hour,
                    "weekday": entry_ts.weekday(),
                    "month": entry_ts.month,
                    "is_holiday": bool(entry_row.get("is_holiday", False)),
                    "is_weekend": bool(entry_row.get("is_weekend", False)),
                    "solar_mw": solar,
                    "wind_total": wind_on + wind_off,
                    "price_vol": price_vol,
                    "entry_price": entry_price,
                })

        total_wins = len(wins)
        total_losses = len(losses)
        win_rate = total_wins / total_signals if total_signals > 0 else 0.0

        # Aggregate clusters
        by_hour: Dict[int, Dict] = defaultdict(lambda: {"count": 0, "sum_loss": 0.0})
        by_weekday: Dict[int, Dict] = defaultdict(lambda: {"count": 0, "sum_loss": 0.0})
        by_month: Dict[int, Dict] = defaultdict(lambda: {"count": 0, "sum_loss": 0.0})
        extreme_weather: List[Dict] = []

        for loss in losses:
            rr = loss["realized_rebound"]
            h = loss["hour"]
            wd = loss["weekday"]
            mo = loss["month"]

            by_hour[h]["count"] += 1
            by_hour[h]["sum_loss"] += rr

            by_weekday[wd]["count"] += 1
            by_weekday[wd]["sum_loss"] += rr

            by_month[mo]["count"] += 1
            by_month[mo]["sum_loss"] += rr

            if loss["price_vol"] > 80 or loss["solar_mw"] > 30000:
                extreme_weather.append({
                    "hour": h,
                    "weekday": wd,
                    "month": mo,
                    "solar_mw": loss["solar_mw"],
                    "wind_total": loss["wind_total"],
                    "price_vol": round(loss["price_vol"], 2),
                    "realized_rebound": round(rr, 2),
                    "entry_price": round(loss["entry_price"], 2),
                })

        def _summarise(d: Dict) -> Dict:
            return {
                k: {
                    "count": v["count"],
                    "avg_loss": round(v["sum_loss"] / v["count"], 2) if v["count"] > 0 else 0.0,
                }
                for k, v in sorted(d.items())
            }

        return {
            "total_signals": total_signals,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "win_rate": round(win_rate, 4),
            "avg_win_eur_mwh": round(float(np.mean(wins)), 2) if wins else None,
            "avg_loss_eur_mwh": round(float(np.mean([l["realized_rebound"] for l in losses])), 2) if losses else None,
            "best_win_eur_mwh": round(float(np.max(wins)), 2) if wins else None,
            "worst_loss_eur_mwh": round(float(np.min([l["realized_rebound"] for l in losses])), 2) if losses else None,
            "clusters": {
                "by_hour": _summarise(by_hour),
                "by_weekday": _summarise(by_weekday),
                "by_month": _summarise(by_month),
                "extreme_weather": extreme_weather[:100],
            },
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/loss-clusters", summary="Analyse losing trade clusters")
async def get_loss_clusters() -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _compute_loss_clusters)
    except Exception as exc:
        logger.exception("GET /analytics/loss-clusters failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Loss cluster analysis failed: {exc}")

    if not result:
        return {
            "total_signals": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "loss_clusters": [],
            "note": "No entry signals in the database yet. Clusters will appear once ENTER signals are generated.",
        }

    logger.info(
        "GET /analytics/loss-clusters: %d signals, win_rate=%.1f%%",
        result.get("total_signals", 0),
        (result.get("win_rate") or 0) * 100,
    )
    return result


@router.get("/regime-performance", summary="Per-regime backtest performance stats")
async def get_regime_performance(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    # Load all futures_signals
    stmt = select(FuturesSignal).order_by(FuturesSignal.timestamp.asc())
    result = await db.execute(stmt)
    signals = result.scalars().all()

    if not signals:
        raise HTTPException(status_code=503, detail="No Futures signals found in database")

    # Try loading regime snapshots
    regime_map: Dict[datetime, str] = {}
    try:
        from app.db.models import RegimeSnapshot
        rs_stmt = select(RegimeSnapshot).order_by(RegimeSnapshot.timestamp.asc())
        rs_result = await db.execute(rs_stmt)
        snapshots = rs_result.scalars().all()
        for snap in snapshots:
            ts = snap.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            regime_map[ts] = getattr(snap, "regime", "UNKNOWN")
    except Exception:
        pass

    if not regime_map:
        signal_counts: Dict[str, int] = defaultdict(int)
        for sig in signals:
            signal_counts[sig.action] += 1
        return {
            "status": "partial",
            "message": "Regime snapshots not yet available — showing signal counts only",
            "total_signals": len(signals),
            "signal_counts": dict(signal_counts),
            "per_regime": {},
        }

    # Align each signal to the nearest regime snapshot
    snap_timestamps = sorted(regime_map.keys())

    def _nearest_regime(ts: datetime) -> str:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        best = min(snap_timestamps, key=lambda t: abs((t - ts).total_seconds()))
        return regime_map[best]

    per_regime: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sig in signals:
        regime = _nearest_regime(sig.timestamp)
        per_regime[regime][sig.action] += 1

    return {
        "status": "ok",
        "total_signals": len(signals),
        "per_regime": {
            regime: dict(action_counts)
            for regime, action_counts in sorted(per_regime.items())
        },
    }


@router.get("/tail-events", summary="Recent tail risk events")
async def get_tail_events(
    limit: int = Query(default=50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Return recent tail risk events logged by the signal engine."""
    try:
        from app.db.models import TailEvent
        stmt = select(TailEvent).order_by(TailEvent.timestamp.desc()).limit(limit)
        rows = (await db.execute(stmt)).scalars().all()
        return [{k: v for k, v in r.__dict__.items() if not k.startswith("_")} for r in rows]
    except Exception as exc:
        logger.exception("GET /analytics/tail-events failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/oos-performance", summary="OOS backtest performance summary")
async def get_oos_performance(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Rolling OOS performance metrics from stored backtest results."""
    try:
        from app.db.models import BacktestResult as BacktestResultDB
        stmt = select(BacktestResultDB).order_by(BacktestResultDB.created_at.desc()).limit(20)
        rows = (await db.execute(stmt)).scalars().all()
        if not rows:
            return {"status": "no_backtest_results", "runs": [], "summary": {}}

        runs = []
        for r in rows:
            runs.append({
                "run_id": r.run_id,
                "strategy": r.strategy,
                "start_date": str(r.start_date)[:10] if r.start_date else None,
                "end_date": str(r.end_date)[:10] if r.end_date else None,
                "total_trades": r.total_trades,
                "win_rate_pct": r.win_rate_pct,
                "sharpe_ratio": r.sharpe_ratio,
                "max_drawdown_pct": r.max_drawdown_pct,
                "total_return_pct": r.total_return_pct,
                "profit_factor": r.profit_factor,
                "worst_trade_eur_mwh": r.worst_trade_eur_mwh,
                "created_at": str(r.created_at)[:19] if r.created_at else None,
            })

        ml_runs = [r for r in runs if r["strategy"] == "ml_rebound"]
        # Exclude look-ahead-biased runs (100% win rate + 0% drawdown = biased)
        honest_ml_runs = [
            r for r in ml_runs
            if not (r.get("win_rate_pct") == 100.0 and (r.get("max_drawdown_pct") or 0.0) == 0.0)
        ]
        summary: Dict[str, Any] = {}
        if ml_runs:
            target = honest_ml_runs if honest_ml_runs else ml_runs
            sharpes = [r["sharpe_ratio"] for r in target if r["sharpe_ratio"] is not None]
            wrs = [r["win_rate_pct"] for r in target if r["win_rate_pct"] is not None]
            pfs = [r["profit_factor"] for r in target if r["profit_factor"] is not None]
            summary = {
                "ml_runs_count": len(target),
                "total_ml_runs_in_db": len(ml_runs),
                "biased_runs_excluded": len(ml_runs) - len(target),
                "avg_sharpe": round(sum(sharpes)/len(sharpes), 4) if sharpes else None,
                "avg_win_rate": round(sum(wrs)/len(wrs), 2) if wrs else None,
                "avg_profit_factor": round(sum(pfs)/len(pfs), 4) if pfs else None,
            }

        return {
            "status": "ok",
            "summary": summary,
            "runs": runs,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.exception("GET /analytics/oos-performance failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/regime-drift", summary="Regime distribution drift over time")
async def get_regime_drift(
    days: int = Query(default=30, ge=7, le=365),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Analyse how the market regime distribution has shifted over the last N days."""
    try:
        from app.db.models import RegimeSnapshot
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        stmt = (
            select(RegimeSnapshot)
            .where(RegimeSnapshot.timestamp >= cutoff)
            .order_by(RegimeSnapshot.timestamp.asc())
        )
        rows = (await db.execute(stmt)).scalars().all()
        if not rows:
            return {"status": "no_data", "regime_counts": {}, "days_analyzed": days, "note": "No regime snapshots in window"}

        from collections import Counter, defaultdict
        regimes = [r.regime for r in rows]
        counts = dict(Counter(regimes))
        total = len(rows)

        mid = len(rows) // 2
        first_half = Counter(r.regime for r in rows[:mid])
        second_half = Counter(r.regime for r in rows[mid:])
        all_regs = set(list(first_half.keys()) + list(second_half.keys()))
        drift = {}
        for reg in all_regs:
            fp = first_half.get(reg, 0) / max(mid, 1) * 100
            sp = second_half.get(reg, 0) / max(len(rows)-mid, 1) * 100
            drift[reg] = {"first_half_pct": round(fp, 1), "second_half_pct": round(sp, 1), "drift_pp": round(sp - fp, 1)}

        conf_by_regime: dict = defaultdict(list)
        for r in rows:
            if r.confidence is not None:
                conf_by_regime[r.regime].append(r.confidence)
        avg_conf = {reg: round(sum(cs)/len(cs), 3) for reg, cs in conf_by_regime.items()}

        return {
            "status": "ok",
            "days_analyzed": days,
            "total_snapshots": total,
            "regime_distribution": {k: {"count": v, "pct": round(v/total*100, 1)} for k, v in counts.items()},
            "drift_analysis": drift,
            "avg_confidence_by_regime": avg_conf,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.exception("GET /analytics/regime-drift failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
