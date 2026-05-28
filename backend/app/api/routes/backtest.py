"""
Backtest routes.

POST /backtest/run                  – run a backtest with given parameters
GET  /backtest/results              – return last N backtest results from DB
GET  /backtest/compare-naive-vs-ml  – compare naive vs ML strategies
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.session import get_db
from app.db.models import HourlyPrice, BacktestResult as BacktestResultDB
from app.api.schemas import (
    BacktestParams,
    BacktestResult,
    BacktestComparison,
    BacktestStrategy,
    CostModelConfig,
    MonthlyPerformance,
)

logger = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


async def _load_historical_data(
    db: AsyncSession,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Load HourlyPrice rows between start and end into a DataFrame."""
    stmt = (
        select(HourlyPrice)
        .where(
            HourlyPrice.timestamp >= start,
            HourlyPrice.timestamp <= end,
            HourlyPrice.price_eur_mwh.isnot(None),
        )
        .order_by(HourlyPrice.timestamp.asc())
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    data = []
    for r in rows:
        data.append({
            "timestamp": r.timestamp,
            "price_eur_mwh": r.price_eur_mwh,
            "load_mw": r.load_mw,
            "wind_onshore_mw": r.wind_onshore_mw,
            "wind_offshore_mw": r.wind_offshore_mw,
            "solar_mw": r.solar_mw,
            "residual_load_mw": r.residual_load_mw,
            "net_export_mw": r.net_export_mw,
            "temperature_c": r.temperature_c,
            "wind_speed_ms": r.wind_speed_ms,
            "solar_radiation_wm2": r.solar_radiation_wm2,
            "cloud_cover_pct": r.cloud_cover_pct,
            "is_holiday": int(r.is_holiday) if r.is_holiday is not None else 0,
            "is_weekend": int(r.is_weekend) if r.is_weekend is not None else 0,
            "hour": r.hour,
            "month": r.month,
        })

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------


def _generate_naive_signals(df: pd.DataFrame) -> pd.Series:
    """
    Naive strategy: enter long whenever price < 0.
    Returns a boolean Series aligned with df index.
    """
    return df["price_eur_mwh"] < 0


def _train_oos_rebound_clf(train_cutoff: datetime, days_back: int = 548):
    """
    Train a ReboundClassifier on data strictly before train_cutoff.
    Returns (clf, error_string_or_None).
    Uses 548 days (~18 months) lookback by default.
    """
    import tempfile, os
    from app.ml.trainer import ModelTrainer
    from app.ml.rebound_classifier import ReboundClassifier

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ModelTrainer(model_dir=tmpdir)
        try:
            train_df = trainer.load_training_data(days_back=days_back, before_ts=train_cutoff)
            if train_df.empty or len(train_df) < 500:
                return None, f"Insufficient training data before {train_cutoff.date()}"
            clf = ReboundClassifier(model_dir=tmpdir)
            clf.train(train_df)
            # Move model files to a real temp path we can keep
            import shutil
            keep_dir = tempfile.mkdtemp(prefix="oos_backtest_")
            for fname in os.listdir(tmpdir):
                shutil.copy(os.path.join(tmpdir, fname), os.path.join(keep_dir, fname))
            clf2 = ReboundClassifier(model_dir=keep_dir)
            clf2.load(keep_dir)
            return clf2, None
        except Exception as exc:
            return None, str(exc)


def _predict_rebound(
    clf,
    df: pd.DataFrame,
) -> pd.Series:
    """Run loaded ReboundClassifier on df, return p_rebound Series aligned to df.index."""
    from app.features.engineering import FeatureEngineer
    fe = FeatureEngineer()
    features_df = fe.build_features(df)
    available_cols = [c for c in fe.FEATURE_COLUMNS if c in features_df.columns]
    valid_mask = features_df[available_cols].notna().all(axis=1)
    X = features_df.loc[valid_mask, available_cols]
    if X.empty:
        return pd.Series(0.0, index=df.index)
    proba = clf.predict_proba(X)
    p_arr = np.array(proba)
    if p_arr.ndim == 2:
        p_arr = p_arr[:, 1]
    return pd.Series(p_arr, index=X.index).reindex(df.index, fill_value=0.0)


def _generate_ml_signals(
    df: pd.DataFrame,
    p_rebound_threshold: float = 0.60,
    min_edge_threshold: float = 10.0,
    cost_model: Optional[CostModelConfig] = None,
    train_cutoff: Optional[datetime] = None,
    walk_forward_days: Optional[int] = None,
) -> tuple[pd.Series, pd.Series]:
    """
    ML strategy: enter when price < 0 AND p_rebound >= threshold AND net_edge >= min_edge.

    train_cutoff: train model on data before this date (OOS split). If None, uses
                  the pre-trained on-disk model (may have look-ahead bias).
    walk_forward_days: if set, retrain at each fold boundary within df.
    """
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

    if walk_forward_days and train_cutoff:
        # Walk-forward: retrain at each fold boundary
        p_rebound_series = pd.Series(0.0, index=df.index)
        ts_col = pd.to_datetime(df["timestamp"])
        fold_start = train_cutoff
        fold_end = ts_col.max().to_pydatetime()
        if fold_end.tzinfo is None:
            fold_end = fold_end.replace(tzinfo=timezone.utc)
        current_cutoff = fold_start if fold_start.tzinfo else fold_start.replace(tzinfo=timezone.utc)

        while current_cutoff < fold_end:
            next_cutoff = current_cutoff + timedelta(days=walk_forward_days)
            fold_mask = (
                ts_col >= pd.Timestamp(current_cutoff)
            ) & (
                ts_col < pd.Timestamp(min(next_cutoff, fold_end + timedelta(hours=1)))
            )
            fold_df = df[fold_mask]
            if not fold_df.empty:
                clf, err = _train_oos_rebound_clf(current_cutoff)
                if clf is not None:
                    fold_preds = _predict_rebound(clf, fold_df)
                    p_rebound_series.update(fold_preds)
                else:
                    logger.warning("Walk-forward fold %s: %s — using synthetic", current_cutoff.date(), err)
                    p_rebound_series.update(_synthetic_p_rebound(fold_df))
            current_cutoff = next_cutoff
    elif train_cutoff:
        # Single OOS split: train before cutoff, predict on all of df
        clf, err = _train_oos_rebound_clf(train_cutoff)
        if clf is not None:
            p_rebound_series = _predict_rebound(clf, df)
            logger.info("OOS model trained on data before %s", train_cutoff.date())
        else:
            logger.warning("OOS model training failed (%s); falling back to synthetic", err)
            p_rebound_series = _synthetic_p_rebound(df)
    else:
        # Pre-trained model (may have look-ahead bias if test overlaps training window)
        try:
            from app.ml.rebound_classifier import ReboundClassifier
            reb_clf = ReboundClassifier()
            reb_clf.load(settings.model_dir)
            p_rebound_series = _predict_rebound(reb_clf, df)
        except Exception:
            p_rebound_series = _synthetic_p_rebound(df)

    entry_signals = (
        (df["price_eur_mwh"] < 0)
        & (p_rebound_series >= p_rebound_threshold)
        & (net_edge >= min_edge_threshold)
    )
    return entry_signals, p_rebound_series


def _synthetic_p_rebound(df: pd.DataFrame) -> pd.Series:
    """Generate synthetic rebound probabilities for backtesting when models are absent."""
    prices = df["price_eur_mwh"]
    # Higher probability for deeper negative prices
    p = np.where(
        prices < 0,
        np.clip(0.4 + abs(prices) / 200.0, 0.0, 0.95),
        0.0,
    )
    return pd.Series(p, index=df.index)


def _simulate_trades(
    df: pd.DataFrame,
    entry_signals: pd.Series,
    params: BacktestParams,
) -> List[Dict[str, Any]]:
    """
    Simulate trades given entry signals and exit rules.

    Returns a list of trade dicts with entry/exit timestamps, prices, and PnL.
    """
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
            # Check exit conditions
            holding_hours = i - entry_idx
            should_exit = False
            exit_reason = "unknown"

            if holding_hours >= params.max_holding_hours:
                should_exit = True
                exit_reason = "time_exit"
            elif price <= entry_price - params.stop_loss_eur_mwh:
                # Stop-loss: price fell further negative (long position losing)
                should_exit = True
                exit_reason = "stop_loss"
            elif price - entry_price >= params.take_profit_eur_mwh:
                # Take-profit: price rebounded enough to cover costs + target
                should_exit = True
                exit_reason = "take_profit"
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

    # Close any open trade at end of period
    if in_trade and entry_idx < len(prices) - 1:
        exit_price = float(prices[-1])
        holding_hours = len(prices) - 1 - entry_idx
        pnl_gross = (exit_price - entry_price) * params.notional_size_mwh
        futures_costs = cost_total * params.notional_size_mwh
        net_pnl = pnl_gross - futures_costs
        trades.append({
            "entry_timestamp": timestamps[entry_idx],
            "exit_timestamp": timestamps[-1],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "holding_hours": holding_hours,
            "pnl_gross": round(pnl_gross, 4),
            "futures_costs": round(futures_costs, 4),
            "net_pnl": round(net_pnl, 4),
            "exit_reason": "end_of_period",
            "is_winner": net_pnl > 0,
        })

    return trades


def _compute_metrics(
    trades: List[Dict[str, Any]],
    df: pd.DataFrame,
    params: BacktestParams,
    run_id: str,
    strategy: BacktestStrategy,
) -> BacktestResult:
    """Compute BacktestResult from a list of simulated trades."""
    now = datetime.now(timezone.utc)
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
            sharpe_ratio=None,
            sortino_ratio=None,
            max_drawdown_pct=0.0,
            profit_factor=None,
            win_rate_pct=0.0,
            avg_trade_eur_mwh=0.0,
            best_trade_eur_mwh=None,
            worst_trade_eur_mwh=None,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            trades_per_month=0.0,
            avg_holding_hours=None,
            equity_curve=[],
            monthly_performance=[],
            created_at=now,
        )

    pnl_list = [t["net_pnl"] for t in trades]
    pnl_arr = np.array(pnl_list)

    winning = [t for t in trades if t["is_winner"]]
    losing = [t for t in trades if not t["is_winner"]]

    win_rate = len(winning) / total_trades * 100.0
    avg_trade = float(np.mean(pnl_arr))
    best_trade = float(np.max(pnl_arr))
    worst_trade = float(np.min(pnl_arr))

    gross_profit = sum(t["net_pnl"] for t in winning)
    gross_loss = abs(sum(t["net_pnl"] for t in losing)) if losing else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    # Initial capital = 100 EUR/MWh * notional
    initial_capital = 100.0 * params.notional_size_mwh
    total_pnl = float(np.sum(pnl_arr))
    total_return_pct = (total_pnl / initial_capital) * 100.0

    # Annualise
    delta_days = (params.end_date - params.start_date).days
    if delta_days > 0:
        years = delta_days / 365.25
        base = 1 + total_return_pct / 100.0
        if base <= 0:
            annualized_return_pct = -100.0  # total loss
        else:
            annualized_return_pct = (base ** (1.0 / max(years, 0.01)) - 1) * 100.0
    else:
        annualized_return_pct = 0.0

    # Sharpe ratio (annualised, assume daily PnL)
    if len(pnl_arr) > 1:
        sharpe = float(np.mean(pnl_arr) / (np.std(pnl_arr, ddof=1) + 1e-9)) * np.sqrt(252)
        # Sortino: only downside std
        downside = pnl_arr[pnl_arr < 0]
        sortino_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-9
        sortino = float(np.mean(pnl_arr) / (sortino_std + 1e-9)) * np.sqrt(252)
    else:
        sharpe = None
        sortino = None

    # Max drawdown from cumulative PnL curve
    cum_pnl = np.cumsum(pnl_arr)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = (running_max - cum_pnl)
    max_dd_abs = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    max_drawdown_pct = (max_dd_abs / (initial_capital + abs(float(running_max.max())) + 1e-9)) * 100.0

    # Trades per month
    months = max(delta_days / 30.0, 0.01)
    trades_per_month = total_trades / months

    # Average holding hours
    holding_hours_list = [t["holding_hours"] for t in trades if t.get("holding_hours") is not None]
    avg_holding_hours = float(np.mean(holding_hours_list)) if holding_hours_list else None

    # Equity curve
    equity_curve = []
    equity = initial_capital
    for t in trades:
        equity += t["net_pnl"]
        equity_curve.append({
            "timestamp": str(t["exit_timestamp"]),
            "equity": round(equity, 2),
            "trade_pnl": round(t["net_pnl"], 2),
        })

    # Monthly performance breakdown
    monthly_perf: Dict[str, Dict] = {}
    for t in trades:
        ts = t["exit_timestamp"]
        if hasattr(ts, "year"):
            period = f"{ts.year}-{ts.month:02d}"
        else:
            period = str(ts)[:7]
        if period not in monthly_perf:
            monthly_perf[period] = {"trades": 0, "pnl": 0.0, "wins": 0}
        monthly_perf[period]["trades"] += 1
        monthly_perf[period]["pnl"] += t["net_pnl"]
        if t["is_winner"]:
            monthly_perf[period]["wins"] += 1

    monthly_list = []
    for period, data in sorted(monthly_perf.items()):
        t_count = data["trades"]
        wr = (data["wins"] / t_count * 100.0) if t_count > 0 else 0.0
        monthly_list.append(
            MonthlyPerformance(
                period=period,
                trades=t_count,
                win_rate_pct=round(wr, 1),
                pnl_eur=round(data["pnl"], 2),
                return_pct=round(data["pnl"] / initial_capital * 100.0, 2),
            )
        )

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
        avg_holding_hours=round(avg_holding_hours, 1) if avg_holding_hours is not None else None,
        equity_curve=equity_curve,
        monthly_performance=monthly_list,
        created_at=now,
    )


async def _run_single_backtest(
    db: AsyncSession,
    params: BacktestParams,
    strategy: BacktestStrategy,
) -> BacktestResult:
    """Run a single backtest for a given strategy and params."""
    run_id = str(uuid.uuid4())[:8]

    df = await _load_historical_data(db, params.start_date, params.end_date)

    if df.empty:
        raise HTTPException(
            status_code=422,
            detail=f"No historical price data found between {params.start_date} and {params.end_date}. "
                   "Run /data/ingest first.",
        )

    if strategy == BacktestStrategy.NAIVE:
        entry_signals = _generate_naive_signals(df)
    elif strategy == BacktestStrategy.ML_REBOUND:
        # Always use OOS split so model never sees test-period data
        train_cutoff = params.start_date
        if train_cutoff.tzinfo is None:
            train_cutoff = train_cutoff.replace(tzinfo=timezone.utc)
        walk_days = params.walk_forward_window_days if params.use_walk_forward else None
        logger.info(
            "ML backtest: OOS cutoff=%s walk_forward=%s walk_days=%s",
            train_cutoff.date(), params.use_walk_forward, walk_days,
        )
        entry_signals, _ = _generate_ml_signals(
            df,
            p_rebound_threshold=params.min_confidence,
            min_edge_threshold=params.cost_model.min_edge_threshold,
            cost_model=params.cost_model,
            train_cutoff=train_cutoff,
            walk_forward_days=walk_days,
        )
    else:
        raise HTTPException(status_code=422, detail=f"Unknown strategy: {strategy}")

    trades = _simulate_trades(df, entry_signals, params)
    result = _compute_metrics(trades, df, params, run_id, strategy)

    # Persist to DB
    try:
        db_record = BacktestResultDB(
            run_id=run_id,
            strategy=strategy.value,
            start_date=params.start_date,
            end_date=params.end_date,
            total_return_pct=result.total_return_pct,
            annualized_return_pct=result.annualized_return_pct,
            sharpe_ratio=result.sharpe_ratio,
            sortino_ratio=result.sortino_ratio,
            max_drawdown_pct=result.max_drawdown_pct,
            profit_factor=result.profit_factor,
            win_rate_pct=result.win_rate_pct,
            avg_trade_eur_mwh=result.avg_trade_eur_mwh,
            worst_trade_eur_mwh=result.worst_trade_eur_mwh,
            total_trades=result.total_trades,
            trades_per_month=result.trades_per_month,
            equity_curve=result.equity_curve,
            monthly_performance=[m.model_dump() for m in result.monthly_performance],
            parameters=params.model_dump(mode="json"),
        )
        db.add(db_record)
        await db.flush()
    except Exception as exc:
        logger.warning("Failed to persist backtest result: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/run", response_model=BacktestResult, summary="Run a backtest")
async def run_backtest(
    params: BacktestParams,
    db: AsyncSession = Depends(get_db),
) -> BacktestResult:
    """
    Run a backtest for the specified strategy and date range.

    Strategies:
    - **naive**: Enter long on every negative-price hour.
    - **ml_rebound**: Enter only when p_rebound >= threshold and net_edge is positive.
    - **both**: Runs both strategies (returns the ML result; use /backtest/compare for side-by-side).
    """
    strategy = params.strategy

    if strategy == BacktestStrategy.NAIVE:
        return await _run_single_backtest(db, params, BacktestStrategy.NAIVE)
    elif strategy == BacktestStrategy.ML_REBOUND:
        return await _run_single_backtest(db, params, BacktestStrategy.ML_REBOUND)
    else:
        # For "both", run ML (primary) and return it; user can call /compare for full comparison
        return await _run_single_backtest(db, params, BacktestStrategy.ML_REBOUND)


@router.get("/results", response_model=List[BacktestResult], summary="Last N backtest results")
async def get_backtest_results(
    n: int = Query(default=10, ge=1, le=100, description="Number of results to return"),
    strategy: Optional[str] = Query(default=None, description="Filter by strategy"),
    db: AsyncSession = Depends(get_db),
) -> List[BacktestResult]:
    """
    Return the most recent N backtest results from the database.
    """
    stmt = (
        select(BacktestResultDB)
        .order_by(BacktestResultDB.created_at.desc())
        .limit(n)
    )
    if strategy:
        stmt = stmt.where(BacktestResultDB.strategy == strategy)

    result = await db.execute(stmt)
    db_rows = result.scalars().all()

    results = []
    for row in db_rows:
        try:
            # Reconstruct parameters from stored JSON
            params_dict = row.parameters or {}
            params_obj = BacktestParams(**params_dict) if params_dict else BacktestParams(
                start_date=row.start_date or datetime.now(timezone.utc) - timedelta(days=90),
                end_date=row.end_date or datetime.now(timezone.utc),
            )
        except Exception:
            params_obj = BacktestParams(
                start_date=row.start_date or datetime.now(timezone.utc) - timedelta(days=90),
                end_date=row.end_date or datetime.now(timezone.utc),
            )

        monthly = []
        if row.monthly_performance:
            for m in row.monthly_performance:
                try:
                    monthly.append(MonthlyPerformance(**m))
                except Exception:
                    pass

        results.append(
            BacktestResult(
                run_id=row.run_id,
                strategy=BacktestStrategy(row.strategy),
                start_date=row.start_date or params_obj.start_date,
                end_date=row.end_date or params_obj.end_date,
                parameters=params_obj,
                total_return_pct=row.total_return_pct or 0.0,
                annualized_return_pct=row.annualized_return_pct or 0.0,
                sharpe_ratio=row.sharpe_ratio,
                sortino_ratio=row.sortino_ratio,
                max_drawdown_pct=row.max_drawdown_pct or 0.0,
                profit_factor=row.profit_factor,
                win_rate_pct=row.win_rate_pct or 0.0,
                avg_trade_eur_mwh=row.avg_trade_eur_mwh or 0.0,
                worst_trade_eur_mwh=row.worst_trade_eur_mwh,
                total_trades=row.total_trades or 0,
                winning_trades=0,
                losing_trades=0,
                trades_per_month=row.trades_per_month or 0.0,
                equity_curve=row.equity_curve or [],
                monthly_performance=monthly,
                created_at=row.created_at or datetime.now(timezone.utc),
            )
        )

    return results


@router.get(
    "/compare-naive-vs-ml",
    response_model=BacktestComparison,
    summary="Compare naive vs ML strategies",
)
async def compare_naive_vs_ml(
    start_date: datetime = Query(
        default=None,
        description="Start date for backtest (UTC). Defaults to 90 days ago.",
    ),
    end_date: datetime = Query(
        default=None,
        description="End date for backtest (UTC). Defaults to now.",
    ),
    p_rebound_threshold: float = Query(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="p_rebound threshold for ML strategy",
    ),
    min_edge_threshold: float = Query(
        default=10.0,
        ge=0.0,
        description="Minimum net edge threshold for ML strategy",
    ),
    db: AsyncSession = Depends(get_db),
) -> BacktestComparison:
    """
    Run both naive and ML strategies over the same period and return a
    side-by-side comparison with improvement metrics.
    """
    now = datetime.now(timezone.utc)

    if start_date is None:
        start_date = now - timedelta(days=90)
    if end_date is None:
        end_date = now

    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    cost_model = CostModelConfig(
        min_edge_threshold=min_edge_threshold,
    )

    base_params = BacktestParams(
        start_date=start_date,
        end_date=end_date,
        cost_model=cost_model,
        min_confidence=p_rebound_threshold,
    )

    naive_params = BacktestParams(
        strategy=BacktestStrategy.NAIVE,
        start_date=start_date,
        end_date=end_date,
        cost_model=cost_model,
        min_confidence=p_rebound_threshold,
    )

    ml_params = BacktestParams(
        strategy=BacktestStrategy.ML_REBOUND,
        start_date=start_date,
        end_date=end_date,
        cost_model=cost_model,
        min_confidence=p_rebound_threshold,
    )

    naive_result = await _run_single_backtest(db, naive_params, BacktestStrategy.NAIVE)
    ml_result = await _run_single_backtest(db, ml_params, BacktestStrategy.ML_REBOUND)

    # Summary table
    def _fmt(v: Optional[float], pct: bool = False) -> str:
        if v is None:
            return "N/A"
        suffix = "%" if pct else ""
        return f"{v:.2f}{suffix}"

    summary_table = [
        {
            "metric": "Total Return",
            "naive": _fmt(naive_result.total_return_pct, True),
            "ml_rebound": _fmt(ml_result.total_return_pct, True),
            "improvement_pct": round(
                ml_result.total_return_pct - naive_result.total_return_pct, 2
            ),
        },
        {
            "metric": "Sharpe Ratio",
            "naive": _fmt(naive_result.sharpe_ratio),
            "ml_rebound": _fmt(ml_result.sharpe_ratio),
            "improvement_pct": round(
                (ml_result.sharpe_ratio or 0) - (naive_result.sharpe_ratio or 0), 4
            ),
        },
        {
            "metric": "Win Rate",
            "naive": _fmt(naive_result.win_rate_pct, True),
            "ml_rebound": _fmt(ml_result.win_rate_pct, True),
            "improvement_pct": round(
                ml_result.win_rate_pct - naive_result.win_rate_pct, 2
            ),
        },
        {
            "metric": "Max Drawdown",
            "naive": _fmt(naive_result.max_drawdown_pct, True),
            "ml_rebound": _fmt(ml_result.max_drawdown_pct, True),
            "improvement_pct": round(
                naive_result.max_drawdown_pct - ml_result.max_drawdown_pct, 2
            ),
        },
        {
            "metric": "Total Trades",
            "naive": str(naive_result.total_trades),
            "ml_rebound": str(ml_result.total_trades),
            "improvement_pct": None,
        },
        {
            "metric": "Avg Trade EUR/MWh",
            "naive": _fmt(naive_result.avg_trade_eur_mwh),
            "ml_rebound": _fmt(ml_result.avg_trade_eur_mwh),
            "improvement_pct": round(
                ml_result.avg_trade_eur_mwh - naive_result.avg_trade_eur_mwh, 4
            ),
        },
        {
            "metric": "Profit Factor",
            "naive": _fmt(naive_result.profit_factor),
            "ml_rebound": _fmt(ml_result.profit_factor),
            "improvement_pct": round(
                (ml_result.profit_factor or 0) - (naive_result.profit_factor or 0), 4
            ),
        },
    ]

    return BacktestComparison(
        generated_at=now,
        results=[naive_result, ml_result],
        summary_table=summary_table,
    )
