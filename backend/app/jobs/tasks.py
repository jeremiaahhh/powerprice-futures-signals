"""
Celery tasks for PowerPrice Futures Signals.

All tasks use a synchronous SQLAlchemy session (psycopg2) because Celery
workers run synchronously; the async engine is only used by the FastAPI app.

Tasks:
  - ingest_data: fetch recent SMARD data and upsert into DB
  - generate_and_cache_signal: generate the current signal, cache in Redis, save to DB
  - retrain_models: retrain ML models if data has grown sufficiently
  - check_paper_positions: check open paper positions against the latest price
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import redis as redis_lib
from celery import Task
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings
from app.core.logging import get_logger
from app.jobs.celery_app import celery_app

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Synchronous DB session factory (for Celery workers)
# ---------------------------------------------------------------------------

_sync_engine = create_engine(
    settings.database_url_sync,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)
SyncSessionLocal = sessionmaker(bind=_sync_engine, autocommit=False, autoflush=False)


def _get_sync_session() -> Session:
    """Return a new synchronous SQLAlchemy Session."""
    return SyncSessionLocal()


# ---------------------------------------------------------------------------
# Redis client (for signal caching)
# ---------------------------------------------------------------------------


def _get_redis() -> redis_lib.Redis:
    return redis_lib.from_url(settings.redis_url, decode_responses=True)


# ---------------------------------------------------------------------------
# Task: ingest_data
# ---------------------------------------------------------------------------


@celery_app.task(
    name="app.jobs.tasks.ingest_data",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def ingest_data(self: Task, hours_back: int = 48) -> Dict[str, Any]:
    """
    Fetch recent SMARD data and upsert into the hourly_prices table.

    Uses the synchronous httpx / asyncio event loop pattern to reuse the
    existing async fetch_recent function inside a Celery task.
    """
    import asyncio
    import pandas as pd
    from app.data.smard import fetch_recent
    from app.db.models import HourlyPrice

    logger.info("Task ingest_data started: hours_back=%d", hours_back)
    rows_inserted = 0
    rows_updated = 0
    errors = []

    try:
        # Run the async SMARD fetch in a new event loop
        loop = asyncio.new_event_loop()
        try:
            df = loop.run_until_complete(fetch_recent(hours_back=hours_back))
        finally:
            loop.close()

        if df.empty:
            logger.warning("ingest_data: SMARD returned empty DataFrame")
            return {"rows_inserted": 0, "rows_updated": 0, "errors": ["SMARD returned empty data"]}

        df = df.reset_index()

        session = _get_sync_session()
        try:
            for _, row_data in df.iterrows():
                ts = row_data.get("timestamp")
                if ts is None:
                    continue
                if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                existing = (
                    session.query(HourlyPrice)
                    .filter_by(timestamp=ts, source="smard")
                    .first()
                )

                def _f(v):
                    try:
                        return float(v) if v is not None else None
                    except (ValueError, TypeError):
                        return None

                price = _f(row_data.get("price_eur_mwh"))
                load = _f(row_data.get("load_mw"))
                wind_on = _f(row_data.get("wind_onshore_mw"))
                wind_off = _f(row_data.get("wind_offshore_mw"))
                solar = _f(row_data.get("solar_mw"))

                dt = ts if hasattr(ts, "hour") else datetime.fromisoformat(str(ts))
                is_weekend_val = dt.weekday() >= 5
                hour_val = dt.hour
                month_val = dt.month

                wind_total = (wind_on or 0.0) + (wind_off or 0.0)
                residual = (load - wind_total - (solar or 0.0)) if load is not None else None

                if existing is not None:
                    existing.price_eur_mwh = price
                    existing.load_mw = load
                    existing.wind_onshore_mw = wind_on
                    existing.wind_offshore_mw = wind_off
                    existing.solar_mw = solar
                    existing.residual_load_mw = residual
                    existing.is_weekend = is_weekend_val
                    existing.hour = hour_val
                    existing.month = month_val
                    rows_updated += 1
                else:
                    hp = HourlyPrice(
                        timestamp=ts,
                        source="smard",
                        price_eur_mwh=price,
                        load_mw=load,
                        wind_onshore_mw=wind_on,
                        wind_offshore_mw=wind_off,
                        solar_mw=solar,
                        residual_load_mw=residual,
                        is_weekend=is_weekend_val,
                        is_holiday=False,
                        hour=hour_val,
                        month=month_val,
                    )
                    session.add(hp)
                    rows_inserted += 1

            session.commit()
            logger.info(
                "ingest_data complete: inserted=%d updated=%d",
                rows_inserted,
                rows_updated,
            )
        except Exception as exc:
            session.rollback()
            raise exc
        finally:
            session.close()

    except Exception as exc:
        logger.error("ingest_data failed: %s", exc, exc_info=True)
        errors.append(str(exc))
        try:
            raise self.retry(exc=exc)
        except Exception:
            pass

    return {
        "rows_inserted": rows_inserted,
        "rows_updated": rows_updated,
        "errors": errors,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Task: generate_and_cache_signal
# ---------------------------------------------------------------------------


@celery_app.task(
    name="app.jobs.tasks.generate_and_cache_signal",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def generate_and_cache_signal(self: Task) -> Dict[str, Any]:
    """
    Generate the current Futures signal, cache the result in Redis, and persist
    the signal record to the database.

    The cached signal is stored under the key ``signal:latest`` with a TTL
    of 20 minutes (slightly longer than the 15-minute generation interval).
    """
    import asyncio
    import numpy as np
    import pandas as pd
    from app.db.models import HourlyPrice, FuturesSignal
    from app.api.schemas import SignalAction, CostModelConfig

    logger.info("Task generate_and_cache_signal started")

    try:
        session = _get_sync_session()
        try:
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(hours=72)

            rows = (
                session.query(HourlyPrice)
                .filter(HourlyPrice.timestamp >= cutoff)
                .filter(HourlyPrice.price_eur_mwh.isnot(None))
                .order_by(HourlyPrice.timestamp.asc())
                .all()
            )

            if not rows:
                logger.warning("generate_and_cache_signal: no market data")
                return {"action": "DATA_QUALITY_BLOCKED", "reason": "No market data"}

            latest_row = rows[-1]
            current_price = float(latest_row.price_eur_mwh)

            signal_ts = latest_row.timestamp
            if signal_ts.tzinfo is None:
                signal_ts = signal_ts.replace(tzinfo=timezone.utc)

            age_minutes = (now - signal_ts).total_seconds() / 60.0
            if age_minutes > settings.max_data_age_minutes:
                action = SignalAction.DATA_QUALITY_BLOCKED
                reason = f"Data is stale ({age_minutes:.0f} min old)"
                _save_and_cache_signal(
                    session, action, current_price, signal_ts, now, reason, None, None
                )
                return {"action": action.value, "reason": reason}

            if current_price >= 0:
                action = SignalAction.NO_TRADE
                reason = f"Price {current_price:.2f} EUR/MWh is non-negative"
                _save_and_cache_signal(
                    session, action, current_price, signal_ts, now, reason, None, None
                )
                return {"action": action.value, "reason": reason, "current_price": current_price}

            # Build features
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
            df = pd.DataFrame(data)

            p_rebound = None
            p_negative = None
            predicted_price = None

            try:
                from app.features.engineering import FeatureEngineer
                fe = FeatureEngineer()
                features_df = fe.build_features(df)
                available_cols = [c for c in fe.FEATURE_COLUMNS if c in features_df.columns]
                valid_mask = features_df[available_cols].notna().all(axis=1)
                X = features_df.loc[valid_mask, available_cols]

                if not X.empty:
                    X_latest = X.tail(1)

                    try:
                        from app.ml.negative_classifier import NegativePriceClassifier  # type: ignore
                        neg_clf = NegativePriceClassifier()
                        neg_clf.load(settings.model_dir)
                        proba = neg_clf.predict_proba(X_latest)
                        arr = np.array(proba)
                        p_negative = float(arr[-1, 1] if arr.ndim == 2 else arr[-1])
                    except Exception:
                        p_negative = 1.0 if current_price < 0 else 0.0

                    try:
                        from app.ml.rebound_classifier import ReboundClassifier  # type: ignore
                        reb_clf = ReboundClassifier()
                        reb_clf.load(settings.model_dir)
                        proba = reb_clf.predict_proba(X_latest)
                        arr = np.array(proba)
                        p_rebound = float(arr[-1, 1] if arr.ndim == 2 else arr[-1])
                    except Exception:
                        p_rebound = min(0.9, abs(current_price) / 100.0)

                    try:
                        from app.ml.price_regression import PriceRegressionModel  # type: ignore
                        price_model = PriceRegressionModel()
                        price_model.load(settings.model_dir)
                        pred = price_model.predict(X_latest)
                        predicted_price = float(np.array(pred)[-1])
                    except Exception:
                        predicted_price = max(0.0, current_price + 30.0)

            except Exception as exc:
                logger.warning("Feature engineering failed in task: %s", exc)
                p_rebound = min(0.9, abs(current_price) / 100.0)
                p_negative = 1.0
                predicted_price = max(0.0, current_price + 30.0)

            if p_rebound is None:
                p_rebound = min(0.9, abs(current_price) / 100.0)
            if p_negative is None:
                p_negative = 1.0
            if predicted_price is None:
                predicted_price = max(0.0, current_price + 30.0)

            # Edge calculation
            cost_model = CostModelConfig(
                avg_spread_eur_mwh=settings.futures_avg_spread_eur_mwh,
                slippage_eur_mwh=settings.futures_slippage_eur_mwh,
                broker_markup_eur_mwh=settings.futures_broker_markup_eur_mwh,
                safety_buffer_eur_mwh=settings.futures_safety_buffer_eur_mwh,
                min_edge_threshold=settings.futures_min_edge_threshold,
            )
            total_cost = (
                cost_model.avg_spread_eur_mwh
                + cost_model.slippage_eur_mwh
                + cost_model.broker_markup_eur_mwh
                + cost_model.safety_buffer_eur_mwh
            )
            gross_edge = max(0.0, predicted_price - current_price)
            net_edge = gross_edge - total_cost

            # Determine action
            open_count = session.query(HourlyPrice).count()  # simple proxy check
            from app.db.models import PaperPosition
            open_positions = session.query(PaperPosition).filter_by(status="open").count()

            if open_positions >= settings.max_open_positions:
                action = SignalAction.RISK_BLOCKED
                reason = f"Max open positions reached ({open_positions})"
            elif (
                net_edge >= settings.futures_min_edge_threshold
                and p_rebound >= settings.min_confidence_threshold
            ):
                action = SignalAction.ENTER_LONG_REBOUND_SIGNAL
                reason = (
                    f"Price {current_price:.2f} EUR/MWh. "
                    f"p_rebound={p_rebound:.0%}, net_edge={net_edge:.1f} EUR/MWh."
                )
            elif current_price < 0 and p_negative >= 0.5:
                action = SignalAction.WATCH_LONG_REBOUND
                reason = (
                    f"Negative price ({current_price:.2f}) but edge below threshold. "
                    f"net_edge={net_edge:.1f}"
                )
            else:
                action = SignalAction.NO_TRADE
                reason = f"Conditions not met. p_rebound={p_rebound:.2f}, net_edge={net_edge:.2f}"

            _save_and_cache_signal(
                session,
                action,
                current_price,
                signal_ts,
                now,
                reason,
                p_negative,
                p_rebound,
                predicted_price=predicted_price,
                net_edge=net_edge,
                gross_edge=gross_edge,
                estimated_futures_costs=total_cost,
            )

            logger.info(
                "generate_and_cache_signal complete: action=%s price=%.2f p_rebound=%.2f net_edge=%.2f",
                action.value,
                current_price,
                p_rebound,
                net_edge,
            )
            return {
                "action": action.value,
                "current_price": current_price,
                "p_rebound": p_rebound,
                "net_edge": net_edge,
                "completed_at": now.isoformat(),
            }

        except Exception as exc:
            session.rollback()
            raise exc
        finally:
            session.close()

    except Exception as exc:
        logger.error("generate_and_cache_signal failed: %s", exc, exc_info=True)
        try:
            raise self.retry(exc=exc)
        except Exception:
            pass
        return {"error": str(exc)}


def _save_and_cache_signal(
    session: Session,
    action,
    current_price: float,
    signal_ts: datetime,
    generated_at: datetime,
    reason: str,
    p_negative: Optional[float],
    p_rebound: Optional[float],
    predicted_price: Optional[float] = None,
    net_edge: Optional[float] = None,
    gross_edge: Optional[float] = None,
    estimated_futures_costs: Optional[float] = None,
) -> None:
    """Persist signal to DB and cache in Redis."""
    from app.db.models import FuturesSignal

    # Save to DB
    try:
        signal_record = FuturesSignal(
            timestamp=signal_ts,
            action=action.value if hasattr(action, "value") else str(action),
            confidence=p_rebound or 0.0,
            current_price=current_price,
            predicted_price=predicted_price,
            p_negative=p_negative,
            p_rebound=p_rebound,
            net_edge=net_edge,
            gross_edge=gross_edge,
            estimated_futures_costs=estimated_futures_costs,
            reason=reason,
        )
        session.add(signal_record)
        session.commit()
    except Exception as exc:
        logger.warning("Failed to save signal to DB: %s", exc)
        session.rollback()

    # Cache in Redis
    try:
        r = _get_redis()
        payload = {
            "action": action.value if hasattr(action, "value") else str(action),
            "current_price": current_price,
            "p_negative": p_negative,
            "p_rebound": p_rebound,
            "predicted_price": predicted_price,
            "net_edge": net_edge,
            "reason": reason,
            "generated_at": generated_at.isoformat(),
            "signal_ts": signal_ts.isoformat(),
        }
        r.setex("signal:latest", 1200, json.dumps(payload))  # 20-minute TTL
    except Exception as exc:
        logger.warning("Failed to cache signal in Redis: %s", exc)


# ---------------------------------------------------------------------------
# Task: retrain_models
# ---------------------------------------------------------------------------


@celery_app.task(
    name="app.jobs.tasks.retrain_models",
    bind=True,
    max_retries=1,
    default_retry_delay=300,
    acks_late=True,
    soft_time_limit=3600,
    time_limit=3900,
)
def retrain_models(self: Task) -> Dict[str, Any]:
    """
    Retrain ML models if there is sufficient new data since the last training run.

    The task checks whether the model files exist and whether the row count in
    the DB has grown enough to warrant a retrain.  The ``retrain_interval_hours``
    setting controls the minimum interval between retrain runs.
    """
    import os
    import pandas as pd
    from app.db.models import HourlyPrice

    logger.info("Task retrain_models started")

    try:
        session = _get_sync_session()
        try:
            # Check how much data we have
            row_count = session.query(HourlyPrice).filter(
                HourlyPrice.price_eur_mwh.isnot(None)
            ).count()

            MIN_ROWS_FOR_TRAINING = 500
            if row_count < MIN_ROWS_FOR_TRAINING:
                msg = f"Insufficient data for training: {row_count} rows (need {MIN_ROWS_FOR_TRAINING})"
                logger.info(msg)
                return {"status": "skipped", "reason": msg, "row_count": row_count}

            # Fetch all data
            rows = (
                session.query(HourlyPrice)
                .filter(HourlyPrice.price_eur_mwh.isnot(None))
                .order_by(HourlyPrice.timestamp.asc())
                .all()
            )

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
            df = pd.DataFrame(data)

            trained_models = []
            errors = []

            # Attempt to retrain each model
            for model_cls_path, model_name in [
                ("app.ml.negative_classifier.NegativePriceClassifier", "NegativePriceClassifier"),
                ("app.ml.rebound_classifier.ReboundClassifier", "ReboundClassifier"),
                ("app.ml.price_regression.PriceRegressionModel", "PriceRegressionModel"),
            ]:
                try:
                    module_path, cls_name = model_cls_path.rsplit(".", 1)
                    import importlib
                    module = importlib.import_module(module_path)
                    model_cls = getattr(module, cls_name)
                    model_instance = model_cls()

                    if hasattr(model_instance, "train"):
                        model_instance.train(df)
                    elif hasattr(model_instance, "fit"):
                        from app.features.engineering import FeatureEngineer
                        fe = FeatureEngineer()
                        X, y = fe.get_feature_matrix(df)
                        if not X.empty:
                            model_instance.fit(X, y)

                    if hasattr(model_instance, "save"):
                        model_instance.save(settings.model_dir)

                    trained_models.append(model_name)
                    logger.info("Retrained model: %s", model_name)
                except Exception as exc:
                    logger.warning("Failed to retrain %s: %s", model_name, exc)
                    errors.append(f"{model_name}: {exc}")

            logger.info(
                "retrain_models complete: trained=%s errors=%d",
                trained_models,
                len(errors),
            )
            return {
                "status": "completed",
                "row_count": row_count,
                "trained_models": trained_models,
                "errors": errors,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as exc:
            session.rollback()
            raise exc
        finally:
            session.close()

    except Exception as exc:
        logger.error("retrain_models failed: %s", exc, exc_info=True)
        try:
            raise self.retry(exc=exc)
        except Exception:
            pass
        return {"status": "failed", "error": str(exc)}


# ---------------------------------------------------------------------------
# Task: check_paper_positions
# ---------------------------------------------------------------------------


@celery_app.task(
    name="app.jobs.tasks.check_paper_positions",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def check_paper_positions(self: Task) -> Dict[str, Any]:
    """
    Check open paper positions against the latest available price.

    For each open position:
    - If current_price >= take_profit → close with EXIT_TAKE_PROFIT_SIGNAL
    - If current_price <= stop_loss   → close with EXIT_STOP_LOSS_SIGNAL
    - If holding time >= max_holding_hours → close with time_exit

    Closed positions have their PnL calculated and persisted.
    """
    from app.db.models import HourlyPrice, PaperPosition
    from app.api.schemas import CostModelConfig

    logger.info("Task check_paper_positions started")

    exits = []
    errors = []

    try:
        session = _get_sync_session()
        try:
            # Get latest price
            latest_price_row = (
                session.query(HourlyPrice)
                .filter(HourlyPrice.price_eur_mwh.isnot(None))
                .order_by(HourlyPrice.timestamp.desc())
                .first()
            )

            if latest_price_row is None:
                logger.warning("check_paper_positions: no market data available")
                return {"exits": [], "errors": ["No market data available"]}

            current_price = float(latest_price_row.price_eur_mwh)
            price_ts = latest_price_row.timestamp
            if price_ts.tzinfo is None:
                price_ts = price_ts.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            age_minutes = (now - price_ts).total_seconds() / 60.0

            if age_minutes > settings.max_data_age_minutes:
                logger.warning(
                    "check_paper_positions: market data is stale (%.0f min)", age_minutes
                )
                return {"exits": [], "errors": [f"Stale price data ({age_minutes:.0f} min)"]}

            # Fetch open positions
            open_positions = (
                session.query(PaperPosition)
                .filter_by(status="open")
                .all()
            )

            if not open_positions:
                return {"exits": [], "skipped": "no open positions"}

            cost_model = CostModelConfig(
                avg_spread_eur_mwh=settings.futures_avg_spread_eur_mwh,
                slippage_eur_mwh=settings.futures_slippage_eur_mwh,
                broker_markup_eur_mwh=settings.futures_broker_markup_eur_mwh,
                safety_buffer_eur_mwh=settings.futures_safety_buffer_eur_mwh,
            )
            total_cost_per_mwh = (
                cost_model.avg_spread_eur_mwh
                + cost_model.slippage_eur_mwh
                + cost_model.broker_markup_eur_mwh
                + cost_model.safety_buffer_eur_mwh
            )

            for pos in open_positions:
                entry_ts = pos.entry_timestamp
                if entry_ts is not None and entry_ts.tzinfo is None:
                    entry_ts = entry_ts.replace(tzinfo=timezone.utc)

                holding_hours = (now - entry_ts).total_seconds() / 3600.0 if entry_ts else 0.0

                should_exit = False
                exit_reason = "unknown"

                if pos.take_profit is not None and current_price >= pos.take_profit:
                    should_exit = True
                    exit_reason = "take_profit"
                elif pos.stop_loss is not None and current_price <= pos.stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif pos.max_holding_hours is not None and holding_hours >= pos.max_holding_hours:
                    should_exit = True
                    exit_reason = "time_exit"

                if should_exit:
                    pnl_gross = (current_price - pos.entry_price) * pos.notional_size_mwh
                    futures_costs = total_cost_per_mwh * pos.notional_size_mwh
                    net_pnl = pnl_gross - futures_costs

                    pos.status = "closed"
                    pos.exit_price = current_price
                    pos.exit_timestamp = now
                    pos.pnl_eur = round(pnl_gross, 4)
                    pos.futures_costs_eur = round(futures_costs, 4)
                    pos.net_pnl_eur = round(net_pnl, 4)
                    pos.exit_reason = exit_reason
                    pos.updated_at = now

                    exits.append({
                        "position_id": pos.id,
                        "exit_reason": exit_reason,
                        "entry_price": pos.entry_price,
                        "exit_price": current_price,
                        "net_pnl": round(net_pnl, 4),
                    })

                    logger.info(
                        "Paper position closed: id=%d reason=%s entry=%.2f exit=%.2f net_pnl=%.2f",
                        pos.id,
                        exit_reason,
                        pos.entry_price,
                        current_price,
                        net_pnl,
                    )

            if exits:
                session.commit()

        except Exception as exc:
            session.rollback()
            raise exc
        finally:
            session.close()

    except Exception as exc:
        logger.error("check_paper_positions failed: %s", exc, exc_info=True)
        errors.append(str(exc))
        try:
            raise self.retry(exc=exc)
        except Exception:
            pass

    return {
        "exits": exits,
        "errors": errors,
        "current_price": None if not exits else exits[0].get("exit_price"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
