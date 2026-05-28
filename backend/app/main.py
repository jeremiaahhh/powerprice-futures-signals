import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.base import Base
from app.db.session import async_engine, get_db
from app.api.routes import health, data, forecast, futures, backtest, paper, regime, analytics, shadow, battery, risk

# Optional new routers (loaded when modules are available)
try:
    from app.api.routes import daemon as daemon_router
    _daemon_router = daemon_router
except ImportError:
    _daemon_router = None

try:
    from app.api.routes import notifications as notifications_router
    _notifications_router = notifications_router
except ImportError:
    _notifications_router = None

try:
    from app.api.routes import adaptation as adaptation_router
    _adaptation_router = adaptation_router
except ImportError:
    _adaptation_router = None

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background scheduler jobs
# ---------------------------------------------------------------------------

async def _job_ingest_data() -> None:
    """Hourly: fetch latest prices + weather and upsert."""
    try:
        import pandas as pd
        from app.data.smard import fetch_recent
        from app.data import openmeteo
        from app.data.ingestion import _merge_sources
        from app.data.holidays import is_german_holiday
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from app.db.models import HourlyPrice

        smard_df, weather_df = await asyncio.gather(
            fetch_recent(hours_back=4),
            openmeteo.fetch_historical(hours_back=4),
            return_exceptions=True,
        )
        if isinstance(smard_df, Exception):
            smard_df = pd.DataFrame()
        if isinstance(weather_df, Exception):
            weather_df = pd.DataFrame()

        merged = _merge_sources(smard_df, pd.DataFrame(), weather_df)
        if merged.empty:
            return

        rows = []
        for ts, row in merged.iterrows():
            dt = pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None)
            wk = dt.weekday()

            def sf(v):
                try:
                    return float(v) if v is not None and not pd.isna(v) else None
                except Exception:
                    return None

            wind_on = sf(row.get("wind_onshore_mw"))
            wind_off = sf(row.get("wind_offshore_mw"))
            solar = sf(row.get("solar_mw"))
            load = sf(row.get("load_mw"))
            residual = None
            if all(v is not None for v in [load, wind_on, solar]):
                residual = load - (wind_on or 0) - (wind_off or 0) - solar

            rows.append({
                "timestamp": dt,
                "source": "smard",
                "price_eur_mwh": sf(row.get("price_eur_mwh")),
                "load_mw": load, "wind_onshore_mw": wind_on,
                "wind_offshore_mw": wind_off, "solar_mw": solar,
                "residual_load_mw": residual,
                "temperature_c": sf(row.get("temperature_c")),
                "wind_speed_ms": sf(row.get("wind_speed_ms")),
                "solar_radiation_wm2": sf(row.get("solar_radiation_wm2")),
                "cloud_cover_pct": sf(row.get("cloud_cover_pct")),
                "is_holiday": is_german_holiday(dt),
                "is_weekend": wk >= 5,
                "hour": dt.hour, "month": dt.month,
            })

        if rows:
            async for db in get_db():
                stmt = (
                    pg_insert(HourlyPrice).values(rows)
                    .on_conflict_do_update(
                        index_elements=["timestamp"],
                        set_={k: pg_insert(HourlyPrice).excluded[k]
                              for k in rows[0] if k != "timestamp"},
                    )
                )
                await db.execute(stmt)
                await db.commit()
                break
            logger.info("Scheduler: upserted %d rows", len(rows))
    except Exception as exc:
        logger.error("Scheduler ingest job failed: %s", exc)


async def _job_generate_signal() -> None:
    """Every 15 min: generate signal and persist to futures_signals table."""
    try:
        from app.api.routes.futures import _generate_signal
        from app.db.models import FuturesSignal
        from app.db.session import get_db

        async for db in get_db():
            sig = await _generate_signal(db, include_explanation=False)
            record = FuturesSignal(
                timestamp=sig.timestamp,
                action=sig.action,
                confidence=sig.confidence,
                current_price=sig.current_price,
                predicted_price=sig.predicted_price,
                p_negative=sig.p_negative,
                p_rebound=sig.p_rebound,
                expected_rebound_eur_mwh=sig.expected_rebound_eur_mwh,
                gross_edge=sig.gross_edge,
                estimated_futures_costs=sig.estimated_futures_costs,
                net_edge=sig.net_edge,
                stop_loss=sig.stop_loss,
                take_profit=sig.take_profit,
                max_holding_hours=sig.max_holding_hours,
                reason=sig.reason,
                risk_warnings=sig.risk_warnings,
            )
            db.add(record)
            await db.commit()
            logger.info("Scheduler: signal persisted: %s", sig.action)
            break
    except Exception as exc:
        logger.error("Scheduler signal job failed: %s", exc)


async def _job_save_regime_snapshot() -> None:
    """Every 15 min: classify current regime and persist a RegimeSnapshot."""
    try:
        import pandas as pd
        from datetime import timedelta
        from sqlalchemy import select
        from app.db.models import HourlyPrice, RegimeSnapshot
        from app.db.session import get_db
        from app.features.engineering import FeatureEngineer
        from app.regime import RegimeClassifier

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=72)

        async for db in get_db():
            stmt = (
                select(HourlyPrice)
                .where(HourlyPrice.timestamp >= cutoff, HourlyPrice.price_eur_mwh.isnot(None))
                .order_by(HourlyPrice.timestamp.asc())
            )
            result = await db.execute(stmt)
            rows = result.scalars().all()

            if not rows:
                logger.warning("Regime snapshot: no data available")
                break

            def sf(v):
                try:
                    return float(v) if v is not None and not pd.isna(v) else None
                except Exception:
                    return None

            data = []
            for r in rows:
                data.append({
                    "timestamp": r.timestamp,
                    "price_eur_mwh": sf(r.price_eur_mwh),
                    "load_mw": sf(r.load_mw),
                    "wind_onshore_mw": sf(r.wind_onshore_mw),
                    "wind_offshore_mw": sf(r.wind_offshore_mw),
                    "solar_mw": sf(r.solar_mw),
                    "residual_load_mw": sf(r.residual_load_mw),
                    "net_export_mw": sf(r.net_export_mw) or 0.0,
                    "temperature_c": sf(r.temperature_c),
                    "wind_speed_ms": sf(r.wind_speed_ms),
                    "solar_radiation_wm2": sf(r.solar_radiation_wm2),
                    "cloud_cover_pct": sf(r.cloud_cover_pct),
                    "battery_net_mw": sf(getattr(r, "battery_net_mw", None)) or 0.0,
                    "is_holiday": int(r.is_holiday) if r.is_holiday is not None else 0,
                    "is_weekend": int(r.is_weekend) if r.is_weekend is not None else 0,
                    "hour": r.hour,
                    "month": r.month,
                })
            df = pd.DataFrame(data)

            fe = FeatureEngineer()
            features_df = fe.build_features(df)
            available_cols = [c for c in fe.FEATURE_COLUMNS if c in features_df.columns]
            valid_mask = features_df[available_cols].notna().all(axis=1)
            valid_rows = features_df[valid_mask]
            if valid_rows.empty:
                logger.warning("Regime snapshot: insufficient features")
                break

            feature_row = valid_rows.iloc[-1]
            clf = RegimeClassifier()
            regime = clf.classify(feature_row)

            # Compute renewable_share from raw last row for snapshot
            last = df.iloc[-1]
            wind_t = (last.get("wind_onshore_mw") or 0.0) + (last.get("wind_offshore_mw") or 0.0)
            sol = last.get("solar_mw") or 0.0
            load = last.get("load_mw") or 0.0
            ren_share = round((wind_t + sol) / load, 4) if load > 0 else None
            prices_24 = df["price_eur_mwh"].dropna().tail(24)
            price_vol = round(float(prices_24.std()), 2) if len(prices_24) >= 2 else None
            hours_neg = int((prices_24 < 0).sum())

            snapshot = RegimeSnapshot(
                timestamp=now,
                regime=regime.regime.value,
                confidence=regime.confidence,
                renewable_share=ren_share,
                price_volatility_24h=price_vol,
                hours_negative_24h=float(hours_neg),
                solar_mw=sol,
                wind_mw=round(wind_t, 1),
                oversupply_index=regime.oversupply_index,
            )
            db.add(snapshot)
            await db.commit()
            logger.info("Scheduler: regime snapshot saved: %s (conf=%.2f)", regime.regime.value, regime.confidence)
            break
    except Exception as exc:
        logger.error("Scheduler regime snapshot job failed: %s", exc)


async def _run_scheduler() -> None:
    """Background coroutine: ingest every hour, signal every 15 min."""
    ingest_interval = 3600   # 1 hour
    signal_interval = 900    # 15 minutes
    last_ingest = 0.0
    last_signal = 0.0
    import time
    # Run immediately on startup
    await _job_ingest_data()
    await _job_generate_signal()
    await _job_save_regime_snapshot()
    last_ingest = time.monotonic()
    last_signal = time.monotonic()

    while True:
        await asyncio.sleep(60)  # check every minute
        now = time.monotonic()
        if now - last_ingest >= ingest_interval:
            await _job_ingest_data()
            last_ingest = now
        if now - last_signal >= signal_interval:
            await _job_generate_signal()
            await _job_save_regime_snapshot()
            last_signal = now


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting PowerPrice Futures Signals API", extra={"env": settings.app_env})
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified")
    # Idempotent column migrations for new fields
    from sqlalchemy import text
    async with async_engine.begin() as conn:
        for ddl in [
            "ALTER TABLE hourly_prices ADD COLUMN IF NOT EXISTS battery_charge_mw FLOAT",
            "ALTER TABLE hourly_prices ADD COLUMN IF NOT EXISTS battery_discharge_mw FLOAT",
            "ALTER TABLE hourly_prices ADD COLUMN IF NOT EXISTS battery_net_mw FLOAT",
            "CREATE TABLE IF NOT EXISTS tail_events (id SERIAL PRIMARY KEY, timestamp TIMESTAMPTZ NOT NULL, event_type VARCHAR(50) NOT NULL, current_price FLOAT, tail_risk_score FLOAT, gap_risk_score FLOAT, negative_price_streak INT, max_price_gap_1h FLOAT, volatility_24h FLOAT, oversupply_stress_index FLOAT, block_reason VARCHAR(100), block_detail TEXT, would_have_entered BOOLEAN, realized_outcome_6h FLOAT, regime VARCHAR(50), created_at TIMESTAMPTZ DEFAULT NOW())",
            "CREATE TABLE IF NOT EXISTS blocked_trades (id SERIAL PRIMARY KEY, timestamp TIMESTAMPTZ NOT NULL, block_reason VARCHAR(50) NOT NULL, current_price FLOAT, p_rebound FLOAT, net_edge FLOAT, tail_risk_score FLOAT, gap_risk_score FLOAT, negative_price_streak INT, regime VARCHAR(50), block_detail TEXT, price_6h_later FLOAT, would_have_won BOOLEAN, missed_pnl FLOAT, created_at TIMESTAMPTZ DEFAULT NOW())",
            "CREATE INDEX IF NOT EXISTS ix_tail_events_timestamp ON tail_events(timestamp)",
            "CREATE INDEX IF NOT EXISTS ix_blocked_trades_timestamp ON blocked_trades(timestamp)",
            "ALTER TABLE shadow_signals ADD COLUMN IF NOT EXISTS tail_risk_score FLOAT",
            "ALTER TABLE shadow_signals ADD COLUMN IF NOT EXISTS sent_to_telegram BOOLEAN DEFAULT FALSE",
            "ALTER TABLE shadow_signals ADD COLUMN IF NOT EXISTS reason_json JSONB",
            "CREATE TABLE IF NOT EXISTS shadow_outcomes (id SERIAL PRIMARY KEY, signal_id INTEGER REFERENCES shadow_signals(id), evaluated_at TIMESTAMPTZ NOT NULL, realized_price_1h FLOAT, realized_price_2h FLOAT, realized_price_4h FLOAT, realized_rebound FLOAT, simulated_pnl FLOAT, would_hit_stop BOOLEAN, would_hit_take_profit BOOLEAN, outcome_status VARCHAR(30), created_at TIMESTAMPTZ DEFAULT NOW())",
            "CREATE INDEX IF NOT EXISTS ix_shadow_outcomes_signal_id ON shadow_outcomes(signal_id)",
            "CREATE TABLE IF NOT EXISTS notification_events (id SERIAL PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL, channel VARCHAR(50) NOT NULL, event_type VARCHAR(50) NOT NULL, signal_id INTEGER, fingerprint VARCHAR(100), payload JSONB, status VARCHAR(20) NOT NULL DEFAULT 'sent', error_message TEXT)",
            "CREATE INDEX IF NOT EXISTS ix_notification_events_created_at ON notification_events(created_at)",
            "CREATE TABLE IF NOT EXISTS daemon_health_logs (id SERIAL PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL, cycle_count INTEGER, consecutive_errors INTEGER DEFAULT 0, last_signal VARCHAR(60), signal_mode VARCHAR(20), rolling_pf FLOAT, rolling_win_rate FLOAT, telegram_sent_today INTEGER DEFAULT 0, notes TEXT)",
            "CREATE INDEX IF NOT EXISTS ix_daemon_health_created_at ON daemon_health_logs(created_at)",
        ]:
            await conn.execute(text(ddl))

    import os
    try:
        os.makedirs(os.path.dirname(settings.daemon_state_file), exist_ok=True)
    except OSError:
        pass  # read-only fs in some environments; daemon writes locally
    scheduler_task = asyncio.create_task(_run_scheduler())
    logger.info("Background scheduler started (ingest=1h, signal=15min)")

    yield

    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutting down")
    await async_engine.dispose()


app = FastAPI(
    title="PowerPrice Futures Signals",
    description="Data-driven signal platform for German electricity price Futures trading. SIGNAL ONLY - no live execution.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(health.router, tags=["Health"])
app.include_router(data.router, prefix="/data", tags=["Data"])
app.include_router(forecast.router, prefix="/forecast", tags=["Forecast"])
app.include_router(futures.router, prefix="/futures", tags=["Futures"])
app.include_router(backtest.router, prefix="/backtest", tags=["Backtest"])
app.include_router(paper.router, prefix="/paper", tags=["Paper Trading"])
app.include_router(regime.router, prefix="/regime", tags=["Regime"])
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
app.include_router(shadow.router, prefix="/shadow", tags=["Shadow"])
app.include_router(battery.router, prefix="/battery", tags=["Battery"])
app.include_router(risk.router, prefix="/risk", tags=["Risk"])
if _daemon_router is not None:
    app.include_router(_daemon_router.router, prefix="/daemon", tags=["Daemon"])
if _notifications_router is not None:
    app.include_router(_notifications_router.router, prefix="/notifications", tags=["Notifications"])
if _adaptation_router is not None:
    app.include_router(_adaptation_router.router, prefix="/adaptation", tags=["Adaptation"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
