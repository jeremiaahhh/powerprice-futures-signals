"""
Signal Daemon – permanent adaptive signal loop.

Run as standalone process:
    python -m app.runtime.signal_daemon

Or via Docker:
    command: python -m app.runtime.signal_daemon

The daemon runs until:
- SIGTERM / SIGINT (graceful shutdown)
- daemon_stop.signal file appears
- MAX_CONSECUTIVE_ERRORS exceeded

SIGNAL ONLY – no live trading, no order execution.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure the backend package is importable when run as __main__
_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent.parent  # backend/
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.core.config import settings
from app.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

_DISCLAIMER = "Signal only. Keine Order ausgeführt."


def _check_on_battery() -> bool:
    """Return True if the system is currently running on battery (not plugged in)."""
    try:
        import psutil
        battery = psutil.sensors_battery()
        if battery is None:
            return False  # no battery → desktop/server, treat as AC
        return not battery.power_plugged
    except Exception:
        return False


class SignalDaemon:
    """
    Permanent adaptive signal loop.

    SIGNAL ONLY – never executes real trades.
    """

    def __init__(self) -> None:
        self._running = False
        self._shutdown_requested = False
        self._started_at: Optional[datetime] = None
        self._last_run_at: Optional[datetime] = None
        self._next_run_at: Optional[datetime] = None
        self._cycle_count = 0
        self._consecutive_errors = 0
        self._last_error: Optional[str] = None
        self._last_signal: Optional[str] = None
        self._last_signal_at: Optional[datetime] = None
        self._last_drift_check: Optional[datetime] = None
        self._last_retrain: Optional[datetime] = None
        self._last_daily_summary: Optional[datetime] = None
        self._signal_mode: str = settings.signal_mode
        self._telegram_sent_today: int = 0
        self._loop_interval_s: int = settings.signal_loop_interval_minutes * 60
        self._on_battery: bool = False

        # Services (lazy init)
        self._notifier: Optional[Any] = None
        self._evaluator: Optional[Any] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _setup_signals(self) -> None:
        def _handle(signum, frame):
            logger.info("Signal %d received — requesting graceful shutdown", signum)
            self._shutdown_requested = True

        signal.signal(signal.SIGTERM, _handle)
        signal.signal(signal.SIGINT, _handle)

    def _check_stop_signal(self) -> bool:
        return Path(settings.daemon_stop_signal_file).exists()

    def _init_services(self) -> None:
        try:
            from app.notifications.notification_service import NotificationService
            self._notifier = NotificationService()
        except Exception as exc:
            logger.warning("NotificationService not available: %s", exc)
            self._notifier = None

        try:
            from app.runtime.shadow_outcome_evaluator import ShadowOutcomeEvaluator
            self._evaluator = ShadowOutcomeEvaluator()
        except Exception as exc:
            logger.warning("ShadowOutcomeEvaluator not available: %s", exc)
            self._evaluator = None

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _write_state(self) -> None:
        try:
            state_path = Path(settings.daemon_state_file)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "running": self._running,
                "pid": os.getpid(),
                "started_at": self._started_at.isoformat() if self._started_at else None,
                "last_run_at": self._last_run_at.isoformat() if self._last_run_at else None,
                "next_run_at": self._next_run_at.isoformat() if self._next_run_at else None,
                "cycle_count": self._cycle_count,
                "consecutive_errors": self._consecutive_errors,
                "last_error": self._last_error,
                "last_signal": self._last_signal,
                "last_signal_at": self._last_signal_at.isoformat() if self._last_signal_at else None,
                "telegram_enabled": settings.telegram_enabled,
                "auto_retrain_enabled": settings.auto_retrain_enabled,
                "signal_mode": self._signal_mode,
                "telegram_sent_today": self._telegram_sent_today,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            state_path.write_text(json.dumps(state, indent=2))
        except Exception as exc:
            logger.debug("State write failed: %s", exc)

    # ------------------------------------------------------------------
    # Individual cycle steps (each wrapped in try/except)
    # ------------------------------------------------------------------

    async def _step_ingest_data(self) -> None:
        """Step 1: Fetch latest market data."""
        try:
            import pandas as pd
            from app.data.smard import fetch_recent
            from app.data import openmeteo
            from app.data.ingestion import _merge_sources
            from app.data.holidays import is_german_holiday
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            from app.db.models import HourlyPrice
            from app.db.session import get_db

            from app.data import entsoe as entsoe_mod

            now_utc = datetime.now(timezone.utc)
            entsoe_start = now_utc - timedelta(hours=6)

            smard_df, weather_df, entsoe_load_df = await asyncio.gather(
                fetch_recent(hours_back=4),
                openmeteo.fetch_historical(hours_back=4),
                entsoe_mod.fetch_actual_load(entsoe_start, now_utc),
                return_exceptions=True,
            )
            if isinstance(smard_df, Exception):
                smard_df = pd.DataFrame()
            if isinstance(weather_df, Exception):
                weather_df = pd.DataFrame()
            if isinstance(entsoe_load_df, Exception):
                entsoe_load_df = pd.DataFrame()

            # Resample ENTSO-E 15-min actual load to hourly mean
            if not entsoe_load_df.empty and "load_mw" in entsoe_load_df.columns:
                try:
                    entsoe_hourly = (
                        entsoe_load_df["load_mw"]
                        .resample("1h")
                        .mean()
                        .rename("load_mw_entsoe")
                    )
                    _entsoe_map = entsoe_hourly.to_dict()
                except Exception:
                    _entsoe_map = {}
            else:
                _entsoe_map = {}

            merged = _merge_sources(smard_df, pd.DataFrame(), weather_df)
            if merged.empty:
                return

            rows = []
            for ts, row in merged.iterrows():
                # Keep UTC timezone so PostgreSQL stores at correct UTC offset
                dt = pd.Timestamp(ts).to_pydatetime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                wk = dt.weekday()

                def sf(v):
                    try:
                        return float(v) if v is not None and not pd.isna(v) else None
                    except Exception:
                        return None

                wind_on = sf(row.get("wind_onshore_mw"))
                wind_off = sf(row.get("wind_offshore_mw"))
                solar = sf(row.get("solar_mw"))
                # Prefer ENTSO-E actual load over Open-Meteo proxy
                dt_utc_floor = dt.replace(minute=0, second=0, microsecond=0)
                load = sf(_entsoe_map.get(dt_utc_floor)) or sf(row.get("load_mw"))
                residual = None
                if all(v is not None for v in [load, wind_on, solar]):
                    residual = load - (wind_on or 0) - (wind_off or 0) - solar

                rows.append({
                    "timestamp": dt, "source": "entsoe" if _entsoe_map.get(dt_utc_floor) else "smard",
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
                            set_={k: pg_insert(HourlyPrice).excluded[k] for k in rows[0] if k != "timestamp"},
                        )
                    )
                    await db.execute(stmt)
                    await db.commit()
                    break
                logger.info("Daemon: ingested %d rows", len(rows))
        except Exception as exc:
            logger.error("Daemon step 1 (ingest) failed: %s", exc)

    async def _step_generate_signal(self) -> Optional[Dict]:
        """Steps 3-6: Build features, detect regime, assess tail risk, generate signal."""
        try:
            from app.api.routes.futures import _generate_signal
            from app.db.session import get_db

            async for db in get_db():
                sig = await _generate_signal(db, include_explanation=False)
                return sig.__dict__ if hasattr(sig, "__dict__") else dict(sig)
        except Exception as exc:
            logger.error("Daemon step 3-6 (signal) failed: %s", exc)
            return None

    async def _step_persist_signal(self, sig: Dict) -> Optional[int]:
        """Step 7: Save signal and shadow signal to DB. Returns signal_id."""
        try:
            from app.db.models import FuturesSignal, ShadowSignal
            from app.db.session import get_db

            action = sig.get("action") or "NO_TRADE"
            # Don't persist consecutive identical NO_TRADE / DATA_QUALITY_BLOCKED
            if action in ("NO_TRADE", "DATA_QUALITY_BLOCKED") and self._last_signal == action:
                return None

            async for db in get_db():
                now = datetime.now(timezone.utc)

                # FuturesSignal
                record = FuturesSignal(
                    timestamp=sig.get("timestamp") or now,
                    action=str(action),
                    confidence=sig.get("confidence") or 0.0,
                    current_price=sig.get("current_price"),
                    predicted_price=sig.get("predicted_price"),
                    p_negative=sig.get("p_negative"),
                    p_rebound=sig.get("p_rebound"),
                    expected_rebound_eur_mwh=sig.get("expected_rebound_eur_mwh"),
                    gross_edge=sig.get("gross_edge"),
                    estimated_futures_costs=sig.get("estimated_futures_costs"),
                    net_edge=sig.get("net_edge"),
                    stop_loss=sig.get("stop_loss"),
                    take_profit=sig.get("take_profit"),
                    max_holding_hours=sig.get("max_holding_hours"),
                    reason=sig.get("reason"),
                    risk_warnings=sig.get("risk_warnings"),
                )
                db.add(record)
                await db.flush()
                signal_id = record.id

                # ShadowSignal (if shadow mode enabled)
                if settings.shadow_mode_enabled:
                    shadow = ShadowSignal(
                        timestamp=sig.get("timestamp") or now,
                        action=str(action),
                        current_price=sig.get("current_price"),
                        predicted_price=sig.get("predicted_price"),
                        p_rebound=sig.get("p_rebound"),
                        p_negative=sig.get("p_negative"),
                        net_edge=sig.get("net_edge"),
                    )
                    # Set optional new columns if they exist on the model
                    try:
                        shadow.tail_risk_score = sig.get("tail_risk_score")
                        shadow.sent_to_telegram = False
                    except Exception:
                        pass
                    db.add(shadow)

                await db.commit()
                logger.info("Daemon: signal persisted: %s (id=%s)", action, signal_id)
                return signal_id
        except Exception as exc:
            logger.error("Daemon step 7 (persist) failed: %s", exc)
            return None

    async def _step_send_telegram(self, sig: Dict, signal_id: Optional[int]) -> None:
        """Step 9: Send Telegram notification if enabled and signal is relevant."""
        if self._notifier is None or not settings.telegram_enabled:
            return
        try:
            sent = await self._notifier.send_signal(sig)
            if sent:
                self._telegram_sent_today += 1
                if signal_id:
                    # Mark sent_to_telegram in shadow_signals
                    try:
                        import psycopg2
                        conn = psycopg2.connect("postgresql://ppuser:pppass@localhost:5432/powerprice")
                        try:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "UPDATE shadow_signals SET sent_to_telegram = TRUE WHERE id = "
                                    "(SELECT id FROM shadow_signals ORDER BY created_at DESC LIMIT 1)"
                                )
                            conn.commit()
                        finally:
                            conn.close()
                    except Exception:
                        pass
        except Exception as exc:
            logger.error("Daemon step 9 (telegram) failed: %s", exc)

    async def _step_evaluate_outcomes(self) -> None:
        """Step 8: Evaluate shadow outcomes and update performance metrics."""
        if self._evaluator is None:
            return
        try:
            count = self._evaluator.evaluate_pending()
            if count > 0:
                logger.debug("Daemon: evaluated %d shadow outcomes", count)
        except Exception as exc:
            logger.error("Daemon step 8 (outcomes) failed: %s", exc)

    async def _step_check_performance(self) -> None:
        """Step 10: Check rolling performance and adjust signal mode."""
        if self._evaluator is None:
            return
        try:
            perf = self._evaluator.compute_rolling_performance(window=settings.signal_mode_rolling_window)
            rolling_pf = perf.get("rolling_pf")

            if rolling_pf is not None:
                if rolling_pf < settings.signal_mode_pf_floor and self._signal_mode != "WATCH_ONLY":
                    self._signal_mode = "WATCH_ONLY"
                    logger.warning(
                        "Signal mode → WATCH_ONLY (rolling PF=%.3f < %.2f)",
                        rolling_pf, settings.signal_mode_pf_floor,
                    )
                    if self._notifier:
                        await self._notifier.send_error_alert(
                            f"Signal mode switched to WATCH_ONLY — rolling PF={rolling_pf:.3f}",
                            {"rolling_win_rate": perf.get("rolling_win_rate"), "sample": perf.get("sample_size")}
                        )
                elif rolling_pf >= settings.signal_mode_pf_recovery and self._signal_mode == "WATCH_ONLY":
                    self._signal_mode = "NORMAL"
                    logger.info(
                        "Signal mode → NORMAL (rolling PF=%.3f >= %.2f)",
                        rolling_pf, settings.signal_mode_pf_recovery,
                    )
        except Exception as exc:
            logger.error("Daemon step 10 (performance) failed: %s", exc)

    async def _step_check_drift(self) -> None:
        """Step 10b: Periodic drift check."""
        if not settings.drift_check_enabled:
            return
        now = datetime.now(timezone.utc)
        if (self._last_drift_check and
                (now - self._last_drift_check).total_seconds() < settings.drift_check_interval_hours * 3600):
            return

        self._last_drift_check = now
        try:
            from app.adaptation.drift_detector import DriftDetector
            detector = DriftDetector()
            report = detector.check()

            if report.has_drift and self._notifier:
                await self._notifier.send_drift_alert({
                    "drift_types": report.drift_types,
                    "severity": report.severity,
                    "details": report.details,
                })

            if report.has_drift and report.severity in ("MEDIUM", "HIGH"):
                await self._step_retrain(reason=f"drift_detected:{report.severity}")
        except Exception as exc:
            logger.error("Daemon step 10b (drift) failed: %s", exc)

    async def _step_retrain(self, reason: str = "scheduled") -> None:
        """Step 11: Run retraining if needed."""
        if not settings.auto_retrain_enabled:
            return
        now = datetime.now(timezone.utc)
        if (self._last_retrain and
                (now - self._last_retrain).total_seconds() < settings.retrain_interval_hours * 3600):
            return

        self._last_retrain = now
        try:
            from app.adaptation.retrain_scheduler import RetrainScheduler
            scheduler = RetrainScheduler()
            if not scheduler.should_retrain(drift_detected="drift" in reason):
                return

            logger.info("Daemon: starting retraining (reason=%s)", reason)
            loop = asyncio.get_event_loop()
            # Run sync training in thread executor to avoid blocking the loop
            report = await loop.run_in_executor(None, scheduler.run, reason)

            if report and self._notifier:
                await self._notifier.send_retrain_report(report)
        except Exception as exc:
            logger.error("Daemon step 11 (retrain) failed: %s", exc)

    async def _step_daily_summary(self) -> None:
        """Send daily summary once per day."""
        if self._notifier is None or not settings.telegram_send_daily_summary:
            return
        now = datetime.now(timezone.utc)
        if (self._last_daily_summary and
                (now - self._last_daily_summary).total_seconds() < 86400):
            return
        # Send only between 07:00 and 09:00 UTC
        if not (7 <= now.hour <= 9):
            return

        self._last_daily_summary = now
        try:
            perf = {"rolling_pf": None, "rolling_win_rate": None}
            if self._evaluator:
                perf = self._evaluator.compute_rolling_performance()

            summary = {
                "signals_today": self._cycle_count,
                "enter_signals": 0,  # TODO: track today's counts
                "blocked_signals": 0,
                "rolling_pf": perf.get("rolling_pf"),
                "rolling_win_rate": (perf.get("rolling_win_rate") or 0) * 100 if perf.get("rolling_win_rate") else None,
                "current_regime": "UNKNOWN",
                "signal_mode": self._signal_mode,
            }
            await self._notifier.send_daily_summary(summary)
        except Exception as exc:
            logger.error("Daemon step daily_summary failed: %s", exc)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run_cycle(self) -> None:
        """Execute one full signal cycle (12 steps)."""
        self._cycle_count += 1
        self._last_run_at = datetime.now(timezone.utc)

        # Step 1: Ingest data
        await self._step_ingest_data()

        # Steps 3-6: Signal generation (includes feature build, regime, tail risk)
        sig = await self._step_generate_signal()

        if sig:
            action = str(sig.get("action") or "NO_TRADE")
            self._last_signal = action
            self._last_signal_at = datetime.now(timezone.utc)

            # Step 7: Persist
            signal_id = await self._step_persist_signal(sig)

            # Step 9: Telegram
            await self._step_send_telegram(sig, signal_id)

        # Step 8: Evaluate outcomes
        await self._step_evaluate_outcomes()

        # Step 10: Performance check
        await self._step_check_performance()

        # Step 10b: Drift check (time-gated)
        await self._step_check_drift()

        # Step 11: Retrain (time-gated)
        await self._step_retrain(reason="time_based")

        # Daily summary
        await self._step_daily_summary()

        # Update state file
        self._write_state()
        logger.info(
            "Daemon cycle %d complete: signal=%s mode=%s errors=%d",
            self._cycle_count, self._last_signal, self._signal_mode, self._consecutive_errors,
        )

    async def run(self) -> None:
        """Main daemon entry point. Runs until stop requested."""
        self._setup_signals()
        self._init_services()
        self._running = True
        self._started_at = datetime.now(timezone.utc)
        os.makedirs(os.path.dirname(settings.daemon_state_file), exist_ok=True)

        logger.info(
            "=== PowerPrice Signal Daemon starting (PID=%d) ===\n"
            "    Interval: %d min | Telegram: %s | AutoRetrain: %s\n"
            "    %s",
            os.getpid(), settings.signal_loop_interval_minutes,
            settings.telegram_enabled, settings.auto_retrain_enabled,
            _DISCLAIMER,
        )
        self._write_state()

        # Start Telegram command handler as background task
        _cmd_task: Optional[asyncio.Task] = None
        if settings.telegram_enabled:
            try:
                from app.notifications.telegram_commands import TelegramCommandHandler
                _tg_client = getattr(self._notifier, "_client", None)
                if _tg_client is None:
                    from app.notifications.telegram_client import TelegramClient
                    _tg_client = TelegramClient()
                _cmd_handler = TelegramCommandHandler(_tg_client)
                _cmd_task = asyncio.create_task(_cmd_handler.poll_forever())
                logger.info("Telegram command handler started (background task)")
            except Exception as exc:
                logger.warning("Could not start Telegram command handler: %s", exc)

        backoff_s = self._loop_interval_s

        while not self._shutdown_requested:
            if self._check_stop_signal():
                logger.info("Daemon: stop signal file detected — shutting down")
                break

            cycle_start = time.monotonic()
            try:
                await self._run_cycle()
                self._consecutive_errors = 0
                backoff_s = self._loop_interval_s
            except Exception as exc:
                self._consecutive_errors += 1
                self._last_error = str(exc)
                logger.exception(
                    "Daemon cycle failed (consecutive_errors=%d): %s",
                    self._consecutive_errors, exc,
                )

                if self._consecutive_errors >= settings.max_consecutive_errors:
                    if self._notifier:
                        try:
                            await self._notifier.send_error_alert(
                                f"Daemon: {self._consecutive_errors} consecutive errors",
                                {"last_error": str(exc)[:200]}
                            )
                        except Exception:
                            pass
                    backoff_s = min(backoff_s * 2, 3600)
                    logger.error(
                        "Daemon backing off to %ds after %d errors",
                        backoff_s, self._consecutive_errors,
                    )

            # Battery-aware interval: extend sleep when on battery power
            self._on_battery = _check_on_battery()
            effective_interval_s = (
                settings.signal_loop_interval_battery_minutes * 60
                if self._on_battery
                else self._loop_interval_s
            )
            if self._on_battery:
                logger.debug("Battery power detected — using extended interval %ds", effective_interval_s)

            # Sleep until next cycle
            elapsed = time.monotonic() - cycle_start
            sleep_s = max(0, effective_interval_s - elapsed)
            self._next_run_at = datetime.now(timezone.utc) + timedelta(seconds=sleep_s)
            self._write_state()

            if sleep_s > 0:
                logger.debug("Daemon sleeping %.0fs until next cycle", sleep_s)
                slept = 0.0
                while slept < sleep_s and not self._shutdown_requested and not self._check_stop_signal():
                    chunk = min(10.0, sleep_s - slept)
                    wall_before = time.time()
                    await asyncio.sleep(chunk)
                    wall_after = time.time()
                    slept += chunk

                    # Wake-from-sleep detection: wall clock jumped more than expected
                    if settings.wake_detection_enabled:
                        expected_wall = chunk + settings.wake_clock_jump_threshold_seconds
                        if wall_after - wall_before > expected_wall:
                            logger.info(
                                "Wake-from-sleep detected (wall jump %.0fs > %.0fs threshold) — "
                                "applying %ds post-wake grace period",
                                wall_after - wall_before,
                                expected_wall,
                                settings.post_wake_grace_seconds,
                            )
                            await asyncio.sleep(settings.post_wake_grace_seconds)
                            logger.info("Post-wake grace period complete — resuming cycle immediately")
                            break  # run next cycle now, don't wait for remaining sleep

        # Cancel Telegram command handler
        if _cmd_task is not None and not _cmd_task.done():
            _cmd_task.cancel()
            try:
                await _cmd_task
            except asyncio.CancelledError:
                pass
            logger.info("Telegram command handler stopped")

        # Graceful shutdown
        self._running = False
        self._write_state()
        logger.info(
            "=== PowerPrice Signal Daemon stopped (%d cycles completed) ===",
            self._cycle_count,
        )

    async def stop(self) -> None:
        self._shutdown_requested = True


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

_daemon_instance: Optional[SignalDaemon] = None


def get_daemon() -> Optional[SignalDaemon]:
    return _daemon_instance


if __name__ == "__main__":
    _daemon_instance = SignalDaemon()
    asyncio.run(_daemon_instance.run())
