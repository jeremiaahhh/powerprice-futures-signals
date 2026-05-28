"""
Telegram command handler — polls getUpdates and dispatches /commands.

Supported commands:
  /hilfe       — Befehlsliste
  /status      — Daemon-Status + aktuelles Signal
  /signal      — Signal jetzt generieren
  /backtest [tage] — ML-Backtest der letzten N Tage (default 30)
  /performance — Rolling PF, Win Rate, Sharpe
  /drift       — Drift-Report
  /modelle     — Modell-Infos (AUC, Zeitpunkt)
  /stop        — Daemon anhalten
  /start       — Daemon starten

Security: only responds to the configured TELEGRAM_CHAT_ID.
SIGNAL ONLY — no live order execution, ever.
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from app.notifications.telegram_client import TelegramClient

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_DISCLAIMER = "Signal only. Keine Order ausgeführt."
_API_BASE = "http://localhost:8000"


def _h(text: str) -> str:
    """HTML-escape a string for Telegram."""
    return html.escape(str(text))


def _fmt_float(v: Any, decimals: int = 2) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


class TelegramCommandHandler:
    """
    Polls Telegram getUpdates and dispatches recognised /commands.

    Run as a background asyncio task alongside the daemon's main loop.
    """

    def __init__(self, client: "TelegramClient") -> None:
        self._client = client
        self._offset: int = 0
        self._running: bool = True
        self._base_url = (
            f"https://api.telegram.org/bot{settings.telegram_bot_token}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def poll_forever(self) -> None:
        """Long-poll getUpdates in a loop until stopped."""
        logger.info("Telegram command handler started (polling)")
        while self._running:
            try:
                updates = await self._get_updates()
                for upd in updates:
                    self._offset = upd["update_id"] + 1
                    await self._dispatch(upd)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Command poll error: %s", exc)
                await asyncio.sleep(5)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Telegram API helpers
    # ------------------------------------------------------------------

    async def _get_updates(self) -> List[Dict]:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=35.0) as c:
                resp = await c.get(
                    f"{self._base_url}/getUpdates",
                    params={"offset": self._offset, "timeout": 30, "allowed_updates": ["message"]},
                )
                if resp.status_code == 200:
                    return resp.json().get("result", [])
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        return []

    async def _reply(self, text: str) -> None:
        await self._client.send_html(text)

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    async def _dispatch(self, update: Dict) -> None:
        msg = update.get("message", {})
        chat_id = str(msg.get("chat", {}).get("id", ""))
        text = (msg.get("text") or "").strip()

        if not text.startswith("/"):
            return

        # Security: only respond to configured chat
        if chat_id != str(settings.telegram_chat_id):
            logger.debug("Ignoring command from unknown chat %s", chat_id)
            return

        parts = text.split()
        cmd = parts[0].lstrip("/").lower().split("@")[0]
        args = parts[1:]

        handlers = {
            "hilfe": self._cmd_hilfe,
            "help": self._cmd_hilfe,
            "info": self._cmd_info,
            "status": self._cmd_status,
            "signal": self._cmd_signal,
            "backtest": self._cmd_backtest,
            "performance": self._cmd_performance,
            "drift": self._cmd_drift,
            "modelle": self._cmd_modelle,
            "stop": self._cmd_stop,
            "start": self._cmd_start,
        }

        handler = handlers.get(cmd)
        if handler:
            try:
                await handler(args)
            except Exception as exc:
                logger.error("Command /%s failed: %s", cmd, exc)
                await self._reply(
                    f"<b>Fehler bei /{_h(cmd)}</b>\n<code>{_h(str(exc)[:300])}</code>"
                )
        else:
            await self._reply(
                f"Unbekannter Befehl: <code>/{_h(cmd)}</code>\n"
                f"Tippe /hilfe für die Befehlsliste."
            )

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _cmd_hilfe(self, args: List[str]) -> None:
        await self._reply(
            "<b>PowerPrice — Befehle</b>\n\n"
            "/info — Vollständige Übersicht aller Befehle &amp; Funktionen\n"
            "/status — Daemon-Status &amp; aktuelles Signal\n"
            "/signal — Signal jetzt generieren\n"
            "/backtest [tage] — ML-Backtest (default: 30 Tage)\n"
            "/performance — Rolling-Performance (PF, Win Rate)\n"
            "/drift — Drift-Report\n"
            "/modelle — Modell-Infos\n"
            "/stop — Daemon anhalten\n"
            "/start — Daemon starten\n"
            "/hilfe — Diese Liste\n\n"
            f"<i>{_DISCLAIMER}</i>"
        )

    async def _cmd_info(self, args: List[str]) -> None:
        await self._reply(
            "<b>PowerPrice Futures Signals — Systemübersicht</b>\n\n"
            "Deutsches Strompreis-ML-Signalsystem. Erkennt negative "
            "Preisphasen und Rebound-Chancen auf Basis von SMARD/ENTSO-E-Daten.\n\n"

            "━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>SIGNAL-BEFEHLE</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "/signal\n"
            "  → Generiert ein Signal auf Basis aktueller Marktdaten.\n"
            "  Gibt aus: Aktion, Preis, p_rebound, Net Edge, Grund.\n\n"
            "/status\n"
            "  → Daemon-Status: läuft/gestoppt, Modus (NORMAL/WATCH_ONLY),\n"
            "  Zyklen, letztes Signal, nächster Lauf, Fehlerzähler.\n\n"

            "━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>ANALYSE-BEFEHLE</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "/backtest [tage]\n"
            "  → ML-Rebound-Backtest der letzten N Tage (7–365, default 30).\n"
            "  Gibt aus: Sharpe Ratio, Profit Factor, Win Rate,\n"
            "  Max Drawdown, Ø Trade, bester/schlechtester Trade.\n\n"
            "/performance\n"
            "  → Rolling-Performance der letzten 30 Tage (Shadow-Trades).\n"
            "  Gibt aus: Profit Factor, Win Rate, Anzahl Trades, Signal-Modus.\n\n"
            "/drift\n"
            "  → Drift-Report der ML-Modelle.\n"
            "  Erkennt Feature-Drift, Prediction-Drift, Performance-Drift.\n"
            "  Schweregrade: LOW / MEDIUM / HIGH.\n\n"
            "/modelle\n"
            "  → Modell-Registry: AUC NegativePriceClassifier,\n"
            "  AUC ReboundClassifier, Trainingszeitpunkt.\n"
            "  Zeigt Produktions- und Kandidaten-Modell.\n\n"

            "━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>STEUERUNG</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "/stop\n"
            "  → Setzt Stop-Signal. Daemon beendet aktuellen Zyklus\n"
            "  und hält dann sauber an.\n\n"
            "/start\n"
            "  → Entfernt Stop-Signal. Daemon läuft beim nächsten\n"
            "  Start (oder LaunchAgent-Neustart) normal weiter.\n\n"

            "━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>ML-MODELLE</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "• <b>NegativePriceClassifier</b> — erkennt negative Preisphasen (AUC 0.97)\n"
            "• <b>ReboundClassifier</b> — schätzt Rebound-Wahrscheinlichkeit (AUC 0.98)\n"
            "• <b>PriceRegressionModel</b> — Preisvorhersage +4h (R²=0.07, nicht primär)\n"
            "Automatisches Retraining alle 24h oder bei Drift-Erkennung.\n\n"

            "━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>SIGNAL-TYPEN</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "HIGH_CONFIDENCE_SIGNAL — p_rebound ≥ 0.85, Preis negativ\n"
            "ENTER_LONG_REBOUND_SIGNAL — p_rebound ≥ 0.70, Net Edge ≥ 30\n"
            "WATCH_LONG_REBOUND — p_rebound ≥ 0.55 (beobachten)\n"
            "NO_TRADE — kein Setup\n"
            "TAIL_RISK_BLOCKED — Tail-Risk zu hoch\n"
            "GAP_RISK_BLOCKED — Gap-Risk &gt; 0.80\n"
            "DATA_QUALITY_BLOCKED — Datenlücken &gt; 20%\n\n"

            f"<i>{_DISCLAIMER}</i>"
        )

    async def _cmd_status(self, args: List[str]) -> None:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=8.0) as c:
                r = await c.get(f"{_API_BASE}/daemon/status")
                s = r.json()
        except Exception as exc:
            await self._reply(f"Status nicht abrufbar: {_h(str(exc))}")
            return

        running = s.get("running", False)
        state = "RUNNING" if running else "STOPPED"
        last_signal = s.get("last_signal") or "—"
        mode = s.get("signal_mode") or "—"
        cycle = s.get("cycle_count") or 0
        last_run = s.get("last_run_at") or "—"
        next_run = s.get("next_run_at") or "—"
        errors = s.get("consecutive_errors") or 0

        def _fmt_ts(ts: str) -> str:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.strftime("%H:%M UTC")
            except Exception:
                return ts[:16] if ts else "—"

        await self._reply(
            f"<b>Daemon-Status: {state}</b>\n\n"
            f"Läuft: <b>{'Ja' if running else 'Nein'}</b>\n"
            f"Modus: <code>{_h(mode)}</code>\n"
            f"Zyklen: {cycle}\n"
            f"Letztes Signal: <code>{_h(last_signal)}</code>\n"
            f"Letzter Lauf: {_fmt_ts(last_run)}\n"
            f"Nächster Lauf: {_fmt_ts(next_run)}\n"
            f"Fehler (konsekutiv): {errors}\n\n"
            f"<i>{_DISCLAIMER}</i>"
        )

    async def _cmd_signal(self, args: List[str]) -> None:
        await self._reply("Generiere Signal…")
        try:
            import httpx
            async with httpx.AsyncClient(timeout=20.0) as c:
                r = await c.get(f"{_API_BASE}/futures/signal")
                sig = r.json()
        except Exception as exc:
            await self._reply(f"Signal-Generierung fehlgeschlagen: {_h(str(exc))}")
            return

        action = sig.get("action") or "—"
        price = sig.get("current_price")
        p_reb = sig.get("p_rebound")
        net_edge = sig.get("net_edge")
        reason = sig.get("reason") or ""

        lines = [
            f"<b>Signal: {_h(action)}</b>",
            "",
            f"Preis: <b>{_fmt_float(price)} EUR/MWh</b>" if price is not None else "Preis: —",
        ]
        if p_reb is not None:
            lines.append(f"p_rebound: {_fmt_float(p_reb, 3)}")
        if net_edge is not None:
            lines.append(f"Net Edge: {_fmt_float(net_edge)} EUR/MWh")
        if reason:
            lines.append(f"\n<i>{_h(reason[:200])}</i>")
        lines.append(f"\n<i>{_DISCLAIMER}</i>")

        await self._reply("\n".join(lines))

    async def _cmd_backtest(self, args: List[str]) -> None:
        days = 30
        if args:
            try:
                days = max(7, min(365, int(args[0])))
            except ValueError:
                pass

        await self._reply(f"Starte ML-Backtest ({days} Tage)…")

        try:
            import httpx
            end_dt = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=days)
            payload = {
                "strategy": "ml_rebound",
                "start_date": start_dt.isoformat(),
                "end_date": end_dt.isoformat(),
                "p_rebound_threshold": settings.futures_p_rebound_entry,
                "min_edge_threshold": settings.futures_min_edge_threshold,
            }
            async with httpx.AsyncClient(timeout=120.0) as c:
                r = await c.post(f"{_API_BASE}/backtest/run", json=payload)
                if r.status_code != 200:
                    await self._reply(f"Backtest-Fehler: HTTP {r.status_code}\n<code>{_h(r.text[:300])}</code>")
                    return
                result = r.json()
        except Exception as exc:
            await self._reply(f"Backtest fehlgeschlagen: {_h(str(exc))}")
            return

        metrics = result.get("metrics") or result
        sharpe = _fmt_float(metrics.get("sharpe_ratio"))
        pf = _fmt_float(metrics.get("profit_factor"))
        wr = _fmt_float(metrics.get("win_rate_pct"))
        dd = _fmt_float(metrics.get("max_drawdown_pct"))
        trades = metrics.get("total_trades") or 0
        avg = _fmt_float(metrics.get("avg_trade_eur_mwh"))
        worst = _fmt_float(metrics.get("worst_trade_eur_mwh"))
        best = _fmt_float(metrics.get("best_trade_eur_mwh"))
        ret = _fmt_float(metrics.get("total_return_pct"))

        await self._reply(
            f"<b>Backtest ML-Rebound — {days} Tage</b>\n\n"
            f"Sharpe Ratio:    <b>{sharpe}</b>\n"
            f"Profit Factor:   <b>{pf}</b>\n"
            f"Win Rate:        <b>{wr}%</b>\n"
            f"Max Drawdown:    {dd}%\n"
            f"Gesamt-Return:   {ret}%\n\n"
            f"Trades gesamt:   {trades}\n"
            f"Ø Trade:         {avg} EUR/MWh\n"
            f"Bester Trade:    {best} EUR/MWh\n"
            f"Schlechtster:    {worst} EUR/MWh\n\n"
            f"<i>{_DISCLAIMER}</i>"
        )

    async def _cmd_performance(self, args: List[str]) -> None:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(f"{_API_BASE}/adaptation/rolling-performance")
                perf = r.json()
        except Exception as exc:
            await self._reply(f"Performance nicht abrufbar: {_h(str(exc))}")
            return

        pf = _fmt_float(perf.get("rolling_pf"))
        wr = _fmt_float((perf.get("rolling_win_rate") or 0) * 100)
        n = perf.get("sample_size") or 0
        mode = perf.get("signal_mode") or "—"
        window = perf.get("window_days") or 30

        await self._reply(
            f"<b>Rolling-Performance ({window}d)</b>\n\n"
            f"Profit Factor:  <b>{pf}</b>\n"
            f"Win Rate:       <b>{wr}%</b>\n"
            f"Trades (Shadow): {n}\n"
            f"Signal-Modus:   <code>{_h(mode)}</code>\n\n"
            f"<i>{_DISCLAIMER}</i>"
        )

    async def _cmd_drift(self, args: List[str]) -> None:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(f"{_API_BASE}/adaptation/drift-report")
                drift = r.json()
        except Exception as exc:
            await self._reply(f"Drift-Report nicht abrufbar: {_h(str(exc))}")
            return

        has_drift = drift.get("has_drift", False)
        severity = drift.get("severity") or "LOW"
        drift_types = drift.get("drift_types") or []
        checked_at = drift.get("checked_at") or ""

        lines = [
            "<b>Drift-Report</b>",
            "",
            f"Drift erkannt: {'Ja' if has_drift else 'Nein'}",
            f"Schweregrad: <b>{_h(severity)}</b>",
        ]
        if drift_types:
            lines.append(f"Typen: {', '.join(_h(t) for t in drift_types)}")
        if checked_at:
            try:
                dt = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
                lines.append(f"Geprüft: {dt.strftime('%d.%m. %H:%M UTC')}")
            except Exception:
                pass
        lines.append(f"\n<i>{_DISCLAIMER}</i>")

        await self._reply("\n".join(lines))

    async def _cmd_modelle(self, args: List[str]) -> None:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(f"{_API_BASE}/adaptation/model-registry")
                reg = r.json()
        except Exception as exc:
            await self._reply(f"Modell-Registry nicht abrufbar: {_h(str(exc))}")
            return

        prod = reg.get("production") or {}
        cand = reg.get("candidate")

        def _model_block(label: str, m: Dict) -> str:
            if not m:
                return f"<b>{label}</b>: —"
            auc_neg = _fmt_float(m.get("auc_neg"), 4)
            auc_reb = _fmt_float(m.get("auc_rebound"), 4)
            trained = m.get("trained_at") or ""
            try:
                dt = datetime.fromisoformat(trained.replace("Z", "+00:00"))
                trained = dt.strftime("%d.%m. %H:%M UTC")
            except Exception:
                pass
            return (
                f"<b>{label}</b>\n"
                f"  AUC Negativ:  {auc_neg}\n"
                f"  AUC Rebound:  {auc_reb}\n"
                f"  Trainiert:    {_h(trained)}"
            )

        lines = ["<b>Modell-Registry</b>", ""]
        lines.append(_model_block("Produktion", prod))
        if cand:
            lines.append("")
            lines.append(_model_block("Kandidat", cand))
        lines.append(f"\n<i>{_DISCLAIMER}</i>")

        await self._reply("\n".join(lines))

    async def _cmd_stop(self, args: List[str]) -> None:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=8.0) as c:
                r = await c.post(f"{_API_BASE}/daemon/stop")
                if r.status_code == 200:
                    await self._reply("<b>Daemon wird angehalten.</b>\n\n<i>Stop-Signal gesetzt. Aktueller Zyklus wird abgeschlossen.</i>")
                else:
                    await self._reply(f"Stop fehlgeschlagen: HTTP {r.status_code}")
        except Exception as exc:
            await self._reply(f"Stop fehlgeschlagen: {_h(str(exc))}")

    async def _cmd_start(self, args: List[str]) -> None:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=8.0) as c:
                r = await c.post(f"{_API_BASE}/daemon/start")
                if r.status_code == 200:
                    await self._reply("<b>Daemon-Stop-Signal entfernt.</b>\n\n<i>Der Daemon läuft weiter oder wird beim nächsten Start normal beginnen.</i>")
                else:
                    await self._reply(f"Start fehlgeschlagen: HTTP {r.status_code}")
        except Exception as exc:
            await self._reply(f"Start fehlgeschlagen: {_h(str(exc))}")
