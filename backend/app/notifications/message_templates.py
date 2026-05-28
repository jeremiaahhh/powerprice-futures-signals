"""
HTML message templates for Telegram notifications.
All messages include the SIGNAL ONLY disclaimer.
Special characters are HTML-escaped.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _esc(s: Any) -> str:
    """HTML-escape a value for safe Telegram output."""
    return html.escape(str(s)) if s is not None else "–"


def _price(v: Optional[float]) -> str:
    if v is None:
        return "–"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f} EUR/MWh"


def _pct(v: Optional[float]) -> str:
    if v is None:
        return "–"
    return f"{v:.1f}%"


def _risk_level(score: Optional[float]) -> str:
    if score is None:
        return "UNKNOWN"
    if score < 0.30:
        return "LOW"
    if score < 0.65:
        return "MEDIUM"
    return "HIGH"


def format_signal(signal: Dict[str, Any]) -> str:
    """Format a Futures signal for Telegram. SIGNAL ONLY."""
    action = signal.get("action", "UNKNOWN")
    price = signal.get("current_price")
    p_rebound = signal.get("p_rebound")
    net_edge = signal.get("net_edge")
    tail_risk = signal.get("tail_risk_score")
    regime = signal.get("regime", "–")
    stop = signal.get("stop_loss")
    take_profit = signal.get("take_profit")
    max_hold = signal.get("max_holding_hours")
    reason = signal.get("reason", "")
    risk_warnings = signal.get("risk_warnings") or []

    lines = [
        "<b>Strompreis Signal DE-LU</b>",
        "",
        f"<b>Signal: {_esc(action)}</b>",
        f"Preis: <b>{_price(price)}</b>",
    ]
    if p_rebound is not None:
        lines.append(f"p_rebound: <b>{p_rebound:.2f}</b>")
    if net_edge is not None:
        lines.append(f"Net Edge: <b>{_price(net_edge)}</b>")
    lines.append(f"Tail Risk: <b>{_risk_level(tail_risk)}</b>")
    lines.append(f"Regime: <code>{_esc(regime)}</code>")
    if stop is not None:
        lines.append(f"Stop: {_price(stop)}")
    if take_profit is not None:
        lines.append(f"Take Profit: {_price(take_profit)}")
    if max_hold is not None:
        lines.append(f"Max Holding: {max_hold}h")

    if reason:
        lines.append("")
        lines.append("<b>Begründung:</b>")
        for part in reason.split(","):
            part = part.strip()
            if part:
                lines.append(f"- {_esc(part)}")

    if risk_warnings:
        lines.append("")
        lines.append("<b>Risiko-Warnungen:</b>")
        for w in risk_warnings[:5]:
            lines.append(f"- {_esc(w)}")

    lines.append("")
    lines.append("<i>Signal only. Keine Order ausgeführt.</i>")
    lines.append(f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>")
    return "\n".join(lines)


def format_error_alert(error: str, context: Optional[Dict] = None) -> str:
    """Format a daemon error alert."""
    lines = [
        "<b>PowerPrice Daemon — Fehler</b>",
        "",
        f"<code>{_esc(error[:500])}</code>",
    ]
    if context:
        for k, v in list(context.items())[:5]:
            lines.append(f"- {_esc(k)}: {_esc(v)}")
    lines.append("")
    lines.append("<i>Signal only. Keine Order ausgeführt.</i>")
    lines.append(f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>")
    return "\n".join(lines)


def format_daily_summary(summary: Dict[str, Any]) -> str:
    """Format daily performance summary."""
    signals_today = summary.get("signals_today", 0)
    enters = summary.get("enter_signals", 0)
    blocked = summary.get("blocked_signals", 0)
    rolling_pf = summary.get("rolling_pf")
    rolling_wr = summary.get("rolling_win_rate")
    regime = summary.get("current_regime", "–")
    signal_mode = summary.get("signal_mode", "NORMAL")

    lines = [
        "<b>PowerPrice Tages-Summary</b>",
        "",
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "",
        f"Signale heute:    <b>{signals_today}</b>",
        f"ENTER-Signale:    <b>{enters}</b>",
        f"Blockiert:        <b>{blocked}</b>",
        "",
        f"Rolling PF (30):  <b>{rolling_pf:.3f}</b>" if rolling_pf else "Rolling PF (30):  –",
        f"Rolling Win Rate: <b>{_pct(rolling_wr)}</b>" if rolling_wr else "Rolling Win Rate: –",
        "",
        f"Aktuelles Regime: <code>{_esc(regime)}</code>",
        f"Signal Mode:      <code>{_esc(signal_mode)}</code>",
        "",
        "<i>Signal only. Keine Order ausgeführt.</i>",
    ]
    return "\n".join(lines)


def format_retrain_report(report: Dict[str, Any]) -> str:
    """Format a model retraining notification."""
    promoted = report.get("promoted", False)
    model = report.get("model", "unknown")
    new_pf = report.get("new_pf")
    old_pf = report.get("old_pf")
    new_wr = report.get("new_win_rate")
    reason = report.get("reason", "")

    status = "Promoviert" if promoted else "Kandidat (nicht promoviert)"

    lines = [
        "<b>Modell Retraining Report</b>",
        "",
        f"Modell: <code>{_esc(model)}</code>",
        f"Status: <b>{status}</b>",
    ]
    if new_pf is not None:
        lines.append(f"Neues PF: <b>{new_pf:.3f}</b>" + (f" (alt: {old_pf:.3f})" if old_pf else ""))
    if new_wr is not None:
        lines.append(f"Neuer Win Rate: <b>{_pct(new_wr)}</b>")
    if reason:
        lines.append("")
        lines.append(f"Grund: {_esc(reason)}")
    lines.append("")
    lines.append("<i>Signal only. Keine Order ausgeführt.</i>")
    lines.append(f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>")
    return "\n".join(lines)


def format_drift_alert(report: Dict[str, Any]) -> str:
    """Format a drift detection alert."""
    drift_types = report.get("drift_types", [])
    severity = report.get("severity", "MEDIUM")
    details = report.get("details", {})

    lines = [
        f"<b>Drift erkannt — Severity: {_esc(severity)}</b>",
        "",
    ]
    for dt in drift_types[:5]:
        lines.append(f"- {_esc(dt)}")
    if details:
        lines.append("")
        for k, v in list(details.items())[:5]:
            lines.append(f"  {_esc(k)}: {_esc(v)}")
    lines.append("")
    lines.append("<i>Retraining wird geprüft.</i>")
    lines.append("<i>Signal only. Keine Order ausgeführt.</i>")
    return "\n".join(lines)
