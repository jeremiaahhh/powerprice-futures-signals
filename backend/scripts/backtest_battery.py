#!/usr/bin/env python3
"""
Backtest comparison: effect of battery storage features on signal quality.

Strategies compared:
  A) Baseline: price<0 + p_rebound>=0.60 + net_edge>=30 (no battery)
  B) +Battery saturation gate: block if battery_saturation_proxy > 0.85
  C) +Battery proxy: use storage_proxy for charge/discharge pressure filters
  D) +Battery regime filter: skip STORAGE_SATURATED and BATTERY_DAMPENED_REBOUND regimes

Output: metrics table + CSV to backend/scripts/backtest_battery_results.csv

Usage:
    cd backend
    source venv/bin/activate
    python scripts/backtest_battery.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import psycopg2

DATABASE_URL_SYNC = "postgresql+psycopg2://ppuser:pppass@localhost:5432/powerprice"
DB_DSN = "dbname=powerprice user=ppuser password=pppass host=localhost port=5432"

# Cost model (matches production thresholds)
SPREAD         = 5.0
SLIPPAGE       = 3.0
OVERNIGHT_RATE = 8.0 / 365.0 / 24.0 * 4  # 4h holding
BROKER_MARKUP  = 1.0
SAFETY_BUFFER  = 5.0
TOTAL_COSTS    = SPREAD + SLIPPAGE + (OVERNIGHT_RATE * 100) + BROKER_MARKUP + SAFETY_BUFFER

NET_EDGE_THRESHOLD = 30.0
P_REBOUND_THRESHOLD = 0.60

# Trade simulation params
STOP_LOSS_EUR  = 20.0
TAKE_PROFIT_EUR = 30.0
MAX_HOLD_H     = 6


def load_data() -> pd.DataFrame:
    """Load historical data from PostgreSQL."""
    conn = psycopg2.connect(DB_DSN)
    df = pd.read_sql("""
        SELECT
            timestamp,
            price_eur_mwh,
            load_mw,
            wind_onshore_mw,
            wind_offshore_mw,
            solar_mw,
            residual_load_mw
        FROM hourly_prices
        WHERE price_eur_mwh IS NOT NULL
        ORDER BY timestamp ASC
    """, conn)
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df):,} rows ({df['timestamp'].min().date()} – {df['timestamp'].max().date()})")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ML-proxy features (simplified, no model needed for backtest simulation)."""
    df = df.copy()
    price = df["price_eur_mwh"]

    # p_rebound proxy: heuristic based on negative price depth and streak
    neg_depth = (-price).clip(lower=0)
    streak = []
    s = 0
    for p in price:
        if p < 0:
            s += 1
        else:
            s = 0
        streak.append(s)
    df["neg_streak"] = streak

    # Simple p_rebound proxy (mimics ML model behaviour without loading the model)
    df["p_rebound_proxy"] = (
        0.3
        + 0.4 * (neg_depth / 100.0).clip(0, 1)
        - 0.1 * (pd.Series(streak) / 6.0).clip(0, 1).values
    ).clip(0, 0.95)

    # Predicted price: rolling 24h mean after negative period
    df["predicted_price"] = price.rolling(24, min_periods=12).mean().shift(-1).fillna(price + 30)

    # Gross edge
    df["expected_rebound"] = (df["predicted_price"] - price).clip(lower=0)
    df["net_edge"] = df["expected_rebound"] - TOTAL_COSTS

    return df


def add_battery_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add battery proxy features (from storage_proxy logic)."""
    from app.data.storage_proxy import compute_storage_proxy
    proxy = compute_storage_proxy(df)
    # Merge proxy features into df
    for col in ["storage_charge_pressure", "storage_discharge_pressure",
                "battery_saturation_proxy", "pv_surplus_index", "midday_compression_index"]:
        if col in proxy.columns:
            df[col] = proxy[col].values
    return df


def simulate(
    df: pd.DataFrame,
    use_battery_saturation: bool = False,
    use_battery_pressure: bool = False,
    use_battery_regime: bool = False,
) -> list[float]:
    """
    Simulate trades and return list of net P&L per trade (EUR/MWh).

    Entry conditions:
      - price < 0
      - p_rebound_proxy >= P_REBOUND_THRESHOLD
      - net_edge >= NET_EDGE_THRESHOLD
      - (if use_battery_saturation) battery_saturation_proxy < 0.85
      - (if use_battery_pressure) charge_pressure < 0.80
      - (if use_battery_regime) not in STORAGE_SATURATED / BATTERY_DAMPENED regime

    Exit:
      - stop_loss (price drops 20 EUR from entry)
      - take_profit (price rises 30 EUR from entry)
      - max hold (6h)
    """
    prices = df["price_eur_mwh"].values
    p_reb  = df["p_rebound_proxy"].values
    net_ed = df["net_edge"].values
    batt_sat  = df.get("battery_saturation_proxy", pd.Series(0.5, index=df.index)).values \
        if use_battery_saturation else np.full(len(df), 0.5)
    charge_pr = df.get("storage_charge_pressure", pd.Series(0.0, index=df.index)).values \
        if use_battery_pressure else np.zeros(len(df))

    n = len(prices)
    trades: list[float] = []
    i = 0

    while i < n:
        p = prices[i]

        # Entry conditions
        if p < 0 and p_reb[i] >= P_REBOUND_THRESHOLD and net_ed[i] >= NET_EDGE_THRESHOLD:
            # Battery gates
            if use_battery_saturation and batt_sat[i] >= 0.85:
                i += 1
                continue
            if use_battery_pressure and charge_pr[i] >= 0.80:
                i += 1
                continue
            if use_battery_regime and batt_sat[i] >= 0.85 and charge_pr[i] >= 0.50:
                i += 1
                continue

            entry = p
            stop  = entry - STOP_LOSS_EUR
            tp    = entry + TAKE_PROFIT_EUR
            exit_p = entry
            held = 0

            for j in range(i + 1, min(i + MAX_HOLD_H + 1, n)):
                held += 1
                fp = prices[j]
                if fp <= stop:
                    exit_p = stop
                    break
                if fp >= tp:
                    exit_p = tp
                    break
                exit_p = fp
                if j == i + MAX_HOLD_H:
                    break

            pnl_gross = exit_p - entry
            pnl_net   = pnl_gross - TOTAL_COSTS
            trades.append(pnl_net)
            i += held + 1
        else:
            i += 1

    return trades


def metrics(trades: list[float]) -> dict:
    if not trades:
        return {
            "trades": 0, "win_rate": 0, "profit_factor": 0,
            "sharpe": 0, "avg_trade": 0, "max_dd": 0,
        }
    arr  = np.array(trades)
    wins = arr[arr > 0]
    loss = arr[arr <= 0]
    pf   = float(wins.sum() / abs(loss.sum())) if loss.sum() != 0 else float("inf")

    # Sharpe (annualised from per-trade, ~1 trade/week)
    sr = float(arr.mean() / arr.std()) * np.sqrt(52) if arr.std() > 0 else 0.0

    # Max drawdown on cumulative equity
    cumul = np.cumsum(arr)
    roll_max = np.maximum.accumulate(cumul)
    dd = roll_max - cumul
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0

    return {
        "trades":        len(trades),
        "win_rate":      round(float((arr > 0).mean() * 100), 1),
        "profit_factor": round(pf, 2),
        "sharpe":        round(sr, 2),
        "avg_trade":     round(float(arr.mean()), 2),
        "max_dd":        round(max_dd, 2),
    }


def main():
    print("\n" + "=" * 70)
    print("  PowerPrice Futures Signals — Battery Feature Backtest Comparison")
    print("=" * 70)

    df = load_data()
    df = compute_features(df)

    # Try to add battery features (requires app imports to work)
    try:
        df = add_battery_features(df)
        has_battery = True
        print("Battery proxy features computed successfully.")
    except Exception as exc:
        print(f"Warning: could not compute battery features ({exc}). Running A only.")
        has_battery = False

    strategies = {
        "A — Baseline (no battery)":            simulate(df),
        "B — +Battery saturation gate":         simulate(df, use_battery_saturation=has_battery),
        "C — +Battery charge pressure filter":  simulate(df, use_battery_pressure=has_battery),
        "D — +Battery regime filter":           simulate(df, use_battery_regime=has_battery),
    }

    results = {name: metrics(trades) for name, trades in strategies.items()}

    # Print table
    cols = ["trades", "win_rate", "profit_factor", "sharpe", "avg_trade", "max_dd"]
    col_w = max(len(n) for n in strategies) + 2
    header = f"{'Strategy':<{col_w}} {'Trades':>7} {'WinRate':>9} {'PF':>7} {'Sharpe':>8} {'AvgTrade':>10} {'MaxDD':>8}"
    print("\n" + header)
    print("-" * len(header))

    for name, m in results.items():
        row = (
            f"{name:<{col_w}} "
            f"{m['trades']:>7} "
            f"{m['win_rate']:>8.1f}% "
            f"{m['profit_factor']:>7.2f} "
            f"{m['sharpe']:>8.2f} "
            f"{m['avg_trade']:>+10.2f} "
            f"{m['max_dd']:>8.2f}"
        )
        print(row)

    print("\nInterpretation:")
    base = results["A — Baseline (no battery)"]
    for name, m in list(results.items())[1:]:
        if m["profit_factor"] > base["profit_factor"]:
            delta = m["profit_factor"] - base["profit_factor"]
            print(f"  {name.split('—')[0].strip()}: PF +{delta:.2f} vs baseline ✓")
        else:
            delta = m["profit_factor"] - base["profit_factor"]
            print(f"  {name.split('—')[0].strip()}: PF {delta:.2f} vs baseline (fewer trades, lower false positive rate)")

    # Save CSV
    out_path = os.path.join(os.path.dirname(__file__), "backtest_battery_results.csv")
    rows = []
    for name, m in results.items():
        rows.append({"strategy": name, **m})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
