#!/usr/bin/env python3
"""
Threshold grid search: maximize profit factor after Futures costs.

Grid:
  p_negative : 0.50 – 0.90  (step 0.05)  →  9 values
  p_rebound  : 0.55 – 0.90  (step 0.05)  →  8 values
  net_edge   : 5 – 30 EUR/MWh (step 5)   →  6 values
Total: 432 combinations.

Metric: profit_factor = gross_wins / abs(gross_losses)
        after deducting actual Futures costs per trade.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import numpy as np
import pandas as pd
import psycopg2

from app.ml.negative_price_classifier import NegativePriceClassifier
from app.ml.rebound_classifier import ReboundClassifier
from app.ml.price_regression import PriceRegressionModel
from app.features.engineering import FeatureEngineer
from app.core.config import settings

DSN = "postgresql://ppuser:pppass@localhost:5432/powerprice"

# ---------------------------------------------------------------------------
# 1. Load full history
# ---------------------------------------------------------------------------

print("Loading full historical data from DB...")
conn = psycopg2.connect(DSN)
with conn.cursor() as cur:
    cur.execute("""
        SELECT timestamp, price_eur_mwh, load_mw, wind_onshore_mw,
               wind_offshore_mw, solar_mw, residual_load_mw, net_export_mw,
               temperature_c, wind_speed_ms, solar_radiation_wm2, cloud_cover_pct,
               is_holiday, is_weekend, hour, month
        FROM hourly_prices
        WHERE price_eur_mwh IS NOT NULL
        ORDER BY timestamp ASC
    """)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
conn.close()

df = pd.DataFrame(rows, columns=cols)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
print(f"Loaded {len(df)} rows  {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
print(f"Negative price hours: {(df['price_eur_mwh'] < 0).sum()}")

# ---------------------------------------------------------------------------
# 2. Build features & batch-predict
# ---------------------------------------------------------------------------

print("\nBuilding features...")
fe = FeatureEngineer()
features_df = fe.build_features(df)
feat_cols = [c for c in fe.FEATURE_COLUMNS if c in features_df.columns]

valid_mask = features_df[feat_cols].notna().all(axis=1)
X_valid = features_df.loc[valid_mask, feat_cols].fillna(0.0)
print(f"Valid feature rows: {valid_mask.sum()} / {len(df)}  (warmup skipped: {(~valid_mask).sum()})")

print("Loading models and generating batch predictions...")
neg_clf = NegativePriceClassifier(model_dir=settings.model_dir)
reb_clf = ReboundClassifier(model_dir=settings.model_dir)
reg_mdl = PriceRegressionModel(model_dir=settings.model_dir)
neg_clf.load()
reb_clf.load()
reg_mdl.load()

X_neg_s = neg_clf.scaler.transform(X_valid)
X_reb_s = reb_clf.scaler.transform(X_valid)
X_reg_s = reg_mdl.scaler.transform(X_valid)

p_neg_valid = neg_clf.model.predict_proba(X_neg_s)[:, 1]
p_reb_valid = reb_clf.model.predict_proba(X_reb_s)[:, 1]
pred_price_valid = reg_mdl.model.predict(X_reg_s)

preds = pd.DataFrame({
    "p_negative":      p_neg_valid,
    "p_rebound":       p_reb_valid,
    "predicted_price": pred_price_valid,
}, index=X_valid.index).reindex(df.index, fill_value=0.0)

# ---------------------------------------------------------------------------
# 3. Prepare numpy arrays for fast simulation
# ---------------------------------------------------------------------------

df_idx = df.set_index("timestamp").sort_index()
preds.index = df_idx.index

prices       = df_idx["price_eur_mwh"].values.astype(float)
is_weekends  = df_idx["is_weekend"].values.astype(bool)
p_neg_arr    = preds["p_negative"].values.astype(float)
p_reb_arr    = preds["p_rebound"].values.astype(float)
pred_p_arr   = preds["predicted_price"].values.astype(float)

# Futures cost constants (mirrors CFDCostModel defaults)
_SPREAD      = 5.0   # EUR/MWh round-trip
_SLIPPAGE    = 3.0
_MARKUP      = 1.0
_SAFETY      = 5.0
_FIXED_COST  = _SPREAD + _SLIPPAGE + _MARKUP + _SAFETY   # 14.0 EUR/MWh
_FIN_HOURLY  = (0.08 / 365.0 / 24.0)                    # annualised 8%
_FIN_WE_MUL  = 1.5
_MAX_HOLD    = 6
_SL_BUF      = 8.0
_TP_MUL      = 2.0
_MIN_TRADES  = 5     # discard configurations with too few trades


def _total_cost(holding_hours: float, is_wknd: bool, price_abs: float) -> float:
    rate = _FIN_HOURLY * (_FIN_WE_MUL if is_wknd else 1.0)
    return _FIXED_COST + rate * holding_hours * price_abs


def simulate(p_neg_thr: float, p_reb_thr: float, net_edge_thr: float) -> list[float]:
    """Return list of net P&L per trade for the given threshold combination."""
    trades: list[float] = []
    n = len(prices)
    in_pos = False
    entry_price = 0.0
    entry_i = 0
    sl = 0.0
    tp = 0.0

    for i in range(n):
        px = prices[i]
        if not np.isfinite(px):
            continue

        if not in_pos:
            if px >= 0.0:
                continue
            if p_neg_arr[i] < p_neg_thr:
                continue
            if p_reb_arr[i] < p_reb_thr:
                continue

            raw_reb = max(0.0, pred_p_arr[i] - px)
            exp_reb = raw_reb * p_reb_arr[i]
            entry_cost = _total_cost(_MAX_HOLD, is_weekends[i], abs(px) or 100.0)
            net_edge = exp_reb - entry_cost
            if net_edge < net_edge_thr:
                continue

            in_pos     = True
            entry_price = px
            entry_i    = i
            sl         = entry_price - _SL_BUF
            tp         = entry_price + net_edge * _TP_MUL
        else:
            hours_held = i - entry_i
            exit_trigger = (
                px <= sl
                or px >= 0.0
                or px >= tp
                or hours_held >= _MAX_HOLD
            )
            if exit_trigger:
                exit_cost = _total_cost(float(hours_held), is_weekends[i], abs(entry_price) or 100.0)
                pnl_net = (px - entry_price) - exit_cost
                trades.append(pnl_net)
                in_pos = False

    return trades


def metrics(pnls: list[float]) -> dict:
    if len(pnls) < _MIN_TRADES:
        return {}
    arr = np.array(pnls)
    wins   = arr[arr > 0]
    losses = arr[arr <= 0]
    gp = wins.sum() if len(wins) > 0 else 0.0
    gl = abs(losses.sum()) if len(losses) > 0 else 0.0
    pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 1.0)
    wr = len(wins) / len(arr) * 100.0
    avg = arr.mean()
    std = arr.std(ddof=1) if len(arr) > 1 else 1e-9
    sharpe = avg / std * np.sqrt(len(arr))
    total_pnl = arr.sum()
    # Max drawdown on cumulative P&L
    cumulative = np.cumsum(arr)
    peak = np.maximum.accumulate(cumulative)
    dd = (peak - cumulative)
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0
    return {
        "profit_factor": round(pf, 3),
        "win_rate":       round(wr, 1),
        "avg_trade":      round(avg, 2),
        "sharpe":         round(sharpe, 3),
        "n_trades":       len(arr),
        "total_pnl":      round(total_pnl, 2),
        "max_drawdown":   round(max_dd, 2),
    }


# ---------------------------------------------------------------------------
# 4. Grid search
# ---------------------------------------------------------------------------

P_NEG_GRID  = np.round(np.arange(0.50, 0.91, 0.05), 2)
P_REB_GRID  = np.round(np.arange(0.55, 0.91, 0.05), 2)
NET_EDG_GRID = [5, 10, 15, 20, 25, 30]

total = len(P_NEG_GRID) * len(P_REB_GRID) * len(NET_EDG_GRID)
print(f"\nGrid search: {len(P_NEG_GRID)} × {len(P_REB_GRID)} × {len(NET_EDG_GRID)} = {total} combinations")
print(f"Data rows: {len(prices)}")

results = []
t0 = time.perf_counter()

for pn in P_NEG_GRID:
    for pr in P_REB_GRID:
        for ne in NET_EDG_GRID:
            pnls = simulate(float(pn), float(pr), float(ne))
            m = metrics(pnls)
            if m:
                m["p_negative"] = pn
                m["p_rebound"]  = pr
                m["net_edge"]   = ne
                results.append(m)

elapsed = time.perf_counter() - t0
print(f"Done in {elapsed:.1f}s  —  {len(results)} valid configurations (≥{_MIN_TRADES} trades)")

if not results:
    print("No configurations produced enough trades. Lower _MIN_TRADES or check predictions.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------

results_df = pd.DataFrame(results).sort_values("profit_factor", ascending=False)

# Format for display
print("\n" + "=" * 90)
print("TOP 20 THRESHOLD COMBINATIONS  —  ranked by Profit Factor after Futures costs")
print("=" * 90)
print(f"{'Rank':>4}  {'p_neg':>6}  {'p_reb':>6}  {'net_edge':>9}  "
      f"{'PF':>6}  {'WinRate':>8}  {'Sharpe':>7}  {'AvgTrade':>9}  "
      f"{'Trades':>7}  {'MaxDD':>8}  {'TotalPnL':>9}")
print("-" * 90)

for rank, (_, row) in enumerate(results_df.head(20).iterrows(), 1):
    pf_str = f"{row['profit_factor']:.3f}" if row['profit_factor'] != float("inf") else "  ∞   "
    print(f"{rank:>4}  {row['p_negative']:>6.2f}  {row['p_rebound']:>6.2f}  "
          f"{row['net_edge']:>9.0f}  {pf_str:>6}  "
          f"{row['win_rate']:>7.1f}%  {row['sharpe']:>7.3f}  "
          f"{row['avg_trade']:>+9.2f}  {int(row['n_trades']):>7}  "
          f"{row['max_drawdown']:>8.2f}  {row['total_pnl']:>+9.2f}")

# Best by profit factor (at least 10 trades for statistical significance)
robust = results_df[results_df["n_trades"] >= 10]
if not robust.empty:
    best = robust.iloc[0]
    print("\n" + "=" * 90)
    print(f"RECOMMENDED (highest PF with ≥10 trades):")
    print(f"  p_negative  = {best['p_negative']:.2f}")
    print(f"  p_rebound   = {best['p_rebound']:.2f}")
    print(f"  net_edge    = {best['net_edge']:.0f} EUR/MWh")
    print(f"  PF          = {best['profit_factor']:.3f}")
    print(f"  Win rate    = {best['win_rate']:.1f}%")
    print(f"  Sharpe      = {best['sharpe']:.3f}")
    print(f"  Avg trade   = {best['avg_trade']:+.2f} EUR/MWh")
    print(f"  Total PnL   = {best['total_pnl']:+.2f} EUR/MWh")
    print(f"  Trades      = {int(best['n_trades'])}")
    print(f"  Max drawdown= {best['max_drawdown']:.2f} EUR/MWh")

# Show sensitivity: fix best p_neg/p_reb, vary net_edge
print("\n" + "=" * 90)
print("SENSITIVITY: top p_neg/p_reb pair — all net_edge values")
print("=" * 90)
if not robust.empty:
    top_pn = robust.iloc[0]["p_negative"]
    top_pr = robust.iloc[0]["p_rebound"]
    sens = results_df[(results_df["p_negative"] == top_pn) & (results_df["p_rebound"] == top_pr)]
    for _, row in sens.sort_values("net_edge").iterrows():
        pf_str = f"{row['profit_factor']:.3f}" if row['profit_factor'] != float("inf") else "  ∞   "
        print(f"  net_edge={row['net_edge']:.0f}  PF={pf_str}  "
              f"WinRate={row['win_rate']:.1f}%  AvgTrade={row['avg_trade']:+.2f}  "
              f"Trades={int(row['n_trades'])}  TotalPnL={row['total_pnl']:+.2f}")

# Save full results
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "threshold_optimization_results.csv")
results_df.to_csv(out_path, index=False)
print(f"\nFull results saved to: {out_path}")
