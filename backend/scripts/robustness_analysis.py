#!/usr/bin/env python3
"""
Comprehensive robustness analysis for ML Rebound strategy.

1. Walk-forward backtest per quarter  (existing models, quarterly windows)
2. Out-of-sample: train on 2024 only → test on 2025
3. Spread ×2 and Slippage ×2 stress test
4. net_edge threshold: 25 vs 30 vs 35 EUR/MWh
5. Worst-month / worst-week analysis
6. Gap filter + extreme-spread guard (no trades on bad data)
"""
from __future__ import annotations

import os
import sys
import shutil
import tempfile
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import numpy as np
import pandas as pd
import psycopg2
from datetime import timezone

from app.ml.negative_price_classifier import NegativePriceClassifier
from app.ml.rebound_classifier import ReboundClassifier
from app.ml.price_regression import PriceRegressionModel
from app.features.engineering import FeatureEngineer
from app.core.config import settings

DSN = "postgresql://ppuser:pppass@localhost:5432/powerprice"

# ---------------------------------------------------------------------------
# Futures cost constants (default)
# ---------------------------------------------------------------------------
_D_SPREAD    = 5.0
_D_SLIPPAGE  = 3.0
_D_MARKUP    = 1.0
_D_SAFETY    = 5.0
_FIN_HOURLY  = 0.08 / 365.0 / 24.0   # 8% p.a.
_FIN_WE_MUL  = 1.5
_MAX_HOLD    = 6
_SL_BUF      = 8.0
_TP_MUL      = 2.0
_P_REB_THR   = 0.60
_NET_EDGE    = 30.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def total_cost(spread, slippage, markup, safety, holding_hours, is_wknd, price_abs):
    rate = _FIN_HOURLY * (_FIN_WE_MUL if is_wknd else 1.0)
    return spread + slippage + markup + safety + rate * holding_hours * price_abs


def simulate(prices, is_weekends, p_neg_arr, p_reb_arr, pred_p_arr, timestamps_ns,
             net_edge_thr=_NET_EDGE,
             spread=_D_SPREAD, slippage=_D_SLIPPAGE,
             markup=_D_MARKUP, safety=_D_SAFETY,
             gap_filter=False, price_gaps_h=None, vol24_arr=None,
             start_ts_ns=None, end_ts_ns=None):
    """Simulate ML rebound strategy, return list of (exit_ts_ns, pnl_net)."""
    n = len(prices)
    trades = []
    in_pos = False
    entry_price = 0.0
    entry_i = 0
    sl = 0.0
    tp = 0.0

    for i in range(n):
        px = prices[i]
        if not np.isfinite(px):
            continue

        ts = timestamps_ns[i]
        if start_ts_ns is not None and ts < start_ts_ns:
            continue
        if end_ts_ns is not None and ts >= end_ts_ns:
            continue

        if not in_pos:
            if px >= 0.0:
                continue
            if p_reb_arr[i] < _P_REB_THR:
                continue

            # Task 6 – gap filter: skip entry if any of last 6 rows has gap > 2h
            if gap_filter and price_gaps_h is not None:
                lookback = max(0, i - 6)
                if any(g > 2.0 for g in price_gaps_h[lookback:i]):
                    continue

            # Task 6 – extreme spread guard: if price vol (24h std) > 100 EUR/MWh, skip
            dyn_spread = spread
            if gap_filter and vol24_arr is not None and np.isfinite(vol24_arr[i]):
                if vol24_arr[i] > 100.0:
                    continue  # too volatile — spread would be extreme

            raw_reb = max(0.0, pred_p_arr[i] - px)
            exp_reb = raw_reb * p_reb_arr[i]
            entry_tc = total_cost(dyn_spread, slippage, markup, safety,
                                  _MAX_HOLD, is_weekends[i], abs(px) or 100.0)
            net_edge = exp_reb - entry_tc
            if net_edge < net_edge_thr:
                continue

            in_pos = True
            entry_price = px
            entry_i = i
            sl = entry_price - _SL_BUF
            tp = entry_price + net_edge * _TP_MUL

        else:
            hours_held = i - entry_i
            if (px <= sl or px >= 0.0 or px >= tp or hours_held >= _MAX_HOLD):
                exit_tc = total_cost(spread, slippage, markup, safety,
                                     float(hours_held), is_weekends[i],
                                     abs(entry_price) or 100.0)
                pnl_net = (px - entry_price) - exit_tc
                trades.append((timestamps_ns[i], pnl_net))
                in_pos = False

    return trades


def calc_metrics(trades, label=""):
    if not trades:
        return {"label": label, "n_trades": 0, "profit_factor": 0.0,
                "win_rate": 0.0, "avg_trade": 0.0, "sharpe": 0.0,
                "total_pnl": 0.0, "max_drawdown": 0.0,
                "best_trade": 0.0, "worst_trade": 0.0}
    pnls = np.array([p for _, p in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gp = wins.sum() if len(wins) else 0.0
    gl = abs(losses.sum()) if len(losses) else 0.0
    pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 1.0)
    wr = len(wins) / len(pnls) * 100.0
    avg = pnls.mean()
    std = pnls.std(ddof=1) if len(pnls) > 1 else 1e-9
    sharpe = avg / std * np.sqrt(len(pnls))
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    max_dd = float((peak - cum).max()) if len(cum) else 0.0
    return {
        "label": label,
        "n_trades": len(pnls),
        "profit_factor": round(min(pf, 999.0), 3),
        "win_rate": round(wr, 1),
        "avg_trade": round(avg, 2),
        "sharpe": round(sharpe, 3),
        "total_pnl": round(pnls.sum(), 2),
        "max_drawdown": round(max_dd, 2),
        "best_trade": round(pnls.max(), 2),
        "worst_trade": round(pnls.min(), 2),
    }


def print_table(rows, title):
    print(f"\n{'=' * 100}")
    print(title)
    print("=" * 100)
    hdr = (f"{'Label':<28}  {'N':>5}  {'PF':>6}  {'WinRate':>8}  "
           f"{'AvgTrade':>9}  {'Sharpe':>7}  {'TotalPnL':>10}  "
           f"{'MaxDD':>8}  {'Best':>8}  {'Worst':>9}")
    print(hdr)
    print("-" * 100)
    for r in rows:
        pf = f"{r['profit_factor']:.3f}" if r['profit_factor'] < 999 else "  ∞   "
        print(f"{r['label']:<28}  {r['n_trades']:>5}  {pf:>6}  "
              f"{r['win_rate']:>7.1f}%  {r['avg_trade']:>+9.2f}  "
              f"{r['sharpe']:>7.3f}  {r['total_pnl']:>+10.2f}  "
              f"{r['max_drawdown']:>8.2f}  {r['best_trade']:>+8.2f}  "
              f"{r['worst_trade']:>+9.2f}")


# ---------------------------------------------------------------------------
# 0. Load full data + generate predictions with production models
# ---------------------------------------------------------------------------

print("Loading full historical data...")
conn = psycopg2.connect(DSN)
with conn.cursor() as cur:
    cur.execute("""
        SELECT timestamp, price_eur_mwh, load_mw, wind_onshore_mw,
               wind_offshore_mw, solar_mw, residual_load_mw, net_export_mw,
               temperature_c, wind_speed_ms, solar_radiation_wm2, cloud_cover_pct,
               is_holiday, is_weekend, hour, month
        FROM hourly_prices WHERE price_eur_mwh IS NOT NULL ORDER BY timestamp ASC
    """)
    cols = [d[0] for d in cur.description]
    rows_all = cur.fetchall()
conn.close()

df_all = pd.DataFrame(rows_all, columns=cols)
df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True)
print(f"Rows: {len(df_all)}  |  {df_all['timestamp'].min().date()} → {df_all['timestamp'].max().date()}")
print(f"Negative hours: {(df_all['price_eur_mwh'] < 0).sum()}")

print("\nBuilding features + generating predictions (production models)...")
fe = FeatureEngineer()
features_df = fe.build_features(df_all)
feat_cols = [c for c in fe.FEATURE_COLUMNS if c in features_df.columns]
valid_mask = features_df[feat_cols].notna().all(axis=1)
X_valid = features_df.loc[valid_mask, feat_cols].fillna(0.0)

neg_clf = NegativePriceClassifier(model_dir=settings.model_dir); neg_clf.load()
reb_clf = ReboundClassifier(model_dir=settings.model_dir); reb_clf.load()
reg_mdl = PriceRegressionModel(model_dir=settings.model_dir); reg_mdl.load()

X_neg_s = neg_clf.scaler.transform(X_valid)
X_reb_s = reb_clf.scaler.transform(X_valid)
X_reg_s = reg_mdl.scaler.transform(X_valid)

preds_all = pd.DataFrame({
    "p_negative":      neg_clf.model.predict_proba(X_neg_s)[:, 1],
    "p_rebound":       reb_clf.model.predict_proba(X_reb_s)[:, 1],
    "predicted_price": reg_mdl.model.predict(X_reg_s),
}, index=X_valid.index).reindex(df_all.index, fill_value=0.0)

df_idx = df_all.set_index("timestamp").sort_index()
preds_all.index = df_idx.index

# Numpy arrays
prices       = df_idx["price_eur_mwh"].values.astype(float)
is_weekends  = df_idx["is_weekend"].values.astype(bool)
p_neg_arr    = preds_all["p_negative"].values.astype(float)
p_reb_arr    = preds_all["p_rebound"].values.astype(float)
pred_p_arr   = preds_all["predicted_price"].values.astype(float)
ts_index     = df_idx.index
# pandas 2.x DatetimeIndex is datetime64[us] → .asi8 gives microseconds;
# pd.Timestamp.value gives nanoseconds. Multiply by 1000 to align to ns.
ts_ns        = ts_index.asi8 * 1000  # now in nanoseconds, same unit as Timestamp.value

# Timestamp gaps (for Task 6)
ts_diffs_h = np.concatenate([[0.0], np.diff(ts_ns) / 3_600_000_000_000])  # ns → hours

# 24h rolling price std (for Task 6 extreme spread guard)
price_series = pd.Series(prices, index=ts_index)
vol24_arr    = price_series.rolling(24, min_periods=12).std().values

# Baseline (full window, no filters)
baseline = calc_metrics(
    simulate(prices, is_weekends, p_neg_arr, p_reb_arr, pred_p_arr, ts_ns),
    "Baseline (full 2Y, net_edge=30)"
)
print(f"Baseline → PF={baseline['profit_factor']:.3f}, "
      f"N={baseline['n_trades']}, WR={baseline['win_rate']:.1f}%, "
      f"Avg={baseline['avg_trade']:+.2f}")

# Quarter boundaries as nanoseconds
quarters = []
for year in [2024, 2025, 2026]:
    for (qnum, (m0, m1)) in enumerate([(1,4),(4,7),(7,10),(10,13)], 1):
        if year == 2026 and m0 >= 7:
            continue
        y1 = year + (1 if m1 > 12 else 0)
        m1r = m1 if m1 <= 12 else 1
        start = pd.Timestamp(f"{year}-{m0:02d}-01", tz="UTC")
        end   = pd.Timestamp(f"{y1}-{m1r:02d}-01", tz="UTC")
        if start >= ts_index[-1]:
            continue
        quarters.append((f"Q{qnum} {year}", start.value, end.value))

# Year boundaries
year_bounds = {
    "2024": (pd.Timestamp("2024-01-01", tz="UTC").value,
             pd.Timestamp("2025-01-01", tz="UTC").value),
    "2025": (pd.Timestamp("2025-01-01", tz="UTC").value,
             pd.Timestamp("2026-01-01", tz="UTC").value),
    "2026 (partial)": (pd.Timestamp("2026-01-01", tz="UTC").value,
                       pd.Timestamp("2027-01-01", tz="UTC").value),
}

# ============================================================
# 1. WALK-FORWARD PER QUARTER
# ============================================================
print("\n\n>>> TASK 1: Walk-forward per quarter  (production models)")
qtr_results = []
for label, s_ns, e_ns in quarters:
    t = simulate(prices, is_weekends, p_neg_arr, p_reb_arr, pred_p_arr, ts_ns,
                 start_ts_ns=s_ns, end_ts_ns=e_ns)
    qtr_results.append(calc_metrics(t, label))

# Also add yearly summary
for yl, (s_ns, e_ns) in year_bounds.items():
    t = simulate(prices, is_weekends, p_neg_arr, p_reb_arr, pred_p_arr, ts_ns,
                 start_ts_ns=s_ns, end_ts_ns=e_ns)
    qtr_results.append(calc_metrics(t, f"Year {yl}"))

print_table(qtr_results, "Walk-forward per quarter (net_edge=30, p_rebound=0.60)")

# ============================================================
# 2. OUT-OF-SAMPLE: train on 2024 → test on 2025
# ============================================================
print("\n\n>>> TASK 2: Out-of-sample  (2024-trained models → 2025 predictions)")

oos_model_dir = tempfile.mkdtemp(prefix="powerprice_oos_")
print(f"Training OOS models in {oos_model_dir} ...")

# 2024 training data
mask_2024 = (df_all["timestamp"] >= "2024-01-01") & (df_all["timestamp"] < "2025-01-01")
df_2024 = df_all[mask_2024].copy().reset_index(drop=True)
print(f"  2024 training rows: {len(df_2024)}, neg hours: {(df_2024['price_eur_mwh'] < 0).sum()}")

try:
    neg_oos = NegativePriceClassifier(model_dir=oos_model_dir); neg_oos.train(df_2024); neg_oos.save()
    reb_oos = ReboundClassifier(model_dir=oos_model_dir);       reb_oos.train(df_2024); reb_oos.save()
    reg_oos = PriceRegressionModel(model_dir=oos_model_dir);    reg_oos.train(df_2024); reg_oos.save()
    print("  OOS models trained OK.")

    # 2025 test data (incl. warmup prefix from Dec 2024 for feature engineering)
    mask_oos_load = df_all["timestamp"] >= "2024-10-01"
    df_oos_raw = df_all[mask_oos_load].copy().reset_index(drop=True)

    fe2 = FeatureEngineer()
    feat_oos = fe2.build_features(df_oos_raw)
    valid_oos = feat_oos[feat_cols].notna().all(axis=1)
    X_oos_valid = feat_oos.loc[valid_oos, feat_cols].fillna(0.0)

    X_n = neg_oos.scaler.transform(X_oos_valid)
    X_r = reb_oos.scaler.transform(X_oos_valid)
    X_g = reg_oos.scaler.transform(X_oos_valid)

    preds_oos = pd.DataFrame({
        "p_negative":      neg_oos.model.predict_proba(X_n)[:, 1],
        "p_rebound":       reb_oos.model.predict_proba(X_r)[:, 1],
        "predicted_price": reg_oos.model.predict(X_g),
    }, index=X_oos_valid.index).reindex(df_oos_raw.index, fill_value=0.0)

    df_oos_idx = df_oos_raw.set_index("timestamp").sort_index()
    preds_oos.index = df_oos_idx.index

    prices_oos      = df_oos_idx["price_eur_mwh"].values.astype(float)
    is_we_oos       = df_oos_idx["is_weekend"].values.astype(bool)
    p_reb_oos       = preds_oos["p_rebound"].values.astype(float)
    p_neg_oos       = preds_oos["p_negative"].values.astype(float)
    pred_p_oos      = preds_oos["predicted_price"].values.astype(float)
    ts_oos_ns       = df_oos_idx.index.asi8 * 1000  # μs → ns

    y2025_s = pd.Timestamp("2025-01-01", tz="UTC").value
    y2025_e = pd.Timestamp("2026-01-01", tz="UTC").value

    oos_rows = []
    oos_in_sample = calc_metrics(
        simulate(prices, is_weekends, p_neg_arr, p_reb_arr, pred_p_arr, ts_ns,
                 start_ts_ns=year_bounds["2025"][0], end_ts_ns=year_bounds["2025"][1]),
        "2025 — PROD models (in-sample)"
    )
    oos_true = calc_metrics(
        simulate(prices_oos, is_we_oos, p_neg_oos, p_reb_oos, pred_p_oos, ts_oos_ns,
                 start_ts_ns=y2025_s, end_ts_ns=y2025_e),
        "2025 — 2024-trained (OOS)"
    )
    oos_rows.append(oos_in_sample)
    oos_rows.append(oos_true)
    print_table(oos_rows, "Out-of-Sample comparison: 2025")

except Exception as exc:
    print(f"  OOS training failed: {exc}")
    import traceback; traceback.print_exc()
finally:
    shutil.rmtree(oos_model_dir, ignore_errors=True)

# ============================================================
# 3. STRESS TEST: spread ×2, slippage ×2
# ============================================================
print("\n\n>>> TASK 3: Stress test — Spread ×2 and Slippage ×2")

stress_scenarios = [
    ("Default (spread=5, slip=3)",    _D_SPREAD,      _D_SLIPPAGE),
    ("Spread ×2 (spread=10, slip=3)", _D_SPREAD * 2,  _D_SLIPPAGE),
    ("Slippage ×2 (spread=5, slip=6)", _D_SPREAD,     _D_SLIPPAGE * 2),
    ("Spread ×2 + Slippage ×2",       _D_SPREAD * 2,  _D_SLIPPAGE * 2),
]
stress_rows = []
for label, sp, sl in stress_scenarios:
    t = simulate(prices, is_weekends, p_neg_arr, p_reb_arr, pred_p_arr, ts_ns,
                 spread=sp, slippage=sl)
    stress_rows.append(calc_metrics(t, label))

print_table(stress_rows, "Cost stress test (net_edge=30, p_rebound=0.60)")

# ============================================================
# 4. NET_EDGE THRESHOLD: 25 vs 30 vs 35
# ============================================================
print("\n\n>>> TASK 4: net_edge threshold sensitivity: 25 / 30 / 35 EUR/MWh")

edge_rows = []
for ne in [20, 25, 30, 35, 40]:
    t = simulate(prices, is_weekends, p_neg_arr, p_reb_arr, pred_p_arr, ts_ns,
                 net_edge_thr=float(ne))
    edge_rows.append(calc_metrics(t, f"net_edge ≥ {ne} EUR/MWh"))

print_table(edge_rows, "net_edge threshold sensitivity (p_rebound=0.60, default costs)")

# ============================================================
# 5. WORST-MONTH / WORST-WEEK ANALYSIS
# ============================================================
print("\n\n>>> TASK 5: Worst-month / worst-week analysis")

all_trades = simulate(prices, is_weekends, p_neg_arr, p_reb_arr, pred_p_arr, ts_ns)

if all_trades:
    ts_arr  = pd.to_datetime(np.array([t for t, _ in all_trades]), unit="ns", utc=True)
    pnl_arr = np.array([p for _, p in all_trades])

    trade_df = pd.DataFrame({"ts": ts_arr, "pnl": pnl_arr})
    trade_df["month"]   = trade_df["ts"].dt.to_period("M")
    trade_df["isoweek"] = trade_df["ts"].dt.to_period("W")

    monthly = trade_df.groupby("month").agg(
        n_trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
        avg_trade=("pnl", "mean"),
        max_loss=("pnl", "min"),
    ).reset_index().sort_values("total_pnl")

    weekly = trade_df.groupby("isoweek").agg(
        n_trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
        avg_trade=("pnl", "mean"),
        max_loss=("pnl", "min"),
    ).reset_index().sort_values("total_pnl")

    print(f"\n{'=' * 90}")
    print("Monthly performance — all months with at least 1 trade (sorted worst → best)")
    print("=" * 90)
    print(f"{'Month':<12}  {'N':>4}  {'TotalPnL':>10}  {'WinRate':>8}  "
          f"{'AvgTrade':>9}  {'WorstTrade':>11}")
    print("-" * 90)
    for _, row in monthly.iterrows():
        print(f"{str(row['month']):<12}  {int(row['n_trades']):>4}  "
              f"{row['total_pnl']:>+10.2f}  {row['win_rate']:>7.1f}%  "
              f"{row['avg_trade']:>+9.2f}  {row['max_loss']:>+11.2f}")

    # Best months
    print(f"\nBest 5 months:")
    for _, row in monthly.sort_values("total_pnl", ascending=False).head(5).iterrows():
        print(f"  {str(row['month']):<12}  total={row['total_pnl']:>+8.2f}  "
              f"N={int(row['n_trades'])}  WR={row['win_rate']:.1f}%")

    print(f"\n{'=' * 90}")
    print("Worst 10 weeks (ISO weeks sorted by total P&L, ascending)")
    print("=" * 90)
    print(f"{'ISOWeek':<22}  {'N':>4}  {'TotalPnL':>10}  {'WinRate':>8}  "
          f"{'AvgTrade':>9}  {'WorstTrade':>11}")
    print("-" * 90)
    for _, row in weekly.head(10).iterrows():
        print(f"{str(row['isoweek']):<22}  {int(row['n_trades']):>4}  "
              f"{row['total_pnl']:>+10.2f}  {row['win_rate']:>7.1f}%  "
              f"{row['avg_trade']:>+9.2f}  {row['max_loss']:>+11.2f}")

    # Max consecutive losses
    win_arr = (pnl_arr > 0).astype(int)
    loss_runs = []
    cur_run = 0
    for w in win_arr:
        if w == 0:
            cur_run += 1
        else:
            if cur_run > 0:
                loss_runs.append(cur_run)
            cur_run = 0
    if cur_run > 0:
        loss_runs.append(cur_run)

    max_consec_losses = max(loss_runs) if loss_runs else 0
    print(f"\nMax consecutive losses: {max_consec_losses}")
    print(f"Total trades: {len(pnl_arr)}, Wins: {(pnl_arr > 0).sum()}, Losses: {(pnl_arr <= 0).sum()}")

# ============================================================
# 6. GAP FILTER + EXTREME-SPREAD GUARD
# ============================================================
print("\n\n>>> TASK 6: Gap filter + extreme-spread guard")

t_baseline = simulate(prices, is_weekends, p_neg_arr, p_reb_arr, pred_p_arr, ts_ns)
t_gapfilter = simulate(prices, is_weekends, p_neg_arr, p_reb_arr, pred_p_arr, ts_ns,
                       gap_filter=True, price_gaps_h=ts_diffs_h, vol24_arr=vol24_arr)

m_base  = calc_metrics(t_baseline,  "No filter (baseline)")
m_filt  = calc_metrics(t_gapfilter, "Gap + extreme-spread filter")

# Count how many candidate entries were blocked
filtered_count = 0
unfiltered_count = 0
for i, px in enumerate(prices):
    if not np.isfinite(px) or px >= 0.0 or p_reb_arr[i] < _P_REB_THR:
        continue
    unfiltered_count += 1
    lookback = max(0, i - 6)
    has_gap = any(g > 2.0 for g in ts_diffs_h[lookback:i])
    extreme_vol = np.isfinite(vol24_arr[i]) and vol24_arr[i] > 100.0
    if has_gap or extreme_vol:
        filtered_count += 1

print(f"\nCandidate entries (price<0, p_reb≥0.60): {unfiltered_count}")
print(f"Blocked by gap or extreme-vol filter:    {filtered_count}")
print(f"Gap-filtered rate:                       {filtered_count/max(1,unfiltered_count)*100:.1f}%")

filter_rows = [m_base, m_filt]
print_table(filter_rows, "Effect of gap + extreme-spread filter on final metrics")

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n\n" + "=" * 100)
print("ROBUSTNESS SUMMARY")
print("=" * 100)
summary = [
    baseline,
    calc_metrics(
        simulate(prices, is_weekends, p_neg_arr, p_reb_arr, pred_p_arr, ts_ns,
                 spread=_D_SPREAD * 2, slippage=_D_SLIPPAGE * 2),
        "Stress (spread×2 + slip×2)"
    ),
    calc_metrics(
        simulate(prices, is_weekends, p_neg_arr, p_reb_arr, pred_p_arr, pred_p_arr,
                 net_edge_thr=25.0),
        "net_edge=25"
    ),
    calc_metrics(
        simulate(prices, is_weekends, p_neg_arr, p_reb_arr, pred_p_arr, ts_ns,
                 net_edge_thr=35.0),
        "net_edge=35"
    ),
    calc_metrics(t_gapfilter, "Gap + vol filter applied"),
]
print_table(summary, "Key scenario comparison")

print("\nRobustness analysis complete.")
