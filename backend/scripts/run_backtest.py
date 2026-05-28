#!/usr/bin/env python3
"""
Run backtest: Naive strategy vs ML Rebound strategy.
"""
import sys
import os
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import psycopg2
import pandas as pd
import numpy as np

from app.backtest.backtester import Backtester
from app.futures.cost_model import CFDCostModel
from app.ml.negative_price_classifier import NegativePriceClassifier
from app.ml.rebound_classifier import ReboundClassifier
from app.ml.price_regression import PriceRegressionModel
from app.features.engineering import FeatureEngineer
from app.core.config import settings

# ---------------------------------------------------------------------------
# Load historical data
# ---------------------------------------------------------------------------

DSN = "postgresql://ppuser:pppass@localhost:5432/powerprice"
DAYS_BACK = 500

print(f"Loading {DAYS_BACK} days of historical data...")

conn = psycopg2.connect(DSN)
start_ts = datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)

with conn.cursor() as cur:
    cur.execute("""
        SELECT timestamp, price_eur_mwh, load_mw, wind_onshore_mw,
               wind_offshore_mw, solar_mw, residual_load_mw, net_export_mw,
               temperature_c, wind_speed_ms, solar_radiation_wm2, cloud_cover_pct,
               is_holiday, is_weekend, hour, month
        FROM hourly_prices
        WHERE timestamp >= %s AND price_eur_mwh IS NOT NULL
        ORDER BY timestamp ASC
    """, (start_ts,))
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()

conn.close()
df = pd.DataFrame(rows, columns=cols)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
print(f"Loaded {len(df)} rows: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
print(f"Negative price hours: {(df['price_eur_mwh'] < 0).sum()}")

# ---------------------------------------------------------------------------
# Load trained models
# ---------------------------------------------------------------------------

model_dir = settings.model_dir
neg_clf = NegativePriceClassifier(model_dir=model_dir)
reb_clf = ReboundClassifier(model_dir=model_dir)
reg_mdl = PriceRegressionModel(model_dir=model_dir)

try:
    neg_clf.load()
    reb_clf.load()
    reg_mdl.load()
    print(f"Models loaded from {model_dir}\n")
except FileNotFoundError as e:
    print(f"ERROR: {e}\nRun train_models.py first.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Generate ML predictions for entire historical window
# ---------------------------------------------------------------------------

print("Generating ML predictions on full dataset...")
fe = FeatureEngineer()
features_df = fe.build_features(df)

# Align to df index for predictions
p_neg_list = []
p_reb_list = []
pred_price_list = []

feature_cols = [c for c in fe.FEATURE_COLUMNS if c in features_df.columns]
X_all = features_df[feature_cols].fillna(0.0)

# Batch predict where features are complete (after warmup period)
valid_mask = features_df[feature_cols].notna().all(axis=1)
X_valid = X_all.loc[valid_mask]

if len(X_valid) > 0:
    from sklearn.preprocessing import StandardScaler
    X_neg_s = neg_clf.scaler.transform(X_valid)
    X_reb_s = reb_clf.scaler.transform(X_valid)
    X_reg_s = reg_mdl.scaler.transform(X_valid)

    p_neg_valid = neg_clf.model.predict_proba(X_neg_s)[:, 1]
    p_reb_valid = reb_clf.model.predict_proba(X_reb_s)[:, 1]
    pred_price_valid = reg_mdl.model.predict(X_reg_s)

    predictions = pd.DataFrame({
        "p_negative":       p_neg_valid,
        "p_rebound":        p_reb_valid,
        "predicted_price":  pred_price_valid,
    }, index=X_valid.index)
else:
    predictions = pd.DataFrame(columns=["p_negative", "p_rebound", "predicted_price"])

# Reindex to match df (fill missing with zeros)
predictions = predictions.reindex(df.index, fill_value=0.0)
# Set the timestamp as index on df for Backtester
df_indexed = df.set_index("timestamp").sort_index()
predictions_indexed = predictions.copy()
predictions_indexed.index = df_indexed.index

print(f"Predictions generated for {valid_mask.sum()} rows (warmup={len(df) - valid_mask.sum()} skipped)")

# ---------------------------------------------------------------------------
# Run Naive strategy backtest
# ---------------------------------------------------------------------------

cost_model = CFDCostModel()
backtester = Backtester(cost_model=cost_model)

print("\n" + "="*60)
print("NAIVE STRATEGY BACKTEST")
print("="*60)

try:
    naive_metrics = backtester.run_naive(df_indexed)
    print(f"Total trades:          {naive_metrics.total_trades}")
    print(f"Win rate:              {naive_metrics.win_rate_pct:.1f}%")
    print(f"Total return:          {naive_metrics.total_return_pct:.2f}%")
    print(f"Annualized return:     {naive_metrics.annualized_return_pct:.2f}%")
    print(f"Sharpe ratio:          {naive_metrics.sharpe_ratio:.3f}")
    print(f"Sortino ratio:         {naive_metrics.sortino_ratio:.3f}")
    print(f"Max drawdown:          {naive_metrics.max_drawdown_pct:.2f}%")
    print(f"Profit factor:         {naive_metrics.profit_factor:.3f}")
    print(f"Avg trade P&L:         {naive_metrics.avg_trade_eur_mwh:.2f} EUR/MWh")
    print(f"Best trade:            {naive_metrics.best_trade_eur_mwh:.2f} EUR/MWh")
    print(f"Worst trade:           {naive_metrics.worst_trade_eur_mwh:.2f} EUR/MWh")
    print(f"Trades per month:      {naive_metrics.trades_per_month:.1f}")
except Exception as e:
    print(f"Naive backtest error: {e}")
    import traceback; traceback.print_exc()

# ---------------------------------------------------------------------------
# Run ML Rebound strategy backtest
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("ML REBOUND STRATEGY BACKTEST")
print("="*60)

try:
    ml_metrics = backtester.run_ml_rebound(
        df_indexed,
        predictions_indexed,
        p_rebound_threshold=0.60,
        min_edge_threshold=30.0,
        max_holding_hours=6,
    )
    print(f"Total trades:          {ml_metrics.total_trades}")
    print(f"Win rate:              {ml_metrics.win_rate_pct:.1f}%")
    print(f"Total return:          {ml_metrics.total_return_pct:.2f}%")
    print(f"Annualized return:     {ml_metrics.annualized_return_pct:.2f}%")
    print(f"Sharpe ratio:          {ml_metrics.sharpe_ratio:.3f}")
    print(f"Sortino ratio:         {ml_metrics.sortino_ratio:.3f}")
    print(f"Max drawdown:          {ml_metrics.max_drawdown_pct:.2f}%")
    print(f"Profit factor:         {ml_metrics.profit_factor:.3f}")
    print(f"Avg trade P&L:         {ml_metrics.avg_trade_eur_mwh:.2f} EUR/MWh")
    print(f"Best trade:            {ml_metrics.best_trade_eur_mwh:.2f} EUR/MWh")
    print(f"Worst trade:           {ml_metrics.worst_trade_eur_mwh:.2f} EUR/MWh")
    print(f"Trades per month:      {ml_metrics.trades_per_month:.1f}")

    if ml_metrics.monthly_performance:
        print("\nMonthly P&L (EUR/MWh):")
        for month, pnl in sorted(ml_metrics.monthly_performance.items()):
            sign = "+" if pnl >= 0 else ""
            print(f"  {month}:  {sign}{pnl:.2f}")
except Exception as e:
    print(f"ML backtest error: {e}")
    import traceback; traceback.print_exc()

print("\nBacktest complete.")
