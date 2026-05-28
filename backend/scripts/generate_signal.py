#!/usr/bin/env python3
"""
Generate the current Futures signal using live data + trained models.
Shows what signal the system would emit right now.
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

from app.signals.signal_engine import SignalEngine
from app.futures.cost_model import CFDCostModel
from app.ml.negative_price_classifier import NegativePriceClassifier
from app.ml.rebound_classifier import ReboundClassifier
from app.ml.price_regression import PriceRegressionModel
from app.features.engineering import FeatureEngineer
from app.core.config import settings

try:
    from app.risk.risk_engine import RiskEngine
except ImportError:
    RiskEngine = None

# ---------------------------------------------------------------------------
# Load recent data (last 72h for lag features)
# ---------------------------------------------------------------------------

DSN = "postgresql://ppuser:pppass@localhost:5432/powerprice"

conn = psycopg2.connect(DSN)
with conn.cursor() as cur:
    cur.execute("""
        SELECT timestamp, price_eur_mwh, load_mw, wind_onshore_mw,
               wind_offshore_mw, solar_mw, residual_load_mw, net_export_mw,
               temperature_c, wind_speed_ms, solar_radiation_wm2, cloud_cover_pct,
               is_holiday, is_weekend, hour, month
        FROM hourly_prices
        WHERE timestamp >= NOW() - INTERVAL '5 days'
          AND price_eur_mwh IS NOT NULL
        ORDER BY timestamp ASC
    """)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
conn.close()

df = pd.DataFrame(rows, columns=cols)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
latest = df.iloc[-1]

print(f"\nLatest data point: {df['timestamp'].max()}")
print(f"Current price:     {latest['price_eur_mwh']:.2f} EUR/MWh")
print(f"Temperature:       {latest['temperature_c']:.1f}°C")
print(f"Wind speed:        {latest['wind_speed_ms']:.1f} m/s")
print(f"Solar radiation:   {latest['solar_radiation_wm2']:.0f} W/m²")
print(f"Is weekend:        {bool(latest['is_weekend'])}")

# ---------------------------------------------------------------------------
# Build features
# ---------------------------------------------------------------------------

fe = FeatureEngineer()
features_df = fe.build_features(df)
feature_cols = [c for c in fe.FEATURE_COLUMNS if c in features_df.columns]
X_current = features_df[feature_cols].fillna(0.0).iloc[-1:]

# ---------------------------------------------------------------------------
# Load models and predict
# ---------------------------------------------------------------------------

model_dir = settings.model_dir
neg_clf = NegativePriceClassifier(model_dir=model_dir)
reb_clf = ReboundClassifier(model_dir=model_dir)
reg_mdl = PriceRegressionModel(model_dir=model_dir)

neg_clf.load()
reb_clf.load()
reg_mdl.load()

X_neg_s = neg_clf.scaler.transform(X_current)
X_reb_s = reb_clf.scaler.transform(X_current)
X_reg_s = reg_mdl.scaler.transform(X_current)

p_negative   = float(neg_clf.model.predict_proba(X_neg_s)[0, 1])
p_rebound    = float(reb_clf.model.predict_proba(X_reb_s)[0, 1])
predicted_price = float(reg_mdl.model.predict(X_reg_s)[0])

print(f"\n{'='*50}")
print("ML SIGNAL ANALYSIS")
print(f"{'='*50}")
print(f"p_negative (prob price will go <0):  {p_negative:.3f}")
print(f"p_rebound  (prob of rebound if neg): {p_rebound:.3f}")
print(f"Predicted price (next period):       {predicted_price:.2f} EUR/MWh")

# ---------------------------------------------------------------------------
# Futures cost model edge calculation
# ---------------------------------------------------------------------------

current_price = float(latest["price_eur_mwh"])
cost_model = CFDCostModel()

if current_price < 0:
    expected_rebound = max(0.0, predicted_price - current_price)
    cost_result = cost_model.calculate_net_edge(
        expected_rebound_eur_mwh=expected_rebound * p_rebound,
        estimated_holding_hours=4.0,
        is_weekend=bool(latest["is_weekend"]),
        notional_price_eur_mwh=abs(current_price),
    )
    print(f"\nCFD Edge (current price IS negative):")
    print(f"  Expected rebound:     {expected_rebound:.2f} EUR/MWh")
    print(f"  Gross edge:           {cost_result.gross_edge:.2f} EUR/MWh")
    print(f"  Total Futures costs:      {cost_result.total_cost:.2f} EUR/MWh")
    print(f"  Net edge:             {cost_result.net_edge:.2f} EUR/MWh")

# ---------------------------------------------------------------------------
# Signal determination
# ---------------------------------------------------------------------------

print(f"\n{'='*50}")
print("SIGNAL DECISION")
print(f"{'='*50}")

P_NEG_THRESHOLD = 0.50
P_REB_THRESHOLD = 0.60
MIN_NET_EDGE    = 10.0

if current_price >= 0:
    if p_negative >= P_NEG_THRESHOLD:
        action = "WATCH_LONG_REBOUND"
        reason = f"Price positive ({current_price:.1f}) but p_negative={p_negative:.2f} elevated — watch for entry"
    else:
        action = "NO_TRADE"
        reason = f"Price positive ({current_price:.1f} EUR/MWh), p_negative={p_negative:.2f} — no opportunity"
elif current_price < 0:
    if p_rebound < P_REB_THRESHOLD:
        action = "NO_TRADE"
        reason = f"Price negative but p_rebound={p_rebound:.2f} < {P_REB_THRESHOLD} threshold"
    else:
        # Calculate edge
        expected_rebound = max(0.0, predicted_price - current_price) * p_rebound
        cost_result = cost_model.calculate_net_edge(
            expected_rebound_eur_mwh=expected_rebound,
            estimated_holding_hours=4.0,
            is_weekend=bool(latest["is_weekend"]),
            notional_price_eur_mwh=abs(current_price),
        )
        if cost_result.net_edge >= MIN_NET_EDGE:
            action = "ENTER_LONG_REBOUND_SIGNAL"
            reason = f"All conditions met: price={current_price:.1f}, p_neg={p_negative:.2f}, p_reb={p_rebound:.2f}, net_edge={cost_result.net_edge:.1f}"
        else:
            action = "RISK_BLOCKED"
            reason = f"Net edge {cost_result.net_edge:.1f} EUR/MWh < threshold {MIN_NET_EDGE}"

print(f"Action: {action}")
print(f"Reason: {reason}")
print(f"\n[SIGNAL_ONLY=true — no real orders, paper tracking only]")
