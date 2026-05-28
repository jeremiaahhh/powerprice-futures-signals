# Architecture Documentation – PowerPrice Futures Signals

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Component Descriptions](#2-component-descriptions)
3. [Data Flow](#3-data-flow)
4. [Signal Generation Logic](#4-signal-generation-logic)
5. [Futures Cost Model](#5-futures-cost-model)
6. [Backtesting Methodology](#6-backtesting-methodology)
7. [Risk Management Rules](#7-risk-management-rules)

---

## 1. System Overview

PowerPrice Futures Signals is a research platform that applies supervised machine learning to European electricity market data to produce directional trading signals on power-price CFDs. The system is composed of five logical layers:

| Layer | Responsibility |
|---|---|
| **Ingestion** | Fetch raw OHLCV-style price data, generation mix, load and weather forecasts from public APIs |
| **Feature Engineering** | Transform raw data into model-ready features (lags, rolling statistics, calendar variables, residual load) |
| **ML Inference** | Run pre-trained models to produce directional probabilities and regime classifications |
| **Signal Engine** | Apply threshold and rule-based logic to model outputs; emit, store and expire paper-trade signals |
| **Presentation** | Serve signals via REST and WebSocket APIs consumed by a React dashboard |

All components run inside Docker containers orchestrated via Docker Compose (development) or Kubernetes (production). Inter-service communication uses HTTP/2 internally and Redis pub/sub for real-time event distribution.

---

## 2. Component Descriptions

### FastAPI Backend (`backend/`)

- Async Python 3.11 application using FastAPI and SQLAlchemy 2.0 async ORM
- Exposes versioned REST endpoints (`/api/v1/...`) and WebSocket endpoints (`/ws/...`)
- Performs JWT-based authentication for write operations (paper-trade management)
- Delegates all long-running computation to Celery tasks; remains non-blocking

### Celery Worker / Beat (`app/jobs/`)

- **Beat** uses a `CELERYBEAT_SCHEDULE` dict to trigger periodic tasks:
  - Every 60 minutes: `ingest_smard_prices`, `ingest_entsoe_data`, `ingest_weather`
  - Every 60 minutes (offset +5 min): `run_feature_pipeline`
  - Every 60 minutes (offset +10 min): `run_signal_generation`
  - Every 24 hours at 02:00 UTC: `retrain_models`
- **Workers** consume tasks from Redis queues, write results to PostgreSQL, publish events to Redis pub/sub channels

### PostgreSQL Database

Tables of interest:

| Table | Description |
|---|---|
| `price_series` | Hourly spot prices per market (time-series, optionally partitioned by month) |
| `generation_mix` | Hourly generation by fuel type per bidding zone |
| `cross_border_flows` | Hourly scheduled flows between countries |
| `weather_observations` | Hourly actual and forecast weather values |
| `feature_snapshots` | Engineered feature vector stored at each inference run |
| `ml_model_registry` | Version metadata, train/eval metrics, file path of serialised artefact |
| `signals` | Emitted signals with entry price, confidence, expiry time, status |
| `paper_trades` | Trade lifecycle: open → filled → closed, with P&L columns |

### Redis

Used for three purposes:
1. **Celery broker** (db 0) – task queuing
2. **Celery result backend** (db 1) – task status and return values
3. **API cache + pub/sub** (db 2) – 5-minute TTL on heavy reads; channel `signals:new` for real-time push to WebSocket clients

### React Frontend

Single-page application built with Vite/TypeScript. Key views:

- **Dashboard** – current signal table, live price ticker, market regime indicator
- **Signal History** – filterable log with confidence scores, P&L column
- **Performance** – equity curve, win rate, Sharpe ratio for paper trades
- **Models** – model version browser with training metrics
- **Data Explorer** – interactive price/feature chart with zoom

---

## 3. Data Flow

### 3.1 Ingestion

```
[External APIs]
      |
      | HTTP GET (hourly Celery beat trigger)
      v
[Celery Worker: ingest_* tasks]
      |
      | Validate, normalise units (EUR/MWh, MW, °C)
      v
[PostgreSQL: raw tables]
      |
      | Redis pub/sub: channel "ingestion:complete"
      v
[Celery Worker: run_feature_pipeline]
```

Each ingestor stores both a raw copy (for audit) and a cleaned copy. Failed requests are retried up to 3 times with exponential backoff; gaps are back-filled on the next successful run.

### 3.2 Feature Engineering

The feature pipeline reads the last `SIGNAL_LOOKBACK_HOURS` (default 168 h = 7 days) of raw data and computes:

**Price features**
- Hourly close price and log-return
- Rolling mean and standard deviation (3 h, 12 h, 24 h, 168 h windows)
- Lag prices: t-1, t-2, t-3, t-6, t-12, t-24, t-48, t-168
- Hour-of-day, day-of-week, month, is_weekend (calendar)
- Price z-score relative to rolling 30-day distribution

**Supply / demand features**
- Total load (actual and forecast)
- Residual load = load − (wind + solar)
- Generation mix fractions (coal, gas, nuclear, wind, solar, hydro)
- Cross-border net position (sum of import/export flows)

**Weather features**
- Onshore and offshore wind speed at hub height (10 m, 100 m)
- Global horizontal irradiance (GHI) forecast
- Temperature deviation from seasonal norm
- Wind ramp rate (delta wind speed t vs t-3)

All features are normalised using the statistics of the training window. The feature vector is stored in `feature_snapshots` for reproducibility.

### 3.3 ML Inference

```
[feature_snapshots: latest row]
      |
      v
[Load model artefacts from /app/models/]
      |
      +---> price_direction model  --> P(up), P(down)
      +---> price_magnitude model  --> expected_delta_eur_mwh
      +---> regime_classifier      --> regime_label {low, normal, spike}
      +---> spike_detector         --> spike_probability
      |
      v
[Signal engine: rule evaluation]
      |
      v
[signals table + Redis pub/sub: "signals:new"]
```

### 3.4 Signal Storage and Expiry

Each signal record contains:
- `market` (e.g., `DE_LU`, `FR`, `NL`)
- `signal_type` (`LONG`, `SHORT`, `SPIKE_WARNING`, `REGIME_CHANGE`, `NO_TRADE`)
- `entry_price_eur_mwh` – mid-market at time of emission
- `confidence` – model probability driving the signal
- `expected_delta_eur_mwh` – magnitude model output
- `valid_until` – timestamp of next settlement hour (default entry + 1 h)
- `status` – `OPEN`, `EXPIRED`, `CLOSED`

A Celery task running every 10 minutes scans for `OPEN` signals past their `valid_until` and closes them, recording the exit price and computing paper-trade P&L.

---

## 4. Signal Generation Logic

### 4.1 LONG signal

Emitted when all of the following conditions are met:

1. `price_direction` model P(up) > `SIGNAL_MIN_CONFIDENCE` (default 0.55)
2. `regime_classifier` output is `normal` or `spike`
3. Residual load is above its 24-hour rolling mean (demand pressure)
4. No open LONG signal already exists for the same market

### 4.2 SHORT signal

Emitted when:

1. `price_direction` model P(down) > `SIGNAL_MIN_CONFIDENCE`
2. `regime_classifier` output is `low` or `normal`
3. Renewable surplus: wind + solar fraction > 0.40 of total generation
4. No open SHORT signal already exists for the same market

### 4.3 SPIKE_WARNING signal

Emitted when:

1. `spike_detector` score > 0.70 AND
2. Residual load > 95th percentile of 30-day rolling distribution OR
3. Cross-border imports are at network capacity (congestion proxy)

SPIKE_WARNING signals do not open paper trades; they are informational alerts.

### 4.4 REGIME_CHANGE signal

Emitted when the `regime_classifier` label changes between consecutive inference runs (e.g., `normal` → `spike`). A cooldown of 3 hours prevents repeated regime-change alerts during oscillation.

### 4.5 NO_TRADE signal

Emitted when:

1. Model confidence is in the neutral band: P(up) between 0.45 and 0.55 AND
2. Regime is `normal`

NO_TRADE signals are stored to maintain a complete signal log but do not open paper trades.

---

## 5. Futures Cost Model

The paper-trading engine simulates realistic Futures trading economics. All calculations are in EUR/MWh with a default lot size of 1 MWh.

### 5.1 Components

**Spread cost**

The bid-ask spread represents the market-maker fee embedded in the Futures price. The platform models it as a fixed half-spread deducted on each side of the trade:

```
spread_cost_entry = FUTURES_SPREAD_EUR_MWH / 2
spread_cost_exit  = FUTURES_SPREAD_EUR_MWH / 2
total_spread_cost = FUTURES_SPREAD_EUR_MWH  (per lot)
```

**Overnight financing charge**

CFDs carry a daily financing charge proportional to the notional value held overnight:

```
daily_financing_rate = FUTURES_FINANCING_RATE_PA / 365

financing_charge_per_day = entry_price_eur_mwh
                           * lot_size_mwh
                           * daily_financing_rate
```

For intraday signals (< 24 h duration) the financing charge is prorated:

```
financing_charge = financing_charge_per_day * (hours_held / 24)
```

**Net P&L formula**

```
gross_pnl = (exit_price - entry_price) * direction * lot_size
net_pnl   = gross_pnl - total_spread_cost - financing_charge
```

Where `direction` = +1 for LONG, -1 for SHORT.

### 5.2 Example

Given:
- Signal: LONG, entry = 85.00 EUR/MWh, exit = 87.50 EUR/MWh
- Lot size: 1 MWh
- Spread: 0.50 EUR/MWh
- Financing rate: 5% p.a., held 2 hours

```
gross_pnl         = (87.50 - 85.00) * 1 * 1     =  2.50 EUR
spread_cost       = 0.50 * 1                      =  0.50 EUR
financing_charge  = 85.00 * 1 * (0.05/365) * (2/24) =  0.00972 EUR

net_pnl           = 2.50 - 0.50 - 0.01            =  1.99 EUR
```

---

## 6. Backtesting Methodology

The backtesting engine replays the signal generation logic over historical data to measure out-of-sample performance.

### 6.1 Walk-Forward Validation

1. Training window: 90 days
2. Test window: 30 days (no-look-ahead)
3. The window slides forward by 30 days and the model is retrained each time
4. This produces non-overlapping out-of-sample periods covering the full history

### 6.2 Metrics Computed

| Metric | Formula |
|---|---|
| Win rate | `winning_trades / total_trades` |
| Average net P&L per trade | `sum(net_pnl) / total_trades` |
| Profit factor | `sum(winning_pnl) / abs(sum(losing_pnl))` |
| Sharpe ratio (annualised) | `mean(daily_pnl) / std(daily_pnl) * sqrt(252)` |
| Maximum drawdown | Maximum peak-to-trough decline in cumulative P&L |
| Calmar ratio | `CAGR / abs(max_drawdown)` |

### 6.3 Signal Quality Metrics

| Metric | Description |
|---|---|
| Precision | Fraction of LONG signals where price actually rose |
| Recall | Fraction of actual up-moves captured by LONG signals |
| F1 score | Harmonic mean of precision and recall |
| Log loss | Calibration quality of model probabilities |
| Brier score | Mean squared error of probability predictions |

### 6.4 Important Caveats

- Backtests do not account for market impact (the platform targets small lot sizes where this is negligible for research purposes)
- Transaction costs are modelled conservatively with the spread; actual broker costs may differ
- Electricity markets have structural changes (policy, infrastructure) that limit the stationarity of learned patterns
- SMARD and ENTSO-E historical data may contain revisions; the platform uses the latest available version

---

## 7. Risk Management Rules

The signal engine applies a rule layer on top of raw model output. These rules limit signal exposure and prevent pathological behaviour.

### 7.1 Signal Gating Rules

| Rule | Description |
|---|---|
| **Max open signals** | At most 1 open directional signal (LONG or SHORT) per market at any time |
| **Confidence threshold** | Signals suppressed if model confidence < `SIGNAL_MIN_CONFIDENCE` |
| **Regime gate** | LONG signals blocked when regime = `spike` and direction confidence < 0.65 |
| **Data freshness gate** | Signals suppressed if the most recent data point is older than 90 minutes |
| **Price spike filter** | Signals suppressed during confirmed price spikes (price > 3σ from rolling mean) to avoid chasing extremes |

### 7.2 Position Sizing

For paper trading, all trades use a fixed lot size of 1 MWh. This simplifies P&L accounting and performance attribution.

### 7.3 Signal Expiry

All signals have a hard expiry of 1 hour (the next settlement interval). Long-duration positions would accumulate overnight financing that is not captured in a single-hour signal framework. Future versions may support multi-hour signals with explicit position-sizing rules.

### 7.4 Model Staleness Protection

If a model artefact is older than `MODEL_RETRAIN_INTERVAL_HOURS * 2` (i.e., two retraining cycles have been missed), the signal engine switches to `NO_TRADE` for all markets and emits a system alert. This prevents stale models from generating signals in changed market conditions.

### 7.5 Database Consistency

All signal and paper-trade writes use database transactions. If any step of the signal emission pipeline fails (e.g., price snapshot unavailable), the signal is not stored and the error is logged for replay on the next cycle. This prevents partially-written signals that could corrupt performance metrics.
