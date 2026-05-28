# PowerPrice Futures Signals

A research platform that turns German day-ahead electricity market data into
directional Futures signals. Ingests live data from SMARD, ENTSO-E, and
Open-Meteo, runs a feature pipeline, scores it through gradient-boosted
models, and emits typed signals (`ENTER`, `WATCH`, `BLOCKED`) with full
audit trail, paper-trading P&L, and a tail-risk gate.

The entire stack runs locally with `docker compose up`. A persistent
**signal daemon** keeps the loop alive across macOS sleep, ENTSO-E outages,
and clock jumps; a **shadow-mode** lane lets new model versions run alongside
live without leaking into the operator's signal feed.

<p>
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-18-61dafb?logo=react" alt="React" />
  <img src="https://img.shields.io/badge/Postgres-16-336791?logo=postgresql" alt="Postgres" />
  <img src="https://img.shields.io/badge/Redis-7-DC382D?logo=redis" alt="Redis" />
  <img src="https://img.shields.io/badge/Celery-5-37814A?logo=celery" alt="Celery" />
  <img src="https://img.shields.io/badge/LightGBM-✓-success" alt="LightGBM" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License" />
</p>

> **SIGNAL ONLY.** No broker integration, no order execution, no managed
> money. Outputs are research artefacts derived from publicly available
> market data. Futures trading is leveraged and can result in losses exceeding
> deposits. Nothing in this repository is financial advice.

---

## Why this project

Most ML projects stop at "model produces good RMSE." Trading-adjacent ML
projects fail there: the model is the easy part, the surrounding plumbing
is what decides whether the signals are trustworthy.

The interesting engineering work in this repo is the plumbing around the
model:

- A **typed signal contract** (`ENTER`, `WATCH`, `BLOCKED`, `NO_SIGNAL`)
  with reasons surfaced to the UI — no opaque scores.
- A **Futures cost model** baked into the entry threshold: spread, slippage,
  overnight financing, and a configurable safety buffer have to be cleared
  before a signal is even considered.
- A **tail-risk gate** that suppresses signals during negative-price runs,
  large overnight gaps, and extreme volatility — the same kind of guard
  that keeps "good" models from blowing up on the days that actually matter.
- A **shadow-mode lane** so a candidate model runs against live data
  *without emitting signals to the operator*, building up an out-of-sample
  P&L record before promotion.
- A **persistent daemon** with state on disk, wall-clock jump detection,
  consecutive-error tripwires, and graceful degradation when ENTSO-E or
  SMARD return stale data.
- A **paper-trading ledger** that closes positions on the next settlement
  hour and writes win-rate, Sharpe, drawdown, and rolling-PF metrics into
  Postgres — the same metrics the auto-retrain scheduler reads to decide
  whether to ship a new model.

The model itself is one swappable component. The rest is the project.

---

## Technical highlights

| Area | What's done | Why it matters |
| --- | --- | --- |
| Architecture | Layered FastAPI backend: routers → services → ML/feature/runtime modules → repositories. All config flows through a single `pydantic-settings` `Settings` object. | Routes stay thin; the same service objects are reusable from Celery tasks, the signal daemon, scripts, and tests. |
| Signal engine | Threshold rules over `p_rebound`, `net_edge`, and `tail_risk_score`. Every signal records the reasons it was emitted *or* blocked. | A signal feed without reasons is unreviewable. This one is. |
| Cost model | Per-signal Futures cost is computed from spread + slippage + overnight financing × holding hours, then compared against a configurable `MIN_EDGE_THRESHOLD`. | Profitable in spreadsheet ≠ profitable after frictions. The threshold is the friction. |
| Tail-risk gate | Rolling volatility window, negative-price streak detector, gap-size monitor, and an aggregate score. Above `MAX_TAIL_RISK_SCORE`, signals are blocked. | The drawdowns that kill systems happen on a handful of days. The gate is for those days. |
| Adaptive retraining | Rolling 365-day training window, drift detection on a 6-hour interval, and a signal-mode escalator (`NORMAL → WATCH_ONLY`) keyed off rolling profit-factor. | The model degrades silently; the system shouldn't. |
| Shadow mode | A parallel pipeline scores live data against a candidate model, writes results to a separate table, and computes an out-of-sample P&L without emitting signals. | New models earn promotion on real data, not just backtests. |
| Signal daemon | Single long-running process (`app.runtime.signal_daemon`) with state file, stop-signal file, wall-clock-jump detection, post-wake grace period, and battery-mode loop interval. Installable as a macOS LaunchAgent or Windows service via `infra/` scripts. | The loop has to survive a closed laptop, a flaky network, and a system clock that jumps an hour. |
| Notifications | Telegram bot with a typed message catalogue, dedup window, minimum signal level filter, blocked-signal pass-through, and an interactive command set (`/status`, `/last`, `/backtest`, `/models`). | Operators get signals out of the dashboard and into where they actually look. |
| Backtest engine | Walk-forward backtester with the same cost model and tail-risk gate the live engine uses; outputs Sharpe, Sortino, win rate, max drawdown, rolling PF, and a calibration report. | Backtests that don't share code with live are best-case fiction. |
| Observability | `structlog` JSON logs, request-id middleware, Prometheus metrics endpoint, daemon health JSON, and a Telegram daily summary. | Production incidents start with "what was the system thinking?" — this answers that. |

---

## Architecture

```mermaid
flowchart LR
    SMARD[SMARD<br/>DE spot + gen mix]
    ENTSOE[ENTSO-E<br/>cross-border, load]
    OPENMETEO[Open-Meteo<br/>wind, solar, temp]

    SMARD & ENTSOE & OPENMETEO --> INGEST[Ingestion<br/>Celery beat tasks]
    INGEST --> PG[(Postgres 16<br/>price + features + signals)]

    PG --> FE[Feature engineering<br/>app/features]
    FE --> ML[ML inference<br/>app/ml]
    ML --> SE[Signal engine<br/>app/signals]
    SE --> RG[Risk + tail-risk gate<br/>app/risk · app/guards]

    RG -->|emit| API[FastAPI<br/>REST + WS]
    RG -->|paper-trade| PAPER[Paper ledger<br/>app/paper]
    RG -->|shadow| SHADOW[Shadow lane<br/>app/runtime/shadow_outcome_evaluator]
    RG -->|notify| TG[Telegram bot<br/>app/notifications]

    DAEMON[Signal daemon<br/>app.runtime.signal_daemon] -.controls.-> SE
    API --> UI[React 18 + Vite<br/>frontend]
```

Each box on the right of the gate is independent — paper trades, shadow
runs, and notifications can be disabled or rewritten without touching the
signal engine.

```
backend/app/
  api/routes/      thin HTTP adapters (futures, forecast, backtest, paper,
                   battery, daemon, telegram, shadow, drift, …)
  core/            settings (pydantic), logging (structlog)
  db/              SQLAlchemy 2.0 async + sync engines, ORM models
  data/            SMARD / ENTSO-E / Open-Meteo clients
  features/        feature engineering + battery features
  ml/              gradient-boosted price + spike + regime models
  signals/         signal engine, threshold rules, reasons
  risk/, guards/   Futures cost model + tail-risk gate
  paper/           paper-trade ledger + P&L
  backtest/        walk-forward backtester (shared cost model)
  regime/          regime classifier + transition detection
  adaptation/      drift detector, retrain scheduler, signal-mode escalator
  runtime/         signal daemon + shadow-mode evaluator
  notifications/   Telegram bot, message templates, command handlers
  jobs/            Celery app + beat schedule
  main.py          FastAPI factory
```

```
frontend/src/
  pages/           Overview, FuturesSignal, Forecast, CostSimulator, Backtest,
                   PaperTrading, DataQuality, BatteryIntelligence,
                   TailRiskMonitor, SignalStability, DaemonStatus,
                   TelegramSettings, ShadowMode, AutoRetraining, DriftMonitor
  components/      Layout, KPICard, SignalCard
  api/             axios client
```

Sequence diagrams for the ingest → feature → score → gate → emit flow and
the daemon lifecycle live in [`docs/architecture.md`](docs/architecture.md).

---

## Demo flow

```bash
# 1. Boot the stack.
cp .env.example .env             # set ENTSOE_API_KEY (free) at minimum
cp backend/.env.example backend/.env
docker compose up --build -d

# 2. Run migrations.
docker compose exec backend alembic upgrade head

# 3. Seed historical data (last 90 days from SMARD/ENTSO-E).
docker compose exec backend python -m app.scripts.bulk_load_historical
```

Then in the browser at **http://localhost:3000**:

1. **Overview** — current signal, live price ticker, daemon status, latest
   shadow-mode entries.
2. **Futures Signal** — the active signal with its full reason set
   (`p_rebound`, `net_edge`, `tail_risk_score`, blocked-by-X), the cost
   breakdown, and the suggested entry / target window.
3. **Cost Simulator** — change spread, overnight rate, holding hours, and
   safety buffer; the threshold the engine uses updates live.
4. **Tail Risk Monitor** — current gate state, the rolling volatility, the
   negative-price streak counter, and the list of signals it blocked.
5. **Backtest** — walk-forward run over a chosen window; compare two
   model versions side by side.
6. **Shadow Mode** — out-of-sample P&L of the candidate vs. the live model.
7. **Auto Retraining** — drift state, rolling profit factor, the
   `NORMAL → WATCH_ONLY` mode escalator, and the last retrain log.
8. **Daemon** — process health, last loop tick, consecutive errors, last
   wake-from-sleep recovery.

API docs: **http://localhost:8000/docs** · Flower (Celery): **http://localhost:5555**.

---

## Local setup

### Docker (recommended)

```bash
cp .env.example .env             # tweak ports / set ENTSOE_API_KEY
cp backend/.env.example backend/.env
docker compose up --build
docker compose exec backend alembic upgrade head
```

| Service       | URL                              |
| ------------- | -------------------------------- |
| Frontend      | http://localhost:3000            |
| Backend       | http://localhost:8000            |
| API docs      | http://localhost:8000/docs       |
| Flower        | http://localhost:5555            |
| Postgres      | localhost:5433                   |
| Redis         | localhost:6380                   |

### Run services manually

Requires Python 3.11+, Node 20+, Postgres 16, Redis 7.

```bash
# 1. Backend
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
alembic upgrade head
uvicorn app.main:app --reload --port 8000

# 2. Celery worker + beat (separate terminals)
celery -A app.jobs.celery_app worker --loglevel=info --concurrency=2
celery -A app.jobs.celery_app beat --loglevel=info

# 3. Signal daemon
python -m app.runtime.signal_daemon

# 4. Frontend
cd frontend
npm install
npm run dev
```

### Daemon as a system service (optional)

- **macOS** — `bash infra/macos/install.sh` registers it as a LaunchAgent
  that survives sleep/wake. `bash infra/macos/uninstall.sh` removes it.
- **Windows** — `infra/windows/install-taskscheduler.bat` registers a
  Scheduled Task; `infra/windows/install-nssm.bat` registers it as a
  Windows service via NSSM.

---

## API overview

All endpoints documented interactively at http://localhost:8000/docs.

| Method | Path                                | Description                                   |
| ------ | ----------------------------------- | --------------------------------------------- |
| GET    | `/health`                           | Service liveness                              |
| GET    | `/api/futures/signal/current`           | Currently active signal with reasons          |
| GET    | `/api/futures/signal/history`           | Past signals (paginated, filterable)          |
| GET    | `/api/forecast/next-hours`          | Price forecast with confidence intervals      |
| GET    | `/api/backtest/runs`                | Walk-forward backtest results                 |
| POST   | `/api/backtest/run`                 | Trigger a new backtest run                    |
| GET    | `/api/paper/positions`              | Open paper-trade positions                    |
| GET    | `/api/paper/performance`            | Aggregate P&L, win rate, Sharpe, drawdown     |
| GET    | `/api/risk/tail-risk`               | Current tail-risk gate state                  |
| GET    | `/api/data/quality`                 | Data freshness + completeness per source      |
| GET    | `/api/daemon/status`                | Loop tick, error counter, last wake event     |
| GET    | `/api/shadow/runs`                  | Shadow-mode candidate-vs-live comparison      |
| GET    | `/api/adaptation/drift`             | Drift detector state + signal-mode escalator  |
| POST   | `/api/notifications/telegram/test`  | Send a test message through the bot           |
| GET    | `/api/battery/intelligence`         | Battery-feature snapshot for arbitrage view   |

### Sample current signal response

```json
{
  "signal_id": "9a1f…",
  "emitted_at": "2024-09-17T14:00:00Z",
  "market": "DE_LU",
  "level": "ENTER",
  "direction": "LONG",
  "p_rebound": 0.74,
  "net_edge_eur_mwh": 38.4,
  "tail_risk_score": 0.21,
  "cost_breakdown": {
    "spread": 5.0,
    "slippage": 3.0,
    "overnight": 0.0,
    "safety_buffer": 5.0,
    "min_edge_threshold": 30.0
  },
  "reasons": ["p_rebound>=0.70", "net_edge>=30.0", "tail_risk<0.65"],
  "expires_at": "2024-09-17T15:00:00Z",
  "model_name": "rebound_lgbm_v17",
  "used_shadow": false
}
```

### Error envelope

```json
{
  "error": {
    "code": "data_stale",
    "message": "SMARD latest price is 102 minutes old (max 90).",
    "details": { "source": "smard", "age_minutes": 102 }
  }
}
```

---

## Environment variables

`.env` (root, used by `docker compose`) and `backend/.env` (used when
running uvicorn directly) — templates at `.env.example` and
`backend/.env.example`. Values shown are the defaults; only `ENTSOE_API_KEY`
is required for live data.

### Application

| Variable | Default | Description |
| --- | --- | --- |
| `APP_ENV` | `development` | `development` or `production` |
| `SIGNAL_ONLY` | `true` | Locks the system to signal mode; cannot be disabled |
| `SECRET_KEY` | *required in prod* | JWT signing key (`openssl rand -hex 32`) |

### Data sources

| Variable | Default | Description |
| --- | --- | --- |
| `ENTSOE_API_KEY` | *empty* | ENTSO-E Transparency Platform API key (free at https://transparency.entsoe.eu) |
| `SMARD_BASE_URL` | `https://www.smard.de/app/chart_data` | |
| `ENTSOE_BASE_URL` | `https://transparency.entsoe.eu/api` | |
| `OPENMETEO_BASE_URL` | `https://api.open-meteo.com/v1/forecast` | |

### Persistence

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+asyncpg://ppuser:pppass@postgres:5432/powerprice` | Async SQLAlchemy DSN |
| `DATABASE_URL_SYNC` | `postgresql+psycopg2://...` | Sync DSN for Alembic |
| `REDIS_URL` | `redis://redis:6379/0` | |
| `CELERY_BROKER_URL` | `redis://redis:6379/1` | |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/2` | |

### Signal engine

| Variable | Default | Description |
| --- | --- | --- |
| `FUTURES_AVG_SPREAD_EUR_MWH` | `5.0` | |
| `FUTURES_SLIPPAGE_EUR_MWH` | `3.0` | |
| `FUTURES_OVERNIGHT_FEE_ANNUAL_PCT` | `8.0` | |
| `FUTURES_SAFETY_BUFFER_EUR_MWH` | `5.0` | |
| `FUTURES_MIN_EDGE_THRESHOLD` | `30.0` | Minimum net edge in EUR/MWh to emit |
| `FUTURES_P_REBOUND_ENTRY` | `0.70` | Threshold for `ENTER` |
| `FUTURES_P_REBOUND_WATCH` | `0.55` | Threshold for `WATCH` |
| `SIGNAL_MODE` | `NORMAL` | `NORMAL` or `WATCH_ONLY` |
| `SHADOW_MODE_ENABLED` | `true` | |

### Tail risk

| Variable | Default | Description |
| --- | --- | --- |
| `MAX_NEGATIVE_PRICE_EUR_MWH` | `-150.0` | |
| `MAX_NEGATIVE_STREAK_HOURS` | `3` | |
| `MAX_GAP_SIZE_EUR_MWH` | `100.0` | |
| `MAX_TAIL_RISK_SCORE` | `0.65` | |

### Daemon

| Variable | Default | Description |
| --- | --- | --- |
| `SIGNAL_DAEMON_ENABLED` | `true` | |
| `SIGNAL_LOOP_INTERVAL_MINUTES` | `15` | On AC power |
| `SIGNAL_LOOP_INTERVAL_BATTERY_MINUTES` | `30` | On battery |
| `DATA_STALE_MAX_MINUTES` | `90` | Abort loop if data older than this |
| `MAX_CONSECUTIVE_ERRORS` | `5` | Stop daemon after this many in a row |
| `WAKE_DETECTION_ENABLED` | `true` | Detect wall-clock jump on system wake |
| `POST_WAKE_GRACE_SECONDS` | `15` | Wait this long after wake before resuming |

### Telegram (optional)

| Variable | Default | Description |
| --- | --- | --- |
| `TELEGRAM_ENABLED` | `false` | |
| `TELEGRAM_BOT_TOKEN` | *empty* | Get one from @BotFather |
| `TELEGRAM_CHAT_ID` | *empty* | https://api.telegram.org/bot<TOKEN>/getUpdates |
| `TELEGRAM_MIN_SIGNAL_LEVEL` | `WATCH` | `WATCH`, `ENTER`, `HIGH_CONFIDENCE` |
| `SIGNAL_DEDUP_MINUTES` | `60` | |

### Adaptive retraining

| Variable | Default | Description |
| --- | --- | --- |
| `AUTO_RETRAIN_ENABLED` | `true` | |
| `ROLLING_TRAINING_DAYS` | `365` | |
| `DRIFT_CHECK_ENABLED` | `true` | |
| `DRIFT_CHECK_INTERVAL_HOURS` | `6` | |

---

## Data sources

| Source | Data | License |
| --- | --- | --- |
| [SMARD](https://www.smard.de) | DE spot prices, generation mix, load | Open Data (BNetzA) |
| [ENTSO-E Transparency](https://transparency.entsoe.eu) | Pan-European load, cross-border flows, prices | Free, API key required |
| [Open-Meteo](https://open-meteo.com) | Wind, solar, temperature forecasts | CC BY 4.0 |

---

## ML models

| Model | Task | Algorithm |
| --- | --- | --- |
| `rebound_lgbm` | Probability of next-hour mean reversion (`p_rebound`) | LightGBM |
| `price_regression` | Expected price move in EUR/MWh | Gradient Boosting |
| `regime_classifier` | Market regime (calm / normal / spike) | Random Forest |
| `spike_detector` | Probability of >2σ price spike | Isolation Forest + XGBoost |
| `negative_price_classifier` | Probability of next-hour negative price | LightGBM |

Models are versioned by timestamp suffix and retrained nightly on a
rolling 365-day window. Drift is checked every 6 hours; a sustained
profit-factor below `signal_mode_pf_floor` switches the mode escalator to
`WATCH_ONLY`.

---

## Tests

```bash
cd backend
pytest -q
```

Coverage:

- `tests/test_signal_engine.py` — threshold rules, reason ordering, blocked states
- `tests/test_futures_cost_model.py` — per-signal cost across spread / overnight / hold-hours
- `tests/test_daemon.py` — wall-clock jump, post-wake grace, consecutive-error tripwire
- `tests/test_drift_detector.py` — drift trigger thresholds
- `tests/test_backtest.py` — walk-forward correctness + cost-model parity with live
- `tests/test_telegram.py` — message templates, dedup, level filter

---

## Project structure

```
powerprice-futures-signals/
├── backend/
│   ├── app/
│   │   ├── api/routes/        FastAPI routers (futures, forecast, paper, …)
│   │   ├── core/              config, logging
│   │   ├── db/                SQLAlchemy models, sessions
│   │   ├── data/              SMARD / ENTSO-E / Open-Meteo clients
│   │   ├── features/          feature engineering
│   │   ├── ml/                price + spike + regime models
│   │   ├── signals/           signal engine
│   │   ├── risk/, guards/     cost model + tail-risk gate
│   │   ├── paper/             paper-trade ledger
│   │   ├── backtest/          walk-forward backtester
│   │   ├── regime/            regime classifier
│   │   ├── adaptation/        drift detector + retrain scheduler
│   │   ├── runtime/           signal daemon + shadow evaluator
│   │   ├── notifications/     Telegram bot
│   │   ├── jobs/              Celery app + beat schedule
│   │   ├── services/          higher-level orchestration
│   │   └── main.py
│   ├── alembic/               migrations
│   ├── scripts/               training, backtests, bulk load
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                  React 18 + Vite + Tailwind
│   ├── src/
│   │   ├── api/               axios client
│   │   ├── components/        Layout, KPICard, SignalCard
│   │   ├── hooks/             useSignal
│   │   └── pages/             15 dashboard pages
│   ├── Dockerfile
│   └── nginx.frontend.conf
├── infra/
│   ├── macos/                 LaunchAgent installer
│   ├── windows/               Scheduled Task + NSSM installers
│   └── nginx.conf             reverse-proxy config
├── k8s/                       Kubernetes manifests
├── docs/architecture.md
├── docker-compose.yml
└── .gitignore
```

---

## Limitations

Intentional v1 boundaries:

- **Single market.** German EPEX day-ahead only; the schema is multi-market
  ready, the data clients aren't wired for others yet.
- **Hourly resolution.** Quarter-hourly (CWE intraday) would need a separate
  data path and a different cost model.
- **Synchronous backtest.** Walk-forward runs in-process; long windows need
  a background queue.
- **No auth on read endpoints.** Single-tenant local tool.

---

## Future work

- Quarter-hourly intraday signal lane
- Additional bidding zones (FR, NL, AT)
- Cross-encoder reranker over the daily candidate set
- OpenTelemetry tracing across the daemon and Celery tasks
- Promotion workflow from shadow → live with required out-of-sample window

---

## License

MIT — see [`LICENSE`](LICENSE). Built in 2025 as a portfolio project.
