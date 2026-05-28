# Battery Storage Intelligence

This document describes how battery storage data is integrated into the
PowerPrice Futures Signals platform as a structural market factor.

---

## Why Battery Storage Affects Negative Electricity Prices

German electricity prices go negative when supply exceeds demand and cannot
be curtailed quickly. Battery storage interacts with this in several ways:

| Situation | Battery effect | Signal impact |
|---|---|---|
| PV midday surplus | Batteries charge → absorb excess | Rebound may be weaker / later |
| Batteries near full | Cannot absorb more supply | Negative prices EXTEND, rebound delayed |
| Evening demand spike | Batteries discharge → add supply | Competes with price rebound |
| Batteries empty + price rebounds | Charge from grid | Dampens rebound speed |
| High wind overnight | Batteries charge → reduce curtailment | Shorter negative windows |

**Net effect on our signal:** When `battery_saturation_proxy > 0.85`, the
HIGH_CONFIDENCE_SIGNAL is blocked because a rebound is unlikely to materialise
quickly when storage is saturated.

---

## Data Sources

### 1. ENTSO-E Transparency Platform (Primary — requires API key)

**What we fetch:**
- `documentType=A75` (Actual Generation Per Production Type)
- `PsrType=B10` — Hydro Pumped Storage (main dispatchable storage in DE)
- `PsrType=B14` — Battery Storage (limited granularity in DE 2024–2025)

**Availability:**
- Pumped hydro (B10): ~30 min delay, reliable
- Battery (B14): Most German batteries are below individual reporting thresholds;
  aggregate data has limited granularity as of 2025

**Setup:**
```bash
# backend/.env
ENTSOE_API_KEY=your-key-from-transparency.entsoe.eu
```
Register at [ENTSO-E Transparency Platform](https://transparency.entsoe.eu).

**Note:** ENTSO-E A75 only provides the *generation* (discharging) side.
The charging side is derived from supply/demand balance as a proxy.

### 2. Proxy (Fallback — always available)

When no ENTSO-E key is configured, all battery data is derived from
price and generation signals. Quality score: **0.35** (clearly labelled).

**Proxy logic:**
```
charge_pressure = f(price < 10 EUR/MWh, PV surplus, midday hour)
discharge_pressure = f(price > 60 EUR/MWh, evening hour, high load)
battery_charging_mw = charge_pressure × installed_power_mw × 0.7
battery_saturation_proxy = rolling 24h integral of net flow / capacity
```

**Limitations:**
- Does not capture operator decisions or ancillary-service obligations
- Ignores grid topology (local congestion vs national balance)
- SoC resets every 24h; true SoC depends on multi-day history
- Installed capacity estimate may lag actual deployment by 1–3 months

### 3. Marktstammdatenregister (MaStR) — Structural capacity data

**What we use:** Installed battery power (MW) and capacity (MWh) for Germany.

**Method:** MaStR REST API with 24h cache. Falls back to milestone
interpolation from known capacity milestones (Fraunhofer ISE / BNetzA).

**Known milestones:**

| Date | Installed Power | Installed Capacity |
|------|----------------|-------------------|
| 2024 Q1 | ~7.5 GW | ~15 GWh |
| 2024 Q4 | ~10.5 GW | ~21 GWh |
| 2025 Q1 | ~11.5 GW | ~23 GWh |
| 2025 Q4 | ~15 GW | ~30 GWh |

---

## Battery Features

The following features are computed by `BatteryFeatureBuilder` and
`compute_storage_proxy`. They are **NOT** in `FeatureEngineer.FEATURE_COLUMNS`
(to avoid retraining existing XGBoost models).

| Feature | Description | Range |
|---|---|---|
| `battery_charging_mw` | Estimated charging power | 0 – installed_power_mw |
| `battery_discharging_mw` | Estimated discharge power | 0 – installed_power_mw |
| `net_battery_flow_mw` | discharge - charge (positive = net gen) | ±installed_power |
| `storage_charge_pressure` | Likelihood batteries are charging | 0 – 1 |
| `storage_discharge_pressure` | Likelihood batteries are discharging | 0 – 1 |
| `battery_saturation_proxy` | Estimated state of charge | 0 – 1 |
| `pv_surplus_after_load` | Excess renewable generation above load | MW |
| `midday_price_compression` | How compressed midday prices are | 0 – 1 |
| `evening_arbitrage_spread` | Evening high–low 6h spread | EUR/MWh |
| `expected_battery_absorption` | Projected charging next hour | MW |
| `expected_battery_release` | Projected discharge next hour | MW |
| `battery_adjusted_residual_load` | Residual load ± battery flows | MW |
| `battery_installed_power_mw` | Current installed capacity | MW |
| `battery_installed_capacity_mwh` | Current installed energy capacity | MWh |

---

## Signal Engine Integration

### HIGH_CONFIDENCE_SIGNAL conditions (all must be true):
```
price < 0
AND p_rebound >= 0.60
AND net_edge >= regime_hc_threshold (35 EUR/MWh default)
AND residual_load_ramp_1h > 0         (load rising)
AND (solar_ramp_1h < 0 OR evening_demand_spike)  (directional confirmation)
AND battery_saturation_proxy < 0.85   ← NEW battery guard
AND expected_battery_absorption < 8000 MW  ← NEW battery guard
```

### ENTER_LONG_REBOUND_SIGNAL (battery does not block, but reason string includes battery state):
```
price < 0
AND p_rebound >= 0.60
AND net_edge >= regime_enter_threshold (30 EUR/MWh default)
```

---

## Battery Regimes

Five battery-specific regimes are added to the regime classifier:

| Regime | Trigger | Enter threshold | HC threshold |
|---|---|---|---|
| `STORAGE_SATURATED` | SoC > 85%, charge_pressure > 50% | 35 EUR/MWh | 45 EUR/MWh |
| `HIGH_STORAGE_ABSORPTION` | charge_pressure > 70%, SoC < 80% | 32 EUR/MWh | 40 EUR/MWh |
| `EVENING_DISCHARGE_PRESSURE` | discharge_pressure > 60%, hour >= 17 | 28 EUR/MWh | 35 EUR/MWh |
| `BATTERY_DAMPENED_REBOUND` | SoC > 70%, expected_absorption > 6 GW | 35 EUR/MWh | 45 EUR/MWh |
| `LOW_STORAGE_IMPACT` | Fallback (no strong battery signal) | 30 EUR/MWh | 35 EUR/MWh |

Battery regimes are checked AFTER VOLATILE and WINTER_LOW but BEFORE
SOLAR_OVERSUPPLY and WIND_OVERSUPPLY in the priority tree.

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /battery/latest` | Installed capacity + latest proxy state |
| `GET /battery/capacity` | Installed capacity estimate with source |
| `GET /battery/flows?hours=48` | Historical charge/discharge flows |
| `GET /battery/proxy?hours=48` | Full proxy feature table |
| `GET /battery/features` | Current battery features for signal engine |
| `GET /battery/regime-impact` | Battery state vs regime thresholds |
| `GET /battery/data-quality` | Data source quality report |

---

## Backtest Comparison

Run `backend/scripts/backtest_battery.py` to compare strategies A–D:

| Strategy | Description |
|---|---|
| A — Baseline | No battery filter (existing logic) |
| B — +Saturation gate | Block if battery_saturation_proxy > 85% |
| C — +Charge pressure | Block if storage_charge_pressure > 80% |
| D — +Regime filter | Block STORAGE_SATURATED + BATTERY_DAMPENED regimes |

Expected outcome: Strategies B–D reduce trade count but improve Profit Factor
by eliminating entries where batteries suppress the expected rebound.

---

## Transparency Statement

Battery storage influence on German electricity prices is **structurally
important** but **difficult to measure in real time**. This module implements
conservative, documented proxies rather than overconfident estimates.

Key principle: **No hard signal decisions are made solely on battery proxy data
with quality_score < 0.5.** The battery features influence:
1. Regime classification (rule-based adjustments to thresholds)
2. HIGH_CONFIDENCE gate (additional prerequisite, not sole criterion)
3. Signal reason string (transparent explanation of battery state)

When data quality is poor, the signal engine falls back to the standard
ENTER_LONG_REBOUND_SIGNAL rather than blocking the trade entirely.
