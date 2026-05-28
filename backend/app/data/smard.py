"""
German electricity price data connector.

Primary source: aWATTar DE API (free, no key, EPEX SPOT Day-Ahead prices for Germany)
  https://api.awattar.de/v1/marketdata

Generation proxies derived from Open-Meteo weather data when SMARD
generation data is unavailable (SMARD API URL format changed in 2024).

aWATTar provides the same Day-Ahead prices as EPEX SPOT / SMARD for DE-LU.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)

AWATTAR_BASE = "https://api.awattar.de/v1/marketdata"

_RETRY_DELAYS = [1.0, 2.0, 4.0]


async def _get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: Optional[dict] = None,
    timeout: float = 30.0,
) -> httpx.Response:
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt, delay in enumerate([0.0] + _RETRY_DELAYS, start=1):
        if delay:
            await asyncio.sleep(delay)
        try:
            resp = await client.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Request failed (attempt %d/%d): %s — retrying in %.0fs",
                attempt, len(_RETRY_DELAYS) + 1, exc, delay,
            )
            last_exc = exc
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            logger.warning("Network error (attempt %d): %s", attempt, exc)
            last_exc = exc
    raise last_exc


def _ms_to_utc(ms: int) -> datetime:
    # Keep UTC tzinfo so PostgreSQL stores as UTC, not as local (CEST) time
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


async def _fetch_awattar_prices(
    start: datetime, end: datetime, client: httpx.AsyncClient
) -> pd.DataFrame:
    """
    Fetch Day-Ahead prices from aWATTar DE API.
    Returns DataFrame with columns: timestamp (UTC-naive, hourly), price_eur_mwh
    """
    start_ms = int((start if start.tzinfo else start.replace(tzinfo=timezone.utc)).timestamp() * 1000)
    end_ms = int((end if end.tzinfo else end.replace(tzinfo=timezone.utc)).timestamp() * 1000)

    try:
        resp = await _get_with_retry(
            client, AWATTAR_BASE, params={"start": start_ms, "end": end_ms},
        )
        data = resp.json().get("data", [])
    except Exception as exc:
        logger.error("aWATTar fetch failed: %s", exc)
        return pd.DataFrame(columns=["timestamp", "price_eur_mwh"])

    if not data:
        logger.warning("aWATTar returned 0 data points")
        return pd.DataFrame(columns=["timestamp", "price_eur_mwh"])

    rows = []
    for entry in data:
        ts = _ms_to_utc(entry["start_timestamp"])
        price = entry.get("marketprice")
        if price is not None:
            rows.append({"timestamp": ts, "price_eur_mwh": float(price)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("h")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    logger.info("aWATTar: fetched %d hourly price records", len(df))
    return df


async def _fetch_generation_proxy(
    start: datetime, end: datetime, client: httpx.AsyncClient
) -> pd.DataFrame:
    """
    Derive wind/solar/load proxies from Open-Meteo weather data.
    Used because SMARD generation API URLs changed in 2024.
    Returns: timestamp, solar_mw, wind_onshore_mw, wind_offshore_mw, load_mw
    """
    now_utc = datetime.now(timezone.utc)
    start_utc = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
    hours_back = max(1, int((now_utc - start_utc).total_seconds() / 3600))
    past_days = min(92, max(1, hours_back // 24 + 2))

    params = {
        "latitude": 51.5,
        "longitude": 10.0,
        "hourly": "shortwave_radiation,wind_speed_10m,temperature_2m,cloud_cover",
        "past_days": past_days,
        "forecast_days": 3,
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }

    try:
        resp = await _get_with_retry(client, settings.openmeteo_base_url, params=params)
        payload = resp.json()
    except Exception as exc:
        logger.error("Open-Meteo generation proxy failed: %s", exc)
        return pd.DataFrame()

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(times, utc=True).floor("h"),
        "solar_radiation_wm2": hourly.get("shortwave_radiation", [None] * len(times)),
        "wind_speed_ms": hourly.get("wind_speed_10m", [None] * len(times)),
        "temperature_c": hourly.get("temperature_2m", [None] * len(times)),
        "cloud_cover_pct": hourly.get("cloud_cover", [None] * len(times)),
    })

    # Proxy generation from weather (Germany installed capacity estimates)
    df["solar_mw"] = (df["solar_radiation_wm2"].fillna(0) * 60.0).clip(0, 70000)
    df["wind_onshore_mw"] = (df["wind_speed_ms"].fillna(0) ** 2.5 * 400).clip(0, 65000)
    df["wind_offshore_mw"] = (df["wind_speed_ms"].fillna(0) ** 2.5 * 90).clip(0, 9000)
    df["load_mw"] = (55000 + (15 - df["temperature_c"].fillna(15)) * 800).clip(35000, 90000)

    _start_utc = pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
    _end_utc = pd.Timestamp(end).tz_localize("UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
    mask = (df["timestamp"] >= _start_utc) & (df["timestamp"] <= _end_utc)
    df = df[mask].reset_index(drop=True)

    logger.info("Generation proxy: %d rows (solar max %.0f MW, wind max %.0f MW)",
                len(df),
                df["solar_mw"].max() if len(df) else 0,
                df["wind_onshore_mw"].max() if len(df) else 0)

    # Battery proxy: Germany ~12 GW installed storage (2025)
    # Charge when renewables exceed load (oversupply), discharge otherwise
    BATT_CAPACITY_MW = 12_000.0
    oversupply = (
        df.get("wind_onshore_mw", pd.Series(0.0, index=df.index)).fillna(0.0)
        + df.get("wind_offshore_mw", pd.Series(0.0, index=df.index)).fillna(0.0)
        + df.get("solar_mw", pd.Series(0.0, index=df.index)).fillna(0.0)
        - df.get("load_mw", pd.Series(55_000.0, index=df.index)).fillna(55_000.0)
    )
    df["battery_charge_mw"] = (oversupply.clip(lower=0) * 0.12).clip(0, BATT_CAPACITY_MW)
    df["battery_discharge_mw"] = ((-oversupply).clip(lower=0) * 0.05).clip(0, BATT_CAPACITY_MW * 0.3)
    df["battery_net_mw"] = df["battery_charge_mw"] - df["battery_discharge_mw"]
    return df


async def fetch_recent(hours_back: int = 48) -> pd.DataFrame:
    """
    Fetch recent hourly electricity market data for Germany.

    Returns DataFrame with columns:
        timestamp, price_eur_mwh, load_mw, wind_onshore_mw,
        wind_offshore_mw, solar_mw

    Price data: aWATTar DE (EPEX SPOT Day-Ahead DE-LU, free API)
    Generation data: Open-Meteo weather-derived proxies
    """
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    start = end - timedelta(hours=hours_back + 1)

    async with httpx.AsyncClient(
        headers={"Accept": "application/json", "User-Agent": "PowerPriceFutures/1.0"},
        follow_redirects=True,
    ) as client:
        results = await asyncio.gather(
            _fetch_awattar_prices(start, end, client),
            _fetch_generation_proxy(start, end, client),
            return_exceptions=True,
        )

    price_df = results[0] if not isinstance(results[0], Exception) else pd.DataFrame(
        columns=["timestamp", "price_eur_mwh"])
    gen_df = results[1] if not isinstance(results[1], Exception) else pd.DataFrame()

    if isinstance(results[0], Exception):
        logger.error("Price fetch error: %s", results[0])
    if isinstance(results[1], Exception):
        logger.error("Generation proxy error: %s", results[1])

    if price_df.empty:
        logger.warning("No price data available — returning empty DataFrame")
        return pd.DataFrame(
            columns=["timestamp", "price_eur_mwh", "load_mw",
                     "wind_onshore_mw", "wind_offshore_mw", "solar_mw"])

    if not gen_df.empty:
        gen_cols = [c for c in ["timestamp", "load_mw", "wind_onshore_mw",
                                "wind_offshore_mw", "solar_mw"] if c in gen_df.columns]
        merged = price_df.merge(gen_df[gen_cols], on="timestamp", how="left")
    else:
        merged = price_df.copy()
        for col in ["load_mw", "wind_onshore_mw", "wind_offshore_mw", "solar_mw"]:
            merged[col] = None

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    cutoff = end - timedelta(hours=hours_back)
    merged = merged[merged["timestamp"] >= cutoff].reset_index(drop=True)

    neg_count = int((merged["price_eur_mwh"] < 0).sum())
    logger.info(
        "fetch_recent: %d rows | EUR/MWh range %.1f–%.1f | negative: %d h",
        len(merged),
        merged["price_eur_mwh"].min() if len(merged) else 0,
        merged["price_eur_mwh"].max() if len(merged) else 0,
        neg_count,
    )
    return merged
