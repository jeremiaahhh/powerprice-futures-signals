"""
Feature engineering module for the PowerPrice Futures Signals platform.

Transforms raw HourlyPrice data into ML-ready feature matrices covering
cyclical time encodings, renewable penetration, lag/rolling statistics,
demand patterns, volatility proxies, and negative-price history signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, List

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# German public holidays (static list; extend or replace with a holidays library
# if the dependency is available).
_GERMAN_HOLIDAYS: List[Tuple[int, int]] = [
    (1, 1),   # New Year
    (5, 1),   # Labour Day
    (10, 3),  # German Unity Day
    (11, 1),  # All Saints (some states)
    (12, 25), # Christmas Day
    (12, 26), # Boxing Day
]


def _is_statutory_holiday(dt: pd.Timestamp) -> bool:
    """Return True if *dt* falls on a known German statutory holiday."""
    return (dt.month, dt.day) in _GERMAN_HOLIDAYS


class FeatureEngineer:
    """Build ML feature matrices from raw HourlyPrice DataFrames.

    The input DataFrame is expected to have the columns that map 1-to-1 to
    the ``HourlyPrice`` SQLAlchemy model:

        timestamp, price_eur_mwh, load_mw, wind_onshore_mw,
        wind_offshore_mw, solar_mw, residual_load_mw, net_export_mw,
        temperature_c, wind_speed_ms, solar_radiation_wm2,
        cloud_cover_pct, is_holiday, is_weekend, hour, month

    All feature derivation is purely pandas/numpy – no external data sources
    are called at inference time.
    """

    # ------------------------------------------------------------------
    # Canonical ordered list of feature columns produced by build_features().
    # Models are trained and served against exactly these names in this order.
    # ------------------------------------------------------------------
    FEATURE_COLUMNS: List[str] = [
        # Cyclical time
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        # Calendar flags
        "is_weekend",
        "is_holiday",
        # Renewable / grid balance
        "renewable_share",
        "residual_load_mw",
        "wind_total_mw",
        # Ramp features
        "solar_ramp_1h",
        "wind_ramp_1h",
        "residual_load_ramp_1h",
        # Lag features (price)
        "price_lag_1h",
        "price_lag_2h",
        "price_lag_4h",
        "price_lag_24h",
        "price_lag_48h",
        # Rolling statistics (price)
        "price_rolling_mean_4h",
        "price_rolling_mean_24h",
        "price_rolling_mean_7d",
        "price_rolling_std_4h",
        "price_rolling_std_24h",
        # Demand / supply pattern flags
        "evening_demand_spike",
        "low_demand_weekend",
        # Volatility
        "price_volatility_24h",
        # Negative-price history
        "hours_in_negative_last_24h",
        "negative_price_streak",
        # Interaction features
        "solar_forecast_evening",
        "wind_x_renewable_share",
        # Passthrough raw features useful for tree models
        "load_mw",
        "solar_mw",
        "wind_speed_ms",
        "temperature_c",
        "solar_radiation_wm2",
        "cloud_cover_pct",
        "net_export_mw",
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a new DataFrame containing all engineered features.

        Parameters
        ----------
        df:
            Raw HourlyPrice DataFrame.  Must be sorted by ``timestamp``
            ascending before calling this method (or sorting will be
            applied internally).

        Returns
        -------
        pd.DataFrame
            Same index as *df*, columns = ``FEATURE_COLUMNS``.  Rows where
            insufficient history exists for lag/rolling features will contain
            NaN values – callers should decide whether to drop or impute them.
        """
        df = df.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.debug("Building features for %d rows", len(df))

        out = pd.DataFrame(index=df.index)

        # ---- Resolve timestamp series --------------------------------
        ts: pd.Series = pd.to_datetime(df["timestamp"])

        # ---- Cyclical time encodings ---------------------------------
        hour = ts.dt.hour.astype(float)
        dow = ts.dt.dayofweek.astype(float)   # Monday=0
        month = ts.dt.month.astype(float)

        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        out["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7.0)
        out["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7.0)
        out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
        out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

        # ---- Calendar flags ------------------------------------------
        # Prefer pre-computed columns when available; derive otherwise.
        if "is_weekend" in df.columns:
            out["is_weekend"] = df["is_weekend"].astype(int)
        else:
            out["is_weekend"] = (dow >= 5).astype(int)

        if "is_holiday" in df.columns:
            out["is_holiday"] = df["is_holiday"].astype(int)
        else:
            out["is_holiday"] = ts.apply(_is_statutory_holiday).astype(int)

        # ---- Renewable / grid balance --------------------------------
        wind_total = (
            df.get("wind_onshore_mw", pd.Series(0.0, index=df.index)).fillna(0.0)
            + df.get("wind_offshore_mw", pd.Series(0.0, index=df.index)).fillna(0.0)
        )
        solar = df.get("solar_mw", pd.Series(0.0, index=df.index)).fillna(0.0)
        load = df.get("load_mw", pd.Series(np.nan, index=df.index))

        out["wind_total_mw"] = wind_total

        # renewable_share = (wind + solar) / load; clamp to [0, 1]
        renewable_gen = wind_total + solar
        with np.errstate(divide="ignore", invalid="ignore"):
            ren_share = np.where(
                load.notna() & (load > 0),
                np.clip(renewable_gen / load, 0.0, 1.0),
                np.nan,
            )
        out["renewable_share"] = ren_share

        if "residual_load_mw" in df.columns and df["residual_load_mw"].notna().any():
            out["residual_load_mw"] = df["residual_load_mw"]
        else:
            # Derive: load - (wind + solar)
            out["residual_load_mw"] = load - renewable_gen

        # ---- Ramp features (1-hour difference) ----------------------
        out["solar_ramp_1h"] = solar.diff(1)
        out["wind_ramp_1h"] = wind_total.diff(1)
        out["residual_load_ramp_1h"] = out["residual_load_mw"].diff(1)

        # ---- Price lag features -------------------------------------
        price = df["price_eur_mwh"]
        out["price_lag_1h"] = price.shift(1)
        out["price_lag_2h"] = price.shift(2)
        out["price_lag_4h"] = price.shift(4)
        out["price_lag_24h"] = price.shift(24)
        out["price_lag_48h"] = price.shift(48)

        # ---- Rolling statistics (price) -----------------------------
        # min_periods keeps early rows as NaN rather than producing
        # misleading statistics from too-small windows.
        out["price_rolling_mean_4h"] = price.rolling(4, min_periods=4).mean()
        out["price_rolling_mean_24h"] = price.rolling(24, min_periods=12).mean()
        out["price_rolling_mean_7d"] = price.rolling(168, min_periods=24).mean()
        out["price_rolling_std_4h"] = price.rolling(4, min_periods=4).std()
        out["price_rolling_std_24h"] = price.rolling(24, min_periods=12).std()

        # ---- Demand / supply pattern flags --------------------------
        weekday_flag = (dow < 5).astype(bool)
        evening_hours = hour.isin([17, 18, 19, 20])
        out["evening_demand_spike"] = (evening_hours & weekday_flag).astype(int)

        weekend_flag = (dow >= 5).astype(bool)
        early_hours = hour.isin(range(0, 7))
        out["low_demand_weekend"] = (weekend_flag & early_hours).astype(int)

        # ---- Volatility proxy ---------------------------------------
        out["price_volatility_24h"] = price.rolling(24, min_periods=12).std()

        # ---- Negative-price history ---------------------------------
        neg_flag = (price < 0.0).astype(int)

        # Count of negative hours in the last 24h (excluding current row)
        out["hours_in_negative_last_24h"] = (
            neg_flag.shift(1).rolling(24, min_periods=1).sum()
        )

        # Current consecutive streak of negative prices (including current row)
        streak = []
        current_streak = 0
        for p in price:
            if pd.isna(p):
                streak.append(np.nan)
                current_streak = 0
            elif p < 0:
                current_streak += 1
                streak.append(current_streak)
            else:
                current_streak = 0
                streak.append(0)
        out["negative_price_streak"] = streak

        # ---- Interaction features -----------------------------------
        # solar_forecast_evening: solar output during afternoon/evening hours
        afternoon_evening = hour.between(14, 20)
        out["solar_forecast_evening"] = solar * afternoon_evening.astype(float)

        # wind × renewable_share
        out["wind_x_renewable_share"] = wind_total * pd.Series(
            ren_share, index=df.index
        ).fillna(0.0)

        # ---- Passthrough raw features --------------------------------
        for col in ["load_mw", "solar_mw", "wind_speed_ms", "temperature_c",
                    "solar_radiation_wm2", "cloud_cover_pct"]:
            out[col] = df.get(col, pd.Series(np.nan, index=df.index))
        # net_export_mw is rarely populated; fill with 0 so it doesn't drop all rows
        out["net_export_mw"] = df.get("net_export_mw", pd.Series(np.nan, index=df.index)).fillna(0.0)

        # ---- Battery storage features (proxy — NOT in FEATURE_COLUMNS to avoid model retraining) ----
        battery_net = df.get("battery_net_mw", pd.Series(0.0, index=df.index)).fillna(0.0)
        out["battery_net_mw"] = battery_net
        out["battery_charging"] = (battery_net > 500.0).astype(int)
        # SOC proxy: rolling 24h sum of net battery flow (MWh), normalized by 12 GWh max
        soc_mwh = battery_net.rolling(24, min_periods=1).sum().clip(lower=0)
        out["battery_soc_proxy"] = (soc_mwh / 12_000.0).clip(0, 1.0)
        out["battery_near_full"] = (out["battery_soc_proxy"] > 0.75).astype(int)

        # Preserve original timestamp for downstream joins
        out["timestamp"] = df["timestamp"].values

        logger.debug(
            "Feature engineering complete: %d rows x %d features",
            len(out),
            len(self.FEATURE_COLUMNS),
        )
        return out

    def get_feature_matrix(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Return ``(X, y)`` for supervised learning.

        Parameters
        ----------
        df:
            Raw HourlyPrice DataFrame.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix with columns == ``FEATURE_COLUMNS``.  Rows that
            contain any NaN in X or y are dropped.
        y : pd.Series
            Target series ``price_eur_mwh`` aligned with X.
        """
        features_df = self.build_features(df)
        y = df["price_eur_mwh"].copy()
        y.index = features_df.index

        # Select only canonical feature columns that are actually present
        available = [c for c in self.FEATURE_COLUMNS if c in features_df.columns]
        X = features_df[available].copy()

        # Drop rows with NaN in either X or y
        valid = X.notna().all(axis=1) & y.notna()
        X = X.loc[valid]
        y = y.loc[valid]

        logger.debug(
            "Feature matrix: %d rows after dropping NaN (dropped %d)",
            len(X),
            (~valid).sum(),
        )
        return X, y

    def get_negative_price_labels(
        self,
        df: pd.DataFrame,
        threshold: float = 0.0,
    ) -> pd.Series:
        """Binary classification labels: 1 if ``price_eur_mwh < threshold``.

        Parameters
        ----------
        df:
            Raw HourlyPrice DataFrame.
        threshold:
            Price level below which the label is 1 (default 0.0 EUR/MWh).

        Returns
        -------
        pd.Series of int (0 or 1), same index as *df*.
        """
        labels = (df["price_eur_mwh"] < threshold).astype(int)
        neg_count = labels.sum()
        logger.debug(
            "Negative-price labels: %d positive / %d total (threshold=%.2f)",
            neg_count,
            len(labels),
            threshold,
        )
        return labels

    def get_rebound_labels(
        self,
        df: pd.DataFrame,
        horizon_hours: int = 6,
        min_rebound_eur: float = 10.0,
    ) -> pd.Series:
        """Binary labels for price rebound after a negative-price period.

        Label is 1 when **both** of the following hold:
        - Current ``price_eur_mwh < 0``
        - ``max(price_eur_mwh[t+1 … t+horizon_hours]) > current_price + min_rebound_eur``

        Parameters
        ----------
        df:
            Raw HourlyPrice DataFrame, sorted by timestamp ascending.
        horizon_hours:
            Look-ahead window (default 6 hours).
        min_rebound_eur:
            Minimum absolute recovery required for a positive label.

        Returns
        -------
        pd.Series of int (0 or 1), same index as *df*.  Rows at the tail
        (where no full look-ahead window exists) are labelled 0 rather
        than NaN to simplify downstream usage; callers may choose to drop
        them.
        """
        df = df.copy().sort_values("timestamp").reset_index(drop=True)
        price = df["price_eur_mwh"].values
        n = len(price)
        labels = np.zeros(n, dtype=int)

        for i in range(n):
            current_p = price[i]
            if pd.isna(current_p) or current_p >= 0:
                continue
            # Look-ahead window
            end = min(i + horizon_hours + 1, n)
            future = price[i + 1 : end]
            if len(future) == 0:
                continue
            max_future = np.nanmax(future)
            if max_future > current_p + min_rebound_eur:
                labels[i] = 1

        result = pd.Series(labels, index=df.index, name="rebound_label")
        logger.debug(
            "Rebound labels: %d positive / %d total (horizon=%dh, min_rebound=%.2f)",
            labels.sum(),
            n,
            horizon_hours,
            min_rebound_eur,
        )
        return result
