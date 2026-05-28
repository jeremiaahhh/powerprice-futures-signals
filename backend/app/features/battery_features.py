"""
Battery storage feature builder.

These features are intentionally NOT added to FeatureEngineer.FEATURE_COLUMNS
to avoid invalidating trained XGBoost models.  They are consumed by:
  - RegimeClassifier   (rule-based — reads any Series key)
  - Signal engine      (explicit checks in futures.py)
  - Battery API routes (GET /battery/features)
  - Backtest comparison scripts

Call BatteryFeatureBuilder.build_features(df) to append battery columns to a
HourlyPrice DataFrame.  If battery_flow_df is supplied (real or proxy) the
raw flow columns are merged; otherwise a price-based proxy is computed inline.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

BATTERY_FEATURE_COLUMNS: list[str] = [
    "battery_charging_mw",
    "battery_discharging_mw",
    "net_battery_flow_mw",
    "storage_charge_pressure",
    "storage_discharge_pressure",
    "battery_saturation_proxy",
    "pv_surplus_after_load",
    "midday_price_compression",
    "evening_arbitrage_spread",
    "expected_battery_absorption",
    "expected_battery_release",
    "battery_adjusted_residual_load",
    "battery_installed_power_mw",
    "battery_installed_capacity_mwh",
]

_CHARGE_PRICE_THR    = 10.0
_DISCHARGE_PRICE_THR = 60.0


class BatteryFeatureBuilder:
    """Append battery storage features to a HourlyPrice DataFrame."""

    def __init__(
        self,
        installed_power_mw:    float = 14_000.0,
        installed_capacity_mwh: float = 28_000.0,
    ):
        self.installed_power_mw    = installed_power_mw
        self.installed_capacity_mwh = installed_capacity_mwh

    def build_features(
        self,
        df: pd.DataFrame,
        battery_flow_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Append BATTERY_FEATURE_COLUMNS to df.

        Parameters
        ----------
        df : HourlyPrice DataFrame (must contain timestamp, price_eur_mwh).
        battery_flow_df : Optional real/proxy battery flow DataFrame.
                          Expected columns: timestamp, charging_mw,
                          discharging_mw, net_battery_flow_mw,
                          storage_charge_pressure, storage_discharge_pressure,
                          battery_saturation_proxy.
        """
        df = df.copy().sort_values("timestamp").reset_index(drop=True)

        def _c(name: str, fill: float = 0.0) -> pd.Series:
            return df.get(name, pd.Series(fill, index=df.index)).fillna(fill)

        price     = _c("price_eur_mwh")
        solar     = _c("solar_mw")
        wind_on   = _c("wind_onshore_mw")
        wind_off  = _c("wind_offshore_mw")
        load      = _c("load_mw", 55_000.0)
        residual  = df.get(
            "residual_load_mw",
            pd.Series(load - solar - wind_on - wind_off, index=df.index),
        ).fillna(load - solar - wind_on - wind_off)
        ts   = pd.to_datetime(df["timestamp"])
        hour = ts.dt.hour.astype(float)

        # ---- Source: real flow data or inline proxy ----------------------
        if battery_flow_df is not None and not battery_flow_df.empty:
            flow_cols = ["timestamp", "charging_mw", "discharging_mw",
                         "net_battery_flow_mw", "storage_charge_pressure",
                         "storage_discharge_pressure", "battery_saturation_proxy"]
            available = [c for c in flow_cols if c in battery_flow_df.columns]
            # Normalize to tz-naive UTC before merging to avoid datetime tz mismatch
            df_ts = df[["timestamp"]].copy()
            df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"]).dt.tz_localize(None)
            flow_subset = battery_flow_df[available].copy()
            flow_subset["timestamp"] = pd.to_datetime(flow_subset["timestamp"]).dt.tz_localize(None)
            merged = df_ts.merge(flow_subset, on="timestamp", how="left")
            charging    = merged.get("charging_mw",              pd.Series(0.0, index=df.index)).fillna(0.0)
            discharging = merged.get("discharging_mw",           pd.Series(0.0, index=df.index)).fillna(0.0)
            net_flow    = merged.get("net_battery_flow_mw",      pd.Series(0.0, index=df.index)).fillna(0.0)
            chg_press   = merged.get("storage_charge_pressure",  pd.Series(0.0, index=df.index)).fillna(0.0)
            dis_press   = merged.get("storage_discharge_pressure",pd.Series(0.0, index=df.index)).fillna(0.0)
            saturation  = merged.get("battery_saturation_proxy", pd.Series(0.5, index=df.index)).fillna(0.5)
        else:
            # Inline price-based proxy
            is_midday   = ((hour >= 9) & (hour <= 16)).astype(float)
            is_evening  = ((hour >= 17) & (hour <= 21)).astype(float)
            pb          = (_CHARGE_PRICE_THR - price.clip(upper=_CHARGE_PRICE_THR)) / (_CHARGE_PRICE_THR + 50)
            chg_press   = (0.5 * pb.clip(0, 1) + 0.3 * ((solar - load).clip(lower=0) / max(self.installed_power_mw, 1)).clip(0, 1) + 0.2 * is_midday).clip(0, 1)
            dis_press   = (0.4 * ((price - _DISCHARGE_PRICE_THR).clip(lower=0) / 100).clip(0, 1) + 0.4 * is_evening + 0.2 * ((load - 55_000) / 20_000).clip(0, 1)).clip(0, 1)
            charging    = (chg_press * self.installed_power_mw * 0.7).clip(0, self.installed_power_mw)
            discharging = (dis_press * self.installed_power_mw * 0.7).clip(0, self.installed_power_mw)
            net_flow    = discharging - charging
            soc_delta   = (-net_flow).rolling(24, min_periods=1).sum()
            saturation  = ((soc_delta + self.installed_capacity_mwh * 0.5) / self.installed_capacity_mwh).clip(0, 1.0)

        # ---- Derived features -------------------------------------------
        total_gen             = solar + wind_on + wind_off
        pv_surplus            = (total_gen - load).clip(lower=0)
        roll_mean_24          = price.rolling(24, min_periods=12).mean()
        is_midday_flag        = ((hour >= 9) & (hour <= 16)).astype(float)
        is_evening_flag       = ((hour >= 17) & (hour <= 21)).astype(float)
        midday_rel            = (price / roll_mean_24.replace(0, np.nan)).fillna(1.0)
        midday_compr          = ((1.0 - midday_rel.clip(0, 2) / 2.0) * is_midday_flag).clip(0, 1)
        roll_min_6            = price.rolling(6, min_periods=1).min()
        roll_max_6            = price.rolling(6, min_periods=1).max()
        eve_arb_spread        = ((roll_max_6 - roll_min_6) * is_evening_flag).clip(lower=0)
        exp_absorption        = (chg_press.shift(-1).fillna(chg_press) * self.installed_power_mw * 0.5).clip(0, self.installed_power_mw)
        exp_release           = (dis_press.shift(-1).fillna(dis_press) * self.installed_power_mw * 0.5).clip(0, self.installed_power_mw)
        batt_adj_residual     = residual + charging - discharging

        df["battery_charging_mw"]          = charging.round(1)
        df["battery_discharging_mw"]       = discharging.round(1)
        df["net_battery_flow_mw"]          = net_flow.round(1)
        df["storage_charge_pressure"]      = chg_press.round(4)
        df["storage_discharge_pressure"]   = dis_press.round(4)
        df["battery_saturation_proxy"]     = saturation.round(4)
        df["pv_surplus_after_load"]        = pv_surplus.round(1)
        df["midday_price_compression"]     = midday_compr.round(4)
        df["evening_arbitrage_spread"]     = eve_arb_spread.round(2)
        df["expected_battery_absorption"]  = exp_absorption.round(1)
        df["expected_battery_release"]     = exp_release.round(1)
        df["battery_adjusted_residual_load"] = batt_adj_residual.round(1)
        df["battery_installed_power_mw"]   = float(self.installed_power_mw)
        df["battery_installed_capacity_mwh"] = float(self.installed_capacity_mwh)

        return df
