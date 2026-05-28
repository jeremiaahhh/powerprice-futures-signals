"""
Market regime classifier for German electricity Futures.

Uses rule-based logic on engineered features to classify the current market
into one of six regimes. Thresholds are informed by 2024–2026 historical data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class RegimeType(str, Enum):
    NORMAL_SUMMER     = "NORMAL_SUMMER"       # balanced renewable mix, moderate prices
    SOLAR_OVERSUPPLY  = "SOLAR_OVERSUPPLY"    # midday PV flood → deep negative hours
    WIND_OVERSUPPLY   = "WIND_OVERSUPPLY"     # sustained high wind → negative overnight
    VOLATILE          = "VOLATILE"            # high price swings, unreliable signals
    WINTER_LOW        = "WINTER_LOW"          # low renewables, few negative opportunities
    STRESS            = "STRESS"              # extreme volatility / market disruption
    STORAGE_SATURATED          = "STORAGE_SATURATED"
    HIGH_STORAGE_ABSORPTION    = "HIGH_STORAGE_ABSORPTION"
    EVENING_DISCHARGE_PRESSURE = "EVENING_DISCHARGE_PRESSURE"
    BATTERY_DAMPENED_REBOUND   = "BATTERY_DAMPENED_REBOUND"
    LOW_STORAGE_IMPACT         = "LOW_STORAGE_IMPACT"


@dataclass
class RegimeResult:
    regime: RegimeType
    confidence: float                    # 0–1
    renewable_share: Optional[float]
    price_volatility_24h: Optional[float]
    hours_negative_24h: Optional[float]
    solar_mw: Optional[float]
    wind_mw: Optional[float]
    oversupply_index: Optional[float]    # (solar + wind - load) if positive → excess
    signal_thresholds: dict = field(default_factory=dict)
    description: str = ""


class RegimeClassifier:
    """
    Classify the current market regime from the latest feature row.

    Inputs come from the FeatureEngineer feature matrix (the last valid row).
    """

    # Regime-specific net_edge overrides (None = use global default)
    _REGIME_THRESHOLDS: dict[RegimeType, dict] = {
        RegimeType.SOLAR_OVERSUPPLY: {"net_edge_enter": 30.0, "net_edge_hc": 35.0},
        RegimeType.WIND_OVERSUPPLY:  {"net_edge_enter": 30.0, "net_edge_hc": 35.0},
        RegimeType.NORMAL_SUMMER:    {"net_edge_enter": 30.0, "net_edge_hc": 35.0},
        RegimeType.VOLATILE:         {"net_edge_enter": 40.0, "net_edge_hc": 50.0},  # tighter in volatile
        RegimeType.WINTER_LOW:       {"net_edge_enter": 25.0, "net_edge_hc": 30.0},  # rare events, accept lower
        RegimeType.STRESS:           {"net_edge_enter": 999.0, "net_edge_hc": 999.0},  # block all entries
        RegimeType.STORAGE_SATURATED:           {"net_edge_enter": 35.0, "net_edge_hc": 45.0},
        RegimeType.HIGH_STORAGE_ABSORPTION:     {"net_edge_enter": 32.0, "net_edge_hc": 40.0},
        RegimeType.EVENING_DISCHARGE_PRESSURE:  {"net_edge_enter": 28.0, "net_edge_hc": 35.0},
        RegimeType.BATTERY_DAMPENED_REBOUND:    {"net_edge_enter": 35.0, "net_edge_hc": 45.0},
        RegimeType.LOW_STORAGE_IMPACT:          {"net_edge_enter": 30.0, "net_edge_hc": 35.0},
    }

    def classify(self, features: pd.Series) -> RegimeResult:
        """Classify regime from the latest feature row."""

        def _f(key: str, default: float = 0.0) -> float:
            v = features.get(key, default)
            return float(v) if v is not None and np.isfinite(float(v if v is not None else 0)) else default

        renewable_share      = _f("renewable_share")
        price_vol            = _f("price_volatility_24h")
        hours_neg            = _f("hours_in_negative_last_24h")
        solar_mw             = _f("solar_mw")
        wind_total           = _f("wind_total_mw", _f("wind_x_renewable_share", 0.0))
        load_mw              = _f("load_mw", 50_000.0)
        hour                 = int(_f("hour", 12))

        oversupply = (solar_mw + wind_total) - load_mw if load_mw > 0 else 0.0

        # Battery state (from proxy features in FeatureEngineer)
        battery_near_full = _f("battery_near_full") > 0.5
        battery_soc_proxy = _f("battery_soc_proxy", 0.5)

        # Battery regime detection (uses BatteryFeatureBuilder features when available)
        charge_pressure   = _f("storage_charge_pressure",    0.0)
        discharge_pressure = _f("storage_discharge_pressure", 0.0)
        battery_saturation = _f("battery_saturation_proxy",  0.5)
        exp_absorption     = _f("expected_battery_absorption", 0.0)

        # STORAGE_SATURATED: batteries near full → rebound timing unreliable
        if battery_saturation > 0.85 and charge_pressure > 0.50:
            return RegimeResult(
                regime=RegimeType.STORAGE_SATURATED,
                confidence=0.80,
                renewable_share=renewable_share,
                price_volatility_24h=price_vol,
                hours_negative_24h=hours_neg,
                solar_mw=solar_mw,
                wind_mw=wind_total,
                oversupply_index=oversupply,
                signal_thresholds=self._REGIME_THRESHOLDS[RegimeType.STORAGE_SATURATED],
                description=(
                    "Battery storage near capacity limit — rebound may be delayed or weaker. "
                    f"SoC proxy={battery_saturation:.0%}, charge_pressure={charge_pressure:.0%}."
                ),
            )

        # HIGH_STORAGE_ABSORPTION: batteries actively charging with capacity to spare
        if charge_pressure > 0.70 and battery_saturation < 0.80:
            return RegimeResult(
                regime=RegimeType.HIGH_STORAGE_ABSORPTION,
                confidence=0.75,
                renewable_share=renewable_share,
                price_volatility_24h=price_vol,
                hours_negative_24h=hours_neg,
                solar_mw=solar_mw,
                wind_mw=wind_total,
                oversupply_index=oversupply,
                signal_thresholds=self._REGIME_THRESHOLDS[RegimeType.HIGH_STORAGE_ABSORPTION],
                description=(
                    "Batteries actively absorbing surplus — supply overhang being managed. "
                    f"charge_pressure={charge_pressure:.0%}, SoC={battery_saturation:.0%}."
                ),
            )

        # EVENING_DISCHARGE_PRESSURE: batteries releasing energy → price spike risk
        if discharge_pressure > 0.60 and hour >= 17:
            return RegimeResult(
                regime=RegimeType.EVENING_DISCHARGE_PRESSURE,
                confidence=0.75,
                renewable_share=renewable_share,
                price_volatility_24h=price_vol,
                hours_negative_24h=hours_neg,
                solar_mw=solar_mw,
                wind_mw=wind_total,
                oversupply_index=oversupply,
                signal_thresholds=self._REGIME_THRESHOLDS[RegimeType.EVENING_DISCHARGE_PRESSURE],
                description=(
                    "Evening battery discharge pressure high — batteries competing with demand rebound. "
                    f"discharge_pressure={discharge_pressure:.0%}."
                ),
            )

        # BATTERY_DAMPENED_REBOUND: high expected absorption next hour → weak rebound
        if battery_saturation > 0.70 and exp_absorption > 6_000:
            return RegimeResult(
                regime=RegimeType.BATTERY_DAMPENED_REBOUND,
                confidence=0.70,
                renewable_share=renewable_share,
                price_volatility_24h=price_vol,
                hours_negative_24h=hours_neg,
                solar_mw=solar_mw,
                wind_mw=wind_total,
                oversupply_index=oversupply,
                signal_thresholds=self._REGIME_THRESHOLDS[RegimeType.BATTERY_DAMPENED_REBOUND],
                description=(
                    "Battery absorption expected to dampen rebound strength. "
                    f"SoC={battery_saturation:.0%}, expected_absorption={exp_absorption:.0f} MW."
                ),
            )

        # --- Decision tree -------------------------------------------

        # STRESS: extreme volatility (market disruption, crisis events)
        if price_vol > 150.0:
            return RegimeResult(
                regime=RegimeType.STRESS,
                confidence=0.95,
                renewable_share=renewable_share,
                price_volatility_24h=price_vol,
                hours_negative_24h=hours_neg,
                solar_mw=solar_mw,
                wind_mw=wind_total,
                oversupply_index=oversupply,
                signal_thresholds=self._REGIME_THRESHOLDS[RegimeType.STRESS],
                description="Extreme price volatility — all entries blocked",
            )

        # VOLATILE: elevated volatility, wider spreads
        if price_vol > 80.0:
            return RegimeResult(
                regime=RegimeType.VOLATILE,
                confidence=0.85,
                renewable_share=renewable_share,
                price_volatility_24h=price_vol,
                hours_negative_24h=hours_neg,
                solar_mw=solar_mw,
                wind_mw=wind_total,
                oversupply_index=oversupply,
                signal_thresholds=self._REGIME_THRESHOLDS[RegimeType.VOLATILE],
                description="Elevated volatility — higher edge threshold required",
            )

        # WINTER_LOW: low renewable penetration, few negative hours
        if renewable_share < 0.25 and hours_neg < 2:
            return RegimeResult(
                regime=RegimeType.WINTER_LOW,
                confidence=0.80,
                renewable_share=renewable_share,
                price_volatility_24h=price_vol,
                hours_negative_24h=hours_neg,
                solar_mw=solar_mw,
                wind_mw=wind_total,
                oversupply_index=oversupply,
                signal_thresholds=self._REGIME_THRESHOLDS[RegimeType.WINTER_LOW],
                description="Winter low-renewable regime — rare events accepted at lower edge",
            )

        # SOLAR_OVERSUPPLY: high solar (midday), or many negative hours PV-driven
        solar_driven = (solar_mw > 25_000 and 9 <= hour <= 16) or \
                       (hours_neg >= 4 and solar_mw > 15_000 and hour < 18)
        if solar_driven:
            batt_note = ""
            confidence = 0.88
            if battery_near_full:
                batt_note = " Batteries near full — extended negative prices likely."
                confidence = min(0.95, confidence + 0.05)
            return RegimeResult(
                regime=RegimeType.SOLAR_OVERSUPPLY,
                confidence=confidence,
                renewable_share=renewable_share,
                price_volatility_24h=price_vol,
                hours_negative_24h=hours_neg,
                solar_mw=solar_mw,
                wind_mw=wind_total,
                oversupply_index=oversupply,
                signal_thresholds=self._REGIME_THRESHOLDS[RegimeType.SOLAR_OVERSUPPLY],
                description=f"Solar oversupply regime — midday PV flood driving negative prices.{batt_note}",
            )

        # WIND_OVERSUPPLY: sustained high wind at any hour
        if wind_total > 35_000 or (wind_total > 20_000 and hours_neg >= 3):
            batt_note = ""
            if battery_near_full:
                batt_note = " Batteries near full — grid cannot absorb more surplus."
            return RegimeResult(
                regime=RegimeType.WIND_OVERSUPPLY,
                confidence=0.82,
                renewable_share=renewable_share,
                price_volatility_24h=price_vol,
                hours_negative_24h=hours_neg,
                solar_mw=solar_mw,
                wind_mw=wind_total,
                oversupply_index=oversupply,
                signal_thresholds=self._REGIME_THRESHOLDS[RegimeType.WIND_OVERSUPPLY],
                description=f"Wind oversupply regime — high wind capacity driving negative prices.{batt_note}",
            )

        return RegimeResult(
            regime=RegimeType.NORMAL_SUMMER,
            confidence=0.70,
            renewable_share=renewable_share,
            price_volatility_24h=price_vol,
            hours_negative_24h=hours_neg,
            solar_mw=solar_mw,
            wind_mw=wind_total,
            oversupply_index=oversupply,
            signal_thresholds=self._REGIME_THRESHOLDS[RegimeType.NORMAL_SUMMER],
            description="Normal summer regime — balanced renewable mix",
        )
