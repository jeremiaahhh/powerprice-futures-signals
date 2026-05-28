"""
Tail Risk Engine — composite risk scorer for Futures rebound signals.

Blocks entries when:
  - Price too deep negative (< max_negative_price threshold)
  - Negative streak too long (> max_streak_hours)
  - Extreme intra-hour gap detected (gap_risk_score > 0.80)
  - Composite tail risk score too high (> max_tail_risk_score)

Designed for hourly German electricity price data (EPEX SPOT DE).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TailRiskAssessment:
    tail_risk_score: float               # 0-1 composite
    gap_risk_score: float                # 0-1
    oversupply_stress_index: float       # 0-1
    rebound_failure_probability: float   # 0-1
    negative_price_streak: int           # consecutive negative hours at current
    max_price_gap_1h: float              # max |Δprice| in last 6h
    volatility_24h: float                # 24h price std
    is_blocked: bool
    block_reason: Optional[str]          # "TAIL_RISK_BLOCKED" | "GAP_RISK_BLOCKED" | None
    block_detail: str
    components: Dict[str, float] = field(default_factory=dict)


class TailRiskEngine:
    """Composite tail risk assessor for negative-price rebound signals."""

    def __init__(
        self,
        max_negative_price: float = -150.0,
        max_streak_hours: int = 3,
        max_gap_size: float = 100.0,
        max_tail_risk_score: float = 0.65,
    ):
        self.max_negative_price = max_negative_price
        self.max_streak_hours = max_streak_hours
        self.max_gap_size = max_gap_size
        self.max_tail_risk_score = max_tail_risk_score

    def assess(self, df: pd.DataFrame, current_price: float) -> TailRiskAssessment:
        """
        Assess tail risk for a potential entry at current_price.

        Parameters
        ----------
        df : HourlyPrice DataFrame, sorted ascending by timestamp.
             Expected columns: timestamp, price_eur_mwh, solar_mw,
             wind_onshore_mw, wind_offshore_mw, load_mw, residual_load_mw
        current_price : Current market price (EUR/MWh), expected < 0 for entry.
        """
        if df.empty:
            return TailRiskAssessment(
                tail_risk_score=0.0, gap_risk_score=0.0,
                oversupply_stress_index=0.0, rebound_failure_probability=0.0,
                negative_price_streak=0, max_price_gap_1h=0.0, volatility_24h=0.0,
                is_blocked=False, block_reason=None, block_detail="no data",
            )

        prices = df["price_eur_mwh"].dropna()

        # 1. Negative price streak (count backward from end)
        streak = 0
        for p in reversed(prices.tolist()):
            if p < 0:
                streak += 1
            else:
                break

        # 2. Gap risk: max |price[t] - price[t-1]| in last 6 bars
        recent_6 = prices.tail(7)
        gaps = recent_6.diff().abs().dropna()
        max_gap = float(gaps.max()) if len(gaps) > 0 else 0.0
        gap_risk_score = min(max_gap / 200.0, 1.0)

        # 3. Oversupply stress index (from last row)
        try:
            last = df.iloc[-1]
            wind = (float(last.get("wind_onshore_mw") or 0) +
                    float(last.get("wind_offshore_mw") or 0))
            solar = float(last.get("solar_mw") or 0)
            load = float(last.get("load_mw") or 55_000)
            oversupply = max((wind + solar - load), 0.0)
            oversupply_stress = min(oversupply / max(load, 1.0), 1.0)
        except Exception:
            oversupply_stress = 0.0

        # 4. Rebound failure probability (streak + depth)
        streak_factor = min(streak / 6.0, 1.0)
        depth_factor = min(abs(current_price) / 300.0, 1.0)
        rebound_failure = min(0.5 * streak_factor + 0.5 * depth_factor, 1.0)

        # 5. Composite tail risk score
        tail_risk_score = min(
            gap_risk_score * 0.35
            + oversupply_stress * 0.25
            + rebound_failure * 0.40,
            1.0,
        )

        # 6. 24h volatility
        prices_24h = prices.tail(24)
        vol_24h = float(prices_24h.std()) if len(prices_24h) >= 2 else 0.0

        components = {
            "gap_risk_score": round(gap_risk_score, 4),
            "oversupply_stress_index": round(oversupply_stress, 4),
            "rebound_failure_probability": round(rebound_failure, 4),
            "streak_factor": round(streak_factor, 4),
            "depth_factor": round(depth_factor, 4),
        }

        # 7. Block decision (ordered by severity)
        is_blocked = False
        block_reason: Optional[str] = None
        block_detail = "ok"

        if current_price < self.max_negative_price:
            is_blocked = True
            block_reason = "GAP_RISK_BLOCKED"
            block_detail = (
                f"Price {current_price:.0f} EUR/MWh below floor "
                f"{self.max_negative_price:.0f} EUR/MWh — extreme tail risk"
            )
        elif streak > self.max_streak_hours:
            is_blocked = True
            block_reason = "TAIL_RISK_BLOCKED"
            block_detail = (
                f"Negative streak {streak}h exceeds limit {self.max_streak_hours}h "
                f"— sustained oversupply, rebound less certain"
            )
        elif gap_risk_score > 0.80:
            is_blocked = True
            block_reason = "GAP_RISK_BLOCKED"
            block_detail = (
                f"Extreme price gap {max_gap:.0f} EUR/MWh detected "
                f"(gap_risk_score={gap_risk_score:.2f}) — gap risk too high"
            )
        elif tail_risk_score > self.max_tail_risk_score:
            is_blocked = True
            block_reason = "TAIL_RISK_BLOCKED"
            block_detail = (
                f"Composite tail_risk_score={tail_risk_score:.2f} "
                f"exceeds limit {self.max_tail_risk_score:.2f}"
            )

        if not is_blocked:
            block_detail = "ok"

        return TailRiskAssessment(
            tail_risk_score=round(tail_risk_score, 4),
            gap_risk_score=round(gap_risk_score, 4),
            oversupply_stress_index=round(oversupply_stress, 4),
            rebound_failure_probability=round(rebound_failure, 4),
            negative_price_streak=streak,
            max_price_gap_1h=round(max_gap, 2),
            volatility_24h=round(vol_24h, 2),
            is_blocked=is_blocked,
            block_reason=block_reason,
            block_detail=block_detail,
            components=components,
        )
