"""
Futures Cost Model for German Electricity Futures.

Models the full cost stack of a Futures trade:
  - Dynamic spread (volatility-adjusted)
  - Slippage
  - Overnight/weekend financing
  - Broker markup
  - Safety buffer

Net edge = expected_rebound - total_costs
"""

import numpy as np
import logging
from typing import Optional

from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class CostModelConfig(BaseModel):
    """All tunable parameters for the Futures cost model."""

    avg_spread_eur_mwh: float = Field(5.0, ge=0, description="Average half-spread in EUR/MWh")
    min_spread_eur_mwh: float = Field(2.0, ge=0, description="Minimum spread floor in EUR/MWh")
    max_spread_eur_mwh: float = Field(15.0, ge=0, description="Maximum spread cap in EUR/MWh")
    volatility_spread_multiplier: float = Field(
        1.5, ge=1.0, description="Scale factor applied to spread under high volatility"
    )
    slippage_eur_mwh: float = Field(3.0, ge=0, description="Expected slippage per side in EUR/MWh")
    overnight_fee_annual_pct: float = Field(
        8.0, ge=0, description="Annualised overnight financing rate (%)"
    )
    weekend_fee_multiplier: float = Field(
        1.5, ge=1.0, description="Multiplier applied to overnight rate over weekends"
    )
    broker_markup_eur_mwh: float = Field(
        1.0, ge=0, description="Fixed broker markup per MWh"
    )
    safety_buffer_eur_mwh: float = Field(
        5.0, ge=0, description="Conservative safety buffer EUR/MWh"
    )
    minimum_edge_threshold: float = Field(
        30.0, description="Minimum net edge (EUR/MWh) required to flag a trade"
    )


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class CostBreakdown(BaseModel):
    """Full cost decomposition and tradability verdict for a single signal."""

    spread_cost: float = Field(..., description="Half-spread round-trip cost in EUR/MWh")
    slippage_cost: float = Field(..., description="Slippage cost in EUR/MWh")
    financing_cost: float = Field(..., description="Overnight financing cost for holding period")
    broker_markup: float = Field(..., description="Broker markup in EUR/MWh")
    safety_buffer: float = Field(..., description="Safety buffer in EUR/MWh")
    total_cost: float = Field(..., description="Sum of all cost components")
    net_edge: float = Field(..., description="expected_rebound - total_cost")
    is_tradeable: bool = Field(..., description="True if net_edge >= minimum_edge_threshold")
    rejection_reason: Optional[str] = Field(None, description="Human-readable reason if not tradeable")


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

class FuturesCostModel:
    """
    Realistic Futures cost model for German electricity Futures.

    Net edge formula:
        net_edge = expected_rebound
                   - spread_cost
                   - slippage_cost
                   - financing_cost
                   - broker_markup
                   - safety_buffer

    All costs are expressed in EUR/MWh and assume a notional size of 1 MWh.
    Callers must scale by their actual notional when computing EUR P&L.
    """

    def __init__(self, config: Optional[CostModelConfig] = None) -> None:
        self.config = config or CostModelConfig()
        logger.info(
            "FuturesCostModel initialised",
            extra={
                "avg_spread": self.config.avg_spread_eur_mwh,
                "min_edge": self.config.minimum_edge_threshold,
            },
        )

    # ------------------------------------------------------------------
    # Individual cost components
    # ------------------------------------------------------------------

    def calculate_spread_cost(self, price_volatility: Optional[float] = None) -> float:
        """
        Compute the round-trip spread cost in EUR/MWh.

        When ``price_volatility`` is provided (expressed as a percentage of
        the reference price, e.g. 15 = 15 %), the spread is scaled linearly
        between ``avg_spread`` and ``avg_spread * volatility_spread_multiplier``
        and then clipped to [min_spread, max_spread].

        Args:
            price_volatility: Optional volatility expressed as a percentage
                (e.g. 15 = 15 %).  When None, the average spread is returned.

        Returns:
            Spread cost in EUR/MWh.
        """
        base = self.config.avg_spread_eur_mwh

        if price_volatility is not None:
            # Linear scaling: full multiplier kicks in at volatility == 10 %
            multiplier = 1.0 + (price_volatility / 10.0) * (
                self.config.volatility_spread_multiplier - 1.0
            )
            spread = base * multiplier
            clipped = float(
                np.clip(spread, self.config.min_spread_eur_mwh, self.config.max_spread_eur_mwh)
            )
            logger.debug(
                "Dynamic spread calculated",
                extra={
                    "volatility": price_volatility,
                    "raw_spread": spread,
                    "clipped_spread": clipped,
                },
            )
            return clipped

        return base

    def calculate_financing_cost(
        self,
        holding_hours: float,
        is_weekend: bool = False,
        notional_price_eur_mwh: float = 100.0,
    ) -> float:
        """
        Estimate the overnight financing cost for holding an open position.

        The annual rate is converted to an hourly rate and applied over the
        expected holding period.  Weekend rates are multiplied by
        ``weekend_fee_multiplier``.

        Args:
            holding_hours: Expected position holding time in hours.
            is_weekend: Whether the position will be held over a weekend.
            notional_price_eur_mwh: Reference price used to convert the
                percentage rate to an absolute EUR/MWh figure (default 100 €).

        Returns:
            Financing cost in EUR/MWh for the given holding period.
        """
        if holding_hours <= 0:
            return 0.0

        daily_rate = self.config.overnight_fee_annual_pct / 365.0 / 100.0
        if is_weekend:
            daily_rate *= self.config.weekend_fee_multiplier

        hourly_rate = daily_rate / 24.0
        cost = hourly_rate * holding_hours * notional_price_eur_mwh

        logger.debug(
            "Financing cost calculated",
            extra={
                "holding_hours": holding_hours,
                "is_weekend": is_weekend,
                "daily_rate_pct": daily_rate * 100,
                "cost_eur_mwh": cost,
            },
        )
        return cost

    # ------------------------------------------------------------------
    # Full cost + edge calculation
    # ------------------------------------------------------------------

    def calculate_net_edge(
        self,
        expected_rebound_eur_mwh: float,
        estimated_holding_hours: float = 4.0,
        price_volatility: Optional[float] = None,
        is_weekend: bool = False,
        notional_price_eur_mwh: float = 100.0,
    ) -> CostBreakdown:
        """
        Calculate the full cost breakdown and net edge for a proposed trade.

        Args:
            expected_rebound_eur_mwh: Expected price recovery in EUR/MWh
                (already probability-weighted by the caller).
            estimated_holding_hours: Expected position duration in hours.
            price_volatility: Current price volatility (%) for dynamic spread.
            is_weekend: Whether the holding period spans a weekend.
            notional_price_eur_mwh: Reference price for financing cost calc.

        Returns:
            ``CostBreakdown`` with all components and tradability verdict.
        """
        spread_cost = self.calculate_spread_cost(price_volatility)
        slippage_cost = self.config.slippage_eur_mwh
        financing_cost = self.calculate_financing_cost(
            estimated_holding_hours, is_weekend, notional_price_eur_mwh
        )
        broker_markup = self.config.broker_markup_eur_mwh
        safety_buffer = self.config.safety_buffer_eur_mwh

        total_cost = spread_cost + slippage_cost + financing_cost + broker_markup + safety_buffer
        net_edge = expected_rebound_eur_mwh - total_cost

        is_tradeable = net_edge >= self.config.minimum_edge_threshold
        rejection_reason: Optional[str] = None
        if not is_tradeable:
            rejection_reason = (
                f"Net edge {net_edge:.2f} EUR/MWh is below minimum threshold "
                f"{self.config.minimum_edge_threshold:.2f} EUR/MWh "
                f"(total costs: {total_cost:.2f}, expected rebound: {expected_rebound_eur_mwh:.2f})"
            )

        breakdown = CostBreakdown(
            spread_cost=round(spread_cost, 4),
            slippage_cost=round(slippage_cost, 4),
            financing_cost=round(financing_cost, 4),
            broker_markup=round(broker_markup, 4),
            safety_buffer=round(safety_buffer, 4),
            total_cost=round(total_cost, 4),
            net_edge=round(net_edge, 4),
            is_tradeable=is_tradeable,
            rejection_reason=rejection_reason,
        )

        logger.debug(
            "Net edge calculated",
            extra={
                "expected_rebound": expected_rebound_eur_mwh,
                "total_cost": total_cost,
                "net_edge": net_edge,
                "is_tradeable": is_tradeable,
            },
        )
        return breakdown

    # ------------------------------------------------------------------
    # Config management
    # ------------------------------------------------------------------

    def update_config(self, new_config: CostModelConfig) -> None:
        """Replace the current configuration with a new one."""
        old_threshold = self.config.minimum_edge_threshold
        self.config = new_config
        logger.info(
            "CostModelConfig updated",
            extra={
                "old_min_edge": old_threshold,
                "new_min_edge": new_config.minimum_edge_threshold,
            },
        )

    def get_config(self) -> CostModelConfig:
        """Return the current configuration (immutable copy via Pydantic)."""
        return self.config.model_copy()
