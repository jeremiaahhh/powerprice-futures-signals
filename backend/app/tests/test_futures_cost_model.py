"""
Tests for the Futures cost model.

Tests cover:
  - Default configuration values
  - Net edge is positive with a good rebound
  - Net edge is negative / insufficient with a small rebound
  - High volatility increases the effective spread
  - Weekend multiplier applies to overnight fee
  - Config update persists
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone

import pytest

from app.api.schemas import CostBreakdown, CostModelConfig


# ---------------------------------------------------------------------------
# Pure cost calculation functions (mirrors routes/futures.py logic)
# ---------------------------------------------------------------------------


def compute_cost_breakdown(
    config: CostModelConfig,
    is_weekend: bool = False,
    volatility_eur_mwh: float = 0.0,
    avg_spread_base: float | None = None,
) -> CostBreakdown:
    """
    Compute the full itemised Futures cost breakdown.

    Parameters
    ----------
    config:
        Active cost model configuration.
    is_weekend:
        Whether the weekend overnight fee multiplier applies.
    volatility_eur_mwh:
        Current 24-hour price volatility. When > avg_spread_base the spread
        is scaled by futures_volatility_spread_multiplier.
    avg_spread_base:
        Reference spread level. Defaults to config.avg_spread_eur_mwh.
    """
    from app.core.config import settings

    # Spread – may be widened under high volatility
    if avg_spread_base is None:
        avg_spread_base = config.avg_spread_eur_mwh

    vol_multiplier = getattr(settings, "futures_volatility_spread_multiplier", 1.5)
    if volatility_eur_mwh > avg_spread_base:
        spread_cost = min(
            avg_spread_base * vol_multiplier,
            getattr(settings, "futures_max_spread_eur_mwh", 15.0),
        )
    else:
        spread_cost = avg_spread_base

    # Slippage
    slippage_cost = config.slippage_eur_mwh

    # Overnight fee
    daily_fee_pct = config.overnight_fee_annual_pct / 365.0
    hourly_fee_pct = daily_fee_pct / 24.0
    base_overnight = hourly_fee_pct * config.holding_hours * 100.0

    if is_weekend:
        overnight_fee = base_overnight * config.weekend_fee_multiplier
    else:
        overnight_fee = base_overnight

    overnight_fee = min(overnight_fee, 5.0)

    broker_markup = config.broker_markup_eur_mwh
    safety_buffer = config.safety_buffer_eur_mwh

    total = spread_cost + slippage_cost + overnight_fee + broker_markup + safety_buffer

    return CostBreakdown(
        spread_cost_eur_mwh=round(spread_cost, 4),
        slippage_cost_eur_mwh=round(slippage_cost, 4),
        overnight_fee_eur_mwh=round(overnight_fee, 4),
        broker_markup_eur_mwh=round(broker_markup, 4),
        safety_buffer_eur_mwh=round(safety_buffer, 4),
        total_eur_mwh=round(total, 4),
        holding_hours_assumed=config.holding_hours,
        is_weekend=is_weekend,
    )


def compute_net_edge(
    entry_price: float,
    predicted_exit_price: float,
    config: CostModelConfig,
    is_weekend: bool = False,
) -> tuple[float, float, float]:
    """Return (gross_edge, futures_costs, net_edge)."""
    gross_edge = max(0.0, predicted_exit_price - entry_price)
    breakdown = compute_cost_breakdown(config, is_weekend=is_weekend)
    futures_costs = breakdown.total_eur_mwh
    net_edge = gross_edge - futures_costs
    return round(gross_edge, 4), round(futures_costs, 4), round(net_edge, 4)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    """Verify that the default CostModelConfig matches expected values."""

    def test_default_avg_spread(self):
        config = CostModelConfig()
        assert config.avg_spread_eur_mwh == 5.0

    def test_default_slippage(self):
        config = CostModelConfig()
        assert config.slippage_eur_mwh == 3.0

    def test_default_overnight_fee(self):
        config = CostModelConfig()
        assert config.overnight_fee_annual_pct == 8.0

    def test_default_weekend_multiplier(self):
        config = CostModelConfig()
        assert config.weekend_fee_multiplier == 1.5

    def test_default_broker_markup(self):
        config = CostModelConfig()
        assert config.broker_markup_eur_mwh == 1.0

    def test_default_safety_buffer(self):
        config = CostModelConfig()
        assert config.safety_buffer_eur_mwh == 5.0

    def test_default_min_edge_threshold(self):
        config = CostModelConfig()
        assert config.min_edge_threshold == 30.0  # raised from 10→30 after OOS analysis

    def test_default_holding_hours(self):
        config = CostModelConfig()
        assert config.holding_hours == 4

    def test_total_cost_components_sum(self):
        """The sum of fixed components (excluding overnight) matches expectations."""
        config = CostModelConfig()
        fixed = (
            config.avg_spread_eur_mwh
            + config.slippage_eur_mwh
            + config.broker_markup_eur_mwh
            + config.safety_buffer_eur_mwh
        )
        assert fixed == pytest.approx(14.0, abs=1e-6)


class TestNetEdgePositive:
    """With a good rebound, net edge should exceed the threshold."""

    def test_strong_rebound_net_edge_positive(self):
        config = CostModelConfig()
        gross, costs, net = compute_net_edge(
            entry_price=-10.0,
            predicted_exit_price=40.0,
            config=config,
        )
        assert gross == pytest.approx(50.0, abs=1e-3)
        assert net > config.min_edge_threshold
        assert net > 0.0

    def test_net_edge_exceeds_minimum_threshold(self):
        config = CostModelConfig(min_edge_threshold=10.0)
        _, _, net = compute_net_edge(
            entry_price=-15.0,
            predicted_exit_price=35.0,
            config=config,
        )
        # gross_edge = 50, total_cost ≈ 14.1, net ≈ 35.9
        assert net >= config.min_edge_threshold

    def test_very_deep_negative_large_edge(self):
        config = CostModelConfig()
        gross, costs, net = compute_net_edge(
            entry_price=-50.0,
            predicted_exit_price=30.0,
            config=config,
        )
        assert gross == pytest.approx(80.0, abs=1e-3)
        assert net > 60.0

    def test_tradeable_flag(self):
        config = CostModelConfig(min_edge_threshold=10.0)
        _, _, net = compute_net_edge(
            entry_price=-10.0,
            predicted_exit_price=30.0,
            config=config,
        )
        is_tradeable = net >= config.min_edge_threshold
        assert is_tradeable is True


class TestNetEdgeNegative:
    """Insufficient rebound should yield a non-tradeable signal."""

    def test_small_rebound_negative_edge(self):
        config = CostModelConfig()
        gross, costs, net = compute_net_edge(
            entry_price=-5.0,
            predicted_exit_price=5.0,
            config=config,
        )
        # gross_edge = 10, total_cost ≈ 14.1, net ≈ -4.1
        assert net < config.min_edge_threshold

    def test_zero_rebound_negative_edge(self):
        config = CostModelConfig()
        gross, costs, net = compute_net_edge(
            entry_price=-5.0,
            predicted_exit_price=-5.0,  # no price movement
            config=config,
        )
        assert gross == pytest.approx(0.0, abs=1e-6)
        assert net < 0.0

    def test_very_tight_edge_not_tradeable(self):
        config = CostModelConfig(min_edge_threshold=15.0)
        _, _, net = compute_net_edge(
            entry_price=-2.0,
            predicted_exit_price=10.0,
            config=config,
        )
        # gross=12, costs≈14, net≈-2 → not tradeable
        is_tradeable = net >= config.min_edge_threshold
        assert is_tradeable is False

    def test_gross_edge_is_always_non_negative(self):
        """Gross edge is max(0, predicted - current); never negative."""
        config = CostModelConfig()
        gross, _, _ = compute_net_edge(
            entry_price=10.0,
            predicted_exit_price=5.0,  # price goes lower
            config=config,
        )
        assert gross >= 0.0


class TestSpreadVolatilityAdjustment:
    """High volatility should widen the effective spread."""

    def test_low_volatility_uses_base_spread(self):
        config = CostModelConfig(avg_spread_eur_mwh=5.0)
        breakdown = compute_cost_breakdown(
            config,
            volatility_eur_mwh=2.0,  # below avg_spread of 5.0
        )
        assert breakdown.spread_cost_eur_mwh == pytest.approx(5.0, abs=1e-4)

    def test_high_volatility_widens_spread(self):
        config = CostModelConfig(avg_spread_eur_mwh=5.0)
        breakdown_low = compute_cost_breakdown(
            config, volatility_eur_mwh=2.0
        )
        breakdown_high = compute_cost_breakdown(
            config, volatility_eur_mwh=10.0  # above avg_spread of 5.0
        )
        assert breakdown_high.spread_cost_eur_mwh > breakdown_low.spread_cost_eur_mwh

    def test_volatility_spread_capped_at_max(self):
        """Spread should not exceed the configured maximum."""
        from app.core.config import settings

        config = CostModelConfig(avg_spread_eur_mwh=5.0)
        breakdown = compute_cost_breakdown(
            config, volatility_eur_mwh=1000.0  # extreme volatility
        )
        assert breakdown.spread_cost_eur_mwh <= settings.futures_max_spread_eur_mwh

    def test_volatility_increases_total_cost(self):
        config = CostModelConfig(avg_spread_eur_mwh=5.0)
        low_vol = compute_cost_breakdown(config, volatility_eur_mwh=1.0)
        high_vol = compute_cost_breakdown(config, volatility_eur_mwh=20.0)
        assert high_vol.total_eur_mwh > low_vol.total_eur_mwh


class TestFinancingCostWeekend:
    """Weekend multiplier should inflate the overnight fee."""

    def test_weekend_overnight_higher_than_weekday(self):
        config = CostModelConfig(overnight_fee_annual_pct=8.0, weekend_fee_multiplier=1.5)
        weekday = compute_cost_breakdown(config, is_weekend=False)
        weekend = compute_cost_breakdown(config, is_weekend=True)
        assert weekend.overnight_fee_eur_mwh >= weekday.overnight_fee_eur_mwh

    def test_weekend_total_cost_higher(self):
        config = CostModelConfig()
        weekday = compute_cost_breakdown(config, is_weekend=False)
        weekend = compute_cost_breakdown(config, is_weekend=True)
        assert weekend.total_eur_mwh >= weekday.total_eur_mwh

    def test_weekend_flag_set_correctly(self):
        config = CostModelConfig()
        breakdown = compute_cost_breakdown(config, is_weekend=True)
        assert breakdown.is_weekend is True

    def test_weekday_flag_set_correctly(self):
        config = CostModelConfig()
        breakdown = compute_cost_breakdown(config, is_weekend=False)
        assert breakdown.is_weekend is False

    def test_weekend_multiplier_one_same_as_weekday(self):
        """With multiplier=1.0 the weekend overnight should equal the weekday overnight."""
        config = CostModelConfig(overnight_fee_annual_pct=8.0, weekend_fee_multiplier=1.0)
        weekday = compute_cost_breakdown(config, is_weekend=False)
        weekend = compute_cost_breakdown(config, is_weekend=True)
        assert weekend.overnight_fee_eur_mwh == pytest.approx(
            weekday.overnight_fee_eur_mwh, abs=1e-6
        )


class TestUpdateConfig:
    """Config update should change all relevant values."""

    def test_update_changes_spread(self):
        original = CostModelConfig(avg_spread_eur_mwh=5.0)
        updated = original.model_copy(update={"avg_spread_eur_mwh": 8.0})
        assert updated.avg_spread_eur_mwh == 8.0
        assert original.avg_spread_eur_mwh == 5.0  # original unchanged

    def test_update_changes_min_edge_threshold(self):
        config = CostModelConfig(min_edge_threshold=10.0)
        updated = config.model_copy(update={"min_edge_threshold": 20.0})
        assert updated.min_edge_threshold == 20.0

    def test_update_affects_cost_breakdown(self):
        original = CostModelConfig(avg_spread_eur_mwh=5.0)
        updated = CostModelConfig(avg_spread_eur_mwh=10.0)
        breakdown_original = compute_cost_breakdown(original)
        breakdown_updated = compute_cost_breakdown(updated)
        assert breakdown_updated.spread_cost_eur_mwh > breakdown_original.spread_cost_eur_mwh
        assert breakdown_updated.total_eur_mwh > breakdown_original.total_eur_mwh

    def test_update_persists_to_file(self):
        """Verify that config can be serialised to JSON and reloaded."""
        config = CostModelConfig(
            avg_spread_eur_mwh=7.0,
            slippage_eur_mwh=4.0,
            min_edge_threshold=15.0,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            path = fh.name
            json.dump(config.model_dump(), fh)

        try:
            with open(path) as fh:
                loaded_data = json.load(fh)
            loaded_config = CostModelConfig(**loaded_data)

            assert loaded_config.avg_spread_eur_mwh == 7.0
            assert loaded_config.slippage_eur_mwh == 4.0
            assert loaded_config.min_edge_threshold == 15.0
        finally:
            os.unlink(path)

    def test_validation_rejects_negative_spread(self):
        """Pydantic validation should reject a negative spread."""
        with pytest.raises(Exception):
            CostModelConfig(avg_spread_eur_mwh=-1.0)

    def test_validation_rejects_weekend_multiplier_below_one(self):
        """Weekend multiplier must be >= 1.0."""
        with pytest.raises(Exception):
            CostModelConfig(weekend_fee_multiplier=0.5)
