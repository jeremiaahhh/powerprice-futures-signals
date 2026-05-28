"""
Risk Management Engine for Futures Signal Generation.

All risk checks are defensive: any single blocking check prevents a trade
signal from being issued.  Non-blocking issues surface as warnings that are
attached to the signal for operator review.

Checks performed:
  1. Spread filter        – spread must be below max_spread_filter
  2. Volatility filter    – volatility must be below max_volatility_pct
  3. Data freshness       – data must be younger than max_data_age_minutes
  4. Missing data         – critical fields must be present
  5. Confidence threshold – p_rebound >= min_confidence
  6. Position limits      – open positions < max_open_positions
  7. Daily loss limit     – daily_pnl must not breach max_daily_loss
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

@dataclass
class DataQualityResult:
    """Passed into the risk engine by the signal pipeline."""

    is_fresh: bool
    age_minutes: float
    missing_fields: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """Outcome of a full risk evaluation."""

    is_allowed: bool
    warnings: List[str] = field(default_factory=list)
    blocking_reasons: List[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0–1, higher = riskier


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class RiskConfig(BaseModel):
    """Tunable parameters for the risk engine."""

    max_spread_filter_eur_mwh: float = Field(
        12.0, ge=0, description="Reject signals when spread exceeds this value"
    )
    max_volatility_pct: float = Field(
        40.0, ge=0, description="Reject signals when 1h price volatility exceeds this %"
    )
    max_data_age_minutes: int = Field(
        90, ge=1, description="Maximum allowable data staleness in minutes"
    )
    min_confidence_threshold: float = Field(
        0.60, ge=0.0, le=1.0, description="Minimum p_rebound to proceed"
    )
    min_p_negative_threshold: float = Field(
        0.50, ge=0.0, le=1.0, description="Minimum p_negative (price is negative)"
    )
    max_open_positions: int = Field(
        3, ge=1, description="Maximum simultaneous open paper positions"
    )
    max_daily_loss_eur: float = Field(
        500.0, ge=0, description="Maximum cumulative intraday loss before blocking new signals"
    )
    warn_volatility_pct: float = Field(
        25.0, ge=0, description="Volatility level that triggers a warning (not a block)"
    )
    warn_spread_eur_mwh: float = Field(
        8.0, ge=0, description="Spread level that triggers a warning (not a block)"
    )


# ---------------------------------------------------------------------------
# Risk engine
# ---------------------------------------------------------------------------

class RiskEngine:
    """
    Risk management for Futures signal generation.

    Each ``_check_*`` method returns a ``(is_blocking, Optional[message])``
    tuple.  If ``is_blocking`` is True the message is added to
    ``blocking_reasons``; otherwise it is a warning.

    The overall ``risk_score`` (0–1) is a weighted average of individual
    sub-scores derived from how close each metric is to its limit.
    """

    # Weights for the composite risk score (must sum to 1.0)
    _SCORE_WEIGHTS = {
        "spread": 0.20,
        "volatility": 0.20,
        "freshness": 0.15,
        "missing_data": 0.10,
        "confidence": 0.20,
        "positions": 0.10,
        "daily_loss": 0.05,
    }

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.config = config or RiskConfig()
        logger.info(
            "RiskEngine initialised",
            extra={
                "max_spread": self.config.max_spread_filter_eur_mwh,
                "max_positions": self.config.max_open_positions,
                "max_daily_loss": self.config.max_daily_loss_eur,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_signal(
        self,
        features: pd.Series,
        cost_breakdown: "CostBreakdown",  # noqa: F821 – imported at call-site
        p_negative: float,
        p_rebound: float,
        data_quality: DataQualityResult,
        open_positions_count: int = 0,
        daily_pnl_eur: float = 0.0,
    ) -> RiskAssessment:
        """
        Run all risk checks and return a consolidated assessment.

        Args:
            features: Current feature vector (pd.Series from feature pipeline).
            cost_breakdown: Output of ``FuturesCostModel.calculate_net_edge``.
            p_negative: Model probability that current price is negative.
            p_rebound: Model probability of a price rebound.
            data_quality: Freshness / completeness result from data pipeline.
            open_positions_count: Number of currently open paper positions.
            daily_pnl_eur: Cumulative daily P&L in EUR (negative = loss).

        Returns:
            ``RiskAssessment`` with verdicts, warnings, and a risk score.
        """
        blocking_reasons: List[str] = []
        warnings: List[str] = []
        sub_scores: dict[str, float] = {}

        # 1. Spread check
        spread = cost_breakdown.spread_cost
        blocked, msg = self._check_spread(spread)
        if blocked and msg:
            blocking_reasons.append(msg)
        elif msg:
            warnings.append(msg)
        sub_scores["spread"] = self._score_ratio(spread, self.config.max_spread_filter_eur_mwh)

        # 2. Volatility check
        volatility = self._extract_volatility(features)
        blocked, msg = self._check_volatility(volatility)
        if blocked and msg:
            blocking_reasons.append(msg)
        elif msg:
            warnings.append(msg)
        sub_scores["volatility"] = self._score_ratio(volatility, self.config.max_volatility_pct)

        # 3. Data freshness check
        blocked, msg = self._check_data_freshness(data_quality)
        if blocked and msg:
            blocking_reasons.append(msg)
        elif msg:
            warnings.append(msg)
        sub_scores["freshness"] = self._score_ratio(
            data_quality.age_minutes, self.config.max_data_age_minutes
        )

        # 4. Missing data check
        blocked, msg = self._check_missing_data(data_quality)
        if blocked and msg:
            blocking_reasons.append(msg)
        elif msg:
            warnings.append(msg)
        sub_scores["missing_data"] = min(1.0, len(data_quality.missing_fields) / 3.0)

        # 5. Confidence check
        blocked, msg = self._check_confidence(p_rebound, p_negative)
        if blocked and msg:
            blocking_reasons.append(msg)
        elif msg:
            warnings.append(msg)
        # Invert: higher p_rebound = lower risk score contribution
        sub_scores["confidence"] = max(0.0, 1.0 - p_rebound)

        # 6. Position limit check
        blocked, msg = self._check_position_limits(open_positions_count)
        if blocked and msg:
            blocking_reasons.append(msg)
        elif msg:
            warnings.append(msg)
        sub_scores["positions"] = self._score_ratio(
            open_positions_count, self.config.max_open_positions
        )

        # 7. Daily loss check
        blocked, msg = self._check_daily_loss(daily_pnl_eur)
        if blocked and msg:
            blocking_reasons.append(msg)
        elif msg:
            warnings.append(msg)
        # daily_pnl_eur is negative for a loss
        loss = max(0.0, -daily_pnl_eur)
        sub_scores["daily_loss"] = self._score_ratio(loss, self.config.max_daily_loss_eur)

        # Composite risk score
        risk_score = float(
            sum(
                sub_scores.get(k, 0.0) * w
                for k, w in self._SCORE_WEIGHTS.items()
            )
        )
        risk_score = round(min(1.0, max(0.0, risk_score)), 4)

        is_allowed = len(blocking_reasons) == 0

        assessment = RiskAssessment(
            is_allowed=is_allowed,
            warnings=warnings,
            blocking_reasons=blocking_reasons,
            risk_score=risk_score,
        )

        logger.info(
            "Risk assessment complete",
            extra={
                "is_allowed": is_allowed,
                "risk_score": risk_score,
                "blocking_count": len(blocking_reasons),
                "warning_count": len(warnings),
                "blocking_reasons": blocking_reasons,
            },
        )
        return assessment

    def update_config(self, new_config: RiskConfig) -> None:
        """Hot-swap the risk configuration."""
        self.config = new_config
        logger.info("RiskEngine config updated", extra={"new_config": new_config.model_dump()})

    def get_config(self) -> RiskConfig:
        """Return current configuration."""
        return self.config.model_copy()

    # ------------------------------------------------------------------
    # Individual checks (returns blocking_flag, optional_message)
    # ------------------------------------------------------------------

    def _check_spread(self, spread: float) -> Tuple[bool, Optional[str]]:
        """Block if spread exceeds max; warn if above soft threshold."""
        if spread > self.config.max_spread_filter_eur_mwh:
            return True, (
                f"Spread {spread:.2f} EUR/MWh exceeds maximum filter "
                f"{self.config.max_spread_filter_eur_mwh:.2f} EUR/MWh"
            )
        if spread > self.config.warn_spread_eur_mwh:
            return False, (
                f"Spread {spread:.2f} EUR/MWh is elevated "
                f"(warning threshold {self.config.warn_spread_eur_mwh:.2f} EUR/MWh)"
            )
        return False, None

    def _check_volatility(self, volatility: float) -> Tuple[bool, Optional[str]]:
        """Block if volatility exceeds max; warn if above soft threshold."""
        if volatility > self.config.max_volatility_pct:
            return True, (
                f"Price volatility {volatility:.1f}% exceeds maximum "
                f"{self.config.max_volatility_pct:.1f}%"
            )
        if volatility > self.config.warn_volatility_pct:
            return False, (
                f"Price volatility {volatility:.1f}% is elevated "
                f"(warning threshold {self.config.warn_volatility_pct:.1f}%)"
            )
        return False, None

    def _check_data_freshness(
        self, data_quality: DataQualityResult
    ) -> Tuple[bool, Optional[str]]:
        """Block if data is stale."""
        if not data_quality.is_fresh:
            return True, (
                f"Data is stale: age {data_quality.age_minutes:.1f} minutes "
                f"(max {self.config.max_data_age_minutes} minutes)"
            )
        if data_quality.age_minutes > self.config.max_data_age_minutes * 0.75:
            return False, (
                f"Data is ageing: {data_quality.age_minutes:.1f} minutes old "
                f"(limit {self.config.max_data_age_minutes} minutes)"
            )
        return False, None

    def _check_missing_data(
        self, data_quality: DataQualityResult
    ) -> Tuple[bool, Optional[str]]:
        """Block if critical fields are missing; warn for non-critical."""
        critical_fields = {"price_eur_mwh", "residual_load_mw", "solar_mw", "wind_onshore_mw"}
        missing_critical = [f for f in data_quality.missing_fields if f in critical_fields]
        missing_non_critical = [
            f for f in data_quality.missing_fields if f not in critical_fields
        ]

        if missing_critical:
            return True, (
                f"Critical feature(s) missing: {', '.join(missing_critical)}"
            )
        if missing_non_critical:
            return False, (
                f"Non-critical feature(s) missing: {', '.join(missing_non_critical)}"
            )
        return False, None

    def _check_confidence(
        self, p_rebound: float, p_negative: float
    ) -> Tuple[bool, Optional[str]]:
        """Block if model confidence falls below thresholds."""
        if p_rebound < self.config.min_confidence_threshold:
            return True, (
                f"p_rebound {p_rebound:.3f} is below minimum confidence threshold "
                f"{self.config.min_confidence_threshold:.3f}"
            )
        if p_negative < self.config.min_p_negative_threshold:
            return True, (
                f"p_negative {p_negative:.3f} is below threshold "
                f"{self.config.min_p_negative_threshold:.3f} – insufficient evidence "
                "that price is actually negative"
            )
        return False, None

    def _check_position_limits(self, open_positions: int) -> Tuple[bool, Optional[str]]:
        """Block if maximum concurrent positions would be exceeded."""
        if open_positions >= self.config.max_open_positions:
            return True, (
                f"Open position limit reached: {open_positions}/{self.config.max_open_positions}"
            )
        if open_positions >= self.config.max_open_positions - 1:
            return False, (
                f"Approaching position limit: {open_positions}/{self.config.max_open_positions} open"
            )
        return False, None

    def _check_daily_loss(self, daily_pnl_eur: float) -> Tuple[bool, Optional[str]]:
        """Block if daily loss exceeds the configured maximum."""
        loss = -daily_pnl_eur  # positive value when in loss
        if loss >= self.config.max_daily_loss_eur:
            return True, (
                f"Daily loss limit breached: -{loss:.2f} EUR "
                f"(max -{self.config.max_daily_loss_eur:.2f} EUR)"
            )
        if loss >= self.config.max_daily_loss_eur * 0.75:
            return False, (
                f"Daily loss approaching limit: -{loss:.2f} EUR "
                f"(limit -{self.config.max_daily_loss_eur:.2f} EUR)"
            )
        return False, None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_volatility(features: pd.Series) -> float:
        """Extract volatility from features, returning 0 if unavailable."""
        for col in ("price_volatility_pct", "price_volatility", "volatility_pct"):
            if col in features.index and pd.notna(features[col]):
                return float(features[col])
        return 0.0

    @staticmethod
    def _score_ratio(value: float, maximum: float) -> float:
        """Return value/maximum clipped to [0, 1]."""
        if maximum <= 0:
            return 0.0
        return float(min(1.0, max(0.0, value / maximum)))
