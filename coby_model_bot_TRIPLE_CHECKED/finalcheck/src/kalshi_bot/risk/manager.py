"""Risk management: pre-trade checks and position sizing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kalshi_bot.exchange.models import OrderRequest

if TYPE_CHECKING:
    from kalshi_bot.config import RiskSettings


@dataclass(slots=True)
class RiskLimits:
    """Configurable risk limits."""

    max_contracts_per_order: int = 200
    max_position_per_market: int = 500
    max_order_notional_cents: int = 50_000
    kelly_fraction: float = 0.25

    @classmethod
    def from_settings(cls, risk: RiskSettings) -> RiskLimits:
        """Build limits from validated risk settings."""
        return cls(
            max_contracts_per_order=risk.max_contracts_per_order,
            max_position_per_market=risk.max_position_per_market,
            max_order_notional_cents=risk.max_order_notional_cents,
            kelly_fraction=risk.kelly_fraction,
        )


@dataclass(slots=True)
class RiskDecision:
    """Result of a pre-trade risk check."""

    approved: bool
    reason: str = ""


class RiskManager:
    """Enforces pre-trade risk limits and sizes positions."""

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    @classmethod
    def from_settings(cls, risk: RiskSettings) -> RiskManager:
        """Build a risk manager from validated risk settings."""
        return cls(RiskLimits.from_settings(risk))

    def check(self, order: OrderRequest, current_position: int) -> RiskDecision:
        """Approve or reject an order against the configured limits."""
        if order.count > self.limits.max_contracts_per_order:
            return RiskDecision(False, "order count exceeds max_contracts_per_order")

        # YES contracts are represented as positive positions and NO contracts
        # as negative positions. Buying NO moves the position negative; selling
        # NO moves it back toward zero.
        side_direction = 1 if order.side.value == "yes" else -1
        action_direction = 1 if order.action.value == "buy" else -1
        signed = order.count * side_direction * action_direction
        projected = current_position + signed
        if abs(projected) > self.limits.max_position_per_market:
            return RiskDecision(False, "projected position exceeds max_position_per_market")

        price = order.yes_price or order.no_price or 0
        notional = price * order.count
        if notional > self.limits.max_order_notional_cents:
            return RiskDecision(False, "order notional exceeds max_order_notional_cents")

        return RiskDecision(True)

    def kelly_size(self, edge_prob: float, price_cents: int, bankroll_cents: int) -> int:
        """Return a fractional-Kelly contract count capped by order limits."""
        p = price_cents / 100.0
        if not 0.0 < p < 1.0:
            return 0
        full_kelly = (edge_prob - p) / (1.0 - p)
        if full_kelly <= 0:
            return 0
        fraction = full_kelly * self.limits.kelly_fraction
        stake_cents = bankroll_cents * fraction
        contracts = int(stake_cents // price_cents)
        return max(0, min(contracts, self.limits.max_contracts_per_order))
