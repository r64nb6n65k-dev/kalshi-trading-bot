"""Kalshi trading-fee math.

General schedule: ``fee = ceil(rate * C * P * (1 - P))`` dollars, rounded up to
the next cent, where ``C`` is contracts and ``P`` the price in dollars. The
default rate is 0.07; check Kalshi's current fee schedule and set
``KALSHI_WEATHER__FEE_RATE`` if it changes.
"""

from __future__ import annotations

import math


def fee_cents(price_cents: int, count: int, rate: float = 0.07) -> int:
    """Total fee in cents for ``count`` contracts filled at ``price_cents``."""
    if count <= 0 or not 0 < price_cents < 100:
        return 0
    p = price_cents / 100.0
    raw = rate * count * p * (1.0 - p) * 100.0
    # Guard against float noise turning an exact cent into the next one.
    return math.ceil(round(raw, 6))


def fee_per_contract(price_cents: int, rate: float = 0.07) -> float:
    """Marginal fee per contract in probability units (cents / 100), unrounded."""
    if not 0 < price_cents < 100:
        return 0.0
    p = price_cents / 100.0
    return rate * p * (1.0 - p)
