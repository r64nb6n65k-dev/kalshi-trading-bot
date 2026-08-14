from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from kalshi_bot.exchange.models import Action, OrderRequest, OrderType, Side
from kalshi_bot.strategies.base import Strategy, StrategyContext


class CobyStrategy(Strategy):
"""
Late-market Kalshi strategy.

Version 1:
- Only considers entries in the final 5 minutes.
- Maximum entry price: 90 cents.
- Take profit: 98 cents.
- Stop trigger: 79 cents.
- Does NOT enter until the external BTC/VWAP signal is connected.
"""

name = "coby_strategy"

def __init__(self, **params: Any) -> None:
super().__init__(**params)

self.entry_max = int(params.get("entry_max", 90))
self.take_profit = int(params.get("take_profit", 98))
self.stop_price = int(params.get("stop_price", 79))
self.entry_window_seconds = int(
params.get("entry_window_seconds", 300)
)
self.size = int(params.get("size", 1))

def _seconds_to_close(self, close_time: str | None) -> float | None:
if not close_time:
return None

try:
close_dt = datetime.fromisoformat(
close_time.replace("Z", "+00:00")
)

if close_dt.tzinfo is None:
close_dt = close_dt.replace(tzinfo=timezone.utc)

now = datetime.now(timezone.utc)
return (close_dt - now).total_seconds()

except ValueError:
return None

def _external_signal(self) -> Side | None:
"""
Placeholder for BTC/Gold market-data logic.

Later this will evaluate:
- underlying price vs Kalshi target
- VWAP
- trend
- momentum / follow-through
- chop
- distance from target

Returns Side.YES, Side.NO, or None.
"""

return None

def on_market_data(
self,
ctx: StrategyContext,
) -> list[OrderRequest]:
m = ctx.market
inventory = ctx.position_for(m.ticker)

# EXIT EXISTING YES POSITION
if inventory > 0 and m.yes_bid is not None:
if m.yes_bid >= self.take_profit:
return [
OrderRequest(
ticker=m.ticker,
action=Action.SELL,
side=Side.YES,
count=inventory,
type=OrderType.LIMIT,
yes_price=m.yes_bid,
)
]

if m.yes_bid <= self.stop_price:
return [
OrderRequest(
ticker=m.ticker,
action=Action.SELL,
side=Side.YES,
count=inventory,
type=OrderType.MARKET,
)
]

return []

# EXIT EXISTING NO POSITION
if inventory < 0 and m.no_bid is not None:
count = abs(inventory)

if m.no_bid >= self.take_profit:
return [
OrderRequest(
ticker=m.ticker,
action=Action.SELL,
side=Side.NO,
count=count,
type=OrderType.LIMIT,
no_price=m.no_bid,
)
]

if m.no_bid <= self.stop_price:
return [
OrderRequest(
ticker=m.ticker,
action=Action.SELL,
side=Side.NO,
count=count,
type=OrderType.MARKET,
)
]

return []

# ENTRY TIMING FILTER
seconds_left = self._seconds_to_close(m.close_time)

if seconds_left is None:
return []

if seconds_left > self.entry_window_seconds:
return []

if seconds_left <= 0:
return []

# EXTERNAL MARKET SIGNAL
signal = self._external_signal()

# Entries remain disabled until BTC/VWAP data is connected.
if signal is None:
return []

# YES ENTRY
if signal is Side.YES:
if m.yes_ask is None:
return []

if m.yes_ask > self.entry_max:
return []

return [
OrderRequest(
ticker=m.ticker,
action=Action.BUY,
side=Side.YES,
count=self.size,
type=OrderType.LIMIT,
yes_price=m.yes_ask,
)
]

# NO ENTRY
if signal is Side.NO:
if m.no_ask is None:
return []

if m.no_ask > self.entry_max:
return []

return [
OrderRequest(
ticker=m.ticker,
action=Action.BUY,
side=Side.NO,
count=self.size,
type=OrderType.LIMIT,
no_price=m.no_ask,
)
]

return []
