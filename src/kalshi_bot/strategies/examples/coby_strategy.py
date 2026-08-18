"""Coby's 15-minute Kalshi live strategy.

Live-entry rules:
- One entry attempt maximum per 15-minute market ticker.
- Only enter during the final ``entry_window_seconds``.
- Never open a new position with 60 seconds or less remaining.
- Enter a YES or NO side only when the executable ask is within the
  configured 90c-95c entry range.
- Cap each live entry at $100 notional and 100 contracts.
- Exit at or above ``take_profit``.
- Exit at or below ``stop_price``.

IMPORTANT:
The framework remains dry-run unless the CLI is explicitly started with
``--live``. Every emitted OrderRequest still passes through the framework's
risk manager before execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from kalshi_bot.exchange.models import (
    Action,
    OrderRequest,
    OrderType,
    Side,
    TimeInForce,
)
from kalshi_bot.strategies.base import Strategy, StrategyContext
from kalshi_bot.telemetry.logging import get_logger


logger = get_logger(__name__)


class CobyStrategy(Strategy):
    """Coby's 15-minute BTC Kalshi strategy with live-order support."""

    name = "coby_strategy"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

        self.entry_min = int(params.get("entry_min", 90))
        self.entry_max = int(params.get("entry_max", 95))
        self.take_profit = int(params.get("take_profit", 98))
        self.stop_price = int(params.get("stop_price", 79))
        self.entry_window_seconds = int(
            params.get("entry_window_seconds", 300)
        )

        # Hard live-entry caps.
        self.max_contracts = int(params.get("max_contracts", 100))
        self.max_notional_cents = int(
            params.get("max_notional_cents", 10_000)
        )

        # One attempted entry per ticker. This prevents repeated orders if an
        # IOC order does not fill or only partially fills.
        self._traded_tickers: set[str] = set()

        # Remember which side we attempted for each ticker so we do not have
        # to infer YES/NO from the sign convention used by position_for().
        self._side_by_ticker: dict[str, Side] = {}

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

    def _entry_count(self, entry_price: int) -> int:
        """Return a whole-contract count that never exceeds the $100 cap."""
        if entry_price <= 0:
            return 0

        count_by_notional = self.max_notional_cents // entry_price
        return max(0, min(self.max_contracts, count_by_notional))

    def _order(
        self,
        *,
        ticker: str,
        action: Action,
        side: Side,
        price: int,
        count: int,
    ) -> OrderRequest:
        """Build an IOC limit order in the repo's existing YES/NO model."""
        kwargs: dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": OrderType.LIMIT,
            "time_in_force": TimeInForce.IMMEDIATE_OR_CANCEL,
        }

        if side is Side.YES:
            kwargs["yes_price"] = price
        else:
            kwargs["no_price"] = price

        return OrderRequest(**kwargs)

    def on_market_data(
        self,
        ctx: StrategyContext,
    ) -> list[OrderRequest]:
        m = ctx.market
        seconds_left = self._seconds_to_close(m.close_time)

        if seconds_left is None:
            return []

        position = ctx.position_for(m.ticker)

        # --------------------------------------------------------------
        # MANAGE AN EXISTING LIVE POSITION
        # --------------------------------------------------------------
        if position != 0:
            side = self._side_by_ticker.get(m.ticker)

            # Conservative fail-safe: if we somehow have a position but do
            # not know which outcome side created it, do not guess.
            if side is None:
                logger.error(
                    "LIVE EXIT BLOCKED | ticker=%s | position=%s | "
                    "reason=UNKNOWN_SIDE",
                    m.ticker,
                    position,
                )
                return []

            count = abs(int(position))
            if count <= 0:
                return []

            bid = m.yes_bid if side is Side.YES else m.no_bid
            if bid is None:
                return []

            if bid >= self.take_profit:
                logger.warning(
                    "LIVE EXIT SIGNAL | ticker=%s | side=%s | "
                    "bid=%dc | reason=TAKE_PROFIT | count=%d",
                    m.ticker,
                    side.value,
                    bid,
                    count,
                )
                return [
                    self._order(
                        ticker=m.ticker,
                        action=Action.SELL,
                        side=side,
                        price=bid,
                        count=count,
                    )
                ]

            if bid <= self.stop_price:
                logger.warning(
                    "LIVE EXIT SIGNAL | ticker=%s | side=%s | "
                    "bid=%dc | reason=STOP | count=%d",
                    m.ticker,
                    side.value,
                    bid,
                    count,
                )
                return [
                    self._order(
                        ticker=m.ticker,
                        action=Action.SELL,
                        side=side,
                        price=bid,
                        count=count,
                    )
                ]

            return []

        # --------------------------------------------------------------
        # NEW ENTRY
        # --------------------------------------------------------------

        # One attempted entry maximum per 15-minute market.
        if m.ticker in self._traded_tickers:
            return []

        # Only enter during the final five minutes, but never during the
        # final 60 seconds.
        if seconds_left <= 60 or seconds_left > self.entry_window_seconds:
            return []

        side: Side | None = None
        entry_price: int | None = None

        # LIVE entry uses the ask because that is the immediately executable
        # buy-side price. The paper version used bids, which can overstate
        # how favorable a real entry would be.
        if (
            m.yes_ask is not None
            and self.entry_min <= m.yes_ask <= self.entry_max
        ):
            side = Side.YES
            entry_price = m.yes_ask
        elif (
            m.no_ask is not None
            and self.entry_min <= m.no_ask <= self.entry_max
        ):
            side = Side.NO
            entry_price = m.no_ask

        if side is None or entry_price is None:
            return []

        count = self._entry_count(entry_price)
        if count <= 0:
            return []

        notional_cents = entry_price * count

        # Second hard check so a parameter mistake cannot push this strategy
        # above the configured live-entry notional.
        if notional_cents > self.max_notional_cents:
            logger.error(
                "LIVE ENTRY BLOCKED | ticker=%s | notional=%dc | limit=%dc",
                m.ticker,
                notional_cents,
                self.max_notional_cents,
            )
            return []

        # Mark before emitting the order so repeated market-data ticks cannot
        # fire duplicate entry orders.
        self._traded_tickers.add(m.ticker)
        self._side_by_ticker[m.ticker] = side

        logger.warning(
            "LIVE ENTRY SIGNAL | ticker=%s | side=%s | entry=%dc | "
            "count=%d | notional=$%.2f | seconds_left=%.1f | "
            "yes_bid=%s | yes_ask=%s | no_bid=%s | no_ask=%s",
            m.ticker,
            side.value,
            entry_price,
            count,
            notional_cents / 100,
            seconds_left,
            m.yes_bid,
            m.yes_ask,
            m.no_bid,
            m.no_ask,
        )

        return [
            self._order(
                ticker=m.ticker,
                action=Action.BUY,
                side=side,
                price=entry_price,
                count=count,
            )
        ]
