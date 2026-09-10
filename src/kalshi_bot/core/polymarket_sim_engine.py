"""Simulation-only execution engine for Polymarket rolling crypto markets."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from kalshi_bot.data.multi_crypto import MultiCryptoPriceFeed
from kalshi_bot.exchange.models import Side
from kalshi_bot.polymarket import PolymarketListing, PolymarketPublicClient
from kalshi_bot.strategies.examples.polymarket_momentum import (
    PolymarketMomentumStrategy,
    SimSignal,
)
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class PendingFill:
    listing: PolymarketListing
    signal: SimSignal
    started: float
    deadline: float


class PolymarketSimEngine:
    """Paper execution only; this class has no order-submission capability."""

    def __init__(
        self,
        *,
        client: PolymarketPublicClient,
        feed: MultiCryptoPriceFeed,
        strategy: PolymarketMomentumStrategy,
        poll_interval: float = 1.0,
        execution_window_seconds: float = 3.0,
    ) -> None:
        self.client = client
        self.feed = feed
        self.strategy = strategy
        self.poll_interval = max(0.25, poll_interval)
        self.execution_window_seconds = max(0.0, execution_window_seconds)
        self._targets: dict[str, float] = {}
        self._known: dict[str, PolymarketListing] = {}
        self._pending: dict[str, PendingFill] = {}

    def _target_for(self, listing: PolymarketListing) -> float | None:
        existing = self._targets.get(listing.slug)
        if existing is not None:
            return existing
        product = self.client.ASSETS[listing.asset]
        ticks = self.feed.snapshot(product)
        if not ticks:
            return None
        opening = min(ticks, key=lambda tick: abs(tick.timestamp.timestamp() - listing.open_time))
        # Do not invent a price-to-beat when the process started mid-market.
        if abs(opening.timestamp.timestamp() - listing.open_time) > 5:
            return None
        self._targets[listing.slug] = opening.price
        logger.warning(
            "OPENING REFERENCE CAPTURED | ticker=%s | price=%.6f | source=%s",
            listing.slug,
            opening.price,
            opening.source,
        )
        return opening.price

    def _try_fill(self, pending: PendingFill, listing: PolymarketListing, now: float) -> bool:
        ask = listing.yes_ask if pending.signal.side is Side.YES else listing.no_ask
        if ask is not None and ask <= pending.signal.limit_price:
            self.strategy.open_position(listing, pending.signal.side, ask)
            logger.warning(
                "POLYMARKET PAPER FILL | ticker=%s | side=%s | signal_ask=%dc | "
                "fixed_limit=%dc | fill=%dc | count=%d",
                listing.slug,
                pending.signal.side.value,
                pending.signal.signal_ask,
                pending.signal.limit_price,
                ask,
                self.strategy.contracts,
            )
            return True
        if now >= pending.deadline:
            logger.warning(
                "POLYMARKET PAPER CANCEL | ticker=%s | reason=NO_FILL_3S_FIXED_LIMIT | "
                "fixed_limit=%dc",
                listing.slug,
                pending.signal.limit_price,
            )
            return True
        return False

    def _take_profit(self, listing: PolymarketListing, now: float) -> None:
        position = self.strategy.positions.get(listing.slug)
        if position is None:
            return
        bid = listing.yes_bid if position.side is Side.YES else listing.no_bid
        if bid is not None and bid >= self.strategy.take_profit:
            self.strategy.close_position(listing.slug, bid, "TAKE_PROFIT_98", now)
            logger.warning(
                "POLYMARKET PAPER EXIT | ticker=%s | side=%s | fill=%dc | reason=TAKE_PROFIT_98",
                listing.slug,
                position.side.value,
                bid,
            )

    async def _settle_missing(self, active: set[str], now: float) -> None:
        for slug in set(self.strategy.positions) - active:
            listing = self._known.get(slug)
            if listing is None or now < listing.close_time:
                continue
            resolved = await self.client.resolved(listing)
            if resolved is None or resolved.resolved_side not in {"yes", "no"}:
                continue
            position = self.strategy.positions.get(slug)
            if position is None:
                continue
            won = position.side.value == resolved.resolved_side
            self.strategy.close_position(
                slug,
                100 if won else 0,
                "SETTLEMENT_WIN" if won else "SETTLEMENT_LOSS",
                now,
            )

    async def run(self, max_cycles: int | None = None) -> None:
        logger.warning(
            "POLYMARKET CRYPTO SIM STARTED | intervals=%s | assets=%s | "
            "bankroll=$%.2f | contracts=%d | live_orders=DISABLED",
            self.client.intervals,
            tuple(self.client.ASSETS),
            self.strategy.bankroll_cents / 100,
            self.strategy.contracts,
        )
        await self.feed.start()
        cycle = 0
        try:
            while max_cycles is None or cycle < max_cycles:
                try:
                    now = time.time()
                    listings = await self.client.get_open_crypto_markets(now)
                    active = {listing.slug for listing in listings}
                    self._known.update({listing.slug: listing for listing in listings})
                    await self._settle_missing(active, now)
                    for listing in listings:
                        self._take_profit(listing, now)
                        pending = self._pending.get(listing.slug)
                        if pending is not None:
                            if self._try_fill(pending, listing, now):
                                self._pending.pop(listing.slug, None)
                            continue
                        if listing.slug in self.strategy.positions:
                            continue
                        product = self.client.ASSETS[listing.asset]
                        signal = self.strategy.evaluate(
                            listing,
                            self._target_for(listing),
                            now,
                            self.feed.snapshot(product),
                        )
                        if signal is not None:
                            pending = PendingFill(
                                listing=listing,
                                signal=signal,
                                started=now,
                                deadline=now + self.execution_window_seconds,
                            )
                            self._pending[listing.slug] = pending
                            if self._try_fill(pending, listing, now):
                                self._pending.pop(listing.slug, None)
                    keep = active | set(self.strategy.positions) | set(self._pending)
                    self.strategy.prune(keep)
                    self._targets = {k: v for k, v in self._targets.items() if k in keep}
                except Exception:
                    logger.exception("POLYMARKET SIM SCAN FAILED; retrying")
                cycle += 1
                if max_cycles is None or cycle < max_cycles:
                    await asyncio.sleep(self.poll_interval)
        finally:
            await self.feed.stop()
