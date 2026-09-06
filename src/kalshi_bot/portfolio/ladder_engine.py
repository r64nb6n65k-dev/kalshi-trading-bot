"""Paper-only same-day strike-ladder arbitrage across all Kalshi categories."""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from itertools import combinations
from typing import Any
from zoneinfo import ZoneInfo

from kalshi_bot.dashboard import record_entry, record_exit, record_model_snapshot
from kalshi_bot.exchange.client import KalshiClient
from kalshi_bot.exchange.models import Market
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)
CENTRAL = ZoneInfo("America/Chicago")
UPPER_WORDS = ("ABOVE", "GREATER", "HIGHER", "AT LEAST", "OVER")


def fee_cents(price: int, count: int, coefficient: float = 0.07) -> int:
    """Conservative Kalshi-style fee estimate, rounded up to a cent."""
    probability = Decimal(price) / Decimal(100)
    dollars = Decimal(str(coefficient)) * Decimal(count) * probability * (1 - probability)
    return int((dollars * 100).to_integral_value(rounding=ROUND_CEILING))


@dataclass(frozen=True)
class LadderOpportunity:
    lower: Market
    higher: Market
    yes_ask: int
    no_ask: int
    count: int
    fees: int

    @property
    def unit_cost(self) -> int:
        return self.yes_ask + self.no_ask

    @property
    def total_cost(self) -> int:
        return self.unit_cost * self.count + self.fees

    @property
    def guaranteed_payout(self) -> int:
        return 100 * self.count

    @property
    def profit(self) -> int:
        return self.guaranteed_payout - self.total_cost

    @property
    def key(self) -> str:
        return f"LADDER:{self.lower.ticker}|{self.higher.ticker}"


@dataclass
class PaperBundle:
    opportunity: LadderOpportunity
    release: datetime


def _close_time(market: Market) -> datetime | None:
    if not market.close_time:
        return None
    try:
        value = datetime.fromisoformat(market.close_time.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _api_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def closes_today(market: Market, now: datetime) -> bool:
    """True only for a future close on the current Central calendar day."""
    close = _close_time(market)
    if close is None or close <= now:
        return False
    return close.astimezone(CENTRAL).date() == now.astimezone(CENTRAL).date()


def resolves_promptly(market: Market, maximum_delay: timedelta = timedelta(hours=24)) -> bool:
    """Reject contracts whose advertised determination may be materially delayed."""
    close = _close_time(market)
    expected = _api_time(market.expected_expiration_time)
    if close is None or expected is None or expected < close:
        return False
    if expected - close > maximum_delay:
        return False
    timer = market.settlement_timer_seconds
    return timer is not None and 0 <= timer <= 7_200


def _release_time(market: Market) -> datetime | None:
    expected = _api_time(market.expected_expiration_time)
    timer = market.settlement_timer_seconds
    if expected is None or timer is None:
        return None
    return expected + timedelta(seconds=max(0, timer))


def _central_day_bounds(now: datetime) -> tuple[int, int]:
    local = now.astimezone(CENTRAL)
    start = datetime.combine(local.date(), time.min, CENTRAL)
    end = start + timedelta(days=1)
    return int(now.timestamp()), int(end.astimezone(UTC).timestamp())


def is_upper_threshold(market: Market) -> bool:
    if market.floor_strike is None or market.cap_strike is not None:
        return False
    text = " ".join(
        filter(None, (market.title, market.subtitle, market.rules_primary))
    ).upper()
    return market.strike_type in {None, "greater", "greater_or_equal"} and any(
        word in text for word in UPPER_WORDS
    )


def rules_signature(market: Market) -> str:
    """Remove numeric strike values while retaining the settlement wording."""
    text = market.rules_primary or market.title or ""
    normalized = re.sub(r"[$â¬Â£]?[-+]?\d[\d,]*(?:\.\d+)?", "#", text.upper())
    return re.sub(r"\s+", " ", normalized).strip()


def compatible(lower: Market, higher: Market) -> bool:
    """Require the same event, close, direction, and normalized settlement rule."""
    return bool(
        lower.event_ticker
        and lower.event_ticker == higher.event_ticker
        and lower.series_ticker == higher.series_ticker
        and lower.close_time == higher.close_time
        and lower.expected_expiration_time == higher.expected_expiration_time
        and lower.strike_type == higher.strike_type
        and is_upper_threshold(lower)
        and is_upper_threshold(higher)
        and float(lower.floor_strike or 0) < float(higher.floor_strike or 0)
        and rules_signature(lower)
        and rules_signature(lower) == rules_signature(higher)
    )


def _price_cents(value: Any) -> int | None:
    try:
        number = Decimal(str(value))
    except Exception:
        return None
    if number <= 1:
        number *= 100
    return int(number.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def executable_ask_depth(book: dict[str, Any], outcome: str, ask: int) -> int:
    """Derive ask-side depth from Kalshi's YES/NO bid-only orderbook."""
    opposing = "no" if outcome == "yes" else "yes"
    levels = book.get(f"{opposing}_dollars") or book.get(opposing) or []
    minimum_bid = 100 - ask
    quantity = 0
    for level in levels:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue
        price = _price_cents(level[0])
        try:
            count = int(Decimal(str(level[1])))
        except Exception:
            continue
        if price is not None and price >= minimum_bid:
            quantity += max(0, count)
    return quantity


class SameDayLadderEngine:
    """Scan every open market but book only depth-confirmed same-day paper pairs."""

    def __init__(
        self,
        client: KalshiClient,
        bankroll_cents: int = 50_000,
        poll_interval: float = 10.0,
        max_pairs: int = 5,
        contracts_per_pair: int = 5,
        minimum_profit_per_pair: int = 3,
    ) -> None:
        self.client = client
        self.cash = bankroll_cents
        self.realized_pnl = 0
        self.poll_interval = max(2.0, poll_interval)
        self.max_pairs = max(1, max_pairs)
        self.contracts_per_pair = max(1, contracts_per_pair)
        self.minimum_profit_per_pair = max(1, minimum_profit_per_pair)
        self.locked: dict[str, PaperBundle] = {}
        self.seen: set[str] = set()
        self._running = False

    async def _books(self, tickers: set[str]) -> dict[str, dict[str, Any]]:
        semaphore = asyncio.Semaphore(12)

        async def fetch(ticker: str) -> tuple[str, dict[str, Any]]:
            async with semaphore:
                return ticker, await self.client.get_orderbook(ticker, depth=100)

        return dict(await asyncio.gather(*(fetch(ticker) for ticker in sorted(tickers))))

    async def scan(self, now: datetime | None = None) -> list[LadderOpportunity]:
        now = now or datetime.now(UTC)
        minimum_close, maximum_close = _central_day_bounds(now)
        markets = await self.client.get_all_markets(
            min_close_ts=minimum_close,
            max_close_ts=maximum_close,
            mve_filter="exclude",
        )
        eligible = [
            m
            for m in markets
            if m.status in {"active", "open"}
            and closes_today(m, now)
            and resolves_promptly(m)
            and is_upper_threshold(m)
        ]
        groups: dict[tuple[str, str], list[Market]] = defaultdict(list)
        for market in eligible:
            groups[(market.event_ticker or "", market.close_time or "")].append(market)

        raw_pairs: list[tuple[Market, Market]] = []
        for group in groups.values():
            ordered = sorted(group, key=lambda item: float(item.floor_strike or 0))
            raw_pairs.extend(
                (lower, higher)
                for lower, higher in combinations(ordered, 2)
                if compatible(lower, higher)
                and f"LADDER:{lower.ticker}|{higher.ticker}" not in self.seen
            )
        books = await self._books({m.ticker for pair in raw_pairs for m in pair})
        opportunities: list[LadderOpportunity] = []
        for lower, higher in raw_pairs:
            if lower.yes_ask is None or higher.no_ask is None:
                continue
            yes_ask, no_ask = int(lower.yes_ask), int(higher.no_ask)
            if not 1 <= yes_ask <= 99 or not 1 <= no_ask <= 99:
                continue
            available = min(
                executable_ask_depth(books.get(lower.ticker, {}), "yes", yes_ask),
                executable_ask_depth(books.get(higher.ticker, {}), "no", no_ask),
            )
            count = min(self.contracts_per_pair, available)
            if count <= 0:
                continue
            fees = fee_cents(yes_ask, count) + fee_cents(no_ask, count)
            opportunity = LadderOpportunity(lower, higher, yes_ask, no_ask, count, fees)
            required = self.minimum_profit_per_pair * count
            if opportunity.profit >= required and opportunity.total_cost <= self.cash:
                opportunities.append(opportunity)
        return sorted(opportunities, key=lambda item: item.profit, reverse=True)

    def _book(self, opportunity: LadderOpportunity) -> None:
        release = _release_time(opportunity.lower)
        if (
            release is None
            or opportunity.key in self.seen
            or opportunity.total_cost > self.cash
        ):
            return
        self.cash -= opportunity.total_cost
        self.seen.add(opportunity.key)
        self.locked[opportunity.key] = PaperBundle(opportunity, release)
        record_entry(
            opportunity.key[:180],
            "LOW_YES_HIGH_NO",
            opportunity.unit_cost,
            count=opportunity.count,
            take_profit=100,
            execution_mode="paper_depth_confirmed_same_day_ladder",
        )
        logger.warning(
            "PAPER SAME-DAY LADDER | %s | count=%d | cost=$%.2f | min_profit=$%.2f",
            opportunity.key,
            opportunity.count,
            opportunity.total_cost / 100,
            opportunity.profit / 100,
        )

    def _release(self, now: datetime) -> None:
        for key, bundle in list(self.locked.items()):
            if now < bundle.release:
                continue
            opportunity = bundle.opportunity
            self.cash += opportunity.guaranteed_payout
            self.realized_pnl += opportunity.profit
            self.locked.pop(key, None)
            record_exit(
                key[:180],
                "LOW_YES_HIGH_NO",
                opportunity.unit_cost,
                100,
                "SAME_DAY_GUARANTEED_MINIMUM",
                opportunity.count,
                opportunity.profit,
                self.realized_pnl,
            )

    async def step(self) -> int:
        now = datetime.now(UTC)
        self._release(now)
        opportunities = await self.scan(now)
        capacity = self.max_pairs - len(self.locked)
        used_tickers = {
            ticker
            for bundle in self.locked.values()
            for ticker in (
                bundle.opportunity.lower.ticker,
                bundle.opportunity.higher.ticker,
            )
        }
        booked = 0
        for opportunity in opportunities:
            if booked >= max(0, capacity):
                break
            pair_tickers = {opportunity.lower.ticker, opportunity.higher.ticker}
            if used_tickers & pair_tickers:
                continue
            self._book(opportunity)
            if opportunity.key in self.locked:
                used_tickers.update(pair_tickers)
                booked += 1
        record_model_snapshot(
            ticker="ALL_MARKET_LADDER",
            decision="MONITOR",
            reason=(
                f"SAME_DAY_OPPORTUNITIES_{len(opportunities)}_LOCKED_{len(self.locked)}_"
                f"CASH_{self.cash}_PNL_{self.realized_pnl}_PAPER_ONLY"
            ),
        )
        return len(opportunities)

    async def run(self, max_cycles: int | None = None) -> None:
        if self._running:
            return
        self._running = True
        cycles = 0
        try:
            while self._running:
                try:
                    await self.step()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("All-market ladder scan failed; retrying")
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    break
                await asyncio.sleep(self.poll_interval)
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False
