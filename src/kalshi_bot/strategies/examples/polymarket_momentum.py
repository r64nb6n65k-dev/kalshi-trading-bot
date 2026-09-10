"""Simulation-only port of the live crypto momentum strategy to Polymarket."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import ClassVar
from zoneinfo import ZoneInfo

from kalshi_bot.dashboard import record_entry, record_exit, record_model_snapshot
from kalshi_bot.exchange.models import Side
from kalshi_bot.polymarket import PolymarketListing
from kalshi_bot.strategies.base import UnderlyingTick
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SimSignal:
    side: Side
    signal_ask: int
    limit_price: int
    detail: str


@dataclass(frozen=True, slots=True)
class SimPosition:
    side: Side
    entry_price: int
    count: int


class PolymarketMomentumStrategy:
    """Same score, risk windows and take-profit rules as the live bot."""

    _CENTRAL: ClassVar[ZoneInfo] = ZoneInfo("America/Chicago")
    _NO_ENTRY_WINDOWS: ClassVar[tuple[tuple[int, int], ...]] = (
        (0, 2 * 60),
        (8 * 60, 10 * 60),
        (19 * 60, 20 * 60),
    )

    def __init__(
        self,
        *,
        contracts: int = 10,
        bankroll_cents: int = 50_000,
        take_profit: int = 98,
        minimum_history: float = 45.0,
        minimum_separation_bps: float = 4.0,
        entry_slippage_cents: int = 2,
        decision_window: float = 15.0,
        drawdown_limit_cents: int = 4_000,
    ) -> None:
        self.contracts = contracts
        self.bankroll_cents = bankroll_cents
        self.take_profit = take_profit
        self.minimum_history = minimum_history
        self.minimum_separation_bps = minimum_separation_bps
        self.entry_slippage_cents = entry_slippage_cents
        self.decision_window = decision_window
        self.drawdown_limit_cents = drawdown_limit_cents
        self.decided: set[str] = set()
        self.positions: dict[str, SimPosition] = {}
        self.total_pnl_cents = 0
        self._daily_day: date | None = None
        self._daily_pnl_cents = 0
        self._daily_peak_cents = 0

    @staticmethod
    def decision_seconds(interval_minutes: int) -> float:
        # The live 15-minute bot observes the first third (5 minutes), then
        # decides with two thirds remaining.  Preserve that timing on 5m too.
        return interval_minutes * 60 * (2 / 3)

    @classmethod
    def entry_block_reason(cls, now: float) -> str | None:
        central = datetime.fromtimestamp(now, UTC).astimezone(cls._CENTRAL)
        minute = central.hour * 60 + central.minute
        for start, end in cls._NO_ENTRY_WINDOWS:
            if start <= minute < end:
                return f"NO_ENTRY_WINDOW_CT | central_time={central:%Y-%m-%d %H:%M:%S %Z}"
        return None

    def _ensure_day(self, now: float) -> None:
        current = datetime.fromtimestamp(now, UTC).astimezone(self._CENTRAL).date()
        if current == self._daily_day:
            return
        self._daily_day = current
        self._daily_pnl_cents = 0
        self._daily_peak_cents = 0

    @property
    def drawdown_cents(self) -> int:
        return self._daily_peak_cents - self._daily_pnl_cents

    def reserved_cents(self) -> int:
        return sum(p.entry_price * p.count for p in self.positions.values())

    @staticmethod
    def _at_or_before(ticks: tuple[UnderlyingTick, ...], timestamp: float) -> UnderlyingTick:
        return min(ticks, key=lambda row: abs(row.timestamp.timestamp() - timestamp))

    def _signal(
        self,
        target: float,
        now: float,
        ticks: tuple[UnderlyingTick, ...],
    ) -> tuple[Side | None, str]:
        if target <= 0:
            return None, "MISSING_OPENING_REFERENCE"
        if len(ticks) < 2:
            return None, "NO_UNDERLYING_FEED_OR_HISTORY"
        latest = ticks[-1]
        first_time = ticks[0].timestamp.timestamp()
        if now - latest.timestamp.timestamp() > 5 or now - first_time < self.minimum_history:
            return None, "STALE_OR_INSUFFICIENT_PRICE_HISTORY"
        short = self._at_or_before(ticks, now - 60)
        long = ticks[0]
        short_bps = (latest.price - short.price) / target * 10_000
        long_bps = (latest.price - long.price) / target * 10_000
        separation_bps = (latest.price - target) / target * 10_000
        if abs(separation_bps) < self.minimum_separation_bps:
            return (
                None,
                f"INSUFFICIENT_SEPARATION | separation_bps={separation_bps:+.2f} "
                f"minimum={self.minimum_separation_bps:.2f}",
            )
        recent_volume = sum(
            tick.size for tick in ticks if tick.timestamp.timestamp() >= now - 60
        )
        older_volume = sum(
            tick.size for tick in ticks if tick.timestamp.timestamp() < now - 60
        )
        older_seconds = max(1.0, now - first_time - 60)
        baseline_volume = older_volume * 60 / older_seconds
        volume_ratio = recent_volume / baseline_volume if baseline_volume > 0 else 1.0
        volume_weight = max(0.5, min(2.0, volume_ratio))
        score = 0.55 * separation_bps + volume_weight * (
            0.30 * short_bps + 0.15 * long_bps
        )
        side = Side.YES if score >= 0 else Side.NO
        return side, (
            f"underlying={latest.price:.6f} source={latest.source} "
            f"target={target:.6f} score={score:+.2f} "
            f"separation_bps={separation_bps:+.2f} "
            f"momentum_60_bps={short_bps:+.2f} "
            f"momentum_long_bps={long_bps:+.2f} volume_ratio={volume_ratio:.2f}"
        )

    def evaluate(
        self,
        listing: PolymarketListing,
        target: float | None,
        now: float,
        ticks: tuple[UnderlyingTick, ...],
    ) -> SimSignal | None:
        self._ensure_day(now)
        if listing.slug in self.decided or listing.slug in self.positions:
            return None
        seconds_left = listing.close_time - now
        decision = self.decision_seconds(listing.interval_minutes)
        if not decision - self.decision_window <= seconds_left <= decision:
            return None
        self.decided.add(listing.slug)
        if target is None:
            self._snapshot(listing, seconds_left, None, "SKIP", "MISSING_OPENING_REFERENCE")
            return None
        side, detail = self._signal(target, now, ticks)
        if side is None:
            self._snapshot(listing, seconds_left, target, "SKIP", detail)
            return None
        ask = listing.yes_ask if side is Side.YES else listing.no_ask
        if ask is None or not 1 <= ask <= 99:
            self._snapshot(listing, seconds_left, target, "SKIP", "NO_EXECUTABLE_ASK")
            return None
        limit_price = min(99, ask + self.entry_slippage_cents)
        block = self.entry_block_reason(now)
        if self.drawdown_cents >= self.drawdown_limit_cents:
            block = (
                f"DAILY_DRAWDOWN_LIMIT_CT | drawdown_cents={self.drawdown_cents} "
                f"limit_cents={self.drawdown_limit_cents}"
            )
        if self.reserved_cents() + limit_price * self.contracts > self.bankroll_cents:
            block = "BANKROLL_CAP"
        if block:
            self._snapshot(
                listing,
                seconds_left,
                target,
                f"SHADOW_BUY_{side.value.upper()}",
                f"{block} | intended_side={side.value} | {detail}",
            )
            return None
        self._snapshot(listing, seconds_left, target, f"BUY_{side.value.upper()}", detail)
        return SimSignal(side=side, signal_ask=ask, limit_price=limit_price, detail=detail)

    def _snapshot(
        self,
        listing: PolymarketListing,
        seconds_left: float,
        target: float | None,
        decision: str,
        reason: str,
    ) -> None:
        record_model_snapshot(
            ticker=listing.slug,
            seconds_left=seconds_left,
            target_price=target,
            yes_bid=listing.yes_bid,
            yes_ask=listing.yes_ask,
            no_bid=listing.no_bid,
            no_ask=listing.no_ask,
            decision=decision,
            reason=f"market_source={listing.source} | {reason}",
        )
        logger.warning("%s | ticker=%s | %s", decision, listing.slug, reason)

    def open_position(self, listing: PolymarketListing, side: Side, price: int) -> None:
        position = SimPosition(side=side, entry_price=price, count=self.contracts)
        self.positions[listing.slug] = position
        record_entry(
            ticker=listing.slug,
            side=side.value,
            entry_price=price,
            count=self.contracts,
            seconds_left=listing.close_time - datetime.now(UTC).timestamp(),
            take_profit=self.take_profit,
            execution_mode="polymarket_paper",
        )

    def close_position(self, slug: str, exit_price: int, reason: str, now: float) -> None:
        position = self.positions.pop(slug, None)
        if position is None:
            return
        pnl = (exit_price - position.entry_price) * position.count
        self.total_pnl_cents += pnl
        self._ensure_day(now)
        self._daily_pnl_cents += pnl
        self._daily_peak_cents = max(self._daily_peak_cents, self._daily_pnl_cents)
        record_exit(
            ticker=slug,
            side=position.side.value,
            entry_price=position.entry_price,
            exit_price=exit_price,
            reason=reason,
            count=position.count,
            pnl_cents=pnl,
            total_pnl_cents=self.total_pnl_cents,
        )

    def prune(self, keep: set[str]) -> None:
        self.decided.intersection_update(keep | set(self.positions))
