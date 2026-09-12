

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
    """Trade the opening-target direction only when market momentum confirms it."""

    _CENTRAL: ClassVar[ZoneInfo] = ZoneInfo("America/Chicago")

    def __init__(
        self,
        *,
        contracts: int = 5,
        bankroll_cents: int = 50_000,
        take_profit: int = 98,
        minimum_entry_price: int = 50,
        maximum_entry_price: int = 80,
        minimum_history: float = 20.0,
        minimum_separation_bps: float = 4.0,
        minimum_volume_ratio: float = 0.25,
        low_volume_override_bps: float = 12.0,
        confirmations: int = 2,
        evaluation_interval: float = 2.0,
        entry_slippage_cents: int = 2,
        final_entry_seconds: float = 60.0,
        drawdown_limit_cents: int = 4_000,
    ) -> None:
        self.contracts = contracts
        self.bankroll_cents = bankroll_cents
        self.take_profit = take_profit
        self.minimum_entry_price = max(1, min(99, minimum_entry_price))
        self.maximum_entry_price = max(
            self.minimum_entry_price,
            min(99, maximum_entry_price),
        )
        self.minimum_history = minimum_history
        self.minimum_separation_bps = minimum_separation_bps
        self.minimum_volume_ratio = max(0.0, minimum_volume_ratio)
        self.low_volume_override_bps = max(
            self.minimum_separation_bps,
            low_volume_override_bps,
        )
        self.confirmations = max(1, confirmations)
        self.evaluation_interval = max(0.25, evaluation_interval)
        self.entry_slippage_cents = entry_slippage_cents
        self.final_entry_seconds = max(0.0, final_entry_seconds)
        self.drawdown_limit_cents = drawdown_limit_cents
        self.decided: set[str] = set()
        self.positions: dict[str, SimPosition] = {}
        self._last_retry_reason: dict[str, str] = {}
        self._last_evaluation_at: dict[str, float] = {}
        self._candidate_side: dict[str, Side] = {}
        self._candidate_readings: dict[str, int] = {}
        self.total_pnl_cents = 0
        self._daily_day: date | None = None
        self._daily_pnl_cents = 0
        self._daily_peak_cents = 0

    @staticmethod
    def decision_seconds(interval_minutes: int) -> float:
        # Observe the first 45 seconds, then monitor continuously until cutoff.
        return max(0.0, interval_minutes * 60 - 45.0)

    @classmethod
    def entry_block_reason(cls, now: float) -> str | None:
        # Five-minute crypto markets trade continuously; there are no clock-based shutdowns.
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
        listing: PolymarketListing,
        target: float,
        now: float,
        ticks: tuple[UnderlyingTick, ...],
    ) -> tuple[Side | None, str]:
        if target <= 0:
            return None, "MISSING_OPENING_REFERENCE"
        if len(ticks) < 2:
            return None, "NO_UNDERLYING_FEED_OR_HISTORY"
        market_ticks = tuple(
            tick
            for tick in ticks
            if listing.open_time <= tick.timestamp.timestamp() <= now
        )
        if len(market_ticks) < 2:
            return None, "NO_MARKET_PRICE_HISTORY"
        latest = market_ticks[-1]
        first_time = market_ticks[0].timestamp.timestamp()
        if now - latest.timestamp.timestamp() > 5 or now - first_time < self.minimum_history:
            return None, "STALE_OR_INSUFFICIENT_PRICE_HISTORY"
        short = self._at_or_before(market_ticks, max(first_time, now - 60))
        long = market_ticks[0]
        short_bps = (latest.price - short.price) / target * 10_000
        long_bps = (latest.price - long.price) / target * 10_000
        separation_bps = (latest.price - target) / target * 10_000
        if abs(separation_bps) < self.minimum_separation_bps:
            return (
                None,
                f"INSUFFICIENT_SEPARATION | separation_bps={separation_bps:+.2f} "
                f"minimum={self.minimum_separation_bps:.2f}",
            )
        if separation_bps * short_bps <= 0 or separation_bps * long_bps <= 0:
            return (
                None,
                f"MOMENTUM_DISAGREEMENT | separation_bps={separation_bps:+.2f} "
                f"momentum_60_bps={short_bps:+.2f} "
                f"momentum_market_bps={long_bps:+.2f}",
            )
        recent_volume = sum(
            tick.size
            for tick in market_ticks
            if tick.timestamp.timestamp() >= max(first_time, now - 60)
        )
        older_volume = sum(
            tick.size
            for tick in market_ticks
            if tick.timestamp.timestamp() < now - 60
        )
        older_seconds = max(1.0, now - first_time - 60)
        baseline_volume = older_volume * 60 / older_seconds
        volume_ratio = recent_volume / baseline_volume if baseline_volume > 0 else 1.0
        if (
            volume_ratio < self.minimum_volume_ratio
            and abs(separation_bps) < self.low_volume_override_bps
        ):
            return (
                None,
                f"LOW_VOLUME_CONFIRMATION | volume_ratio={volume_ratio:.2f} "
                f"minimum={self.minimum_volume_ratio:.2f} "
                f"separation_bps={separation_bps:+.2f} "
                f"override_bps={self.low_volume_override_bps:.2f}",
            )
        volume_weight = max(0.5, min(2.0, volume_ratio))
        score = 0.55 * separation_bps + volume_weight * (
            0.30 * short_bps + 0.15 * long_bps
        )
        # The opening target owns direction. Momentum and volume may confirm or defer,
        # but they may never reverse the side selected by the settlement reference.
        side = Side.YES if separation_bps > 0 else Side.NO
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
        if seconds_left > decision:
            return None
        if seconds_left <= self.final_entry_seconds:
            self.decided.add(listing.slug)
            self._last_retry_reason.pop(listing.slug, None)
            self._snapshot(
                listing,
                seconds_left,
                target,
                "SKIP",
                f"ENTRY_WINDOW_EXPIRED | final_entry_seconds={self.final_entry_seconds:.0f}",
            )
            return None
        if target is None:
            self._snapshot_retryable(
                listing,
                seconds_left,
                None,
                "MISSING_OPENING_REFERENCE",
            )
            return None
        last_evaluation = self._last_evaluation_at.get(listing.slug)
        if last_evaluation is not None and now - last_evaluation < self.evaluation_interval:
            return None
        self._last_evaluation_at[listing.slug] = now
        side, detail = self._signal(listing, target, now, ticks)
        if side is None:
            self._candidate_side.pop(listing.slug, None)
            self._candidate_readings.pop(listing.slug, None)
            self._snapshot_retryable(listing, seconds_left, target, detail)
            return None
        previous_side = self._candidate_side.get(listing.slug)
        readings = (
            self._candidate_readings.get(listing.slug, 0) + 1
            if side is previous_side
            else 1
        )
        self._candidate_side[listing.slug] = side
        self._candidate_readings[listing.slug] = readings
        if readings < self.confirmations:
            self._snapshot_retryable(
                listing,
                seconds_left,
                target,
                f"SIGNAL_CONFIRMING | predicted={side.value} | "
                f"readings={readings}/{self.confirmations} | {detail}",
            )
            return None
        ask = listing.yes_ask if side is Side.YES else listing.no_ask
        if ask is None or not 1 <= ask <= 99:
            self._snapshot_retryable(
                listing,
                seconds_left,
                target,
                "NO_EXECUTABLE_ASK",
            )
            return None
        if ask < self.minimum_entry_price:
            self._snapshot_retryable(
                listing,
                seconds_left,
                target,
                f"ENTRY_PRICE_BELOW_FLOOR | predicted={side.value} | ask={ask}c "
                f"minimum={self.minimum_entry_price}c | {detail}",
            )
            return None
        if ask > self.maximum_entry_price:
            self._snapshot_retryable(
                listing,
                seconds_left,
                target,
                f"ENTRY_PRICE_ABOVE_CAP | predicted={side.value} | ask={ask}c "
                f"maximum={self.maximum_entry_price}c | {detail}",
            )
            return None
        limit_price = min(
            self.maximum_entry_price,
            ask + max(0, self.entry_slippage_cents),
        )
        block = self.entry_block_reason(now)
        if self.drawdown_cents >= self.drawdown_limit_cents:
            block = (
                f"DAILY_DRAWDOWN_LIMIT_CT | drawdown_cents={self.drawdown_cents} "
                f"limit_cents={self.drawdown_limit_cents}"
            )
        if self.reserved_cents() + limit_price * self.contracts > self.bankroll_cents:
            block = "BANKROLL_CAP"
        if block:
            self._snapshot_retryable(
                listing,
                seconds_left,
                target,
                f"{block} | intended_side={side.value} | {detail}",
                decision=f"SHADOW_BUY_{side.value.upper()}",
            )
            return None
        self._last_retry_reason.pop(listing.slug, None)
        self._snapshot(listing, seconds_left, target, f"BUY_{side.value.upper()}", detail)
        return SimSignal(side=side, signal_ask=ask, limit_price=limit_price, detail=detail)

    def _snapshot_retryable(
        self,
        listing: PolymarketListing,
        seconds_left: float,
        target: float | None,
        reason: str,
        *,
        decision: str = "SKIP_RETRYING",
    ) -> None:
        """Record a temporary skip once per reason while leaving the market eligible."""
        reason_code = reason.split(" | ", 1)[0]
        if self._last_retry_reason.get(listing.slug) == reason_code:
            return
        self._last_retry_reason[listing.slug] = reason_code
        self._snapshot(listing, seconds_left, target, decision, reason)

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

    def open_position(
        self,
        listing: PolymarketListing,
        side: Side,
        price: int,
        *,
        count: int | None = None,
        execution_mode: str = "polymarket_paper",
    ) -> None:
        actual_count = self.contracts if count is None else count
        self.decided.add(listing.slug)
        self._last_retry_reason.pop(listing.slug, None)
        self._last_evaluation_at.pop(listing.slug, None)
        self._candidate_side.pop(listing.slug, None)
        self._candidate_readings.pop(listing.slug, None)
        position = SimPosition(side=side, entry_price=price, count=actual_count)
        self.positions[listing.slug] = position
        record_entry(
            ticker=listing.slug,
            side=side.value,
            entry_price=price,
            count=actual_count,
            seconds_left=listing.close_time - datetime.now(UTC).timestamp(),
            take_profit=self.take_profit,
            execution_mode=execution_mode,
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
        self._last_retry_reason = {
            slug: reason
            for slug, reason in self._last_retry_reason.items()
            if slug in keep
        }
        self._last_evaluation_at = {
            slug: evaluated_at
            for slug, evaluated_at in self._last_evaluation_at.items()
            if slug in keep
        }
        self._candidate_side = {
            slug: side for slug, side in self._candidate_side.items() if slug in keep
        }
        self._candidate_readings = {
            slug: readings
            for slug, readings in self._candidate_readings.items()
            if slug in keep
        }
