

"""Simulation-only port of the live crypto momentum strategy to Polymarket."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from itertools import pairwise
from typing import ClassVar
from zoneinfo import ZoneInfo

from kalshi_bot.dashboard import record_entry, record_exit, record_model_snapshot
from kalshi_bot.exchange.models import Side
from kalshi_bot.polymarket import PolymarketListing
from kalshi_bot.strategies.base import UnderlyingTick
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)

# Printed by the live engine at startup so deployment logs prove which strategy
# Northflank actually installed.
STRATEGY_VERSION = "poly-5m-confidence-v6"


@dataclass(frozen=True, slots=True)
class SimSignal:
    side: Side
    signal_ask: int
    limit_price: int
    detail: str
    maximum_entry_price: int = 90


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
        maximum_entry_price: int = 90,
        minimum_history: float = 20.0,
        minimum_separation_bps: float = 4.0,
        minimum_volume_ratio: float = 0.25,
        low_volume_override_bps: float = 12.0,
        confirmations: int = 2,
        confirmation_seconds: float = 2.0,
        evaluation_interval: float = 2.0,
        maximum_tick_age_seconds: float = 12.0,
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
        self.confirmation_seconds = max(1.0, confirmation_seconds)
        self.evaluation_interval = max(0.25, evaluation_interval)
        # A candidate must remain alive long enough to complete confirmation.
        # The old five-second freshness cutoff conflicted with the six-second
        # confirmation period and repeatedly erased candidates in quiet markets.
        self.maximum_tick_age_seconds = max(
            self.confirmation_seconds + self.evaluation_interval,
            maximum_tick_age_seconds,
        )
        self.entry_slippage_cents = entry_slippage_cents
        self.final_entry_seconds = max(0.0, final_entry_seconds)
        self.drawdown_limit_cents = drawdown_limit_cents
        self.decided: set[str] = set()
        self.positions: dict[str, SimPosition] = {}
        self._last_retry_reason: dict[str, str] = {}
        self._last_evaluation_at: dict[str, float] = {}
        self._candidate_side: dict[str, Side] = {}
        self._candidate_readings: dict[str, int] = {}
        self._candidate_at: dict[str, float] = {}
        self._candidate_price: dict[str, float] = {}
        self._candidate_tick_at: dict[str, float] = {}
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
    def _metric(detail: str, name: str, default: float = 0.0) -> float:
        match = re.search(rf"(?:^|\s){re.escape(name)}=([+-]?\d+(?:\.\d+)?)", detail)
        return float(match.group(1)) if match else default

    def _dynamic_entry_ceiling(self, detail: str) -> int:
        """Use chart confidence to decide how much the strategy may pay."""
        confidence = self._metric(detail, "confidence")
        if confidence >= 72.0:
            ceiling = 90
        elif confidence >= 65.0:
            ceiling = 84
        elif confidence >= 58.0:
            ceiling = 76
        elif confidence >= 50.0:
            ceiling = 65
        else:
            return self.minimum_entry_price - 1
        return min(self.maximum_entry_price, ceiling)

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
            sorted(
                (
                    tick
                    for tick in ticks
                    if listing.open_time <= tick.timestamp.timestamp() <= now
                ),
                key=lambda tick: tick.timestamp,
            )
        )
        if len(market_ticks) < 2:
            return None, "NO_MARKET_PRICE_HISTORY"
        latest = market_ticks[-1]
        first_time = market_ticks[0].timestamp.timestamp()
        if (
            now - latest.timestamp.timestamp() > self.maximum_tick_age_seconds
            or now - first_time < self.minimum_history
        ):
            return None, "STALE_OR_INSUFFICIENT_PRICE_HISTORY"
        very_short = self._at_or_before(market_ticks, max(first_time, now - 15))
        medium = self._at_or_before(market_ticks, max(first_time, now - 30))
        short = self._at_or_before(market_ticks, max(first_time, now - 60))
        long = market_ticks[0]
        very_short_bps = (latest.price - very_short.price) / target * 10_000
        medium_bps = (latest.price - medium.price) / target * 10_000
        short_bps = (latest.price - short.price) / target * 10_000
        long_bps = (latest.price - long.price) / target * 10_000
        separation_bps = (latest.price - target) / target * 10_000
        direction = 1.0 if separation_bps >= 0 else -1.0
        # Compare adjacent equal-duration windows. Missing early volume is neutral,
        # not a reason to throw away the entire opportunity.
        elapsed = now - first_time
        volume_window = min(45.0, elapsed / 2)
        recent_start = now - volume_window
        baseline_start = recent_start - volume_window
        recent_volume = sum(
            tick.size
            for tick in market_ticks
            if tick.timestamp.timestamp() >= recent_start
        )
        baseline_volume = sum(
            tick.size
            for tick in market_ticks
            if baseline_start <= tick.timestamp.timestamp() < recent_start
        )
        volume_ratio = recent_volume / baseline_volume if baseline_volume > 0 else 1.0
        prices = [tick.price for tick in market_ticks]
        path = sum(abs(right - left) for left, right in pairwise(prices))
        efficiency = abs(latest.price - long.price) / path if path > 0 else 0.0
        crossings = sum(
            1
            for left, right in pairwise(prices)
            if (left - target) * (right - target) < 0
        )
        total_size = sum(max(0.0, tick.size) for tick in market_ticks)
        vwap = (
            sum(tick.price * max(0.0, tick.size) for tick in market_ticks) / total_size
            if total_size > 0
            else sum(prices) / len(prices)
        )
        vwap_distance_bps = (latest.price - vwap) / target * 10_000
        midpoint = first_time + (now - first_time) / 2
        early = tuple(
            tick for tick in market_ticks if tick.timestamp.timestamp() < midpoint
        )
        late = tuple(
            tick for tick in market_ticks if tick.timestamp.timestamp() >= midpoint
        )

        def weighted_price(rows: tuple[UnderlyingTick, ...]) -> float:
            size = sum(max(0.0, tick.size) for tick in rows)
            if size > 0:
                return sum(tick.price * max(0.0, tick.size) for tick in rows) / size
            return sum(tick.price for tick in rows) / len(rows)

        early_vwap = weighted_price(early) if early else long.price
        late_vwap = weighted_price(late) if late else latest.price
        vwap_slope_bps = (late_vwap - early_vwap) / target * 10_000
        volume_weight = max(0.5, min(2.0, volume_ratio))
        score = 0.35 * separation_bps + volume_weight * (
            0.25 * very_short_bps
            + 0.20 * medium_bps
            + 0.15 * short_bps
            + 0.05 * long_bps
        )
        alignment = (
            0.34 * math.tanh(direction * separation_bps / 6.0)
            + 0.16 * math.tanh(direction * very_short_bps / 2.0)
            + 0.14 * math.tanh(direction * medium_bps / 3.0)
            + 0.10 * math.tanh(direction * short_bps / 5.0)
            + 0.06 * math.tanh(direction * long_bps / 6.0)
            + 0.10 * math.tanh(direction * vwap_distance_bps / 3.0)
            + 0.10 * math.tanh(direction * vwap_slope_bps / 2.0)
        )
        trend_bonus = 5.0 * min(1.0, efficiency / 0.35)
        volume_bonus = 2.0 * min(1.0, max(0.0, volume_ratio))
        crossing_penalty = min(6.0, crossings * 1.5)
        confidence = max(
            0.0,
            min(90.0, 50.0 + 32.0 * alignment + trend_bonus + volume_bonus - crossing_penalty),
        )
        # The opening target owns direction. Momentum and volume may confirm or defer,
        # but they may never reverse the side selected by the settlement reference.
        side = Side.YES if separation_bps > 0 else Side.NO
        return side, (
            f"underlying={latest.price:.6f} source={latest.source} "
            f"target={target:.6f} score={score:+.2f} confidence={confidence:.1f} "
            f"separation_bps={separation_bps:+.2f} "
            f"momentum_15_bps={very_short_bps:+.2f} "
            f"momentum_30_bps={medium_bps:+.2f} "
            f"momentum_60_bps={short_bps:+.2f} "
            f"momentum_long_bps={long_bps:+.2f} "
            f"vwap={vwap:.6f} vwap_distance_bps={vwap_distance_bps:+.2f} "
            f"vwap_slope_bps={vwap_slope_bps:+.2f} "
            f"volume_ratio={volume_ratio:.2f} crossings={crossings} "
            f"efficiency={efficiency:.2f}"
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
            self._clear_candidate(listing.slug)
            self._snapshot_retryable(listing, seconds_left, target, detail)
            return None
        market_ticks = tuple(
            sorted(
                (
                    tick
                    for tick in ticks
                    if listing.open_time <= tick.timestamp.timestamp() <= now
                ),
                key=lambda tick: tick.timestamp,
            )
        )
        latest = market_ticks[-1]
        latest_tick_at = latest.timestamp.timestamp()
        previous_side = self._candidate_side.get(listing.slug)
        if side is not previous_side:
            self._start_candidate(listing.slug, side, now, latest.price, latest_tick_at)
            readings = 1
        else:
            candidate_at = self._candidate_at[listing.slug]
            candidate_price = self._candidate_price[listing.slug]
            candidate_tick_at = self._candidate_tick_at[listing.slug]
            elapsed_confirmation = now - candidate_at
            if (
                elapsed_confirmation < self.confirmation_seconds
                or latest_tick_at <= candidate_tick_at
            ):
                self._snapshot_retryable(
                    listing,
                    seconds_left,
                    target,
                    f"SIGNAL_CONFIRMING | predicted={side.value} | readings=1/"
                    f"{self.confirmations} | elapsed={elapsed_confirmation:.1f}s/"
                    f"{self.confirmation_seconds:.1f}s | waiting_for_new_tick=true | {detail}",
                )
                return None
            direction = 1.0 if side is Side.YES else -1.0
            confirmation_move_bps = (
                direction * (latest.price - candidate_price) / target * 10_000
            )
            # Flat is acceptable.  Reset only when the selected direction has
            # materially weakened; this keeps trade frequency while preventing
            # a stale opening burst from receiving automatic confirmation.
            if confirmation_move_bps < -0.50:
                self._start_candidate(
                    listing.slug,
                    side,
                    now,
                    latest.price,
                    latest_tick_at,
                )
                self._snapshot_retryable(
                    listing,
                    seconds_left,
                    target,
                    f"SIGNAL_RESET_WEAKENED | predicted={side.value} | "
                    f"candidate_price={candidate_price:.6f} "
                    f"latest_price={latest.price:.6f} "
                    f"confirmation_move_bps={confirmation_move_bps:+.2f} | {detail}",
                )
                return None
            readings = self._candidate_readings.get(listing.slug, 1) + 1
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
        dynamic_ceiling = self._dynamic_entry_ceiling(detail)
        if dynamic_ceiling < self.minimum_entry_price:
            self._snapshot_retryable(
                listing,
                seconds_left,
                target,
                f"LOW_CHART_CONFIDENCE | predicted={side.value} "
                f"confidence={self._metric(detail, 'confidence'):.1f}% minimum=50.0% | {detail}",
            )
            return None
        if ask > dynamic_ceiling:
            self._snapshot_retryable(
                listing,
                seconds_left,
                target,
                f"ENTRY_PRICE_ABOVE_CAP | predicted={side.value} | ask={ask}c "
                f"maximum={dynamic_ceiling}c | {detail}",
            )
            return None
        limit_price = min(
            dynamic_ceiling,
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
        return SimSignal(
            side=side,
            signal_ask=ask,
            limit_price=limit_price,
            detail=f"{detail} entry_ceiling={dynamic_ceiling}c",
            maximum_entry_price=dynamic_ceiling,
        )

    def _start_candidate(
        self,
        slug: str,
        side: Side,
        now: float,
        price: float,
        tick_at: float,
    ) -> None:
        self._candidate_side[slug] = side
        self._candidate_readings[slug] = 1
        self._candidate_at[slug] = now
        self._candidate_price[slug] = price
        self._candidate_tick_at[slug] = tick_at

    def _clear_candidate(self, slug: str) -> None:
        self._candidate_side.pop(slug, None)
        self._candidate_readings.pop(slug, None)
        self._candidate_at.pop(slug, None)
        self._candidate_price.pop(slug, None)
        self._candidate_tick_at.pop(slug, None)

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
        self._clear_candidate(listing.slug)
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
        self._candidate_at = {
            slug: candidate_at
            for slug, candidate_at in self._candidate_at.items()
            if slug in keep
        }
        self._candidate_price = {
            slug: price for slug, price in self._candidate_price.items() if slug in keep
        }
        self._candidate_tick_at = {
            slug: tick_at
            for slug, tick_at in self._candidate_tick_at.items()
            if slug in keep
        }
