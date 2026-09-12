

"""Chart-driven direction strategy for Polymarket rolling crypto markets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
import math
import statistics
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
    """Predict settlement direction from the underlying chart, not market odds."""

    _CENTRAL: ClassVar[ZoneInfo] = ZoneInfo("America/Chicago")

    def __init__(
        self,
        *,
        contracts: int = 5,
        bankroll_cents: int = 50_000,
        take_profit: int = 98,
        maximum_entry_price: int = 76,
        minimum_entry_price: int = 50,
        minimum_history: float = 20.0,
        minimum_confidence: float = 0.58,
        late_minimum_confidence: float = 0.65,
        early_window_seconds: float = 60.0,
        confirmations: int = 2,
        evaluation_interval: float = 2.0,
        entry_slippage_cents: int = 2,
        final_entry_seconds: float = 60.0,
        drawdown_limit_cents: int = 4_000,
    ) -> None:
        self.contracts = contracts
        self.bankroll_cents = bankroll_cents
        self.take_profit = take_profit
        self.maximum_entry_price = max(1, min(99, maximum_entry_price))
        self.minimum_entry_price = max(
            1,
            min(self.maximum_entry_price, minimum_entry_price),
        )
        self.minimum_history = minimum_history
        self.minimum_confidence = max(0.50, min(0.99, minimum_confidence))
        self.late_minimum_confidence = max(
            self.minimum_confidence,
            min(0.99, late_minimum_confidence),
        )
        self.early_window_seconds = max(0.0, early_window_seconds)
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
        # Observe the first 25 seconds, then evaluate through the final cutoff.
        # A 5m market therefore becomes eligible with 4:35 remaining.
        return max(0.0, interval_minutes * 60 - 25.0)

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

    @staticmethod
    def _bounded(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _squash(value: float, scale: float) -> float:
        """Map a chart measurement to -1..+1 without one spike dominating."""
        return math.tanh(value / max(1e-9, scale))

    @staticmethod
    def _vwap(ticks: tuple[UnderlyingTick, ...]) -> float:
        total_size = sum(max(0.0, tick.size) for tick in ticks)
        if total_size <= 0:
            return sum(tick.price for tick in ticks) / len(ticks)
        return sum(tick.price * max(0.0, tick.size) for tick in ticks) / total_size

    @staticmethod
    def _trend_bps(
        ticks: tuple[UnderlyingTick, ...], target: float, projection_seconds: float = 30.0
    ) -> float:
        """Least-squares chart slope, projected forward and expressed in bps."""
        if len(ticks) < 2:
            return 0.0
        origin = ticks[0].timestamp.timestamp()
        xs = [tick.timestamp.timestamp() - origin for tick in ticks]
        ys = [tick.price for tick in ticks]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denominator = sum((value - mean_x) ** 2 for value in xs)
        if denominator <= 0:
            return 0.0
        slope = sum(
            (x_value - mean_x) * (y_value - mean_y)
            for x_value, y_value in zip(xs, ys, strict=True)
        ) / denominator
        return slope * projection_seconds / target * 10_000

    @staticmethod
    def _target_crossings(ticks: tuple[UnderlyingTick, ...], target: float) -> int:
        sides: list[int] = []
        for tick in ticks:
            side = 1 if tick.price > target else -1 if tick.price < target else 0
            if side and (not sides or side != sides[-1]):
                sides.append(side)
        return max(0, len(sides) - 1)

    def _signal(
        self,
        listing: PolymarketListing,
        target: float,
        now: float,
        ticks: tuple[UnderlyingTick, ...],
    ) -> tuple[Side | None, float, str]:
        if target <= 0:
            return None, 0.50, "MISSING_OPENING_REFERENCE"
        if len(ticks) < 2:
            return None, 0.50, "NO_UNDERLYING_FEED_OR_HISTORY"

        market_ticks = tuple(
            tick
            for tick in ticks
            if listing.open_time <= tick.timestamp.timestamp() <= now
        )
        if len(market_ticks) < 2:
            return None, 0.50, "NO_MARKET_CHART_HISTORY"
        latest = market_ticks[-1]
        first_time = market_ticks[0].timestamp.timestamp()
        observed_seconds = now - first_time
        if now - latest.timestamp.timestamp() > 5 or observed_seconds < self.minimum_history:
            return None, 0.50, "STALE_OR_INSUFFICIENT_PRICE_HISTORY"

        def return_bps(seconds: float) -> float:
            earlier = self._at_or_before(market_ticks, max(first_time, now - seconds))
            return (latest.price - earlier.price) / target * 10_000

        momentum_15_bps = return_bps(15)
        momentum_30_bps = return_bps(30)
        momentum_60_bps = return_bps(60)
        separation_bps = (latest.price - target) / target * 10_000

        one_second_returns = [
            (current.price - previous.price)
            / target
            * 10_000
            / math.sqrt(
                max(
                    0.001,
                    current.timestamp.timestamp() - previous.timestamp.timestamp(),
                )
            )
            for previous, current in zip(market_ticks, market_ticks[1:])
        ]
        sigma_bps = (
            statistics.pstdev(one_second_returns)
            if len(one_second_returns) >= 3
            else abs(momentum_15_bps) / math.sqrt(max(1.0, observed_seconds))
        )
        sigma_bps = max(0.10, sigma_bps)
        seconds_left = max(1.0, listing.close_time - now)
        expected_remaining_bps = max(1.0, sigma_bps * math.sqrt(seconds_left))

        chart_60 = tuple(
            tick for tick in market_ticks if tick.timestamp.timestamp() >= now - 60
        )
        trend_bps = self._trend_bps(chart_60, target)
        market_vwap = self._vwap(market_ticks)
        vwap_distance_bps = (latest.price - market_vwap) / target * 10_000
        midpoint = listing.open_time + (now - listing.open_time) / 2
        early_ticks = tuple(
            tick for tick in market_ticks if tick.timestamp.timestamp() <= midpoint
        )
        late_ticks = tuple(
            tick for tick in market_ticks if tick.timestamp.timestamp() > midpoint
        )
        vwap_slope_bps = 0.0
        if early_ticks and late_ticks:
            vwap_slope_bps = (
                (self._vwap(late_ticks) - self._vwap(early_ticks)) / target * 10_000
            )

        recent_start = max(listing.open_time, now - 15)
        recent_volume = sum(
            tick.size for tick in market_ticks if tick.timestamp.timestamp() >= recent_start
        )
        older_volume = sum(
            tick.size for tick in market_ticks if tick.timestamp.timestamp() < recent_start
        )
        recent_seconds = max(1.0, now - recent_start)
        older_seconds = max(0.0, recent_start - listing.open_time)
        if older_volume > 0 and older_seconds >= 5:
            volume_ratio = (recent_volume / recent_seconds) / (older_volume / older_seconds)
        else:
            volume_ratio = 1.0

        distance_evidence = self._squash(separation_bps, expected_remaining_bps)
        momentum_evidence = (
            self._squash(momentum_15_bps, sigma_bps * math.sqrt(15))
            + self._squash(momentum_30_bps, sigma_bps * math.sqrt(30))
            + self._squash(momentum_60_bps, sigma_bps * math.sqrt(60))
        ) / 3
        trend_evidence = self._squash(trend_bps, sigma_bps * math.sqrt(30))
        vwap_evidence = (
            self._squash(
                vwap_distance_bps,
                sigma_bps * math.sqrt(max(5.0, observed_seconds)),
            )
            + self._squash(
                vwap_slope_bps,
                sigma_bps * math.sqrt(max(5.0, observed_seconds)),
            )
        ) / 2

        progress = self._bounded(
            (now - listing.open_time) / max(1.0, listing.interval_minutes * 60),
            0.0,
            1.0,
        )
        distance_weight = 0.42 + 0.28 * progress
        momentum_weight = 0.28 - 0.10 * progress
        trend_weight = 0.18 - 0.08 * progress
        vwap_weight = 1.0 - distance_weight - momentum_weight - trend_weight
        raw_score = (
            distance_weight * distance_evidence
            + momentum_weight * momentum_evidence
            + trend_weight * trend_evidence
            + vwap_weight * vwap_evidence
        )

        crossings = self._target_crossings(market_ticks, target)
        travel = sum(
            abs(current.price - previous.price)
            for previous, current in zip(market_ticks, market_ticks[1:])
        )
        efficiency = (
            abs(latest.price - market_ticks[0].price) / travel if travel > 0 else 0.0
        )
        chart_quality = (0.65 + 0.35 * efficiency) / (1.0 + 0.12 * crossings)
        volume_boost = self._bounded(
            1.0 + 0.08 * math.log2(max(0.25, volume_ratio)),
            0.85,
            1.15,
        )
        score = self._bounded(raw_score * chart_quality * volume_boost, -1.0, 1.0)
        confidence = 0.50 + 0.50 * abs(score)
        if abs(score) < 1e-9:
            return None, confidence, "NO_DIRECTIONAL_CHART_EDGE"

        side = Side.YES if score >= 0 else Side.NO
        return side, confidence, (
            f"underlying={latest.price:.6f} source={latest.source} "
            f"target={target:.6f} chart_score={score:+.3f} "
            f"confidence={confidence:.1%} "
            f"separation_bps={separation_bps:+.2f} "
            f"momentum_15_bps={momentum_15_bps:+.2f} "
            f"momentum_30_bps={momentum_30_bps:+.2f} "
            f"momentum_60_bps={momentum_60_bps:+.2f} "
            f"trend_bps={trend_bps:+.2f} "
            f"vwap={market_vwap:.6f} vwap_distance_bps={vwap_distance_bps:+.2f} "
            f"vwap_slope_bps={vwap_slope_bps:+.2f} "
            f"volume_ratio={volume_ratio:.2f} recent_volume={recent_volume:.4f} "
            f"volatility_bps={sigma_bps:.3f} crossings={crossings} "
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
        last_evaluation = self._last_evaluation_at.get(listing.slug)
        if last_evaluation is not None and now - last_evaluation < self.evaluation_interval:
            return None
        self._last_evaluation_at[listing.slug] = now
        if target is None:
            self._snapshot_retryable(
                listing,
                seconds_left,
                None,
                "MISSING_OPENING_REFERENCE",
            )
            return None
        side, confidence, detail = self._signal(listing, target, now, ticks)
        if side is None:
            self._snapshot_retryable(listing, seconds_left, target, detail)
            return None
        elapsed_seconds = max(0.0, now - listing.open_time)
        required_confidence = (
            self.minimum_confidence
            if elapsed_seconds <= self.early_window_seconds
            else self.late_minimum_confidence
        )
        if confidence < required_confidence:
            self._candidate_side.pop(listing.slug, None)
            self._candidate_readings.pop(listing.slug, None)
            self._snapshot_retryable(
                listing,
                seconds_left,
                target,
                f"LOW_CHART_CONFIDENCE | confidence={confidence:.1%} "
                f"minimum={required_confidence:.1%} | elapsed_seconds={elapsed_seconds:.0f} "
                f"| predicted={side.value} | {detail}",
            )
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
                f"CHART_CONFIRMING | predicted={side.value} | readings={readings}/"
                f"{self.confirmations} | {detail}",
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
        block: str | None = None
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
