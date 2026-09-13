"""Settlement-aligned Polymarket five-minute crypto forecast strategy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from statistics import fmean, pstdev

from kalshi_bot.dashboard import record_entry, record_exit, record_model_snapshot
from kalshi_bot.exchange.models import Side
from kalshi_bot.polymarket import PolymarketListing
from kalshi_bot.strategies.base import UnderlyingTick
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)

# Printed by the live engine at startup so deployment logs prove which strategy
# Northflank actually installed.
STRATEGY_VERSION = "poly-5m-chainlink-forecast-v12"


@dataclass(frozen=True, slots=True)
class SimSignal:
    side: Side
    signal_ask: int
    limit_price: int
    detail: str
    maximum_entry_price: int = 65


@dataclass(frozen=True, slots=True)
class SimPosition:
    side: Side
    entry_price: int
    count: int


class PolymarketMomentumStrategy:
    """Forecast the closing Chainlink TWAP from price action and trade that side."""

    def __init__(
        self,
        *,
        contracts: int = 5,
        bankroll_cents: int = 50_000,
        take_profit: int = 98,
        minimum_entry_price: int = 50,
        maximum_entry_price: int = 65,
        minimum_history: float = 45.0,
        confirmations: int = 2,
        confirmation_seconds: float = 2.0,
        evaluation_interval: float = 2.0,
        maximum_tick_age_seconds: float = 12.0,
        maximum_reference_age_seconds: float = 8.0,
        entry_slippage_cents: int = 2,
        final_entry_seconds: float = 60.0,
        minimum_model_edge: float = 0.02,
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
        self.maximum_reference_age_seconds = max(
            self.confirmation_seconds + self.evaluation_interval,
            maximum_reference_age_seconds,
        )
        self.entry_slippage_cents = entry_slippage_cents
        self.final_entry_seconds = max(0.0, final_entry_seconds)
        self.minimum_model_edge = max(0.0, minimum_model_edge)
        self.decided: set[str] = set()
        self.positions: dict[str, SimPosition] = {}
        self._last_retry_reason: dict[str, str] = {}
        self._last_evaluation_at: dict[str, float] = {}
        self._candidate_side: dict[str, Side] = {}
        self._candidate_readings: dict[str, int] = {}
        self._candidate_at: dict[str, float] = {}
        self._candidate_tick_at: dict[str, float] = {}
        self.total_pnl_cents = 0

    @staticmethod
    def decision_seconds(interval_minutes: int) -> float:
        # Observe the first 45 seconds, then monitor continuously until cutoff.
        return max(0.0, interval_minutes * 60 - 45.0)

    @classmethod
    def entry_block_reason(cls, now: float) -> str | None:
        # Five-minute crypto markets trade continuously; there are no clock-based shutdowns.
        return None

    def reserved_cents(self) -> int:
        return sum(p.entry_price * p.count for p in self.positions.values())

    def _dynamic_entry_ceiling(self, model_probability: float) -> int:
        """Highest quote supported by forecast probability after taker fees."""
        affordable = [
            price
            for price in range(self.minimum_entry_price, self.maximum_entry_price + 1)
            if model_probability >= self.fee_adjusted_break_even(price) + self.minimum_model_edge
        ]
        return max(affordable, default=self.minimum_entry_price - 1)

    @staticmethod
    def _at_or_before(ticks: tuple[UnderlyingTick, ...], timestamp: float) -> UnderlyingTick:
        return min(ticks, key=lambda row: abs(row.timestamp.timestamp() - timestamp))

    @classmethod
    def _change_bps(
        cls,
        ticks: tuple[UnderlyingTick, ...],
        now: float,
        seconds: float,
    ) -> float:
        latest = ticks[-1]
        earlier = cls._at_or_before(ticks, now - seconds)
        return (latest.price - earlier.price) / earlier.price * 10_000

    @staticmethod
    def _slope_bps_per_minute(ticks: tuple[UnderlyingTick, ...]) -> float:
        if len(ticks) < 2:
            return 0.0
        origin = ticks[0].timestamp.timestamp()
        times = [(tick.timestamp.timestamp() - origin) / 60 for tick in ticks]
        prices = [tick.price / ticks[0].price * 10_000 for tick in ticks]
        mean_time, mean_price = fmean(times), fmean(prices)
        denominator = sum((value - mean_time) ** 2 for value in times)
        if denominator <= 0:
            return 0.0
        return (
            sum(
                (timestamp - mean_time) * (price - mean_price)
                for timestamp, price in zip(times, prices, strict=True)
            )
            / denominator
        )

    @staticmethod
    def _window(
        ticks: tuple[UnderlyingTick, ...], now: float, seconds: float
    ) -> tuple[UnderlyingTick, ...]:
        rows = tuple(tick for tick in ticks if tick.timestamp.timestamp() >= now - seconds)
        return rows if len(rows) >= 2 else ticks[-2:]

    @staticmethod
    def _weighted_price(ticks: tuple[UnderlyingTick, ...]) -> float:
        positive_size = sum(max(0.0, tick.size) for tick in ticks)
        if positive_size > 0:
            return sum(tick.price * max(0.0, tick.size) for tick in ticks) / positive_size
        return fmean(tick.price for tick in ticks)

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    @staticmethod
    def fee_adjusted_break_even(price_cents: int) -> float:
        """Win probability required for a taker buy held to settlement.

        Polymarket's crypto taker fee is 0.07 * p * (1-p) USDC per share.
        """
        price = price_cents / 100
        return min(1.0, price + 0.07 * price * (1.0 - price))

    def _signal(
        self,
        listing: PolymarketListing,
        target: float,
        now: float,
        ticks: tuple[UnderlyingTick, ...],
        reference_ticks: tuple[UnderlyingTick, ...],
    ) -> tuple[Side | None, str, float]:
        if target <= 0:
            return None, "MISSING_OPENING_REFERENCE", 0.5
        spot = tuple(
            sorted(
                (tick for tick in ticks if tick.timestamp.timestamp() <= now),
                key=lambda x: x.timestamp,
            )
        )
        reference = tuple(
            sorted(
                (tick for tick in reference_ticks if tick.timestamp.timestamp() <= now),
                key=lambda tick: tick.timestamp,
            )
        )
        market_spot = tuple(
            tick for tick in spot if tick.timestamp.timestamp() >= listing.open_time
        )
        market_reference = tuple(
            tick for tick in reference if tick.timestamp.timestamp() >= listing.open_time - 0.5
        )
        if len(spot) < 2 or len(market_spot) < 2:
            return None, "NO_COINBASE_PRICE_HISTORY", 0.5
        if len(reference) < 2 or len(market_reference) < 2:
            return None, "NO_CHAINLINK_TWAP_HISTORY", 0.5
        latest_spot, latest_reference = spot[-1], reference[-1]
        history_seconds = now - market_spot[0].timestamp.timestamp()
        if now - latest_spot.timestamp.timestamp() > self.maximum_tick_age_seconds:
            return None, "STALE_COINBASE_PRICE", 0.5
        if now - latest_reference.timestamp.timestamp() > self.maximum_reference_age_seconds:
            return None, "STALE_CHAINLINK_TWAP", 0.5
        if history_seconds < self.minimum_history:
            return None, "INSUFFICIENT_PRICE_HISTORY", 0.5

        opening_spot = self._at_or_before(spot, listing.open_time)
        reference_separation_bps = (latest_reference.price - target) / target * 10_000
        spot_move_bps = (latest_spot.price - opening_spot.price) / opening_spot.price * 10_000
        reference_10_bps = self._change_bps(reference, now, 10)
        reference_30_bps = self._change_bps(reference, now, 30)
        reference_60_bps = self._change_bps(reference, now, 60)
        momentum_5_bps = self._change_bps(spot, now, 5)
        momentum_15_bps = self._change_bps(spot, now, 15)
        momentum_30_bps = self._change_bps(spot, now, 30)
        momentum_60_bps = self._change_bps(spot, now, 60)
        momentum_120_bps = self._change_bps(spot, now, 120)

        elapsed = now - market_spot[0].timestamp.timestamp()
        volume_window = max(1.0, min(45.0, elapsed / 2))
        recent_start = now - volume_window
        baseline_start = recent_start - volume_window
        recent_volume = sum(
            tick.size for tick in spot if tick.timestamp.timestamp() >= recent_start
        )
        baseline_volume = sum(
            tick.size
            for tick in spot
            if baseline_start <= tick.timestamp.timestamp() < recent_start
        )
        volume_ratio = recent_volume / baseline_volume if baseline_volume > 0 else 1.0
        prices = [tick.price for tick in market_spot]
        path = sum(abs(right - left) for left, right in pairwise(prices))
        efficiency = abs(latest_spot.price - market_spot[0].price) / path if path > 0 else 0.0
        crossings = sum(
            1
            for left, right in pairwise(prices)
            if (left - opening_spot.price) * (right - opening_spot.price) < 0
        )
        market_vwap = self._weighted_price(market_spot)
        vwap_distance_bps = (latest_spot.price - market_vwap) / latest_spot.price * 10_000
        midpoint = listing.open_time + elapsed / 2
        early = tuple(tick for tick in market_spot if tick.timestamp.timestamp() < midpoint)
        late = tuple(tick for tick in market_spot if tick.timestamp.timestamp() >= midpoint)
        early_vwap = self._weighted_price(early) if early else market_spot[0].price
        late_vwap = self._weighted_price(late) if late else latest_spot.price
        vwap_slope_bps = (late_vwap - early_vwap) / early_vwap * 10_000

        recent_ticks = self._window(spot, now, 45)
        signed_volume = 0.0
        total_volume = 0.0
        for left, right in pairwise(recent_ticks):
            size = max(0.0, right.size)
            total_volume += size
            signed_volume += size * (
                1.0 if right.price > left.price else -1.0 if right.price < left.price else 0.0
            )
        volume_imbalance = signed_volume / total_volume if total_volume > 0 else 0.0

        returns_bps = [
            (right.price - left.price) / left.price * 10_000
            for left, right in pairwise(self._window(spot, now, 120))
            if left.price > 0
        ]
        one_second_volatility_bps = pstdev(returns_bps) if len(returns_bps) > 1 else 0.0
        seconds_left = max(1.0, listing.close_time - now)
        expected_noise_bps = max(0.75, one_second_volatility_bps * math.sqrt(seconds_left))

        slope_30 = self._slope_bps_per_minute(self._window(spot, now, 30))
        slope_60 = self._slope_bps_per_minute(self._window(spot, now, 60))
        slope_120 = self._slope_bps_per_minute(self._window(spot, now, 120))
        trend_per_minute = 0.50 * slope_30 + 0.30 * slope_60 + 0.20 * slope_120
        horizon_minutes = min(1.5, seconds_left / 60)
        trend_projection = 0.35 * trend_per_minute * horizon_minutes

        # If a short move opposes an established trend while price remains on
        # the trend side of VWAP, treat it as a retrace rather than a reversal.
        retrace_projection = 0.0
        if slope_120 * slope_30 < 0 and vwap_distance_bps * slope_120 > 0:
            retrace_projection = 0.12 * slope_120 * horizon_minutes

        anchor = 0.72 * reference_separation_bps + 0.28 * spot_move_bps
        vwap_projection = 0.12 * vwap_distance_bps + 0.08 * vwap_slope_bps
        flow_projection = 0.18 * expected_noise_bps * volume_imbalance
        reference_projection = 0.18 * reference_10_bps + 0.10 * reference_30_bps
        raw_projection_bps = (
            anchor
            + trend_projection
            + retrace_projection
            + vwap_projection
            + flow_projection
            + reference_projection
        )
        # Chop is uncertainty, not a veto. It pulls the estimate toward 50/50
        # without preventing a decision on the market.
        chop_factor = max(0.55, 1.0 - 0.05 * crossings)
        efficiency_factor = 0.75 + 0.25 * min(1.0, efficiency / 0.35)
        projected_bps = raw_projection_bps * chop_factor * efficiency_factor
        up_probability = max(
            0.02,
            min(0.98, self._normal_cdf(projected_bps / expected_noise_bps)),
        )
        side = Side.YES if up_probability >= 0.5 else Side.NO
        confidence = 100 * (up_probability if side is Side.YES else 1 - up_probability)
        regime = "TREND"
        if slope_120 * slope_30 < 0:
            regime = "RETRACE" if retrace_projection else "REVERSAL"
        elif efficiency < 0.12:
            regime = "RANGE"
        selected_probability = up_probability if side is Side.YES else 1 - up_probability

        return (
            side,
            (
                f"reference={latest_reference.price:.6f} "
                f"reference_source={latest_reference.source} "
                f"spot={latest_spot.price:.6f} spot_source={latest_spot.source} "
                f"target={target:.6f} predicted={side.value} confidence={confidence:.1f} "
                f"model_up={up_probability * 100:.1f} projected_finish_bps={projected_bps:+.2f} "
                f"expected_noise_bps={expected_noise_bps:.2f} regime={regime} "
                f"reference_separation_bps={reference_separation_bps:+.2f} "
                f"spot_move_bps={spot_move_bps:+.2f} "
                f"reference_10_bps={reference_10_bps:+.2f} "
                f"reference_30_bps={reference_30_bps:+.2f} "
                f"reference_60_bps={reference_60_bps:+.2f} "
                f"momentum_5_bps={momentum_5_bps:+.2f} "
                f"momentum_15_bps={momentum_15_bps:+.2f} "
                f"momentum_30_bps={momentum_30_bps:+.2f} "
                f"momentum_60_bps={momentum_60_bps:+.2f} "
                f"momentum_120_bps={momentum_120_bps:+.2f} "
                f"slope_30={slope_30:+.2f} slope_60={slope_60:+.2f} slope_120={slope_120:+.2f} "
                f"vwap={market_vwap:.6f} vwap_distance_bps={vwap_distance_bps:+.2f} "
                f"vwap_slope_bps={vwap_slope_bps:+.2f} "
                f"volume_ratio={volume_ratio:.2f} volume_imbalance={volume_imbalance:+.2f} "
                f"crossings={crossings} efficiency={efficiency:.2f}"
            ),
            selected_probability,
        )

    def evaluate(
        self,
        listing: PolymarketListing,
        target: float | None,
        now: float,
        ticks: tuple[UnderlyingTick, ...],
        reference_ticks: tuple[UnderlyingTick, ...],
    ) -> SimSignal | None:
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
        side, detail, model_probability = self._signal(listing, target, now, ticks, reference_ticks)
        if side is None:
            self._clear_candidate(listing.slug)
            self._snapshot_retryable(listing, seconds_left, target, detail)
            return None
        market_reference = tuple(
            sorted(
                (
                    tick
                    for tick in reference_ticks
                    if listing.open_time - 0.5 <= tick.timestamp.timestamp() <= now
                ),
                key=lambda tick: tick.timestamp,
            )
        )
        latest = market_reference[-1]
        latest_tick_at = latest.timestamp.timestamp()
        previous_side = self._candidate_side.get(listing.slug)
        if side is not previous_side:
            self._start_candidate(listing.slug, side, now, latest_tick_at)
            readings = 1
        else:
            candidate_at = self._candidate_at[listing.slug]
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
        dynamic_ceiling = self._dynamic_entry_ceiling(model_probability)
        if dynamic_ceiling < self.minimum_entry_price:
            self._snapshot_retryable(
                listing,
                seconds_left,
                target,
                f"NEGATIVE_FEE_ADJUSTED_EDGE | predicted={side.value} | ask={ask}c "
                f"model_probability={model_probability * 100:.1f}% "
                f"minimum_entry={self.minimum_entry_price}c | {detail}",
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
        break_even = self.fee_adjusted_break_even(ask)
        model_edge = model_probability - break_even
        if model_edge < self.minimum_model_edge:
            self._snapshot_retryable(
                listing,
                seconds_left,
                target,
                f"NEGATIVE_FEE_ADJUSTED_EDGE | predicted={side.value} | ask={ask}c "
                f"model_probability={model_probability * 100:.1f}% "
                f"break_even={break_even * 100:.1f}% edge={model_edge * 100:+.1f}% | {detail}",
            )
            return None
        limit_price = min(
            dynamic_ceiling,
            ask + max(0, self.entry_slippage_cents),
        )
        block = self.entry_block_reason(now)
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
        execution_detail = (
            f"{detail} entry_ceiling={dynamic_ceiling}c "
            f"fee_adjusted_break_even={break_even * 100:.1f}% "
            f"model_edge={model_edge * 100:+.1f}%"
        )
        self._snapshot(
            listing,
            seconds_left,
            target,
            f"BUY_{side.value.upper()}",
            execution_detail,
        )
        return SimSignal(
            side=side,
            signal_ask=ask,
            limit_price=limit_price,
            detail=execution_detail,
            maximum_entry_price=dynamic_ceiling,
        )

    def _start_candidate(
        self,
        slug: str,
        side: Side,
        now: float,
        tick_at: float,
    ) -> None:
        self._candidate_side[slug] = side
        self._candidate_readings[slug] = 1
        self._candidate_at[slug] = now
        self._candidate_tick_at[slug] = tick_at

    def _clear_candidate(self, slug: str) -> None:
        self._candidate_side.pop(slug, None)
        self._candidate_readings.pop(slug, None)
        self._candidate_at.pop(slug, None)
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
            slug: reason for slug, reason in self._last_retry_reason.items() if slug in keep
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
            slug: readings for slug, readings in self._candidate_readings.items() if slug in keep
        }
        self._candidate_at = {
            slug: candidate_at for slug, candidate_at in self._candidate_at.items() if slug in keep
        }
        self._candidate_tick_at = {
            slug: tick_at for slug, tick_at in self._candidate_tick_at.items() if slug in keep
        }
