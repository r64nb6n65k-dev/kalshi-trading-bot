"""Settlement-aligned Polymarket five-minute crypto forecast strategy."""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from statistics import fmean, median, pstdev

from kalshi_bot.dashboard import record_entry, record_exit, record_model_snapshot
from kalshi_bot.exchange.models import Side
from kalshi_bot.polymarket import PolymarketListing
from kalshi_bot.strategies.base import UnderlyingTick
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)

# Printed by the live engine at startup so deployment logs prove which strategy
# Northflank actually installed.
STRATEGY_VERSION = "poly-5m-focused-exhaustion-v16-stoploss-exposure-cap"


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
    close_time: float = 0.0


@dataclass(frozen=True, slots=True)
class MarketPulse:
    observed_at: float
    close_time: float
    side: Side
    reference_60_bps: float
    momentum_60_bps: float


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
        signal_memory_size: int = 4,
        signal_memory_interval_seconds: float = 8.0,
        exhaustion_reference_60_bps: float = 4.0,
        maximum_exhaustion_reference_60_bps: float = 5.5,
        market_exhaustion_reference_bps: float = 3.0,
        market_exhaustion_momentum_bps: float = 1.5,
        preferred_entry_start_seconds: float = 210.0,
        preferred_entry_end_seconds: float = 150.0,
        stop_loss_gap_cents: int = 15,
        stop_loss_confirmations: int = 3,
        stop_loss_confirmation_seconds: float = 6.0,
        max_concurrent_correlated: int = 1,
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
        self.signal_memory_size = max(4, signal_memory_size)
        self.signal_memory_interval_seconds = max(2.0, signal_memory_interval_seconds)
        self.exhaustion_reference_60_bps = max(0.5, exhaustion_reference_60_bps)
        self.maximum_exhaustion_reference_60_bps = max(
            self.exhaustion_reference_60_bps,
            maximum_exhaustion_reference_60_bps,
        )
        self.market_exhaustion_reference_bps = max(0.5, market_exhaustion_reference_bps)
        self.market_exhaustion_momentum_bps = max(0.0, market_exhaustion_momentum_bps)
        self.preferred_entry_start_seconds = max(
            preferred_entry_end_seconds,
            preferred_entry_start_seconds,
        )
        self.preferred_entry_end_seconds = max(0.0, preferred_entry_end_seconds)

        # Stop-loss: exit is NOT triggered by a single adverse tick. The bid
        # must stay at or below (entry_price - stop_loss_gap_cents) for
        # `stop_loss_confirmations` separate reads, each read separated by at
        # least `stop_loss_confirmation_seconds`, before we cut the position.
        # This is what protects against the 5-min market jitter that was
        # stopping out trades that would have recovered, while still capping
        # the loss on a genuinely adverse move instead of riding it to zero.
        self.stop_loss_gap_cents = max(1, stop_loss_gap_cents)
        self.stop_loss_confirmations = max(1, stop_loss_confirmations)
        self.stop_loss_confirmation_seconds = max(0.0, stop_loss_confirmation_seconds)

        # Correlated exposure cap: BTC/ETH/SOL/XRP 5-minute markets that
        # settle at the same close_time move together. Without this, "four
        # independent trades" is really one leveraged directional bet wearing
        # four hats. Default of 1 means only one position open per shared
        # settlement window across all assets.
        self.max_concurrent_correlated = max(1, max_concurrent_correlated)

        self.decided: set[str] = set()
        self.positions: dict[str, SimPosition] = {}
        self._stop_streak: dict[str, int] = {}
        self._stop_last_confirmed_at: dict[str, float] = {}
        self._last_retry_reason: dict[str, str] = {}
        self._last_evaluation_at: dict[str, float] = {}
        self._candidate_side: dict[str, Side] = {}
        self._candidate_readings: dict[str, int] = {}
        self._candidate_at: dict[str, float] = {}
        self._candidate_tick_at: dict[str, float] = {}
        self._prediction_memory: dict[str, deque[tuple[float, Side]]] = {}
        self._preferred_side: dict[str, tuple[Side, float]] = {}
        self._contrarian_side: dict[str, tuple[Side, float, str]] = {}
        self._market_pulses: dict[tuple[str, int], MarketPulse] = {}
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
    def _opposite(side: Side) -> Side:
        return Side.NO if side is Side.YES else Side.YES

    def _remember_prediction(self, slug: str, now: float, side: Side) -> tuple[Side, ...]:
        history = self._prediction_memory.setdefault(
            slug,
            deque(maxlen=self.signal_memory_size),
        )
        if (
            not history
            or history[-1][1] is not side
            or now - history[-1][0] >= self.signal_memory_interval_seconds
        ):
            history.append((now, side))
        return tuple(row[1] for row in history)

    def _memory_reversal(self, history: tuple[Side, ...], current: Side) -> bool:
        """Identify the 3-to-1 late direction flip that lost 7 of 10 observed trades."""
        if len(history) < 4:
            return False
        recent = history[-4:]
        counts = Counter(recent)
        return counts[current] == 3 and counts[self._opposite(current)] == 1

    def _market_exhaustion(
        self,
        listing: PolymarketListing,
        now: float,
        current: Side,
    ) -> bool:
        """Detect a synchronized crypto move that is more likely exhausted than confirming."""
        peers = [
            pulse
            for pulse in self._market_pulses.values()
            if abs(pulse.close_time - listing.close_time) < 1.0
            and now - pulse.observed_at <= 6.0
            and pulse.side is current
        ]
        if len(peers) < 3:
            return False
        direction = 1.0 if current is Side.YES else -1.0
        aligned_reference = [direction * pulse.reference_60_bps for pulse in peers]
        aligned_momentum = [direction * pulse.momentum_60_bps for pulse in peers]
        return (
            median(aligned_reference) >= self.market_exhaustion_reference_bps
            and median(aligned_momentum) >= self.market_exhaustion_momentum_bps
        )

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
        raw_up_probability = max(
            0.02,
            min(0.98, self._normal_cdf(projected_bps / expected_noise_bps)),
        )
        raw_side = Side.YES if raw_up_probability >= 0.5 else Side.NO
        raw_selected_probability = (
            raw_up_probability if raw_side is Side.YES else 1 - raw_up_probability
        )
        regime = "TREND"
        if slope_120 * slope_30 < 0:
            regime = "RETRACE" if retrace_projection else "REVERSAL"
        elif efficiency < 0.12:
            regime = "RANGE"

        direction = 1.0 if raw_side is Side.YES else -1.0
        aligned_reference_60 = direction * reference_60_bps
        history = self._remember_prediction(listing.slug, now, raw_side)
        self._market_pulses[(listing.asset, listing.interval_minutes)] = MarketPulse(
            observed_at=now,
            close_time=listing.close_time,
            side=raw_side,
            reference_60_bps=reference_60_bps,
            momentum_60_bps=momentum_60_bps,
        )

        override_reasons: list[str] = []
        if (
            self.exhaustion_reference_60_bps
            <= aligned_reference_60
            <= self.maximum_exhaustion_reference_60_bps
        ):
            override_reasons.append("EXTENDED_REFERENCE_60")
        if self._market_exhaustion(listing, now, raw_side):
            override_reasons.append("SYNCHRONIZED_MARKET_EXHAUSTION")
        if override_reasons and self._memory_reversal(history, raw_side):
            override_reasons.append("MEMORY_CONFIRMED_EXHAUSTION")

        locked_contrarian = self._contrarian_side.get(listing.slug)
        if locked_contrarian is not None:
            side, selected_probability, locked_reason = locked_contrarian
            override_reasons = [f"LOCKED_{locked_reason}"]
        elif override_reasons:
            side = self._opposite(raw_side)
            # Calibrated conservatively below the in-sample reversal rate.  The
            # stronger the overextension and the more independent warnings,
            # the more room the contrarian entry has beneath the 65c cap.
            selected_probability = min(
                0.78,
                max(
                    0.62,
                    0.62
                    + 0.015 * max(0.0, aligned_reference_60 - self.exhaustion_reference_60_bps)
                    + 0.02 * (len(override_reasons) - 1),
                ),
            )
            self._contrarian_side[listing.slug] = (
                side,
                selected_probability,
                ",".join(override_reasons),
            )
        else:
            side = raw_side
            selected_probability = raw_selected_probability

        seconds_left = max(1.0, listing.close_time - now)
        if self.preferred_entry_end_seconds <= seconds_left <= self.preferred_entry_start_seconds:
            self._preferred_side[listing.slug] = (side, selected_probability)

        up_probability = selected_probability if side is Side.YES else 1 - selected_probability
        confidence = 100 * selected_probability

        # Chainlink determines settlement.  Forecasting a side opposite the
        # current Chainlink-to-target position is not an executable signal;
        # keep watching until the settlement reference agrees.
        reference_direction = 1.0 if reference_separation_bps > 0 else -1.0
        predicted_direction = 1.0 if side is Side.YES else -1.0
        if (
            not override_reasons
            and (reference_separation_bps == 0 or reference_direction != predicted_direction)
        ):
            return (
                None,
                (
                    f"REFERENCE_SIDE_DISAGREEMENT | predicted={side.value} "
                    f"reference_separation_bps={reference_separation_bps:+.2f} "
                    f"target={target:.6f} reference={latest_reference.price:.6f}"
                ),
                selected_probability,
            )

        return (
            side,
            (
                f"reference={latest_reference.price:.6f} "
                f"reference_source={latest_reference.source} "
                f"spot={latest_spot.price:.6f} spot_source={latest_spot.source} "
                f"target={target:.6f} predicted={side.value} raw_predicted={raw_side.value} "
                f"prediction_override={','.join(override_reasons) if override_reasons else 'NONE'} "
                f"confidence={confidence:.1f} "
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
        if self.correlated_positions_open(listing) >= self.max_concurrent_correlated:
            block = "CORRELATED_EXPOSURE_CAP"
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
        self._prediction_memory.pop(listing.slug, None)
        self._preferred_side.pop(listing.slug, None)
        self._contrarian_side.pop(listing.slug, None)
        self._stop_streak.pop(listing.slug, None)
        self._stop_last_confirmed_at.pop(listing.slug, None)
        position = SimPosition(
            side=side,
            entry_price=price,
            count=actual_count,
            close_time=listing.close_time,
        )
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
        self._stop_streak.pop(slug, None)
        self._stop_last_confirmed_at.pop(slug, None)
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

    def stop_loss_exit_price(
        self, listing: PolymarketListing, now: float
    ) -> int | None:
        """Return a sell price if a confirmed stop-loss should fire, else None.

        Deliberately NOT a single-tick trigger. A single bad print in a 5-min
        market is often noise, not information. We only cut the position once
        the bid has stayed at/under the stop threshold for
        `stop_loss_confirmations` reads that are each at least
        `stop_loss_confirmation_seconds` apart, so the position has to be
        underwater for a real stretch of wall-clock time before we act.
        """
        position = self.positions.get(listing.slug)
        if position is None:
            return None
        bid = listing.yes_bid if position.side is Side.YES else listing.no_bid
        if bid is None:
            return None
        stop_threshold = max(1, position.entry_price - self.stop_loss_gap_cents)
        if bid > stop_threshold:
            self._stop_streak[listing.slug] = 0
            return None
        last_confirmed = self._stop_last_confirmed_at.get(listing.slug)
        if (
            last_confirmed is not None
            and now - last_confirmed < self.stop_loss_confirmation_seconds
        ):
            # Same adverse stretch, but too soon after the last confirmed
            # read to count as an independent observation.
            return None
        self._stop_last_confirmed_at[listing.slug] = now
        streak = self._stop_streak.get(listing.slug, 0) + 1
        self._stop_streak[listing.slug] = streak
        if streak < self.stop_loss_confirmations:
            return None
        # Confirmed: exit at (or just under) the current bid so the order
        # actually fills, rather than dumping at a fixed floor price.
        return max(1, bid - 1)

    def correlated_positions_open(self, listing: PolymarketListing) -> int:
        """Count open positions settling in the same window as `listing`.

        BTC/ETH/SOL/XRP 5-minute markets sharing a close_time move together;
        this is used to cap how many of them can be open at once.
        """
        return sum(
            1
            for slug, position in self.positions.items()
            if slug != listing.slug and abs(position.close_time - listing.close_time) < 1.0
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
        self._prediction_memory = {
            slug: history for slug, history in self._prediction_memory.items() if slug in keep
        }
        self._preferred_side = {
            slug: preferred for slug, preferred in self._preferred_side.items() if slug in keep
        }
        self._contrarian_side = {
            slug: contrarian for slug, contrarian in self._contrarian_side.items() if slug in keep
        }
        self._stop_streak = {
            slug: streak for slug, streak in self._stop_streak.items() if slug in keep
        }
        self._stop_last_confirmed_at = {
            slug: at for slug, at in self._stop_last_confirmed_at.items() if slug in keep
        }
