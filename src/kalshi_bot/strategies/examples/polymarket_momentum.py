"""Probability/edge strategy for Polymarket rolling crypto markets."""

from __future__ import annotations

import math
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

STRATEGY_VERSION = "poly-5m-chainlink-price-confidence-v4-stop-loss"


@dataclass(frozen=True, slots=True)
class SimSignal:
    side: Side
    signal_ask: int
    limit_price: int
    detail: str
    model_probability: float
    required_edge_cents: float
    required_probability: float
    maximum_entry_price: int


@dataclass(frozen=True, slots=True)
class SimPosition:
    side: Side
    entry_price: int
    count: int


class PolymarketMomentumStrategy:
    """Predict settlement from Chainlink TWAP, volatility and momentum, then buy edge."""

    _CENTRAL: ClassVar[ZoneInfo] = ZoneInfo("America/Chicago")
    # Previously blocked 00:00-02:00, 08:00-10:00, and 19:00-20:00 Central
    # (5 hours/day). Removed: no data showed those hours performed worse,
    # so the block was just cutting volume for no proven reason. Runs 24/7
    # now; revisit if overnight/off-peak hours turn out to actually be bad
    # once there's real data on them.
    _NO_ENTRY_WINDOWS: ClassVar[tuple[tuple[int, int], ...]] = ()

    def __init__(
        self,
        *,
        contracts: int = 10,
        bankroll_cents: int = 50_000,
        take_profit: int = 98,
        minimum_history: float = 45.0,
        minimum_separation_bps: float = 0.0,
        minimum_entry_price: int = 1,
        maximum_entry_price: int = 85,
        entry_slippage_cents: int = 2,
        decision_window: float = 15.0,
        final_entry_seconds: float = 45.0,
        drawdown_limit_cents: int = 4_000,
        minimum_model_probability: float = 0.55,
        base_required_edge_cents: float = 0.0,
        signal_confirmations: int = 2,
        signal_confirmation_seconds: float = 2.0,
        max_probability_deterioration: float = 0.08,
        minimum_probability_margin: float = 0.10,
        minimum_confirmed_entry_seconds: float = 0.0,
    ) -> None:
        self.contracts = contracts
        self.bankroll_cents = bankroll_cents
        self.take_profit = take_profit
        self.minimum_history = minimum_history
        # Retained only for constructor compatibility. Probability/volatility now
        # replaces the old fixed 4-bps entry gate.
        self.minimum_separation_bps = max(0.0, minimum_separation_bps)
        self.entry_slippage_cents = entry_slippage_cents
        self.decision_window = decision_window
        self.final_entry_seconds = max(0.0, final_entry_seconds)
        self.drawdown_limit_cents = drawdown_limit_cents

        # There is no longer a 50-75c strategy band. These are only technical
        # exchange/economic bounds. The model edge determines the usable price.
        self.minimum_entry_price = max(1, min(99, minimum_entry_price))
        economic_max = max(self.minimum_entry_price, min(99, self.take_profit - 1))
        self.maximum_entry_price = max(
            self.minimum_entry_price,
            min(economic_max, maximum_entry_price),
        )

        self.minimum_model_probability = min(
            0.95, max(0.50, minimum_model_probability)
        )
        self.base_required_edge_cents = max(0.0, base_required_edge_cents)
        self.signal_confirmations = max(1, signal_confirmations)
        self.signal_confirmation_seconds = max(0.0, signal_confirmation_seconds)
        self.max_probability_deterioration = max(0.0, max_probability_deterioration)
        # Live data shows model confidence barely above the required bar is
        # close to a coin flip (~50% win rate), while confidence clearing the
        # bar by 10+ points wins ~71%. Raised from 0.08 -> 0.10: whatever the
        # cutoff, trades that barely clear it underperform (that cohort just
        # moves with the cutoff), so 0.08 still had a coin-flip band right
        # above it. 0.10 is the point where win rate and trade volume both
        # hold up -- going higher (0.15+) trims volume faster than it adds
        # win rate and nets less total profit despite a higher win rate.
        self.minimum_probability_margin = max(0.0, min(0.49, minimum_probability_margin))
        # Live data: signals that confirm quickly (early in the decision
        # window, 170-200s still left) win ~74% of the time. Signals that
        # take a long time to confirm (under 110s left when they finally
        # do) win only ~33% -- a slow confirmation isn't a signal that
        # "eventually got good," it's usually a weak/noisy setup that kept
        # resetting until it limped through once. Require a real
        # confirmation within the first part of the window; abandon slow
        # ones rather than keep retrying them.
        self.minimum_confirmed_entry_seconds = max(0.0, minimum_confirmed_entry_seconds)

        self.evaluation_interval = 1.0
        self.decided: set[str] = set()
        self.positions: dict[str, SimPosition] = {}
        self._last_retry_reason: dict[str, str] = {}
        self.total_pnl_cents = 0
        self._daily_day: date | None = None
        self._daily_pnl_cents = 0
        self._daily_peak_cents = 0

        self._signal_side: dict[str, Side] = {}
        self._signal_streak: dict[str, int] = {}
        self._signal_last_confirmed_at: dict[str, float] = {}
        self._signal_probability: dict[str, float] = {}

        # Stop-loss: require two confirmations spaced ~1s apart instead of
        # three spaced 6s apart (previously confirmations=3, seconds=6.0).
        # A single-confirmation stop (confirmations=1) reacted to one-tick
        # noise and closed positions that would have recovered above the
        # threshold. Two quick confirmations filter that noise out while
        # still exiting in ~1-2s instead of the original 12-18s, which was
        # letting price keep falling during the confirmation window (avg
        # ~19c extra slippage beyond the intended 15c threshold, worst case
        # ~56c observed in live trading).
        self.stop_loss_gap_cents = 15
        self.stop_loss_confirmations = 2
        self.stop_loss_confirmation_seconds = 1.0
        self._stop_streak: dict[str, int] = {}
        self._stop_last_confirmed_at: dict[str, float] = {}

        # Trailing stop removed. It tested well on a looser trade pool
        # (0.08 margin, +$0.28/+21%), but retested against the pool that's
        # actually live now (0.10 margin) it loses money: $1.91-2.14 with
        # trailing vs $2.48 without, across every gap tested. The tighter
        # filter selects cleaner trades that mostly run straight to 98
        # without needing protection along the way -- so trailing just
        # clips profit here instead of saving reversals. Flat 15c stop
        # from entry only, same as the base mechanism always had.

    @staticmethod
    def decision_seconds(interval_minutes: int) -> float:
        # Five-minute markets begin evaluating at 3:20 remaining.
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
    def _at_or_before(
        ticks: tuple[UnderlyingTick, ...], timestamp: float
    ) -> UnderlyingTick:
        return min(ticks, key=lambda row: abs(row.timestamp.timestamp() - timestamp))

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    @staticmethod
    def _persistence_ratio(
        ticks: tuple[UnderlyingTick, ...],
        now: float,
        direction: int,
        window_seconds: float = 45.0,
    ) -> float:
        recent = [
            tick
            for tick in ticks
            if tick.timestamp.timestamp() >= now - window_seconds
        ]
        if len(recent) < 3:
            return 0.50
        aligned = 0
        moves = 0
        for previous, current in zip(recent, recent[1:]):
            delta = current.price - previous.price
            if delta == 0:
                continue
            moves += 1
            if delta * direction > 0:
                aligned += 1
        return aligned / moves if moves else 0.50

    @staticmethod
    def _realized_volatility_bps_per_sqrt_second(
        ticks: tuple[UnderlyingTick, ...],
        now: float,
        window_seconds: float = 90.0,
    ) -> float:
        """RMS log-return volatility normalized to one sqrt-second."""
        recent = [
            tick
            for tick in ticks
            if tick.timestamp.timestamp() >= now - window_seconds
        ]
        if len(recent) < 4:
            return 0.75

        normalized_sq: list[float] = []
        for previous, current in zip(recent, recent[1:]):
            dt = current.timestamp.timestamp() - previous.timestamp.timestamp()
            if dt <= 0 or previous.price <= 0 or current.price <= 0:
                continue
            ret_bps = math.log(current.price / previous.price) * 10_000
            normalized_sq.append((ret_bps / math.sqrt(dt)) ** 2)
        if not normalized_sq:
            return 0.75

        sigma = math.sqrt(sum(normalized_sq) / len(normalized_sq))
        # Avoid false certainty during an unusually quiet few seconds, and cap
        # corrupted/noisy bursts from making every market untradeable.
        return max(0.45, min(4.0, sigma))

    @staticmethod
    def _volume_ratio(
        ticks: tuple[UnderlyingTick, ...], now: float
    ) -> float:
        if not ticks:
            return 1.0
        first_time = ticks[0].timestamp.timestamp()
        recent_volume = sum(
            tick.size for tick in ticks if tick.timestamp.timestamp() >= now - 60
        )
        older_volume = sum(
            tick.size for tick in ticks if tick.timestamp.timestamp() < now - 60
        )
        older_seconds = max(1.0, now - first_time - 60)
        baseline = older_volume * 60 / older_seconds
        if baseline <= 0:
            return 1.0
        return max(0.25, min(4.0, recent_volume / baseline))

    def _required_probability(self, entry_price: int) -> float:
        """Directional-confidence gate from the 2026-09-15 trade replay.

        Normal entries at 70c+ require p_side >= 0.74. Cheaper entries are
        allowed only when the model is substantially stronger (>= 0.80);
        those sub-70c entries must also have positive model edge.
        """
        if entry_price < 70:
            return max(self.minimum_model_probability, 0.80)
        return max(self.minimum_model_probability, 0.74)

    def _required_edge_cents(self, entry_price: int) -> float:
        """Compatibility helper: edge is logged, but no longer gates entry."""
        return 0.0

    def _maximum_price_for_probability(self, probability: float) -> int:
        max_price = 0
        for price in range(self.minimum_entry_price, self.maximum_entry_price + 1):
            if probability >= self._required_probability(price):
                max_price = price
        return max_price

    def _probability(
        self,
        target: float,
        now: float,
        seconds_left: float,
        ticks: tuple[UnderlyingTick, ...],
        reference_ticks: tuple[UnderlyingTick, ...],
    ) -> tuple[float | None, str]:
        if target <= 0:
            return None, "MISSING_OPENING_REFERENCE"
        if not reference_ticks:
            return None, "NO_CHAINLINK_TWAP_HISTORY"

        latest_ref = reference_ticks[-1]
        if now - latest_ref.timestamp.timestamp() > 15:
            return None, "STALE_CHAINLINK_TWAP"

        ref_first_time = reference_ticks[0].timestamp.timestamp()
        spot_fresh = bool(ticks) and now - ticks[-1].timestamp.timestamp() <= 5
        spot_history_ok = len(ticks) >= 2
        history_start = min(
            ref_first_time,
            ticks[0].timestamp.timestamp() if ticks else ref_first_time,
        )
        if now - history_start < self.minimum_history:
            return None, "INSUFFICIENT_PRICE_HISTORY"

        separation_bps = (latest_ref.price - target) / target * 10_000

        if len(reference_ticks) >= 2:
            ref_60 = self._at_or_before(reference_ticks, now - 60)
            twap_momentum_60 = (latest_ref.price - ref_60.price) / target * 10_000
        else:
            twap_momentum_60 = 0.0

        if spot_history_ok and spot_fresh:
            latest_spot = ticks[-1]
            spot_30 = self._at_or_before(ticks, now - 30)
            spot_60 = self._at_or_before(ticks, now - 60)
            spot_momentum_30 = (latest_spot.price - spot_30.price) / target * 10_000
            spot_momentum_60 = (latest_spot.price - spot_60.price) / target * 10_000
            sigma_source = ticks
            volume_ratio = self._volume_ratio(ticks, now)
            spot_price = latest_spot.price
            spot_source = latest_spot.source
            spot_status = "FRESH"
            confidence_shrink = 0.85
        else:
            # Chainlink determines settlement, so a temporarily stale Coinbase
            # helper feed should not kill an otherwise valid setup. Fall back to
            # Chainlink-only direction/volatility and shrink confidence toward 50%.
            spot_momentum_30 = 0.0
            spot_momentum_60 = 0.0
            sigma_source = reference_ticks if len(reference_ticks) >= 4 else ticks
            volume_ratio = 1.0
            spot_price = ticks[-1].price if ticks else float("nan")
            spot_source = ticks[-1].source if ticks else "UNAVAILABLE"
            spot_status = "STALE_FALLBACK_CHAINLINK" if ticks else "MISSING_FALLBACK_CHAINLINK"
            confidence_shrink = 0.65

        sigma = self._realized_volatility_bps_per_sqrt_second(
            sigma_source if sigma_source else reference_ticks, now
        )

        # Chainlink is the settlement feed and therefore gets most of the drift
        # weight. Coinbase only refines the estimate when it is actually fresh.
        if spot_fresh and spot_history_ok:
            raw_drift_per_second = (
                0.60 * (twap_momentum_60 / 60.0)
                + 0.25 * (spot_momentum_30 / 30.0)
                + 0.15 * (spot_momentum_60 / 60.0)
            )
            volume_multiplier = max(0.80, min(1.20, 0.90 + 0.10 * volume_ratio))
        else:
            raw_drift_per_second = twap_momentum_60 / 60.0
            volume_multiplier = 1.0

        drift_horizon = min(max(0.0, seconds_left), 60.0)
        drift_adjustment = raw_drift_per_second * drift_horizon * 0.40 * volume_multiplier
        drift_adjustment = max(-8.0, min(8.0, drift_adjustment))

        forecast_separation_bps = separation_bps + drift_adjustment
        remaining_sigma_bps = sigma * math.sqrt(max(1.0, seconds_left))
        z_score = forecast_separation_bps / max(1.0, remaining_sigma_bps)
        raw_yes_probability = self._normal_cdf(z_score)

        direction = 1 if forecast_separation_bps >= 0 else -1
        persistence_source = ticks if spot_fresh and len(ticks) >= 3 else reference_ticks
        persistence = self._persistence_ratio(persistence_source, now, direction)
        persistence_adjustment = direction * (persistence - 0.50) * 0.08
        raw_yes_probability = max(
            0.01, min(0.99, raw_yes_probability + persistence_adjustment)
        )

        yes_probability = 0.50 + (raw_yes_probability - 0.50) * confidence_shrink
        yes_probability = max(0.05, min(0.95, yes_probability))

        detail = (
            f"chainlink_twap={latest_ref.price:.6f} target={target:.6f} "
            f"chainlink_sep_bps={separation_bps:+.2f} forecast_sep_bps={forecast_separation_bps:+.2f} "
            f"spot={spot_price:.6f} spot_source={spot_source} spot_status={spot_status} "
            f"spot_mom_30_bps={spot_momentum_30:+.2f} spot_mom_60_bps={spot_momentum_60:+.2f} "
            f"twap_mom_60_bps={twap_momentum_60:+.2f} sigma_bps_sqrt_s={sigma:.3f} "
            f"remaining_sigma_bps={remaining_sigma_bps:.2f} drift_adj_bps={drift_adjustment:+.2f} "
            f"persistence={persistence:.2f} volume_ratio={volume_ratio:.2f} "
            f"confidence_shrink={confidence_shrink:.2f} p_yes={yes_probability:.3f}"
        )
        return yes_probability, detail

    def _confirm_probability(
        self,
        slug: str,
        side: Side,
        probability: float,
        now: float,
    ) -> tuple[bool, int, str | None]:
        previous_side = self._signal_side.get(slug)
        previous_probability = self._signal_probability.get(slug)

        if previous_side is not side:
            self._signal_side[slug] = side
            self._signal_streak[slug] = 1
            self._signal_last_confirmed_at[slug] = now
            self._signal_probability[slug] = probability
            return self.signal_confirmations <= 1, 1, None

        if (
            previous_probability is not None
            and probability + self.max_probability_deterioration < previous_probability
        ):
            self._signal_streak[slug] = 1
            self._signal_last_confirmed_at[slug] = now
            self._signal_probability[slug] = probability
            return False, 1, (
                f"PROBABILITY_DETERIORATING | previous={previous_probability:.3f} "
                f"current={probability:.3f}"
            )

        last = self._signal_last_confirmed_at.get(slug)
        if last is not None and now - last < self.signal_confirmation_seconds:
            return False, self._signal_streak.get(slug, 1), None

        self._signal_last_confirmed_at[slug] = now
        self._signal_probability[slug] = probability
        streak = self._signal_streak.get(slug, 0) + 1
        self._signal_streak[slug] = streak
        return streak >= self.signal_confirmations, streak, None

    def evaluate(
        self,
        listing: PolymarketListing,
        target: float | None,
        now: float,
        ticks: tuple[UnderlyingTick, ...],
        reference_ticks: tuple[UnderlyingTick, ...] | None = None,
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
            self._clear_signal_confirmation(listing.slug)
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
                listing, seconds_left, None, "MISSING_OPENING_REFERENCE"
            )
            return None

        yes_probability, detail = self._probability(
            target,
            now,
            seconds_left,
            ticks,
            reference_ticks or (),
        )
        if yes_probability is None:
            self._clear_signal_confirmation(listing.slug)
            self._snapshot_retryable(listing, seconds_left, target, detail)
            return None

        side = Side.YES if yes_probability >= 0.50 else Side.NO
        model_probability = (
            yes_probability if side is Side.YES else 1.0 - yes_probability
        )
        ask = listing.yes_ask if side is Side.YES else listing.no_ask
        if ask is None or not 1 <= ask <= 99:
            self._snapshot_retryable(
                listing, seconds_left, target, "NO_EXECUTABLE_ASK"
            )
            return None
        if ask < self.minimum_entry_price or ask > self.maximum_entry_price:
            self._snapshot_retryable(
                listing,
                seconds_left,
                target,
                (
                    f"TECHNICAL_PRICE_BOUND | predicted={side.value} ask={ask}c "
                    f"bounds={self.minimum_entry_price}-{self.maximum_entry_price}c"
                ),
            )
            return None

        required_probability = self._required_probability(ask)
        # The replay thresholds are absolute p_side gates. Do not stack the
        # older probability-margin rule on top of them or 74% would silently
        # become 84%.
        required_with_margin = required_probability
        edge_cents = model_probability * 100.0 - ask
        maximum_entry_price = self._maximum_price_for_probability(model_probability)
        low_price_edge_failed = ask < 70 and edge_cents <= 0.0
        if (
            model_probability < required_probability
            or low_price_edge_failed
            or maximum_entry_price < self.minimum_entry_price
        ):
            self._snapshot_retryable(
                listing,
                seconds_left,
                target,
                (
                    f"PRICE_ADJUSTED_CONFIDENCE_TOO_LOW | predicted={side.value} ask={ask}c "
                    f"p_side={model_probability:.3f} required_p={required_probability:.3f} "
                    f"required_with_margin={required_with_margin:.3f} "
                    f"market_edge={edge_cents:+.2f}c low_price_positive_edge_required={ask < 70} "
                    f"max_confidence_price={maximum_entry_price}c | {detail}"
                ),
            )
            return None

        confirmed, streak, confirmation_reason = self._confirm_probability(
            listing.slug, side, model_probability, now
        )
        if not confirmed:
            reason = confirmation_reason or "PROBABILITY_CONFIRMING"
            self._snapshot_retryable(
                listing,
                seconds_left,
                target,
                (
                    f"{reason} | predicted={side.value} p_side={model_probability:.3f} "
                    f"market_edge={edge_cents:+.2f}c required_p={required_probability:.3f} streak={streak}/{self.signal_confirmations} | {detail}"
                ),
            )
            return None

        if seconds_left < self.minimum_confirmed_entry_seconds:
            # Confirmed, but took too long to get there -- data shows these
            # are the weak setups, not late bloomers. Abandon rather than
            # keep retrying; more time passing only pushes it later.
            self.decided.add(listing.slug)
            self._clear_signal_confirmation(listing.slug)
            self._snapshot(
                listing,
                seconds_left,
                target,
                "SKIP",
                (
                    f"CONFIRMATION_TOO_SLOW | predicted={side.value} p_side={model_probability:.3f} "
                    f"seconds_left={seconds_left:.0f} minimum_confirmed_entry_seconds="
                    f"{self.minimum_confirmed_entry_seconds:.0f} | {detail}"
                ),
            )
            return None

        limit_price = min(maximum_entry_price, ask + self.entry_slippage_cents)
        if limit_price < ask:
            self._snapshot_retryable(
                listing,
                seconds_left,
                target,
                "MODEL_EDGE_MOVED_BELOW_EXECUTABLE_PRICE",
            )
            return None

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
                (
                    f"{block} | intended_side={side.value} p_side={model_probability:.3f} "
                    f"market_edge={edge_cents:+.2f}c required_p={required_probability:.3f} | {detail}"
                ),
                decision=f"SHADOW_BUY_{side.value.upper()}",
            )
            return None

        self._last_retry_reason.pop(listing.slug, None)
        self._snapshot(
            listing,
            seconds_left,
            target,
            f"BUY_{side.value.upper()}",
            (
                f"p_side={model_probability:.3f} ask={ask}c market_edge={edge_cents:+.2f}c "
                f"required_p={required_probability:.3f} required_with_margin={required_with_margin:.3f} "
                f"max_confidence_price={maximum_entry_price}c "
                f"probability_confirmed={streak}/{self.signal_confirmations} | {detail}"
            ),
        )
        return SimSignal(
            side=side,
            signal_ask=ask,
            limit_price=limit_price,
            detail=detail,
            model_probability=model_probability,
            required_edge_cents=0.0,
            required_probability=required_probability,
            maximum_entry_price=maximum_entry_price,
        )

    def _clear_signal_confirmation(self, slug: str) -> None:
        self._signal_side.pop(slug, None)
        self._signal_streak.pop(slug, None)
        self._signal_last_confirmed_at.pop(slug, None)
        self._signal_probability.pop(slug, None)

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
        self._clear_signal_confirmation(listing.slug)
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

    def log_open_position(
        self, listing: PolymarketListing, target: float | None, now: float
    ) -> None:
        position = self.positions.get(listing.slug)
        if position is None:
            return
        bid = listing.yes_bid if position.side is Side.YES else listing.no_bid
        threshold = max(1, position.entry_price - self.stop_loss_gap_cents)
        record_model_snapshot(
            ticker=listing.slug,
            seconds_left=max(0.0, listing.close_time - now),
            target_price=target,
            yes_bid=listing.yes_bid,
            yes_ask=listing.yes_ask,
            no_bid=listing.no_bid,
            no_ask=listing.no_ask,
            decision="HOLD_OPEN_POSITION",
            reason=(
                f"market_source={listing.source} | side={position.side.value} "
                f"entry={position.entry_price}c current_bid={bid}c count={position.count} "
                f"stop_threshold={threshold}c "
                f"stop_streak={self._stop_streak.get(listing.slug, 0)}"
                f"/{self.stop_loss_confirmations} take_profit={self.take_profit}c"
            ),
        )

    def stop_loss_exit_price(
        self, listing: PolymarketListing, now: float
    ) -> int | None:
        """Trigger the retained 15-cent stop after two quick confirmations."""
        position = self.positions.get(listing.slug)
        if position is None:
            return None
        bid = listing.yes_bid if position.side is Side.YES else listing.no_bid
        ask = listing.yes_ask if position.side is Side.YES else listing.no_ask
        # A vanished bid (no resting buy orders) is not "no information" --
        # in a fast-resolving market it usually means the book has cleared
        # out because the outcome is becoming obvious against us. Previously
        # a missing bid caused this method to return early without even
        # resetting the streak, freezing the stop indefinitely and letting
        # the position ride unprotected all the way to settlement. Fall back
        # to the ask (still live even when the bid disappears) so the stop
        # keeps evaluating instead of going inert.
        effective_price = bid if bid is not None else ask
        if effective_price is None:
            return None

        threshold = max(1, position.entry_price - self.stop_loss_gap_cents)

        if effective_price > threshold:
            self._stop_streak[listing.slug] = 0
            return None
        last = self._stop_last_confirmed_at.get(listing.slug)
        if last is not None and now - last < self.stop_loss_confirmation_seconds:
            return None
        self._stop_last_confirmed_at[listing.slug] = now
        streak = self._stop_streak.get(listing.slug, 0) + 1
        self._stop_streak[listing.slug] = streak
        if streak < self.stop_loss_confirmations:
            return None
        # If we only have an ask (no bid to sell into), exit at that price
        # rather than subtracting a cent we have no basis for.
        return max(1, effective_price - 1) if bid is not None else max(1, effective_price)

    def close_position(self, slug: str, exit_price: int, reason: str, now: float) -> None:
        position = self.positions.pop(slug, None)
        self._stop_streak.pop(slug, None)
        self._stop_last_confirmed_at.pop(slug, None)
        self._clear_signal_confirmation(slug)
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
        self._signal_side = {
            slug: side for slug, side in self._signal_side.items() if slug in keep
        }
        self._signal_streak = {
            slug: streak for slug, streak in self._signal_streak.items() if slug in keep
        }
        self._signal_last_confirmed_at = {
            slug: at
            for slug, at in self._signal_last_confirmed_at.items()
            if slug in keep
        }
        self._signal_probability = {
            slug: probability
            for slug, probability in self._signal_probability.items()
            if slug in keep
        }
        self._stop_streak = {
            slug: streak for slug, streak in self._stop_streak.items() if slug in keep
        }
        self._stop_last_confirmed_at = {
            slug: at
            for slug, at in self._stop_last_confirmed_at.items()
            if slug in keep
        }
