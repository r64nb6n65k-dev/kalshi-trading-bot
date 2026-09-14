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
STRATEGY_VERSION = "poly-5m-chainlink-forecast-v11-stop-loss"

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
    def __init__(self, *, contracts: int = 5, bankroll_cents: int = 50_000,
                 take_profit: int = 98, minimum_entry_price: int = 50,
                 maximum_entry_price: int = 65, minimum_history: float = 45.0,
                 confirmations: int = 2, confirmation_seconds: float = 2.0,
                 evaluation_interval: float = 2.0, maximum_tick_age_seconds: float = 12.0,
                 maximum_reference_age_seconds: float = 8.0, entry_slippage_cents: int = 2,
                 final_entry_seconds: float = 60.0, minimum_model_edge: float = 0.0) -> None:
        self.contracts, self.bankroll_cents, self.take_profit = contracts, bankroll_cents, take_profit
        self.minimum_entry_price = max(1, min(99, minimum_entry_price))
        self.maximum_entry_price = max(self.minimum_entry_price, min(99, maximum_entry_price))
        self.minimum_history = minimum_history
        self.confirmations = max(1, confirmations)
        self.confirmation_seconds = max(1.0, confirmation_seconds)
        self.evaluation_interval = max(.25, evaluation_interval)
        self.maximum_tick_age_seconds = max(self.confirmation_seconds + self.evaluation_interval, maximum_tick_age_seconds)
        self.maximum_reference_age_seconds = max(self.confirmation_seconds + self.evaluation_interval, maximum_reference_age_seconds)
        self.entry_slippage_cents, self.final_entry_seconds = entry_slippage_cents, max(0., final_entry_seconds)
        self.minimum_model_edge = max(0., minimum_model_edge)
        self.decided: set[str] = set(); self.positions: dict[str, SimPosition] = {}
        self._last_retry_reason: dict[str, str] = {}; self._last_evaluation_at: dict[str, float] = {}
        self._candidate_side: dict[str, Side] = {}; self._candidate_readings: dict[str, int] = {}
        self._candidate_at: dict[str, float] = {}; self._candidate_tick_at: dict[str, float] = {}
        self.total_pnl_cents = 0
        # Only retained behavior from the current strategy: confirmed entry-relative stop.
        self.stop_loss_gap_cents = 15; self.stop_loss_confirmations = 3
        self.stop_loss_confirmation_seconds = 6.0
        self._stop_streak: dict[str, int] = {}; self._stop_last_confirmed_at: dict[str, float] = {}

    @staticmethod
    def decision_seconds(interval_minutes: int) -> float:
        return max(0., interval_minutes * 60 - 45.)

    @classmethod
    def entry_block_reason(cls, now: float) -> str | None:
        return None

    def reserved_cents(self) -> int:
        return sum(p.entry_price * p.count for p in self.positions.values())

    def _dynamic_entry_ceiling(self, probability: float) -> int:
        prices = [p for p in range(self.minimum_entry_price, self.maximum_entry_price + 1)
                  if probability >= self.fee_adjusted_break_even(p) + self.minimum_model_edge]
        return max(prices, default=self.minimum_entry_price - 1)

    @staticmethod
    def _at_or_before(ticks: tuple[UnderlyingTick, ...], timestamp: float) -> UnderlyingTick:
        return min(ticks, key=lambda x: abs(x.timestamp.timestamp() - timestamp))

    @classmethod
    def _change_bps(cls, ticks: tuple[UnderlyingTick, ...], now: float, seconds: float) -> float:
        latest = ticks[-1]; earlier = cls._at_or_before(ticks, now - seconds)
        return (latest.price - earlier.price) / earlier.price * 10_000

    @staticmethod
    def _slope_bps_per_minute(ticks: tuple[UnderlyingTick, ...]) -> float:
        if len(ticks) < 2: return 0.
        origin = ticks[0].timestamp.timestamp()
        ts = [(x.timestamp.timestamp() - origin) / 60 for x in ticks]
        ps = [x.price / ticks[0].price * 10_000 for x in ticks]
        mt, mp = fmean(ts), fmean(ps); den = sum((x - mt) ** 2 for x in ts)
        return 0. if den <= 0 else sum((t - mt) * (p - mp) for t, p in zip(ts, ps, strict=True)) / den

    @staticmethod
    def _window(ticks: tuple[UnderlyingTick, ...], now: float, seconds: float) -> tuple[UnderlyingTick, ...]:
        rows = tuple(x for x in ticks if x.timestamp.timestamp() >= now - seconds)
        return rows if len(rows) >= 2 else ticks[-2:]

    @staticmethod
    def _weighted_price(ticks: tuple[UnderlyingTick, ...]) -> float:
        size = sum(max(0., x.size) for x in ticks)
        return sum(x.price * max(0., x.size) for x in ticks) / size if size else fmean(x.price for x in ticks)

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return .5 * (1. + math.erf(value / math.sqrt(2.)))

    @staticmethod
    def fee_adjusted_break_even(price_cents: int) -> float:
        p = price_cents / 100
        return min(1., p + .07 * p * (1. - p))

    def _signal(self, listing: PolymarketListing, target: float, now: float,
                ticks: tuple[UnderlyingTick, ...], refs: tuple[UnderlyingTick, ...]) -> tuple[Side | None, str, float]:
        if target <= 0: return None, "MISSING_OPENING_REFERENCE", .5
        spot = tuple(sorted((x for x in ticks if x.timestamp.timestamp() <= now), key=lambda x: x.timestamp))
        reference = tuple(sorted((x for x in refs if x.timestamp.timestamp() <= now), key=lambda x: x.timestamp))
        ms = tuple(x for x in spot if x.timestamp.timestamp() >= listing.open_time)
        mr = tuple(x for x in reference if x.timestamp.timestamp() >= listing.open_time - .5)
        if len(spot) < 2 or len(ms) < 2: return None, "NO_COINBASE_PRICE_HISTORY", .5
        if len(reference) < 2 or len(mr) < 2: return None, "NO_CHAINLINK_TWAP_HISTORY", .5
        ls, lr = spot[-1], reference[-1]; history = now - ms[0].timestamp.timestamp()
        if now - ls.timestamp.timestamp() > self.maximum_tick_age_seconds: return None, "STALE_COINBASE_PRICE", .5
        if now - lr.timestamp.timestamp() > self.maximum_reference_age_seconds: return None, "STALE_CHAINLINK_TWAP", .5
        if history < self.minimum_history: return None, "INSUFFICIENT_PRICE_HISTORY", .5
        opening = self._at_or_before(spot, listing.open_time)
        sep = (lr.price - target) / target * 10_000; move = (ls.price - opening.price) / opening.price * 10_000
        r10, r30, r60 = (self._change_bps(reference, now, n) for n in (10, 30, 60))
        m5, m15, m30, m60, m120 = (self._change_bps(spot, now, n) for n in (5, 15, 30, 60, 120))
        elapsed = now - ms[0].timestamp.timestamp(); vw = max(1., min(45., elapsed / 2)); recent = now - vw
        rv = sum(x.size for x in spot if x.timestamp.timestamp() >= recent)
        bv = sum(x.size for x in spot if recent - vw <= x.timestamp.timestamp() < recent)
        volume_ratio = rv / bv if bv > 0 else 1.; prices = [x.price for x in ms]
        path = sum(abs(b - a) for a, b in pairwise(prices)); efficiency = abs(ls.price - ms[0].price) / path if path else 0.
        crossings = sum(1 for a, b in pairwise(prices) if (a - opening.price) * (b - opening.price) < 0)
        vwap = self._weighted_price(ms); vd = (ls.price - vwap) / ls.price * 10_000
        midpoint = listing.open_time + elapsed / 2
        early = tuple(x for x in ms if x.timestamp.timestamp() < midpoint); late = tuple(x for x in ms if x.timestamp.timestamp() >= midpoint)
        ev = self._weighted_price(early) if early else ms[0].price; lv = self._weighted_price(late) if late else ls.price
        vws = (lv - ev) / ev * 10_000
        recent_ticks = self._window(spot, now, 45); signed = total = 0.
        for a, b in pairwise(recent_ticks):
            s = max(0., b.size); total += s; signed += s * (1. if b.price > a.price else -1. if b.price < a.price else 0.)
        imbalance = signed / total if total else 0.
        returns = [(b.price - a.price) / a.price * 10_000 for a, b in pairwise(self._window(spot, now, 120)) if a.price > 0]
        noise = max(.75, (pstdev(returns) if len(returns) > 1 else 0.) * math.sqrt(max(1., listing.close_time - now)))
        s30, s60, s120 = (self._slope_bps_per_minute(self._window(spot, now, n)) for n in (30, 60, 120))
        horizon = min(1.5, max(1., listing.close_time - now) / 60); trend = (.5 * s30 + .3 * s60 + .2 * s120) * horizon * .35
        retrace = .12 * s120 * horizon if s120 * s30 < 0 and vd * s120 > 0 else 0.
        raw = .72 * sep + .28 * move + trend + retrace + .12 * vd + .08 * vws + .18 * noise * imbalance + .18 * r10 + .10 * r30
        projected = raw * max(.55, 1. - .05 * crossings) * (.75 + .25 * min(1., efficiency / .35))
        up = max(.02, min(.98, self._normal_cdf(projected / noise))); side = Side.YES if up >= .5 else Side.NO
        confidence = 100 * (up if side is Side.YES else 1 - up); regime = "TREND"
        if s120 * s30 < 0: regime = "RETRACE" if retrace else "REVERSAL"
        elif efficiency < .12: regime = "RANGE"
        selected = up if side is Side.YES else 1 - up
        detail = (f"reference={lr.price:.6f} reference_source={lr.source} spot={ls.price:.6f} spot_source={ls.source} target={target:.6f} "
                  f"predicted={side.value} confidence={confidence:.1f} model_up={up*100:.1f} projected_finish_bps={projected:+.2f} expected_noise_bps={noise:.2f} regime={regime} "
                  f"reference_separation_bps={sep:+.2f} spot_move_bps={move:+.2f} reference_10_bps={r10:+.2f} reference_30_bps={r30:+.2f} reference_60_bps={r60:+.2f} "
                  f"momentum_5_bps={m5:+.2f} momentum_15_bps={m15:+.2f} momentum_30_bps={m30:+.2f} momentum_60_bps={m60:+.2f} momentum_120_bps={m120:+.2f} "
                  f"slope_30={s30:+.2f} slope_60={s60:+.2f} slope_120={s120:+.2f} vwap={vwap:.6f} vwap_distance_bps={vd:+.2f} vwap_slope_bps={vws:+.2f} volume_ratio={volume_ratio:.2f} volume_imbalance={imbalance:+.2f} crossings={crossings} efficiency={efficiency:.2f}")
        return side, detail, selected

    def evaluate(self, listing: PolymarketListing, target: float | None, now: float,
                 ticks: tuple[UnderlyingTick, ...], reference_ticks: tuple[UnderlyingTick, ...]) -> SimSignal | None:
        if listing.slug in self.decided or listing.slug in self.positions: return None
        left = listing.close_time - now; decision = self.decision_seconds(listing.interval_minutes)
        if left > decision: return None
        if left <= self.final_entry_seconds:
            self.decided.add(listing.slug); self._last_retry_reason.pop(listing.slug, None)
            self._snapshot(listing, left, target, "SKIP", f"ENTRY_WINDOW_EXPIRED | final_entry_seconds={self.final_entry_seconds:.0f}"); return None
        if target is None: self._snapshot_retryable(listing, left, None, "MISSING_OPENING_REFERENCE"); return None
        last = self._last_evaluation_at.get(listing.slug)
        if last is not None and now - last < self.evaluation_interval: return None
        self._last_evaluation_at[listing.slug] = now
        side, detail, probability = self._signal(listing, target, now, ticks, reference_ticks)
        if side is None: self._clear_candidate(listing.slug); self._snapshot_retryable(listing, left, target, detail); return None
        mr = tuple(sorted((x for x in reference_ticks if listing.open_time - .5 <= x.timestamp.timestamp() <= now), key=lambda x: x.timestamp))
        if not mr: self._snapshot_retryable(listing, left, target, "NO_CHAINLINK_TWAP_HISTORY"); return None
        tick_at = mr[-1].timestamp.timestamp(); previous = self._candidate_side.get(listing.slug)
        if side is not previous:
            self._start_candidate(listing.slug, side, now, tick_at); readings = 1
        else:
            elapsed = now - self._candidate_at[listing.slug]
            if elapsed < self.confirmation_seconds or tick_at <= self._candidate_tick_at[listing.slug]:
                self._snapshot_retryable(listing, left, target, f"SIGNAL_CONFIRMING | predicted={side.value} | readings=1/{self.confirmations} | elapsed={elapsed:.1f}s/{self.confirmation_seconds:.1f}s | waiting_for_new_tick=true | {detail}"); return None
            readings = self._candidate_readings.get(listing.slug, 1) + 1; self._candidate_readings[listing.slug] = readings
        if readings < self.confirmations:
            self._snapshot_retryable(listing, left, target, f"SIGNAL_CONFIRMING | predicted={side.value} | readings={readings}/{self.confirmations} | {detail}"); return None
        ask = listing.yes_ask if side is Side.YES else listing.no_ask
        if ask is None or not 1 <= ask <= 99: self._snapshot_retryable(listing, left, target, "NO_EXECUTABLE_ASK"); return None
        if ask < self.minimum_entry_price:
            self._snapshot_retryable(listing, left, target, f"ENTRY_PRICE_BELOW_FLOOR | predicted={side.value} | ask={ask}c minimum={self.minimum_entry_price}c | {detail}"); return None
        ceiling = self._dynamic_entry_ceiling(probability)
        if ceiling < self.minimum_entry_price:
            self._snapshot_retryable(listing, left, target, f"NEGATIVE_FEE_ADJUSTED_EDGE | predicted={side.value} | ask={ask}c model_probability={probability*100:.1f}% minimum_entry={self.minimum_entry_price}c | {detail}"); return None
        if ask > ceiling:
            self._snapshot_retryable(listing, left, target, f"ENTRY_PRICE_ABOVE_CAP | predicted={side.value} | ask={ask}c maximum={ceiling}c | {detail}"); return None
        be = self.fee_adjusted_break_even(ask); edge = probability - be
        if edge < self.minimum_model_edge:
            self._snapshot_retryable(listing, left, target, f"NEGATIVE_FEE_ADJUSTED_EDGE | predicted={side.value} | ask={ask}c model_probability={probability*100:.1f}% break_even={be*100:.1f}% edge={edge*100:+.1f}% | {detail}"); return None
        limit = min(ceiling, ask + max(0, self.entry_slippage_cents)); block = self.entry_block_reason(now)
        if self.reserved_cents() + limit * self.contracts > self.bankroll_cents: block = "BANKROLL_CAP"
        if block:
            self._snapshot_retryable(listing, left, target, f"{block} | intended_side={side.value} | {detail}", decision=f"SHADOW_BUY_{side.value.upper()}"); return None
        self._last_retry_reason.pop(listing.slug, None); execution = f"{detail} entry_ceiling={ceiling}c fee_adjusted_break_even={be*100:.1f}% model_edge={edge*100:+.1f}%"
        self._snapshot(listing, left, target, f"BUY_{side.value.upper()}", execution)
        return SimSignal(side, ask, limit, execution, ceiling)

    def _start_candidate(self, slug: str, side: Side, now: float, tick_at: float) -> None:
        self._candidate_side[slug] = side; self._candidate_readings[slug] = 1; self._candidate_at[slug] = now; self._candidate_tick_at[slug] = tick_at

    def _clear_candidate(self, slug: str) -> None:
        for mapping in (self._candidate_side, self._candidate_readings, self._candidate_at, self._candidate_tick_at): mapping.pop(slug, None)

    def _snapshot_retryable(self, listing: PolymarketListing, seconds_left: float, target: float | None, reason: str, *, decision: str = "SKIP_RETRYING") -> None:
        code = reason.split(" | ", 1)[0]
        if self._last_retry_reason.get(listing.slug) == code: return
        self._last_retry_reason[listing.slug] = code; self._snapshot(listing, seconds_left, target, decision, reason)

    def _snapshot(self, listing: PolymarketListing, seconds_left: float, target: float | None, decision: str, reason: str) -> None:
        record_model_snapshot(ticker=listing.slug, seconds_left=seconds_left, target_price=target, yes_bid=listing.yes_bid, yes_ask=listing.yes_ask, no_bid=listing.no_bid, no_ask=listing.no_ask, decision=decision, reason=f"market_source={listing.source} | {reason}")
        logger.warning("%s | ticker=%s | %s", decision, listing.slug, reason)

    def open_position(self, listing: PolymarketListing, side: Side, price: int, *, count: int | None = None, execution_mode: str = "polymarket_paper") -> None:
        count = self.contracts if count is None else count; self.decided.add(listing.slug); self._last_retry_reason.pop(listing.slug, None); self._last_evaluation_at.pop(listing.slug, None); self._clear_candidate(listing.slug)
        self.positions[listing.slug] = SimPosition(side, price, count)
        record_entry(ticker=listing.slug, side=side.value, entry_price=price, count=count, seconds_left=listing.close_time - datetime.now(UTC).timestamp(), take_profit=self.take_profit, execution_mode=execution_mode)

    def close_position(self, slug: str, exit_price: int, reason: str, now: float) -> None:
        position = self.positions.pop(slug, None); self._stop_streak.pop(slug, None); self._stop_last_confirmed_at.pop(slug, None)
        if position is None: return
        pnl = (exit_price - position.entry_price) * position.count; self.total_pnl_cents += pnl
        record_exit(ticker=slug, side=position.side.value, entry_price=position.entry_price, exit_price=exit_price, reason=reason, count=position.count, pnl_cents=pnl, total_pnl_cents=self.total_pnl_cents)

    def log_open_position(self, listing: PolymarketListing, target: float | None, now: float) -> None:
        p = self.positions.get(listing.slug)
        if p is None: return
        bid = listing.yes_bid if p.side is Side.YES else listing.no_bid; threshold = max(1, p.entry_price - self.stop_loss_gap_cents)
        record_model_snapshot(ticker=listing.slug, seconds_left=max(0., listing.close_time-now), target_price=target, yes_bid=listing.yes_bid, yes_ask=listing.yes_ask, no_bid=listing.no_bid, no_ask=listing.no_ask, decision="HOLD_OPEN_POSITION", reason=f"market_source={listing.source} | side={p.side.value} entry={p.entry_price}c current_bid={bid}c count={p.count} stop_threshold={threshold}c stop_streak={self._stop_streak.get(listing.slug, 0)}/{self.stop_loss_confirmations} take_profit={self.take_profit}c")

    def stop_loss_exit_price(self, listing: PolymarketListing, now: float) -> int | None:
        p = self.positions.get(listing.slug)
        if p is None: return None
        bid = listing.yes_bid if p.side is Side.YES else listing.no_bid
        if bid is None: return None
        threshold = max(1, p.entry_price - self.stop_loss_gap_cents)
        if bid > threshold: self._stop_streak[listing.slug] = 0; return None
        last = self._stop_last_confirmed_at.get(listing.slug)
        if last is not None and now - last < self.stop_loss_confirmation_seconds: return None
        self._stop_last_confirmed_at[listing.slug] = now; streak = self._stop_streak.get(listing.slug, 0) + 1; self._stop_streak[listing.slug] = streak
        return max(1, bid - 1) if streak >= self.stop_loss_confirmations else None

    def prune(self, keep: set[str]) -> None:
        self.decided.intersection_update(keep | set(self.positions))
        for mapping in (self._last_retry_reason, self._last_evaluation_at, self._candidate_side, self._candidate_readings, self._candidate_at, self._candidate_tick_at, self._stop_streak, self._stop_last_confirmed_at):
            for slug in list(mapping):
                if slug not in keep: mapping.pop(slug, None)
