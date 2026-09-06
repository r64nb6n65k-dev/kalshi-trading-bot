"""Paper-only multi-market crypto portfolio engine.

The engine scans open Kalshi crypto markets for complementary and strike-ladder
arbitrage.  BTC/ETH 15-minute directional trades use a confirmed contract-price
rebound and conservative resting-limit paper fills.  Hard bankroll, liquidity,
fee and settlement checks always remain authoritative.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal
from itertools import pairwise
from pathlib import Path
from typing import Any

from kalshi_bot.dashboard import record_entry, record_exit, record_model_snapshot
from kalshi_bot.exchange.client import KalshiClient
from kalshi_bot.exchange.models import Market, Side
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)

CRYPTO_TICKER_TOKENS = {
    "BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "AVAX", "ADA", "LTC",
    "BCH", "LINK", "DOT", "MATIC",
}
CRYPTO_NAMES = (
    "BITCOIN", "ETHEREUM", "SOLANA", "DOGECOIN", "CARDANO", "LITECOIN",
    "BITCOIN CASH", "CHAINLINK", "POLKADOT", "POLYGON", "CRYPTOCURRENCY",
)
UPPER_WORDS = ("ABOVE", "GREATER", "HIGHER", "AT LEAST", "OVER")
SCALP_SERIES = ("KXBTC15M", "KXETH15M")


def fee_cents(price: int, count: int, coefficient: float = 0.07) -> int:
    """Return a conservative Kalshi-style fee estimate, rounded up."""
    probability = Decimal(price) / Decimal(100)
    dollars = Decimal(str(coefficient)) * Decimal(count) * probability * (1 - probability)
    return int((dollars * 100).to_integral_value(rounding=ROUND_CEILING))


def _ticker_text(market: Market) -> str:
    return " ".join(filter(None, (market.series_ticker, market.event_ticker, market.ticker))).upper()


def is_crypto_market(market: Market) -> bool:
    """Identify crypto without matching generic prose words such as LINK or DOT."""
    ticker_text = _ticker_text(market)
    compact = re.sub(r"[^A-Z0-9]", "", ticker_text)
    if any(series in compact for series in SCALP_SERIES):
        return True
    tokens = set(re.findall(r"[A-Z]+", ticker_text))
    if tokens & CRYPTO_TICKER_TOKENS:
        return True
    if any(re.search(rf"KX{token}(?:\d|$)", compact) for token in CRYPTO_TICKER_TOKENS):
        return True
    prose = " ".join(filter(None, (market.title, market.subtitle))).upper()
    return any(re.search(rf"\b{re.escape(name)}\b", prose) for name in CRYPTO_NAMES)


def is_upper_threshold(market: Market) -> bool:
    if market.floor_strike is None or market.cap_strike is not None:
        return False
    text = " ".join(filter(None, (market.title, market.subtitle, market.rules_primary))).upper()
    return market.strike_type in {None, "greater", "greater_or_equal"} and any(
        word in text for word in UPPER_WORDS
    )


def is_scalp_market(market: Market) -> bool:
    compact = re.sub(r"[^A-Z0-9]", "", _ticker_text(market))
    return any(series in compact for series in SCALP_SERIES)


@dataclass
class PaperPosition:
    ticker: str
    side: Side
    entry: int
    count: int
    opened: datetime
    take_profit: int
    stop: int
    module: str
    score_key: str
    fee_paid: int


@dataclass
class PendingMaker:
    ticker: str
    side: Side
    price: int
    count: int
    created: datetime
    expires: datetime
    score: float
    score_key: str
    crossed_samples: int = 0


@dataclass
class LockedBundle:
    key: str
    module: str
    side: str
    unit_cost: int
    count: int
    total_cost: int
    gross_payout: int
    release: datetime


class AdaptiveScorer:
    """Small explainable learner; it may rank setups but cannot waive hard rules."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.stats: dict[str, dict[str, float]] = {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self.stats = raw
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

    def score(self, key: str, structural_edge: float, fill_quality: float) -> float:
        stat = self.stats.get(key, {"wins": 1.0, "losses": 1.0, "pnl": 0.0, "n": 0.0})
        win_rate = stat["wins"] / max(1.0, stat["wins"] + stat["losses"])
        confidence = min(1.0, stat["n"] / 50.0)
        structural = max(0.0, min(1.0, structural_edge / 10.0))
        quality = max(0.0, min(1.0, fill_quality))
        learned = 0.5 + confidence * (win_rate - 0.5)
        return max(0.0, min(5.0, 5.0 * (0.45 * structural + 0.35 * quality + 0.20 * learned)))

    def samples(self, key: str) -> int:
        return int(self.stats.get(key, {}).get("n", 0.0))

    def update(self, key: str, pnl_cents: int) -> None:
        stat = self.stats.setdefault(key, {"wins": 1.0, "losses": 1.0, "pnl": 0.0, "n": 0.0})
        stat["wins" if pnl_cents > 0 else "losses"] += 1
        stat["pnl"] += pnl_cents
        stat["n"] += 1
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.stats, sort_keys=True), encoding="utf-8")
        except OSError:
            logger.exception("Could not persist adaptive scorer state")


class MultiCryptoPortfolioEngine:
    """Scan and paper-trade safely identifiable open crypto markets."""

    def __init__(self, client: KalshiClient, bankroll_cents: int = 50_000, poll_interval: float = 2.0) -> None:
        self.client = client
        self.starting_bankroll = bankroll_cents
        self.cash = bankroll_cents
        self.realized_pnl = 0
        self.poll_interval = max(0.5, poll_interval)
        self.reserve = 20_000
        self.max_deployed = 30_000
        self.max_positions = 6
        self.max_per_trade = 5_000
        self.daily_loss_limit = 5_000
        self.positions: dict[str, PaperPosition] = {}
        self.locked: dict[str, LockedBundle] = {}
        self.pending: dict[str, PendingMaker] = {}
        self.cooldown: dict[str, datetime] = {}
        self.history: dict[str, deque[tuple[datetime, float]]] = defaultdict(lambda: deque(maxlen=300))
        self.rejection_totals: Counter[str] = Counter()
        volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
        self.scorer = AdaptiveScorer(Path(volume or "/tmp") / "crypto_adaptive_score.json")
        self._running = False

    @property
    def deployed(self) -> int:
        positions = sum(p.entry * p.count + p.fee_paid for p in self.positions.values())
        return positions + sum(bundle.total_cost for bundle in self.locked.values())

    def available(self) -> int:
        return max(0, min(self.cash - self.reserve, self.max_deployed - self.deployed))

    @staticmethod
    def _valid_book(m: Market) -> bool:
        values = (m.yes_bid, m.yes_ask, m.no_bid, m.no_ask)
        return all(v is not None and 1 <= int(v) <= 99 for v in values)

    @staticmethod
    def _seconds_left(m: Market, now: datetime) -> float | None:
        close = MultiCryptoPortfolioEngine._close_time(m)
        return None if close is None else (close - now).total_seconds()

    @staticmethod
    def _close_time(m: Market) -> datetime | None:
        if not m.close_time:
            return None
        try:
            close = datetime.fromisoformat(m.close_time.replace("Z", "+00:00"))
            return close.replace(tzinfo=UTC) if close.tzinfo is None else close.astimezone(UTC)
        except (TypeError, ValueError):
            return None

    def _count_for(self, unit_cost: int, cap: int = 50) -> int:
        if unit_cost <= 0:
            return 0
        return max(0, min(cap, self.max_per_trade // unit_cost, self.available() // unit_cost))

    def _cool(self, key: str, now: datetime, seconds: int = 300) -> bool:
        previous = self.cooldown.get(key)
        return previous is not None and (now - previous).total_seconds() < seconds

    @staticmethod
    def _snapshot(ticker: str, **values: Any) -> None:
        """Record useful telemetry while remaining compatible with older dashboards."""
        try:
            record_model_snapshot(ticker=ticker, **values)
        except TypeError:
            record_model_snapshot(ticker, decision=values.get("decision", "MONITOR"), reason=values.get("reason", ""))

    def _book_locked_trade(
        self, key: str, module: str, side: str, unit_cost: int, count: int,
        gross_payout: int, fees: int, now: datetime, release: datetime,
    ) -> None:
        total_cost = unit_cost * count + fees
        pnl = gross_payout * count - total_cost
        if count <= 0 or pnl <= 0 or total_cost > self.available() or len(self.positions) + len(self.locked) >= self.max_positions:
            return
        self.cash -= total_cost
        self.cooldown[key] = now
        self.locked[key] = LockedBundle(key, module, side, unit_cost, count, total_cost, gross_payout, release)
        record_entry(key[:180], side, unit_cost, count=count, stop_price=None, take_profit=gross_payout, execution_mode="paper_locked")
        logger.warning("PAPER LOCKED %s | %s | count=%d | expected_pnl=$%.2f", module, key, count, pnl / 100)

    def _release_locked(self, now: datetime) -> None:
        for key, bundle in list(self.locked.items()):
            if now < bundle.release:
                continue
            proceeds = bundle.gross_payout * bundle.count
            pnl = proceeds - bundle.total_cost
            self.cash += proceeds
            self.realized_pnl += pnl
            self.locked.pop(key, None)
            record_exit(key[:180], bundle.side, bundle.unit_cost, bundle.gross_payout, bundle.module, bundle.count, pnl, self.realized_pnl)
            self.scorer.update(bundle.module, pnl)

    def _same_market_arbitrage(self, markets: Iterable[Market], now: datetime) -> None:
        for m in markets:
            if not self._valid_book(m) or self._cool("PAIR:" + m.ticker, now):
                continue
            cost = int(m.yes_ask) + int(m.no_ask)
            count = self._count_for(cost)
            if count <= 0:
                continue
            fees = fee_cents(int(m.yes_ask), count) + fee_cents(int(m.no_ask), count)
            if 100 * count - cost * count - fees >= max(2 * count, 10):
                release = self._close_time(m)
                if release is not None:
                    self._book_locked_trade("PAIR:" + m.ticker, "YES_NO_ARBITRAGE", "PAIR", cost, count, 100, fees, now, release)

    def _ladder_arbitrage(self, markets: Iterable[Market], now: datetime) -> None:
        groups: dict[tuple[str, str], list[Market]] = defaultdict(list)
        for m in markets:
            if is_upper_threshold(m) and self._valid_book(m) and m.close_time:
                groups[(m.event_ticker or m.series_ticker or "", m.close_time)].append(m)
        for group in groups.values():
            ordered = sorted(group, key=lambda m: float(m.floor_strike or 0))
            for lower, higher in pairwise(ordered):
                key = f"LADDER:{lower.ticker}|{higher.ticker}"
                if self._cool(key, now):
                    continue
                cost = int(lower.yes_ask) + int(higher.no_ask)
                count = self._count_for(cost)
                fees = fee_cents(int(lower.yes_ask), count) + fee_cents(int(higher.no_ask), count)
                release = self._close_time(lower)
                if release is not None and count and 100 * count - cost * count - fees >= max(2 * count, 10):
                    self._book_locked_trade(key, "STRIKE_LADDER_ARBITRAGE", "LOW_YES_HIGH_NO", cost, count, 100, fees, now, release)

    def _update_history(self, markets: Iterable[Market], now: datetime) -> None:
        for m in markets:
            if self._valid_book(m):
                midpoint = (int(m.yes_bid) + int(m.yes_ask)) / 2
                hist = self.history[m.ticker]
                hist.append((now, midpoint))
                cutoff = now - timedelta(minutes=3)
                while hist and hist[0][0] < cutoff:
                    hist.popleft()

    def _manage_pending(self, by_ticker: dict[str, Market], now: datetime) -> None:
        for ticker, order in list(self.pending.items()):
            m = by_ticker.get(ticker)
            if m is None or not self._valid_book(m) or now >= order.expires:
                self.pending.pop(ticker, None)
                self.rejection_totals["PENDING_EXPIRED"] += 1
                continue
            ask = int(m.yes_ask if order.side is Side.YES else m.no_ask)
            if ask > order.price:
                order.crossed_samples = 0
                continue
            order.crossed_samples += 1
            if order.crossed_samples < 2:
                continue
            fee = fee_cents(order.price, order.count)
            cost = order.price * order.count + fee
            if cost > self.available() or len(self.positions) + len(self.locked) >= self.max_positions:
                self.pending.pop(ticker, None)
                self.rejection_totals["FILL_RISK_LIMIT"] += 1
                continue
            position = PaperPosition(ticker, order.side, order.price, order.count, now, min(99, order.price + 9), max(1, order.price - 5), "BTC_ETH_REBOUND", order.score_key, fee)
            self.positions[ticker] = position
            self.cash -= cost
            self.pending.pop(ticker, None)
            record_entry(ticker, order.side.value, order.price, count=order.count, stop_price=position.stop, take_profit=position.take_profit, execution_mode="paper_ask_cross_confirmed")
            self._snapshot(ticker, decision="ENTRY", reason=f"REBOUND_FILL_{order.side.value.upper()}_SCORE_{order.score:.2f}")

    def _manage_positions(self, by_ticker: dict[str, Market], now: datetime) -> None:
        for ticker, p in list(self.positions.items()):
            m = by_ticker.get(ticker)
            if m is None or not self._valid_book(m):
                continue
            bid = int(m.yes_bid if p.side is Side.YES else m.no_bid)
            held = (now - p.opened).total_seconds()
            reason = None
            if bid >= p.take_profit:
                exit_price, reason = p.take_profit, "REBOUND_TAKE_PROFIT"
            elif bid <= p.stop:
                exit_price, reason = bid, "REBOUND_STOP"
            elif held >= 120:
                exit_price, reason = bid, "REBOUND_TIME_EXIT"
            else:
                seconds = self._seconds_left(m, now)
                if seconds is not None and seconds <= 120:
                    exit_price, reason = bid, "FINAL_TWO_MINUTE_EXIT"
            if reason is None:
                continue
            exit_fee = fee_cents(exit_price, p.count)
            proceeds = exit_price * p.count - exit_fee
            cost = p.entry * p.count + p.fee_paid
            pnl = proceeds - cost
            self.cash += proceeds
            self.realized_pnl += pnl
            self.positions.pop(ticker, None)
            self.cooldown["MM:" + ticker] = now
            record_exit(ticker, p.side.value, p.entry, exit_price, reason, p.count, pnl, self.realized_pnl)
            self.scorer.update(p.score_key, pnl)

    def _reject(self, cycle: Counter[str], reason: str) -> None:
        cycle[reason] += 1
        self.rejection_totals[reason] += 1

    def _quote_rebounds(self, markets: Iterable[Market], now: datetime) -> Counter[str]:
        cycle: Counter[str] = Counter()
        if len(self.positions) + len(self.locked) + len(self.pending) >= self.max_positions:
            cycle["PORTFOLIO_FULL"] += 1
            return cycle
        if self.realized_pnl <= -self.daily_loss_limit:
            cycle["DAILY_STOP"] += 1
            return cycle

        candidates: list[tuple[float, Market, Side, int, str]] = []
        for m in markets:
            if not is_scalp_market(m):
                self._reject(cycle, "NOT_BTC_ETH_15M")
                continue
            cycle["SCALP_MARKET"] += 1
            if m.ticker in self.positions or m.ticker in self.pending or self._cool("MM:" + m.ticker, now, 180):
                self._reject(cycle, "ALREADY_ACTIVE_OR_COOLING")
                continue
            if not self._valid_book(m):
                self._reject(cycle, "INVALID_BOOK")
                continue
            if (m.volume or 0) < 500:
                self._reject(cycle, "LOW_VOLUME")
                continue
            seconds = self._seconds_left(m, now)
            if seconds is None or not 180 < seconds < 780:
                self._reject(cycle, "OUTSIDE_TIME_WINDOW")
                continue
            hist = self.history[m.ticker]
            recent_points = [(ts, value) for ts, value in hist if ts >= now - timedelta(seconds=90)]
            if len(recent_points) < 15:
                self._reject(cycle, "INSUFFICIENT_HISTORY")
                continue
            yes_mids = [value for _, value in recent_points]
            for side, mids, bid, ask in (
                (Side.YES, yes_mids, int(m.yes_bid), int(m.yes_ask)),
                (Side.NO, [100 - value for value in yes_mids], int(m.no_bid), int(m.no_ask)),
            ):
                spread = ask - bid
                price = max(1, min(99, min(ask - 1, bid + 1)))
                if not 35 <= price <= 65:
                    self._reject(cycle, "PRICE_OUTSIDE_35_65")
                    continue
                if not 1 <= spread <= 8:
                    self._reject(cycle, "SPREAD_OUTSIDE_1_8")
                    continue
                low_index = min(range(len(mids)), key=mids.__getitem__)
                low = mids[low_index]
                rebound = mids[-1] - low
                pre_drop = max(mids[: low_index + 1]) - low
                confirmed = len(mids) >= 3 and mids[-3] < mids[-2] <= mids[-1]
                if low_index > len(mids) - 3:
                    self._reject(cycle, "LOW_TOO_RECENT")
                    continue
                if pre_drop < 3:
                    self._reject(cycle, "NO_3C_PANIC")
                    continue
                if rebound < 2:
                    self._reject(cycle, "NO_2C_REBOUND")
                    continue
                if not confirmed:
                    self._reject(cycle, "REBOUND_NOT_CONFIRMED")
                    continue
                key = f"BTC_ETH_REBOUND:{side.value.upper()}"
                structural = rebound + pre_drop
                depth_quality = min(1.0, (m.volume or 0) / 10_000)
                score = self.scorer.score(key, structural, depth_quality)
                required_score = 1.65 if self.scorer.samples(key) < 20 else 2.10
                if score < required_score:
                    self._reject(cycle, "SCORE_TOO_LOW")
                    continue
                count = self._count_for(price)
                if count <= 0:
                    self._reject(cycle, "NO_AVAILABLE_CAPITAL")
                    continue
                candidates.append((score, m, side, price, key))
                cycle["QUALIFIED"] += 1

        for score, m, side, price, key in sorted(candidates, reverse=True, key=lambda item: item[0]):
            if len(self.positions) + len(self.locked) + len(self.pending) >= self.max_positions:
                break
            count = self._count_for(price)
            if count <= 0:
                continue
            self.pending[m.ticker] = PendingMaker(m.ticker, side, price, count, now, now + timedelta(seconds=60), score, key)
            self._snapshot(
                m.ticker,
                seconds_left=self._seconds_left(m, now),
                yes_bid=m.yes_bid,
                yes_ask=m.yes_ask,
                no_bid=m.no_bid,
                no_ask=m.no_ask,
                decision="QUOTE",
                reason=f"REBOUND_{side.value.upper()}_LIMIT_{price}_SCORE_{score:.2f}_TTL_60",
            )
        return cycle

    async def _fetch_open_markets(self) -> list[Market]:
        result = self.client.get_open_crypto_markets()
        if inspect.isawaitable(result):
            result = await result
        return list(result or [])

    def _record_heartbeat(self, raw_count: int, crypto_count: int, cycle: Counter[str]) -> None:
        priorities = (
            "SCALP_MARKET", "QUALIFIED", "INVALID_BOOK", "LOW_VOLUME",
            "OUTSIDE_TIME_WINDOW", "INSUFFICIENT_HISTORY", "PRICE_OUTSIDE_35_65",
            "SPREAD_OUTSIDE_1_8", "LOW_TOO_RECENT", "NO_3C_PANIC",
            "NO_2C_REBOUND", "REBOUND_NOT_CONFIRMED", "SCORE_TOO_LOW",
        )
        detail = "_".join(f"{name}_{cycle.get(name, 0)}" for name in priorities)
        reason = (
            f"RAW_{raw_count}_CRYPTO_{crypto_count}_OPEN_{len(self.positions)}_"
            f"LOCKED_{len(self.locked)}_PENDING_{len(self.pending)}_"
            f"CASH_{self.cash}_PNL_{self.realized_pnl}_{detail}"
        )
        self._snapshot("CRYPTO_PORTFOLIO", decision="MONITOR", reason=reason)

    async def run(self, max_cycles: int | None = None) -> None:
        """Run until stopped; max_cycles is intended for tests and diagnostics."""
        if self._running:
            return
        self._running = True
        cycle_number = 0
        try:
            while self._running:
                try:
                    now = datetime.now(UTC)
                    raw_markets = await self._fetch_open_markets()
                    markets = [m for m in raw_markets if is_crypto_market(m)]
                    by_ticker = {m.ticker: m for m in markets}
                    self._release_locked(now)
                    self._manage_pending(by_ticker, now)
                    self._manage_positions(by_ticker, now)
                    self._update_history(markets, now)
                    self._same_market_arbitrage(markets, now)
                    self._ladder_arbitrage(markets, now)
                    cycle_rejections = self._quote_rebounds(markets, now)
                    self._record_heartbeat(len(raw_markets), len(markets), cycle_rejections)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Crypto portfolio scan failed; retrying")
                cycle_number += 1
                if max_cycles is not None and cycle_number >= max_cycles:
                    break
                await asyncio.sleep(self.poll_interval)
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False
