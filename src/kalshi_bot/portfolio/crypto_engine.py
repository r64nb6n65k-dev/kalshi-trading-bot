"""Conservative paper-only multi-market crypto portfolio engine.

It scans all open Kalshi crypto markets for complementary arbitrage and
monotonic strike-ladder arbitrage. Directional scalps are restricted to BTC
and ETH 15-minute markets and require a confirmed contract-price rebound.
The adaptive scorer learns only from completed paper positions and can never
override hard bankroll, fee, liquidity, or settlement-safety checks.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from itertools import pairwise
from pathlib import Path

from kalshi_bot.dashboard import record_entry, record_exit, record_model_snapshot
from kalshi_bot.exchange.client import KalshiClient
from kalshi_bot.exchange.models import Market, Side
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)

CRYPTO_WORDS = (
    "BTC",
    "BITCOIN",
    "ETH",
    "ETHEREUM",
    "SOL",
    "SOLANA",
    "XRP",
    "DOGE",
    "BNB",
    "AVAX",
    "CARDANO",
    "ADA",
    "LTC",
    "LITECOIN",
    "BCH",
    "CHAINLINK",
    "LINK",
    "POLKADOT",
    "DOT",
    "POLYGON",
    "MATIC",
    "CRYPTO",
)
UPPER_WORDS = ("ABOVE", "GREATER", "HIGHER", "AT LEAST", "OVER")
SCALP_SERIES = ("KXBTC15M", "KXETH15M")


def fee_cents(price: int, count: int, coefficient: float = 0.07) -> int:
    """Conservative Kalshi-style fee estimate, rounded up to a cent."""
    probability = Decimal(price) / Decimal(100)
    dollars = Decimal(str(coefficient)) * Decimal(count) * probability * (1 - probability)
    return int((dollars * 100).to_integral_value(rounding=ROUND_CEILING))


def is_crypto_market(market: Market) -> bool:
    ticker_text = " ".join(
        filter(None, (market.series_ticker, market.event_ticker, market.ticker))
    ).upper()
    if re.search(
        r"(?:^|[^A-Z])(KX)?(?:BTC|ETH|SOL|XRP|DOGE|BNB|AVAX|ADA|LTC|BCH|LINK|DOT|MATIC)(?:[^A-Z]|$)",
        ticker_text,
    ):
        return True
    prose = " ".join(filter(None, (market.title, market.subtitle))).upper()
    return any(re.search(rf"\b{re.escape(word)}\b", prose) for word in CRYPTO_WORDS)


def is_upper_threshold(market: Market) -> bool:
    if market.floor_strike is None or market.cap_strike is not None:
        return False
    text = " ".join(filter(None, (market.title, market.subtitle, market.rules_primary))).upper()
    return market.strike_type in {None, "greater", "greater_or_equal"} and any(
        w in text for w in UPPER_WORDS
    )


def is_scalp_market(market: Market) -> bool:
    text = " ".join(
        filter(None, (market.series_ticker, market.event_ticker, market.ticker))
    ).upper()
    return any(series in text for series in SCALP_SERIES)


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
    volume_at_quote: int
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
    """Small, explainable online learner; hard rules remain authoritative."""

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
    """Scan and paper-trade every safely identifiable open crypto market."""

    def __init__(
        self, client: KalshiClient, bankroll_cents: int = 50_000, poll_interval: float = 2.0
    ) -> None:
        self.client = client
        self.starting_bankroll = bankroll_cents
        self.cash = bankroll_cents
        self.realized_pnl = 0
        self.poll_interval = poll_interval
        self.reserve = 20_000
        self.max_deployed = 30_000
        self.max_positions = 6
        self.max_per_trade = 5_000
        self.daily_loss_limit = 5_000
        self.positions: dict[str, PaperPosition] = {}
        self.locked: dict[str, LockedBundle] = {}
        self.pending: dict[str, PendingMaker] = {}
        self.cooldown: dict[str, datetime] = {}
        self.history: dict[str, deque[tuple[datetime, float]]] = defaultdict(
            lambda: deque(maxlen=60)
        )
        volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
        state_path = Path(volume or "/tmp") / "crypto_adaptive_score.json"
        self.scorer = AdaptiveScorer(state_path)
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
        return all(v is not None and 1 <= v <= 99 for v in values)

    @staticmethod
    def _seconds_left(m: Market, now: datetime) -> float | None:
        if not m.close_time:
            return None
        try:
            close = datetime.fromisoformat(m.close_time.replace("Z", "+00:00"))
            if close.tzinfo is None:
                close = close.replace(tzinfo=UTC)
            return (close - now).total_seconds()
        except ValueError:
            return None

    @staticmethod
    def _close_time(m: Market) -> datetime | None:
        if not m.close_time:
            return None
        try:
            close = datetime.fromisoformat(m.close_time.replace("Z", "+00:00"))
            return close.replace(tzinfo=UTC) if close.tzinfo is None else close
        except ValueError:
            return None

    def _count_for(self, unit_cost: int, cap: int = 50) -> int:
        if unit_cost <= 0:
            return 0
        return max(0, min(cap, self.max_per_trade // unit_cost, self.available() // unit_cost))

    def _cool(self, key: str, now: datetime, seconds: int = 300) -> bool:
        previous = self.cooldown.get(key)
        return previous is not None and (now - previous).total_seconds() < seconds

    def _book_locked_trade(
        self,
        key: str,
        module: str,
        side: str,
        unit_cost: int,
        count: int,
        gross_payout: int,
        fees: int,
        now: datetime,
        release: datetime,
    ) -> None:
        total_cost = unit_cost * count + fees
        pnl = gross_payout * count - total_cost
        if (
            count <= 0
            or pnl <= 0
            or total_cost > self.available()
            or len(self.positions) + len(self.locked) >= self.max_positions
        ):
            return
        self.cash -= total_cost
        self.cooldown[key] = now
        synthetic = key[:180]
        self.locked[key] = LockedBundle(
            key, module, side, unit_cost, count, total_cost, gross_payout, release
        )
        record_entry(
            synthetic,
            side,
            unit_cost,
            count=count,
            stop_price=None,
            take_profit=gross_payout,
            execution_mode="paper_locked",
        )
        logger.warning(
            "PAPER LOCKED %s | %s | count=%d | expected_pnl=$%.2f | capital reserved",
            module,
            key,
            count,
            pnl / 100,
        )

    def _release_locked(self, now: datetime) -> None:
        for key, bundle in list(self.locked.items()):
            if now < bundle.release:
                continue
            proceeds = bundle.gross_payout * bundle.count
            pnl = proceeds - bundle.total_cost
            self.cash += proceeds
            self.realized_pnl += pnl
            self.locked.pop(key, None)
            record_exit(
                key[:180],
                bundle.side,
                bundle.unit_cost,
                bundle.gross_payout,
                bundle.module,
                bundle.count,
                pnl,
                self.realized_pnl,
            )
            self.scorer.update(bundle.module, pnl)

    def _same_market_arbitrage(self, markets: Iterable[Market], now: datetime) -> None:
        for m in markets:
            if not self._valid_book(m) or self._cool("PAIR:" + m.ticker, now):
                continue
            assert m.yes_ask is not None and m.no_ask is not None
            release = self._close_time(m)
            if release is None:
                continue
            cost = m.yes_ask + m.no_ask
            count = self._count_for(cost)
            if count <= 0:
                continue
            fees = fee_cents(m.yes_ask, count) + fee_cents(m.no_ask, count)
            if 100 * count - cost * count - fees >= max(2 * count, 10):
                self._book_locked_trade(
                    "PAIR:" + m.ticker,
                    "YES_NO_ARBITRAGE",
                    "PAIR",
                    cost,
                    count,
                    100,
                    fees,
                    now,
                    release,
                )

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
                assert lower.yes_ask is not None and higher.no_ask is not None
                release = self._close_time(lower)
                if release is None:
                    continue
                cost = lower.yes_ask + higher.no_ask
                count = self._count_for(cost)
                fees = fee_cents(lower.yes_ask, count) + fee_cents(higher.no_ask, count)
                if count and 100 * count - cost * count - fees >= max(2 * count, 10):
                    self._book_locked_trade(
                        key,
                        "STRIKE_LADDER_ARBITRAGE",
                        "LOW_YES_HIGH_NO",
                        cost,
                        count,
                        100,
                        fees,
                        now,
                        release,
                    )

    def _update_history(self, markets: Iterable[Market], now: datetime) -> None:
        for m in markets:
            if self._valid_book(m):
                assert m.yes_bid is not None and m.yes_ask is not None
                self.history[m.ticker].append((now, (m.yes_bid + m.yes_ask) / 2))

    def _manage_pending(self, by_ticker: dict[str, Market], now: datetime) -> None:
        for ticker, order in list(self.pending.items()):
            m = by_ticker.get(ticker)
            if m is None or not self._valid_book(m) or now >= order.expires:
                self.pending.pop(ticker, None)
                continue
            ask = m.yes_ask if order.side is Side.YES else m.no_ask
            age = (now - order.created).total_seconds()
            volume_advanced = (m.volume or 0) > order.volume_at_quote
            if ask is None or ask > order.price or not volume_advanced or age < 2:
                order.crossed_samples = 0
                continue
            order.crossed_samples += 1
            if order.crossed_samples < 2:
                continue
            fee = fee_cents(order.price, order.count, coefficient=0.07)
            cost = order.price * order.count + fee
            if (
                cost > self.available()
                or len(self.positions) + len(self.locked) >= self.max_positions
            ):
                self.pending.pop(ticker, None)
                continue
            position = PaperPosition(
                ticker,
                order.side,
                order.price,
                order.count,
                now,
                min(99, order.price + 9),
                max(1, order.price - 5),
                "BTC_ETH_REBOUND",
                order.score_key,
                fee,
            )
            self.positions[ticker] = position
            self.cash -= cost
            self.pending.pop(ticker, None)
            record_entry(
                ticker,
                order.side.value,
                order.price,
                count=order.count,
                stop_price=position.stop,
                take_profit=position.take_profit,
                execution_mode="paper_volume_confirmed_cross",
            )

    def _manage_positions(self, by_ticker: dict[str, Market], now: datetime) -> None:
        for ticker, p in list(self.positions.items()):
            m = by_ticker.get(ticker)
            if m is None or not self._valid_book(m):
                continue
            bid = m.yes_bid if p.side is Side.YES else m.no_bid
            assert bid is not None
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
            exit_fee = fee_cents(exit_price, p.count, coefficient=0.07)
            proceeds = exit_price * p.count - exit_fee
            cost = p.entry * p.count + p.fee_paid
            pnl = proceeds - cost
            self.cash += proceeds
            self.realized_pnl += pnl
            self.positions.pop(ticker, None)
            self.cooldown["MM:" + ticker] = now
            record_exit(
                ticker, p.side.value, p.entry, exit_price, reason, p.count, pnl, self.realized_pnl
            )
            self.scorer.update(p.score_key, pnl)

    def _quote_rebounds(self, markets: Iterable[Market], now: datetime) -> None:
        if (
            len(self.positions) + len(self.locked) + len(self.pending) >= self.max_positions
            or self.realized_pnl <= -self.daily_loss_limit
        ):
            return
        candidates: list[tuple[float, Market, Side, int, str]] = []
        for m in markets:
            if (
                m.ticker in self.positions
                or m.ticker in self.pending
                or self._cool("MM:" + m.ticker, now, 180)
            ):
                continue
            if not is_scalp_market(m) or not self._valid_book(m) or (m.volume or 0) < 1_000:
                continue
            seconds = self._seconds_left(m, now)
            if seconds is None or not 180 < seconds < 780:
                continue
            assert None not in (m.yes_bid, m.yes_ask, m.no_bid, m.no_ask)
            hist = self.history[m.ticker]
            if len(hist) < 20:
                continue
            yes_mids = [value for _, value in hist]
            for side, mids, bid, ask in (
                (Side.YES, yes_mids, int(m.yes_bid), int(m.yes_ask)),
                (Side.NO, [100 - value for value in yes_mids], int(m.no_bid), int(m.no_ask)),
            ):
                price = min(ask - 1, bid + 1)
                if not 50 <= price <= 60 or ask - bid < 3:
                    continue
                recent = mids[-10:]
                low_index = min(range(len(recent)), key=recent.__getitem__)
                low = recent[low_index]
                rebound = recent[-1] - low
                pre_drop = max(recent[: low_index + 1]) - low
                confirmed = recent[-3] < recent[-2] <= recent[-1]
                if low_index > len(recent) - 3 or rebound < 2 or pre_drop < 3 or not confirmed:
                    continue
                key = f"BTC_ETH_REBOUND:{side.value.upper()}"
                structural = rebound + pre_drop
                depth_quality = min(1.0, (m.volume or 0) / 10_000)
                score = self.scorer.score(key, structural, depth_quality)
                if score >= 2.25:
                    candidates.append((score, m, side, price, key))
        for score, m, side, price, key in sorted(candidates, reverse=True, key=lambda x: x[0]):
            if len(self.positions) + len(self.locked) + len(self.pending) >= self.max_positions:
                break
            count = self._count_for(price, cap=20)
            if count <= 0:
                break
            self.pending[m.ticker] = PendingMaker(
                m.ticker,
                side,
                price,
                count,
                now,
                datetime.fromtimestamp(now.timestamp() + 20, UTC),
                score,
                key,
                m.volume or 0,
            )
            record_model_snapshot(
                ticker=m.ticker,
                yes_bid=m.yes_bid,
                yes_ask=m.yes_ask,
                no_bid=m.no_bid,
                no_ask=m.no_ask,
                decision="QUOTE",
                reason=f"BTC_ETH_REBOUND_{side.value.upper()}_AI_{score:.2f}",
            )

    async def step(self) -> int:
        now = datetime.now(UTC)
        all_open = await self.client.get_open_crypto_markets()
        markets = [m for m in all_open if is_crypto_market(m)]
        by_ticker = {m.ticker: m for m in markets}
        self._update_history(markets, now)
        self._release_locked(now)
        self._manage_pending(by_ticker, now)
        self._manage_positions(by_ticker, now)
        self._same_market_arbitrage(markets, now)
        self._ladder_arbitrage(markets, now)
        self._quote_rebounds(markets, now)
        record_model_snapshot(
            ticker="CRYPTO_PORTFOLIO",
            decision="MONITOR",
            reason=(
                f"MARKETS_{len(markets)}_OPEN_{len(self.positions)}_"
                f"LOCKED_{len(self.locked)}_PENDING_{len(self.pending)}_"
                f"CASH_{self.cash}_PNL_{self.realized_pnl}"
            ),
        )
        return len(markets)

    async def run(self, max_cycles: int | None = None) -> None:
        self._running = True
        cycle = 0
        logger.warning(
            "MULTI CRYPTO PAPER ENGINE | bankroll=$%.2f | live trading locked", self.cash / 100
        )
        while self._running:
            try:
                count = await self.step()
                logger.info(
                    "Crypto scan | markets=%d | cash=$%.2f | pnl=$%.2f",
                    count,
                    self.cash / 100,
                    self.realized_pnl / 100,
                )
            except Exception:
                logger.exception("Crypto portfolio scan failed; retrying")
            cycle += 1
            if max_cycles is not None and cycle >= max_cycles:
                break
            await asyncio.sleep(self.poll_interval)
        self._running = False

    def stop(self) -> None:
        self._running = False
