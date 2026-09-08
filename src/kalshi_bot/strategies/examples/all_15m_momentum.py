"""One-decision momentum strategy for every Kalshi 15-minute market."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, ClassVar

from kalshi_bot.dashboard import record_model_snapshot
from kalshi_bot.exchange.models import (
    Action,
    Market,
    Order,
    OrderRequest,
    OrderType,
    Side,
    TimeInForce,
)
from kalshi_bot.strategies.base import UnderlyingTick
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)


class All15mMomentumStrategy:
    """Choose YES or NO once near 10:00, then hold unless the bid reaches 98c."""

    name = "all_15m_momentum"
    _PRODUCT_ALIASES: ClassVar[dict[str, str]] = {
        "GOLD": "XAU_USD",
        "SILVER": "XAG_USD",
        "COPPER": "XCU_USD",
        "WTI": "WTICO_USD",
        "NATGAS": "NATGAS_USD",
        "PALLADIUM": "XPD_USD",
        "PLATINUM": "XPT_USD",
        "EURUSD": "EURUSD-USD",
        "GBPUSD": "GBPUSD-USD",
        "USDJPY": "USDJPY-USD",
        "INX": "INX-USD",
        "NDQ": "NDQ-USD",
    }

    def __init__(self, **params: Any) -> None:
        self.contracts = int(params.get("contracts", 5))
        self.bankroll_cents = int(params.get("bankroll_cents", 50_000))
        self.decision_seconds = float(params.get("decision_seconds", 600))
        self.decision_window = float(params.get("decision_window", 15))
        self.take_profit = int(params.get("take_profit", 98))
        self.minimum_history = float(params.get("minimum_history", 45))
        self.minimum_separation_bps = float(params.get("minimum_separation_bps", 4.0))
        self.entry_slippage_cents = int(params.get("entry_slippage_cents", 2))
        self._decided: set[str] = set()
        self._pending: set[str] = set()
        self._sides: dict[str, Side] = {}
        self._counts: dict[str, int] = {}
        self._entry_costs: dict[str, int] = {}

    @staticmethod
    def seconds_to_close(market: Market) -> float | None:
        if not market.close_time:
            return None
        try:
            close = datetime.fromisoformat(market.close_time.replace("Z", "+00:00"))
            if close.tzinfo is None:
                close = close.replace(tzinfo=UTC)
            return (close - datetime.now(UTC)).total_seconds()
        except ValueError:
            return None

    @classmethod
    def product_for(cls, market: Market) -> str | None:
        series = (market.series_ticker or market.ticker.split("-")[0]).upper()
        match = re.fullmatch(r"KX([A-Z0-9]+)15M", series)
        if not match:
            return None
        product = match.group(1)
        # Leader/comparison markets do not have one numeric target/underlying.
        if product in {"CRYPTOCOMP", "CRYPTOLEAD"}:
            return None
        return cls._PRODUCT_ALIASES.get(product, f"{product}-USD")

    @staticmethod
    def _at_or_before(ticks: tuple[UnderlyingTick, ...], timestamp: float) -> UnderlyingTick:
        return min(ticks, key=lambda item: abs(item.timestamp.timestamp() - timestamp))

    def _signal(
        self, market: Market, now: float, ticks: tuple[UnderlyingTick, ...]
    ) -> tuple[Side | None, str]:
        target = float(market.floor_strike) if market.floor_strike is not None else None
        if target is None or target <= 0 or len(ticks) < 2:
            return None, "MISSING_OR_STALE_PRICE_HISTORY_OR_STRIKE"
        latest = ticks[-1]
        first_time = ticks[0].timestamp.timestamp()
        if now - latest.timestamp.timestamp() > 5 or now - first_time < self.minimum_history:
            return None, "MISSING_OR_STALE_PRICE_HISTORY_OR_STRIKE"
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
        recent_volume = sum(tick.size for tick in ticks if tick.timestamp.timestamp() >= now - 60)
        older_volume = sum(tick.size for tick in ticks if tick.timestamp.timestamp() < now - 60)
        older_seconds = max(1.0, now - first_time - 60)
        baseline_volume = older_volume * 60 / older_seconds
        volume_ratio = recent_volume / baseline_volume if baseline_volume > 0 else 1.0
        volume_weight = max(0.5, min(2.0, volume_ratio))
        score = 0.55 * separation_bps + volume_weight * (0.30 * short_bps + 0.15 * long_bps)
        side = Side.YES if score >= 0 else Side.NO
        detail = (
            f"underlying={latest.price:.4f} target={target:.4f} score={score:+.2f} "
            f"separation_bps={separation_bps:+.2f} momentum_60_bps={short_bps:+.2f} "
            f"momentum_long_bps={long_bps:+.2f} volume_ratio={volume_ratio:.2f}"
        )
        return side, detail

    def reserved_cents(self) -> int:
        return sum(self._entry_costs.values())

    def entry_price_for(self, ticker: str) -> int:
        count = self._counts.get(ticker, 0)
        return round(self._entry_costs.get(ticker, 0) / count) if count else 0

    def count_for(self, ticker: str) -> int:
        return self._counts.get(ticker, 0)

    def reconcile(self, active_tickers: set[str]) -> None:
        """Release bankroll reservations after a market closes or disappears."""
        for ticker in set(self._entry_costs) - active_tickers:
            self._entry_costs.pop(ticker, None)
            self._counts.pop(ticker, None)
            self._sides.pop(ticker, None)
            self._pending.discard(ticker)

    def orders_for(
        self,
        market: Market,
        position: int,
        now: float,
        underlying_ticks: tuple[UnderlyingTick, ...],
    ) -> list[OrderRequest]:
        ticker = market.ticker
        seconds_left = self.seconds_to_close(market)
        if seconds_left is None or ticker in self._pending:
            return []

        if position != 0:
            side = self._sides.get(ticker, Side.YES if position > 0 else Side.NO)
            bid = market.yes_bid if side is Side.YES else market.no_bid
            if bid is not None and bid >= self.take_profit:
                self._pending.add(ticker)
                return [self._order(ticker, Action.SELL, side, self.take_profit, abs(position))]
            return []

        if ticker in self._decided:
            return []
        if not (
            self.decision_seconds - self.decision_window <= seconds_left <= self.decision_seconds
        ):
            return []

        side, detail = self._signal(market, now, underlying_ticks)
        self._decided.add(ticker)
        if side is None:
            logger.warning("SKIP | ticker=%s | %s", ticker, detail)
            record_model_snapshot(
                ticker=ticker,
                seconds_left=seconds_left,
                target_price=market.floor_strike,
                yes_bid=market.yes_bid,
                yes_ask=market.yes_ask,
                no_bid=market.no_bid,
                no_ask=market.no_ask,
                decision="SKIP",
                reason=detail,
            )
            return []
        ask = market.yes_ask if side is Side.YES else market.no_ask
        if ask is None or not 1 <= ask <= 99:
            logger.warning("SKIP | ticker=%s | side=%s | no executable ask", ticker, side.value)
            return []
        limit_price = min(99, ask + self.entry_slippage_cents)
        cost = limit_price * self.contracts
        if self.reserved_cents() + cost > self.bankroll_cents:
            logger.warning(
                "SKIP | ticker=%s | bankroll cap | needed=%d available=%d",
                ticker,
                cost,
                self.bankroll_cents - self.reserved_cents(),
            )
            return []
        logger.warning(
            "SIGNAL | ticker=%s | side=%s | ask=%dc | limit=%dc | %s",
            ticker,
            side.value,
            ask,
            limit_price,
            detail,
        )
        latest_price = underlying_ticks[-1].price
        target = float(market.floor_strike or 0)
        short = self._at_or_before(underlying_ticks, now - 60)
        record_model_snapshot(
            ticker=ticker,
            seconds_left=seconds_left,
            target_price=target,
            separation=latest_price - target,
            yes_bid=market.yes_bid,
            yes_ask=market.yes_ask,
            no_bid=market.no_bid,
            no_ask=market.no_ask,
            momentum_60=latest_price - short.price,
            decision=f"BUY_{side.value.upper()}",
            reason=detail,
        )
        self._pending.add(ticker)
        return [self._order(ticker, Action.BUY, side, limit_price, self.contracts)]

    @staticmethod
    def _order(ticker: str, action: Action, side: Side, price: int, count: int) -> OrderRequest:
        kwargs: dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": OrderType.LIMIT,
            "time_in_force": TimeInForce.IMMEDIATE_OR_CANCEL,
        }
        kwargs["yes_price" if side is Side.YES else "no_price"] = price
        return OrderRequest(**kwargs)

    def on_order_result(self, request: OrderRequest, result: Order | None) -> None:
        ticker = request.ticker
        self._pending.discard(ticker)
        if result is None or int(result.fill_count or 0) <= 0:
            return
        count = int(result.fill_count)
        price = int(
            result.outcome_fill_price
            or (request.yes_price if request.side is Side.YES else request.no_price)
            or 0
        )
        if request.action is Action.BUY:
            self._sides[ticker] = request.side
            self._counts[ticker] = count
            self._entry_costs[ticker] = price * count
        else:
            remaining = max(0, self._counts.get(ticker, count) - count)
            self._counts[ticker] = remaining
            if remaining == 0:
                self._entry_costs.pop(ticker, None)
