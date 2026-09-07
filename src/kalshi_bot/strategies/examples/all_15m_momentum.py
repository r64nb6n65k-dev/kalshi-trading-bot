"""One-decision momentum strategy for every Kalshi 15-minute market."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from kalshi_bot.exchange.models import (
    Action,
    Market,
    Order,
    OrderRequest,
    OrderType,
    Side,
    TimeInForce,
)
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Observation:
    timestamp: float
    midpoint: float
    volume: int


class All15mMomentumStrategy:
    """Choose YES or NO once near 10:00, then hold unless the bid reaches 98c."""

    name = "all_15m_momentum"

    def __init__(self, **params: Any) -> None:
        self.contracts = int(params.get("contracts", 20))
        self.bankroll_cents = int(params.get("bankroll_cents", 50_000))
        self.decision_seconds = float(params.get("decision_seconds", 600))
        self.decision_window = float(params.get("decision_window", 15))
        self.take_profit = int(params.get("take_profit", 98))
        self.minimum_history = float(params.get("minimum_history", 45))
        self._history: dict[str, deque[Observation]] = defaultdict(lambda: deque(maxlen=900))
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

    @staticmethod
    def _midpoint(market: Market) -> float | None:
        if market.yes_bid is not None and market.yes_ask is not None:
            return (market.yes_bid + market.yes_ask) / 2
        return float(market.last_price) if market.last_price is not None else None

    def observe(self, market: Market, now: float) -> None:
        midpoint = self._midpoint(market)
        if midpoint is not None:
            self._history[market.ticker].append(Observation(now, midpoint, int(market.volume or 0)))

    def _at_or_before(self, history: deque[Observation], timestamp: float) -> Observation:
        return min(history, key=lambda item: abs(item.timestamp - timestamp))

    def _signal(self, market: Market, now: float) -> tuple[Side, str] | None:
        history = self._history[market.ticker]
        if len(history) < 2 or now - history[0].timestamp < self.minimum_history:
            return None
        latest = history[-1]
        short = self._at_or_before(history, now - 60)
        long = history[0]
        short_momentum = latest.midpoint - short.midpoint
        long_momentum = latest.midpoint - long.midpoint
        separation = latest.midpoint - 50.0
        volume_change = max(0, latest.volume - long.volume)

        # Direction is a weighted vote. Volume strengthens the observed move;
        # it never invents a direction on its own.
        directional_move = 0.65 * short_momentum + 0.35 * long_momentum
        volume_weight = min(2.0, 1.0 + volume_change / 500.0)
        # Separation is supporting evidence, not a command to chase whichever
        # side is already expensive. Fresh momentum remains the primary vote.
        score = directional_move * volume_weight + 0.05 * separation
        side = Side.YES if score >= 0 else Side.NO
        detail = (
            f"score={score:.2f} short={short_momentum:+.2f} long={long_momentum:+.2f} "
            f"separation={separation:+.2f} volume_delta={volume_change}"
        )
        return side, detail

    def reserved_cents(self) -> int:
        return sum(self._entry_costs.values())

    def reconcile(self, active_tickers: set[str]) -> None:
        """Release bankroll reservations after a market closes or disappears."""
        for ticker in set(self._entry_costs) - active_tickers:
            self._entry_costs.pop(ticker, None)
            self._counts.pop(ticker, None)
            self._sides.pop(ticker, None)
            self._pending.discard(ticker)

    def orders_for(self, market: Market, position: int, now: float) -> list[OrderRequest]:
        self.observe(market, now)
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

        signal = self._signal(market, now)
        self._decided.add(ticker)
        if signal is None:
            logger.warning("SKIP | ticker=%s | insufficient momentum history", ticker)
            return []
        side, detail = signal
        ask = market.yes_ask if side is Side.YES else market.no_ask
        if ask is None or not 1 <= ask <= 99:
            logger.warning("SKIP | ticker=%s | side=%s | no executable ask", ticker, side.value)
            return []
        cost = ask * self.contracts
        if self.reserved_cents() + cost > self.bankroll_cents:
            logger.warning(
                "SKIP | ticker=%s | bankroll cap | needed=%d available=%d",
                ticker,
                cost,
                self.bankroll_cents - self.reserved_cents(),
            )
            return []
        logger.warning(
            "SIGNAL | ticker=%s | side=%s | ask=%dc | %s", ticker, side.value, ask, detail
        )
        self._pending.add(ticker)
        return [self._order(ticker, Action.BUY, side, ask, self.contracts)]

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
