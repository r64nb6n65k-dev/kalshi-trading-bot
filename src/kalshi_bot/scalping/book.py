"""In-memory Kalshi binary order book.

Kalshi publishes bids for YES and NO.  A NO bid at N cents is the same
economic price level as a YES ask at 100-N cents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any


def dollars_to_cents(value: Any) -> int:
    return int(
        (Decimal(str(value)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )


def contracts(value: Any) -> int:
    return max(0, int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)))


@dataclass(slots=True)
class BookChange:
    side: str
    price: int
    delta: int
    old_count: int
    new_count: int


@dataclass(slots=True)
class BinaryOrderBook:
    ticker: str
    yes: dict[int, int] = field(default_factory=dict)
    no: dict[int, int] = field(default_factory=dict)
    sequence: int = 0
    ready: bool = False

    def apply(self, message: dict[str, Any]) -> BookChange | None:
        kind = message.get("type")
        payload = message.get("msg") or {}
        if payload.get("market_ticker") not in (None, self.ticker):
            return None

        seq = int(message.get("seq") or self.sequence)
        if kind == "orderbook_snapshot":
            self.yes = self._levels(payload.get("yes_dollars_fp") or payload.get("yes") or [])
            self.no = self._levels(payload.get("no_dollars_fp") or payload.get("no") or [])
            self.sequence = seq
            self.ready = True
            return None

        if kind != "orderbook_delta" or not self.ready:
            return None
        if seq and self.sequence and seq <= self.sequence:
            return None

        side = str(payload.get("side", "")).lower()
        if side not in {"yes", "no"}:
            return None
        price_raw = payload.get("price_dollars")
        if price_raw is None:
            price_raw = payload.get("price")
        if price_raw is None:
            return None
        price = dollars_to_cents(price_raw) if Decimal(str(price_raw)) <= 1 else int(price_raw)
        delta_raw = payload.get("delta_fp")
        if delta_raw is None:
            delta_raw = payload.get("delta", 0)
        delta = int(Decimal(str(delta_raw)))

        levels = self.yes if side == "yes" else self.no
        old = levels.get(price, 0)
        new = max(0, old + delta)
        if new:
            levels[price] = new
        else:
            levels.pop(price, None)
        self.sequence = seq
        return BookChange(side=side, price=price, delta=delta, old_count=old, new_count=new)

    @staticmethod
    def _levels(raw: list[Any]) -> dict[int, int]:
        result: dict[int, int] = {}
        for price_raw, count_raw in raw:
            price = dollars_to_cents(price_raw) if Decimal(str(price_raw)) <= 1 else int(price_raw)
            count = contracts(count_raw)
            if 0 < price < 100 and count > 0:
                result[price] = count
        return result

    def bid(self, outcome: str) -> int | None:
        levels = self.yes if outcome == "yes" else self.no
        return max(levels, default=None)

    def ask(self, outcome: str) -> int | None:
        opposite = self.no if outcome == "yes" else self.yes
        best = max(opposite, default=None)
        return None if best is None else 100 - best

    def bid_size(self, outcome: str, price: int | None = None) -> int:
        levels = self.yes if outcome == "yes" else self.no
        selected = self.bid(outcome) if price is None else price
        return 0 if selected is None else levels.get(selected, 0)

    def ask_size(self, outcome: str) -> int:
        opposite = "no" if outcome == "yes" else "yes"
        opposite_bid = self.bid(opposite)
        return self.bid_size(opposite, opposite_bid)

    @property
    def yes_bid(self) -> int | None:
        return self.bid("yes")

    @property
    def yes_ask(self) -> int | None:
        return self.ask("yes")

    @property
    def spread(self) -> int | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return self.yes_ask - self.yes_bid
