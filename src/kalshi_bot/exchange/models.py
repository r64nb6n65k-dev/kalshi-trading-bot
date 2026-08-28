"""Typed data models for Kalshi API objects.

Compatibility build for the current Kalshi V2 API. It preserves the original
framework's integer-cent / whole-contract interface while mapping current V2
fixed-point and dollar fields into those legacy fields.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


def _dollars_to_cents(value: Any) -> int:
    if value is None:
        return 0
    return int(
        (Decimal(str(value)) * 100).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )


def _fp_to_int(value: Any) -> int:
    if value is None:
        return 0
    return int(
        Decimal(str(value)).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )


class Side(str, Enum):
    YES = "yes"
    NO = "no"


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class TimeInForce(str, Enum):
    GOOD_TILL_CANCELED = "good_till_canceled"
    IMMEDIATE_OR_CANCEL = "immediate_or_cancel"
    FILL_OR_KILL = "fill_or_kill"


class Market(BaseModel):
    ticker: str
    event_ticker: str | None = None
    title: str | None = None
    status: str | None = None
    yes_bid: int | None = None
    yes_ask: int | None = None
    no_bid: int | None = None
    no_ask: int | None = None
    last_price: int | None = None
    volume: int | None = None
    open_interest: int | None = None
    close_time: str | None = None
    floor_strike: float | None = None
    rules_primary: str | None = None
    yes_bid_size: int | None = None
    yes_ask_size: int | None = None
    exchange_index: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _map_v2_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        d = dict(data)

        # Current Kalshi V2 market responses expose prices as dollar strings
        # such as "0.9300". Preserve any legacy integer-cent field if it is
        # already present; otherwise map the V2 dollar field into cents.
        for legacy, v2 in (
            ("yes_bid", "yes_bid_dollars"),
            ("yes_ask", "yes_ask_dollars"),
            ("no_bid", "no_bid_dollars"),
            ("no_ask", "no_ask_dollars"),
            ("last_price", "last_price_dollars"),
        ):
            if d.get(legacy) is None and d.get(v2) is not None:
                d[legacy] = _dollars_to_cents(d[v2])

        if d.get("volume") is None and d.get("volume_fp") is not None:
            d["volume"] = _fp_to_int(d["volume_fp"])

        if d.get("open_interest") is None and d.get("open_interest_fp") is not None:
            d["open_interest"] = _fp_to_int(d["open_interest_fp"])

        if d.get("yes_bid_size") is None and d.get("yes_bid_size_fp") is not None:
            d["yes_bid_size"] = _fp_to_int(d["yes_bid_size_fp"])

        if d.get("yes_ask_size") is None and d.get("yes_ask_size_fp") is not None:
            d["yes_ask_size"] = _fp_to_int(d["yes_ask_size_fp"])

        return d


class OrderRequest(BaseModel):
    ticker: str
    action: Action
    side: Side
    count: int = Field(gt=0)
    type: OrderType = OrderType.LIMIT
    yes_price: int | None = Field(default=None, ge=1, le=99)
    no_price: int | None = Field(default=None, ge=1, le=99)
    time_in_force: TimeInForce | None = None
    client_order_id: str | None = None
    post_only: bool | None = None
    exchange_index: int | None = None

    def to_payload(self) -> dict[str, object]:
        return self.model_dump(exclude_none=True, mode="json")


class Order(BaseModel):
    order_id: str
    ticker: str
    status: str | None = None
    side: Side | None = None
    action: Action | None = None
    yes_price: int | None = None
    no_price: int | None = None
    count: int | None = None
    remaining_count: int | None = None
    client_order_id: str | None = None
    fill_count: int = 0
    average_fill_price: str | None = None
    outcome_fill_price: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _map_v2_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)

        if "remaining_count" not in d and "remaining_count_fp" in d:
            d["remaining_count"] = _fp_to_int(d.get("remaining_count_fp"))
        if "count" not in d and "initial_count_fp" in d:
            d["count"] = _fp_to_int(d.get("initial_count_fp"))
        if "fill_count" not in d and "fill_count_fp" in d:
            d["fill_count"] = _fp_to_int(d.get("fill_count_fp"))

        if "yes_price" not in d and d.get("yes_price_dollars") is not None:
            d["yes_price"] = _dollars_to_cents(d["yes_price_dollars"])
        if "no_price" not in d and d.get("no_price_dollars") is not None:
            d["no_price"] = _dollars_to_cents(d["no_price_dollars"])

        return d


class Position(BaseModel):
    """Signed position: positive YES, negative NO."""

    ticker: str
    position: int = 0
    market_exposure: int = 0
    realized_pnl: int = 0
    avg_price: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _map_v2_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)

        if "position" not in d and "position_fp" in d:
            d["position"] = _fp_to_int(d.get("position_fp"))

        if "market_exposure" not in d and d.get("market_exposure_dollars") is not None:
            d["market_exposure"] = _dollars_to_cents(d["market_exposure_dollars"])

        if "realized_pnl" not in d and d.get("realized_pnl_dollars") is not None:
            d["realized_pnl"] = _dollars_to_cents(d["realized_pnl_dollars"])

        return d

    @property
    def is_flat(self) -> bool:
        return self.position == 0

    def unrealized_pnl(self, mark_price: int) -> int:
        return round((mark_price - self.avg_price) * self.position)


class Fill(BaseModel):
    ticker: str
    side: Side
    action: Action
    count: int = Field(gt=0)
    price: int = Field(ge=1, le=99)
    order_id: str | None = None
    trade_id: str | None = None

    @property
    def notional_cents(self) -> int:
        return self.price * self.count

    @property
    def signed_count(self) -> int:
        return self.count if self.action is Action.BUY else -self.count


class OrderBookLevel(BaseModel):
    price: int = Field(ge=1, le=99)
    count: int = Field(ge=0)


class OrderBook(BaseModel):
    ticker: str
    yes: list[OrderBookLevel] = Field(default_factory=list)
    no: list[OrderBookLevel] = Field(default_factory=list)

    @property
    def best_yes_bid(self) -> int | None:
        return max((lvl.price for lvl in self.yes), default=None)

    @property
    def best_yes_ask(self) -> int | None:
        best_no = max((lvl.price for lvl in self.no), default=None)
        return None if best_no is None else 100 - best_no

    @property
    def mid_price(self) -> float | None:
        bid, ask = self.best_yes_bid, self.best_yes_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    @property
    def spread(self) -> int | None:
        bid, ask = self.best_yes_bid, self.best_yes_ask
        if bid is None or ask is None:
            return None
        return ask - bid


class Balance(BaseModel):
    balance: int = 0

    @property
    def balance_dollars(self) -> float:
        return self.balance / 100.0


class Portfolio(BaseModel):
    balance_cents: int = 0
    positions: list[Position] = Field(default_factory=list)

    def position_for(self, ticker: str) -> Position | None:
        for pos in self.positions:
            if pos.ticker == ticker:
                return pos
        return None

    def market_value_cents(self, marks: dict[str, int]) -> int:
        equity = self.balance_cents
        for pos in self.positions:
            mark = marks.get(pos.ticker)
            if mark is not None:
                equity += pos.position * mark
        return equity
