"""Conservative queue-aware maker fill simulator.

A paper order starts behind the displayed quantity at its price. Only public
trade volume at that exact price consumes the simulated queue. Cancellations
ahead are deliberately ignored, so reported fills err on the conservative side.
"""

from __future__ import annotations

from dataclasses import dataclass

from kalshi_bot.scalping.book import BinaryOrderBook


@dataclass(slots=True)
class PaperOrder:
    order_id: str
    outcome: str
    action: str
    price: int
    remaining: int
    queue_ahead: int
    created_ns: int


@dataclass(slots=True)
class PaperFill:
    order_id: str
    outcome: str
    action: str
    price: int
    count: int


class QueuePaperBroker:
    def __init__(self) -> None:
        self.orders: dict[str, PaperOrder] = {}
        self._counter = 0

    def place(
        self,
        book: BinaryOrderBook,
        outcome: str,
        action: str,
        price: int,
        count: int,
        now_ns: int,
    ) -> PaperOrder:
        self._counter += 1
        order_id = f"paper-maker-{self._counter}"
        # BUY outcome orders rest on that outcome's bid book.  SELL outcome
        # orders rest as bids on the opposite outcome at the complementary price.
        level_outcome = outcome if action == "buy" else ("no" if outcome == "yes" else "yes")
        level_price = price if action == "buy" else 100 - price
        queue = book.bid_size(level_outcome, level_price)
        order = PaperOrder(order_id, outcome, action, price, count, queue, now_ns)
        self.orders[order_id] = order
        return order

    def cancel(self, order_id: str) -> None:
        self.orders.pop(order_id, None)

    def on_trade(
        self,
        yes_price: int,
        no_price: int,
        count: int,
        taker_side: str,
    ) -> list[PaperFill]:
        fills: list[PaperFill] = []
        for order in list(self.orders.values()):
            # Aggressive NO orders consume YES bids; aggressive YES orders
            # consume NO bids / YES asks.
            maker_is_hit = (
                (order.outcome == "yes" and order.action == "buy" and taker_side == "no")
                or (order.outcome == "yes" and order.action == "sell" and taker_side == "yes")
                or (order.outcome == "no" and order.action == "buy" and taker_side == "yes")
                or (order.outcome == "no" and order.action == "sell" and taker_side == "no")
            )
            trade_price = yes_price if order.outcome == "yes" else no_price
            if not maker_is_hit or order.price != trade_price:
                continue
            ahead_consumed = min(order.queue_ahead, count)
            order.queue_ahead -= ahead_consumed
            available = count - ahead_consumed
            if available <= 0:
                continue
            fill_count = min(order.remaining, available)
            order.remaining -= fill_count
            fills.append(
                PaperFill(
                    order.order_id,
                    order.outcome,
                    order.action,
                    order.price,
                    fill_count,
                )
            )
            if order.remaining == 0:
                self.orders.pop(order.order_id, None)
        return fills
