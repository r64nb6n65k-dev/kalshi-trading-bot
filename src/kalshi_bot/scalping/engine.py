"""Event-driven BTC 15-minute order-book recorder and paper scalper."""

from __future__ import annotations

import asyncio
import csv
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kalshi_bot.exchange.client import KalshiClient
from kalshi_bot.exchange.models import Market
from kalshi_bot.exchange.websocket import KalshiWebSocket
from kalshi_bot.scalping.book import BinaryOrderBook, contracts, dollars_to_cents
from kalshi_bot.scalping.paper import PaperFill, QueuePaperBroker
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ScalpConfig:
    quote_size: int = 10
    max_inventory: int = 20
    min_spread_cents: int = 1
    profit_cents: int = 1
    min_top_depth: int = 20
    max_seconds_left: int = 720
    min_seconds_left: int = 120
    reprice_seconds: float = 5.0
    output_csv: str = "btc_orderbook_data_v2.csv"


class BtcOrderBookScalpingEngine:
    """Runs a deliberately conservative dry-run maker simulation.

    Live mode is intentionally refused in this first build.  The existing
    general engine cannot reconcile asynchronous maker fills safely.  Enabling
    live trading before recorded queue statistics validate the strategy would
    turn paper assumptions into real losses.
    """

    def __init__(
        self,
        client: KalshiClient,
        websocket: KalshiWebSocket,
        config: ScalpConfig | None = None,
        dry_run: bool = True,
        series_ticker: str = "KXBTC15M",
    ) -> None:
        if not dry_run:
            raise RuntimeError(
                "BTC order-book scalper live mode is locked until queue-aware "
                "dry-run validation passes"
            )
        self.client = client
        self.websocket = websocket
        self.config = config or ScalpConfig()
        self.series_ticker = series_ticker
        self.broker = QueuePaperBroker()
        self.inventory = {"yes": 0, "no": 0}
        self.cost_cents = {"yes": 0, "no": 0}
        self.realized_cents = 0
        self._quote_ids: dict[tuple[str, str], str] = {}
        self._quote_time: dict[tuple[str, str], float] = {}
        self._writer: csv.DictWriter[str] | None = None
        self._file: Any = None

    async def resolve_market(self) -> Market:
        markets = await self.client.get_markets(
            status="open", series_ticker=self.series_ticker, limit=100
        )
        if not markets:
            raise RuntimeError(f"No open {self.series_ticker} market found")
        return min(markets, key=lambda m: m.close_time or "")

    @staticmethod
    def seconds_left(market: Market) -> float:
        if not market.close_time:
            return 0.0
        close = datetime.fromisoformat(market.close_time.replace("Z", "+00:00"))
        return max(0.0, (close - datetime.now(timezone.utc)).total_seconds())

    def _open_csv(self) -> None:
        path = Path(self.config.output_csv)
        exists = path.exists() and path.stat().st_size > 0
        self._file = path.open("a", newline="", encoding="utf-8")
        fields = [
            "timestamp", "ticker", "seconds_left", "yes_bid", "yes_ask",
            "yes_bid_size", "yes_ask_size", "spread", "yes_inventory",
            "no_inventory", "realized_cents", "marked_total_cents", "event",
            "detail",
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=fields)
        if not exists:
            self._writer.writeheader()

    def _record(
        self,
        book: BinaryOrderBook,
        seconds_left: float,
        event: str,
        detail: str = "",
    ) -> None:
        assert self._writer is not None and self._file is not None
        self._writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": book.ticker,
            "seconds_left": round(seconds_left, 3),
            "yes_bid": book.yes_bid,
            "yes_ask": book.yes_ask,
            "yes_bid_size": book.bid_size("yes"),
            "yes_ask_size": book.ask_size("yes"),
            "spread": book.spread,
            "yes_inventory": self.inventory["yes"],
            "no_inventory": self.inventory["no"],
            "realized_cents": self.realized_cents,
            "marked_total_cents": self._marked_total_cents(book),
            "event": event,
            "detail": detail,
        })
        self._file.flush()

    def _cancel_key(self, key: tuple[str, str]) -> None:
        order_id = self._quote_ids.pop(key, None)
        self._quote_time.pop(key, None)
        if order_id:
            self.broker.cancel(order_id)

    def _cancel_all(self) -> None:
        for key in list(self._quote_ids):
            self._cancel_key(key)

    def _place(
        self,
        book: BinaryOrderBook,
        outcome: str,
        action: str,
        price: int,
        count: int,
    ) -> None:
        key = (outcome, action)
        current_id = self._quote_ids.get(key)
        current = self.broker.orders.get(current_id or "")
        now = time.monotonic()
        quote_age = now - self._quote_time.get(key, now)
        if current and current.price == price and quote_age < self.config.reprice_seconds:
            return
        self._cancel_key(key)
        order = self.broker.place(book, outcome, action, price, count, time.time_ns())
        self._quote_ids[key] = order.order_id
        self._quote_time[key] = now

    def _quote(self, book: BinaryOrderBook, seconds_left: float) -> None:
        cfg = self.config
        if (
            not book.ready
            or book.spread is None
            or book.spread < cfg.min_spread_cents
            or not cfg.min_seconds_left < seconds_left <= cfg.max_seconds_left
            or book.yes_bid is None
            or book.yes_ask is None
            or book.bid_size("yes") < cfg.min_top_depth
            or book.bid_size("no") < cfg.min_top_depth
        ):
            self._cancel_all()
            return

        for outcome in ("yes", "no"):
            held = self.inventory[outcome]
            if held > 0:
                entry_average = self.cost_cents[outcome] // held
                exit_price = min(99, entry_average + cfg.profit_cents)
                # Never cross the book in the simulator; this measures maker exits.
                self._place(book, outcome, "sell", exit_price, held)
                self._cancel_key((outcome, "buy"))
            elif sum(self.inventory.values()) < cfg.max_inventory:
                self._cancel_key((outcome, "sell"))
                bid = book.bid(outcome)
                if bid is not None:
                    self._place(book, outcome, "buy", bid, cfg.quote_size)

    def _apply_fill(self, fill: PaperFill) -> None:
        key = (fill.outcome, fill.action)
        if fill.action == "buy":
            opposite = "no" if fill.outcome == "yes" else "yes"
            opposite_held = self.inventory[opposite]
            paired = min(fill.count, opposite_held)
            if paired:
                opposite_avg = self.cost_cents[opposite] / opposite_held
                self.realized_cents += round(
                    (100 - fill.price - opposite_avg) * paired
                )
                self.inventory[opposite] -= paired
                self.cost_cents[opposite] -= round(opposite_avg * paired)
                if self.inventory[opposite] == 0:
                    self.cost_cents[opposite] = 0
                    self._cancel_key((opposite, "sell"))
            unpaired = fill.count - paired
            self.inventory[fill.outcome] += unpaired
            self.cost_cents[fill.outcome] += fill.price * unpaired
        else:
            held = self.inventory[fill.outcome]
            closed = min(held, fill.count)
            avg = self.cost_cents[fill.outcome] / held if held else 0.0
            self.realized_cents += round((fill.price - avg) * closed)
            self.inventory[fill.outcome] -= closed
            self.cost_cents[fill.outcome] -= round(avg * closed)
            if self.inventory[fill.outcome] == 0:
                self.cost_cents[fill.outcome] = 0
        if fill.order_id not in self.broker.orders:
            self._quote_ids.pop(key, None)
            self._quote_time.pop(key, None)

    def _marked_total_cents(self, book: BinaryOrderBook) -> int:
        """Return realized P&L plus executable bid value of open inventory."""
        total = self.realized_cents
        for outcome in ("yes", "no"):
            held = self.inventory[outcome]
            if held <= 0:
                continue
            exit_bid = book.bid(outcome)
            if exit_bid is None:
                exit_bid = 0
            total += (exit_bid * held) - self.cost_cents[outcome]
        return total

    async def run(self, max_events: int | None = None) -> None:
        market = await self.resolve_market()
        book = BinaryOrderBook(market.ticker)
        self._open_csv()
        events = 0
        logger.warning(
            "BTC SCALPER PAPER MODE | ticker=%s | queue-aware fills | live locked",
            market.ticker,
        )
        try:
            async for message in self.websocket.stream(
                ["orderbook_delta", "trade"],
                [market.ticker],
                subscription_params={"use_yes_price": False},
            ):
                change = book.apply(message)
                remaining = self.seconds_left(market)
                if message.get("type") == "orderbook_snapshot":
                    self._record(book, remaining, "SNAPSHOT")
                fills: list[PaperFill] = []
                if message.get("type") == "trade":
                    payload = message.get("msg") or {}
                    fills = self.broker.on_trade(
                        yes_price=dollars_to_cents(payload.get("yes_price_dollars", 0)),
                        no_price=dollars_to_cents(payload.get("no_price_dollars", 0)),
                        count=contracts(payload.get("count_fp", 0)),
                        taker_side=str(payload.get("taker_side", "")).lower(),
                    )
                for fill in fills:
                    self._apply_fill(fill)
                    detail = (
                        f"{fill.action} {fill.count} {fill.outcome.upper()} "
                        f"@ {fill.price}c"
                    )
                    logger.warning("PAPER MAKER FILL | %s", detail)
                    self._record(book, remaining, "PAPER_FILL", detail)
                if change is not None:
                    self._quote(book, remaining)
                    self._record(
                        book,
                        remaining,
                        "BOOK_DELTA",
                        f"{change.side} {change.price}c {change.delta:+d}",
                    )
                    events += 1
                elif fills:
                    self._quote(book, remaining)
                if remaining <= 0 or (max_events is not None and events >= max_events):
                    break
        finally:
            self._cancel_all()
            marked_total_cents = self._marked_total_cents(book)
            if self._file is not None:
                self._record(
                    book,
                    self.seconds_left(market),
                    "FINAL_MARK",
                    f"marked_total_cents={marked_total_cents}",
                )
                self._file.close()
            logger.warning(
                "BTC SCALPER STOPPED | realized=$%.2f | marked_total=$%.2f | "
                "YES=%d | NO=%d | events=%d",
                self.realized_cents / 100.0,
                marked_total_cents / 100.0,
                self.inventory["yes"],
                self.inventory["no"],
                events,
            )
