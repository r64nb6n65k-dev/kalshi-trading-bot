"""Resilient public Coinbase BNB/USD trade stream for strategy filtering."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from contextlib import suppress
from datetime import UTC, datetime, timedelta

import websockets

from kalshi_bot.strategies.base import UnderlyingTick
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)


class BnbPriceFeed:
    """Maintain a rolling window of public Coinbase BNB/USD trades.

    Coinbase BNB-USD is one of the constituent markets used by CF Benchmarks
    for BNBUSD_RTI. It is still only one constituent, not the licensed RTI
    itself, so missing or stale data remains a hard entry veto.
    """

    def __init__(
        self,
        ws_url: str,
        product_id: str = "BNB-USD",
        history_minutes: int = 20,
    ) -> None:
        self.ws_url = ws_url
        self.product_id = product_id
        self._history = timedelta(minutes=history_minutes)
        self._ticks: deque[UnderlyingTick] = deque()
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping = False
            self._task = asyncio.create_task(self._run(), name="bnb-price-feed")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    def snapshot(self) -> tuple[UnderlyingTick, ...]:
        return tuple(self._ticks)

    def _append(self, tick: UnderlyingTick) -> None:
        if self._ticks:
            last = self._ticks[-1]
            last_second = int(last.timestamp.timestamp())
            tick_second = int(tick.timestamp.timestamp())
            if tick_second < last_second:
                return
            if tick_second == last_second:
                combined_size = last.size + tick.size
                combined_price = (
                    (last.price * last.size + tick.price * tick.size) / combined_size
                    if combined_size > 0
                    else tick.price
                )
                self._ticks[-1] = UnderlyingTick(
                    price=combined_price,
                    size=combined_size,
                    timestamp=tick.timestamp,
                    source=tick.source,
                )
                return
        self._ticks.append(tick)
        cutoff = tick.timestamp - self._history
        while self._ticks and self._ticks[0].timestamp < cutoff:
            self._ticks.popleft()

    def _handle_message(self, message: dict[str, object]) -> None:
        if message.get("channel") != "market_trades":
            return
        events = message.get("events")
        if not isinstance(events, list):
            return
        for event in events:
            if not isinstance(event, dict):
                continue
            trades = event.get("trades")
            if not isinstance(trades, list):
                continue
            for trade in trades:
                if not isinstance(trade, dict):
                    continue
                if trade.get("product_id") != self.product_id:
                    continue
                try:
                    timestamp = datetime.fromisoformat(
                        str(trade["time"]).replace("Z", "+00:00")
                    )
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=UTC)
                    self._append(
                        UnderlyingTick(
                            price=float(trade["price"]),
                            size=float(trade["size"]),
                            timestamp=timestamp,
                            source="COINBASE_BNBUSD_CONSTITUENT",
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    logger.exception("Invalid BNB market-trade message")

    async def _run(self) -> None:
        delay = 1.0
        while not self._stopping:
            try:
                async with websockets.connect(
                    self.ws_url,
                    open_timeout=10,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "subscribe",
                                "product_ids": [self.product_id],
                                "channel": "market_trades",
                            }
                        )
                    )
                    await ws.send(
                        json.dumps({"type": "subscribe", "channel": "heartbeats"})
                    )
                    logger.warning(
                        "BNB FEED CONNECTED | source=COINBASE_BNBUSD_CONSTITUENT | product=%s",
                        self.product_id,
                    )
                    delay = 1.0
                    async for raw in ws:
                        message = json.loads(raw)
                        if isinstance(message, dict):
                            self._handle_message(message)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "BNB FEED DISCONNECTED | retry_seconds=%.1f",
                    delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
