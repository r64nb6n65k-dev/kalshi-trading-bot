"""One Coinbase WebSocket feed for multiple crypto products."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from contextlib import suppress
from datetime import UTC, datetime, timedelta

import websockets

from kalshi_bot.strategies.base import UnderlyingTick
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)


class MultiCryptoPriceFeed:
    def __init__(self, ws_url: str, product_ids: tuple[str, ...], history_minutes: int = 20):
        self.ws_url = ws_url
        self.product_ids = product_ids
        self._history = timedelta(minutes=history_minutes)
        self._ticks: dict[str, deque[UnderlyingTick]] = defaultdict(deque)
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping = False
            self._task = asyncio.create_task(self._run(), name="multi-crypto-price-feed")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    def snapshot(self, product_id: str) -> tuple[UnderlyingTick, ...]:
        return tuple(self._ticks.get(product_id, ()))

    def _append(self, product_id: str, tick: UnderlyingTick) -> None:
        ticks = self._ticks[product_id]
        if ticks and int(ticks[-1].timestamp.timestamp()) == int(tick.timestamp.timestamp()):
            last = ticks[-1]
            size = last.size + tick.size
            price = (last.price * last.size + tick.price * tick.size) / size if size else tick.price
            ticks[-1] = UnderlyingTick(price, size, tick.timestamp, tick.source)
        elif not ticks or tick.timestamp >= ticks[-1].timestamp:
            ticks.append(tick)
        cutoff = tick.timestamp - self._history
        while ticks and ticks[0].timestamp < cutoff:
            ticks.popleft()

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
                product_id = str(trade.get("product_id", ""))
                if product_id not in self.product_ids:
                    continue
                try:
                    timestamp = datetime.fromisoformat(str(trade["time"]).replace("Z", "+00:00"))
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=UTC)
                    self._append(
                        product_id,
                        UnderlyingTick(
                            float(trade["price"]),
                            float(trade["size"]),
                            timestamp,
                            f"COINBASE_{product_id}",
                        ),
                    )
                except (KeyError, TypeError, ValueError):
                    logger.exception("Invalid Coinbase market-trade message")

    async def _run(self) -> None:
        delay = 1.0
        while not self._stopping:
            try:
                async with websockets.connect(
                    self.ws_url, open_timeout=10, ping_interval=20, ping_timeout=20
                ) as ws:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "subscribe",
                                "product_ids": list(self.product_ids),
                                "channel": "market_trades",
                            }
                        )
                    )
                    await ws.send(json.dumps({"type": "subscribe", "channel": "heartbeats"}))
                    logger.warning("CRYPTO PRICE FEED CONNECTED | products=%s", self.product_ids)
                    delay = 1.0
                    async for raw in ws:
                        message = json.loads(raw)
                        if isinstance(message, dict):
                            self._handle_message(message)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("CRYPTO PRICE FEED DISCONNECTED | retry=%.1fs", delay)
                await asyncio.sleep(delay)
                delay = min(30.0, delay * 2)
