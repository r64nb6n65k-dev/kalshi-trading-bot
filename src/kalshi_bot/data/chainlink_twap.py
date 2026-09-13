"""Polymarket's public Chainlink 60-second TWAP stream.

These are the reference prices used by the rolling crypto markets.  The
stream is available through Polymarket RTDS without Chainlink credentials.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation

import websockets

from kalshi_bot.strategies.base import UnderlyingTick
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)


class ChainlinkTwapFeed:
    """Maintain exact Chainlink 60-second TWAP updates for supported assets."""

    def __init__(
        self,
        product_ids: tuple[str, ...],
        *,
        ws_url: str = "wss://ws-live-data.polymarket.com",
        history_minutes: int = 30,
    ) -> None:
        self.ws_url = ws_url
        self.product_ids = tuple(dict.fromkeys(product_ids))
        self._symbol_to_product = {
            product.lower().replace("-", "/"): product for product in self.product_ids
        }
        self._history = timedelta(minutes=history_minutes)
        self._ticks: dict[str, deque[UnderlyingTick]] = defaultdict(deque)
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping = False
            self._task = asyncio.create_task(self._run(), name="chainlink-twap-feed")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    def snapshot(self, product_id: str) -> tuple[UnderlyingTick, ...]:
        return tuple(self._ticks.get(product_id, ()))

    def opening_reference(
        self,
        product_id: str,
        open_time: float,
        *,
        tolerance_seconds: float = 5.0,
    ) -> float | None:
        """Return the first Chainlink observation at/just after a market opens."""
        ticks = self.snapshot(product_id)
        candidates = [
            tick
            for tick in ticks
            if open_time - 0.5 <= tick.timestamp.timestamp() <= open_time + tolerance_seconds
        ]
        if not candidates:
            return None
        # The resolution comparison uses the observation at the boundary.  An
        # update just after the boundary is preferable to a pre-open update.
        after = [tick for tick in candidates if tick.timestamp.timestamp() >= open_time]
        chosen = min(
            after or candidates, key=lambda tick: abs(tick.timestamp.timestamp() - open_time)
        )
        return chosen.price

    def _append(self, product_id: str, tick: UnderlyingTick) -> None:
        ticks = self._ticks[product_id]
        if ticks and tick.timestamp < ticks[-1].timestamp:
            return
        if ticks and tick.timestamp == ticks[-1].timestamp:
            ticks[-1] = tick
        else:
            ticks.append(tick)
        cutoff = tick.timestamp - self._history
        while ticks and ticks[0].timestamp < cutoff:
            ticks.popleft()

    def _handle_message(self, raw: str) -> None:
        if raw == "PONG":
            return
        try:
            message = json.loads(raw)
        except (TypeError, ValueError):
            return
        if not isinstance(message, dict) or message.get("type") != "update":
            return
        if message.get("topic") not in {
            "crypto_prices_twap_sixty",
            "prices.crypto.chainlink.twap",
        }:
            return
        payload = message.get("payload")
        if not isinstance(payload, dict):
            return
        symbol = str(payload.get("symbol", "")).lower()
        product_id = self._symbol_to_product.get(symbol)
        if product_id is None:
            return
        try:
            value = Decimal(str(payload["value"]))
            timestamp_ms = int(payload["timestamp"])
            if value <= 0:
                return
        except (KeyError, InvalidOperation, TypeError, ValueError):
            logger.warning("INVALID CHAINLINK TWAP UPDATE | payload=%r", payload)
            return
        self._append(
            product_id,
            UnderlyingTick(
                price=float(value),
                size=1.0,
                timestamp=datetime.fromtimestamp(timestamp_ms / 1000, UTC),
                source="CHAINLINK_TWAP_60S",
            ),
        )

    async def _heartbeat(self, ws: object) -> None:
        while True:
            await asyncio.sleep(5)
            await ws.send("PING")  # type: ignore[attr-defined]

    async def _run(self) -> None:
        delay = 1.0
        subscription = {
            "action": "subscribe",
            "subscriptions": [{"topic": "crypto_prices_twap_sixty", "type": "update"}],
        }
        while not self._stopping:
            try:
                async with websockets.connect(
                    self.ws_url,
                    open_timeout=10,
                    ping_interval=None,
                    close_timeout=5,
                ) as ws:
                    await ws.send(json.dumps(subscription, separators=(",", ":")))
                    heartbeat = asyncio.create_task(self._heartbeat(ws))
                    logger.warning(
                        "CHAINLINK TWAP FEED CONNECTED | products=%s | window=60s",
                        self.product_ids,
                    )
                    delay = 1.0
                    try:
                        async for raw in ws:
                            if isinstance(raw, str):
                                self._handle_message(raw)
                    finally:
                        heartbeat.cancel()
                        with suppress(asyncio.CancelledError):
                            await heartbeat
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("CHAINLINK TWAP FEED DISCONNECTED | retry=%.1fs", delay)
                await asyncio.sleep(delay)
                delay = min(30.0, delay * 2)
