"""Resilient Pyth XAU/USD price feed for Kalshi gold markets."""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from kalshi_bot.strategies.base import UnderlyingTick
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)


class GoldPriceFeed:
    """Maintain a rolling history of the Pyth XAU/USD settlement feed."""

    def __init__(
        self,
        base_url: str,
        price_feed_id: str,
        api_key: str = "",
        poll_interval: float = 1.0,
        history_minutes: int = 20,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.price_feed_id = price_feed_id.removeprefix("0x")
        self.api_key = api_key.strip()
        self.poll_interval = poll_interval
        self._history = timedelta(minutes=history_minutes)
        self._ticks: deque[UnderlyingTick] = deque()
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping = False
            self._task = asyncio.create_task(self._run(), name="gold-price-feed")

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
        if self._ticks and tick.timestamp < self._ticks[-1].timestamp:
            return
        if self._ticks and tick.timestamp == self._ticks[-1].timestamp:
            self._ticks[-1] = tick
            return
        self._ticks.append(tick)
        cutoff = tick.timestamp - self._history
        while self._ticks and self._ticks[0].timestamp < cutoff:
            self._ticks.popleft()

    def _parse(self, payload: dict[str, Any]) -> UnderlyingTick:
        parsed = payload.get("parsed")
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("Pyth response did not contain a parsed price")
        price_data = parsed[0].get("price")
        if not isinstance(price_data, dict):
            raise ValueError("Pyth response did not contain price data")
        price = float(price_data["price"]) * (10.0 ** int(price_data["expo"]))
        timestamp = datetime.fromtimestamp(int(price_data["publish_time"]), tz=UTC)
        return UnderlyingTick(
            price=price,
            size=1.0,
            timestamp=timestamp,
            source="PYTH_XAUUSD_SETTLEMENT_SOURCE",
        )

    async def _run(self) -> None:
        delay = 1.0
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        url = f"{self.base_url}/v2/updates/price/latest"
        params = [("ids[]", self.price_feed_id)]
        async with httpx.AsyncClient(
            timeout=10.0,
            headers=headers,
            trust_env=False,
        ) as client:
            while not self._stopping:
                try:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    self._append(self._parse(response.json()))
                    delay = 1.0
                    await asyncio.sleep(self.poll_interval)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("GOLD FEED ERROR | retry_seconds=%.1f", delay)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)
