"""Resilient XAU/USD price feed for Kalshi gold markets."""

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
    """Maintain a rolling history of the Gold-API XAU/USD spot feed."""

    def __init__(
        self,
        base_url: str,
        price_feed_id: str,
        api_key: str = "",
        poll_interval: float = 1.0,
        history_minutes: int = 20,
    ) -> None:
        # Preserve compatibility with the existing CLI configuration.
        del base_url, price_feed_id, api_key

        self.base_url = "https://api.gold-api.com"
        self.poll_interval = poll_interval
        self._history = timedelta(minutes=history_minutes)
        self._ticks: deque[UnderlyingTick] = deque()
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping = False
            self._task = asyncio.create_task(
                self._run(),
                name="gold-price-feed",
            )

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
        if payload.get("symbol") != "XAU":
            raise ValueError("Gold API response was not XAU")

        if payload.get("currency") != "USD":
            raise ValueError("Gold API response was not denominated in USD")

        price = float(payload["price"])
        if price <= 0:
            raise ValueError("Gold API returned a non-positive price")

        updated_at = payload.get("updatedAt")
        if not isinstance(updated_at, str):
            raise ValueError(
                "Gold API response did not contain an update timestamp"
            )

        timestamp = datetime.fromisoformat(
            updated_at.replace("Z", "+00:00")
        )
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        return UnderlyingTick(
            price=price,
            size=1.0,
            timestamp=timestamp,
            source="GOLD_API_XAUUSD_SPOT",
        )

    async def _run(self) -> None:
        delay = 1.0
        url = f"{self.base_url}/price/XAU"

        async with httpx.AsyncClient(
            timeout=10.0,
            trust_env=False,
        ) as client:
            while not self._stopping:
                try:
                    response = await client.get(url)
                    response.raise_for_status()

                    tick = self._parse(response.json())
                    self._append(tick)

                    delay = 1.0
                    await asyncio.sleep(self.poll_interval)

                except asyncio.CancelledError:
                    raise

                except Exception:
                    logger.exception(
                        "GOLD FEED ERROR | retry_seconds=%.1f",
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)
