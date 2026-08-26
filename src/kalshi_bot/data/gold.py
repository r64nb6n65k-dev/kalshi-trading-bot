"""Resilient OANDA XAU/USD price feed for Kalshi gold markets."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from kalshi_bot.strategies.base import UnderlyingTick
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)


class GoldPriceFeed:
    """Maintain a rolling history of OANDA practice XAU/USD prices."""

    def __init__(
        self,
        base_url: str,
        price_feed_id: str,
        api_key: str = "",
        poll_interval: float = 1.0,
        history_minutes: int = 20,
    ) -> None:
        # Keep the old constructor shape so no other project file must change.
        del base_url, price_feed_id, poll_interval
        self.api_key = api_key.strip()
        self._rest_url = "https://api-fxpractice.oanda.com"
        self._stream_url = "https://stream-fxpractice.oanda.com"
        self._history = timedelta(minutes=history_minutes)
        self._ticks: deque[UnderlyingTick] = deque()
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        if not self.api_key:
            raise ValueError("OANDA API token is missing")
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

    @staticmethod
    def _parse_time(value: Any) -> datetime:
        if not isinstance(value, str):
            raise ValueError("OANDA price did not contain a timestamp")
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return timestamp

    def _parse(self, payload: dict[str, Any]) -> UnderlyingTick | None:
        if payload.get("type") != "PRICE":
            return None
        if payload.get("instrument") != "XAU_USD":
            return None
        if payload.get("status") not in (None, "tradeable"):
            return None

        bids = payload.get("bids")
        asks = payload.get("asks")
        if not isinstance(bids, list) or not bids:
            raise ValueError("OANDA price did not contain a bid")
        if not isinstance(asks, list) or not asks:
            raise ValueError("OANDA price did not contain an ask")

        bid = float(bids[0]["price"])
        ask = float(asks[0]["price"])
        if bid <= 0 or ask <= 0 or ask < bid:
            raise ValueError("OANDA returned an invalid XAU/USD quote")

        return UnderlyingTick(
            price=(bid + ask) / 2.0,
            size=1.0,
            timestamp=self._parse_time(payload.get("time")),
            source="OANDA_PRACTICE_XAUUSD_MID",
        )

    async def _account_id(self, client: httpx.AsyncClient) -> str:
        response = await client.get(f"{self._rest_url}/v3/accounts")
        response.raise_for_status()
        accounts = response.json().get("accounts")
        if not isinstance(accounts, list) or not accounts:
            raise ValueError("OANDA token has no practice account")
        account_id = accounts[0].get("id")
        if not isinstance(account_id, str) or not account_id:
            raise ValueError("OANDA response did not contain an account ID")
        return account_id

    async def _stream(self, client: httpx.AsyncClient) -> None:
        account_id = await self._account_id(client)
        url = f"{self._stream_url}/v3/accounts/{account_id}/pricing/stream"
        params = {"instruments": "XAU_USD", "snapshot": "true"}
        async with client.stream("GET", url, params=params) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if self._stopping:
                    return
                if not line.strip():
                    continue
                tick = self._parse(json.loads(line))
                if tick is not None:
                    self._append(tick)

    async def _run(self) -> None:
        delay = 1.0
        headers = {"Authorization": f"Bearer {self.api_key}"}
        timeout = httpx.Timeout(10.0, read=None)
        async with httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            trust_env=False,
        ) as client:
            while not self._stopping:
                try:
                    await self._stream(client)
                    if not self._stopping:
                        raise ConnectionError("OANDA pricing stream ended")
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "GOLD FEED ERROR | provider=OANDA retry_seconds=%.1f",
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)
