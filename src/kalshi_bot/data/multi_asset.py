"""Live price and activity feeds for Kalshi 15-minute commodity markets."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import httpx
import websockets

from kalshi_bot.config import Settings
from kalshi_bot.exchange.websocket import KalshiWebSocket
from kalshi_bot.strategies.base import UnderlyingTick
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)


class MultiAssetPriceFeed:
    """Use OANDA for commodities with Kalshi underlying streams as fallback."""

    CRYPTO_PRODUCTS: ClassVar[tuple[str, ...]] = (
        "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "XRP-USD",
        "ADA-USD", "BCH-USD", "BNB-USD", "HYPE-USD", "NEAR-USD",
        "TON-USD", "ZEC-USD",
    )
    OANDA_PRODUCTS: ClassVar[tuple[str, ...]] = (
        "XAU_USD", "XAG_USD", "XCU_USD", "WTICO_USD", "BCO_USD", "NATGAS_USD",
    )
    ALIASES: ClassVar[dict[str, str]] = {
        "GOLD": "XAU_USD", "XAU": "XAU_USD",
        "SILVER": "XAG_USD", "XAG": "XAG_USD",
        "COPPER": "XCU_USD", "XCU": "XCU_USD",
        "OIL": "WTICO_USD", "CRUDE": "WTICO_USD", "WTI": "WTICO_USD",
        "WTICO": "WTICO_USD", "CL": "WTICO_USD",
        "BRENT": "BCO_USD", "BCO": "BCO_USD", "NATGAS": "NATGAS_USD",
        "CU": "XCU_USD", "PYTHOIL": "WTICO_USD",
        "PALLADIUM": "XPD_USD", "XPD": "XPD_USD",
        "PLATINUM": "XPT_USD", "XPT": "XPT_USD",
        "EUR": "EURUSD-USD", "EURUSD": "EURUSD-USD",
        "GBP": "GBPUSD-USD", "GBPUSD": "GBPUSD-USD",
        "USD_JPY": "USDJPY-USD", "USDJPY": "USDJPY-USD",
        "INX": "INX-USD", "SPX": "INX-USD", "SP500": "INX-USD",
        "US500": "INX-USD",
        "NDQ": "NDQ-USD", "NDX": "NDQ-USD", "NASDAQ100": "NDQ-USD",
        "US100": "NDQ-USD",
    }

    def __init__(self, settings: Settings, history_minutes: int = 20) -> None:
        self._coinbase_url = settings.btc_ws_url
        self._oanda_token = settings.pyth_api_key.strip()
        self._kalshi_ws = KalshiWebSocket.from_settings(settings)
        self._history = timedelta(minutes=history_minutes)
        self._primary: dict[str, deque[UnderlyingTick]] = defaultdict(deque)
        self._fallback: dict[str, deque[UnderlyingTick]] = defaultdict(deque)
        self._tasks: list[asyncio.Task[None]] = []
        self._stopping = False
        self._pyth_products_seen: set[str] = set()

    @classmethod
    def _key(cls, value: str) -> str:
        raw = value.upper().strip()
        if raw in cls.OANDA_PRODUCTS or raw.endswith("-USD"):
            return raw
        if "." in raw:
            raw = raw.rsplit(".", 1)[-1]
        raw = raw.replace("/", "_")
        if raw in cls.OANDA_PRODUCTS:
            return raw
        if raw.endswith("_USD"):
            raw = raw[:-4]
        return cls.ALIASES.get(raw, f"{raw}-USD")

    async def start(self) -> None:
        if self._tasks and not all(task.done() for task in self._tasks):
            return
        self._stopping = False
        self._tasks = [
            # Kalshi's underlying-value stream is an independent fallback for
            # commodity prices and also closely tracks the reference used by the
            # market itself.  OANDA is preferred when it is fresh.
            asyncio.create_task(self._run_pyth(), name="kalshi-commodity-fallback"),
        ]
        if self._oanda_token:
            self._tasks.append(asyncio.create_task(self._run_oanda(), name="commodity-feed"))
        else:
            logger.warning(
                "OANDA COMMODITY PRIMARY FEED DISABLED | missing=KALSHI_PYTH_API_KEY | using=KALSHI_UNDERLYING_FALLBACK"
            )

    async def stop(self) -> None:
        self._stopping = True
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._tasks = []

    @staticmethod
    def _fresh(rows: deque[UnderlyingTick]) -> bool:
        return bool(rows) and (datetime.now(UTC) - rows[-1].timestamp).total_seconds() <= 5

    def snapshot(self, product_id: str) -> tuple[UnderlyingTick, ...]:
        key = self._key(product_id)
        primary = self._primary.get(key, deque())
        return tuple(primary if self._fresh(primary) else self._fallback.get(key, ()))

    def _append(
        self,
        store: dict[str, deque[UnderlyingTick]],
        product: str,
        tick: UnderlyingTick,
        aggregate_second: bool = False,
    ) -> None:
        rows = store[self._key(product)]
        if rows and tick.timestamp < rows[-1].timestamp:
            return
        if aggregate_second and rows and int(rows[-1].timestamp.timestamp()) == int(
            tick.timestamp.timestamp()
        ):
            last = rows[-1]
            size = last.size + tick.size
            price = (last.price * last.size + tick.price * tick.size) / size if size else tick.price
            rows[-1] = UnderlyingTick(price, size, tick.timestamp, tick.source)
        elif rows and tick.timestamp == rows[-1].timestamp:
            rows[-1] = tick
        else:
            rows.append(tick)
        cutoff = tick.timestamp - self._history
        while rows and rows[0].timestamp < cutoff:
            rows.popleft()

    def _handle_coinbase(self, message: dict[str, object]) -> None:
        if message.get("channel") != "market_trades":
            return
        events = message.get("events")
        if not isinstance(events, list):
            return
        for event in events:
            trades = event.get("trades") if isinstance(event, dict) else None
            if not isinstance(trades, list):
                continue
            for trade in trades:
                if not isinstance(trade, dict):
                    continue
                product = str(trade.get("product_id", ""))
                if product not in self.CRYPTO_PRODUCTS:
                    continue
                try:
                    timestamp = datetime.fromisoformat(str(trade["time"]).replace("Z", "+00:00"))
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=UTC)
                    self._append(
                        self._primary,
                        product,
                        UnderlyingTick(
                            float(trade["price"]), float(trade["size"]), timestamp,
                            f"COINBASE_{product}",
                        ),
                        True,
                    )
                except (KeyError, TypeError, ValueError):
                    logger.exception("INVALID COINBASE TRADE | product=%s", product)

    async def _run_coinbase(self) -> None:
        delay = 1.0
        while not self._stopping:
            try:
                async with websockets.connect(
                    self._coinbase_url, open_timeout=10, ping_interval=20, ping_timeout=20
                ) as ws:
                    await ws.send(json.dumps({
                        "type": "subscribe", "product_ids": list(self.CRYPTO_PRODUCTS),
                        "channel": "market_trades",
                    }))
                    await ws.send(json.dumps({"type": "subscribe", "channel": "heartbeats"}))
                    logger.warning(
                        "LIVE-STYLE CRYPTO FEED CONNECTED | products=%s",
                        self.CRYPTO_PRODUCTS,
                    )
                    delay = 1.0
                    async for raw in ws:
                        message = json.loads(raw)
                        if isinstance(message, dict):
                            self._handle_coinbase(message)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("COINBASE FEED ERROR | retry_seconds=%.1f", delay)
                await asyncio.sleep(delay)
                delay = min(30.0, delay * 2)

    def _handle_oanda(self, payload: dict[str, Any]) -> None:
        instrument = str(payload.get("instrument", ""))
        if payload.get("type") != "PRICE" or instrument not in self.OANDA_PRODUCTS:
            return
        if payload.get("status") not in (None, "tradeable"):
            return
        try:
            bid = float(payload["bids"][0]["price"])
            ask = float(payload["asks"][0]["price"])
            timestamp = datetime.fromisoformat(str(payload["time"]).replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            if bid <= 0 or ask < bid:
                raise ValueError("invalid quote")
            self._append(
                self._primary, instrument,
                UnderlyingTick((bid + ask) / 2, 1.0, timestamp, f"OANDA_{instrument}_MID"),
            )
        except (KeyError, IndexError, TypeError, ValueError):
            logger.exception("INVALID OANDA PRICE | instrument=%s", instrument)

    async def _oanda_setup(self, client: httpx.AsyncClient) -> tuple[str, tuple[str, ...]]:
        response = await client.get("https://api-fxpractice.oanda.com/v3/accounts")
        response.raise_for_status()
        accounts = response.json().get("accounts", [])
        if not accounts:
            raise ValueError("OANDA token has no practice account")
        account_id = str(accounts[0]["id"])
        response = await client.get(
            f"https://api-fxpractice.oanda.com/v3/accounts/{account_id}/instruments"
        )
        response.raise_for_status()
        available = {
            str(row.get("name")) for row in response.json().get("instruments", [])
            if isinstance(row, dict)
        }
        selected = tuple(product for product in self.OANDA_PRODUCTS if product in available)
        if not selected:
            raise ValueError("No requested commodity instruments are enabled")
        return account_id, selected

    async def _run_oanda(self) -> None:
        delay = 1.0
        headers = {"Authorization": f"Bearer {self._oanda_token}"}
        async with httpx.AsyncClient(
            headers=headers, timeout=httpx.Timeout(10, read=None), trust_env=False
        ) as client:
            while not self._stopping:
                try:
                    account_id, instruments = await self._oanda_setup(client)
                    logger.warning("COMMODITY FEED CONNECTED | instruments=%s", instruments)
                    url = f"https://stream-fxpractice.oanda.com/v3/accounts/{account_id}/pricing/stream"
                    async with client.stream(
                        "GET", url,
                        params={"instruments": ",".join(instruments), "snapshot": "true"},
                    ) as response:
                        response.raise_for_status()
                        delay = 1.0
                        async for line in response.aiter_lines():
                            if self._stopping:
                                return
                            if line.strip():
                                payload = json.loads(line)
                                if isinstance(payload, dict):
                                    self._handle_oanda(payload)
                    if not self._stopping:
                        raise ConnectionError("OANDA stream ended")
                except asyncio.CancelledError:
                    raise
                except ValueError as exc:
                    # Some OANDA regions/accounts do not offer commodity CFDs.
                    # Kalshi Pyth remains the commodity feed in that case, so do
                    # not flood Railway logs by retrying an unsupported account.
                    logger.warning(
                        "OANDA COMMODITY FEED DISABLED | reason=%s | using=KALSHI_PYTH",
                        exc,
                    )
                    return
                except Exception:
                    logger.exception("OANDA FEED ERROR | retry_seconds=%.1f", delay)
                    await asyncio.sleep(delay)
                    delay = min(30.0, delay * 2)

    def _handle_pyth(self, message: dict[str, Any]) -> None:
        payload = message.get("msg")
        if message.get("type") != "pyth_value" or not isinstance(payload, dict):
            return
        try:
            product = str(payload["underlying_ticker"])
            tick = UnderlyingTick(
                float(payload["value_usd"]), 1.0,
                datetime.fromtimestamp(int(payload["source_ts_ms"]) / 1000, tz=UTC),
                f"KALSHI_PYTH:{product}",
            )
            if tick.price > 0:
                self._append(self._fallback, product, tick)
                key = self._key(product)
                if key not in self._pyth_products_seen:
                    self._pyth_products_seen.add(key)
                    logger.warning(
                        "PYTH PRODUCT ACTIVE | underlying=%s | mapped_product=%s",
                        product,
                        key,
                    )
        except (KeyError, TypeError, ValueError, OverflowError):
            logger.exception("INVALID PYTH MESSAGE")

    async def _run_pyth(self) -> None:
        # Kalshi creates a pyth_value subscription with zero underlyings when
        # this parameter is omitted.  "all" tracks every underlying currently
        # exposed by Kalshi, including metals and energy products.
        await self._run_kalshi_channel(
            "pyth_value", self._handle_pyth, {"underlying_tickers": ["all"]}
        )

    def _handle_cf(self, message: dict[str, Any]) -> None:
        payload = message.get("msg")
        if message.get("type") != "cfbenchmarks_value" or not isinstance(payload, dict):
            return
        try:
            index = str(payload["index_id"])
            raw = json.loads(str(payload["data"]))
            base = "BTC" if index.upper() == "BRTI" else index.upper().split("USD", 1)[0]
            tick = UnderlyingTick(
                float(raw["value"]), 1.0,
                datetime.fromtimestamp(int(raw["time"]) / 1000, tz=UTC),
                f"KALSHI_CFBENCHMARKS:{index}",
            )
            if tick.price > 0:
                self._append(self._fallback, f"{base}-USD", tick)
        except (KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError):
            logger.exception("INVALID CF MESSAGE")

    async def _run_cf(self) -> None:
        await self._run_kalshi_channel(
            "cfbenchmarks_value", self._handle_cf, {"index_ids": ["all"]}
        )

    async def _run_kalshi_channel(
        self,
        channel: str,
        handler: Any,
        params: dict[str, Any] | None = None,
    ) -> None:
        delay = 1.0
        while not self._stopping:
            try:
                async for message in self._kalshi_ws.stream(
                    [channel], subscription_params=params
                ):
                    if self._stopping:
                        return
                    if isinstance(message, dict):
                        if message.get("type") == "subscribed":
                            logger.warning(
                                "%s FEED CONNECTED | subscription=%s",
                                channel.upper(),
                                params or {},
                            )
                        handler(message)
                if not self._stopping:
                    raise ConnectionError(f"{channel} stream ended")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("%s FEED ERROR | retry_seconds=%.1f", channel, delay)
                await asyncio.sleep(delay)
                delay = min(30.0, delay * 2)
