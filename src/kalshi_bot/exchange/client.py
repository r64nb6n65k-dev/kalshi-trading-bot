"""Async Kalshi REST client with RSA-PSS request signing.

Wraps the public market-data endpoints and the authenticated portfolio/trading
endpoints under ``/trade-api/v2``. Auth headers are attached automatically to
every request via an httpx event hook.

Example:
    ```python
    from kalshi_bot.config import load_settings
    from kalshi_bot.exchange.client import KalshiClient

    settings = load_settings()
    async with KalshiClient.from_settings(settings) as client:
        balance = await client.get_balance()
        markets = await client.get_markets(status="open", limit=10)
    ```

Part of the Kalshi Trading Bot by Viprasol Tech Private Limited.
"""

from __future__ import annotations

import time
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

import httpx
from cryptography.hazmat.primitives.asymmetric import rsa

from kalshi_bot.config import Settings
from kalshi_bot.exchange.auth import build_auth_headers, load_private_key
from kalshi_bot.exchange.models import (
    Balance,
    Market,
    Order,
    OrderRequest,
    Position,
)

API_PREFIX = ""


class KalshiError(RuntimeError):
    """Raised when the Kalshi API returns an error response."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"Kalshi API error {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class KalshiClient:
    """Async REST client for the Kalshi trading API."""

    def __init__(
        self,
        base_url: str,
        api_key_id: str | None = None,
        private_key: rsa.RSAPrivateKey | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key_id = api_key_id
        self._private_key = private_key
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            event_hooks={"request": [self._sign_request]},
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> KalshiClient:
        """Build a client from :class:`~kalshi_bot.config.Settings`.

        Loads the RSA private key only when an API key ID is configured, so the
        client can still be used for public market data without credentials.
        """
        private_key = None
        if settings.api_key_id:
            private_key = load_private_key(settings.private_key_path)
        return cls(
            base_url=settings.rest_base_url,
            api_key_id=settings.api_key_id or None,
            private_key=private_key,
            timeout=settings.request_timeout,
        )

    @property
    def authenticated(self) -> bool:
        """Whether credentials are present for signed requests."""
        return self._api_key_id is not None and self._private_key is not None

    async def _sign_request(self, request: httpx.Request) -> None:
        """httpx event hook: attach Kalshi auth headers to every request.

        The signed path includes the ``/trade-api/v2`` prefix and excludes the
        query string.
        """
        if not self.authenticated:
            return
        assert self._api_key_id is not None and self._private_key is not None
        path = urlsplit(str(request.url)).path
        timestamp_ms = int(time.time() * 1000)
        headers = build_auth_headers(
            self._api_key_id, self._private_key, timestamp_ms, request.method, path
        )
        request.headers.update(headers)

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = await self._client.request(method, API_PREFIX + path, **kwargs)
        if response.is_error:
            raise KalshiError(response.status_code, response.text)
        return response.json()  # type: ignore[no-any-return]

    # --- Market data (public) -------------------------------------------

    async def get_markets(self, **params: Any) -> list[Market]:
        """List markets. Filter by ``status``, ``event_ticker``, ``series_ticker``, ``limit``."""
        data = await self._request("GET", "/markets", params=params)
        return [Market.model_validate(m) for m in data.get("markets", [])]

    async def get_market(self, ticker: str) -> Market:
        """Fetch a single market by ticker."""
        data = await self._request("GET", f"/markets/{ticker}")
        return Market.model_validate(data["market"])

    async def get_orderbook(self, ticker: str, depth: int | None = None) -> dict[str, Any]:
        """Fetch the order book for a market."""
        params = {"depth": depth} if depth is not None else {}
        data = await self._request("GET", f"/markets/{ticker}/orderbook", params=params)
        return data.get("orderbook", {})  # type: ignore[no-any-return]

    # --- Portfolio / trading (authenticated) ----------------------------

    async def get_balance(self) -> Balance:
        """Get the portfolio cash balance (in cents)."""
        data = await self._request("GET", "/portfolio/balance")
        return Balance.model_validate(data)

    async def get_positions(self, **params: Any) -> list[Position]:
        """List open positions."""
        data = await self._request("GET", "/portfolio/positions", params=params)
        return [Position.model_validate(p) for p in data.get("market_positions", [])]

    async def create_order(self, order: OrderRequest) -> Order:
        """Create an order using Kalshi's current V2 event-order endpoint."""

        action = order.action.value
        outcome = order.side.value

        # Kalshi V2 uses a single YES-side book:
        # bid = buy YES / sell NO
        # ask = sell YES / buy NO
        if action == "buy" and outcome == "yes":
            book_side = "bid"
            price_cents = order.yes_price
        elif action == "sell" and outcome == "yes":
            book_side = "ask"
            price_cents = order.yes_price
        elif action == "buy" and outcome == "no":
            book_side = "ask"
            price_cents = None if order.no_price is None else 100 - order.no_price
        elif action == "sell" and outcome == "no":
            book_side = "bid"
            price_cents = None if order.no_price is None else 100 - order.no_price
        else:
            raise ValueError(
                f"Unsupported order direction: action={action}, side={outcome}"
            )

        if price_cents is None:
            raise ValueError("Order price is required")

        raw_tif = (
            order.time_in_force.value
            if order.time_in_force is not None
            else "good_till_canceled"
        )

        tif_map = {
            "ioc": "immediate_or_cancel",
            "immediate_or_cancel": "immediate_or_cancel",
            "gtc": "good_till_canceled",
            "good_till_canceled": "good_till_canceled",
            "fok": "fill_or_kill",
            "fill_or_kill": "fill_or_kill",
        }

        time_in_force = tif_map.get(raw_tif)
        if time_in_force is None:
            raise ValueError(f"Unsupported time_in_force: {raw_tif}")

        payload: dict[str, Any] = {
            "ticker": order.ticker,
            "side": book_side,
            "count": f"{order.count:.2f}",
            "price": f"{price_cents / 100:.4f}",
            "time_in_force": time_in_force,
            "self_trade_prevention_type": "taker_at_cross",
            "post_only": bool(order.post_only) if order.post_only is not None else False,
        }

        if order.client_order_id is not None:
            payload["client_order_id"] = order.client_order_id

        created = await self._request(
            "POST",
            "/portfolio/events/orders",
            json=payload,
        )

        order_id = created["order_id"]

        # The V2 create endpoint returns a compact response. Fetch the full
        # order through the existing read endpoint so the rest of the bot can
        # keep using the current Order model unchanged.
        data = await self._request(
            "GET",
            f"/portfolio/orders/{order_id}",
        )
        return Order.model_validate(data["order"])

    async def cancel_order(self, order_id: str) -> Order:
        """Cancel an order by ID (``DELETE /portfolio/orders/{id}``)."""
        data = await self._request("DELETE", f"/portfolio/orders/{order_id}")
        return Order.model_validate(data["order"])

    async def get_orders(self, **params: Any) -> list[Order]:
        """List orders, optionally filtered by ``ticker``/``status``."""
        data = await self._request("GET", "/portfolio/orders", params=params)
        return [Order.model_validate(o) for o in data.get("orders", [])]

    # --- Lifecycle ------------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> KalshiClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
