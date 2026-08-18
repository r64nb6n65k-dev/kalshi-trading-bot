"""Async Kalshi REST client with RSA-PSS request signing.

This version uses Kalshi's V2 event-order endpoints for create/cancel and logs
the matching-engine result returned by order creation, including fill_count,
remaining_count, and average_fill_price.
"""

from __future__ import annotations

import time
import uuid
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
from kalshi_bot.telemetry.logging import get_logger


logger = get_logger(__name__)

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
    def from_settings(cls, settings: Settings) -> "KalshiClient":
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
        return self._api_key_id is not None and self._private_key is not None

    async def _sign_request(self, request: httpx.Request) -> None:
        if not self.authenticated:
            return

        assert self._api_key_id is not None
        assert self._private_key is not None

        path = urlsplit(str(request.url)).path
        timestamp_ms = int(time.time() * 1000)

        headers = build_auth_headers(
            self._api_key_id,
            self._private_key,
            timestamp_ms,
            request.method,
            path,
        )
        request.headers.update(headers)

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response = await self._client.request(
            method,
            API_PREFIX + path,
            **kwargs,
        )

        if response.is_error:
            raise KalshiError(response.status_code, response.text)

        return response.json()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # MARKET DATA
    # ------------------------------------------------------------------

    async def get_markets(self, **params: Any) -> list[Market]:
        data = await self._request("GET", "/markets", params=params)
        return [
            Market.model_validate(m)
            for m in data.get("markets", [])
        ]

    async def get_market(self, ticker: str) -> Market:
        data = await self._request("GET", f"/markets/{ticker}")
        return Market.model_validate(data["market"])

    async def get_orderbook(
        self,
        ticker: str,
        depth: int | None = None,
    ) -> dict[str, Any]:
        params = {"depth": depth} if depth is not None else {}
        data = await self._request(
            "GET",
            f"/markets/{ticker}/orderbook",
            params=params,
        )
        return data.get("orderbook", {})  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # PORTFOLIO
    # ------------------------------------------------------------------

    async def get_balance(self) -> Balance:
        data = await self._request("GET", "/portfolio/balance")
        return Balance.model_validate(data)

    async def get_positions(self, **params: Any) -> list[Position]:
        data = await self._request(
            "GET",
            "/portfolio/positions",
            params=params,
        )
        return [
            Position.model_validate(p)
            for p in data.get("market_positions", [])
        ]

    # ------------------------------------------------------------------
    # ORDERS
    # ------------------------------------------------------------------

    async def create_order(self, order: OrderRequest) -> Order:
        """Create an order with Kalshi's V2 event-order endpoint.

        V2 exposes a single YES-side order book:
        - bid = buy YES
        - ask = sell YES

        Buying NO at N cents is therefore represented as selling YES at
        (100-N) cents. Selling NO is represented as buying YES at (100-N).
        """

        action = order.action.value
        outcome = order.side.value

        if action == "buy" and outcome == "yes":
            book_side = "bid"
            price_cents = order.yes_price

        elif action == "sell" and outcome == "yes":
            book_side = "ask"
            price_cents = order.yes_price

        elif action == "buy" and outcome == "no":
            book_side = "ask"
            price_cents = (
                None
                if order.no_price is None
                else 100 - order.no_price
            )

        elif action == "sell" and outcome == "no":
            book_side = "bid"
            price_cents = (
                None
                if order.no_price is None
                else 100 - order.no_price
            )

        else:
            raise ValueError(
                "Unsupported order direction: "
                f"action={action}, side={outcome}"
            )

        if price_cents is None:
            raise ValueError("Order price is required")

        if not 0 < price_cents < 100:
            raise ValueError(
                f"Order price must be between 1 and 99 cents: {price_cents}"
            )

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
            raise ValueError(
                f"Unsupported time_in_force: {raw_tif}"
            )

        client_order_id = (
            order.client_order_id
            if order.client_order_id
            else str(uuid.uuid4())
        )

        payload: dict[str, Any] = {
            "ticker": order.ticker,
            "client_order_id": client_order_id,
            "side": book_side,
            "count": f"{order.count:.2f}",
            "price": f"{price_cents / 100:.4f}",
            "time_in_force": time_in_force,
            "self_trade_prevention_type": "taker_at_cross",
            "post_only": (
                bool(order.post_only)
                if order.post_only is not None
                else False
            ),
        }

        created = await self._request(
            "POST",
            "/portfolio/events/orders",
            json=payload,
        )

        fill_count = int(float(created.get("fill_count", "0")))
        remaining_count = int(
            float(created.get("remaining_count", "0"))
        )
        average_fill_price = created.get("average_fill_price")

        logger.warning(
            "LIVE ORDER RESULT | ticker=%s | order_id=%s | "
            "client_order_id=%s | action=%s | outcome=%s | "
            "book_side=%s | requested_count=%d | "
            "fill_count=%d | remaining_count=%d | "
            "limit_price=%dc | average_fill_price=%s",
            order.ticker,
            created.get("order_id"),
            client_order_id,
            action,
            outcome,
            book_side,
            order.count,
            fill_count,
            remaining_count,
            price_cents,
            average_fill_price,
        )

        if fill_count >= order.count:
            status = "executed"
        elif remaining_count > 0:
            status = "resting"
        else:
            # For IOC, zero remaining means the unfilled remainder has
            # already been canceled by the exchange.
            status = "canceled"

        return Order(
            order_id=created["order_id"],
            ticker=order.ticker,
            status=status,
            side=order.side,
            action=order.action,
            yes_price=order.yes_price,
            no_price=order.no_price,
            count=order.count,
            remaining_count=remaining_count,
            client_order_id=client_order_id,
        )

    async def cancel_order(self, order_id: str) -> Order:
        """Cancel an order using Kalshi's V2 event-order endpoint."""

        data = await self._request(
            "DELETE",
            f"/portfolio/events/orders/{order_id}",
        )

        return Order(
            order_id=data["order_id"],
            ticker="",
            status="canceled",
            client_order_id=data.get("client_order_id"),
        )

    async def get_orders(self, **params: Any) -> list[Order]:
        """List orders using the existing read endpoint."""

        data = await self._request(
            "GET",
            "/portfolio/orders",
            params=params,
        )

        return [
            Order.model_validate(o)
            for o in data.get("orders", [])
        ]

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "KalshiClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
