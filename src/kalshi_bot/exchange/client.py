"""Async Kalshi REST client with RSA-PSS request signing.

Current V2 event-order implementation with confirmed fill metadata and
reduce-only protection on exits.
"""

from __future__ import annotations

import time
import uuid
from decimal import Decimal, ROUND_HALF_UP
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

import httpx
from cryptography.hazmat.primitives.asymmetric import rsa

from kalshi_bot.config import Settings
from kalshi_bot.exchange.auth import build_auth_headers, load_private_key
from kalshi_bot.exchange.models import Balance, Market, Order, OrderRequest, Position
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)
API_PREFIX = ""


def _dollars_to_cents(value: str) -> int:
    return int(
        (Decimal(str(value)) * 100).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )


class KalshiError(RuntimeError):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"Kalshi API error {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class KalshiClient:
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

        request.headers.update(
            build_auth_headers(
                self._api_key_id,
                self._private_key,
                timestamp_ms,
                request.method,
                path,
            )
        )

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

        return response.json()

    async def get_markets(self, **params: Any) -> list[Market]:
        data = await self._request("GET", "/markets", params=params)
        return [Market.model_validate(m) for m in data.get("markets", [])]

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
        return data.get("orderbook", {})

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

    async def create_order(self, order: OrderRequest) -> Order:
        """Create an order through the standard Predictions order endpoint.

        The bot intentionally uses /portfolio/orders here instead of the newer
        /portfolio/events/orders exchange-sharded endpoint.  The standard
        endpoint is the documented Predictions quick-start path and matches
        this account/API-key flow.
        """

        client_order_id = order.client_order_id or str(uuid.uuid4())

        payload = order.to_payload()
        payload["client_order_id"] = client_order_id

        data = await self._request(
            "POST",
            "/portfolio/orders",
            json=payload,
        )

        created = data.get("order", data)
        result = Order.model_validate(created)

        # Preserve the bot's requested values when the API response omits
        # legacy compatibility fields.
        result.ticker = result.ticker or order.ticker
        result.side = result.side or order.side
        result.action = result.action or order.action
        result.client_order_id = result.client_order_id or client_order_id
        result.count = result.count if result.count is not None else order.count

        # The strategy already falls back to the requested limit price if an
        # immediate fill response does not contain an explicit average price.
        if result.fill_count > 0 and result.outcome_fill_price is None:
            result.outcome_fill_price = (
                order.yes_price if order.side is Side.YES else order.no_price
            )

        logger.warning(
            "LIVE ORDER RESULT | ticker=%s | order_id=%s | action=%s | "
            "side=%s | requested=%d | fill_count=%d | remaining=%s | "
            "outcome_fill=%s",
            order.ticker,
            result.order_id,
            order.action.value,
            order.side.value,
            order.count,
            result.fill_count,
            result.remaining_count,
            result.outcome_fill_price,
        )

        return result

    async def cancel_order(self, order_id: str) -> Order:
        data = await self._request(
            "DELETE",
            f"/portfolio/orders/{order_id}",
        )

        canceled = data.get("order", data)
        if isinstance(canceled, dict) and canceled.get("order_id"):
            return Order.model_validate(canceled)

        return Order(
            order_id=order_id,
            ticker="",
            status="canceled",
        )

    async def get_orders(self, **params: Any) -> list[Order]:
        data = await self._request(
            "GET",
            "/portfolio/orders",
            params=params,
        )

        return [
            Order.model_validate(o)
            for o in data.get("orders", [])
        ]

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
