"""Live Polymarket CLOB execution for the rolling crypto momentum strategy.

Safety properties:
- live execution requires BOTH the CLI --live flag and POLY_LIVE=true
- geoblock is checked from the machine actually running the bot
- entries use fixed-price FOK orders; partial entries are not accepted
- ambiguous network/order responses are never blindly retried with a new signed order
- exits use FOK limit sells and remain open until a confirmed match
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, replace
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Any

import httpx

from kalshi_bot.data.multi_crypto import MultiCryptoPriceFeed
from kalshi_bot.exchange.models import Side
from kalshi_bot.polymarket import PolymarketListing, PolymarketPublicClient
from kalshi_bot.strategies.examples.polymarket_momentum import (
    PolymarketMomentumStrategy,
    SimSignal,
)
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required deployment variable: {name}")
    return value


def _response_value(response: Any, *keys: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        for key in keys:
            if key in response:
                return response[key]
        return default
    for key in keys:
        if hasattr(response, key):
            return getattr(response, key)
    return default


def _average_fill_cents(response: Any, side: str, fallback_cents: int) -> int:
    """Derive matched average price from maker/taker amounts when possible."""
    try:
        making = Decimal(
            str(_response_value(response, "makingAmount", "making_amount"))
        )
        taking = Decimal(
            str(_response_value(response, "takingAmount", "taking_amount"))
        )
        if making <= 0 or taking <= 0:
            return fallback_cents

        price = making / taking if side == "BUY" else taking / making
        cents = int((price * 100).quantize(Decimal("1")))
        return max(1, min(99, cents))
    except (InvalidOperation, TypeError, ValueError, ZeroDivisionError):
        return fallback_cents


@dataclass(slots=True)
class PendingLiveEntry:
    listing: PolymarketListing
    signal: SimSignal
    deadline: float
    signed_order: Any | None = None
    shares: int = 0
    submitted: bool = False
    uncertain: bool = False


class PolymarketTradingClient:
    """Thin async wrapper around Polymarket's official CLOB V2 Python SDK."""

    HOST = "https://clob.polymarket.com"
    CHAIN_ID = 137
    SIGNATURE_TYPE_DEPOSIT_WALLET = 3

    def __init__(self) -> None:
        try:
            from py_clob_client_v2 import ApiCreds, ClobClient
        except ImportError as exc:
            raise RuntimeError(
                "py-clob-client-v2 is not installed. "
                "Deploy from the updated pyproject.toml."
            ) from exc

        self._ApiCreds = ApiCreds
        self._ClobClient = ClobClient
        self.private_key = _require("POLYMARKET_SIGNER_PRIVATE_KEY")
        self.wallet = _require("POLYMARKET_WALLET_ADDRESS")
        self.host = os.getenv("POLYMARKET_CLOB_URL", self.HOST).rstrip("/")

        api_key = os.getenv("POLYMARKET_CLOB_API_KEY", "").strip()
        api_secret = os.getenv("POLYMARKET_CLOB_SECRET", "").strip()
        api_passphrase = os.getenv(
            "POLYMARKET_CLOB_PASSPHRASE", ""
        ).strip()

        if api_key and api_secret and api_passphrase:
            creds = ApiCreds(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            )
        elif any((api_key, api_secret, api_passphrase)):
            raise RuntimeError(
                "Set all three CLOB variables or none: "
                "POLYMARKET_CLOB_API_KEY, "
                "POLYMARKET_CLOB_SECRET, "
                "POLYMARKET_CLOB_PASSPHRASE"
            )
        else:
            temp = ClobClient(
                host=self.host,
                chain_id=self.CHAIN_ID,
                key=self.private_key,
                signature_type=self.SIGNATURE_TYPE_DEPOSIT_WALLET,
                funder=self.wallet,
                use_server_time=True,
            )
            creds = temp.create_or_derive_api_key()

        self._client = ClobClient(
            host=self.host,
            chain_id=self.CHAIN_ID,
            key=self.private_key,
            creds=creds,
            signature_type=self.SIGNATURE_TYPE_DEPOSIT_WALLET,
            funder=self.wallet,
            use_server_time=True,
            retry_on_error=False,
        )

    async def check_geoblock(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=8.0) as http:
            response = await http.get(
                "https://polymarket.com/api/geoblock"
            )
            response.raise_for_status()
            payload = response.json()

        if not isinstance(payload, dict):
            raise RuntimeError(
                "Unexpected Polymarket geoblock response"
            )

        return payload

    async def collateral_balance(self) -> float | None:
        try:
            from py_clob_client_v2 import (
                AssetType,
                BalanceAllowanceParams,
            )

            result = await asyncio.to_thread(
                self._client.get_balance_allowance,
                BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL
                ),
            )

            raw = _response_value(result, "balance")

            if raw is None and isinstance(result, dict):
                raw = result.get("balance")

            if raw is None:
                return None

            value = Decimal(str(raw))

            if value > Decimal("10000"):
                value /= Decimal("1000000")

            return float(value)

        except Exception:
            logger.exception(
                "Could not read Polymarket collateral balance"
            )
            return None

    def _token_id(
        self,
        listing: PolymarketListing,
        side: Side,
    ) -> str:
        token = (
            listing.up_token_id
            if side is Side.YES
            else listing.down_token_id
        )

        if not token:
            raise RuntimeError(
                f"No CLOB token id available for "
                f"{listing.slug} {side.value}"
            )

        return token

    async def build_limit_order(
        self,
        listing: PolymarketListing,
        side: Side,
        *,
        action: str,
        price_cents: int,
        shares: int,
    ) -> Any:
        from py_clob_client_v2 import (
            OrderArgs,
            PartialCreateOrderOptions,
        )
        from py_clob_client_v2 import Side as ClobSide

        token_id = self._token_id(listing, side)
        clob_side = (
            ClobSide.BUY
            if action == "BUY"
            else ClobSide.SELL
        )

        args = OrderArgs(
            token_id=token_id,
            price=price_cents / 100,
            side=clob_side,
            size=shares,
        )

        return await asyncio.to_thread(
            self._client.create_order,
            args,
            PartialCreateOrderOptions(),
        )

    async def post_fok(self, signed_order: Any) -> Any:
        from py_clob_client_v2 import OrderType

        return await asyncio.to_thread(
            self._client.post_order,
            signed_order,
            OrderType.FOK,
            False,
            False,
        )

    async def executable_buy_quote(
        self,
        listing: PolymarketListing,
        side: Side,
        *,
        slippage_cents: int,
    ) -> tuple[int, int, float] | None:
        """Return fresh best ask, limit price, and executable depth."""
        token_id = self._token_id(listing, side)

        book = await asyncio.to_thread(
            self._client.get_order_book,
            token_id,
        )

        asks = _response_value(
            book,
            "asks",
            default=[],
        ) or []

        levels: list[tuple[int, float]] = []

        for level in asks:
            try:
                price = Decimal(
                    str(_response_value(level, "price"))
                )
                size = float(
                    _response_value(
                        level,
                        "size",
                        default=0,
                    )
                )
                cents = int(
                    (price * 100).to_integral_value(
                        rounding=ROUND_CEILING
                    )
                )
            except (InvalidOperation, TypeError, ValueError):
                continue

            if 1 <= cents <= 99 and size > 0:
                levels.append((cents, size))

        if not levels:
            return None

        best_ask = min(
            price for price, _ in levels
        )

        limit_price = min(
            99,
            best_ask + max(0, slippage_cents),
        )

        executable_depth = sum(
            size
            for price, size in levels
            if price <= limit_price
        )

        return (
            best_ask,
            limit_price,
            executable_depth,
        )
       
