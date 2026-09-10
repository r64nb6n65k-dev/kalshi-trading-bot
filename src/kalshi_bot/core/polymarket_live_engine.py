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
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from kalshi_bot.data.multi_crypto import MultiCryptoPriceFeed
from kalshi_bot.exchange.models import Side
from kalshi_bot.polymarket import PolymarketListing, PolymarketPublicClient
from kalshi_bot.strategies.examples.polymarket_momentum import PolymarketMomentumStrategy, SimSignal
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
        raise RuntimeError(f"Missing required Railway variable: {name}")
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
    """Derive matched average price from maker/taker amounts when possible.

    For a BUY, maker amount is collateral and taking amount is shares.
    For a SELL, maker amount is shares and taking amount is collateral.
    The ratio is scale-invariant whether the SDK returns human units or 1e6 fixed units.
    """
    try:
        making = Decimal(str(_response_value(response, "makingAmount", "making_amount")))
        taking = Decimal(str(_response_value(response, "takingAmount", "taking_amount")))
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
        except ImportError as exc:  # pragma: no cover - dependency is installed on Railway
            raise RuntimeError(
                "py-clob-client-v2 is not installed. Deploy from the updated pyproject.toml."
            ) from exc

        self._ApiCreds = ApiCreds
        self._ClobClient = ClobClient
        self.private_key = _require("POLYMARKET_SIGNER_PRIVATE_KEY")
        self.wallet = _require("POLYMARKET_WALLET_ADDRESS")
        self.host = os.getenv("POLYMARKET_CLOB_URL", self.HOST).rstrip("/")

        api_key = os.getenv("POLYMARKET_CLOB_API_KEY", "").strip()
        api_secret = os.getenv("POLYMARKET_CLOB_SECRET", "").strip()
        api_passphrase = os.getenv("POLYMARKET_CLOB_PASSPHRASE", "").strip()

        if api_key and api_secret and api_passphrase:
            creds = ApiCreds(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            )
        elif any((api_key, api_secret, api_passphrase)):
            raise RuntimeError(
                "Set all three CLOB variables or none: POLYMARKET_CLOB_API_KEY, "
                "POLYMARKET_CLOB_SECRET, POLYMARKET_CLOB_PASSPHRASE"
            )
        else:
            # Derive deterministic CLOB credentials from the signer. We never print them.
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
            response = await http.get("https://polymarket.com/api/geoblock")
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected Polymarket geoblock response")
        return payload

    async def collateral_balance(self) -> float | None:
        try:
            from py_clob_client_v2 import AssetType, BalanceAllowanceParams

            result = await asyncio.to_thread(
                self._client.get_balance_allowance,
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
            )
            raw = _response_value(result, "balance")
            if raw is None and isinstance(result, dict):
                raw = result.get("balance")
            if raw is None:
                return None
            value = Decimal(str(raw))
            # Balance endpoint commonly returns 6-decimal fixed units.
            if value > Decimal("10000"):
                value /= Decimal("1000000")
            return float(value)
        except Exception:
            logger.exception("Could not read Polymarket collateral balance")
            return None

    def _token_id(self, listing: PolymarketListing, side: Side) -> str:
        token = listing.up_token_id if side is Side.YES else listing.down_token_id
        if not token:
            raise RuntimeError(f"No CLOB token id available for {listing.slug} {side.value}")
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
        from py_clob_client_v2 import OrderArgs, PartialCreateOrderOptions
        from py_clob_client_v2 import Side as ClobSide

        token_id = self._token_id(listing, side)
        clob_side = ClobSide.BUY if action == "BUY" else ClobSide.SELL
        args = OrderArgs(
            token_id=token_id,
            price=price_cents / 100,
            side=clob_side,
            size=shares,
        )
        # Let the V2 client fetch the market's actual tick size / neg-risk metadata.
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


class PolymarketLiveEngine:
    def __init__(
        self,
        *,
        market_client: PolymarketPublicClient,
        trading_client: PolymarketTradingClient,
        feed: MultiCryptoPriceFeed,
        strategy: PolymarketMomentumStrategy,
        poll_interval: float = 1.0,
        execution_window_seconds: float = 3.0,
        live_enabled: bool = False,
    ) -> None:
        self.market_client = market_client
        self.trading_client = trading_client
        self.feed = feed
        self.strategy = strategy
        self.poll_interval = max(0.25, poll_interval)
        self.execution_window_seconds = max(0.5, execution_window_seconds)
        self.live_enabled = live_enabled and _env_bool("POLY_LIVE", False)
        self._targets: dict[str, float] = {}
        self._known: dict[str, PolymarketListing] = {}
        self._pending: dict[str, PendingLiveEntry] = {}
        self._exit_uncertain: set[str] = set()

    def _target_for(self, listing: PolymarketListing) -> float | None:
        existing = self._targets.get(listing.slug)
        if existing is not None:
            return existing
        product = self.market_client.ASSETS[listing.asset]
        ticks = self.feed.snapshot(product)
        if not ticks:
            return None
        opening = min(ticks, key=lambda tick: abs(tick.timestamp.timestamp() - listing.open_time))
        if abs(opening.timestamp.timestamp() - listing.open_time) > 5:
            return None
        self._targets[listing.slug] = opening.price
        logger.warning(
            "OPENING REFERENCE CAPTURED | ticker=%s | price=%.6f | source=%s",
            listing.slug,
            opening.price,
            opening.source,
        )
        return opening.price

    @staticmethod
    def _response_ok(response: Any) -> bool:
        return bool(_response_value(response, "success", "ok", default=False))

    @staticmethod
    def _status(response: Any) -> str:
        return str(_response_value(response, "status", default="")).lower()

    @staticmethod
    def _error(response: Any) -> str:
        return str(_response_value(response, "errorMsg", "error_msg", default="") or "")

    @classmethod
    def _explicit_no_fill(cls, response: Any) -> bool:
        error = cls._error(response).lower()
        status = cls._status(response)
        return (not cls._response_ok(response)) and (
            "no orders found" in error
            or "no match" in error
            or "fok" in error and "fill" in error
            or status == "unmatched"
        )

    async def _submit_entry(self, pending: PendingLiveEntry, now: float) -> bool:
        if now >= pending.deadline:
            logger.warning(
                "POLYMARKET LIVE CANCEL | ticker=%s | reason=NO_FILL_3S_FIXED_LIMIT | limit=%dc",
                pending.listing.slug,
                pending.signal.limit_price,
            )
            return True
        if pending.uncertain:
            return True
        try:
            if pending.signed_order is None:
                pending.signed_order = await self.trading_client.build_limit_order(
                    pending.listing,
                    pending.signal.side,
                    action="BUY",
                    price_cents=pending.signal.limit_price,
                    shares=self.strategy.contracts,
                )
            response = await self.trading_client.post_fok(pending.signed_order)
        except Exception:
            # Reusing the same signed order is normally idempotent, but a transport error after
            # submission is still ambiguous. Do not create/sign another order for this signal.
            pending.uncertain = True
            logger.exception(
                "POLYMARKET ENTRY RESPONSE UNCERTAIN | ticker=%s | no automatic resubmission",
                pending.listing.slug,
            )
            return True

        status = self._status(response)
        if self._response_ok(response) and status == "matched":
            fill_cents = _average_fill_cents(response, "BUY", pending.signal.limit_price)
            self.strategy.open_position(
                pending.listing,
                pending.signal.side,
                fill_cents,
                count=self.strategy.contracts,
                execution_mode="polymarket_live",
            )
            logger.warning(
                "POLYMARKET LIVE FILL | ticker=%s | side=%s | fill=%dc | count=%d | order=%s",
                pending.listing.slug,
                pending.signal.side.value,
                fill_cents,
                self.strategy.contracts,
                _response_value(response, "orderID", "order_id", default="?"),
            )
            return True

        if self._explicit_no_fill(response):
            # Same signed FOK can be tried again within the original 3-second fixed-price window.
            return False

        # 'delayed' or any unexpected accepted state is ambiguous. Never fire a second order.
        logger.error(
            "POLYMARKET ENTRY NOT CONFIRMED | ticker=%s | status=%s | error=%s | "
            "no automatic duplicate",
            pending.listing.slug,
            status,
            self._error(response),
        )
        return True

    async def _take_profit(self, listing: PolymarketListing, now: float) -> None:
        position = self.strategy.positions.get(listing.slug)
        if position is None or listing.slug in self._exit_uncertain:
            return
        bid = listing.yes_bid if position.side is Side.YES else listing.no_bid
        if bid is None or bid < self.strategy.take_profit:
            return
        try:
            signed = await self.trading_client.build_limit_order(
                listing,
                position.side,
                action="SELL",
                price_cents=self.strategy.take_profit,
                shares=position.count,
            )
            response = await self.trading_client.post_fok(signed)
        except Exception:
            self._exit_uncertain.add(listing.slug)
            logger.exception(
                "POLYMARKET EXIT RESPONSE UNCERTAIN | ticker=%s | position frozen to avoid double sell",
                listing.slug,
            )
            return

        if self._response_ok(response) and self._status(response) == "matched":
            fill_cents = _average_fill_cents(response, "SELL", self.strategy.take_profit)
            self.strategy.close_position(listing.slug, fill_cents, "TAKE_PROFIT_98", now)
            logger.warning(
                "POLYMARKET LIVE EXIT | ticker=%s | side=%s | fill=%dc | reason=TAKE_PROFIT_98",
                listing.slug,
                position.side.value,
                fill_cents,
            )
            return

        if self._explicit_no_fill(response):
            return
        self._exit_uncertain.add(listing.slug)
        logger.error(
            "POLYMARKET EXIT NOT CONFIRMED | ticker=%s | status=%s | error=%s | "
            "position frozen to avoid duplicate sell",
            listing.slug,
            self._status(response),
            self._error(response),
        )

    async def _settle_missing(self, active: set[str], now: float) -> None:
        # Resolution bookkeeping only. Winning outcome tokens are worth $1 and losing tokens $0.
        # We deliberately do not send an order after resolution.
        for slug in set(self.strategy.positions) - active:
            listing = self._known.get(slug)
            if listing is None or now < listing.close_time:
                continue
            resolved = await self.market_client.resolved(listing)
            if resolved is None or resolved.resolved_side not in {"yes", "no"}:
                continue
            position = self.strategy.positions.get(slug)
            if position is None:
                continue
            won = position.side.value == resolved.resolved_side
            self.strategy.close_position(
                slug,
                100 if won else 0,
                "SETTLEMENT_WIN" if won else "SETTLEMENT_LOSS",
                now,
            )
            self._exit_uncertain.discard(slug)

    async def preflight(self) -> None:
        if not self.live_enabled:
            raise RuntimeError(
                "Live trading is locked. Start with --live AND set Railway variable POLY_LIVE=true."
            )
        geo = await self.trading_client.check_geoblock()
        if bool(geo.get("blocked")):
            raise RuntimeError(
                "Polymarket reports this Railway server IP as geoblocked: "
                f"country={geo.get('country')} region={geo.get('region')}. "
                "Move the Railway service to an eligible European region; do not use a VPN/proxy."
            )
        balance = await self.trading_client.collateral_balance()
        logger.warning(
            "POLYMARKET PREFLIGHT OK | server_country=%s | server_region=%s | wallet=%s...%s | "
            "reported_collateral=%s",
            geo.get("country"),
            geo.get("region"),
            self.trading_client.wallet[:6],
            self.trading_client.wallet[-4:],
            "unknown" if balance is None else f"${balance:.2f}",
        )

    async def run(self, max_cycles: int | None = None) -> None:
        await self.preflight()
        logger.warning(
            "POLYMARKET LIVE STARTED | intervals=%s | assets=%s | bankroll=$%.2f | "
            "contracts=%d | LIVE_ORDERS=ENABLED",
            self.market_client.intervals,
            tuple(self.market_client.ASSETS),
            self.strategy.bankroll_cents / 100,
            self.strategy.contracts,
        )
        await self.feed.start()
        cycle = 0
        try:
            while max_cycles is None or cycle < max_cycles:
                try:
                    now = time.time()
                    listings = await self.market_client.get_open_crypto_markets(now)
                    # Live trading requires Gamma/CLOB token IDs, even if a US read-only quote exists.
                    listings = [
                        row for row in listings if row.up_token_id is not None and row.down_token_id is not None
                    ]
                    active = {listing.slug for listing in listings}
                    self._known.update({listing.slug: listing for listing in listings})
                    await self._settle_missing(active, now)
                    for listing in listings:
                        await self._take_profit(listing, now)
                        pending = self._pending.get(listing.slug)
                        if pending is not None:
                            done = await self._submit_entry(pending, now)
                            if done:
                                self._pending.pop(listing.slug, None)
                            continue
                        if listing.slug in self.strategy.positions:
                            continue
                        product = self.market_client.ASSETS[listing.asset]
                        signal = self.strategy.evaluate(
                            listing,
                            self._target_for(listing),
                            now,
                            self.feed.snapshot(product),
                        )
                        if signal is not None:
                            pending = PendingLiveEntry(
                                listing=listing,
                                signal=signal,
                                deadline=now + self.execution_window_seconds,
                            )
                            self._pending[listing.slug] = pending
                            if await self._submit_entry(pending, now):
                                self._pending.pop(listing.slug, None)
                    keep = active | set(self.strategy.positions) | set(self._pending)
                    self.strategy.prune(keep)
                    self._targets = {k: v for k, v in self._targets.items() if k in keep}
                except Exception:
                    logger.exception("POLYMARKET LIVE SCAN FAILED; retrying")
                cycle += 1
                if max_cycles is None or cycle < max_cycles:
                    await asyncio.sleep(self.poll_interval)
        finally:
            await self.feed.stop()
