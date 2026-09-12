"""Live Polymarket execution for the rolling crypto momentum strategy."""

from __future__ import annotations

import asyncio
import inspect
import os
import time
from dataclasses import dataclass, replace
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Any

import httpx

from kalshi_bot.data.multi_crypto import MultiCryptoPriceFeed
from kalshi_bot.dashboard import record_model_snapshot
from kalshi_bot.exchange.models import Side
from kalshi_bot.polymarket import PolymarketListing, PolymarketPublicClient
from kalshi_bot.strategies.examples.polymarket_momentum import PolymarketMomentumStrategy, SimSignal
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required deployment variable: {name}")
    return value


def _value(obj: Any, *keys: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return next((obj[key] for key in keys if key in obj), default)
    return next((getattr(obj, key) for key in keys if hasattr(obj, key)), default)


def _fill_cents(response: Any, side: str, fallback: int) -> int:
    try:
        making = Decimal(str(_value(response, "makingAmount", "making_amount")))
        taking = Decimal(str(_value(response, "takingAmount", "taking_amount")))
        if making <= 0 or taking <= 0:
            return fallback
        price = making / taking if side == "BUY" else taking / making
        return max(1, min(99, int((price * 100).quantize(Decimal("1")))))
    except (InvalidOperation, TypeError, ValueError, ZeroDivisionError):
        return fallback


@dataclass(slots=True)
class PendingLiveEntry:
    listing: PolymarketListing
    signal: SimSignal
    deadline: float
    signed_order: Any | None = None
    shares: int = 0
    uncertain: bool = False


class PolymarketTradingClient:
    HOST = "https://clob.polymarket.com"
    CHAIN_ID = 137
    SIGNATURE_TYPE = 3

    def __init__(self) -> None:
        try:
            from py_clob_client_v2 import ApiCreds, ClobClient
        except ImportError as exc:
            raise RuntimeError("py-clob-client-v2 is not installed") from exc
        self.private_key = _require("POLYMARKET_SIGNER_PRIVATE_KEY")
        self.wallet = _require("POLYMARKET_WALLET_ADDRESS")
        self.host = os.getenv("POLYMARKET_CLOB_URL", self.HOST).rstrip("/")
        key = os.getenv("POLYMARKET_CLOB_API_KEY", "").strip()
        secret = os.getenv("POLYMARKET_CLOB_SECRET", "").strip()
        passphrase = os.getenv("POLYMARKET_CLOB_PASSPHRASE", "").strip()
        if key and secret and passphrase:
            creds = ApiCreds(api_key=key, api_secret=secret, api_passphrase=passphrase)
        elif any((key, secret, passphrase)):
            raise RuntimeError("Set all three Polymarket CLOB credential variables or none")
        else:
            temporary = ClobClient(
                host=self.host, chain_id=self.CHAIN_ID, key=self.private_key,
                signature_type=self.SIGNATURE_TYPE, funder=self.wallet, use_server_time=True,
            )
            creds = temporary.create_or_derive_api_key()
        self._client = ClobClient(
            host=self.host, chain_id=self.CHAIN_ID, key=self.private_key, creds=creds,
            signature_type=self.SIGNATURE_TYPE, funder=self.wallet,
            use_server_time=True, retry_on_error=False,
        )
        self._secure_client: Any | None = None
        self._secure_client_lock = asyncio.Lock()
        self._redemption_lock = asyncio.Lock()

    async def _secure(self) -> Any:
        if self._secure_client is not None:
            return self._secure_client
        async with self._secure_client_lock:
            if self._secure_client is not None:
                return self._secure_client
            try:
                from polymarket import AsyncSecureClient, RelayerApiKey
            except ImportError as exc:
                raise RuntimeError(
                    "polymarket-client is required for automatic redemption"
                ) from exc
            relayer_key = _require("POLYMARKET_RELAYER_API_KEY")
            relayer_address = _require("POLYMARKET_RELAYER_API_KEY_ADDRESS")
            self._secure_client = await AsyncSecureClient.create(
                private_key=self.private_key,
                wallet=self.wallet,
                api_key=RelayerApiKey(
                    key=relayer_key,
                    address=relayer_address,
                ),
            )
            return self._secure_client

    async def prepare_redemption(self) -> None:
        """Validate the gasless redemption credentials during startup."""
        await self._secure()

    async def redeemable_condition_ids(self) -> tuple[str, ...]:
        """Return every currently redeemable condition held by the wallet."""
        client = await self._secure()
        found: set[str] = set()
        pages = client.list_positions(
            user=self.wallet,
            status="REDEEMABLE",
            page_size=100,
        )
        async for page in pages:
            for position in page.items:
                if not bool(_value(position, "redeemable", default=False)):
                    continue
                condition_id = str(_value(position, "condition_id", "conditionId", default=""))
                if condition_id:
                    found.add(condition_id)
        return tuple(sorted(found))

    async def redeem_positions(self, condition_id: str) -> Any:
        """Redeem both outcome balances for one resolved condition and wait."""
        async with self._redemption_lock:
            client = await self._secure()
            transaction = await client.redeem_positions(condition_id=condition_id)
            return await transaction.wait()

    async def close(self) -> None:
        client = self._secure_client
        if client is None:
            return
        closer = getattr(client, "aclose", None) or getattr(client, "close", None)
        if closer is None:
            return
        result = closer()
        if inspect.isawaitable(result):
            await result

    async def check_geoblock(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get("https://polymarket.com/api/geoblock")
            response.raise_for_status()
            result = response.json()
        if not isinstance(result, dict):
            raise RuntimeError("Unexpected Polymarket geoblock response")
        return result

    async def collateral_balance(self) -> float | None:
        try:
            from py_clob_client_v2 import AssetType, BalanceAllowanceParams
            result = await asyncio.to_thread(
                self._client.get_balance_allowance,
                BalanceAllowanceParams(asset_type=AssetType.COLLATERAL),
            )
            raw = _value(result, "balance")
            if raw is None:
                return None
            amount = Decimal(str(raw))
            return float(amount / 1_000_000 if amount > 10_000 else amount)
        except Exception:
            logger.exception("Could not read Polymarket collateral balance")
            return None

    def _token(self, listing: PolymarketListing, side: Side) -> str:
        token = listing.up_token_id if side is Side.YES else listing.down_token_id
        if not token:
            raise RuntimeError(f"No token id for {listing.slug} {side.value}")
        return token

    async def executable_buy_quote(
        self, listing: PolymarketListing, side: Side, *,
        slippage_cents: int, maximum_price_cents: int,
    ) -> tuple[int, int, float] | None:
        book = await asyncio.to_thread(self._client.get_order_book, self._token(listing, side))
        levels: list[tuple[int, float]] = []
        for row in _value(book, "asks", default=[]) or []:
            try:
                price = Decimal(str(_value(row, "price")))
                cents = int((price * 100).to_integral_value(rounding=ROUND_CEILING))
                size = float(_value(row, "size", default=0))
            except (InvalidOperation, TypeError, ValueError):
                continue
            if 1 <= cents <= 99 and size > 0:
                levels.append((cents, size))
        if not levels:
            return None
        ask = min(price for price, _ in levels)
        limit = min(
            max(1, min(99, maximum_price_cents)),
            ask + max(0, slippage_cents),
        )
        depth = sum(size for price, size in levels if price <= limit)
        return ask, limit, depth

    async def build_limit_order(
        self, listing: PolymarketListing, side: Side, *, action: str,
        price_cents: int, shares: int,
    ) -> Any:
        from py_clob_client_v2 import OrderArgs, PartialCreateOrderOptions
        from py_clob_client_v2 import Side as ClobSide
        args = OrderArgs(
            token_id=self._token(listing, side), price=price_cents / 100,
            side=ClobSide.BUY if action == "BUY" else ClobSide.SELL, size=shares,
        )
        return await asyncio.to_thread(
            self._client.create_order, args, PartialCreateOrderOptions()
        )

    async def post_fok(self, order: Any) -> Any:
        from py_clob_client_v2 import OrderType
        return await asyncio.to_thread(
            self._client.post_order, order, OrderType.FOK, False, False
        )


class PolymarketLiveEngine:
    def __init__(
        self, *, market_client: PolymarketPublicClient,
        trading_client: PolymarketTradingClient, feed: MultiCryptoPriceFeed,
        strategy: PolymarketMomentumStrategy, poll_interval: float = 1.0,
        execution_window_seconds: float = 3.0, live_enabled: bool = False,
    ) -> None:
        self.market_client, self.trading_client = market_client, trading_client
        self.feed, self.strategy = feed, strategy
        self.poll_interval = max(0.25, poll_interval)
        self.execution_window_seconds = max(0.5, execution_window_seconds)
        self.live_enabled = live_enabled and _env_bool("POLY_LIVE")
        self.maximum_entry_price = max(
            1, min(99, int(os.getenv("POLY_MAX_ENTRY_PRICE", "85")))
        )
        self.auto_redeem = os.getenv("POLY_AUTO_REDEEM", "true").strip().lower() in {
            "1", "true", "yes", "on",
        }
        self.redeem_scan_seconds = max(
            5.0, float(os.getenv("POLY_REDEEM_SCAN_SECONDS", "15"))
        )
        self._targets: dict[str, float] = {}
        self._known: dict[str, PolymarketListing] = {}
        self._pending: dict[str, PendingLiveEntry] = {}
        self._exit_uncertain: set[str] = set()
        self._next_redeem_scan = 0.0
        self._redeemed_conditions: set[str] = set()
        self._redeem_retry_at: dict[str, float] = {}

    @staticmethod
    def _ok(response: Any) -> bool:
        return bool(_value(response, "success", "ok", default=False))

    @staticmethod
    def _status(response: Any) -> str:
        return str(_value(response, "status", default="")).lower()

    @staticmethod
    def _error(response: Any) -> str:
        return str(_value(response, "errorMsg", "error_msg", default="") or "")

    @classmethod
    def _no_fill(cls, response: Any) -> bool:
        error, status = cls._error(response).lower(), cls._status(response)
        return not cls._ok(response) and (
            "no match" in error or "no orders found" in error
            or ("fok" in error and "fill" in error) or status == "unmatched"
        )

    @staticmethod
    def _no_fill_exception(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(text in message for text in (
            "couldn't be fully filled", "could not be fully filled",
            "fully filled or killed", "no match",
        ))

    @staticmethod
    def _insufficient_collateral_exception(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(text in message for text in (
            "not enough balance", "insufficient balance", "insufficient collateral",
            "balance or allowance", "insufficient funds",
        ))

    async def _redeem_condition(self, condition_id: str, now: float) -> bool:
        if condition_id in self._redeemed_conditions:
            return True
        if now < self._redeem_retry_at.get(condition_id, 0.0):
            return False
        try:
            outcome = await self.trading_client.redeem_positions(condition_id)
        except Exception:
            self._redeem_retry_at[condition_id] = now + self.redeem_scan_seconds
            logger.exception(
                "POLYMARKET REDEEM FAILED | condition_id=%s | will_retry=true",
                condition_id,
            )
            return False
        self._redeemed_conditions.add(condition_id)
        self._redeem_retry_at.pop(condition_id, None)
        tx_hash = _value(outcome, "transaction_hash", "transactionHash", default="unknown")
        balance = await self.trading_client.collateral_balance()
        logger.warning(
            "POLYMARKET AUTO REDEEMED | condition_id=%s | tx=%s | collateral=%s",
            condition_id,
            tx_hash,
            "unknown" if balance is None else f"${balance:.2f}",
        )
        return True

    async def _redeem_wallet_positions(self, now: float, *, force: bool = False) -> None:
        if not self.auto_redeem or (not force and now < self._next_redeem_scan):
            return
        self._next_redeem_scan = now + self.redeem_scan_seconds
        try:
            condition_ids = await self.trading_client.redeemable_condition_ids()
        except Exception:
            logger.exception("POLYMARKET REDEEM SCAN FAILED; will retry")
            return
        for condition_id in condition_ids:
            await self._redeem_condition(condition_id, now)

    def _target(self, listing: PolymarketListing) -> float | None:
        if listing.slug in self._targets:
            return self._targets[listing.slug]
        ticks = self.feed.snapshot(self.market_client.ASSETS[listing.asset])
        if not ticks:
            return None
        opening = min(ticks, key=lambda x: abs(x.timestamp.timestamp() - listing.open_time))
        if abs(opening.timestamp.timestamp() - listing.open_time) > 5:
            return None
        self._targets[listing.slug] = opening.price
        logger.warning(
            "OPENING REFERENCE CAPTURED | ticker=%s | price=%.6f",
            listing.slug, opening.price,
        )
        return opening.price

    async def _submit(self, pending: PendingLiveEntry, now: float) -> bool:
        if now >= pending.deadline or pending.uncertain:
            return True
        try:
            if pending.signed_order is None:
                quote = await self.trading_client.executable_buy_quote(
                    pending.listing, pending.signal.side,
                    slippage_cents=self.strategy.entry_slippage_cents,
                    maximum_price_cents=self.maximum_entry_price,
                )
                if quote is None:
                    return False
                ask, limit, depth = quote
                record_model_snapshot(
                    ticker=pending.listing.slug,
                    seconds_left=pending.listing.close_time - now,
                    target_price=self._targets.get(pending.listing.slug),
                    yes_ask=ask if pending.signal.side is Side.YES else None,
                    no_ask=ask if pending.signal.side is Side.NO else None,
                    decision=f"CLOB_QUOTE_{pending.signal.side.value.upper()}",
                    reason=(
                        f"fresh_clob_ask={ask}c | limit={limit}c | "
                        f"maximum={self.maximum_entry_price}c | depth={depth:.4f}"
                    ),
                )
                if ask > self.maximum_entry_price:
                    logger.warning(
                        "LIVE CANCEL | ticker=%s | reason=FRESH_CLOB_ASK_ABOVE_MAX "
                        "| ask=%dc | maximum=%dc",
                        pending.listing.slug, ask, self.maximum_entry_price,
                    )
                    return True
                pending.shares = max(self.strategy.contracts, (100 + limit - 1) // limit)
                cost = limit * pending.shares
                if self.strategy.reserved_cents() + cost > self.strategy.bankroll_cents:
                    logger.warning(
                        "LIVE CANCEL | ticker=%s | reason=BANKROLL_CAP",
                        pending.listing.slug,
                    )
                    return True
                if depth + 1e-9 < pending.shares:
                    logger.warning(
                        "LIVE WAIT | ticker=%s | reason=INSUFFICIENT_DEPTH",
                        pending.listing.slug,
                    )
                    return False
                pending.signal = replace(pending.signal, signal_ask=ask, limit_price=limit)
                pending.signed_order = await self.trading_client.build_limit_order(
                    pending.listing, pending.signal.side, action="BUY",
                    price_cents=limit, shares=pending.shares,
                )
            response = await self.trading_client.post_fok(pending.signed_order)
        except Exception as exc:
            if self._no_fill_exception(exc):
                pending.signed_order = None
                logger.warning("LIVE RETRY | ticker=%s | reason=FOK_NO_FILL", pending.listing.slug)
                return False
            if self._insufficient_collateral_exception(exc):
                logger.error(
                    "LIVE CANCEL | ticker=%s | reason=INSUFFICIENT_COLLATERAL",
                    pending.listing.slug,
                )
                return True
            pending.uncertain = True
            self.strategy.decided.add(pending.listing.slug)
            logger.exception("ENTRY RESPONSE UNCERTAIN | ticker=%s", pending.listing.slug)
            return True
        if self._ok(response) and self._status(response) == "matched":
            fill = _fill_cents(response, "BUY", pending.signal.limit_price)
            self.strategy.open_position(
                pending.listing, pending.signal.side, fill, count=pending.shares,
                execution_mode="polymarket_live",
            )
            logger.warning(
                "POLYMARKET LIVE FILL | ticker=%s | fill=%dc | count=%d",
                pending.listing.slug, fill, pending.shares,
            )
            return True
        if self._no_fill(response):
            pending.signed_order = None
            return False
        logger.error(
            "ENTRY NOT CONFIRMED | ticker=%s | status=%s",
            pending.listing.slug, self._status(response),
        )
        # The exchange may have accepted an order even when its response is not
        # recognizable.  Lock this market rather than risk a duplicate buy.
        self.strategy.decided.add(pending.listing.slug)
        return True

    async def _take_profit(self, listing: PolymarketListing, now: float) -> None:
        position = self.strategy.positions.get(listing.slug)
        if position is None or listing.slug in self._exit_uncertain:
            return
        bid = listing.yes_bid if position.side is Side.YES else listing.no_bid
        if bid is None or bid < self.strategy.take_profit:
            return
        try:
            order = await self.trading_client.build_limit_order(
                listing, position.side, action="SELL",
                price_cents=self.strategy.take_profit, shares=position.count,
            )
            response = await self.trading_client.post_fok(order)
        except Exception as exc:
            if self._no_fill_exception(exc):
                return
            self._exit_uncertain.add(listing.slug)
            logger.exception("EXIT RESPONSE UNCERTAIN | ticker=%s", listing.slug)
            return
        if self._ok(response) and self._status(response) == "matched":
            fill = _fill_cents(response, "SELL", self.strategy.take_profit)
            self.strategy.close_position(listing.slug, fill, "TAKE_PROFIT_98", now)
            logger.warning("POLYMARKET LIVE EXIT | ticker=%s | fill=%dc", listing.slug, fill)

    async def _settle_missing(self, active: set[str], now: float) -> None:
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
            if won and self.auto_redeem:
                if not resolved.condition_id:
                    logger.error(
                        "POLYMARKET REDEEM WAIT | ticker=%s | reason=MISSING_CONDITION_ID",
                        slug,
                    )
                    continue
                if not await self._redeem_condition(resolved.condition_id, now):
                    continue
            self.strategy.close_position(
                slug, 100 if won else 0,
                "SETTLEMENT_WIN" if won else "SETTLEMENT_LOSS", now,
            )
            self._exit_uncertain.discard(slug)

    async def preflight(self) -> None:
        if not self.live_enabled:
            raise RuntimeError("Start with --live and set POLY_LIVE=true")
        geo = await self.trading_client.check_geoblock()
        if geo.get("blocked"):
            raise RuntimeError(f"Server is geoblocked: {geo.get('country')} / {geo.get('region')}")
        balance = await self.trading_client.collateral_balance()
        if self.auto_redeem:
            await self.trading_client.prepare_redemption()
        logger.warning(
            "POLYMARKET PREFLIGHT OK | collateral=%s",
            "unknown" if balance is None else f"${balance:.2f}",
        )

    async def run(self, max_cycles: int | None = None) -> None:
        await self.preflight()
        await self._redeem_wallet_positions(time.time(), force=True)
        logger.warning("POLYMARKET LIVE STARTED | LIVE_ORDERS=ENABLED")
        await self.feed.start()
        cycle = 0
        try:
            while max_cycles is None or cycle < max_cycles:
                try:
                    now = time.time()
                    await self._redeem_wallet_positions(now)
                    listings = await self.market_client.get_open_crypto_markets(now)
                    listings = [x for x in listings if x.up_token_id and x.down_token_id]
                    self._known.update({x.slug: x for x in listings})
                    await self._settle_missing({x.slug for x in listings}, now)
                    for listing in listings:
                        await self._take_profit(listing, now)
                        pending = self._pending.get(listing.slug)
                        if pending:
                            if await self._submit(pending, now):
                                self._pending.pop(listing.slug, None)
                            continue
                        if listing.slug in self.strategy.positions:
                            continue
                        product = self.market_client.ASSETS[listing.asset]
                        signal = self.strategy.evaluate(
                            listing, self._target(listing), now, self.feed.snapshot(product)
                        )
                        if signal:
                            pending = PendingLiveEntry(
                                listing, signal, now + self.execution_window_seconds
                            )
                            self._pending[listing.slug] = pending
                            if await self._submit(pending, now):
                                self._pending.pop(listing.slug, None)
                    keep = (
                        {x.slug for x in listings}
                        | set(self.strategy.positions)
                        | set(self._pending)
                    )
                    self.strategy.prune(keep)
                    self._targets = {
                        key: value for key, value in self._targets.items() if key in keep
                    }
                except Exception:
                    logger.exception("POLYMARKET LIVE SCAN FAILED; retrying")
                cycle += 1
                if max_cycles is None or cycle < max_cycles:
                    await asyncio.sleep(self.poll_interval)
        finally:
            await self.feed.stop()
            await self.trading_client.close()

