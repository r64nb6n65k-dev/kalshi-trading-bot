"""Public Polymarket market data used by the simulation-only engine.

The US gateway is always queried first.  Polymarket's rolling crypto listings
are not consistently present in its public category response, so simulation
may fall back to Polymarket's public Gamma feed for the same rolling slugs.
No authenticated endpoint and no order endpoint exists in this module.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import httpx


def _cents(value: Any) -> int | None:
    if value is None:
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not 0 <= price <= 1:
        return None
    return max(0, min(100, round(price * 100)))


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


@dataclass(frozen=True, slots=True)
class PolymarketListing:
    slug: str
    asset: str
    interval_minutes: int
    open_time: float
    close_time: float
    yes_bid: int | None
    yes_ask: int | None
    no_bid: int | None
    no_ask: int | None
    active: bool
    closed: bool
    source: str
    up_token_id: str | None = None
    down_token_id: str | None = None
    resolved_side: str | None = None

    @property
    def seconds_left(self) -> float:
        return self.close_time - time.time()


class PolymarketPublicClient:
    """Read-only discovery and quote client for rolling crypto markets."""

    ASSETS: ClassVar[dict[str, str]] = {
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "SOL": "SOL-USD",
        "XRP": "XRP-USD",
    }

    def __init__(
        self,
        *,
        intervals: tuple[int, ...] = (5, 15),
        us_base_url: str = "https://gateway.polymarket.us",
        gamma_base_url: str = "https://gamma-api.polymarket.com",
        timeout: float = 8.0,
        transport: httpx.AsyncBaseTransport | None = None,
        international_only: bool = False,
    ) -> None:
        invalid = set(intervals) - {5, 15}
        if invalid:
            raise ValueError(f"Unsupported intervals: {sorted(invalid)}")
        self.intervals = tuple(sorted(set(intervals)))
        self.us_base_url = us_base_url.rstrip("/")
        self.gamma_base_url = gamma_base_url.rstrip("/")
        self._http = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )
        self._next_us_probe = 0.0
        self.international_only = international_only

    async def __aenter__(self) -> PolymarketPublicClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    @staticmethod
    def candidate_slug(asset: str, interval_minutes: int, now: float) -> str:
        seconds = interval_minutes * 60
        start = int(now) // seconds * seconds
        return f"{asset.lower()}-updown-{interval_minutes}m-{start}"

    async def get_open_crypto_markets(self, now: float | None = None) -> list[PolymarketListing]:
        timestamp = time.time() if now is None else now
        candidates = [
            (asset, interval, self.candidate_slug(asset, interval, timestamp))
            for asset in self.ASSETS
            for interval in self.intervals
        ]
        rows: list[PolymarketListing | None] = [None] * len(candidates)
        if not self.international_only and timestamp >= self._next_us_probe:
            us_rows = await asyncio.gather(
                *(
                    self._from_us_search(asset, interval, slug)
                    for asset, interval, slug in candidates
                )
            )
            rows = list(us_rows)
            # Re-check periodically because the US public catalog is still
            # expanding, but do not double the request count on every tick.
            self._next_us_probe = timestamp + 300
        missing = [index for index, row in enumerate(rows) if row is None]
        fallback_rows = await asyncio.gather(
            *(
                self._from_gamma(*candidates[index])
                for index in missing
            )
        )
        for index, row in zip(missing, fallback_rows, strict=True):
            rows[index] = row
        return [row for row in rows if row is not None and row.active and not row.closed]

    async def refresh(self, listing: PolymarketListing) -> PolymarketListing | None:
        return await self._load_listing(listing.asset, listing.interval_minutes, listing.slug)

    async def _load_listing(
        self, asset: str, interval_minutes: int, slug: str
    ) -> PolymarketListing | None:
        # Query the regulated US data service first.  Search is used because the
        # documented single-market route is not available for every listing.
        us = await self._from_us_search(asset, interval_minutes, slug)
        if us is not None:
            return us
        return await self._from_gamma(asset, interval_minutes, slug)

    async def _from_us_search(
        self, asset: str, interval_minutes: int, slug: str
    ) -> PolymarketListing | None:
        try:
            response = await self._http.get(
                f"{self.us_base_url}/v1/search",
                params={"query": slug, "limit": 25},
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            match = self._find_slug(response.json(), slug)
            return self._parse_us(match, asset, interval_minutes) if match else None
        except (httpx.HTTPError, TypeError, ValueError):
            return None

    @classmethod
    def _find_slug(cls, value: Any, slug: str) -> dict[str, Any] | None:
        if isinstance(value, dict):
            if value.get("slug") == slug and ("marketSides" in value or "bestBidQuote" in value):
                return value
            for child in value.values():
                found = cls._find_slug(child, slug)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = cls._find_slug(child, slug)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _times(slug: str, interval_minutes: int) -> tuple[float, float] | None:
        try:
            opened = float(int(slug.rsplit("-", 1)[-1]))
        except (TypeError, ValueError):
            return None
        return opened, opened + interval_minutes * 60

    @classmethod
    def _parse_us(
        cls, row: dict[str, Any], asset: str, interval_minutes: int
    ) -> PolymarketListing | None:
        slug = str(row.get("slug", ""))
        times = cls._times(slug, interval_minutes)
        if times is None:
            return None
        yes_bid = _cents((row.get("bestBidQuote") or {}).get("value"))
        yes_ask = _cents((row.get("bestAskQuote") or {}).get("value"))
        if yes_bid is None or yes_ask is None:
            sides = row.get("marketSides", [])
            if isinstance(sides, list):
                for side in sides:
                    if isinstance(side, dict) and side.get("long") is True:
                        yes_ask = yes_ask or _cents((side.get("quote") or {}).get("value"))
                        yes_bid = yes_bid or _cents(side.get("price"))
        no_bid = None if yes_ask is None else 100 - yes_ask
        no_ask = None if yes_bid is None else 100 - yes_bid
        status = str(row.get("status", row.get("ep3Status", ""))).upper()
        result = str(row.get("result", "")).lower()
        resolved = result if result in {"yes", "no", "up", "down"} else None
        return PolymarketListing(
            slug=slug,
            asset=asset,
            interval_minutes=interval_minutes,
            open_time=times[0],
            close_time=times[1],
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            active=bool(row.get("active", True)),
            closed=bool(row.get("closed", False)) or "RESOLVED" in status,
            source="POLYMARKET_US",
            resolved_side=resolved,
        )

    async def _from_gamma(
        self, asset: str, interval_minutes: int, slug: str
    ) -> PolymarketListing | None:
        try:
            response = await self._http.get(f"{self.gamma_base_url}/events/slug/{slug}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            event = response.json()
            markets = event.get("markets", []) if isinstance(event, dict) else []
            row = markets[0] if isinstance(markets, list) and markets else None
            if not isinstance(row, dict):
                return None
            return self._parse_gamma(row, asset, interval_minutes)
        except (httpx.HTTPError, TypeError, ValueError):
            return None

    @classmethod
    def _parse_gamma(
        cls, row: dict[str, Any], asset: str, interval_minutes: int
    ) -> PolymarketListing | None:
        slug = str(row.get("slug", ""))
        times = cls._times(slug, interval_minutes)
        if times is None:
            return None
        outcomes = [str(x).lower() for x in _json_list(row.get("outcomes"))]
        prices = _json_list(row.get("outcomePrices"))
        tokens = [str(x) for x in _json_list(row.get("clobTokenIds"))]
        try:
            up_index = next(i for i, value in enumerate(outcomes) if value in {"up", "yes"})
        except StopIteration:
            up_index = 0
        try:
            down_index = next(i for i, value in enumerate(outcomes) if value in {"down", "no"})
        except StopIteration:
            down_index = 1 if len(tokens) > 1 and up_index == 0 else 0
        yes_bid = _cents(row.get("bestBid"))
        yes_ask = _cents(row.get("bestAsk"))
        if yes_bid is None and up_index < len(prices):
            yes_bid = _cents(prices[up_index])
        if yes_ask is None:
            yes_ask = yes_bid
        no_bid = None if yes_ask is None else 100 - yes_ask
        no_ask = None if yes_bid is None else 100 - yes_bid
        resolved = cls._gamma_result(outcomes, prices, bool(row.get("closed", False)))
        return PolymarketListing(
            slug=slug,
            asset=asset,
            interval_minutes=interval_minutes,
            open_time=times[0],
            close_time=times[1],
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            active=bool(row.get("active", True)),
            closed=bool(row.get("closed", False)),
            source="POLYMARKET_PUBLIC",
            up_token_id=tokens[up_index] if up_index < len(tokens) else None,
            down_token_id=tokens[down_index] if down_index < len(tokens) else None,
            resolved_side=resolved,
        )

    @staticmethod
    def _gamma_result(outcomes: list[str], prices: list[Any], closed: bool) -> str | None:
        if not closed or not outcomes or len(outcomes) != len(prices):
            return None
        cents = [_cents(value) for value in prices]
        if 100 not in cents:
            return None
        winner = outcomes[cents.index(100)]
        if winner in {"up", "yes"}:
            return "yes"
        if winner in {"down", "no"}:
            return "no"
        return None

    async def resolved(self, listing: PolymarketListing) -> PolymarketListing | None:
        refreshed = await self.refresh(listing)
        if refreshed is None:
            return None
        if refreshed.resolved_side in {"yes", "no"}:
            return refreshed
        return replace(refreshed, closed=refreshed.closed)
