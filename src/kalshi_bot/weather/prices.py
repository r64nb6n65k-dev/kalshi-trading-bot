"""Price markets (crypto, commodities, stock indexes) decided near the close.

Same idea as weather: no predicting. In the last few minutes before a
market closes, the bot compares the live price with the market's line. It
bets only when the price is so far from the line that, at the speed this
asset has actually been moving, it would take a 3-standard-deviation move
in the time left to cross it. That's roughly a 1-in-700 event; the bot calls
it 97% to leave room for surprises.

Prices come from your existing feed (``data/multi_asset.py``): Coinbase
trades for crypto, plus Kalshi's own stream of the CF Benchmarks indexes the
crypto markets settle on, and Kalshi's underlying stream for commodities
and indexes.
"""

from __future__ import annotations

import asyncio
import itertools
import math
from collections.abc import Sequence
from datetime import datetime

from kalshi_bot.data.multi_asset import MultiAssetPriceFeed
from kalshi_bot.exchange.models import Market, Side
from kalshi_bot.strategies.base import UnderlyingTick
from kalshi_bot.weather.live import Verdict

# Series name (after "KX") prefix -> price feed product.
PRODUCTS: dict[str, str] = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "DOGE": "DOGE-USD",
    "XRP": "XRP-USD", "ADA": "ADA-USD", "BCH": "BCH-USD", "BNB": "BNB-USD",
    "HYPE": "HYPE-USD", "NEAR": "NEAR-USD", "TON": "TON-USD", "ZEC": "ZEC-USD",
    "GOLD": "XAU_USD", "SILVER": "XAG_USD", "COPPER": "XCU_USD", "WTI": "WTICO_USD",
    "NATGAS": "NATGAS_USD", "INX": "INX-USD", "NASDAQ100": "NDQ-USD",
}  # fmt: skip
CRYPTO = {"BTC", "ETH", "SOL", "DOGE", "XRP", "ADA", "BCH", "BNB", "HYPE", "NEAR", "TON", "ZEC"}
INDEXES = {"INX", "NASDAQ100"}


def asset_for(series_ticker: str) -> str | None:
    """'KXBTC15M' -> 'BTC'. Longest matching name wins (NASDAQ100 before NAS...)."""
    s = series_ticker.upper()
    s = s[2:] if s.startswith("KX") else s
    hits = [k for k in PRODUCTS if s.startswith(k)]
    return max(hits, key=len) if hits else None


def vol_per_sqrt_second(ticks: Sequence[UnderlyingTick], now: datetime) -> float | None:
    """Realised volatility (log-return per sqrt-second) over the recent ticks.

    Needs 30+ ticks spanning 5+ minutes, else ``None`` (not enough evidence).
    """
    rows = [t for t in ticks if t.price > 0 and (now - t.timestamp).total_seconds() <= 900]
    if len(rows) < 30:
        return None
    span = (rows[-1].timestamp - rows[0].timestamp).total_seconds()
    if span < 300:
        return None
    sq, dt = 0.0, 0.0
    for a, b in itertools.pairwise(rows):
        gap = (b.timestamp - a.timestamp).total_seconds()
        if gap <= 0:
            continue
        sq += math.log(b.price / a.price) ** 2
        dt += gap
    if dt <= 0:
        return None
    # Floor so a quiet patch never makes the bot overconfident.
    return max(math.sqrt(sq / dt), 2e-5)


def _fmt(x: float) -> str:
    """99000 -> '99,000'; 0.6123 -> '0.6123'."""
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:,.2f}" if abs(x) >= 10 else f"{x:.4g}"


def contract_kind(market: Market) -> str | None:
    """'greater' / 'less' / 'between' (or the >=/<= forms), inferred from the
    strikes when Kalshi uses another label (e.g. on 15-minute up/down markets)."""
    st = (market.strike_type or "").lower()
    if st in ("greater", "greater_or_equal", "less", "less_or_equal", "between"):
        return st
    fl, cap = market.floor_strike, market.cap_strike
    if fl is not None and cap is not None:
        return "between"
    if fl is not None:
        return "greater_or_equal"
    if cap is not None:
        return "less_or_equal"
    return None


def price_verdict(
    market: Market, price: float, vol: float, seconds_left: float, z_needed: float
) -> Verdict | None:
    """Verdict if the price sits ``z_needed`` standard moves from the line(s)."""
    if seconds_left <= 0:
        return None
    move = price * vol * math.sqrt(max(seconds_left, 30.0))  # 1 std dev of the move left
    st = contract_kind(market) or ""
    fl, cap = market.floor_strike, market.cap_strike

    def z(line: float) -> float:
        return (price - line) / move

    why = f"price {_fmt(price)}, needs a {{z:.1f}}-std move in {seconds_left / 60:.1f} min"
    if st in ("greater", "greater_or_equal") and fl is not None:
        zz = z(fl)
        if abs(zz) >= z_needed:
            side = Side.YES if zz > 0 else Side.NO
            return Verdict(side, 0.97, f"line {_fmt(fl)}: " + why.format(z=abs(zz)))
    elif st in ("less", "less_or_equal") and cap is not None:
        zz = z(cap)
        if abs(zz) >= z_needed:
            side = Side.YES if zz < 0 else Side.NO
            return Verdict(side, 0.97, f"line {_fmt(cap)}: " + why.format(z=abs(zz)))
    elif st == "between" and fl is not None and cap is not None:
        lo, hi = z(fl), z(cap)
        if lo >= z_needed and -hi >= z_needed:
            return Verdict(Side.YES, 0.97, f"inside {_fmt(fl)}-{_fmt(cap)}: " + why.format(
                z=min(lo, -hi)))  # fmt: skip
        if -lo >= z_needed or hi >= z_needed:
            return Verdict(Side.NO, 0.97, f"outside {_fmt(fl)}-{_fmt(cap)}: " + why.format(
                z=max(-lo, hi)))  # fmt: skip
    return None


class DecidedPriceFeed(MultiAssetPriceFeed):
    """Your multi-asset feed with its crypto streams switched on as well.

    The base ``start()`` only runs the commodity streams; this also runs
    Coinbase trades and Kalshi's CF Benchmarks index stream.
    """

    async def start(self) -> None:
        await super().start()
        self._tasks.append(asyncio.create_task(self._run_coinbase(), name="crypto-coinbase"))
        self._tasks.append(asyncio.create_task(self._run_cf(), name="crypto-cf-benchmarks"))
