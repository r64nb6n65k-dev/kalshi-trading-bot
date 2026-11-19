"""Live strategy: bet what the weather has already decided.

No forecasts against Kalshi's traders. The bot only bets when live airport
readings (the same NWS data the markets settle on) show an outcome is
already settled, or nearly so, and then it buys the favourite.

Daily high:
  * The thermometer already passed a range, so that range can't win. Bet NO.
  * After 4 PM, the high hasn't moved, it's 3°F+ cooler now, and the NWS
    hourly forecast for the rest of the day stays 3°F+ below the high:
    the high is locked. Bet YES on its range.
Daily low: the mirror image (after 10 AM, with the rest of the day forecast
  to stay 3°F+ above the low so far).
Rain ("Where will it rain today?"):
  * The city's airport already measured rain today. Bet YES.
  * It's raining right now and the NWS gives 80%+ for the rest of the day. Bet YES.
  * After 8 PM: no rain yet, not raining, and NWS gives 5% or less for the
    rest of the day. Bet NO.
Hourly ("35° or above at 5 AM?"): within the last hour, current airport
  readings plus the NWS forecast change put the reading 5°F+ away from the
  strike. Bet the side it's on.

A bet is placed only if the price is below the chance of winning (minus a
2-cent cushion) and at most 97 cents.
"""

from __future__ import annotations

from dataclasses import dataclass

from kalshi_bot.exchange.models import Action, Market, OrderRequest, OrderType, Side, TimeInForce
from kalshi_bot.weather.brackets import Bracket, bracket_from_text
from kalshi_bot.weather.observations import ObsSummary, RainSummary
from kalshi_bot.weather.settings import WeatherSettings
from kalshi_bot.weather.stations import Kind
from kalshi_bot.weather.strategy import Budget, Decision

LOCK_MARGIN_F = 3.0  # how far below the high (above the low) the rest of the day must stay


def rnd(x: float) -> int:
    """Round half up, like the climate report's whole degrees."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


@dataclass(slots=True)
class Verdict:
    side: Side
    p_win: float  # chance our side wins
    why: str


def daily_verdicts(
    kind: Kind,
    brackets: list[tuple[str, Bracket]],
    obs: ObsSummary,
    rest_of_day: list[float],
    local_hour: float,
) -> dict[str, Verdict]:
    """Verdicts per ticker for today's high/low ranges, from what's been observed."""
    out: dict[str, Verdict] = {}
    if kind is Kind.HIGH:
        if obs.max_lo is None or obs.max_hi is None:
            return out
        floor = rnd(obs.max_lo)  # the final high can't be below this
        for ticker, b in brackets:
            if b.hi is not None and b.hi < floor:
                out[ticker] = Verdict(Side.NO, 0.995, f"already hit {obs.max_lo:.1f}°")
        now = obs.latest_temp_f
        locked = (
            local_hour >= 16
            and now is not None
            and now <= obs.max_lo - LOCK_MARGIN_F
            and (not rest_of_day or max(rest_of_day) <= obs.max_lo - LOCK_MARGIN_F)
        )
        if locked:
            # Readings come every 5 minutes; allow half a degree for a peak between them.
            lo_v, hi_v = floor, rnd(obs.max_hi + 0.5)
            why = (
                f"high locked: {obs.max_lo:.1f}° so far, now {now:.0f}°, rest of day ≤ "
                f"{max(rest_of_day, default=now):.0f}°"
            )
            for ticker, b in brackets:
                if b.contains(lo_v) and b.contains(hi_v):
                    out[ticker] = Verdict(Side.YES, 0.95, why)
                elif b.lo is not None and b.lo > hi_v:
                    out[ticker] = Verdict(Side.NO, 0.97, why)
        return out

    if obs.min_lo is None or obs.min_hi is None:
        return out
    ceiling = rnd(obs.min_hi)  # the final low can't be above this
    for ticker, b in brackets:
        if b.lo is not None and b.lo > ceiling:
            out[ticker] = Verdict(Side.NO, 0.995, f"already dropped to {obs.min_hi:.1f}°")
    now = obs.latest_temp_f
    locked = (
        local_hour >= 10
        and now is not None
        and now >= obs.min_hi + LOCK_MARGIN_F
        and (not rest_of_day or min(rest_of_day) >= obs.min_hi + LOCK_MARGIN_F)
    )
    if locked:
        lo_v, hi_v = rnd(obs.min_lo - 0.5), ceiling
        why = (
            f"low locked: {obs.min_hi:.1f}° so far, now {now:.0f}°, rest of day ≥ "
            f"{min(rest_of_day, default=now):.0f}°"
        )
        for ticker, b in brackets:
            if b.contains(lo_v) and b.contains(hi_v):
                out[ticker] = Verdict(Side.YES, 0.95, why)
            elif b.hi is not None and b.hi < lo_v:
                out[ticker] = Verdict(Side.NO, 0.97, why)
    return out


def rain_verdict(rain: RainSummary, pop_rest: float | None, local_hour: float) -> Verdict | None:
    if rain.inches >= 0.01:
        return Verdict(Side.YES, 0.97, f'{rain.inches:.2f}" of rain measured today')
    if rain.raining_now and pop_rest is not None and pop_rest >= 0.8:
        return Verdict(Side.YES, 0.9, f"raining now, NWS {pop_rest:.0%} for the rest of the day")
    if (
        local_hour >= 20
        and not rain.raining_now
        and pop_rest is not None
        and pop_rest <= 0.05
        and rain.latest_time is not None
    ):
        return Verdict(Side.NO, 0.95, f"dry all day, NWS {pop_rest:.0%} for the rest of the day")
    return None


def hourly_verdict(market: Market, projected: float, gap: float) -> Verdict | None:
    """Side for an 'X° or above/below' contract if the projection is ``gap``°F+ away."""
    b = bracket_from_text(market.yes_sub_title or market.subtitle)
    if b is None:
        return None
    if b.hi is None and b.lo is not None:  # "X or above"
        if projected - b.lo >= gap:
            return Verdict(Side.YES, 0.95, f"{projected:.0f}° expected vs {b.lo}° line")
        if b.lo - projected >= gap:
            return Verdict(Side.NO, 0.95, f"{projected:.0f}° expected vs {b.lo}° line")
    if b.lo is None and b.hi is not None:  # "X or below"
        if b.hi - projected >= gap:
            return Verdict(Side.YES, 0.95, f"{projected:.0f}° expected vs {b.hi}° line")
        if projected - b.hi >= gap:
            return Verdict(Side.NO, 0.95, f"{projected:.0f}° expected vs {b.hi}° line")
    return None


def lock_bet(
    market: Market,
    v: Verdict,
    event_ticker: str,
    day_iso: str,
    positions: dict[str, int],
    budget: Budget,
    ws: WeatherSettings,
) -> Decision | None:
    """Buy our side if it's priced below its chance of winning (minus the cushion)."""
    if positions.get(market.ticker):
        return None  # one bet per market
    if v.side is Side.YES:
        price = market.yes_ask
    else:
        price = market.no_ask or (100 - market.yes_bid if market.yes_bid else None)
    if not price:
        return None
    limit = min(ws.live_max_price, int(v.p_win * 100) - ws.live_cushion_cents)
    if price > limit:
        return None
    count = min(ws.live_stake_cents // price, ws.max_contracts_per_order)
    if v.side is Side.YES and market.yes_ask_size:
        count = min(count, market.yes_ask_size)
    room = budget.remaining(ws, market.ticker, event_ticker, day_iso)
    count = min(count, max(0, room) // price)
    if count <= 0:
        return None
    budget.add(market.ticker, event_ticker, day_iso, count * price)
    return Decision(
        order=OrderRequest(
            ticker=market.ticker,
            action=Action.BUY,
            side=v.side,
            count=count,
            type=OrderType.LIMIT,
            yes_price=price if v.side is Side.YES else None,
            no_price=price if v.side is Side.NO else None,
            time_in_force=TimeInForce.IMMEDIATE_OR_CANCEL,
            exchange_index=market.exchange_index,
        ),
        reason=f"{v.why} -> {v.side.value.upper()} at {price}c ({v.p_win:.0%} sure)",
        kind="entry",
        p_model=v.p_win if v.side is Side.YES else 1 - v.p_win,
        p_used=v.p_win,
        edge=v.p_win - price / 100,
        price=price,
    )
