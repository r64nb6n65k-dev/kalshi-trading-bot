"""The simple strategy: bet the National Weather Service forecast.

The whole idea, in four steps:

1. Take the NWS forecast high for the day (for example 64 degrees).
2. Turn it into odds for each Kalshi range. The NWS is usually within a
   couple of degrees, so 64-65 is the most likely range, with the ranges
   next to it a bit less likely, and so on.
3. Buy ONE bet per city per day, on the most likely range, but only if
   Kalshi's price is at least ``simple_margin`` (5 cents) below those odds
   after fees. If it's too expensive, skip it.
4. Bet before the day starts, then hold until Kalshi settles it. No
   selling early, no betting against the range, no long shots.

How wide the odds are (how far off the NWS usually is) comes from
``calibration.py``: 2.6 degrees for tomorrow by default, or whatever
``weather-calibrate`` measured for that city.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np

from kalshi_bot.exchange.models import Action, OrderRequest, OrderType, Side, TimeInForce
from kalshi_bot.weather.brackets import WeatherEvent
from kalshi_bot.weather.calibration import KindCalibration, lead_bucket
from kalshi_bot.weather.fees import fee_per_contract
from kalshi_bot.weather.model import ModelResult
from kalshi_bot.weather.settings import WeatherSettings
from kalshi_bot.weather.stations import Kind, Station, climate_day_window
from kalshi_bot.weather.strategy import Budget, Decision


def forecast_odds(
    station: Station,
    kind: Kind,
    day: date,
    now: datetime,
    nws_forecast: float,
    calib: KindCalibration,
) -> ModelResult | None:
    """Odds for every whole-degree outcome, centered on the NWS forecast.

    Returned as a :class:`ModelResult` so the dashboard, logging and report
    work the same as for the model strategy.
    """
    _, end = climate_day_window(station, day)
    lead_h = (end - now).total_seconds() / 3600.0
    if lead_h <= 0:
        return None
    bucket = lead_bucket(lead_h)
    sigma = calib.sigma_for(bucket)
    center = nws_forecast + calib.source_offset("nws", kind) + calib.bias

    # Exact normal odds for each whole degree, expressed as a weighted list of
    # outcomes (2,000 "votes" spread by probability) so ModelResult.prob works.
    lo, hi = math.floor(center - 6 * sigma), math.ceil(center + 6 * sigma)
    temps = np.arange(lo, hi + 1)

    def cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf((x - center) / (sigma * math.sqrt(2.0))))

    p = np.array([cdf(t + 0.5) - cdf(t - 0.5) for t in temps])
    votes = np.round(p / p.sum() * 2000).astype(int)
    values = np.repeat(temps, votes)

    return ModelResult(
        station=station.code,
        kind=kind,
        day=day,
        lead_hours=lead_h,
        bucket=bucket,
        mu=center,
        sigma_cal=sigma,
        ens_std=0.0,
        n_members=0,
        intraday=now >= climate_day_window(station, day)[0],
        centers={"nws": nws_forecast},
        values=values,
        notes=["NWS forecast"],
    )


def decide_simple(
    event: WeatherEvent,
    probs: dict[str, float],
    positions: dict[str, int],
    budget: Budget,
    ws: WeatherSettings,
    forecast: float,
    day_started: bool,
) -> list[Decision]:
    """Zero or one bet for this city-day. Every 'no' has a plain reason."""
    if day_started:
        return []  # only bet before the day starts
    prefix = event.event_ticker + "-"
    if any(t.startswith(prefix) and n != 0 for t, n in positions.items()):
        return []  # already have our one bet on this city-day

    pick = max(event.brackets, key=lambda b: probs.get(b.ticker, 0.0))
    p = probs.get(pick.ticker, 0.0)
    ask = pick.market.yes_ask
    if not ask or not ws.min_price <= ask <= ws.max_price:
        return []
    edge = p - ask / 100 - fee_per_contract(ask, ws.fee_rate)
    if edge < ws.simple_margin:
        return []

    count = ws.simple_stake_cents // ask
    if pick.market.yes_ask_size:
        count = min(count, pick.market.yes_ask_size)
    count = min(count, ws.max_contracts_per_order)
    room = budget.remaining(ws, pick.ticker, event.event_ticker, event.day.isoformat())
    count = min(count, max(0, room) // ask)
    if count <= 0:
        return []
    budget.add(pick.ticker, event.event_ticker, event.day.isoformat(), count * ask)

    label = pick.bracket.label()
    return [
        Decision(
            order=OrderRequest(
                ticker=pick.ticker,
                action=Action.BUY,
                side=Side.YES,
                count=count,
                type=OrderType.LIMIT,
                yes_price=ask,
                time_in_force=TimeInForce.IMMEDIATE_OR_CANCEL,
                exchange_index=pick.market.exchange_index,
            ),
            reason=(
                f"NWS forecast {forecast:.0f}° -> most likely {label}° ({p:.0%}); "
                f"Kalshi price {ask}c"
            ),
            kind="entry",
            p_model=p,
            p_used=p,
            edge=edge,
            price=ask,
        )
    ]
