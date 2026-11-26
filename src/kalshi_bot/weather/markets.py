"""Hourly temperature and daily rain markets, under the same simple rule.

Rule (shared with the daily highs/lows in ``simple.py``): turn the National
Weather Service forecast into odds, and make at most ONE bet per market
unit, only when Kalshi's price is at least ``simple_margin`` below those odds
after fees. Hold to settlement.

* **Hourly temperature** ("NYC 35° or above at 5am?"): settles a few minutes
  after the hour on a blend of 5-8 metro airports. Odds = NWS hourly forecast
  for that hour, +/- a spread that grows with how far ahead the hour is. One
  bet per city-hour, either YES or NO on whichever strike is best. No bets in
  the last ``hourly_min_lead_minutes`` before the reading.
* **Daily rain** ("Where will it rain today?" has one yes/no market per city):
  odds = NWS chance of rain over the city's local day. One bet per
  city-day, YES or NO. No bets later than ``rain_cutoff_hours`` into the day.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from kalshi_bot.exchange.models import (
    Action,
    Market,
    OrderRequest,
    OrderType,
    Side,
    TimeInForce,
)
from kalshi_bot.weather.brackets import bracket_from_text, parse_event_date
from kalshi_bot.weather.fees import fee_per_contract
from kalshi_bot.weather.settings import WeatherSettings
from kalshi_bot.weather.stations import Station
from kalshi_bot.weather.strategy import Budget, Decision


def _place(code: str, name: str, lat: float, lon: float, std: int) -> Station:
    """A forecast location that isn't a daily-temperature settlement station."""
    return Station(code, "", name, lat, lon, std, "", "", "")


# Hourly series -> forecast point near the middle of the airport blend.
HOURLY: dict[str, Station] = {
    "KXTEMPNYCHS": _place("NYC-H", "New York City hourly", 40.74, -73.94, -5),
    "KXTEMPCHIHS": _place("CHI-H", "Chicago hourly", 41.88, -87.80, -6),
    "KXTEMPMIAH": _place("MIA-H", "Miami hourly", 25.85, -80.25, -5),
    "KXTEMPLAXHS": _place("LAX-H", "LA coastal hourly", 33.95, -118.40, -8),
}

# Kalshi's own live index (the number hourly markets settle on) is read from
# GET /live_data/weather/{city}. Candidate city IDs per series: the first one
# Kalshi recognises is remembered.
HOURLY_INDEX_IDS: dict[str, list[str]] = {
    "KXTEMPNYCHS": ["nyc", "new-york", "newyork", "new_york", "ny"],
    "KXTEMPCHIHS": ["chicago", "chi"],
    "KXTEMPMIAH": ["miami", "mia"],
    "KXTEMPLAXHS": ["la-coastal", "la_coastal", "lacoastal", "los-angeles", "la", "lax"],
}

# Fallback if the index can't be read: the index's member airports, averaged.
# NYC and LA lists from minutetemp.com (Kalshi re-estimates small per-station
# offsets weekly, which this ignores); Chicago and Miami are best guesses.
HOURLY_STATIONS: dict[str, list[str]] = {
    "KXTEMPNYCHS": ["KLGA", "KEWR", "KTEB", "KCDW", "KHPN", "KFRG", "KSMQ", "KISP"],
    "KXTEMPCHIHS": ["KORD", "KMDW", "KPWK", "KDPA", "KARR"],
    "KXTEMPMIAH": ["KMIA", "KFLL", "KOPF", "KTMB", "KHWO"],
    "KXTEMPLAXHS": ["KHHR", "KLAX", "KSMO", "KLGB", "KSNA"],
}

RAIN_SERIES = "KXRAIN"

# Rain is settled at the city's main airport; the station nearest downtown
# can be a different one (Tampa downtown -> Peter O. Knight, not KTPA).
RAIN_STATIONS: dict[str, str] = {"Tampa": "KTPA", "New Orleans": "KMSY"}

# City name (as Kalshi writes it) -> (lat, lon, standard UTC offset).
RAIN_CITIES: dict[str, tuple[float, float, int]] = {
    "Albuquerque": (35.08, -106.65, -7), "Atlanta": (33.75, -84.39, -5),
    "Austin": (30.27, -97.74, -6), "Baltimore": (39.29, -76.61, -5),
    "Birmingham": (33.52, -86.80, -6), "Boise": (43.62, -116.20, -7),
    "Boston": (42.36, -71.06, -5), "Buffalo": (42.89, -78.88, -5),
    "Charleston": (32.78, -79.93, -5), "Charlotte": (35.23, -80.84, -5),
    "Chicago": (41.88, -87.63, -6), "Cincinnati": (39.10, -84.51, -5),
    "Cleveland": (41.50, -81.69, -5), "College Station": (30.63, -96.33, -6),
    "Columbus": (39.96, -83.00, -5), "Corpus Christi": (27.80, -97.40, -6),
    "Dallas": (32.78, -96.80, -6), "Denver": (39.74, -104.99, -7),
    "Detroit": (42.33, -83.05, -5), "El Paso": (31.76, -106.49, -7),
    "Fort Lauderdale": (26.12, -80.14, -5), "Fort Worth": (32.76, -97.33, -6),
    "Honolulu": (21.31, -157.86, -10), "Houston": (29.76, -95.37, -6),
    "Indianapolis": (39.77, -86.16, -5), "Jacksonville": (30.33, -81.66, -5),
    "Kansas City": (39.10, -94.58, -6), "Key West": (24.56, -81.78, -5),
    "Las Vegas": (36.17, -115.14, -8), "Little Rock": (34.75, -92.29, -6),
    "Los Angeles": (34.05, -118.24, -8), "Louisville": (38.25, -85.76, -5),
    "Memphis": (35.15, -90.05, -6), "Miami": (25.76, -80.19, -5),
    "Milwaukee": (43.04, -87.91, -6), "Minneapolis": (44.98, -93.27, -6),
    "Napa": (38.30, -122.29, -8), "Nashville": (36.16, -86.78, -6),
    "New Orleans": (29.95, -90.07, -6), "New York": (40.71, -74.01, -5),
    "Oklahoma City": (35.47, -97.52, -6), "Omaha": (41.26, -95.93, -6),
    "Orlando": (28.54, -81.38, -5), "Philadelphia": (39.95, -75.17, -5),
    "Phoenix": (33.45, -112.07, -7), "Pittsburgh": (40.44, -80.00, -5),
    "Portland": (45.52, -122.68, -8), "Raleigh": (35.78, -78.64, -5),
    "Reno": (39.53, -119.81, -8), "Richmond": (37.54, -77.44, -5),
    "Sacramento": (38.58, -121.49, -8), "Salt Lake City": (40.76, -111.89, -7),
    "San Antonio": (29.42, -98.49, -6), "San Diego": (32.72, -117.16, -8),
    "San Francisco": (37.77, -122.42, -8), "San Jose": (37.34, -121.89, -8),
    "Savannah": (32.08, -81.09, -5), "Seattle": (47.61, -122.33, -8),
    "Springfield": (37.21, -93.29, -6), "St. Louis": (38.63, -90.20, -6),
    "Saint Louis": (38.63, -90.20, -6), "Tampa": (27.95, -82.46, -5),
    "Tucson": (32.22, -110.97, -7), "Tulsa": (36.15, -95.99, -6),
    "Washington": (38.91, -77.04, -5), "West Palm Beach": (26.71, -80.05, -5),
}  # fmt: skip


def rain_city(market: Market) -> str | None:
    """Longest known city name found in the market's labels."""
    text = " ".join(x for x in (market.yes_sub_title, market.subtitle, market.title) if x).lower()
    hits = [c for c in RAIN_CITIES if re.search(rf"\b{re.escape(c.lower())}\b", text)]
    return max(hits, key=len) if hits else None


def rain_place(city: str) -> Station:
    lat, lon, std = RAIN_CITIES[city]
    return _place("R-" + city.replace(" ", "").replace(".", ""), f"{city} rain", lat, lon, std)


def _eastern_offset(day: date) -> int:
    """US Eastern UTC offset (DST from 2nd Sunday of March to 1st Sunday of Nov)."""
    mar = date(day.year, 3, 8)
    dst_start = mar + timedelta(days=(6 - mar.weekday()) % 7)
    nov = date(day.year, 11, 1)
    dst_end = nov + timedelta(days=(6 - nov.weekday()) % 7)
    return -4 if dst_start <= day < dst_end else -5


_HOUR_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})(\d{2})(?:-|$)")


def hourly_target(market: Market) -> datetime | None:
    """When the hourly reading is taken: market close time, rounded to the hour.

    Falls back to the event ticker (``...-26NOV1605`` = 5 AM Eastern).
    """
    if market.close_time:
        try:
            t = datetime.fromisoformat(market.close_time.replace("Z", "+00:00"))
            t = t.astimezone(UTC)
            return (t + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
        except ValueError:
            pass
    m = _HOUR_RE.search((market.event_ticker or market.ticker).upper())
    d = parse_event_date(market.event_ticker or market.ticker)
    if not m or d is None:
        return None
    hour = int(m.group(4))
    local = datetime(d.year, d.month, d.day, hour, tzinfo=UTC)
    return local - timedelta(hours=_eastern_offset(d))


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def hourly_sigma(lead_hours: float) -> float:
    """How far off an NWS hourly forecast typically is, by hours ahead."""
    return min(4.5, max(1.5, 1.5 + 0.15 * lead_hours))


def threshold_prob(market: Market, mu: float, sigma: float) -> float | None:
    """P(YES) for an 'X° or above' / 'X° or below' contract."""
    br = bracket_from_text(market.yes_sub_title or market.subtitle)
    if br is None:
        return None
    if br.hi is None and br.lo is not None:
        return 1.0 - _phi((br.lo - mu) / sigma)
    if br.lo is None and br.hi is not None:
        return _phi((br.hi - mu) / sigma)
    if br.lo is not None and br.hi is not None:
        return _phi((br.hi - mu) / sigma) - _phi((br.lo - mu) / sigma)
    return None


@dataclass(slots=True)
class Candidate:
    market: Market
    p: float  # model probability of YES
    label: str


def edges(c: Candidate, ws: WeatherSettings) -> tuple[float | None, float | None, int | None]:
    """(YES edge, NO edge, NO ask) after fees; None where not tradable."""
    m = c.market
    ye = None
    if m.yes_ask and ws.min_price <= m.yes_ask <= ws.max_price:
        ye = c.p - m.yes_ask / 100 - fee_per_contract(m.yes_ask, ws.fee_rate)
    no_ask = m.no_ask or (100 - m.yes_bid if m.yes_bid else None)
    ne = None
    if no_ask and ws.min_price <= no_ask <= ws.max_price:
        ne = (1 - c.p) - no_ask / 100 - fee_per_contract(no_ask, ws.fee_rate)
    return ye, ne, no_ask


def decide_binary(
    cands: list[Candidate],
    unit_prefix: str,
    event_ticker: str,
    day_iso: str,
    positions: dict[str, int],
    budget: Budget,
    ws: WeatherSettings,
    why: str,
) -> list[Decision]:
    """At most one bet (YES or NO) across ``cands``; none if the unit already has one."""
    if any(t.startswith(unit_prefix) and n != 0 for t, n in positions.items()):
        return []
    best: tuple[float, Candidate, Side, int] | None = None
    for c in cands:
        ye, ne, no_ask = edges(c, ws)
        if ye is not None and c.market.yes_ask and (best is None or ye > best[0]):
            best = (ye, c, Side.YES, c.market.yes_ask)
        if ne is not None and no_ask and (best is None or ne > best[0]):
            best = (ne, c, Side.NO, no_ask)
    if best is None or best[0] < ws.simple_margin:
        return []
    edge, c, side, price = best
    count = min(ws.simple_stake_cents // price, ws.max_contracts_per_order)
    if side is Side.YES and c.market.yes_ask_size:
        count = min(count, c.market.yes_ask_size)
    room = budget.remaining(ws, c.market.ticker, event_ticker, day_iso)
    count = min(count, max(0, room) // price)
    if count <= 0:
        return []
    budget.add(c.market.ticker, event_ticker, day_iso, count * price)
    p_side = c.p if side is Side.YES else 1 - c.p
    return [
        Decision(
            order=OrderRequest(
                ticker=c.market.ticker,
                action=Action.BUY,
                side=side,
                count=count,
                type=OrderType.LIMIT,
                yes_price=price if side is Side.YES else None,
                no_price=price if side is Side.NO else None,
                time_in_force=TimeInForce.IMMEDIATE_OR_CANCEL,
                exchange_index=c.market.exchange_index,
            ),
            reason=f"{why}: {side.value.upper()} on {c.label} ({p_side:.0%} vs {price}c)",
            kind="entry",
            p_model=c.p,
            p_used=c.p,
            edge=edge,
            price=price,
        )
    ]


def candidate_rows(cands: list[Candidate], ws: WeatherSettings) -> list[dict[str, object]]:
    """Dashboard/log rows for a set of yes/no contracts."""
    rows: list[dict[str, object]] = []
    for c in cands:
        ye, ne, no_ask = edges(c, ws)
        m = c.market
        rows.append(
            {
                "ticker": m.ticker,
                "label": c.label,
                "p": round(c.p, 4),
                "yes_bid": m.yes_bid,
                "yes_ask": m.yes_ask,
                "no_bid": m.no_bid,
                "no_ask": no_ask,
                "edge_yes": None if ye is None else round(ye, 4),
                "edge_no": None if ne is None else round(ne, 4),
                "volume": m.volume,
            }
        )
    return rows
