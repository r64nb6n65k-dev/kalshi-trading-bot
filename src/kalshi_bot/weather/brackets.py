"""Turn Kalshi temperature markets into integer brackets and validate them.

The NWS climate report gives whole degrees Fahrenheit, so each market is an
inclusive integer range ``[lo, hi]`` (``None`` for an open tail). The bracket
is read two independent ways, from the human label ("65° to 66°", "73° or
above") and from the strike fields. If they disagree, or the event's brackets
don't tile the integers exactly once, the event is rejected rather than
traded.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from kalshi_bot.exchange.models import Market

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"],
        start=1,
    )
}
# Optional trailing hour for hourly markets: KXTEMPNYCHS-26NOV1605.
_EVENT_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})(?:\d{2})?(?:-|$)")
# Labels may use an en dash between numbers.
_RANGE_RE = re.compile(r"(-?\d+)\s*°?\s*(?:F\s*)?(?:to|-|\u2013|through)\s*(-?\d+)\s*°?", re.I)
_ABOVE_RE = re.compile(r"(-?\d+)\s*°?\s*(?:F\s*)?or\s+(?:above|higher|more|greater)", re.I)
_BELOW_RE = re.compile(r"(-?\d+)\s*°?\s*(?:F\s*)?or\s+(?:below|lower|less|fewer)", re.I)
_GT_RE = re.compile(r"(?:above|greater than|over|>)\s*(-?\d+)\s*°", re.I)
_LT_RE = re.compile(r"(?:below|less than|under|<)\s*(-?\d+)\s*°", re.I)


@dataclass(frozen=True, slots=True)
class Bracket:
    """Inclusive integer range ``[lo, hi]``; ``None`` means unbounded."""

    lo: int | None
    hi: int | None

    def contains(self, value: int) -> bool:
        return (self.lo is None or value >= self.lo) and (self.hi is None or value <= self.hi)

    def label(self) -> str:
        if self.lo is None:
            return f"<={self.hi}"
        if self.hi is None:
            return f">={self.lo}"
        if self.lo == self.hi:
            return f"{self.lo}"
        return f"{self.lo}-{self.hi}"


@dataclass(slots=True)
class BracketMarket:
    market: Market
    bracket: Bracket

    @property
    def ticker(self) -> str:
        return self.market.ticker


@dataclass(slots=True)
class WeatherEvent:
    """All brackets for one series + climate date."""

    event_ticker: str
    series_ticker: str
    day: date
    brackets: list[BracketMarket] = field(default_factory=list)
    valid: bool = True
    problem: str = ""

    @property
    def close_time(self) -> str | None:
        times = [b.market.close_time for b in self.brackets if b.market.close_time]
        return min(times) if times else None


def parse_event_date(event_ticker: str) -> date | None:
    """``KXHIGHNY-26OCT28`` -> ``date(2026, 10, 28)``."""
    m = _EVENT_DATE_RE.search(event_ticker.upper())
    if not m:
        return None
    yy, mon, dd = m.groups()
    month = _MONTHS.get(mon)
    if month is None:
        return None
    try:
        return date(2000 + int(yy), month, int(dd))
    except ValueError:
        return None


def bracket_from_text(text: str | None) -> Bracket | None:
    if not text:
        return None
    if m := _RANGE_RE.search(text):
        a, b = int(m.group(1)), int(m.group(2))
        return Bracket(min(a, b), max(a, b))
    if m := _ABOVE_RE.search(text):
        return Bracket(int(m.group(1)), None)
    if m := _BELOW_RE.search(text):
        return Bracket(None, int(m.group(1)))
    if m := _GT_RE.search(text):
        return Bracket(int(m.group(1)) + 1, None)
    if m := _LT_RE.search(text):
        return Bracket(None, int(m.group(1)) - 1)
    return None


def bracket_from_strikes(market: Market) -> Bracket | None:
    st = (market.strike_type or "").lower()
    fl, cap = market.floor_strike, market.cap_strike
    if st == "between" and fl is not None and cap is not None:
        return Bracket(math.ceil(fl), math.floor(cap))
    if st == "greater" and fl is not None:
        # strictly greater than floor
        return Bracket(math.floor(fl) + 1, None)
    if st == "greater_or_equal" and fl is not None:
        return Bracket(math.ceil(fl), None)
    if st == "less" and cap is not None:
        return Bracket(None, math.ceil(cap) - 1)
    if st == "less_or_equal" and cap is not None:
        return Bracket(None, math.floor(cap))
    return None


def parse_bracket(market: Market) -> tuple[Bracket | None, str]:
    """Return ``(bracket, problem)``. ``problem`` is empty when OK."""
    text = market.yes_sub_title or market.subtitle
    from_text = bracket_from_text(text)
    from_strikes = bracket_from_strikes(market)
    if from_text and from_strikes and from_text != from_strikes:
        return None, (
            f"{market.ticker}: label '{text}' -> {from_text.label()} but strikes "
            f"({market.strike_type} {market.floor_strike}/{market.cap_strike}) -> "
            f"{from_strikes.label()}"
        )
    chosen = from_text or from_strikes
    if chosen is None:
        return None, f"{market.ticker}: cannot parse bracket from '{text}'"
    return chosen, ""


def validate_partition(brackets: list[Bracket]) -> str:
    """Empty string if the brackets cover every integer exactly once."""
    if not brackets:
        return "no brackets"
    lows = [b for b in brackets if b.lo is None]
    highs = [b for b in brackets if b.hi is None]
    if len(lows) != 1 or len(highs) != 1:
        return f"expected one lower and one upper tail, got {len(lows)}/{len(highs)}"
    middle = sorted(
        (b for b in brackets if b.lo is not None and b.hi is not None), key=lambda b: b.lo or 0
    )
    ordered = [lows[0], *middle, highs[0]]
    for prev, nxt in itertools.pairwise(ordered):
        assert prev.hi is not None and nxt.lo is not None
        if nxt.lo != prev.hi + 1:
            return f"gap/overlap between {prev.label()} and {nxt.label()}"
    for b in middle:
        assert b.lo is not None and b.hi is not None
        if b.hi < b.lo:
            return f"inverted bracket {b.label()}"
    return ""


def group_events(markets: list[Market]) -> list[WeatherEvent]:
    """Group markets into validated :class:`WeatherEvent` objects."""
    by_event: dict[str, list[Market]] = defaultdict(list)
    for m in markets:
        if m.event_ticker:
            by_event[m.event_ticker].append(m)

    events: list[WeatherEvent] = []
    for event_ticker, rows in sorted(by_event.items()):
        day = parse_event_date(event_ticker)
        series = rows[0].series_ticker or event_ticker.split("-")[0]
        ev = WeatherEvent(event_ticker=event_ticker, series_ticker=series, day=day or date.min)
        if day is None:
            ev.valid, ev.problem = False, f"cannot parse date from {event_ticker}"
            events.append(ev)
            continue
        problems: list[str] = []
        for m in rows:
            br, problem = parse_bracket(m)
            if br is None:
                problems.append(problem)
            else:
                ev.brackets.append(BracketMarket(market=m, bracket=br))
        if problems:
            ev.valid, ev.problem = False, "; ".join(problems)
        else:
            gap = validate_partition([b.bracket for b in ev.brackets])
            if gap:
                ev.valid, ev.problem = False, gap
        ev.brackets.sort(key=lambda b: -10_000 if b.bracket.lo is None else b.bracket.lo)
        events.append(ev)
    return events
