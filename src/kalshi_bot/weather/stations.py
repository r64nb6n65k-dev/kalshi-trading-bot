# ruff: noqa: E501  (the station table is intentionally one row per city)
"""Settlement stations and the NWS "climate day".

Kalshi daily temperature markets settle on the NWS Daily Climate Report (CLI)
for one specific station. The CLI day runs midnight-to-midnight *local standard
time* all year. During daylight saving time that means 1:00 AM to 12:59 AM
local clock time. Because standard time has a fixed UTC offset, the climate
day is always the same fixed UTC window for a station, which is what
:func:`climate_day_window` returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import Enum


class Kind(str, Enum):
    """Which daily extreme a market settles on."""

    HIGH = "high"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class Station:
    """One settlement station and the Kalshi series that settle on it."""

    code: str  # short city code used in config, e.g. "NYC"
    icao: str  # station identifier used by NWS / IEM, e.g. "KNYC"
    name: str
    lat: float
    lon: float
    std_utc_offset_hours: int  # local STANDARD time offset, e.g. -5 for Eastern
    high_series: str
    low_series: str
    # Words expected in Kalshi's rules for this market, "|"-separated
    # alternatives. A safety check that the station matches the contract.
    rules_hint: str

    def series_for(self, kind: Kind) -> str:
        return self.high_series if kind is Kind.HIGH else self.low_series

    def rules_match(self, rules: str) -> bool:
        text = rules.lower()
        return any(h.strip().lower() in text for h in self.rules_hint.split("|") if h.strip())


# Every Kalshi daily-temperature city (Nov 2026). Settlement stations from
# Kalshi's rules as compiled at vweatherstation.com/prediction-markets; series
# from kalshi.com/category/climate/daily-temperature. A series that doesn't
# exist (e.g. a low market for a city that only has highs) simply has no
# markets and is skipped. "" means Kalshi has no such series.
# Override any with KALSHI_WEATHER__SERIES_OVERRIDES='{"NYC:low": "KXLOWNEW"}'.
#   code  icao   name                          lat      lon     std  high          low          rules hints
_TABLE: list[tuple[str, str, str, float, float, int, str, str, str]] = [
    ("NYC", "KNYC", "New York (Central Park)", 40.7789, -73.9692, -5, "KXHIGHNY", "KXLOWTNYC", "Central Park|New York"),
    ("CHI", "KMDW", "Chicago (Midway)", 41.7861, -87.7522, -6, "KXHIGHCHI", "KXLOWTCHI", "Midway|Chicago"),
    ("AUS", "KAUS", "Austin (Bergstrom)", 30.1831, -97.6799, -6, "KXHIGHAUS", "KXLOWTAUS", "Austin"),
    ("MIA", "KMIA", "Miami (International)", 25.7881, -80.3169, -5, "KXHIGHMIA", "KXLOWTMIA", "Miami"),
    ("DEN", "KDEN", "Denver (International)", 39.8466, -104.6562, -7, "KXHIGHDEN", "KXLOWTDEN", "Denver"),
    ("LAX", "KLAX", "Los Angeles (LAX)", 33.9382, -118.3866, -8, "KXHIGHLAX", "KXLOWTLAX", "Los Angeles|LAX"),
    ("PHX", "KPHX", "Phoenix (Sky Harbor)", 33.4277, -112.0038, -7, "KXHIGHTPHX", "KXLOWTPHX", "Phoenix"),
    ("SFO", "KSFO", "San Francisco (SFO)", 37.6197, -122.3647, -8, "KXHIGHTSFO", "KXLOWTSFO", "San Francisco|SFO"),
    ("SEA", "KSEA", "Seattle (Sea-Tac)", 47.4447, -122.3144, -8, "KXHIGHTSEA", "KXLOWTSEA", "Seattle|Sea-Tac"),
    ("LV", "KLAS", "Las Vegas (Harry Reid)", 36.0719, -115.1634, -8, "KXHIGHTLV", "KXLOWTLV", "Las Vegas"),
    ("SAN", "KSAN", "San Diego (Lindbergh)", 32.7336, -117.1831, -8, "KXHIGHTSAN", "KXLOWTSAN", "San Diego"),
    ("MIN", "KMSP", "Minneapolis (MSP)", 44.8831, -93.2289, -6, "KXHIGHTMIN", "KXLOWTMIN", "Minneapolis|MSP"),
    ("ATL", "KATL", "Atlanta (Hartsfield)", 33.6301, -84.4418, -5, "KXHIGHTATL", "KXLOWTATL", "Atlanta"),
    ("BOS", "KBOS", "Boston (Logan)", 42.3606, -71.0097, -5, "KXHIGHTBOS", "KXLOWTBOS", "Boston|Logan"),
    ("HOU", "KHOU", "Houston (Hobby)", 29.6375, -95.2825, -6, "KXHIGHTHOU", "KXLOWTHOU", "Hobby|Houston"),
    ("OKC", "KOKC", "Oklahoma City (Will Rogers)", 35.3889, -97.6006, -6, "KXHIGHTOKC", "KXLOWTOKC", "Oklahoma City"),
    ("DAL", "KDFW", "Dallas (DFW)", 32.8978, -97.0189, -6, "KXHIGHTDAL", "KXLOWTDAL", "Dallas|DFW|Fort Worth"),
    ("DC", "KDCA", "Washington DC (Reagan)", 38.8483, -77.0342, -5, "", "KXLOWTDC", "Washington|Reagan|DCA"),
    ("TTN", "KTTN", "Trenton (Mercer)", 40.2767, -74.8156, -5, "", "KXLOWTTTN", "Trenton"),
]  # fmt: skip

STATIONS: dict[str, Station] = {row[0]: Station(*row) for row in _TABLE}

ALL_STATION_CODES = ",".join(STATIONS)


def resolve_stations(
    codes: list[str],
    series_overrides: dict[str, str] | None = None,
) -> list[Station]:
    """Return configured stations, applying any ``CODE:kind`` series overrides."""
    overrides = {k.upper(): v for k, v in (series_overrides or {}).items()}
    out: list[Station] = []
    for raw in codes:
        code = raw.strip().upper()
        if not code:
            continue
        if code not in STATIONS:
            raise ValueError(f"Unknown station code '{code}'. Known: {', '.join(STATIONS)}")
        st = STATIONS[code]
        high = overrides.get(f"{code}:HIGH", st.high_series)
        low = overrides.get(f"{code}:LOW", st.low_series)
        if high != st.high_series or low != st.low_series:
            st = Station(
                code=st.code,
                icao=st.icao,
                name=st.name,
                lat=st.lat,
                lon=st.lon,
                std_utc_offset_hours=st.std_utc_offset_hours,
                high_series=high,
                low_series=low,
                rules_hint=st.rules_hint,
            )
        out.append(st)
    return out


def climate_day_window(station: Station, day: date) -> tuple[datetime, datetime]:
    """UTC ``[start, end)`` of the NWS climate day ``day`` for ``station``."""
    start_local_std = datetime(day.year, day.month, day.day, tzinfo=UTC)
    start = start_local_std - timedelta(hours=station.std_utc_offset_hours)
    return start, start + timedelta(hours=24)


def climate_today(station: Station, now: datetime) -> date:
    """The climate day currently in progress at ``station``."""
    local_std = now.astimezone(UTC) + timedelta(hours=station.std_utc_offset_hours)
    return local_std.date()
