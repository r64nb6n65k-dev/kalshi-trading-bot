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
    # Text expected somewhere in Kalshi's rules for this market. Used only as a
    # safety check that the configured station matches the contract.
    rules_hint: str

    def series_for(self, kind: Kind) -> str:
        return self.high_series if kind is Kind.HIGH else self.low_series


# Series tickers come from Kalshi's public listings as of late 2026. Run
# ``kalshi-bot weather-check`` to confirm them; override any of them with
# KALSHI_WEATHER__SERIES_OVERRIDES='{"NYC:low": "KXLOWNEW"}'.
STATIONS: dict[str, Station] = {
    "NYC": Station(
        code="NYC",
        icao="KNYC",
        name="New York (Central Park)",
        lat=40.7789,
        lon=-73.9692,
        std_utc_offset_hours=-5,
        high_series="KXHIGHNY",
        low_series="KXLOWTNYC",
        rules_hint="Central Park",
    ),
    "CHI": Station(
        code="CHI",
        icao="KMDW",
        name="Chicago (Midway)",
        lat=41.7861,
        lon=-87.7522,
        std_utc_offset_hours=-6,
        high_series="KXHIGHCHI",
        low_series="KXLOWTCHI",
        rules_hint="Midway",
    ),
    "AUS": Station(
        code="AUS",
        icao="KAUS",
        name="Austin (Bergstrom)",
        lat=30.1831,
        lon=-97.6799,
        std_utc_offset_hours=-6,
        high_series="KXHIGHAUS",
        low_series="KXLOWTAUS",
        rules_hint="Austin",
    ),
    "MIA": Station(
        code="MIA",
        icao="KMIA",
        name="Miami (International)",
        lat=25.7881,
        lon=-80.3169,
        std_utc_offset_hours=-5,
        high_series="KXHIGHMIA",
        low_series="KXLOWTMIA",
        rules_hint="Miami",
    ),
    "DEN": Station(
        code="DEN",
        icao="KDEN",
        name="Denver (International)",
        lat=39.8466,
        lon=-104.6562,
        std_utc_offset_hours=-7,
        high_series="KXHIGHDEN",
        low_series="KXLOWTDEN",
        rules_hint="Denver",
    ),
}


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
