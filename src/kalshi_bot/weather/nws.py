"""api.weather.gov client: gridpoint forecast extremes and station observations.

The NWS API is free and needs no key, but it does require a User-Agent that
identifies you (set ``KALSHI_WEATHER__USER_AGENT`` to something with your
email address).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from kalshi_bot.weather.observations import ObsReading, parse_observation
from kalshi_bot.weather.stations import Station, climate_day_window

NWS_BASE = "https://api.weather.gov"
_DURATION = re.compile(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?")


def parse_valid_time(valid: str) -> tuple[datetime, datetime] | None:
    """``2026-10-28T11:00:00+00:00/PT13H`` -> (start, end)."""
    try:
        start_s, dur_s = valid.split("/")
        start = datetime.fromisoformat(start_s.replace("Z", "+00:00"))
    except ValueError:
        return None
    m = _DURATION.fullmatch(dur_s)
    if not m:
        return None
    d, h, mi = (int(x) if x else 0 for x in m.groups())
    return start, start + timedelta(days=d, hours=h, minutes=mi)


def _to_f(value: float, uom: str) -> float:
    return value if uom.endswith("degF") else value * 9.0 / 5.0 + 32.0


def extremes_by_climate_day(
    grid_props: dict[str, Any],
    station: Station,
    key: str,
) -> dict[date, float]:
    """Map NWS ``maxTemperature``/``minTemperature`` periods onto climate days.

    Each period is assigned to the climate day containing its midpoint (the
    daytime max period lands on that day; the overnight min period ending in
    the morning lands on that morning's day).
    """
    block = grid_props.get(key) or {}
    uom = str(block.get("uom") or "wmoUnit:degC")
    out: dict[date, float] = {}
    for row in block.get("values") or []:
        if row.get("value") is None:
            continue
        span = parse_valid_time(str(row.get("validTime", "")))
        if span is None:
            continue
        mid = span[0] + (span[1] - span[0]) / 2
        local_std = mid.astimezone(UTC) + timedelta(hours=station.std_utc_offset_hours)
        day = local_std.date()
        start, end = climate_day_window(station, day)
        if start <= mid < end:
            out[day] = _to_f(float(row["value"]), uom)
    return out


@dataclass(slots=True)
class GridForecast:
    highs: dict[date, float]
    lows: dict[date, float]


class NwsClient:
    def __init__(self, user_agent: str, timeout: float = 20.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=NWS_BASE,
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept": "application/geo+json"},
            follow_redirects=True,
        )
        self._grid_urls: dict[str, str] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, url: str, **params: Any) -> dict[str, Any]:
        resp = await self._client.get(url, params=params or None)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data

    async def grid_url(self, station: Station) -> str:
        if station.code not in self._grid_urls:
            data = await self._get(f"/points/{station.lat:.4f},{station.lon:.4f}")
            self._grid_urls[station.code] = str(data["properties"]["forecastGridData"])
        return self._grid_urls[station.code]

    async def grid_forecast(self, station: Station) -> GridForecast:
        data = await self._get(await self.grid_url(station))
        props = data.get("properties") or {}
        return GridForecast(
            highs=extremes_by_climate_day(props, station, "maxTemperature"),
            lows=extremes_by_climate_day(props, station, "minTemperature"),
        )

    async def observations(
        self,
        station: Station,
        start: datetime,
        end: datetime,
    ) -> list[ObsReading]:
        data = await self._get(
            f"/stations/{station.icao}/observations",
            start=start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=end.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            limit=500,
        )
        readings: list[ObsReading] = []
        for feat in data.get("features") or []:
            readings.extend(parse_observation(feat.get("properties") or {}))
        return readings
