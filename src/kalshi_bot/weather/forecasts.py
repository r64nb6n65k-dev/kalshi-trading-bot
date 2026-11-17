"""Open-Meteo forecast fetching (ensembles, deterministic models, archives).

Every model is requested on its own, so any key in the ``hourly`` block that
starts with ``temperature_2m`` is a member (``temperature_2m`` is the control,
``temperature_2m_memberNN`` the perturbed members), whatever suffix
Open-Meteo uses. Model IDs are tried in order within a group, and the first
that returns data is remembered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import httpx
import numpy as np

from kalshi_bot.telemetry.logging import get_logger
from kalshi_bot.weather.stations import Station

logger = get_logger(__name__)

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


@dataclass(slots=True)
class HourlySet:
    """Hourly °F series for one model: ``values[m, t]`` for member m."""

    source: str  # e.g. "ens:gfs_seamless", "det:ncep_nbm_conus"
    times: np.ndarray  # unix seconds, shape (T,)
    values: np.ndarray  # shape (M, T), NaN where missing

    @property
    def is_ensemble(self) -> bool:
        return self.source.startswith("ens:")

    @property
    def n_members(self) -> int:
        return int(self.values.shape[0])


@dataclass(slots=True)
class ForecastBundle:
    station: Station
    fetched_at: datetime
    hourly: list[HourlySet] = field(default_factory=list)
    nws_highs: dict[date, float] = field(default_factory=dict)
    nws_lows: dict[date, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def parse_hourly(payload: dict[str, Any], source: str) -> HourlySet | None:
    hourly = payload.get("hourly") or {}
    raw_times = hourly.get("time") or []
    if not raw_times:
        return None
    if isinstance(raw_times[0], (int, float)):
        times = np.asarray(raw_times, dtype=np.int64)
    else:
        times = np.asarray(
            [
                int(datetime.fromisoformat(str(t)).replace(tzinfo=UTC).timestamp())
                for t in raw_times
            ],
            dtype=np.int64,
        )
    keys = sorted(k for k in hourly if k.startswith("temperature_2m"))
    rows: list[np.ndarray] = []
    for k in keys:
        vals = np.asarray([np.nan if v is None else float(v) for v in hourly[k]], dtype=np.float64)
        if vals.shape[0] == times.shape[0] and np.isfinite(vals).any():
            rows.append(vals)
    if not rows:
        return None
    return HourlySet(source=source, times=times, values=np.vstack(rows))


class OpenMeteoClient:
    def __init__(self, api_key: str = "", timeout: float = 30.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        self._resolved: dict[tuple[str, ...], str] = {}
        self._api_key = api_key.strip()

    def _url(self, url: str) -> str:
        # Paid plans use the same paths on "customer-" prefixed hosts.
        return url.replace("https://", "https://customer-", 1) if self._api_key else url

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _fetch(
        self,
        url: str,
        station: Station,
        model: str,
        **extra: Any,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "latitude": f"{station.lat:.4f}",
            "longitude": f"{station.lon:.4f}",
            "hourly": "temperature_2m",
            "models": model,
            "temperature_unit": "fahrenheit",
            "timezone": "GMT",
            "timeformat": "unixtime",
        }
        params.update(extra)
        if self._api_key:
            params["apikey"] = self._api_key
        resp = await self._client.get(self._url(url), params=params)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        if data.get("error"):
            raise RuntimeError(str(data.get("reason")))
        return data

    async def fetch_group(
        self,
        url: str,
        station: Station,
        group: list[str],
        prefix: str,
        errors: list[str],
        **extra: Any,
    ) -> HourlySet | None:
        """Fetch the first model in ``group`` that returns data."""
        key = (url, *group)
        candidates = [self._resolved[key]] if key in self._resolved else list(group)
        for model in candidates:
            try:
                data = await self._fetch(url, station, model, **extra)
            except Exception as exc:
                errors.append(f"{prefix}:{model}: {exc}")
                continue
            hs = parse_hourly(data, f"{prefix}:{model}")
            if hs is not None:
                self._resolved[key] = model
                return hs
            errors.append(f"{prefix}:{model}: empty response")
        return None

    async def forecast_sets(
        self,
        station: Station,
        ensemble_groups: list[list[str]],
        deterministic_groups: list[list[str]],
        errors: list[str],
    ) -> list[HourlySet]:
        out: list[HourlySet] = []
        for group in ensemble_groups:
            hs = await self.fetch_group(
                ENSEMBLE_URL, station, group, "ens", errors, forecast_days=4, past_days=1
            )
            if hs is not None:
                out.append(hs)
        for group in deterministic_groups:
            hs = await self.fetch_group(
                FORECAST_URL, station, group, "det", errors, forecast_days=4, past_days=1
            )
            if hs is not None:
                out.append(hs)
        return out

    async def historical_set(
        self,
        station: Station,
        model: str,
        start: date,
        end: date,
    ) -> HourlySet | None:
        """Archived short-lead forecasts (for calibration)."""
        data = await self._fetch(
            HISTORICAL_FORECAST_URL,
            station,
            model,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        )
        return parse_hourly(data, f"det:{model}")


def window_extreme(
    hs: HourlySet,
    start: datetime,
    end: datetime,
    kind_high: bool,
    min_hours: int = 20,
) -> np.ndarray:
    """Per-member max (or min) over hours in ``[start, end)``; NaN if sparse."""
    t0, t1 = int(start.timestamp()), int(end.timestamp())
    mask = (hs.times >= t0) & (hs.times < t1)
    vals = hs.values[:, mask]
    out = np.full(hs.n_members, np.nan)
    if vals.shape[1] == 0:
        return out
    counts = np.isfinite(vals).sum(axis=1)
    with np.errstate(all="ignore"):
        ext = np.nanmax(vals, axis=1) if kind_high else np.nanmin(vals, axis=1)
    ok = counts >= min(min_hours, vals.shape[1])
    out[ok] = ext[ok]
    return out
