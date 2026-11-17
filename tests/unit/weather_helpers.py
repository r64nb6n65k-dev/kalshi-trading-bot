"""Synthetic fixtures shared by the weather tests (no network)."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np

from kalshi_bot.exchange.models import Market
from kalshi_bot.weather.forecasts import ForecastBundle, HourlySet
from kalshi_bot.weather.stations import STATIONS, Station

NYC: Station = STATIONS["NYC"]
DAY = date(2026, 10, 28)


def diurnal_set(
    source: str,
    center_max: float,
    members: int = 30,
    spread: float = 1.5,
    amp: float = 8.0,
    start: datetime | None = None,
    hours: int = 24 * 5,
    seed: int = 1,
) -> HourlySet:
    """Hourly temps peaking at 20Z (3 PM EST) with max ~= center_max."""
    rng = np.random.default_rng(seed)
    t0 = start or datetime(2026, 10, 27, 0, tzinfo=UTC)
    times = np.array([int((t0 + timedelta(hours=h)).timestamp()) for h in range(hours)])
    hour_utc = (times // 3600) % 24
    base = (center_max - amp) + amp * np.cos(2 * math.pi * (hour_utc - 20) / 24.0)
    offsets = rng.normal(0, spread, members) if members > 1 else np.zeros(1)
    values = base[None, :] + offsets[:, None]
    return HourlySet(source=source, times=times, values=values)


def bundle(center_max: float = 70.0, with_nws: bool = True) -> ForecastBundle:
    b = ForecastBundle(station=NYC, fetched_at=datetime(2026, 10, 27, 12, tzinfo=UTC))
    b.hourly = [
        diurnal_set("ens:gfs_seamless", center_max, seed=1),
        diurnal_set("ens:ecmwf_ifs025", center_max, seed=2),
        diurnal_set("det:ncep_nbm_conus", center_max, members=1),
    ]
    if with_nws:
        b.nws_highs = {DAY: center_max + 0.8}
        b.nws_lows = {DAY: center_max - 16 - 0.8}
    return b


def market(
    ticker: str,
    label: str,
    *,
    strike_type: str,
    floor: float | None = None,
    cap: float | None = None,
    yes_bid: int | None = None,
    yes_ask: int | None = None,
    event: str = "KXHIGHNY-26OCT28",
    rules: str | None = "If the highest temperature recorded in Central Park, New York ...",
    close_time: str = "2026-10-29T04:59:00Z",
    yes_ask_size: int | None = 100,
) -> Market:
    data: dict[str, Any] = {
        "ticker": ticker,
        "event_ticker": event,
        "series_ticker": event.split("-")[0],
        "yes_sub_title": label,
        "strike_type": strike_type,
        "floor_strike": floor,
        "cap_strike": cap,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": None if yes_ask is None else 100 - yes_ask,
        "no_ask": None if yes_bid is None else 100 - yes_bid,
        "close_time": close_time,
        "rules_primary": rules,
        "yes_ask_size": yes_ask_size,
        "status": "active",
    }
    return Market.model_validate(data)


def ladder(
    quotes: dict[str, tuple[int, int]] | None = None,
    event: str = "KXHIGHNY-26OCT28",
    rules: str | None = "highest temperature in Central Park",
) -> list[Market]:
    """<=63, 64-65, 66-67, 68-69, 70-71, 72-73, 74-75, >=76."""
    q = quotes or {}
    rows: list[Market] = [
        market(
            f"{event}-T64",
            "63° or below",
            strike_type="less",
            cap=64,
            yes_bid=q.get("lo", (1, 3))[0],
            yes_ask=q.get("lo", (1, 3))[1],
            event=event,
            rules=rules,
        ),
    ]
    for lo in range(64, 76, 2):
        key = f"{lo}"
        bid, ask = q.get(key, (10, 14))
        rows.append(
            market(
                f"{event}-B{lo}.5",
                f"{lo}° to {lo + 1}°",
                strike_type="between",
                floor=lo,
                cap=lo + 1,
                yes_bid=bid,
                yes_ask=ask,
                event=event,
                rules=rules,
            )
        )
    rows.append(
        market(
            f"{event}-T75",
            "76° or above",
            strike_type="greater",
            floor=75,
            yes_bid=q.get("hi", (1, 3))[0],
            yes_ask=q.get("hi", (1, 3))[1],
            event=event,
            rules=rules,
        )
    )
    return rows
