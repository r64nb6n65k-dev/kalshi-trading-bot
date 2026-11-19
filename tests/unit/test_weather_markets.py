"""Hourly temperature and daily rain markets."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from kalshi_bot.exchange.models import Market, Side
from kalshi_bot.weather.brackets import parse_event_date
from kalshi_bot.weather.dashboard import build_data, market_type
from kalshi_bot.weather.ledger import day_of
from kalshi_bot.weather.markets import (
    Candidate,
    decide_binary,
    hourly_target,
    rain_city,
    threshold_prob,
)
from kalshi_bot.weather.nws import GridForecast, periods
from kalshi_bot.weather.settings import WeatherSettings
from kalshi_bot.weather.strategy import Budget
from tests.unit.test_weather_engine import _engine
from tests.unit.weather_helpers import DAY, ladder

NOW = datetime(2026, 10, 27, 12, tzinfo=UTC)


def _m(ticker: str, label: str, bid: int, ask: int, **kw: Any) -> Market:
    return Market.model_validate(
        {
            "ticker": ticker,
            "event_ticker": ticker.rsplit("-", 1)[0],
            "yes_sub_title": label,
            "yes_bid": bid,
            "yes_ask": ask,
            "no_bid": 100 - ask,
            "no_ask": 100 - bid,
            "yes_ask_size": 100,
            **kw,
        }
    )


def test_dates_with_hour_suffix() -> None:
    assert parse_event_date("KXTEMPNYCHS-26NOV1605") == date(2026, 11, 16)
    assert day_of("KXTEMPNYCHS-26NOV1605-T35") == "2026-11-16"
    assert market_type("KXTEMPNYCHS-26NOV1605-T35") == "Hourly"
    assert market_type("KXRAIN-26NOV16-HOU") == "Rain"
    assert market_type("KXLOWTDEN-26NOV16-B30.5") == "Daily low"


def test_hourly_target_time() -> None:
    m = _m("KXTEMPNYCHS-26NOV1605-T35", "35° or above", 40, 44, close_time="2026-11-16T09:59:59Z")
    assert hourly_target(m) == datetime(2026, 11, 16, 10, tzinfo=UTC)
    # Without a close time: 05 in the ticker is 5 AM Eastern (EST in November).
    m2 = _m("KXTEMPNYCHS-26NOV1605-T35", "35° or above", 40, 44)
    assert hourly_target(m2) == datetime(2026, 11, 16, 10, tzinfo=UTC)
    m3 = _m("KXTEMPNYCHS-26JUL0105-T80", "80° or above", 40, 44)  # EDT
    assert hourly_target(m3) == datetime(2026, 7, 1, 9, tzinfo=UTC)


def test_threshold_probabilities() -> None:
    at = _m("X-1", "35° or above", 1, 2)
    below = _m("X-2", "35° or below", 1, 2)
    assert threshold_prob(at, 35.0, 2.0) == pytest.approx(0.5)
    assert threshold_prob(at, 30.0, 2.0) < 0.01
    assert threshold_prob(below, 30.0, 2.0) > 0.99


@pytest.mark.parametrize(
    ("label", "city"),
    [
        ("Will it rain in West Palm Beach on Nov 17, 2026?", "West Palm Beach"),
        ("St. Louis", "St. Louis"),
        ("Las Vegas", "Las Vegas"),
        ("Gotham", None),
    ],
)
def test_rain_city_matching(label: str, city: str | None) -> None:
    assert rain_city(_m("KXRAIN-26NOV17-X", label, 1, 2)) == city


def test_grid_periods_and_rain_chance() -> None:
    props = {
        "probabilityOfPrecipitation": {
            "uom": "wmoUnit:percent",
            "values": [
                {"validTime": "2026-10-28T06:00:00+00:00/PT12H", "value": 20},
                {"validTime": "2026-10-28T18:00:00+00:00/PT12H", "value": 60},
            ],
        },
        "temperature": {
            "uom": "wmoUnit:degC",
            "values": [{"validTime": "2026-10-28T10:00:00+00:00/PT1H", "value": 0}],
        },
    }
    g = GridForecast(
        {}, {}, periods(props, "temperature"), periods(props, "probabilityOfPrecipitation")
    )
    assert g.temp_at(datetime(2026, 10, 28, 10, 30, tzinfo=UTC)) == pytest.approx(32.0)
    day = (datetime(2026, 10, 28, 6, tzinfo=UTC), datetime(2026, 10, 29, 6, tzinfo=UTC))
    # max 0.6, independent 1-0.8*0.4=0.68 -> midpoint 0.64
    assert g.rain_chance(*day) == pytest.approx(0.64)


def test_decide_binary_one_bet_best_side() -> None:
    ws = WeatherSettings()
    cands = [
        Candidate(_m("E-1", "35° or above", 40, 44), 0.13, "35°+"),
        Candidate(_m("E-2", "33° or above", 50, 54), 0.40, "33°+"),
    ]
    ds = decide_binary(cands, "E-", "E", "2026-10-28", {}, Budget(), ws, "test")
    assert len(ds) == 1
    d = ds[0]
    # 35+: NO costs 60c for an 87% chance -> best edge
    assert d.order.ticker == "E-1" and d.order.side is Side.NO and d.order.no_price == 60
    assert d.order.count == ws.simple_stake_cents // 60
    # Already have a bet on this unit -> nothing.
    assert decide_binary(cands, "E-", "E", "2026-10-28", {"E-2": 3}, Budget(), ws, "t") == []


class FakeKalshiAll:
    authenticated = False

    def __init__(self) -> None:
        self.by_series: dict[str, list[Market]] = {
            "KXHIGHNY": ladder({"70": (18, 20)}),
            "KXTEMPNYCHS": [
                _m("KXTEMPNYCHS-26OCT2805-T35", "35° or above", 40, 44,
                   close_time="2026-10-28T10:00:00Z"),
                _m("KXTEMPNYCHS-26OCT2805-T33", "33° or above", 45, 49,
                   close_time="2026-10-28T10:00:00Z"),
            ],
            "KXRAIN": [
                _m("KXRAIN-26OCT28-HOU", "Houston", 50, 54),
                _m("KXRAIN-26OCT28-GOT", "Gotham", 50, 54),
            ],
        }  # fmt: skip

    async def get_all_markets(self, **params: Any) -> list[Market]:
        return self.by_series.get(params.get("series_ticker", ""), [])

    async def get_market(self, ticker: str) -> Market:
        return Market(ticker=ticker, result="no", status="finalized")


class FakeNwsAll:
    async def grid_forecast(self, _: Any) -> GridForecast:
        return GridForecast(
            highs={DAY: 70.8},
            lows={},
            temps=[(datetime(2026, 10, 27, tzinfo=UTC), datetime(2026, 10, 30, tzinfo=UTC), 30.0)],
            pops=[
                (
                    datetime(2026, 10, 28, 6, tzinfo=UTC),
                    datetime(2026, 10, 28, 18, tzinfo=UTC),
                    0.9,
                ),
                (
                    datetime(2026, 10, 28, 18, tzinfo=UTC),
                    datetime(2026, 10, 29, 6, tzinfo=UTC),
                    0.5,
                ),
            ],
        )

    async def aclose(self) -> None:
        return None


def test_engine_trades_daily_hourly_and_rain(tmp_path: Path) -> None:
    eng = _engine(tmp_path, "paper", [], strategy="simple")
    eng.client = FakeKalshiAll()  # type: ignore[assignment]
    eng.nws = FakeNwsAll()  # type: ignore[assignment]
    eng.ws = eng.ws.model_copy(update={"min_price": 12, "max_price": 88})
    snaps = asyncio.run(eng.cycle())
    assert len(snaps) == 1  # the daily high event
    kinds = {s["kind"] for s in eng.last_snapshots}
    assert kinds == {"high", "hourly", "rain"}
    bets = eng.ledger.signed_positions()
    assert bets["KXHIGHNY-26OCT28-B70.5"] > 0  # YES on the forecast range
    assert bets["KXTEMPNYCHS-26OCT2805-T35"] < 0  # NO: forecast 30°, strike 35
    assert bets["KXRAIN-26OCT28-HOU"] > 0  # YES: ~90% rain, price 54c
    assert not any("GOT" in t for t in bets)
    assert any("Gotham" in w for w in eng.warnings)

    # Same-day settlement check covers the hourly bet (no waiting until tomorrow).
    eng.now_fn = lambda: datetime(2026, 10, 28, 11, tzinfo=UTC)
    eng._last_settle = float("-inf")
    asyncio.run(eng._settle_paper())
    assert "KXTEMPNYCHS-26OCT2805-T35" not in eng.ledger.positions

    d = build_data(eng)
    types = {r["type"]: r for r in d["scoreboard"]}
    assert types["Hourly"]["wins"] == 1  # NO bet, result "no"
    assert types["Rain"]["losses"] == 1 and types["Daily high"]["losses"] == 1
    assert {"Hourly", "Rain", "Daily high"} <= set(types)
