"""The simple strategy: bet the NWS forecast, one bet per city-day, hold."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from kalshi_bot.exchange.models import Market, Side
from kalshi_bot.weather.brackets import group_events
from kalshi_bot.weather.calibration import KindCalibration
from kalshi_bot.weather.settings import WeatherSettings
from kalshi_bot.weather.simple import decide_simple, forecast_odds
from kalshi_bot.weather.stations import Kind
from kalshi_bot.weather.strategy import Budget
from tests.unit.test_weather_engine import _engine
from tests.unit.weather_helpers import DAY, NYC, ladder

BEFORE = datetime(2026, 10, 27, 15, tzinfo=UTC)  # day before the Oct 28 climate day
DURING = datetime(2026, 10, 28, 15, tzinfo=UTC)


def _odds(fc: float = 70.8, now: datetime = BEFORE):  # type: ignore[no-untyped-def]
    res = forecast_odds(NYC, Kind.HIGH, DAY, now, fc, KindCalibration())
    assert res is not None
    return res


def _setup(quotes: dict[str, tuple[int, int]], fc: float = 70.8):  # type: ignore[no-untyped-def]
    ev = group_events(ladder(quotes))[0]
    res = _odds(fc)
    return ev, {b.ticker: res.prob(b.bracket) for b in ev.brackets}


def test_odds_center_on_forecast_and_sum_to_one() -> None:
    res = _odds(70.8)
    ev = group_events(ladder())[0]
    probs = {b.bracket.label(): res.prob(b.bracket) for b in ev.brackets}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
    assert max(probs, key=lambda k: probs[k]) == "70-71"
    assert res.mean == pytest.approx(70.8, abs=0.1)
    assert res.std == pytest.approx(2.6, abs=0.15)  # default next-day spread


def test_buys_the_most_likely_range_when_cheap() -> None:
    ev, probs = _setup({"70": (18, 20)})
    ds = decide_simple(ev, probs, {}, Budget(), WeatherSettings(), 70.8, day_started=False)
    assert len(ds) == 1
    d = ds[0]
    assert d.order.ticker.endswith("B70.5") and d.order.side is Side.YES
    assert d.order.yes_price == 20 and d.order.count == 25  # $5 / 20c
    assert "NWS forecast 71°" in d.reason


@pytest.mark.parametrize(
    ("quotes", "positions", "started", "why"),
    [
        ({"70": (28, 30)}, {}, False, "too expensive: odds ~30%, price 30c"),
        ({"70": (8, 10)}, {}, False, "long shot under 12c"),
        ({"70": (18, 20)}, {"KXHIGHNY-26OCT28-B68.5": 5}, False, "already bet this city-day"),
        ({"70": (18, 20)}, {}, True, "day already started"),
    ],
)
def test_skips(
    quotes: dict[str, tuple[int, int]], positions: dict[str, int], started: bool, why: str
) -> None:
    ev, probs = _setup(quotes)
    assert decide_simple(ev, probs, positions, Budget(), WeatherSettings(), 70.8, started) == [], (
        why
    )


class NoOpenMeteo:
    async def forecast_sets(self, *_: Any, **__: Any) -> Any:
        raise AssertionError("simple strategy must not call Open-Meteo")

    async def aclose(self) -> None:
        return None


def test_engine_simple_bets_once_and_reports_settlement(tmp_path: Path) -> None:
    eng = _engine(tmp_path, "paper", ladder({"70": (18, 20)}), result=None, strategy="simple")
    eng.openmeteo = NoOpenMeteo()  # type: ignore[assignment]
    eng.ws = eng.ws.model_copy(update={"min_price": 12, "max_price": 88})
    snaps = asyncio.run(eng.cycle())
    assert [d.order.ticker for s in snaps for d in s.decisions] == ["KXHIGHNY-26OCT28-B70.5"]
    assert eng.last_snapshots[0]["source"] == "NWS forecast"
    # Second cycle: still only one bet for this city-day.
    eng.strategy.last_entry.clear()
    snaps = asyncio.run(eng.cycle())
    assert not [d for s in snaps for d in s.decisions]
    assert list(eng.ledger.positions) == ["KXHIGHNY-26OCT28-B70.5"]

    # Next day: Kalshi hasn't decided yet -> reported as waiting, not silently ignored.
    eng.now_fn = lambda: datetime(2026, 10, 29, 15, tzinfo=UTC)
    eng._last_settle = float("-inf")
    asyncio.run(eng._settle_paper())
    st = eng.settle_status
    assert st["past_day"] == 1 and st["settled_now"] == 0 and len(st["waiting"]) == 1


def test_settlement_errors_are_visible(tmp_path: Path) -> None:
    eng = _engine(tmp_path, "paper", ladder({"70": (18, 20)}), strategy="simple")
    asyncio.run(eng.cycle())
    assert eng.ledger.positions

    async def boom(_: str) -> Market:
        raise RuntimeError("401 unauthorized")

    eng.client.get_market = boom  # type: ignore[method-assign]
    eng.now_fn = lambda: datetime(2026, 10, 29, 15, tzinfo=UTC)
    eng._last_settle = float("-inf")
    asyncio.run(eng._settle_paper())
    assert "401 unauthorized" in eng.settle_status["errors"][0]
