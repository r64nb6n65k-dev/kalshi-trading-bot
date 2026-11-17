"""End-to-end engine cycle with fake Kalshi / Open-Meteo / NWS clients."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kalshi_bot.exchange.models import Market
from kalshi_bot.risk.manager import RiskManager
from kalshi_bot.weather.datalog import DataLog
from kalshi_bot.weather.engine import WeatherEngine
from kalshi_bot.weather.forecasts import HourlySet
from kalshi_bot.weather.nws import GridForecast
from kalshi_bot.weather.observations import ObsReading
from kalshi_bot.weather.settings import WeatherSettings
from kalshi_bot.weather.stations import Station
from tests.unit.weather_helpers import DAY, bundle, ladder

NOW = datetime(2026, 10, 27, 12, tzinfo=UTC)


class FakeKalshi:
    authenticated = False

    def __init__(self, markets: list[Market], result: str | None = None) -> None:
        self.markets = markets
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def get_all_markets(self, **params: Any) -> list[Market]:
        self.calls.append(params)
        return self.markets if params.get("series_ticker") == "KXHIGHNY" else []

    async def get_market(self, ticker: str) -> Market:
        return Market(ticker=ticker, result=self.result)


class FakeOpenMeteo:
    async def forecast_sets(self, *_: Any, **__: Any) -> list[HourlySet]:
        return bundle(70.0).hourly

    async def aclose(self) -> None:
        return None


class FakeNws:
    async def grid_forecast(self, _: Station) -> GridForecast:
        return GridForecast(highs={DAY: 70.8}, lows={})

    async def observations(self, *_: Any) -> list[ObsReading]:
        return []

    async def aclose(self) -> None:
        return None


def _engine(tmp_path: Path, mode: str, markets: list[Market], result: str | None = None):  # type: ignore[no-untyped-def]
    ws = WeatherSettings(
        stations="NYC",
        kinds="high",
        data_dir=str(tmp_path),
        samples=10_000,
        partition_arb=False,
        settle_check_interval=1,
    )
    return WeatherEngine(
        FakeKalshi(markets, result),  # type: ignore[arg-type]
        ws,
        RiskManager(),
        mode=mode,
        now_fn=lambda: NOW,
        openmeteo=FakeOpenMeteo(),  # type: ignore[arg-type]
        nws=FakeNws(),  # type: ignore[arg-type]
    )


def test_paper_cycle_trades_logs_and_settles(tmp_path: Path) -> None:
    # The 70-71 bracket is offered at 5c although the model gives it ~25%.
    eng = _engine(tmp_path, "paper", ladder({"70": (2, 5)}), result="yes")
    snaps = asyncio.run(eng.cycle())
    assert len(snaps) == 1
    s = snaps[0]
    assert abs(sum(s.probs.values()) - 1.0) < 1e-9
    entries = [d for d in s.decisions if d.kind == "entry"]
    assert any(d.order.ticker.endswith("B70.5") for d in entries)
    rows = list(DataLog(tmp_path).rows())
    assert {r["type"] for r in rows} >= {"model", "order"}
    # Paper position recorded; settled only once the climate day is over.
    assert eng.ledger.positions
    eng.now_fn = lambda: datetime(2026, 10, 29, 12, tzinfo=UTC)
    eng._last_settle = float("-inf")
    asyncio.run(eng._settle_paper())
    assert not eng.ledger.positions and eng.ledger.realized_cents != 0


def test_log_mode_never_trades(tmp_path: Path) -> None:
    eng = _engine(tmp_path, "log", ladder({"70": (2, 5)}))
    snaps = asyncio.run(eng.cycle())
    assert snaps and all(not s.decisions for s in snaps)
    assert not eng.ledger.positions
    assert [r["type"] for r in DataLog(tmp_path).rows()] == ["model"]


def test_rules_mismatch_blocks_event(tmp_path: Path) -> None:
    markets = ladder({"70": (2, 5)}, rules="highest temperature at LaGuardia Airport")
    eng = _engine(tmp_path, "paper", markets)
    assert asyncio.run(eng.cycle()) == []
