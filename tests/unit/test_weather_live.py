"""Live strategy: only bet outcomes the readings show are already decided."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from kalshi_bot.exchange.models import Market, Side
from kalshi_bot.weather.brackets import Bracket
from kalshi_bot.weather.live import (
    Verdict,
    daily_verdicts,
    hourly_verdict,
    lock_bet,
    rain_verdict,
)
from kalshi_bot.weather.nws import GridForecast
from kalshi_bot.weather.observations import ObsReading, ObsSummary, RainSummary
from kalshi_bot.weather.settings import WeatherSettings
from kalshi_bot.weather.stations import Kind
from kalshi_bot.weather.strategy import Budget
from tests.unit.test_weather_engine import _engine
from tests.unit.weather_helpers import ladder

T = datetime(2026, 10, 28, 23, tzinfo=UTC)
LADDER = [
    ("A", Bracket(None, 59)),
    ("B", Bracket(60, 61)),
    ("C", Bracket(62, 63)),
    ("D", Bracket(64, 65)),
    ("E", Bracket(66, None)),
]


def _obs(max_lo: float, now: float, min_hi: float = 45.0) -> ObsSummary:
    return ObsSummary(
        window_start=T, window_end=T, max_lo=max_lo, max_hi=max_lo + 0.2,
        min_lo=min_hi - 0.2, min_hi=min_hi, latest_time=T, latest_temp_f=now, n_readings=40,
    )  # fmt: skip


def test_ranges_already_passed_are_certain_no() -> None:
    v = daily_verdicts(Kind.HIGH, LADDER, _obs(62.4, 62.0), [64.0], local_hour=13)
    assert set(v) == {"A", "B"}  # 62.4 rounds to 62: <=59 and 60-61 are dead
    assert all(x.side is Side.NO and x.p_win >= 0.99 for x in v.values())


def test_high_locks_in_the_evening() -> None:
    v = daily_verdicts(Kind.HIGH, LADDER, _obs(62.4, 57.0), [55.0, 52.0], local_hour=18)
    assert v["C"].side is Side.YES and v["C"].p_win == 0.95
    assert v["D"].side is Side.NO and v["E"].side is Side.NO  # can't reach 64+
    assert v["A"].side is Side.NO


@pytest.mark.parametrize(
    ("now", "rest", "hour", "why"),
    [
        (57.0, [55.0], 14, "too early in the day"),
        (61.0, [55.0], 18, "not 3° off the high yet"),
        (57.0, [61.0], 18, "forecast says it could warm back up"),
    ],
)
def test_high_not_locked(now: float, rest: list[float], hour: float, why: str) -> None:
    v = daily_verdicts(Kind.HIGH, LADDER, _obs(62.4, now), rest, local_hour=hour)
    assert "C" not in v, why


def test_low_locks_mid_day_and_rules_out_warmer_ranges() -> None:
    lows = [("L1", Bracket(None, 40)), ("L2", Bracket(41, 42)), ("L3", Bracket(43, 44)),
            ("L4", Bracket(45, None))]  # fmt: skip
    obs = _obs(60.0, 55.0, min_hi=42.3)
    v = daily_verdicts(Kind.LOW, lows, obs, [50.0, 48.0], local_hour=12)
    assert v["L2"].side is Side.YES  # 41-42 holds 41..42
    assert v["L3"].side is Side.NO and v["L4"].side is Side.NO  # already dropped to 42.3
    assert v["L1"].side is Side.NO


def test_rain_verdicts() -> None:
    assert rain_verdict(RainSummary(0.12, False, T), 0.1, 9).side is Side.YES  # type: ignore[union-attr]
    assert rain_verdict(RainSummary(0.0, True, T), 0.85, 9).side is Side.YES  # type: ignore[union-attr]
    assert rain_verdict(RainSummary(0.0, True, T), 0.5, 9) is None  # raining, but not sure
    assert rain_verdict(RainSummary(0.0, False, T), 0.03, 21).side is Side.NO  # type: ignore[union-attr]
    assert rain_verdict(RainSummary(0.0, False, T), 0.03, 14) is None  # too early to call dry


def _m(label: str, bid: int, ask: int, ticker: str = "X-1", **kw: Any) -> Market:
    return Market.model_validate(
        {"ticker": ticker, "event_ticker": ticker.rsplit("-", 1)[0], "yes_sub_title": label,
         "yes_bid": bid, "yes_ask": ask, "no_bid": 100 - ask, "no_ask": 100 - bid,
         "yes_ask_size": 100, **kw}
    )  # fmt: skip


def test_hourly_verdict_needs_a_big_gap() -> None:
    m = _m("65° or above", 10, 14)
    assert hourly_verdict(m, 58.0, 5.0).side is Side.NO  # type: ignore[union-attr]
    assert hourly_verdict(m, 62.0, 5.0) is None
    assert hourly_verdict(m, 71.0, 5.0).side is Side.YES  # type: ignore[union-attr]


def test_lock_bet_buys_favourite_but_not_overpriced() -> None:
    ws = WeatherSettings()
    v = Verdict(Side.YES, 0.95, "locked")
    d = lock_bet(_m("62° to 63°", 88, 90), v, "E", "2026-10-28", {}, Budget(), ws)
    assert d is not None and d.order.yes_price == 90 and d.order.count == 1000 // 90
    assert lock_bet(_m("62° to 63°", 93, 94), v, "E", "2026-10-28", {}, Budget(), ws) is None
    assert lock_bet(_m("62° to 63°", 88, 90), v, "E", "d", {"X-1": 3}, Budget(), ws) is None
    sure_no = Verdict(Side.NO, 0.995, "passed")
    d2 = lock_bet(_m("59° or below", 4, 6), sure_no, "E", "d", {}, Budget(), ws)
    assert d2 is not None and d2.order.side is Side.NO and d2.order.no_price == 96


class FakeKalshiLive:
    authenticated = False

    def __init__(self) -> None:
        self.by_series: dict[str, list[Market]] = {
            "KXHIGHNY": ladder({"lo": (88, 90)}),
            "KXTEMPNYCHS": [
                _m("65° or above", 10, 14, "KXTEMPNYCHS-26OCT2819-T64.99",
                   close_time="2026-10-29T00:00:00Z"),
            ],
            "KXRAIN": [_m("Miami", 60, 64, "KXRAIN-26OCT28-MIA")],
        }  # fmt: skip

    async def get_all_markets(self, **params: Any) -> list[Market]:
        return self.by_series.get(params.get("series_ticker", ""), [])

    async def get_market(self, ticker: str) -> Market:
        return Market(ticker=ticker)


def _inst(t: datetime, f: float) -> ObsReading:
    return ObsReading(t, "inst", f - 0.1, f + 0.1)


class FakeNwsLive:
    async def grid_forecast(self, _: Any) -> GridForecast:
        flat = (datetime(2026, 10, 27, tzinfo=UTC), datetime(2026, 10, 30, tzinfo=UTC), 50.0)
        return GridForecast({}, {}, [flat], [(flat[0], flat[1], 0.2)])

    async def observations_icao(self, icao: str, *_: Any) -> list[ObsReading]:
        return [
            _inst(datetime(2026, 10, 28, 19, tzinfo=UTC), 62.4),
            _inst(datetime(2026, 10, 28, 22, 50, tzinfo=UTC), 58.0),
            ObsReading(datetime(2026, 10, 28, 15, 51, tzinfo=UTC), "precip", 0.12, 0.12),
        ]

    async def nearest_station(self, _: Any) -> str:
        return "KMIA"

    async def aclose(self) -> None:
        return None


def test_engine_live_bets_only_decided_outcomes(tmp_path: Path) -> None:
    eng = _engine(tmp_path, "paper", [], strategy="live")
    eng.client = FakeKalshiLive()  # type: ignore[assignment]
    eng.nws = FakeNwsLive()  # type: ignore[assignment]
    eng.now_fn = lambda: T  # 6 PM in New York
    eng.ws = eng.ws.model_copy(update={"hourly": True})
    asyncio.run(eng.cycle())
    pos = eng.ledger.signed_positions()
    # High locked at 62.4 (rest of day 50°): YES on "63 or below" at 90c.
    assert pos["KXHIGHNY-26OCT28-T64"] > 0
    # Hourly: 58° now, flat forecast, 65+ line is 7° away -> NO at 90c.
    assert pos["KXTEMPNYCHS-26OCT2819-T64.99"] < 0
    # Miami measured 0.12" today -> YES at 64c.
    assert pos["KXRAIN-26OCT28-MIA"] > 0
    kinds = {s["kind"] for s in eng.last_snapshots}
    assert kinds == {"high", "hourly", "rain"}
    rain = next(s for s in eng.last_snapshots if s["kind"] == "rain")
    assert rain["hot"] and '0.12"' in rain["brackets"][0]["label"]


@pytest.mark.parametrize(
    ("price", "bought"),
    # This morning's real NYC hourly YES buys, all "95% sure".
    [(1, False), (8, False), (24, False), (56, False), (88, True), (91, True), (93, True)],
)
def test_never_buy_what_kalshi_calls_a_long_shot(price: int, bought: bool) -> None:
    v = Verdict(Side.YES, 0.95, "sure")
    d = lock_bet(_m("45° or above", max(price - 2, 0), price), v, "E", "d", {}, Budget(),
                 WeatherSettings())  # fmt: skip
    assert (d is not None) is bought


class FakeKalshiIndex(FakeKalshiLive):
    """Kalshi knows the NYC index as 'new-york'; 'nyc' is rejected."""

    def __init__(self, value: float, minutes_old: float) -> None:
        super().__init__()
        self.value, self.minutes_old = value, minutes_old
        self.asked: list[str] = []

    async def get_weather_index(self, city: str, last_sec: int = 1200) -> dict[str, Any]:
        self.asked.append(city)
        if city != "new-york":
            raise RuntimeError("400 unknown city")
        t = int((T.timestamp() - self.minutes_old * 60) * 1000)
        return {"city": city, "timeseries": [
            {"t": t - 60_000, "v": self.value - 0.2, "status": "normal"},
            {"t": t, "v": self.value, "status": "normal"},
            {"t": t + 60_000, "status": "incomplete"},
        ]}  # fmt: skip


def _hourly_engine(tmp_path: Path, client: FakeKalshiIndex):  # type: ignore[no-untyped-def]
    eng = _engine(tmp_path, "paper", [], strategy="live")
    eng.client = client  # type: ignore[assignment]
    eng.nws = FakeNwsLive()  # type: ignore[assignment]
    eng.now_fn = lambda: T
    eng.ws = eng.ws.model_copy(update={"hourly": True, "rain": False})
    return eng


def test_hourly_reads_kalshis_own_index(tmp_path: Path) -> None:
    k = FakeKalshiIndex(61.5, minutes_old=3)
    eng = _hourly_engine(tmp_path, k)
    asyncio.run(eng.cycle())
    snap = next(s for s in eng.last_snapshots if s["kind"] == "hourly")
    assert "Kalshi index 61.5°" in snap["evidence"]
    # 61.5 vs the 65 line is 3.5° (index gap is 3°) -> NO at 90c.
    assert eng.ledger.signed_positions()["KXTEMPNYCHS-26OCT2819-T64.99"] < 0
    assert eng._index_ids["KXTEMPNYCHS"] == "new-york"
    # Next cycle goes straight to the ID that worked.
    k.asked.clear()
    asyncio.run(eng.cycle())
    assert k.asked == ["new-york"]


def test_hourly_close_call_is_skipped(tmp_path: Path) -> None:
    eng = _hourly_engine(tmp_path, FakeKalshiIndex(63.0, minutes_old=3))
    asyncio.run(eng.cycle())
    assert not [t for t in eng.ledger.positions if t.startswith("KXTEMP")]  # only 2° away


def test_stale_index_falls_back_to_member_airports(tmp_path: Path) -> None:
    eng = _hourly_engine(tmp_path, FakeKalshiIndex(61.5, minutes_old=40))
    asyncio.run(eng.cycle())
    snap = next(s for s in eng.last_snapshots if s["kind"] == "hourly")
    assert "index unavailable" in snap["evidence"]
