"""Sports and price markets under the "already decided" rules."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from kalshi_bot.exchange.models import Market, Side
from kalshi_bot.strategies.base import UnderlyingTick
from kalshi_bot.weather.dashboard import market_type
from kalshi_bot.weather.prices import asset_for, price_verdict, vol_per_sqrt_second
from kalshi_bot.weather.sports import (
    LEAGUES,
    Game,
    Team,
    decided,
    league_for,
    match_game,
    parse_scoreboard,
    verdicts_for,
)
from tests.unit.test_weather_engine import _engine

NOW = datetime(2026, 11, 23, 3, 0, tzinfo=UTC)  # 10 PM Eastern, Nov 22


def _mk(ticker: str, label: str, bid: int, ask: int, **kw: Any) -> Market:
    return Market.model_validate(
        {"ticker": ticker, "event_ticker": ticker.rsplit("-", 1)[0], "yes_sub_title": label,
         "yes_bid": bid, "yes_ask": ask, "no_bid": 100 - ask, "no_ask": 100 - bid,
         "yes_ask_size": 200, "status": "active", **kw}
    )  # fmt: skip


def espn_payload(home: tuple[str, str, int], away: tuple[str, str, int], period: int,
                 clock: float, state: str = "in") -> dict[str, Any]:  # fmt: skip
    def comp(t: tuple[str, str, int], side: str) -> dict[str, Any]:
        team = {"abbreviation": t[0], "displayName": t[1], "location": t[1].split()[0]}
        return {"homeAway": side, "score": str(t[2]), "team": team}

    return {"events": [{
        "date": "2026-11-23T00:30Z",
        "competitions": [{
            "competitors": [comp(home, "home"), comp(away, "away")],
            "status": {"clock": clock, "period": period,
                       "type": {"state": state, "shortDetail": f"Q{period} {int(clock)}s"}},
        }],
    }]}  # fmt: skip


def _game(sport_scores: tuple[int, int], period: int, clock: float, state: str = "in") -> Game:
    bos = Team("BOS", ["boston celtics", "boston"], sport_scores[0], False)
    gsw = Team("GS", ["golden state warriors", "golden state"], sport_scores[1], True)
    return Game(state, period, clock, "detail", date(2026, 11, 22), [bos, gsw])


# --- sports ----------------------------------------------------------------
def test_league_detection() -> None:
    assert league_for("KXNBAGAME") is LEAGUES["NBA"]
    assert league_for("KXNCAAFGAME") is LEAGUES["NCAAF"]
    assert league_for("KXNBASPREAD") is None
    assert market_type("KXNBAGAME-26NOV22BOSGSW-BOS") == "Sports"
    assert market_type("KXBTC15M-26NOV211430-30") == "Crypto"
    assert market_type("KXGOLD15M-26NOV211430-30") == "Commodities"
    assert market_type("KXHIGHNY-26NOV21-B60.5") == "Daily high"


def test_parse_and_match_with_alias() -> None:
    games = parse_scoreboard(espn_payload(("GS", "Golden State Warriors", 90),
                                          ("BOS", "Boston Celtics", 105), 4, 100.0))  # fmt: skip
    assert len(games) == 1 and games[0].day == date(2026, 11, 22)
    ms = [_mk("KXNBAGAME-26NOV22BOSGSW-BOS", "Boston", 90, 92),
          _mk("KXNBAGAME-26NOV22BOSGSW-GSW", "Golden State", 8, 10)]  # fmt: skip
    found = match_game("KXNBAGAME-26NOV22BOSGSW", ms, games)
    assert found is not None
    _, mapping = found
    assert mapping["KXNBAGAME-26NOV22BOSGSW-BOS"].abbr == "BOS"  # type: ignore[union-attr]
    assert mapping["KXNBAGAME-26NOV22BOSGSW-GSW"].abbr == "GS"  # type: ignore[union-attr]
    # Wrong date (a week later) doesn't match.
    assert match_game("KXNBAGAME-26NOV29BOSGSW", ms, games) is None


@pytest.mark.parametrize(
    ("league", "scores", "period", "clock", "state", "expect"),
    [
        ("NBA", (105, 90), 4, 100.0, "in", True),  # (15-3)^2=144 > 100s
        ("NBA", (100, 90), 4, 100.0, "in", False),  # (10-3)^2=49 < 100s
        ("NBA", (100, 90), 3, 100.0, "in", False),  # a whole quarter left
        ("NFL", (31, 14), 4, 400.0, "in", True),  # 17 pts, 6:40 left
        ("NFL", (24, 14), 4, 400.0, "in", False),  # 10 pts, 6:40 left
        ("NFL", (24, 14), 4, 90.0, "in", True),  # 10 pts, 1:30 left
        ("NHL", (4, 2), 3, 60.0, "in", True),  # 2 goals, 1:00 left
        ("NHL", (3, 2), 3, 30.0, "in", False),  # 1 goal
        ("MLB", (7, 2), 9, 0.0, "in", True),  # 5 runs in the 9th
        ("MLB", (7, 2), 7, 0.0, "in", False),
        ("EPL", (2, 0), 2, 86 * 60.0, "in", True),  # 2 goals, 86'
        ("EPL", (1, 0), 2, 88 * 60.0, "in", False),
        ("NBA", (99, 98), 4, 0.0, "post", True),  # final, any margin
    ],
)
def test_decided_rules(league: str, scores: tuple[int, int], period: int, clock: float,
                       state: str, expect: bool) -> None:  # fmt: skip
    out = decided(LEAGUES[league], _game(scores, period, clock, state))
    assert (out is not None) is expect
    if out is not None:
        assert out[0] is not None and out[0].abbr == "BOS"


def test_verdicts_yes_on_winner_no_on_loser_and_tie() -> None:
    g = _game((2, 0), 2, 87 * 60.0)
    a, b = g.teams
    mapping = {"E-BOS": a, "E-GSW": b, "E-TIE": None}
    v = verdicts_for(mapping, decided(LEAGUES["EPL"], g))  # type: ignore[arg-type]
    assert v["E-BOS"].side is Side.YES
    assert v["E-GSW"].side is Side.NO and v["E-TIE"].side is Side.NO


# --- prices ----------------------------------------------------------------
def _ticks(prices: list[float], end: datetime, step: float = 10.0) -> list[UnderlyingTick]:
    n = len(prices)
    return [
        UnderlyingTick(p, 1.0, end - timedelta(seconds=step * (n - 1 - i)), "TEST")
        for i, p in enumerate(prices)
    ]


def test_asset_mapping_and_volatility() -> None:
    assert asset_for("KXBTC15M") == "BTC" and asset_for("KXETHD") == "ETH"
    assert asset_for("KXHIGHNY") is None
    flat = _ticks([100_000.0 + (5 if i % 2 else -5) for i in range(60)], NOW)
    vol = vol_per_sqrt_second(flat, NOW)
    assert vol is not None and 2e-5 <= vol < 1e-4
    assert vol_per_sqrt_second(flat[:10], NOW) is None  # not enough history


@pytest.mark.parametrize(
    ("strike_type", "floor", "cap", "price", "side"),
    [
        ("greater", 99_000.0, None, 100_000.0, Side.YES),  # $1,000 above, 2 min left
        ("greater", 101_000.0, None, 100_000.0, Side.NO),
        ("greater", 99_950.0, None, 100_000.0, None),  # $50: too close
        ("less", None, 101_000.0, 100_000.0, Side.YES),
        ("between", 99_000.0, 101_000.0, 100_000.0, Side.YES),
        ("between", 101_000.0, 102_000.0, 100_000.0, Side.NO),
    ],
)
def test_price_verdicts(strike_type: str, floor: float | None, cap: float | None,
                        price: float, side: Side | None) -> None:  # fmt: skip
    m = _mk("KXBTC15M-26NOV230315-X", "BTC", 50, 52, strike_type=strike_type,
            floor_strike=floor, cap_strike=cap)  # fmt: skip
    v = price_verdict(m, price, 5e-5, 120.0, 3.0)
    assert (None if v is None else v.side) is side


# --- engine fast loop ----------------------------------------------------
class FakeKalshiAll:
    authenticated = False

    def __init__(self) -> None:
        self.sports = [
            _mk("KXNBAGAME-26NOV22BOSGSW-BOS", "Boston", 88, 90),
            _mk("KXNBAGAME-26NOV22BOSGSW-GSW", "Golden State", 10, 12),
        ]
        close = (NOW + timedelta(minutes=2)).isoformat().replace("+00:00", "Z")
        self.closing = [
            _mk("KXBTC15M-26NOV222200-00", "BTC", 92, 94, strike_type="greater",
                floor_strike=99_000.0, close_time=close, series_ticker="KXBTC15M"),
            _mk("KXHIGHNY-26NOV22-B50.5", "50-51", 50, 52, close_time=close),  # not a price mkt
        ]  # fmt: skip

    async def get_series_list(self, category: str) -> list[dict[str, Any]]:
        assert category == "Sports"
        return [{"ticker": "KXNBAGAME"}, {"ticker": "KXNBASPREAD"}]

    async def get_all_markets(self, **params: Any) -> list[Market]:
        if params.get("series_ticker") == "KXNBAGAME":
            return self.sports
        if "max_close_ts" in params:
            return self.closing
        return []

    async def get_market(self, ticker: str) -> Market:
        return next(m for m in self.sports + self.closing if m.ticker == ticker)


class FakeEspn:
    async def games(self, league: Any) -> list[Game]:
        return parse_scoreboard(espn_payload(("GS", "Golden State Warriors", 90),
                                             ("BOS", "Boston Celtics", 105), 4, 100.0))  # fmt: skip

    async def aclose(self) -> None:
        return None


class FakeFeed:
    def snapshot(self, product: str) -> tuple[UnderlyingTick, ...]:
        assert product == "BTC-USD"
        return tuple(_ticks([100_000.0 + (20 if i % 2 else -20) for i in range(60)], NOW))

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def test_fast_cycle_bets_sports_and_crypto(tmp_path: Path) -> None:
    eng = _engine(tmp_path, "paper", [], strategy="live")
    eng.client = FakeKalshiAll()  # type: ignore[assignment]
    eng.espn = FakeEspn()
    eng.price_feed = FakeFeed()
    eng.now_fn = lambda: NOW
    asyncio.run(eng.fast_cycle())
    pos = eng.ledger.signed_positions()
    assert pos["KXNBAGAME-26NOV22BOSGSW-BOS"] > 0  # YES on Boston at 90c
    assert pos["KXNBAGAME-26NOV22BOSGSW-GSW"] < 0  # NO on Golden State at 90c
    assert pos["KXBTC15M-26NOV222200-00"] > 0  # BTC $1,000 over the line, 2 min left
    assert "KXHIGHNY-26NOV22-B50.5" not in pos
    kinds = {s["kind"] for s in eng.last_snapshots}
    assert kinds == {"sports", "crypto"}
    # Second pass: no repeat bets.
    asyncio.run(eng.fast_cycle())
    assert eng.ledger.signed_positions() == pos


def test_fast_cycle_respects_toggles(tmp_path: Path) -> None:
    eng = _engine(tmp_path, "paper", [], strategy="live")
    eng.client = FakeKalshiAll()  # type: ignore[assignment]
    eng.espn = FakeEspn()
    eng.price_feed = FakeFeed()
    eng.now_fn = lambda: NOW
    eng.ws = eng.ws.model_copy(update={"sports": False, "prices": False})
    asyncio.run(eng.fast_cycle())
    assert not eng.ledger.positions


def test_unlabelled_contract_type_is_inferred_from_strikes() -> None:
    m = _mk("KXBTC15M-26NOV230315-15", "BTC", 50, 52, strike_type="custom",
            floor_strike=99_000.0)  # fmt: skip
    v = price_verdict(m, 100_000.0, 5e-5, 120.0, 3.0)
    assert v is not None and v.side is Side.YES


def test_price_status_explains_skips(tmp_path: Path) -> None:
    eng = _engine(tmp_path, "paper", [], strategy="live")
    k = FakeKalshiAll()
    close = k.closing[0].close_time
    # Decided, but Kalshi already asks 98c: too expensive.
    k.closing[0] = _mk(
        "KXBTC15M-26NOV222200-00", "BTC", 97, 98, strike_type="greater",
        floor_strike=99_000.0, close_time=close, series_ticker="KXBTC15M",
    )  # fmt: skip
    eng.client = k  # type: ignore[assignment]
    eng.espn = FakeEspn()
    eng.price_feed = FakeFeed()
    eng.now_fn = lambda: NOW
    eng.ws = eng.ws.model_copy(update={"sports": False})
    asyncio.run(eng.fast_cycle())
    st = eng.price_status
    assert st["price_markets"] == 1 and st["decided"] == 1 and st["bets"] == 0
    assert any("too expensive" in r for r in st["skipped"])
    assert st["feeds"]["BTC"] is not None
