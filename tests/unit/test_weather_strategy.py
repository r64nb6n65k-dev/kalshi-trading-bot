"""Strategy decisions, paper ledger and report scoring."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from kalshi_bot.exchange.models import Action, Side
from kalshi_bot.risk.manager import RiskManager
from kalshi_bot.weather.brackets import group_events
from kalshi_bot.weather.calibration import CalibrationStore, KindCalibration, fit_history
from kalshi_bot.weather.ledger import PaperLedger
from kalshi_bot.weather.report import score_rows
from kalshi_bot.weather.settings import WeatherSettings
from kalshi_bot.weather.stations import Kind
from kalshi_bot.weather.strategy import Budget, WeatherStrategy
from tests.unit.weather_helpers import ladder

UTC = UTC
NOW = datetime(2026, 10, 27, 12, tzinfo=UTC)


def _event(quotes: dict[str, tuple[int, int]]):  # type: ignore[no-untyped-def]
    return group_events(ladder(quotes))[0]


def _mid_probs(ev) -> dict[str, float]:  # type: ignore[no-untyped-def]
    raw = {b.ticker: (b.market.yes_bid + b.market.yes_ask) / 200 for b in ev.brackets}
    s = sum(raw.values())
    return {k: v / s for k, v in raw.items()}


def _strategy(**kw: object) -> WeatherStrategy:
    ws = WeatherSettings(partition_arb=False, **kw)  # type: ignore[arg-type]
    return WeatherStrategy(ws, RiskManager())


def test_buys_yes_when_model_beats_ask() -> None:
    ev = _event({"70": (26, 30)})
    probs = _mid_probs(ev)
    t70 = next(b.ticker for b in ev.brackets if b.bracket.lo == 70)
    probs[t70] = 0.45
    ds = _strategy().decide(ev, probs, {}, Budget(), 50_000, NOW)
    assert len(ds) == 1
    d = ds[0]
    assert d.order.ticker == t70 and d.order.side is Side.YES and d.order.action is Action.BUY
    assert d.order.yes_price == 30
    assert d.order.count == 50  # capped by max_contracts_per_order and $15 market budget
    assert d.edge is not None and d.edge > 0.06


def test_buys_no_on_overpriced_bracket() -> None:
    ev = _event({"64": (20, 24)})
    probs = _mid_probs(ev)
    t64 = next(b.ticker for b in ev.brackets if b.bracket.lo == 64)
    probs[t64] = 0.01
    ds = _strategy().decide(ev, probs, {}, Budget(), 50_000, NOW)
    assert [(d.order.ticker, d.order.side) for d in ds] == [(t64, Side.NO)]
    assert ds[0].order.no_price == 80


def test_no_trade_without_edge_and_cooldown() -> None:
    ev = _event({"70": (26, 30)})
    probs = _mid_probs(ev)
    strat = _strategy()
    assert strat.decide(ev, probs, {}, Budget(), 50_000, NOW) == []
    t70 = next(b.ticker for b in ev.brackets if b.bracket.lo == 70)
    probs[t70] = 0.45
    assert strat.decide(ev, probs, {}, Budget(), 50_000, NOW)
    # Same signal again inside the cooldown -> nothing.
    assert strat.decide(ev, probs, {}, Budget(), 50_000, NOW + timedelta(seconds=30)) == []


def test_day_budget_blocks_entries() -> None:
    ev = _event({"70": (26, 30)})
    probs = _mid_probs(ev)
    t70 = next(b.ticker for b in ev.brackets if b.bracket.lo == 70)
    probs[t70] = 0.45
    b = Budget()
    b.add("OTHER-CITY", "OTHER", "2026-10-28", 10_000)  # day cap already used elsewhere
    assert _strategy().decide(ev, probs, {}, b, 50_000, NOW) == []


def test_exit_when_model_turns() -> None:
    ev = _event({"70": (40, 44)})
    probs = _mid_probs(ev)
    t70 = next(b.ticker for b in ev.brackets if b.bracket.lo == 70)
    probs[t70] = 0.05
    ds = _strategy().decide(ev, probs, {t70: 10}, Budget(), 50_000, NOW)
    assert len(ds) == 1 and ds[0].kind == "exit"
    assert ds[0].order.action is Action.SELL and ds[0].order.count == 10
    assert ds[0].order.yes_price == 40


def test_partition_arbitrage_yes_set() -> None:
    quotes = dict.fromkeys(("lo", "64", "66", "68", "70", "72", "74", "hi"), (5, 9))
    ev = _event(quotes)  # 8 asks x 9c = 72c for a guaranteed 100c
    strat = WeatherStrategy(WeatherSettings(), RiskManager())
    ds = strat.decide(ev, _mid_probs(ev), {}, Budget(), 50_000, NOW)
    assert len(ds) == 8 and all(d.kind == "arb" and d.order.side is Side.YES for d in ds)
    assert len({d.order.count for d in ds}) == 1


# --- ledger ----------------------------------------------------------------
def test_paper_ledger_round_trip(tmp_path: Path) -> None:
    led = PaperLedger.load(tmp_path / "l.json", 10_000)
    led.apply_fill("KXHIGHNY-26OCT28-B70.5", Side.YES, Action.BUY, 10, 30, 2)
    led.apply_fill("KXHIGHNY-26OCT28-B64.5", Side.NO, Action.BUY, 5, 80, 1)
    assert led.signed_positions() == {"KXHIGHNY-26OCT28-B70.5": 10, "KXHIGHNY-26OCT28-B64.5": -5}
    b = led.budget()
    assert b.by_event["KXHIGHNY-26OCT28"] == 700 and b.by_day["2026-10-28"] == 700
    assert led.settle("KXHIGHNY-26OCT28-B70.5", "yes") == 700
    assert led.settle("KXHIGHNY-26OCT28-B64.5", "no") == 100
    led2 = PaperLedger.load(tmp_path / "l.json", 10_000)
    assert led2.realized_cents == 800 - 3 and not led2.positions
    assert led2.bankroll_cents() == 10_797


def test_paper_ledger_partial_sell(tmp_path: Path) -> None:
    led = PaperLedger.load(tmp_path / "l.json", 10_000)
    led.apply_fill("T-26OCT28-X", Side.YES, Action.BUY, 10, 30, 0)
    led.apply_fill("T-26OCT28-X", Side.YES, Action.SELL, 4, 50, 0)
    assert led.realized_cents == 4 * 50 - 120
    assert led.positions["T-26OCT28-X"].count == 6


# --- calibration / report ----------------------------------------------------
def test_fit_history_recovers_bias() -> None:
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(60)]
    actuals = {d: 50.0 + (i % 7) for i, d in enumerate(days)}
    forecasts = {
        d: {"gfs": actuals[d] - 1.5 + ((i % 3) - 1), "ecmwf": actuals[d] + 0.5}
        for i, d in enumerate(days)
    }
    biases, resid = fit_history(forecasts, actuals, min_n=30)
    assert biases["gfs"] == pytest.approx(1.5, abs=0.2)
    assert biases["ecmwf"] == pytest.approx(-0.5, abs=0.01)
    assert len(resid) == 60


def test_calibration_store_round_trip(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path / "c.json")
    store.put("NYC", Kind.HIGH, KindCalibration(bias=0.4, source_bias={"gfs_seamless": 1.1}))
    store.save()
    cal = CalibrationStore(tmp_path / "c.json").get("NYC", Kind.HIGH)
    assert cal.bias == 0.4
    assert cal.source_offset("ens:gfs_seamless", Kind.HIGH) == 1.1
    assert cal.source_offset("nws", Kind.HIGH) == 0.0
    assert cal.source_offset("det:other", Kind.LOW) == -0.8


def test_score_rows_prefers_sharper_correct_model() -> None:
    rows = [
        {
            "bucket": "d1",
            "intraday": False,
            "brackets": [
                {"ticker": "A", "p": 0.8, "yes_bid": 45, "yes_ask": 55, "no_ask": 55},
                {"ticker": "B", "p": 0.2, "yes_bid": 45, "yes_ask": 55, "no_ask": 55},
            ],
        }
    ]
    rep = score_rows(rows, {"A": "yes", "B": "no"}, min_edge=0.06, fee_rate=0.07)
    s = rep.buckets["d1"]
    assert s.n == 2 and s.brier_model < s.brier_market
    assert rep.signals == 2 and rep.signal_wins == 2
    assert rep.signal_pnl_cents > 0
