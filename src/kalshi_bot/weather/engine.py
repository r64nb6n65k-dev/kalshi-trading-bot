"""Weather engine: fetch -> model -> decide -> execute -> log, on a loop.

Modes:

* ``log``: data collection only. Runs the model and records model and market
  snapshots. Never places orders.
* ``paper``: everything in ``log``, plus simulated fills in a persistent paper
  ledger that settles against real Kalshi results.
* ``live``: real immediate-or-cancel orders through :class:`KalshiClient`.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from kalshi_bot.exchange.client import KalshiClient
from kalshi_bot.exchange.models import Action, Side
from kalshi_bot.risk.manager import RiskManager
from kalshi_bot.telemetry.logging import get_logger
from kalshi_bot.weather.brackets import WeatherEvent, group_events
from kalshi_bot.weather.calibration import CalibrationStore
from kalshi_bot.weather.datalog import DataLog
from kalshi_bot.weather.fees import fee_cents, fee_per_contract
from kalshi_bot.weather.forecasts import ForecastBundle, OpenMeteoClient
from kalshi_bot.weather.ledger import PaperLedger, budget_from_live, day_of
from kalshi_bot.weather.model import ModelResult, build_distribution
from kalshi_bot.weather.nws import NwsClient
from kalshi_bot.weather.observations import ObsReading, summarize
from kalshi_bot.weather.settings import WeatherSettings
from kalshi_bot.weather.stations import (
    Kind,
    Station,
    climate_day_window,
    climate_today,
    resolve_stations,
)
from kalshi_bot.weather.strategy import Budget, Decision, WeatherStrategy

logger = get_logger(__name__)

MODES = ("log", "paper", "live")


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


@dataclass(slots=True)
class EventSnapshot:
    station: Station
    kind: Kind
    event: WeatherEvent
    model: ModelResult
    probs: dict[str, float]
    decisions: list[Decision]


def snapshot_dict(s: EventSnapshot, fee_rate: float) -> dict[str, Any]:
    """JSON-safe summary of one modeled event (for the dashboard)."""
    m = s.model
    rows = []
    for bm in s.event.brackets:
        mk = bm.market
        p = s.probs[bm.ticker]
        no_ask = mk.no_ask or (100 - mk.yes_bid if mk.yes_bid else None)
        rows.append(
            {
                "ticker": bm.ticker,
                "label": bm.bracket.label(),
                "p": round(p, 4),
                "yes_bid": mk.yes_bid,
                "yes_ask": mk.yes_ask,
                "no_bid": mk.no_bid,
                "no_ask": no_ask,
                "edge_yes": None
                if not mk.yes_ask
                else round(p - mk.yes_ask / 100 - fee_per_contract(mk.yes_ask, fee_rate), 4),
                "edge_no": None
                if not no_ask
                else round((1 - p) - no_ask / 100 - fee_per_contract(no_ask, fee_rate), 4),
            }
        )
    return {
        "event": s.event.event_ticker,
        "station": s.station.code,
        "city": s.station.name,
        "kind": s.kind.value,
        "day": s.event.day.isoformat(),
        "mean": round(m.mean, 1),
        "std": round(m.std, 1),
        "lead_hours": round(m.lead_hours, 1),
        "intraday": m.intraday,
        "obs_lo": None if m.obs_lo is None else round(m.obs_lo, 1),
        "obs_hi": None if m.obs_hi is None else round(m.obs_hi, 1),
        "brackets": rows,
        "decisions": [d.reason for d in s.decisions],
    }


class WeatherEngine:
    def __init__(
        self,
        client: KalshiClient,
        settings: WeatherSettings,
        risk: RiskManager,
        mode: str = "paper",
        now_fn: Callable[[], datetime] | None = None,
        openmeteo: OpenMeteoClient | None = None,
        nws: NwsClient | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "live" and not client.authenticated:
            raise ValueError("live mode needs KALSHI_API_KEY_ID and a private key")
        self.client = client
        self.ws = settings
        self.risk = risk
        self.mode = mode
        self.now_fn = now_fn or (lambda: datetime.now(UTC))
        self.data_dir = settings.data_path()
        self.stations = resolve_stations(settings.station_codes(), settings.series_overrides)
        self.kinds = [Kind(k) for k in settings.kind_names()]
        self.calibration = CalibrationStore(self.data_dir / "calibration.json")
        self.datalog = DataLog(self.data_dir)
        self.ledger = PaperLedger.load(
            self.data_dir / "paper_ledger.json", settings.paper_bankroll_cents
        )
        self.strategy = WeatherStrategy(settings, risk)
        self.openmeteo = openmeteo or OpenMeteoClient(settings.openmeteo_api_key)
        self.nws = nws or NwsClient(settings.user_agent)
        self._bundles: dict[str, tuple[float, ForecastBundle]] = {}
        self._obs: dict[str, tuple[float, list[ObsReading]]] = {}
        self._last_log: dict[str, float] = {}
        self._warned: set[str] = set()
        self._last_settle = float("-inf")  # monotonic clock can start near 0
        self._running = False
        # Read by the dashboard thread; replaced wholesale, never mutated.
        self.started_at = datetime.now(UTC)
        self.cycles = 0
        self.last_cycle_at: datetime | None = None
        self.last_error: str | None = None
        self.last_snapshots: list[dict[str, Any]] = []
        self.live_account: dict[str, Any] | None = None

    async def aclose(self) -> None:
        await self.openmeteo.aclose()
        await self.nws.aclose()

    def _warn_once(self, key: str, msg: str, *args: object) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg, *args)

    # ------------------------------------------------------------------
    # data
    # ------------------------------------------------------------------
    async def forecast(self, st: Station) -> ForecastBundle | None:
        cached = self._bundles.get(st.code)
        if cached and time.monotonic() - cached[0] < self.ws.forecast_refresh:
            return cached[1]
        bundle = ForecastBundle(station=st, fetched_at=self.now_fn())
        bundle.hourly = await self.openmeteo.forecast_sets(
            st, self.ws.ensemble_models, self.ws.deterministic_models, bundle.errors
        )
        if self.ws.use_nws_gridpoint:
            try:
                grid = await self.nws.grid_forecast(st)
                bundle.nws_highs, bundle.nws_lows = grid.highs, grid.lows
            except Exception as exc:
                bundle.errors.append(f"nws grid: {exc}")
        for err in bundle.errors:
            self._warn_once(f"fx:{st.code}:{err[:60]}", "%s forecast issue: %s", st.code, err)
        if not bundle.hourly and not bundle.nws_highs:
            if cached and (self.now_fn() - cached[1].fetched_at) < timedelta(hours=3):
                logger.warning("%s: forecast refresh failed; reusing previous", st.code)
                return cached[1]
            return None
        self._bundles[st.code] = (time.monotonic(), bundle)
        return bundle

    async def observations(self, st: Station, now: datetime) -> list[ObsReading]:
        cached = self._obs.get(st.code)
        if cached and time.monotonic() - cached[0] < self.ws.obs_refresh:
            return cached[1]
        start, _ = climate_day_window(st, climate_today(st, now))
        try:
            readings = await self.nws.observations(st, start - timedelta(hours=1), now)
        except Exception as exc:
            logger.warning("%s observations failed: %s", st.code, exc)
            return cached[1] if cached else []
        self._obs[st.code] = (time.monotonic(), readings)
        return readings

    async def events(self, st: Station, kind: Kind, now: datetime) -> list[WeatherEvent]:
        series = st.series_for(kind)
        markets = await self.client.get_all_markets(series_ticker=series, status="open")
        out: list[WeatherEvent] = []
        for ev in group_events(markets):
            if not ev.valid:
                self._warn_once(
                    f"bad:{ev.event_ticker}", "Skipping %s: %s", ev.event_ticker, ev.problem
                )
                continue
            _, end = climate_day_window(st, ev.day)
            if end <= now or (end - now).total_seconds() / 3600 > self.ws.max_lead_hours:
                continue
            rules = next(
                (b.market.rules_primary for b in ev.brackets if b.market.rules_primary), None
            )
            if rules is None:
                self._warn_once(
                    f"norules:{series}", "%s: rules text unavailable; station not verified", series
                )
            elif st.rules_hint.lower() not in rules.lower():
                self._warn_once(
                    f"rules:{series}",
                    "%s rules do not mention '%s' -- station mismatch? Not trading. Rules: %s",
                    series,
                    st.rules_hint,
                    rules[:240],
                )
                continue
            out.append(ev)
        return out

    # ------------------------------------------------------------------
    # one cycle
    # ------------------------------------------------------------------
    async def _account(self) -> tuple[dict[str, int], Budget, int]:
        if self.mode == "live":
            live = await self.client.get_positions()
            balance = (await self.client.get_balance()).balance
            self.live_account = {
                "balance_cents": balance,
                "positions": [
                    {
                        "ticker": p.ticker,
                        "count": p.position,
                        "cost_cents": abs(p.market_exposure),
                        "realized_cents": p.realized_pnl,
                    }
                    for p in live
                    if p.position
                ],
            }
            return {p.ticker: p.position for p in live}, budget_from_live(live), balance
        return self.ledger.signed_positions(), self.ledger.budget(), self.ledger.bankroll_cents()

    def _can_trade(self, ev: WeatherEvent, now: datetime) -> bool:
        if self.mode == "log":
            return False
        close = _parse_ts(ev.close_time)
        if close is None:
            return True
        return now < close - timedelta(minutes=self.ws.stop_trading_minutes_before_close)

    async def cycle(self) -> list[EventSnapshot]:
        now = self.now_fn()
        positions, budget, bankroll = await self._account()
        snapshots: list[EventSnapshot] = []
        for st in self.stations:
            try:
                bundle = await self.forecast(st)
            except Exception:
                logger.exception("%s: forecast fetch crashed", st.code)
                continue
            if bundle is None:
                self._warn_once(f"nofx:{st.code}:{now:%Y%m%d%H}", "%s: no forecast data", st.code)
                continue
            readings = await self.observations(st, now)
            today = climate_today(st, now)
            for kind in self.kinds:
                try:
                    events = await self.events(st, kind, now)
                except Exception as exc:
                    logger.warning("%s %s: market fetch failed: %s", st.code, kind.value, exc)
                    continue
                for ev in events:
                    obs = None
                    if ev.day == today:
                        start, end = climate_day_window(st, ev.day)
                        obs = summarize(readings, start, end)
                    res = build_distribution(
                        st,
                        kind,
                        ev.day,
                        now,
                        bundle,
                        self.calibration.get(st.code, kind),
                        self.ws,
                        obs,
                    )
                    if res is None:
                        continue
                    probs = {bm.ticker: res.prob(bm.bracket) for bm in ev.brackets}
                    decisions: list[Decision] = []
                    if self._can_trade(ev, now):
                        decisions = self.strategy.decide(
                            ev, probs, positions, budget, bankroll, now, intraday=res.intraday
                        )
                        for d in decisions:
                            await self._execute(d, positions, ev)
                    self._maybe_log(st, kind, ev, res, probs)
                    snapshots.append(EventSnapshot(st, kind, ev, res, probs, decisions))
        if self.mode == "paper":
            await self._settle_paper()
        self.last_snapshots = [snapshot_dict(s, self.ws.fee_rate) for s in snapshots]
        self.last_cycle_at = datetime.now(UTC)
        self.cycles += 1
        return snapshots

    def _maybe_log(
        self,
        st: Station,
        kind: Kind,
        ev: WeatherEvent,
        res: ModelResult,
        probs: dict[str, float],
    ) -> None:
        last = self._last_log.get(ev.event_ticker, float("-inf"))
        if time.monotonic() - last < self.ws.log_interval:
            return
        self._last_log[ev.event_ticker] = time.monotonic()
        self.datalog.write(
            "model",
            event=ev.event_ticker,
            series=ev.series_ticker,
            **res.summary(),
            brackets=[
                {
                    "ticker": bm.ticker,
                    "label": bm.bracket.label(),
                    "lo": bm.bracket.lo,
                    "hi": bm.bracket.hi,
                    "p": round(probs[bm.ticker], 4),
                    "yes_bid": bm.market.yes_bid,
                    "yes_ask": bm.market.yes_ask,
                    "no_bid": bm.market.no_bid,
                    "no_ask": bm.market.no_ask,
                    "volume": bm.market.volume,
                    "oi": bm.market.open_interest,
                }
                for bm in ev.brackets
            ],
        )

    async def _execute(self, d: Decision, positions: dict[str, int], ev: WeatherEvent) -> None:
        o = d.order
        cur = positions.get(o.ticker, 0)
        verdict = self.risk.check(o, cur)
        if not verdict.approved:
            logger.warning("Risk veto %s: %s", o.ticker, verdict.reason)
            return
        signed = o.count * (1 if o.side is Side.YES else -1) * (1 if o.action is Action.BUY else -1)
        if self.mode == "paper":
            fee = fee_cents(d.price, o.count, self.ws.fee_rate)
            self.ledger.apply_fill(o.ticker, o.side, o.action, o.count, d.price, fee)
            positions[o.ticker] = cur + signed
            filled, price = o.count, d.price
        else:
            try:
                resp = await self.client.create_order(o)
            except Exception:
                logger.exception("Live order failed: %s", o)
                self.datalog.write("order_error", ticker=o.ticker, reason=d.reason)
                return
            filled = resp.fill_count
            price = resp.outcome_fill_price or d.price
            if filled:
                positions[o.ticker] = cur + (signed // o.count) * filled
        logger.warning(
            "%s %s %s %s x%d @%dc (filled %d) | %s",
            self.mode.upper(),
            d.kind,
            o.action.value,
            o.side.value,
            o.count,
            d.price,
            filled,
            d.reason,
        )
        self.datalog.write(
            "order",
            mode=self.mode,
            kind=d.kind,
            event=ev.event_ticker,
            ticker=o.ticker,
            action=o.action.value,
            side=o.side.value,
            count=o.count,
            limit=d.price,
            filled=filled,
            fill_price=price,
            p_model=d.p_model,
            p_used=d.p_used,
            edge=d.edge,
            reason=d.reason,
        )

    async def _settle_paper(self) -> None:
        if time.monotonic() - self._last_settle < self.ws.settle_check_interval:
            return
        self._last_settle = time.monotonic()
        today = self.now_fn().astimezone(UTC).date().isoformat()
        for ticker in list(self.ledger.positions):
            if day_of(ticker) >= today:
                continue  # climate day not over yet
            try:
                m = await self.client.get_market(ticker)
            except Exception as exc:
                logger.debug("settle lookup %s failed: %s", ticker, exc)
                continue
            if m.result in ("yes", "no"):
                pnl = self.ledger.settle(ticker, m.result)
                logger.warning("PAPER SETTLED %s -> %s | pnl %+dc", ticker, m.result, pnl)
                self.datalog.write("settle", ticker=ticker, result=m.result, pnl=pnl)

    # ------------------------------------------------------------------
    async def run(self, max_cycles: int | None = None) -> None:
        self._running = True
        logger.warning(
            "Weather engine | mode=%s | stations=%s | kinds=%s",
            self.mode,
            ",".join(s.code for s in self.stations),
            ",".join(k.value for k in self.kinds),
        )
        n = 0
        try:
            while self._running:
                try:
                    snaps = await self.cycle()
                    logger.info("cycle %d: %d events modeled", n + 1, len(snaps))
                except Exception:
                    logger.exception("cycle failed; continuing")
                    self.last_error = (
                        f"{datetime.now(UTC):%H:%M:%S} UTC: {traceback.format_exc(limit=1)}"
                    )
                n += 1
                if max_cycles is not None and n >= max_cycles:
                    break
                await asyncio.sleep(self.ws.poll_interval)
        finally:
            self._running = False
            await self.aclose()

    def stop(self) -> None:
        self._running = False
