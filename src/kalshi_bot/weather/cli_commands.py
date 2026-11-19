"""Typer commands for the weather bot, registered onto the main ``kalshi-bot`` app.

kalshi-bot weather-check                 # verify every data source + market parsing
kalshi-bot weather-calibrate             # fit biases/spread from history
kalshi-bot weather-show                  # model vs market table, right now
kalshi-bot weather-run --log-only        # data logger (no orders)
kalshi-bot weather-run                   # paper trading (default)
kalshi-bot weather-run --live            # real orders (also needs KALSHI_DRY_RUN=false)
kalshi-bot weather-report                # model vs market scores + paper P&L
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table

from kalshi_bot.config import load_settings
from kalshi_bot.exchange.client import KalshiClient
from kalshi_bot.risk.manager import RiskManager
from kalshi_bot.weather.brackets import group_events
from kalshi_bot.weather.calibration import CalibrationStore
from kalshi_bot.weather.climate import fetch_cli
from kalshi_bot.weather.dashboard import start_weather_dashboard
from kalshi_bot.weather.datalog import DataLog
from kalshi_bot.weather.engine import WeatherEngine
from kalshi_bot.weather.fees import fee_per_contract
from kalshi_bot.weather.forecasts import OpenMeteoClient
from kalshi_bot.weather.ledger import PaperLedger
from kalshi_bot.weather.nws import NwsClient
from kalshi_bot.weather.observations import summarize
from kalshi_bot.weather.report import calibrate_from_log, calibrate_history, evaluate
from kalshi_bot.weather.stations import Kind, climate_day_window, climate_today, resolve_stations


def _ok(flag: bool) -> str:
    return "[green]OK[/]" if flag else "[red]FAIL[/]"


def register_weather_commands(app: typer.Typer, console: Console) -> None:
    @app.command("weather-check")
    def weather_check() -> None:
        """Check Kalshi series, NWS, Open-Meteo and IEM for every configured station."""

        # Plain one-line-per-check output: readable in Railway's log viewer on a phone.
        def say(msg: str) -> None:
            print(f"[weather-check] {msg}", flush=True)

        def ok(flag: bool) -> str:
            return "OK  " if flag else "FAIL"

        async def _run() -> None:
            say("starting")
            settings = load_settings()
            ws = settings.weather
            stations = resolve_stations(ws.station_codes(), ws.series_overrides)
            now = datetime.now(UTC)
            say(f"stations={','.join(s.code for s in stations)} kinds={ws.kinds}")
            ua_set = "set KALSHI" not in ws.user_agent
            say(f"user agent set: {ua_set}")
            om = OpenMeteoClient(ws.openmeteo_api_key)
            nws = NwsClient(ws.user_agent)
            try:
                async with KalshiClient.from_settings(settings) as client:
                    for st in stations:
                        say(f"===== {st.code} ({st.icao}) =====")
                        for kind_name in ws.kind_names():
                            series = st.series_for(Kind(kind_name))
                            try:
                                mk = await client.get_all_markets(
                                    series_ticker=series, status="open"
                                )
                                evs = group_events(mk)
                                good = [e for e in evs if e.valid]
                                bad = [e for e in evs if not e.valid]
                                rules = next((m.rules_primary for m in mk if m.rules_primary), "")
                                if not rules:
                                    hint = "no rules text"
                                elif st.rules_match(rules):
                                    hint = "rules ok"
                                else:
                                    hint = f"RULES LACK '{st.rules_hint}': {rules[:120]}"
                                say(
                                    f"{ok(bool(good))} kalshi {series}: {len(mk)} markets, "
                                    f"{len(good)} good events, {len(bad)} bad; {hint}"
                                )
                                for e in bad[:2]:
                                    say(f"     bad {e.event_ticker}: {e.problem[:160]}")
                                if good:
                                    labels = ", ".join(b.bracket.label() for b in good[0].brackets)
                                    say(f"     {good[0].event_ticker}: {labels}")
                            except Exception as exc:
                                say(f"FAIL kalshi {series}: {exc!r}"[:300])
                        try:
                            g = await nws.grid_forecast(st)
                            hi = ", ".join(
                                f"{d:%m-%d}:{v:.0f}" for d, v in sorted(g.highs.items())[:3]
                            )
                            lo = ", ".join(
                                f"{d:%m-%d}:{v:.0f}" for d, v in sorted(g.lows.items())[:3]
                            )
                            say(f"{ok(bool(g.highs))} nws forecast: highs {hi} | lows {lo}")
                        except Exception as exc:
                            say(f"FAIL nws forecast: {exc!r}"[:300])
                        try:
                            start, end = climate_day_window(st, climate_today(st, now))
                            rd = await nws.observations(st, start - timedelta(hours=1), now)
                            s = summarize(rd, start, end)
                            latest = (
                                f"{s.latest_temp_f:.1f}F" if s.latest_temp_f is not None else "-"
                            )
                            say(
                                f"{ok(s.n_readings > 0)} nws observations: {s.n_readings} "
                                f"readings today, latest {latest}"
                            )
                        except Exception as exc:
                            say(f"FAIL nws observations: {exc!r}"[:300])
                        errs: list[str] = []
                        sets = await om.forecast_sets(
                            st, ws.ensemble_models, ws.deterministic_models, errs
                        )
                        say(
                            f"{ok(bool(sets))} open-meteo: "
                            + (", ".join(f"{h.source}({h.n_members})" for h in sets) or "nothing")
                        )
                        for err in errs[:4]:
                            say(f"     tried {err[:160]}")
                        try:
                            cli = await fetch_cli(st.icao, [now.year], ws.user_agent)
                            last = max(cli) if cli else None
                            say(
                                f"{ok(last is not None)} climate archive: {len(cli)} days, "
                                f"last {last} {cli[last] if last else ''}"
                            )
                        except Exception as exc:
                            say(f"FAIL climate archive: {exc!r}"[:300])
            except Exception as exc:
                say(f"CRASHED: {exc!r}")
                raise
            finally:
                await om.aclose()
                await nws.aclose()
            say("done")

        asyncio.run(_run())

    @app.command("weather-calibrate")
    def weather_calibrate(
        source: str = typer.Option("history", help="'history' (archives) or 'log' (own logs)."),
        days: int = typer.Option(180, help="Days of history for --source history."),
    ) -> None:
        """Fit per-station forecast bias and error spread."""

        async def _run() -> None:
            ws = load_settings().weather
            stations = resolve_stations(ws.station_codes(), ws.series_overrides)
            kinds = [Kind(k) for k in ws.kind_names()]
            store = CalibrationStore(ws.data_path() / "calibration.json")
            if source == "history":
                outcomes = await calibrate_history(stations, kinds, ws, store, days=days)
            elif source == "log":
                outcomes = await calibrate_from_log(
                    stations, kinds, ws, store, DataLog(ws.data_path())
                )
            else:
                raise typer.BadParameter("source must be 'history' or 'log'")
            t = Table(title=f"Calibration ({source}) -> {store.path}")
            for c in ("Station", "Kind", "N", "Applied", "Source bias", "Bias", "Sigma d0/d1/d2"):
                t.add_column(c)
            for o in outcomes:
                t.add_row(
                    o.station,
                    o.kind,
                    str(o.n_days),
                    _ok(o.applied) + (f" {o.note}" if o.note else ""),
                    ", ".join(f"{k}:{v:+.1f}" for k, v in o.source_bias.items()),
                    f"{o.bias:+.2f}",
                    "/".join(f"{o.sigma.get(b, 0):.1f}" for b in ("d0", "d1", "d2")),
                )
            console.print(t)

        asyncio.run(_run())

    @app.command("weather-show")
    def weather_show(
        station: str = typer.Option("", help="Only this station code, e.g. NYC."),
    ) -> None:
        """Print model probabilities vs. the live book for every open event."""

        async def _run() -> None:
            settings = load_settings()
            ws = settings.weather
            if station:
                ws = ws.model_copy(update={"stations": station})
            async with KalshiClient.from_settings(settings) as client:
                engine = WeatherEngine(
                    client, ws, RiskManager.from_settings(settings.risk), mode="log"
                )
                try:
                    snaps = await engine.cycle()
                finally:
                    await engine.aclose()
            if not snaps:
                console.print("[yellow]No events modeled. Run weather-check.[/]")
            for s in snaps:
                m = s.model
                title = (
                    f"{s.event.event_ticker}  {s.station.name} {s.kind.value}  "
                    f"mean {m.mean:.1f}F sd {m.std:.1f}  lead {m.lead_hours:.0f}h"
                    + (f"  obs [{m.obs_lo:.1f},{m.obs_hi:.1f}]" if m.obs_lo is not None else "")
                    + (f"  anomaly {m.anomaly:+.1f}" if m.anomaly is not None else "")
                )
                t = Table(title=title)
                for c in ("Bracket", "Model", "Bid", "Ask", "YES edge", "NO edge"):
                    t.add_column(c, justify="right")
                for bm in s.event.brackets:
                    p = s.probs[bm.ticker]
                    mk = bm.market
                    ye = (
                        f"{p - mk.yes_ask / 100 - fee_per_contract(mk.yes_ask, ws.fee_rate):+.3f}"
                        if mk.yes_ask
                        else "-"
                    )
                    no_ask = mk.no_ask or (100 - mk.yes_bid if mk.yes_bid else None)
                    ne = (
                        f"{(1 - p) - no_ask / 100 - fee_per_contract(no_ask, ws.fee_rate):+.3f}"
                        if no_ask
                        else "-"
                    )
                    t.add_row(
                        bm.bracket.label(),
                        f"{p:.3f}",
                        str(mk.yes_bid or "-"),
                        str(mk.yes_ask or "-"),
                        ye,
                        ne,
                    )
                console.print(t)
                if m.notes:
                    console.print(f"[dim]{'; '.join(m.notes)}[/]")

        asyncio.run(_run())

    @app.command("weather-run")
    def weather_run(
        log_only: bool = typer.Option(False, "--log-only", help="Collect data; never trade."),
        live: bool = typer.Option(False, "--live", help="Send real orders."),
        cycles: int = typer.Option(0, help="Stop after N cycles (0 = forever)."),
        dashboard: bool = typer.Option(
            True, "--dashboard/--no-dashboard", help="Serve the web dashboard on $PORT."
        ),
    ) -> None:
        """Run the weather engine (paper by default) with its web dashboard."""
        settings = load_settings()
        if live and log_only:
            raise typer.BadParameter("--live and --log-only are mutually exclusive")
        if live and settings.dry_run:
            console.print("[red]Refusing live mode: set KALSHI_DRY_RUN=false as well as --live.[/]")
            raise typer.Exit(code=1)
        mode = "live" if live else ("log" if log_only else "paper")

        async def _run() -> None:
            async with KalshiClient.from_settings(settings) as client:
                engine = WeatherEngine(
                    client, settings.weather, RiskManager.from_settings(settings.risk), mode=mode
                )
                if dashboard:
                    start_weather_dashboard(engine)
                await engine.run(max_cycles=None if cycles == 0 else cycles)

        asyncio.run(_run())

    @app.command("weather-report")
    def weather_report() -> None:
        """Score logged model probabilities vs. market prices on settled events."""

        async def _run() -> None:
            settings = load_settings()
            ws = settings.weather
            data_dir = ws.data_path()
            async with KalshiClient.from_settings(settings) as client:
                rep = await evaluate(client, DataLog(data_dir), data_dir, ws)
            t = Table(title=f"Model vs market ({rep.tickers_scored} settled brackets)")
            for c in (
                "Lead",
                "Snapshots",
                "Brier model",
                "Brier market",
                "LogLoss model",
                "LogLoss market",
            ):
                t.add_column(c, justify="right")
            for b, s in sorted(rep.buckets.items()):
                better = s.brier_model < s.brier_market
                t.add_row(
                    b,
                    str(s.n),
                    f"[{'green' if better else 'red'}]{s.brier_model:.4f}[/]",
                    f"{s.brier_market:.4f}",
                    f"{s.logloss_model:.4f}",
                    f"{s.logloss_market:.4f}",
                )
            console.print(t)
            r = Table(title="Model reliability (forecast prob vs observed frequency)")
            for c in ("Bin", "Observed", "N"):
                r.add_column(c, justify="right")
            for mid, freq, n in rep.reliability:
                r.add_row(f"{mid:.2f}", f"{freq:.3f}", str(n))
            console.print(r)
            console.print(
                f"Hypothetical signals (first edge >= {ws.min_edge:.2f} per bracket, "
                f"1 contract, held): {rep.signals} trades, {rep.signal_wins} winners, "
                f"P&L {rep.signal_pnl_cents / 100:+.2f} USD"
            )
            led = PaperLedger.load(data_dir / "paper_ledger.json", ws.paper_bankroll_cents)
            console.print(
                f"Paper ledger: realized {led.realized_cents / 100:+.2f} USD "
                f"(fees {led.fees_cents / 100:.2f}), open cost "
                f"{led.open_cost_cents() / 100:.2f} USD in {len(led.positions)} markets"
            )

        asyncio.run(_run())
