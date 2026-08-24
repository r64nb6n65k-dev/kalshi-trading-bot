"""Command-line interface for the Kalshi Trading Bot.

Run ``kalshi-bot --help`` after installing, or ``python -m kalshi_bot --help``.

Includes CobyStrategy and the Coby dashboard.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from kalshi_bot import __version__
from kalshi_bot.backtesting.data import random_walk_snapshots
from kalshi_bot.backtesting.engine import Backtester
from kalshi_bot.backtesting.report import render as render_report
from kalshi_bot.config import load_settings
from kalshi_bot.core.engine import TradingEngine
from kalshi_bot.dashboard import start_dashboard
from kalshi_bot.data.bnb import BnbPriceFeed
from kalshi_bot.exchange.client import KalshiClient
from kalshi_bot.risk.manager import RiskManager
from kalshi_bot.scalping.engine import BtcOrderBookScalpingEngine
from kalshi_bot.strategies.base import Strategy
from kalshi_bot.strategies.examples.arbitrage import ArbitrageYesNo
from kalshi_bot.strategies.examples.coby_strategy import CobyStrategy
from kalshi_bot.strategies.examples.fair_value import FairValue
from kalshi_bot.strategies.examples.market_maker import MarketMaker
from kalshi_bot.strategies.examples.mean_reversion import MeanReversion
from kalshi_bot.strategies.examples.momentum import Momentum

app = typer.Typer(
    add_completion=False,
    help="Kalshi Trading Bot â open-source framework by Viprasol Tech.",
)
console = Console()


STRATEGIES: dict[str, type[Strategy]] = {
    "market_maker": MarketMaker,
    "momentum": Momentum,
    "mean_reversion": MeanReversion,
    "arbitrage": ArbitrageYesNo,
    "fair_value": FairValue,
    "coby_strategy": CobyStrategy,
}


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"kalshi-trading-bot [bold cyan]{__version__}[/] â by Viprasol Tech")


@app.command()
def strategies() -> None:
    """List registered strategies."""
    table = Table(title="Registered strategies")
    table.add_column("Name", style="cyan")
    table.add_column("Class")
    table.add_column("Summary")

    summaries = {
        "market_maker": "Quote both sides around mid with an inventory cap.",
        "momentum": "Trade in the direction of short-term last-price momentum.",
        "mean_reversion": "Fade moves beyond N standard deviations from the mean.",
        "arbitrage": "Buy YES+NO when their combined ask is below 100c.",
        "fair_value": "Buy YES below your fair probability, Kelly-sized.",
        "coby_strategy": "Coby's 15-minute BNB strategy.",
    }

    for name, cls in STRATEGIES.items():
        table.add_row(name, cls.__name__, summaries.get(name, ""))

    console.print(table)


@app.command()
def backtest(
    strategy: str = typer.Argument(
        "momentum",
        help=f"One of: {', '.join(STRATEGIES)}",
    ),
    ticks: int = typer.Option(
        200,
        help="Number of synthetic snapshots to replay.",
    ),
    balance: int = typer.Option(
        100_000,
        help="Starting balance in cents.",
    ),
    seed: int = typer.Option(
        42,
        help="RNG seed for the synthetic random walk.",
    ),
) -> None:
    """Backtest a strategy on synthetic data and print a metrics report."""
    if strategy not in STRATEGIES:
        choices = ", ".join(STRATEGIES)
        console.print(f"[red]Unknown strategy '{strategy}'. Choose from: {choices}[/]")
        raise typer.Exit(code=1)

    snapshots = random_walk_snapshots(n=ticks, seed=seed)
    bt = Backtester(starting_balance_cents=balance)
    report = bt.run(STRATEGIES[strategy](), snapshots)
    render_report(
        report,
        console=console,
        title=f"Backtest: {strategy} ({ticks} ticks)",
    )


@app.command()
def markets(
    status: str = typer.Option(
        "open",
        help="Filter by market status.",
    ),
    limit: int = typer.Option(
        10,
        help="Maximum markets to list.",
    ),
) -> None:
    """List markets from Kalshi."""

    async def _run() -> None:
        settings = load_settings()

        async with KalshiClient.from_settings(settings) as client:
            rows = await client.get_markets(
                status=status,
                limit=limit,
            )

            table = Table(title=f"Kalshi markets ({settings.environment.value})")
            table.add_column("Ticker", style="cyan")
            table.add_column("Title")
            table.add_column("Yes Bid", justify="right")
            table.add_column("Yes Ask", justify="right")

            for m in rows:
                table.add_row(
                    m.ticker,
                    m.title or "-",
                    str(m.yes_bid),
                    str(m.yes_ask),
                )

            console.print(table)

    asyncio.run(_run())


@app.command()
def balance() -> None:
    """Show portfolio balance."""

    async def _run() -> None:
        settings = load_settings()

        async with KalshiClient.from_settings(settings) as client:
            if not client.authenticated:
                console.print("[red]No credentials configured. Set KALSHI_API_KEY_ID.[/]")
                raise typer.Exit(code=1)

            bal = await client.get_balance()

            console.print(f"Balance: [bold green]${bal.balance / 100:,.2f}[/]")

    asyncio.run(_run())


@app.command()
def run(
    strategy: str = typer.Argument(
        ...,
        help=f"One of: {', '.join(STRATEGIES)}",
    ),
    ticker: str = typer.Argument(
        ...,
        help="Market ticker or series ticker to trade.",
    ),
    ticks: int = typer.Option(
        0,
        help="Number of poll cycles before stopping (0 runs continuously).",
    ),
    live: bool = typer.Option(
        False,
        "--live",
        help="Send real orders (default is dry-run).",
    ),
) -> None:
    """Run a strategy live or in dry-run mode."""
    if strategy not in STRATEGIES:
        choices = ", ".join(STRATEGIES)
        console.print(f"[red]Unknown strategy '{strategy}'. Choose from: {choices}[/]")
        raise typer.Exit(code=1)

    # Keep the web dashboard running in the same Railway process.
    start_dashboard()

    async def _run() -> None:
        settings = load_settings()

        # Dry-run unless --live is explicitly passed.
        dry_run = not live

        async with KalshiClient.from_settings(settings) as client:
            bnb_feed = None
            strategy_params = {}
            if strategy == "coby_strategy":
                bnb_feed = BnbPriceFeed(
                    ws_url=settings.bnb_ws_url,
                    product_id=settings.bnb_product_id,
                )
                strategy_params["max_bnb_age_seconds"] = settings.bnb_max_age_seconds
            engine = TradingEngine(
                client=client,
                strategy=STRATEGIES[strategy](**strategy_params),
                risk=RiskManager.from_settings(settings.risk),
                dry_run=dry_run,
                poll_interval=settings.poll_interval,
                bnb_feed=bnb_feed,
            )

            await engine.run(
                ticker,
                max_ticks=None if ticks == 0 else ticks,
            )

    asyncio.run(_run())


@app.command("btc-scalper")
def btc_scalper(
    events: int = typer.Option(
        0,
        help="Order-book deltas to record before stopping (0 runs continuously).",
    ),
    live: bool = typer.Option(
        False,
        "--live",
        help="Reserved for the validated live build; currently safety-locked.",
    ),
) -> None:
    """Record and queue-simulate the KXBTC15M order book."""

    async def _run() -> None:
        settings = load_settings()
        if live:
            console.print(
                "[red]Live maker trading is safety-locked until dry-run queue validation passes.[/]"
            )
            raise typer.Exit(code=2)
        async with KalshiClient.from_settings(settings) as client:
            from kalshi_bot.exchange.websocket import KalshiWebSocket

            websocket = KalshiWebSocket.from_settings(settings)
            engine = BtcOrderBookScalpingEngine(
                client=client,
                websocket=websocket,
                dry_run=True,
            )
            await engine.run(max_events=None if events == 0 else events)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
