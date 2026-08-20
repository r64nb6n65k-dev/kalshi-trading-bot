from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kalshi_bot.exchange.models import Action, Market, Position, Side
from kalshi_bot.strategies.base import StrategyContext, UnderlyingTick
from kalshi_bot.strategies.examples.coby_strategy import CobyStrategy


def _close_time(seconds: int = 180) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


def _ctx(prices: list[float], *, target: float = 100.0) -> StrategyContext:
    now = datetime.now(UTC)
    start = now - timedelta(seconds=len(prices) - 1)
    ticks = tuple(
        UnderlyingTick(
            price=price,
            size=1.0,
            timestamp=start + timedelta(seconds=index),
            source="TEST",
        )
        for index, price in enumerate(prices)
    )
    market = Market(
        ticker="KXBTC15M-TEST",
        close_time=_close_time(),
        floor_strike=target,
        yes_bid=91,
        yes_ask=92,
        no_bid=8,
        no_ask=9,
        yes_bid_size=1_000,
        yes_ask_size=1_000,
    )
    return StrategyContext(
        market=market,
        positions={market.ticker: Position(ticker=market.ticker)},
        balance=100_000,
        underlying_ticks=ticks,
    )


def test_requires_three_consecutive_qualified_readings() -> None:
    strategy = CobyStrategy(
        minimum_history_seconds=30,
        required_confirmations=3,
    )
    ctx = _ctx([110.0 + index for index in range(61)])

    assert strategy.on_market_data(ctx) == []
    assert strategy.on_market_data(ctx) == []
    orders = strategy.on_market_data(ctx)

    assert len(orders) == 1
    assert orders[0].action is Action.BUY
    assert orders[0].side is Side.YES


def test_choppy_target_crossings_block_entry() -> None:
    strategy = CobyStrategy(
        minimum_history_seconds=30,
        minimum_distance=5,
        volatility_multiplier=0.1,
        minimum_efficiency=0.0,
        minimum_trend_change=-100,
        required_confirmations=1,
    )
    prices = [90.0 if index % 2 == 0 else 110.0 for index in range(61)]
    prices[-1] = 120.0

    assert strategy.on_market_data(_ctx(prices)) == []


def test_exit_does_not_require_btc_feed() -> None:
    strategy = CobyStrategy()
    ticker = "KXBTC15M-TEST"
    strategy._side_by_ticker[ticker] = Side.YES
    strategy._filled_count_by_ticker[ticker] = 10
    market = Market(
        ticker=ticker,
        close_time=_close_time(),
        yes_bid=79,
        yes_ask=80,
    )
    ctx = StrategyContext(
        market=market,
        positions={ticker: Position(ticker=ticker, position=10)},
        balance=100_000,
    )

    orders = strategy.on_market_data(ctx)

    assert len(orders) == 1
    assert orders[0].action is Action.SELL
    assert orders[0].yes_price == 1
