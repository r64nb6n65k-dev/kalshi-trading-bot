from __future__ import annotations

from typing import Any, cast

from kalshi_bot.core.engine import TradingEngine
from kalshi_bot.exchange.client import KalshiClient
from kalshi_bot.exchange.models import Action, Market, Side
from kalshi_bot.risk.manager import RiskManager
from kalshi_bot.strategies.examples.coby_strategy import CobyStrategy


async def test_paper_orders_fill_and_update_strategy_state() -> None:
    strategy = CobyStrategy(max_contracts=2, max_notional_cents=1_000)
    engine = TradingEngine(
        client=cast(KalshiClient, cast(Any, None)),
        strategy=strategy,
        risk=RiskManager(),
        dry_run=True,
    )
    market = Market(
        ticker="KXBTC15M-TEST",
        yes_bid=91,
        yes_ask=92,
        no_bid=8,
        no_ask=9,
    )
    buy = strategy._order(
        ticker=market.ticker,
        action=Action.BUY,
        side=Side.YES,
        price=95,
        count=2,
    )

    entry = await engine._submit(buy, 0, market, 180.0)

    assert entry is not None
    assert entry.outcome_fill_price == 92
    assert strategy._filled_count_by_ticker[market.ticker] == 2

    market.yes_bid = 79
    sell = strategy._order(
        ticker=market.ticker,
        action=Action.SELL,
        side=Side.YES,
        price=1,
        count=2,
    )

    exit_fill = await engine._submit(sell, 2, market, 120.0)

    assert exit_fill is not None
    assert exit_fill.outcome_fill_price == 79
    assert market.ticker in strategy._closed_tickers
