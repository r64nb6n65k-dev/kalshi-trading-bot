"""Scanner/execution loop for all open Kalshi 15-minute markets."""

from __future__ import annotations

import asyncio
import time

from kalshi_bot.exchange.client import KalshiClient
from kalshi_bot.exchange.models import Action, Market, Order, OrderRequest, Position, Side
from kalshi_bot.risk.manager import RiskManager
from kalshi_bot.strategies.examples.all_15m_momentum import All15mMomentumStrategy
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)


class All15mEngine:
    def __init__(
        self,
        client: KalshiClient,
        strategy: All15mMomentumStrategy,
        risk: RiskManager,
        *,
        dry_run: bool,
        poll_interval: float,
    ) -> None:
        self.client, self.strategy, self.risk = client, strategy, risk
        self.dry_run, self.poll_interval = dry_run, poll_interval
        self._paper_positions: dict[str, int] = {}

    async def _positions(self) -> dict[str, Position]:
        if not self.dry_run and self.client.authenticated:
            return {p.ticker: p for p in await self.client.get_positions()}
        return {
            ticker: Position(ticker=ticker, position=value)
            for ticker, value in self._paper_positions.items()
            if value
        }

    async def _submit(self, request: OrderRequest, market: Market, position: int) -> None:
        decision = self.risk.check(request, position)
        if not decision.approved:
            logger.warning("ORDER VETO | ticker=%s | %s", request.ticker, decision.reason)
            self.strategy.on_order_result(request, None)
            return
        if self.dry_run:
            price = (
                (market.yes_ask if request.side is Side.YES else market.no_ask)
                if request.action is Action.BUY
                else (market.yes_bid if request.side is Side.YES else market.no_bid)
            )
            if price is None:
                self.strategy.on_order_result(request, None)
                return
            result = Order(
                order_id=f"paper-{request.ticker}-{time.time_ns()}",
                ticker=request.ticker,
                status="executed",
                side=request.side,
                action=request.action,
                count=request.count,
                remaining_count=0,
                fill_count=request.count,
                outcome_fill_price=price,
            )
            delta = request.count if request.side is Side.YES else -request.count
            if request.action is Action.SELL:
                delta = -delta
            self._paper_positions[request.ticker] = (
                self._paper_positions.get(request.ticker, 0) + delta
            )
            logger.warning(
                "PAPER FILL | ticker=%s | action=%s | side=%s | fill=%dc | count=%d",
                request.ticker,
                request.action.value,
                request.side.value,
                price,
                request.count,
            )
        else:
            try:
                result = await self.client.create_order(request)
            except Exception:
                logger.exception("LIVE ORDER FAILED | ticker=%s", request.ticker)
                self.strategy.on_order_result(request, None)
                return
        self.strategy.on_order_result(request, result)

    async def run(self, max_cycles: int | None = None) -> None:
        cycle = 0
        logger.warning(
            "ALL-15M ENGINE STARTED | dry_run=%s | bankroll=$%.2f | contracts=%d",
            self.dry_run,
            self.strategy.bankroll_cents / 100,
            self.strategy.contracts,
        )
        while max_cycles is None or cycle < max_cycles:
            try:
                markets = await self.client.get_open_15m_markets()
                active_tickers = {market.ticker for market in markets}
                self.strategy.reconcile(active_tickers)
                for ticker in set(self._paper_positions) - active_tickers:
                    self._paper_positions.pop(ticker, None)
                positions = await self._positions()
                now = time.time()
                for market in markets:
                    position = positions.get(market.ticker)
                    current = position.position if position else 0
                    for order in self.strategy.orders_for(market, current, now):
                        await self._submit(order, market, current)
            except Exception:
                logger.exception("ALL-15M SCAN FAILED; retrying")
            cycle += 1
            if max_cycles is None or cycle < max_cycles:
                await asyncio.sleep(self.poll_interval)
