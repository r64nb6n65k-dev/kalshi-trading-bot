"""Scanner/execution loop for all open Kalshi 15-minute markets."""

from __future__ import annotations

import asyncio
import time

from kalshi_bot.dashboard import record_entry, record_exit, update_open_count
from kalshi_bot.data.multi_asset import MultiAssetPriceFeed
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
        price_feed: MultiAssetPriceFeed,
    ) -> None:
        self.client, self.strategy, self.risk = client, strategy, risk
        self.dry_run, self.poll_interval = dry_run, poll_interval
        self.price_feed = price_feed
        self._paper_positions: dict[str, int] = {}
        self._exit_values: dict[str, int] = {}
        self._exit_counts: dict[str, int] = {}
        self._entry_prices: dict[str, int] = {}
        self._total_pnl_cents = 0

    async def _positions(self) -> dict[str, Position]:
        if not self.dry_run and self.client.authenticated:
            return {p.ticker: p for p in await self.client.get_positions()}
        return {
            ticker: Position(ticker=ticker, position=value)
            for ticker, value in self._paper_positions.items()
            if value
        }

    async def _settle_inactive_positions(self, active_tickers: set[str]) -> None:
        """Record official Kalshi settlement before releasing a tracked position."""
        for ticker in set(self._paper_positions) - active_tickers:
            signed_count = self._paper_positions.get(ticker, 0)
            if signed_count == 0:
                self._paper_positions.pop(ticker, None)
                continue
            try:
                market = await self.client.get_market(ticker)
            except Exception:
                logger.exception("SETTLEMENT LOOKUP FAILED | ticker=%s", ticker)
                continue
            result = (market.result or "").lower()
            if result not in {"yes", "no"}:
                logger.info(
                    "AWAITING SETTLEMENT | ticker=%s | status=%s",
                    ticker,
                    market.status,
                )
                continue

            side = Side.YES if signed_count > 0 else Side.NO
            count = abs(signed_count)
            won = result == side.value
            exit_price = 100 if won else 0
            entry_price = self._entry_prices.get(ticker, self.strategy.entry_price_for(ticker))
            pnl = (exit_price - entry_price) * count
            self._total_pnl_cents += pnl
            record_exit(
                ticker=ticker,
                side=side.value,
                entry_price=entry_price,
                exit_price=exit_price,
                reason="SETTLEMENT_WIN" if won else "SETTLEMENT_LOSS",
                count=count,
                pnl_cents=pnl,
                total_pnl_cents=self._total_pnl_cents,
            )
            logger.warning(
                "SETTLEMENT RECORDED | ticker=%s | side=%s | result=%s | exit=%dc | pnl=$%.2f",
                ticker,
                side.value,
                result,
                exit_price,
                pnl / 100,
            )
            self._paper_positions.pop(ticker, None)
            self._entry_prices.pop(ticker, None)
            self._exit_values.pop(ticker, None)
            self._exit_counts.pop(ticker, None)

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
        entry_price = self._entry_prices.get(request.ticker, 0)
        self.strategy.on_order_result(request, result)
        fill_count = int(result.fill_count or 0)
        fill_price = int(result.outcome_fill_price or 0)
        if fill_count <= 0 or fill_price <= 0:
            return
        if not self.dry_run:
            delta = fill_count if request.side is Side.YES else -fill_count
            if request.action is Action.SELL:
                delta = -delta
            self._paper_positions[request.ticker] = (
                self._paper_positions.get(request.ticker, 0) + delta
            )
        if request.action is Action.BUY:
            self._entry_prices[request.ticker] = fill_price
            record_entry(
                ticker=request.ticker,
                side=request.side.value,
                entry_price=fill_price,
                count=fill_count,
                seconds_left=self.strategy.seconds_to_close(market),
                stop_price=None,
                take_profit=self.strategy.take_profit,
                execution_mode="paper" if self.dry_run else "live",
            )
            return

        self._exit_values[request.ticker] = (
            self._exit_values.get(request.ticker, 0) + fill_price * fill_count
        )
        self._exit_counts[request.ticker] = self._exit_counts.get(request.ticker, 0) + fill_count
        remaining = self.strategy.count_for(request.ticker)
        if remaining > 0:
            update_open_count(request.ticker, remaining)
            return
        exited = self._exit_counts.pop(request.ticker)
        average_exit = round(self._exit_values.pop(request.ticker) / exited)
        pnl = (average_exit - entry_price) * exited
        self._total_pnl_cents += pnl
        record_exit(
            ticker=request.ticker,
            side=request.side.value,
            entry_price=entry_price,
            exit_price=average_exit,
            reason="TAKE_PROFIT_98",
            count=exited,
            pnl_cents=pnl,
            total_pnl_cents=self._total_pnl_cents,
        )
        self._entry_prices.pop(request.ticker, None)
        self._paper_positions.pop(request.ticker, None)

    async def run(self, max_cycles: int | None = None) -> None:
        cycle = 0
        logger.warning(
            "ALL-15M ENGINE STARTED | dry_run=%s | bankroll=$%.2f | contracts=%d",
            self.dry_run,
            self.strategy.bankroll_cents / 100,
            self.strategy.contracts,
        )
        await self.price_feed.start()
        try:
            while max_cycles is None or cycle < max_cycles:
                try:
                    markets = await self.client.get_open_15m_markets()
                    active_tickers = {market.ticker for market in markets}
                    await self._settle_inactive_positions(active_tickers)
                    self.strategy.reconcile(active_tickers | set(self._paper_positions))
                    positions = await self._positions()
                    now = time.time()
                    for market in markets:
                        position = positions.get(market.ticker)
                        current = position.position if position else 0
                        product = self.strategy.product_for(market)
                        ticks = self.price_feed.snapshot(product) if product else ()
                        for order in self.strategy.orders_for(market, current, now, ticks):
                            await self._submit(order, market, current)
                except Exception:
                    logger.exception("ALL-15M SCAN FAILED; retrying")
                cycle += 1
                if max_cycles is None or cycle < max_cycles:
                    await asyncio.sleep(self.poll_interval)
        finally:
            await self.price_feed.stop()
