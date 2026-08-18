"""Live trading engine with confirmed execution callbacks."""

from __future__ import annotations

import asyncio

from kalshi_bot.exchange.client import KalshiClient
from kalshi_bot.exchange.models import Order, OrderRequest, Position
from kalshi_bot.risk.manager import RiskManager
from kalshi_bot.strategies.base import Strategy, StrategyContext
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)


class TradingEngine:
    def __init__(
        self,
        client: KalshiClient,
        strategy: Strategy,
        risk: RiskManager,
        dry_run: bool = True,
        poll_interval: float = 2.0,
    ) -> None:
        self.client = client
        self.strategy = strategy
        self.risk = risk
        self.dry_run = dry_run
        self.poll_interval = poll_interval
        self._running = False

    async def _snapshot(self, ticker: str) -> StrategyContext:
        if ticker == "KXBTC15M":
            markets = await self.client.get_markets(
                status="open",
                series_ticker="KXBTC15M",
                limit=100,
            )
            if not markets:
                raise RuntimeError("No open KXBTC15M market found")
            market = min(markets, key=lambda m: m.close_time or "")
        else:
            market = await self.client.get_market(ticker)

        positions: dict[str, Position] = {}
        balance = 0
        if self.client.authenticated:
            for pos in await self.client.get_positions():
                positions[pos.ticker] = pos
            balance = (await self.client.get_balance()).balance

        return StrategyContext(market=market, positions=positions, balance=balance)

    def _notify_order_result(
        self,
        request: OrderRequest,
        result: Order | None,
        seconds_left: float | None,
    ) -> None:
        callback = getattr(self.strategy, "on_order_result", None)
        if callable(callback):
            callback(request, result, seconds_left)

    async def _submit(
        self,
        order: OrderRequest,
        current_position: int,
        seconds_left: float | None = None,
    ) -> Order | None:
        decision = self.risk.check(order, current_position)
        if not decision.approved:
            logger.warning(
                "Order vetoed by risk manager: %s | %s",
                decision.reason,
                order,
            )
            self._notify_order_result(order, None, seconds_left)
            return None

        if self.dry_run:
            logger.info("[DRY-RUN] would submit: %s", order.to_payload())
            return None

        try:
            order_resp = await self.client.create_order(order)
        except Exception:
            # Clear the strategy's pending-order flag so one API error does not
            # permanently freeze entry or exit attempts.
            logger.exception("Live order submission failed | %s", order)
            self._notify_order_result(order, None, seconds_left)
            return None

        logger.info(
            "Submitted order %s for %s | fill_count=%d | remaining_count=%s",
            order_resp.order_id,
            order.ticker,
            order_resp.fill_count,
            order_resp.remaining_count,
        )
        self._notify_order_result(order, order_resp, seconds_left)
        return order_resp

    async def run(self, ticker: str, max_ticks: int | None = None) -> None:
        self._running = True
        self.strategy.on_start()
        logger.info(
            "Engine started | strategy=%s | ticker=%s | dry_run=%s",
            self.strategy.name,
            ticker,
            self.dry_run,
        )
        tick = 0
        try:
            while self._running:
                ctx = await self._snapshot(ticker)

                seconds_left = None
                helper = getattr(self.strategy, "_seconds_to_close", None)
                if callable(helper):
                    seconds_left = helper(ctx.market.close_time)

                for order in self.strategy.on_market_data(ctx):
                    await self._submit(
                        order,
                        ctx.position_for(order.ticker),
                        seconds_left,
                    )

                tick += 1
                if max_ticks is not None and tick >= max_ticks:
                    break
                await asyncio.sleep(self.poll_interval)
        finally:
            self.strategy.on_stop()
            self._running = False
            logger.info("Engine stopped after %d tick(s)", tick)

    def stop(self) -> None:
        self._running = False
