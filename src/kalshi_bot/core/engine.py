"""Live trading engine with confirmed execution callbacks."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import time

from kalshi_bot.data.bitcoin import BitcoinPriceFeed
from kalshi_bot.exchange.client import KalshiClient
from kalshi_bot.exchange.models import Market, Order, OrderRequest, Position, Side
from kalshi_bot.risk.manager import RiskManager
from kalshi_bot.strategies.base import Strategy, StrategyContext
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)


def tick_id() -> int:
    return time.time_ns()


class TradingEngine:
    def __init__(
        self,
        client: KalshiClient,
        strategy: Strategy,
        risk: RiskManager,
        dry_run: bool = True,
        poll_interval: float = 1.0,
        underlying_feed: BitcoinPriceFeed | None = None,
    ) -> None:
        self.client = client
        self.strategy = strategy
        self.risk = risk
        self.dry_run = dry_run
        self.poll_interval = poll_interval
        self.underlying_feed = underlying_feed
        self._running = False

    async def _snapshot(self, ticker: str) -> StrategyContext:
        if ticker == "KXBTC15M":
            market = None

            while self._running and market is None:
                now = datetime.now(timezone.utc)

                # First use Kalshi's open-market view. If that view lags during a
                # 15-minute rollover, fall back to the full series instead of
                # selecting an expired contract or crashing the Railway process.
                markets = await self.client.get_markets(
                    status="open",
                    series_ticker="KXBTC15M",
                    limit=100,
                )

                def future_candidates(items: list[Market]) -> list[tuple[datetime, Market]]:
                    candidates: list[tuple[datetime, Market]] = []
                    for candidate in items:
                        if not candidate.close_time:
                            continue
                        try:
                            close_dt = datetime.fromisoformat(
                                candidate.close_time.replace("Z", "+00:00")
                            )
                            if close_dt.tzinfo is None:
                                close_dt = close_dt.replace(tzinfo=timezone.utc)
                        except ValueError:
                            continue
                        if close_dt > now:
                            candidates.append((close_dt, candidate))
                    return candidates

                active_markets = future_candidates(markets)

                if not active_markets:
                    all_series_markets = await self.client.get_markets(
                        series_ticker="KXBTC15M",
                        limit=100,
                    )
                    active_markets = future_candidates(all_series_markets)

                if active_markets:
                    market = min(active_markets, key=lambda item: item[0])[1]
                    break

                logger.warning(
                    "No current KXBTC15M market returned yet; retrying in %.1fs",
                    self.poll_interval,
                )
                await asyncio.sleep(self.poll_interval)

            if market is None:
                raise RuntimeError("Trading engine stopped before a KXBTC15M market was found")
        else:
            market = await self.client.get_market(ticker)

        positions: dict[str, Position] = {}
        balance = 0
        if self.client.authenticated:
            for pos in await self.client.get_positions():
                positions[pos.ticker] = pos
            balance = (await self.client.get_balance()).balance

        underlying_ticks = (
            self.underlying_feed.snapshot() if self.underlying_feed is not None else ()
        )
        return StrategyContext(
            market=market,
            positions=positions,
            balance=balance,
            underlying_ticks=underlying_ticks,
        )

    def _notify_order_result(
        self,
        request: OrderRequest,
        result: Order | None,
        seconds_left: float | None,
        is_live: bool = True,
    ) -> None:
        callback = getattr(self.strategy, "on_order_result", None)
        if callable(callback):
            callback(request, result, seconds_left, is_live)

    async def _submit(
        self,
        order: OrderRequest,
        current_position: int,
        market: Market,
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
            if order.side is Side.YES:
                fill_price = market.yes_ask if order.action.value == "buy" else market.yes_bid
            else:
                fill_price = market.no_ask if order.action.value == "buy" else market.no_bid
            if fill_price is None:
                logger.warning("PAPER NO FILL | missing executable quote | %s", order)
                self._notify_order_result(order, None, seconds_left, is_live=False)
                return None
            paper_fill = Order(
                order_id=f"paper-{order.ticker}-{tick_id()}",
                ticker=order.ticker,
                status="executed",
                side=order.side,
                action=order.action,
                count=order.count,
                remaining_count=0,
                fill_count=order.count,
                outcome_fill_price=fill_price,
            )
            logger.warning(
                "PAPER FILL | ticker=%s | action=%s | side=%s | fill=%dc | count=%d",
                order.ticker,
                order.action.value,
                order.side.value,
                fill_price,
                order.count,
            )
            self._notify_order_result(order, paper_fill, seconds_left, is_live=False)
            return paper_fill

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
        if self.underlying_feed is not None:
            await self.underlying_feed.start()
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
                        ctx.market,
                        seconds_left,
                    )

                tick += 1
                if max_ticks is not None and tick >= max_ticks:
                    break
                await asyncio.sleep(self.poll_interval)
        finally:
            if self.underlying_feed is not None:
                await self.underlying_feed.stop()
            self.strategy.on_stop()
            self._running = False
            logger.info("Engine stopped after %d tick(s)", tick)

    def stop(self) -> None:
        self._running = False
