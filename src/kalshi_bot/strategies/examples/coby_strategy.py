"""Coby's 15-minute Kalshi live strategy with confirmed-fill tracking."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from kalshi_bot.dashboard import record_entry, record_exit, update_open_count
from kalshi_bot.exchange.models import (
    Action,
    Order,
    OrderRequest,
    OrderType,
    Side,
    TimeInForce,
)
from kalshi_bot.strategies.base import Strategy, StrategyContext
from kalshi_bot.telemetry.logging import get_logger

logger = get_logger(__name__)


class CobyStrategy(Strategy):
    name = "coby_strategy"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

        self.entry_min = int(params.get("entry_min", 90))
        self.entry_max = int(params.get("entry_max", 95))
        self.take_profit = int(params.get("take_profit", 98))
        self.stop_price = int(params.get("stop_price", 79))
        self.entry_window_seconds = int(params.get("entry_window_seconds", 300))
        self.max_contracts = int(params.get("max_contracts", 200))
        self.max_notional_cents = int(params.get("max_notional_cents", 20_000))
        self.final_exit_seconds = int(params.get("final_exit_seconds", 60))

        self.trading_timezone = ZoneInfo(
            str(params.get("trading_timezone", "America/Chicago"))
        )
        self.trading_start_hour = int(params.get("trading_start_hour", 6))
        self.trading_end_hour = int(params.get("trading_end_hour", 19))

        # For a triggered stop or final-minute exit, use an aggressively
        # marketable limit. Kalshi's V2 order API is limit-order based; 1c
        # prioritizes getting out. Neither trigger guarantees a fill.
        self.stop_exit_floor = int(params.get("stop_exit_floor", 1))

        self._traded_tickers: set[str] = set()
        self._closed_tickers: set[str] = set()
        self._side_by_ticker: dict[str, Side] = {}
        self._entry_price_by_ticker: dict[str, int] = {}
        self._filled_count_by_ticker: dict[str, int] = {}
        self._pending_action: dict[str, Action] = {}
        self._exit_reason: dict[str, str] = {}

        self._exit_fill_value_by_ticker: dict[str, int] = {}
        self._exit_fill_count_by_ticker: dict[str, int] = {}
        self._total_pnl_cents = 0

    def _seconds_to_close(self, close_time: str | None) -> float | None:
        if not close_time:
            return None
        try:
            close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            if close_dt.tzinfo is None:
                close_dt = close_dt.replace(tzinfo=timezone.utc)
            return (close_dt - datetime.now(timezone.utc)).total_seconds()
        except ValueError:
            return None

    def _within_trading_hours(self) -> bool:
        local_hour = datetime.now(self.trading_timezone).hour
        return self.trading_start_hour <= local_hour < self.trading_end_hour

    def _entry_count(self, limit_price: int) -> int:
        if limit_price <= 0:
            return 0
        return max(
            0,
            min(self.max_contracts, self.max_notional_cents // limit_price),
        )

    def _order(
        self,
        *,
        ticker: str,
        action: Action,
        side: Side,
        price: int,
        count: int,
    ) -> OrderRequest:
        kwargs: dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": OrderType.LIMIT,
            "time_in_force": TimeInForce.IMMEDIATE_OR_CANCEL,
        }
        if side is Side.YES:
            kwargs["yes_price"] = price
        else:
            kwargs["no_price"] = price
        return OrderRequest(**kwargs)

    def _known_position_count(self, ticker: str, ctx_position: int) -> int:
        if ticker in self._filled_count_by_ticker:
            return self._filled_count_by_ticker[ticker]
        return abs(int(ctx_position)) if ctx_position != 0 else 0

    def _recover_side_from_position(self, ticker: str, ctx_position: int) -> Side | None:
        side = self._side_by_ticker.get(ticker)
        if side is not None:
            return side
        if ctx_position > 0:
            side = Side.YES
        elif ctx_position < 0:
            side = Side.NO
        else:
            return None
        self._side_by_ticker[ticker] = side
        self._filled_count_by_ticker[ticker] = abs(int(ctx_position))
        logger.warning(
            "LIVE POSITION RECOVERED | ticker=%s | side=%s | count=%d",
            ticker,
            side.value,
            abs(int(ctx_position)),
        )
        return side

    def on_order_result(
        self,
        request: OrderRequest,
        result: Order | None,
        seconds_left: float | None = None,
    ) -> None:
        ticker = request.ticker
        self._pending_action.pop(ticker, None)

        if result is None:
            return

        fill_count = int(result.fill_count or 0)
        if fill_count <= 0:
            logger.warning(
                "LIVE NO FILL | ticker=%s | action=%s | side=%s",
                ticker,
                request.action.value,
                request.side.value,
            )
            return

        fill_price = result.outcome_fill_price
        if fill_price is None:
            fill_price = (
                request.yes_price
                if request.side is Side.YES
                else request.no_price
            )

        if fill_price is None:
            logger.error(
                "LIVE FILL WITHOUT PRICE | ticker=%s | action=%s",
                ticker,
                request.action.value,
            )
            return

        fill_price = int(fill_price)

        if request.action is Action.BUY:
            self._traded_tickers.add(ticker)
            self._side_by_ticker[ticker] = request.side
            self._entry_price_by_ticker[ticker] = fill_price
            self._filled_count_by_ticker[ticker] = fill_count

            record_entry(
                ticker=ticker,
                side=request.side.value,
                entry_price=fill_price,
                count=fill_count,
                seconds_left=seconds_left,
            )
            logger.warning(
                "LIVE ENTRY FILLED | ticker=%s | side=%s | fill=%dc | count=%d",
                ticker,
                request.side.value,
                fill_price,
                fill_count,
            )
            return

        if request.action is Action.SELL:
            side = self._side_by_ticker.get(ticker, request.side)
            old_count = self._filled_count_by_ticker.get(ticker, fill_count)
            remaining = max(0, old_count - fill_count)

            self._exit_fill_value_by_ticker[ticker] = (
                self._exit_fill_value_by_ticker.get(ticker, 0)
                + fill_price * fill_count
            )
            self._exit_fill_count_by_ticker[ticker] = (
                self._exit_fill_count_by_ticker.get(ticker, 0)
                + fill_count
            )

            if remaining > 0:
                self._filled_count_by_ticker[ticker] = remaining
                update_open_count(ticker, remaining)
                logger.warning(
                    "LIVE EXIT PARTIAL | ticker=%s | side=%s | fill=%dc | "
                    "filled=%d | remaining=%d",
                    ticker,
                    side.value,
                    fill_price,
                    fill_count,
                    remaining,
                )
                return

            self._filled_count_by_ticker.pop(ticker, None)
            self._closed_tickers.add(ticker)

            entry_price = self._entry_price_by_ticker.pop(ticker, None)
            reason = self._exit_reason.pop(ticker, "EXIT")
            total_exit_count = self._exit_fill_count_by_ticker.pop(ticker, 0)
            total_exit_value = self._exit_fill_value_by_ticker.pop(ticker, 0)

            if total_exit_count > 0:
                avg_exit = round(total_exit_value / total_exit_count)
            else:
                avg_exit = fill_price

            if entry_price is not None:
                pnl_cents = (avg_exit - entry_price) * total_exit_count
                self._total_pnl_cents += pnl_cents
                record_exit(
                    ticker=ticker,
                    side=side.value,
                    entry_price=entry_price,
                    exit_price=avg_exit,
                    reason=reason,
                    count=total_exit_count,
                    pnl_cents=pnl_cents,
                    total_pnl_cents=self._total_pnl_cents,
                )
            else:
                logger.warning(
                    "LIVE EXIT FILLED BUT ENTRY PRICE UNKNOWN | ticker=%s | "
                    "side=%s | exit=%dc | count=%d",
                    ticker,
                    side.value,
                    avg_exit,
                    total_exit_count,
                )

            self._side_by_ticker.pop(ticker, None)

            logger.warning(
                "LIVE EXIT FILLED | ticker=%s | side=%s | avg_exit=%dc | "
                "count=%d | reason=%s",
                ticker,
                side.value,
                avg_exit,
                total_exit_count,
                reason,
            )

    def _log_entry_check(
        self,
        *,
        ticker: str,
        seconds_left: float | None,
        yes_bid: int | None,
        yes_ask: int | None,
        no_bid: int | None,
        no_ask: int | None,
        reason: str,
    ) -> None:
        logger.info(
            "ENTRY CHECK | ticker=%s | seconds_left=%s | yes_bid=%s | "
            "yes_ask=%s | no_bid=%s | no_ask=%s | reason=%s",
            ticker,
            "None" if seconds_left is None else f"{seconds_left:.1f}",
            yes_bid,
            yes_ask,
            no_bid,
            no_ask,
            reason,
        )

    def on_market_data(self, ctx: StrategyContext) -> list[OrderRequest]:
        m = ctx.market
        seconds_left = self._seconds_to_close(m.close_time)
        if seconds_left is None:
            return []

        ctx_position = ctx.position_for(m.ticker)

        if m.ticker in self._closed_tickers:
            return []

        count = self._known_position_count(m.ticker, ctx_position)

        if count > 0:
            side = self._recover_side_from_position(m.ticker, ctx_position)
            if side is None:
                logger.error(
                    "LIVE EXIT BLOCKED | ticker=%s | count=%d | reason=UNKNOWN_SIDE",
                    m.ticker,
                    count,
                )
                return []

            if self._pending_action.get(m.ticker) is Action.SELL:
                return []

            bid = m.yes_bid if side is Side.YES else m.no_bid

            if seconds_left <= self.final_exit_seconds:
                self._pending_action[m.ticker] = Action.SELL
                self._exit_reason[m.ticker] = "FINAL_60_SECOND_EXIT"
                logger.warning(
                    "LIVE EXIT SIGNAL | ticker=%s | side=%s | bid=%s | "
                    "limit=%dc | seconds_left=%.1f | "
                    "reason=FINAL_60_SECOND_EXIT | count=%d",
                    m.ticker,
                    side.value,
                    bid,
                    self.stop_exit_floor,
                    seconds_left,
                    count,
                )
                return [
                    self._order(
                        ticker=m.ticker,
                        action=Action.SELL,
                        side=side,
                        price=self.stop_exit_floor,
                        count=count,
                    )
                ]

            if bid is None:
                return []

            if bid >= self.take_profit:
                self._pending_action[m.ticker] = Action.SELL
                self._exit_reason[m.ticker] = "TAKE_PROFIT"
                logger.warning(
                    "LIVE EXIT SIGNAL | ticker=%s | side=%s | bid=%dc | "
                    "limit=%dc | reason=TAKE_PROFIT | count=%d",
                    m.ticker,
                    side.value,
                    bid,
                    self.take_profit,
                    count,
                )
                return [
                    self._order(
                        ticker=m.ticker,
                        action=Action.SELL,
                        side=side,
                        price=self.take_profit,
                        count=count,
                    )
                ]

            if bid <= self.stop_price:
                self._pending_action[m.ticker] = Action.SELL
                self._exit_reason[m.ticker] = "STOP"
                logger.warning(
                    "LIVE EXIT SIGNAL | ticker=%s | side=%s | bid=%dc | "
                    "limit=%dc | reason=STOP | count=%d",
                    m.ticker,
                    side.value,
                    bid,
                    self.stop_exit_floor,
                    count,
                )
                return [
                    self._order(
                        ticker=m.ticker,
                        action=Action.SELL,
                        side=side,
                        price=self.stop_exit_floor,
                        count=count,
                    )
                ]

            return []

        if not self._within_trading_hours():
            self._log_entry_check(
                ticker=m.ticker,
                seconds_left=seconds_left,
                yes_bid=m.yes_bid,
                yes_ask=m.yes_ask,
                no_bid=m.no_bid,
                no_ask=m.no_ask,
                reason="OUTSIDE_TRADING_HOURS_6AM_7PM_CENTRAL",
            )
            return []

        if m.ticker in self._traded_tickers:
            self._log_entry_check(
                ticker=m.ticker,
                seconds_left=seconds_left,
                yes_bid=m.yes_bid,
                yes_ask=m.yes_ask,
                no_bid=m.no_bid,
                no_ask=m.no_ask,
                reason="ALREADY_TRADED",
            )
            return []

        if self._pending_action.get(m.ticker) is Action.BUY:
            self._log_entry_check(
                ticker=m.ticker,
                seconds_left=seconds_left,
                yes_bid=m.yes_bid,
                yes_ask=m.yes_ask,
                no_bid=m.no_bid,
                no_ask=m.no_ask,
                reason="ENTRY_PENDING",
            )
            return []

        if seconds_left <= 60:
            self._log_entry_check(
                ticker=m.ticker,
                seconds_left=seconds_left,
                yes_bid=m.yes_bid,
                yes_ask=m.yes_ask,
                no_bid=m.no_bid,
                no_ask=m.no_ask,
                reason="FINAL_60_SECONDS",
            )
            return []

        if seconds_left > self.entry_window_seconds:
            self._log_entry_check(
                ticker=m.ticker,
                seconds_left=seconds_left,
                yes_bid=m.yes_bid,
                yes_ask=m.yes_ask,
                no_bid=m.no_bid,
                no_ask=m.no_ask,
                reason="TOO_EARLY",
            )
            return []

        side: Side | None = None
        observed_ask: int | None = None

        if m.yes_ask is not None and self.entry_min <= m.yes_ask <= self.entry_max:
            side, observed_ask = Side.YES, m.yes_ask
        elif m.no_ask is not None and self.entry_min <= m.no_ask <= self.entry_max:
            side, observed_ask = Side.NO, m.no_ask

        if side is None or observed_ask is None:
            self._log_entry_check(
                ticker=m.ticker,
                seconds_left=seconds_left,
                yes_bid=m.yes_bid,
                yes_ask=m.yes_ask,
                no_bid=m.no_bid,
                no_ask=m.no_ask,
                reason="ASK_NOT_IN_90_95",
            )
            return []

        limit_price = self.entry_max
        count = self._entry_count(limit_price)
        if count <= 0:
            return []

        self._pending_action[m.ticker] = Action.BUY

        logger.warning(
            "LIVE ENTRY SIGNAL | ticker=%s | side=%s | observed_ask=%dc | "
            "limit=%dc | count=%d | seconds_left=%.1f",
            m.ticker,
            side.value,
            observed_ask,
            limit_price,
            count,
            seconds_left,
        )

        return [
            self._order(
                ticker=m.ticker,
                action=Action.BUY,
                side=side,
                price=limit_price,
                count=count,
            )
        ]
