"""Coby's original late-favorite strategy for Kalshi gold 15-minute markets.

The strategy enters the favored contract at 90-95 cents during the final five
minutes, takes profit at 98 cents, triggers a hard stop at 79 cents, and exits
any remaining position during the final 60 seconds. Live order results are
tracked from confirmed fills, including partial exits.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from kalshi_bot.dashboard import (
    get_open_position,
    record_entry,
    record_exit,
    record_model_snapshot,
    update_open_count,
)
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
    """Trade a strongly favored gold contract late in each 15-minute market."""

    name = "coby_strategy"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

        self.entry_window_seconds = int(params.get("entry_window_seconds", 300))
        self.final_exit_seconds = int(params.get("final_exit_seconds", 60))
        self.minimum_entry_price = int(params.get("minimum_entry_price", 90))
        self.maximum_entry_price = int(params.get("maximum_entry_price", 95))
        self.take_profit = int(params.get("take_profit", 98))
        self.hard_stop = int(params.get("hard_stop", 79))
        self.required_confirmations = int(params.get("required_confirmations", 2))
        self.max_contracts = int(params.get("max_contracts", 100))
        self.max_notional_cents = int(params.get("max_notional_cents", 20_000))
        self.live_ioc_slippage_cents = int(
            params.get("live_ioc_slippage_cents", 2)
        )
        self.stop_exit_floor = int(params.get("stop_exit_floor", 1))

        self.use_trading_hours = bool(params.get("use_trading_hours", False))
        self.trading_timezone = ZoneInfo(
            str(params.get("trading_timezone", "America/Chicago"))
        )
        self.trading_start_hour = int(params.get("trading_start_hour", 6))
        self.trading_end_hour = int(params.get("trading_end_hour", 19))

        self._traded_tickers: set[str] = set()
        self._closed_tickers: set[str] = set()
        self._side_by_ticker: dict[str, Side] = {}
        self._entry_price_by_ticker: dict[str, int] = {}
        self._filled_count_by_ticker: dict[str, int] = {}
        self._pending_action: dict[str, Action] = {}
        self._exit_reason: dict[str, str] = {}
        self._confirmation_side_by_ticker: dict[str, Side] = {}
        self._confirmation_count_by_ticker: dict[str, int] = {}
        self._exit_fill_value_by_ticker: dict[str, int] = {}
        self._exit_fill_count_by_ticker: dict[str, int] = {}
        self._total_pnl_cents = 0

    @staticmethod
    def _seconds_to_close(close_time: str | None) -> float | None:
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
        if not self.use_trading_hours:
            return True
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
        exchange_index: int | None = None,
        emergency_exit: bool = False,
    ) -> OrderRequest:
        if emergency_exit:
            execution_price = max(1, int(price))
        else:
            slippage = max(0, self.live_ioc_slippage_cents)
            if action is Action.BUY:
                execution_price = min(99, int(price) + slippage)
            else:
                execution_price = max(1, int(price) - slippage)

        kwargs: dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": OrderType.LIMIT,
            "time_in_force": TimeInForce.IMMEDIATE_OR_CANCEL,
            "exchange_index": exchange_index,
        }
        if side is Side.YES:
            kwargs["yes_price"] = execution_price
        else:
            kwargs["no_price"] = execution_price
        return OrderRequest(**kwargs)

    def _snapshot(
        self,
        ctx: StrategyContext,
        seconds_left: float | None,
        decision: str,
        reason: str,
    ) -> None:
        market = ctx.market
        latest = ctx.underlying_ticks[-1] if ctx.underlying_ticks else None
        target = float(market.floor_strike) if market.floor_strike is not None else None
        separation = latest.price - target if latest is not None and target is not None else None
        record_model_snapshot(
            ticker=market.ticker,
            seconds_left=seconds_left,
            gold_price=latest.price if latest is not None else None,
            target_price=target,
            separation=separation,
            yes_bid=market.yes_bid,
            yes_ask=market.yes_ask,
            no_bid=market.no_bid,
            no_ask=market.no_ask,
            decision=decision,
            reason=reason,
        )

    def _known_position_count(self, ticker: str, ctx_position: int) -> int:
        if ticker in self._filled_count_by_ticker:
            return self._filled_count_by_ticker[ticker]
        return abs(int(ctx_position)) if ctx_position else 0

    def _recover_side_from_position(
        self,
        ticker: str,
        ctx_position: int,
    ) -> Side | None:
        side = self._side_by_ticker.get(ticker)
        if side is not None:
            return side
        if ctx_position > 0:
            side = Side.YES
        elif ctx_position < 0:
            side = Side.NO
        else:
            return None

        count = abs(int(ctx_position))
        self._side_by_ticker[ticker] = side
        self._filled_count_by_ticker[ticker] = count
        self._traded_tickers.add(ticker)

        saved = get_open_position(ticker)
        if saved is not None and saved.get("entry_price") is not None:
            self._entry_price_by_ticker[ticker] = int(saved["entry_price"])

        logger.warning(
            "LIVE POSITION RECOVERED | ticker=%s | side=%s | count=%d",
            ticker,
            side.value,
            count,
        )
        return side

    def _reset_confirmation(self, ticker: str) -> None:
        self._confirmation_side_by_ticker.pop(ticker, None)
        self._confirmation_count_by_ticker[ticker] = 0

    def _manage_position(
        self,
        ctx: StrategyContext,
        seconds_left: float,
        count: int,
        side: Side,
    ) -> list[OrderRequest]:
        market = ctx.market
        ticker = market.ticker

        if self._pending_action.get(ticker) is Action.SELL:
            return []

        bid = market.yes_bid if side is Side.YES else market.no_bid

        if seconds_left <= self.final_exit_seconds:
            self._pending_action[ticker] = Action.SELL
            self._exit_reason[ticker] = "FINAL_60_SECOND_EXIT"
            self._snapshot(ctx, seconds_left, "SELL", "FINAL_60_SECOND_EXIT")
            return [
                self._order(
                    ticker=ticker,
                    action=Action.SELL,
                    side=side,
                    price=self.stop_exit_floor,
                    count=count,
                    exchange_index=market.exchange_index,
                    emergency_exit=True,
                )
            ]

        if bid is None or not 1 <= int(bid) <= 99:
            self._snapshot(ctx, seconds_left, "HOLD", "NO_EXECUTABLE_BID")
            return []

        if int(bid) >= self.take_profit:
            self._pending_action[ticker] = Action.SELL
            self._exit_reason[ticker] = "TAKE_PROFIT_98C"
            self._snapshot(ctx, seconds_left, "SELL", "TAKE_PROFIT_98C")
            return [
                self._order(
                    ticker=ticker,
                    action=Action.SELL,
                    side=side,
                    price=self.take_profit,
                    count=count,
                    exchange_index=market.exchange_index,
                )
            ]

        if int(bid) <= self.hard_stop:
            self._pending_action[ticker] = Action.SELL
            self._exit_reason[ticker] = "HARD_STOP_79C"
            self._snapshot(ctx, seconds_left, "SELL", "HARD_STOP_79C")
            return [
                self._order(
                    ticker=ticker,
                    action=Action.SELL,
                    side=side,
                    price=self.stop_exit_floor,
                    count=count,
                    exchange_index=market.exchange_index,
                    emergency_exit=True,
                )
            ]

        self._snapshot(ctx, seconds_left, "HOLD", "WAITING_FOR_98C_OR_79C")
        return []

    def on_market_data(self, ctx: StrategyContext) -> list[OrderRequest]:
        market = ctx.market
        ticker = market.ticker
        seconds_left = self._seconds_to_close(market.close_time)

        if seconds_left is None:
            self._snapshot(ctx, seconds_left, "WAIT", "NO_CLOSE_TIME")
            return []

        ctx_position = ctx.position_for(ticker)
        count = self._known_position_count(ticker, ctx_position)

        if count > 0:
            side = self._recover_side_from_position(ticker, ctx_position)
            if side is None:
                logger.error(
                    "LIVE EXIT BLOCKED | ticker=%s | count=%d | reason=UNKNOWN_SIDE",
                    ticker,
                    count,
                )
                return []
            return self._manage_position(ctx, seconds_left, count, side)

        if ticker in self._closed_tickers or ticker in self._traded_tickers:
            self._snapshot(ctx, seconds_left, "CLOSED", "ONE_TRADE_PER_MARKET")
            return []

        if self._pending_action.get(ticker) is Action.BUY:
            return []

        if not self._within_trading_hours():
            self._reset_confirmation(ticker)
            self._snapshot(ctx, seconds_left, "WAIT", "OUTSIDE_TRADING_HOURS")
            return []

        if seconds_left > self.entry_window_seconds:
            self._reset_confirmation(ticker)
            self._snapshot(ctx, seconds_left, "WAIT", "WAITING_FOR_FINAL_5_MINUTES")
            return []

        if seconds_left <= self.final_exit_seconds:
            self._reset_confirmation(ticker)
            self._snapshot(ctx, seconds_left, "SKIP", "FINAL_60_SECONDS_NO_ENTRY")
            return []

        choices: list[tuple[int, Side]] = []
        if (
            market.yes_ask is not None
            and self.minimum_entry_price <= int(market.yes_ask) <= self.maximum_entry_price
        ):
            choices.append((int(market.yes_ask), Side.YES))
        if (
            market.no_ask is not None
            and self.minimum_entry_price <= int(market.no_ask) <= self.maximum_entry_price
        ):
            choices.append((int(market.no_ask), Side.NO))

        if not choices:
            self._reset_confirmation(ticker)
            self._snapshot(ctx, seconds_left, "WAIT", "NO_FAVORITE_BETWEEN_90C_AND_95C")
            return []

        observed_ask, side = max(choices, key=lambda item: item[0])
        previous_side = self._confirmation_side_by_ticker.get(ticker)
        confirmations = self._confirmation_count_by_ticker.get(ticker, 0)
        confirmations = confirmations + 1 if previous_side is side else 1
        self._confirmation_side_by_ticker[ticker] = side
        self._confirmation_count_by_ticker[ticker] = confirmations

        if confirmations < self.required_confirmations:
            self._snapshot(
                ctx,
                seconds_left,
                "WAIT",
                f"CONFIRMING_{side.value.upper()}_{confirmations}_OF_{self.required_confirmations}",
            )
            return []

        count = self._entry_count(observed_ask)
        if count <= 0:
            self._snapshot(ctx, seconds_left, "SKIP", "POSITION_SIZE_ZERO")
            return []

        self._pending_action[ticker] = Action.BUY
        self._snapshot(ctx, seconds_left, "BUY_" + side.value.upper(), "ORIGINAL_LATE_FAVORITE_ENTRY")
        logger.warning(
            "GOLD LATE FAVORITE ENTRY | ticker=%s | side=%s | ask=%dc | "
            "confirmations=%d | seconds_left=%.1f | count=%d",
            ticker,
            side.value,
            observed_ask,
            confirmations,
            seconds_left,
            count,
        )
        return [
            self._order(
                ticker=ticker,
                action=Action.BUY,
                side=side,
                price=observed_ask,
                count=count,
                exchange_index=market.exchange_index,
            )
        ]

    def on_order_result(
        self,
        request: OrderRequest,
        result: Order | None,
        seconds_left: float | None = None,
        is_live: bool | None = None,
    ) -> None:
        ticker = request.ticker
        self._pending_action.pop(ticker, None)

        if result is None:
            logger.warning(
                "ORDER RESULT MISSING | ticker=%s | action=%s | side=%s",
                ticker,
                request.action.value,
                request.side.value,
            )
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
            self._reset_confirmation(ticker)

            record_entry(
                ticker=ticker,
                side=request.side.value,
                entry_price=fill_price,
                count=fill_count,
                seconds_left=seconds_left,
                stop_price=self.hard_stop,
                take_profit=self.take_profit,
                execution_mode="live" if is_live else "paper",
            )
            logger.warning(
                "LIVE ENTRY FILLED | ticker=%s | side=%s | fill=%dc | count=%d",
                ticker,
                request.side.value,
                fill_price,
                fill_count,
            )
            return

        side = self._side_by_ticker.get(ticker, request.side)
        old_count = self._filled_count_by_ticker.get(ticker, fill_count)
        remaining = max(0, old_count - fill_count)

        self._exit_fill_value_by_ticker[ticker] = (
            self._exit_fill_value_by_ticker.get(ticker, 0)
            + fill_price * fill_count
        )
        self._exit_fill_count_by_ticker[ticker] = (
            self._exit_fill_count_by_ticker.get(ticker, 0) + fill_count
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
        avg_exit = (
            round(total_exit_value / total_exit_count)
            if total_exit_count > 0
            else fill_price
        )

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
