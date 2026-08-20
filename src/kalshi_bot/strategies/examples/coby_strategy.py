"""Coby's 15-minute BTC strategy with underlying-market filters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from math import fsum
from typing import Any

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


@dataclass(frozen=True)
class BtcAnalysis:
    price: float
    target: float
    distance: float
    required_distance: float
    vwap: float
    trend_change: float
    efficiency: float
    target_crossings: int
    age_seconds: float
    history_seconds: float
    source: str


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
        self.minimum_distance = float(params.get("minimum_distance", 30.0))
        self.volatility_multiplier = float(params.get("volatility_multiplier", 1.5))
        self.analysis_seconds = int(params.get("analysis_seconds", 180))
        self.minimum_history_seconds = int(params.get("minimum_history_seconds", 60))
        self.max_btc_age_seconds = float(params.get("max_btc_age_seconds", 3.0))
        self.max_target_crossings = int(params.get("max_target_crossings", 1))
        self.minimum_efficiency = float(params.get("minimum_efficiency", 0.25))
        self.minimum_trend_change = float(params.get("minimum_trend_change", 5.0))
        self.required_confirmations = int(params.get("required_confirmations", 3))
        self.max_spread_cents = int(params.get("max_spread_cents", 4))

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
        self._qualified_count: dict[str, int] = {}

        self._exit_fill_value_by_ticker: dict[str, int] = {}
        self._exit_fill_count_by_ticker: dict[str, int] = {}
        self._total_pnl_cents = 0

    def _seconds_to_close(self, close_time: str | None) -> float | None:
        if not close_time:
            return None
        try:
            close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            if close_dt.tzinfo is None:
                close_dt = close_dt.replace(tzinfo=UTC)
            return (close_dt - datetime.now(UTC)).total_seconds()
        except ValueError:
            return None

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
        is_live: bool = True,
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
            fill_price = request.yes_price if request.side is Side.YES else request.no_price

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

            if is_live:
                record_entry(
                    ticker=ticker,
                    side=request.side.value,
                    entry_price=fill_price,
                    count=fill_count,
                    seconds_left=seconds_left,
                )
            logger.warning(
                "%s ENTRY FILLED | ticker=%s | side=%s | fill=%dc | count=%d",
                "LIVE" if is_live else "PAPER",
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
                self._exit_fill_value_by_ticker.get(ticker, 0) + fill_price * fill_count
            )
            self._exit_fill_count_by_ticker[ticker] = (
                self._exit_fill_count_by_ticker.get(ticker, 0) + fill_count
            )

            if remaining > 0:
                self._filled_count_by_ticker[ticker] = remaining
                if is_live:
                    update_open_count(ticker, remaining)
                logger.warning(
                    "LIVE EXIT PARTIAL | ticker=%s | side=%s | fill=%dc | filled=%d | remaining=%d",
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
                if is_live:
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
                "%s EXIT FILLED | ticker=%s | side=%s | avg_exit=%dc | "
                "count=%d | reason=%s | pnl_cents=%s",
                "LIVE" if is_live else "PAPER",
                ticker,
                side.value,
                avg_exit,
                total_exit_count,
                reason,
                (avg_exit - entry_price) * total_exit_count
                if entry_price is not None
                else "UNKNOWN",
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

    def _btc_analysis(self, ctx: StrategyContext) -> BtcAnalysis | None:
        target = ctx.market.floor_strike
        if target is None or not ctx.underlying_ticks:
            return None

        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=self.analysis_seconds)
        ticks = [tick for tick in ctx.underlying_ticks if tick.timestamp >= cutoff]
        if not ticks:
            return None

        latest = ticks[-1]
        age = (now - latest.timestamp).total_seconds()
        history_seconds = (latest.timestamp - ticks[0].timestamp).total_seconds()

        # Use one closing price per second for direction/choppiness so bursts of
        # trades do not dominate those statistics.
        second_prices: dict[int, float] = {}
        for tick in ticks:
            second_prices[int(tick.timestamp.timestamp())] = tick.price
        prices = list(second_prices.values())
        if len(prices) < 2:
            return None

        recent_30 = prices[-30:]
        recent_range = max(recent_30) - min(recent_30)
        required_distance = max(
            self.minimum_distance,
            self.volatility_multiplier * recent_range,
        )

        path = fsum(abs(b - a) for a, b in pairwise(prices))
        efficiency = abs(prices[-1] - prices[0]) / path if path > 0 else 0.0

        window = min(15, max(1, len(prices) // 2))
        recent_mean = fsum(prices[-window:]) / window
        prior = prices[-2 * window : -window]
        prior_mean = fsum(prior) / len(prior) if prior else prices[0]
        trend_change = recent_mean - prior_mean

        target_crossings = 0
        previous_sign = prices[0] >= target
        for price in prices[1:]:
            current_sign = price >= target
            if current_sign != previous_sign:
                target_crossings += 1
            previous_sign = current_sign

        weighted = [(tick.price, tick.size) for tick in ticks if tick.size > 0]
        total_size = fsum(size for _, size in weighted)
        vwap = (
            fsum(price * size for price, size in weighted) / total_size
            if total_size > 0
            else fsum(prices) / len(prices)
        )

        return BtcAnalysis(
            price=latest.price,
            target=float(target),
            distance=abs(latest.price - float(target)),
            required_distance=required_distance,
            vwap=vwap,
            trend_change=trend_change,
            efficiency=efficiency,
            target_crossings=target_crossings,
            age_seconds=age,
            history_seconds=history_seconds,
            source=latest.source,
        )

    def _btc_rejection(self, analysis: BtcAnalysis, side: Side) -> str | None:
        direction = 1 if side is Side.YES else -1
        signed_target_distance = (analysis.price - analysis.target) * direction
        signed_vwap_distance = (analysis.price - analysis.vwap) * direction
        signed_trend = analysis.trend_change * direction

        if analysis.age_seconds > self.max_btc_age_seconds:
            return "BTC_DATA_STALE"
        if analysis.history_seconds < self.minimum_history_seconds:
            return "BTC_HISTORY_WARMING_UP"
        if signed_target_distance <= 0:
            return "BTC_WRONG_SIDE_OF_TARGET"
        if analysis.distance < analysis.required_distance:
            return "BTC_DISTANCE_TOO_SMALL"
        if analysis.target_crossings > self.max_target_crossings:
            return "BTC_CHOPPY_TARGET_CROSSINGS"
        if analysis.efficiency < self.minimum_efficiency:
            return "BTC_CHOPPY_LOW_EFFICIENCY"
        if signed_trend < self.minimum_trend_change:
            return "BTC_TREND_NOT_CONFIRMED"
        if signed_vwap_distance <= 0:
            return "BTC_WRONG_SIDE_OF_VWAP"
        return None

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
            self._qualified_count[m.ticker] = 0
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

        selected_bid = m.yes_bid if side is Side.YES else m.no_bid
        if selected_bid is None or observed_ask - selected_bid > self.max_spread_cents:
            self._qualified_count[m.ticker] = 0
            self._log_entry_check(
                ticker=m.ticker,
                seconds_left=seconds_left,
                yes_bid=m.yes_bid,
                yes_ask=m.yes_ask,
                no_bid=m.no_bid,
                no_ask=m.no_ask,
                reason="SPREAD_TOO_WIDE",
            )
            return []

        limit_price = self.entry_max
        count = self._entry_count(limit_price)
        if count <= 0:
            return []

        available_at_ask = m.yes_ask_size if side is Side.YES else m.yes_bid_size
        if available_at_ask is None or available_at_ask < count:
            self._qualified_count[m.ticker] = 0
            self._log_entry_check(
                ticker=m.ticker,
                seconds_left=seconds_left,
                yes_bid=m.yes_bid,
                yes_ask=m.yes_ask,
                no_bid=m.no_bid,
                no_ask=m.no_ask,
                reason="INSUFFICIENT_ENTRY_LIQUIDITY",
            )
            return []

        analysis = self._btc_analysis(ctx)
        if analysis is None:
            self._qualified_count[m.ticker] = 0
            self._log_entry_check(
                ticker=m.ticker,
                seconds_left=seconds_left,
                yes_bid=m.yes_bid,
                yes_ask=m.yes_ask,
                no_bid=m.no_bid,
                no_ask=m.no_ask,
                reason="BTC_DATA_OR_TARGET_MISSING",
            )
            return []

        rejection = self._btc_rejection(analysis, side)
        if rejection is not None:
            self._qualified_count[m.ticker] = 0
            logger.info(
                "BTC FILTER | ticker=%s | side=%s | price=%.2f | target=%.2f | "
                "distance=%.2f | required=%.2f | vwap=%.2f | trend=%.2f | "
                "efficiency=%.3f | crossings=%d | age=%.2f | source=%s | reason=%s",
                m.ticker,
                side.value,
                analysis.price,
                analysis.target,
                analysis.distance,
                analysis.required_distance,
                analysis.vwap,
                analysis.trend_change,
                analysis.efficiency,
                analysis.target_crossings,
                analysis.age_seconds,
                analysis.source,
                rejection,
            )
            return []

        confirmations = self._qualified_count.get(m.ticker, 0) + 1
        self._qualified_count[m.ticker] = confirmations
        if confirmations < self.required_confirmations:
            logger.info(
                "BTC FILTER QUALIFIED | ticker=%s | confirmation=%d/%d | "
                "price=%.2f | target=%.2f | distance=%.2f | required=%.2f",
                m.ticker,
                confirmations,
                self.required_confirmations,
                analysis.price,
                analysis.target,
                analysis.distance,
                analysis.required_distance,
            )
            return []

        self._pending_action[m.ticker] = Action.BUY

        logger.warning(
            "ENTRY SIGNAL | ticker=%s | side=%s | observed_ask=%dc | "
            "limit=%dc | count=%d | seconds_left=%.1f | btc=%.2f | "
            "target=%.2f | distance=%.2f | required=%.2f | source=%s",
            m.ticker,
            side.value,
            observed_ask,
            limit_price,
            count,
            seconds_left,
            analysis.price,
            analysis.target,
            analysis.distance,
            analysis.required_distance,
            analysis.source,
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
