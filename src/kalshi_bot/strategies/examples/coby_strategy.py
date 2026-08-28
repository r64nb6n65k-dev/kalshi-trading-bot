"""Coby's 15-minute Kalshi live strategy with confirmed-fill tracking."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import math
import statistics
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
    name = "coby_strategy"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

        self.min_confidence = float(params.get("min_confidence", 0.72))
        self.min_edge_cents = float(params.get("min_edge_cents", 7.0))

        self.min_history_seconds = float(
            params.get(
                "minimum_history_seconds",
                params.get("min_history_seconds", 60),
            )
        )

        self.required_confirmations = int(
            params.get("required_confirmations", 3)
        )

        self.minimum_distance = float(
            params.get("minimum_distance", 3.00)
        )

        self.hard_minimum_separation = float(
            params.get("hard_minimum_separation", 3.00)
        )

        self.normal_minimum_separation = float(
            params.get("normal_minimum_separation", 3.00)
        )

        self.high_volume_minimum_separation = float(
            params.get("high_volume_minimum_separation", 3.00)
        )

        self.low_volume_ratio = float(
            params.get("low_volume_ratio", 0.75)
        )

        self.high_volume_ratio = float(
            params.get("high_volume_ratio", 1.50)
        )

        self.volume_baseline_seconds = float(
            params.get("volume_baseline_seconds", 300.0)
        )

        self.volatility_multiplier = float(
            params.get("volatility_multiplier", 0.35)
        )

        self.maximum_target_crossings = int(
            params.get("maximum_target_crossings", 2)
        )

        self.crossing_ignore_seconds = float(
            params.get("crossing_ignore_seconds", 180.0)
        )

        self.crossing_skip_enable_seconds = float(
            params.get("crossing_skip_enable_seconds", 180.0)
        )

        self.minimum_efficiency = float(
            params.get("minimum_efficiency", 0.12)
        )

        self.minimum_trend_change = float(
            params.get("minimum_trend_change", -0.10)
        )

        self.max_gold_age_seconds = float(
            params.get("max_gold_age_seconds", 5)
        )

        self.take_profit = int(
            params.get("take_profit", 98)
        )

        self.dynamic_stop_gap = int(
            params.get("dynamic_stop_gap", 13)
        )

        self.legacy_stop_cap = int(
            params.get("legacy_stop_cap", 79)
        )

        self.max_contracts = int(
            params.get("max_contracts", 200)
        )

        self.max_notional_cents = int(
            params.get("max_notional_cents", 20_000)
        )

        self.final_exit_seconds = int(
            params.get("final_exit_seconds", 60)
        )

        # Observe the first 5 minutes.
        # Entry becomes possible from 10:00 remaining until the
        # final 60-second no-entry window.
        self.entry_window_seconds = int(
            params.get("entry_window_seconds", 600)
        )

        self.min_entry_price = int(
            params.get("min_entry_price", 70)
        )

        # ============================================================
        # ADAPTIVE CHAOS FILTER
        # ============================================================
        #
        # High movement alone is not considered chaos.
        #
        # For a market to trigger EXTREME_CHAOS_TEMP_BLOCK it must:
        #
        # 1. Travel an unusually large distance over the rolling window.
        # 2. Have very poor directional efficiency.
        # 3. Spend a large percentage of its travel reversing against
        #    its eventual net direction.
        #
        # These thresholds were tightened so ordinary gold chop does not
        # spend most of the entry window classified as "extreme chaos."
        #
        # Once genuine extreme chaos IS detected, recovery is intentionally
        # slower to prevent a few clean seconds from immediately reopening
        # the market.

        self.chaos_window_seconds = float(
            params.get("chaos_window_seconds", 60.0)
        )

        self.chaos_history_size = int(
            params.get("chaos_history_size", 20)
        )

        self.chaos_min_baseline_markets = int(
            params.get("chaos_min_baseline_markets", 4)
        )

        # UPDATED: was 2.25
        self.chaos_travel_multiplier = float(
            params.get("chaos_travel_multiplier", 2.75)
        )

        # UPDATED: was 0.30
        self.chaos_efficiency_ceiling = float(
            params.get("chaos_efficiency_ceiling", 0.22)
        )

        # UPDATED: was 0.45
        self.chaos_reversal_ratio = float(
            params.get("chaos_reversal_ratio", 0.55)
        )

        # UPDATED: was 8.00
        self.chaos_min_travel = float(
            params.get("chaos_min_travel", 12.00)
        )

        # A gold contract-price stop must persist and be confirmed by
        # adverse movement in XAU/USD.
        #
        # One thin Kalshi bid is not enough to dump a trade.
        self.stop_confirmations = int(
            params.get("stop_confirmations", 3)
        )

        self.stop_adverse_move = float(
            params.get("stop_adverse_move", 1.00)
        )

        # Chaos is observed from market open, but it cannot block entry
        # until the normal entry window itself has opened.
        self.chaos_skip_enable_seconds_left = int(
            params.get(
                "chaos_skip_enable_seconds_left",
                self.entry_window_seconds,
            )
        )

        # UPDATED: was 3.
        #
        # Requiring 15 consecutive clean evaluations makes recovery
        # substantially more deliberate after genuine extreme chaos.
        #
        # At roughly 1 evaluation per second this is approximately
        # a 15-second clean recovery period.
        self.chaos_recovery_confirmations = int(
            params.get("chaos_recovery_confirmations", 15)
        )

        self.use_trading_hours = bool(
            params.get("use_trading_hours", False)
        )

        self.trading_timezone = ZoneInfo(
            str(
                params.get(
                    "trading_timezone",
                    "America/Chicago",
                )
            )
        )

        self.trading_start_hour = int(
            params.get("trading_start_hour", 6)
        )

        self.trading_end_hour = int(
            params.get("trading_end_hour", 19)
        )

        self.stop_exit_floor = int(
            params.get("stop_exit_floor", 1)
        )

        self._traded_tickers: set[str] = set()
        self._closed_tickers: set[str] = set()

        self._chop_disqualified_tickers: set[str] = set()
        self._chaos_disqualified_tickers: set[str] = set()

        self._crossing_count_by_ticker: dict[str, int] = {}
        self._last_target_side_by_ticker: dict[str, int] = {}

        # Rolling completed-market memory for adaptive chaos detection.
        self._chaos_history: deque[float] = deque(
            maxlen=max(1, self.chaos_history_size)
        )

        self._peak_travel_by_ticker: dict[str, float] = {}
        self._remembered_chaos_tickers: set[str] = set()
        self._chaos_clean_count_by_ticker: dict[str, int] = {}

        self._side_by_ticker: dict[str, Side] = {}
        self._entry_price_by_ticker: dict[str, int] = {}
        self._filled_count_by_ticker: dict[str, int] = {}

        self._pending_action: dict[str, Action] = {}
        self._exit_reason: dict[str, str] = {}

        self._exit_fill_value_by_ticker: dict[str, int] = {}
        self._exit_fill_count_by_ticker: dict[str, int] = {}

        self._total_pnl_cents = 0

        self._confirmation_side_by_ticker: dict[str, Side] = {}
        self._confirmation_count_by_ticker: dict[str, int] = {}

        self._stop_confirmation_count_by_ticker: dict[str, int] = {}

    def _seconds_to_close(
        self,
        close_time: str | None,
    ) -> float | None:
        if not close_time:
            return None

        try:
            close_dt = datetime.fromisoformat(
                close_time.replace("Z", "+00:00")
            )

            if close_dt.tzinfo is None:
                close_dt = close_dt.replace(tzinfo=timezone.utc)

            return (
                close_dt - datetime.now(timezone.utc)
            ).total_seconds()

        except ValueError:
            return None

    def _within_trading_hours(self) -> bool:
        if not self.use_trading_hours:
            return True

        local_hour = datetime.now(
            self.trading_timezone
        ).hour

        return (
            self.trading_start_hour
            <= local_hour
            < self.trading_end_hour
        )

    def _entry_count(
        self,
        limit_price: int,
    ) -> int:
        if limit_price <= 0:
            return 0

        return max(
            0,
            min(
                self.max_contracts,
                self.max_notional_cents // limit_price,
            ),
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

    def _known_position_count(
        self,
        ticker: str,
        ctx_position: int,
    ) -> int:
        if ticker in self._filled_count_by_ticker:
            return self._filled_count_by_ticker[ticker]

        return (
            abs(int(ctx_position))
            if ctx_position != 0
            else 0
        )

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

        self._side_by_ticker[ticker] = side
        self._filled_count_by_ticker[ticker] = abs(
            int(ctx_position)
        )

        saved = get_open_position(ticker)

        if (
            saved is not None
            and saved.get("entry_price") is not None
        ):
            self._entry_price_by_ticker[ticker] = int(
                saved["entry_price"]
            )

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
        is_live: bool | None = None,
    ) -> None:
        ticker = request.ticker

        self._pending_action.pop(
            ticker,
            None,
        )

        if result is None:
            return

        fill_count = int(
            result.fill_count or 0
        )

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
                stop_price=max(
                    1,
                    min(
                        self.legacy_stop_cap,
                        fill_price - self.dynamic_stop_gap,
                    ),
                ),
                take_profit=self.take_profit,
                execution_mode=(
                    "live"
                    if is_live
                    else "paper"
                ),
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
            side = self._side_by_ticker.get(
                ticker,
                request.side,
            )

            old_count = self._filled_count_by_ticker.get(
                ticker,
                fill_count,
            )

            remaining = max(
                0,
                old_count - fill_count,
            )

            self._exit_fill_value_by_ticker[ticker] = (
                self._exit_fill_value_by_ticker.get(
                    ticker,
                    0,
                )
                + fill_price * fill_count
            )

            self._exit_fill_count_by_ticker[ticker] = (
                self._exit_fill_count_by_ticker.get(
                    ticker,
                    0,
                )
                + fill_count
            )

            if remaining > 0:
                self._filled_count_by_ticker[ticker] = remaining

                update_open_count(
                    ticker,
                    remaining,
                )

                logger.warning(
                    "LIVE EXIT PARTIAL | ticker=%s | side=%s | fill=%dc | filled=%d | remaining=%d",
                    ticker,
                    side.value,
                    fill_price,
                    fill_count,
                    remaining,
                )

                return

            self._filled_count_by_ticker.pop(
                ticker,
                None,
            )

            self._closed_tickers.add(
                ticker
            )

            entry_price = self._entry_price_by_ticker.pop(
                ticker,
                None,
            )

            reason = self._exit_reason.pop(
                ticker,
                "EXIT",
            )

            total_exit_count = self._exit_fill_count_by_ticker.pop(
                ticker,
                0,
            )

            total_exit_value = self._exit_fill_value_by_ticker.pop(
                ticker,
                0,
            )

            avg_exit = (
                round(
                    total_exit_value
                    / total_exit_count
                )
                if total_exit_count > 0
                else fill_price
            )

            if entry_price is not None:
                pnl_cents = (
                    avg_exit - entry_price
                ) * total_exit_count

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
                    "LIVE EXIT FILLED BUT ENTRY PRICE UNKNOWN | ticker=%s | side=%s | exit=%dc | count=%d",
                    ticker,
                    side.value,
                    avg_exit,
                    total_exit_count,
                )

            self._side_by_ticker.pop(
                ticker,
                None,
            )

            logger.warning(
                "LIVE EXIT FILLED | ticker=%s | side=%s | avg_exit=%dc | count=%d | reason=%s",
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
            "ENTRY CHECK | ticker=%s | seconds_left=%s | "
            "yes_bid=%s | yes_ask=%s | no_bid=%s | no_ask=%s | "
            "reason=%s",
            ticker,
            (
                "None"
                if seconds_left is None
                else f"{seconds_left:.1f}"
            ),
            yes_bid,
            yes_ask,
            no_bid,
            no_ask,
            reason,
        )

    def on_market_data(
        self,
        ctx: StrategyContext,
    ) -> list[OrderRequest]:
        m = ctx.market

        seconds_left = self._seconds_to_close(
            m.close_time
        )

        if seconds_left is None:
            return []

        ctx_position = ctx.position_for(
            m.ticker
        )

        if m.ticker in self._closed_tickers:
            return []

        count = self._known_position_count(
            m.ticker,
            ctx_position,
        )

        # ============================================================
        # POSITION MANAGEMENT
        # ============================================================

        if count > 0:
            side = self._recover_side_from_position(
                m.ticker,
                ctx_position,
            )

            if side is None:
                logger.error(
                    "LIVE EXIT BLOCKED | ticker=%s | count=%d | reason=UNKNOWN_SIDE",
                    m.ticker,
                    count,
                )

                return []

            if (
                self._pending_action.get(m.ticker)
                is Action.SELL
            ):
                return []

            bid = (
                m.yes_bid
                if side is Side.YES
                else m.no_bid
            )

            if seconds_left <= self.final_exit_seconds:
                self._pending_action[
                    m.ticker
                ] = Action.SELL

                self._exit_reason[
                    m.ticker
                ] = "FINAL_60_SECOND_EXIT"

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
                self._pending_action[
                    m.ticker
                ] = Action.SELL

                self._exit_reason[
                    m.ticker
                ] = "TAKE_PROFIT"

                return [
                    self._order(
                        ticker=m.ticker,
                        action=Action.SELL,
                        side=side,
                        price=self.take_profit,
                        count=count,
                    )
                ]

            entry_price = self._entry_price_by_ticker.get(
                m.ticker
            )

            dynamic_stop = (
                max(
                    1,
                    min(
                        self.legacy_stop_cap,
                        entry_price - self.dynamic_stop_gap,
                    ),
                )
                if entry_price is not None
                else max(
                    1,
                    int(bid) - self.dynamic_stop_gap,
                )
            )

            ticks = ctx.underlying_ticks

            adverse_underlying = False

            if (
                ticks
                and m.floor_strike is not None
            ):
                latest_tick = ticks[-1]

                cutoff = (
                    latest_tick.timestamp.timestamp()
                    - 15.0
                )

                old_tick = next(
                    (
                        tick
                        for tick in reversed(ticks)
                        if tick.timestamp.timestamp()
                        <= cutoff
                    ),
                    ticks[0],
                )

                momentum_15 = (
                    latest_tick.price
                    - old_tick.price
                )

                target = float(
                    m.floor_strike
                )

                adverse_underlying = (
                    latest_tick.price <= target
                    or momentum_15
                    <= -self.stop_adverse_move
                    if side is Side.YES
                    else latest_tick.price >= target
                    or momentum_15
                    >= self.stop_adverse_move
                )

            if (
                bid <= dynamic_stop
                and adverse_underlying
            ):
                stop_checks = (
                    self._stop_confirmation_count_by_ticker.get(
                        m.ticker,
                        0,
                    )
                    + 1
                )

                self._stop_confirmation_count_by_ticker[
                    m.ticker
                ] = stop_checks

            else:
                self._stop_confirmation_count_by_ticker[
                    m.ticker
                ] = 0

            if (
                self._stop_confirmation_count_by_ticker[
                    m.ticker
                ]
                >= self.stop_confirmations
            ):
                self._pending_action[
                    m.ticker
                ] = Action.SELL

                self._exit_reason[
                    m.ticker
                ] = "CONFIRMED_GOLD_DYNAMIC_STOP"

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

        # ============================================================
        # ENTRY ELIGIBILITY
        # ============================================================

        if not self._within_trading_hours():
            self._log_entry_check(
                ticker=m.ticker,
                seconds_left=seconds_left,
                yes_bid=m.yes_bid,
                yes_ask=m.yes_ask,
                no_bid=m.no_bid,
                no_ask=m.no_ask,
                reason=(
                    "OUTSIDE_TRADING_HOURS_6AM_7PM_CENTRAL"
                ),
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

        if (
            m.ticker
            in self._chop_disqualified_tickers
        ):
            self._log_entry_check(
                ticker=m.ticker,
                seconds_left=seconds_left,
                yes_bid=m.yes_bid,
                yes_ask=m.yes_ask,
                no_bid=m.no_bid,
                no_ask=m.no_ask,
                reason="SKIP_MARKET_CHOP_DETECTED",
            )

            return []

        if (
            self._pending_action.get(m.ticker)
            is Action.BUY
        ):
            return []

        ticks = ctx.underlying_ticks

        if not ticks:
            self._log_entry_check(
                ticker=m.ticker,
                seconds_left=seconds_left,
                yes_bid=m.yes_bid,
                yes_ask=m.yes_ask,
                no_bid=m.no_bid,
                no_ask=m.no_ask,
                reason="NO_GOLD_DATA",
            )

            return []

        latest = ticks[-1]

        age = (
            datetime.now(timezone.utc)
            - latest.timestamp
        ).total_seconds()

        if (
            age > self.max_gold_age_seconds
            or m.floor_strike is None
        ):
            self._log_entry_check(
                ticker=m.ticker,
                seconds_left=seconds_left,
                yes_bid=m.yes_bid,
                yes_ask=m.yes_ask,
                no_bid=m
