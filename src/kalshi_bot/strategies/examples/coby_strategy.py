"""Coby's 15-minute Kalshi live strategy with confirmed-fill tracking."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import statistics
from typing import Any
from zoneinfo import ZoneInfo

from kalshi_bot.dashboard import get_open_position, record_entry, record_exit, record_model_snapshot, update_open_count
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
            params.get("minimum_history_seconds", params.get("min_history_seconds", 90))
        )
        self.required_confirmations = int(params.get("required_confirmations", 3))
        self.minimum_distance = float(params.get("minimum_distance", 15.0))
        self.hard_minimum_separation = float(params.get("hard_minimum_separation", 30.0))
        self.volatility_multiplier = float(params.get("volatility_multiplier", 0.35))
        self.maximum_target_crossings = int(params.get("maximum_target_crossings", 2))

        # Ignore ALL target crossings during the first 3 minutes.
        # Crossing count begins from zero after 3 minutes have elapsed.
        self.crossing_ignore_seconds = float(params.get("crossing_ignore_seconds", 180.0))
        self.crossing_skip_enable_seconds = float(params.get("crossing_skip_enable_seconds", 180.0))

        self.minimum_efficiency = float(params.get("minimum_efficiency", 0.12))
        self.minimum_trend_change = float(params.get("minimum_trend_change", -5.0))

        # Stability filters. These are designed to reject a market that looks
        # strong for a few snapshots but has not shown durable directional behavior.
        self.stability_lookback_seconds = float(params.get("stability_lookback_seconds", 90.0))
        self.minimum_stability_efficiency = float(
            params.get("minimum_stability_efficiency", 0.25)
        )
        self.minimum_same_side_ratio = float(params.get("minimum_same_side_ratio", 0.90))
        self.minimum_stability_net_move = float(
            params.get("minimum_stability_net_move", 10.0)
        )
        self.maximum_retracement_ratio = float(params.get("maximum_retracement_ratio", 0.35))
        self.maximum_retracement_floor = float(
            params.get("maximum_retracement_floor", 15.0)
        )

        self.max_btc_age_seconds = float(params.get("max_btc_age_seconds", 5))
        self.take_profit = int(params.get("take_profit", 98))
        self.dynamic_stop_gap = int(params.get("dynamic_stop_gap", 13))
        self.legacy_stop_cap = int(params.get("legacy_stop_cap", 79))
        self.max_contracts = int(params.get("max_contracts", 200))
        self.max_notional_cents = int(params.get("max_notional_cents", 20_000))
        self.final_exit_seconds = int(params.get("final_exit_seconds", 60))

        # Observe the first 10 minutes, then allow entries from 5:00 remaining
        # down to the final 60-second no-entry window.
        self.entry_window_seconds = int(params.get("entry_window_seconds", 300))

        # Only allow an entry when the chosen YES/NO contract ask is 70c or higher.
        self.min_entry_price = int(params.get("min_entry_price", 70))

        # The selected contract must stay at/above the entry floor continuously
        # before the normal 3-step confirmation sequence can begin.
        self.minimum_entry_price_hold_seconds = float(
            params.get("minimum_entry_price_hold_seconds", 30.0)
        )

        self.use_trading_hours = bool(params.get("use_trading_hours", False))
        self.trading_timezone = ZoneInfo(
            str(params.get("trading_timezone", "America/Chicago"))
        )
        self.trading_start_hour = int(params.get("trading_start_hour", 6))
        self.trading_end_hour = int(params.get("trading_end_hour", 19))

        self.stop_exit_floor = int(params.get("stop_exit_floor", 1))

        self._traded_tickers: set[str] = set()
        self._closed_tickers: set[str] = set()

        # If this market ever triggers CHOPPY_TARGET_CROSSINGS, it is
        # disqualified from entry for the rest of that 15-minute ticker.
        self._chop_disqualified_tickers: set[str] = set()

        # Target-crossing state is tracked per 15-minute ticker.
        self._crossing_count_by_ticker: dict[str, int] = {}
        self._last_target_side_by_ticker: dict[str, int] = {}

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

        # Timestamp of when the chosen contract first met the >=70c floor.
        # Uses market-data timestamps rather than wall-clock time so dry/live
        # behavior is measured consistently.
        self._price_floor_side_by_ticker: dict[str, Side] = {}
        self._price_floor_since_by_ticker: dict[str, datetime] = {}

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
        if not self.use_trading_hours:
            return True
        local_hour = datetime.now(self.trading_timezone).hour
        return self.trading_start_hour <= local_hour < self.trading_end_hour

    def _entry_count(self, limit_price: int) -> int:
        if limit_price <= 0:
            return 0
        return max(0, min(self.max_contracts, self.max_notional_cents // limit_price))

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
        saved = get_open_position(ticker)
        if saved is not None and saved.get("entry_price") is not None:
            self._entry_price_by_ticker[ticker] = int(saved["entry_price"])
        logger.warning(
            "LIVE POSITION RECOVERED | ticker=%s | side=%s | count=%d",
            ticker,
            side.value,
            abs(int(ctx_position)),
        )
        return side

    def _reset_confirmation(self, ticker: str) -> None:
        self._confirmation_count_by_ticker[ticker] = 0
        self._confirmation_side_by_ticker.pop(ticker, None)

    def _reset_price_floor_timer(self, ticker: str) -> None:
        self._price_floor_side_by_ticker.pop(ticker, None)
        self._price_floor_since_by_ticker.pop(ticker, None)

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
            self._reset_price_floor_timer(ticker)

            record_entry(
                ticker=ticker,
                side=request.side.value,
                entry_price=fill_price,
                count=fill_count,
                seconds_left=seconds_left,
                stop_price=max(1, min(self.legacy_stop_cap, fill_price - self.dynamic_stop_gap)),
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
                update_open_count(ticker, remaining)
                logger.warning(
                    "LIVE EXIT PARTIAL | ticker=%s | side=%s | fill=%dc | filled=%d | remaining=%d",
                    ticker, side.value, fill_price, fill_count, remaining,
                )
                return

            self._filled_count_by_ticker.pop(ticker, None)
            self._closed_tickers.add(ticker)

            entry_price = self._entry_price_by_ticker.pop(ticker, None)
            reason = self._exit_reason.pop(ticker, "EXIT")
            total_exit_count = self._exit_fill_count_by_ticker.pop(ticker, 0)
            total_exit_value = self._exit_fill_value_by_ticker.pop(ticker, 0)

            avg_exit = round(total_exit_value / total_exit_count) if total_exit_count > 0 else fill_price

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
                    "LIVE EXIT FILLED BUT ENTRY PRICE UNKNOWN | ticker=%s | side=%s | exit=%dc | count=%d",
                    ticker, side.value, avg_exit, total_exit_count,
                )

            self._side_by_ticker.pop(ticker, None)

            logger.warning(
                "LIVE EXIT FILLED | ticker=%s | side=%s | avg_exit=%dc | count=%d | reason=%s",
                ticker, side.value, avg_exit, total_exit_count, reason,
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
            "ENTRY CHECK | ticker=%s | seconds_left=%s | yes_bid=%s | yes_ask=%s | no_bid=%s | no_ask=%s | reason=%s",
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

        # ============================================================
        # OPEN POSITION MANAGEMENT
        # ============================================================

        if count > 0:
            side = self._recover_side_from_position(m.ticker, ctx_position)
            if side is None:
                logger.error(
                    "LIVE EXIT BLOCKED | ticker=%s | count=%d | reason=UNKNOWN_SIDE",
                    m.ticker, count,
                )
                return []

            if self._pending_action.get(m.ticker) is Action.SELL:
                return []

            bid = m.yes_bid if side is Side.YES else m.no_bid

            if seconds_left <= self.final_exit_seconds:
                self._pending_action[m.ticker] = Action.SELL
                self._exit_reason[m.ticker] = "FINAL_60_SECOND_EXIT"
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
                return [
                    self._order(
                        ticker=m.ticker,
                        action=Action.SELL,
                        side=side,
                        price=self.take_profit,
                        count=count,
                    )
                ]

            entry_price = self._entry_price_by_ticker.get(m.ticker)
            dynamic_stop = (
                max(1, min(self.legacy_stop_cap, entry_price - self.dynamic_stop_gap))
                if entry_price is not None
                else max(1, int(bid) - self.dynamic_stop_gap)
            )
            if bid <= dynamic_stop:
                self._pending_action[m.ticker] = Action.SELL
                self._exit_reason[m.ticker] = "DYNAMIC_STOP"
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
        # ENTRY FILTERS
        # ============================================================

        if not self._within_trading_hours():
            self._log_entry_check(
                ticker=m.ticker, seconds_left=seconds_left,
                yes_bid=m.yes_bid, yes_ask=m.yes_ask,
                no_bid=m.no_bid, no_ask=m.no_ask,
                reason="OUTSIDE_TRADING_HOURS_6AM_7PM_CENTRAL",
            )
            return []

        if m.ticker in self._traded_tickers:
            self._log_entry_check(
                ticker=m.ticker, seconds_left=seconds_left,
                yes_bid=m.yes_bid, yes_ask=m.yes_ask,
                no_bid=m.no_bid, no_ask=m.no_ask,
                reason="ALREADY_TRADED",
            )
            return []

        if m.ticker in self._chop_disqualified_tickers:
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

        if self._pending_action.get(m.ticker) is Action.BUY:
            return []

        if seconds_left <= 60:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            self._log_entry_check(
                ticker=m.ticker, seconds_left=seconds_left,
                yes_bid=m.yes_bid, yes_ask=m.yes_ask,
                no_bid=m.no_bid, no_ask=m.no_ask,
                reason="FINAL_60_SECONDS",
            )
            return []

        ticks = ctx.underlying_ticks
        if not ticks:
            self._log_entry_check(
                ticker=m.ticker, seconds_left=seconds_left,
                yes_bid=m.yes_bid, yes_ask=m.yes_ask,
                no_bid=m.no_bid, no_ask=m.no_ask,
                reason="NO_BTC_DATA",
            )
            return []

        latest = ticks[-1]
        age = (datetime.now(timezone.utc) - latest.timestamp).total_seconds()
        if age > self.max_btc_age_seconds or m.floor_strike is None:
            self._log_entry_check(
                ticker=m.ticker, seconds_left=seconds_left,
                yes_bid=m.yes_bid, yes_ask=m.yes_ask,
                no_bid=m.no_bid, no_ask=m.no_ask,
                reason="BTC_STALE_OR_NO_TARGET",
            )
            return []

        history = (latest.timestamp - ticks[0].timestamp).total_seconds()
        if history < self.min_history_seconds:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            return []

        def old_price(sec: float) -> float:
            cutoff = latest.timestamp.timestamp() - sec
            for tick in reversed(ticks):
                if tick.timestamp.timestamp() <= cutoff:
                    return tick.price
            return ticks[0].price

        mom15 = latest.price - old_price(15)
        mom60 = latest.price - old_price(60)

        recent = [
            x.price
            for x in ticks
            if (latest.timestamp - x.timestamp).total_seconds() <= 60
        ]
        diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        vol = statistics.pstdev(diffs) if len(diffs) >= 2 else 0.0

        drift = 0.65 * mom15 / 15.0 + 0.35 * mom60 / 60.0
        projected = latest.price + drift * min(seconds_left, 300) * 0.35
        sigma = max(5.0, max(vol, 0.35) * math.sqrt(max(seconds_left, 1)))
        z = (projected - float(m.floor_strike)) / sigma
        model_yes = max(0.01, min(0.99, 0.5 * (1 + math.erf(z / math.sqrt(2)))))
        model_no = 1 - model_yes

        target = float(m.floor_strike)
        separation = latest.price - target

        recent_ticks = [
            x
            for x in ticks
            if (latest.timestamp - x.timestamp).total_seconds() <= 60
        ]
        path = [x.price for x in recent_ticks]
        travel = sum(abs(path[i] - path[i - 1]) for i in range(1, len(path)))
        efficiency = abs(path[-1] - path[0]) / travel if travel > 0 and len(path) > 1 else 1.0

        # ============================================================
        # TARGET-CROSSING LOGIC
        # ============================================================

        elapsed_seconds = max(0.0, 900.0 - seconds_left)

        if elapsed_seconds < self.crossing_ignore_seconds:
            self._crossing_count_by_ticker[m.ticker] = 0
            self._last_target_side_by_ticker.pop(m.ticker, None)
        else:
            if separation > 0:
                current_target_side = 1
            elif separation < 0:
                current_target_side = -1
            else:
                current_target_side = 0

            if current_target_side != 0:
                previous_target_side = self._last_target_side_by_ticker.get(m.ticker)

                if previous_target_side is None:
                    self._last_target_side_by_ticker[m.ticker] = current_target_side
                elif current_target_side != previous_target_side:
                    self._crossing_count_by_ticker[m.ticker] = (
                        self._crossing_count_by_ticker.get(m.ticker, 0) + 1
                    )
                    self._last_target_side_by_ticker[m.ticker] = current_target_side

        crossings = self._crossing_count_by_ticker.get(m.ticker, 0)

        dynamic_distance = max(
            self.minimum_distance,
            self.volatility_multiplier
            * max(vol, 0.35)
            * math.sqrt(max(1.0, min(seconds_left, 60.0))),
        )

        common = dict(
            ticker=m.ticker,
            seconds_left=seconds_left,
            btc_price=latest.price,
            target_price=target,
            separation=separation,
            yes_bid=m.yes_bid,
            yes_ask=m.yes_ask,
            no_bid=m.no_bid,
            no_ask=m.no_ask,
            model_yes=model_yes * 100,
            model_no=model_no * 100,
            edge_yes=None if m.yes_ask is None else model_yes * 100 - m.yes_ask,
            edge_no=None if m.no_ask is None else model_no * 100 - m.no_ask,
            momentum_15=mom15,
            momentum_60=mom60,
            volatility=vol,
        )

        if (
            elapsed_seconds >= self.crossing_skip_enable_seconds
            and crossings >= self.maximum_target_crossings
        ):
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            self._chop_disqualified_tickers.add(m.ticker)
            record_model_snapshot(
                **common,
                decision="SKIP_MARKET",
                reason="SKIP_MARKET_2_PLUS_TARGET_CROSSINGS",
            )
            logger.warning(
                "MODEL MARKET SKIP | ticker=%s | reason=2_PLUS_TARGET_CROSSINGS | "
                "crossings=%d | elapsed_seconds=%.1f",
                m.ticker,
                crossings,
                elapsed_seconds,
            )
            return []

        # ============================================================
        # REVISED ENTRY WINDOW: FINAL FIVE MINUTES ONLY
        # ============================================================

        if seconds_left > self.entry_window_seconds:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="OBSERVATION_ONLY_OVER_5_MIN",
            )
            return []

        # ============================================================
        # SEPARATION FILTERS
        # ============================================================

        if abs(separation) < self.hard_minimum_separation:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="HARD_MINIMUM_SEPARATION_UNDER_30",
            )
            return []

        if abs(separation) < dynamic_distance:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="TOO_CLOSE_TO_TARGET",
            )
            return []

        if efficiency < self.minimum_efficiency:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="LOW_DIRECTIONAL_EFFICIENCY",
            )
            return []

        # ============================================================
        # MODEL SIDE SELECTION
        # ============================================================

        choices = []
        if m.yes_ask is not None:
            choices.append((model_yes * 100 - m.yes_ask, model_yes, Side.YES, m.yes_ask))
        if m.no_ask is not None:
            choices.append((model_no * 100 - m.no_ask, model_no, Side.NO, m.no_ask))

        if not choices:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(**common, decision="WAIT", reason="NO_EXECUTABLE_ASK")
            return []

        edge, confidence, side, observed_ask = max(choices, key=lambda x: x[0])

        if side is Side.YES and separation <= 0:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="MODEL_TARGET_DIRECTION_CONFLICT",
            )
            return []

        if side is Side.NO and separation >= 0:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="MODEL_TARGET_DIRECTION_CONFLICT",
            )
            return []

        # ============================================================
        # NEW 90-SECOND BTC STABILITY FILTER
        # ============================================================

        stability_ticks = [
            x
            for x in ticks
            if (latest.timestamp - x.timestamp).total_seconds()
            <= self.stability_lookback_seconds
        ]

        if len(stability_ticks) < 2:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="INSUFFICIENT_STABILITY_HISTORY",
            )
            return []

        stability_span = (
            stability_ticks[-1].timestamp - stability_ticks[0].timestamp
        ).total_seconds()

        if stability_span < self.stability_lookback_seconds * 0.90:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="INSUFFICIENT_STABILITY_HISTORY",
            )
            return []

        stability_path = [x.price for x in stability_ticks]
        stability_travel = sum(
            abs(stability_path[i] - stability_path[i - 1])
            for i in range(1, len(stability_path))
        )
        stability_efficiency = (
            abs(stability_path[-1] - stability_path[0]) / stability_travel
            if stability_travel > 0
            else 1.0
        )

        if stability_efficiency < self.minimum_stability_efficiency:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="UNSTABLE_90_SECOND_PATH",
            )
            return []

        if side is Side.YES:
            correct_side_count = sum(1 for x in stability_ticks if x.price > target)
            directional_distances = [x.price - target for x in stability_ticks]
            directional_net_move = stability_path[-1] - stability_path[0]
        else:
            correct_side_count = sum(1 for x in stability_ticks if x.price < target)
            directional_distances = [target - x.price for x in stability_ticks]
            directional_net_move = stability_path[0] - stability_path[-1]

        same_side_ratio = correct_side_count / len(stability_ticks)

        if same_side_ratio < self.minimum_same_side_ratio:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="TARGET_SIDE_NOT_STABLE",
            )
            return []

        if directional_net_move < self.minimum_stability_net_move:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="NO_SUSTAINED_DIRECTIONAL_PROGRESS",
            )
            return []

        favorable_distances = [max(0.0, distance) for distance in directional_distances]
        peak_favorable_distance = max(favorable_distances)
        current_favorable_distance = max(0.0, directional_distances[-1])
        recent_retracement = max(
            0.0,
            peak_favorable_distance - current_favorable_distance,
        )
        allowed_retracement = max(
            self.maximum_retracement_floor,
            peak_favorable_distance * self.maximum_retracement_ratio,
        )

        if recent_retracement > allowed_retracement:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="SEPARATION_RETRACING_TOO_FAST",
            )
            return []

        # ============================================================
        # MOMENTUM ALIGNMENT
        # ============================================================

        # Both 15-second and 60-second momentum must agree with the chosen side.
        if side is Side.YES and (mom15 <= 0 or mom60 <= 0):
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="MOMENTUM_DIRECTION_NOT_ALIGNED",
            )
            return []

        if side is Side.NO and (mom15 >= 0 or mom60 >= 0):
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="MOMENTUM_DIRECTION_NOT_ALIGNED",
            )
            return []

        trend_change = mom15 - (mom60 / 4.0)

        if side is Side.YES and trend_change < self.minimum_trend_change and mom15 < 0:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="MOMENTUM_WEAKENING",
            )
            return []

        if side is Side.NO and -trend_change < self.minimum_trend_change and mom15 > 0:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="MOMENTUM_WEAKENING",
            )
            return []

        # ============================================================
        # MODEL CONFIDENCE / EDGE
        # ============================================================

        if confidence < self.min_confidence:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="MODEL_CONFIDENCE_LOW",
            )
            return []

        if edge < self.min_edge_cents:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="EDGE_TOO_SMALL",
            )
            return []

        # ============================================================
        # CONTRACT PRICE FLOOR + 30-SECOND STABILITY
        # ============================================================

        if observed_ask < self.min_entry_price:
            self._reset_confirmation(m.ticker)
            self._reset_price_floor_timer(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason=f"ENTRY_PRICE_BELOW_{self.min_entry_price}C",
            )
            return []

        tracked_floor_side = self._price_floor_side_by_ticker.get(m.ticker)

        if tracked_floor_side is not side:
            self._price_floor_side_by_ticker[m.ticker] = side
            self._price_floor_since_by_ticker[m.ticker] = latest.timestamp
            self._reset_confirmation(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="ENTRY_PRICE_STABILITY_TIMER_STARTED",
            )
            return []

        floor_since = self._price_floor_since_by_ticker.get(m.ticker)

        if floor_since is None:
            self._price_floor_since_by_ticker[m.ticker] = latest.timestamp
            self._reset_confirmation(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="ENTRY_PRICE_STABILITY_TIMER_STARTED",
            )
            return []

        floor_hold_seconds = (latest.timestamp - floor_since).total_seconds()

        if floor_hold_seconds < self.minimum_entry_price_hold_seconds:
            self._reset_confirmation(m.ticker)
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason="ENTRY_PRICE_NOT_STABLE_30_SECONDS",
            )
            return []

        # ============================================================
        # FINAL THREE-STEP CONFIRMATION
        # ============================================================

        previous_side = self._confirmation_side_by_ticker.get(m.ticker)
        confirmations = self._confirmation_count_by_ticker.get(m.ticker, 0)
        confirmations = confirmations + 1 if previous_side is side else 1
        self._confirmation_side_by_ticker[m.ticker] = side
        self._confirmation_count_by_ticker[m.ticker] = confirmations

        if confirmations < self.required_confirmations:
            record_model_snapshot(
                **common,
                decision="WAIT",
                reason=f"CONFIRMING_{confirmations}_OF_{self.required_confirmations}",
            )
            return []

        count = self._entry_count(observed_ask)
        if count <= 0:
            return []

        self._pending_action[m.ticker] = Action.BUY
        record_model_snapshot(
            **common,
            decision="BUY_" + side.value.upper(),
            reason="MODEL_EDGE_CONFIRMED_STABLE",
        )
        logger.warning(
            "MODEL ENTRY | ticker=%s | side=%s | ask=%dc | confidence=%.1f%% | "
            "edge=%.1fc | confirmations=%d | seconds_left=%.1f | "
            "stability_efficiency=%.3f | same_side_ratio=%.3f | "
            "directional_move=%.2f | retracement=%.2f | price_floor_hold=%.1fs",
            m.ticker,
            side.value,
            observed_ask,
            confidence * 100,
            edge,
            confirmations,
            seconds_left,
            stability_efficiency,
            same_side_ratio,
            directional_net_move,
            recent_retracement,
            floor_hold_seconds,
        )
        return [
            self._order(
                ticker=m.ticker,
                action=Action.BUY,
                side=side,
                price=observed_ask,
                count=count,
            )
        ]
