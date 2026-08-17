from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from kalshi_bot.exchange.models import Side
from kalshi_bot.strategies.base import Strategy, StrategyContext
from kalshi_bot.telemetry.logging import get_logger
from kalshi_bot.dashboard import record_entry, record_exit

logger = get_logger(__name__)


class CobyStrategy(Strategy):
    """
    PAPER-ONLY Kalshi dry-run strategy.

    Uses Kalshi data only:
    - Considers entries in the final 5 minutes.
    - Chooses whichever side Kalshi currently prices as more likely.
    - Maximum entry price: 90 cents.
    - Take profit: 98 cents.
    - Stop trigger: 79 cents.
    - Simulates positions internally and NEVER emits a real order.

    This build is intentionally for collecting dry-run data.
    """

    name = "coby_strategy"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

        self.entry_min = int(params.get("entry_min", 90))
        self.entry_max = int(params.get("entry_max",95))
        self.take_profit = int(params.get("take_profit", 98))
        self.stop_price = int(params.get("stop_price", 79))
        self.entry_window_seconds = int(
            params.get("entry_window_seconds", 300)
        )
        self.size = int(params.get("size", 1))

        self._paper_ticker: str | None = None
        self._paper_side: Side | None = None
        self._paper_entry_price: int | None = None
        self._paper_entry_time: datetime | None = None
        self._traded_tickers: set[str] = set()
        self._paper_total_pnl_cents = 0

    def _seconds_to_close(self, close_time: str | None) -> float | None:
        if not close_time:
            return None

        try:
            close_dt = datetime.fromisoformat(
                close_time.replace("Z", "+00:00")
            )
            if close_dt.tzinfo is None:
                close_dt = close_dt.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            return (close_dt - now).total_seconds()
        except ValueError:
            return None

    def _reset_position(self) -> None:
        self._paper_ticker = None
        self._paper_side = None
        self._paper_entry_price = None
        self._paper_entry_time = None

    def _log_exit(self, exit_price: int, reason: str) -> None:
        if (
            self._paper_ticker is None
            or self._paper_side is None
            or self._paper_entry_price is None
        ):
            return

        pnl_per_contract = exit_price - self._paper_entry_price
        trade_pnl = pnl_per_contract * self.size
        self._paper_total_pnl_cents += trade_pnl

        logger.info(
            "PAPER EXIT | ticker=%s | side=%s | entry=%dc | exit=%dc | "
            "reason=%s | count=%d | pnl=%+dc | total_pnl=%+dc",
            self._paper_ticker,
            self._paper_side.value,
            self._paper_entry_price,
            exit_price,
            reason,
            self.size,
            trade_pnl,
            self._paper_total_pnl_cents,
        )
                record_exit(
            ticker=self._paper_ticker,
            side=self._paper_side.value,
            entry_price=self._paper_entry_price,
            exit_price=exit_price,
            reason=reason,
            count=self.size,
            pnl_cents=trade_pnl,
            total_pnl_cents=self._paper_total_pnl_cents,
        )
        self._reset_position()

    def on_market_data(self, ctx: StrategyContext) -> list:
        m = ctx.market
        seconds_left = self._seconds_to_close(m.close_time)

        # If Kalshi has rolled to a new 15-minute ticker while a paper trade
        # is still open, record it as unresolved rather than inventing a result.
        if self._paper_ticker is not None and self._paper_ticker != m.ticker:
            logger.warning(
                "PAPER ROLLOVER_UNRESOLVED | ticker=%s | side=%s | entry=%dc",
                self._paper_ticker,
                self._paper_side.value if self._paper_side else "UNKNOWN",
                self._paper_entry_price or 0,
            )
            self._reset_position()

        # Manage an existing paper position.
        if self._paper_ticker == m.ticker and self._paper_side is not None:
            bid = m.yes_bid if self._paper_side is Side.YES else m.no_bid

            if bid is None:
                return []

            if bid >= self.take_profit:
                self._log_exit(bid, "TAKE_PROFIT")
                return []

            if bid <= self.stop_price:
                self._log_exit(bid, "STOP")
                return []

            return []

        # One simulated entry maximum per 15-minute market.
        if m.ticker in self._traded_tickers:
            return []

        # Entry timing filter.
        if seconds_left is None:
            return []

        if seconds_left > self.entry_window_seconds or seconds_left <= 0:
            return []

        # Need both sides to decide which side Kalshi currently favors.
        if m.yes_ask is None or m.no_ask is None:
            return []

        if m.yes_ask >= m.no_ask:
            side = Side.YES
            entry_price = m.yes_ask
        else:
            side = Side.NO
            entry_price = m.no_ask

        # Preserve the existing 90-cent minimum entry rule.
        if entry_price < self.entry_min or entry_price > self.entry_max:
            return []

        self._paper_ticker = m.ticker
        self._paper_side = side
        self._paper_entry_price = entry_price
        self._paper_entry_time = datetime.now(timezone.utc)
        self._traded_tickers.add(m.ticker)

        logger.info(
            "PAPER ENTER | ticker=%s | side=%s | entry=%dc | count=%d | "
            "seconds_left=%.1f | yes_bid=%s | yes_ask=%s | no_bid=%s | no_ask=%s",
            m.ticker,
            side.value,
            entry_price,
            self.size,
            seconds_left,
            m.yes_bid,
            m.yes_ask,
            m.no_bid,
            m.no_ask,
        )

        # PAPER ONLY: never emit a real OrderRequest.
        return []
