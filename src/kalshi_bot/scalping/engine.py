"""Kalshi 15-minute momentum scalper (replaces the old BTC maker scalper).

Drop-in replacement for ``scalping/engine.py``: the existing start command
keeps working and runs this scalper in paper mode::

    python -m kalshi_bot.cli btc-scalper

What it does
------------
* Scans every 15-minute crypto and commodity series on Kalshi (not just BTC).
* Profiles each series' liquidity (trades/min, spread, depth) and sorts it
  into HIGH / MEDIUM / THIN. THIN markets are watch-only by default.
* Enters only when price is moving fast for that market, real taker volume
  confirms it, the order book leans the same way, and the move holds for a
  second check. Rising price buys YES, falling price buys NO.
* Exits: +10c take profit, -5c stop (two checks) or -8c hard stop, stall
  exit after 3 minutes, exit with 60s left, and close-out on shutdown.
* Records every order-book/trade message, one row per market per second,
  every trade and every settlement result.

Settings (environment variables, all optional)
----------------------------------------------
SCALPER_CONTRACTS=5          contracts per trade (MEDIUM tier uses half)
SCALPER_TAKE_PROFIT=10       cents per contract
SCALPER_STOP_LOSS=5          cents per contract
SCALPER_SERIES=              e.g. KXBTC15M,KXETH15M (empty = all 15m series)
SCALPER_CATEGORIES=Crypto,Commodities
SCALPER_TRADE_THIN=false     also trade THIN markets
SCALPER_DAILY_LOSS=10        dollars; no new entries after this daily loss
SCALPER_MAX_POSITIONS=4
SCALPER_RECORD_RAW=true      raw message recording (needed for replay)
SCALPER_DATA_DIR=            default: $RAILWAY_VOLUME_MOUNT_PATH/scalper
KALSHI_SCALPER_LIVE=false    set to true to send REAL orders

Reports and replay (no trading)::

    python -m kalshi_bot.scalping.engine report
    python -m kalshi_bot.scalping.engine replay --take-profit 8 --stop-loss 6

Fees: Kalshi taker fee ceil(7% x contracts x P x (1-P)), ~2c/contract each
way at mid prices. A +10c win nets ~+6c, a stop costs ~-9 to -10c, so the
strategy must win ~60% of trades. Paper fills are optimistic (no latency).
"""

from __future__ import annotations

# ==========================================================================
# CONFIG
# Configuration for the Kalshi momentum scalper.
#
# Every threshold lives here so it can be tuned from recorded data (see
# ``scalper-replay``) without touching strategy code. The liquidity thresholds
# are starting guesses: run in paper mode first, then look at the
# ``trade_rate``, ``depth`` and ``spread`` columns in the tick log and adjust.
# ==========================================================================
from dataclasses import dataclass, field

HIGH = "HIGH"
MEDIUM = "MEDIUM"
THIN = "THIN"
WARMUP = "WARMUP"


@dataclass(slots=True)
class TierRules:
    """Entry rules for one liquidity tier."""

    tradeable: bool
    # Minimum mid-price move (cents) over the momentum window.
    min_move_cents: float
    # The move must also be this multiple of the market's typical 30s move,
    # so a 4c move only counts in a market that normally moves 1-2c.
    vol_multiple: float
    # Directional taker volume over the flow window must reach the larger of
    # this floor and ``volume_surge`` x the market's normal volume for the
    # same window.
    min_aggressive_contracts: int
    volume_surge: float
    # (buyers - sellers) / total taker volume in the trade direction.
    min_flow_ratio: float
    # (support - resistance) / total resting depth near the top of book.
    min_book_imbalance: float
    max_spread_cents: int
    # Position size as a fraction of the configured contract count.
    size_fraction: float


def _default_tiers() -> dict[str, TierRules]:
    return {
        HIGH: TierRules(
            tradeable=True,
            min_move_cents=4,
            vol_multiple=2.0,
            min_aggressive_contracts=25,
            volume_surge=1.0,
            min_flow_ratio=0.40,
            min_book_imbalance=0.10,
            max_spread_cents=2,
            size_fraction=1.0,
        ),
        MEDIUM: TierRules(
            tradeable=True,
            min_move_cents=5,
            vol_multiple=2.5,
            min_aggressive_contracts=10,
            volume_surge=1.2,
            min_flow_ratio=0.50,
            min_book_imbalance=0.20,
            max_spread_cents=3,
            size_fraction=0.5,
        ),
        THIN: TierRules(
            # Watch-only by default: in thin books a "move" is often a quote
            # jumping, and the exit may not fill.
            tradeable=False,
            min_move_cents=6,
            vol_multiple=3.0,
            min_aggressive_contracts=5,
            volume_surge=1.5,
            min_flow_ratio=0.60,
            min_book_imbalance=0.30,
            max_spread_cents=3,
            size_fraction=0.4,
        ),
    }


@dataclass(slots=True)
class ScalperConfig:
    # ---- sizing / exits --------------------------------------------------
    contracts: int = 5
    take_profit_cents: int = 10
    stop_loss_cents: int = 5
    # Beyond this adverse move, exit immediately without waiting for a
    # second confirmation.
    hard_stop_cents: int = 8
    stop_confirmations: int = 2
    stop_confirm_seconds: float = 1.0
    # Exit if the trade hasn't made this much progress after this long.
    stall_seconds: float = 180.0
    stall_min_progress_cents: int = 3
    # Never hold into the last minute: settlement flips are where scalps die.
    exit_before_close_seconds: float = 60.0

    # ---- entry filters ---------------------------------------------------
    min_seconds_left_entry: float = 180.0
    min_market_age_seconds: float = 45.0
    min_entry_price: int = 10
    max_entry_price: int = 85
    # Skip if price already ran this far in the last 60s (late chase).
    max_run_cents: float = 25.0
    # Share of the recent price path (in cents) that went in the trade
    # direction, measured over ``consistency_window`` seconds.
    consistency_min: float = 0.75
    consistency_window: float = 15.0
    signal_confirmations: int = 2
    signal_confirm_seconds: float = 1.0
    # Abandon a building signal if the ask runs this far past where it began.
    max_chase_cents: int = 2

    # ---- execution -------------------------------------------------------
    entry_slippage_cents: int = 1
    stop_slippage_cents: int = 3
    taker_fee_rate: float = 0.07

    # ---- feature windows (seconds) ---------------------------------------
    fast_window: float = 10.0
    move_window: float = 30.0
    slow_window: float = 60.0
    flow_window: float = 30.0
    depth_band_cents: int = 3

    # ---- risk ------------------------------------------------------------
    cooldown_after_exit_seconds: float = 30.0
    cooldown_after_stop_seconds: float = 90.0
    max_entries_per_market: int = 3
    max_open_positions: int = 4
    daily_loss_limit_cents: int = 1_000
    # Consecutive live order errors before entries pause.
    max_order_errors: int = 3
    error_pause_seconds: float = 300.0

    # ---- liquidity profiling (per series, rolling) -----------------------
    profile_window_seconds: float = 600.0
    profile_warmup_seconds: float = 120.0
    high_min_trade_rate: float = 300.0  # contracts per minute
    high_min_depth: float = 500.0  # contracts within depth band, both sides
    high_max_spread: float = 2.0
    medium_min_trade_rate: float = 50.0
    medium_min_depth: float = 100.0
    medium_max_spread: float = 3.0

    tiers: dict[str, TierRules] = field(default_factory=_default_tiers)

    def size_for(self, tier: str) -> int:
        rules = self.tiers.get(tier)
        if rules is None:
            return 0
        return max(1, round(self.contracts * rules.size_fraction))



# ==========================================================================
# FEES
# Kalshi taker fee: ceil(rate x contracts x P x (1 - P)) in cents.
#
# Matches the fees in the adaptive bot's trade log, e.g. 2 contracts at 75c:
# 0.07 x 2 x 0.75 x 0.25 = 2.6c -> 3c.
# ==========================================================================
import math


def taker_fee_cents(count: int, price_cents: int, rate: float = 0.07) -> int:
    return taker_fee_for_levels([(price_cents, count)], rate)


def taker_fee_for_levels(levels: list[tuple[int, int]], rate: float = 0.07) -> int:
    raw = sum(rate * count * price * (100 - price) / 100.0 for price, count in levels if count > 0)
    return math.ceil(raw - 1e-9) if raw > 0 else 0



# ==========================================================================
# MARKET STATE
# Per-market price history and per-series liquidity profiles.
# ==========================================================================
import statistics
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from decimal import Decimal
from typing import Any

from kalshi_bot.scalping.book import BinaryOrderBook, contracts, dollars_to_cents


@dataclass(slots=True)
class MarketInfo:
    ticker: str
    series: str
    open_ts: float
    close_ts: float
    exchange_index: int | None = None
    floor_strike: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MarketInfo:
        return cls(**{f.name: data.get(f.name) for f in fields(cls)})


@dataclass(slots=True)
class TradePrint:
    ts: float
    yes_price: int
    count: int
    taker_side: str  # "yes" = buyer lifted YES ask, "no" = buyer lifted NO ask


def _price_cents(raw: Any) -> int:
    value = Decimal(str(raw))
    return dollars_to_cents(value) if value <= 1 else int(value)


def parse_trade(payload: dict[str, Any], received_ts: float) -> TradePrint | None:
    yes_raw = payload.get("yes_price_dollars")
    if yes_raw is None:
        yes_raw = payload.get("yes_price")
    if yes_raw is None:
        no_raw = payload.get("no_price_dollars", payload.get("no_price"))
        if no_raw is None:
            return None
        yes_price = 100 - _price_cents(no_raw)
    else:
        yes_price = _price_cents(yes_raw)
    count_raw = payload.get("count_fp", payload.get("count", 0))
    count = contracts(count_raw)
    side = str(payload.get("taker_side", "")).lower()
    if count <= 0 or side not in {"yes", "no"}:
        return None
    return TradePrint(received_ts, yes_price, count, side)


def side_depth(book: BinaryOrderBook, outcome: str, band: int) -> int:
    levels = book.yes if outcome == "yes" else book.no
    best = book.bid(outcome)
    if best is None:
        return 0
    return sum(size for price, size in levels.items() if price > best - band)


def book_crossed(book: BinaryOrderBook) -> bool:
    """A real binary book always has best YES bid + best NO bid < 100.

    If not, the local book is corrupt (wrong price convention, missed delta)
    and nothing may be traded or exited against it.
    """
    yes, no = book.bid("yes"), book.bid("no")
    return yes is not None and no is not None and yes + no >= 100


def ask_levels(book: BinaryOrderBook, outcome: str) -> list[tuple[int, int]]:
    """Ascending (price, size) levels a buyer of ``outcome`` can lift."""
    opposite = book.no if outcome == "yes" else book.yes
    return sorted((100 - price, size) for price, size in opposite.items())


def bid_levels(book: BinaryOrderBook, outcome: str) -> list[tuple[int, int]]:
    """Descending (price, size) levels a seller of ``outcome`` can hit."""
    levels = book.yes if outcome == "yes" else book.no
    return sorted(levels.items(), reverse=True)


@dataclass(slots=True)
class TickerState:
    info: MarketInfo
    book: BinaryOrderBook
    # (ts, mid) sampled on every engine step while the book is two-sided.
    mids: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=2000))
    trades: deque[TradePrint] = field(default_factory=lambda: deque(maxlen=5000))
    first_seen: float = 0.0

    @classmethod
    def create(cls, info: MarketInfo, now: float) -> TickerState:
        return cls(info=info, book=BinaryOrderBook(info.ticker), first_seen=now)

    def mid(self) -> float | None:
        bid, ask = self.book.yes_bid, self.book.yes_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def sample(self, now: float) -> None:
        mid = self.mid()
        if mid is not None:
            self.mids.append((now, mid))
        horizon = now - 120.0
        while self.trades and self.trades[0].ts < horizon:
            self.trades.popleft()

    def mid_at(self, ts: float) -> float | None:
        """Latest sampled mid at or before ``ts``."""
        found = None
        for sample_ts, mid in reversed(self.mids):
            if sample_ts <= ts:
                found = mid
                break
        return found

    def history_covers(self, seconds: float, now: float) -> bool:
        return bool(self.mids) and self.mids[0][0] <= now - seconds

    def flow(self, window: float, now: float) -> tuple[int, int]:
        yes = no = 0
        start = now - window
        for trade in reversed(self.trades):
            if trade.ts < start:
                break
            if trade.taker_side == "yes":
                yes += trade.count
            else:
                no += trade.count
        return yes, no

    def seconds_left(self, now: float) -> float:
        return self.info.close_ts - now

    def age(self, now: float) -> float:
        return now - max(self.info.open_ts, self.first_seen)


@dataclass(slots=True)
class ProfileStats:
    tier: str
    trade_rate: float  # contracts / minute
    spread: float | None
    depth: float | None
    typical_move: float  # median |30s mid change|, cents
    observed_seconds: float


class SeriesProfile:
    """Rolling liquidity and volatility profile for one series (e.g. KXBTC15M).

    It spans consecutive 15-minute markets, so a fresh market inherits what
    the series normally looks like instead of starting blind.
    """

    _SAMPLE_EVERY = 5.0

    def __init__(self, series: str, now: float) -> None:
        self.series = series
        self.started = now
        self._samples: deque[tuple[float, str, float, int, int]] = deque()
        self._trades: deque[tuple[float, int]] = deque()
        self._last_sample = 0.0
        self._stats: ProfileStats | None = None
        self._stats_ts = 0.0

    def on_trade(self, trade: TradePrint) -> None:
        self._trades.append((trade.ts, trade.count))

    def sample(self, state: TickerState, now: float, band: int) -> None:
        if now - self._last_sample < self._SAMPLE_EVERY:
            return
        mid = state.mid()
        spread = state.book.spread
        if mid is None or spread is None or spread <= 0:
            return
        depth = side_depth(state.book, "yes", band) + side_depth(state.book, "no", band)
        self._samples.append((now, state.info.ticker, mid, spread, depth))
        self._last_sample = now

    def _trim(self, now: float, window: float) -> None:
        horizon = now - window
        while self._samples and self._samples[0][0] < horizon:
            self._samples.popleft()
        while self._trades and self._trades[0][0] < horizon:
            self._trades.popleft()

    def stats(self, now: float, cfg: ScalperConfig) -> ProfileStats:
        if self._stats is not None and now - self._stats_ts < 5.0:
            return self._stats
        self._trim(now, cfg.profile_window_seconds)
        observed = now - self.started
        span = max(1.0, min(observed, cfg.profile_window_seconds))
        trade_rate = sum(count for _, count in self._trades) * 60.0 / span

        spreads = [s[3] for s in self._samples]
        depths = [s[4] for s in self._samples]
        spread = statistics.median(spreads) if spreads else None
        depth = statistics.median(depths) if depths else None

        moves: list[float] = []
        samples = list(self._samples)
        for i, (ts, ticker, mid, _, _) in enumerate(samples):
            for j in range(i - 1, -1, -1):
                prev_ts, prev_ticker, prev_mid, _, _ = samples[j]
                if ts - prev_ts >= 30.0:
                    if prev_ticker == ticker and ts - prev_ts <= 40.0:
                        moves.append(abs(mid - prev_mid))
                    break
        typical = max(1.0, statistics.median(moves)) if moves else 1.0

        if observed < cfg.profile_warmup_seconds or spread is None or depth is None:
            tier = WARMUP
        elif (
            trade_rate >= cfg.high_min_trade_rate
            and depth >= cfg.high_min_depth
            and spread <= cfg.high_max_spread
        ):
            tier = HIGH
        elif (
            trade_rate >= cfg.medium_min_trade_rate
            and depth >= cfg.medium_min_depth
            and spread <= cfg.medium_max_spread
        ):
            tier = MEDIUM
        else:
            tier = THIN

        self._stats = ProfileStats(tier, trade_rate, spread, depth, typical, observed)
        self._stats_ts = now
        return self._stats



# ==========================================================================
# SIGNALS
# Momentum features and the entry decision.
#
# A long entry needs all of these at once:
#
# * price moving fast (30s mid move above both a floor and a multiple of the
#   market's own typical 30s move),
# * the move still going (10s move not against it) and not already exhausted,
# * the last 15 seconds moving mostly one way (size-weighted consistency),
# * real taker volume confirming it, above the market's normal volume,
# * the order book leaning the same way,
# * a tight enough spread and room left to the take-profit target.
# ==========================================================================
from dataclasses import asdict, dataclass
from typing import Any



@dataclass(slots=True)
class Features:
    yes_bid: int | None
    yes_ask: int | None
    spread: int | None
    mid: float | None
    move_fast: float | None
    move: float | None
    move_slow: float | None
    up_cents: float
    down_cents: float
    yes_flow: int
    no_flow: int
    yes_depth: int
    no_depth: int
    seconds_left: float
    age: float

    @property
    def imbalance(self) -> float:
        total = self.yes_depth + self.no_depth
        return 0.0 if total == 0 else (self.yes_depth - self.no_depth) / total

    @property
    def flow_ratio(self) -> float:
        total = self.yes_flow + self.no_flow
        return 0.0 if total == 0 else (self.yes_flow - self.no_flow) / total

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["imbalance"] = round(self.imbalance, 4)
        data["flow_ratio"] = round(self.flow_ratio, 4)
        return data


@dataclass(slots=True)
class Signal:
    side: str  # "yes" or "no"
    ask: int
    threshold: float
    detail: str


def compute_features(state: TickerState, now: float, cfg: ScalperConfig) -> Features:
    mid = state.mid()

    def move(window: float) -> float | None:
        if mid is None or not state.history_covers(window, now):
            return None
        past = state.mid_at(now - window)
        return None if past is None else mid - past

    # Size-weighted consistency over the recent part of the move: how much of
    # the path went with the move vs against it. Measured over a shorter window
    # than the move itself so a clean breakout out of a choppy range qualifies.
    up = down = 0.0
    window = cfg.consistency_window
    if mid is not None and state.history_covers(window, now):
        step = window / 6.0
        prev = state.mid_at(now - window)
        t = now - window + step
        while t <= now + 1e-9 and prev is not None:
            current = state.mid_at(t)
            if current is not None:
                if current > prev:
                    up += current - prev
                elif current < prev:
                    down += prev - current
                prev = current
            t += step

    yes_flow, no_flow = state.flow(cfg.flow_window, now)
    return Features(
        yes_bid=state.book.yes_bid,
        yes_ask=state.book.yes_ask,
        spread=state.book.spread,
        mid=mid,
        move_fast=move(cfg.fast_window),
        move=move(cfg.move_window),
        move_slow=move(cfg.slow_window),
        up_cents=up,
        down_cents=down,
        yes_flow=yes_flow,
        no_flow=no_flow,
        yes_depth=side_depth(state.book, "yes", cfg.depth_band_cents),
        no_depth=side_depth(state.book, "no", cfg.depth_band_cents),
        seconds_left=state.seconds_left(now),
        age=state.age(now),
    )


def evaluate(
    f: Features,
    profile: ProfileStats,
    rules: TierRules,
    cfg: ScalperConfig,
) -> tuple[Signal | None, str]:
    """Return (signal, reason). ``reason`` explains a rejection."""
    if f.yes_bid is None or f.yes_ask is None or f.spread is None or f.mid is None:
        return None, "BOOK_ONE_SIDED"
    if f.move is None or f.move_fast is None:
        return None, "HISTORY_SHORT"
    if f.spread <= 0:
        return None, f"BOOK_CROSSED spread={f.spread}c"
    if f.spread > rules.max_spread_cents:
        return None, f"SPREAD_WIDE {f.spread}c>{rules.max_spread_cents}c"

    threshold = max(rules.min_move_cents, rules.vol_multiple * profile.typical_move)
    if abs(f.move) < threshold:
        return None, f"NO_MOMENTUM move={f.move:+.1f}c need={threshold:.1f}c"

    side = "yes" if f.move > 0 else "no"
    sign = 1 if side == "yes" else -1

    if sign * f.move_fast < 0:
        return None, f"FADING move_fast={f.move_fast:+.1f}c"
    if f.move_slow is not None and abs(f.move_slow) > cfg.max_run_cents:
        return None, f"EXHAUSTED move_60s={f.move_slow:+.1f}c"

    with_move = f.up_cents if side == "yes" else f.down_cents
    against_move = f.down_cents if side == "yes" else f.up_cents
    moving = with_move + against_move
    consistency = with_move / moving if moving else 0.0
    if consistency < cfg.consistency_min:
        return None, f"CHOPPY consistency={consistency:.2f}"

    directional = f.yes_flow if side == "yes" else f.no_flow
    normal = profile.trade_rate * cfg.flow_window / 60.0
    need_volume = max(rules.min_aggressive_contracts, rules.volume_surge * normal)
    if directional < need_volume:
        return None, f"NO_VOLUME taker={directional} need={need_volume:.0f}"
    if sign * f.flow_ratio < rules.min_flow_ratio:
        return None, f"FLOW_MIXED ratio={sign * f.flow_ratio:+.2f}"
    if sign * f.imbalance < rules.min_book_imbalance:
        return None, f"BOOK_AGAINST imbalance={sign * f.imbalance:+.2f}"

    ask = f.yes_ask if side == "yes" else 100 - f.yes_bid
    if ask < cfg.min_entry_price or ask > cfg.max_entry_price:
        return None, f"PRICE_OUT_OF_RANGE {side}_ask={ask}c"
    if ask + cfg.take_profit_cents > 99:
        return None, f"NO_ROOM {side}_ask={ask}c"

    detail = (
        f"{side.upper()} ask={ask}c move30={f.move:+.1f}c move10={f.move_fast:+.1f}c "
        f"need={threshold:.1f}c consistency={consistency:.2f} taker={directional} "
        f"flow={sign * f.flow_ratio:+.2f} book={sign * f.imbalance:+.2f} spread={f.spread}c"
    )
    return Signal(side, ask, threshold, detail), "SIGNAL"



# ==========================================================================
# CORE
# Exchange-independent scalper logic.
#
# The core never talks to Kalshi. It consumes WebSocket messages and a clock,
# and returns order *intents*. A broker (paper or live) executes each intent
# and reports the fill back. The same core therefore runs live, in paper mode,
# and in ``scalper-replay`` over recorded data, with identical decisions.
# ==========================================================================
import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo


logger = logging.getLogger(__name__)
_CENTRAL = ZoneInfo("America/Chicago")

Listener = Callable[[str, dict[str, Any]], None]


@dataclass(slots=True)
class Intent:
    intent_id: int
    ticker: str
    action: str  # "buy" | "sell"
    side: str  # "yes" | "no"
    count: int
    limit_price: int  # outcome price in cents
    reason: str
    exchange_index: int | None = None


@dataclass(slots=True)
class Fill:
    count: int
    avg_price: float
    fee_cents: int


@dataclass(slots=True)
class Position:
    ticker: str
    series: str
    side: str
    tier: str
    count: int
    entry_price: float
    entry_fee: int
    entry_ts: float
    features: dict[str, Any]
    profile: dict[str, Any]
    exit_proceeds: float = 0.0
    exit_fees: int = 0
    exited: int = 0
    stop_streak: int = 0
    last_stop_check: float = 0.0
    peak_bid: int = 0
    trough_bid: int = 100
    awaiting_settlement: bool = False
    last_exit_reason: str = ""

    @property
    def remaining(self) -> int:
        return self.count - self.exited


@dataclass(slots=True)
class PendingSignal:
    side: str
    first_ts: float
    first_ask: int
    last_ts: float
    streak: int


@dataclass(slots=True)
class CoreStats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    net_cents: int = 0
    fees_cents: int = 0


class ScalperCore:
    def __init__(
        self,
        cfg: ScalperConfig,
        mode: str = "paper",
        listener: Listener | None = None,
    ) -> None:
        self.cfg = cfg
        self.mode = mode
        self.listener = listener
        self.markets: dict[str, TickerState] = {}
        self.profiles: dict[str, SeriesProfile] = {}
        self.positions: dict[str, Position] = {}
        self.pending_orders: dict[str, Intent] = {}
        self.pending_signals: dict[str, PendingSignal] = {}
        self.cooldown_until: dict[str, float] = {}
        self.entries: dict[str, int] = {}
        self.blocked: set[str] = set()
        self.stats = CoreStats()
        self.closed_trades: list[dict[str, Any]] = []
        self._ids = itertools.count(1)
        self._day: date | None = None
        self.daily_net_cents = 0
        self._order_errors = 0
        self._paused_until = 0.0
        self.last_reason: dict[str, str] = {}
        self._entry_context: dict[int, dict[str, Any]] = {}
        self._crossed_logged: dict[str, float] = {}

    # ------------------------------------------------------------------ io
    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.listener is not None:
            try:
                self.listener(kind, data)
            except Exception:  # never let telemetry break trading
                logger.exception("listener failed for %s", kind)

    def add_market(self, info: MarketInfo, now: float) -> None:
        if info.ticker in self.markets:
            return
        self.markets[info.ticker] = TickerState.create(info, now)
        self.profiles.setdefault(info.series, SeriesProfile(info.series, now))

    def drop_market(self, ticker: str) -> None:
        if ticker in self.positions or ticker in self.pending_orders:
            return
        self.markets.pop(ticker, None)
        self.pending_signals.pop(ticker, None)
        self.cooldown_until.pop(ticker, None)
        self.entries.pop(ticker, None)
        self.last_reason.pop(ticker, None)

    def reset_books(self) -> None:
        """Called after a reconnect; books stay unusable until a new snapshot."""
        for state in self.markets.values():
            state.book.ready = False
            state.book.yes.clear()
            state.book.no.clear()
            state.book.sequence = 0
            state.mids.clear()
        self.pending_signals.clear()

    def on_message(self, message: dict[str, Any], now: float) -> None:
        payload = message.get("msg") or {}
        ticker = payload.get("market_ticker")
        state = self.markets.get(ticker) if ticker else None
        if state is None:
            return
        kind = message.get("type")
        if kind in ("orderbook_snapshot", "orderbook_delta"):
            state.book.apply(message)
        elif kind == "trade":
            trade = parse_trade(payload, now)
            if trade is not None:
                state.trades.append(trade)
                self.profiles[state.info.series].on_trade(trade)

    # ---------------------------------------------------------------- loop
    def step(self, now: float) -> list[Intent]:
        self._roll_day(now)
        intents: list[Intent] = []
        for ticker, state in list(self.markets.items()):
            state.sample(now)
            if state.book.ready:
                self.profiles[state.info.series].sample(state, now, self.cfg.depth_band_cents)
            if ticker in self.pending_orders:
                continue
            if state.book.ready and book_crossed(state.book):
                self.last_reason[ticker] = (
                    f"BOOK_CROSSED yes_bid={state.book.bid('yes')} no_bid={state.book.bid('no')}"
                )
                self.pending_signals.pop(ticker, None)
                if now - self._crossed_logged.get(ticker, 0.0) > 60:
                    self._crossed_logged[ticker] = now
                    logger.error(
                        "CROSSED ORDER BOOK | %s | yes_bid=%s no_bid=%s | not trading this market",
                        ticker, state.book.bid("yes"), state.book.bid("no"),
                    )
                continue
            position = self.positions.get(ticker)
            if position is not None:
                self.last_reason[ticker] = f"HOLDING {position.side.upper()} x{position.remaining}"
                intent = self._manage(position, state, now)
            else:
                intent = self._consider_entry(state, now)
            if intent is not None:
                self.pending_orders[ticker] = intent
                intents.append(intent)
        return intents

    def tick_rows(self, now: float) -> list[dict[str, Any]]:
        """One row per market for the data recorder."""
        rows = []
        for ticker, state in self.markets.items():
            if not state.book.ready:
                continue
            f = compute_features(state, now, self.cfg)
            stats = self.profiles[state.info.series].stats(now, self.cfg)
            position = self.positions.get(ticker)
            rows.append({
                "ts": round(now, 3),
                "series": state.info.series,
                "ticker": ticker,
                "seconds_left": round(f.seconds_left, 1),
                "yes_bid": f.yes_bid,
                "yes_ask": f.yes_ask,
                "spread": f.spread,
                "yes_bid_size": state.book.bid_size("yes"),
                "yes_ask_size": state.book.ask_size("yes"),
                "yes_depth": f.yes_depth,
                "no_depth": f.no_depth,
                "move_10s": _r(f.move_fast),
                "move_30s": _r(f.move),
                "move_60s": _r(f.move_slow),
                "up_cents_15s": _r(f.up_cents),
                "down_cents_15s": _r(f.down_cents),
                "yes_taker_30s": f.yes_flow,
                "no_taker_30s": f.no_flow,
                "tier": stats.tier,
                "trade_rate": round(stats.trade_rate, 1),
                "profile_spread": stats.spread,
                "profile_depth": stats.depth,
                "typical_move": round(stats.typical_move, 2),
                "position": f"{position.side}:{position.remaining}" if position else "",
                "reason": self.last_reason.get(ticker, ""),
            })
        return rows

    # ------------------------------------------------------------- entries
    def _consider_entry(self, state: TickerState, now: float) -> Intent | None:
        cfg = self.cfg
        ticker = state.info.ticker

        def reject(reason: str) -> None:
            self.last_reason[ticker] = reason
            self.pending_signals.pop(ticker, None)

        if not state.book.ready:
            return reject("BOOK_NOT_READY")
        if ticker in self.blocked:
            return reject("BLOCKED_EXISTING_POSITION")
        if now < self._paused_until:
            return reject("PAUSED_ORDER_ERRORS")
        if self.daily_net_cents <= -cfg.daily_loss_limit_cents:
            return reject("DAILY_LOSS_LIMIT")
        if len(self.positions) + self._pending_buys() >= cfg.max_open_positions:
            return reject("MAX_POSITIONS")
        if now < self.cooldown_until.get(ticker, 0.0):
            return reject("COOLDOWN")
        if self.entries.get(ticker, 0) >= cfg.max_entries_per_market:
            return reject("MAX_ENTRIES_THIS_MARKET")
        if state.age(now) < cfg.min_market_age_seconds:
            return reject("MARKET_TOO_NEW")
        if state.seconds_left(now) < cfg.min_seconds_left_entry:
            return reject("TOO_CLOSE_TO_EXPIRY")

        stats = self.profiles[state.info.series].stats(now, cfg)
        if stats.tier == WARMUP:
            return reject("PROFILING_MARKET")
        rules = cfg.tiers[stats.tier]
        if not rules.tradeable:
            return reject(f"WATCH_ONLY tier={stats.tier}")

        features = compute_features(state, now, cfg)
        signal, reason = evaluate(features, stats, rules, cfg)
        if signal is None:
            return reject(reason)

        pending = self.pending_signals.get(ticker)
        if pending is None or pending.side != signal.side:
            self.pending_signals[ticker] = PendingSignal(signal.side, now, signal.ask, now, 1)
            pending = self.pending_signals[ticker]
            self._emit("signal", {"ticker": ticker, "detail": signal.detail, "tier": stats.tier})
        elif now - pending.last_ts >= cfg.signal_confirm_seconds:
            pending.streak += 1
            pending.last_ts = now

        if signal.ask - pending.first_ask > cfg.max_chase_cents:
            self.pending_signals.pop(ticker, None)
            self.last_reason[ticker] = f"CHASED ask={signal.ask}c first={pending.first_ask}c"
            return None
        if pending.streak < cfg.signal_confirmations:
            self.last_reason[ticker] = f"SIGNAL_BUILDING {pending.streak}/{cfg.signal_confirmations}"
            return None

        self.pending_signals.pop(ticker, None)
        count = cfg.size_for(stats.tier)
        limit = min(99, signal.ask + cfg.entry_slippage_cents)
        self.last_reason[ticker] = f"ENTER {signal.detail}"
        intent = Intent(
            intent_id=next(self._ids),
            ticker=ticker,
            action="buy",
            side=signal.side,
            count=count,
            limit_price=limit,
            reason="ENTRY",
            exchange_index=state.info.exchange_index,
        )
        intent_context = {
            "features": features.to_dict(),
            "profile": {
                "tier": stats.tier,
                "trade_rate": round(stats.trade_rate, 1),
                "spread": stats.spread,
                "depth": stats.depth,
                "typical_move": round(stats.typical_move, 2),
            },
            "detail": signal.detail,
        }
        self._entry_context[intent.intent_id] = intent_context
        return intent

    def _pending_buys(self) -> int:
        return sum(1 for i in self.pending_orders.values() if i.action == "buy")

    # --------------------------------------------------------------- exits
    def _manage(self, pos: Position, state: TickerState, now: float) -> Intent | None:
        cfg = self.cfg
        if pos.awaiting_settlement:
            return None
        if now >= state.info.close_ts:
            pos.awaiting_settlement = True
            self._emit("awaiting_settlement", {"ticker": pos.ticker, "remaining": pos.remaining})
            logger.warning("HELD INTO CLOSE | %s %s x%d", pos.ticker, pos.side, pos.remaining)
            return None
        if not state.book.ready:
            return None

        bid = state.book.bid(pos.side)
        if bid is not None:
            pos.peak_bid = max(pos.peak_bid, bid)
            pos.trough_bid = min(pos.trough_bid, bid)
        entry = pos.entry_price

        def sell(limit: int, reason: str) -> Intent:
            pos.last_exit_reason = reason
            return Intent(
                intent_id=next(self._ids),
                ticker=pos.ticker,
                action="sell",
                side=pos.side,
                count=pos.remaining,
                limit_price=max(1, min(99, limit)),
                reason=reason,
                exchange_index=state.info.exchange_index,
            )

        if state.seconds_left(now) <= cfg.exit_before_close_seconds:
            if bid is None:
                return None
            return sell(bid - cfg.stop_slippage_cents, "PRE_CLOSE")

        target = round(entry + cfg.take_profit_cents)
        if bid is not None and bid >= target:
            return sell(target, "TAKE_PROFIT")

        # A vanished bid counts as adverse so a dead book can't freeze the stop.
        adverse = entry - (bid if bid is not None else 0)
        if bid is not None and adverse >= cfg.hard_stop_cents:
            return sell(bid - cfg.stop_slippage_cents, "HARD_STOP")
        if adverse >= cfg.stop_loss_cents:
            if now - pos.last_stop_check >= cfg.stop_confirm_seconds:
                pos.stop_streak += 1
                pos.last_stop_check = now
            if pos.stop_streak >= cfg.stop_confirmations and bid is not None:
                return sell(bid - cfg.stop_slippage_cents, "STOP_LOSS")
            return None
        pos.stop_streak = 0

        if (
            now - pos.entry_ts >= cfg.stall_seconds
            and bid is not None
            and bid < entry + cfg.stall_min_progress_cents
        ):
            return sell(bid - cfg.stop_slippage_cents, "STALL")
        return None

    def flatten_intents(self, reason: str = "SHUTDOWN") -> list[Intent]:
        """Exit orders for every open position that can still be sold."""
        intents = []
        for ticker, pos in self.positions.items():
            state = self.markets.get(ticker)
            if pos.awaiting_settlement or ticker in self.pending_orders or state is None:
                continue
            ok = state.book.ready and not book_crossed(state.book)
            bid = state.book.bid(pos.side) if ok else None
            if bid is None:
                continue
            pos.last_exit_reason = reason
            intent = Intent(
                intent_id=next(self._ids),
                ticker=ticker,
                action="sell",
                side=pos.side,
                count=pos.remaining,
                limit_price=max(1, bid - self.cfg.stop_slippage_cents),
                reason=reason,
                exchange_index=state.info.exchange_index,
            )
            self.pending_orders[ticker] = intent
            intents.append(intent)
        return intents

    # --------------------------------------------------------- fill events
    def on_fill(self, intent: Intent, fill: Fill, now: float) -> None:
        self.pending_orders.pop(intent.ticker, None)
        self._order_errors = 0
        context = self._entry_context.pop(intent.intent_id, None)
        state = self.markets.get(intent.ticker)

        if intent.action == "buy":
            self.entries[intent.ticker] = self.entries.get(intent.ticker, 0) + 1
            if fill.count <= 0:
                self.cooldown_until[intent.ticker] = now + 5.0
                self.last_reason[intent.ticker] = "ENTRY_NOT_FILLED"
                self._emit("entry_missed", {"ticker": intent.ticker, "limit": intent.limit_price})
                return
            ctx = context or {}
            pos = Position(
                ticker=intent.ticker,
                series=state.info.series if state else "",
                side=intent.side,
                tier=ctx.get("profile", {}).get("tier", ""),
                count=fill.count,
                entry_price=fill.avg_price,
                entry_fee=fill.fee_cents,
                entry_ts=now,
                features=ctx.get("features", {}),
                profile=ctx.get("profile", {}),
                peak_bid=0,
            )
            self.positions[intent.ticker] = pos
            logger.warning(
                "ENTRY %s | %s %s x%d @ %.1fc fee=%dc | %s",
                self.mode.upper(), intent.ticker, intent.side.upper(), fill.count,
                fill.avg_price, fill.fee_cents, ctx.get("detail", ""),
            )
            self._emit("entry", {
                "ticker": intent.ticker,
                "side": intent.side,
                "count": fill.count,
                "price": fill.avg_price,
                "fee": fill.fee_cents,
                "tier": pos.tier,
                "seconds_left": state.seconds_left(now) if state else None,
                "target": round(fill.avg_price + self.cfg.take_profit_cents),
                "stop": round(fill.avg_price - self.cfg.stop_loss_cents),
                "detail": ctx.get("detail", ""),
            })
            return

        pos = self.positions.get(intent.ticker)
        if pos is None or fill.count <= 0:
            return
        count = min(fill.count, pos.remaining)
        pos.exit_proceeds += fill.avg_price * count
        pos.exit_fees += fill.fee_cents
        pos.exited += count
        if pos.remaining <= 0:
            self._close(pos, intent.reason, now)

    def on_order_failed(self, intent: Intent, error: str, now: float) -> None:
        self.pending_orders.pop(intent.ticker, None)
        self._entry_context.pop(intent.intent_id, None)
        self._order_errors += 1
        logger.error("ORDER FAILED | %s %s %s | %s", intent.ticker, intent.action, intent.reason, error)
        self._emit("order_failed", {"ticker": intent.ticker, "action": intent.action, "error": error})
        if intent.action == "buy":
            self.cooldown_until[intent.ticker] = now + 30.0
        if self._order_errors >= self.cfg.max_order_errors:
            self._paused_until = now + self.cfg.error_pause_seconds
            logger.error("ENTRIES PAUSED %.0fs after %d order errors", self.cfg.error_pause_seconds, self._order_errors)

    def settle(self, ticker: str, result: str, now: float) -> None:
        pos = self.positions.get(ticker)
        if pos is None or result not in {"yes", "no"}:
            return
        price = 100.0 if result == pos.side else 0.0
        remaining = pos.remaining
        pos.exit_proceeds += price * remaining
        pos.exited += remaining
        self._close(pos, "SETTLEMENT_WIN" if price else "SETTLEMENT_LOSS", now)

    def _close(self, pos: Position, reason: str, now: float) -> None:
        self.positions.pop(pos.ticker, None)
        cost = pos.entry_price * pos.count
        gross = round(pos.exit_proceeds - cost)
        fees = pos.entry_fee + pos.exit_fees
        net = gross - fees
        exit_price = pos.exit_proceeds / pos.count if pos.count else 0.0

        self.stats.trades += 1
        self.stats.net_cents += net
        self.stats.fees_cents += fees
        if net > 0:
            self.stats.wins += 1
        else:
            self.stats.losses += 1
        self.daily_net_cents += net

        stopped = reason in {"STOP_LOSS", "HARD_STOP"}
        self.cooldown_until[pos.ticker] = now + (
            self.cfg.cooldown_after_stop_seconds if stopped else self.cfg.cooldown_after_exit_seconds
        )

        record = {
            "mode": self.mode,
            "series": pos.series,
            "ticker": pos.ticker,
            "tier": pos.tier,
            "side": pos.side,
            "count": pos.count,
            "entry_price": round(pos.entry_price, 2),
            "exit_price": round(exit_price, 2),
            "entry_fee": pos.entry_fee,
            "exit_fee": pos.exit_fees,
            "gross_cents": gross,
            "net_cents": net,
            "reason": reason,
            "entry_ts": datetime.fromtimestamp(pos.entry_ts, UTC).isoformat(),
            "exit_ts": datetime.fromtimestamp(now, UTC).isoformat(),
            "hold_seconds": round(now - pos.entry_ts, 1),
            "best_bid": pos.peak_bid,
            "worst_bid": pos.trough_bid if pos.trough_bid < 100 else None,
            "move_30s": pos.features.get("move"),
            "move_10s": pos.features.get("move_fast"),
            "flow_ratio": pos.features.get("flow_ratio"),
            "imbalance": pos.features.get("imbalance"),
            "spread": pos.features.get("spread"),
            "seconds_left_at_entry": _r(pos.features.get("seconds_left")),
            "trade_rate": pos.profile.get("trade_rate"),
            "typical_move": pos.profile.get("typical_move"),
            "daily_net_cents": self.daily_net_cents,
        }
        self.closed_trades.append(record)
        logger.warning(
            "EXIT %s | %s %s x%d %.1fc -> %.1fc | %s | net=%+dc (fees %dc) | day=%+dc",
            self.mode.upper(), pos.ticker, pos.side.upper(), pos.count, pos.entry_price,
            exit_price, reason, net, fees, self.daily_net_cents,
        )
        self._emit("exit", record)

    # ------------------------------------------------------------- helpers
    def _roll_day(self, now: float) -> None:
        today = datetime.fromtimestamp(now, UTC).astimezone(_CENTRAL).date()
        if today != self._day:
            self._day = today
            self.daily_net_cents = 0

    def summary(self) -> dict[str, Any]:
        s = self.stats
        return {
            "trades": s.trades,
            "wins": s.wins,
            "losses": s.losses,
            "win_rate": round(100 * s.wins / s.trades, 1) if s.trades else 0.0,
            "net_cents": s.net_cents,
            "fees_cents": s.fees_cents,
            "open_positions": len(self.positions),
        }


def _r(value: Any) -> Any:
    return None if value is None else round(float(value), 2)



# ==========================================================================
# BROKERS
# Order execution: paper (book-walking) and live (Kalshi IOC orders).
#
# Every order is immediate-or-cancel. Momentum entries need to be fast, and
# IOC exits avoid managing resting orders that could fill after the bot has
# already decided to leave.
# ==========================================================================
import logging
from typing import TYPE_CHECKING, Protocol

from kalshi_bot.scalping.book import BinaryOrderBook

if TYPE_CHECKING:
    from kalshi_bot.exchange.client import KalshiClient

logger = logging.getLogger(__name__)


class Broker(Protocol):
    async def execute(self, intent: Intent, book: BinaryOrderBook | None) -> Fill: ...


def simulate_ioc(intent: Intent, book: BinaryOrderBook, fee_rate: float = 0.07) -> Fill:
    """Fill against displayed liquidity only, never beyond the limit price."""
    if book_crossed(book):
        return Fill(0, 0.0, 0)
    if intent.action == "buy":
        levels = [(p, s) for p, s in ask_levels(book, intent.side) if p <= intent.limit_price]
    else:
        levels = [(p, s) for p, s in bid_levels(book, intent.side) if p >= intent.limit_price]
    remaining = intent.count
    taken: list[tuple[int, int]] = []
    for price, size in levels:
        if remaining <= 0:
            break
        qty = min(size, remaining)
        taken.append((price, qty))
        remaining -= qty
    filled = sum(q for _, q in taken)
    if filled == 0:
        return Fill(0, 0.0, 0)
    avg = sum(p * q for p, q in taken) / filled
    return Fill(filled, avg, taker_fee_for_levels(taken, fee_rate))


class PaperBroker:
    """Fills at the displayed book at decision time.

    This is optimistic about latency: a live IOC lands ~100-300ms later, when
    a fast-moving book may have moved. Treat paper results as an upper bound.
    """

    def __init__(self, fee_rate: float = 0.07) -> None:
        self.fee_rate = fee_rate

    async def execute(self, intent: Intent, book: BinaryOrderBook | None) -> Fill:
        if book is None or not book.ready:
            return Fill(0, 0.0, 0)
        return simulate_ioc(intent, book, self.fee_rate)


class LiveBroker:
    def __init__(self, client: KalshiClient, fee_rate: float = 0.07) -> None:
        self.client = client
        self.fee_rate = fee_rate

    async def execute(self, intent: Intent, book: BinaryOrderBook | None) -> Fill:
        from kalshi_bot.exchange.models import Action, OrderRequest, Side, TimeInForce

        side = Side.YES if intent.side == "yes" else Side.NO
        request = OrderRequest(
            ticker=intent.ticker,
            action=Action.BUY if intent.action == "buy" else Action.SELL,
            side=side,
            count=intent.count,
            yes_price=intent.limit_price if side is Side.YES else None,
            no_price=intent.limit_price if side is Side.NO else None,
            time_in_force=TimeInForce.IMMEDIATE_OR_CANCEL,
            exchange_index=intent.exchange_index,
        )
        order = await self.client.create_order(request)
        filled = int(order.fill_count or 0)
        if filled <= 0:
            return Fill(0, 0.0, 0)
        price = order.outcome_fill_price
        if price is None:
            # Fall back to the limit, which is the worst price IOC could fill at.
            price = intent.limit_price
            logger.warning("No fill price returned for %s; booking at limit %dc", intent.ticker, price)
        # Kalshi does not return the fee on the order response; estimate it with
        # the published taker formula. Reconcile against account fills if needed.
        return Fill(filled, float(price), taker_fee_cents(filled, int(round(price)), self.fee_rate))



# ==========================================================================
# RECORDER
# Market data and trade recording.
#
# Files written under the data directory (one set per UTC day):
#
# * ``raw/YYYY-MM-DD.jsonl.gz``    every WebSocket message plus market metadata
#                                  and settlement results; ``scalper-replay``
#                                  re-runs the strategy over these.
# * ``ticks/YYYY-MM-DD.csv.gz``    one row per market per second: prices, depth,
#                                  momentum, taker flow, liquidity tier.
# * ``scalper_trades.csv``         every closed scalp with entry features.
# * ``scalper_events.csv``         signals, entries, missed fills, order errors.
# * ``market_results.csv``         settlement result of every market seen.
# ==========================================================================
import csv
import gzip
import io
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def default_data_dir() -> Path:
    volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
    base = Path(volume) if volume else Path("data")
    return Path(os.getenv("SCALPER_DATA_DIR", str(base / "scalper")))


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")


class _DailyGzip:
    def __init__(self, folder: Path, suffix: str) -> None:
        self.folder = folder
        self.suffix = suffix
        self.day = ""
        self.handle: gzip.GzipFile | None = None
        self.text: io.TextIOWrapper | None = None

    def get(self, ts: float) -> tuple[io.TextIOWrapper, bool]:
        day = _day(ts)
        new_file = False
        if day != self.day or self.text is None:
            self.close()
            self.folder.mkdir(parents=True, exist_ok=True)
            path = self.folder / f"{day}{self.suffix}"
            new_file = not path.exists()
            self.handle = gzip.open(path, "ab")  # type: ignore[assignment]
            self.text = io.TextIOWrapper(self.handle, encoding="utf-8", newline="")  # type: ignore[arg-type]
            self.day = day
        return self.text, new_file

    def flush(self) -> None:
        if self.text is not None:
            self.text.flush()
            assert self.handle is not None
            self.handle.flush()

    def close(self) -> None:
        if self.text is not None:
            try:
                self.text.close()
            except Exception:
                pass
        self.text = None
        self.handle = None


class Recorder:
    TICK_FIELDS = [
        "ts", "series", "ticker", "seconds_left", "yes_bid", "yes_ask", "spread",
        "yes_bid_size", "yes_ask_size", "yes_depth", "no_depth", "move_10s",
        "move_30s", "move_60s", "up_cents_15s", "down_cents_15s", "yes_taker_30s",
        "no_taker_30s", "tier", "trade_rate", "profile_spread", "profile_depth",
        "typical_move", "position", "reason",
    ]
    TRADE_FIELDS = [
        "mode", "series", "ticker", "tier", "side", "count", "entry_price",
        "exit_price", "entry_fee", "exit_fee", "gross_cents", "net_cents", "reason",
        "entry_ts", "exit_ts", "hold_seconds", "best_bid", "worst_bid", "move_30s",
        "move_10s", "flow_ratio", "imbalance", "spread", "seconds_left_at_entry",
        "trade_rate", "typical_move", "daily_net_cents",
    ]

    def __init__(self, data_dir: Path | str | None = None, record_raw: bool = True) -> None:
        self.dir = Path(data_dir) if data_dir else default_data_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.record_raw = record_raw
        self._raw = _DailyGzip(self.dir / "raw", ".jsonl.gz")
        self._ticks = _DailyGzip(self.dir / "ticks", ".csv.gz")
        self._tick_writer: csv.DictWriter[str] | None = None
        self._tick_day = ""
        self._last_flush = time.monotonic()

    # -------------------------------------------------------------- raw
    def _raw_line(self, record: dict[str, Any], ts: float) -> None:
        if not self.record_raw:
            return
        handle, _ = self._raw.get(ts)
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def message(self, message: dict[str, Any], ts: float) -> None:
        self._raw_line({"rt": round(ts, 3), "m": message}, ts)

    def market(self, info: dict[str, Any], ts: float) -> None:
        self._raw_line({"rt": round(ts, 3), "market": info}, ts)

    def reset(self, ts: float) -> None:
        """Mark a reconnect so replay drops the stale book like the live bot did."""
        self._raw_line({"rt": round(ts, 3), "reset": 1}, ts)

    def result(self, ticker: str, series: str, result: str, ts: float) -> None:
        self._raw_line({"rt": round(ts, 3), "result": {"ticker": ticker, "result": result}}, ts)
        self._append_csv(
            "market_results.csv",
            ["recorded_at", "series", "ticker", "result"],
            {"recorded_at": datetime.fromtimestamp(ts, UTC).isoformat(), "series": series,
             "ticker": ticker, "result": result},
        )

    # ------------------------------------------------------------ ticks
    def ticks(self, rows: list[dict[str, Any]], ts: float) -> None:
        if not rows:
            return
        handle, new_file = self._ticks.get(ts)
        if self._tick_writer is None or self._tick_day != self._ticks.day:
            self._tick_writer = csv.DictWriter(handle, fieldnames=self.TICK_FIELDS, extrasaction="ignore")
            self._tick_day = self._ticks.day
            if new_file:
                self._tick_writer.writeheader()
        self._tick_writer.writerows(rows)

    # ----------------------------------------------------------- events
    def trade(self, record: dict[str, Any]) -> None:
        self._append_csv("scalper_trades.csv", self.TRADE_FIELDS, record)

    def event(self, kind: str, data: dict[str, Any], ts: float) -> None:
        self._append_csv(
            "scalper_events.csv",
            ["ts", "event", "ticker", "data"],
            {"ts": datetime.fromtimestamp(ts, UTC).isoformat(), "event": kind,
             "ticker": data.get("ticker", ""), "data": json.dumps(data, default=str)},
        )

    def _append_csv(self, name: str, fields: list[str], row: dict[str, Any]) -> None:
        path = self.dir / name
        new_file = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    # ------------------------------------------------------------ admin
    def maybe_flush(self, every: float = 2.0) -> None:
        now = time.monotonic()
        if now - self._last_flush >= every:
            self.flush()
            self._last_flush = now

    def flush(self) -> None:
        self._raw.flush()
        self._ticks.flush()

    def close(self) -> None:
        self._raw.close()
        self._ticks.close()


def read_raw(path: Path) -> list[dict[str, Any]]:
    """Read a raw day file, tolerating a truncated final gzip member."""
    records: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (EOFError, gzip.BadGzipFile, OSError):
        pass
    return records



# ==========================================================================
# RUNNER
# Live/paper runner: market discovery, WebSocket feed, step loop, settlement.
# ==========================================================================
import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import Any


logger = logging.getLogger(__name__)

SERIES_PATTERN = re.compile(r"KX[A-Z0-9]+15M")


def _ts(raw: str | None) -> float | None:
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()


class ScalperRunner:
    def __init__(
        self,
        client: Any,
        websocket: Any,
        core: ScalperCore,
        broker: Broker,
        recorder: Recorder | None,
        categories: tuple[str, ...] = ("Crypto", "Commodities"),
        series: tuple[str, ...] = (),
        step_seconds: float = 0.5,
        dashboard: bool = True,
    ) -> None:
        self.client = client
        self.websocket = websocket
        self.core = core
        self.broker = broker
        self.recorder = recorder
        self.categories = categories
        self.fixed_series = tuple(s.upper() for s in series)
        self.step_seconds = step_seconds
        self.dashboard = dashboard
        self.series: set[str] = set(self.fixed_series)
        self._series_refreshed = 0.0
        self._ws: Any = None
        self._ws_ids = 0
        self._subscribed: set[str] = set()
        self._awaiting_result: dict[str, tuple[str, float]] = {}
        self._stop = asyncio.Event()
        self._seq: dict[int, int] = {}
        self._last_reconnect = 0.0
        self._last_tick_write = 0.0
        self._last_status = 0.0
        self.discover_every = 5.0
        self.settle_every = 20.0
        core.listener = self._on_core_event

    # ------------------------------------------------------------ events
    def _on_core_event(self, kind: str, data: dict[str, Any]) -> None:
        now = time.time()
        if self.recorder is not None:
            if kind == "exit":
                self.recorder.trade(data)
            self.recorder.event(kind, data, now)
        if not self.dashboard:
            return
        try:
            from kalshi_bot import dashboard as dash

            mode = f"scalp-{self.core.mode}"
            if kind == "entry":
                dash.record_entry(
                    data["ticker"], data["side"], int(round(data["price"])), data["count"],
                    data.get("seconds_left"), data.get("stop"), data.get("target"), mode,
                )
            elif kind == "exit":
                dash.record_exit(
                    data["ticker"], data["side"], int(round(data["entry_price"])),
                    int(round(data["exit_price"])), data["reason"], data["count"],
                    data["net_cents"], self.core.stats.net_cents,
                )
            elif kind == "signal":
                dash.record_model_snapshot(ticker=data["ticker"], decision="SIGNAL", reason=data["detail"])
        except Exception:
            logger.exception("dashboard update failed")

    # -------------------------------------------------------- discovery
    async def _refresh_series(self, now: float) -> None:
        if self.fixed_series or now - self._series_refreshed < 600:
            return
        found: set[str] = set()
        for category in self.categories:
            try:
                data = await self.client._request("GET", "/series", params={"category": category})
            except Exception:
                logger.exception("series discovery failed | %s", category)
                continue
            for item in data.get("series", []):
                ticker = str(item.get("ticker", "")).upper()
                if SERIES_PATTERN.fullmatch(ticker):
                    found.add(ticker)
        if found:
            if found != self.series:
                logger.warning("SCALPER SERIES | %d | %s", len(found), sorted(found))
            self.series = found
            self._series_refreshed = now

    def _needs_refresh(self, series: str, now: float) -> bool:
        current = [s for s in self.core.markets.values() if s.info.series == series and s.info.close_ts > now]
        return not current or min(s.info.close_ts for s in current) - now < 30

    async def discover(self) -> None:
        semaphore = asyncio.Semaphore(4)
        full_refresh_at = 0.0
        while not self._stop.is_set():
            now = time.time()
            await self._refresh_series(now)
            full = now >= full_refresh_at
            if full:
                full_refresh_at = now + 120

            async def fetch(series: str) -> None:
                async with semaphore:
                    try:
                        markets = await self.client.get_markets(status="open", series_ticker=series, limit=20)
                    except Exception:
                        logger.exception("market poll failed | %s", series)
                        return
                for market in markets:
                    open_ts, close_ts = _ts(market.open_time), _ts(market.close_time)
                    if close_ts is None or close_ts <= time.time():
                        continue
                    if open_ts is not None and open_ts > time.time() + 5:
                        continue
                    self._add_market(MarketInfo(
                        ticker=market.ticker,
                        series=series,
                        open_ts=open_ts or time.time(),
                        close_ts=close_ts,
                        exchange_index=market.exchange_index,
                        floor_strike=market.floor_strike,
                    ))

            targets = [s for s in sorted(self.series) if full or self._needs_refresh(s, now)]
            await asyncio.gather(*(fetch(s) for s in targets))
            self._expire_markets(time.time())
            new = [t for t in self._active_tickers() if t not in self._subscribed]
            if new and self._ws is not None:
                try:
                    await self._subscribe(new)
                except Exception:
                    logger.exception("subscribe failed")
            await self._sleep(self.discover_every)

    def _add_market(self, info: MarketInfo) -> None:
        if info.ticker in self.core.markets:
            return
        now = time.time()
        self.core.add_market(info, now)
        if self.recorder is not None:
            self.recorder.market(info.to_dict(), now)
        logger.info("MARKET ADDED | %s closes in %.0fs", info.ticker, info.close_ts - now)

    def _expire_markets(self, now: float) -> None:
        for ticker, state in list(self.core.markets.items()):
            if state.info.close_ts <= now and ticker not in self._awaiting_result:
                self._awaiting_result[ticker] = (state.info.series, now)
            if state.info.close_ts + 5 <= now:
                self.core.drop_market(ticker)

    # -------------------------------------------------------- settlement
    async def settlements(self) -> None:
        while not self._stop.is_set():
            await self._sleep(self.settle_every)
            for ticker, (series, closed_at) in list(self._awaiting_result.items()):
                try:
                    market = await self.client.get_market(ticker)
                except Exception:
                    logger.exception("settlement poll failed | %s", ticker)
                    continue
                result = (market.result or "").lower()
                if result in {"yes", "no"}:
                    now = time.time()
                    self._awaiting_result.pop(ticker, None)
                    if self.recorder is not None:
                        self.recorder.result(ticker, series, result, now)
                    self.core.settle(ticker, result, now)
                    self.core.drop_market(ticker)
                elif time.time() - closed_at > 3 * 3600:
                    logger.warning("giving up on settlement result for %s", ticker)
                    self._awaiting_result.pop(ticker, None)

    # --------------------------------------------------------- websocket
    async def feed(self) -> None:
        import websockets

        backoff = 1.0
        while not self._stop.is_set():
            tickers = self._active_tickers()
            if not tickers:
                await self._sleep(1.0)
                continue
            try:
                headers = self.websocket._handshake_headers()
                async with websockets.connect(
                    self.websocket._ws_url, additional_headers=headers, ping_interval=10, max_size=None
                ) as ws:
                    self._ws = ws
                    self._subscribed = set()
                    self._seq = {}
                    self.core.reset_books()
                    if self.recorder is not None:
                        self.recorder.reset(time.time())
                    await self._subscribe(tickers)
                    logger.warning("WEBSOCKET CONNECTED | %d markets", len(tickers))
                    backoff = 1.0
                    async for raw in ws:
                        now = time.time()
                        message = json.loads(raw)
                        if self._sequence_gap(message):
                            logger.warning("ORDERBOOK SEQUENCE GAP | reconnecting")
                            break
                        kind = message.get("type")
                        if kind == "error":
                            logger.error("WEBSOCKET ERROR | %s", message)
                            continue
                        if kind in {"orderbook_snapshot", "orderbook_delta", "trade"}:
                            self.core.on_message(message, now)
                            if self.recorder is not None:
                                self.recorder.message(message, now)
                        new = [t for t in self._active_tickers() if t not in self._subscribed]
                        if new:
                            await self._subscribe(new)
                        if self._stop.is_set():
                            break
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("websocket failed; reconnecting in %.0fs", backoff)
            finally:
                self._ws = None
            if self._stop.is_set():
                break
            self.core.reset_books()
            if self.recorder is not None:
                self.recorder.reset(time.time())
            await self._sleep(backoff)
            backoff = min(30.0, backoff * 2)

    def _active_tickers(self) -> list[str]:
        now = time.time()
        return sorted(t for t, s in self.core.markets.items() if s.info.close_ts > now)

    async def _subscribe(self, tickers: list[str]) -> None:
        if self._ws is None or not tickers:
            return
        self._ws_ids += 1
        await self._ws.send(json.dumps({
            "id": self._ws_ids,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta", "trade"],
                "market_tickers": tickers,
                # NO-side levels must arrive as NO prices: the book derives
                # YES asks as 100 - best NO bid. Without this flag Kalshi sends
                # NO levels in YES terms and the book comes out crossed.
                "use_yes_price": False,
            },
        }))
        self._subscribed.update(tickers)

    def _sequence_gap(self, message: dict[str, Any]) -> bool:
        if message.get("type") not in {"orderbook_snapshot", "orderbook_delta"}:
            return False
        sid, seq = message.get("sid"), message.get("seq")
        if sid is None or seq is None:
            return False
        last = self._seq.get(sid)
        self._seq[sid] = int(seq)
        return last is not None and int(seq) != last + 1

    # -------------------------------------------------------------- step
    async def stepper(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            for intent in self.core.step(now):
                asyncio.create_task(self._execute(intent))
            if self.recorder is not None:
                if now - self._last_tick_write >= 1.0:
                    self.recorder.ticks(self.core.tick_rows(now), now)
                    self._last_tick_write = now
                self.recorder.maybe_flush()
            if now - self._last_status >= 60:
                self._last_status = now
                self._log_status(now)
            await asyncio.sleep(self.step_seconds)

    async def _execute(self, intent: Intent) -> None:
        state = self.core.markets.get(intent.ticker)
        book = state.book if state else None
        try:
            fill = await self.broker.execute(intent, book)
        except Exception as exc:  # live API error
            self.core.on_order_failed(intent, repr(exc), time.time())
            return
        self.core.on_fill(intent, fill if fill else Fill(0, 0.0, 0), time.time())

    def _log_status(self, now: float) -> None:
        tiers: dict[str, list[str]] = {}
        for series, profile in self.core.profiles.items():
            stats = profile.stats(now, self.core.cfg)
            tiers.setdefault(stats.tier, []).append(
                f"{series}(rate={stats.trade_rate:.0f}/m spread={stats.spread} depth={stats.depth})"
            )
        s = self.core.summary()
        logger.warning(
            "SCALPER %s | trades=%d win=%.1f%% net=%+dc today=%+dc open=%d | %s",
            self.core.mode.upper(), s["trades"], s["win_rate"], s["net_cents"],
            self.core.daily_net_cents, s["open_positions"],
            " ".join(f"{k}: {', '.join(v)}" for k, v in sorted(tiers.items())),
        )

    # --------------------------------------------------------------- run
    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            pass

    async def check_existing_positions(self) -> None:
        """Live safety: never trade a market where the account already holds a position."""
        try:
            positions = await self.client.get_positions(count_filter="position")
        except Exception:
            logger.exception("could not read existing positions")
            return
        for pos in positions:
            if pos.position != 0:
                self.core.blocked.add(pos.ticker)
        if self.core.blocked:
            logger.warning("EXISTING POSITIONS (not traded) | %s", sorted(self.core.blocked))

    async def flatten(self) -> None:
        """Best-effort exit of open positions on shutdown (e.g. a redeploy)."""
        for attempt in range(3):
            intents = self.core.flatten_intents()
            if not intents:
                break
            await asyncio.gather(*(self._execute(i) for i in intents))
            await asyncio.sleep(0.5)
        if self.core.positions:
            logger.error(
                "OPEN POSITIONS LEFT AT SHUTDOWN (close manually if live) | %s",
                {t: f"{p.side} x{p.remaining}" for t, p in self.core.positions.items()},
            )

    def _install_signal_handlers(self) -> None:
        import signal

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except (NotImplementedError, RuntimeError, ValueError):
                pass

    def _task_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("scalper task crashed; shutting down so the host restarts it", exc_info=exc)
            self._stop.set()

    async def run(self, max_seconds: float | None = None) -> None:
        self._install_signal_handlers()
        if self.core.mode == "live":
            await self.check_existing_positions()
        tasks = [
            asyncio.create_task(self.discover()),
            asyncio.create_task(self.feed()),
            asyncio.create_task(self.stepper()),
            asyncio.create_task(self.settlements()),
        ]
        for task in tasks:
            task.add_done_callback(self._task_done)
        try:
            if max_seconds:
                await self._sleep(max_seconds)
            else:
                await self._stop.wait()
        finally:
            self._stop.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await asyncio.wait_for(self.flatten(), timeout=15)
            except Exception:
                logger.exception("flatten on shutdown failed")
            if self.recorder is not None:
                self.recorder.close()
            logger.warning("SCALPER STOPPED | %s", self.core.summary())



# ==========================================================================
# REPLAY
# Re-run the scalper over recorded raw data with any settings.
#
# Use it to tune thresholds: record a few days in paper mode, then replay the
# same market data with a different take-profit, stop or liquidity rule and
# compare results. Fills use the recorded book at decision time, same as paper.
# ==========================================================================
from collections.abc import Iterable
from pathlib import Path



def replay_records(records: Iterable[dict], cfg: ScalperConfig, step: float = 0.5) -> ScalperCore:
    core = ScalperCore(cfg, mode="replay")
    next_step: float | None = None

    def run_step(now: float) -> None:
        for intent in core.step(now):
            state = core.markets.get(intent.ticker)
            if state is None or not state.book.ready:
                core.on_fill(intent, Fill(0, 0.0, 0), now)
                continue
            core.on_fill(intent, simulate_ioc(intent, state.book, cfg.taker_fee_rate), now)
        for ticker, state in list(core.markets.items()):
            if state.info.close_ts + 5 <= now:
                core.drop_market(ticker)

    for record in records:
        rt = float(record.get("rt", 0.0))
        if next_step is None:
            next_step = rt
        if rt - next_step > 10 * step:
            next_step = rt - step
        while next_step <= rt:
            run_step(next_step)
            next_step += step

        if "market" in record:
            core.add_market(MarketInfo.from_dict(record["market"]), rt)
        elif "m" in record:
            core.on_message(record["m"], rt)
        elif "reset" in record:
            core.reset_books()
        elif "result" in record:
            result = record["result"]
            core.settle(result["ticker"], str(result["result"]).lower(), rt)
            core.drop_market(result["ticker"])
    return core


def raw_files(data_dir: Path, days: list[str] | None = None) -> list[Path]:
    folder = data_dir / "raw"
    files = sorted(folder.glob("*.jsonl.gz"))
    if days:
        wanted = set(days)
        files = [f for f in files if f.name.split(".")[0] in wanted]
    return files


def replay_files(files: list[Path], cfg: ScalperConfig, step: float = 0.5) -> ScalperCore:
    def records() -> Iterable[dict]:
        for path in files:
            yield from read_raw(path)

    return replay_records(records(), cfg, step)



# ==========================================================================
# REPORT
# Summaries of recorded scalper data.
#
# * Scalp performance overall, by liquidity tier, series and exit reason.
# * Liquidity profile of each series (to tune the tier thresholds).
# * Hold rate: when one side first trades at >= 90c, how often does it win,
#   by series and time left.
# ==========================================================================
import csv
import gzip
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tick_rows(data_dir: Path):
    for path in sorted((data_dir / "ticks").glob("*.csv.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                yield from csv.DictReader(handle)
        except (EOFError, OSError):
            continue


def _num(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def trade_summary(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    trades = _rows(data_dir / "scalper_trades.csv")
    out: dict[str, list[dict[str, Any]]] = {}
    for key in ("mode", "tier", "series", "reason"):
        groups: dict[str, list[float]] = defaultdict(list)
        for t in trades:
            groups[t.get(key, "") or "-"].append(float(t["net_cents"]))
        table = []
        for name, nets in sorted(groups.items()):
            wins = [n for n in nets if n > 0]
            losses = [n for n in nets if n <= 0]
            table.append({
                key: name,
                "trades": len(nets),
                "win_rate": round(100 * len(wins) / len(nets), 1),
                "net_$": round(sum(nets) / 100, 2),
                "avg_win_c": round(statistics.mean(wins), 1) if wins else 0,
                "avg_loss_c": round(statistics.mean(losses), 1) if losses else 0,
            })
        out[key] = table
    return out


def liquidity_summary(data_dir: Path) -> list[dict[str, Any]]:
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    tiers: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in _tick_rows(data_dir):
        series = row["series"]
        for col in ("trade_rate", "spread", "yes_depth", "no_depth", "typical_move"):
            value = _num(row.get(col))
            if value is not None:
                per[series][col].append(value)
        tiers[series][row.get("tier", "")] += 1
    table = []
    for series, cols in sorted(per.items()):
        depth = [a + b for a, b in zip(cols["yes_depth"], cols["no_depth"], strict=False)]
        total = sum(tiers[series].values()) or 1
        table.append({
            "series": series,
            "trade_rate/min": round(statistics.median(cols["trade_rate"]), 1) if cols["trade_rate"] else None,
            "spread": statistics.median(cols["spread"]) if cols["spread"] else None,
            "depth": round(statistics.median(depth)) if depth else None,
            "typical_30s_move": round(statistics.median(cols["typical_move"]), 2) if cols["typical_move"] else None,
            "tier_mix": " ".join(f"{k}:{100 * v // total}%" for k, v in sorted(tiers[series].items())),
        })
    return table


def hold_rate_at(data_dir: Path, level: int = 90) -> list[dict[str, Any]]:
    results = {r["ticker"]: r["result"] for r in _rows(data_dir / "market_results.csv")}
    first_touch: dict[str, tuple[str, str, float]] = {}
    for row in _tick_rows(data_dir):
        ticker = row["ticker"]
        if ticker in first_touch or ticker not in results:
            continue
        ask, bid = _num(row.get("yes_ask")), _num(row.get("yes_bid"))
        left = _num(row.get("seconds_left"))
        if ask is None or bid is None or left is None or left <= 0:
            continue
        if ask >= level:
            first_touch[ticker] = (row["series"], "yes", left)
        elif 100 - bid >= level:
            first_touch[ticker] = (row["series"], "no", left)

    buckets = [(0, 60, "<1m"), (60, 180, "1-3m"), (180, 300, "3-5m"), (300, 600, "5-10m"), (600, 10**6, "10m+")]
    stats: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for ticker, (series, side, left) in first_touch.items():
        label = next(lbl for lo, hi, lbl in buckets if lo <= left < hi)
        held = int(results[ticker] == side)
        for key in ((series, label), ("ALL", label)):
            stats[key][0] += 1
            stats[key][1] += held
    order = {lbl: i for i, (_, _, lbl) in enumerate(buckets)}
    return [
        {"series": s, "time_left": lbl, "markets": n, "held_%": round(100 * h / n, 1)}
        for (s, lbl), (n, h) in sorted(stats.items(), key=lambda kv: (kv[0][0] != "ALL", kv[0][0], order[kv[0][1]]))
    ]


# ==========================================================================
# ENTRY POINTS
# ==========================================================================


def _env(name: str, default: str) -> str:
    import os

    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    return _env(name, "true" if default else "false").lower() in {"1", "true", "yes", "on"}


def config_from_env(**overrides: Any) -> ScalperConfig:
    contracts = int(overrides.get("contracts") or _env("SCALPER_CONTRACTS", "5"))
    take_profit = int(overrides.get("take_profit") or _env("SCALPER_TAKE_PROFIT", "10"))
    stop_loss = int(overrides.get("stop_loss") or _env("SCALPER_STOP_LOSS", "5"))
    daily_loss = float(overrides.get("daily_loss") or _env("SCALPER_DAILY_LOSS", "10"))
    max_positions = int(overrides.get("max_positions") or _env("SCALPER_MAX_POSITIONS", "4"))
    trade_thin = overrides.get("trade_thin")
    if trade_thin is None:
        trade_thin = _env_bool("SCALPER_TRADE_THIN")

    cfg = ScalperConfig(
        contracts=max(1, contracts),
        take_profit_cents=take_profit,
        stop_loss_cents=stop_loss,
        hard_stop_cents=max(stop_loss + 3, 8),
        daily_loss_limit_cents=int(daily_loss * 100),
        max_open_positions=max(1, max_positions),
    )
    cfg.max_entry_price = min(cfg.max_entry_price, 97 - take_profit)
    if trade_thin:
        cfg.tiers[THIN].tradeable = True
    return cfg


class BtcOrderBookScalpingEngine:
    """Kept under the old name so the existing ``btc-scalper`` command runs it.

    The CLI always passes ``dry_run=True``; live trading is switched on only by
    ``KALSHI_SCALPER_LIVE=true`` in the environment.
    """

    def __init__(
        self,
        client: Any,
        websocket: Any,
        config: ScalperConfig | None = None,
        dry_run: bool = True,
        series_ticker: str | None = None,
    ) -> None:
        self.client = client
        self.websocket = websocket
        self.config = config or config_from_env()
        self.live = _env_bool("KALSHI_SCALPER_LIVE")
        series = _env("SCALPER_SERIES", series_ticker or "")
        self.series = tuple(s.strip().upper() for s in series.split(",") if s.strip())
        self.categories = tuple(
            c.strip() for c in _env("SCALPER_CATEGORIES", "Crypto,Commodities").split(",") if c.strip()
        )

    async def run(self, max_events: int | None = None) -> None:
        """Run until stopped. ``max_events`` (the old --events option) is treated as seconds."""
        try:
            from kalshi_bot.dashboard import start_dashboard

            start_dashboard()
        except Exception:
            logger.exception("dashboard not started")

        if not getattr(self.client, "authenticated", True):
            raise RuntimeError("Kalshi API credentials are required (the order-book WebSocket is authenticated)")

        mode = "live" if self.live else "paper"
        if self.live:
            balance = await self.client.get_balance()
            logger.warning("LIVE SCALPER | real orders enabled | balance $%.2f", balance.balance / 100)
        logger.warning(
            "MOMENTUM SCALPER %s | contracts=%d tp=%dc stop=%dc series=%s",
            mode.upper(), self.config.contracts, self.config.take_profit_cents,
            self.config.stop_loss_cents, ",".join(self.series) or "all 15m",
        )
        core = ScalperCore(self.config, mode=mode)
        broker: Broker = (
            LiveBroker(self.client, self.config.taker_fee_rate)
            if self.live
            else PaperBroker(self.config.taker_fee_rate)
        )
        runner = ScalperRunner(
            client=self.client,
            websocket=self.websocket,
            core=core,
            broker=broker,
            recorder=Recorder(_env("SCALPER_DATA_DIR", "") or None, record_raw=_env_bool("SCALPER_RECORD_RAW", True)),
            categories=self.categories,
            series=self.series,
        )
        await runner.run(max_seconds=float(max_events) if max_events else None)


def _print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n== {title} ==")
    if not rows:
        print("  (no data yet)")
        return
    keys = list(rows[0])
    widths = {k: max(len(str(k)), *(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    print("  " + "  ".join(str(k).ljust(widths[k]) for k in keys))
    for row in rows:
        print("  " + "  ".join(str(row.get(k, "")).ljust(widths[k]) for k in keys))


def main(argv: list[str] | None = None) -> None:
    import argparse
    from collections import defaultdict

    parser = argparse.ArgumentParser(prog="python -m kalshi_bot.scalping.engine")
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="Summarise recorded trades, liquidity and hold rates")
    rep.add_argument("--data-dir", default="")
    rep.add_argument("--level", type=int, default=90)
    rpl = sub.add_parser("replay", help="Re-run recorded data with different settings")
    rpl.add_argument("--data-dir", default="")
    rpl.add_argument("--days", default="", help="Comma-separated UTC days, YYYY-MM-DD")
    rpl.add_argument("--contracts", type=int)
    rpl.add_argument("--take-profit", type=int)
    rpl.add_argument("--stop-loss", type=int)
    rpl.add_argument("--daily-loss", type=float)
    rpl.add_argument("--max-positions", type=int)
    rpl.add_argument("--trade-thin", action="store_true", default=None)
    args = parser.parse_args(argv)

    folder = Path(args.data_dir) if args.data_dir else default_data_dir()
    if args.command == "report":
        for key, rows in trade_summary(folder).items():
            _print_table(f"Scalps by {key}", rows)
        _print_table("Market liquidity (medians)", liquidity_summary(folder))
        _print_table(f"Hold rate after first touch of {args.level}c", hold_rate_at(folder, args.level))
        return

    files = raw_files(folder, [d.strip() for d in args.days.split(",") if d.strip()] or None)
    if not files:
        raise SystemExit(f"No raw recordings in {folder / 'raw'}")
    cfg = config_from_env(
        contracts=args.contracts, take_profit=args.take_profit, stop_loss=args.stop_loss,
        daily_loss=args.daily_loss, max_positions=args.max_positions, trade_thin=args.trade_thin,
    )
    logging.basicConfig(level=logging.ERROR)
    core = replay_files(files, cfg)
    _print_table("Replay summary", [core.summary()])
    for key in ("tier", "series", "reason"):
        groups: dict[str, list[int]] = defaultdict(list)
        for trade in core.closed_trades:
            groups[str(trade[key])].append(int(trade["net_cents"]))
        _print_table(f"By {key}", [
            {key: k, "trades": len(v), "win_rate": round(100 * sum(n > 0 for n in v) / len(v), 1),
             "net_$": round(sum(v) / 100, 2)}
            for k, v in sorted(groups.items())
        ])


if __name__ == "__main__":
    main()
