"""Event-level weather strategy: compare model probabilities with the book.

For every bracket in an event:

* **Entry.** Buy YES (or NO) when the probability, shrunk toward the market mid
  by ``confidence``, beats the ask plus the taker fee by ``min_edge``. Size is
  fractional Kelly, capped by per-market, per-event and per-day budgets and by
  the displayed ask size.
* **Exit.** If held contracts now bid above what the model thinks they're
  worth (by ``exit_margin`` after fees), sell them back (reduce-only).
* **Partition arbitrage.** The brackets partition the outcome, so exactly one
  YES pays 100c. If all YES asks plus fees cost less than 100c, buy one of
  each. If all NO asks plus fees cost less than ``(N-1) * 100c``, buy NO on
  every bracket.

Orders are immediate-or-cancel limits at the touch, so nothing rests on the
book.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from kalshi_bot.exchange.models import Action, OrderRequest, OrderType, Side, TimeInForce
from kalshi_bot.risk.manager import RiskManager
from kalshi_bot.weather.brackets import WeatherEvent
from kalshi_bot.weather.fees import fee_per_contract
from kalshi_bot.weather.settings import WeatherSettings

P_CLIP = (0.005, 0.995)


@dataclass(slots=True)
class Decision:
    order: OrderRequest
    reason: str
    kind: str  # "entry" | "exit" | "arb"
    p_model: float | None = None
    p_used: float | None = None
    edge: float | None = None
    price: int = 0


@dataclass(slots=True)
class Budget:
    """Cents already committed (cost basis of open positions)."""

    by_market: dict[str, int] = field(default_factory=dict)
    by_event: dict[str, int] = field(default_factory=dict)
    by_day: dict[str, int] = field(default_factory=dict)

    def remaining(self, ws: WeatherSettings, ticker: str, event: str, day: str) -> int:
        return min(
            ws.max_cost_per_market_cents - self.by_market.get(ticker, 0),
            ws.max_cost_per_event_cents - self.by_event.get(event, 0),
            ws.max_cost_per_day_cents - self.by_day.get(day, 0),
        )

    def add(self, ticker: str, event: str, day: str, cents: int) -> None:
        self.by_market[ticker] = self.by_market.get(ticker, 0) + cents
        self.by_event[event] = self.by_event.get(event, 0) + cents
        self.by_day[day] = self.by_day.get(day, 0) + cents


def _clip(p: float) -> float:
    return min(P_CLIP[1], max(P_CLIP[0], p))


def _order(
    ticker: str, side: Side, action: Action, count: int, price: int, idx: int | None
) -> OrderRequest:
    return OrderRequest(
        ticker=ticker,
        action=action,
        side=side,
        count=count,
        type=OrderType.LIMIT,
        yes_price=price if side is Side.YES else None,
        no_price=price if side is Side.NO else None,
        time_in_force=TimeInForce.IMMEDIATE_OR_CANCEL,
        exchange_index=idx,
    )


def _no_ask(yes_bid: int | None, no_ask: int | None) -> int | None:
    if no_ask is not None and no_ask > 0:
        return no_ask
    return None if yes_bid is None or yes_bid <= 0 else 100 - yes_bid


def _no_bid(yes_ask: int | None, no_bid: int | None) -> int | None:
    if no_bid is not None and no_bid > 0:
        return no_bid
    return None if yes_ask is None or yes_ask >= 100 else 100 - yes_ask


class WeatherStrategy:
    def __init__(self, settings: WeatherSettings, risk: RiskManager) -> None:
        self.ws = settings
        self.risk = risk
        self.last_entry: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    def decide(
        self,
        event: WeatherEvent,
        probs: dict[str, float],
        positions: dict[str, int],
        budget: Budget,
        bankroll_cents: int,
        now: datetime,
        intraday: bool = False,
    ) -> list[Decision]:
        ws = self.ws
        day = event.day.isoformat()
        weight = ws.intraday_confidence if intraday else ws.confidence
        out: list[Decision] = []

        if ws.partition_arb and event.valid:
            out.extend(self._partition_arb(event, budget, day))
            if out:
                return out

        for bm in event.brackets:
            m = bm.market
            p_model = probs.get(m.ticker)
            if p_model is None:
                continue
            p = _clip(p_model)
            mid: float | None = None
            if m.yes_bid and m.yes_ask and m.yes_ask > m.yes_bid:
                mid = (m.yes_bid + m.yes_ask) / 200.0
            p_used = weight * p + (1 - weight) * mid if mid is not None else p
            pos = positions.get(m.ticker, 0)

            # ---- exits ------------------------------------------------
            if ws.exit_on_reversal and pos != 0:
                ex = self._exit(
                    m.ticker, pos, p_used, p, m.yes_ask, m.yes_bid, m.no_bid, m.exchange_index
                )
                if ex is not None:
                    out.append(ex)
                    # Don't flip straight back in after selling.
                    self.last_entry[m.ticker] = now
                    continue

            # ---- entries ----------------------------------------------
            last = self.last_entry.get(m.ticker)
            if last is not None and (now - last).total_seconds() < ws.reentry_cooldown_s:
                continue

            cands: list[tuple[float, Side, int, float, int | None]] = []
            if m.yes_ask and pos >= 0 and ws.min_price <= m.yes_ask <= ws.max_price:
                edge = p_used - m.yes_ask / 100 - fee_per_contract(m.yes_ask, ws.fee_rate)
                cands.append((edge, Side.YES, m.yes_ask, p_used, m.yes_ask_size))
            no_ask = _no_ask(m.yes_bid, m.no_ask)
            if no_ask and pos <= 0 and ws.min_price <= no_ask <= ws.max_price:
                edge = (1 - p_used) - no_ask / 100 - fee_per_contract(no_ask, ws.fee_rate)
                # Displayed NO-ask depth == YES-bid depth; not in the market
                # payload, so leave uncapped here (risk caps still apply).
                cands.append((edge, Side.NO, no_ask, 1 - p_used, None))
            if not cands:
                continue
            edge, side, price, win_p, depth = max(cands, key=lambda c: c[0])
            if edge < ws.min_edge:
                continue

            count = self.risk.kelly_size(win_p, price, bankroll_cents)
            count = min(count, ws.max_contracts_per_order)
            if depth:
                count = min(count, depth)
            room = budget.remaining(ws, m.ticker, event.event_ticker, day)
            count = min(count, max(0, room) // price)
            if count <= 0:
                continue
            budget.add(m.ticker, event.event_ticker, day, count * price)
            self.last_entry[m.ticker] = now
            out.append(
                Decision(
                    order=_order(m.ticker, side, Action.BUY, count, price, m.exchange_index),
                    reason=(
                        f"{bm.bracket.label()} model={p:.3f} used={p_used:.3f} "
                        f"{side.value}@{price}c edge={edge:+.3f}"
                    ),
                    kind="entry",
                    p_model=p,
                    p_used=p_used,
                    edge=edge,
                    price=price,
                )
            )
        return out

    # ------------------------------------------------------------------
    def _exit(
        self,
        ticker: str,
        pos: int,
        p_used: float,
        p_model: float,
        yes_ask: int | None,
        yes_bid: int | None,
        no_bid: int | None,
        idx: int | None,
    ) -> Decision | None:
        ws = self.ws
        if pos > 0 and yes_bid:
            hurdle = yes_bid / 100 - fee_per_contract(yes_bid, ws.fee_rate) - ws.exit_margin
            if p_used < hurdle:
                return Decision(
                    order=_order(ticker, Side.YES, Action.SELL, pos, yes_bid, idx),
                    reason=f"exit YES: worth {p_used:.3f} < bid {yes_bid}c",
                    kind="exit",
                    p_model=p_model,
                    p_used=p_used,
                    edge=hurdle - p_used,
                    price=yes_bid,
                )
        if pos < 0:
            nb = _no_bid(yes_ask, no_bid)
            if nb:
                hurdle = nb / 100 - fee_per_contract(nb, ws.fee_rate) - ws.exit_margin
                if (1 - p_used) < hurdle:
                    return Decision(
                        order=_order(ticker, Side.NO, Action.SELL, -pos, nb, idx),
                        reason=f"exit NO: worth {1 - p_used:.3f} < bid {nb}c",
                        kind="exit",
                        p_model=p_model,
                        p_used=p_used,
                        edge=hurdle - (1 - p_used),
                        price=nb,
                    )
        return None

    def _partition_arb(self, event: WeatherEvent, budget: Budget, day: str) -> list[Decision]:
        ws = self.ws
        mkts = [b.market for b in event.brackets]
        n = len(mkts)
        if n < 2:
            return []

        # Buy every YES: pays exactly 100c.
        asks = [m.yes_ask for m in mkts]
        if all(a is not None and 0 < a < 100 for a in asks):
            cost = sum(a / 100 + fee_per_contract(a, ws.fee_rate) for a in asks if a)
            if cost < 1.0 - 0.01:
                sizes = [m.yes_ask_size or 1 for m in mkts]
                set_cost = sum(a for a in asks if a)
                room = min(budget.remaining(ws, m.ticker, event.event_ticker, day) for m in mkts)
                count = min(min(sizes), ws.max_contracts_per_order, max(0, room) // set_cost)
                if count > 0:
                    for m, a in zip(mkts, asks, strict=True):
                        budget.add(m.ticker, event.event_ticker, day, count * (a or 0))
                    return [
                        Decision(
                            order=_order(
                                m.ticker, Side.YES, Action.BUY, count, a, m.exchange_index
                            ),
                            reason=f"partition arb YES set cost {cost:.3f}",
                            kind="arb",
                            edge=1.0 - cost,
                            price=a,
                        )
                        for m, a in zip(mkts, asks, strict=True)
                        if a is not None
                    ]

        # Buy every NO: pays exactly (n-1) * 100c.
        nos = [_no_ask(m.yes_bid, m.no_ask) for m in mkts]
        if all(x is not None and 0 < x < 100 for x in nos):
            cost = sum(x / 100 + fee_per_contract(x, ws.fee_rate) for x in nos if x)
            if cost < (n - 1) - 0.01:
                set_cost = sum(x for x in nos if x)
                room = min(budget.remaining(ws, m.ticker, event.event_ticker, day) for m in mkts)
                count = min(ws.max_contracts_per_order, max(0, room) // set_cost, 5)
                if count > 0:
                    for m, x in zip(mkts, nos, strict=True):
                        budget.add(m.ticker, event.event_ticker, day, count * (x or 0))
                    return [
                        Decision(
                            order=_order(m.ticker, Side.NO, Action.BUY, count, x, m.exchange_index),
                            reason=f"partition arb NO set cost {cost:.3f} vs {n - 1}",
                            kind="arb",
                            edge=(n - 1) - cost,
                            price=x,
                        )
                        for m, x in zip(mkts, nos, strict=True)
                        if x is not None
                    ]
        return []
