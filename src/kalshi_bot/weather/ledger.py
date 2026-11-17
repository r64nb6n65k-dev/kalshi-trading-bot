"""Paper-trading ledger persisted to JSON, with settlement and P&L.

Positions use the repo convention: positive = YES contracts, negative = NO.
Cost basis is tracked in cents, so the per-market, per-event and per-day
budgets work the same way in paper and live mode.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from kalshi_bot.exchange.models import Action, Position, Side
from kalshi_bot.weather.brackets import parse_event_date
from kalshi_bot.weather.strategy import Budget


def event_of(ticker: str) -> str:
    return ticker.rsplit("-", 1)[0]


def day_of(ticker: str) -> str:
    d = parse_event_date(ticker)
    return d.isoformat() if d else "unknown"


@dataclass(slots=True)
class PaperPosition:
    ticker: str
    count: int = 0  # signed
    cost_cents: int = 0  # cost basis of the open contracts
    fees_cents: int = 0


@dataclass(slots=True)
class PaperLedger:
    path: Path
    starting_cents: int
    positions: dict[str, PaperPosition] = field(default_factory=dict)
    realized_cents: int = 0
    fees_cents: int = 0
    history: list[dict[str, object]] = field(default_factory=list)

    # --- persistence -----------------------------------------------------
    @classmethod
    def load(cls, path: Path, starting_cents: int) -> PaperLedger:
        led = cls(path=path, starting_cents=starting_cents)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            led.starting_cents = int(raw.get("starting_cents", starting_cents))
            led.realized_cents = int(raw.get("realized_cents", 0))
            led.fees_cents = int(raw.get("fees_cents", 0))
            led.history = list(raw.get("history", []))
            for t, p in (raw.get("positions") or {}).items():
                led.positions[t] = PaperPosition(**p)
        return led

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "starting_cents": self.starting_cents,
            "realized_cents": self.realized_cents,
            "fees_cents": self.fees_cents,
            "positions": {t: asdict(p) for t, p in self.positions.items()},
            "history": self.history[-5000:],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    # --- trading ---------------------------------------------------------
    def apply_fill(
        self, ticker: str, side: Side, action: Action, count: int, price: int, fee: int
    ) -> None:
        pos = self.positions.setdefault(ticker, PaperPosition(ticker=ticker))
        sign = 1 if side is Side.YES else -1
        if action is Action.BUY:
            pos.count += sign * count
            pos.cost_cents += count * price
        else:
            held = abs(pos.count)
            count = min(count, held)
            if count == 0:
                return
            basis = round(pos.cost_cents * count / held)
            self.realized_cents += count * price - basis
            pos.cost_cents -= basis
            pos.count -= sign * count
        pos.fees_cents += fee
        self.fees_cents += fee
        self.realized_cents -= fee
        self.history.append(
            {
                "t": datetime.now(UTC).isoformat(timespec="seconds"),
                "type": "fill",
                "ticker": ticker,
                "side": side.value,
                "action": action.value,
                "count": count,
                "price": price,
                "fee": fee,
            }
        )
        if pos.count == 0:
            self.positions.pop(ticker, None)
        self.save()

    def settle(self, ticker: str, result: str) -> int:
        """Settle at 100/0. Returns P&L in cents."""
        pos = self.positions.pop(ticker, None)
        if pos is None:
            return 0
        yes_won = result.lower() == "yes"
        won = (pos.count > 0 and yes_won) or (pos.count < 0 and not yes_won)
        payout = abs(pos.count) * 100 if won else 0
        pnl = payout - pos.cost_cents
        self.realized_cents += pnl
        self.history.append(
            {
                "t": datetime.now(UTC).isoformat(timespec="seconds"),
                "type": "settle",
                "ticker": ticker,
                "result": result,
                "count": pos.count,
                "pnl": pnl,
            }
        )
        self.save()
        return pnl

    # --- views -----------------------------------------------------------
    def signed_positions(self) -> dict[str, int]:
        return {t: p.count for t, p in self.positions.items()}

    def open_cost_cents(self) -> int:
        return sum(p.cost_cents for p in self.positions.values())

    def bankroll_cents(self) -> int:
        return self.starting_cents + self.realized_cents - self.open_cost_cents()

    def budget(self) -> Budget:
        return budget_from_costs({t: p.cost_cents for t, p in self.positions.items()})


def budget_from_costs(costs: dict[str, int]) -> Budget:
    b = Budget()
    for ticker, cents in costs.items():
        if cents > 0:
            b.add(ticker, event_of(ticker), day_of(ticker), cents)
    return b


def budget_from_live(positions: list[Position]) -> Budget:
    return budget_from_costs({p.ticker: abs(p.market_exposure) for p in positions if p.position})
