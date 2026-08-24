# BTC 15-minute order-book scalper

This build preserves the existing BNB strategy and adds a separate,
queue-aware BTC order-book recorder and paper scalper.

## Run on Railway or locally

Use this start command:

```bash
python -m kalshi_bot.cli btc-scalper
```

Do **not** append `--live`. Live mode is deliberately locked in this first
build. The engine needs authenticated Kalshi WebSocket access so it can receive
full order-book snapshots and deltas, but it does not submit orders.

The process writes `btc_orderbook_data.csv`. It records top-of-book state,
spread, depth, simulated maker fills, inventory, and realized paper P&L.

## Why fills are conservative

A simulated order starts behind all displayed contracts already resting at its
price. It is not counted as filled merely because price touches the quote.
Negative book deltas must first consume the displayed queue ahead of it.

This is still an estimate: book deltas do not perfectly distinguish trades
from cancellations. The collected data is meant to measure whether the idea is
worth implementing live, not to manufacture an attractive backtest.

## Current safeguards

- Separate from `coby_strategy`; the BNB bot is unchanged.
- Only quotes when the spread is at least 3 cents.
- Requires at least 20 contracts on both best bids.
- Uses 10-contract paper quotes and caps inventory at 20.
- Begins with 12 minutes remaining and stops quoting at 2 minutes.
- Targets a 2-cent maker scalp.
- Cancels stale quotes every 5 seconds.
- Refuses `--live`.

## Next gate

Collect at least several hundred complete BTC 15-minute markets. Live support
should only be added after reviewing fill rate, one-sided inventory duration,
worst adverse move after fills, exit time, and net results after the exact
current KXBTC15M fee schedule.
