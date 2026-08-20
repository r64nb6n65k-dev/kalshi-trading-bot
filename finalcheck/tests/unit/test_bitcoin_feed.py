from __future__ import annotations

from kalshi_bot.data.bitcoin import BitcoinPriceFeed


def test_parses_coinbase_advanced_trade_message() -> None:
    feed = BitcoinPriceFeed("wss://example.invalid")
    feed._handle_message(
        {
            "channel": "market_trades",
            "events": [
                {
                    "type": "update",
                    "trades": [
                        {
                            "trade_id": "1",
                            "product_id": "BTC-USD",
                            "price": "69588.25",
                            "size": "0.125",
                            "side": "BUY",
                            "time": "2026-08-20T01:00:00.000Z",
                        }
                    ],
                }
            ],
        }
    )

    ticks = feed.snapshot()
    assert len(ticks) == 1
    assert ticks[0].price == 69_588.25
    assert ticks[0].size == 0.125
    assert ticks[0].source == "COINBASE_PROXY"


def test_aggregates_trades_into_one_volume_weighted_second() -> None:
    feed = BitcoinPriceFeed("wss://example.invalid")
    message = {
        "channel": "market_trades",
        "events": [
            {
                "type": "update",
                "trades": [
                    {
                        "product_id": "BTC-USD",
                        "price": "100",
                        "size": "1",
                        "time": "2026-08-20T01:00:00.100Z",
                    },
                    {
                        "product_id": "BTC-USD",
                        "price": "110",
                        "size": "3",
                        "time": "2026-08-20T01:00:00.900Z",
                    },
                ],
            }
        ],
    }

    feed._handle_message(message)

    ticks = feed.snapshot()
    assert len(ticks) == 1
    assert ticks[0].price == 107.5
    assert ticks[0].size == 4.0
