from kalshi_bot.data.bnb import BnbPriceFeed


def test_bnb_feed_parses_coinbase_market_trade() -> None:
    feed = BnbPriceFeed("wss://example.invalid", product_id="BNB-USD")
    feed._handle_message(
        {
            "channel": "market_trades",
            "events": [
                {
                    "trades": [
                        {
                            "product_id": "BNB-USD",
                            "price": "694.25",
                            "size": "2.5",
                            "time": "2026-08-22T12:00:00.123Z",
                        }
                    ]
                }
            ],
        }
    )
    ticks = feed.snapshot()
    assert len(ticks) == 1
    assert ticks[0].price == 694.25
    assert ticks[0].size == 2.5
    assert ticks[0].source == "COINBASE_BNBUSD_CONSTITUENT"


def test_bnb_feed_ignores_other_products() -> None:
    feed = BnbPriceFeed("wss://example.invalid", product_id="BNB-USD")
    feed._handle_message(
        {
            "channel": "market_trades",
            "events": [
                {
                    "trades": [
                        {
                            "product_id": "BTC-USD",
                            "price": "70000",
                            "size": "0.1",
                            "time": "2026-08-22T12:00:00.123Z",
                        }
                    ]
                }
            ],
        }
    )
    assert feed.snapshot() == ()


def test_bnb_feed_ignores_non_trade_message() -> None:
    feed = BnbPriceFeed("wss://example.invalid")
    feed._handle_message({"channel": "heartbeats"})
    assert feed.snapshot() == ()
