from kalshi_bot.scalping.book import BinaryOrderBook
from kalshi_bot.scalping.paper import QueuePaperBroker


def snapshot() -> dict:
    return {
        "type": "orderbook_snapshot",
        "seq": 1,
        "msg": {
            "market_ticker": "KXBTC15M-TEST",
            "yes_dollars_fp": [["0.40", "100.00"], ["0.41", "25.00"]],
            "no_dollars_fp": [["0.56", "30.00"], ["0.55", "80.00"]],
        },
    }


def test_binary_book_derives_asks_from_opposite_bids() -> None:
    book = BinaryOrderBook("KXBTC15M-TEST")
    book.apply(snapshot())
    assert book.yes_bid == 41
    assert book.yes_ask == 44
    assert book.bid("no") == 56
    assert book.ask("no") == 59
    assert book.spread == 3
    assert book.bid_size("yes") == 25
    assert book.ask_size("yes") == 30


def test_delta_updates_one_price_level() -> None:
    book = BinaryOrderBook("KXBTC15M-TEST")
    book.apply(snapshot())
    change = book.apply({
        "type": "orderbook_delta",
        "seq": 2,
        "msg": {
            "market_ticker": "KXBTC15M-TEST",
            "side": "yes",
            "price_dollars": "0.41",
            "delta_fp": "-10.00",
        },
    })
    assert change is not None
    assert change.old_count == 25
    assert change.new_count == 15
    assert book.bid_size("yes") == 15


def test_paper_fill_requires_trade_volume_to_clear_queue() -> None:
    book = BinaryOrderBook("KXBTC15M-TEST")
    book.apply(snapshot())
    broker = QueuePaperBroker()
    order = broker.place(book, "yes", "buy", 41, 10, 1)
    assert order.queue_ahead == 25
    assert broker.on_trade(41, 59, 20, "no") == []
    fills = broker.on_trade(41, 59, 10, "no")
    assert len(fills) == 1
    assert fills[0].count == 5
    assert broker.orders[order.order_id].remaining == 5


def test_wrong_taker_side_does_not_fill_maker() -> None:
    book = BinaryOrderBook("KXBTC15M-TEST")
    book.apply(snapshot())
    broker = QueuePaperBroker()
    broker.place(book, "yes", "buy", 41, 10, 1)
    assert broker.on_trade(41, 59, 100, "yes") == []
