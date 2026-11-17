"""Weather trading package for Kalshi daily high/low temperature markets.

Pipeline:

    forecasts + observations  ->  model (bracket probabilities)
                              ->  strategy (edge after fees, Kelly sizing)
                              ->  engine (paper or live orders, JSONL logging)

See ``WEATHER_BOT.md`` in the repository root for the operating guide.
"""

from __future__ import annotations

__all__ = [
    "brackets",
    "calibration",
    "climate",
    "datalog",
    "engine",
    "fees",
    "forecasts",
    "ledger",
    "model",
    "observations",
    "report",
    "settings",
    "stations",
    "strategy",
]
