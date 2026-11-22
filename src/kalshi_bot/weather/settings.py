"""Weather bot settings, loaded through ``KALSHI_WEATHER__*`` env variables.

Kept in its own module (plain pydantic, no pydantic-settings import) so the
weather package can be used and tested without the full app config.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_DATA_DIR = "./data/weather"


class WeatherSettings(BaseModel):
    """Everything the weather engine, model and strategy need."""

    # --- Universe -----------------------------------------------------------
    stations: str = Field(
        default="NYC,CHI,AUS,MIA,DEN,LAX,PHX,SFO,SEA,LV,SAN,MIN,ATL,BOS,HOU,OKC,DAL,DC,TTN",
        description="Comma-separated codes (see stations.py).",
    )
    kinds: str = Field(default="high,low", description="'high', 'low' or 'high,low'.")
    hourly: bool = Field(default=True, description="Trade the hourly temperature markets.")
    rain: bool = Field(default=True, description="Trade the daily 'Where will it rain?' markets.")
    hourly_min_lead_minutes: float = Field(
        default=45.0,
        ge=0,
        description="Don't bet an hourly market this close to its reading time.",
    )
    rain_cutoff_hours: float = Field(
        default=6.0,
        description="Stop betting a day's rain market this many hours after local midnight.",
    )
    series_overrides: dict[str, str] = Field(
        default_factory=dict,
        description='Override series tickers, e.g. {"NYC:low": "KXLOWTNYC"}.',
    )
    max_lead_hours: float = Field(
        default=52.0,
        gt=0,
        description="Ignore events whose climate day ends more than this many hours out.",
    )

    # --- Data sources -------------------------------------------------------
    user_agent: str = Field(
        default="kalshi-weather-bot (set KALSHI_WEATHER__USER_AGENT to your email)",
        description="api.weather.gov requires a User-Agent that identifies you.",
    )
    data_dir: str = Field(
        default=DEFAULT_DATA_DIR,
        description="Where logs/ledger live. Default uses the Railway volume when mounted.",
    )
    openmeteo_api_key: str = Field(
        default="",
        description="Optional paid Open-Meteo key (free tier is non-commercial, ~10k calls/day).",
    )
    # Each inner list is one model *group*: the first ID that works is used, the
    # rest are alternates (Open-Meteo has renamed IDs over time).
    ensemble_models: list[list[str]] = Field(
        default_factory=lambda: [
            ["gfs_seamless", "gfs025", "ncep_gefs025"],
            ["ecmwf_ifs025", "ecmwf_ifs04", "ecmwf_ifs"],
            ["icon_seamless", "dwd_icon_seamless_eps", "icon_global"],
            ["gem_global", "cmc_gem_geps"],
        ]
    )
    deterministic_models: list[list[str]] = Field(
        default_factory=lambda: [["ncep_nbm_conus"], ["gfs_seamless"], ["ecmwf_ifs025"]]
    )
    use_nws_gridpoint: bool = True
    nws_weight: float = Field(default=2.0, ge=0)
    nbm_weight: float = Field(default=2.0, ge=0)
    ensemble_weight: float = Field(default=1.0, ge=0)
    deterministic_weight: float = Field(default=1.0, ge=0)

    # --- Refresh cadence (seconds) ------------------------------------------
    poll_interval: float = Field(default=60.0, gt=0)
    forecast_refresh: float = Field(default=3600.0, gt=0)
    obs_refresh: float = Field(default=300.0, gt=0)
    log_interval: float = Field(
        default=600.0, gt=0, description="Seconds between logged snapshots per event."
    )
    settle_check_interval: float = Field(default=900.0, gt=0)

    # --- Model --------------------------------------------------------------
    samples: int = Field(default=40_000, ge=1_000)
    tail_weight: float = Field(
        default=0.05,
        ge=0,
        le=0.5,
        description="Share of samples drawn with doubled spread (fat tails).",
    )
    obs_persistence: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="How much of today's observed-vs-forecast anomaly carries forward.",
    )

    # --- Which strategy ------------------------------------------------------
    strategy: str = Field(
        default="live",
        description=(
            "'live' = only bet outcomes live airport readings show are already decided "
            "(see live.py). 'simple' = bet the NWS forecast before the day starts. "
            "'model' = the older multi-model strategy."
        ),
    )

    # --- Beyond weather (same "already decided" idea) ----------------------------
    sports: bool = Field(default=True, description="Game-winner markets decided by live scores.")
    prices: bool = Field(
        default=True, description="Crypto/commodity/index markets decided by price near close."
    )
    fast_poll_interval: float = Field(
        default=15.0, gt=0, description="Seconds between sports/price checks."
    )
    price_window_seconds: float = Field(
        default=900.0, gt=0, description="Only judge price markets this close to their close."
    )
    price_stop_seconds: float = Field(
        default=10.0, ge=0, description="Stop betting a price market this close to its close."
    )
    price_z: float = Field(
        default=3.0, gt=0, description="Standard moves the price must sit from the line."
    )

    # --- Live strategy ----------------------------------------------------------
    live_stake_cents: int = Field(default=1_000, gt=0, description="Money per bet ($10).")
    live_max_price: int = Field(default=97, ge=1, le=99, description="Never pay more than this.")
    live_cushion_cents: int = Field(
        default=2, ge=0, description="Price must be this far below the chance of winning."
    )
    live_min_price: int = Field(
        default=60,
        ge=1,
        le=99,
        description=(
            "Only buy a side Kalshi also prices as the favourite (at least this many cents). "
            "If the bot is sure but the market says it's a long shot, the bot is missing something."
        ),
    )
    hourly_live_gap: float = Field(
        default=3.0,
        gt=0,
        description="Hourly, using Kalshi's own index: projected reading must be this many °F "
        "from the line.",
    )
    hourly_fallback_gap: float = Field(
        default=5.0,
        gt=0,
        description="Hourly, when the index can't be read and airports are used instead.",
    )
    hourly_live_minutes: float = Field(
        default=60.0, gt=0, description="Hourly: only bet within this many minutes of the reading."
    )

    # --- Simple strategy --------------------------------------------------------
    simple_margin: float = Field(
        default=0.05,
        ge=0,
        description="Buy only if the forecast odds beat Kalshi's price + fee by this (0.05 = 5c).",
    )
    simple_stake_cents: int = Field(default=500, gt=0, description="Money per bet ($5).")

    # --- Model strategy (strategy='model') -----------------------------------
    min_edge: float = Field(
        default=0.08,
        ge=0,
        description="Required edge after fees, in probability units (0.08 = 8c).",
    )
    confidence: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Weight on the model vs the market mid before the day starts (1 = model only).",
    )
    intraday_confidence: float = Field(
        default=0.25,
        ge=0,
        le=1,
        description=(
            "Weight on the model once the climate day is under way. Traders watch the same "
            "observations, so the market mid gets more say."
        ),
    )
    # Never buy a contract priced below min_price: cheap long shots (either side)
    # are where the market is usually right and the model most often wrong.
    min_price: int = Field(default=12, ge=1, le=99)
    max_price: int = Field(default=88, ge=1, le=99)
    fee_rate: float = Field(default=0.07, ge=0)
    max_contracts_per_order: int = Field(default=50, gt=0)
    max_cost_per_market_cents: int = Field(default=1_500, gt=0)
    max_cost_per_event_cents: int = Field(default=2_500, gt=0)
    max_cost_per_day_cents: int = Field(
        default=60_000,
        gt=0,
        description="Cap across ALL cities for one date (weather systems are correlated).",
    )
    paper_bankroll_cents: int = Field(default=50_000, gt=0)
    reentry_cooldown_s: float = Field(
        default=3600.0, ge=0, description="No new entry on a market this soon after any trade."
    )
    exit_on_reversal: bool = True
    exit_margin: float = Field(default=0.08, ge=0)
    partition_arb: bool = True
    stop_trading_minutes_before_close: float = Field(default=5.0, ge=0)

    def station_codes(self) -> list[str]:
        return [s.strip().upper() for s in self.stations.split(",") if s.strip()]

    def kind_names(self) -> list[str]:
        return [k.strip().lower() for k in self.kinds.split(",") if k.strip()]

    def data_path(self) -> Path:
        """Resolved data directory.

        With the default setting and a Railway volume mounted
        (``RAILWAY_VOLUME_MOUNT_PATH``), data goes to ``<volume>/weather`` so it
        survives redeploys. Falls back to the local default if that isn't writable.
        """
        volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
        if self.data_dir == DEFAULT_DATA_DIR and volume:
            candidate = Path(volume) / "weather"
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                probe = candidate / ".write_test"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
                return candidate
            except OSError:
                pass
        path = Path(self.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
