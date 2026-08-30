"""Typed configuration loaded from environment variables and ``.env``.

All settings are validated at startup via pydantic. Secrets (API key ID and the
path to the RSA private key) come from the environment -- never hard-code them.
"""

from __future__ import annotations

from enum import Enum
import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Kalshi trading environments."""

    PROD = "prod"
    DEMO = "demo"


REST_BASE_URLS: dict[Environment, str] = {
    Environment.PROD: "https://api.elections.kalshi.com/trade-api/v2",
    Environment.DEMO: "https://external-api.demo.kalshi.co/trade-api/v2",
}

WS_BASE_URLS: dict[Environment, str] = {
    Environment.PROD: "wss://external-api-ws.kalshi.com/trade-api/ws/v2",
    Environment.DEMO: "wss://demo-api.kalshi.co/trade-api/ws/v2",
}


class RiskSettings(BaseModel):
    """Risk limits configurable through ``KALSHI_RISK__*`` variables."""

    max_contracts_per_order: int = Field(default=200, gt=0)
    max_position_per_market: int = Field(default=500, gt=0)
    max_order_notional_cents: int = Field(default=50_000, gt=0)
    kelly_fraction: float = Field(default=0.25, gt=0.0, le=1.0)


class Settings(BaseSettings):
    """Runtime configuration for the bot."""

    model_config = SettingsConfigDict(
        env_prefix="KALSHI_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    api_key_id: str = Field(default="", description="Kalshi API key ID (UUID).")
    private_key_path: str = Field(
        default="./secrets/kalshi_private_key.pem",
        description="Path to the RSA private key PEM file.",
    )

    environment: Environment = Field(
        default=Environment.PROD,
        description="Trading environment: 'demo' (sandbox) or 'prod'.",
    )

    dry_run: bool = Field(
        default=True,
        description="If true, simulate orders instead of sending them to Kalshi.",
    )

    request_timeout: float = Field(
        default=10.0,
        gt=0.0,
        description="HTTP timeout in seconds.",
    )

    poll_interval: float = Field(
        default=1.0,
        gt=0.0,
        description="Seconds between market polls in the live engine.",
    )

    btc_ws_url: str = Field(
        default="wss://advanced-trade-ws.coinbase.com",
        description="Public BTC/USD WebSocket used as a BRTI proxy.",
    )
    btc_product_id: str = Field(default="BTC-USD")
    btc_max_age_seconds: float = Field(
        default=3.0,
        gt=0.0,
        description="Reject entries when the BTC/USD proxy feed is older than this.",
    )

    # Keep the historical field name so the existing Railway
    # KALSHI_PYTH_API_KEY variable continues to supply the OANDA token.
    pyth_api_key: str = Field(default="")
    gold_max_age_seconds: float = Field(
        default=3.0,
        gt=0.0,
        description="Reject entries when the OANDA XAU/USD feed is older than this.",
    )

    risk: RiskSettings = Field(default_factory=RiskSettings)

    @field_validator("api_key_id")
    @classmethod
    def _strip_key(cls, value: str) -> str:
        """Trim accidental whitespace from a pasted key ID."""
        return value.strip()

    @property
    def rest_base_url(self) -> str:
        """REST base URL for the configured environment."""
        return REST_BASE_URLS[self.environment]

    @property
    def ws_base_url(self) -> str:
        """WebSocket base URL for the configured environment."""
        return WS_BASE_URLS[self.environment]


def load_settings() -> Settings:
    """Load and validate settings from the environment.

    Railway can provide the Kalshi RSA private key directly through
    ``KALSHI_PRIVATE_KEY``. When present, write it to a private temporary PEM
    file and point the existing authentication code at that file. This keeps
    the REST and WebSocket authentication paths unchanged.
    """
    settings = Settings()

    private_key_pem = os.getenv("KALSHI_PRIVATE_KEY", "").strip()
    if private_key_pem:
        # Also tolerate a value pasted with literal \n sequences.
        private_key_pem = private_key_pem.replace("\\n", "\n")
        key_path = Path("/tmp/kalshi_private_key.pem")
        key_path.write_text(private_key_pem + "\n", encoding="utf-8")
        key_path.chmod(0o600)
        settings.private_key_path = str(key_path)

    return settings
