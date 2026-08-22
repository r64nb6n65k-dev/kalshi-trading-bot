"""Typed configuration loaded from environment variables and ``.env``.

All settings are validated at startup via pydantic. Secrets (API key ID and the
path to the RSA private key) come from the environment -- never hard-code them.
"""

from __future__ import annotations

from enum import Enum

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

    bnb_ws_url: str = Field(
        default="wss://advanced-trade-ws.coinbase.com",
        description="Public Coinbase BNB/USD WebSocket; Coinbase is a BNBUSD_RTI constituent.",
    )

    bnb_product_id: str = Field(default="BNB-USD")

    bnb_max_age_seconds: float = Field(
        default=3.0,
        gt=0.0,
        description="Reject entries when the BNB proxy stream is older than this.",
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
    """Load and validate settings from the environment."""
    return Settings()
