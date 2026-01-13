"""Configuration management for the trading bot."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Polymarket
    polymarket_api_key: str = Field(default="", description="Polymarket API key")
    polymarket_secret: str = Field(default="", description="Polymarket API secret")
    polymarket_passphrase: str = Field(default="", description="Polymarket API passphrase")

    # Binance
    binance_api_key: str = Field(default="", description="Binance API key")
    binance_api_secret: str = Field(default="", description="Binance API secret")

    # Ethereum/Polygon (for Chainlink)
    polygon_rpc_url: str = Field(
        default="https://polygon-rpc.com",
        description="Polygon RPC endpoint for Chainlink data",
    )

    # Database
    database_url: str = Field(
        default="postgresql://localhost/polybot",
        description="PostgreSQL connection string",
    )

    # Trading parameters
    max_position_usd: float = Field(default=100.0, description="Maximum position size in USD")
    daily_loss_limit_usd: float = Field(default=50.0, description="Daily loss limit in USD")
    min_edge_threshold: float = Field(
        default=0.05, description="Minimum edge (probability difference) to trade"
    )
    kelly_fraction: float = Field(
        default=0.25, description="Fraction of Kelly criterion to use for position sizing"
    )

    # RL parameters
    rl_exploration_rate: float = Field(
        default=0.1, description="Epsilon for epsilon-greedy exploration"
    )
    rl_learning_rate: float = Field(default=3e-4, description="RL agent learning rate")
    rl_batch_size: int = Field(default=64, description="Batch size for RL training")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")


settings = Settings()
