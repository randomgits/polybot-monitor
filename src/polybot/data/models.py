"""Data models for PolyBot market state."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class PriceUpdate:
    """A single price update from any source."""

    price: float
    timestamp: datetime
    source: str  # "binance", "chainlink", "polymarket"

    @property
    def age_seconds(self) -> float:
        """Seconds since this update."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()


@dataclass
class BTC15MinMarket:
    """Represents a BTC 15-minute prediction market on Polymarket."""

    market_id: str
    condition_id: str
    question: str
    token_id: str  # YES token ID
    no_token_id: str  # NO token ID
    yes_price: float
    no_price: float
    start_price: float  # BTC price at market open
    end_time: datetime
    slug: str = ""

    @property
    def time_to_expiry_seconds(self) -> float:
        """Seconds until market resolution."""
        delta = self.end_time - datetime.now(timezone.utc)
        return max(0, delta.total_seconds())

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return abs(self.yes_price + self.no_price - 1.0)

    @property
    def is_expired(self) -> bool:
        """Whether market has expired."""
        return self.time_to_expiry_seconds <= 0


@dataclass
class MarketState:
    """
    Complete market state for decision making.

    Contains all features needed by the probability model and RL agent.
    13 core features that can be converted to array for RL.
    """

    timestamp: datetime

    # BTC prices from different sources
    btc_price_binance: float
    btc_price_chainlink: float  # Resolution source!

    # Polymarket odds
    polymarket_yes_price: float
    polymarket_no_price: float
    polymarket_spread: float

    # Market timing
    time_to_expiry: float  # Seconds
    market_start_price: float  # BTC price at window open

    # Volatility (annualized)
    volatility_1m: float = 0.5
    volatility_5m: float = 0.5
    volatility_15m: float = 0.5

    # Order flow signals
    binance_orderbook_imbalance: float = 0.0  # -1 to 1
    chainlink_binance_spread: float = 0.0  # $ difference

    # Position tracking
    current_position: float = 0.0  # Shares held
    current_position_value: float = 0.0  # USD value

    # Market metadata
    market_id: str = ""
    yes_token_id: str = ""
    no_token_id: str = ""

    def to_array(self) -> list[float]:
        """
        Convert to feature array for RL agent.

        Returns 13 normalized features.
        """
        # Normalize prices relative to a baseline (e.g., 100k BTC)
        price_scale = 100_000.0

        return [
            # BTC prices (normalized)
            self.btc_price_binance / price_scale,
            self.btc_price_chainlink / price_scale,
            (self.btc_price_chainlink - self.market_start_price) / price_scale,  # Change from start

            # Polymarket odds (already 0-1)
            self.polymarket_yes_price,
            self.polymarket_no_price,
            self.polymarket_spread,

            # Time (normalized to 0-1 over 15 min)
            min(1.0, self.time_to_expiry / 900.0),

            # Volatility (already annualized fraction)
            self.volatility_1m,
            self.volatility_5m,
            self.volatility_15m,

            # Order flow (already -1 to 1)
            self.binance_orderbook_imbalance,

            # Price spread (normalized)
            self.chainlink_binance_spread / 100.0,  # Per $100

            # Position (normalized by typical size)
            self.current_position / 100.0,
        ]

    @classmethod
    def empty(cls) -> "MarketState":
        """Create an empty state for initialization."""
        return cls(
            timestamp=datetime.now(timezone.utc),
            btc_price_binance=0.0,
            btc_price_chainlink=0.0,
            polymarket_yes_price=0.5,
            polymarket_no_price=0.5,
            polymarket_spread=0.0,
            time_to_expiry=900.0,
            market_start_price=0.0,
        )


@dataclass
class Opportunity:
    """A detected trading opportunity."""

    timestamp: datetime
    market_id: str
    market_question: str

    # Prices
    btc_price_binance: float
    btc_price_chainlink: float
    market_start_price: float

    # Polymarket
    yes_price: float
    no_price: float
    spread: float
    time_to_expiry: float

    # Signal
    model_probability: float
    market_probability: float
    edge: float
    recommended_action: str
    confidence: float
    kelly_fraction: float

    # Metadata
    volatility: float = 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "market_id": self.market_id,
            "market_question": self.market_question,
            "btc_price_binance": self.btc_price_binance,
            "btc_price_chainlink": self.btc_price_chainlink,
            "market_start_price": self.market_start_price,
            "btc_change_pct": ((self.btc_price_chainlink - self.market_start_price) / self.market_start_price * 100) if self.market_start_price > 0 else 0,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "spread": self.spread,
            "time_to_expiry_seconds": self.time_to_expiry,
            "time_to_expiry_formatted": f"{int(self.time_to_expiry // 60)}:{int(self.time_to_expiry % 60):02d}",
            "model_probability": self.model_probability,
            "market_probability": self.market_probability,
            "edge_pct": self.edge * 100,
            "recommended_action": self.recommended_action,
            "confidence": self.confidence,
            "kelly_fraction": self.kelly_fraction,
            "volatility": self.volatility,
        }
