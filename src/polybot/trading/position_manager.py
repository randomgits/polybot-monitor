"""Position manager for tracking market windows and start prices.

The key insight: BTC 15-min markets resolve based on whether BTC price
at window END >= price at window START. We must capture the start price
at exactly the window start time using Chainlink (the resolution source).
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()


@dataclass
class MarketWindow:
    """Tracks a single 15-minute market window."""

    market_id: str
    slug: str
    question: str

    # Window timing
    window_start: datetime  # When the 15-min window starts
    window_end: datetime  # When market resolves

    # Start price (captured at window_start)
    start_price_chainlink: Optional[float] = None
    start_price_captured_at: Optional[datetime] = None

    # Resolution
    end_price_chainlink: Optional[float] = None
    resolved: bool = False
    resolution: Optional[str] = None  # "UP" or "DOWN"

    # Token IDs for trading
    yes_token_id: str = ""
    no_token_id: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = asdict(self)
        d["window_start"] = self.window_start.isoformat()
        d["window_end"] = self.window_end.isoformat()
        if self.start_price_captured_at:
            d["start_price_captured_at"] = self.start_price_captured_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "MarketWindow":
        """Create from dictionary."""
        data["window_start"] = datetime.fromisoformat(data["window_start"])
        data["window_end"] = datetime.fromisoformat(data["window_end"])
        if data.get("start_price_captured_at"):
            data["start_price_captured_at"] = datetime.fromisoformat(data["start_price_captured_at"])
        return cls(**data)

    @property
    def has_start_price(self) -> bool:
        """Whether we've captured the start price."""
        return self.start_price_chainlink is not None

    @property
    def is_window_started(self) -> bool:
        """Whether the trading window has started."""
        return datetime.now(timezone.utc) >= self.window_start

    @property
    def is_expired(self) -> bool:
        """Whether the market has expired."""
        return datetime.now(timezone.utc) >= self.window_end

    @property
    def time_to_start(self) -> float:
        """Seconds until window starts."""
        delta = self.window_start - datetime.now(timezone.utc)
        return max(0, delta.total_seconds())

    @property
    def time_to_end(self) -> float:
        """Seconds until window ends."""
        delta = self.window_end - datetime.now(timezone.utc)
        return max(0, delta.total_seconds())


class PositionManager:
    """
    Manages market windows and tracks start prices.

    Key responsibilities:
    1. Track when each market window starts
    2. Capture Chainlink price at window start (the "strike price")
    3. Track open positions
    4. Handle market resolution
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("/tmp/polybot_positions")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.windows_file = self.data_dir / "market_windows.json"

        # Active windows being tracked
        self.windows: dict[str, MarketWindow] = {}  # market_id -> window

        # Price history for start price capture
        self.price_history: list[tuple[datetime, float]] = []  # (timestamp, chainlink_price)
        self.max_history = 300  # Keep 5 minutes of history

        self._load_state()

        logger.info(
            "[PositionManager] Initialized",
            active_windows=len(self.windows),
        )

    def _load_state(self) -> None:
        """Load saved state."""
        try:
            if self.windows_file.exists():
                with open(self.windows_file) as f:
                    data = json.load(f)
                    for w_data in data.get("windows", []):
                        window = MarketWindow.from_dict(w_data)
                        if not window.resolved:
                            self.windows[window.market_id] = window
                    logger.info(f"[PositionManager] Loaded {len(self.windows)} windows")
        except Exception as e:
            logger.error(f"[PositionManager] Error loading state: {e}")

    def _save_state(self) -> None:
        """Save state to file."""
        try:
            with open(self.windows_file, "w") as f:
                json.dump({
                    "windows": [w.to_dict() for w in self.windows.values()],
                }, f, indent=2)
        except Exception as e:
            logger.error(f"[PositionManager] Error saving state: {e}")

    def record_price(self, chainlink_price: float) -> None:
        """Record a price point for start price capture."""
        now = datetime.now(timezone.utc)
        self.price_history.append((now, chainlink_price))

        # Trim old history
        if len(self.price_history) > self.max_history:
            self.price_history = self.price_history[-self.max_history:]

    def track_market(
        self,
        market_id: str,
        slug: str,
        question: str,
        window_start: datetime,
        window_end: datetime,
        yes_token_id: str,
        no_token_id: str,
    ) -> MarketWindow:
        """Start tracking a new market window."""
        if market_id in self.windows:
            return self.windows[market_id]

        window = MarketWindow(
            market_id=market_id,
            slug=slug,
            question=question,
            window_start=window_start,
            window_end=window_end,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
        )

        self.windows[market_id] = window
        self._save_state()

        logger.info(
            "[PositionManager] Tracking new market",
            market_id=market_id,
            window_start=window_start.isoformat(),
            window_end=window_end.isoformat(),
        )

        return window

    def try_capture_start_price(
        self,
        market_id: str,
        current_chainlink_price: float,
    ) -> Optional[float]:
        """
        Try to capture the start price for a market.

        Should be called frequently. Will capture price when window starts.
        Returns the start price if captured, None otherwise.
        """
        if market_id not in self.windows:
            return None

        window = self.windows[market_id]

        # Already captured
        if window.has_start_price:
            return window.start_price_chainlink

        # Check if window has started
        now = datetime.now(timezone.utc)
        if now < window.window_start:
            return None

        # Window just started! Capture the price.
        # Try to find the price closest to window start from history
        best_price = current_chainlink_price
        best_time_diff = abs((now - window.window_start).total_seconds())

        for ts, price in self.price_history:
            time_diff = abs((ts - window.window_start).total_seconds())
            if time_diff < best_time_diff:
                best_time_diff = time_diff
                best_price = price

        window.start_price_chainlink = best_price
        window.start_price_captured_at = now
        self._save_state()

        logger.info(
            "[PositionManager] Captured start price",
            market_id=market_id,
            start_price=f"${best_price:,.2f}",
            capture_delay=f"{best_time_diff:.1f}s",
        )

        return best_price

    def get_start_price(self, market_id: str) -> Optional[float]:
        """Get the start price for a market."""
        if market_id not in self.windows:
            return None
        return self.windows[market_id].start_price_chainlink

    def resolve_market(
        self,
        market_id: str,
        end_price_chainlink: float,
    ) -> Optional[str]:
        """
        Resolve a market and return the outcome.

        Returns "UP" if end_price >= start_price, "DOWN" otherwise.
        """
        if market_id not in self.windows:
            return None

        window = self.windows[market_id]

        if not window.has_start_price:
            logger.warning(f"[PositionManager] Cannot resolve {market_id}: no start price")
            return None

        window.end_price_chainlink = end_price_chainlink
        window.resolved = True

        if end_price_chainlink >= window.start_price_chainlink:
            window.resolution = "UP"
        else:
            window.resolution = "DOWN"

        self._save_state()

        logger.info(
            "[PositionManager] Market resolved",
            market_id=market_id,
            resolution=window.resolution,
            start_price=f"${window.start_price_chainlink:,.2f}",
            end_price=f"${end_price_chainlink:,.2f}",
            change=f"${end_price_chainlink - window.start_price_chainlink:+,.2f}",
        )

        return window.resolution

    def get_window(self, market_id: str) -> Optional[MarketWindow]:
        """Get a market window."""
        return self.windows.get(market_id)

    def cleanup_expired(self) -> list[str]:
        """Remove expired windows that are resolved or too old."""
        now = datetime.now(timezone.utc)
        to_remove = []

        for market_id, window in self.windows.items():
            # Remove if resolved and expired
            if window.resolved:
                to_remove.append(market_id)
            # Remove if expired for more than 5 minutes without resolution
            elif window.is_expired:
                age = (now - window.window_end).total_seconds()
                if age > 300:  # 5 minutes
                    to_remove.append(market_id)

        for market_id in to_remove:
            del self.windows[market_id]

        if to_remove:
            self._save_state()
            logger.info(f"[PositionManager] Cleaned up {len(to_remove)} windows")

        return to_remove

    def get_active_windows(self) -> list[MarketWindow]:
        """Get all active (non-resolved) windows."""
        return [w for w in self.windows.values() if not w.resolved]

    def get_status(self) -> dict:
        """Get position manager status."""
        active = self.get_active_windows()
        return {
            "active_windows": len(active),
            "windows_with_start_price": sum(1 for w in active if w.has_start_price),
            "price_history_length": len(self.price_history),
            "windows": [
                {
                    "market_id": w.market_id,
                    "question": w.question[:50],
                    "has_start_price": w.has_start_price,
                    "start_price": w.start_price_chainlink,
                    "time_to_end": w.time_to_end,
                }
                for w in active
            ],
        }
