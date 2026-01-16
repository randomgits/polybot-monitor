"""Paper trading module for backtesting and simulation.

Tracks hypothetical trades without real money to validate strategy performance.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()


@dataclass
class PaperTrade:
    """A single paper trade."""

    trade_id: str
    market_id: str
    market_question: str

    # Entry
    entry_time: datetime
    entry_side: str  # "YES" or "NO"
    entry_price: float
    size_usd: float
    shares: float

    # Market state at entry
    btc_chainlink_at_entry: float
    btc_start_price: float
    time_to_expiry_at_entry: float
    model_probability: float
    market_probability: float
    edge: float
    confidence: float

    # Exit (filled after resolution)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    outcome: Optional[str] = None  # "WIN", "LOSS"
    pnl: Optional[float] = None

    # Status
    status: str = "OPEN"  # "OPEN", "CLOSED", "EXPIRED"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["entry_time"] = self.entry_time.isoformat()
        if self.exit_time:
            d["exit_time"] = self.exit_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "PaperTrade":
        """Create from dictionary."""
        data["entry_time"] = datetime.fromisoformat(data["entry_time"])
        if data.get("exit_time"):
            data["exit_time"] = datetime.fromisoformat(data["exit_time"])
        return cls(**data)


@dataclass
class TradingStats:
    """Aggregate trading statistics."""

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0

    # By side
    yes_trades: int = 0
    yes_wins: int = 0
    no_trades: int = 0
    no_wins: int = 0

    # Running metrics
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0
    current_streak: int = 0  # Positive = wins, negative = losses

    @property
    def win_rate(self) -> float:
        """Win rate as percentage."""
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades * 100

    @property
    def avg_pnl_per_trade(self) -> float:
        """Average P&L per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "win_rate": self.win_rate,
            "avg_pnl_per_trade": self.avg_pnl_per_trade,
        }


class PaperTrader:
    """
    Paper trading system for strategy validation.

    Features:
    - Tracks hypothetical trades based on signals
    - Records entry/exit with full market state
    - Calculates P&L after market resolution
    - Persists trade history to file
    - Provides performance statistics
    """

    def __init__(
        self,
        initial_balance: float = 1000.0,
        max_position_pct: float = 0.10,
        min_edge_to_trade: float = 0.05,
        min_confidence: float = 0.3,
        data_dir: Optional[Path] = None,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_position_pct = max_position_pct
        self.min_edge_to_trade = min_edge_to_trade
        self.min_confidence = min_confidence

        # Data persistence
        self.data_dir = data_dir or Path("/tmp/polybot_paper")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.trades_file = self.data_dir / "paper_trades.json"
        self.stats_file = self.data_dir / "paper_stats.json"

        # State
        self.trades: list[PaperTrade] = []
        self.open_trades: dict[str, PaperTrade] = {}  # market_id -> trade
        self.stats = TradingStats()
        self._trade_counter = 0

        # Load existing data
        self._load_state()

        logger.info(
            "[PaperTrader] Initialized",
            balance=self.balance,
            total_trades=self.stats.total_trades,
            total_pnl=self.stats.total_pnl,
        )

    def _load_state(self) -> None:
        """Load saved state from files."""
        try:
            if self.trades_file.exists():
                with open(self.trades_file) as f:
                    data = json.load(f)
                    self.trades = [PaperTrade.from_dict(t) for t in data.get("trades", [])]
                    self.balance = data.get("balance", self.initial_balance)
                    self._trade_counter = data.get("trade_counter", len(self.trades))

                    # Rebuild open trades index
                    for trade in self.trades:
                        if trade.status == "OPEN":
                            self.open_trades[trade.market_id] = trade

                    logger.info(f"[PaperTrader] Loaded {len(self.trades)} trades")

            if self.stats_file.exists():
                with open(self.stats_file) as f:
                    data = json.load(f)
                    self.stats = TradingStats(
                        total_trades=data.get("total_trades", 0),
                        wins=data.get("wins", 0),
                        losses=data.get("losses", 0),
                        total_pnl=data.get("total_pnl", 0.0),
                        yes_trades=data.get("yes_trades", 0),
                        yes_wins=data.get("yes_wins", 0),
                        no_trades=data.get("no_trades", 0),
                        no_wins=data.get("no_wins", 0),
                        max_drawdown=data.get("max_drawdown", 0.0),
                        peak_pnl=data.get("peak_pnl", 0.0),
                        current_streak=data.get("current_streak", 0),
                    )
        except Exception as e:
            logger.error(f"[PaperTrader] Error loading state: {e}")

    def _save_state(self) -> None:
        """Save state to files."""
        try:
            # Save trades
            with open(self.trades_file, "w") as f:
                json.dump({
                    "trades": [t.to_dict() for t in self.trades],
                    "balance": self.balance,
                    "trade_counter": self._trade_counter,
                }, f, indent=2)

            # Save stats
            with open(self.stats_file, "w") as f:
                json.dump(self.stats.to_dict(), f, indent=2)

        except Exception as e:
            logger.error(f"[PaperTrader] Error saving state: {e}")

    def should_trade(
        self,
        edge: float,
        confidence: float,
        time_to_expiry: float,
        spread: float,
        market_id: str,
    ) -> bool:
        """Determine if we should enter a trade."""
        # Already have position in this market
        if market_id in self.open_trades:
            return False

        # Edge too small
        if abs(edge) < self.min_edge_to_trade:
            return False

        # Confidence too low
        if confidence < self.min_confidence:
            return False

        # Too close to expiry (need at least 60s)
        if time_to_expiry < 60:
            return False

        # Spread too wide (>5%)
        if spread > 0.05:
            return False

        return True

    def calculate_position_size(self, kelly_fraction: float) -> float:
        """Calculate position size in USD."""
        # Use Kelly fraction but cap at max position
        position_pct = min(kelly_fraction, self.max_position_pct)
        return self.balance * position_pct

    def open_trade(
        self,
        market_id: str,
        market_question: str,
        side: str,  # "YES" or "NO"
        price: float,
        btc_chainlink: float,
        btc_start_price: float,
        time_to_expiry: float,
        model_probability: float,
        market_probability: float,
        edge: float,
        confidence: float,
        kelly_fraction: float,
    ) -> Optional[PaperTrade]:
        """Open a new paper trade."""
        if market_id in self.open_trades:
            logger.warning(f"[PaperTrader] Already have position in {market_id}")
            return None

        # Calculate position size
        size_usd = self.calculate_position_size(kelly_fraction)
        if size_usd < 5:  # Minimum trade size
            logger.debug(f"[PaperTrader] Position size too small: ${size_usd:.2f}")
            return None

        # Calculate shares (how much we'd buy)
        # Price is what we pay per share, payout is $1 if we win
        shares = size_usd / price

        # Create trade
        self._trade_counter += 1
        trade = PaperTrade(
            trade_id=f"PT-{self._trade_counter:05d}",
            market_id=market_id,
            market_question=market_question,
            entry_time=datetime.now(timezone.utc),
            entry_side=side,
            entry_price=price,
            size_usd=size_usd,
            shares=shares,
            btc_chainlink_at_entry=btc_chainlink,
            btc_start_price=btc_start_price,
            time_to_expiry_at_entry=time_to_expiry,
            model_probability=model_probability,
            market_probability=market_probability,
            edge=edge,
            confidence=confidence,
        )

        self.trades.append(trade)
        self.open_trades[market_id] = trade
        self._save_state()

        logger.info(
            f"[PaperTrader] Opened {side} position",
            trade_id=trade.trade_id,
            market=market_question[:40],
            size=f"${size_usd:.2f}",
            price=f"{price:.3f}",
            edge=f"{edge*100:.1f}%",
        )

        return trade

    def close_trade(
        self,
        market_id: str,
        final_btc_price: float,
        btc_start_price: float,
    ) -> Optional[PaperTrade]:
        """Close a trade when market resolves."""
        if market_id not in self.open_trades:
            return None

        trade = self.open_trades[market_id]

        # Determine outcome
        btc_went_up = final_btc_price >= btc_start_price

        if trade.entry_side == "YES":
            won = btc_went_up
        else:  # NO
            won = not btc_went_up

        # Calculate P&L
        if won:
            # Win: get $1 per share, minus what we paid
            pnl = trade.shares * (1.0 - trade.entry_price)
            # Apply ~1.5% fee on winnings (simplified)
            pnl *= 0.985
            trade.outcome = "WIN"
        else:
            # Loss: lose entire stake
            pnl = -trade.size_usd
            trade.outcome = "LOSS"

        trade.exit_time = datetime.now(timezone.utc)
        trade.exit_price = 1.0 if won else 0.0
        trade.pnl = pnl
        trade.status = "CLOSED"

        # Update balance
        self.balance += pnl

        # Update stats
        self.stats.total_trades += 1
        self.stats.total_pnl += pnl

        if won:
            self.stats.wins += 1
            self.stats.current_streak = max(1, self.stats.current_streak + 1)
        else:
            self.stats.losses += 1
            self.stats.current_streak = min(-1, self.stats.current_streak - 1)

        if trade.entry_side == "YES":
            self.stats.yes_trades += 1
            if won:
                self.stats.yes_wins += 1
        else:
            self.stats.no_trades += 1
            if won:
                self.stats.no_wins += 1

        # Track drawdown
        if self.stats.total_pnl > self.stats.peak_pnl:
            self.stats.peak_pnl = self.stats.total_pnl
        drawdown = self.stats.peak_pnl - self.stats.total_pnl
        if drawdown > self.stats.max_drawdown:
            self.stats.max_drawdown = drawdown

        # Remove from open trades
        del self.open_trades[market_id]
        self._save_state()

        logger.info(
            f"[PaperTrader] Closed trade: {trade.outcome}",
            trade_id=trade.trade_id,
            pnl=f"${pnl:+.2f}",
            total_pnl=f"${self.stats.total_pnl:+.2f}",
            win_rate=f"{self.stats.win_rate:.1f}%",
        )

        return trade

    def expire_trade(self, market_id: str) -> Optional[PaperTrade]:
        """Mark a trade as expired (market closed without resolution data)."""
        if market_id not in self.open_trades:
            return None

        trade = self.open_trades[market_id]
        trade.status = "EXPIRED"
        trade.exit_time = datetime.now(timezone.utc)

        del self.open_trades[market_id]
        self._save_state()

        logger.warning(f"[PaperTrader] Trade expired: {trade.trade_id}")
        return trade

    def get_open_positions(self) -> list[PaperTrade]:
        """Get all open positions."""
        return list(self.open_trades.values())

    def get_stats(self) -> dict:
        """Get trading statistics."""
        return {
            **self.stats.to_dict(),
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "return_pct": (self.balance - self.initial_balance) / self.initial_balance * 100,
            "open_positions": len(self.open_trades),
        }

    def get_recent_trades(self, limit: int = 20) -> list[dict]:
        """Get recent trades."""
        return [t.to_dict() for t in reversed(self.trades[-limit:])]

    def execute_trade(
        self,
        market_id: str,
        action: str,  # "BUY_YES", "BUY_NO", "HOLD"
        probability: float,
        edge: float,
        confidence: float,
        kelly_fraction: float,
        yes_price: float,
        no_price: float,
        time_to_expiry: float,
        spread: float,
        market_question: str = "",
        btc_chainlink: float = 0.0,
        btc_start_price: float = 0.0,
    ) -> Optional[PaperTrade]:
        """
        Execute a trade if conditions are met.

        Convenience method that combines should_trade check and open_trade.

        Returns the trade if executed, None otherwise.
        """
        if action == "HOLD":
            return None

        # Determine side and price
        if action == "BUY_YES":
            side = "YES"
            price = yes_price
        elif action == "BUY_NO":
            side = "NO"
            price = no_price
        else:
            logger.warning(f"[PaperTrader] Unknown action: {action}")
            return None

        # Check if we should trade
        if not self.should_trade(edge, confidence, time_to_expiry, spread, market_id):
            return None

        # Open the trade
        return self.open_trade(
            market_id=market_id,
            market_question=market_question,
            side=side,
            price=price,
            btc_chainlink=btc_chainlink,
            btc_start_price=btc_start_price,
            time_to_expiry=time_to_expiry,
            model_probability=probability,
            market_probability=yes_price if side == "YES" else 1 - no_price,
            edge=edge,
            confidence=confidence,
            kelly_fraction=kelly_fraction,
        )

    def reset(self) -> None:
        """Reset paper trading state."""
        self.balance = self.initial_balance
        self.trades = []
        self.open_trades = {}
        self.stats = TradingStats()
        self._trade_counter = 0
        self._save_state()
        logger.info("[PaperTrader] Reset complete")
