"""Risk management for position sizing and loss limits."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog

# TradingSignal no longer needed - RL makes decisions, risk manager just enforces limits

logger = structlog.get_logger()


@dataclass
class RiskLimits:
    """Risk limit configuration."""

    max_position_usd: float = 100.0  # Max position size per market
    daily_loss_limit_usd: float = 50.0  # Max daily loss before stopping
    max_drawdown_pct: float = 0.20  # Max drawdown from peak (20%)
    min_time_to_expiry: float = 30.0  # Don't trade with < 30s left
    max_spread_pct: float = 0.05  # Don't trade if spread > 5%
    max_concurrent_positions: int = 1  # Only one position at a time


@dataclass
class Position:
    """Current position in a market."""

    market_id: str
    token_id: str
    side: str  # "YES" or "NO"
    size: float  # Number of shares
    entry_price: float
    entry_time: datetime
    current_value: float = 0.0

    @property
    def pnl(self) -> float:
        """Unrealized P&L."""
        cost = self.size * self.entry_price
        return self.current_value - cost


@dataclass
class RiskState:
    """Current risk state tracking."""

    daily_pnl: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    trades_today: int = 0
    last_reset_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    @property
    def total_position_value(self) -> float:
        """Total value of all positions."""
        return sum(p.current_value for p in self.positions.values())


class RiskManager:
    """
    Manages risk limits and position sizing.

    Responsibilities:
    - Enforce position size limits
    - Track daily P&L and stop trading if limit hit
    - Enforce drawdown limits
    - Validate trades before execution
    """

    def __init__(self, limits: RiskLimits | None = None, initial_capital: float = 1000.0):
        self.limits = limits or RiskLimits()
        self.initial_capital = initial_capital
        self.state = RiskState(
            current_equity=initial_capital,
            peak_equity=initial_capital,
        )

    def can_trade(self) -> tuple[bool, str]:
        """
        Check if trading is allowed given current risk state.

        Returns:
            (can_trade, reason)
        """
        # Check daily loss limit
        if self.state.daily_pnl <= -self.limits.daily_loss_limit_usd:
            return False, f"Daily loss limit hit: ${self.state.daily_pnl:.2f}"

        # Check drawdown limit
        if self.state.current_drawdown >= self.limits.max_drawdown_pct:
            return False, f"Max drawdown hit: {self.state.current_drawdown:.1%}"

        # Check concurrent positions
        if len(self.state.positions) >= self.limits.max_concurrent_positions:
            return False, f"Max concurrent positions: {len(self.state.positions)}"

        return True, "OK"

    def calculate_position_size(
        self,
        available_capital: float,
        position_pct: float = 0.10,
    ) -> float:
        """
        Calculate position size using fixed percentage of capital.

        Simpler than Kelly - RL learns WHEN to trade, risk manager controls HOW MUCH.

        Args:
            available_capital: Current capital available
            position_pct: Percentage of capital per trade (default 10%)
        """
        # Base position size as percentage of capital
        position_size = available_capital * position_pct

        # Cap at max position size per trade
        max_size = min(self.limits.max_position_usd, available_capital * 0.5)
        position_size = min(position_size, max_size)

        # Scale down if approaching daily loss limit
        if self.limits.daily_loss_limit_usd > 0:
            loss_ratio = abs(min(0, self.state.daily_pnl)) / self.limits.daily_loss_limit_usd
            risk_scaling = max(0.25, 1 - loss_ratio)
            position_size *= risk_scaling

        # Scale down if in drawdown
        if self.state.current_drawdown > 0.1:
            drawdown_scaling = max(0.5, 1 - self.state.current_drawdown)
            position_size *= drawdown_scaling

        # Minimum viable trade size
        min_trade_size = 1.0
        if position_size < min_trade_size:
            return 0.0

        return round(position_size, 2)

    def validate_trade(
        self,
        time_to_expiry: float,
        spread: float,
    ) -> tuple[bool, str]:
        """
        Validate a potential trade against risk rules.

        Note: This only checks risk limits, NOT probability model recommendations.
        The RL agent decides whether to trade; this just enforces safety limits.

        Returns:
            (is_valid, reason)
        """
        # Check if trading is allowed (position limits, daily loss, drawdown)
        can_trade, reason = self.can_trade()
        if not can_trade:
            return False, reason

        # Check time to expiry
        if time_to_expiry < self.limits.min_time_to_expiry:
            return False, f"Too close to expiry: {time_to_expiry:.0f}s"

        # Check spread
        if spread > self.limits.max_spread_pct:
            return False, f"Spread too wide: {spread:.1%}"

        return True, "OK"

    def record_trade(
        self,
        market_id: str,
        token_id: str,
        side: str,
        size: float,
        price: float,
        fees: float = 0.0,
    ) -> None:
        """Record a new position and deduct fees from equity."""
        position = Position(
            market_id=market_id,
            token_id=token_id,
            side=side,
            size=size,
            entry_price=price,
            entry_time=datetime.now(timezone.utc),
            current_value=size * price,
        )
        self.state.positions[market_id] = position
        self.state.trades_today += 1

        # Deduct fees from equity
        if fees > 0:
            self.state.current_equity -= fees
            self.state.daily_pnl -= fees

        logger.info(
            "Position opened",
            market_id=market_id,
            side=side,
            size=size,
            price=price,
            fees=fees,
        )

    def close_position(self, market_id: str, exit_price: float) -> float:
        """
        Close a position and return realized P&L.
        """
        if market_id not in self.state.positions:
            logger.warning("No position to close", market_id=market_id)
            return 0.0

        position = self.state.positions.pop(market_id)
        realized_pnl = (exit_price - position.entry_price) * position.size

        # Update state
        self.state.daily_pnl += realized_pnl
        self.state.current_equity += realized_pnl
        self.state.peak_equity = max(self.state.peak_equity, self.state.current_equity)

        logger.info(
            "Position closed",
            market_id=market_id,
            entry_price=position.entry_price,
            exit_price=exit_price,
            pnl=realized_pnl,
            daily_pnl=self.state.daily_pnl,
        )

        return realized_pnl

    def update_position_value(self, market_id: str, current_price: float) -> None:
        """Update mark-to-market value of a position."""
        if market_id in self.state.positions:
            position = self.state.positions[market_id]
            position.current_value = position.size * current_price

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of each day)."""
        self.state.daily_pnl = 0.0
        self.state.trades_today = 0
        self.state.last_reset_date = datetime.now(timezone.utc)
        logger.info("Daily risk counters reset")

    def get_summary(self) -> dict:
        """Get summary of current risk state."""
        return {
            "current_equity": self.state.current_equity,
            "daily_pnl": self.state.daily_pnl,
            "drawdown": self.state.current_drawdown,
            "positions": len(self.state.positions),
            "trades_today": self.state.trades_today,
            "can_trade": self.can_trade()[0],
        }
