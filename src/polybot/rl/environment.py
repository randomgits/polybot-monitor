"""Gymnasium environment for Polymarket BTC 15-min trading."""

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from polybot.data.models import MarketState


class PolymarketEnv(gym.Env):
    """
    Gymnasium environment for Polymarket BTC 15-min prediction market trading.

    This environment can be used for:
    1. Offline training on historical data
    2. Online learning during live trading

    Observation Space:
    - BTC prices (Binance, Chainlink)
    - Volatility features (1m, 5m, 15m)
    - Polymarket odds
    - Time to expiry
    - Orderbook imbalance
    - Current position
    - Probability model estimate

    Action Space:
    - 0: HOLD (do nothing)
    - 1: BUY_YES (bet on UP)
    - 2: BUY_NO (bet on DOWN)
    - 3: CLOSE (close current position)

    Reward:
    - Actual P&L from real trades (passed in via set_reward)
    - Normalized by position size (percentage returns)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        initial_capital: float = 1000.0,
        max_position_pct: float = 0.1,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.render_mode = render_mode

        # Experience buffer for online learning
        self.experience_buffer: deque[dict[str, Any]] = deque(maxlen=10000)

        # Current state (will be updated by external data)
        self._current_state: MarketState | None = None
        self._model_probability: float = 0.5  # From probability model

        # Pending reward from actual P&L (set by bot when trades resolve)
        self._pending_reward: float = 0.0
        self._pending_done: bool = False

        # Define observation space
        # 13 features as defined in MarketState.to_array()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),  # 13 from MarketState + 1 for model probability
            dtype=np.float32,
        )

        # Define action space
        self.action_space = spaces.Discrete(4)  # HOLD, BUY_YES, BUY_NO, CLOSE

        # Action labels for logging
        self.action_labels = ["HOLD", "BUY_YES", "BUY_NO", "CLOSE"]

    def set_state(self, state: MarketState, model_probability: float) -> None:
        """
        Update the environment with current market state.

        Called by the data aggregator during live trading.
        """
        self._current_state = state
        self._model_probability = model_probability

    def set_reward(self, pnl: float, position_size: float, done: bool = False) -> None:
        """
        Set reward from actual P&L (called by bot when trades resolve).

        Args:
            pnl: Actual profit/loss in USD
            position_size: Position size in USD (for normalization)
            done: Whether this ends an episode (market resolved)
        """
        # Normalize to percentage return
        if position_size > 0:
            self._pending_reward = pnl / position_size
        else:
            self._pending_reward = 0.0
        self._pending_done = done

    def _get_observation(self) -> np.ndarray:
        """Convert current state to observation array."""
        if self._current_state is None:
            return np.zeros(14, dtype=np.float32)

        state_array = self._current_state.to_array()
        # Add model probability as additional feature
        obs = state_array + [self._model_probability]
        return np.array(obs, dtype=np.float32)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        self._pending_reward = 0.0
        self._pending_done = False

        obs = self._get_observation()
        info = {}

        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        The environment doesn't track positions - it receives actual P&L
        from the bot via set_reward() when trades resolve.

        Args:
            action: 0=HOLD, 1=BUY_YES, 2=BUY_NO, 3=CLOSE

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self._current_state is None:
            return self._get_observation(), 0.0, False, False, {}

        # Use pending reward from actual P&L (set by bot)
        reward = self._pending_reward
        terminated = self._pending_done

        # Reset pending reward after consuming it
        self._pending_reward = 0.0
        self._pending_done = False

        info: dict[str, Any] = {
            "action": self.action_labels[action],
            "reward": reward,
        }

        # Store experience for replay
        self._store_experience(action, reward, terminated)

        obs = self._get_observation()

        return obs, reward, terminated, False, info

    def _store_experience(
        self, action: int, reward: float, done: bool
    ) -> None:
        """Store experience for replay buffer."""
        if self._current_state is None:
            return

        experience = {
            "state": self._get_observation().copy(),
            "action": action,
            "reward": reward,
            "done": done,
            "timestamp": self._current_state.timestamp,
        }
        self.experience_buffer.append(experience)

    def get_experience_batch(self, batch_size: int) -> list[dict[str, Any]]:
        """Get a batch of experiences for training."""
        import random

        if len(self.experience_buffer) < batch_size:
            return list(self.experience_buffer)
        return random.sample(list(self.experience_buffer), batch_size)

    def render(self) -> None:
        """Render current state."""
        if self.render_mode != "human" or self._current_state is None:
            return

        print(f"\n=== Polymarket Environment ===")
        print(f"Buffer size: {len(self.experience_buffer)}")
        print(f"BTC (Chainlink): ${self._current_state.btc_price_chainlink:,.2f}")
        print(f"YES Price: {self._current_state.polymarket_yes_price:.3f}")
        print(f"Time to expiry: {self._current_state.time_to_expiry:.0f}s")
