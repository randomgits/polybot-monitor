"""RL Agent wrapper using Stable Baselines3."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback

from polybot.rl.environment import PolymarketEnv

logger = structlog.get_logger()


class RLStats:
    """Track and persist RL training statistics."""

    def __init__(self, stats_path: Path | None = None):
        self.stats_path = stats_path or Path("./models/rl_stats.json")
        self.stats = {
            "total_predictions": 0,
            "exploration_count": 0,
            "exploitation_count": 0,
            "action_counts": {"HOLD": 0, "BUY_YES": 0, "BUY_NO": 0, "CLOSE": 0},
            "total_training_steps": 0,
            "training_sessions": 0,
            "total_reward": 0.0,
            "episode_rewards": [],
            "last_save": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._load()

    def _load(self) -> None:
        """Load stats from disk."""
        if self.stats_path.exists():
            try:
                with open(self.stats_path) as f:
                    saved = json.load(f)
                    self.stats.update(saved)
                logger.info("Loaded RL stats", path=str(self.stats_path))
            except Exception as e:
                logger.warning("Failed to load RL stats", error=str(e))

    def save(self) -> None:
        """Save stats to disk."""
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats["last_save"] = datetime.now(timezone.utc).isoformat()
        with open(self.stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)

    def record_prediction(self, action: str, explored: bool) -> None:
        """Record a prediction."""
        self.stats["total_predictions"] += 1
        if explored:
            self.stats["exploration_count"] += 1
        else:
            self.stats["exploitation_count"] += 1
        self.stats["action_counts"][action] += 1

        # Auto-save every 10 predictions
        if self.stats["total_predictions"] % 10 == 0:
            self.save()

    def record_training(self, steps: int, reward: float) -> None:
        """Record a training step."""
        self.stats["total_training_steps"] += steps
        self.stats["training_sessions"] += 1
        self.stats["total_reward"] += reward

    def record_episode(self, reward: float) -> None:
        """Record episode completion."""
        self.stats["episode_rewards"].append(reward)
        # Keep only last 100 episodes
        if len(self.stats["episode_rewards"]) > 100:
            self.stats["episode_rewards"] = self.stats["episode_rewards"][-100:]

    @property
    def exploration_rate(self) -> float:
        """Calculate actual exploration rate."""
        total = self.stats["total_predictions"]
        if total == 0:
            return 0.0
        return self.stats["exploration_count"] / total

    def summary(self) -> dict:
        """Get summary statistics."""
        recent_rewards = self.stats["episode_rewards"][-10:] if self.stats["episode_rewards"] else []
        return {
            "predictions": self.stats["total_predictions"],
            "exploration_pct": f"{self.exploration_rate * 100:.1f}%",
            "training_steps": self.stats["total_training_steps"],
            "avg_reward_10": np.mean(recent_rewards) if recent_rewards else 0.0,
            "action_dist": self.stats["action_counts"],
        }


class TradingCallback(BaseCallback):
    """Callback for logging training progress."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def _on_step(self) -> bool:
        # Log episode completion
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][i]
                    ep_reward = info.get("episode", {}).get("r", 0)
                    ep_length = info.get("episode", {}).get("l", 0)
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)

                    if len(self.episode_rewards) % 10 == 0:
                        recent_rewards = self.episode_rewards[-10:]
                        avg_reward = np.mean(recent_rewards)
                        logger.info(
                            "Training progress",
                            episodes=len(self.episode_rewards),
                            avg_reward_10=avg_reward,
                        )
        return True


class TradingAgent:
    """
    RL Agent for Polymarket trading.

    Wraps Stable Baselines3 algorithms (PPO or DQN) with trading-specific
    functionality like online learning and action selection.
    """

    def __init__(
        self,
        env: PolymarketEnv,
        algorithm: str = "PPO",
        learning_rate: float = 3e-4,
        exploration_rate: float = 0.1,
        model_path: Path | None = None,
        verbose: bool = True,
    ):
        self.env = env
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.model_path = model_path
        self.verbose = verbose

        self._model: PPO | DQN | None = None
        self._training_steps = 0
        self._last_save_step = 0
        self._save_interval = 100  # Save every N training steps

        # Stats tracking
        if model_path is None:
            stats_path = Path("./models/rl_stats.json")
        else:
            # Handle both ./models/trading_agent and ./models/trading_agent.zip
            stats_path = model_path.parent / "rl_stats.json"
        self.stats = RLStats(stats_path)

        # Initialize or load model
        # Stable Baselines3 saves models with .zip extension
        model_zip = model_path.with_suffix(".zip") if model_path else None
        if model_path and (model_path.exists() or (model_zip and model_zip.exists())):
            load_path = model_zip if (model_zip and model_zip.exists()) else model_path
            self.load(load_path)
        else:
            self._create_model()

    def _create_model(self) -> None:
        """Create a new model."""
        # Larger network now that features are normalized
        # (128, 64) gives ~10k parameters - good for 14 features, 4 actions
        policy_kwargs = {
            "net_arch": [128, 64],
        }

        if self.algorithm == "PPO":
            self._model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                n_steps=256,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=0,
                policy_kwargs=policy_kwargs,
            )
        elif self.algorithm == "DQN":
            self._model = DQN(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                buffer_size=10000,
                learning_starts=100,
                batch_size=64,
                gamma=0.99,
                exploration_fraction=0.3,
                exploration_final_eps=self.exploration_rate,
                verbose=0,
                policy_kwargs=policy_kwargs,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        logger.info("Created new model", algorithm=self.algorithm, net_arch=[128, 64])

    @property
    def model(self) -> PPO | DQN:
        """Get the underlying model."""
        if self._model is None:
            raise RuntimeError("Model not initialized")
        return self._model

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, dict[str, Any]]:
        """
        Predict action for given observation.

        Args:
            observation: Current state observation
            deterministic: If True, use greedy action selection

        Returns:
            (action, info dict)
        """
        # Epsilon-greedy exploration during online learning
        explored = False
        if not deterministic and np.random.random() < self.exploration_rate:
            action = self.env.action_space.sample()
            explored = True
        else:
            action, _states = self.model.predict(observation, deterministic=deterministic)
            action = int(action)

        action_name = self.env.action_labels[action]

        # Record stats
        self.stats.record_prediction(action_name, explored)

        info = {
            "exploration": explored,
            "action_name": action_name,
            "buffer_size": len(self.env.experience_buffer),
            "total_predictions": self.stats.stats["total_predictions"],
        }

        if self.verbose:
            mode = "EXPLORE" if explored else "EXPLOIT"
            logger.info(
                f"RL prediction: {action_name}",
                mode=mode,
                buffer_size=len(self.env.experience_buffer),
                exploration_rate=f"{self.stats.exploration_rate * 100:.1f}%",
            )

        return action, info

    def train_offline(
        self,
        total_timesteps: int = 10000,
        callback: BaseCallback | None = None,
    ) -> None:
        """
        Train on collected experience (offline training).

        This is used during the bootstrap phase with historical data.
        """
        if callback is None:
            callback = TradingCallback()

        logger.info("Starting offline training", timesteps=total_timesteps)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        self._training_steps += total_timesteps
        logger.info("Offline training complete", total_steps=self._training_steps)

    def train_online(self, batch_size: int = 64) -> dict[str, float]:
        """
        Perform one step of online learning from experience buffer.

        This is called periodically during live trading to update the policy.
        """
        buffer_size = len(self.env.experience_buffer)
        if buffer_size < batch_size:
            if self.verbose:
                logger.debug(
                    "Skipping training - insufficient data",
                    buffer_size=buffer_size,
                    required=batch_size,
                )
            return {"skipped": True, "buffer_size": buffer_size}

        # For PPO, we need to do a mini-training step
        # This is a simplified version - full online RL is more complex
        if isinstance(self.model, PPO):
            # PPO requires on-policy data, so we just continue training
            self.model.learn(total_timesteps=batch_size, reset_num_timesteps=False)
        elif isinstance(self.model, DQN):
            # DQN can use off-policy data from replay buffer
            self.model.learn(total_timesteps=batch_size, reset_num_timesteps=False)

        self._training_steps += batch_size

        # Record stats
        self.stats.record_training(batch_size, 0.0)  # TODO: track actual reward

        # Auto-save periodically
        if self._training_steps - self._last_save_step >= self._save_interval:
            self.save()
            self.stats.save()
            self._last_save_step = self._training_steps

        result = {
            "skipped": False,
            "steps": batch_size,
            "total_steps": self._training_steps,
            "buffer_size": buffer_size,
        }

        if self.verbose:
            logger.info(
                "Online training step completed",
                steps=batch_size,
                total_steps=self._training_steps,
                buffer_size=buffer_size,
            )

        return result

    def save(self, path: Path | None = None) -> None:
        """Save model and stats to disk."""
        save_path = path or self.model_path
        if save_path is None:
            save_path = Path("./models/trading_agent")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        self.stats.save()
        logger.info(
            "Model saved",
            path=str(save_path),
            training_steps=self._training_steps,
            predictions=self.stats.stats["total_predictions"],
        )

    def load(self, path: Path) -> None:
        """Load model from disk."""
        if self.algorithm == "PPO":
            self._model = PPO.load(str(path), env=self.env)
        elif self.algorithm == "DQN":
            self._model = DQN.load(str(path), env=self.env)

        logger.info(
            "Model loaded",
            path=str(path),
            stats=self.stats.summary(),
        )

    def print_status(self) -> None:
        """Print current RL agent status."""
        summary = self.stats.summary()
        print("\n" + "=" * 50)
        print("RL AGENT STATUS")
        print("=" * 50)
        print(f"Algorithm:        {self.algorithm}")
        print(f"Model path:       {self.model_path}")
        print(f"Training steps:   {self._training_steps}")
        print(f"Predictions:      {summary['predictions']}")
        print(f"Exploration:      {summary['exploration_pct']}")
        print(f"Buffer size:      {len(self.env.experience_buffer)}")
        print(f"Avg reward (10):  {summary['avg_reward_10']:.4f}")
        print(f"Action counts:    {summary['action_dist']}")
        print("=" * 50 + "\n")

    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """Get action probabilities for logging/analysis."""
        if isinstance(self.model, PPO):
            obs_tensor = self.model.policy.obs_to_tensor(observation)[0]
            with self.model.policy.evaluate_actions(obs_tensor, None) as result:
                # This is a simplified version
                pass
        # Return uniform for now
        return np.array([0.25, 0.25, 0.25, 0.25])


def create_agent(
    env: PolymarketEnv,
    algorithm: str = "PPO",
    model_path: Path | None = None,
) -> TradingAgent:
    """Factory function to create a trading agent."""
    return TradingAgent(
        env=env,
        algorithm=algorithm,
        model_path=model_path,
    )
