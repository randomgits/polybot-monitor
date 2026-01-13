"""Main trading bot that orchestrates all components."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import structlog

from polybot.config import settings
from polybot.data.aggregator import DataAggregator
from polybot.data.chainlink import ChainlinkClient
from polybot.data.coinbase import CoinbaseClient
from polybot.data.models import MarketState
from polybot.data.polymarket import PolymarketClient
from polybot.execution.order_manager import OrderManager, OrderSide, OrderType, TradeExecutor
from polybot.execution.risk_manager import RiskLimits, RiskManager
from polybot.rl.agent import TradingAgent
from polybot.rl.environment import PolymarketEnv
from polybot.strategy.probability_model import ProbabilityModel, TradingSignal

logger = structlog.get_logger()


class TradingMode:
    """Trading mode configuration."""

    PAPER = "paper"  # Paper trading (no real money)
    PROBABILITY_MODEL = "probability_model"  # Only use probability model
    RL_ONLY = "rl_only"  # Only use RL agent
    HYBRID = "hybrid"  # Probability model + RL


class PolyBot:
    """
    Main trading bot for Polymarket BTC 15-min markets.

    Orchestrates:
    - Data aggregation from Polymarket, Binance, Chainlink
    - Signal generation from probability model
    - RL agent for decision making
    - Order execution and risk management
    """

    def __init__(
        self,
        mode: str = TradingMode.PAPER,
        initial_capital: float = 1000.0,
        use_rl: bool = True,
        model_path: Path | None = None,
    ):
        self.mode = mode
        self.initial_capital = initial_capital
        self.use_rl = use_rl
        self.model_path = model_path or Path("./models/trading_agent")

        # Components (initialized in start())
        self._polymarket: PolymarketClient | None = None
        self._coinbase: CoinbaseClient | None = None
        self._chainlink: ChainlinkClient | None = None
        self._aggregator: DataAggregator | None = None
        self._probability_model: ProbabilityModel | None = None
        self._risk_manager: RiskManager | None = None
        self._order_manager: OrderManager | None = None
        self._executor: TradeExecutor | None = None
        self._rl_env: PolymarketEnv | None = None
        self._rl_agent: TradingAgent | None = None

        # State tracking
        self._running = False
        self._current_state: MarketState | None = None
        self._current_signal: TradingSignal | None = None
        self._last_action_time: datetime | None = None
        self._action_cooldown_seconds = 5.0  # Minimum time between actions
        self._current_market_id: str | None = None  # Track current market for expiry detection

        # Statistics
        self._stats = {
            "signals_generated": 0,
            "trades_executed": 0,
            "total_pnl": 0.0,
            "markets_traded": set(),
            "wins": 0,
            "losses": 0,
        }

    async def start(self) -> None:
        """Initialize and start the trading bot."""
        logger.info("Starting PolyBot", mode=self.mode, capital=self.initial_capital)

        # Initialize clients
        self._polymarket = PolymarketClient(
            api_key=settings.polymarket_api_key,
            api_secret=settings.polymarket_secret,
            passphrase=settings.polymarket_passphrase,
        )
        await self._polymarket.__aenter__()

        self._coinbase = CoinbaseClient()
        self._chainlink = ChainlinkClient(rpc_url=settings.polygon_rpc_url)

        # Initialize data aggregator
        self._aggregator = DataAggregator(
            self._polymarket,
            self._coinbase,
            self._chainlink,
        )
        self._aggregator.on_state_update(self._on_state_update)

        # Initialize strategy components
        self._probability_model = ProbabilityModel(
            min_edge_threshold=settings.min_edge_threshold,
            kelly_fraction=settings.kelly_fraction,
        )

        # Initialize risk and execution
        risk_limits = RiskLimits(
            max_position_usd=settings.max_position_usd,
            daily_loss_limit_usd=settings.daily_loss_limit_usd,
        )
        self._risk_manager = RiskManager(
            limits=risk_limits,
            initial_capital=self.initial_capital,
        )

        paper_trading = self.mode == TradingMode.PAPER
        self._order_manager = OrderManager(
            self._polymarket,
            paper_trading=paper_trading,
        )
        self._executor = TradeExecutor(self._order_manager, self._risk_manager)

        # Initialize RL components if enabled
        if self.use_rl:
            self._rl_env = PolymarketEnv(
                initial_capital=self.initial_capital,
                max_position_pct=0.1,
            )
            self._rl_agent = TradingAgent(
                env=self._rl_env,
                algorithm="PPO",
                learning_rate=settings.rl_learning_rate,
                exploration_rate=settings.rl_exploration_rate,
                model_path=self.model_path,
            )

        # Start data aggregation
        await self._aggregator.start()

        self._running = True
        logger.info("PolyBot started successfully")

        # Start main trading loop
        await self._trading_loop()

    async def stop(self) -> None:
        """Stop the trading bot."""
        logger.info("Stopping PolyBot")
        self._running = False

        if self._aggregator:
            await self._aggregator.stop()

        if self._polymarket:
            await self._polymarket.__aexit__(None, None, None)

        # Save RL model and stats
        if self._rl_agent:
            self._rl_agent.save()
            self._rl_agent.stats.save()
            self._rl_agent.print_status()

        logger.info("PolyBot stopped", stats=self._stats)

    async def _trading_loop(self) -> None:
        """Main trading loop."""
        online_training_interval = 10  # Train every N signals (more frequent for learning)
        status_print_interval = 50  # Print RL status every N signals
        last_training = 0
        last_status_print = 0

        while self._running:
            try:
                # Process any pending state updates
                await asyncio.sleep(0.1)

                signals = self._stats["signals_generated"]

                # Periodic online training - use smaller batch to train sooner
                if (
                    self.use_rl
                    and self._rl_agent
                    and signals - last_training >= online_training_interval
                ):
                    # Use buffer size as batch, minimum 8 experiences
                    buffer_size = len(self._rl_env.experience_buffer) if self._rl_env else 0
                    batch_size = max(8, min(buffer_size, 32))
                    result = self._rl_agent.train_online(batch_size=batch_size)
                    if not result.get("skipped"):
                        last_training = signals
                        logger.info("Online training step", **result)

                # Periodic RL status print
                if (
                    self.use_rl
                    and self._rl_agent
                    and signals - last_status_print >= status_print_interval
                    and signals > 0
                ):
                    self._rl_agent.print_status()
                    last_status_print = signals

            except Exception as e:
                logger.error("Error in trading loop", error=str(e))
                await asyncio.sleep(1)

    def _on_state_update(self, state: MarketState) -> None:
        """Handle state update from data aggregator."""
        self._current_state = state

        # Check if market changed (previous market expired)
        market = self._aggregator.current_market
        if market and self._current_market_id and market.market_id != self._current_market_id:
            # Market changed - resolve any open positions from old market
            asyncio.create_task(self._resolve_expired_positions(self._current_market_id))

        # Update current market tracking
        if market:
            self._current_market_id = market.market_id

        # Generate signal from probability model
        signal = self._probability_model.generate_signal(state)
        self._current_signal = signal
        self._stats["signals_generated"] += 1

        # Update RL environment
        if self._rl_env:
            self._rl_env.set_state(state, signal.model_up_probability)

        # Check if we should trade
        asyncio.create_task(self._maybe_trade(state, signal))

    async def _resolve_expired_positions(self, market_id: str) -> None:
        """Resolve positions when a market expires."""
        if market_id not in self._risk_manager.state.positions:
            return

        position = self._risk_manager.state.positions[market_id]
        position_size = position.size * position.entry_price  # USD value

        # For now, we simulate resolution based on whether market went UP or DOWN
        # In reality, we'd fetch the actual resolution from Polymarket
        # Using last known Coinbase price vs start price as proxy
        if self._current_state:
            went_up = self._current_state.btc_price_binance >= self._current_state.market_start_price
        else:
            went_up = True  # Default assumption

        # Calculate payout
        if position.side == "YES":
            exit_price = 1.0 if went_up else 0.0
        else:  # NO position
            exit_price = 0.0 if went_up else 1.0

        # Close position with resolution price
        pnl = self._risk_manager.close_position(market_id, exit_price)

        # Update stats
        self._stats["total_pnl"] += pnl
        if pnl > 0:
            self._stats["wins"] += 1
        else:
            self._stats["losses"] += 1

        # Pass actual P&L to RL environment (normalized by position size)
        if self._rl_env:
            self._rl_env.set_reward(pnl, position_size, done=True)

        # Record reward in RL agent stats
        if self._rl_agent:
            self._rl_agent.stats.record_episode(pnl)

        logger.info(
            "Position resolved at market expiry",
            market_id=market_id,
            side=position.side,
            outcome="UP" if went_up else "DOWN",
            pnl=pnl,
            pnl_pct=f"{(pnl / position_size * 100):.1f}%" if position_size > 0 else "N/A",
            total_pnl=self._stats["total_pnl"],
            record=f"{self._stats['wins']}W-{self._stats['losses']}L",
        )

    async def _close_current_position(self, market_id: str) -> float:
        """Close current position at market price."""
        if market_id not in self._risk_manager.state.positions:
            return 0.0

        position = self._risk_manager.state.positions[market_id]
        position_size = position.size * position.entry_price  # USD value

        # Get current market price for the position side
        if self._current_state:
            if position.side == "YES":
                exit_price = self._current_state.polymarket_yes_price
            else:
                exit_price = self._current_state.polymarket_no_price
        else:
            exit_price = 0.5

        pnl = self._risk_manager.close_position(market_id, exit_price)
        self._stats["total_pnl"] += pnl

        # Pass actual P&L to RL environment (normalized by position size)
        if self._rl_env:
            self._rl_env.set_reward(pnl, position_size, done=False)

        # Record reward in RL agent stats
        if self._rl_agent:
            self._rl_agent.stats.record_episode(pnl)

        logger.info(
            "Position closed",
            market_id=market_id,
            side=position.side,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=f"{(pnl / position_size * 100):.1f}%" if position_size > 0 else "N/A",
        )

        return pnl

    async def _maybe_trade(self, state: MarketState, signal: TradingSignal) -> None:
        """
        Process RL agent decision and execute if appropriate.

        Key design: RL agent decides WHEN to trade. Risk manager decides HOW MUCH.
        Probability model is just a feature, not a decision maker.
        """
        # Cooldown check
        now = datetime.now(timezone.utc)
        if self._last_action_time:
            elapsed = (now - self._last_action_time).total_seconds()
            if elapsed < self._action_cooldown_seconds:
                return

        # Get current market info
        market = self._aggregator.current_market
        if not market:
            return

        # Get action from RL agent
        if not (self.use_rl and self._rl_agent and self._rl_env):
            return  # RL is required now

        obs = self._rl_env._get_observation()
        action, info = self._rl_agent.predict(obs, deterministic=False)
        action_name = info.get("action_name", "HOLD")

        # Log RL decision
        logger.info(
            "RL decision",
            btc_vs_target=f"${state.btc_price_binance - state.market_start_price:+.2f}",
            yes_price=f"{state.polymarket_yes_price:.3f}",
            model_prob=f"{signal.model_up_probability:.3f}",
            time_left=f"{state.time_to_expiry:.0f}s",
            rl_action=action_name,
            explored=info.get("exploration"),
        )

        # ALWAYS store experience for learning (even for HOLD)
        # This is critical - RL needs to learn from all decisions
        _, reward, _, _, step_info = self._rl_env.step(action)

        # Handle HOLD - no trade but experience was stored
        if action == 0:  # HOLD
            return

        # Handle CLOSE
        if action == 3:  # CLOSE
            if market.market_id in self._risk_manager.state.positions:
                pnl = await self._close_current_position(market.market_id)
                self._last_action_time = now
            return

        # Map action to string
        final_action = "BUY_YES" if action == 1 else "BUY_NO"

        # Check if we already have a position in this market
        has_position = market.market_id in self._risk_manager.state.positions
        if has_position:
            current_pos = self._risk_manager.state.positions[market.market_id]
            # If action is opposite to current position, close first
            if (final_action == "BUY_YES" and current_pos.side == "NO") or \
               (final_action == "BUY_NO" and current_pos.side == "YES"):
                await self._close_current_position(market.market_id)
            elif current_pos.side == ("YES" if final_action == "BUY_YES" else "NO"):
                # Already have position in same direction, skip
                return

        # Execute trade - RL decides, risk manager just enforces limits
        order = await self._executor.execute_rl_action(
            action=final_action,
            market_id=market.market_id,
            yes_token_id=market.token_id,
            no_token_id=market.token_id,  # TODO: Get actual NO token ID
            yes_price=state.polymarket_yes_price,
            no_price=state.polymarket_no_price,
            time_to_expiry=state.time_to_expiry,
            spread=state.polymarket_spread,
            available_capital=self._risk_manager.state.current_equity,
        )

        if order:
            self._last_action_time = now
            self._stats["trades_executed"] += 1
            self._stats["markets_traded"].add(market.market_id)

            logger.info(
                "Trade executed",
                action=final_action,
                order_id=order.order_id,
                size=order.filled_size,
                price=order.filled_price,
            )

    def get_status(self) -> dict:
        """Get current bot status."""
        return {
            "running": self._running,
            "mode": self.mode,
            "capital": self._risk_manager.state.current_equity if self._risk_manager else 0,
            "position": self._risk_manager.state.total_position_value if self._risk_manager else 0,
            "daily_pnl": self._risk_manager.state.daily_pnl if self._risk_manager else 0,
            "stats": {
                "signals": self._stats["signals_generated"],
                "trades": self._stats["trades_executed"],
                "total_pnl": self._stats["total_pnl"],
                "markets": len(self._stats["markets_traded"]),
            },
            "current_signal": {
                "model_prob": self._current_signal.model_up_probability,
                "market_prob": self._current_signal.market_up_probability,
                "edge": self._current_signal.edge,
                "action": self._current_signal.recommended_action,
            }
            if self._current_signal
            else None,
        }


async def main():
    """Entry point for running the bot."""
    import signal

    bot = PolyBot(
        mode=TradingMode.PAPER,
        initial_capital=1000.0,
        use_rl=True,
    )

    # Handle shutdown gracefully
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(bot.stop()))

    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
