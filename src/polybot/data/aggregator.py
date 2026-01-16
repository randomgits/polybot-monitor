"""
Data aggregator that combines all data sources into MarketState.

Orchestrates:
- Binance/exchange price feeds (real-time BTC prices)
- Polymarket order books (prediction market odds)
- Chainlink oracle (resolution source)
- Position management for start price tracking
- Paper trading for strategy validation
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from polybot.data.models import MarketState, BTC15MinMarket, Opportunity
from polybot.data.binance import BinanceClient, BinanceTick
from polybot.data.polymarket import PolymarketClient, OrderBookUpdate
from polybot.data.chainlink import ChainlinkClient, ChainlinkPriceUpdate
from polybot.trading.position_manager import PositionManager
from polybot.trading.paper_trader import PaperTrader

logger = logging.getLogger(__name__)


class DataAggregator:
    """
    Aggregates data from multiple sources into unified MarketState.

    Manages:
    - Market discovery (finding active BTC 15-min markets)
    - Real-time data streaming from all sources
    - State updates to registered callbacks
    """

    def __init__(
        self,
        polymarket_client: PolymarketClient,
        binance_client: BinanceClient,
        chainlink_client: ChainlinkClient,
        enable_paper_trading: bool = True,
        paper_trading_balance: float = 1000.0,
        data_dir: Optional[Path] = None,
    ):
        self._polymarket = polymarket_client
        self._binance = binance_client
        self._chainlink = chainlink_client

        # Current state
        self._current_market: Optional[BTC15MinMarket] = None
        self._current_state: Optional[MarketState] = None

        # Price caches
        self._btc_price_binance: float = 0.0
        self._btc_price_chainlink: float = 0.0
        self._polymarket_yes_price: float = 0.5
        self._polymarket_no_price: float = 0.5

        # State update callbacks
        self._callbacks: list[Callable[[MarketState], None]] = []

        # Running state
        self._running = False
        self._tasks: list[asyncio.Task] = []

        # Opportunity tracking
        self._opportunities: list[Opportunity] = []
        self._max_opportunities = 100

        # Position management for start price tracking
        self._position_manager = PositionManager(data_dir=data_dir)

        # Paper trading
        self._enable_paper_trading = enable_paper_trading
        self._paper_trader: Optional[PaperTrader] = None
        if enable_paper_trading:
            self._paper_trader = PaperTrader(
                initial_balance=paper_trading_balance,
                data_dir=data_dir,
            )

        # Track captured start price per market
        self._market_start_prices: dict[str, float] = {}

    @property
    def current_market(self) -> Optional[BTC15MinMarket]:
        """Currently tracked market."""
        return self._current_market

    @property
    def current_state(self) -> Optional[MarketState]:
        """Current aggregated market state."""
        return self._current_state

    @property
    def opportunities(self) -> list[Opportunity]:
        """Detected trading opportunities."""
        return self._opportunities

    @property
    def paper_trader(self) -> Optional[PaperTrader]:
        """Paper trader instance."""
        return self._paper_trader

    @property
    def position_manager(self) -> PositionManager:
        """Position manager instance."""
        return self._position_manager

    def on_state_update(self, callback: Callable[[MarketState], None]) -> None:
        """Register callback for state updates."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, state: MarketState) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    async def start(self) -> None:
        """Start data aggregation."""
        logger.info("[Aggregator] Starting data aggregation...")
        self._running = True

        # Discover market first
        await self._discover_market()

        # Start data streams
        self._tasks = [
            asyncio.create_task(self._run_binance_stream()),
            asyncio.create_task(self._run_chainlink_stream()),
            asyncio.create_task(self._run_polymarket_stream()),
            asyncio.create_task(self._run_market_monitor()),
        ]

        logger.info("[Aggregator] Data streams started")

    async def stop(self) -> None:
        """Stop data aggregation."""
        logger.info("[Aggregator] Stopping...")
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks = []
        logger.info("[Aggregator] Stopped")

    async def _discover_market(self) -> None:
        """Find and subscribe to the best BTC 15-min market."""
        try:
            markets = await self._polymarket.get_btc_15m_markets()

            if not markets:
                logger.warning("[Aggregator] No active BTC 15-min markets found")
                return

            # Pick the market with most time remaining (but not too far out)
            # Prefer markets with 5-15 minutes remaining
            best_market = None
            for market in markets:
                time_remaining = market.time_to_expiry_seconds
                if 60 < time_remaining < 900:  # 1-15 minutes
                    if best_market is None or time_remaining > best_market.time_to_expiry_seconds:
                        best_market = market

            # Fallback to any market
            if best_market is None and markets:
                best_market = markets[0]

            if best_market:
                self._current_market = best_market
                await self._polymarket.subscribe_to_market(best_market)

                # Fetch initial order books
                await self._polymarket.get_order_book(best_market.token_id)
                await self._polymarket.get_order_book(best_market.no_token_id)

                # Register market with position manager for start price tracking
                if best_market.window_start:
                    self._position_manager.track_market(
                        market_id=best_market.market_id,
                        slug=best_market.slug,
                        question=best_market.question,
                        window_start=best_market.window_start,
                        window_end=best_market.end_time,
                        yes_token_id=best_market.token_id,
                        no_token_id=best_market.no_token_id,
                    )

                logger.info(
                    f"[Aggregator] Tracking market: {best_market.question[:50]}... "
                    f"(expires in {best_market.time_to_expiry_seconds:.0f}s)"
                )

        except Exception as e:
            logger.error(f"[Aggregator] Market discovery error: {e}")

    async def _run_binance_stream(self) -> None:
        """Stream Binance price updates."""
        try:
            async for tick in self._binance.run():
                if not self._running:
                    break

                self._btc_price_binance = tick.last_price
                self._update_state()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[Aggregator] Binance stream error: {e}")

    async def _run_chainlink_stream(self) -> None:
        """Poll Chainlink price updates."""
        await self._chainlink.connect()

        try:
            async for update in self._chainlink.run_price_stream(poll_interval=5.0):
                if not self._running:
                    break

                self._btc_price_chainlink = update.price

                # Record price for start price capture
                self._position_manager.record_price(update.price)

                # Try to capture start price for current market
                if self._current_market:
                    captured = self._position_manager.try_capture_start_price(
                        self._current_market.market_id,
                        update.price,
                    )
                    if captured and self._current_market.market_id not in self._market_start_prices:
                        self._market_start_prices[self._current_market.market_id] = captured
                        logger.info(
                            f"[Aggregator] Captured start price for market: ${captured:,.2f}"
                        )

                self._update_state()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[Aggregator] Chainlink stream error: {e}")

    async def _run_polymarket_stream(self) -> None:
        """Stream Polymarket order book updates."""
        try:
            async for update in self._polymarket.run_orderbook_stream():
                if not self._running:
                    break

                # Update prices based on token
                if self._current_market:
                    if update.token_id == self._current_market.token_id:
                        self._polymarket_yes_price = update.mid_price
                    elif update.token_id == self._current_market.no_token_id:
                        self._polymarket_no_price = update.mid_price

                self._update_state()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[Aggregator] Polymarket stream error: {e}")

    async def _run_market_monitor(self) -> None:
        """Monitor for market expiry and switch to new markets."""
        while self._running:
            try:
                await asyncio.sleep(30.0)  # Check every 30 seconds

                if self._current_market:
                    time_remaining = self._current_market.time_to_expiry_seconds

                    if time_remaining <= 0:
                        logger.info("[Aggregator] Market expired, discovering new market...")
                        await self._discover_market()

                    elif time_remaining < 60:
                        logger.info(f"[Aggregator] Market expiring in {time_remaining:.0f}s")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Aggregator] Market monitor error: {e}")

    def _update_state(self) -> None:
        """Update and broadcast current market state."""
        if not self._current_market:
            return

        # Get volatility from Binance
        vol_1m = self._binance.calculate_volatility(1)
        vol_5m = self._binance.calculate_volatility(5)
        vol_15m = self._binance.calculate_volatility(15)

        # Get order book imbalance
        imbalance = self._binance.calculate_orderbook_imbalance()

        # Calculate Chainlink-Binance spread
        cl_spread = self._btc_price_chainlink - self._btc_price_binance

        # Get Polymarket spread
        pm_spread = self._polymarket.get_spread()

        # Get start price - prioritize captured price from position manager
        market_id = self._current_market.market_id
        start_price = (
            self._market_start_prices.get(market_id)  # Captured at window start
            or self._position_manager.get_start_price(market_id)  # From position manager
            or self._current_market.start_price  # From API
            or self._btc_price_chainlink  # Fallback
        )

        # Build state
        state = MarketState(
            timestamp=datetime.now(timezone.utc),
            btc_price_binance=self._btc_price_binance,
            btc_price_chainlink=self._btc_price_chainlink,
            polymarket_yes_price=self._polymarket_yes_price,
            polymarket_no_price=self._polymarket_no_price,
            polymarket_spread=pm_spread,
            time_to_expiry=self._current_market.time_to_expiry_seconds,
            market_start_price=start_price,
            volatility_1m=vol_1m,
            volatility_5m=vol_5m,
            volatility_15m=vol_15m,
            binance_orderbook_imbalance=imbalance,
            chainlink_binance_spread=cl_spread,
            market_id=self._current_market.market_id,
            yes_token_id=self._current_market.token_id,
            no_token_id=self._current_market.no_token_id,
        )

        self._current_state = state
        self._notify_callbacks(state)

    def record_opportunity(self, opportunity: Opportunity) -> None:
        """Record a detected opportunity."""
        self._opportunities.append(opportunity)

        # Keep only recent opportunities
        if len(self._opportunities) > self._max_opportunities:
            self._opportunities = self._opportunities[-self._max_opportunities:]

    def get_status(self) -> dict:
        """Get current aggregator status."""
        # Get captured start price for current market
        start_price = None
        if self._current_market:
            market_id = self._current_market.market_id
            start_price = (
                self._market_start_prices.get(market_id)
                or self._position_manager.get_start_price(market_id)
                or self._current_market.start_price
            )

        return {
            "running": self._running,
            "market": {
                "id": self._current_market.market_id if self._current_market else None,
                "question": self._current_market.question[:50] if self._current_market else None,
                "time_to_expiry": self._current_market.time_to_expiry_seconds if self._current_market else None,
                "start_price": start_price,
                "window_started": self._current_market.is_window_started if self._current_market else None,
            } if self._current_market else None,
            "prices": {
                "btc_binance": self._btc_price_binance,
                "btc_chainlink": self._btc_price_chainlink,
                "btc_spread": self._btc_price_chainlink - self._btc_price_binance,
                "yes_price": self._polymarket_yes_price,
                "no_price": self._polymarket_no_price,
            },
            "volatility": {
                "1m": self._binance.calculate_volatility(1),
                "5m": self._binance.calculate_volatility(5),
                "15m": self._binance.calculate_volatility(15),
            },
            "connections": {
                "binance": self._binance.is_connected,
                "chainlink": self._chainlink.is_connected,
                "polymarket": self._current_market is not None,
            },
            "opportunities_count": len(self._opportunities),
            "paper_trading": self._paper_trader.get_stats() if self._paper_trader else None,
            "position_manager": self._position_manager.get_status(),
        }
