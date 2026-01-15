"""
Binance WebSocket client for real-time BTC price feeds.

Supports multiple exchanges with automatic fallback and proxy support
for bypassing geo-restrictions.
"""

import asyncio
import json
import logging
import os
import time
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncGenerator, Callable, Optional

import websockets

logger = logging.getLogger(__name__)


@dataclass
class BinanceTick:
    """Represents a single price ticker update."""

    symbol: str
    timestamp_ms: int
    last_price: float
    best_bid: float
    best_ask: float
    price_change_pct_24h: float
    high_24h: float
    low_24h: float
    volume_24h: float
    exchange: str = "binance"

    @property
    def timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp_ms / 1000, tz=timezone.utc)

    @property
    def mid_price(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return self.last_price


@dataclass
class VolatilityTracker:
    """Tracks price returns for volatility calculation."""

    prices: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add_price(self, price: float, timestamp_ms: int) -> None:
        self.prices.append(price)
        self.timestamps.append(timestamp_ms)

    def calculate_volatility(self, window_minutes: int) -> float:
        """Calculate annualized volatility over the given window."""
        if len(self.prices) < 2:
            return 0.5  # Default volatility

        now_ms = int(time.time() * 1000)
        window_ms = window_minutes * 60 * 1000

        # Filter prices within window
        window_prices = []
        for price, ts in zip(self.prices, self.timestamps):
            if now_ms - ts <= window_ms:
                window_prices.append(price)

        if len(window_prices) < 2:
            return 0.5

        # Calculate log returns
        returns = []
        for i in range(1, len(window_prices)):
            if window_prices[i - 1] > 0:
                ret = math.log(window_prices[i] / window_prices[i - 1])
                returns.append(ret)

        if len(returns) < 2:
            return 0.5

        # Calculate standard deviation of returns
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        std_ret = math.sqrt(variance)

        # Annualize: assume ~1 tick per second, 31,536,000 seconds per year
        # More conservative: use sqrt(samples_per_year) where samples = 60 * window_minutes
        samples_per_year = 365.25 * 24 * 60 * (60 / window_minutes)
        annualized_vol = std_ret * math.sqrt(samples_per_year)

        # Clamp to reasonable range
        return max(0.1, min(2.0, annualized_vol))


def get_proxy_url() -> Optional[str]:
    """Get proxy URL from environment variables."""
    proxy_host = os.getenv("BINANCE_PROXY_HOST", os.getenv("PROXY_HOST", ""))
    proxy_port = os.getenv("BINANCE_PROXY_PORT", os.getenv("PROXY_PORT", ""))
    proxy_user = os.getenv("BINANCE_PROXY_USER", os.getenv("PROXY_USER", ""))
    proxy_pass = os.getenv("BINANCE_PROXY_PASS", os.getenv("PROXY_PASS", ""))

    if not proxy_host or not proxy_port:
        return None

    if proxy_user and proxy_pass:
        return f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"
    return f"http://{proxy_host}:{proxy_port}"


def get_exchange_configs() -> list[dict]:
    """Get exchange configurations with proxy support."""
    configs = []
    proxy_url = get_proxy_url()

    # If proxy is configured, try Binance first
    if proxy_url:
        configs.append({
            "name": "Binance (proxy)",
            "ws_url": "wss://stream.binance.com:9443/ws/btcusdt@ticker",
            "subscribe_msg": None,
            "parser": "binance",
            "proxy": proxy_url,
        })

    # Fallback exchanges
    configs.extend([
        {
            "name": "Binance.US",
            "ws_url": "wss://stream.binance.us:9443/ws/btcusd@ticker",
            "subscribe_msg": None,
            "parser": "binance",
            "proxy": None,
        },
        {
            "name": "Kraken",
            "ws_url": "wss://ws.kraken.com",
            "subscribe_msg": {"event": "subscribe", "pair": ["XBT/USD"], "subscription": {"name": "ticker"}},
            "parser": "kraken",
            "proxy": None,
        },
        {
            "name": "Coinbase",
            "ws_url": "wss://ws-feed.exchange.coinbase.com",
            "subscribe_msg": {"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker"]},
            "parser": "coinbase",
            "proxy": None,
        },
    ])

    return configs


class BinanceClient:
    """
    WebSocket client for BTC price feeds with multi-exchange fallback.

    Provides:
    - Real-time BTC/USD prices
    - Volatility calculation (1m, 5m, 15m windows)
    - Order book imbalance (from bid/ask)
    - Automatic reconnection and fallback
    """

    def __init__(self):
        self._last_tick: Optional[BinanceTick] = None
        self._connected = False
        self._current_exchange: str = ""
        self._volatility_tracker = VolatilityTracker()
        self._callbacks: list[Callable[[BinanceTick], None]] = []
        self._running = False

    @property
    def current_price(self) -> float:
        """Current BTC price."""
        return self._last_tick.last_price if self._last_tick else 0.0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def calculate_volatility(self, window_minutes: int) -> float:
        """Get annualized volatility for the given window."""
        return self._volatility_tracker.calculate_volatility(window_minutes)

    def calculate_orderbook_imbalance(self) -> float:
        """
        Calculate order book imbalance from best bid/ask.

        Returns value from -1 (all sells) to 1 (all buys).
        """
        if not self._last_tick:
            return 0.0

        bid = self._last_tick.best_bid
        ask = self._last_tick.best_ask

        if bid <= 0 or ask <= 0:
            return 0.0

        # Simple imbalance based on price position between bid and ask
        mid = (bid + ask) / 2
        last = self._last_tick.last_price

        if last <= bid:
            return -1.0
        elif last >= ask:
            return 1.0
        else:
            # Normalize to -1 to 1
            return (last - mid) / ((ask - bid) / 2) if ask != bid else 0.0

    def on_tick(self, callback: Callable[[BinanceTick], None]) -> None:
        """Register a callback for price updates."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, tick: BinanceTick) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(tick)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _parse_binance(self, msg: dict, exchange_name: str = "Binance") -> Optional[BinanceTick]:
        """Parse Binance ticker message."""
        try:
            return BinanceTick(
                symbol=msg.get("s", "BTCUSD"),
                timestamp_ms=int(msg.get("E", time.time() * 1000)),
                last_price=float(msg.get("c", 0)),
                best_bid=float(msg.get("b", 0)),
                best_ask=float(msg.get("a", 0)),
                price_change_pct_24h=float(msg.get("P", 0)),
                high_24h=float(msg.get("h", 0)),
                low_24h=float(msg.get("l", 0)),
                volume_24h=float(msg.get("v", 0)),
                exchange=exchange_name,
            )
        except (ValueError, TypeError):
            return None

    def _parse_kraken(self, msg, exchange_name: str = "Kraken") -> Optional[BinanceTick]:
        """Parse Kraken ticker message."""
        try:
            if not isinstance(msg, list) or len(msg) < 2:
                return None
            data = msg[1]
            if not isinstance(data, dict):
                return None

            return BinanceTick(
                symbol="XBTUSD",
                timestamp_ms=int(time.time() * 1000),
                last_price=float(data.get("c", [0])[0]),
                best_bid=float(data.get("b", [0])[0]),
                best_ask=float(data.get("a", [0])[0]),
                price_change_pct_24h=0.0,
                high_24h=float(data.get("h", [0, 0])[1]),
                low_24h=float(data.get("l", [0, 0])[1]),
                volume_24h=float(data.get("v", [0, 0])[1]),
                exchange=exchange_name,
            )
        except (ValueError, TypeError, IndexError):
            return None

    def _parse_coinbase(self, msg: dict, exchange_name: str = "Coinbase") -> Optional[BinanceTick]:
        """Parse Coinbase ticker message."""
        try:
            if msg.get("type") != "ticker":
                return None
            return BinanceTick(
                symbol="BTCUSD",
                timestamp_ms=int(time.time() * 1000),
                last_price=float(msg.get("price", 0)),
                best_bid=float(msg.get("best_bid", 0)),
                best_ask=float(msg.get("best_ask", 0)),
                price_change_pct_24h=0.0,
                high_24h=float(msg.get("high_24h", 0)),
                low_24h=float(msg.get("low_24h", 0)),
                volume_24h=float(msg.get("volume_24h", 0)),
                exchange=exchange_name,
            )
        except (ValueError, TypeError):
            return None

    async def _try_exchange(self, config: dict) -> AsyncGenerator[BinanceTick, None]:
        """Try connecting to a specific exchange."""
        name = config["name"]
        url = config["ws_url"]
        subscribe_msg = config.get("subscribe_msg")
        parser = config["parser"]
        proxy = config.get("proxy")

        connect_kwargs = {
            "ping_interval": 20,
            "ping_timeout": 20,
            "open_timeout": 15,
            "close_timeout": 5,
        }

        # Add proxy if configured
        if proxy:
            connect_kwargs["proxy"] = proxy
            logger.info(f"[{name}] Connecting via proxy...")

        logger.info(f"[{name}] Connecting to {url}...")

        async with websockets.connect(url, **connect_kwargs) as ws:
            self._connected = True
            self._current_exchange = name

            if subscribe_msg:
                await ws.send(json.dumps(subscribe_msg))

            logger.info(f"[{name}] Connected to BTC price feed")

            while True:
                raw = await ws.recv()
                msg = json.loads(raw)

                tick = None
                if parser == "binance":
                    if isinstance(msg, dict) and "c" in msg:
                        tick = self._parse_binance(msg, name)
                elif parser == "kraken":
                    if isinstance(msg, list):
                        tick = self._parse_kraken(msg, name)
                elif parser == "coinbase":
                    if isinstance(msg, dict):
                        tick = self._parse_coinbase(msg, name)

                if tick and tick.last_price > 0:
                    self._last_tick = tick
                    self._volatility_tracker.add_price(tick.last_price, tick.timestamp_ms)
                    self._notify_callbacks(tick)
                    yield tick

    async def connect(self) -> None:
        """Connect to price feed (for compatibility)."""
        pass

    async def disconnect(self) -> None:
        """Disconnect from price feed."""
        self._running = False
        self._connected = False

    async def run(self) -> AsyncGenerator[BinanceTick, None]:
        """
        Async generator yielding price ticks.

        Tries multiple exchanges with automatic fallback.
        """
        self._running = True
        failed_exchanges: set[str] = set()
        consecutive_failures = 0
        exchange_configs = get_exchange_configs()

        while self._running:
            for config in exchange_configs:
                name = config["name"]

                if name in failed_exchanges:
                    continue

                try:
                    self._connected = False
                    async for tick in self._try_exchange(config):
                        if not self._running:
                            return
                        consecutive_failures = 0
                        yield tick

                except asyncio.CancelledError:
                    self._connected = False
                    return

                except Exception as e:
                    self._connected = False
                    error_str = str(e)

                    # Check for permanent errors
                    if "451" in error_str:
                        logger.info(f"[{name}] Geo-blocked (HTTP 451), trying next...")
                        failed_exchanges.add(name)
                        continue

                    if "403" in error_str or "401" in error_str:
                        logger.info(f"[{name}] Access denied, trying next...")
                        failed_exchanges.add(name)
                        continue

                    logger.info(f"[{name}] Connection error: {e}, trying next...")
                    consecutive_failures += 1
                    await asyncio.sleep(1.0)
                    continue

            # All exchanges failed
            if consecutive_failures > 0:
                logger.info("[Price Feed] All exchanges failed, retrying in 5s...")
                await asyncio.sleep(5.0)
                if consecutive_failures > 10:
                    failed_exchanges.clear()
                    consecutive_failures = 0


# Alias for compatibility with original bot.py import
CoinbaseClient = BinanceClient
