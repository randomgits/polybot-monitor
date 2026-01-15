"""
Chainlink BTC/USD price feed client.

Polymarket BTC 15-minute markets resolve using Chainlink as the oracle source.
This client fetches prices directly from the Chainlink aggregator on Polygon.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Chainlink BTC/USD Price Feed on Polygon
# https://data.chain.link/polygon/mainnet/crypto-usd/btc-usd
CHAINLINK_BTC_USD_POLYGON = "0xc907E116054Ad103354f2D350FD2514433D57F6f"

# Default Polygon RPC endpoints
DEFAULT_RPC_URLS = [
    "https://polygon-rpc.com",
    "https://rpc-mainnet.matic.network",
    "https://matic-mainnet.chainstacklabs.com",
    "https://polygon-mainnet.public.blastapi.io",
]

# Chainlink Aggregator V3 ABI (minimal for latestRoundData)
AGGREGATOR_V3_ABI = [
    {
        "inputs": [],
        "name": "latestRoundData",
        "outputs": [
            {"name": "roundId", "type": "uint80"},
            {"name": "answer", "type": "int256"},
            {"name": "startedAt", "type": "uint256"},
            {"name": "updatedAt", "type": "uint256"},
            {"name": "answeredInRound", "type": "uint80"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
]


@dataclass
class ChainlinkPriceUpdate:
    """Price update from Chainlink oracle."""

    price: float
    timestamp: datetime
    round_id: int
    decimals: int = 8

    @property
    def age_seconds(self) -> float:
        """Seconds since this update."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()


class ChainlinkClient:
    """
    Client for Chainlink BTC/USD price feed on Polygon.

    Uses web3.py to read from the Chainlink aggregator contract.
    Falls back to HTTP JSON-RPC if web3 is not available.
    """

    def __init__(self, rpc_url: str = ""):
        self.rpc_url = rpc_url or os.getenv("POLYGON_RPC_URL", DEFAULT_RPC_URLS[0])
        self._web3 = None
        self._contract = None
        self._decimals: int = 8
        self._last_update: Optional[ChainlinkPriceUpdate] = None
        self._callbacks: list[Callable[[ChainlinkPriceUpdate], None]] = []
        self._running = False
        self._connected = False

    @property
    def current_price(self) -> float:
        """Current BTC price from Chainlink."""
        return self._last_update.price if self._last_update else 0.0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def on_price_update(self, callback: Callable[[ChainlinkPriceUpdate], None]) -> None:
        """Register callback for price updates."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, update: ChainlinkPriceUpdate) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def connect(self) -> None:
        """Initialize web3 connection."""
        try:
            from web3 import Web3
            from web3.middleware import geth_poa_middleware

            # Try multiple RPC endpoints
            rpc_urls = [self.rpc_url] + [url for url in DEFAULT_RPC_URLS if url != self.rpc_url]

            for rpc_url in rpc_urls:
                try:
                    self._web3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))

                    # Polygon uses PoA, need this middleware
                    self._web3.middleware_onion.inject(geth_poa_middleware, layer=0)

                    if self._web3.is_connected():
                        self.rpc_url = rpc_url
                        logger.info(f"[Chainlink] Connected to Polygon RPC: {rpc_url}")

                        # Initialize contract
                        self._contract = self._web3.eth.contract(
                            address=Web3.to_checksum_address(CHAINLINK_BTC_USD_POLYGON),
                            abi=AGGREGATOR_V3_ABI,
                        )

                        # Get decimals
                        self._decimals = self._contract.functions.decimals().call()
                        self._connected = True
                        return

                except Exception as e:
                    logger.debug(f"[Chainlink] Failed to connect to {rpc_url}: {e}")
                    continue

            logger.error("[Chainlink] Failed to connect to any Polygon RPC")
            self._connected = False

        except ImportError:
            logger.warning("[Chainlink] web3 not installed, using HTTP fallback")
            self._connected = True  # Will use HTTP fallback

    async def disconnect(self) -> None:
        """Close connection."""
        self._running = False
        self._connected = False
        self._web3 = None
        self._contract = None

    async def get_latest_price(self) -> Optional[ChainlinkPriceUpdate]:
        """Fetch latest price from Chainlink aggregator."""
        try:
            if self._contract:
                # Use web3 contract call
                round_data = self._contract.functions.latestRoundData().call()
                round_id, answer, _, updated_at, _ = round_data

                price = answer / (10 ** self._decimals)
                timestamp = datetime.fromtimestamp(updated_at, tz=timezone.utc)

                update = ChainlinkPriceUpdate(
                    price=price,
                    timestamp=timestamp,
                    round_id=round_id,
                    decimals=self._decimals,
                )

            else:
                # Fallback to HTTP JSON-RPC
                update = await self._fetch_via_http()

            if update:
                self._last_update = update
                self._notify_callbacks(update)

            return update

        except Exception as e:
            logger.error(f"[Chainlink] Error fetching price: {e}")
            return self._last_update

    async def _fetch_via_http(self) -> Optional[ChainlinkPriceUpdate]:
        """Fetch price via HTTP JSON-RPC (fallback when web3 not available)."""
        import httpx

        # Encode latestRoundData() call
        # Function selector: 0xfeaf968c
        call_data = "0xfeaf968c"

        payload = {
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [
                {
                    "to": CHAINLINK_BTC_USD_POLYGON,
                    "data": call_data,
                },
                "latest",
            ],
            "id": 1,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self.rpc_url, json=payload)
                response.raise_for_status()
                result = response.json()

                if "result" not in result:
                    return None

                # Decode result (5 uint values packed)
                hex_result = result["result"]
                if hex_result == "0x" or len(hex_result) < 66:
                    return None

                # Remove 0x prefix and decode
                data = hex_result[2:]

                # Each value is 32 bytes (64 hex chars)
                round_id = int(data[0:64], 16)
                answer = int(data[64:128], 16)
                # started_at = int(data[128:192], 16)
                updated_at = int(data[192:256], 16)
                # answered_in_round = int(data[256:320], 16)

                # Handle signed int256 for answer
                if answer > 2**255:
                    answer = answer - 2**256

                price = answer / (10 ** self._decimals)
                timestamp = datetime.fromtimestamp(updated_at, tz=timezone.utc)

                return ChainlinkPriceUpdate(
                    price=price,
                    timestamp=timestamp,
                    round_id=round_id,
                    decimals=self._decimals,
                )

        except Exception as e:
            logger.error(f"[Chainlink] HTTP fetch error: {e}")
            return None

    async def run_price_stream(self, poll_interval: float = 5.0):
        """
        Poll Chainlink for price updates.

        Args:
            poll_interval: Seconds between polls (default 5s, Chainlink updates ~every heartbeat)
        """
        self._running = True

        while self._running:
            try:
                update = await self.get_latest_price()
                if update:
                    yield update

            except asyncio.CancelledError:
                return

            except Exception as e:
                logger.error(f"[Chainlink] Stream error: {e}")

            await asyncio.sleep(poll_interval)
