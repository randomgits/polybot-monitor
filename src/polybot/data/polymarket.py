"""
Polymarket CLOB client for BTC 15-minute prediction markets.

Provides:
- Market discovery (finding active BTC 15-min markets)
- Real-time order book data via WebSocket
- REST API for market metadata
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator, Callable, Optional

import httpx
import websockets

from polybot.data.models import BTC15MinMarket

logger = logging.getLogger(__name__)

# API endpoints
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def get_proxy_url() -> Optional[str]:
    """Get HTTP proxy URL for Polymarket API."""
    proxy_host = os.getenv("PROXY_HOST", "")
    proxy_port = os.getenv("PROXY_PORT", "")
    proxy_user = os.getenv("PROXY_USER", "")
    proxy_pass = os.getenv("PROXY_PASS", "")

    if not proxy_host or not proxy_port:
        return None

    if proxy_user and proxy_pass:
        return f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"
    return f"http://{proxy_host}:{proxy_port}"


@dataclass
class OrderBookUpdate:
    """Order book update from WebSocket."""

    token_id: str
    timestamp_ms: int
    best_bid: float
    best_ask: float
    bid_size: float
    ask_size: float

    @property
    def mid_price(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid or self.best_ask

    @property
    def spread(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return self.best_ask - self.best_bid
        return 0.0


def clean_json_response(data):
    """Recursively clean JSON response from API quirks."""
    if isinstance(data, dict):
        return {k: clean_json_response(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_response(item) for item in data]
    elif isinstance(data, str):
        # Try to parse nested JSON strings
        if data.startswith("{") or data.startswith("["):
            try:
                return clean_json_response(json.loads(data))
            except json.JSONDecodeError:
                pass
        # Convert string booleans
        if data.lower() == "true":
            return True
        if data.lower() == "false":
            return False
        if data.lower() == "null":
            return None
        # Try to convert numeric strings
        try:
            if "." in data:
                return float(data)
            return int(data)
        except ValueError:
            pass
    return data


class PolymarketClient:
    """
    Client for Polymarket BTC 15-minute prediction markets.

    Provides both REST API for market discovery and WebSocket for real-time updates.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase

        self._http_client: Optional[httpx.AsyncClient] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected = False
        self._current_market: Optional[BTC15MinMarket] = None
        self._order_books: dict[str, OrderBookUpdate] = {}
        self._callbacks: list[Callable[[OrderBookUpdate], None]] = []
        self._running = False

        # Proxy configuration
        self._proxy_url = get_proxy_url()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Initialize HTTP client."""
        transport = None
        if self._proxy_url:
            transport = httpx.AsyncHTTPTransport(proxy=self._proxy_url)
            logger.info(f"[Polymarket] Using proxy: {self._proxy_url.split('@')[-1]}")

        self._http_client = httpx.AsyncClient(
            timeout=30.0,
            transport=transport,
        )
        logger.info("[Polymarket] HTTP client initialized")

    async def disconnect(self) -> None:
        """Close connections."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._connected = False

    def on_orderbook_update(self, callback: Callable[[OrderBookUpdate], None]) -> None:
        """Register callback for order book updates."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, update: OrderBookUpdate) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def get_btc_15m_markets(self) -> list[BTC15MinMarket]:
        """
        Discover active BTC 15-minute prediction markets.

        BTC 15-min markets use timestamp-based slugs like 'btc-updown-15m-{timestamp}'
        and must be queried directly - they don't appear in general market listings.

        Returns list of markets sorted by expiry time (soonest first).
        """
        if not self._http_client:
            await self.connect()

        markets = []
        now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())

        # Calculate timestamps for current and upcoming 15-minute windows
        interval = 15 * 60  # 15 minutes in seconds
        current_window_end = ((now_ts // interval) + 1) * interval

        # Check current and next few windows
        timestamps_to_check = [
            current_window_end,
            current_window_end + interval,
            current_window_end + interval * 2,
        ]

        for ts in timestamps_to_check:
            slug = f"btc-updown-15m-{ts}"

            try:
                response = await self._http_client.get(
                    f"{GAMMA_API_URL}/markets",
                    params={"slug": slug}
                )
                response.raise_for_status()
                data = clean_json_response(response.json())

                if not data:
                    continue

                # Handle both list and single object responses
                market_list = data if isinstance(data, list) else [data]

                for market_data in market_list:
                    question = market_data.get("question", "")

                    # Parse end time - use endDate (full timestamp) not endDateIso (date only)
                    end_time_str = market_data.get("endDate") or market_data.get("end_date")
                    if not end_time_str:
                        continue

                    try:
                        # Handle various datetime formats from API
                        end_time_str_clean = end_time_str.replace("Z", "+00:00")
                        end_time = datetime.fromisoformat(end_time_str_clean)
                        # Ensure timezone-aware for comparison
                        if end_time.tzinfo is None:
                            end_time = end_time.replace(tzinfo=timezone.utc)
                    except (ValueError, AttributeError):
                        continue

                    # Skip expired or closed markets
                    if end_time <= now:
                        continue

                    if market_data.get("closed"):
                        continue

                    # Get token IDs
                    clob_token_ids = market_data.get("clobTokenIds", [])
                    if not clob_token_ids or len(clob_token_ids) < 2:
                        continue

                    # Parse prices
                    outcome_prices = market_data.get("outcomePrices", ["0.5", "0.5"])
                    try:
                        yes_price = float(outcome_prices[0]) if outcome_prices else 0.5
                        no_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else 0.5
                    except (ValueError, TypeError):
                        yes_price = 0.5
                        no_price = 0.5

                    # Parse start price from question (fallback only)
                    start_price = self._parse_strike_from_question(question)

                    # Parse window start time (eventStartTime)
                    window_start = None
                    window_start_str = market_data.get("eventStartTime")
                    if window_start_str:
                        try:
                            window_start_clean = window_start_str.replace("Z", "+00:00")
                            window_start = datetime.fromisoformat(window_start_clean)
                            if window_start.tzinfo is None:
                                window_start = window_start.replace(tzinfo=timezone.utc)
                        except (ValueError, AttributeError):
                            pass

                    market = BTC15MinMarket(
                        market_id=str(market_data.get("id", "")),
                        condition_id=str(market_data.get("conditionId", "")),
                        question=question,
                        token_id=str(clob_token_ids[0]),  # YES token
                        no_token_id=str(clob_token_ids[1]),  # NO token
                        yes_price=yes_price,
                        no_price=no_price,
                        start_price=start_price,
                        end_time=end_time,
                        window_start=window_start,
                        slug=slug,
                    )
                    markets.append(market)
                    logger.info(f"[Polymarket] Found market: {slug} expires at {end_time}")

            except httpx.HTTPError as e:
                logger.debug(f"[Polymarket] No market at {slug}: {e}")
            except Exception as e:
                logger.error(f"[Polymarket] Error fetching market {slug}: {e}")

        # Sort by expiry time (soonest first)
        markets.sort(key=lambda m: m.end_time)

        logger.info(f"[Polymarket] Found {len(markets)} active BTC 15-min markets")
        return markets

    def _parse_strike_from_question(self, question: str) -> float:
        """Extract strike price from market question."""
        # Match patterns like "$95,000", "$95000", "$95K"
        patterns = [
            r"\$([0-9,]+(?:\.[0-9]+)?)\s*([KMB])?",
            r"([0-9,]+(?:\.[0-9]+)?)\s*(?:USD|dollars)",
        ]

        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                try:
                    price_str = match.group(1).replace(",", "")
                    price = float(price_str)

                    # Handle K/M/B suffixes
                    suffix = match.group(2) if len(match.groups()) > 1 else None
                    if suffix:
                        suffix = suffix.upper()
                        if suffix == "K":
                            price *= 1_000
                        elif suffix == "M":
                            price *= 1_000_000
                        elif suffix == "B":
                            price *= 1_000_000_000

                    return price
                except (ValueError, IndexError):
                    continue

        return 0.0

    async def get_order_book(self, token_id: str) -> Optional[OrderBookUpdate]:
        """Fetch current order book for a token."""
        if not self._http_client:
            await self.connect()

        try:
            response = await self._http_client.get(
                f"{CLOB_API_URL}/book",
                params={"token_id": token_id}
            )
            response.raise_for_status()
            data = clean_json_response(response.json())

            bids = data.get("bids", [])
            asks = data.get("asks", [])

            best_bid = float(bids[0]["price"]) if bids else 0.0
            best_ask = float(asks[0]["price"]) if asks else 1.0
            bid_size = float(bids[0]["size"]) if bids else 0.0
            ask_size = float(asks[0]["size"]) if asks else 0.0

            update = OrderBookUpdate(
                token_id=token_id,
                timestamp_ms=int(time.time() * 1000),
                best_bid=best_bid,
                best_ask=best_ask,
                bid_size=bid_size,
                ask_size=ask_size,
            )

            self._order_books[token_id] = update
            return update

        except Exception as e:
            logger.error(f"[Polymarket] Error fetching order book: {e}")
            return None

    async def subscribe_to_market(self, market: BTC15MinMarket) -> None:
        """Subscribe to WebSocket updates for a market."""
        self._current_market = market

        try:
            logger.info(f"[Polymarket] Subscribing to market: {market.market_id[:20]}...")

            # Connect to WebSocket
            self._ws = await websockets.connect(
                CLOB_WS_URL,
                ping_interval=20,
                ping_timeout=20,
                open_timeout=15,
            )

            # Subscribe to both YES and NO tokens
            for token_id in [market.token_id, market.no_token_id]:
                subscribe_msg = {
                    "type": "subscribe",
                    "channel": "market",
                    "assets_ids": [token_id],
                }
                await self._ws.send(json.dumps(subscribe_msg))

            self._connected = True
            logger.info("[Polymarket] WebSocket subscribed successfully")

        except Exception as e:
            logger.error(f"[Polymarket] WebSocket subscription error: {e}")
            self._connected = False

    async def run_orderbook_stream(self) -> AsyncGenerator[OrderBookUpdate, None]:
        """
        Stream order book updates via WebSocket.

        Falls back to REST polling if WebSocket fails.
        """
        self._running = True

        while self._running:
            try:
                # Try WebSocket first
                if self._ws and self._connected:
                    try:
                        raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
                        msg = json.loads(raw)

                        if msg.get("type") == "book":
                            token_id = msg.get("asset_id", "")
                            bids = msg.get("bids", [])
                            asks = msg.get("asks", [])

                            best_bid = float(bids[0]["price"]) if bids else 0.0
                            best_ask = float(asks[0]["price"]) if asks else 1.0
                            bid_size = float(bids[0]["size"]) if bids else 0.0
                            ask_size = float(asks[0]["size"]) if asks else 0.0

                            update = OrderBookUpdate(
                                token_id=token_id,
                                timestamp_ms=int(time.time() * 1000),
                                best_bid=best_bid,
                                best_ask=best_ask,
                                bid_size=bid_size,
                                ask_size=ask_size,
                            )

                            self._order_books[token_id] = update
                            self._notify_callbacks(update)
                            yield update

                    except asyncio.TimeoutError:
                        # No message received, continue
                        pass

                else:
                    # Fallback to REST polling
                    if self._current_market:
                        for token_id in [self._current_market.token_id, self._current_market.no_token_id]:
                            update = await self.get_order_book(token_id)
                            if update:
                                self._notify_callbacks(update)
                                yield update

                    await asyncio.sleep(1.0)  # Poll every second

            except asyncio.CancelledError:
                return

            except Exception as e:
                logger.error(f"[Polymarket] Stream error: {e}")
                self._connected = False
                await asyncio.sleep(2.0)

    def get_current_prices(self) -> tuple[float, float]:
        """Get current YES and NO prices from cached order books."""
        if not self._current_market:
            return 0.5, 0.5

        yes_book = self._order_books.get(self._current_market.token_id)
        no_book = self._order_books.get(self._current_market.no_token_id)

        yes_price = yes_book.mid_price if yes_book else 0.5
        no_price = no_book.mid_price if no_book else 0.5

        return yes_price, no_price

    def get_spread(self) -> float:
        """Get combined spread from both YES and NO order books."""
        if not self._current_market:
            return 0.0

        yes_book = self._order_books.get(self._current_market.token_id)
        no_book = self._order_books.get(self._current_market.no_token_id)

        yes_spread = yes_book.spread if yes_book else 0.0
        no_spread = no_book.spread if no_book else 0.0

        return (yes_spread + no_spread) / 2
