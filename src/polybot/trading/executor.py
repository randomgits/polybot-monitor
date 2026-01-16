"""Trade executor for Polymarket CLOB.

Handles actual order placement and execution on Polymarket.
"""

import hmac
import hashlib
import time
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from enum import Enum

import httpx
import structlog

logger = structlog.get_logger()

# Polymarket CLOB API
CLOB_API_URL = "https://clob.polymarket.com"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


@dataclass
class Order:
    """Represents an order on Polymarket."""

    order_id: str
    token_id: str
    side: OrderSide
    size: float  # Number of shares
    price: float  # Price per share
    order_type: OrderType
    status: str  # "LIVE", "FILLED", "CANCELED", etc.
    filled_size: float = 0.0
    created_at: Optional[datetime] = None


@dataclass
class OrderResult:
    """Result of an order operation."""

    success: bool
    order_id: Optional[str] = None
    message: str = ""
    filled_size: float = 0.0
    avg_price: float = 0.0


class TradeExecutor:
    """
    Executes trades on Polymarket CLOB.

    IMPORTANT: Requires API credentials to trade.
    Set environment variables:
    - POLYMARKET_API_KEY
    - POLYMARKET_API_SECRET
    - POLYMARKET_PASSPHRASE

    The executor supports:
    - Market orders (immediate execution)
    - Limit orders (price-specific)
    - Order cancellation
    - Position queries
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
        dry_run: bool = True,  # Default to dry run for safety
    ):
        self.api_key = api_key or os.getenv("POLYMARKET_API_KEY")
        self.api_secret = api_secret or os.getenv("POLYMARKET_API_SECRET")
        self.passphrase = passphrase or os.getenv("POLYMARKET_PASSPHRASE")

        self.dry_run = dry_run
        self._client: Optional[httpx.AsyncClient] = None

        # Track orders
        self.pending_orders: dict[str, Order] = {}
        self.filled_orders: list[Order] = []

        if not all([self.api_key, self.api_secret, self.passphrase]):
            logger.warning(
                "[Executor] API credentials not configured - dry run mode only"
            )
            self.dry_run = True

        mode = "DRY RUN" if self.dry_run else "LIVE"
        logger.info(f"[Executor] Initialized in {mode} mode")

    async def connect(self) -> None:
        """Initialize HTTP client."""
        if not self._client:
            self._client = httpx.AsyncClient(timeout=30.0)

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _sign_request(
        self,
        method: str,
        path: str,
        body: str = "",
    ) -> dict:
        """
        Sign a request for Polymarket API authentication.

        Returns headers dict with signature.
        """
        timestamp = str(int(time.time() * 1000))

        # Create signature
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            self.api_secret.encode() if self.api_secret else b"",
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

        return {
            "POLY-API-KEY": self.api_key or "",
            "POLY-SIGNATURE": signature,
            "POLY-TIMESTAMP": timestamp,
            "POLY-PASSPHRASE": self.passphrase or "",
            "Content-Type": "application/json",
        }

    async def place_market_order(
        self,
        token_id: str,
        side: str,  # "BUY" or "SELL"
        size_usd: float,
    ) -> OrderResult:
        """
        Place a market order.

        Args:
            token_id: The token to buy/sell (YES or NO token)
            side: "BUY" or "SELL"
            size_usd: Amount in USD to trade

        Returns:
            OrderResult with execution details
        """
        if self.dry_run:
            logger.info(
                f"[Executor] DRY RUN: Would place {side} order",
                token_id=token_id[:20] + "...",
                size_usd=f"${size_usd:.2f}",
            )
            return OrderResult(
                success=True,
                order_id=f"DRY-{int(time.time())}",
                message="Dry run - no order placed",
                filled_size=size_usd,
                avg_price=0.5,  # Simulated
            )

        if not self._client:
            await self.connect()

        try:
            # Get current best price from orderbook
            book_response = await self._client.get(
                f"{CLOB_API_URL}/book",
                params={"token_id": token_id},
            )
            book_response.raise_for_status()
            book = book_response.json()

            # Determine price based on side
            if side == "BUY":
                # Buy at best ask
                asks = book.get("asks", [])
                if not asks:
                    return OrderResult(success=False, message="No asks in orderbook")
                best_price = float(asks[0]["price"])
            else:
                # Sell at best bid
                bids = book.get("bids", [])
                if not bids:
                    return OrderResult(success=False, message="No bids in orderbook")
                best_price = float(bids[0]["price"])

            # Calculate shares
            shares = size_usd / best_price

            # Place order
            order_data = {
                "tokenID": token_id,
                "side": side,
                "size": str(shares),
                "price": str(best_price),
                "type": "LIMIT",  # Polymarket requires limit orders
                "timeInForce": "IOC",  # Immediate or cancel
            }

            path = "/order"
            body = str(order_data)
            headers = self._sign_request("POST", path, body)

            response = await self._client.post(
                f"{CLOB_API_URL}{path}",
                json=order_data,
                headers=headers,
            )

            if response.status_code == 200:
                result = response.json()
                order_id = result.get("orderID", "")

                logger.info(
                    f"[Executor] Order placed",
                    order_id=order_id,
                    side=side,
                    size=f"${size_usd:.2f}",
                    price=f"{best_price:.3f}",
                )

                return OrderResult(
                    success=True,
                    order_id=order_id,
                    message="Order placed",
                    filled_size=shares,
                    avg_price=best_price,
                )
            else:
                error_msg = response.text
                logger.error(f"[Executor] Order failed: {error_msg}")
                return OrderResult(
                    success=False,
                    message=f"Order failed: {error_msg}",
                )

        except Exception as e:
            logger.error(f"[Executor] Error placing order: {e}")
            return OrderResult(success=False, message=str(e))

    async def place_limit_order(
        self,
        token_id: str,
        side: str,
        shares: float,
        price: float,
    ) -> OrderResult:
        """Place a limit order at a specific price."""
        if self.dry_run:
            logger.info(
                f"[Executor] DRY RUN: Would place limit {side}",
                token_id=token_id[:20] + "...",
                shares=shares,
                price=f"{price:.3f}",
            )
            return OrderResult(
                success=True,
                order_id=f"DRY-{int(time.time())}",
                message="Dry run - no order placed",
            )

        if not self._client:
            await self.connect()

        try:
            order_data = {
                "tokenID": token_id,
                "side": side,
                "size": str(shares),
                "price": str(price),
                "type": "LIMIT",
                "timeInForce": "GTC",  # Good til canceled
            }

            path = "/order"
            body = str(order_data)
            headers = self._sign_request("POST", path, body)

            response = await self._client.post(
                f"{CLOB_API_URL}{path}",
                json=order_data,
                headers=headers,
            )

            if response.status_code == 200:
                result = response.json()
                order_id = result.get("orderID", "")
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    message="Limit order placed",
                )
            else:
                return OrderResult(
                    success=False,
                    message=f"Order failed: {response.text}",
                )

        except Exception as e:
            logger.error(f"[Executor] Error placing limit order: {e}")
            return OrderResult(success=False, message=str(e))

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if self.dry_run:
            logger.info(f"[Executor] DRY RUN: Would cancel order {order_id}")
            return True

        if not self._client:
            await self.connect()

        try:
            path = f"/order/{order_id}"
            headers = self._sign_request("DELETE", path)

            response = await self._client.delete(
                f"{CLOB_API_URL}{path}",
                headers=headers,
            )

            if response.status_code == 200:
                logger.info(f"[Executor] Order canceled: {order_id}")
                return True
            else:
                logger.error(f"[Executor] Cancel failed: {response.text}")
                return False

        except Exception as e:
            logger.error(f"[Executor] Error canceling order: {e}")
            return False

    async def get_positions(self) -> list[dict]:
        """Get current positions."""
        if self.dry_run:
            return []

        if not self._client:
            await self.connect()

        try:
            path = "/positions"
            headers = self._sign_request("GET", path)

            response = await self._client.get(
                f"{CLOB_API_URL}{path}",
                headers=headers,
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"[Executor] Failed to get positions: {response.text}")
                return []

        except Exception as e:
            logger.error(f"[Executor] Error getting positions: {e}")
            return []

    async def get_balance(self) -> float:
        """Get USDC balance."""
        if self.dry_run:
            return 0.0

        if not self._client:
            await self.connect()

        try:
            path = "/balance"
            headers = self._sign_request("GET", path)

            response = await self._client.get(
                f"{CLOB_API_URL}{path}",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                return float(data.get("balance", 0))
            else:
                logger.error(f"[Executor] Failed to get balance: {response.text}")
                return 0.0

        except Exception as e:
            logger.error(f"[Executor] Error getting balance: {e}")
            return 0.0

    def get_status(self) -> dict:
        """Get executor status."""
        return {
            "mode": "DRY RUN" if self.dry_run else "LIVE",
            "has_credentials": all([self.api_key, self.api_secret, self.passphrase]),
            "pending_orders": len(self.pending_orders),
            "filled_orders": len(self.filled_orders),
        }
