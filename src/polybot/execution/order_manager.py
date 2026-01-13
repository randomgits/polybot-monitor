"""Order manager for Polymarket order execution."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

from polybot.data.polymarket import PolymarketClient

logger = structlog.get_logger()


class OrderStatus(Enum):
    """Order status."""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Order:
    """Order representation."""

    order_id: str
    market_id: str
    token_id: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: float | None  # None for market orders
    status: OrderStatus
    created_at: datetime
    filled_size: float = 0.0
    filled_price: float = 0.0
    fees_paid: float = 0.0


class OrderManager:
    """
    Manages order execution on Polymarket.

    Features:
    - Limit order placement (preferred to avoid taker fees)
    - Market order execution
    - Order tracking and status updates
    - Paper trading mode for testing
    """

    def __init__(
        self,
        polymarket_client: PolymarketClient,
        paper_trading: bool = True,
    ):
        self.client = polymarket_client
        self.paper_trading = paper_trading
        self._orders: dict[str, Order] = {}
        self._order_counter = 0

    async def place_order(
        self,
        market_id: str,
        token_id: str,
        side: OrderSide,
        size: float,
        price: float | None = None,
        order_type: OrderType = OrderType.LIMIT,
    ) -> Order:
        """
        Place an order on Polymarket.

        For live trading, this will submit to Polymarket CLOB.
        For paper trading, this simulates execution.

        Args:
            market_id: Market identifier
            token_id: Token ID (YES or NO token)
            side: BUY or SELL
            size: Number of shares (in USDC terms)
            price: Limit price (required for limit orders)
            order_type: LIMIT or MARKET
        """
        self._order_counter += 1
        order_id = f"order_{self._order_counter}"

        order = Order(
            order_id=order_id,
            market_id=market_id,
            token_id=token_id,
            side=side,
            order_type=order_type,
            size=size,
            price=price,
            status=OrderStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )

        self._orders[order_id] = order

        if self.paper_trading:
            await self._simulate_execution(order)
        else:
            await self._submit_to_polymarket(order)

        return order

    async def _simulate_execution(self, order: Order) -> None:
        """Simulate order execution for paper trading."""
        # Get current orderbook to simulate fill
        orderbook = await self.client.get_market_orderbook(order.token_id)

        if not orderbook:
            order.status = OrderStatus.REJECTED
            logger.warning("Order rejected - no orderbook", order_id=order.order_id)
            return

        # Determine fill price
        if order.order_type == OrderType.MARKET:
            # Market order fills at best available price
            if order.side == OrderSide.BUY:
                fill_price = orderbook.best_ask or 0.5
            else:
                fill_price = orderbook.best_bid or 0.5
        else:
            # Limit order - check if it would fill
            if order.side == OrderSide.BUY:
                if order.price and orderbook.best_ask and order.price >= orderbook.best_ask:
                    fill_price = orderbook.best_ask
                else:
                    # Order would rest on book - simulate immediate fill at limit price
                    fill_price = order.price or 0.5
            else:
                if order.price and orderbook.best_bid and order.price <= orderbook.best_bid:
                    fill_price = orderbook.best_bid
                else:
                    fill_price = order.price or 0.5

        # Simulate slippage (0.1% for paper trading)
        slippage = 0.001
        if order.side == OrderSide.BUY:
            fill_price *= 1 + slippage
        else:
            fill_price *= 1 - slippage

        # Calculate fees (taker fee varies by price, max ~3.15% at 50/50)
        # In Polymarket, even limit orders incur taker fees if they execute immediately
        # (i.e., cross the spread). Only resting orders that add liquidity avoid fees.
        # For paper trading, we assume immediate execution = taker fees
        fee_rate = self._calculate_taker_fee(fill_price)
        fees = order.size * fee_rate

        # Fill the order
        order.filled_size = order.size
        order.filled_price = fill_price
        order.fees_paid = fees
        order.status = OrderStatus.FILLED

        logger.info(
            "Order filled (paper)",
            order_id=order.order_id,
            side=order.side.value,
            size=order.size,
            price=fill_price,
            fees=fees,
        )

    async def _submit_to_polymarket(self, order: Order) -> None:
        """
        Submit order to Polymarket CLOB.

        NOTE: This requires proper authentication and signing.
        Implementation depends on Polymarket's specific API requirements.
        """
        # TODO: Implement actual Polymarket order submission
        # This requires:
        # 1. EIP-712 signature for order
        # 2. Proper nonce management
        # 3. API key authentication
        # See: https://docs.polymarket.com/developers/CLOB/trading

        logger.warning(
            "Live trading not implemented yet",
            order_id=order.order_id,
        )
        order.status = OrderStatus.REJECTED

    def _calculate_taker_fee(self, price: float) -> float:
        """
        Calculate Polymarket taker fee based on price.

        Fees are highest at 50% (0.50 price) and decrease toward extremes.
        Max fee is ~3.15% at 50/50 odds.
        """
        # Fee formula approximation (simplified)
        # Actual formula may differ - check Polymarket docs
        distance_from_50 = abs(price - 0.5)
        max_fee = 0.0315  # 3.15%

        # Linear decay from max fee at 50% to 0 at extremes
        fee = max_fee * (1 - 2 * distance_from_50)
        return max(0, fee)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]
        if order.status not in [OrderStatus.PENDING, OrderStatus.OPEN]:
            return False

        if self.paper_trading:
            order.status = OrderStatus.CANCELLED
            logger.info("Order cancelled (paper)", order_id=order_id)
            return True
        else:
            # TODO: Implement Polymarket cancel
            return False

    def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_open_orders(self, market_id: str | None = None) -> list[Order]:
        """Get all open orders, optionally filtered by market."""
        orders = [
            o
            for o in self._orders.values()
            if o.status in [OrderStatus.PENDING, OrderStatus.OPEN]
        ]
        if market_id:
            orders = [o for o in orders if o.market_id == market_id]
        return orders

    def get_filled_orders(self, market_id: str | None = None) -> list[Order]:
        """Get all filled orders, optionally filtered by market."""
        orders = [o for o in self._orders.values() if o.status == OrderStatus.FILLED]
        if market_id:
            orders = [o for o in orders if o.market_id == market_id]
        return orders


class TradeExecutor:
    """
    High-level trade execution that combines OrderManager and RiskManager.

    Note: The RL agent decides WHEN to trade. This class handles HOW to execute
    while enforcing risk limits. No probability model filtering.
    """

    def __init__(
        self,
        order_manager: OrderManager,
        risk_manager: Any,  # Avoid circular import
    ):
        self.orders = order_manager
        self.risk = risk_manager

    async def execute_rl_action(
        self,
        action: str,  # "BUY_YES" or "BUY_NO"
        market_id: str,
        yes_token_id: str,
        no_token_id: str,
        yes_price: float,
        no_price: float,
        time_to_expiry: float,
        spread: float,
        available_capital: float,
    ) -> Order | None:
        """
        Execute an RL trading decision.

        The RL agent decides whether to trade. This method only enforces
        risk limits and handles execution mechanics.

        Args:
            action: "BUY_YES" or "BUY_NO"
            market_id: Market identifier
            yes_token_id: YES token ID
            no_token_id: NO token ID
            yes_price: Current YES price
            no_price: Current NO price
            time_to_expiry: Seconds until resolution
            spread: Current bid-ask spread
            available_capital: Capital available for trading

        Returns:
            Order if executed, None if rejected by risk limits
        """
        # Validate against risk limits only (not probability model)
        is_valid, reason = self.risk.validate_trade(time_to_expiry, spread)
        if not is_valid:
            logger.debug("Trade rejected by risk limits", reason=reason)
            return None

        # Calculate position size (fixed percentage)
        size = self.risk.calculate_position_size(available_capital)
        if size < 1.0:
            logger.debug("Position size too small", size=size)
            return None

        # Determine token and price based on action
        if action == "BUY_YES":
            token_id = yes_token_id
            price = yes_price + 0.001  # Slightly above market
            token_side = "YES"
        elif action == "BUY_NO":
            token_id = no_token_id
            price = no_price + 0.001
            token_side = "NO"
        else:
            logger.warning("Invalid action for execution", action=action)
            return None

        # Place limit order
        order = await self.orders.place_order(
            market_id=market_id,
            token_id=token_id,
            side=OrderSide.BUY,
            size=size,
            price=price,
            order_type=OrderType.LIMIT,
        )

        # Record in risk manager if filled
        if order.status == OrderStatus.FILLED:
            self.risk.record_trade(
                market_id=market_id,
                token_id=token_id,
                side=token_side,
                size=order.filled_size,
                price=order.filled_price,
                fees=order.fees_paid,
            )

        return order
