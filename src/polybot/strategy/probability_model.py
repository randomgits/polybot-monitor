"""Probability model for BTC 15-min binary option pricing.

This is the bootstrap strategy that estimates the "true" probability of BTC
going up vs down, compared to Polymarket's implied probability.
"""

import math
from dataclasses import dataclass

import structlog

from polybot.data.models import MarketState

logger = structlog.get_logger()


@dataclass
class TradingSignal:
    """Trading signal from the probability model."""

    model_up_probability: float  # Our estimate of P(up)
    market_up_probability: float  # Polymarket's implied P(up)
    edge: float  # model - market (positive = we think UP is underpriced)
    recommended_action: str  # "BUY_YES", "BUY_NO", "HOLD"
    confidence: float  # 0-1 confidence in the signal
    kelly_fraction: float  # Optimal position size as fraction of bankroll


class BinaryOptionPricer:
    """
    Prices BTC 15-min markets as binary options using Black-Scholes-like model.

    Key insight: The market is essentially a digital/binary option that pays $1
    if BTC price at end >= price at start, $0 otherwise.

    For a binary option, the probability of being in-the-money is approximately:
    P(S_T >= K) = N(d2)

    where:
    - S = current price
    - K = strike (start price for the 15-min window)
    - T = time to expiry
    - sigma = volatility
    - N = cumulative normal distribution
    - d2 = (ln(S/K) + (r - sigma^2/2)*T) / (sigma * sqrt(T))

    For our use case:
    - K = start price of 15-min window (resolution threshold)
    - S = current Chainlink price (resolution source!)
    - We use realized volatility from Binance
    """

    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate

    def _norm_cdf(self, x: float) -> float:
        """Cumulative distribution function for standard normal."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def calculate_up_probability(
        self,
        current_price: float,
        strike_price: float,
        time_to_expiry_seconds: float,
        volatility: float,
    ) -> float:
        """
        Calculate probability of BTC ending >= strike price.

        Args:
            current_price: Current BTC price (from Chainlink, the resolution source!)
            strike_price: Start price for the 15-min window
            time_to_expiry_seconds: Seconds until market resolution
            volatility: Annualized volatility

        Returns:
            Probability in [0, 1]
        """
        if strike_price <= 0 or current_price <= 0:
            return 0.5

        if time_to_expiry_seconds <= 0:
            # At expiry, just check current price
            return 1.0 if current_price >= strike_price else 0.0

        # Convert to years
        T = time_to_expiry_seconds / (365.25 * 24 * 3600)

        # Avoid division by zero for very short times
        if T < 1e-10 or volatility < 1e-10:
            return 1.0 if current_price >= strike_price else 0.0

        # Calculate d2
        try:
            d2 = (
                math.log(current_price / strike_price)
                + (self.risk_free_rate - 0.5 * volatility**2) * T
            ) / (volatility * math.sqrt(T))

            return self._norm_cdf(d2)

        except (ValueError, ZeroDivisionError):
            return 0.5


class ProbabilityModel:
    """
    Main probability model that generates trading signals.

    Combines binary option pricing with additional market microstructure signals.
    """

    def __init__(
        self,
        min_edge_threshold: float = 0.05,
        kelly_fraction: float = 0.25,
        max_position_fraction: float = 0.1,
    ):
        self.pricer = BinaryOptionPricer()
        self.min_edge_threshold = min_edge_threshold
        self.kelly_fraction = kelly_fraction
        self.max_position_fraction = max_position_fraction

    def generate_signal(self, state: MarketState) -> TradingSignal:
        """
        Generate trading signal from current market state.

        Uses multiple models and combines them:
        1. Binary option pricing (Black-Scholes)
        2. Price momentum (Chainlink vs start price)
        3. Orderbook imbalance (Binance)
        """
        # Use 15-minute volatility for pricing
        # BTC annualized volatility is typically 40-80%, floor at 30% to avoid
        # over-confident predictions when volatility data is sparse
        volatility = max(state.volatility_15m, 0.30)

        # Get model probability using CHAINLINK price (resolution source!)
        # This is critical: Polymarket resolves on Chainlink, not Binance
        model_prob = self.pricer.calculate_up_probability(
            current_price=state.btc_price_chainlink,
            strike_price=state.market_start_price,
            time_to_expiry_seconds=state.time_to_expiry,
            volatility=volatility,
        )

        # Adjust for orderbook imbalance (momentum signal)
        # Positive imbalance = more buy pressure = slightly higher probability of UP
        imbalance_adjustment = state.binance_orderbook_imbalance * 0.02
        model_prob = max(0.001, min(0.999, model_prob + imbalance_adjustment))

        # Market implied probability from Polymarket
        market_prob = state.polymarket_yes_price

        # Calculate edge
        edge = model_prob - market_prob

        # Determine action and confidence
        action, confidence = self._determine_action(edge, state)

        # Calculate Kelly fraction for position sizing
        kelly = self._calculate_kelly(model_prob, market_prob, edge)

        return TradingSignal(
            model_up_probability=model_prob,
            market_up_probability=market_prob,
            edge=edge,
            recommended_action=action,
            confidence=confidence,
            kelly_fraction=kelly,
        )

    def _determine_action(self, edge: float, state: MarketState) -> tuple[str, float]:
        """Determine action and confidence based on edge."""
        abs_edge = abs(edge)

        # Don't trade if edge is below threshold
        if abs_edge < self.min_edge_threshold:
            return "HOLD", 0.0

        # Don't trade if too close to expiry (high slippage risk)
        if state.time_to_expiry < 30:  # 30 seconds
            return "HOLD", 0.0

        # Don't trade if spread is too wide
        if state.polymarket_spread > 0.05:  # 5% spread
            return "HOLD", 0.0

        # Calculate confidence based on edge magnitude
        confidence = min(1.0, abs_edge / 0.20)  # Max confidence at 20% edge

        if edge > 0:
            return "BUY_YES", confidence
        else:
            return "BUY_NO", confidence

    def _calculate_kelly(
        self, model_prob: float, market_prob: float, edge: float
    ) -> float:
        """
        Calculate Kelly criterion position size, accounting for fees.

        For binary bets:
        kelly = (p * b - q) / b

        where:
        - p = probability of winning
        - q = 1 - p = probability of losing
        - b = net odds (payout ratio after fees)

        Polymarket fees are highest at 50/50 (~3.15%) and decrease toward extremes.
        """
        if abs(edge) < self.min_edge_threshold:
            return 0.0

        # Calculate taker fee based on price (same formula as order_manager)
        distance_from_50 = abs(market_prob - 0.5)
        max_fee = 0.0315  # 3.15%
        fee_rate = max_fee * (1 - 2 * distance_from_50)
        fee_rate = max(0, fee_rate)

        if edge > 0:
            # Betting on YES
            p = model_prob
            q = 1 - p
            if market_prob <= 0 or market_prob >= 1:
                return 0.0

            # Gross odds: win (1 - market_prob) for risking market_prob
            # Net payout after fees: (1 - market_prob) * (1 - fee_rate)
            gross_win = 1 - market_prob
            net_win = gross_win * (1 - fee_rate)
            b = net_win / market_prob  # Net odds ratio
        else:
            # Betting on NO
            p = 1 - model_prob
            q = 1 - p
            market_no_prob = 1 - market_prob
            if market_no_prob <= 0 or market_no_prob >= 1:
                return 0.0

            gross_win = market_prob  # Win market_prob for risking (1 - market_prob)
            net_win = gross_win * (1 - fee_rate)
            b = net_win / market_no_prob  # Net odds ratio

        # Kelly formula: f* = (p*b - q) / b
        if b <= 0:
            return 0.0

        kelly = (p * b - q) / b

        # Apply fractional Kelly (reduces variance significantly)
        # Use 0.25 by default (quarter Kelly is more conservative)
        kelly = kelly * self.kelly_fraction

        # Additional safety: scale down when edge is uncertain (close to threshold)
        edge_confidence = min(1.0, abs(edge) / (2 * self.min_edge_threshold))
        kelly = kelly * edge_confidence

        # Cap at max position fraction
        kelly = max(0, min(self.max_position_fraction, kelly))

        return kelly


def demo():
    """Demo the probability model."""
    from datetime import datetime, timezone

    model = ProbabilityModel(min_edge_threshold=0.03)

    # Create a sample market state
    state = MarketState(
        timestamp=datetime.now(timezone.utc),
        btc_price_binance=95000.0,
        btc_price_chainlink=95050.0,  # Slightly higher on Chainlink
        polymarket_yes_price=0.48,  # Market thinks slight DOWN bias
        polymarket_no_price=0.52,
        polymarket_spread=0.02,
        time_to_expiry=600,  # 10 minutes left
        volatility_1m=0.15,
        volatility_5m=0.18,
        volatility_15m=0.20,
        binance_orderbook_imbalance=0.1,  # Slight buy pressure
        chainlink_binance_spread=50.0,
        current_position=0.0,
        current_position_value=0.0,
        market_start_price=95000.0,  # Started at 95k
    )

    signal = model.generate_signal(state)

    print("=== Probability Model Signal ===")
    print(f"Model P(UP): {signal.model_up_probability:.3f}")
    print(f"Market P(UP): {signal.market_up_probability:.3f}")
    print(f"Edge: {signal.edge:+.3f}")
    print(f"Action: {signal.recommended_action}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Kelly fraction: {signal.kelly_fraction:.3f}")


if __name__ == "__main__":
    demo()
