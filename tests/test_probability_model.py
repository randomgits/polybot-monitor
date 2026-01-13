"""Tests for the probability model."""

import pytest
from datetime import datetime, timezone

from polybot.data.models import MarketState
from polybot.strategy.probability_model import BinaryOptionPricer, ProbabilityModel


class TestBinaryOptionPricer:
    """Tests for binary option pricing."""

    def test_at_the_money(self):
        """Test pricing when current price equals strike."""
        pricer = BinaryOptionPricer()

        # At the money with 15 minutes left
        prob = pricer.calculate_up_probability(
            current_price=95000.0,
            strike_price=95000.0,
            time_to_expiry_seconds=900,  # 15 minutes
            volatility=0.20,
        )

        # Should be close to 50%
        assert 0.45 < prob < 0.55

    def test_in_the_money(self):
        """Test pricing when current price is above strike."""
        pricer = BinaryOptionPricer()

        # 1% above strike
        prob = pricer.calculate_up_probability(
            current_price=95950.0,
            strike_price=95000.0,
            time_to_expiry_seconds=900,
            volatility=0.20,
        )

        # Should be > 50%
        assert prob > 0.55

    def test_out_of_the_money(self):
        """Test pricing when current price is below strike."""
        pricer = BinaryOptionPricer()

        # 1% below strike
        prob = pricer.calculate_up_probability(
            current_price=94050.0,
            strike_price=95000.0,
            time_to_expiry_seconds=900,
            volatility=0.20,
        )

        # Should be < 50%
        assert prob < 0.45

    def test_near_expiry(self):
        """Test pricing near expiry."""
        pricer = BinaryOptionPricer()

        # In the money, 10 seconds left
        prob = pricer.calculate_up_probability(
            current_price=95100.0,
            strike_price=95000.0,
            time_to_expiry_seconds=10,
            volatility=0.20,
        )

        # Should be very high (close to 1)
        assert prob > 0.90

    def test_at_expiry(self):
        """Test pricing at expiry."""
        pricer = BinaryOptionPricer()

        # In the money at expiry
        prob = pricer.calculate_up_probability(
            current_price=95100.0,
            strike_price=95000.0,
            time_to_expiry_seconds=0,
            volatility=0.20,
        )

        assert prob == 1.0

        # Out of the money at expiry
        prob = pricer.calculate_up_probability(
            current_price=94900.0,
            strike_price=95000.0,
            time_to_expiry_seconds=0,
            volatility=0.20,
        )

        assert prob == 0.0


class TestProbabilityModel:
    """Tests for the full probability model."""

    @pytest.fixture
    def sample_state(self) -> MarketState:
        """Create a sample market state."""
        return MarketState(
            timestamp=datetime.now(timezone.utc),
            btc_price_binance=95000.0,
            btc_price_chainlink=95050.0,
            polymarket_yes_price=0.48,
            polymarket_no_price=0.52,
            polymarket_spread=0.02,
            time_to_expiry=600,
            volatility_1m=0.15,
            volatility_5m=0.18,
            volatility_15m=0.20,
            binance_orderbook_imbalance=0.1,
            chainlink_binance_spread=50.0,
            current_position=0.0,
            current_position_value=0.0,
            market_start_price=95000.0,
        )

    def test_generates_signal(self, sample_state):
        """Test that model generates a valid signal."""
        model = ProbabilityModel(min_edge_threshold=0.03)
        signal = model.generate_signal(sample_state)

        assert 0 <= signal.model_up_probability <= 1
        assert 0 <= signal.market_up_probability <= 1
        assert signal.recommended_action in ["HOLD", "BUY_YES", "BUY_NO"]
        assert 0 <= signal.confidence <= 1
        assert signal.kelly_fraction >= 0

    def test_hold_on_small_edge(self, sample_state):
        """Test that model recommends HOLD when edge is small."""
        model = ProbabilityModel(min_edge_threshold=0.10)  # High threshold

        # Set market price close to model price
        sample_state.polymarket_yes_price = 0.50

        signal = model.generate_signal(sample_state)

        # Should hold due to small edge
        assert signal.recommended_action == "HOLD"

    def test_buy_yes_when_underpriced(self, sample_state):
        """Test that model recommends BUY_YES when YES is underpriced."""
        model = ProbabilityModel(min_edge_threshold=0.03)

        # Set Chainlink price significantly above start (should be UP)
        sample_state.btc_price_chainlink = 96000.0  # 1% up
        sample_state.polymarket_yes_price = 0.40  # Market thinks DOWN

        signal = model.generate_signal(sample_state)

        assert signal.edge > 0
        assert signal.recommended_action == "BUY_YES"

    def test_buy_no_when_overpriced(self, sample_state):
        """Test that model recommends BUY_NO when YES is overpriced."""
        model = ProbabilityModel(min_edge_threshold=0.03)

        # Set Chainlink price significantly below start (should be DOWN)
        sample_state.btc_price_chainlink = 94000.0  # 1% down
        sample_state.polymarket_yes_price = 0.60  # Market thinks UP

        signal = model.generate_signal(sample_state)

        assert signal.edge < 0
        assert signal.recommended_action == "BUY_NO"

    def test_hold_near_expiry(self, sample_state):
        """Test that model recommends HOLD near expiry."""
        model = ProbabilityModel(min_edge_threshold=0.03)

        # Very close to expiry
        sample_state.time_to_expiry = 10  # 10 seconds

        signal = model.generate_signal(sample_state)

        # Should hold due to slippage risk
        assert signal.recommended_action == "HOLD"

    def test_hold_wide_spread(self, sample_state):
        """Test that model recommends HOLD when spread is wide."""
        model = ProbabilityModel(min_edge_threshold=0.03)

        # Wide spread
        sample_state.polymarket_spread = 0.10  # 10%

        signal = model.generate_signal(sample_state)

        # Should hold due to wide spread
        assert signal.recommended_action == "HOLD"
