# PolyBot - Polymarket BTC 15-min Trading Bot

An algorithmic trading bot for Polymarket BTC 15-minute prediction markets using online reinforcement learning with a probability model as a feature.

## Current Status

**Working:**
- Data aggregation from Polymarket, Binance (via CryptoCompare), and Chainlink
- PPO-based RL agent making trading decisions
- Paper trading mode with position tracking
- Online learning from actual P&L (normalized to percentage returns)
- Probability model provides signal as feature input to RL

**Architecture:**
- RL agent decides WHEN to trade (HOLD, BUY_YES, BUY_NO, CLOSE)
- Risk manager decides HOW MUCH (fixed 10% of capital)
- Rewards come from actual trade P&L, not simulated positions

## Environment Variables

Create a `.env` file in the project root:

```bash
# Required for live trading (optional for paper trading)
POLYMARKET_API_KEY=
POLYMARKET_SECRET=
POLYMARKET_PASSPHRASE=

# Optional - uses public endpoints if not set
BINANCE_API_KEY=
BINANCE_API_SECRET=

# Chainlink data (default works, but rate-limited)
POLYGON_RPC_URL=https://polygon-rpc.com

# Trading parameters
MAX_POSITION_USD=100.0          # Max position per market
DAILY_LOSS_LIMIT_USD=50.0       # Stop trading after this daily loss

# RL parameters
RL_EXPLORATION_RATE=0.1         # Random action probability
RL_LEARNING_RATE=0.0003         # PPO learning rate
RL_BATCH_SIZE=64                # Training batch size

# Logging
LOG_LEVEL=INFO
```

## Key Insights

- **Resolution Source**: Polymarket BTC 15-min markets resolve based on **Chainlink BTC/USD**, not Binance
- **Fee Structure**: Taker fees up to 3.15% at 50/50 odds - pure latency arbitrage is not profitable
- **Edge Source**: RL learns patterns in price movement vs market odds

## Installation

```bash
# Install dependencies with uv
uv sync

# Run paper trading
uv run python -m polybot run --mode paper --capital 1000
```

## Usage

### Test Data Connections

```bash
# Test all data sources
uv run python -m polybot test-data

# Test specific source
uv run python -m polybot test-data --source chainlink
```

### Run Paper Trading

```bash
# Default paper trading with RL
uv run python -m polybot run --mode paper --capital 1000

# Paper trading with probability model only (no RL)
uv run python -m polybot run --mode paper --no-rl
```

### Monitor Status

The bot logs status updates to stdout. Key metrics:
- Current capital and P&L
- Active positions
- Signal edge (model vs market probability)
- Trade execution details

## Project Structure

```
polybot/
├── src/polybot/
│   ├── data/              # Data layer
│   │   ├── polymarket.py  # Polymarket CLOB API client
│   │   ├── binance.py     # Binance WebSocket client
│   │   ├── chainlink.py   # Chainlink price feed (resolution source!)
│   │   ├── aggregator.py  # Combines all data sources
│   │   └── models.py      # Data models (MarketState, etc.)
│   │
│   ├── strategy/          # Strategy layer
│   │   └── probability_model.py  # Binary option pricing
│   │
│   ├── execution/         # Execution layer
│   │   ├── order_manager.py   # Order submission
│   │   └── risk_manager.py    # Position sizing, limits
│   │
│   ├── rl/                # Reinforcement learning
│   │   ├── environment.py # Gymnasium environment
│   │   └── agent.py       # Stable Baselines3 wrapper
│   │
│   ├── bot.py             # Main trading bot
│   ├── config.py          # Configuration
│   └── __main__.py        # CLI entry point
│
├── tests/                 # Test files
├── models/                # Saved RL models
├── logs/                  # Trading logs
├── pyproject.toml         # Dependencies
└── .env                   # Environment variables
```

## Configuration

Edit `.env` or set environment variables:

```bash
# Trading parameters
MAX_POSITION_USD=100.0      # Max position per market
DAILY_LOSS_LIMIT_USD=50.0   # Stop trading after this loss
MIN_EDGE_THRESHOLD=0.05     # Minimum edge to trade (5%)
KELLY_FRACTION=0.25         # Use 25% of Kelly criterion

# RL parameters
RL_EXPLORATION_RATE=0.1     # Epsilon for exploration
RL_LEARNING_RATE=0.0003     # Learning rate
```

## How It Works

### Data Flow

```
Chainlink (resolution) ─┐
Binance (price/vol)    ─┼──> Aggregator ──> MarketState (13 features)
Polymarket (odds)      ─┘                         │
                                                  │
                        ┌───────────────┐         │
                        │ Probability   │─────────┼──> 14 features
                        │ Model         │ (1 feat)│
                        └───────────────┘         │
                                                  ▼
                                          ┌───────────────┐
                                          │ RL Agent      │──> Action
                                          │ (PPO)         │   (HOLD/YES/NO/CLOSE)
                                          └───────┬───────┘
                                                  │
                                                  ▼
                                          ┌───────────────┐
                                          │ Risk Manager  │──> Position Size (10%)
                                          └───────┬───────┘
                                                  │
                                                  ▼
                                          ┌───────────────┐
                                          │ Order Manager │──> Execute
                                          └───────┬───────┘
                                                  │
                                                  ▼
                                          ┌───────────────┐
                                          │ Actual P&L    │──> Reward (normalized %)
                                          └───────────────┘
```

### Probability Model

The model prices BTC 15-min markets as binary options:

```
P(UP) = N(d2)

where d2 = (ln(S/K) + (r - σ²/2)T) / (σ√T)

S = current Chainlink price (resolution source!)
K = start price for 15-min window
T = time to expiry
σ = realized volatility from Binance
```

### RL Agent

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Network**: 128 → 64 hidden layers
- **State**: 14 normalized features:
  1. BTC price vs market start (%)
  2. Chainlink price vs start (%)
  3. Chainlink-Binance spread (bps)
  4. Polymarket YES price
  5. Polymarket NO price
  6. Bid-ask spread
  7. Time to expiry (normalized)
  8. Volatility 1m, 5m, 15m
  9. Orderbook imbalance
  10. Current position
  11. Above/below target (binary)
  12. Probability model estimate
- **Actions**: HOLD, BUY_YES, BUY_NO, CLOSE
- **Reward**: Actual P&L from trades, normalized by position size (percentage returns)

## Development

```bash
# Run tests
uv run pytest

# Format code
uv run ruff format src/

# Type check
uv run mypy src/
```

## License

MIT
