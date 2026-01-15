# PolyBot Architecture & Operations Guide

A monitoring and trading bot for Polymarket's BTC 15-minute prediction markets.

## Table of Contents

1. [Overview](#overview)
2. [How BTC 15-Min Markets Work](#how-btc-15-min-markets-work)
3. [Architecture](#architecture)
4. [Data Sources](#data-sources)
5. [Key Learnings & Gotchas](#key-learnings--gotchas)
6. [Deployment](#deployment)
7. [Configuration](#configuration)
8. [API Reference](#api-reference)

---

## Overview

PolyBot monitors Polymarket's BTC 15-minute prediction markets and identifies trading opportunities by comparing:

- **Binance BTC price** (real-time market data)
- **Chainlink BTC/USD price** (the actual resolution source)
- **Polymarket YES/NO prices** (market odds)

The core insight: Polymarket resolves these markets using **Chainlink data**, not Binance. Any divergence between Binance prices and Chainlink prices creates potential arbitrage opportunities.

### Market Mechanics

```
Market Question: "Will BTC be UP or DOWN at 2:00 AM ET compared to 1:45 AM ET?"

Resolution:
- Uses Chainlink BTC/USD data stream (https://data.chain.link/streams/btc-usd)
- If end_price >= start_price → resolves to "Up" (YES wins)
- If end_price < start_price → resolves to "Down" (NO wins)
```

---

## How BTC 15-Min Markets Work

### Market Lifecycle

1. **Creation**: Markets are created ~15-30 minutes before the prediction window
2. **Trading Window**: Users buy YES or NO shares based on their BTC price prediction
3. **Resolution**: At window end, Chainlink price determines the outcome
4. **Settlement**: Winning shares pay $1, losing shares pay $0

### Slug Pattern

Markets use timestamp-based slugs:
```
btc-updown-15m-{unix_timestamp}
```

Where `unix_timestamp` is the end time of the 15-minute window.

**Example**: For a market ending at 2:00 AM UTC on Jan 15, 2026:
```
btc-updown-15m-1768459500
```

### Timestamp Calculation

```python
interval = 15 * 60  # 15 minutes in seconds
now_ts = int(datetime.now(timezone.utc).timestamp())
current_window_end = ((now_ts // interval) + 1) * interval

# Check current and next windows
timestamps = [
    current_window_end,           # Current window
    current_window_end + interval, # Next window
    current_window_end + interval * 2,  # Window after
]
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PolyBot System                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Binance    │  │  Chainlink   │  │  Polymarket  │          │
│  │  WebSocket   │  │  HTTP/RPC    │  │  REST + WS   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └────────────┬────┴─────────────────┘                   │
│                      │                                          │
│              ┌───────▼───────┐                                  │
│              │ DataAggregator │                                  │
│              │               │                                  │
│              │ - Combines    │                                  │
│              │   all feeds   │                                  │
│              │ - Calculates  │                                  │
│              │   volatility  │                                  │
│              │ - Detects     │                                  │
│              │   opportunities│                                  │
│              └───────┬───────┘                                  │
│                      │                                          │
│              ┌───────▼───────┐                                  │
│              │ProbabilityModel│                                  │
│              │               │                                  │
│              │ - Estimates   │                                  │
│              │   true prob   │                                  │
│              │ - Calculates  │                                  │
│              │   edge        │                                  │
│              │ - Kelly sizing│                                  │
│              └───────┬───────┘                                  │
│                      │                                          │
│              ┌───────▼───────┐                                  │
│              │   FastAPI     │                                  │
│              │   Dashboard   │                                  │
│              └───────────────┘                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `PolymarketClient` | `data/polymarket.py` | Market discovery, order book data |
| `BinanceClient` | `data/binance.py` | Real-time BTC price via WebSocket |
| `ChainlinkClient` | `data/chainlink.py` | Resolution price from Polygon |
| `DataAggregator` | `data/aggregator.py` | Combines all data sources |
| `ProbabilityModel` | `strategy/probability.py` | Calculates true probability & edge |
| `FastAPI App` | `api.py` | HTTP dashboard and API |

---

## Data Sources

### 1. Binance (Real-time BTC Price)

**Purpose**: Fast price discovery, volatility calculation

**Connection Methods** (in priority order):
1. `wss://stream.binance.com:9443/ws/btcusdt@ticker` (Global, via proxy)
2. `wss://stream.binance.us:9443/ws/btcusd@ticker` (US fallback)

**Important**: Binance global is geo-blocked in many regions. See [Proxy Requirements](#proxy-requirements).

### 2. Chainlink (Resolution Source)

**Purpose**: This is the ACTUAL price used for market resolution

**Endpoint**: Polygon RPC → Chainlink Price Feed Contract
```
Contract: 0xc907E116054Ad103354f2D350FD2514433D57F6f (BTC/USD on Polygon)
RPC: https://polygon-rpc.com
```

**Critical**: Always compare Polymarket odds against Chainlink price, not Binance!

### 3. Polymarket (Market Data)

**APIs Used**:
- **Gamma API**: `https://gamma-api.polymarket.com` - Market discovery
- **CLOB API**: `https://clob.polymarket.com` - Order book data
- **WebSocket**: `wss://ws-subscriptions-clob.polymarket.com/ws/market` - Real-time updates

---

## Key Learnings & Gotchas

### DO's

#### 1. Use German ISP Proxies for Binance
```bash
# Binance global is blocked in US/UK but works from Germany
PROXY_HOST=isp.decodo.com
PROXY_PORT=10051
PROXY_USER=your_user
PROXY_PASS=your_pass
```

ISP proxies (not datacenter) are required as Binance blocks known datacenter IPs.

#### 2. Query Markets by Timestamp Slug
```python
# BTC 15-min markets don't appear in general listings
# Must query by specific slug
slug = f"btc-updown-15m-{timestamp}"
response = await client.get(f"{GAMMA_API}/markets", params={"slug": slug})
```

#### 3. Use `endDate` Not `endDateIso`
```python
# endDateIso = "2026-01-15" (date only, no time!)
# endDate = "2026-01-15T06:45:00Z" (full timestamp)
end_time_str = market_data.get("endDate")  # Correct!
```

#### 4. Ensure Timezone-Aware Datetimes
```python
# Always make datetimes timezone-aware for comparison
if end_time.tzinfo is None:
    end_time = end_time.replace(tzinfo=timezone.utc)
```

#### 5. Convert IDs to Strings
```python
# The clean_json_response function converts "1234" to 1234 (int)
# But Pydantic models expect strings
market_id=str(market_data.get("id", ""))
token_id=str(clob_token_ids[0])
```

#### 6. Use REST Polling as Fallback
```python
# WebSocket can be unreliable, always have REST fallback
if not ws_connected:
    update = await self.get_order_book(token_id)  # REST API
```

#### 7. Start Services in Background for Health Checks
```python
# Don't block startup - health checks need fast response
asyncio.create_task(initialize_services())  # Background
```

### DON'Ts

#### 1. Don't Use Binance Price for Resolution Logic
```python
# WRONG: Binance price for decision
if binance_price > start_price:
    buy_yes()

# RIGHT: Use Chainlink (resolution source)
if chainlink_price > start_price:
    buy_yes()
```

#### 2. Don't Assume Markets Are Listed in General API
```python
# WRONG: Search all markets
response = await client.get(f"{GAMMA_API}/markets", params={"active": True})

# RIGHT: Query specific timestamp slugs
for ts in timestamps_to_check:
    slug = f"btc-updown-15m-{ts}"
    response = await client.get(f"{GAMMA_API}/markets", params={"slug": slug})
```

#### 3. Don't Use Editable Install in Docker
```dockerfile
# WRONG: Editable install creates issues
RUN pip install -e .

# RIGHT: Standard install
RUN pip install --no-cache-dir .
```

#### 4. Don't Use Exec Form for Variable Expansion
```dockerfile
# WRONG: Variables not expanded in exec form
CMD ["python", "-m", "uvicorn", "app:app", "--port", "$PORT"]

# RIGHT: Shell form expands variables
CMD python -m uvicorn polybot.api:app --host 0.0.0.0 --port ${PORT:-8080}
```

#### 5. Don't Include Heavy ML Dependencies in Production
```toml
# Move to optional dependencies to reduce build time
[project.optional-dependencies]
rl = ["stable-baselines3>=2.2.1", "gymnasium>=0.29.0"]
```

#### 6. Don't Trust `dict.get("key", {})` with None Values
```python
# WRONG: Returns None if key exists with None value
market = status_data.get("market", {})

# RIGHT: Handle None explicitly
market = status_data.get("market") or {}
```

---

## Deployment

### Railway Deployment

1. **Create Project**
```bash
railway login
railway init
railway link
```

2. **Set Environment Variables**
```bash
railway variables set PROXY_HOST=isp.decodo.com
railway variables set PROXY_PORT=10051
railway variables set PROXY_USER=your_user
railway variables set PROXY_PASS=your_pass
```

3. **Deploy**
```bash
railway up
```

4. **Generate Domain**
```bash
railway domain
```

### Required Files

**Dockerfile**:
```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ git && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-cache-dir .
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
EXPOSE 8080
CMD python -m uvicorn polybot.api:app --host 0.0.0.0 --port ${PORT:-8080}
```

**railway.json**:
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "healthcheckPath": "/health",
    "healthcheckTimeout": 120,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `PROXY_HOST` | Yes* | Proxy hostname for Binance |
| `PROXY_PORT` | Yes* | Proxy port |
| `PROXY_USER` | No | Proxy username |
| `PROXY_PASS` | No | Proxy password |
| `MIN_EDGE_THRESHOLD` | No | Minimum edge to signal (default: 0.05) |
| `KELLY_FRACTION` | No | Kelly criterion fraction (default: 0.25) |
| `PORT` | No | Server port (default: 8080) |

*Required for Binance global access from geo-blocked regions

### Proxy Recommendations

For Binance access, use **German ISP proxies**:
- Residential/ISP proxies work best
- Datacenter proxies are often blocked
- Germany, Netherlands, and Switzerland have reliable access

Tested provider: **Decodo** (isp.decodo.com)

---

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | HTML dashboard (auto-refresh) |
| `/health` | GET | Health check for load balancers |
| `/status` | GET | Full system status JSON |
| `/opportunities` | GET | Recent trading opportunities |
| `/signals` | GET | Recent generated signals |
| `/docs` | GET | Swagger API documentation |

### Response Examples

**GET /health**
```json
{
  "status": "ok",
  "uptime_seconds": 3600.5,
  "connections": {
    "binance": true,
    "chainlink": true,
    "polymarket": true
  }
}
```

**GET /status**
```json
{
  "running": true,
  "market": {
    "market_id": "1181211",
    "question": "Bitcoin Up or Down - January 15, 1:45AM-2:00AM ET",
    "time_to_expiry_seconds": 845.2,
    "yes_price": 0.52,
    "no_price": 0.48,
    "spread": 0.0
  },
  "prices": {
    "btc_binance": 96500.25,
    "btc_chainlink": 96485.10,
    "btc_spread": -15.15,
    "chainlink_age_seconds": 0.5
  },
  "volatility": {
    "1m": 0.85,
    "5m": 0.42,
    "15m": 0.28
  },
  "connections": {
    "binance": true,
    "chainlink": true,
    "polymarket": true
  },
  "signals_generated": 147,
  "opportunities_count": 3,
  "uptime_seconds": 3600.5
}
```

---

## Trading Strategy Notes

### Edge Calculation

```python
# Model estimates true probability based on:
# - Current BTC price vs start price
# - Time remaining
# - Recent volatility

model_prob = estimate_probability(chainlink_price, start_price, time_left, volatility)
market_prob = yes_price  # Market's implied probability

edge = model_prob - market_prob

# Signal if edge exceeds threshold (default 5%)
if abs(edge) > 0.05:
    action = "BUY_YES" if edge > 0 else "BUY_NO"
```

### Kelly Criterion Sizing

```python
# Kelly fraction for optimal bet sizing
kelly = (edge * win_prob - (1 - win_prob)) / odds
position_size = bankroll * kelly * fraction  # fraction = 0.25 for safety
```

### Key Insight: Chainlink vs Binance Spread

The spread between Chainlink and Binance prices is the primary alpha source:
- If Chainlink > Binance and market expects DOWN → opportunity
- If Chainlink < Binance and market expects UP → opportunity

Monitor the spread closely - it typically ranges from -$50 to +$50.

---

## Troubleshooting

### "Polymarket Disconnected"

1. Check if BTC 15-min markets are active (they can be paused for maintenance)
2. Verify timestamp calculation is correct for current UTC time
3. Check logs for market discovery errors

### "Binance Access Denied"

1. Verify proxy credentials
2. Try a different proxy region (Germany recommended)
3. Check if IP is blacklisted (switch proxy)

### "Chainlink Stale Price"

1. Polygon RPC might be slow - try alternative RPCs
2. Check `chainlink_age_seconds` in status
3. Prices older than 60s should be treated as stale

### Build Timeout on Railway

1. Remove heavy dependencies (PyTorch, TensorFlow)
2. Use multi-stage Docker builds
3. Move ML dependencies to optional extras

---

## Future Improvements

1. **WebSocket Reliability**: Implement reconnection logic with exponential backoff
2. **Multiple RPC Fallbacks**: Add backup Polygon RPC endpoints
3. **Historical Analysis**: Store signals for backtesting
4. **Telegram Alerts**: Notify on high-confidence opportunities
5. **Auto-Trading**: Integrate with Polymarket CLOB for execution

---

*Last updated: January 2026*
