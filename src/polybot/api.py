"""
FastAPI monitoring dashboard for PolyBot.

Provides endpoints to:
- View current market state
- See detected opportunities
- Monitor bot health and connections
- View signal/trade statistics
"""

import asyncio
import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from polybot.data.models import MarketState, Opportunity
from polybot.data.binance import BinanceClient
from polybot.data.polymarket import PolymarketClient
from polybot.data.chainlink import ChainlinkClient
from polybot.data.aggregator import DataAggregator
from polybot.strategy.probability_model import ProbabilityModel, TradingSignal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global state
class AppState:
    """Global application state."""

    aggregator: Optional[DataAggregator] = None
    probability_model: Optional[ProbabilityModel] = None
    polymarket_client: Optional[PolymarketClient] = None
    binance_client: Optional[BinanceClient] = None
    chainlink_client: Optional[ChainlinkClient] = None

    # Statistics
    signals_generated: int = 0
    opportunities_detected: int = 0
    start_time: datetime = datetime.now(timezone.utc)

    # Recent signals
    recent_signals: list[dict] = []
    max_signals: int = 100


state = AppState()


# Pydantic models for API responses
class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    connections: dict


class MarketResponse(BaseModel):
    market_id: Optional[str]
    question: Optional[str]
    time_to_expiry_seconds: float
    yes_price: float
    no_price: float
    spread: float


class PricesResponse(BaseModel):
    btc_binance: float
    btc_chainlink: float
    btc_spread: float
    chainlink_age_seconds: float


class SignalResponse(BaseModel):
    timestamp: str
    model_probability: float
    market_probability: float
    edge_pct: float
    recommended_action: str
    confidence: float
    kelly_fraction: float


class StatusResponse(BaseModel):
    running: bool
    market: Optional[MarketResponse]
    prices: PricesResponse
    volatility: dict
    connections: dict
    signals_generated: int
    opportunities_count: int
    uptime_seconds: float


class OpportunityResponse(BaseModel):
    timestamp: str
    market_id: str
    market_question: str
    btc_price_binance: float
    btc_price_chainlink: float
    btc_change_pct: float
    yes_price: float
    no_price: float
    time_to_expiry_formatted: str
    model_probability: float
    market_probability: float
    edge_pct: float
    recommended_action: str
    confidence: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler - initialize and cleanup."""
    logger.info("Starting PolyBot monitoring service...")

    # Initialize clients
    state.polymarket_client = PolymarketClient()
    state.binance_client = BinanceClient()
    state.chainlink_client = ChainlinkClient()

    await state.polymarket_client.connect()

    # Initialize aggregator
    state.aggregator = DataAggregator(
        state.polymarket_client,
        state.binance_client,
        state.chainlink_client,
    )

    # Initialize probability model
    state.probability_model = ProbabilityModel(
        min_edge_threshold=float(os.getenv("MIN_EDGE_THRESHOLD", "0.05")),
        kelly_fraction=float(os.getenv("KELLY_FRACTION", "0.25")),
    )

    # Register state update callback
    state.aggregator.on_state_update(on_state_update)

    # Start aggregator
    await state.aggregator.start()

    logger.info("PolyBot monitoring service started")

    yield

    # Cleanup
    logger.info("Shutting down PolyBot monitoring service...")
    await state.aggregator.stop()
    await state.polymarket_client.disconnect()
    await state.chainlink_client.disconnect()


def on_state_update(market_state: MarketState) -> None:
    """Handle state update from aggregator."""
    if not state.probability_model:
        return

    # Generate signal
    signal = state.probability_model.generate_signal(market_state)
    state.signals_generated += 1

    # Record signal
    signal_data = {
        "timestamp": market_state.timestamp.isoformat(),
        "btc_binance": market_state.btc_price_binance,
        "btc_chainlink": market_state.btc_price_chainlink,
        "yes_price": market_state.polymarket_yes_price,
        "no_price": market_state.polymarket_no_price,
        "time_to_expiry": market_state.time_to_expiry,
        "model_probability": signal.model_up_probability,
        "market_probability": signal.market_up_probability,
        "edge": signal.edge,
        "recommended_action": signal.recommended_action,
        "confidence": signal.confidence,
        "kelly_fraction": signal.kelly_fraction,
    }

    state.recent_signals.append(signal_data)
    if len(state.recent_signals) > state.max_signals:
        state.recent_signals = state.recent_signals[-state.max_signals:]

    # Record opportunity if edge is significant
    if abs(signal.edge) >= 0.05:  # 5% edge
        state.opportunities_detected += 1

        opportunity = Opportunity(
            timestamp=market_state.timestamp,
            market_id=market_state.market_id,
            market_question=state.aggregator.current_market.question if state.aggregator.current_market else "",
            btc_price_binance=market_state.btc_price_binance,
            btc_price_chainlink=market_state.btc_price_chainlink,
            market_start_price=market_state.market_start_price,
            yes_price=market_state.polymarket_yes_price,
            no_price=market_state.polymarket_no_price,
            spread=market_state.polymarket_spread,
            time_to_expiry=market_state.time_to_expiry,
            model_probability=signal.model_up_probability,
            market_probability=signal.market_up_probability,
            edge=signal.edge,
            recommended_action=signal.recommended_action,
            confidence=signal.confidence,
            kelly_fraction=signal.kelly_fraction,
            volatility=market_state.volatility_15m,
        )

        state.aggregator.record_opportunity(opportunity)


# Create FastAPI app
app = FastAPI(
    title="PolyBot Monitor",
    description="Monitoring dashboard for PolyBot BTC 15-min prediction market bot",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Dashboard home page."""
    uptime = (datetime.now(timezone.utc) - state.start_time).total_seconds()

    # Get current status
    status_data = state.aggregator.get_status() if state.aggregator else {}
    market = status_data.get("market", {})
    prices = status_data.get("prices", {})
    connections = status_data.get("connections", {})

    # Format recent opportunities
    opportunities_html = ""
    if state.aggregator:
        for opp in reversed(state.aggregator.opportunities[-10:]):
            opp_dict = opp.to_dict()
            color = "green" if opp.recommended_action == "BUY_YES" else "red" if opp.recommended_action == "BUY_NO" else "gray"
            opportunities_html += f"""
            <tr>
                <td>{opp_dict['timestamp'][:19]}</td>
                <td>{opp_dict['time_to_expiry_formatted']}</td>
                <td>${opp_dict['btc_price_chainlink']:,.2f}</td>
                <td>{opp_dict['btc_change_pct']:+.2f}%</td>
                <td>{opp_dict['yes_price']:.3f}</td>
                <td>{opp_dict['model_probability']:.3f}</td>
                <td style="color: {color}; font-weight: bold;">{opp_dict['edge_pct']:+.1f}%</td>
                <td style="color: {color}; font-weight: bold;">{opp_dict['recommended_action']}</td>
            </tr>
            """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PolyBot Monitor</title>
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
            h1, h2 {{ color: #00d4ff; }}
            .card {{ background: #16213e; border-radius: 8px; padding: 20px; margin: 10px 0; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }}
            .stat {{ text-align: center; }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #00d4ff; }}
            .stat-label {{ font-size: 12px; color: #888; text-transform: uppercase; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #333; }}
            th {{ color: #00d4ff; }}
            .connected {{ color: #4caf50; }}
            .disconnected {{ color: #f44336; }}
            .price-up {{ color: #4caf50; }}
            .price-down {{ color: #f44336; }}
        </style>
    </head>
    <body>
        <h1>PolyBot Monitor</h1>

        <div class="grid">
            <div class="card stat">
                <div class="stat-value">{uptime/3600:.1f}h</div>
                <div class="stat-label">Uptime</div>
            </div>
            <div class="card stat">
                <div class="stat-value">{state.signals_generated:,}</div>
                <div class="stat-label">Signals Generated</div>
            </div>
            <div class="card stat">
                <div class="stat-value">{state.opportunities_detected:,}</div>
                <div class="stat-label">Opportunities Detected</div>
            </div>
            <div class="card stat">
                <div class="stat-value">{market.get('time_to_expiry', 0):.0f}s</div>
                <div class="stat-label">Time to Expiry</div>
            </div>
        </div>

        <div class="card">
            <h2>Current Market</h2>
            <p><strong>Question:</strong> {market.get('question', 'No market')}</p>
            <p><strong>Market ID:</strong> {market.get('id', 'N/A')[:20]}...</p>
        </div>

        <div class="grid">
            <div class="card">
                <h2>BTC Prices</h2>
                <table>
                    <tr><td>Binance</td><td><strong>${prices.get('btc_binance', 0):,.2f}</strong></td></tr>
                    <tr><td>Chainlink (resolution)</td><td><strong>${prices.get('btc_chainlink', 0):,.2f}</strong></td></tr>
                    <tr><td>Spread</td><td class="{'price-up' if prices.get('btc_spread', 0) > 0 else 'price-down'}">${prices.get('btc_spread', 0):+.2f}</td></tr>
                </table>
            </div>

            <div class="card">
                <h2>Polymarket Odds</h2>
                <table>
                    <tr><td>YES Price</td><td><strong>{prices.get('yes_price', 0.5):.3f}</strong></td></tr>
                    <tr><td>NO Price</td><td><strong>{prices.get('no_price', 0.5):.3f}</strong></td></tr>
                </table>
            </div>

            <div class="card">
                <h2>Connections</h2>
                <table>
                    <tr><td>Binance</td><td class="{'connected' if connections.get('binance') else 'disconnected'}">{'Connected' if connections.get('binance') else 'Disconnected'}</td></tr>
                    <tr><td>Chainlink</td><td class="{'connected' if connections.get('chainlink') else 'disconnected'}">{'Connected' if connections.get('chainlink') else 'Disconnected'}</td></tr>
                    <tr><td>Polymarket</td><td class="{'connected' if connections.get('polymarket') else 'disconnected'}">{'Connected' if connections.get('polymarket') else 'Disconnected'}</td></tr>
                </table>
            </div>
        </div>

        <div class="card">
            <h2>Recent Opportunities (5%+ Edge)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Expiry</th>
                        <th>BTC Price</th>
                        <th>Change</th>
                        <th>YES</th>
                        <th>Model P</th>
                        <th>Edge</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    {opportunities_html if opportunities_html else '<tr><td colspan="8">No opportunities detected yet</td></tr>'}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>API Endpoints</h2>
            <ul>
                <li><a href="/health" style="color: #00d4ff;">/health</a> - Health check</li>
                <li><a href="/status" style="color: #00d4ff;">/status</a> - Full status</li>
                <li><a href="/opportunities" style="color: #00d4ff;">/opportunities</a> - Recent opportunities</li>
                <li><a href="/signals" style="color: #00d4ff;">/signals</a> - Recent signals</li>
                <li><a href="/docs" style="color: #00d4ff;">/docs</a> - API documentation</li>
            </ul>
        </div>

        <p style="color: #666; font-size: 12px; text-align: center;">
            Auto-refreshes every 5 seconds | PolyBot v1.0.0
        </p>
    </body>
    </html>
    """
    return html


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    uptime = (datetime.now(timezone.utc) - state.start_time).total_seconds()
    status_data = state.aggregator.get_status() if state.aggregator else {}

    return HealthResponse(
        status="ok" if state.aggregator and state.aggregator._running else "starting",
        uptime_seconds=uptime,
        connections=status_data.get("connections", {}),
    )


@app.get("/status", response_model=StatusResponse)
async def status():
    """Full status endpoint."""
    uptime = (datetime.now(timezone.utc) - state.start_time).total_seconds()
    status_data = state.aggregator.get_status() if state.aggregator else {}

    market_data = status_data.get("market", {})
    prices = status_data.get("prices", {})

    return StatusResponse(
        running=state.aggregator._running if state.aggregator else False,
        market=MarketResponse(
            market_id=market_data.get("id"),
            question=market_data.get("question"),
            time_to_expiry_seconds=market_data.get("time_to_expiry", 0) or 0,
            yes_price=prices.get("yes_price", 0.5),
            no_price=prices.get("no_price", 0.5),
            spread=0.0,
        ) if market_data.get("id") else None,
        prices=PricesResponse(
            btc_binance=prices.get("btc_binance", 0),
            btc_chainlink=prices.get("btc_chainlink", 0),
            btc_spread=prices.get("btc_spread", 0),
            chainlink_age_seconds=0,
        ),
        volatility=status_data.get("volatility", {}),
        connections=status_data.get("connections", {}),
        signals_generated=state.signals_generated,
        opportunities_count=status_data.get("opportunities_count", 0),
        uptime_seconds=uptime,
    )


@app.get("/opportunities")
async def opportunities():
    """Get recent opportunities."""
    if not state.aggregator:
        return {"opportunities": []}

    return {
        "count": len(state.aggregator.opportunities),
        "opportunities": [opp.to_dict() for opp in reversed(state.aggregator.opportunities[-50:])]
    }


@app.get("/signals")
async def signals():
    """Get recent signals."""
    return {
        "count": len(state.recent_signals),
        "signals": list(reversed(state.recent_signals[-50:]))
    }


@app.get("/market")
async def market():
    """Get current market details."""
    if not state.aggregator or not state.aggregator.current_market:
        raise HTTPException(status_code=404, detail="No active market")

    market = state.aggregator.current_market
    return {
        "market_id": market.market_id,
        "condition_id": market.condition_id,
        "question": market.question,
        "slug": market.slug,
        "yes_token_id": market.token_id,
        "no_token_id": market.no_token_id,
        "yes_price": market.yes_price,
        "no_price": market.no_price,
        "start_price": market.start_price,
        "end_time": market.end_time.isoformat(),
        "time_to_expiry_seconds": market.time_to_expiry_seconds,
        "is_expired": market.is_expired,
    }


@app.get("/signal/current")
async def current_signal():
    """Get current trading signal."""
    if not state.aggregator or not state.aggregator.current_state or not state.probability_model:
        raise HTTPException(status_code=404, detail="No current state")

    signal = state.probability_model.generate_signal(state.aggregator.current_state)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_probability": signal.model_up_probability,
        "market_probability": signal.market_up_probability,
        "edge": signal.edge,
        "edge_pct": signal.edge * 100,
        "recommended_action": signal.recommended_action,
        "confidence": signal.confidence,
        "kelly_fraction": signal.kelly_fraction,
        "should_trade": signal.recommended_action != "HOLD",
    }


# Entry point for running directly
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
