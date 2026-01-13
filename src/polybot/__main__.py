"""CLI entry point for PolyBot."""

import asyncio
import sys
from pathlib import Path

import structlog


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    import logging

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )


def main() -> None:
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PolyBot - Polymarket BTC 15-min Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in paper trading mode
  uv run python -m polybot --mode paper

  # Run with RL disabled (probability model only)
  uv run python -m polybot --mode paper --no-rl

  # Run with custom capital
  uv run python -m polybot --mode paper --capital 500

  # Test data connections
  uv run python -m polybot test-data
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command (default)
    run_parser = subparsers.add_parser("run", help="Run the trading bot")
    run_parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode (default: paper)",
    )
    run_parser.add_argument(
        "--capital",
        type=float,
        default=1000.0,
        help="Initial capital in USD (default: 1000)",
    )
    run_parser.add_argument(
        "--no-rl",
        action="store_true",
        help="Disable RL agent, use probability model only",
    )
    run_parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to load/save RL model",
    )

    # Test data command
    test_parser = subparsers.add_parser("test-data", help="Test data connections")
    test_parser.add_argument(
        "--source",
        choices=["polymarket", "coinbase", "chainlink", "all"],
        default="all",
        help="Data source to test",
    )

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest on historical data")
    backtest_parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    backtest_parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD)",
    )

    # RL status command
    rl_parser = subparsers.add_parser("rl-status", help="Show RL agent status")
    rl_parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("./models/trading_agent"),
        help="Path to RL model",
    )

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Default to run if no command specified
    if args.command is None:
        args.command = "run"
        args.mode = "paper"
        args.capital = 1000.0
        args.no_rl = False
        args.model_path = None

    # Execute command
    if args.command == "run":
        run_bot(args)
    elif args.command == "test-data":
        test_data(args)
    elif args.command == "backtest":
        run_backtest(args)
    elif args.command == "rl-status":
        show_rl_status(args)
    else:
        parser.print_help()


def run_bot(args) -> None:
    """Run the trading bot."""
    from polybot.bot import PolyBot, TradingMode

    mode = TradingMode.PAPER if args.mode == "paper" else TradingMode.PROBABILITY_MODEL

    bot = PolyBot(
        mode=mode,
        initial_capital=args.capital,
        use_rl=not args.no_rl,
        model_path=args.model_path,
    )

    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\nShutting down...")
        asyncio.run(bot.stop())


def test_data(args) -> None:
    """Test data source connections."""

    async def _test():
        print("Testing data connections...\n")

        if args.source in ["polymarket", "all"]:
            print("=== Polymarket ===")
            try:
                from polybot.data.polymarket import PolymarketClient

                async with PolymarketClient() as client:
                    markets = await client.get_btc_15m_markets()
                    print(f"Found {len(markets)} BTC 15-min markets")
                    for m in markets[:3]:
                        print(f"  - {m.question[:50]}...")
                        print(f"    YES: {m.yes_price:.3f}, Expiry: {m.time_to_expiry_seconds:.0f}s")
                print("✓ Polymarket OK\n")
            except Exception as e:
                print(f"✗ Polymarket ERROR: {e}\n")

        if args.source in ["coinbase", "all"]:
            print("=== Coinbase ===")
            try:
                from polybot.data.coinbase import CoinbaseClient
                import asyncio as aio

                client = CoinbaseClient()
                await client.connect()

                # Run the websocket for a few seconds to receive data
                async def collect_data():
                    for _ in range(50):  # Check 50 times over ~5 seconds
                        if client.current_price > 0:
                            break
                        await aio.sleep(0.1)

                # Start the stream in background and collect data
                stream_task = aio.create_task(client.run())
                await collect_data()
                stream_task.cancel()

                if client.current_price > 0:
                    print(f"Current BTC price: ${client.current_price:,.2f}")
                    print(f"Volatility (15m): {client.calculate_volatility(15):.4f}")
                    print("✓ Coinbase OK\n")
                else:
                    print("Warning: No price data received yet")
                    print("✓ Coinbase connected (data pending)\n")

                await client.disconnect()
            except Exception as e:
                print(f"✗ Coinbase ERROR: {e}\n")

        if args.source in ["chainlink", "all"]:
            print("=== Chainlink ===")
            try:
                from polybot.data.chainlink import ChainlinkClient

                client = ChainlinkClient()
                await client.connect()

                update = await client.get_latest_price()
                if update:
                    print(f"Chainlink BTC/USD: ${update.price:,.2f}")
                    print(f"Last update: {update.timestamp.isoformat()}")
                print("✓ Chainlink OK\n")

                await client.disconnect()
            except Exception as e:
                print(f"✗ Chainlink ERROR: {e}\n")

        print("Data connection test complete.")

    asyncio.run(_test())


def run_backtest(args) -> None:
    """Run backtest on historical data."""
    print("Backtest not yet implemented.")
    print("To run a backtest, you need to:")
    print("1. Collect historical data using the data pipeline")
    print("2. Implement the backtest runner")
    print("3. Run: uv run python -m polybot backtest --start-date 2024-01-01")


def show_rl_status(args) -> None:
    """Show RL agent status and statistics."""
    import json

    print("\n" + "=" * 60)
    print("RL AGENT STATUS")
    print("=" * 60)

    model_path = args.model_path
    stats_path = model_path.parent / "rl_stats.json"

    # Check if model exists
    model_zip = model_path.with_suffix(".zip")
    if model_zip.exists():
        print(f"Model file:       {model_zip}")
        print(f"Model size:       {model_zip.stat().st_size / 1024:.1f} KB")
    elif model_path.exists():
        print(f"Model file:       {model_path}")
    else:
        print(f"Model file:       NOT FOUND at {model_path}")
        print("\nNo trained model found. Run the bot to start training.")
        print("=" * 60 + "\n")
        return

    # Load and display stats
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

        print(f"\nTraining Statistics:")
        print(f"  Created:          {stats.get('created_at', 'N/A')}")
        print(f"  Last saved:       {stats.get('last_save', 'N/A')}")
        print(f"  Total predictions: {stats.get('total_predictions', 0)}")
        print(f"  Training steps:   {stats.get('total_training_steps', 0)}")
        print(f"  Training sessions: {stats.get('training_sessions', 0)}")

        exploration = stats.get('exploration_count', 0)
        exploitation = stats.get('exploitation_count', 0)
        total = exploration + exploitation
        if total > 0:
            print(f"  Exploration rate: {exploration / total * 100:.1f}%")

        print(f"\nAction Distribution:")
        for action, count in stats.get('action_counts', {}).items():
            pct = count / total * 100 if total > 0 else 0
            print(f"  {action:10s}: {count:5d} ({pct:5.1f}%)")

        rewards = stats.get('episode_rewards', [])
        if rewards:
            import numpy as np
            print(f"\nReward Statistics (last {len(rewards)} episodes):")
            print(f"  Mean reward:      {np.mean(rewards):.4f}")
            print(f"  Std reward:       {np.std(rewards):.4f}")
            print(f"  Min reward:       {np.min(rewards):.4f}")
            print(f"  Max reward:       {np.max(rewards):.4f}")
    else:
        print(f"\nStats file not found at {stats_path}")
        print("Run the bot to generate training statistics.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
