"""
Compatibility module - CoinbaseClient is an alias for BinanceClient.

The original bot.py imports CoinbaseClient, but we use BinanceClient
with multi-exchange fallback support.
"""

from polybot.data.binance import BinanceClient

# Alias for compatibility
CoinbaseClient = BinanceClient

__all__ = ["CoinbaseClient"]
