#!/usr/bin/env python3
"""
Kraken Momentum Trader - Main Entry Point

A cryptocurrency momentum trading application that:
- Runs on Replit (short-term testing) or AWS (production)
- Connects to Kraken API for automated trading
- Uses momentum strategy with proper risk management

IMPORTANT: Start with paper trading for 3+ months before any real capital.
"""

import argparse
import signal
import sys
from typing import Optional

from src.config.settings import get_settings, reload_settings
from src.config.platform import detect_platform, get_platform_config
from src.core.trader import Trader
from src.observability.logger import configure_logging, get_logger


def setup_signal_handlers(trader: Trader) -> None:
    """Setup signal handlers for graceful shutdown."""

    def handle_signal(signum, frame):
        logger = get_logger("main")
        logger.info(f"Received signal {signum}, initiating shutdown")
        trader.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def print_banner() -> None:
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║           KRAKEN MOMENTUM TRADER                             ║
║                                                              ║
║  WARNING: Trading cryptocurrencies involves significant      ║
║  risk of loss. Paper trade for 3+ months before using        ║
║  real capital.                                               ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_status(trader: Trader) -> None:
    """Print current trader status."""
    status = trader.get_status()
    print("\n=== Trader Status ===")
    print(f"State: {status['state']}")
    print(f"Paper Trading: {status['paper_trading']}")
    print(f"Has Position: {status['has_position']}")

    if status['position']:
        pos = status['position']
        print(f"  - Pair: {pos['pair']}")
        print(f"  - Side: {pos['side']}")
        print(f"  - Entry: ${pos['entry_price']:.2f}")
        print(f"  - Size: {pos['size']:.6f}")

    print(f"\n=== Risk Limits ===")
    limits = status['risk_limits']
    print(f"Daily Trades Remaining: {limits['daily_trades_remaining']}")
    print(f"Daily Loss Remaining: ${limits['daily_loss_remaining']:.2f}")
    print(f"Exposure Available: ${limits['exposure_available']:.2f}")
    print(f"Circuit Breaker: {limits['circuit_breaker']}")

    print(f"\n=== Performance ===")
    metrics = status['metrics']
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Total P&L: ${metrics['total_pnl']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_percent']:.1f}%")
    print()


def run_trader(config_path: Optional[str] = None) -> int:
    """
    Run the trading bot.

    Args:
        config_path: Optional path to config file.

    Returns:
        Exit code.
    """
    # Load settings
    if config_path:
        settings = reload_settings(config_path)
    else:
        settings = get_settings()

    # Configure logging
    configure_logging(
        level=settings.logging.level,
        format_type=settings.logging.format,
        include_timestamps=settings.logging.include_timestamps
    )

    logger = get_logger("main")

    # Print startup info
    print_banner()

    platform = detect_platform()
    platform_config = get_platform_config()

    logger.info(
        "Starting Kraken Momentum Trader",
        platform=platform.value,
        paper_trading=settings.trading.paper_trading,
        pairs=settings.trading.pairs
    )

    print(f"Platform: {platform.value}")
    print(f"Paper Trading: {settings.trading.paper_trading}")
    print(f"Trading Pairs: {', '.join(settings.trading.pairs)}")
    print(f"Check Interval: {settings.trading.check_interval_seconds}s")
    print()

    # Safety check
    if not settings.trading.paper_trading:
        print("=" * 60)
        print("WARNING: LIVE TRADING MODE ENABLED")
        print("Real money will be used for trades!")
        print("=" * 60)
        response = input("Type 'CONFIRM' to continue: ")
        if response.strip() != "CONFIRM":
            print("Aborting.")
            return 1

    try:
        # Create and run trader
        trader = Trader(settings=settings)

        # Setup signal handlers
        setup_signal_handlers(trader)

        # Print initial status
        if trader.initialize():
            print_status(trader)

        # Run the trading loop
        trader.run()

        # Print final status
        print_status(trader)

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


def validate_strategy(config_path: Optional[str] = None) -> int:
    """
    Check if strategy meets validation criteria.

    Args:
        config_path: Optional path to config file.

    Returns:
        Exit code (0 if validated, 1 if not).
    """
    from src.persistence.file_store import StateStore
    from src.observability.metrics import MetricsTracker

    print("=== Strategy Validation ===\n")

    # Load metrics
    store = StateStore()
    metrics_data = store.load_metrics()

    if not metrics_data:
        print("No metrics data found. Run paper trading first.")
        return 1

    tracker = MetricsTracker.from_dict(metrics_data)
    is_valid, reason = tracker.is_strategy_validated(
        min_sharpe=1.0,
        max_drawdown=20.0,
        min_trades=100
    )

    metrics = tracker.get_metrics()

    print(f"Total Trades: {metrics.total_trades}")
    print(f"Win Rate: {metrics.win_rate:.1f}%")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown_percent:.1f}%")
    print()

    if is_valid:
        print("VALIDATION PASSED")
        print("Strategy meets criteria for live trading consideration.")
        return 0
    else:
        print(f"VALIDATION FAILED: {reason}")
        print("Continue paper trading until criteria are met.")
        return 1


def show_config() -> None:
    """Display current configuration."""
    settings = get_settings()

    print("=== Current Configuration ===\n")

    print("[Trading]")
    print(f"  Pairs: {settings.trading.pairs}")
    print(f"  Check Interval: {settings.trading.check_interval_seconds}s")
    print(f"  Paper Trading: {settings.trading.paper_trading}")

    print("\n[Strategy]")
    print(f"  Name: {settings.strategy.name}")
    print(f"  RSI Period: {settings.strategy.rsi_period}")
    print(f"  RSI Oversold: {settings.strategy.rsi_oversold}")
    print(f"  RSI Overbought: {settings.strategy.rsi_overbought}")
    print(f"  EMA Short: {settings.strategy.ema_short_period}")
    print(f"  EMA Long: {settings.strategy.ema_long_period}")

    print("\n[Risk]")
    print(f"  Max Position: ${settings.risk.max_position_size_usd} ({settings.risk.max_position_percent}%)")
    print(f"  Max Exposure: {settings.risk.max_total_exposure_percent}%")
    print(f"  Stop Loss: {settings.risk.stop_loss_percent}%")
    print(f"  Take Profit: {settings.risk.take_profit_percent}%")
    print(f"  Max Daily Loss: {settings.risk.max_daily_loss_percent}%")
    print(f"  Max Daily Trades: {settings.risk.max_daily_trades}")
    print(f"  Circuit Breaker: {settings.risk.circuit_breaker_consecutive_losses} losses")

    print("\n[Exchange]")
    print(f"  Name: {settings.exchange.name}")
    print(f"  Rate Limit: {settings.exchange.rate_limit_calls_per_minute} calls/min")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Kraken Momentum Trader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run the trader
  python main.py --config config.yaml  # Use specific config
  python main.py --validate         # Validate strategy metrics
  python main.py --show-config      # Show current config
        """
    )

    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate strategy meets live trading criteria"
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show current configuration"
    )

    args = parser.parse_args()

    if args.show_config:
        show_config()
        return 0

    if args.validate:
        return validate_strategy(args.config)

    return run_trader(args.config)


if __name__ == "__main__":
    sys.exit(main())
