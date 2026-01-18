#!/usr/bin/env python3
"""
Kraken Momentum Trader - Main Entry Point

A cryptocurrency momentum trading application that:
- Runs on Replit (short-term testing) or AWS (production)
- Connects to Kraken API for automated trading
- Uses momentum strategy with proper risk management

IMPORTANT: Start with paper trading for 2+ weeks before any real capital.
"""

import argparse
import signal
import sys
from datetime import datetime, timezone
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
================================================================
           KRAKEN MOMENTUM TRADER
================================================================
  WARNING: Trading cryptocurrencies involves significant
  risk of loss. Paper trade for 2+ weeks before using
  real capital.
================================================================
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
        print(f"  - Avg Entry: ${pos['entry_price']:.2f}")
        print(f"  - Total Size: {pos['size']:.6f}")
        print(f"  - Total Cost: ${pos['total_cost_usd']:.2f}")
        print(f"  - Entries: {pos['num_entries']} (Martingale)")
        print(f"  - Peak Price: ${pos['peak_price']:.2f}")
        print(f"  - Trailing Stop: {'Active' if pos['trailing_stop_active'] else 'Inactive'}")

    print(f"\n=== Risk Limits ===")
    limits = status['risk_limits']
    print(f"Exposure Available: ${limits['exposure_available']:.2f}")
    print(f"Max Position Size: ${limits['max_position_size']:.2f}")
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
        min_trades=20
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
    print(f"  Initial Stop Loss: {settings.risk.initial_stop_loss_percent}% (0=disabled)")
    print(f"  Trailing Stop: {settings.risk.trailing_stop_percent}% from peak")
    print(f"  Trailing Activation: {settings.risk.trailing_stop_activation_percent}% profit")
    print(f"  Circuit Breaker: {settings.risk.circuit_breaker_consecutive_losses} losses -> {settings.risk.circuit_breaker_cooldown_hours}h cooldown")

    print("\n[Martingale]")
    mg = settings.risk.martingale
    print(f"  Enabled: {mg.enabled}")
    print(f"  Max Entries: {mg.max_entries}")
    print(f"  Size Multiplier: {mg.size_multiplier}x")
    print(f"  Add-on Trigger: {mg.add_on_drop_percent}% drop from avg entry")
    print(f"  Require RSI Oversold: {mg.require_rsi_oversold}")
    print(f"  Require EMA Trend: {mg.require_ema_trend}")

    print("\n[Exchange]")
    print(f"  Name: {settings.exchange.name}")
    print(f"  Rate Limit: {settings.exchange.rate_limit_calls_per_minute} calls/min")


def run_backtest(
    pair: str = "BTC/USD",
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: int = 60,
    capital: float = 10000.0,
    export: bool = False
) -> int:
    """
    Run a backtest on historical data.

    Args:
        pair: Trading pair (e.g., "BTC/USD").
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        interval: Candle interval in minutes.
        capital: Initial capital.
        export: Whether to export results to files.

    Returns:
        Exit code.
    """
    from datetime import timedelta
    from src.backtest import BacktestConfig, BacktestRunner

    print_banner()
    print("=== BACKTEST MODE ===\n")

    # Parse dates
    if end:
        end_date = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_date = datetime.now(timezone.utc)

    if start:
        start_date = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        # Default to 6 months ago
        start_date = end_date - timedelta(days=180)

    print(f"Pair: {pair}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Interval: {interval} minutes")
    print(f"Initial Capital: ${capital:,.2f}")
    print()

    try:
        # Create config
        config = BacktestConfig(
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            initial_capital=capital
        )

        # Run backtest
        runner = BacktestRunner(config)
        results = runner.run()

        # Print report
        runner.print_report()

        # Export if requested
        if export:
            runner.export_results()

        # Return 0 if profitable, 1 otherwise
        net_pnl = results.metrics.total_pnl - results.metrics.total_fees
        return 0 if net_pnl > 0 else 1

    except Exception as e:
        print(f"Backtest error: {e}")
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Kraken Momentum Trader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Run the trader
  python main.py --config config.yaml         # Use specific config
  python main.py --validate                   # Validate strategy metrics
  python main.py --show-config                # Show current config

Backtest Examples:
  python main.py --backtest --pair BTC/USD --start 2024-01-01 --end 2024-06-30
  python main.py --backtest --pair ETH/USD --start 2023-01-01 --end 2023-12-31 --interval 240
  python main.py --backtest --pair BTC/USD --export  # Export results to CSV/JSON
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

    # Backtest arguments
    parser.add_argument(
        "--backtest", "-b",
        action="store_true",
        help="Run backtest on historical data"
    )
    parser.add_argument(
        "--pair", "-p",
        default="BTC/USD",
        help="Trading pair for backtest (default: BTC/USD)"
    )
    parser.add_argument(
        "--start", "-s",
        help="Start date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", "-e",
        help="End date for backtest (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=60,
        help="Candle interval in minutes (default: 60)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital for backtest (default: 10000)"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export backtest results to CSV/JSON files"
    )

    args = parser.parse_args()

    if args.show_config:
        show_config()
        return 0

    if args.validate:
        return validate_strategy(args.config)

    if args.backtest:
        return run_backtest(
            pair=args.pair,
            start=args.start,
            end=args.end,
            interval=args.interval,
            capital=args.capital,
            export=args.export
        )

    return run_trader(args.config)


if __name__ == "__main__":
    sys.exit(main())
