#!/usr/bin/env python3
"""
Run scalping strategy backtest across multiple pairs.

Usage:
    python run_scalping_backtest.py
    python run_scalping_backtest.py --pairs BTC/USD ETH/USD
    python run_scalping_backtest.py --days 14 --interval 15
    python run_scalping_backtest.py --take-profit 2.0 --stop-loss 1.0
"""

import argparse
import sys
from datetime import datetime

from src.backtest.scalping_backtester import (
    ScalpingBacktester,
    ScalpingBacktestConfig,
)
from src.strategy.scalping_strategy import ScalpingConfig


# Default pairs to test (high liquidity)
DEFAULT_PAIRS = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "XRP/USD",
    "DOGE/USD",
    "ADA/USD",
    "AVAX/USD",
    "DOT/USD",
    "LINK/USD",
    "MATIC/USD",
    "ATOM/USD",
    "NEAR/USD",
]


def main():
    parser = argparse.ArgumentParser(
        description="Backtest scalping strategy across multiple pairs"
    )

    # Pair selection
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=DEFAULT_PAIRS,
        help="Trading pairs to test (default: top 10 by volume)"
    )

    # Time range
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of historical data to test (default: 30)"
    )

    # Candle interval
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        choices=[1, 5, 15, 30, 60],
        help="Candle interval in minutes (default: 5)"
    )

    # Strategy parameters
    parser.add_argument(
        "--take-profit",
        type=float,
        default=1.5,
        help="Take profit percentage (default: 1.5)"
    )

    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.75,
        help="Stop loss percentage (default: 0.75)"
    )

    parser.add_argument(
        "--min-confirmations",
        type=int,
        default=2,
        help="Minimum indicator confirmations for entry (default: 2)"
    )

    # Capital
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital in USD (default: 10000)"
    )

    parser.add_argument(
        "--position-size",
        type=float,
        default=20.0,
        help="Position size as percentage of capital (default: 20)"
    )

    # Fee
    parser.add_argument(
        "--fee",
        type=float,
        default=0.26,
        help="Trading fee percentage (default: 0.26 for Kraken)"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save report to file"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SCALPING STRATEGY BACKTESTER")
    print("=" * 60)
    print()
    print(f"Pairs: {', '.join(args.pairs)}")
    print(f"Period: Last {args.days} days")
    print(f"Candle interval: {args.interval} minutes")
    print(f"Take profit: {args.take_profit}%")
    print(f"Stop loss: {args.stop_loss}%")
    print(f"Fee: {args.fee}%")
    print(f"Net target: {args.take_profit - (args.fee * 2):.2f}%")
    print()

    # Create configs
    scalping_config = ScalpingConfig(
        take_profit_percent=args.take_profit,
        stop_loss_percent=args.stop_loss,
        min_confirmations=args.min_confirmations,
        fee_percent=args.fee,
    )

    backtest_config = ScalpingBacktestConfig(
        initial_capital=args.capital,
        position_size_percent=args.position_size,
        fee_percent=args.fee,
        candle_interval=args.interval,
    )

    # Create backtester
    backtester = ScalpingBacktester(
        config=backtest_config,
        scalping_config=scalping_config
    )

    # Run backtest
    print("Running backtest...")
    print("-" * 60)

    try:
        results = backtester.run_multi_pair_backtest(
            pairs=args.pairs,
            days=args.days
        )
    except Exception as e:
        print(f"Error running backtest: {e}")
        sys.exit(1)

    print()

    # Generate report
    report = backtester.generate_report(
        results,
        output_path=args.output
    )

    print(report)

    # Get profitable pairs
    profitable = backtester.get_profitable_pairs(results)

    if profitable:
        print()
        print("=" * 60)
        print("PROFITABLE PAIRS FOR SCALPING")
        print("=" * 60)
        for pair in profitable:
            r = results[pair]
            print(f"  {pair}: {r.total_trades} trades, {r.win_rate:.1%} win rate, {r.total_pnl_percent:.2f}% P&L")
        print()
        print("To use these pairs, update your config or run:")
        print(f"  --pairs {' '.join(profitable)}")
    else:
        print()
        print("No profitable pairs found with current settings.")
        print("Consider adjusting take-profit/stop-loss or trying different intervals.")

    return 0 if profitable else 1


if __name__ == "__main__":
    sys.exit(main())
