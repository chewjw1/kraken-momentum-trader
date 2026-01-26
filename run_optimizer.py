#!/usr/bin/env python3
"""
Parameter Optimizer CLI

Finds optimal trading strategy parameters using Bayesian optimization (Optuna)
with 2 years of historical data from Binance via CCXT.

Usage:
    # Run with defaults (200 trials, BTC/USD + ETH/USD, 2 years)
    python run_optimizer.py

    # Specify pairs and trials
    python run_optimizer.py --pairs BTC/USD ETH/USD SOL/USD --trials 100

    # Quick test run
    python run_optimizer.py --pairs BTC/USD --trials 10

    # Full optimization with export
    python run_optimizer.py --pairs BTC/USD ETH/USD SOL/USD --trials 200 --export

    # Custom date range
    python run_optimizer.py --start 2024-01-22 --end 2026-01-22 --trials 200
"""

import argparse
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backtest.optimizer import OptimizerConfig, ParameterOptimizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize trading strategy parameters using Bayesian optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_optimizer.py --pairs BTC/USD ETH/USD --trials 10
  python run_optimizer.py --pairs BTC/USD --trials 200 --export
  python run_optimizer.py --start 2024-01-01 --end 2026-01-01 --objective sharpe
        """
    )

    # Pairs
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["BTC/USD", "ETH/USD"],
        help="Trading pairs to optimize across (default: BTC/USD ETH/USD)"
    )

    # Trials
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=200,
        help="Number of optimization trials (default: 200)"
    )

    # Timeout
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Timeout in seconds (default: 7200 = 2 hours)"
    )

    # Date range
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Default: 2 years ago"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Default: today"
    )

    # Objective
    parser.add_argument(
        "--objective",
        choices=["composite", "sharpe", "return", "profit_factor"],
        default="composite",
        help="Objective function to optimize (default: composite)"
    )

    # Capital
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital for backtests (default: 10000)"
    )

    # Interval
    parser.add_argument(
        "--interval",
        type=int,
        choices=[1, 5, 15, 30, 60, 240, 1440],
        default=60,
        help="Candle interval in minutes (default: 60)"
    )

    # Data source
    parser.add_argument(
        "--source",
        choices=["kraken", "ccxt", "yfinance"],
        default="kraken",
        help="Data source: 'kraken' (hourly, 1mo), 'ccxt' (Binance, may be geo-blocked), 'yfinance' (daily, 5+ years). Default: kraken"
    )

    # Export
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export optimized config to config/config.optimized.yaml"
    )
    parser.add_argument(
        "--export-path",
        type=str,
        default=None,
        help="Custom path for exported config"
    )

    # Study
    parser.add_argument(
        "--study-name",
        type=str,
        default="momentum_optimization",
        help="Name for the Optuna study (default: momentum_optimization)"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="data/optimization/study.db",
        help="Path for study database (default: data/optimization/study.db)"
    )

    # Per-pair optimization
    parser.add_argument(
        "--per-pair",
        action="store_true",
        help="Optimize parameters separately for each pair (recommended)"
    )

    # Other options
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear CCXT data cache before starting"
    )

    return parser.parse_args()


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("PARAMETER OPTIMIZER")
    print("=" * 60)

    # Parse dates
    if args.end:
        end_date = parse_date(args.end)
    else:
        end_date = datetime.now(timezone.utc)

    if args.start:
        start_date = parse_date(args.start)
    else:
        start_date = end_date - timedelta(days=730)  # 2 years

    print(f"\nConfiguration:")
    print(f"  Pairs: {', '.join(args.pairs)}")
    print(f"  Trials: {args.trials}" + (" per pair" if args.per_pair else " total"))
    print(f"  Mode: {'Per-pair (separate params)' if args.per_pair else 'Global (shared params)'}")
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    print(f"  Interval: {args.interval} minutes")
    print(f"  Objective: {args.objective}")
    print(f"  Data Source: {args.source}")
    print(f"  Initial Capital: ${args.capital:,.2f}")
    print(f"  Study: {args.study_name}")

    # Create optimizer config
    config = OptimizerConfig(
        n_trials=args.trials,
        timeout_seconds=args.timeout,
        objective=args.objective,
        pairs=args.pairs,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        interval=args.interval,
        data_source=args.source,
        study_name=args.study_name,
        storage_path=args.storage,
        export_config_path=args.export_path or "config/config.optimized.yaml",
    )

    # Create optimizer
    try:
        optimizer = ParameterOptimizer(config)
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install required dependencies:")
        print("  pip install ccxt optuna")
        sys.exit(1)

    # Clear cache if requested
    if args.clear_cache:
        print("\nClearing CCXT cache...")
        optimizer.data_provider.clear_cache()

    # Run optimization
    try:
        print("\n" + "-" * 60)

        if args.per_pair:
            # Per-pair optimization (separate parameters for each crypto)
            results = optimizer.optimize_per_pair(
                show_progress=not args.no_progress
            )

            # Print summary
            optimizer.print_per_pair_results()

            # Export if requested
            if args.export or args.export_path:
                export_path = optimizer.export_per_pair_config(args.export_path)
                print(f"\nOptimized per-pair config exported to: {export_path}")
                print("\nNote: Per-pair config requires code changes to load pair-specific settings.")
        else:
            # Global optimization (shared parameters for all pairs)
            best_params, best_score = optimizer.optimize(
                show_progress=not args.no_progress
            )

            # Print results
            optimizer.print_results()

            # Export if requested
            if args.export or args.export_path:
                export_path = optimizer.export_config(args.export_path)
                print(f"\nOptimized config exported to: {export_path}")
                print("\nTo use the optimized config:")
                print(f"  python main.py --backtest --config {export_path}")

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        if hasattr(optimizer, '_per_pair_results') and optimizer._per_pair_results:
            print("Partial per-pair results saved.")
            optimizer.print_per_pair_results()
        elif optimizer.study and len(optimizer.study.trials) > 0:
            print("Partial results are saved and can be resumed.")
            optimizer.print_results()
        sys.exit(1)

    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nOptimization complete!")


if __name__ == "__main__":
    main()
