"""
Backtesting framework for the Kraken Momentum Trader.

This module provides tools to validate the momentum + Martingale strategy
against historical data and measure profitability over specific time periods.

Components:
- BacktestConfig: Configuration for backtest parameters
- HistoricalDataManager: Fetches and caches OHLC data from Kraken
- SimulatedPosition: Position tracking for backtest simulation
- BacktestEngine: Core simulation loop
- ResultsReporter: Generates reports and exports
- BacktestRunner: Orchestrates the complete backtest process

Usage:
    from src.backtest import BacktestRunner, BacktestConfig
    from datetime import datetime

    config = BacktestConfig(
        pair="BTC/USD",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),
        initial_capital=10000.0
    )

    runner = BacktestRunner(config)
    results = runner.run()
    runner.print_report()
    runner.export_results()
"""

from .config import BacktestConfig
from .data import HistoricalDataManager
from .engine import BacktestEngine, BacktestResults, EquityPoint
from .position import SimulatedPosition, SimulatedEntry, CompletedTrade
from .reporter import ResultsReporter, PeriodStats
from .runner import BacktestRunner, run_backtest

__all__ = [
    # Config
    "BacktestConfig",

    # Data
    "HistoricalDataManager",

    # Engine
    "BacktestEngine",
    "BacktestResults",
    "EquityPoint",

    # Position
    "SimulatedPosition",
    "SimulatedEntry",
    "CompletedTrade",

    # Reporter
    "ResultsReporter",
    "PeriodStats",

    # Runner
    "BacktestRunner",
    "run_backtest",
]
