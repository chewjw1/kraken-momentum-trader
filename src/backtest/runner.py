"""
Backtest runner - orchestrates the backtesting process.
"""

from datetime import datetime, timezone
from typing import Optional

from ..exchange.kraken_client import KrakenClient
from ..observability.logger import get_logger
from ..strategy.momentum_strategy import MomentumStrategy

from .config import BacktestConfig
from .data import HistoricalDataManager
from .engine import BacktestEngine, BacktestResults
from .reporter import ResultsReporter

logger = get_logger(__name__)


class BacktestRunner:
    """
    Orchestrates the complete backtesting process.

    Coordinates:
    - Historical data fetching and caching
    - Backtest engine execution
    - Results reporting and export
    """

    def __init__(
        self,
        config: BacktestConfig,
        client: Optional[KrakenClient] = None,
        strategy: Optional[MomentumStrategy] = None
    ):
        """
        Initialize the backtest runner.

        Args:
            config: Backtest configuration.
            client: Optional KrakenClient for data fetching.
            strategy: Optional MomentumStrategy instance.
        """
        self.config = config
        self.client = client
        self.strategy = strategy or MomentumStrategy()

        # Initialize components
        self.data_manager = HistoricalDataManager(
            cache_dir=config.cache_dir,
            use_cache=config.use_cache,
            client=client
        )
        self.engine = BacktestEngine(config, self.strategy)
        self.results: Optional[BacktestResults] = None

    def run(self, show_progress: bool = True) -> BacktestResults:
        """
        Run the complete backtest.

        Args:
            show_progress: Whether to show progress during data fetching.

        Returns:
            BacktestResults with all metrics and trades.
        """
        logger.info(
            "Starting backtest",
            pair=self.config.pair,
            start=self.config.start_date.isoformat(),
            end=self.config.end_date.isoformat(),
            interval=self.config.interval
        )

        # Progress callback
        def progress_callback(current: int, total: int) -> None:
            if show_progress:
                pct = (current / total * 100) if total > 0 else 0
                print(f"\rFetching data: {current}/{total} requests ({pct:.0f}%)...", end="", flush=True)

        # Fetch historical data
        print(f"Fetching historical data for {self.config.pair}...")
        candles = self.data_manager.get_ohlc_range(
            pair=self.config.pair,
            start=self.config.start_date,
            end=self.config.end_date,
            interval=self.config.interval,
            progress_callback=progress_callback if show_progress else None
        )

        if show_progress:
            print()  # New line after progress

        if not candles:
            raise ValueError("No historical data available for the specified period")

        print(f"Loaded {len(candles)} candles from {candles[0].timestamp} to {candles[-1].timestamp}")

        # Run backtest
        print(f"Running backtest simulation...")
        self.results = self.engine.run(candles)

        logger.info(
            "Backtest completed",
            trades=len(self.results.trades),
            win_rate=self.results.metrics.win_rate,
            net_pnl=self.results.metrics.total_pnl - self.results.metrics.total_fees
        )

        return self.results

    def print_report(self) -> None:
        """Print the backtest report to console."""
        if not self.results:
            raise ValueError("No results available. Run backtest first.")

        reporter = ResultsReporter(self.results)
        reporter.print_summary()

    def export_results(
        self,
        output_dir: str = "data/backtest_results",
        export_csv: bool = True,
        export_json: bool = True,
        export_equity: bool = True
    ) -> None:
        """
        Export backtest results to files.

        Args:
            output_dir: Directory for output files.
            export_csv: Whether to export trades CSV.
            export_json: Whether to export full JSON.
            export_equity: Whether to export equity curve CSV.
        """
        if not self.results:
            raise ValueError("No results available. Run backtest first.")

        reporter = ResultsReporter(self.results)

        # Generate filename prefix
        pair_clean = self.config.pair.replace("/", "_")
        start_str = self.config.start_date.strftime("%Y%m%d")
        end_str = self.config.end_date.strftime("%Y%m%d")
        prefix = f"{output_dir}/{pair_clean}_{start_str}_{end_str}"

        if export_csv:
            reporter.export_trades_csv(f"{prefix}_trades.csv")

        if export_json:
            reporter.export_trades_json(f"{prefix}_results.json")

        if export_equity:
            reporter.export_equity_curve_csv(f"{prefix}_equity.csv")

    def get_summary(self) -> dict:
        """
        Get backtest summary as dictionary.

        Returns:
            Summary dictionary.
        """
        if not self.results:
            raise ValueError("No results available. Run backtest first.")

        reporter = ResultsReporter(self.results)
        return reporter.get_summary_dict()


def run_backtest(
    pair: str = "BTC/USD",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    interval: int = 60,
    initial_capital: float = 10000.0,
    show_report: bool = True,
    export_results: bool = False,
    output_dir: str = "data/backtest_results"
) -> BacktestResults:
    """
    Convenience function to run a backtest.

    Args:
        pair: Trading pair.
        start_date: Start date (defaults to 6 months ago).
        end_date: End date (defaults to now).
        interval: Candle interval in minutes.
        initial_capital: Starting capital.
        show_report: Whether to print report to console.
        export_results: Whether to export results to files.
        output_dir: Directory for exported files.

    Returns:
        BacktestResults.
    """
    # Default date range: last 6 months
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    if start_date is None:
        from datetime import timedelta
        start_date = end_date - timedelta(days=180)

    # Create config
    config = BacktestConfig(
        pair=pair,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        initial_capital=initial_capital
    )

    # Run backtest
    runner = BacktestRunner(config)
    results = runner.run()

    # Print report
    if show_report:
        runner.print_report()

    # Export results
    if export_results:
        runner.export_results(output_dir)

    return results
