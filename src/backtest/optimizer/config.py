"""
Optimizer configuration dataclass.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional


@dataclass
class OptimizerConfig:
    """Configuration for parameter optimization."""

    # Optimization settings
    n_trials: int = 200
    timeout_seconds: int = 7200  # 2 hours max
    n_jobs: int = 1  # Parallel jobs (1 = sequential)

    # Objective function
    objective: str = "composite"  # "composite", "sharpe", "return", "profit_factor"

    # Trading pairs to optimize across (multi-pair helps avoid overfitting)
    pairs: List[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])

    # Date range for backtesting
    start_date: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) - timedelta(days=730)
    )
    end_date: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Backtest settings
    initial_capital: float = 10000.0
    interval: int = 60  # hourly candles
    fee_percent: float = 0.26  # Kraken taker fee

    # Data source
    data_source: str = "ccxt"  # "ccxt" (Binance) or "kraken"
    cache_dir: str = "data/cache/ccxt"

    # Study persistence
    study_name: str = "momentum_optimization"
    storage_path: str = "data/optimization/study.db"

    # Output
    export_config_path: Optional[str] = None  # e.g., "config/config.optimized.yaml"

    def __post_init__(self):
        """Validate configuration."""
        if self.n_trials < 1:
            raise ValueError("n_trials must be at least 1")

        if self.timeout_seconds < 60:
            raise ValueError("timeout_seconds must be at least 60")

        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")

        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        valid_objectives = ["composite", "sharpe", "return", "profit_factor"]
        if self.objective not in valid_objectives:
            raise ValueError(f"objective must be one of {valid_objectives}")

        valid_intervals = [1, 5, 15, 30, 60, 240, 1440]
        if self.interval not in valid_intervals:
            raise ValueError(f"interval must be one of {valid_intervals}")

        if not self.pairs:
            raise ValueError("At least one pair must be specified")

    @property
    def period_days(self) -> int:
        """Get the number of days in the optimization period."""
        delta = self.end_date - self.start_date
        return delta.days

    def to_dict(self) -> dict:
        """Serialize config to dictionary."""
        return {
            "n_trials": self.n_trials,
            "timeout_seconds": self.timeout_seconds,
            "n_jobs": self.n_jobs,
            "objective": self.objective,
            "pairs": self.pairs,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "interval": self.interval,
            "fee_percent": self.fee_percent,
            "data_source": self.data_source,
            "cache_dir": self.cache_dir,
            "study_name": self.study_name,
            "storage_path": self.storage_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OptimizerConfig":
        """Deserialize config from dictionary."""
        config_data = data.copy()

        # Parse datetime strings
        if "start_date" in config_data and isinstance(config_data["start_date"], str):
            config_data["start_date"] = datetime.fromisoformat(config_data["start_date"])
        if "end_date" in config_data and isinstance(config_data["end_date"], str):
            config_data["end_date"] = datetime.fromisoformat(config_data["end_date"])

        return cls(**config_data)
