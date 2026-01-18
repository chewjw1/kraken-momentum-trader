"""
Backtest configuration dataclass.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    # Trading pair to backtest
    pair: str = "BTC/USD"

    # Date range for backtest
    start_date: datetime = field(default_factory=lambda: datetime(2024, 1, 1))
    end_date: datetime = field(default_factory=datetime.now)

    # Candle interval in minutes (1, 5, 15, 30, 60, 240, 1440)
    interval: int = 60

    # Capital and fees
    initial_capital: float = 10000.0
    fee_percent: float = 0.26  # Kraken taker fee

    # Caching
    use_cache: bool = True
    cache_dir: str = "data/cache/ohlc"

    # Simulation settings
    lookback_candles: int = 50  # Number of candles for indicator calculation

    def __post_init__(self):
        """Validate configuration."""
        valid_intervals = [1, 5, 15, 30, 60, 240, 1440, 10080, 21600]
        if self.interval not in valid_intervals:
            raise ValueError(f"Invalid interval {self.interval}. Must be one of {valid_intervals}")

        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")

        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        if self.fee_percent < 0 or self.fee_percent > 10:
            raise ValueError("fee_percent must be between 0 and 10")

    @property
    def period_days(self) -> int:
        """Get the number of days in the backtest period."""
        delta = self.end_date - self.start_date
        return delta.days

    def to_dict(self) -> dict:
        """Serialize config to dictionary."""
        return {
            "pair": self.pair,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "interval": self.interval,
            "initial_capital": self.initial_capital,
            "fee_percent": self.fee_percent,
            "use_cache": self.use_cache,
            "cache_dir": self.cache_dir,
            "lookback_candles": self.lookback_candles,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BacktestConfig":
        """Deserialize config from dictionary."""
        return cls(
            pair=data.get("pair", "BTC/USD"),
            start_date=datetime.fromisoformat(data["start_date"]) if "start_date" in data else datetime(2024, 1, 1),
            end_date=datetime.fromisoformat(data["end_date"]) if "end_date" in data else datetime.now(),
            interval=data.get("interval", 60),
            initial_capital=data.get("initial_capital", 10000.0),
            fee_percent=data.get("fee_percent", 0.26),
            use_cache=data.get("use_cache", True),
            cache_dir=data.get("cache_dir", "data/cache/ohlc"),
            lookback_candles=data.get("lookback_candles", 50),
        )
