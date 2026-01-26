"""
Configuration loading and management.
Loads settings from YAML files and environment variables.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import yaml

from .platform import detect_platform, Platform


@dataclass
class TradingConfig:
    """Trading-specific configuration."""
    pairs: list[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])
    check_interval_seconds: int = 60
    paper_trading: bool = True  # CRITICAL: Default to paper trading
    paper_trading_capital: float = 10000.0  # Starting capital for paper trading


@dataclass
class StrategyConfig:
    """Strategy-specific configuration."""
    name: str = "momentum"
    momentum_threshold: float = 0.02
    min_expected_gain: float = 0.015

    # RSI
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70

    # EMA
    ema_short_period: int = 12
    ema_long_period: int = 26

    # Volume
    volume_sma_period: int = 20
    volume_threshold: float = 1.5


@dataclass
class MartingaleConfig:
    """Martingale strategy configuration."""
    enabled: bool = True
    max_entries: int = 4  # Maximum total entries (initial + add-ons)
    size_multiplier: float = 1.25  # Each add-on is 1.25x previous
    add_on_drop_percent: float = 5.0  # Add when price drops 5% from avg entry
    require_rsi_oversold: bool = True  # Check RSI before adding
    require_ema_trend: bool = False  # Check EMA trend before adding


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size_usd: float = 5000.0
    max_position_percent: float = 40.0
    max_total_exposure_percent: float = 200.0  # Increased for Martingale pyramiding

    # Trailing stop configuration
    use_trailing_stop: bool = True
    trailing_stop_percent: float = 5.0  # Exit when price drops this % from peak
    trailing_stop_activation_percent: float = 5.0  # Minimum profit before trailing activates
    initial_stop_loss_percent: float = 0.0  # Initial stop loss (0 = disabled)

    # Take profit configuration (0 = disabled)
    take_profit_percent: float = 0.0  # Exit when profit reaches this %

    # Martingale configuration
    martingale: MartingaleConfig = None

    circuit_breaker_consecutive_losses: int = 5  # Increased for Martingale
    circuit_breaker_cooldown_hours: int = 2  # Reduced for Martingale

    def __post_init__(self):
        if self.martingale is None:
            self.martingale = MartingaleConfig()


@dataclass
class ExchangeConfig:
    """Exchange-specific configuration."""
    name: str = "kraken"
    api_version: str = "0"
    rate_limit_calls_per_minute: int = 15
    retry_attempts: int = 3
    retry_delay_seconds: float = 2.0


@dataclass
class PersistenceConfig:
    """Persistence configuration."""
    type: str = "sqlite"  # "sqlite" or "file"
    db_path: str = "data/trading.db"
    # Legacy JSON paths (for migration)
    state_file: str = "data/state.json"
    trades_file: str = "data/trades.json"
    metrics_file: str = "data/metrics.json"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"
    include_timestamps: bool = True


@dataclass
class DiscordConfig:
    """Discord notification configuration."""
    enabled: bool = False
    webhook_url: str = ""


@dataclass
class DailySummaryConfig:
    """Daily summary scheduler configuration."""
    enabled: bool = True
    hour: int = 17  # 5 PM
    minute: int = 0
    timezone: str = "America/New_York"


@dataclass
class NotificationConfig:
    """Notification configuration."""
    discord: DiscordConfig = field(default_factory=DiscordConfig)
    daily_summary: DailySummaryConfig = field(default_factory=DailySummaryConfig)


@dataclass
class MLConfig:
    """Machine Learning configuration."""
    enabled: bool = True
    confidence_threshold: float = 0.6  # Min confidence to approve signals
    model_dir: str = "data/ml/models"

    # Auto-retraining settings
    retrain_after_trades: int = 50  # Retrain after N new trades
    min_samples_for_retrain: int = 100  # Min samples required
    performance_threshold: float = 0.4  # Retrain if win rate drops below
    check_interval_seconds: int = 3600  # Check triggers every hour


@dataclass
class Settings:
    """Main application settings container."""
    trading: TradingConfig = field(default_factory=TradingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    ml: MLConfig = field(default_factory=MLConfig)


def _find_config_file() -> Optional[Path]:
    """
    Find the configuration file.

    Returns:
        Path to config file or None if not found.
    """
    # Check environment variable first
    env_config = os.environ.get("CONFIG_PATH")
    if env_config:
        path = Path(env_config)
        if path.exists():
            return path

    # Check common locations
    search_paths = [
        Path("config/config.yaml"),
        Path("config.yaml"),
        Path(__file__).parent.parent.parent / "config" / "config.yaml",
    ]

    # Add production config for AWS
    if detect_platform() == Platform.AWS:
        search_paths.insert(0, Path("config/config.prod.yaml"))

    for path in search_paths:
        if path.exists():
            return path

    return None


def _dict_to_config(data: dict, config_class, existing=None):
    """
    Convert dictionary to dataclass, preserving defaults for missing keys.

    Args:
        data: Dictionary with configuration data.
        config_class: The dataclass type to create.
        existing: Existing instance to update (optional).

    Returns:
        Instance of config_class with data applied.
    """
    if existing is None:
        existing = config_class()

    if data is None:
        return existing

    for key, value in data.items():
        if hasattr(existing, key):
            setattr(existing, key, value)

    return existing


def load_settings(config_path: Optional[str] = None) -> Settings:
    """
    Load settings from configuration file and environment.

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        Settings instance with loaded configuration.
    """
    settings = Settings()

    # Find config file
    if config_path:
        path = Path(config_path)
    else:
        path = _find_config_file()

    # Load from file if found
    if path and path.exists():
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if data:
            settings.trading = _dict_to_config(
                data.get("trading"), TradingConfig, settings.trading
            )
            settings.strategy = _dict_to_config(
                data.get("strategy"), StrategyConfig, settings.strategy
            )
            # Handle risk config with nested martingale
            risk_data = data.get("risk", {})
            # Extract martingale data before processing risk config
            martingale_data = risk_data.pop("martingale", None) if isinstance(risk_data, dict) else None
            settings.risk = _dict_to_config(
                risk_data, RiskConfig, settings.risk
            )
            # Handle nested martingale config separately
            if martingale_data and isinstance(martingale_data, dict):
                # Create fresh MartingaleConfig and apply dict values
                martingale_config = MartingaleConfig()
                settings.risk.martingale = _dict_to_config(
                    martingale_data, MartingaleConfig, martingale_config
                )
            settings.exchange = _dict_to_config(
                data.get("exchange"), ExchangeConfig, settings.exchange
            )
            settings.persistence = _dict_to_config(
                data.get("persistence"), PersistenceConfig, settings.persistence
            )
            settings.logging = _dict_to_config(
                data.get("logging"), LoggingConfig, settings.logging
            )

            # Handle notifications config with nested discord and daily_summary
            notif_data = data.get("notifications", {})
            if notif_data and isinstance(notif_data, dict):
                discord_data = notif_data.get("discord", {})
                daily_summary_data = notif_data.get("daily_summary", {})

                settings.notifications.discord = _dict_to_config(
                    discord_data, DiscordConfig, settings.notifications.discord
                )
                settings.notifications.daily_summary = _dict_to_config(
                    daily_summary_data, DailySummaryConfig, settings.notifications.daily_summary
                )

            # Handle ML config
            settings.ml = _dict_to_config(
                data.get("ml"), MLConfig, settings.ml
            )

    # Override from environment variables
    _apply_env_overrides(settings)

    # Safety check: ensure paper trading is enabled unless explicitly disabled
    if os.environ.get("ENABLE_LIVE_TRADING", "").lower() != "true":
        settings.trading.paper_trading = True

    return settings


def _apply_env_overrides(settings: Settings) -> None:
    """
    Apply environment variable overrides to settings.

    Args:
        settings: Settings instance to modify.
    """
    # Trading overrides
    if pairs := os.environ.get("TRADING_PAIRS"):
        settings.trading.pairs = [p.strip() for p in pairs.split(",")]

    if interval := os.environ.get("CHECK_INTERVAL_SECONDS"):
        settings.trading.check_interval_seconds = int(interval)

    # Risk overrides
    if max_pos := os.environ.get("MAX_POSITION_SIZE_USD"):
        settings.risk.max_position_size_usd = float(max_pos)

    if stop_loss := os.environ.get("STOP_LOSS_PERCENT"):
        settings.risk.stop_loss_percent = float(stop_loss)

    if take_profit := os.environ.get("TAKE_PROFIT_PERCENT"):
        settings.risk.take_profit_percent = float(take_profit)

    # Logging overrides
    if log_level := os.environ.get("LOG_LEVEL"):
        settings.logging.level = log_level.upper()

    # Notification overrides
    if discord_webhook := os.environ.get("DISCORD_WEBHOOK_URL"):
        settings.notifications.discord.webhook_url = discord_webhook
        settings.notifications.discord.enabled = True

    if discord_enabled := os.environ.get("DISCORD_NOTIFICATIONS_ENABLED"):
        settings.notifications.discord.enabled = discord_enabled.lower() == "true"


# Global settings instance (lazy loaded)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance.

    Returns:
        Settings instance.
    """
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reload_settings(config_path: Optional[str] = None) -> Settings:
    """
    Reload settings from configuration.

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        New Settings instance.
    """
    global _settings
    _settings = load_settings(config_path)
    return _settings
