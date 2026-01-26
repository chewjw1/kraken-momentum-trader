"""
Per-pair configuration loader.

Loads pair-specific settings from optimized config files,
allowing different parameters (including Martingale) per trading pair.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any
import yaml

from .settings import (
    Settings, get_settings, StrategyConfig, RiskConfig,
    MartingaleConfig, MLConfig
)


@dataclass
class PairSettings:
    """Settings specific to a trading pair."""
    pair: str
    strategy: StrategyConfig
    risk: RiskConfig
    ml: MLConfig

    # Convenience properties
    @property
    def martingale_enabled(self) -> bool:
        return self.risk.martingale.enabled

    @property
    def take_profit_percent(self) -> float:
        return self.risk.take_profit_percent


class PairConfigManager:
    """
    Manages per-pair configurations.

    Loads pair-specific settings from config.optimized.yaml
    and provides access to per-pair parameters.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pair config manager.

        Args:
            config_path: Path to per-pair config file (e.g., config.optimized.yaml)
        """
        self._pair_configs: Dict[str, PairSettings] = {}
        self._base_settings = get_settings()
        self._config_path = config_path

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> bool:
        """
        Load per-pair configuration from a YAML file.

        Args:
            config_path: Path to configuration file.

        Returns:
            True if loaded successfully.
        """
        path = Path(config_path)
        if not path.exists():
            return False

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if not data or 'pair_parameters' not in data:
            return False

        pair_params = data['pair_parameters']

        for pair, params in pair_params.items():
            self._pair_configs[pair] = self._create_pair_settings(pair, params)

        return True

    def _create_pair_settings(self, pair: str, params: dict) -> PairSettings:
        """
        Create PairSettings from config dict.

        Args:
            pair: Trading pair name.
            params: Parameter dictionary from config.

        Returns:
            PairSettings instance.
        """
        base = self._base_settings

        # Strategy config
        strat_params = params.get('strategy', {})
        strategy = StrategyConfig(
            name=base.strategy.name,
            momentum_threshold=base.strategy.momentum_threshold,
            min_expected_gain=base.strategy.min_expected_gain,
            rsi_period=base.strategy.rsi_period,
            rsi_oversold=strat_params.get('rsi_oversold', base.strategy.rsi_oversold),
            rsi_overbought=strat_params.get('rsi_overbought', base.strategy.rsi_overbought),
            ema_short_period=base.strategy.ema_short_period,
            ema_long_period=base.strategy.ema_long_period,
            volume_sma_period=base.strategy.volume_sma_period,
            volume_threshold=strat_params.get('volume_threshold', base.strategy.volume_threshold),
        )

        # Martingale config
        mart_params = params.get('risk', {}).get('martingale', {})
        martingale = MartingaleConfig(
            enabled=mart_params.get('enabled', base.risk.martingale.enabled),
            max_entries=mart_params.get('max_entries', base.risk.martingale.max_entries),
            size_multiplier=mart_params.get('size_multiplier', base.risk.martingale.size_multiplier),
            add_on_drop_percent=mart_params.get('add_on_drop_percent', base.risk.martingale.add_on_drop_percent),
            require_rsi_oversold=base.risk.martingale.require_rsi_oversold,
            require_ema_trend=base.risk.martingale.require_ema_trend,
        )

        # Risk config
        risk_params = params.get('risk', {})
        risk = RiskConfig(
            max_position_size_usd=base.risk.max_position_size_usd,
            max_position_percent=base.risk.max_position_percent,
            max_total_exposure_percent=base.risk.max_total_exposure_percent,
            use_trailing_stop=base.risk.use_trailing_stop,
            trailing_stop_percent=risk_params.get('trailing_stop_percent', base.risk.trailing_stop_percent),
            trailing_stop_activation_percent=risk_params.get(
                'trailing_stop_activation_percent',
                base.risk.trailing_stop_activation_percent
            ),
            initial_stop_loss_percent=base.risk.initial_stop_loss_percent,
            take_profit_percent=risk_params.get('take_profit_percent', base.risk.take_profit_percent),
            martingale=martingale,
            circuit_breaker_consecutive_losses=base.risk.circuit_breaker_consecutive_losses,
            circuit_breaker_cooldown_hours=base.risk.circuit_breaker_cooldown_hours,
        )

        # ML config
        ml_params = params.get('ml', {})
        ml = MLConfig(
            enabled=base.ml.enabled,
            confidence_threshold=ml_params.get('confidence_threshold', base.ml.confidence_threshold),
            model_dir=base.ml.model_dir,
            retrain_after_trades=base.ml.retrain_after_trades,
            min_samples_for_retrain=base.ml.min_samples_for_retrain,
            performance_threshold=base.ml.performance_threshold,
            check_interval_seconds=base.ml.check_interval_seconds,
        )

        return PairSettings(
            pair=pair,
            strategy=strategy,
            risk=risk,
            ml=ml,
        )

    def get_pair_settings(self, pair: str) -> PairSettings:
        """
        Get settings for a specific pair.

        Args:
            pair: Trading pair name.

        Returns:
            PairSettings for the pair, or default settings if not configured.
        """
        if pair in self._pair_configs:
            return self._pair_configs[pair]

        # Return default settings if pair not configured
        return PairSettings(
            pair=pair,
            strategy=self._base_settings.strategy,
            risk=self._base_settings.risk,
            ml=self._base_settings.ml,
        )

    def has_pair_config(self, pair: str) -> bool:
        """Check if pair has specific configuration."""
        return pair in self._pair_configs

    def get_all_pairs(self) -> list:
        """Get list of all configured pairs."""
        return list(self._pair_configs.keys())

    def get_martingale_status(self) -> Dict[str, bool]:
        """Get Martingale enabled status for all pairs."""
        return {
            pair: settings.martingale_enabled
            for pair, settings in self._pair_configs.items()
        }

    def summary(self) -> str:
        """Get a summary of per-pair configurations."""
        lines = ["Per-Pair Configuration Summary:", "=" * 50]

        for pair, settings in self._pair_configs.items():
            mart = settings.risk.martingale
            lines.append(f"\n{pair}:")
            lines.append(f"  RSI: {settings.strategy.rsi_oversold}/{settings.strategy.rsi_overbought}")
            lines.append(f"  Take Profit: {settings.risk.take_profit_percent}%")
            lines.append(f"  Trailing Stop: {settings.risk.trailing_stop_percent}% (activation: {settings.risk.trailing_stop_activation_percent}%)")
            lines.append(f"  Martingale: {'ENABLED' if mart.enabled else 'DISABLED'}")
            if mart.enabled:
                lines.append(f"    - Max Entries: {mart.max_entries}")
                lines.append(f"    - Size Multiplier: {mart.size_multiplier}x")
                lines.append(f"    - Add On Drop: {mart.add_on_drop_percent}%")

        return "\n".join(lines)


# Global pair config manager instance
_pair_config_manager: Optional[PairConfigManager] = None


def get_pair_config_manager() -> PairConfigManager:
    """Get the global pair config manager instance."""
    global _pair_config_manager
    if _pair_config_manager is None:
        _pair_config_manager = PairConfigManager()
    return _pair_config_manager


def load_pair_configs(config_path: str) -> PairConfigManager:
    """
    Load per-pair configurations from file.

    Args:
        config_path: Path to per-pair config file.

    Returns:
        PairConfigManager instance.
    """
    global _pair_config_manager
    _pair_config_manager = PairConfigManager(config_path)
    return _pair_config_manager


def get_pair_settings(pair: str) -> PairSettings:
    """
    Get settings for a specific pair.

    Args:
        pair: Trading pair name.

    Returns:
        PairSettings for the pair.
    """
    return get_pair_config_manager().get_pair_settings(pair)
