"""
Parameter space definitions for Optuna optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import optuna

try:
    import optuna as _optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    _optuna = None
    OPTUNA_AVAILABLE = False


@dataclass
class ParameterRange:
    """Definition of a parameter range."""
    name: str
    param_type: str  # "int", "float", "bool", "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[list] = None
    log: bool = False  # Use log scale for sampling
    conditional_on: Optional[str] = None  # Only sample if this param is True


# Default parameter ranges for momentum strategy optimization
DEFAULT_PARAMETER_RANGES = [
    # RSI parameters - wider range for sideways markets
    ParameterRange(
        name="rsi_oversold",
        param_type="int",
        low=35,
        high=55,  # Higher = more signals in sideways markets
        step=5
    ),
    ParameterRange(
        name="rsi_overbought",
        param_type="int",
        low=50,  # Lower = quicker exits in sideways markets
        high=75,
        step=5
    ),

    # Volume parameters - lower threshold for more trades
    ParameterRange(
        name="volume_threshold",
        param_type="float",
        low=0.3,
        high=1.5,  # Lowered max - don't require huge volume spikes
        step=0.2
    ),

    # Take profit - NEW for sideways markets
    ParameterRange(
        name="take_profit_percent",
        param_type="float",
        low=3.0,  # Take profits at 3-12%
        high=12.0,
        step=1.0
    ),

    # ML confidence threshold
    ParameterRange(
        name="ml_confidence_threshold",
        param_type="float",
        low=0.3,
        high=0.9,
        step=0.05
    ),

    # Trailing stop parameters
    ParameterRange(
        name="trailing_stop_percent",
        param_type="float",
        low=2.0,
        high=10.0,
        step=0.5
    ),
    ParameterRange(
        name="trailing_stop_activation_percent",
        param_type="float",
        low=1.0,
        high=8.0,
        step=0.5
    ),

    # Martingale parameters - tuned for sideways/ranging markets
    ParameterRange(
        name="martingale_enabled",
        param_type="bool"
    ),
    ParameterRange(
        name="martingale_max_entries",
        param_type="int",
        low=2,
        high=5,  # Reduced max - too many entries = too much exposure
        step=1,
        conditional_on="martingale_enabled"
    ),
    ParameterRange(
        name="martingale_size_multiplier",
        param_type="float",
        low=1.1,
        high=1.5,  # Reduced - less aggressive pyramiding
        step=0.1,
        conditional_on="martingale_enabled"
    ),
    ParameterRange(
        name="martingale_add_on_drop_percent",
        param_type="float",
        low=2.0,  # Lower - catch smaller dips in sideways markets
        high=6.0,  # Reduced max
        step=0.5,
        conditional_on="martingale_enabled"
    ),
]


# Martingale-focused parameter ranges (forces Martingale enabled)
MARTINGALE_FOCUSED_RANGES = [
    # RSI parameters - same as default
    ParameterRange(
        name="rsi_oversold",
        param_type="int",
        low=35,
        high=55,
        step=5
    ),
    ParameterRange(
        name="rsi_overbought",
        param_type="int",
        low=50,
        high=75,
        step=5
    ),
    ParameterRange(
        name="volume_threshold",
        param_type="float",
        low=0.3,
        high=1.5,
        step=0.2
    ),
    ParameterRange(
        name="take_profit_percent",
        param_type="float",
        low=3.0,
        high=12.0,
        step=1.0
    ),
    ParameterRange(
        name="ml_confidence_threshold",
        param_type="float",
        low=0.3,
        high=0.9,
        step=0.05
    ),
    ParameterRange(
        name="trailing_stop_percent",
        param_type="float",
        low=2.0,
        high=10.0,
        step=0.5
    ),
    ParameterRange(
        name="trailing_stop_activation_percent",
        param_type="float",
        low=1.0,
        high=8.0,
        step=0.5
    ),
    # Martingale - ALWAYS ENABLED for this space
    ParameterRange(
        name="martingale_max_entries",
        param_type="int",
        low=2,
        high=5,
        step=1
    ),
    ParameterRange(
        name="martingale_size_multiplier",
        param_type="float",
        low=1.1,
        high=1.5,
        step=0.05  # Finer granularity
    ),
    ParameterRange(
        name="martingale_add_on_drop_percent",
        param_type="float",
        low=2.0,  # Small dips for sideways markets
        high=5.0,
        step=0.5
    ),
]


class ParameterSpace:
    """
    Manages parameter space for Optuna optimization.

    Handles:
    - Parameter range definitions
    - Sampling parameters from Optuna trials
    - Conditional parameters (e.g., Martingale sub-params only if enabled)
    - Converting sampled params to strategy config format
    """

    def __init__(
        self,
        parameter_ranges: Optional[list] = None,
        force_martingale: bool = False
    ):
        """
        Initialize parameter space.

        Args:
            parameter_ranges: List of ParameterRange objects. Uses defaults if None.
            force_martingale: If True, always enable Martingale (for focused testing).
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "optuna is required for ParameterSpace. "
                "Install it with: pip install optuna"
            )

        self.ranges = parameter_ranges or DEFAULT_PARAMETER_RANGES
        self._ranges_by_name = {r.name: r for r in self.ranges}
        self.force_martingale = force_martingale

    def sample(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """
        Sample parameters from the trial.

        Args:
            trial: Optuna trial object.

        Returns:
            Dictionary of sampled parameter values.
        """
        params = {}

        # First pass: sample non-conditional parameters
        for param_range in self.ranges:
            if param_range.conditional_on is not None:
                continue
            params[param_range.name] = self._sample_param(trial, param_range)

        # If force_martingale, ensure it's enabled
        if self.force_martingale:
            params["martingale_enabled"] = True

        # Second pass: sample conditional parameters
        for param_range in self.ranges:
            if param_range.conditional_on is None:
                continue

            # Check if condition is met (or if force_martingale for martingale params)
            condition_value = params.get(param_range.conditional_on, False)
            if condition_value or (self.force_martingale and "martingale" in param_range.name):
                params[param_range.name] = self._sample_param(trial, param_range)
            else:
                # Use default value when condition not met
                params[param_range.name] = self._get_default_value(param_range)

        return params

    def _sample_param(
        self,
        trial: "optuna.Trial",
        param_range: ParameterRange
    ) -> Any:
        """Sample a single parameter from the trial."""
        name = param_range.name

        if param_range.param_type == "int":
            return trial.suggest_int(
                name,
                int(param_range.low),
                int(param_range.high),
                step=int(param_range.step) if param_range.step else 1,
                log=param_range.log
            )

        elif param_range.param_type == "float":
            return trial.suggest_float(
                name,
                param_range.low,
                param_range.high,
                step=param_range.step,
                log=param_range.log
            )

        elif param_range.param_type == "bool":
            return trial.suggest_categorical(name, [True, False])

        elif param_range.param_type == "categorical":
            return trial.suggest_categorical(name, param_range.choices)

        else:
            raise ValueError(f"Unknown param_type: {param_range.param_type}")

    def _get_default_value(self, param_range: ParameterRange) -> Any:
        """Get default value for a parameter when condition is not met."""
        if param_range.param_type == "int":
            return int(param_range.low)
        elif param_range.param_type == "float":
            return param_range.low
        elif param_range.param_type == "bool":
            return False
        elif param_range.param_type == "categorical":
            return param_range.choices[0] if param_range.choices else None
        return None

    def to_strategy_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert sampled parameters to strategy configuration format.

        Args:
            params: Dictionary of sampled parameter values.

        Returns:
            Nested dictionary matching the config.yaml structure.
        """
        return {
            "strategy": {
                "rsi_oversold": params.get("rsi_oversold", 30),
                "rsi_overbought": params.get("rsi_overbought", 70),
                "volume_threshold": params.get("volume_threshold", 1.5),
            },
            "ml": {
                "confidence_threshold": params.get("ml_confidence_threshold", 0.6),
            },
            "risk": {
                "trailing_stop_percent": params.get("trailing_stop_percent", 5.0),
                "trailing_stop_activation_percent": params.get(
                    "trailing_stop_activation_percent", 5.0
                ),
                "take_profit_percent": params.get("take_profit_percent", 0.0),
                "martingale": {
                    "enabled": params.get("martingale_enabled", True),
                    "max_entries": params.get("martingale_max_entries", 4),
                    "size_multiplier": params.get("martingale_size_multiplier", 1.25),
                    "add_on_drop_percent": params.get("martingale_add_on_drop_percent", 5.0),
                },
            },
        }

    def get_param_names(self) -> list:
        """Get list of all parameter names."""
        return [r.name for r in self.ranges]

    def get_param_info(self) -> Dict[str, dict]:
        """Get information about all parameters."""
        info = {}
        for r in self.ranges:
            info[r.name] = {
                "type": r.param_type,
                "low": r.low,
                "high": r.high,
                "step": r.step,
                "choices": r.choices,
                "conditional_on": r.conditional_on,
            }
        return info
