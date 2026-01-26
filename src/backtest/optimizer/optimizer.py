"""
Main parameter optimizer using Optuna for Bayesian optimization.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
import yaml

if TYPE_CHECKING:
    import optuna

from ...observability.logger import get_logger
from ... import config as config_module
from ...config.settings import (
    Settings, get_settings, StrategyConfig, RiskConfig,
    MartingaleConfig, MLConfig
)
from ...strategy.momentum_strategy import MomentumStrategy
from ..config import BacktestConfig
from ..engine import BacktestEngine
from ..ccxt_data_provider import CCXTDataProvider, YFinanceDataProvider
from ..data import HistoricalDataManager

from .config import OptimizerConfig
from .parameter_space import ParameterSpace
from .objective import ObjectiveFunction, BacktestMetrics

try:
    import optuna
    from optuna.trial import TrialState
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    TrialState = None
    OPTUNA_AVAILABLE = False

logger = get_logger(__name__)


@contextmanager
def patch_settings(new_settings: Settings):
    """
    Context manager to temporarily patch global settings.

    This is necessary because MomentumStrategy reads from get_settings()
    for risk parameters (trailing stops, martingale, etc.).

    Args:
        new_settings: Settings object to use temporarily.
    """
    import src.config.settings as settings_module

    old_settings = settings_module._settings
    try:
        settings_module._settings = new_settings
        yield
    finally:
        settings_module._settings = old_settings


class ParameterOptimizer:
    """
    Main optimizer for trading strategy parameters.

    Uses Optuna for Bayesian optimization to find optimal parameters
    by running backtests across multiple pairs and time periods.
    """

    def __init__(self, config: OptimizerConfig):
        """
        Initialize the optimizer.

        Args:
            config: Optimizer configuration.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "optuna is required for ParameterOptimizer. "
                "Install it with: pip install optuna"
            )

        self.config = config
        self.parameter_space = ParameterSpace()
        self.objective_fn = ObjectiveFunction(objective_type=config.objective)

        # Data storage
        self._historical_data: Dict[str, List] = {}  # pair -> candles
        self._ccxt_provider: Optional[CCXTDataProvider] = None
        self._kraken_provider: Optional[HistoricalDataManager] = None
        self._yfinance_provider: Optional[YFinanceDataProvider] = None

        # Study
        self.study: Optional[optuna.Study] = None
        self._best_params: Optional[Dict[str, Any]] = None
        self._best_score: Optional[float] = None

        # Ensure directories exist
        Path(config.storage_path).parent.mkdir(parents=True, exist_ok=True)
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    @property
    def ccxt_provider(self) -> CCXTDataProvider:
        """Lazy-load the CCXT data provider."""
        if self._ccxt_provider is None:
            self._ccxt_provider = CCXTDataProvider(
                cache_dir=self.config.cache_dir,
                use_cache=True
            )
        return self._ccxt_provider

    @property
    def kraken_provider(self) -> HistoricalDataManager:
        """Lazy-load the Kraken data provider."""
        if self._kraken_provider is None:
            self._kraken_provider = HistoricalDataManager(
                cache_dir="data/cache/ohlc",
                use_cache=True
            )
        return self._kraken_provider

    @property
    def yfinance_provider(self) -> YFinanceDataProvider:
        """Lazy-load the yfinance data provider."""
        if self._yfinance_provider is None:
            self._yfinance_provider = YFinanceDataProvider(
                cache_dir="data/cache/yfinance",
                use_cache=True
            )
        return self._yfinance_provider

    def prefetch_data(self, show_progress: bool = True) -> None:
        """
        Pre-fetch historical data for all pairs.

        This is done once before optimization to avoid re-fetching during trials.

        Args:
            show_progress: Whether to show progress during fetching.
        """
        logger.info(
            "Pre-fetching historical data for optimization",
            pairs=self.config.pairs,
            start=self.config.start_date.isoformat(),
            end=self.config.end_date.isoformat(),
            data_source=self.config.data_source
        )

        for pair in self.config.pairs:
            print(f"Fetching data for {pair} (source: {self.config.data_source})...")

            def progress_callback(current: int, total: int) -> None:
                if show_progress:
                    pct = (current / total * 100) if total > 0 else 0
                    print(f"\r  Progress: {current}/{total} requests ({pct:.0f}%)...", end="", flush=True)

            # Use appropriate data source
            if self.config.data_source == "kraken":
                candles = self.kraken_provider.get_ohlc_range(
                    pair=pair,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    interval=self.config.interval,
                    progress_callback=progress_callback if show_progress else None
                )
            elif self.config.data_source == "yfinance":
                # yfinance only supports daily data - override interval
                if self.config.interval != 1440:
                    print(f"  Note: yfinance only supports daily candles, using 1440m interval")
                candles = self.yfinance_provider.get_ohlc_range(
                    pair=pair,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    interval=1440,  # Force daily
                    progress_callback=progress_callback if show_progress else None
                )
            else:  # ccxt (Binance)
                candles = self.ccxt_provider.get_ohlc_range(
                    pair=pair,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    interval=self.config.interval,
                    progress_callback=progress_callback if show_progress else None
                )

            if show_progress:
                print()  # New line after progress

            if not candles:
                raise ValueError(f"Failed to fetch data for {pair}")

            self._historical_data[pair] = candles
            print(f"  Loaded {len(candles)} candles for {pair}")
            logger.info(f"Loaded {len(candles)} candles for {pair}")

    def _apply_params_to_settings(self, params: Dict[str, Any]) -> Settings:
        """
        Create a modified Settings object with the trial parameters.

        Args:
            params: Dictionary of sampled parameter values.

        Returns:
            Modified Settings object.
        """
        # Start with current settings
        settings = get_settings()

        # Apply strategy parameters
        strategy_config = StrategyConfig(
            name=settings.strategy.name,
            momentum_threshold=settings.strategy.momentum_threshold,
            min_expected_gain=settings.strategy.min_expected_gain,
            rsi_period=settings.strategy.rsi_period,
            rsi_oversold=params.get("rsi_oversold", settings.strategy.rsi_oversold),
            rsi_overbought=params.get("rsi_overbought", settings.strategy.rsi_overbought),
            ema_short_period=settings.strategy.ema_short_period,
            ema_long_period=settings.strategy.ema_long_period,
            volume_sma_period=settings.strategy.volume_sma_period,
            volume_threshold=params.get("volume_threshold", settings.strategy.volume_threshold),
        )

        # Apply martingale parameters
        martingale_config = MartingaleConfig(
            enabled=params.get("martingale_enabled", settings.risk.martingale.enabled),
            max_entries=params.get("martingale_max_entries", settings.risk.martingale.max_entries),
            size_multiplier=params.get("martingale_size_multiplier", settings.risk.martingale.size_multiplier),
            add_on_drop_percent=params.get("martingale_add_on_drop_percent", settings.risk.martingale.add_on_drop_percent),
            require_rsi_oversold=settings.risk.martingale.require_rsi_oversold,
            require_ema_trend=settings.risk.martingale.require_ema_trend,
        )

        # Apply risk parameters
        risk_config = RiskConfig(
            max_position_size_usd=settings.risk.max_position_size_usd,
            max_position_percent=settings.risk.max_position_percent,
            max_total_exposure_percent=settings.risk.max_total_exposure_percent,
            use_trailing_stop=settings.risk.use_trailing_stop,
            trailing_stop_percent=params.get("trailing_stop_percent", settings.risk.trailing_stop_percent),
            trailing_stop_activation_percent=params.get(
                "trailing_stop_activation_percent",
                settings.risk.trailing_stop_activation_percent
            ),
            initial_stop_loss_percent=settings.risk.initial_stop_loss_percent,
            take_profit_percent=params.get("take_profit_percent", settings.risk.take_profit_percent),
            martingale=martingale_config,
            circuit_breaker_consecutive_losses=settings.risk.circuit_breaker_consecutive_losses,
            circuit_breaker_cooldown_hours=settings.risk.circuit_breaker_cooldown_hours,
        )

        # Apply ML parameters
        ml_config = MLConfig(
            enabled=settings.ml.enabled,
            confidence_threshold=params.get("ml_confidence_threshold", settings.ml.confidence_threshold),
            model_dir=settings.ml.model_dir,
            retrain_after_trades=settings.ml.retrain_after_trades,
            min_samples_for_retrain=settings.ml.min_samples_for_retrain,
            performance_threshold=settings.ml.performance_threshold,
            check_interval_seconds=settings.ml.check_interval_seconds,
        )

        # Create new settings with modified configs
        new_settings = Settings(
            trading=settings.trading,
            strategy=strategy_config,
            risk=risk_config,
            exchange=settings.exchange,
            persistence=settings.persistence,
            logging=settings.logging,
            notifications=settings.notifications,
            ml=ml_config,
        )

        return new_settings

    def _run_backtest_for_pair(
        self,
        pair: str,
        params: Dict[str, Any]
    ) -> Optional[BacktestMetrics]:
        """
        Run a single backtest for a pair with given parameters.

        Args:
            pair: Trading pair.
            params: Parameter values for this trial.

        Returns:
            BacktestMetrics or None if backtest failed.
        """
        if pair not in self._historical_data:
            logger.error(f"No historical data for {pair}")
            return None

        candles = self._historical_data[pair]

        # Create backtest config
        bt_config = BacktestConfig(
            pair=pair,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            interval=self.config.interval,
            initial_capital=self.config.initial_capital,
            fee_percent=self.config.fee_percent,
            use_cache=False,  # We already have data in memory
        )

        try:
            # Create modified settings and patch globally
            modified_settings = self._apply_params_to_settings(params)

            # Use context manager to patch settings during backtest
            with patch_settings(modified_settings):
                # Create strategy - it will read from patched settings
                strategy = MomentumStrategy(config=modified_settings.strategy)

                # Create and run engine - it will also read from patched settings
                engine = BacktestEngine(bt_config, strategy)

                results = engine.run(candles)

            # Convert to BacktestMetrics
            return ObjectiveFunction.from_backtest_results(
                results, initial_capital=self.config.initial_capital
            )

        except Exception as e:
            logger.error(f"Backtest failed for {pair}: {e}")
            return None

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object.

        Returns:
            Score to maximize.
        """
        # Sample parameters
        params = self.parameter_space.sample(trial)

        # Run backtests for all pairs
        pair_metrics: Dict[str, BacktestMetrics] = {}

        for pair in self.config.pairs:
            metrics = self._run_backtest_for_pair(pair, params)
            if metrics:
                pair_metrics[pair] = metrics

                # Report intermediate results
                trial.set_user_attr(f"{pair}_return", metrics.total_return)
                trial.set_user_attr(f"{pair}_sharpe", metrics.sharpe_ratio)
                trial.set_user_attr(f"{pair}_trades", metrics.total_trades)

        if not pair_metrics:
            return float('-inf')

        # Calculate aggregate score
        score = self.objective_fn.aggregate_multi_pair_scores(pair_metrics)

        # Store metrics for this trial
        trial.set_user_attr("pair_count", len(pair_metrics))
        trial.set_user_attr("total_trades", sum(m.total_trades for m in pair_metrics.values()))
        trial.set_user_attr("avg_return", sum(m.total_return for m in pair_metrics.values()) / len(pair_metrics))

        return score

    def optimize(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        show_progress: bool = True
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run the optimization.

        Args:
            n_trials: Number of trials (overrides config if set).
            timeout: Timeout in seconds (overrides config if set).
            show_progress: Whether to show progress.

        Returns:
            Tuple of (best_params, best_score).
        """
        n_trials = n_trials or self.config.n_trials
        timeout = timeout or self.config.timeout_seconds

        # Pre-fetch data if not already done
        if not self._historical_data:
            self.prefetch_data(show_progress)

        # Create or load study
        storage_url = f"sqlite:///{self.config.storage_path}"

        self.study = optuna.create_study(
            study_name=self.config.study_name,
            storage=storage_url,
            direction="maximize",
            load_if_exists=True,  # Resume if exists
        )

        print(f"\nStarting optimization with {n_trials} trials...")
        print(f"Study storage: {self.config.storage_path}")
        print(f"Existing trials: {len(self.study.trials)}")

        # Run optimization
        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
            n_jobs=self.config.n_jobs,
        )

        # Get best results
        self._best_params = self.study.best_params
        self._best_score = self.study.best_value

        logger.info(
            "Optimization completed",
            best_score=self._best_score,
            best_params=self._best_params,
            total_trials=len(self.study.trials)
        )

        return self._best_params, self._best_score

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get the best parameters found."""
        return self._best_params

    def get_best_score(self) -> Optional[float]:
        """Get the best score achieved."""
        return self._best_score

    def export_config(self, output_path: Optional[str] = None) -> str:
        """
        Export optimized parameters to a YAML config file.

        Args:
            output_path: Path for output file. Uses config default if not specified.

        Returns:
            Path to the exported file.
        """
        if self._best_params is None:
            raise ValueError("No optimization results available. Run optimize() first.")

        output_path = output_path or self.config.export_config_path
        if not output_path:
            output_path = "config/config.optimized.yaml"

        # Convert params to config structure
        config_dict = self.parameter_space.to_strategy_config(self._best_params)

        # Add metadata
        config_dict["_optimization"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "study_name": self.config.study_name,
            "n_trials": len(self.study.trials) if self.study else 0,
            "best_score": self._best_score,
            "objective": self.config.objective,
            "pairs": self.config.pairs,
            "period_days": self.config.period_days,
        }

        # Write to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        print(f"Exported optimized config to {output_path}")
        logger.info(f"Exported optimized config to {output_path}")

        return output_path

    def print_results(self) -> None:
        """Print optimization results summary."""
        if self.study is None or self._best_params is None:
            print("No results available. Run optimize() first.")
            return

        print("\n" + "=" * 60)
        print("OPTIMIZATION RESULTS")
        print("=" * 60)

        print(f"\nBest Score: {self._best_score:.4f}")
        print(f"Total Trials: {len(self.study.trials)}")
        print(f"Completed Trials: {len([t for t in self.study.trials if t.state == TrialState.COMPLETE])}")

        print("\n--- Best Parameters ---")
        for name, value in sorted(self._best_params.items()):
            if isinstance(value, float):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")

        # Best trial details
        best_trial = self.study.best_trial
        print("\n--- Best Trial Details ---")
        print(f"  Trial Number: {best_trial.number}")
        for key, value in best_trial.user_attrs.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print("\n" + "=" * 60)

    def get_study_dataframe(self):
        """Get study results as a pandas DataFrame."""
        if self.study is None:
            return None

        try:
            return self.study.trials_dataframe()
        except Exception:
            return None

    def _objective_single_pair(self, trial: optuna.Trial, pair: str) -> float:
        """
        Objective function for single-pair optimization.

        Args:
            trial: Optuna trial object.
            pair: The pair to optimize for.

        Returns:
            Score to maximize.
        """
        params = self.parameter_space.sample(trial)
        metrics = self._run_backtest_for_pair(pair, params)

        if metrics is None:
            return float('-inf')

        # Report results
        trial.set_user_attr("total_return", metrics.total_return)
        trial.set_user_attr("sharpe_ratio", metrics.sharpe_ratio)
        trial.set_user_attr("total_trades", metrics.total_trades)
        trial.set_user_attr("max_drawdown", metrics.max_drawdown_percent)
        trial.set_user_attr("win_rate", metrics.win_rate)

        return self.objective_fn.calculate_score(metrics)

    def optimize_per_pair(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, Tuple[Dict[str, Any], float]]:
        """
        Run separate optimization for each pair.

        Args:
            n_trials: Number of trials per pair.
            timeout: Timeout in seconds per pair.
            show_progress: Whether to show progress.

        Returns:
            Dictionary mapping pair -> (best_params, best_score).
        """
        n_trials = n_trials or self.config.n_trials
        timeout = timeout or self.config.timeout_seconds

        # Pre-fetch data if not already done
        if not self._historical_data:
            self.prefetch_data(show_progress)

        results: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._per_pair_results = {}  # Store for export

        for pair in self.config.pairs:
            pair_clean = pair.replace("/", "_")
            study_name = f"{self.config.study_name}_{pair_clean}"
            storage_path = self.config.storage_path.replace(".db", f"_{pair_clean}.db")
            storage_url = f"sqlite:///{storage_path}"

            print(f"\n{'=' * 60}")
            print(f"OPTIMIZING: {pair}")
            print(f"{'=' * 60}")

            # Create study for this pair
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction="maximize",
                load_if_exists=True,
            )

            print(f"Study: {study_name}")
            print(f"Storage: {storage_path}")
            print(f"Existing trials: {len(study.trials)}")

            # Create objective function for this pair
            def objective(trial, p=pair):
                return self._objective_single_pair(trial, p)

            # Run optimization
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=show_progress,
                n_jobs=self.config.n_jobs,
            )

            # Store results
            best_params = study.best_params
            best_score = study.best_value
            results[pair] = (best_params, best_score)

            # Store for export
            self._per_pair_results[pair] = {
                "params": best_params,
                "score": best_score,
                "study": study,
                "trials": len(study.trials),
                "best_trial": study.best_trial,
            }

            # Print pair results
            print(f"\n--- {pair} Best Parameters ---")
            for name, value in sorted(best_params.items()):
                if isinstance(value, float):
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {value}")
            print(f"  Best Score: {best_score:.4f}")

            # Print best trial details
            best_trial = study.best_trial
            print(f"\n--- {pair} Best Trial Details ---")
            for key, value in best_trial.user_attrs.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

            logger.info(
                f"Per-pair optimization completed for {pair}",
                best_score=best_score,
                best_params=best_params,
                total_trials=len(study.trials)
            )

        return results

    def export_per_pair_config(self, output_path: Optional[str] = None) -> str:
        """
        Export per-pair optimized parameters to a YAML config file.

        Args:
            output_path: Path for output file.

        Returns:
            Path to the exported file.
        """
        if not hasattr(self, '_per_pair_results') or not self._per_pair_results:
            raise ValueError("No per-pair results available. Run optimize_per_pair() first.")

        output_path = output_path or self.config.export_config_path
        if not output_path:
            output_path = "config/config.optimized.yaml"

        # Build config with per-pair parameters
        config_dict = {
            "pair_parameters": {}
        }

        for pair, result in self._per_pair_results.items():
            params = result["params"]
            pair_config = self.parameter_space.to_strategy_config(params)

            # Flatten the nested structure for per-pair config
            config_dict["pair_parameters"][pair] = {
                "strategy": pair_config.get("strategy", {}),
                "ml": pair_config.get("ml", {}),
                "risk": pair_config.get("risk", {}),
                "_optimization": {
                    "score": result["score"],
                    "trials": result["trials"],
                    "best_return": result["best_trial"].user_attrs.get("total_return", 0),
                    "best_trades": result["best_trial"].user_attrs.get("total_trades", 0),
                }
            }

        # Add metadata
        config_dict["_optimization"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "study_name": self.config.study_name,
            "objective": self.config.objective,
            "pairs": self.config.pairs,
            "period_days": self.config.period_days,
            "mode": "per_pair",
        }

        # Write to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        print(f"\nExported per-pair optimized config to {output_path}")
        logger.info(f"Exported per-pair optimized config to {output_path}")

        return output_path

    def print_per_pair_results(self) -> None:
        """Print per-pair optimization results summary."""
        if not hasattr(self, '_per_pair_results') or not self._per_pair_results:
            print("No per-pair results available. Run optimize_per_pair() first.")
            return

        print("\n" + "=" * 70)
        print("PER-PAIR OPTIMIZATION RESULTS SUMMARY")
        print("=" * 70)

        # Summary table header
        print(f"\n{'Pair':<12} {'Score':>10} {'Return':>10} {'Trades':>8} {'Win Rate':>10}")
        print("-" * 52)

        for pair, result in self._per_pair_results.items():
            score = result["score"]
            best_trial = result["best_trial"]
            total_return = best_trial.user_attrs.get("total_return", 0)
            total_trades = best_trial.user_attrs.get("total_trades", 0)
            win_rate = best_trial.user_attrs.get("win_rate", 0)

            # win_rate is already stored as percentage (0-100)
            print(f"{pair:<12} {score:>10.2f} {total_return*100:>9.2f}% {total_trades:>8} {win_rate:>9.1f}%")

        print("-" * 52)
        print("\n" + "=" * 70)
