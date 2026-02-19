#!/usr/bin/env python3
"""
Scalping Strategy Parameter Optimizer

Finds optimal scalping parameters using Bayesian optimization (Optuna)
with historical data from Binance via CCXT.

Usage:
    # Quick test (10 trials)
    python run_scalping_optimizer.py --pairs BTC/USD --trials 10

    # Full optimization for all pairs
    python run_scalping_optimizer.py --pairs BTC/USD ETH/USD SOL/USD --trials 50

    # Per-pair optimization (recommended)
    python run_scalping_optimizer.py --per-pair --pairs BTC/USD ETH/USD SOL/USD --trials 30
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import optuna
    from optuna.trial import TrialState
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False

from src.backtest.ccxt_data_provider import CCXTDataProvider
from src.backtest.data import HistoricalDataManager
from src.strategy.scalping_strategy import ScalpingStrategy, ScalpingConfig
from src.strategy.regime_detector import RegimeDetector, RegimeConfig, MarketRegime, REGIME_ADJUSTMENTS
from src.strategy.base_strategy import MarketData, Position, SignalType
from src.exchange.kraken_client import OHLC
from src.observability.logger import get_logger, configure_logging

logger = get_logger(__name__)


@dataclass
class ScalpingOptimizerConfig:
    """Configuration for scalping optimizer."""
    n_trials: int = 50
    timeout_seconds: int = 3600
    pairs: List[str] = None
    start_date: datetime = None
    end_date: datetime = None
    initial_capital: float = 10000.0
    interval: int = 60  # 1-hour candles
    fee_percent: float = 0.16  # Maker fee (use limit orders)
    position_size_percent: float = 20.0
    storage_path: str = "data/optimization/scalping_study.db"
    study_name: str = "scalping_optimization"
    cache_dir: str = "data/cache/ccxt"

    def __post_init__(self):
        if self.pairs is None:
            self.pairs = ["BTC/USD", "ETH/USD"]
        if self.end_date is None:
            self.end_date = datetime.now(timezone.utc)
        if self.start_date is None:
            self.start_date = self.end_date - timedelta(days=90)


# Scalping parameter ranges for optimization
SCALPING_PARAMETER_RANGES = {
    # Take profit (main target)
    "take_profit_percent": {"low": 2.0, "high": 8.0, "step": 0.5},
    # Stop loss
    "stop_loss_percent": {"low": 1.0, "high": 4.0, "step": 0.5},
    # RSI settings
    "rsi_period": {"low": 5, "high": 14, "step": 1, "type": "int"},
    "rsi_oversold": {"low": 20, "high": 40, "step": 5, "type": "int"},
    "rsi_overbought": {"low": 60, "high": 80, "step": 5, "type": "int"},
    # Bollinger settings
    "bb_period": {"low": 10, "high": 30, "step": 5, "type": "int"},
    "bb_std_dev": {"low": 1.5, "high": 2.5, "step": 0.25},
    # VWAP threshold
    "vwap_threshold_percent": {"low": 0.1, "high": 1.0, "step": 0.1},
    # Volume spike
    "volume_spike_threshold": {"low": 1.0, "high": 2.5, "step": 0.25},
    # Confirmations required
    "min_confirmations": {"low": 1, "high": 4, "step": 1, "type": "int"},
}


class ScalpingBacktestRunner:
    """Runs scalping backtests with given parameters."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_size_percent: float = 20.0,
        fee_percent: float = 0.26
    ):
        self.initial_capital = initial_capital
        self.position_size_percent = position_size_percent
        self.fee_percent = fee_percent

    def run(
        self,
        candles: List[OHLC],
        pair: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a backtest with given parameters.

        Returns:
            Dictionary with metrics: total_return, sharpe_ratio, win_rate, etc.
        """
        # Create strategy with these params
        config = ScalpingConfig(
            take_profit_percent=params.get("take_profit_percent", 4.5),
            stop_loss_percent=params.get("stop_loss_percent", 2.0),
            rsi_period=params.get("rsi_period", 7),
            rsi_oversold=params.get("rsi_oversold", 30.0),
            rsi_overbought=params.get("rsi_overbought", 70.0),
            bb_period=params.get("bb_period", 20),
            bb_std_dev=params.get("bb_std_dev", 2.0),
            vwap_threshold_percent=params.get("vwap_threshold_percent", 0.3),
            volume_spike_threshold=params.get("volume_spike_threshold", 1.5),
            min_confirmations=params.get("min_confirmations", 2),
            fee_percent=self.fee_percent,
        )

        strategy = ScalpingStrategy(config)

        # Run backtest
        capital = self.initial_capital
        peak_capital = capital
        trades = []

        in_position = False
        entry_price = 0.0
        entry_time = None
        position_size_usd = 0.0

        lookback = max(config.bb_period, config.rsi_period) + 5

        for i in range(lookback, len(candles)):
            window = candles[i - lookback:i + 1]
            current = candles[i]

            market_data = MarketData(
                pair=pair,
                ohlc=window,
                prices=[c.close for c in window],
                volumes=[c.volume for c in window],
                ticker=None
            )

            if in_position:
                position = Position(
                    pair=pair,
                    side="long",
                    entry_price=entry_price,
                    current_price=current.close,
                    size=position_size_usd / entry_price,
                    entry_time=entry_time
                )

                signal = strategy.analyze(market_data, position)

                if signal.signal_type in (SignalType.SELL, SignalType.CLOSE_LONG):
                    # Exit
                    exit_price = current.close
                    gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    net_pnl_pct = gross_pnl_pct - (self.fee_percent * 2)
                    pnl_usd = position_size_usd * (net_pnl_pct / 100)

                    capital += pnl_usd
                    peak_capital = max(peak_capital, capital)

                    trades.append({
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl_percent": net_pnl_pct,
                        "pnl_usd": pnl_usd,
                        "win": pnl_usd > 0
                    })

                    in_position = False
            else:
                signal = strategy.analyze(market_data, None)

                if signal.signal_type == SignalType.BUY:
                    entry_price = current.close
                    entry_time = current.timestamp
                    position_size_usd = capital * (self.position_size_percent / 100)
                    in_position = True

        # Calculate metrics
        return self._calculate_metrics(trades, capital)

    def _calculate_metrics(self, trades: List[dict], final_capital: float) -> Dict[str, Any]:
        """Calculate performance metrics from trades."""
        if not trades:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_return": 0.0,
                "total_pnl_percent": 0.0,
                "avg_pnl_percent": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_percent": 0.0,
                "score": float("-inf"),
            }

        wins = sum(1 for t in trades if t["win"])
        losses = len(trades) - wins
        win_rate = wins / len(trades) if trades else 0

        total_pnl_pct = sum(t["pnl_percent"] for t in trades)
        avg_pnl_pct = total_pnl_pct / len(trades)
        total_return = (final_capital - self.initial_capital) / self.initial_capital

        # Profit factor
        gross_profit = sum(t["pnl_usd"] for t in trades if t["pnl_usd"] > 0)
        gross_loss = abs(sum(t["pnl_usd"] for t in trades if t["pnl_usd"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe estimate
        returns = [t["pnl_percent"] for t in trades]
        if len(returns) > 1:
            avg_ret = sum(returns) / len(returns)
            variance = sum((r - avg_ret) ** 2 for r in returns) / (len(returns) - 1)
            std_dev = variance ** 0.5
            sharpe = avg_ret / std_dev if std_dev > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0
        running = self.initial_capital
        for t in trades:
            running += t["pnl_usd"]
            peak = max(peak, running)
            dd = (peak - running) / peak * 100
            max_dd = max(max_dd, dd)

        # Composite score (same formula as momentum optimizer)
        # Penalize too few trades
        trade_penalty = 0 if len(trades) >= 20 else (20 - len(trades)) * 0.5

        score = (
            total_return * 100 * 0.4 +  # Weight returns
            sharpe * 10 * 0.3 +          # Weight risk-adjusted returns
            profit_factor * 5 * 0.2 -    # Weight consistency
            max_dd * 0.1 -               # Penalize drawdown
            trade_penalty                # Penalize too few trades
        )

        return {
            "total_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate * 100,  # As percentage
            "total_return": total_return,
            "total_pnl_percent": total_pnl_pct,
            "avg_pnl_percent": avg_pnl_pct,
            "profit_factor": profit_factor if profit_factor != float("inf") else 999.0,
            "sharpe_ratio": sharpe,
            "max_drawdown_percent": max_dd,
            "score": score,
        }


class ScalpingOptimizer:
    """Optimizer for scalping strategy parameters."""

    def __init__(self, config: ScalpingOptimizerConfig):
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "optuna is required for optimization. "
                "Install it with: pip install optuna"
            )

        self.config = config
        self._historical_data: Dict[str, List[OHLC]] = {}
        self.study: Optional[optuna.Study] = None
        self._best_params: Optional[Dict[str, Any]] = None
        self._best_score: Optional[float] = None
        self._per_pair_results: Dict[str, Any] = {}

        # Ensure directories exist
        Path(config.storage_path).parent.mkdir(parents=True, exist_ok=True)
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    def _fetch_pair_data(
        self, pair: str, show_progress: bool = True
    ) -> List[OHLC]:
        """Fetch data for a pair, trying CCXT cache then Kraken API."""
        def progress_cb(current: int, total: int) -> None:
            if show_progress and total > 0:
                pct = (current / total * 100)
                print(f"\r    Progress: {current}/{total} ({pct:.0f}%)...", end="", flush=True)

        # Check CCXT cache first (no network call)
        cache_dir = Path(self.config.cache_dir)
        if cache_dir.exists():
            for cache_file in cache_dir.glob("*.json"):
                pair_clean = pair.replace("/", "_")
                if pair_clean in cache_file.name:
                    try:
                        provider = CCXTDataProvider(
                            cache_dir=self.config.cache_dir, use_cache=True
                        )
                        # Only check cache -- don't fetch from network
                        candles = provider._load_from_cache(
                            pair, self.config.start_date, self.config.end_date,
                            self.config.interval
                        )
                        if candles and len(candles) > 100:
                            print(f"    Loaded {len(candles)} candles from CCXT cache")
                            return candles
                    except Exception:
                        pass

        # Use Kraken API with pagination
        print(f"    Using Kraken API (paginated)...")
        kraken_dm = HistoricalDataManager(
            cache_dir=self.config.cache_dir,
            use_cache=True,
        )
        candles = kraken_dm.get_ohlc_range(
            pair=pair,
            start=self.config.start_date,
            end=self.config.end_date,
            interval=self.config.interval,
            progress_callback=progress_cb if show_progress else None,
        )
        if show_progress:
            print()
        return candles

    def prefetch_data(self, show_progress: bool = True) -> None:
        """Pre-fetch historical data for all pairs."""
        print(f"\nFetching historical data...")
        print(f"  Period: {self.config.start_date.date()} to {self.config.end_date.date()}")
        print(f"  Interval: {self.config.interval} minutes")

        for pair in self.config.pairs:
            print(f"\n  Fetching {pair}...")

            candles = self._fetch_pair_data(pair, show_progress)

            if not candles:
                raise ValueError(f"Failed to fetch data for {pair}")

            self._historical_data[pair] = candles
            print(f"    Loaded {len(candles)} candles for {pair}")

    def _sample_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample parameters from the trial."""
        params = {}

        for name, ranges in SCALPING_PARAMETER_RANGES.items():
            param_type = ranges.get("type", "float")

            if param_type == "int":
                params[name] = trial.suggest_int(
                    name,
                    int(ranges["low"]),
                    int(ranges["high"]),
                    step=int(ranges.get("step", 1))
                )
            else:
                params[name] = trial.suggest_float(
                    name,
                    ranges["low"],
                    ranges["high"],
                    step=ranges.get("step")
                )

        return params

    def _objective_single_pair(self, trial: optuna.Trial, pair: str) -> float:
        """Objective function for single pair optimization."""
        params = self._sample_params(trial)

        candles = self._historical_data.get(pair)
        if not candles:
            return float("-inf")

        runner = ScalpingBacktestRunner(
            initial_capital=self.config.initial_capital,
            position_size_percent=self.config.position_size_percent,
            fee_percent=self.config.fee_percent
        )

        metrics = runner.run(candles, pair, params)

        # Store metrics as user attributes
        trial.set_user_attr("total_trades", metrics["total_trades"])
        trial.set_user_attr("win_rate", metrics["win_rate"])
        trial.set_user_attr("total_return", metrics["total_return"])
        trial.set_user_attr("sharpe_ratio", metrics["sharpe_ratio"])
        trial.set_user_attr("max_drawdown", metrics["max_drawdown_percent"])
        trial.set_user_attr("profit_factor", metrics["profit_factor"])

        return metrics["score"]

    def optimize_per_pair(
        self,
        n_trials: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, Tuple[Dict[str, Any], float]]:
        """
        Run separate optimization for each pair.

        Returns:
            Dictionary mapping pair -> (best_params, best_score)
        """
        n_trials = n_trials or self.config.n_trials

        # Pre-fetch data
        if not self._historical_data:
            self.prefetch_data(show_progress)

        results: Dict[str, Tuple[Dict[str, Any], float]] = {}

        for pair in self.config.pairs:
            pair_clean = pair.replace("/", "_")
            study_name = f"{self.config.study_name}_{pair_clean}"
            storage_path = self.config.storage_path.replace(".db", f"_{pair_clean}.db")
            storage_url = f"sqlite:///{storage_path}"

            print(f"\n{'=' * 60}")
            print(f"OPTIMIZING: {pair}")
            print(f"{'=' * 60}")

            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction="maximize",
                load_if_exists=True,
            )

            print(f"  Study: {study_name}")
            print(f"  Existing trials: {len(study.trials)}")

            def objective(trial, p=pair):
                return self._objective_single_pair(trial, p)

            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=self.config.timeout_seconds,
                show_progress_bar=show_progress,
            )

            best_params = study.best_params
            best_score = study.best_value
            results[pair] = (best_params, best_score)

            self._per_pair_results[pair] = {
                "params": best_params,
                "score": best_score,
                "study": study,
                "trials": len(study.trials),
                "best_trial": study.best_trial,
            }

            # Print results
            print(f"\n--- {pair} Best Parameters ---")
            for name, value in sorted(best_params.items()):
                if isinstance(value, float):
                    print(f"    {name}: {value:.2f}")
                else:
                    print(f"    {name}: {value}")
            print(f"    Score: {best_score:.2f}")

            # Print metrics
            best_trial = study.best_trial
            print(f"\n--- {pair} Best Metrics ---")
            print(f"    Return: {best_trial.user_attrs.get('total_return', 0)*100:.2f}%")
            print(f"    Trades: {best_trial.user_attrs.get('total_trades', 0)}")
            print(f"    Win Rate: {best_trial.user_attrs.get('win_rate', 0):.1f}%")
            print(f"    Profit Factor: {best_trial.user_attrs.get('profit_factor', 0):.2f}")
            print(f"    Max Drawdown: {best_trial.user_attrs.get('max_drawdown', 0):.1f}%")

        return results

    def print_summary(self) -> None:
        """Print optimization results summary."""
        if not self._per_pair_results:
            print("No results available.")
            return

        print(f"\n{'=' * 70}")
        print("SCALPING OPTIMIZATION SUMMARY")
        print(f"{'=' * 70}")

        print(f"\n{'Pair':<12} {'Score':>10} {'Return':>10} {'Trades':>8} {'Win Rate':>10} {'PF':>8}")
        print("-" * 60)

        for pair, result in self._per_pair_results.items():
            score = result["score"]
            bt = result["best_trial"]
            ret = bt.user_attrs.get("total_return", 0) * 100
            trades = bt.user_attrs.get("total_trades", 0)
            wr = bt.user_attrs.get("win_rate", 0)
            pf = bt.user_attrs.get("profit_factor", 0)

            print(f"{pair:<12} {score:>10.2f} {ret:>9.2f}% {trades:>8} {wr:>9.1f}% {pf:>8.2f}")

        print("-" * 60)

        # Best parameters per pair
        print(f"\n{'=' * 70}")
        print("BEST PARAMETERS BY PAIR")
        print(f"{'=' * 70}")

        for pair, result in self._per_pair_results.items():
            params = result["params"]
            print(f"\n{pair}:")
            print(f"  Take Profit: {params.get('take_profit_percent', 0):.1f}%")
            print(f"  Stop Loss: {params.get('stop_loss_percent', 0):.1f}%")
            print(f"  RSI: period={params.get('rsi_period', 7)}, oversold={params.get('rsi_oversold', 30)}, overbought={params.get('rsi_overbought', 70)}")
            print(f"  BB: period={params.get('bb_period', 20)}, std_dev={params.get('bb_std_dev', 2.0):.2f}")
            print(f"  VWAP threshold: {params.get('vwap_threshold_percent', 0.3):.2f}%")
            print(f"  Volume spike: {params.get('volume_spike_threshold', 1.5):.2f}x")
            print(f"  Min confirmations: {params.get('min_confirmations', 2)}")

    def export_config(self, output_path: str = "config/scalping.optimized.yaml") -> str:
        """Export optimized parameters to YAML."""
        import yaml

        if not self._per_pair_results:
            raise ValueError("No results to export.")

        config = {
            "_optimization": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "study_name": self.config.study_name,
                "period_days": (self.config.end_date - self.config.start_date).days,
            },
            "pair_parameters": {}
        }

        for pair, result in self._per_pair_results.items():
            params = result["params"]
            config["pair_parameters"][pair] = {
                "take_profit_percent": params.get("take_profit_percent", 4.5),
                "stop_loss_percent": params.get("stop_loss_percent", 2.0),
                "rsi_period": params.get("rsi_period", 7),
                "rsi_oversold": params.get("rsi_oversold", 30),
                "rsi_overbought": params.get("rsi_overbought", 70),
                "bb_period": params.get("bb_period", 20),
                "bb_std_dev": params.get("bb_std_dev", 2.0),
                "vwap_threshold_percent": params.get("vwap_threshold_percent", 0.3),
                "volume_spike_threshold": params.get("volume_spike_threshold", 1.5),
                "min_confirmations": params.get("min_confirmations", 2),
                "_metrics": {
                    "score": result["score"],
                    "return": result["best_trial"].user_attrs.get("total_return", 0),
                    "win_rate": result["best_trial"].user_attrs.get("win_rate", 0),
                    "trades": result["best_trial"].user_attrs.get("total_trades", 0),
                }
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"\nExported optimized config to {output_path}")
        return output_path


class RegimeAwareOptimizer:
    """
    Optimizes scalping parameters per market regime.

    Segments historical data into bull/bear/sideways windows using the
    RegimeDetector, then runs separate Optuna studies for each regime.
    The output is a config with per-regime parameter sets that the live
    trader can switch between automatically.
    """

    def __init__(self, config: ScalpingOptimizerConfig):
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna is required. Install: pip install optuna")

        self.config = config
        self._historical_data: Dict[str, List[OHLC]] = {}
        self._regime_detector = RegimeDetector(RegimeConfig(
            fast_sma_period=50,
            slow_sma_period=200,
            min_data_points=200,
        ))
        self._results: Dict[str, Dict[str, Any]] = {}  # pair -> regime -> results

        # Re-use the same data fetch logic as ScalpingOptimizer
        self._data_fetcher = ScalpingOptimizer(config)

    def prefetch_data(self, show_progress: bool = True) -> None:
        """Pre-fetch historical data for all pairs."""
        self._data_fetcher.prefetch_data(show_progress)
        self._historical_data = self._data_fetcher._historical_data

    def segment_by_regime(self, pair: str) -> Dict[MarketRegime, List[List[OHLC]]]:
        """
        Segment candle data into windows by detected regime.

        Returns a dict mapping regime -> list of contiguous candle windows
        belonging to that regime. Each window is at least 200 candles.
        """
        candles = self._historical_data[pair]
        if len(candles) < 250:
            return {MarketRegime.UNKNOWN: [candles]}

        # Classify each candle (using rolling window)
        min_pts = 200
        regime_labels = []
        for i in range(len(candles)):
            if i < min_pts:
                regime_labels.append(MarketRegime.UNKNOWN)
                continue
            window = candles[max(0, i - min_pts):i + 1]
            closes = [c.close for c in window]
            highs = [c.high for c in window]
            lows = [c.low for c in window]
            result = self._regime_detector.detect(closes, highs, lows)
            regime_labels.append(result.regime)

        # Group contiguous runs of the same regime into windows
        segments: Dict[MarketRegime, List[List[OHLC]]] = {
            MarketRegime.BULL: [],
            MarketRegime.BEAR: [],
            MarketRegime.SIDEWAYS: [],
        }

        current_regime = regime_labels[min_pts] if len(regime_labels) > min_pts else MarketRegime.UNKNOWN
        window_start = min_pts

        for i in range(min_pts + 1, len(candles)):
            if regime_labels[i] != current_regime or i == len(candles) - 1:
                # End of segment
                window = candles[window_start:i]
                if len(window) >= 50 and current_regime in segments:  # Min 50 candles
                    segments[current_regime].append(window)
                current_regime = regime_labels[i]
                window_start = i

        # Print segment summary
        for regime, windows in segments.items():
            total = sum(len(w) for w in windows)
            print(f"    {regime.value}: {len(windows)} windows, {total} total candles")

        return segments

    def optimize_per_regime(
        self,
        n_trials: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run per-pair, per-regime optimization.

        Returns nested dict: pair -> regime -> {params, score, metrics}
        """
        n_trials = n_trials or self.config.n_trials

        if not self._historical_data:
            self.prefetch_data(show_progress)

        all_results: Dict[str, Dict[str, Any]] = {}

        for pair in self.config.pairs:
            print(f"\n{'=' * 60}")
            print(f"REGIME-AWARE OPTIMIZATION: {pair}")
            print(f"{'=' * 60}")

            segments = self.segment_by_regime(pair)
            pair_results: Dict[str, Any] = {}

            for regime, windows in segments.items():
                if not windows:
                    print(f"\n  {regime.value}: No data -- skipping")
                    continue

                # Concatenate all windows for this regime
                all_candles = []
                for w in windows:
                    all_candles.extend(w)

                if len(all_candles) < 100:
                    print(f"\n  {regime.value}: Only {len(all_candles)} candles -- skipping")
                    continue

                print(f"\n  --- Optimizing for {regime.value} regime ({len(all_candles)} candles) ---")

                pair_clean = pair.replace("/", "_")
                study_name = f"regime_{pair_clean}_{regime.value}"
                storage_path = self.config.storage_path.replace(
                    ".db", f"_regime_{pair_clean}_{regime.value}.db"
                )
                storage_url = f"sqlite:///{storage_path}"
                Path(storage_path).parent.mkdir(parents=True, exist_ok=True)

                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage_url,
                    direction="maximize",
                    load_if_exists=True,
                )

                runner = ScalpingBacktestRunner(
                    initial_capital=self.config.initial_capital,
                    position_size_percent=self.config.position_size_percent,
                    fee_percent=self.config.fee_percent,
                )

                def objective(trial, c=all_candles, p=pair):
                    params = {}
                    for name, ranges in SCALPING_PARAMETER_RANGES.items():
                        param_type = ranges.get("type", "float")
                        if param_type == "int":
                            params[name] = trial.suggest_int(
                                name, int(ranges["low"]), int(ranges["high"]),
                                step=int(ranges.get("step", 1))
                            )
                        else:
                            params[name] = trial.suggest_float(
                                name, ranges["low"], ranges["high"],
                                step=ranges.get("step")
                            )
                    metrics = runner.run(c, p, params)
                    trial.set_user_attr("total_trades", metrics["total_trades"])
                    trial.set_user_attr("win_rate", metrics["win_rate"])
                    trial.set_user_attr("total_return", metrics["total_return"])
                    trial.set_user_attr("sharpe_ratio", metrics["sharpe_ratio"])
                    trial.set_user_attr("profit_factor", metrics["profit_factor"])
                    trial.set_user_attr("max_drawdown", metrics["max_drawdown_percent"])
                    return metrics["score"]

                study.optimize(
                    objective,
                    n_trials=n_trials,
                    timeout=self.config.timeout_seconds,
                    show_progress_bar=show_progress,
                )

                best = study.best_trial
                pair_results[regime.value] = {
                    "params": study.best_params,
                    "score": study.best_value,
                    "return": best.user_attrs.get("total_return", 0),
                    "win_rate": best.user_attrs.get("win_rate", 0),
                    "trades": best.user_attrs.get("total_trades", 0),
                    "profit_factor": best.user_attrs.get("profit_factor", 0),
                    "max_drawdown": best.user_attrs.get("max_drawdown", 0),
                }

                print(f"    Best score: {study.best_value:.2f}")
                print(f"    Return: {best.user_attrs.get('total_return', 0)*100:.2f}%")
                print(f"    Win rate: {best.user_attrs.get('win_rate', 0):.1f}%")

            all_results[pair] = pair_results

        self._results = all_results
        return all_results

    def print_summary(self) -> None:
        """Print regime-aware optimization summary."""
        print(f"\n{'=' * 70}")
        print("REGIME-AWARE OPTIMIZATION SUMMARY")
        print(f"{'=' * 70}")

        for pair, regimes in self._results.items():
            print(f"\n  {pair}:")
            print(f"  {'Regime':<12} {'Score':>8} {'Return':>10} {'Trades':>8} {'WinRate':>10} {'PF':>8}")
            print(f"  {'-'*58}")

            for regime, result in regimes.items():
                ret = result.get("return", 0) * 100
                trades = result.get("trades", 0)
                wr = result.get("win_rate", 0)
                pf = result.get("profit_factor", 0)
                score = result.get("score", 0)
                print(f"  {regime:<12} {score:>8.1f} {ret:>9.2f}% {trades:>8} {wr:>9.1f}% {pf:>8.2f}")

    def export_config(self, output_path: str = "config/scalping.regime.yaml") -> str:
        """Export regime-aware config to YAML."""
        import yaml

        config = {
            "_optimization": {
                "type": "regime_aware",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "period_days": (self.config.end_date - self.config.start_date).days,
                "regimes": ["bull", "bear", "sideways"],
            },
            "regime_parameters": {},
        }

        for pair, regimes in self._results.items():
            config["regime_parameters"][pair] = {}
            for regime, result in regimes.items():
                params = result.get("params", {})
                config["regime_parameters"][pair][regime] = {
                    **params,
                    "_metrics": {
                        "score": result.get("score", 0),
                        "return": result.get("return", 0),
                        "win_rate": result.get("win_rate", 0),
                        "trades": result.get("trades", 0),
                    }
                }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"\nExported regime-aware config to {output_path}")
        return output_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize scalping strategy parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--pairs", nargs="+",
        default=["BTC/USD", "ETH/USD", "SOL/USD"],
        help="Trading pairs to optimize"
    )
    parser.add_argument(
        "--trials", "-n", type=int, default=50,
        help="Number of trials per pair"
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        choices=[1, 5, 15, 30, 60, 240],
        help="Candle interval in minutes (default: 60)"
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Days of historical data (default: 90, use 730 for 2 years)"
    )
    parser.add_argument(
        "--per-pair", action="store_true",
        help="Optimize each pair separately (recommended)"
    )
    parser.add_argument(
        "--regime-aware", action="store_true",
        help="Segment data by bull/bear/sideways and optimize per regime"
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export optimized config"
    )
    parser.add_argument(
        "--export-path", type=str, default="config/scalping.optimized.yaml",
        help="Path for exported config"
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable progress bar"
    )

    return parser.parse_args()


def parse_date(date_str: str) -> datetime:
    """Parse date string."""
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main():
    """Main entry point."""
    args = parse_args()

    configure_logging(level="WARNING", format_type="json")

    print("=" * 60)
    print("SCALPING STRATEGY OPTIMIZER")
    print("=" * 60)

    # Parse dates
    end_date = parse_date(args.end) if args.end else datetime.now(timezone.utc)
    start_date = parse_date(args.start) if args.start else end_date - timedelta(days=args.days)

    mode = "REGIME-AWARE" if args.regime_aware else "PER-PAIR"
    print(f"\nConfiguration:")
    print(f"  Mode: {mode}")
    print(f"  Pairs: {', '.join(args.pairs)}")
    print(f"  Trials: {args.trials} per pair/regime")
    print(f"  Period: {start_date.date()} to {end_date.date()} ({args.days} days)")
    print(f"  Interval: {args.interval} minutes")

    config = ScalpingOptimizerConfig(
        n_trials=args.trials,
        pairs=args.pairs,
        start_date=start_date,
        end_date=end_date,
        interval=args.interval,
    )

    try:
        if args.regime_aware:
            # Regime-aware optimization: segments data by bull/bear/sideways
            optimizer = RegimeAwareOptimizer(config)
            results = optimizer.optimize_per_regime(
                show_progress=not args.no_progress
            )
            optimizer.print_summary()
            if args.export:
                export_path = args.export_path.replace(
                    ".yaml", ".regime.yaml"
                ) if "regime" not in args.export_path else args.export_path
                optimizer.export_config(export_path)
        else:
            # Standard per-pair optimization
            optimizer = ScalpingOptimizer(config)
            results = optimizer.optimize_per_pair(
                show_progress=not args.no_progress
            )
            optimizer.print_summary()
            if args.export:
                optimizer.export_config(args.export_path)

    except ImportError as e:
        print(f"\nError: {e}")
        print("Install: pip install optuna ccxt")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted.")
        if hasattr(optimizer, '_per_pair_results') and optimizer._per_pair_results:
            optimizer.print_summary()
        elif hasattr(optimizer, '_results') and optimizer._results:
            optimizer.print_summary()
        sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nOptimization complete!")


if __name__ == "__main__":
    main()
