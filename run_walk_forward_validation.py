#!/usr/bin/env python3
"""
Walk-Forward Validation — the gold standard for detecting overfitting.

Splits 2 years of data into 4 x 6-month windows:
  Window 1: Train on months 1-6,   Test on months 7-12
  Window 2: Train on months 7-12,  Test on months 13-18
  Window 3: Train on months 13-18, Test on months 19-24

For each window:
  1. Optimize params on training period (using Optuna, 50 trials)
  2. Run those params on the test period (completely unseen data)
  3. Record OOS performance

If returns are real: OOS performance should be positive across most windows.
If overfitting: OOS performance will be random/negative.

Also tests: current config params on each 6-month window independently
to check for regime-dependent performance.
"""

import argparse
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.exchange.kraken_client import OHLC
from src.strategy.scalping_strategy import ScalpingStrategy, ScalpingConfig
from src.strategy.regime_detector import RegimeDetector, RegimeConfig, MarketRegime, REGIME_ADJUSTMENTS
from src.strategy.base_strategy import MarketData, Position, SignalType

# Reuse the optimizer's backtest runner (now with trailing stops)
from run_scalping_optimizer import ScalpingBacktestRunner, SCALPING_PARAMETER_RANGES

import yaml


def load_config():
    with open("config/scalping.yaml") as f:
        return yaml.safe_load(f)


def load_pair_data(pair: str, start: datetime, end: datetime) -> List[OHLC]:
    """Load data for a pair from Kraken CSV files."""
    from src.backtest.kraken_csv_provider import KrakenCSVProvider
    provider = KrakenCSVProvider()
    return provider.get_ohlc_range(pair, start, end, interval=240)


def get_current_params(config: dict, pair: str) -> Dict[str, Any]:
    """Extract current optimized params for a pair from config."""
    pp = config.get("pair_parameters", {}).get(pair, {})
    return {
        "take_profit_percent": pp.get("take_profit_percent", 4.5),
        "stop_loss_percent": pp.get("stop_loss_percent", 2.5),
        "rsi_period": pp.get("rsi_period", 10),
        "rsi_oversold": pp.get("rsi_oversold", 28),
        "rsi_overbought": pp.get("rsi_overbought", 70),
        "bb_period": pp.get("bb_period", 20),
        "bb_std_dev": pp.get("bb_std_dev", 2.0),
        "vwap_threshold_percent": pp.get("vwap_threshold_percent", 0.5),
        "volume_spike_threshold": pp.get("volume_spike_threshold", 1.5),
        "min_confirmations": pp.get("min_confirmations", 3),
        "stoch_k_period": pp.get("stoch_k_period", 13),
        "stoch_oversold": pp.get("stoch_oversold", 23),
        "stoch_overbought": pp.get("stoch_overbought", 75),
        "atr_period": pp.get("atr_period", 14),
        "atr_stop_multiplier": pp.get("atr_stop_multiplier", 2.5),
        "atr_tp_multiplier": pp.get("atr_tp_multiplier", 2.75),
        "short_min_confirmations": pp.get("short_min_confirmations", 3),
    }


def quick_optimize(candles: List[OHLC], pair: str, n_trials: int = 50) -> Dict[str, Any]:
    """Run a quick Optuna optimization and return best params."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    runner = ScalpingBacktestRunner(
        initial_capital=10000.0,
        position_size_percent=20.0,
        fee_percent=0.16
    )

    def objective(trial):
        params = {}
        for name, ranges in SCALPING_PARAMETER_RANGES.items():
            param_type = ranges.get("type", "float")
            if param_type == "int":
                params[name] = trial.suggest_int(name, int(ranges["low"]), int(ranges["high"]), step=int(ranges.get("step", 1)))
            else:
                params[name] = trial.suggest_float(name, ranges["low"], ranges["high"], step=ranges.get("step"))

        metrics = runner.run(candles, pair, params)
        return metrics.get("score", float("-inf"))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params


def run_backtest(candles: List[OHLC], pair: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Run backtest with given params."""
    runner = ScalpingBacktestRunner(
        initial_capital=10000.0,
        position_size_percent=20.0,
        fee_percent=0.16
    )
    return runner.run(candles, pair, params)


def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--pairs", type=str, default=None, help="Comma-separated pairs (default: all)")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per window")
    args = parser.parse_args()

    config = load_config()
    all_pairs = config.get("pairs", [])
    pairs = args.pairs.split(",") if args.pairs else all_pairs

    end = datetime(2026, 3, 2, tzinfo=timezone.utc)
    start = datetime(2024, 3, 2, tzinfo=timezone.utc)

    # Define 6-month windows
    windows = [
        ("H1 2024", datetime(2024, 3, 2, tzinfo=timezone.utc), datetime(2024, 9, 2, tzinfo=timezone.utc)),
        ("H2 2024", datetime(2024, 9, 2, tzinfo=timezone.utc), datetime(2025, 3, 2, tzinfo=timezone.utc)),
        ("H1 2025", datetime(2025, 3, 2, tzinfo=timezone.utc), datetime(2025, 9, 2, tzinfo=timezone.utc)),
        ("H2 2025", datetime(2025, 9, 2, tzinfo=timezone.utc), datetime(2026, 3, 2, tzinfo=timezone.utc)),
    ]

    # Walk-forward pairs: train on window N, test on window N+1
    wf_pairs_def = [
        ("Train H1→Test H2 2024", 0, 1),
        ("Train H2 2024→Test H1 2025", 1, 2),
        ("Train H1→Test H2 2025", 2, 3),
    ]

    print("=" * 90)
    print("WALK-FORWARD VALIDATION + STABILITY ANALYSIS")
    print(f"Period: {start.date()} to {end.date()} | {len(pairs)} pairs")
    print("=" * 90)

    # ============================================================
    # PART 1: Current config stability across 6-month windows
    # ============================================================
    print("\n" + "=" * 90)
    print("PART 1: CURRENT CONFIG — STABILITY ACROSS TIME WINDOWS")
    print("Do current params work across ALL time periods, or just the training period?")
    print("=" * 90)

    pair_window_results = {}

    for pair in pairs:
        print(f"\n  Loading {pair}...", end="", flush=True)
        candles = load_pair_data(pair, start, end)
        print(f" {len(candles)} candles")

        params = get_current_params(config, pair)
        pair_window_results[pair] = []

        for wname, wstart, wend in windows:
            window_candles = [c for c in candles if wstart <= c.timestamp < wend]
            if len(window_candles) < 100:
                pair_window_results[pair].append({"window": wname, "return": 0, "trades": 0, "wr": 0})
                continue

            metrics = run_backtest(window_candles, pair, params)
            r = metrics.get("total_return", 0) * 100
            trades = metrics.get("total_trades", 0)
            wr = metrics.get("win_rate", 0)
            pair_window_results[pair].append({"window": wname, "return": r, "trades": trades, "wr": wr})

    # Print stability table
    print(f"\n{'Pair':>10} | {'H1 2024':>10} | {'H2 2024':>10} | {'H1 2025':>10} | {'H2 2025':>10} | {'Positive':>8}")
    print("-" * 76)

    stability_scores = {}
    for pair in pairs:
        results = pair_window_results[pair]
        returns = [r["return"] for r in results]
        positive_count = sum(1 for r in returns if r > 0)
        stability_scores[pair] = positive_count

        cols = []
        for r in results:
            val = f"{r['return']:+.1f}%"
            cols.append(f"{val:>10}")

        print(f"{pair:>10} | {' | '.join(cols)} | {positive_count}/4")

    total_positive = sum(stability_scores.values())
    total_windows = len(pairs) * 4
    print(f"\n  Stability score: {total_positive}/{total_windows} windows positive ({total_positive/total_windows*100:.0f}%)")
    print("  (Overfitting indicator: if most returns are concentrated in 1-2 windows)")

    # ============================================================
    # PART 2: Walk-forward validation (train/test on sequential windows)
    # ============================================================
    print("\n" + "=" * 90)
    print("PART 2: WALK-FORWARD — Train on one period, test on the NEXT (unseen)")
    print("This is the hardest test. Params are optimized fresh, then tested on future data.")
    print("=" * 90)

    wf_results = {pair: [] for pair in pairs}

    for wf_name, train_idx, test_idx in wf_pairs_def:
        train_name, train_start, train_end = windows[train_idx]
        test_name, test_start, test_end = windows[test_idx]

        print(f"\n  {wf_name}:")

        for pair in pairs:
            candles = load_pair_data(pair, start, end)

            train_candles = [c for c in candles if train_start <= c.timestamp < train_end]
            test_candles = [c for c in candles if test_start <= c.timestamp < test_end]

            if len(train_candles) < 100 or len(test_candles) < 100:
                print(f"    {pair}: insufficient data")
                wf_results[pair].append({"wf": wf_name, "train_return": 0, "test_return": 0, "trades": 0})
                continue

            # Optimize on training window
            best_params = quick_optimize(train_candles, pair, n_trials=args.trials)

            # Test on completely unseen data
            train_metrics = run_backtest(train_candles, pair, best_params)
            test_metrics = run_backtest(test_candles, pair, best_params)

            train_r = train_metrics.get("total_return", 0) * 100
            test_r = test_metrics.get("total_return", 0) * 100
            test_trades = test_metrics.get("total_trades", 0)
            test_wr = test_metrics.get("win_rate", 0)

            decay = ((test_r / train_r) * 100) if train_r != 0 else 0

            print(f"    {pair:>10}: Train {train_r:+7.1f}% → Test {test_r:+7.1f}%  "
                  f"({test_trades} trades, {test_wr:.0f}% WR, {decay:.0f}% retention)")

            wf_results[pair].append({
                "wf": wf_name,
                "train_return": train_r,
                "test_return": test_r,
                "trades": test_trades,
                "wr": test_wr,
            })

    # Summary
    print("\n" + "=" * 90)
    print("WALK-FORWARD SUMMARY")
    print("=" * 90)

    print(f"\n{'Pair':>10} | {'WF1 OOS':>10} | {'WF2 OOS':>10} | {'WF3 OOS':>10} | {'Avg OOS':>10} | {'Positive':>8}")
    print("-" * 76)

    all_oos = []
    for pair in pairs:
        oos_returns = [r["test_return"] for r in wf_results[pair]]
        avg_oos = sum(oos_returns) / len(oos_returns) if oos_returns else 0
        positive = sum(1 for r in oos_returns if r > 0)
        all_oos.extend(oos_returns)

        cols = [f"{r:+.1f}%".rjust(10) for r in oos_returns]
        print(f"{pair:>10} | {' | '.join(cols)} | {f'{avg_oos:+.1f}%':>10} | {positive}/3")

    total_oos_positive = sum(1 for r in all_oos if r > 0)
    avg_all_oos = sum(all_oos) / len(all_oos) if all_oos else 0

    print(f"\n  Walk-forward OOS: {total_oos_positive}/{len(all_oos)} windows positive ({total_oos_positive/len(all_oos)*100:.0f}%)")
    print(f"  Average OOS return per window: {avg_all_oos:+.1f}%")
    print()

    # Verdict
    print("=" * 90)
    print("OVERFITTING VERDICT")
    print("=" * 90)
    stability_pct = total_positive / total_windows * 100
    wf_pct = total_oos_positive / len(all_oos) * 100 if all_oos else 0

    print(f"  Config stability: {stability_pct:.0f}% of windows profitable")
    print(f"  Walk-forward OOS: {wf_pct:.0f}% of windows profitable")
    print()

    if wf_pct >= 70 and stability_pct >= 70:
        print("  VERDICT: LIKELY REAL EDGE")
        print("  Strategy is profitable on unseen future data across multiple time periods.")
        print("  Overfitting risk: LOW")
    elif wf_pct >= 50 and stability_pct >= 50:
        print("  VERDICT: MODERATE EDGE (some overfitting possible)")
        print("  Strategy shows some predictive power but inconsistently.")
        print("  Consider: reduce parameter count, use simpler model.")
    else:
        print("  VERDICT: LIKELY OVERFIT")
        print("  Strategy fails to generalize to unseen time periods.")
        print("  Returns are likely an artifact of parameter optimization.")


if __name__ == "__main__":
    main()
