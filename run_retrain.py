#!/usr/bin/env python3
"""
Re-train strategy parameters with corrected execution model.

Key fixes over previous optimizer:
  - 12h pairs (AVAX/ATOM/SOL/POL) aggregate 3x 4h candles before analysis
  - One decision per candle (already correct in backtest runner)
  - Stop floor at per-pair static stop_loss_percent (in strategy code)

Training: Q4 2024 + Q1 2025 (~1,092 x 4h candles = 6 months)
Validation: Q1 2026 fresh (~540 x 4h candles = 3 months, held out)

Usage:
    python3 run_retrain.py                  # Full retrain, 50 trials/pair
    python3 run_retrain.py --trials 20      # Quick test
    python3 run_retrain.py --validate-only  # Just compare current vs new on Q1 2026
"""
import argparse
import csv
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.exchange.kraken_client import OHLC
from src.strategy.scalping_strategy import ScalpingStrategy, ScalpingConfig
from src.strategy.regime_detector import RegimeDetector, RegimeConfig, MarketRegime, REGIME_ADJUSTMENTS
from src.strategy.base_strategy import MarketData, Position, SignalType
from src.observability.logger import configure_logging

import yaml

PAIRS_12H = {"AVAX/USD", "ATOM/USD", "SOL/USD", "POL/USD"}
PAIRS_4H = {"BTC/USD", "ETH/USD", "DOT/USD", "NEAR/USD", "LINK/USD", "XRP/USD"}

PARAMETER_RANGES = {
    "take_profit_percent": {"low": 1.5, "high": 10.0, "step": 0.5},
    "stop_loss_percent": {"low": 0.5, "high": 4.0, "step": 0.5},
    "rsi_period": {"low": 5, "high": 14, "step": 1, "type": "int"},
    "rsi_oversold": {"low": 20, "high": 40, "step": 5, "type": "int"},
    "rsi_overbought": {"low": 60, "high": 80, "step": 5, "type": "int"},
    "bb_period": {"low": 10, "high": 30, "step": 5, "type": "int"},
    "bb_std_dev": {"low": 1.5, "high": 2.5, "step": 0.25},
    "vwap_threshold_percent": {"low": 0.1, "high": 1.0, "step": 0.1},
    "volume_spike_threshold": {"low": 1.0, "high": 2.5, "step": 0.25},
    "min_confirmations": {"low": 1, "high": 4, "step": 1, "type": "int"},
    "stoch_k_period": {"low": 5, "high": 21, "step": 2, "type": "int"},
    "stoch_oversold": {"low": 15, "high": 30, "step": 5, "type": "int"},
    "stoch_overbought": {"low": 70, "high": 85, "step": 5, "type": "int"},
    "atr_period": {"low": 10, "high": 20, "step": 2, "type": "int"},
    "atr_stop_multiplier": {"low": 1.0, "high": 3.0, "step": 0.25},
    "atr_tp_multiplier": {"low": 1.5, "high": 4.0, "step": 0.25},
    "short_min_confirmations": {"low": 2, "high": 4, "step": 1, "type": "int"},
}


def load_csv_candles(csv_path: Path) -> List[OHLC]:
    candles = []
    with open(csv_path) as f:
        for row in csv.reader(f):
            ts, o, h, l, c, v = row
            candles.append(OHLC(
                timestamp=datetime.fromtimestamp(int(ts), tz=timezone.utc),
                open=float(o), high=float(h), low=float(l),
                close=float(c), volume=float(v),
                vwap=(float(h) + float(l) + float(c)) / 3, count=0,
            ))
    return candles


def aggregate_to_12h(candles_4h: List[OHLC]) -> List[OHLC]:
    """Aggregate 4h candles into 12h candles (groups of 3)."""
    result = []
    i = 0
    while i + 2 < len(candles_4h):
        c1, c2, c3 = candles_4h[i], candles_4h[i + 1], candles_4h[i + 2]
        result.append(OHLC(
            timestamp=c1.timestamp,
            open=c1.open,
            high=max(c1.high, c2.high, c3.high),
            low=min(c1.low, c2.low, c3.low),
            close=c3.close,
            volume=c1.volume + c2.volume + c3.volume,
            vwap=(max(c1.high, c2.high, c3.high) + min(c1.low, c2.low, c3.low) + c3.close) / 3,
            count=0,
        ))
        i += 3
    return result


def load_pair_data(pair: str, data_dirs: List[Path], use_12h: bool) -> List[OHLC]:
    """Load and concatenate candle data from multiple directories."""
    pair_key = pair.replace("/", "_")
    all_candles = []
    for d in data_dirs:
        csv_path = d / f"{pair_key}_4h.csv"
        if csv_path.exists():
            candles = load_csv_candles(csv_path)
            all_candles.extend(candles)
    all_candles.sort(key=lambda c: c.timestamp)
    if use_12h:
        all_candles = aggregate_to_12h(all_candles)
    return all_candles


def run_backtest(candles: List[OHLC], pair: str, params: Dict[str, Any],
                 regime_candles: Optional[List[OHLC]] = None,
                 initial_capital: float = 10000.0) -> Dict[str, Any]:
    """Run backtest with given parameters. One decision per candle."""
    config = ScalpingConfig(
        take_profit_percent=params.get("take_profit_percent", 4.5),
        stop_loss_percent=params.get("stop_loss_percent", 2.0),
        rsi_period=params.get("rsi_period", 10),
        rsi_oversold=params.get("rsi_oversold", 28.0),
        rsi_overbought=params.get("rsi_overbought", 70.0),
        bb_period=params.get("bb_period", 20),
        bb_std_dev=params.get("bb_std_dev", 2.0),
        vwap_threshold_percent=params.get("vwap_threshold_percent", 0.5),
        volume_spike_threshold=params.get("volume_spike_threshold", 1.5),
        min_confirmations=params.get("min_confirmations", 3),
        fee_percent=0.16,
        stoch_k_period=params.get("stoch_k_period", 13),
        stoch_oversold=params.get("stoch_oversold", 23.0),
        stoch_overbought=params.get("stoch_overbought", 75.0),
        macd_fast=params.get("macd_fast", 12),
        macd_slow=params.get("macd_slow", 26),
        macd_signal=params.get("macd_signal", 9),
        atr_period=params.get("atr_period", 14),
        atr_stop_multiplier=params.get("atr_stop_multiplier", 2.5),
        atr_tp_multiplier=params.get("atr_tp_multiplier", 2.75),
        use_atr_stops=True,
        shorting_enabled=True,
        short_min_confirmations=params.get("short_min_confirmations", 3),
    )

    strategy = ScalpingStrategy(config)
    base_config = config
    regime_detector = RegimeDetector(RegimeConfig())
    current_regime = MarketRegime.UNKNOWN

    capital = initial_capital
    peak_capital = capital
    trades = []
    in_position = False
    entry_price = 0.0
    entry_time = None
    position_size_usd = 0.0
    position_side = "long"
    position_size_pct = 20.0

    lookback = max(config.bb_period, config.rsi_period,
                   config.macd_slow + config.macd_signal, config.atr_period) + 5

    for i in range(lookback, len(candles)):
        if i % 50 == 0 and i >= 200:
            r_source = regime_candles if regime_candles else candles
            all_so_far = r_source[:min(i + 1, len(r_source))]
            closes = [c.close for c in all_so_far]
            highs = [c.high for c in all_so_far]
            lows = [c.low for c in all_so_far]
            result = regime_detector.detect(closes, highs, lows)
            if result.regime != current_regime:
                current_regime = result.regime
                adj = REGIME_ADJUSTMENTS.get(current_regime, REGIME_ADJUSTMENTS[MarketRegime.UNKNOWN])
                tp_mult = adj.get('take_profit_multiplier', 1.0)
                sl_mult = adj.get('stop_loss_multiplier', 1.0)
                conf_offset = adj.get('min_confirmations_offset', 0)
                ema_enabled = adj.get('ema_filter_enabled', True)
                adjusted = ScalpingConfig(
                    take_profit_percent=base_config.take_profit_percent * tp_mult,
                    stop_loss_percent=base_config.stop_loss_percent * sl_mult,
                    rsi_period=base_config.rsi_period,
                    rsi_oversold=base_config.rsi_oversold,
                    rsi_overbought=base_config.rsi_overbought,
                    bb_period=base_config.bb_period,
                    bb_std_dev=base_config.bb_std_dev,
                    vwap_threshold_percent=base_config.vwap_threshold_percent,
                    volume_spike_threshold=base_config.volume_spike_threshold,
                    min_confirmations=max(1, base_config.min_confirmations + conf_offset),
                    fee_percent=base_config.fee_percent,
                    ema_filter_enabled=ema_enabled,
                    stoch_k_period=base_config.stoch_k_period,
                    stoch_d_period=base_config.stoch_d_period,
                    stoch_oversold=base_config.stoch_oversold,
                    stoch_overbought=base_config.stoch_overbought,
                    macd_fast=base_config.macd_fast,
                    macd_slow=base_config.macd_slow,
                    macd_signal=base_config.macd_signal,
                    atr_period=base_config.atr_period,
                    atr_stop_multiplier=base_config.atr_stop_multiplier,
                    atr_tp_multiplier=base_config.atr_tp_multiplier,
                    use_atr_stops=base_config.use_atr_stops,
                    shorting_enabled=base_config.shorting_enabled,
                    short_min_confirmations=base_config.short_min_confirmations,
                )
                strategy = ScalpingStrategy(adjusted)

        window = candles[i - lookback:i + 1]
        current = candles[i]
        market_data = MarketData(
            pair=pair, ohlc=window,
            prices=[c.close for c in window],
            volumes=[c.volume for c in window],
            ticker=None,
        )

        if in_position:
            position = Position(
                pair=pair, side=position_side,
                entry_price=entry_price, current_price=current.close,
                size=position_size_usd / entry_price, entry_time=entry_time,
            )
            signal = strategy.analyze(market_data, position)
            should_exit = False
            if position_side == "long":
                should_exit = signal.signal_type in (SignalType.SELL, SignalType.CLOSE_LONG)
            elif position_side == "short":
                should_exit = signal.signal_type == SignalType.CLOSE_SHORT

            if should_exit:
                exit_price = current.close
                if position_side == "short":
                    gross_pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                else:
                    gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                net_pnl_pct = gross_pnl_pct - (0.16 * 2)
                pnl_usd = position_size_usd * (net_pnl_pct / 100)
                capital += pnl_usd
                peak_capital = max(peak_capital, capital)
                trades.append({
                    "pnl_percent": net_pnl_pct, "pnl_usd": pnl_usd,
                    "win": pnl_usd > 0, "side": position_side,
                })
                in_position = False
        else:
            signal = strategy.analyze(market_data, None)

            # Signal-strength sizing (matches production)
            strength_scale = 0.5 + 0.5 * getattr(signal, 'strength', 0.5)

            if signal.signal_type == SignalType.BUY:
                entry_price = current.close
                entry_time = current.timestamp
                position_size_usd = capital * (position_size_pct / 100) * strength_scale
                position_side = "long"
                in_position = True
            elif signal.signal_type == SignalType.SELL_SHORT:
                entry_price = current.close
                entry_time = current.timestamp
                position_size_usd = capital * (position_size_pct / 100) * strength_scale
                position_side = "short"
                in_position = True

    return _calc_metrics(trades, capital, initial_capital)


def _calc_metrics(trades, final_capital, initial_capital):
    if not trades:
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_return": 0, "total_pnl_pct": 0, "avg_pnl_pct": 0,
                "profit_factor": 0, "sharpe": 0, "max_dd_pct": 0, "score": float("-inf"),
                "long_trades": 0, "short_trades": 0}

    wins = sum(1 for t in trades if t["win"])
    losses = len(trades) - wins
    wr = wins / len(trades)
    total_return = (final_capital - initial_capital) / initial_capital
    total_pnl_pct = sum(t["pnl_percent"] for t in trades)

    gross_profit = sum(t["pnl_usd"] for t in trades if t["pnl_usd"] > 0)
    gross_loss = abs(sum(t["pnl_usd"] for t in trades if t["pnl_usd"] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 5.0
    pf = min(pf, 5.0)

    returns = [t["pnl_percent"] for t in trades]
    avg_ret = sum(returns) / len(returns)
    if len(returns) > 1:
        var = sum((r - avg_ret)**2 for r in returns) / (len(returns) - 1)
        std = var ** 0.5
        sharpe = avg_ret / std if std > 0.001 else 0
    else:
        sharpe = 0
    sharpe = max(min(sharpe, 10), -10)
    if len(trades) < 20:
        sharpe *= len(trades) / 20.0

    peak = initial_capital
    running = initial_capital
    max_dd = 0
    for t in trades:
        running += t["pnl_usd"]
        peak = max(peak, running)
        dd = (peak - running) / peak * 100
        max_dd = max(max_dd, dd)

    # Trade count penalty
    trade_penalty = 0
    if len(trades) < 15:
        trade_penalty = (15 - len(trades)) * 5.0
    if len(trades) < 5:
        trade_penalty += 30.0

    score = (
        total_return * 100 * 0.35 +
        sharpe * 10 * 0.2 +
        pf * 5 * 0.15 +
        wr * 100 * 0.15 -
        max_dd * 0.15 -
        trade_penalty
    )

    return {
        "total_trades": len(trades), "wins": wins, "losses": losses,
        "win_rate": wr * 100, "total_return": total_return,
        "total_pnl_pct": total_pnl_pct, "avg_pnl_pct": avg_ret,
        "profit_factor": pf, "sharpe": sharpe, "max_dd_pct": max_dd,
        "score": score,
        "long_trades": sum(1 for t in trades if t["side"] == "long"),
        "short_trades": sum(1 for t in trades if t["side"] == "short"),
    }


def get_current_params(config: dict, pair: str) -> Dict[str, Any]:
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


def sample_params(trial) -> Dict[str, Any]:
    params = {}
    for name, r in PARAMETER_RANGES.items():
        if r.get("type") == "int":
            params[name] = trial.suggest_int(name, int(r["low"]), int(r["high"]), step=int(r.get("step", 1)))
        else:
            params[name] = trial.suggest_float(name, r["low"], r["high"], step=r.get("step"))
    return params


def optimize_pair(pair: str, train_candles: List[OHLC], val_candles: List[OHLC],
                  btc_train: Optional[List[OHLC]], btc_val: Optional[List[OHLC]],
                  n_trials: int = 50) -> Dict[str, Any]:
    """Optimize a single pair. Train/val split for anti-overfit."""
    def objective(trial):
        params = sample_params(trial)
        # Use 70/30 split within training data for inner validation
        inner_split = int(len(train_candles) * 0.7)
        inner_train = train_candles[:inner_split]
        inner_val = train_candles[inner_split:]
        btc_it = btc_train[:inner_split] if btc_train else None
        btc_iv = btc_train[inner_split:] if btc_train else None

        m_train = run_backtest(inner_train, pair, params, regime_candles=btc_it)
        m_val = run_backtest(inner_val, pair, params, regime_candles=btc_iv)

        return 0.4 * m_train["score"] + 0.6 * m_val["score"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    # Evaluate on held-out validation set
    val_metrics = run_backtest(val_candles, pair, best_params, regime_candles=btc_val)
    train_metrics = run_backtest(train_candles, pair, best_params, regime_candles=btc_train)

    return {
        "params": best_params,
        "train": train_metrics,
        "val": val_metrics,
        "study_score": study.best_value,
    }


def main():
    configure_logging(level="ERROR", format_type="json")

    parser = argparse.ArgumentParser(description="Re-train strategy parameters")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per pair")
    parser.add_argument("--validate-only", action="store_true", help="Just evaluate current params on Q1 2026")
    parser.add_argument("--pairs", type=str, default=None, help="Comma-separated pairs to optimize")
    args = parser.parse_args()

    base = Path("/home/user/kraken-momentum-trader")
    train_dirs = [base / "data/q4_2024", base / "data/q1_2025"]
    val_dir = base / "data/q1_2026_fresh"

    with open(base / "config/scalping.yaml") as f:
        config = yaml.safe_load(f)

    all_pairs = config.get("pairs", [])
    pairs = args.pairs.split(",") if args.pairs else all_pairs

    print(f"\n{'='*80}")
    print(f"  PARAMETER RE-TRAINING (corrected execution model)")
    print(f"{'='*80}")
    print(f"  Training: Q4 2024 + Q1 2025  |  Validation: Q1 2026 (held out)")
    print(f"  Trials: {args.trials} per pair  |  12h aggregation for {', '.join(sorted(PAIRS_12H))}")
    print(f"  Pairs: {', '.join(pairs)}")
    print(f"{'='*80}\n")

    # Load BTC data for regime detection
    use_12h_btc = "BTC/USD" in PAIRS_12H
    btc_train = load_pair_data("BTC/USD", train_dirs, use_12h_btc)
    btc_val = load_pair_data("BTC/USD", [val_dir], use_12h_btc)
    print(f"  BTC regime ref: {len(btc_train)} train, {len(btc_val)} val candles\n")

    results = {}

    for pair in pairs:
        use_12h = pair in PAIRS_12H
        interval_label = "12h" if use_12h else "4h"
        train_candles = load_pair_data(pair, train_dirs, use_12h)
        val_candles = load_pair_data(pair, [val_dir], use_12h)

        print(f"  {pair} ({interval_label}): {len(train_candles)} train, {len(val_candles)} val candles")

        # Evaluate CURRENT params on validation set
        current_params = get_current_params(config, pair)
        current_val = run_backtest(val_candles, pair, current_params, regime_candles=btc_val)
        current_train = run_backtest(train_candles, pair, current_params, regime_candles=btc_train)

        if args.validate_only:
            results[pair] = {
                "current_params": current_params,
                "current_train": current_train,
                "current_val": current_val,
                "new_params": None,
                "new_train": None,
                "new_val": None,
            }
            continue

        # Optimize
        print(f"    Optimizing ({args.trials} trials)...")
        opt = optimize_pair(
            pair, train_candles, val_candles, btc_train, btc_val,
            n_trials=args.trials,
        )

        results[pair] = {
            "current_params": current_params,
            "current_train": current_train,
            "current_val": current_val,
            "new_params": opt["params"],
            "new_train": opt["train"],
            "new_val": opt["val"],
        }

        cr = current_val
        nr = opt["val"]
        print(f"    Current  → Val: {cr['total_return']*100:+.2f}%  ({cr['total_trades']} trades, {cr['win_rate']:.0f}% WR, PF {cr['profit_factor']:.2f})")
        print(f"    New      → Val: {nr['total_return']*100:+.2f}%  ({nr['total_trades']} trades, {nr['win_rate']:.0f}% WR, PF {nr['profit_factor']:.2f})")
        delta = (nr['total_return'] - cr['total_return']) * 100
        print(f"    Delta: {delta:+.2f}%\n")

    # Summary
    print(f"\n{'='*80}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Pair':>10} | {'Curr Train':>11} | {'Curr Val':>10} | {'New Train':>10} | {'New Val':>10} | {'Delta':>8}")
    print(f"  {'-'*10}-+-{'-'*11}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    total_curr_val = 0
    total_new_val = 0

    for pair in pairs:
        r = results[pair]
        ct = r["current_train"]["total_return"] * 100
        cv = r["current_val"]["total_return"] * 100
        total_curr_val += cv

        if r["new_val"] is not None:
            nt = r["new_train"]["total_return"] * 100
            nv = r["new_val"]["total_return"] * 100
            total_new_val += nv
            delta = nv - cv
            print(f"  {pair:>10} | {ct:>+10.2f}% | {cv:>+9.2f}% | {nt:>+9.2f}% | {nv:>+9.2f}% | {delta:>+7.2f}%")
        else:
            total_new_val += cv
            print(f"  {pair:>10} | {ct:>+10.2f}% | {cv:>+9.2f}% |     -     |     -     |    -")

    print(f"  {'-'*10}-+-{'-'*11}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    print(f"  {'TOTAL':>10} | {'':>11} | {total_curr_val:>+9.2f}% | {'':>10} | {total_new_val:>+9.2f}% | {total_new_val-total_curr_val:>+7.2f}%")

    # Which pairs improved?
    improved = []
    regressed = []
    for pair in pairs:
        r = results[pair]
        if r["new_val"] is None:
            continue
        delta = (r["new_val"]["total_return"] - r["current_val"]["total_return"]) * 100
        if delta > 0.5:
            improved.append((pair, delta))
        elif delta < -0.5:
            regressed.append((pair, delta))

    if improved:
        print(f"\n  Improved: {', '.join(f'{p} ({d:+.1f}%)' for p, d in improved)}")
    if regressed:
        print(f"  Regressed: {', '.join(f'{p} ({d:+.1f}%)' for p, d in regressed)}")

    # Export new params for improved pairs
    if not args.validate_only and improved:
        print(f"\n  --- RECOMMENDED CONFIG UPDATES ---")
        for pair in pairs:
            r = results[pair]
            if r["new_val"] is None:
                continue
            delta = (r["new_val"]["total_return"] - r["current_val"]["total_return"]) * 100
            if delta > 0.5:
                print(f"\n  {pair}:")
                p = r["new_params"]
                print(f"    take_profit_percent: {p.get('take_profit_percent', 4.5)}")
                print(f"    stop_loss_percent: {p.get('stop_loss_percent', 2.0)}")
                print(f"    rsi_period: {p.get('rsi_period', 10)}")
                print(f"    rsi_oversold: {p.get('rsi_oversold', 28)}")
                print(f"    rsi_overbought: {p.get('rsi_overbought', 70)}")
                print(f"    bb_period: {p.get('bb_period', 20)}")
                print(f"    bb_std_dev: {p.get('bb_std_dev', 2.0)}")
                print(f"    vwap_threshold_percent: {p.get('vwap_threshold_percent', 0.5)}")
                print(f"    volume_spike_threshold: {p.get('volume_spike_threshold', 1.5)}")
                print(f"    min_confirmations: {p.get('min_confirmations', 3)}")
                print(f"    stoch_k_period: {p.get('stoch_k_period', 13)}")
                print(f"    stoch_oversold: {p.get('stoch_oversold', 23)}")
                print(f"    stoch_overbought: {p.get('stoch_overbought', 75)}")
                print(f"    atr_period: {p.get('atr_period', 14)}")
                print(f"    atr_stop_multiplier: {p.get('atr_stop_multiplier', 2.5)}")
                print(f"    atr_tp_multiplier: {p.get('atr_tp_multiplier', 2.75)}")
                print(f"    short_min_confirmations: {p.get('short_min_confirmations', 3)}")

    # Save full results
    output_path = base / "data/retrain_results.yaml"
    export = {}
    for pair in pairs:
        r = results[pair]
        entry = {
            "current_val_return": r["current_val"]["total_return"],
            "current_val_trades": r["current_val"]["total_trades"],
            "current_val_wr": r["current_val"]["win_rate"],
        }
        if r["new_val"] is not None:
            entry["new_params"] = r["new_params"]
            entry["new_val_return"] = r["new_val"]["total_return"]
            entry["new_val_trades"] = r["new_val"]["total_trades"]
            entry["new_val_wr"] = r["new_val"]["win_rate"]
        export[pair] = entry
    with open(output_path, "w") as f:
        yaml.dump(export, f, default_flow_style=False)
    print(f"\n  Full results saved to {output_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
