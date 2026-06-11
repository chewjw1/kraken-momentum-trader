#!/usr/bin/env python3
"""
Focused re-train: only tune the 6 most impactful parameters.

The full 16-parameter optimization overfits (too many degrees of freedom).
This version locks indicator periods at current values and only tunes:
  1. take_profit_percent
  2. stop_loss_percent
  3. atr_stop_multiplier
  4. atr_tp_multiplier
  5. min_confirmations
  6. short_min_confirmations

Training: Q4 2024 + Q1 2025  |  Validation: Q1 2026 fresh
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

# Only tune these 6 parameters (the "risk/reward" knobs)
TUNABLE_RANGES = {
    "take_profit_percent": {"low": 1.5, "high": 10.0, "step": 0.5},
    "stop_loss_percent": {"low": 1.0, "high": 4.0, "step": 0.5},
    "atr_stop_multiplier": {"low": 1.0, "high": 3.0, "step": 0.25},
    "atr_tp_multiplier": {"low": 1.5, "high": 4.0, "step": 0.25},
    "min_confirmations": {"low": 2, "high": 4, "step": 1, "type": "int"},
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
    pair_key = pair.replace("/", "_")
    all_candles = []
    for d in data_dirs:
        csv_path = d / f"{pair_key}_4h.csv"
        if csv_path.exists():
            all_candles.extend(load_csv_candles(csv_path))
    all_candles.sort(key=lambda c: c.timestamp)
    if use_12h:
        all_candles = aggregate_to_12h(all_candles)
    return all_candles


def run_backtest(candles: List[OHLC], pair: str, params: Dict[str, Any],
                 regime_candles: Optional[List[OHLC]] = None,
                 initial_capital: float = 10000.0) -> Dict[str, Any]:
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
        macd_fast=12, macd_slow=26, macd_signal=9,
        atr_period=params.get("atr_period", 14),
        atr_stop_multiplier=params.get("atr_stop_multiplier", 2.5),
        atr_tp_multiplier=params.get("atr_tp_multiplier", 2.75),
        use_atr_stops=True, shorting_enabled=True,
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
            strength_scale = 0.5 + 0.5 * getattr(signal, 'strength', 0.5)

            if signal.signal_type == SignalType.BUY:
                entry_price = current.close
                entry_time = current.timestamp
                position_size_usd = capital * 0.20 * strength_scale
                position_side = "long"
                in_position = True
            elif signal.signal_type == SignalType.SELL_SHORT:
                entry_price = current.close
                entry_time = current.timestamp
                position_size_usd = capital * 0.20 * strength_scale
                position_side = "short"
                in_position = True

    return _calc_metrics(trades, capital, initial_capital)


def _calc_metrics(trades, final_capital, initial_capital):
    if not trades:
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_return": 0, "total_pnl_pct": 0, "profit_factor": 0,
                "sharpe": 0, "max_dd_pct": 0, "score": float("-inf"),
                "long_trades": 0, "short_trades": 0}

    wins = sum(1 for t in trades if t["win"])
    wr = wins / len(trades)
    total_return = (final_capital - initial_capital) / initial_capital

    gross_profit = sum(t["pnl_usd"] for t in trades if t["pnl_usd"] > 0)
    gross_loss = abs(sum(t["pnl_usd"] for t in trades if t["pnl_usd"] < 0))
    pf = min(gross_profit / gross_loss if gross_loss > 0 else 5.0, 5.0)

    returns = [t["pnl_percent"] for t in trades]
    avg_ret = sum(returns) / len(returns)
    if len(returns) > 1:
        var = sum((r - avg_ret)**2 for r in returns) / (len(returns) - 1)
        sharpe = avg_ret / (var ** 0.5) if var > 0.00001 else 0
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

    trade_penalty = max(0, (15 - len(trades)) * 5.0) if len(trades) < 15 else 0
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
        "total_trades": len(trades), "wins": wins, "losses": len(trades) - wins,
        "win_rate": wr * 100, "total_return": total_return,
        "total_pnl_pct": sum(returns), "profit_factor": pf,
        "sharpe": sharpe, "max_dd_pct": max_dd, "score": score,
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


def main():
    configure_logging(level="ERROR", format_type="json")

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--pairs", type=str, default=None)
    parser.add_argument("--train-dirs", type=str, default="data/q4_2024,data/q1_2025",
                        help="Comma-separated training data directories")
    parser.add_argument("--val-dir", type=str, default="data/q1_2026_fresh",
                        help="Held-out validation data directory")
    args = parser.parse_args()

    base = Path("/home/user/kraken-momentum-trader")
    train_dirs = [base / d for d in args.train_dirs.split(",")]
    val_dir = base / args.val_dir

    with open(base / "config/scalping.yaml") as f:
        config = yaml.safe_load(f)

    all_pairs = config.get("pairs", [])
    pairs = args.pairs.split(",") if args.pairs else all_pairs

    print(f"\n{'='*80}")
    print(f"  FOCUSED RE-TRAIN (6 params only: TP, SL, ATR mults, confirmations)")
    print(f"{'='*80}")
    print(f"  Training: Q4 2024 + Q1 2025  |  Validation: Q1 2026 (held out)")
    print(f"  Trials: {args.trials} per pair  |  12h aggregation for 12h pairs")
    print(f"{'='*80}\n")

    btc_train = load_pair_data("BTC/USD", train_dirs, False)
    btc_val = load_pair_data("BTC/USD", [val_dir], False)

    results = {}

    for pair in pairs:
        use_12h = pair in PAIRS_12H
        train_candles = load_pair_data(pair, train_dirs, use_12h)
        val_candles = load_pair_data(pair, [val_dir], use_12h)
        label = "12h" if use_12h else "4h"

        # Current params baseline
        current_params = get_current_params(config, pair)
        current_val = run_backtest(val_candles, pair, current_params, regime_candles=btc_val)
        current_train = run_backtest(train_candles, pair, current_params, regime_candles=btc_train)

        # Lock indicator params, only tune risk/reward
        locked = {k: v for k, v in current_params.items() if k not in TUNABLE_RANGES}

        def make_objective(p=pair, tc=train_candles, btc_t=btc_train, lk=locked):
            def objective(trial):
                params = dict(lk)
                for name, r in TUNABLE_RANGES.items():
                    if r.get("type") == "int":
                        params[name] = trial.suggest_int(name, int(r["low"]), int(r["high"]), step=int(r.get("step", 1)))
                    else:
                        params[name] = trial.suggest_float(name, r["low"], r["high"], step=r.get("step"))

                # Inner 70/30 split for anti-overfit
                split = int(len(tc) * 0.7)
                m_train = run_backtest(tc[:split], p, params, regime_candles=btc_t[:split] if btc_t else None)
                m_val = run_backtest(tc[split:], p, params, regime_candles=btc_t[split:] if btc_t else None)

                # Heavier validation weight
                return 0.3 * m_train["score"] + 0.7 * m_val["score"]
            return objective

        print(f"  {pair} ({label}): {len(train_candles)} train, {len(val_candles)} val  ", end="", flush=True)
        study = optuna.create_study(direction="maximize")
        study.optimize(make_objective(), n_trials=args.trials, show_progress_bar=False)

        # Build new params = locked indicators + tuned risk/reward
        new_params = dict(locked)
        new_params.update(study.best_params)

        new_val = run_backtest(val_candles, pair, new_params, regime_candles=btc_val)
        new_train = run_backtest(train_candles, pair, new_params, regime_candles=btc_train)

        delta = (new_val["total_return"] - current_val["total_return"]) * 100
        marker = "BETTER" if delta > 0.5 else ("WORSE" if delta < -0.5 else "~SAME")

        print(f"curr={current_val['total_return']*100:+.2f}%  new={new_val['total_return']*100:+.2f}%  Δ={delta:+.2f}% [{marker}]")

        results[pair] = {
            "current_params": current_params,
            "current_val": current_val,
            "current_train": current_train,
            "new_params": new_params,
            "new_val": new_val,
            "new_train": new_train,
            "delta": delta,
        }

    # Summary
    print(f"\n{'='*80}")
    print(f"  RESULTS COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Pair':>10} | {'CurrTrain':>10} | {'CurrVal':>10} | {'NewTrain':>10} | {'NewVal':>10} | {'Delta':>8} | Action")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*12}")

    total_curr = 0
    total_new = 0
    adopt_pairs = []

    for pair in pairs:
        r = results[pair]
        ct = r["current_train"]["total_return"] * 100
        cv = r["current_val"]["total_return"] * 100
        nt = r["new_train"]["total_return"] * 100
        nv = r["new_val"]["total_return"] * 100
        d = r["delta"]
        total_curr += cv
        total_new += nv

        # Adopt new params if: better on val AND not dramatically overfit (train improvement < 3x val improvement)
        adopt = d > 0.5
        if adopt and nt - ct > 0 and nv - cv > 0:
            train_delta = nt - ct
            val_delta = nv - cv
            if train_delta > val_delta * 3:
                adopt = False

        action = "ADOPT" if adopt else "KEEP"
        if adopt:
            adopt_pairs.append(pair)
        print(f"  {pair:>10} | {ct:>+9.2f}% | {cv:>+9.2f}% | {nt:>+9.2f}% | {nv:>+9.2f}% | {d:>+7.2f}% | {action}")

    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*12}")
    total_adopted = sum(results[p]["new_val"]["total_return"] * 100 if p in adopt_pairs else results[p]["current_val"]["total_return"] * 100 for p in pairs)
    print(f"  {'TOTAL':>10} | {'':>10} | {total_curr:>+9.2f}% | {'':>10} | {total_new:>+9.2f}% | {total_new-total_curr:>+7.2f}%")
    print(f"  {'SELECTIVE':>10} | {'':>10} | {'':>10} | {'':>10} | {total_adopted:>+9.2f}% | {total_adopted-total_curr:>+7.2f}% | (adopt only improved)")

    # Print new params for adopted pairs
    if adopt_pairs:
        print(f"\n  ADOPTED NEW PARAMS FOR: {', '.join(adopt_pairs)}")
        for pair in adopt_pairs:
            r = results[pair]
            p = r["new_params"]
            print(f"\n  {pair}:")
            for key in TUNABLE_RANGES:
                old = r["current_params"][key]
                new = p[key]
                changed = " <--" if old != new else ""
                print(f"    {key}: {old} -> {new}{changed}")

    print(f"\n{'='*80}\n")

    # Save results
    with open(base / "data/retrain_focused_results.yaml", "w") as f:
        export = {}
        for pair in pairs:
            r = results[pair]
            export[pair] = {
                "adopt": pair in adopt_pairs,
                "delta_pct": r["delta"],
                "current_val_return": r["current_val"]["total_return"],
                "new_val_return": r["new_val"]["total_return"],
                "new_params": {k: r["new_params"][k] for k in TUNABLE_RANGES},
            }
        yaml.dump(export, f, default_flow_style=False)


if __name__ == "__main__":
    main()
