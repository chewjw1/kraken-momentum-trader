#!/usr/bin/env python3
"""
Multi-timeframe realistic backtest.

Generates signals on 4h candles (where parameters are optimized),
but simulates execution on 1-minute candles for realistic:
  - Intra-bar stop-loss triggers (flash wicks)
  - Intra-bar take-profit triggers
  - Slippage modeling
  - Real spread costs

This bridges the gap between 4h backtest results and real-world execution.
"""

import yaml
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict
from src.backtest.kraken_csv_provider import KrakenCSVProvider
from src.strategy.scalping_strategy import ScalpingStrategy, ScalpingConfig
from src.strategy.base_strategy import MarketData, Position
from src.exchange.kraken_client import OHLC


@dataclass
class MTFTrade:
    pair: str
    side: str  # "long" or "short"
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    pnl_pct: float
    exit_reason: str
    hold_minutes: int
    signal_candle_time: datetime  # 4h candle that triggered entry


def aggregate_to_4h(candles_1m: List[OHLC], interval_min: int = 240) -> List[OHLC]:
    """Aggregate 1-minute candles into 4h candles."""
    if not candles_1m:
        return []

    result = []
    bucket_start = None
    o = h = l = c = v = 0.0
    count = 0

    for candle in candles_1m:
        ts = candle.timestamp
        # Bucket boundary: floor to nearest interval
        bucket = ts.replace(
            hour=(ts.hour // (interval_min // 60)) * (interval_min // 60),
            minute=0, second=0, microsecond=0
        )

        if bucket_start is None:
            bucket_start = bucket
            o = candle.open
            h = candle.high
            l = candle.low
            c = candle.close
            v = candle.volume
            count = getattr(candle, 'count', 0) or 0
        elif bucket == bucket_start:
            h = max(h, candle.high)
            l = min(l, candle.low)
            c = candle.close
            v += candle.volume
            count += getattr(candle, 'count', 0) or 0
        else:
            result.append(OHLC(
                timestamp=bucket_start,
                open=o, high=h, low=l, close=c,
                vwap=c, volume=v, count=count
            ))
            bucket_start = bucket
            o = candle.open
            h = candle.high
            l = candle.low
            c = candle.close
            v = candle.volume
            count = getattr(candle, 'count', 0) or 0

    # Final bucket
    if bucket_start is not None:
        result.append(OHLC(
            timestamp=bucket_start,
            open=o, high=h, low=l, close=c,
            vwap=c, volume=v, count=count
        ))

    return result


def run_mtf_backtest(
    pair: str,
    candles_1m: List[OHLC],
    scalping_config: ScalpingConfig,
    initial_capital: float = 10000.0,
    position_size_pct: float = 0.20,
    fee_pct: float = 0.0016,
    slippage_pct: float = 0.02,
    lookback_4h: int = 50,
    signal_interval: int = 240,
) -> Dict:
    """
    Multi-timeframe backtest.

    1. Aggregate 1m -> signal_interval candles (4h or 12h)
    2. Generate signals on aggregated candles
    3. On signal, enter at next 1m candle open (with slippage)
    4. Check stops/TPs every 1-minute candle
    """
    # Step 1: Aggregate to signal interval
    candles_4h = aggregate_to_4h(candles_1m, interval_min=signal_interval)
    hours = signal_interval // 60

    if len(candles_4h) < lookback_4h + 10:
        return {"pair": pair, "error": f"Only {len(candles_4h)} {hours}h candles"}

    # Build index: signal candle timestamp -> slice of 1m candles
    candle_1m_by_4h = {}
    c4h_idx = 0
    for i, c1m in enumerate(candles_1m):
        ts = c1m.timestamp
        bucket = ts.replace(
            hour=(ts.hour // hours) * hours,
            minute=0, second=0, microsecond=0
        )
        if bucket not in candle_1m_by_4h:
            candle_1m_by_4h[bucket] = []
        candle_1m_by_4h[bucket].append(i)

    strategy = ScalpingStrategy(scalping_config)
    trades: List[MTFTrade] = []
    capital = initial_capital
    peak_capital = capital

    # Position state
    in_position = False
    pos_side = ""
    entry_price = 0.0
    entry_time = None
    signal_candle_time = None
    pos_size_usd = 0.0

    # Dynamic stop/TP from ATR
    dynamic_sl = scalping_config.stop_loss_percent
    dynamic_tp = scalping_config.take_profit_percent

    # Trailing stop state
    trailing_stop_price = 0.0  # Current trailing stop level
    best_price = 0.0  # Best price seen since entry (high for long, low for short)
    breakeven_triggered = False

    # Iterate 4h candles for signals
    for i4h in range(lookback_4h, len(candles_4h)):
        candle_4h = candles_4h[i4h]
        lookback = candles_4h[i4h - lookback_4h:i4h + 1]
        market_data = MarketData(
            pair=pair,
            ohlc=lookback,
            prices=[c.close for c in lookback],
            volumes=[c.volume for c in lookback],
            ticker=None
        )

        # Get 1-minute candles for this 4h period
        bucket_ts = candle_4h.timestamp
        minute_indices = candle_1m_by_4h.get(bucket_ts, [])
        if not minute_indices:
            continue

        if in_position:
            # Check stops on every 1-minute candle in this signal window
            for midx in minute_indices:
                c1m = candles_1m[midx]

                if pos_side == "long":
                    tp_price = entry_price * (1 + dynamic_tp / 100)

                    # Update best price seen
                    if c1m.high > best_price:
                        best_price = c1m.high

                    # Trailing stop logic
                    unrealized_pct = ((best_price - entry_price) / entry_price) * 100
                    tp_progress = unrealized_pct / dynamic_tp if dynamic_tp > 0 else 0

                    if tp_progress >= 0.6:
                        # Trail at 50% of best unrealized profit
                        trail_price = entry_price * (1 + unrealized_pct * 0.5 / 100)
                        trailing_stop_price = max(trailing_stop_price, trail_price)
                    elif tp_progress >= 0.4 and not breakeven_triggered:
                        # Move stop to breakeven (entry + fees)
                        trailing_stop_price = entry_price * (1 + fee_pct * 100 * 2 / 100 + 0.05)
                        breakeven_triggered = True

                    # Effective stop: max of original SL and trailing stop
                    original_sl = entry_price * (1 - dynamic_sl / 100)
                    effective_sl = max(original_sl, trailing_stop_price)

                    if c1m.low <= effective_sl:
                        exit_price = effective_sl * (1 - slippage_pct / 100)
                        gross_pnl = ((exit_price - entry_price) / entry_price) * 100
                        net_pnl = gross_pnl - fee_pct * 100 * 2
                        pnl_usd = pos_size_usd * (net_pnl / 100)
                        capital += pnl_usd
                        hold_min = int((c1m.timestamp - entry_time).total_seconds() / 60)
                        reason = "trailing_stop_1m" if trailing_stop_price > original_sl else "stop_loss_1m"
                        trades.append(MTFTrade(pair, "long", entry_time, c1m.timestamp,
                                               entry_price, exit_price, net_pnl,
                                               reason, hold_min, signal_candle_time))
                        in_position = False
                        break

                    if c1m.high >= tp_price:
                        exit_price = tp_price * (1 - slippage_pct / 100)
                        gross_pnl = ((exit_price - entry_price) / entry_price) * 100
                        net_pnl = gross_pnl - fee_pct * 100 * 2
                        pnl_usd = pos_size_usd * (net_pnl / 100)
                        capital += pnl_usd
                        hold_min = int((c1m.timestamp - entry_time).total_seconds() / 60)
                        trades.append(MTFTrade(pair, "long", entry_time, c1m.timestamp,
                                               entry_price, exit_price, net_pnl,
                                               "take_profit_1m", hold_min, signal_candle_time))
                        in_position = False
                        break

                else:  # short
                    tp_price = entry_price * (1 - dynamic_tp / 100)

                    # Update best price (lowest for shorts)
                    if best_price == 0.0 or c1m.low < best_price:
                        best_price = c1m.low

                    # Trailing stop logic for shorts
                    unrealized_pct = ((entry_price - best_price) / entry_price) * 100
                    tp_progress = unrealized_pct / dynamic_tp if dynamic_tp > 0 else 0

                    if tp_progress >= 0.6:
                        trail_price = entry_price * (1 - unrealized_pct * 0.5 / 100)
                        if trailing_stop_price == 0.0 or trail_price < trailing_stop_price:
                            trailing_stop_price = trail_price
                    elif tp_progress >= 0.4 and not breakeven_triggered:
                        trailing_stop_price = entry_price * (1 - fee_pct * 100 * 2 / 100 - 0.05)
                        breakeven_triggered = True

                    original_sl = entry_price * (1 + dynamic_sl / 100)
                    effective_sl = min(original_sl, trailing_stop_price) if trailing_stop_price > 0 else original_sl

                    if c1m.high >= effective_sl:
                        exit_price = effective_sl * (1 + slippage_pct / 100)
                        gross_pnl = ((entry_price - exit_price) / entry_price) * 100
                        net_pnl = gross_pnl - fee_pct * 100 * 2
                        pnl_usd = pos_size_usd * (net_pnl / 100)
                        capital += pnl_usd
                        hold_min = int((c1m.timestamp - entry_time).total_seconds() / 60)
                        reason = "trailing_stop_1m" if trailing_stop_price > 0 and trailing_stop_price < original_sl else "stop_loss_1m"
                        trades.append(MTFTrade(pair, "short", entry_time, c1m.timestamp,
                                               entry_price, exit_price, net_pnl,
                                               reason, hold_min, signal_candle_time))
                        in_position = False
                        break

                    if c1m.low <= tp_price:
                        exit_price = tp_price * (1 + slippage_pct / 100)
                        gross_pnl = ((entry_price - exit_price) / entry_price) * 100
                        net_pnl = gross_pnl - fee_pct * 100 * 2
                        pnl_usd = pos_size_usd * (net_pnl / 100)
                        capital += pnl_usd
                        hold_min = int((c1m.timestamp - entry_time).total_seconds() / 60)
                        trades.append(MTFTrade(pair, "short", entry_time, c1m.timestamp,
                                               entry_price, exit_price, net_pnl,
                                               "take_profit_1m", hold_min, signal_candle_time))
                        in_position = False
                        break

            if in_position:
                # Still in position — check signal-based exit on 4h
                position = Position(
                    pair=pair, side=pos_side,
                    entry_price=entry_price, current_price=candle_4h.close,
                    size=pos_size_usd / entry_price, entry_time=entry_time
                )
                signal = strategy.analyze(market_data, position)
                exit_signals = ("sell", "close_long") if pos_side == "long" else ("close_short",)
                if signal.signal_type.value in exit_signals:
                    # Exit at last 1m close in this 4h window
                    last_1m = candles_1m[minute_indices[-1]]
                    if pos_side == "long":
                        exit_price = last_1m.close * (1 - slippage_pct / 100)
                        gross_pnl = ((exit_price - entry_price) / entry_price) * 100
                    else:
                        exit_price = last_1m.close * (1 + slippage_pct / 100)
                        gross_pnl = ((entry_price - exit_price) / entry_price) * 100
                    net_pnl = gross_pnl - fee_pct * 100 * 2
                    pnl_usd = pos_size_usd * (net_pnl / 100)
                    capital += pnl_usd
                    hold_min = int((last_1m.timestamp - entry_time).total_seconds() / 60)
                    trades.append(MTFTrade(pair, pos_side, entry_time, last_1m.timestamp,
                                           entry_price, exit_price, net_pnl,
                                           f"signal_exit_{signal.reason}", hold_min, signal_candle_time))
                    in_position = False

        else:
            # Check for new entry signal on 4h candle
            signal = strategy.analyze(market_data, None)

            if signal.signal_type.value in ("buy", "sell_short"):
                # Enter at first 1m candle of NEXT 4h window
                next_4h_idx = i4h + 1
                if next_4h_idx < len(candles_4h):
                    next_bucket = candles_4h[next_4h_idx].timestamp
                    next_indices = candle_1m_by_4h.get(next_bucket, [])
                    if next_indices:
                        entry_1m = candles_1m[next_indices[0]]
                        pos_side = "long" if signal.signal_type.value == "buy" else "short"

                        if pos_side == "long":
                            entry_price = entry_1m.open * (1 + slippage_pct / 100)
                        else:
                            entry_price = entry_1m.open * (1 - slippage_pct / 100)

                        entry_time = entry_1m.timestamp
                        signal_candle_time = candle_4h.timestamp
                        pos_size_usd = capital * position_size_pct
                        in_position = True

                        # Reset trailing stop state
                        trailing_stop_price = 0.0
                        best_price = entry_price
                        breakeven_triggered = False

                        # Calculate ATR-based dynamic stops
                        atr_values = []
                        for j in range(max(0, len(lookback) - scalping_config.atr_period), len(lookback)):
                            atr_values.append(lookback[j].high - lookback[j].low)
                        if atr_values:
                            atr = sum(atr_values) / len(atr_values)
                            atr_pct = (atr / entry_price) * 100
                            dynamic_sl = atr_pct * scalping_config.atr_stop_multiplier
                            dynamic_tp = atr_pct * scalping_config.atr_tp_multiplier
                        else:
                            dynamic_sl = scalping_config.stop_loss_percent
                            dynamic_tp = scalping_config.take_profit_percent

    # Calculate metrics
    if not trades:
        return {
            "pair": pair, "total_pnl_pct": 0, "trades": 0, "wins": 0,
            "win_rate": 0, "profit_factor": 0, "max_dd_pct": 0,
            "sharpe": 0, "avg_hold_min": 0, "sl_exits": 0, "tp_exits": 0,
            "trail_exits": 0, "signal_exits": 0, "final_capital": capital,
        }

    wins = sum(1 for t in trades if t.pnl_pct > 0)
    losses = len(trades) - wins
    total_pnl = sum(t.pnl_pct for t in trades)

    gross_profit = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
    gross_loss = abs(sum(t.pnl_pct for t in trades if t.pnl_pct < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    running = initial_capital
    peak = initial_capital
    max_dd = 0
    for t in trades:
        running += pos_size_usd * (t.pnl_pct / 100)
        peak = max(peak, running)
        dd = (peak - running) / peak * 100
        max_dd = max(max_dd, dd)

    # Sharpe
    returns = [t.pnl_pct for t in trades]
    avg_r = sum(returns) / len(returns)
    if len(returns) > 1:
        var = sum((r - avg_r) ** 2 for r in returns) / (len(returns) - 1)
        sharpe = avg_r / (var ** 0.5) if var > 0 else 0
    else:
        sharpe = 0

    sl_exits = sum(1 for t in trades if t.exit_reason == "stop_loss_1m")
    tp_exits = sum(1 for t in trades if "take_profit" in t.exit_reason)
    trail_exits = sum(1 for t in trades if "trailing_stop" in t.exit_reason)
    signal_exits = sum(1 for t in trades if "signal_exit" in t.exit_reason)
    avg_hold = sum(t.hold_minutes for t in trades) / len(trades)

    return {
        "pair": pair,
        "total_pnl_pct": total_pnl,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(trades) * 100 if trades else 0,
        "profit_factor": pf,
        "max_dd_pct": max_dd,
        "sharpe": sharpe,
        "avg_hold_min": avg_hold,
        "sl_exits": sl_exits,
        "tp_exits": tp_exits,
        "trail_exits": trail_exits,
        "signal_exits": signal_exits,
        "final_capital": capital,
        "long_trades": sum(1 for t in trades if t.side == "long"),
        "short_trades": sum(1 for t in trades if t.side == "short"),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-timeframe backtest: 4h signals + 1m execution")
    parser.add_argument("--days", type=int, default=365, help="Days of data (default: 365)")
    parser.add_argument("--pairs", nargs="+", default=None, help="Override pairs")
    parser.add_argument("--slippage", type=float, default=0.02, help="Slippage percent (default: 0.02)")
    parser.add_argument("--fee", type=float, default=0.16, help="Fee percent (default: 0.16 maker)")
    args = parser.parse_args()

    with open("config/scalping.yaml") as f:
        config = yaml.safe_load(f)

    pairs = args.pairs or config["pairs"]
    pair_params = config.get("pair_parameters", {})
    provider = KrakenCSVProvider()

    end = datetime(2026, 3, 2, tzinfo=timezone.utc)
    start = end - timedelta(days=args.days)

    print("=" * 78)
    print(f"MULTI-TIMEFRAME BACKTEST: 4h signals + 1m execution ({args.days} days)")
    print(f"Period: {start.date()} to {end.date()}")
    print(f"Slippage: {args.slippage}% | Fee: {args.fee}% (maker)")
    print("=" * 78)

    portfolio_pnl = 0
    total_trades = 0
    total_wins = 0
    results = []

    for pair in pairs:
        print(f"\n  Loading {pair} 1-minute data...", end="", flush=True)
        candles_1m = provider.get_ohlc_range(pair, start, end, interval=1)
        print(f" {len(candles_1m)} candles", flush=True)

        if not candles_1m or len(candles_1m) < 10000:
            print(f"  {pair}: Insufficient 1m data")
            continue

        pp = pair_params.get(pair, {})
        sc = ScalpingConfig(
            take_profit_percent=pp.get("take_profit_percent", 4.5),
            stop_loss_percent=pp.get("stop_loss_percent", 2.5),
            rsi_period=pp.get("rsi_period", 10),
            rsi_oversold=pp.get("rsi_oversold", 28),
            rsi_overbought=pp.get("rsi_overbought", 70),
            bb_period=pp.get("bb_period", 20),
            bb_std_dev=pp.get("bb_std_dev", 2.0),
            vwap_threshold_percent=pp.get("vwap_threshold_percent", 0.5),
            volume_spike_threshold=pp.get("volume_spike_threshold", 1.5),
            min_confirmations=pp.get("min_confirmations", 3),
            stoch_k_period=pp.get("stoch_k_period", 13),
            stoch_oversold=pp.get("stoch_oversold", 23),
            stoch_overbought=pp.get("stoch_overbought", 75),
            atr_period=pp.get("atr_period", 14),
            atr_stop_multiplier=pp.get("atr_stop_multiplier", 2.5),
            atr_tp_multiplier=pp.get("atr_tp_multiplier", 2.75),
            short_min_confirmations=pp.get("short_min_confirmations", 3),
        )

        # Per-pair signal interval (4h default, 12h for some pairs)
        pair_interval = pp.get("candle_interval", config.get("strategy", {}).get("candle_interval", 240))
        interval_label = f"{pair_interval // 60}h"
        print(f"  Running MTF backtest ({interval_label} signals)...", end="", flush=True)
        r = run_mtf_backtest(
            pair, candles_1m, sc,
            initial_capital=10000.0,
            position_size_pct=0.20,
            fee_pct=args.fee / 100,
            slippage_pct=args.slippage,
            signal_interval=pair_interval,
        )
        results.append(r)

        if r.get("error"):
            print(f" ERROR: {r['error']}")
            continue

        portfolio_pnl += r["total_pnl_pct"]
        total_trades += r["trades"]
        total_wins += r["wins"]

        status = "OK" if r["total_pnl_pct"] > 0 else "LOSS"
        print(f" done")
        print(f"    {pair:>10}: {r['total_pnl_pct']:+7.2f}%  |  {r['trades']:>3} trades  "
              f"|  {r['win_rate']:.1f}% WR  |  PF {r['profit_factor']:.2f}  "
              f"|  DD {r['max_dd_pct']:.1f}%  |  Sharpe {r['sharpe']:.2f}  [{status}]")
        print(f"               Exits: {r['tp_exits']} TP / {r['sl_exits']} SL / {r['trail_exits']} trail / {r['signal_exits']} signal  "
              f"|  Avg hold: {r['avg_hold_min']:.0f}m  |  L:{r['long_trades']} S:{r['short_trades']}")

    print()
    print("=" * 78)
    print("PORTFOLIO SUMMARY")
    print("=" * 78)

    # Sort by PnL
    results_ok = [r for r in results if not r.get("error")]
    results_ok.sort(key=lambda x: x["total_pnl_pct"], reverse=True)

    for r in results_ok:
        status = "OK" if r["total_pnl_pct"] > 0 else "LOSS"
        print(f"  {r['pair']:>10}: {r['total_pnl_pct']:+7.2f}%  "
              f"|  {r['trades']:>3} trades  |  {r['win_rate']:.1f}% WR  "
              f"|  PF {r['profit_factor']:.2f}  |  DD {r['max_dd_pct']:.1f}%  [{status}]")

    profitable = sum(1 for r in results_ok if r["total_pnl_pct"] > 0)
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    avg_return = portfolio_pnl / len(results_ok) if results_ok else 0

    print(f"\n  Total: {portfolio_pnl:+.2f}%  |  Avg/pair: {avg_return:+.2f}%  "
          f"|  {total_trades} trades  |  {overall_wr:.1f}% WR")
    print(f"  Profitable pairs: {profitable}/{len(results_ok)}")
    print(f"  (4h signal generation + 1-minute stop/TP execution simulation)")


if __name__ == "__main__":
    main()
