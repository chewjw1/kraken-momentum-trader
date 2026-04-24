#!/usr/bin/env python3
"""
Q1 2026 Backtest — validates scalping strategy against Q1 2026 data.

Loads CSV data from data/q1_2026/, runs per-pair backtests using each
pair's production config from config/scalping.yaml, and reports results.
"""

import csv
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.exchange.kraken_client import OHLC
from src.strategy.scalping_strategy import ScalpingStrategy, ScalpingConfig
from src.strategy.base_strategy import MarketData, Position
from src.strategy.regime_detector import RegimeDetector, RegimeConfig, MarketRegime, REGIME_ADJUSTMENTS
from src.observability.logger import configure_logging


def load_csv_candles(filepath: str) -> list[OHLC]:
    candles = []
    with open(filepath) as f:
        for row in csv.reader(f):
            ts, o, h, l, c, v = row
            candles.append(OHLC(
                timestamp=datetime.fromtimestamp(int(ts), tz=timezone.utc),
                open=float(o), high=float(h), low=float(l),
                close=float(c), volume=float(v),
                vwap=(float(h) + float(l) + float(c)) / 3,
                count=0
            ))
    return candles


def build_scalping_config(default_cfg: dict, pair_params: dict, indicators: dict) -> ScalpingConfig:
    pp = pair_params or {}
    return ScalpingConfig(
        take_profit_percent=pp.get('take_profit_percent', default_cfg.get('take_profit_percent', 4.5)),
        stop_loss_percent=pp.get('stop_loss_percent', default_cfg.get('stop_loss_percent', 2.5)),
        min_confirmations=pp.get('min_confirmations', default_cfg.get('min_confirmations', 3)),
        rsi_period=pp.get('rsi_period', indicators.get('rsi', {}).get('period', 10)),
        rsi_oversold=pp.get('rsi_oversold', indicators.get('rsi', {}).get('oversold', 28)),
        rsi_overbought=pp.get('rsi_overbought', indicators.get('rsi', {}).get('overbought', 70)),
        bb_period=pp.get('bb_period', indicators.get('bollinger', {}).get('period', 20)),
        bb_std_dev=pp.get('bb_std_dev', indicators.get('bollinger', {}).get('std_dev', 2.0)),
        vwap_threshold_percent=pp.get('vwap_threshold_percent', indicators.get('vwap', {}).get('threshold_percent', 0.5)),
        volume_spike_threshold=pp.get('volume_spike_threshold', indicators.get('volume', {}).get('spike_threshold', 1.5)),
        stoch_k_period=pp.get('stoch_k_period', indicators.get('stochastic', {}).get('k_period', 13)),
        stoch_d_period=indicators.get('stochastic', {}).get('d_period', 3),
        stoch_oversold=pp.get('stoch_oversold', indicators.get('stochastic', {}).get('oversold', 23)),
        stoch_overbought=pp.get('stoch_overbought', indicators.get('stochastic', {}).get('overbought', 75)),
        macd_fast=indicators.get('macd', {}).get('fast_period', 12),
        macd_slow=indicators.get('macd', {}).get('slow_period', 26),
        macd_signal=indicators.get('macd', {}).get('signal_period', 9),
        obv_sma_period=indicators.get('obv', {}).get('sma_period', 20),
        atr_period=pp.get('atr_period', indicators.get('atr', {}).get('period', 14)),
        atr_stop_multiplier=pp.get('atr_stop_multiplier', indicators.get('atr', {}).get('stop_multiplier', 2.5)),
        atr_tp_multiplier=pp.get('atr_tp_multiplier', indicators.get('atr', {}).get('tp_multiplier', 2.75)),
        use_atr_stops=indicators.get('atr', {}).get('use_dynamic_stops', True),
        fee_percent=0.16,
        shorting_enabled=True,
        short_min_confirmations=pp.get('short_min_confirmations', 3),
        ema_filter_enabled=True,
    )


def run_backtest(pair: str, candles: list[OHLC], strategy: ScalpingStrategy,
                 capital: float = 10000.0, position_pct: float = 25.0,
                 fee_pct: float = 0.16, slippage_pct: float = 0.05) -> dict:
    lookback = 50
    trades = []
    peak_capital = capital
    max_drawdown = 0.0

    in_position = False
    entry_price = 0.0
    entry_time = None
    pos_size_usd = 0.0
    pos_side = "long"
    cooldown_until = 0
    consecutive_losses = 0

    regime_detector = RegimeDetector(RegimeConfig(
        bull_slope_threshold=0.02, bear_slope_threshold=-0.02
    ))
    current_regime = MarketRegime.UNKNOWN
    position_scale = 1.0  # regime-adjusted position scale
    base_config = ScalpingConfig(
        take_profit_percent=strategy.config.take_profit_percent,
        stop_loss_percent=strategy.config.stop_loss_percent,
        min_confirmations=strategy.config.min_confirmations,
        rsi_period=strategy.config.rsi_period,
        rsi_oversold=strategy.config.rsi_oversold,
        rsi_overbought=strategy.config.rsi_overbought,
        bb_period=strategy.config.bb_period,
        bb_std_dev=strategy.config.bb_std_dev,
        vwap_threshold_percent=strategy.config.vwap_threshold_percent,
        volume_spike_threshold=strategy.config.volume_spike_threshold,
        stoch_k_period=strategy.config.stoch_k_period,
        stoch_d_period=strategy.config.stoch_d_period,
        stoch_oversold=strategy.config.stoch_oversold,
        stoch_overbought=strategy.config.stoch_overbought,
        macd_fast=strategy.config.macd_fast,
        macd_slow=strategy.config.macd_slow,
        macd_signal=strategy.config.macd_signal,
        atr_period=strategy.config.atr_period,
        atr_stop_multiplier=strategy.config.atr_stop_multiplier,
        atr_tp_multiplier=strategy.config.atr_tp_multiplier,
        use_atr_stops=strategy.config.use_atr_stops,
        fee_percent=fee_pct,
        shorting_enabled=strategy.config.shorting_enabled,
        short_min_confirmations=strategy.config.short_min_confirmations,
    )

    # Apply initial UNKNOWN regime adjustments immediately
    init_adj = REGIME_ADJUSTMENTS[MarketRegime.UNKNOWN]
    tp_m = init_adj.get('take_profit_multiplier', 1.0)
    sl_m = init_adj.get('stop_loss_multiplier', 1.0)
    conf_off = init_adj.get('min_confirmations_offset', 0)
    s_conf_off = init_adj.get('short_confirmations_offset', 0)
    position_scale = init_adj.get('position_scale_multiplier', 1.0)
    from src.strategy.scalping_strategy import ScalpingConfig as SC
    strategy = ScalpingStrategy(SC(
        take_profit_percent=base_config.take_profit_percent * tp_m,
        stop_loss_percent=base_config.stop_loss_percent * sl_m,
        min_confirmations=max(1, base_config.min_confirmations + conf_off),
        rsi_period=base_config.rsi_period,
        rsi_oversold=base_config.rsi_oversold,
        rsi_overbought=base_config.rsi_overbought,
        bb_period=base_config.bb_period,
        bb_std_dev=base_config.bb_std_dev,
        vwap_threshold_percent=base_config.vwap_threshold_percent,
        volume_spike_threshold=base_config.volume_spike_threshold,
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
        fee_percent=fee_pct,
        ema_filter_enabled=init_adj.get('ema_filter_enabled', True),
        shorting_enabled=init_adj.get('shorting_enabled', base_config.shorting_enabled),
        short_min_confirmations=max(1, base_config.short_min_confirmations + s_conf_off),
        trend_short_enabled=init_adj.get('trend_short_enabled', False),
    ))

    for i in range(lookback, len(candles)):
        if i % 20 == 0 and i >= 100:
            closes = [c.close for c in candles[:i+1]]
            highs = [c.high for c in candles[:i+1]]
            lows = [c.low for c in candles[:i+1]]
            result = regime_detector.detect(closes, highs, lows)
            if result.regime != current_regime:
                current_regime = result.regime
                adj = REGIME_ADJUSTMENTS.get(result.regime, REGIME_ADJUSTMENTS[MarketRegime.UNKNOWN])
                tp_m = adj.get('take_profit_multiplier', 1.0)
                sl_m = adj.get('stop_loss_multiplier', 1.0)
                conf_off = adj.get('min_confirmations_offset', 0)
                s_conf_off = adj.get('short_confirmations_offset', 0)
                position_scale = adj.get('position_scale_multiplier', 1.0)
                from src.strategy.scalping_strategy import ScalpingConfig as SC
                strategy = ScalpingStrategy(SC(
                    take_profit_percent=base_config.take_profit_percent * tp_m,
                    stop_loss_percent=base_config.stop_loss_percent * sl_m,
                    min_confirmations=max(1, base_config.min_confirmations + conf_off),
                    rsi_period=base_config.rsi_period,
                    rsi_oversold=base_config.rsi_oversold,
                    rsi_overbought=base_config.rsi_overbought,
                    bb_period=base_config.bb_period,
                    bb_std_dev=base_config.bb_std_dev,
                    vwap_threshold_percent=base_config.vwap_threshold_percent,
                    volume_spike_threshold=base_config.volume_spike_threshold,
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
                    fee_percent=fee_pct,
                    ema_filter_enabled=adj.get('ema_filter_enabled', True),
                    shorting_enabled=adj.get('shorting_enabled', base_config.shorting_enabled),
                    short_min_confirmations=max(1, base_config.short_min_confirmations + s_conf_off),
                    trend_short_enabled=adj.get('trend_short_enabled', False),
                ))

        window = candles[i - lookback:i + 1]
        current = candles[i]
        md = MarketData(pair=pair, ohlc=window,
                        prices=[c.close for c in window],
                        volumes=[c.volume for c in window],
                        ticker=None)

        if in_position:
            # Intra-candle stop: floor at 2.5% (crypto noise), cap at 8%
            stop_pct_abs = max(strategy.config.stop_loss_percent, 2.5)
            stop_pct_abs = min(stop_pct_abs, 8.0)
            intra_stopped = False
            intra_exit_price = 0.0
            if pos_side == "long" and current.low <= entry_price * (1 - stop_pct_abs / 100):
                intra_stopped = True
                intra_exit_price = entry_price * (1 - stop_pct_abs / 100)
            elif pos_side == "short" and current.high >= entry_price * (1 + stop_pct_abs / 100):
                intra_stopped = True
                intra_exit_price = entry_price * (1 + stop_pct_abs / 100)

            if intra_stopped:
                if pos_side == "short":
                    gross_pnl = ((entry_price - intra_exit_price) / entry_price) * 100
                else:
                    gross_pnl = ((intra_exit_price - entry_price) / entry_price) * 100
                exit_price = intra_exit_price
                reason = f"Intra-candle stop: {gross_pnl - (fee_pct * 2):.2f}%"
                net_pnl = gross_pnl - (fee_pct * 2)
                pnl_usd = pos_size_usd * (net_pnl / 100)
                capital += pnl_usd
                peak_capital = max(peak_capital, capital)
                dd = (peak_capital - capital) / peak_capital * 100
                max_drawdown = max(max_drawdown, dd)
                trades.append({
                    'entry_time': entry_time.isoformat(),
                    'exit_time': current.timestamp.isoformat(),
                    'side': pos_side, 'entry_price': entry_price,
                    'exit_price': exit_price, 'pnl_pct': round(net_pnl, 3),
                    'pnl_usd': round(pnl_usd, 2), 'reason': reason,
                })
                in_position = False
                consecutive_losses += 1
                continue

            position = Position(pair=pair, side=pos_side, entry_price=entry_price,
                                current_price=current.close,
                                size=pos_size_usd / entry_price, entry_time=entry_time)
            signal = strategy.analyze(md, position)
            exit_signals = ("sell", "close_long") if pos_side == "long" else ("close_short",)

            if signal.signal_type.value in exit_signals:
                if pos_side == "short":
                    exit_price = current.close * (1 + slippage_pct / 100)
                    gross_pnl = ((entry_price - exit_price) / entry_price) * 100
                else:
                    exit_price = current.close * (1 - slippage_pct / 100)
                    gross_pnl = ((exit_price - entry_price) / entry_price) * 100

                net_pnl = gross_pnl - (fee_pct * 2)
                pnl_usd = pos_size_usd * (net_pnl / 100)
                capital += pnl_usd
                peak_capital = max(peak_capital, capital)
                dd = (peak_capital - capital) / peak_capital * 100
                max_drawdown = max(max_drawdown, dd)

                trades.append({
                    'entry_time': entry_time.isoformat(),
                    'exit_time': current.timestamp.isoformat(),
                    'side': pos_side,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': round(net_pnl, 3),
                    'pnl_usd': round(pnl_usd, 2),
                    'reason': signal.reason[:80],
                })
                in_position = False
                if net_pnl > 0:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1
        else:
            signal = strategy.analyze(md, None)
            scaled_pct = position_pct * position_scale
            if signal.signal_type.value == "buy":
                entry_price = current.close * (1 + slippage_pct / 100)
                entry_time = current.timestamp
                pos_size_usd = capital * (scaled_pct / 100)
                pos_side = "long"
                in_position = True
            elif signal.signal_type.value == "sell_short":
                entry_price = current.close * (1 - slippage_pct / 100)
                entry_time = current.timestamp
                pos_size_usd = capital * (scaled_pct / 100)
                pos_side = "short"
                in_position = True

    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    gross_profit = sum(t['pnl_usd'] for t in wins)
    gross_loss = abs(sum(t['pnl_usd'] for t in losses))
    total_pnl = sum(t['pnl_usd'] for t in trades)
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    wr = len(wins) / len(trades) * 100 if trades else 0

    return {
        'pair': pair,
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(wr, 1),
        'total_pnl_usd': round(total_pnl, 2),
        'total_pnl_pct': round((capital - 10000) / 10000 * 100, 2),
        'profit_factor': round(pf, 2),
        'max_drawdown_pct': round(max_drawdown, 2),
        'final_capital': round(capital, 2),
        'trade_details': trades,
        'regime': current_regime.value,
    }


def main():
    configure_logging(level="WARNING", format_type="json")

    with open("config/scalping.yaml") as f:
        config = yaml.safe_load(f)

    strategy_cfg = config.get('strategy', {})
    indicators = config.get('indicators', {})
    pair_params = config.get('pair_parameters', {})
    pairs = config.get('pairs', [])
    position_pct = config.get('position', {}).get('size_percent', 25.0)
    fee_pct = config.get('fees', {}).get('maker_percent', 0.16)

    data_dir = Path("data/q1_2026")

    print("=" * 80)
    print("  Q1 2026 BACKTEST — SCALPING STRATEGY (Current Production Config)")
    print("=" * 80)
    print(f"  Capital: $10,000 | Position: {position_pct}% | Fee: {fee_pct}%")
    print(f"  Data: {data_dir}")
    print("=" * 80)

    results = []
    portfolio_capital = 10000.0
    portfolio_pnl = 0.0

    for pair in pairs:
        fname = pair.replace("/", "_") + "_4h.csv"
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"\n  {pair}: NO DATA FILE ({fpath})")
            continue

        candles = load_csv_candles(str(fpath))
        pp = pair_params.get(pair, {})
        sc = build_scalping_config(strategy_cfg, pp, indicators)
        strategy = ScalpingStrategy(sc)

        interval = pp.get('candle_interval', strategy_cfg.get('candle_interval', 240))
        if interval == 720:
            agg_candles = []
            for j in range(0, len(candles) - 2, 3):
                group = candles[j:j+3]
                agg_candles.append(OHLC(
                    timestamp=group[0].timestamp,
                    open=group[0].open,
                    high=max(c.high for c in group),
                    low=min(c.low for c in group),
                    close=group[-1].close,
                    vwap=sum(c.vwap * c.volume for c in group) / max(sum(c.volume for c in group), 1e-10),
                    volume=sum(c.volume for c in group),
                    count=0
                ))
            candles = agg_candles

        result = run_backtest(pair, candles, strategy,
                              capital=10000.0, position_pct=position_pct,
                              fee_pct=fee_pct)
        results.append(result)
        portfolio_pnl += result['total_pnl_usd']

        status = "+" if result['total_pnl_usd'] > 0 else ""
        print(f"\n  {pair:12s}  {result['trades']:3d} trades  "
              f"WR {result['win_rate']:5.1f}%  "
              f"PF {result['profit_factor']:5.2f}  "
              f"P&L {status}${result['total_pnl_usd']:8.2f}  "
              f"({status}{result['total_pnl_pct']:6.2f}%)  "
              f"DD {result['max_drawdown_pct']:5.2f}%  "
              f"[{result['regime']}]")

        for t in result['trade_details'][:3]:
            print(f"    {t['side']:5s} ${t['entry_price']:.4f} -> ${t['exit_price']:.4f}  "
                  f"{'+' if t['pnl_pct'] > 0 else ''}{t['pnl_pct']:.2f}%  {t['reason'][:50]}")
        if len(result['trade_details']) > 3:
            print(f"    ... {len(result['trade_details']) - 3} more trades")

    print("\n" + "=" * 80)
    total_trades = sum(r['trades'] for r in results)
    total_wins = sum(r['wins'] for r in results)
    profitable_pairs = sum(1 for r in results if r['total_pnl_usd'] > 0)
    print(f"  PORTFOLIO SUMMARY")
    print(f"  Total Trades: {total_trades}")
    print(f"  Total Wins: {total_wins} ({total_wins/total_trades*100:.1f}% WR)" if total_trades else "  No trades")
    print(f"  Profitable Pairs: {profitable_pairs}/{len(results)}")
    print(f"  Portfolio P&L: {'+'if portfolio_pnl > 0 else ''}${portfolio_pnl:.2f}")
    print("=" * 80)

    with open("data/q1_2026/backtest_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Detailed results saved to data/q1_2026/backtest_results.json")

    return 0 if portfolio_pnl > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
