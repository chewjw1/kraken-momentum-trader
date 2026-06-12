#!/usr/bin/env python3
"""
Replay historical data through the REAL ScalpingTrader.

Feeds Q1 2025 Kraken 4h OHLCV data through the actual live trading code
path — same regime detection, strategy rebuilds, position management,
order execution, and state persistence. Catches bugs that simplified
backtests miss.

Usage:
    python3 run_replay.py [data_dir]
    python3 run_replay.py data/q1_2025
"""

import csv
import sys
import os
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
from dataclasses import dataclass

from src.exchange.kraken_client import (
    KrakenClient, OHLC, Ticker, Order, OrderSide, OrderType, Balance,
)
from src.observability.logger import configure_logging

# Prevent live runner from loading real credentials
os.environ["KRAKEN_API_KEY"] = "replay-test"
os.environ["KRAKEN_API_SECRET"] = "replay-test"

from run_scalping_live import ScalpingTrader


class ReplayKrakenClient(KrakenClient):
    """Mock KrakenClient that serves historical candle data."""

    def __init__(self, candle_data: dict[str, list[OHLC]], capital: float = 10000.0):
        # Skip real __init__ entirely — no API calls, no rate limiter
        self.paper_trading = True
        self._paper_trading_capital = capital
        self._paper_balances = {"USD": capital}
        self._paper_orders = {}
        self._paper_order_counter = 0
        self._paper_balances_initialized = True
        self._session = MagicMock()

        # Historical data keyed by pair
        self._candle_data = candle_data
        # Current candle index — advanced externally
        self._current_idx = 0
        # How many candles to serve per get_ohlc call (lookback window)
        self._lookback = 540

    def get_ohlc(self, pair: str, interval: int = 240, since=None) -> list[OHLC]:
        pair_key = pair.replace("/", "_")
        candles = self._candle_data.get(pair_key, [])
        end = min(self._current_idx + 1, len(candles))
        start = max(0, end - self._lookback)
        return candles[start:end]

    def get_ticker(self, pair: str) -> Ticker:
        pair_key = pair.replace("/", "_")
        candles = self._candle_data.get(pair_key, [])
        if self._current_idx < len(candles):
            c = candles[self._current_idx]
        else:
            c = candles[-1]
        return Ticker(
            pair=pair, ask=c.close * 1.001, bid=c.close * 0.999,
            last=c.close, volume_24h=c.volume * 6, vwap_24h=c.vwap,
            trades_24h=1000, low_24h=c.low, high_24h=c.high,
            timestamp=c.timestamp,
        )

    def get_balances(self) -> dict:
        return {"USD": Balance(asset="USD", total=self._paper_balances.get("USD", 0), available=self._paper_balances.get("USD", 0))}

    # **kwargs absorbs margin params (leverage/reduce_only) the trader passes
    # for shorts — fills here are simulated, so they have no replay effect.
    def place_order(self, pair, side, order_type, volume, price=None, stop_price=None, validate_only=False, post_only=False, **kwargs) -> Order:
        self._paper_order_counter += 1
        pair_key = pair.replace("/", "_")
        candles = self._candle_data.get(pair_key, [])
        if self._current_idx < len(candles):
            fill_price = candles[self._current_idx].close
        else:
            fill_price = price or 0.0

        # Simulate slippage
        if side == OrderSide.BUY:
            fill_price *= 1.0005
        else:
            fill_price *= 0.9995

        fee = fill_price * volume * 0.0016  # 0.16% maker fee

        return Order(
            order_id=f"REPLAY-{self._paper_order_counter}",
            pair=pair, side=side, order_type=order_type,
            price=fill_price, volume=volume, filled_volume=volume,
            status="closed", created_at=datetime.now(timezone.utc),
            cost=fill_price * volume, fee=fee,
        )

    def place_maker_order(self, pair, side, volume, price_offset_percent=0.0, **kwargs) -> Order:
        return self.place_order(pair, side, OrderType.LIMIT, volume)

    def close(self):
        pass


def load_candles(data_dir: Path) -> dict[str, list[OHLC]]:
    """Load all CSV files into OHLC lists."""
    candle_data = {}
    for csv_file in sorted(data_dir.glob("*_4h.csv")):
        pair_key = csv_file.stem.replace("_4h", "")
        candles = []
        with open(csv_file) as f:
            for row in csv.reader(f):
                ts, o, h, l, c, v = row
                candles.append(OHLC(
                    timestamp=datetime.fromtimestamp(int(ts), tz=timezone.utc),
                    open=float(o), high=float(h), low=float(l),
                    close=float(c), volume=float(v),
                    vwap=(float(h) + float(l) + float(c)) / 3, count=0,
                ))
        candle_data[pair_key] = candles
        print(f"  Loaded {pair_key}: {len(candles)} candles")
    return candle_data


def main():
    configure_logging(level="WARNING", format_type="json")

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/q1_2025")
    print(f"\n{'='*70}")
    print(f"  REPLAY TEST — feeding {data_dir} through live ScalpingTrader")
    print(f"{'='*70}\n")

    # Load historical data
    candle_data = load_candles(data_dir)
    if not candle_data:
        print("No data found!")
        return 1

    num_candles = len(next(iter(candle_data.values())))

    # Clean replay state
    replay_data_dir = Path("data/replay_test")
    if replay_data_dir.exists():
        shutil.rmtree(replay_data_dir)
    replay_data_dir.mkdir(parents=True, exist_ok=True)

    # Create the mock client
    mock_client = ReplayKrakenClient(candle_data, capital=10000.0)

    # Create trader — monkey-patch the client before __init__ fetches balances
    original_init = KrakenClient.__init__
    KrakenClient.__init__ = lambda self, **kwargs: None
    trader = ScalpingTrader(
        config_path="config/scalping.yaml",
        data_dir=str(replay_data_dir),
        paper_trading=True,
    )
    KrakenClient.__init__ = original_init

    # Replace client with our mock
    trader.client = mock_client
    trader.capital = 10000.0
    trader.initial_capital = 10000.0

    # Track events for reporting
    events = []
    prev_regime = trader._current_regime
    prev_positions = set()

    print(f"\n  Stepping through {num_candles} candles...\n")
    print(f"  {'Candle':>7} | {'Date':>10} | {'Regime':>10} | {'Positions':>10} | {'Capital':>12} | Event")
    print(f"  {'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*40}")

    for i in range(num_candles):
        mock_client._current_idx = i

        # Get current date from BTC candle
        btc_candles = candle_data.get("BTC_USD", [])
        current_date = btc_candles[i].timestamp.strftime("%Y-%m-%d") if i < len(btc_candles) else "?"

        # Snapshot state before processing
        capital_before = trader.capital
        positions_before = set(trader.positions.keys())
        regime_before = trader._current_regime

        # Run one cycle: regime detection + process all pairs
        try:
            trader._update_regime()
        except Exception as e:
            events.append((i, current_date, f"REGIME ERROR: {e}"))

        for pair in trader.pairs:
            try:
                trader._process_pair(pair)
            except Exception as e:
                events.append((i, current_date, f"ERROR {pair}: {e}"))
                print(f"  {i:>7} | {current_date:>10} | {'':>10} | {'':>10} | {'':>12} | ERROR {pair}: {e}")

        # Detect events
        regime_after = trader._current_regime
        positions_after = set(trader.positions.keys())
        capital_after = trader.capital

        # Regime change
        if regime_after != regime_before:
            event = f"REGIME {regime_before.value} -> {regime_after.value}"
            events.append((i, current_date, event))
            print(f"  {i:>7} | {current_date:>10} | {regime_after.value:>10} | {len(positions_after):>10} | ${capital_after:>11.2f} | {event}")

        # New positions opened
        opened = positions_after - positions_before
        for pair in opened:
            pos = trader.positions[pair]
            side = pos.get('side', 'long')
            entry = pos.get('entry_price', 0)
            size_usd = pos.get('size_usd', 0)
            event = f"OPEN {side:5s} {pair} @ ${entry:.2f} (${size_usd:.0f})"
            events.append((i, current_date, event))
            print(f"  {i:>7} | {current_date:>10} | {regime_after.value:>10} | {len(positions_after):>10} | ${capital_after:>11.2f} | {event}")

        # Positions closed
        closed = positions_before - positions_after
        for pair in closed:
            pnl_change = capital_after - capital_before
            event = f"CLOSE {pair} P&L ${pnl_change:+.2f}"
            events.append((i, current_date, event))
            print(f"  {i:>7} | {current_date:>10} | {regime_after.value:>10} | {len(positions_after):>10} | ${capital_after:>11.2f} | {event}")

    # Final summary
    initial = trader.initial_capital
    total_pnl = trader.capital - initial
    total_trades = trader.metrics['total_trades']
    wins = trader.metrics['wins']
    losses = trader.metrics['losses']
    wr = (wins / total_trades * 100) if total_trades > 0 else 0
    pnl_pct = (total_pnl / initial * 100) if initial > 0 else 0

    print(f"\n{'='*70}")
    print(f"  REPLAY RESULTS")
    print(f"{'='*70}")
    print(f"  Initial capital:  ${initial:,.2f}")
    print(f"  Final capital:    ${trader.capital:,.2f}")
    print(f"  Total P&L:        ${total_pnl:+,.2f} ({pnl_pct:+.2f}%)")
    print(f"  Trades:           {total_trades} ({wins}W / {losses}L, {wr:.1f}% WR)")
    print(f"  Open positions:   {len(trader.positions)}")
    print(f"  Final regime:     {trader._current_regime.value}")

    if trader.positions:
        print(f"\n  Open positions at end:")
        for pair, pos in trader.positions.items():
            print(f"    {pair}: {pos.get('side','?')} @ ${pos.get('entry_price',0):.2f}")

    # Regime change summary
    regime_changes = [e for e in events if e[2].startswith("REGIME")]
    print(f"\n  Regime changes:   {len(regime_changes)}")
    for _, date, event in regime_changes:
        print(f"    {date}: {event}")

    # Trade summary by pair
    print(f"\n  Trades by pair:")
    pair_trades = {}
    for _, _, event in events:
        if event.startswith("CLOSE"):
            pair = event.split()[1]
            pnl = float(event.split("$")[1])
            if pair not in pair_trades:
                pair_trades[pair] = {"count": 0, "pnl": 0.0}
            pair_trades[pair]["count"] += 1
            pair_trades[pair]["pnl"] += pnl
    for pair in sorted(pair_trades.keys()):
        info = pair_trades[pair]
        s = "+" if info["pnl"] > 0 else ""
        print(f"    {pair:12s}  {info['count']:3d} trades  {s}${info['pnl']:.2f}")

    # Save state for inspection
    trader._save_state()
    print(f"\n  State saved to: {replay_data_dir}/state.json")

    # Check for anomalies
    print(f"\n{'='*70}")
    print(f"  ANOMALY CHECK")
    print(f"{'='*70}")
    anomalies = []

    # Check if any pair never traded
    traded_pairs = set()
    for _, _, event in events:
        if event.startswith("OPEN"):
            pair = event.split()[2]
            traded_pairs.add(pair)
    for pair in trader.pairs:
        if pair not in traded_pairs:
            anomalies.append(f"  WARN: {pair} never traded in {num_candles} candles")

    # Check for positions that were never closed
    if trader.positions:
        for pair in trader.positions:
            anomalies.append(f"  WARN: {pair} still open at end of replay")

    # Check capital sanity
    if trader.capital < initial * 0.5:
        anomalies.append(f"  WARN: Capital dropped below 50% (${trader.capital:.2f})")
    if trader.capital > initial * 2:
        anomalies.append(f"  WARN: Capital more than doubled (${trader.capital:.2f}) — suspiciously high")

    # Per-pair pattern checks (catch noise-trading bugs)
    fee_pct = 0.32  # round-trip 0.16% maker fee
    pair_trade_data: dict = {}
    for _, _, event in events:
        if event.startswith("CLOSE"):
            parts = event.split()
            pair = parts[1]
            try:
                pnl = float(event.split("$")[1])
            except (IndexError, ValueError):
                continue
            pair_trade_data.setdefault(pair, []).append(pnl)

    for pair, pnls in pair_trade_data.items():
        if len(pnls) >= 5:
            losses = [p for p in pnls if p < 0]
            wins = [p for p in pnls if p > 0]
            wr = len(wins) / len(pnls) * 100

            # Excessive trade count vs candle count
            trades_per_100_candles = (len(pnls) / num_candles) * 100
            if trades_per_100_candles > 10:  # >1 trade per 10 candles is excessive
                anomalies.append(
                    f"  WARN: {pair} traded {len(pnls)}x in {num_candles} candles "
                    f"({trades_per_100_candles:.1f} per 100) — possible noise trading"
                )

            # Win rate below 15% likely indicates fee-bleed pattern
            if wr < 15 and len(pnls) >= 10:
                anomalies.append(
                    f"  WARN: {pair} win rate {wr:.1f}% over {len(pnls)} trades — likely fee-bleed bug"
                )

            # Many losses clustered at fee level (within 0.1% of round-trip fee)
            fee_level_losses = [p for p in losses if abs(abs(p / max(1, sum(abs(x) for x in pnls) / len(pnls))) - fee_pct/100) < 0.001]
            if losses:
                avg_loss_pct = sum(losses) / len(losses)
                # If avg loss is suspiciously close to fee level
                # (Hard to compute exact pct without size; just flag if > 80% of trades are losses)
                if len(losses) / len(pnls) > 0.8 and len(pnls) >= 10:
                    anomalies.append(
                        f"  WARN: {pair} {len(losses)}/{len(pnls)} losses (avg ${avg_loss_pct:.2f}) — investigate"
                    )

    if anomalies:
        for a in anomalies:
            print(a)
    else:
        print("  No anomalies detected.")

    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
