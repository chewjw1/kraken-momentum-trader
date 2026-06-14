#!/usr/bin/env python3
"""
Concurrency stress test for live concurrent pair processing.

Validates the _state_lock + capital-reservation model: when many pairs all
try to enter at once in a thread pool (with artificially slow fills to force
overlap), concurrent entries must NOT double-allocate capital, the reservation
counter must always net to zero, and no thread may raise.

This is the only genuinely new failure mode introduced by live.concurrent_pairs
> 1, so it gets a dedicated decisive test rather than relying on the (instant-
fill, sequential) replay path.

Usage:  python3 scripts/test_concurrency.py
"""

import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ["KRAKEN_API_KEY"] = "test"
os.environ["KRAKEN_API_SECRET"] = "test"

from src.exchange.kraken_client import KrakenClient, Ticker, OHLC
from src.strategy.base_strategy import MarketData
from src.strategy.scalping_strategy import SignalType
from src.observability.logger import configure_logging
from run_scalping_live import ScalpingTrader

FAILURES = []


def check(cond, msg):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {msg}")
    if not cond:
        FAILURES.append(msg)


def build_trader(concurrent_pairs):
    original_init = KrakenClient.__init__
    KrakenClient.__init__ = lambda self, **kw: None
    trader = ScalpingTrader(
        config_path="config/scalping.yaml",
        data_dir="data/concurrency_test",
        paper_trading=True,
    )
    KrakenClient.__init__ = original_init

    # Force the LIVE concurrent dispatch path.
    trader.paper_trading = False
    trader.live_concurrent_pairs = concurrent_pairs
    trader.client = MagicMock()

    trader.capital = 10000.0
    trader.initial_capital = 10000.0
    trader.position_size_pct = 20.0  # -> ~5 positions fit in capital
    trader.disaster_stop_pct = 0.0   # don't interfere
    trader._candles_since_flip = 10 ** 9  # no flip block
    trader._block_entry_regimes = set()
    trader._block_short_regimes = set()

    # Every pair gets a fresh advancing candle each cycle so the one-decision
    # guard passes exactly once per pair per cycle.
    cycle = {"n": 0}

    def fake_market_data(pair):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=4 * cycle["n"])
        c = OHLC(timestamp=ts, open=100.0, high=101.0, low=99.0,
                 close=100.0, volume=1000.0, vwap=100.0, count=10)
        ticker = Ticker(pair=pair, ask=100.1, bid=99.9, last=100.0,
                        volume_24h=6000.0, vwap_24h=100.0, trades_24h=100,
                        low_24h=99.0, high_24h=101.0, timestamp=ts)
        return MarketData(pair=pair, ohlc=[c] * 30,
                          prices=[100.0] * 30, volumes=[1000.0] * 30,
                          ticker=ticker)

    trader._get_market_data = fake_market_data
    trader._compute_indicator_snapshot = lambda *a, **k: None
    trader.pair_manager.is_pair_enabled = lambda pair: True
    trader._correlation_blocked = lambda pair: None
    trader._volume_filter_blocked = lambda md: None
    trader._place_server_stop = lambda *a, **k: None
    trader._save_state = lambda: None  # avoid disk contention noise

    # Every pair signals BUY (a strategy stub).
    buy_signal = SimpleNamespace(
        signal_type=SimpleNamespace(value="buy"),
        strength=0.8,
        reason="stress-test buy",
    )
    strat_stub = SimpleNamespace(analyze=lambda md, pos=None: buy_signal)
    trader._get_strategy_for_pair = lambda pair: strat_stub

    # Slow fill to force thread overlap; records peak concurrency observed.
    inflight = {"now": 0, "peak": 0}
    inflight_lock = threading.Lock()

    def slow_entry(pair, side, size, price):
        with inflight_lock:
            inflight["now"] += 1
            inflight["peak"] = max(inflight["peak"], inflight["now"])
        time.sleep(0.05)
        with inflight_lock:
            inflight["now"] -= 1
        return {
            "order_id": f"O-{pair}",
            "fill_price": price,
            "fill_volume": size,
            "fee": price * size * 0.0016,
        }

    trader._execute_entry_order = slow_entry
    return trader, cycle, inflight


def run_case(concurrent_pairs):
    print(f"\n=== concurrent_pairs={concurrent_pairs} ===")
    trader, cycle, inflight = build_trader(concurrent_pairs)

    for n in range(3):
        cycle["n"] = n
        # Clear positions each cycle to re-contend for full capital.
        trader.positions.clear()
        trader._last_processed_candle_ts.clear()
        trader._process_all_pairs()

        deployed = sum(p["size_usd"] for p in trader.positions.values())
        check(deployed <= trader.capital + 1e-6,
              f"cycle {n}: deployed ${deployed:.2f} <= capital ${trader.capital:.2f} "
              f"({len(trader.positions)} positions)")

    check(abs(trader._reserved_usd) < 1e-6,
          f"reservation counter net zero after run (={trader._reserved_usd:.6f})")
    if concurrent_pairs > 1:
        check(inflight["peak"] > 1,
              f"fills actually overlapped (peak in-flight={inflight['peak']})")


def main():
    configure_logging(level="CRITICAL", format_type="json")
    print("=" * 70)
    print("  CONCURRENCY STRESS TEST — capital reservation under contention")
    print("=" * 70)
    run_case(1)   # sequential baseline
    run_case(10)  # max contention: 10 pairs, pool of 10

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"  RESULT: FAIL — {len(FAILURES)} check(s) failed")
        print("=" * 70)
        return 1
    print("  RESULT: PASS — no double-allocation, reservations balanced.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
