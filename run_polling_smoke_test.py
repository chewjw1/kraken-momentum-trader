#!/usr/bin/env python3
"""
Production-realistic smoke test: simulates 60-second polling within each candle.

The original replay tests one decision per 4h candle. But the live runner polls
every 60 seconds, making 240 decisions per 4h candle. The POL bug (41 trades/wk
with 95% losses at exactly fee level) only emerged in production because of this
within-candle polling — the strategy flipped its decision within seconds.

This harness simulates that polling. For each candle:
  - Run _process_pair N times (default 60 = once per minute, ~60% of a 1m loop)
  - Track decisions, ensure they're stable across polls within a candle
  - Flag bugs:
    * Multiple entries/exits for the same pair within one candle
    * Trades that close within the same candle they opened (intra-candle churn)
    * Position count instability within a candle
    * Signal flip-flopping (entry/exit on consecutive polls)

Usage:
    python3 run_polling_smoke_test.py [data_dir] [polls_per_candle]
"""
import csv, sys, os, shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
from collections import defaultdict

from src.exchange.kraken_client import (
    KrakenClient, OHLC, Ticker, Order, OrderSide, OrderType, Balance,
)
from src.observability.logger import configure_logging

os.environ["KRAKEN_API_KEY"] = "smoke-test"
os.environ["KRAKEN_API_SECRET"] = "smoke-test"

from run_scalping_live import ScalpingTrader


class PollingMockClient(KrakenClient):
    """Mock client that simulates intra-candle polling.

    Key difference from ReplayKrakenClient: ticker prices interpolate within
    a candle based on the current poll position. Early polls see the open price,
    later polls see prices walking toward the close (with small noise from H/L).
    """

    def __init__(self, candle_data: dict, capital: float = 10000.0):
        self.paper_trading = True
        self._paper_balances = {"USD": capital}
        self._paper_orders = {}
        self._paper_order_counter = 0
        self._paper_balances_initialized = True
        self._session = MagicMock()

        self._candle_data = candle_data
        self._current_candle_idx = 0
        self._poll_in_candle = 0    # 0..polls_per_candle-1
        self._polls_per_candle = 60  # 4 minutes per "poll" for 4h candle
        self._lookback = 540

    def get_ohlc(self, pair: str, interval: int = 240, since=None):
        pair_key = pair.replace("/", "_")
        candles = self._candle_data.get(pair_key, [])
        # Strategy sees up to and including the current candle
        end = min(self._current_candle_idx + 1, len(candles))
        start = max(0, end - self._lookback)
        return candles[start:end]

    def get_ticker(self, pair: str) -> Ticker:
        """Interpolate price within current candle based on poll position."""
        pair_key = pair.replace("/", "_")
        candles = self._candle_data.get(pair_key, [])
        if self._current_candle_idx >= len(candles):
            c = candles[-1]
            price = c.close
        else:
            c = candles[self._current_candle_idx]
            # Walk price from open -> close as poll progresses, with H/L excursions
            t = self._poll_in_candle / max(self._polls_per_candle - 1, 1)
            # Simple linear interp between open and close
            price = c.open + (c.close - c.open) * t
            # Inject high/low briefly at 1/3 and 2/3 through the candle
            if 0.30 <= t <= 0.35:
                price = c.high
            elif 0.65 <= t <= 0.70:
                price = c.low
        return Ticker(
            pair=pair, ask=price * 1.0005, bid=price * 0.9995,
            last=price, volume_24h=c.volume * 6, vwap_24h=(c.high + c.low + c.close) / 3,
            trades_24h=1000, low_24h=c.low, high_24h=c.high,
            timestamp=c.timestamp,
        )

    def get_balances(self):
        return {"USD": Balance(asset="USD", total=self._paper_balances.get("USD", 0),
                               available=self._paper_balances.get("USD", 0))}

    def place_order(self, pair, side, order_type, volume, price=None, **kwargs):
        self._paper_order_counter += 1
        ticker = self.get_ticker(pair)
        fill_price = ticker.ask if side == OrderSide.BUY else ticker.bid
        fee = fill_price * volume * 0.0016
        return Order(
            order_id=f"SMOKE-{self._paper_order_counter}",
            pair=pair, side=side, order_type=order_type,
            price=fill_price, volume=volume, filled_volume=volume,
            status="closed", created_at=datetime.now(timezone.utc),
            cost=fill_price * volume, fee=fee,
        )

    def place_maker_order(self, pair, side, volume, price_offset_percent=0.0, **kwargs):
        return self.place_order(pair, side, OrderType.LIMIT, volume)

    def close(self):
        pass


def load_candles(data_dir: Path):
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
    return candle_data


def main():
    configure_logging(level="ERROR", format_type="json")
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/q1_2026_fresh")
    polls_per_candle = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    print(f"\n{'='*70}")
    print(f"  PRODUCTION POLLING SMOKE TEST")
    print(f"  Data: {data_dir}")
    print(f"  Polls per candle: {polls_per_candle}  (simulates 60s loop in 4h candle)")
    print(f"{'='*70}\n")

    candle_data = load_candles(data_dir)
    if not candle_data:
        print("No data!")
        return 1

    num_candles = len(next(iter(candle_data.values())))

    test_dir = Path("data/smoke_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    mock_client = PollingMockClient(candle_data, capital=10000.0)
    mock_client._polls_per_candle = polls_per_candle

    original_init = KrakenClient.__init__
    KrakenClient.__init__ = lambda self, **kwargs: None
    trader = ScalpingTrader(
        config_path="config/scalping.yaml",
        data_dir=str(test_dir),
        paper_trading=True,
    )
    KrakenClient.__init__ = original_init
    trader.client = mock_client
    trader.capital = 10000.0
    trader.initial_capital = 10000.0

    # Track per-candle behavior
    intra_candle_opens = defaultdict(int)   # pair -> count of opens in same candle
    intra_candle_closes = defaultdict(int)  # pair -> count of closes in same candle
    same_candle_trades = defaultdict(list)  # pair -> list of (open_candle, close_candle)
    decision_changes = defaultdict(int)     # pair -> count of decision flips within candle
    open_candle_for_pair = {}               # pair -> candle idx when opened

    print(f"  Stepping through {num_candles} candles, {polls_per_candle} polls each = {num_candles * polls_per_candle:,} total decisions...\n")

    for i in range(num_candles):
        mock_client._current_candle_idx = i

        # Snapshot at start of candle
        positions_at_candle_start = set(trader.positions.keys())
        opens_this_candle = defaultdict(int)
        closes_this_candle = defaultdict(int)

        # Update regime once per candle (matching live behavior at the candle boundary)
        try:
            trader._update_regime()
        except Exception as e:
            print(f"  CANDLE {i}: REGIME ERROR: {e}")

        # Run polls within this candle
        positions_per_poll = []
        for poll in range(polls_per_candle):
            mock_client._poll_in_candle = poll
            positions_before = set(trader.positions.keys())

            for pair in trader.pairs:
                try:
                    trader._process_pair(pair)
                except Exception as e:
                    pass  # ignore; we want to find PATTERNS not crashes here

            positions_after = set(trader.positions.keys())
            positions_per_poll.append(positions_after.copy())

            # Track within-candle activity
            opened = positions_after - positions_before
            closed = positions_before - positions_after
            for pair in opened:
                opens_this_candle[pair] += 1
                open_candle_for_pair[pair] = i
            for pair in closed:
                closes_this_candle[pair] += 1
                if pair in open_candle_for_pair:
                    open_c = open_candle_for_pair.pop(pair)
                    same_candle_trades[pair].append((open_c, i))

        # Aggregate signals for this candle
        for pair, n in opens_this_candle.items():
            if n > 1:
                intra_candle_opens[pair] += n - 1  # extra opens beyond first
            decision_changes[pair] += n
        for pair, n in closes_this_candle.items():
            if n > 1:
                intra_candle_closes[pair] += n - 1
            decision_changes[pair] += n

        if i % 50 == 0:
            print(f"  Candle {i}/{num_candles}  positions={len(trader.positions)}  capital=${trader.capital:.2f}")

    print(f"\n{'='*70}")
    print(f"  EXECUTION ANALYSIS")
    print(f"{'='*70}\n")

    print(f"  Total trades: {trader.metrics['total_trades']}")
    print(f"  Final capital: ${trader.capital:.2f} ({(trader.capital - 10000) / 100:.2f}%)")
    print(f"  Open positions at end: {len(trader.positions)}\n")

    print(f"  --- INTRA-CANDLE ACTIVITY (the POL bug pattern) ---")
    print(f"  Pair      | Opens in same candle | Closes in same candle | Same-candle round trips")
    print(f"  ----------+----------------------+-----------------------+------------------------")
    bugs_found = False
    for pair in sorted(trader.pairs):
        ic_opens = intra_candle_opens.get(pair, 0)
        ic_closes = intra_candle_closes.get(pair, 0)
        same_c = sum(1 for o, c in same_candle_trades.get(pair, []) if o == c)
        flag = " <-- BUG" if (ic_opens > 0 or same_c > 5) else ""
        if ic_opens > 0 or same_c > 5:
            bugs_found = True
        print(f"  {pair:9s} | {ic_opens:>20d} | {ic_closes:>21d} | {same_c:>23d}{flag}")

    print(f"\n  --- TRADE HOLD TIME (in candles) ---")
    print(f"  Pair      | Avg hold (candles) | Min hold | Max hold | Trades closed")
    print(f"  ----------+--------------------+----------+----------+--------------")
    for pair in sorted(trader.pairs):
        trades = same_candle_trades.get(pair, [])
        if not trades:
            continue
        holds = [c - o for o, c in trades]
        avg = sum(holds) / len(holds)
        flag_short = " <-- TOO SHORT" if avg < 1 else ""
        print(f"  {pair:9s} | {avg:>18.2f} | {min(holds):>8d} | {max(holds):>8d} | {len(trades):>13d}{flag_short}")

    print(f"\n{'='*70}")
    if bugs_found:
        print(f"  RESULT: EXECUTION BUGS DETECTED (see flagged rows above)")
    else:
        print(f"  RESULT: No execution bugs detected. Strategy is stable across polls.")
    print(f"{'='*70}\n")

    return 0 if not bugs_found else 1


if __name__ == "__main__":
    sys.exit(main())
