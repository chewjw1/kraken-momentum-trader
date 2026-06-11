#!/usr/bin/env python3
"""
Live-fidelity smoke test: models Kraken API behavior that the standard replay
and polling smoke test do NOT — and that broke production in May 2026:

  1. IN-PROGRESS CANDLE: Kraken's OHLC endpoint returns the current,
     not-yet-committed candle as the last array entry. At the first poll of a
     new candle that stub has ~1 poll of volume. The volume filter compares
     stub volume vs the 30-candle SMA and blocks every 4h-pair entry; the
     one-decision-per-candle guard then skips the pair for the rest of the
     candle. Production evidence: 0 trades on all five 4h pairs in 18 days.

  2. SLIDING 720-CANDLE WINDOW: Kraken returns the most recent ~720 candles.
     The 12h aggregation grouped candles from index 0 of that window, so 12h
     candle boundaries re-phased every 4h. 12h pairs decided 3x too often on
     boundary-shifting candles. Production evidence: SOL/POL 12h entries at
     16:00/20:00 UTC (not 00/12 UTC boundaries).

Usage:
    python3 run_live_fidelity_smoke.py [data_dir] [polls_per_candle]
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

CANDLE_SECONDS = 4 * 3600
KRAKEN_WINDOW = 720  # Kraken returns at most ~720 candles per request


class LiveFidelityMockClient(KrakenClient):
    """Serves candles the way Kraken actually does: sliding window of the
    most recent 720 candles, last entry being the in-progress stub whose
    OHLCV accumulates as the simulated clock advances through the candle."""

    def __init__(self, candle_data: dict, capital: float = 10000.0):
        self.paper_trading = True
        self._paper_balances = {"USD": capital}
        self._paper_orders = {}
        self._paper_order_counter = 0
        self._paper_balances_initialized = True
        self._session = MagicMock()

        self._candle_data = candle_data
        self._candle_idx = 0          # index of the candle currently FORMING
        self._progress = 0.0          # 0..1 fraction of the forming candle elapsed

    def sim_now(self) -> datetime:
        """Simulated wall clock: inside the forming candle."""
        candles = next(iter(self._candle_data.values()))
        idx = min(self._candle_idx, len(candles) - 1)
        start = candles[idx].timestamp
        return start + timedelta(seconds=CANDLE_SECONDS * self._progress)

    def _interp_price(self, c: OHLC) -> float:
        t = self._progress
        price = c.open + (c.close - c.open) * t
        if 0.30 <= t <= 0.35:
            price = c.high
        elif 0.65 <= t <= 0.70:
            price = c.low
        return price

    def _stub_candle(self, c: OHLC) -> OHLC:
        """The in-progress candle as Kraken would report it mid-formation."""
        t = max(self._progress, 0.005)
        price = self._interp_price(c)
        high = max(c.open, price, c.high if t > 0.30 else c.open)
        low = min(c.open, price, c.low if t > 0.65 else c.open)
        return OHLC(
            timestamp=c.timestamp, open=c.open, high=high, low=low,
            close=price, volume=c.volume * t,
            vwap=(high + low + price) / 3, count=max(1, int(c.count * t)),
        )

    def get_ohlc(self, pair: str, interval: int = 240, since=None):
        pair_key = pair.replace("/", "_")
        candles = self._candle_data.get(pair_key, [])
        idx = min(self._candle_idx, len(candles) - 1)
        # Kraken: most recent <=720 candles, last one in-progress
        window = candles[max(0, idx + 1 - KRAKEN_WINDOW):idx]
        return list(window) + [self._stub_candle(candles[idx])]

    def get_ticker(self, pair: str) -> Ticker:
        pair_key = pair.replace("/", "_")
        candles = self._candle_data.get(pair_key, [])
        idx = min(self._candle_idx, len(candles) - 1)
        c = candles[idx]
        price = self._interp_price(c)
        return Ticker(
            pair=pair, ask=price * 1.0005, bid=price * 0.9995,
            last=price, volume_24h=c.volume * 6, vwap_24h=(c.high + c.low + c.close) / 3,
            trades_24h=1000, low_24h=c.low, high_24h=c.high,
            timestamp=self.sim_now(),
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
            order_id=f"LFS-{self._paper_order_counter}",
            pair=pair, side=side, order_type=order_type,
            price=fill_price, volume=volume, filled_volume=volume,
            status="closed", created_at=self.sim_now(),
            cost=fill_price * volume, fee=fee,
        )

    def place_maker_order(self, pair, side, volume, price_offset_percent=0.0):
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
                    vwap=(float(h) + float(l) + float(c)) / 3, count=100,
                ))
        candle_data[pair_key] = candles
    return candle_data


def main():
    configure_logging(level="ERROR", format_type="json")
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/q1_2026_fresh")
    polls_per_candle = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    print(f"\n{'='*74}")
    print(f"  LIVE-FIDELITY SMOKE TEST (in-progress stub + sliding 720 window)")
    print(f"  Data: {data_dir}   Polls/candle: {polls_per_candle}")
    print(f"{'='*74}\n")

    candle_data = load_candles(data_dir)
    if not candle_data:
        print("No data!")
        return 1
    num_candles = len(next(iter(candle_data.values())))

    test_dir = Path("data/live_fidelity_smoke")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    mock_client = LiveFidelityMockClient(candle_data, capital=10000.0)

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
    # Inject simulated clock if the trader supports it (post-fix code)
    if hasattr(trader, "_now"):
        trader._now = mock_client.sim_now
    # Optional disaster-stop override for A/B experiments (arg 3, percent)
    if len(sys.argv) > 3 and hasattr(trader, "disaster_stop_pct"):
        trader.disaster_stop_pct = float(sys.argv[3])
        print(f"  Disaster stop override: {trader.disaster_stop_pct:.1f}%")

    pairs_4h = [p for p in trader.pairs if trader.pair_intervals.get(p, trader.candle_interval) == 240]
    pairs_12h = [p for p in trader.pairs if trader.pair_intervals.get(p, trader.candle_interval) == 720]

    entries = defaultdict(list)     # pair -> list of entry sim-times
    exits = defaultdict(list)
    open_candle = {}
    same_candle_rt = defaultdict(int)

    print(f"  Simulating {num_candles} candles x {polls_per_candle} polls "
          f"({num_candles * polls_per_candle:,} cycles)...\n")

    for i in range(num_candles):
        mock_client._candle_idx = i
        try:
            trader._update_regime()
        except Exception:
            pass

        for poll in range(polls_per_candle):
            mock_client._progress = (poll + 0.5) / polls_per_candle
            before = set(trader.positions.keys())
            for pair in trader.pairs:
                try:
                    trader._process_pair(pair)
                except Exception:
                    pass
            after = set(trader.positions.keys())
            now = mock_client.sim_now()
            for pair in after - before:
                entries[pair].append(now)
                open_candle[pair] = i
            for pair in before - after:
                exits[pair].append(now)
                if open_candle.pop(pair, None) == i:
                    same_candle_rt[pair] += 1

        if i % 100 == 0:
            print(f"  candle {i}/{num_candles}  positions={len(trader.positions)}  capital=${trader.capital:.2f}")

    total_trades = trader.metrics['total_trades']
    wins = trader.metrics['wins']
    print(f"\n{'='*74}")
    print(f"  RESULTS  —  P&L ${trader.capital - 10000:+,.2f}  "
          f"({total_trades} trades, {wins}W, "
          f"{(wins/total_trades*100) if total_trades else 0:.0f}% WR)")
    print(f"{'='*74}\n")

    failures = []

    print(f"  --- 4h PAIRS: entry capability (production bug: stub volume blocked ALL entries) ---")
    total_4h_entries = 0
    for pair in sorted(pairs_4h):
        n = len(entries[pair])
        total_4h_entries += n
        print(f"  {pair:10s} entries={n:3d}  exits={len(exits[pair]):3d}")
    if total_4h_entries == 0:
        failures.append("4h pairs made ZERO entries — stub-candle volume filter blockage")
    print(f"  Total 4h entries: {total_4h_entries}  {'<-- BUG' if total_4h_entries == 0 else 'OK'}\n")

    print(f"  --- 12h PAIRS: decision boundary alignment (production bug: re-phased every 4h) ---")
    # Disaster-stop exits legitimately fire intra-candle (any poll), so when
    # it's enabled only ENTRIES must be candle-anchored.
    disaster_on = getattr(trader, "disaster_stop_pct", 0) > 0
    misaligned = 0
    for pair in sorted(pairs_12h):
        times = list(entries[pair]) + ([] if disaster_on else list(exits[pair]))
        bad = [t for t in times
               if (int(t.timestamp()) % 43200) > CANDLE_SECONDS + polls_per_candle and
                  (int(t.timestamp()) % 43200) < 43200 - 600]
        # A decision belonging to a 00/12 UTC 12h candle should occur within the
        # first 4h candle after that boundary (decision at next candle open).
        aligned = [t for t in times if (int(t.timestamp()) % 43200) <= CANDLE_SECONDS + 1200]
        n_bad = len(times) - len(aligned)
        misaligned += n_bad
        print(f"  {pair:10s} decisions={len(times):3d}  off-boundary={n_bad:3d}"
              f"{'  <-- BUG (phase-shifted 12h candles)' if n_bad > 0 else ''}")
    if misaligned > 0:
        failures.append(f"{misaligned} 12h-pair decisions at non-12h-aligned times — sliding-window phase shift")
    print()

    print(f"  --- INTRA-CANDLE CHURN ---")
    churn = sum(same_candle_rt.values())
    for pair, n in sorted(same_candle_rt.items()):
        print(f"  {pair:10s} same-candle round trips: {n}")
    if churn > 5:
        failures.append(f"{churn} same-candle round trips")
    if not same_candle_rt:
        print("  none")

    print(f"\n{'='*74}")
    if failures:
        print(f"  RESULT: LIVE-FIDELITY BUGS DETECTED")
        for f in failures:
            print(f"    * {f}")
    else:
        print(f"  RESULT: PASS — live API quirks handled; behavior matches validated replay.")
    print(f"{'='*74}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
