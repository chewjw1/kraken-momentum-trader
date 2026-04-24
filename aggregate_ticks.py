#!/usr/bin/env python3
"""
Aggregate Kraken tick data into 4h OHLCV candles.

Input: Raw CSV files with format: timestamp,price,volume
Output: 4h OHLCV CSVs with format: timestamp,open,high,low,close,volume
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

INTERVAL = 4 * 3600  # 4 hours in seconds

PAIR_MAP = {
    "XBTUSD": "BTC_USD",
    "ETHUSD": "ETH_USD",
    "SOLUSD": "SOL_USD",
    "XRPUSD": "XRP_USD",
    "AVAXUSD": "AVAX_USD",
    "LINKUSD": "LINK_USD",
    "NEARUSD": "NEAR_USD",
    "DOTUSD": "DOT_USD",
    "ATOMUSD": "ATOM_USD",
    "POLUSD": "POL_USD",
}


def aggregate_to_4h(input_path: str) -> list[dict]:
    """Read tick data and aggregate into 4h OHLCV candles."""
    buckets = defaultdict(list)

    with open(input_path) as f:
        reader = csv.reader(f)
        for row in reader:
            ts = int(row[0].split('.')[0])
            price = float(row[1])
            volume = float(row[2])
            bucket = (ts // INTERVAL) * INTERVAL
            buckets[bucket].append((ts, price, volume))

    candles = []
    for bucket_ts in sorted(buckets.keys()):
        trades = buckets[bucket_ts]
        trades.sort(key=lambda t: t[0])
        prices = [t[1] for t in trades]
        volumes = [t[2] for t in trades]

        candles.append({
            'timestamp': bucket_ts,
            'open': prices[0],
            'high': max(prices),
            'low': min(prices),
            'close': prices[-1],
            'volume': sum(volumes),
        })

    return candles


def main():
    raw_dir = Path("/tmp/kraken_raw")
    out_dir = Path("data/q1_2025")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Aggregating Kraken tick data → 4h OHLCV candles")
    print("=" * 60)

    for raw_name, display_name in PAIR_MAP.items():
        raw_path = raw_dir / f"{raw_name}.csv"
        if not raw_path.exists():
            print(f"  {display_name}: MISSING ({raw_path})")
            continue

        print(f"  {display_name}: aggregating...", end=" ", flush=True)
        candles = aggregate_to_4h(str(raw_path))

        out_path = out_dir / f"{display_name}_4h.csv"
        with open(out_path, 'w', newline='') as f:
            w = csv.writer(f)
            for c in candles:
                w.writerow([c['timestamp'], c['open'], c['high'], c['low'],
                            c['close'], c['volume']])

        first = candles[0]
        last = candles[-1]
        change = (last['close'] - first['open']) / first['open'] * 100
        print(f"{len(candles)} candles, "
              f"${first['open']:.2f} → ${last['close']:.2f} ({change:+.1f}%)")

    print("=" * 60)
    print(f"  Output: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
