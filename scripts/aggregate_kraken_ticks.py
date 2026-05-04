#!/usr/bin/env python3
"""
Aggregate Kraken tick-level trade data into 4h OHLCV candles.

Input: CSV from Kraken bulk download (Price, Volume, Timestamp, Type, Misc, TradeID)
Output: CSV in our format: timestamp(int), open, high, low, close, volume
"""
import csv, os, sys
from datetime import datetime, timezone

INTERVAL_SECONDS = 4 * 3600  # 4h candles

def aggregate(input_csv, output_csv):
    candles = {}  # bucket_start -> [open, high, low, close, volume, first_ts]
    with open(input_csv) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            try:
                price = float(row[0])
                volume = float(row[1])
                ts = float(row[2])
            except (ValueError, IndexError):
                continue
            bucket = int(ts // INTERVAL_SECONDS) * INTERVAL_SECONDS
            if bucket not in candles:
                candles[bucket] = [price, price, price, price, volume, ts]
            else:
                c = candles[bucket]
                c[1] = max(c[1], price)  # high
                c[2] = min(c[2], price)  # low
                c[3] = price             # close (last trade)
                c[4] += volume

    with open(output_csv, 'w') as f:
        writer = csv.writer(f)
        for bucket in sorted(candles):
            o, h, l, c, v, _ = candles[bucket]
            writer.writerow([bucket, f"{o:.6f}".rstrip('0').rstrip('.'),
                             f"{h:.6f}".rstrip('0').rstrip('.'),
                             f"{l:.6f}".rstrip('0').rstrip('.'),
                             f"{c:.6f}".rstrip('0').rstrip('.'),
                             f"{v:.4f}".rstrip('0').rstrip('.')])

# Map Kraken filename -> our pair filename
PAIR_MAP = {
    'XBTUSD.csv': 'BTC_USD_4h.csv',
    'ETHUSD.csv': 'ETH_USD_4h.csv',
    'SOLUSD.csv': 'SOL_USD_4h.csv',
    'AVAXUSD.csv': 'AVAX_USD_4h.csv',
    'NEARUSD.csv': 'NEAR_USD_4h.csv',
    'DOTUSD.csv': 'DOT_USD_4h.csv',
    'ATOMUSD.csv': 'ATOM_USD_4h.csv',
    'LINKUSD.csv': 'LINK_USD_4h.csv',
    'XRPUSD.csv': 'XRP_USD_4h.csv',
    'POLUSD.csv': 'POL_USD_4h.csv',
}

def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    for src_name, dst_name in PAIR_MAP.items():
        src = os.path.join(input_dir, src_name)
        dst = os.path.join(output_dir, dst_name)
        if not os.path.exists(src):
            print(f"  SKIP {src_name} (not found)")
            continue
        size_mb = os.path.getsize(src) / (1024*1024)
        print(f"  {src_name} -> {dst_name} ({size_mb:.1f} MB)... ", end='', flush=True)
        aggregate(src, dst)
        with open(dst) as f:
            n = sum(1 for _ in f)
        print(f"{n} candles")

if __name__ == '__main__':
    main()
