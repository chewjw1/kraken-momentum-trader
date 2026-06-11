#!/usr/bin/env python3
"""Fetch recent 4h OHLC from Kraken's public API into our CSV format.

Kraken returns the last ~720 candles; the final entry is the in-progress
candle and is dropped. Usage:
    python3 scripts/fetch_kraken_ohlc.py <out_dir> <since_epoch>
"""
import json, sys, time, urllib.request

PAIRS = {
    "XBTUSD": "BTC_USD", "ETHUSD": "ETH_USD", "SOLUSD": "SOL_USD",
    "AVAXUSD": "AVAX_USD", "NEARUSD": "NEAR_USD", "DOTUSD": "DOT_USD",
    "ATOMUSD": "ATOM_USD", "LINKUSD": "LINK_USD", "XRPUSD": "XRP_USD",
    "POLUSD": "POL_USD",
}
INTERVAL_S = 4 * 3600

def main():
    out_dir, since = sys.argv[1], int(sys.argv[2])
    import os
    os.makedirs(out_dir, exist_ok=True)
    now = time.time()
    counts = {}
    for kraken_name, our_name in PAIRS.items():
        url = f"https://api.kraken.com/0/public/OHLC?pair={kraken_name}&interval=240"
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
        if data.get("error"):
            print(f"  {kraken_name}: ERROR {data['error']}")
            continue
        key = [k for k in data["result"] if k != "last"][0]
        rows = []
        for c in data["result"][key]:
            ts = int(c[0])
            if ts < since:
                continue
            if ts + INTERVAL_S > now:
                continue  # in-progress candle
            rows.append((ts, c[1], c[2], c[3], c[4], c[6]))
        path = f"{out_dir}/{our_name}_4h.csv"
        with open(path, "w") as f:
            for row in rows:
                f.write(",".join(str(x) for x in row) + "\n")
        counts[our_name] = (len(rows), rows[0][0] if rows else 0, rows[-1][0] if rows else 0)
        print(f"  {our_name}: {len(rows)} candles  "
              f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(rows[0][0])) if rows else '-'} -> "
              f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(rows[-1][0])) if rows else '-'}")
        time.sleep(1.1)  # public API rate limit
    lens = {n for n, (l, _, _) in counts.items()}
    print(f"\nDone. {len(counts)} pairs.")

if __name__ == "__main__":
    main()
