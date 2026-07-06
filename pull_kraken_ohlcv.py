#!/usr/bin/env python3
"""
Pull real 4h OHLCV data from Kraken's public API.

Run on seedbox (or anywhere with internet access):
    python3 pull_kraken_ohlcv.py

Outputs CSV files to data/q4_2024/ and data/q1_2026/ in the format:
    timestamp,open,high,low,close,volume

No API key required — uses Kraken's public OHLC endpoint.
"""

import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

# Kraken pair names (API format → display format)
PAIRS = {
    "XXBTZUSD": "BTC_USD",
    "XETHZUSD": "ETH_USD",
    "SOLUSD": "SOL_USD",
    "XXRPZUSD": "XRP_USD",
    "AVAXUSD": "AVAX_USD",
    "LINKUSD": "LINK_USD",
    "NEARUSD": "NEAR_USD",
    "DOTUSD": "DOT_USD",
    "ATOMUSD": "ATOM_USD",
    "POLUSD": "POL_USD",
}

# Time periods to pull
PERIODS = {
    "q4_2024": {
        "start": datetime(2024, 10, 1, tzinfo=timezone.utc),
        "end": datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
    },
    "q1_2026": {
        "start": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "end": datetime(2026, 3, 31, 23, 59, 59, tzinfo=timezone.utc),
    },
}

INTERVAL = 240  # 4 hours in minutes
BASE_URL = "https://api.kraken.com/0/public/OHLC"


def fetch_ohlc(kraken_pair: str, since_ts: int, interval: int = 240) -> list:
    """Fetch OHLC data from Kraken public API."""
    url = f"{BASE_URL}?pair={kraken_pair}&interval={interval}&since={since_ts}"

    req = urllib.request.Request(url, headers={"User-Agent": "KrakenOHLC/1.0"})

    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())

            if data.get("error") and len(data["error"]) > 0:
                print(f"    API error: {data['error']}")
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                return []

            result = data.get("result", {})
            # The key might be the pair name or slightly different
            for key in result:
                if key != "last":
                    return result[key]
            return []

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            if attempt < 3:
                time.sleep(2 ** (attempt + 1))
            else:
                return []

    return []


def pull_pair(kraken_pair: str, display_name: str, start_dt: datetime,
              end_dt: datetime, output_dir: str) -> int:
    """Pull all 4h candles for a pair within a date range."""
    all_candles = []
    since = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    print(f"  {display_name}: pulling from {start_dt.date()} to {end_dt.date()}...")

    while since < end_ts:
        candles = fetch_ohlc(kraken_pair, since, INTERVAL)
        if not candles:
            break

        for c in candles:
            # Kraken OHLC format: [time, open, high, low, close, vwap, volume, count]
            ts = int(c[0])
            if ts > end_ts:
                break
            if ts >= int(start_dt.timestamp()):
                all_candles.append({
                    "timestamp": ts,
                    "open": c[1],
                    "high": c[2],
                    "low": c[3],
                    "close": c[4],
                    "volume": c[6],
                })

        # Move to after the last candle we got
        last_ts = int(candles[-1][0])
        if last_ts <= since:
            break
        since = last_ts + 1

        time.sleep(1)  # Rate limit: 1 request per second

    if not all_candles:
        print(f"    No data returned for {display_name}")
        return 0

    # Deduplicate by timestamp
    seen = set()
    unique = []
    for c in all_candles:
        if c["timestamp"] not in seen:
            seen.add(c["timestamp"])
            unique.append(c)
    unique.sort(key=lambda x: x["timestamp"])

    # Write CSV
    filepath = os.path.join(output_dir, f"{display_name}_4h.csv")
    with open(filepath, "w") as f:
        for c in unique:
            f.write(f"{c['timestamp']},{c['open']},{c['high']},{c['low']},"
                    f"{c['close']},{c['volume']}\n")

    first_close = unique[0]["close"]
    last_close = unique[-1]["close"]
    print(f"    {len(unique)} candles: ${first_close} -> ${last_close}")
    return len(unique)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Pull Kraken 4h OHLCV for the strategy's 10 pairs. "
                    "With no window args, pulls the hardcoded PERIODS "
                    "(q4_2024, q1_2026) — original behavior.")
    parser.add_argument("--days", type=int, default=0,
                        help="pull the last N days ending now (e.g. 90 for a "
                             "drift-monitor window)")
    parser.add_argument("--start", default=None, help="window start YYYY-MM-DD")
    parser.add_argument("--end", default=None,
                        help="window end YYYY-MM-DD (default: now)")
    parser.add_argument("--output", default=None,
                        help="output dir (required with --days/--start), "
                             "e.g. data/paper_window")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.days or args.start:
        if not args.output:
            parser.error("--output is required with --days/--start")
        now = datetime.now(timezone.utc)
        if args.start:
            start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            start = now - timedelta(days=args.days)
        end = (datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
               if args.end else now)
        periods = {os.path.basename(args.output.rstrip("/")): {"start": start, "end": end}}
        out_base = os.path.dirname(os.path.abspath(args.output)) or script_dir
    else:
        periods = PERIODS
        out_base = os.path.join(script_dir, "data")

    for period_name, period in periods.items():
        output_dir = os.path.join(out_base, period_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"  Pulling {period_name.upper()} data")
        print(f"  {period['start'].date()} to {period['end'].date()}")
        print(f"  Output: {output_dir}")
        print(f"{'=' * 60}")

        total_candles = 0
        for kraken_pair, display_name in PAIRS.items():
            count = pull_pair(kraken_pair, display_name, period["start"],
                              period["end"], output_dir)
            total_candles += count
            time.sleep(2)  # Extra delay between pairs

        print(f"\n  Total: {total_candles} candles for {period_name}")

    print(f"\n{'=' * 60}")
    print("  Done.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
