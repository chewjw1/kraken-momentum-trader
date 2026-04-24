#!/usr/bin/env python3
"""
Build Q4 2024 (bull market) OHLCV 4h CSVs from real web-sourced data.

BTC: Real daily closes from StatMuse (Oct-Dec 2024).
Altcoins: Known monthly anchor prices from public data, interpolated to daily.
All daily prices split into 6x 4h candles with realistic intraday variation.
"""

import csv
import math
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

random.seed(42)

# BTC real daily closes from StatMuse (reverse chronological → chronological)
BTC_DAILY = {
    # October 2024
    "2024-10-07": 62236.66, "2024-10-08": 62131.97, "2024-10-09": 60582.10,
    "2024-10-10": 60274.50, "2024-10-11": 62445.09, "2024-10-12": 63193.02,
    "2024-10-13": 62851.37, "2024-10-14": 66046.13, "2024-10-15": 67041.11,
    "2024-10-16": 67612.72, "2024-10-17": 67399.83, "2024-10-18": 68418.79,
    "2024-10-19": 68362.73, "2024-10-20": 69001.71, "2024-10-21": 67367.85,
    "2024-10-22": 67361.40, "2024-10-23": 66432.20, "2024-10-24": 68161.05,
    "2024-10-25": 66642.41, "2024-10-26": 67014.70, "2024-10-27": 67929.30,
    "2024-10-28": 69907.75, "2024-10-29": 72720.49, "2024-10-30": 72339.54,
    "2024-10-31": 70215.19,
    # November 2024
    "2024-11-06": 75639.08, "2024-11-07": 75904.86, "2024-11-08": 76545.48,
    "2024-11-09": 76778.87, "2024-11-10": 80474.19, "2024-11-11": 88701.49,
    "2024-11-12": 87955.81, "2024-11-13": 90584.17, "2024-11-14": 87250.43,
    "2024-11-15": 91066.01, "2024-11-16": 90558.48, "2024-11-17": 89845.85,
    "2024-11-18": 90542.64, "2024-11-19": 92343.79, "2024-11-20": 94339.50,
    "2024-11-21": 98504.73, "2024-11-22": 98997.66, "2024-11-23": 97777.28,
    "2024-11-24": 98013.82, "2024-11-25": 93102.30, "2024-11-26": 91985.32,
    "2024-11-27": 95962.53, "2024-11-28": 95652.47, "2024-11-29": 97461.52,
    "2024-11-30": 96449.06,
    # December 2024
    "2024-12-07": 99923.34, "2024-12-08": 101236.01, "2024-12-09": 97432.72,
    "2024-12-10": 96675.43, "2024-12-11": 101173.03, "2024-12-12": 100043.00,
    "2024-12-13": 101459.25, "2024-12-14": 101372.97, "2024-12-15": 104298.70,
    "2024-12-16": 106029.72, "2024-12-17": 106140.60, "2024-12-18": 100041.54,
    "2024-12-19": 97490.95, "2024-12-20": 97755.93, "2024-12-21": 97224.73,
    "2024-12-22": 95104.93, "2024-12-23": 94686.24, "2024-12-24": 98676.10,
    "2024-12-25": 99299.19, "2024-12-26": 95795.52, "2024-12-27": 94164.86,
    "2024-12-28": 95163.93, "2024-12-29": 93530.23, "2024-12-30": 92643.21,
    "2024-12-31": 93429.20,
}

# Fill missing BTC days via linear interpolation
def fill_btc_daily():
    dates = sorted(BTC_DAILY.keys())
    start = datetime.strptime("2024-10-01", "%Y-%m-%d")
    end = datetime.strptime("2024-12-31", "%Y-%m-%d")

    # Add approximate start prices (Oct 1-6 from known data)
    extra = {
        "2024-10-01": 63300, "2024-10-02": 62000, "2024-10-03": 61200,
        "2024-10-04": 61800, "2024-10-05": 62100, "2024-10-06": 62300,
        # Nov 1-5
        "2024-11-01": 70500, "2024-11-02": 69500, "2024-11-03": 69000,
        "2024-11-04": 69200, "2024-11-05": 70300,
        # Dec 1-6
        "2024-12-01": 97000, "2024-12-02": 95700, "2024-12-03": 95800,
        "2024-12-04": 97200, "2024-12-05": 96400, "2024-12-06": 99800,
    }
    all_prices = {**BTC_DAILY, **extra}

    # Interpolate any remaining gaps
    known_dates = sorted(all_prices.keys())
    result = {}
    for i, d in enumerate(known_dates):
        result[d] = all_prices[d]

    # Fill any remaining gaps between known dates
    current = start
    while current <= end:
        ds = current.strftime("%Y-%m-%d")
        if ds not in result:
            # Find nearest known before and after
            before_d, before_p = None, None
            after_d, after_p = None, None
            for kd in known_dates:
                if kd <= ds:
                    before_d, before_p = kd, all_prices[kd]
                if kd > ds and after_d is None:
                    after_d, after_p = kd, all_prices[kd]
            if before_p and after_p:
                bd = datetime.strptime(before_d, "%Y-%m-%d")
                ad = datetime.strptime(after_d, "%Y-%m-%d")
                frac = (current - bd).days / max((ad - bd).days, 1)
                result[ds] = before_p + (after_p - before_p) * frac
            elif before_p:
                result[ds] = before_p
        current += timedelta(days=1)

    return dict(sorted(result.items()))


# Altcoin anchor prices (known monthly opens/closes from public data)
# Format: list of (date_str, price) tuples — will interpolate between them
ALTCOIN_ANCHORS = {
    "ETH/USD": [
        ("2024-10-01", 2600), ("2024-10-15", 2650), ("2024-10-31", 2520),
        ("2024-11-06", 2700), ("2024-11-11", 3380), ("2024-11-15", 3180),
        ("2024-11-22", 3450), ("2024-11-30", 3610),
        ("2024-12-06", 4000), ("2024-12-10", 3850), ("2024-12-17", 4100),
        ("2024-12-20", 3480), ("2024-12-25", 3500), ("2024-12-31", 3350),
    ],
    "SOL/USD": [
        ("2024-10-01", 153), ("2024-10-15", 155), ("2024-10-31", 165),
        ("2024-11-06", 170), ("2024-11-11", 210), ("2024-11-18", 235),
        ("2024-11-22", 263), ("2024-11-30", 230),
        ("2024-12-06", 230), ("2024-12-15", 220), ("2024-12-22", 195),
        ("2024-12-31", 190),
    ],
    "XRP/USD": [
        ("2024-10-01", 0.55), ("2024-10-31", 0.52),
        ("2024-11-06", 0.55), ("2024-11-15", 0.95), ("2024-11-22", 1.40),
        ("2024-11-30", 1.80), ("2024-12-05", 2.45),
        ("2024-12-10", 2.25), ("2024-12-17", 2.60), ("2024-12-25", 2.20),
        ("2024-12-31", 2.10),
    ],
    "AVAX/USD": [
        ("2024-10-01", 27.0), ("2024-10-31", 28.5),
        ("2024-11-06", 29.5), ("2024-11-12", 36.0), ("2024-11-22", 45.0),
        ("2024-11-30", 48.0), ("2024-12-06", 55.0), ("2024-12-17", 52.0),
        ("2024-12-25", 42.0), ("2024-12-31", 40.0),
    ],
    "LINK/USD": [
        ("2024-10-01", 11.5), ("2024-10-31", 12.0),
        ("2024-11-06", 12.5), ("2024-11-12", 15.0), ("2024-11-22", 17.5),
        ("2024-11-30", 18.0), ("2024-12-06", 23.0), ("2024-12-12", 28.0),
        ("2024-12-17", 30.0), ("2024-12-25", 24.0), ("2024-12-31", 22.0),
    ],
    "NEAR/USD": [
        ("2024-10-01", 4.80), ("2024-10-31", 5.00),
        ("2024-11-06", 5.20), ("2024-11-12", 6.20), ("2024-11-22", 7.50),
        ("2024-11-30", 7.00), ("2024-12-06", 8.20),
        ("2024-12-17", 7.40), ("2024-12-25", 5.60), ("2024-12-31", 5.30),
    ],
    "DOT/USD": [
        ("2024-10-01", 4.30), ("2024-10-31", 4.50),
        ("2024-11-06", 4.80), ("2024-11-12", 5.80), ("2024-11-22", 8.50),
        ("2024-11-30", 9.50), ("2024-12-06", 10.50),
        ("2024-12-17", 9.50), ("2024-12-25", 7.50), ("2024-12-31", 7.00),
    ],
    "ATOM/USD": [
        ("2024-10-01", 4.40), ("2024-10-31", 4.60),
        ("2024-11-06", 5.00), ("2024-11-12", 6.50), ("2024-11-22", 9.00),
        ("2024-11-30", 9.50), ("2024-12-06", 10.50),
        ("2024-12-17", 9.00), ("2024-12-25", 7.00), ("2024-12-31", 6.50),
    ],
    "POL/USD": [
        ("2024-10-01", 0.38), ("2024-10-31", 0.36),
        ("2024-11-06", 0.37), ("2024-11-12", 0.43), ("2024-11-22", 0.55),
        ("2024-11-30", 0.60), ("2024-12-06", 0.68),
        ("2024-12-17", 0.60), ("2024-12-25", 0.48), ("2024-12-31", 0.45),
    ],
}


def interpolate_daily(anchors: list[tuple[str, float]]) -> dict[str, float]:
    """Interpolate between anchor points to get daily prices."""
    result = {}
    for i in range(len(anchors) - 1):
        d1 = datetime.strptime(anchors[i][0], "%Y-%m-%d")
        d2 = datetime.strptime(anchors[i + 1][0], "%Y-%m-%d")
        p1, p2 = anchors[i][1], anchors[i + 1][1]
        days = (d2 - d1).days
        for j in range(days):
            d = d1 + timedelta(days=j)
            frac = j / max(days, 1)
            price = p1 + (p2 - p1) * frac
            noise = random.gauss(0, abs(p2 - p1) * 0.02)
            result[d.strftime("%Y-%m-%d")] = price + noise
    result[anchors[-1][0]] = anchors[-1][1]
    return dict(sorted(result.items()))


def daily_to_4h_candles(daily_prices: dict[str, float], volatility_pct: float = 1.5) -> list:
    """Split daily closes into 6x 4h OHLCV candles per day."""
    candles = []
    dates = sorted(daily_prices.keys())

    for i, date_str in enumerate(dates):
        close_target = daily_prices[date_str]
        prev_close = daily_prices[dates[i - 1]] if i > 0 else close_target

        day_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        # Generate 6 intraday path points
        day_change = (close_target - prev_close) / prev_close
        current = prev_close

        for h in range(6):
            ts = day_dt + timedelta(hours=h * 4)

            # Drift toward target + noise
            remaining = 6 - h
            drift = (close_target - current) / remaining if remaining > 0 else 0
            noise = random.gauss(0, close_target * volatility_pct / 100)

            c = current + drift + noise
            intra_vol = close_target * volatility_pct / 100 * 0.5
            o = current
            h_price = max(o, c) + abs(random.gauss(0, intra_vol * 0.5))
            l_price = min(o, c) - abs(random.gauss(0, intra_vol * 0.5))

            vol = random.uniform(50, 500) * (close_target / 100)

            candles.append({
                'timestamp': int(ts.timestamp()),
                'open': round(o, 6),
                'high': round(h_price, 6),
                'low': round(l_price, 6),
                'close': round(c, 6),
                'volume': round(vol, 2),
            })
            current = c

    return candles


def write_csv(candles: list, filepath: str):
    with open(filepath, 'w', newline='') as f:
        w = csv.writer(f)
        for c in candles:
            w.writerow([c['timestamp'], c['open'], c['high'], c['low'], c['close'], c['volume']])


def main():
    out_dir = Path(__file__).parent

    # BTC from real daily data
    btc_daily = fill_btc_daily()
    print(f"BTC: {len(btc_daily)} daily prices, {list(btc_daily.values())[0]:.0f} -> {list(btc_daily.values())[-1]:.0f}")
    btc_candles = daily_to_4h_candles(btc_daily, volatility_pct=2.0)
    write_csv(btc_candles, str(out_dir / "BTC_USD_4h.csv"))
    print(f"  -> {len(btc_candles)} 4h candles written")

    # Altcoins from anchor interpolation
    for pair, anchors in ALTCOIN_ANCHORS.items():
        symbol = pair.replace("/", "_")
        daily = interpolate_daily(anchors)
        first_p = list(daily.values())[0]
        last_p = list(daily.values())[-1]
        change = (last_p - first_p) / first_p * 100
        print(f"{pair}: {len(daily)} days, ${first_p:.2f} -> ${last_p:.2f} ({change:+.1f}%)")

        vol_pct = 2.5 if first_p < 1 else (2.0 if first_p < 10 else 1.5)
        candles = daily_to_4h_candles(daily, volatility_pct=vol_pct)
        write_csv(candles, str(out_dir / f"{symbol}_4h.csv"))
        print(f"  -> {len(candles)} 4h candles written")

    print("\nDone! Q4 2024 bull market data generated.")


if __name__ == "__main__":
    main()
