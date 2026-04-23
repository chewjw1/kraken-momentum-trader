"""
Generate realistic Q1 2026 OHLCV data using geometric Brownian motion
anchored to known price points from live trading records.

Price anchors from actual dashboard data:
- BTC: ~$93k Jan 1, ~$95k late Jan, ~$88k Feb, ~$83k mid-Mar, ~$78k late Mar
- ETH: ~$3400 Jan 1, ~$3300 late Jan, ~$2500 Feb, ~$2278 mid-Mar, ~$2330 late Mar
- SOL: ~$190 Jan 1, ~$150 Feb, ~$105 mid-Mar, ~$86 late Mar
- AVAX: ~$38 Jan 1, ~$24 Feb, ~$8.97 early Mar, ~$10.47 mid-Mar, ~$9.29 late Mar
- NEAR: ~$5.5 Jan 1, ~$3.8 Feb, ~$2.0 Mar, ~$1.39 late Mar
- ATOM: ~$7.5 Jan 1, ~$5 Feb, ~$2.5 Mar, ~$1.87 late Mar
- DOT: ~$7.5 Jan 1, ~$5 Feb, ~$1.55 early Mar, ~$1.30 late Mar
- LINK: ~$22 Jan 1, ~$17 Feb, ~$8.91 early Mar, ~$9.27 late Mar
- XRP: ~$2.3 Jan 1, ~$2.8 late Jan, ~$2.5 Feb, ~$1.7 Mar, ~$1.42 late Mar
- POL: ~$0.48 Jan 1, ~$0.30 Feb, ~$0.10 Mar, ~$0.09 late Mar
"""
import csv
import os
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# 4h candles from Jan 1 to Mar 31, 2026
start = datetime(2026, 1, 1)
end = datetime(2026, 3, 31, 23, 59, 59)
interval_hours = 4
n_candles = int((end - start).total_seconds() / (interval_hours * 3600))

# Price anchors: (day_of_quarter, price) — interpolated via log-space
# Day 0 = Jan 1, Day 89 = Mar 31
pairs = {
    'BTC_USD': {
        'anchors': [(0, 93000), (15, 95000), (25, 100000), (35, 92000), (45, 88000),
                    (55, 85000), (65, 84000), (70, 82000), (75, 83000), (80, 80000),
                    (85, 78000), (89, 78000)],
        'volatility': 0.018
    },
    'ETH_USD': {
        'anchors': [(0, 3400), (15, 3300), (25, 3500), (35, 2800), (45, 2500),
                    (55, 2400), (65, 2278), (70, 2173), (75, 2200), (80, 2300),
                    (85, 2330), (89, 2330)],
        'volatility': 0.022
    },
    'SOL_USD': {
        'anchors': [(0, 190), (15, 175), (25, 170), (35, 150), (45, 130),
                    (55, 105), (60, 95), (65, 86.61), (75, 82.24), (80, 84),
                    (85, 86), (89, 86)],
        'volatility': 0.028
    },
    'AVAX_USD': {
        'anchors': [(0, 38), (15, 30), (25, 26), (35, 22), (45, 16),
                    (55, 12), (60, 8.97), (65, 9.72), (70, 10.47),
                    (80, 9.5), (85, 9.29), (89, 9.29)],
        'volatility': 0.032
    },
    'NEAR_USD': {
        'anchors': [(0, 5.5), (15, 5.0), (25, 4.5), (35, 3.8), (45, 3.0),
                    (55, 2.5), (65, 2.0), (75, 1.6), (80, 1.5), (85, 1.39), (89, 1.39)],
        'volatility': 0.030
    },
    'ATOM_USD': {
        'anchors': [(0, 7.5), (15, 7.0), (25, 6.5), (35, 5.0), (45, 4.0),
                    (55, 3.2), (65, 2.5), (75, 2.0), (80, 1.9), (85, 1.87), (89, 1.87)],
        'volatility': 0.028
    },
    'DOT_USD': {
        'anchors': [(0, 7.5), (15, 7.0), (25, 6.0), (35, 4.5), (45, 3.5),
                    (55, 2.5), (60, 1.55), (70, 1.40), (75, 1.30), (80, 1.25),
                    (85, 1.25), (89, 1.25)],
        'volatility': 0.032
    },
    'LINK_USD': {
        'anchors': [(0, 22), (15, 19), (25, 17), (35, 14), (45, 12),
                    (55, 10), (60, 8.91), (62, 8.44), (65, 8.94), (67, 9.41),
                    (70, 9.85), (73, 8.74), (80, 9.0), (85, 9.27), (89, 9.27)],
        'volatility': 0.028
    },
    'XRP_USD': {
        'anchors': [(0, 2.3), (10, 2.5), (20, 2.8), (30, 2.6), (40, 2.4),
                    (50, 2.1), (60, 1.8), (70, 1.6), (80, 1.5), (85, 1.42), (89, 1.42)],
        'volatility': 0.025
    },
    'POL_USD': {
        'anchors': [(0, 0.48), (15, 0.40), (25, 0.35), (35, 0.30), (45, 0.22),
                    (55, 0.15), (60, 0.10), (65, 0.093), (70, 0.094), (75, 0.093),
                    (80, 0.092), (85, 0.09), (89, 0.09)],
        'volatility': 0.035
    },
}

os.makedirs('data/q1_2026', exist_ok=True)

for pair_name, config in pairs.items():
    anchors = config['anchors']
    vol = config['volatility']
    
    # Interpolate anchor prices across all candles using log-space
    anchor_days = np.array([a[0] for a in anchors], dtype=float)
    anchor_prices = np.array([a[1] for a in anchors], dtype=float)
    log_prices = np.log(anchor_prices)
    
    candle_days = np.linspace(0, 89, n_candles)
    trend = np.exp(np.interp(candle_days, anchor_days, log_prices))
    
    # Add mean-reverting noise around trend
    noise = np.zeros(n_candles)
    for i in range(1, n_candles):
        mean_revert = -0.1 * noise[i-1]  # Pull back toward trend
        noise[i] = noise[i-1] + mean_revert + np.random.normal(0, vol)
    
    closes = trend * np.exp(noise)
    
    # Generate OHLCV from closes
    rows = []
    for i in range(n_candles):
        ts = start + timedelta(hours=i * interval_hours)
        ts_unix = int(ts.timestamp())
        c = closes[i]
        
        # Intra-candle variation
        intra_vol = vol * 0.6
        o = c * np.exp(np.random.normal(0, intra_vol * 0.5))
        h = max(o, c) * (1 + abs(np.random.normal(0, intra_vol)))
        l = min(o, c) * (1 - abs(np.random.normal(0, intra_vol)))
        
        # Ensure OHLC consistency
        h = max(h, o, c)
        l = min(l, o, c)
        
        # Volume: higher in volatile periods
        base_vol = 1e6 * (c / 10000)  # Scale with price
        v = base_vol * (1 + abs(np.random.normal(0, 0.5)))
        
        rows.append([ts_unix, round(o, 6), round(h, 6), round(l, 6), round(c, 6), round(v, 2)])
    
    fpath = f'data/q1_2026/{pair_name}_4h.csv'
    with open(fpath, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    
    print(f'{pair_name}: {len(rows)} candles, ${rows[0][1]:.2f} -> ${rows[-1][4]:.2f}')

print(f'\nGenerated {len(pairs)} pair files in data/q1_2026/')
