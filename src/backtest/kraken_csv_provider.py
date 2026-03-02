"""
Data providers for historical backtesting:
1. KrakenCSVProvider - Loads Kraken's downloadable OHLCVT ZIP/CSV files
2. CryptoCompareProvider - Fetches from CryptoCompare free API (no key needed)

Kraken CSV format (per support article):
  Columns: timestamp, open, high, low, close, volume, trades
  ZIP contains CSVs for intervals: 1, 5, 15, 30, 60, 240, 720, 1440 minutes
  Files named like: XBTUSD_240.csv (pair_interval.csv)
"""

import csv
import io
import json
import time
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Callable

import requests

from ..exchange.kraken_client import OHLC
from ..observability.logger import get_logger

logger = get_logger(__name__)

# Kraken uses internal pair names in their CSV files
# Map from our format (BTC/USD) to possible Kraken CSV filenames
PAIR_TO_KRAKEN_NAMES = {
    "BTC/USD": ["XBTUSD", "BTCUSD", "XXBTZUSD"],
    "ETH/USD": ["ETHUSD", "XETHZUSD"],
    "SOL/USD": ["SOLUSD"],
    "XRP/USD": ["XRPUSD", "XXRPZUSD"],
    "LINK/USD": ["LINKUSD"],
    "AVAX/USD": ["AVAXUSD"],
    "DOT/USD": ["DOTUSD"],
    "POL/USD": ["POLUSD", "MATICUSD"],  # Polygon rebranded from MATIC
    "ATOM/USD": ["ATOMUSD"],
    "NEAR/USD": ["NEARUSD"],
    "ADA/USD": ["ADAUSD"],
    "DOGE/USD": ["DOGEUSD", "XDGUSD"],
    "LTC/USD": ["LTCUSD", "XLTCZUSD"],
}

# CryptoCompare symbol mapping
PAIR_TO_CC_SYMBOL = {
    "BTC/USD": "BTC",
    "ETH/USD": "ETH",
    "SOL/USD": "SOL",
    "XRP/USD": "XRP",
    "LINK/USD": "LINK",
    "AVAX/USD": "AVAX",
    "DOT/USD": "DOT",
    "POL/USD": "POL",
    "ATOM/USD": "ATOM",
    "NEAR/USD": "NEAR",
    "ADA/USD": "ADA",
    "DOGE/USD": "DOGE",
    "LTC/USD": "LTC",
}


class KrakenCSVProvider:
    """
    Loads Kraken's downloadable historical OHLCVT data from ZIP or CSV files.

    Expected directory structure:
        data/kraken_historical/
            Kraken_OHLCVT.zip          (single ZIP with all pairs)
            or individual ZIPs/CSVs:
            XBTUSD_240.csv
            ETHUSD.zip
            etc.

    CSV format: timestamp,open,high,low,close,volume,trades
    No header row.
    """

    def __init__(self, data_dir: str = "data/kraken_historical"):
        self.data_dir = Path(data_dir)
        self._zip_cache = {}  # Cache opened ZIP contents

    def get_ohlc_range(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        interval: int = 240,
        progress_callback: Optional[Callable] = None,
    ) -> List[OHLC]:
        """
        Load OHLCV data for a pair from Kraken CSV files.

        Args:
            pair: Trading pair (e.g. "BTC/USD")
            start: Start datetime
            end: End datetime
            interval: Candle interval in minutes (1, 5, 15, 30, 60, 240, 720, 1440)
            progress_callback: Optional callback(current, total)

        Returns:
            List of OHLC candles sorted by timestamp
        """
        if not self.data_dir.exists():
            logger.warning(f"Kraken historical data directory not found: {self.data_dir}")
            return []

        kraken_names = PAIR_TO_KRAKEN_NAMES.get(pair, [pair.replace("/", "")])
        candles = []

        # Strategy 1: Look for individual CSV files (PAIRNAME_INTERVAL.csv)
        for name in kraken_names:
            csv_file = self.data_dir / f"{name}_{interval}.csv"
            if csv_file.exists():
                logger.info(f"Loading {pair} from CSV: {csv_file.name}")
                candles = self._parse_csv_file(csv_file, start, end)
                if candles:
                    break

        # Strategy 2: Look for pair-specific ZIP files
        if not candles:
            for name in kraken_names:
                for zip_file in self.data_dir.glob("*.zip"):
                    candles = self._load_from_zip(zip_file, name, interval, start, end)
                    if candles:
                        logger.info(f"Loading {pair} from ZIP: {zip_file.name}")
                        break
                if candles:
                    break

        if not candles:
            logger.warning(f"No Kraken CSV data found for {pair} at {interval}m interval")
            return []

        # Filter to requested range
        start_ts = start.timestamp()
        end_ts = end.timestamp()
        filtered = [c for c in candles if start_ts <= c.timestamp.timestamp() <= end_ts]
        filtered.sort(key=lambda c: c.timestamp)

        logger.info(f"Loaded {len(filtered)} candles for {pair} from Kraken CSV ({interval}m)")

        if progress_callback and filtered:
            progress_callback(len(filtered), len(filtered))

        return filtered

    def _load_from_zip(
        self,
        zip_path: Path,
        kraken_name: str,
        interval: int,
        start: datetime,
        end: datetime,
    ) -> List[OHLC]:
        """Extract and parse a CSV from inside a ZIP file."""
        target_filename = f"{kraken_name}_{interval}.csv"

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # List all files and find matching one
                names = zf.namelist()

                # Try exact match first
                match = None
                for name in names:
                    basename = name.split("/")[-1]  # Handle subdirectories in ZIP
                    if basename.lower() == target_filename.lower():
                        match = name
                        break

                # Try partial match (some ZIPs have different naming)
                if not match:
                    for name in names:
                        basename = name.split("/")[-1].lower()
                        if kraken_name.lower() in basename and f"_{interval}" in basename:
                            match = name
                            break

                if not match:
                    return []

                with zf.open(match) as f:
                    text = io.TextIOWrapper(f, encoding="utf-8")
                    return self._parse_csv_stream(text, start, end)

        except (zipfile.BadZipFile, KeyError) as e:
            logger.warning(f"Error reading ZIP {zip_path}: {e}")
            return []

    def _parse_csv_file(
        self, csv_path: Path, start: datetime, end: datetime
    ) -> List[OHLC]:
        """Parse a standalone CSV file."""
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                return self._parse_csv_stream(f, start, end)
        except Exception as e:
            logger.error(f"Error reading CSV {csv_path}: {e}")
            return []

    def _parse_csv_stream(
        self, stream, start: datetime, end: datetime
    ) -> List[OHLC]:
        """Parse CSV data from a file-like stream."""
        candles = []
        start_ts = start.timestamp()
        end_ts = end.timestamp()

        reader = csv.reader(stream)
        for row in reader:
            if len(row) < 6:
                continue

            try:
                ts = int(row[0])

                # Quick filter before creating objects
                if ts < start_ts or ts > end_ts:
                    continue

                candles.append(OHLC(
                    timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    vwap=float(row[4]),  # Use close as VWAP approximation
                    count=int(row[6]) if len(row) > 6 else 0,
                ))
            except (ValueError, IndexError):
                continue  # Skip malformed rows

        return candles

    def list_available_pairs(self, interval: int = 240) -> List[str]:
        """List pairs available in the historical data directory."""
        if not self.data_dir.exists():
            return []

        available = set()

        # Check standalone CSVs
        for csv_file in self.data_dir.glob(f"*_{interval}.csv"):
            name = csv_file.stem.replace(f"_{interval}", "")
            for pair, names in PAIR_TO_KRAKEN_NAMES.items():
                if name.upper() in [n.upper() for n in names]:
                    available.add(pair)

        # Check ZIPs
        for zip_path in self.data_dir.glob("*.zip"):
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    for entry in zf.namelist():
                        basename = entry.split("/")[-1]
                        if f"_{interval}" in basename and basename.endswith(".csv"):
                            name = basename.split(f"_{interval}")[0]
                            for pair, names in PAIR_TO_KRAKEN_NAMES.items():
                                if name.upper() in [n.upper() for n in names]:
                                    available.add(pair)
            except zipfile.BadZipFile:
                continue

        return sorted(available)


class CryptoCompareProvider:
    """
    CryptoCompare free API data provider.

    - No API key required (rate-limited to ~50 calls/min)
    - histohour endpoint: up to 2000 candles per request, paginatable
    - Supports all major crypto pairs
    - Aggregates hourly data into any interval (4h, etc.)
    """

    BASE_URL = "https://min-api.cryptocompare.com/data/v2"
    MAX_CANDLES_PER_REQUEST = 2000
    RATE_LIMIT_DELAY = 1.2  # Stay well under 50 req/min

    def __init__(self, cache_dir: str = "data/cache/cryptocompare"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_ohlc_range(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        interval: int = 240,
        progress_callback: Optional[Callable] = None,
    ) -> List[OHLC]:
        """
        Fetch OHLCV data from CryptoCompare.

        Fetches hourly data and aggregates to the requested interval.

        Args:
            pair: Trading pair (e.g. "BTC/USD")
            start: Start datetime
            end: End datetime
            interval: Target interval in minutes (60, 240, 1440)
            progress_callback: Optional callback(current, total)
        """
        # Check cache first
        cached = self._load_cache(pair, start, end, interval)
        if cached:
            logger.info(f"Loaded {len(cached)} candles from CryptoCompare cache", pair=pair)
            if progress_callback:
                progress_callback(len(cached), len(cached))
            return cached

        symbol = PAIR_TO_CC_SYMBOL.get(pair)
        if not symbol:
            logger.warning(f"No CryptoCompare symbol mapping for {pair}")
            return []

        # Fetch hourly data (finest granularity with good depth)
        hourly_candles = self._fetch_hourly(symbol, start, end, progress_callback)
        if not hourly_candles:
            return []

        # Aggregate to requested interval
        if interval == 60:
            result = hourly_candles
        else:
            result = self._aggregate_candles(hourly_candles, interval)

        # Cache
        if result:
            self._save_cache(pair, start, end, interval, result)

        logger.info(f"Fetched {len(result)} candles for {pair} from CryptoCompare ({interval}m)")
        return result

    def _fetch_hourly(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        progress_callback: Optional[Callable] = None,
    ) -> List[OHLC]:
        """Fetch hourly candles with pagination."""
        all_candles = []
        end_ts = int(end.timestamp())
        start_ts = int(start.timestamp())

        # Estimate total requests needed
        total_hours = (end_ts - start_ts) / 3600
        total_requests = max(1, int(total_hours / self.MAX_CANDLES_PER_REQUEST) + 1)
        requests_made = 0

        current_to_ts = end_ts

        while current_to_ts > start_ts:
            try:
                resp = requests.get(
                    f"{self.BASE_URL}/histohour",
                    params={
                        "fsym": symbol,
                        "tsym": "USD",
                        "limit": self.MAX_CANDLES_PER_REQUEST,
                        "toTs": current_to_ts,
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("Response") != "Success":
                    logger.error(f"CryptoCompare error: {data.get('Message', 'unknown')}")
                    break

                raw_candles = data.get("Data", {}).get("Data", [])
                if not raw_candles:
                    break

                for c in raw_candles:
                    ts = c["time"]
                    if ts < start_ts:
                        continue
                    if ts > end_ts:
                        continue
                    # Skip zero-volume candles (no trades happened)
                    if c.get("volumefrom", 0) == 0 and c.get("volumeto", 0) == 0:
                        continue

                    all_candles.append(OHLC(
                        timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
                        open=float(c["open"]),
                        high=float(c["high"]),
                        low=float(c["low"]),
                        close=float(c["close"]),
                        volume=float(c.get("volumefrom", 0)),
                        vwap=float(c["close"]),  # Approximation
                        count=0,
                    ))

                # Move pagination window back
                earliest_ts = raw_candles[0]["time"]
                if earliest_ts >= current_to_ts:
                    break  # No progress
                current_to_ts = earliest_ts - 1

                requests_made += 1
                if progress_callback:
                    progress_callback(requests_made, total_requests)

                # Rate limit
                time.sleep(self.RATE_LIMIT_DELAY)

            except requests.RequestException as e:
                logger.error(f"CryptoCompare API error: {e}")
                break

        # Sort and deduplicate
        all_candles.sort(key=lambda c: c.timestamp)
        seen = set()
        unique = []
        for c in all_candles:
            ts = int(c.timestamp.timestamp())
            if ts not in seen:
                seen.add(ts)
                unique.append(c)

        return unique

    def _aggregate_candles(
        self, hourly: List[OHLC], target_interval: int
    ) -> List[OHLC]:
        """Aggregate hourly candles into larger intervals."""
        if not hourly:
            return []

        candles_per_bar = target_interval // 60
        result = []

        # Group by interval boundaries
        for i in range(0, len(hourly), candles_per_bar):
            group = hourly[i : i + candles_per_bar]
            if not group:
                continue

            result.append(OHLC(
                timestamp=group[0].timestamp,
                open=group[0].open,
                high=max(c.high for c in group),
                low=min(c.low for c in group),
                close=group[-1].close,
                volume=sum(c.volume for c in group),
                vwap=group[-1].close,
                count=sum(c.count for c in group),
            ))

        return result

    def _cache_key(self, pair: str, start: datetime, end: datetime, interval: int) -> str:
        pair_clean = pair.replace("/", "_")
        return f"cc_{pair_clean}_{interval}m_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.json"

    def _load_cache(
        self, pair: str, start: datetime, end: datetime, interval: int
    ) -> Optional[List[OHLC]]:
        cache_file = self.cache_dir / self._cache_key(pair, start, end, interval)
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            candles = []
            for c in data.get("candles", []):
                candles.append(OHLC(
                    timestamp=datetime.fromisoformat(c["timestamp"]),
                    open=c["open"], high=c["high"], low=c["low"], close=c["close"],
                    volume=c["volume"], vwap=c["vwap"], count=c["count"],
                ))
            return candles if len(candles) > 100 else None
        except Exception:
            return None

    def _save_cache(
        self, pair: str, start: datetime, end: datetime, interval: int, candles: List[OHLC]
    ) -> None:
        cache_file = self.cache_dir / self._cache_key(pair, start, end, interval)
        try:
            data = {
                "pair": pair, "interval": interval, "source": "cryptocompare",
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "candle_count": len(candles),
                "candles": [
                    {"timestamp": c.timestamp.isoformat(), "open": c.open, "high": c.high,
                     "low": c.low, "close": c.close, "vwap": c.vwap, "volume": c.volume,
                     "count": c.count}
                    for c in candles
                ],
            }
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save CryptoCompare cache: {e}")
