"""
Historical data manager for backtesting.
Handles fetching and caching OHLC data from Kraken.
"""

import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

from ..exchange.kraken_client import KrakenClient, OHLC
from ..observability.logger import get_logger

logger = get_logger(__name__)


class HistoricalDataManager:
    """
    Manages historical OHLC data for backtesting.

    Features:
    - Fetches data from Kraken API with pagination
    - Caches data to disk to avoid repeated API calls
    - Rate limiting to respect API limits
    """

    # Kraken returns ~720 candles per request
    CANDLES_PER_REQUEST = 720
    RATE_LIMIT_DELAY = 1.0  # seconds between API calls

    def __init__(
        self,
        cache_dir: str = "data/cache/ohlc",
        use_cache: bool = True,
        client: Optional[KrakenClient] = None
    ):
        """
        Initialize the data manager.

        Args:
            cache_dir: Directory for caching OHLC data.
            use_cache: Whether to use disk caching.
            client: Optional KrakenClient instance (creates one if not provided).
        """
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

        # Create cache directory if it doesn't exist
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Use provided client or create one (paper trading mode for data only)
        self.client = client or KrakenClient(paper_trading=True)

    def get_ohlc_range(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        interval: int = 60,
        progress_callback: Optional[callable] = None
    ) -> List[OHLC]:
        """
        Get OHLC data for a date range.

        Args:
            pair: Trading pair (e.g., "BTC/USD").
            start: Start datetime.
            end: End datetime.
            interval: Candle interval in minutes.
            progress_callback: Optional callback(current, total) for progress updates.

        Returns:
            List of OHLC candles sorted by timestamp.
        """
        # Try to load from cache first
        if self.use_cache:
            cached_data = self._load_from_cache(pair, start, end, interval)
            if cached_data:
                logger.info(
                    f"Loaded {len(cached_data)} candles from cache",
                    pair=pair,
                    interval=interval
                )
                return cached_data

        # Fetch from API
        logger.info(
            f"Fetching OHLC data from Kraken API",
            pair=pair,
            start=start.isoformat(),
            end=end.isoformat(),
            interval=interval
        )

        all_candles: List[OHLC] = []
        current_since = int(start.timestamp())
        end_timestamp = int(end.timestamp())

        # Estimate total requests for progress
        interval_seconds = interval * 60
        total_candles_needed = (end_timestamp - current_since) // interval_seconds
        total_requests = max(1, total_candles_needed // self.CANDLES_PER_REQUEST)
        requests_made = 0
        consecutive_errors = 0
        max_consecutive_errors = 5  # Give up after 5 consecutive failures

        while current_since < end_timestamp:
            try:
                candles = self.client.get_ohlc(pair, interval=interval, since=current_since)

                if not candles:
                    logger.warning("No candles returned from API, breaking loop")
                    break

                consecutive_errors = 0  # Reset on success

                # Filter candles within our date range
                start_ts = start.timestamp()
                filtered = [c for c in candles if start_ts <= c.timestamp.timestamp() <= end_timestamp]
                all_candles.extend(filtered)

                # Update since to last candle timestamp
                last_timestamp = int(candles[-1].timestamp.timestamp())

                # If we're not making progress or have reached end, break
                if last_timestamp <= current_since or last_timestamp >= end_timestamp:
                    break

                current_since = last_timestamp + 1
                requests_made += 1

                # Progress callback
                if progress_callback:
                    progress_callback(requests_made, total_requests)

                # Log progress
                if requests_made % 10 == 0:
                    logger.info(
                        f"Fetched {len(all_candles)} candles so far",
                        requests=requests_made
                    )

                # Rate limiting
                time.sleep(self.RATE_LIMIT_DELAY)

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error fetching OHLC data: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        f"Kraken API failed after {max_consecutive_errors} "
                        f"consecutive errors, aborting fetch for {pair}"
                    )
                    break
                # Wait longer on error
                time.sleep(self.RATE_LIMIT_DELAY * 5)
                continue

        # Sort by timestamp and remove duplicates
        all_candles.sort(key=lambda c: c.timestamp)
        unique_candles = self._deduplicate_candles(all_candles)

        logger.info(
            f"Fetched {len(unique_candles)} unique candles",
            pair=pair,
            interval=interval,
            requests=requests_made
        )

        # Cache the data
        if self.use_cache and unique_candles:
            self._save_to_cache(pair, start, end, interval, unique_candles)

        return unique_candles

    def _deduplicate_candles(self, candles: List[OHLC]) -> List[OHLC]:
        """Remove duplicate candles based on timestamp."""
        seen = set()
        unique = []
        for candle in candles:
            ts = candle.timestamp.timestamp()
            if ts not in seen:
                seen.add(ts)
                unique.append(candle)
        return unique

    def _get_cache_filename(self, pair: str, start: datetime, end: datetime, interval: int) -> str:
        """Generate cache filename."""
        pair_clean = pair.replace("/", "_")
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        return f"{pair_clean}_{interval}m_{start_str}_{end_str}.json"

    def _load_from_cache(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        interval: int
    ) -> Optional[List[OHLC]]:
        """
        Try to load data from cache.

        Returns None if cache miss or invalid data.
        """
        cache_file = self.cache_dir / self._get_cache_filename(pair, start, end, interval)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            # Validate cache metadata
            if data.get("pair") != pair or data.get("interval") != interval:
                logger.warning("Cache metadata mismatch, ignoring cache")
                return None

            # Parse candles
            candles = []
            for c in data.get("candles", []):
                candles.append(OHLC(
                    timestamp=datetime.fromisoformat(c["timestamp"]),
                    open=c["open"],
                    high=c["high"],
                    low=c["low"],
                    close=c["close"],
                    vwap=c["vwap"],
                    volume=c["volume"],
                    count=c["count"]
                ))

            # Verify we have data for the requested range
            if not candles:
                return None

            cache_start = candles[0].timestamp
            cache_end = candles[-1].timestamp

            # Allow some tolerance for cache bounds
            start_ok = cache_start <= start + timedelta(hours=1)
            end_ok = cache_end >= end - timedelta(hours=1)

            if not (start_ok and end_ok):
                logger.info(
                    "Cache doesn't fully cover requested range",
                    cache_start=cache_start.isoformat(),
                    cache_end=cache_end.isoformat()
                )
                return None

            return candles

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        interval: int,
        candles: List[OHLC]
    ) -> None:
        """Save data to cache."""
        cache_file = self.cache_dir / self._get_cache_filename(pair, start, end, interval)

        try:
            data = {
                "pair": pair,
                "interval": interval,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "candle_count": len(candles),
                "candles": [
                    {
                        "timestamp": c.timestamp.isoformat(),
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "vwap": c.vwap,
                        "volume": c.volume,
                        "count": c.count
                    }
                    for c in candles
                ]
            }

            with open(cache_file, "w") as f:
                json.dump(data, f)

            logger.info(f"Cached {len(candles)} candles to {cache_file}")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def clear_cache(self, pair: Optional[str] = None) -> int:
        """
        Clear cached data.

        Args:
            pair: If specified, only clear cache for this pair.

        Returns:
            Number of files deleted.
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            if pair:
                pair_clean = pair.replace("/", "_")
                if not cache_file.name.startswith(pair_clean):
                    continue

            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info(f"Cleared {count} cache files")
        return count

    def get_available_cached_ranges(self) -> List[dict]:
        """
        Get list of available cached data ranges.

        Returns:
            List of dicts with pair, interval, start, end, candle_count.
        """
        ranges = []
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    ranges.append({
                        "pair": data.get("pair"),
                        "interval": data.get("interval"),
                        "start": data.get("start"),
                        "end": data.get("end"),
                        "candle_count": data.get("candle_count"),
                        "file": str(cache_file)
                    })
            except Exception:
                continue

        return ranges
