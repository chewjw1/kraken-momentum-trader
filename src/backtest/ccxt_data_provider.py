"""
Historical data providers for backtesting.
Supports multiple sources: CCXT (Binance), yfinance (Yahoo), and more.
"""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Dict

from ..exchange.kraken_client import OHLC
from ..observability.logger import get_logger

logger = get_logger(__name__)


# Pair mapping from Kraken format to yfinance format
KRAKEN_TO_YFINANCE = {
    "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD",
    "SOL/USD": "SOL-USD",
    "XRP/USD": "XRP-USD",
    "ADA/USD": "ADA-USD",
    "DOGE/USD": "DOGE-USD",
    "AVAX/USD": "AVAX-USD",
    "LINK/USD": "LINK-USD",
    "DOT/USD": "DOT-USD",
    "MATIC/USD": "MATIC-USD",
    "LTC/USD": "LTC-USD",
    "UNI/USD": "UNI-USD",
    "ATOM/USD": "ATOM-USD",
}

# Pair mapping from Kraken format to Binance format
KRAKEN_TO_BINANCE = {
    "BTC/USD": "BTC/USDT",
    "ETH/USD": "ETH/USDT",
    "SOL/USD": "SOL/USDT",
    "XRP/USD": "XRP/USDT",
    "ADA/USD": "ADA/USDT",
    "DOGE/USD": "DOGE/USDT",
    "AVAX/USD": "AVAX/USDT",
    "LINK/USD": "LINK/USDT",
    "DOT/USD": "DOT/USDT",
    "MATIC/USD": "MATIC/USDT",
    "LTC/USD": "LTC/USDT",
    "UNI/USD": "UNI/USDT",
    "ATOM/USD": "ATOM/USDT",
}


class CCXTDataProvider:
    """
    CCXT-based data provider using Binance for historical data.

    Features:
    - Fetches OHLCV data from Binance via CCXT
    - Support pagination for large date ranges
    - Caches to disk to avoid re-fetching
    - Maps Kraken pairs to Binance format
    """

    # Binance returns up to 1000 candles per request
    CANDLES_PER_REQUEST = 1000
    RATE_LIMIT_DELAY = 0.2  # seconds between API calls (Binance is generous)

    def __init__(
        self,
        cache_dir: str = "data/cache/ccxt",
        use_cache: bool = True
    ):
        """
        Initialize the CCXT data provider.

        Args:
            cache_dir: Directory for caching OHLCV data.
            use_cache: Whether to use disk caching.
        """
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self._exchange = None

        # Create cache directory if it doesn't exist
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def exchange(self):
        """Lazy-load the CCXT Binance exchange."""
        if self._exchange is None:
            try:
                import ccxt
                self._exchange = ccxt.binance({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                    }
                })
            except ImportError:
                raise ImportError(
                    "ccxt is required for CCXTDataProvider. "
                    "Install it with: pip install ccxt"
                )
        return self._exchange

    def map_pair(self, kraken_pair: str) -> str:
        """
        Map Kraken pair format to Binance format.

        Args:
            kraken_pair: Pair in Kraken format (e.g., "BTC/USD").

        Returns:
            Pair in Binance format (e.g., "BTC/USDT").
        """
        return KRAKEN_TO_BINANCE.get(kraken_pair, kraken_pair.replace("/USD", "/USDT"))

    def interval_to_timeframe(self, interval: int) -> str:
        """
        Convert interval in minutes to CCXT timeframe string.

        Args:
            interval: Candle interval in minutes.

        Returns:
            CCXT timeframe string (e.g., "1h").
        """
        mapping = {
            1: '1m',
            5: '5m',
            15: '15m',
            30: '30m',
            60: '1h',
            240: '4h',
            1440: '1d',
            10080: '1w',
        }
        return mapping.get(interval, '1h')

    def get_ohlc_range(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        interval: int = 60,
        progress_callback: Optional[callable] = None
    ) -> List[OHLC]:
        """
        Get OHLCV data for a date range from Binance.

        Args:
            pair: Trading pair in Kraken format (e.g., "BTC/USD").
            start: Start datetime.
            end: End datetime.
            interval: Candle interval in minutes.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            List of OHLC candles sorted by timestamp.
        """
        # Try to load from cache first
        if self.use_cache:
            cached_data = self._load_from_cache(pair, start, end, interval)
            if cached_data:
                logger.info(
                    f"Loaded {len(cached_data)} candles from CCXT cache",
                    pair=pair,
                    interval=interval
                )
                return cached_data

        # Map pair to Binance format
        binance_pair = self.map_pair(pair)
        timeframe = self.interval_to_timeframe(interval)

        logger.info(
            f"Fetching OHLCV data from Binance via CCXT",
            pair=pair,
            binance_pair=binance_pair,
            start=start.isoformat(),
            end=end.isoformat(),
            interval=interval
        )

        all_candles: List[OHLC] = []
        current_since = int(start.timestamp() * 1000)  # CCXT uses milliseconds
        end_timestamp = int(end.timestamp() * 1000)

        # Estimate total requests for progress
        interval_ms = interval * 60 * 1000
        total_candles_needed = (end_timestamp - current_since) // interval_ms
        total_requests = max(1, total_candles_needed // self.CANDLES_PER_REQUEST)
        requests_made = 0
        consecutive_errors = 0
        max_consecutive_errors = 5  # Give up after 5 consecutive failures

        while current_since < end_timestamp:
            try:
                # Fetch OHLCV data
                ohlcv = self.exchange.fetch_ohlcv(
                    binance_pair,
                    timeframe=timeframe,
                    since=current_since,
                    limit=self.CANDLES_PER_REQUEST
                )

                if not ohlcv:
                    logger.warning("No candles returned from Binance, breaking loop")
                    break

                consecutive_errors = 0  # Reset on success

                # Convert to OHLC dataclass format
                for candle in ohlcv:
                    timestamp_ms = candle[0]
                    if timestamp_ms >= end_timestamp:
                        break

                    all_candles.append(OHLC(
                        timestamp=datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc),
                        open=float(candle[1]),
                        high=float(candle[2]),
                        low=float(candle[3]),
                        close=float(candle[4]),
                        volume=float(candle[5]),
                        vwap=float(candle[4]),  # Use close as VWAP approximation
                        count=0  # Binance doesn't provide trade count in OHLCV
                    ))

                # Update since to last candle timestamp + 1
                last_timestamp = ohlcv[-1][0]

                # If we're not making progress or have reached end, break
                if last_timestamp <= current_since or last_timestamp >= end_timestamp:
                    break

                current_since = last_timestamp + 1
                requests_made += 1

                # Progress callback
                if progress_callback:
                    progress_callback(requests_made, total_requests)

                # Log progress periodically
                if requests_made % 20 == 0:
                    logger.info(
                        f"Fetched {len(all_candles)} candles so far",
                        requests=requests_made
                    )

                # Rate limiting
                time.sleep(self.RATE_LIMIT_DELAY)

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error fetching OHLCV data from Binance: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        f"Binance API unreachable after {max_consecutive_errors} "
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
            f"Fetched {len(unique_candles)} unique candles from Binance",
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
        return f"ccxt_{pair_clean}_{interval}m_{start_str}_{end_str}.json"

    def _load_from_cache(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        interval: int
    ) -> Optional[List[OHLC]]:
        """Try to load data from cache."""
        cache_file = self.cache_dir / self._get_cache_filename(pair, start, end, interval)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            # Validate cache metadata
            if data.get("pair") != pair or data.get("interval") != interval:
                logger.warning("CCXT cache metadata mismatch, ignoring cache")
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
            start_ok = cache_start <= start + timedelta(hours=2)
            end_ok = cache_end >= end - timedelta(hours=2)

            if not (start_ok and end_ok):
                logger.info(
                    "CCXT cache doesn't fully cover requested range",
                    cache_start=cache_start.isoformat(),
                    cache_end=cache_end.isoformat()
                )
                return None

            return candles

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to load CCXT cache: {e}")
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
                "source": "binance",
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
            logger.warning(f"Failed to save CCXT cache: {e}")

    def clear_cache(self, pair: Optional[str] = None) -> int:
        """
        Clear cached data.

        Args:
            pair: If specified, only clear cache for this pair.

        Returns:
            Number of files deleted.
        """
        count = 0
        for cache_file in self.cache_dir.glob("ccxt_*.json"):
            if pair:
                pair_clean = pair.replace("/", "_")
                if pair_clean not in cache_file.name:
                    continue

            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info(f"Cleared {count} CCXT cache files")
        return count

    def get_available_pairs(self) -> List[str]:
        """
        Get list of available trading pairs from Binance.

        Returns:
            List of available Kraken-format pairs.
        """
        try:
            markets = self.exchange.load_markets()
            available = []
            for kraken_pair in KRAKEN_TO_BINANCE.keys():
                binance_pair = KRAKEN_TO_BINANCE[kraken_pair]
                if binance_pair in markets:
                    available.append(kraken_pair)
            return available
        except Exception as e:
            logger.error(f"Error loading markets: {e}")
            return list(KRAKEN_TO_BINANCE.keys())


class YFinanceDataProvider:
    """
    Yahoo Finance data provider using yfinance library.

    Provides 5+ years of historical crypto data.
    Note: Only supports daily candles (no hourly/minute data).
    """

    def __init__(
        self,
        cache_dir: str = "data/cache/yfinance",
        use_cache: bool = True
    ):
        """
        Initialize the yfinance data provider.

        Args:
            cache_dir: Directory for caching OHLCV data.
            use_cache: Whether to use disk caching.
        """
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self._yf = None

        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def yf(self):
        """Lazy-load yfinance."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                raise ImportError(
                    "yfinance is required for YFinanceDataProvider. "
                    "Install it with: pip install yfinance"
                )
        return self._yf

    def map_pair(self, kraken_pair: str) -> str:
        """Map Kraken pair to yfinance ticker."""
        return KRAKEN_TO_YFINANCE.get(kraken_pair, kraken_pair.replace("/", "-"))

    def get_ohlc_range(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        interval: int = 1440,  # Default to daily (yfinance only supports daily for crypto)
        progress_callback: Optional[callable] = None
    ) -> List[OHLC]:
        """
        Get OHLCV data for a date range from Yahoo Finance.

        Args:
            pair: Trading pair in Kraken format (e.g., "BTC/USD").
            start: Start datetime.
            end: End datetime.
            interval: Candle interval in minutes (only 1440/daily supported).
            progress_callback: Optional callback (not used, instant fetch).

        Returns:
            List of OHLC candles sorted by timestamp.
        """
        # yfinance only supports daily data for crypto
        if interval != 1440:
            logger.warning(
                f"yfinance only supports daily candles for crypto. "
                f"Requested {interval}m, using daily (1440m)."
            )
            interval = 1440

        # Try cache first
        if self.use_cache:
            cached = self._load_from_cache(pair, start, end, interval)
            if cached:
                logger.info(f"Loaded {len(cached)} candles from yfinance cache", pair=pair)
                return cached

        # Map to yfinance ticker
        ticker = self.map_pair(pair)

        logger.info(
            f"Fetching OHLCV data from Yahoo Finance",
            pair=pair,
            ticker=ticker,
            start=start.isoformat(),
            end=end.isoformat()
        )

        try:
            # Fetch data
            stock = self.yf.Ticker(ticker)
            df = stock.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d"
            )

            if df.empty:
                logger.warning(f"No data returned from yfinance for {ticker}")
                return []

            # Convert to OHLC objects
            candles = []
            for idx, row in df.iterrows():
                # Handle timezone - yfinance returns timezone-aware index
                ts = idx.to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                candles.append(OHLC(
                    timestamp=ts,
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row['Volume']),
                    vwap=float(row['Close']),  # Use close as VWAP approximation
                    count=0
                ))

            logger.info(f"Fetched {len(candles)} daily candles from yfinance", pair=pair)

            # Cache the data
            if self.use_cache and candles:
                self._save_to_cache(pair, start, end, interval, candles)

            return candles

        except Exception as e:
            logger.error(f"Error fetching data from yfinance: {e}")
            return []

    def _get_cache_filename(self, pair: str, start: datetime, end: datetime, interval: int) -> str:
        """Generate cache filename."""
        pair_clean = pair.replace("/", "_")
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        return f"yfinance_{pair_clean}_{interval}m_{start_str}_{end_str}.json"

    def _load_from_cache(
        self,
        pair: str,
        start: datetime,
        end: datetime,
        interval: int
    ) -> Optional[List[OHLC]]:
        """Try to load data from cache."""
        cache_file = self.cache_dir / self._get_cache_filename(pair, start, end, interval)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)

            if data.get("pair") != pair:
                return None

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

            return candles if candles else None

        except Exception as e:
            logger.warning(f"Failed to load yfinance cache: {e}")
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
                "source": "yfinance",
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
            logger.warning(f"Failed to save yfinance cache: {e}")

    def get_available_pairs(self) -> List[str]:
        """Get list of available trading pairs."""
        return list(KRAKEN_TO_YFINANCE.keys())

    def clear_cache(self, pair: Optional[str] = None) -> int:
        """Clear cached data."""
        count = 0
        for cache_file in self.cache_dir.glob("yfinance_*.json"):
            if pair:
                pair_clean = pair.replace("/", "_")
                if pair_clean not in cache_file.name:
                    continue
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info(f"Cleared {count} yfinance cache files")
        return count
