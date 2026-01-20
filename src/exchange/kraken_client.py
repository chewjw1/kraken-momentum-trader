"""
Kraken API client wrapper.
Handles authentication, rate limiting, and API interactions.
"""

import base64
import hashlib
import hmac
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

import requests

from ..config.platform import get_kraken_credentials
from ..config.settings import get_settings
from ..observability.logger import get_logger
from .rate_limiter import MultiEndpointRateLimiter

logger = get_logger(__name__)


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop-loss"
    TAKE_PROFIT = "take-profit"
    STOP_LOSS_LIMIT = "stop-loss-limit"
    TAKE_PROFIT_LIMIT = "take-profit-limit"


@dataclass
class Ticker:
    """Price ticker data."""
    pair: str
    ask: float
    bid: float
    last: float
    volume_24h: float
    vwap_24h: float
    trades_24h: int
    low_24h: float
    high_24h: float
    timestamp: datetime


@dataclass
class Balance:
    """Account balance for an asset."""
    asset: str
    total: float
    available: float


@dataclass
class Order:
    """Order information."""
    order_id: str
    pair: str
    side: OrderSide
    order_type: OrderType
    price: Optional[float]
    volume: float
    filled_volume: float
    status: str
    created_at: datetime
    cost: float = 0.0
    fee: float = 0.0


@dataclass
class OHLC:
    """OHLC candlestick data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    vwap: float
    volume: float
    count: int


class KrakenAPIError(Exception):
    """Kraken API error."""

    def __init__(self, message: str, errors: list[str] = None):
        super().__init__(message)
        self.errors = errors or []


class KrakenClient:
    """
    Kraken exchange API client.

    Handles both public and private API endpoints with
    proper authentication, rate limiting, and error handling.
    """

    BASE_URL = "https://api.kraken.com"
    API_VERSION = "0"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper_trading: bool = True
    ):
        """
        Initialize the Kraken client.

        Args:
            api_key: Kraken API key. If None, loaded from environment.
            api_secret: Kraken API secret. If None, loaded from environment.
            paper_trading: If True, simulate orders instead of executing.
        """
        self.paper_trading = paper_trading

        # Load credentials
        if api_key and api_secret:
            self._api_key = api_key
            self._api_secret = api_secret
        else:
            self._api_key, self._api_secret = get_kraken_credentials()

        # Validate credentials for non-paper trading
        if not paper_trading and (not self._api_key or not self._api_secret):
            raise ValueError("API credentials required for live trading")

        # Initialize rate limiter
        settings = get_settings()
        self._rate_limiter = MultiEndpointRateLimiter(
            public_calls_per_minute=60,
            private_calls_per_minute=settings.exchange.rate_limit_calls_per_minute
        )

        # Retry settings
        self._retry_attempts = settings.exchange.retry_attempts
        self._retry_delay = settings.exchange.retry_delay_seconds

        # Session for connection pooling
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "KrakenMomentumTrader/1.0"
        })

        # Paper trading state
        self._paper_balances: dict[str, float] = {
            "USD": 10000.0,
            "BTC": 0.0,
            "ETH": 0.0,
            "XRP": 0.0,
            "SOL": 0.0,
            "DOGE": 0.0,
            "ADA": 0.0,
            "MATIC": 0.0,
        }
        self._paper_orders: dict[str, Order] = {}
        self._paper_order_counter = 0

        logger.info(
            "Kraken client initialized",
            paper_trading=paper_trading,
            has_credentials=bool(self._api_key)
        )

    def _get_signature(self, url_path: str, data: dict, nonce: str) -> str:
        """
        Generate API signature for authenticated requests.

        Args:
            url_path: API endpoint path.
            data: Request data.
            nonce: Request nonce.

        Returns:
            Base64 encoded signature.
        """
        post_data = urllib.parse.urlencode(data)
        encoded = (nonce + post_data).encode()
        message = url_path.encode() + hashlib.sha256(encoded).digest()

        signature = hmac.new(
            base64.b64decode(self._api_secret),
            message,
            hashlib.sha512
        )

        return base64.b64encode(signature.digest()).decode()

    def _request(
        self,
        method: str,
        endpoint: str,
        data: dict = None,
        private: bool = False
    ) -> dict:
        """
        Make an API request with retry logic.

        Args:
            method: HTTP method.
            endpoint: API endpoint.
            data: Request data.
            private: Whether this is a private endpoint.

        Returns:
            API response data.

        Raises:
            KrakenAPIError: If the request fails.
        """
        url = f"{self.BASE_URL}/{self.API_VERSION}/{endpoint}"
        data = data or {}

        # Rate limiting
        endpoint_type = "private" if private else "public"
        self._rate_limiter.acquire_sync(endpoint_type)

        headers = {}

        if private:
            if not self._api_key or not self._api_secret:
                raise KrakenAPIError("API credentials required for private endpoints")

            nonce = str(int(time.time() * 1000))
            data["nonce"] = nonce

            url_path = f"/{self.API_VERSION}/{endpoint}"
            signature = self._get_signature(url_path, data, nonce)

            headers["API-Key"] = self._api_key
            headers["API-Sign"] = signature

        # Retry logic
        last_error = None
        for attempt in range(self._retry_attempts):
            try:
                if method.upper() == "GET":
                    response = self._session.get(url, params=data, headers=headers)
                else:
                    response = self._session.post(url, data=data, headers=headers)

                response.raise_for_status()
                result = response.json()

                if result.get("error"):
                    errors = result["error"]
                    # Check for rate limit errors
                    if any("EAPI:Rate limit" in e for e in errors):
                        logger.warning("Rate limit hit, waiting before retry")
                        time.sleep(self._retry_delay * (attempt + 1))
                        continue
                    raise KrakenAPIError(f"API error: {errors}", errors)

                return result.get("result", {})

            except requests.RequestException as e:
                last_error = e
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self._retry_attempts})",
                    error=str(e),
                    endpoint=endpoint
                )
                if attempt < self._retry_attempts - 1:
                    time.sleep(self._retry_delay * (attempt + 1))

        raise KrakenAPIError(f"Request failed after {self._retry_attempts} attempts: {last_error}")

    # =========================================================================
    # Public API Methods
    # =========================================================================

    def get_ticker(self, pair: str) -> Ticker:
        """
        Get current ticker data for a trading pair.

        Args:
            pair: Trading pair (e.g., "BTC/USD").

        Returns:
            Ticker with current price data.
        """
        kraken_pair = self._normalize_pair(pair)
        result = self._request("GET", "public/Ticker", {"pair": kraken_pair})

        if not result:
            raise KrakenAPIError(f"No ticker data for {pair}")

        # Kraken returns data keyed by their pair name
        data = list(result.values())[0]

        return Ticker(
            pair=pair,
            ask=float(data["a"][0]),
            bid=float(data["b"][0]),
            last=float(data["c"][0]),
            volume_24h=float(data["v"][1]),
            vwap_24h=float(data["p"][1]),
            trades_24h=int(data["t"][1]),
            low_24h=float(data["l"][1]),
            high_24h=float(data["h"][1]),
            timestamp=datetime.now(timezone.utc)
        )

    def get_ohlc(
        self,
        pair: str,
        interval: int = 60,
        since: Optional[int] = None
    ) -> list[OHLC]:
        """
        Get OHLC candlestick data.

        Args:
            pair: Trading pair.
            interval: Candlestick interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600).
            since: Return data since this timestamp.

        Returns:
            List of OHLC data.
        """
        kraken_pair = self._normalize_pair(pair)
        params = {"pair": kraken_pair, "interval": interval}
        if since:
            params["since"] = since

        result = self._request("GET", "public/OHLC", params)

        # Data is keyed by pair name
        pair_key = [k for k in result.keys() if k != "last"][0]
        data = result[pair_key]

        ohlc_list = []
        for candle in data:
            ohlc_list.append(OHLC(
                timestamp=datetime.fromtimestamp(candle[0], tz=timezone.utc),
                open=float(candle[1]),
                high=float(candle[2]),
                low=float(candle[3]),
                close=float(candle[4]),
                vwap=float(candle[5]),
                volume=float(candle[6]),
                count=int(candle[7])
            ))

        return ohlc_list

    def get_server_time(self) -> datetime:
        """
        Get Kraken server time.

        Returns:
            Server time as datetime.
        """
        result = self._request("GET", "public/Time")
        return datetime.fromtimestamp(result["unixtime"], tz=timezone.utc)

    # =========================================================================
    # Private API Methods
    # =========================================================================

    def get_balances(self) -> dict[str, Balance]:
        """
        Get account balances.

        Returns:
            Dictionary mapping asset to Balance.
        """
        if self.paper_trading:
            return self._get_paper_balances()

        result = self._request("POST", "private/Balance", private=True)

        balances = {}
        for asset, amount in result.items():
            # Normalize asset names (XXBT -> BTC, ZUSD -> USD)
            normalized = self._normalize_asset(asset)
            balances[normalized] = Balance(
                asset=normalized,
                total=float(amount),
                available=float(amount)  # Kraken doesn't separate available
            )

        return balances

    def get_open_orders(self) -> list[Order]:
        """
        Get open orders.

        Returns:
            List of open orders.
        """
        if self.paper_trading:
            return [o for o in self._paper_orders.values() if o.status == "open"]

        result = self._request("POST", "private/OpenOrders", private=True)

        orders = []
        for order_id, data in result.get("open", {}).items():
            orders.append(self._parse_order(order_id, data))

        return orders

    def place_order(
        self,
        pair: str,
        side: OrderSide,
        order_type: OrderType,
        volume: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        validate_only: bool = False
    ) -> Order:
        """
        Place an order.

        Args:
            pair: Trading pair.
            side: Buy or sell.
            order_type: Order type.
            volume: Order volume in base currency.
            price: Limit price (for limit orders).
            stop_price: Stop price (for stop orders).
            validate_only: If True, validate but don't submit.

        Returns:
            The placed Order.
        """
        logger.trade(
            action="place_order",
            pair=pair,
            side=side.value,
            price=price or 0.0,
            amount=volume,
            order_type=order_type.value,
            paper_trading=self.paper_trading
        )

        if self.paper_trading:
            return self._place_paper_order(pair, side, order_type, volume, price)

        kraken_pair = self._normalize_pair(pair)

        data = {
            "pair": kraken_pair,
            "type": side.value,
            "ordertype": order_type.value,
            "volume": str(volume),
        }

        if price and order_type in (OrderType.LIMIT, OrderType.STOP_LOSS_LIMIT, OrderType.TAKE_PROFIT_LIMIT):
            data["price"] = str(price)

        if stop_price and order_type in (OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT, OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_LIMIT):
            data["price2"] = str(stop_price)

        if validate_only:
            data["validate"] = True

        result = self._request("POST", "private/AddOrder", data, private=True)

        order_id = result["txid"][0]

        return Order(
            order_id=order_id,
            pair=pair,
            side=side,
            order_type=order_type,
            price=price,
            volume=volume,
            filled_volume=0.0,
            status="open",
            created_at=datetime.now(timezone.utc)
        )

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            True if cancelled successfully.
        """
        logger.info(f"Cancelling order {order_id}", paper_trading=self.paper_trading)

        if self.paper_trading:
            if order_id in self._paper_orders:
                self._paper_orders[order_id].status = "cancelled"
                return True
            return False

        result = self._request(
            "POST", "private/CancelOrder",
            {"txid": order_id},
            private=True
        )

        return result.get("count", 0) > 0

    def get_trade_history(self, start: Optional[int] = None, end: Optional[int] = None) -> list[dict]:
        """
        Get trade history.

        Args:
            start: Start timestamp.
            end: End timestamp.

        Returns:
            List of trade records.
        """
        if self.paper_trading:
            return []

        data = {}
        if start:
            data["start"] = start
        if end:
            data["end"] = end

        result = self._request("POST", "private/TradesHistory", data, private=True)

        return list(result.get("trades", {}).values())

    # =========================================================================
    # Paper Trading Simulation
    # =========================================================================

    def _get_paper_balances(self) -> dict[str, Balance]:
        """Get simulated paper trading balances."""
        balances = {}
        for asset, amount in self._paper_balances.items():
            balances[asset] = Balance(
                asset=asset,
                total=amount,
                available=amount
            )
        return balances

    def _place_paper_order(
        self,
        pair: str,
        side: OrderSide,
        order_type: OrderType,
        volume: float,
        price: Optional[float]
    ) -> Order:
        """Simulate order placement in paper trading mode."""
        # Get current price if not specified
        if price is None:
            ticker = self.get_ticker(pair)
            price = ticker.ask if side == OrderSide.BUY else ticker.bid

        # Calculate cost
        cost = volume * price
        fee = cost * 0.0026  # Assume 0.26% fee

        # Check balance
        base, quote = pair.split("/")

        if side == OrderSide.BUY:
            if self._paper_balances.get(quote, 0) < cost + fee:
                raise KrakenAPIError("Insufficient balance for paper trade")
            self._paper_balances[quote] = self._paper_balances.get(quote, 0) - cost - fee
            self._paper_balances[base] = self._paper_balances.get(base, 0) + volume
        else:
            if self._paper_balances.get(base, 0) < volume:
                raise KrakenAPIError("Insufficient balance for paper trade")
            self._paper_balances[base] = self._paper_balances.get(base, 0) - volume
            self._paper_balances[quote] = self._paper_balances.get(quote, 0) + cost - fee

        # Create order record
        self._paper_order_counter += 1
        order_id = f"PAPER-{self._paper_order_counter:06d}"

        order = Order(
            order_id=order_id,
            pair=pair,
            side=side,
            order_type=order_type,
            price=price,
            volume=volume,
            filled_volume=volume,  # Instant fill for market orders
            status="closed",
            created_at=datetime.now(timezone.utc),
            cost=cost,
            fee=fee
        )

        self._paper_orders[order_id] = order

        logger.info(
            f"Paper order executed: {side.value} {volume} {pair} @ {price}",
            order_id=order_id,
            cost=cost,
            fee=fee
        )

        return order

    def set_paper_balance(self, asset: str, amount: float) -> None:
        """
        Set paper trading balance for testing.

        Args:
            asset: Asset symbol.
            amount: Balance amount.
        """
        if not self.paper_trading:
            raise ValueError("Can only set balances in paper trading mode")
        self._paper_balances[asset] = amount

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _normalize_pair(self, pair: str) -> str:
        """
        Convert pair format to Kraken format.

        Args:
            pair: Pair like "BTC/USD".

        Returns:
            Kraken format like "XBTUSD".
        """
        # Remove slash and handle common conversions
        normalized = pair.replace("/", "")

        # Kraken uses XBT instead of BTC
        normalized = normalized.replace("BTC", "XBT")

        return normalized

    def _normalize_asset(self, asset: str) -> str:
        """
        Normalize Kraken asset name to standard format.

        Args:
            asset: Kraken asset name (e.g., "XXBT", "ZUSD").

        Returns:
            Normalized name (e.g., "BTC", "USD").
        """
        # Remove Kraken prefixes
        if asset.startswith("X") or asset.startswith("Z"):
            asset = asset[1:]

        # Handle XBT -> BTC
        if asset == "XBT":
            asset = "BTC"

        return asset

    def _parse_order(self, order_id: str, data: dict) -> Order:
        """Parse order data from API response."""
        descr = data.get("descr", {})

        return Order(
            order_id=order_id,
            pair=descr.get("pair", ""),
            side=OrderSide(descr.get("type", "buy")),
            order_type=OrderType(descr.get("ordertype", "market")),
            price=float(descr.get("price", 0)) if descr.get("price") else None,
            volume=float(data.get("vol", 0)),
            filled_volume=float(data.get("vol_exec", 0)),
            status=data.get("status", "unknown"),
            created_at=datetime.fromtimestamp(data.get("opentm", 0), tz=timezone.utc),
            cost=float(data.get("cost", 0)),
            fee=float(data.get("fee", 0))
        )

    def close(self) -> None:
        """Close the client session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
