"""
Kraken API client wrapper.
Handles authentication, rate limiting, and API interactions.
"""

import base64
import hashlib
import hmac
import math
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
    """Kraken API error.

    `network` is True when the failure was at the transport layer (timeout,
    connection reset) — meaning Kraken MAY have received and executed the
    request. API-level errors (network=False) are definitive: the exchange
    rejected the request and nothing executed.
    """

    def __init__(self, message: str, errors: list[str] = None, network: bool = False):
        super().__init__(message)
        self.errors = errors or []
        self.network = network


class KrakenClient:
    """
    Kraken exchange API client.

    Handles both public and private API endpoints with
    proper authentication, rate limiting, and error handling.
    """

    BASE_URL = "https://api.kraken.com"
    API_VERSION = "0"

    # int32 tag attached to every order this bot places (AddOrder `userref`).
    # Lets startup reconciliation cancel OUR stale orders while leaving any
    # manually placed orders on the account untouched, and enables recovery
    # of orders whose AddOrder response was lost to a network failure.
    BOT_USERREF = 84119231

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper_trading: bool = True,
        paper_trading_capital: float = 10000.0
    ):
        """
        Initialize the Kraken client.

        Args:
            api_key: Kraken API key. If None, loaded from environment.
            api_secret: Kraken API secret. If None, loaded from environment.
            paper_trading: If True, simulate orders instead of executing.
            paper_trading_capital: Starting USD balance for paper trading.
        """
        self.paper_trading = paper_trading
        self._paper_trading_capital = paper_trading_capital

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

        # Paper trading state - will be initialized from real balances if credentials available
        self._paper_balances: dict[str, float] = {
            "USD": paper_trading_capital,
            "BTC": 0.0,
            "ETH": 0.0,
            "XRP": 0.0,
            "SOL": 0.0,
            "DOGE": 0.0,
            "ADA": 0.0,
            "AVAX": 0.0,
        }
        self._paper_orders: dict[str, Order] = {}
        self._paper_order_counter = 0
        self._paper_balances_initialized = False

        # Per-pair trading constraints from AssetPairs (price/volume precision,
        # order minimums, allowed margin leverage). Lazy cache.
        self._pair_info: dict[str, dict] = {}

        # If paper trading with valid credentials, fetch real balances
        if paper_trading and self._api_key and self._api_secret:
            self._init_paper_balances_from_real()

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
        private: bool = False,
        idempotent: bool = True
    ) -> dict:
        """
        Make an API request with retry logic.

        Args:
            method: HTTP method.
            endpoint: API endpoint.
            data: Request data.
            private: Whether this is a private endpoint.
            idempotent: If False (AddOrder), network failures are NOT retried —
                a timeout after Kraken received the order would double-execute
                on retry. The caller must resolve the ambiguity (see
                _recover_ambiguous_order). API-error responses are always safe
                to retry where marked: an error reply proves nothing executed.

        Returns:
            API response data.

        Raises:
            KrakenAPIError: If the request fails. `.network` is True when the
                outcome is ambiguous (transport failure on the final attempt).
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
                    # Rate-limit and EService (busy/unavailable) replies are
                    # definitive "nothing executed" — safe to retry even for
                    # AddOrder.
                    if any("EAPI:Rate limit" in e for e in errors):
                        logger.warning("Rate limit hit, waiting before retry")
                        time.sleep(self._retry_delay * (attempt + 1))
                        continue
                    if any(e.startswith("EService:") for e in errors) \
                            and attempt < self._retry_attempts - 1:
                        logger.warning(f"Kraken busy ({errors}), retrying")
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
                if not idempotent:
                    # Outcome unknown — do NOT blind-retry a possibly-executed
                    # order. Surface the ambiguity to the caller.
                    break
                if attempt < self._retry_attempts - 1:
                    time.sleep(self._retry_delay * (attempt + 1))

        raise KrakenAPIError(
            f"Request failed after network error(s): {last_error}",
            network=True,
        )

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
        validate_only: bool = False,
        post_only: bool = False,
        leverage: Optional[int] = None,
        reduce_only: bool = False
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
            post_only: If True, order will only be placed if it would be a maker order.
                       Rejected if it would take liquidity. Guarantees 0.16% maker fee.
            leverage: Margin leverage (e.g. 2). Required to open/close short
                      positions on Kraken spot margin; omit for plain spot orders.
            reduce_only: If True, order may only reduce an existing margin
                         position (used when covering shorts so a size mismatch
                         can never flip into an unintended long margin position).

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
            paper_trading=self.paper_trading,
            post_only=post_only,
            leverage=leverage or 0,
            reduce_only=reduce_only
        )

        if self.paper_trading:
            return self._place_paper_order(
                pair, side, order_type, volume, price, post_only,
                leverage=leverage, reduce_only=reduce_only,
            )

        kraken_pair = self._normalize_pair(pair)

        # Live volume must respect the pair's lot precision (Kraken rejects
        # excess decimals) and minimum order size. Rounded DOWN so we never
        # order more than the strategy sized or the balance covers.
        volume = self._normalize_volume(pair, volume)

        data = {
            "pair": kraken_pair,
            "type": side.value,
            "ordertype": order_type.value,
            "volume": str(volume),
            # Tag every bot order so startup reconciliation can distinguish
            # our resting orders from anything placed manually on the account.
            "userref": str(self.BOT_USERREF),
        }

        # For plain stop-loss/take-profit, `price` IS the trigger price.
        # For *-limit variants, `price` is the trigger and `price2` the limit.
        if price and order_type in (OrderType.LIMIT, OrderType.STOP_LOSS,
                                    OrderType.TAKE_PROFIT,
                                    OrderType.STOP_LOSS_LIMIT,
                                    OrderType.TAKE_PROFIT_LIMIT):
            data["price"] = str(price)

        if stop_price and order_type in (OrderType.STOP_LOSS_LIMIT, OrderType.TAKE_PROFIT_LIMIT):
            data["price2"] = str(stop_price)

        if validate_only:
            data["validate"] = True

        # Post-only flag ensures maker fee (0.16% instead of 0.26%)
        if post_only and order_type == OrderType.LIMIT:
            data["oflags"] = "post"

        # Margin params (shorts). Kraken requires leverage on BOTH the opening
        # sell and the covering buy so the buy nets against the margin position
        # instead of executing as a spot purchase.
        if leverage:
            data["leverage"] = str(leverage)
        if reduce_only:
            data["reduce_only"] = "true"

        try:
            result = self._request("POST", "private/AddOrder", data,
                                   private=True, idempotent=False)
        except KrakenAPIError as e:
            if not e.network or validate_only:
                raise
            # Transport failure AFTER possibly sending the order: Kraken may
            # or may not have executed it. Look for it by userref before
            # concluding anything — blind retry could double the position.
            recovered = self._recover_ambiguous_order(pair, side, volume)
            if recovered is not None:
                logger.warning(
                    f"AddOrder network failure but order WAS placed — recovered "
                    f"{recovered.order_id} via userref scan",
                    pair=pair, side=side.value,
                )
                return recovered
            raise

        if validate_only:
            return Order(
                order_id="VALIDATE-ONLY", pair=pair, side=side,
                order_type=order_type, price=price, volume=volume,
                filled_volume=0.0, status="validated",
                created_at=datetime.now(timezone.utc),
            )

        if "txid" not in result or not result["txid"]:
            raise KrakenAPIError("No order ID returned from exchange")
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

    def place_maker_order(
        self,
        pair: str,
        side: OrderSide,
        volume: float,
        price_offset_percent: float = 0.0,
        leverage: Optional[int] = None,
        reduce_only: bool = False
    ) -> Order:
        """
        Place a maker (limit) order to get lower fees (0.16% vs 0.26%).

        For BUY: places limit at current bid (or slightly below)
        For SELL: places limit at current ask (or slightly above)

        Args:
            pair: Trading pair.
            side: Buy or sell.
            volume: Order volume.
            price_offset_percent: Offset from bid/ask (0.0 = at bid/ask, 0.1 = 0.1% better)
            leverage: Margin leverage — required for short entries/covers.
            reduce_only: Restrict order to reducing an existing margin position.

        Returns:
            The placed Order.
        """
        ticker = self.get_ticker(pair)

        if side == OrderSide.BUY:
            # Place at bid or slightly below for better fill priority
            price = ticker.bid * (1 - price_offset_percent / 100)
        else:
            # Place at ask or slightly above for better fill priority
            price = ticker.ask * (1 + price_offset_percent / 100)

        # Round to the pair's allowed price precision. A blanket round(_, 2)
        # breaks both extremes: BTC/USD only allows 1 decimal (live order would
        # be rejected as invalid price) and POL/USD allows 5 (rounding $0.084
        # to $0.08 is a 5% mispricing that distorts paper fills).
        return self.place_order(
            pair=pair,
            side=side,
            order_type=OrderType.LIMIT,
            volume=volume,
            price=round(price, self._get_pair_decimals(pair)),
            post_only=True,
            leverage=leverage,
            reduce_only=reduce_only
        )

    def _get_pair_info(self, pair: str) -> Optional[dict]:
        """
        Trading constraints for a pair from AssetPairs, cached after first
        lookup. Returns None (not cached) if the lookup fails, so a transient
        API failure doesn't poison the cache.

        Keys: pair_decimals (price precision), lot_decimals (volume
        precision), ordermin (minimum volume), leverage_buy / leverage_sell
        (allowed margin leverages — empty list means no margin support).
        """
        cached = self._pair_info.get(pair)
        if cached is not None:
            return cached
        try:
            result = self._request(
                "GET", "public/AssetPairs", {"pair": self._normalize_pair(pair)}
            )
            raw = list(result.values())[0]
            info = {
                "pair_decimals": int(raw.get("pair_decimals", 2)),
                "lot_decimals": int(raw.get("lot_decimals", 8)),
                "ordermin": float(raw.get("ordermin", 0) or 0),
                "leverage_buy": [int(x) for x in raw.get("leverage_buy", [])],
                "leverage_sell": [int(x) for x in raw.get("leverage_sell", [])],
            }
            self._pair_info[pair] = info
            return info
        except Exception as e:
            logger.warning(f"Could not fetch AssetPairs info for {pair}: {e}")
            return None

    def _get_pair_decimals(self, pair: str) -> int:
        """
        Price decimals Kraken accepts for a pair (AssetPairs.pair_decimals).
        Falls back to the legacy value of 2 if the lookup fails so order
        placement degrades rather than breaks.
        """
        info = self._get_pair_info(pair)
        return info["pair_decimals"] if info else 2

    def _normalize_volume(self, pair: str, volume: float) -> float:
        """
        Round volume DOWN to the pair's lot_decimals (Kraken rejects volumes
        with excess precision) and enforce the pair minimum.

        Raises KrakenAPIError if the volume is below the pair's ordermin —
        the order would be rejected by the exchange anyway, and this surfaces
        the reason instead of an opaque EGeneral:Invalid arguments.
        """
        info = self._get_pair_info(pair)
        if not info:
            return volume
        factor = 10 ** info["lot_decimals"]
        normalized = math.floor(volume * factor) / factor
        if info["ordermin"] and normalized < info["ordermin"]:
            raise KrakenAPIError(
                f"Volume {normalized} below pair minimum {info['ordermin']} for {pair}"
            )
        return normalized

    def margin_leverage_for(self, pair: str, side: OrderSide,
                            preferred: int) -> Optional[int]:
        """
        Usable margin leverage for a pair/side, or None if the pair has no
        margin support (shorts impossible on Kraken without margin).

        Prefers `preferred` if allowed, else the smallest allowed leverage —
        leverage only affects collateral, never position size, so smaller is
        strictly safer.
        """
        info = self._get_pair_info(pair)
        if not info:
            return None
        allowed = info["leverage_sell"] if side == OrderSide.SELL else info["leverage_buy"]
        if not allowed:
            return None
        return preferred if preferred in allowed else min(allowed)

    def query_order(self, order_id: str) -> dict:
        """
        Current state of an order (QueryOrders).

        Returns dict with: status (pending/open/closed/canceled/expired),
        vol (requested), vol_exec (filled), price (avg fill price), fee.
        """
        if self.paper_trading:
            o = self._paper_orders.get(order_id)
            if o is None:
                return {"status": "unknown", "vol": 0.0, "vol_exec": 0.0,
                        "price": 0.0, "fee": 0.0}
            return {"status": o.status, "vol": o.volume,
                    "vol_exec": o.filled_volume, "price": o.price or 0.0,
                    "fee": o.fee}

        result = self._request("POST", "private/QueryOrders",
                               {"txid": order_id}, private=True)
        raw = result.get(order_id, {})
        return {
            "status": raw.get("status", "unknown"),
            "vol": float(raw.get("vol", 0) or 0),
            "vol_exec": float(raw.get("vol_exec", 0) or 0),
            "price": float(raw.get("price", 0) or 0),
            "fee": float(raw.get("fee", 0) or 0),
        }

    def get_open_positions(self) -> dict:
        """
        Open margin positions (OpenPositions), keyed by position txid.
        Empty in paper mode — paper margin is simulated locally.
        """
        if self.paper_trading:
            return {}
        return self._request("POST", "private/OpenPositions", {}, private=True) or {}

    def get_open_orders_raw(self) -> dict:
        """Raw OpenOrders payload {txid: data} including userref/descr."""
        if self.paper_trading:
            return {}
        result = self._request("POST", "private/OpenOrders", {}, private=True)
        return result.get("open", {})

    def _recover_ambiguous_order(self, pair: str, side: OrderSide,
                                 volume: float) -> Optional[Order]:
        """
        After a network failure on AddOrder, determine whether the order
        actually reached the exchange by scanning recent orders carrying our
        BOT_USERREF for a matching pair/side/volume.

        Returns the recovered Order, or None if no matching order was placed
        (caller may then safely treat the AddOrder as failed).
        """
        try:
            time.sleep(2)  # let the exchange settle the in-flight request
            candidates: dict[str, dict] = {}
            open_result = self._request(
                "POST", "private/OpenOrders",
                {"userref": str(self.BOT_USERREF)}, private=True)
            candidates.update(open_result.get("open", {}))
            closed_result = self._request(
                "POST", "private/ClosedOrders",
                {"userref": str(self.BOT_USERREF),
                 "start": int(time.time()) - 300}, private=True)
            candidates.update(closed_result.get("closed", {}))

            kraken_pair = self._normalize_pair(pair)
            for txid, raw in candidates.items():
                descr = raw.get("descr", {})
                if descr.get("type") != side.value:
                    continue
                if descr.get("pair", "").replace("/", "") not in (kraken_pair, pair.replace("/", "")):
                    continue
                if abs(float(raw.get("vol", 0)) - volume) > volume * 0.001:
                    continue
                return Order(
                    order_id=txid, pair=pair, side=side,
                    order_type=OrderType.LIMIT,
                    price=float(raw.get("price", 0) or 0) or None,
                    volume=volume,
                    filled_volume=float(raw.get("vol_exec", 0) or 0),
                    status=raw.get("status", "open"),
                    created_at=datetime.now(timezone.utc),
                    fee=float(raw.get("fee", 0) or 0),
                )
        except Exception as e:
            logger.error(f"Ambiguous-order recovery scan failed: {e}")
        return None

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

    def _init_paper_balances_from_real(self) -> None:
        """Initialize paper trading balances from real Kraken account."""
        try:
            logger.info("Fetching real balances for paper trading simulation...")

            self.paper_trading = False
            try:
                real_balances = self.get_balances()
            finally:
                self.paper_trading = True

            for asset, balance in real_balances.items():
                self._paper_balances[asset] = balance.total

            self._paper_balances_initialized = True

            usd_balance = self._paper_balances.get("USD", 0)
            logger.info(f"Paper trading initialized with real balance: ${usd_balance:.2f}")

        except Exception as e:
            logger.warning(f"Could not fetch real balances, using default: {e}")
            self._paper_balances_initialized = False

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
        price: Optional[float],
        post_only: bool = False,
        leverage: Optional[int] = None,
        reduce_only: bool = False
    ) -> Order:
        """Simulate order placement in paper trading mode.

        Margin semantics: leverage is accepted (and logged via place_order) but
        does not change paper bookkeeping — volume/notional are already sized
        by the strategy layer. reduce_only marks an order as closing an
        existing position, which bypasses the spot balance gate (covering a
        short is not a funded spot purchase).
        """
        # Get current price if not specified
        if price is None:
            ticker = self.get_ticker(pair)
            price = ticker.ask if side == OrderSide.BUY else ticker.bid

        # Calculate cost
        cost = volume * price
        # Use maker fee (0.16%) for limit/post-only orders, taker fee (0.26%) for market
        fee_rate = 0.0016 if (order_type == OrderType.LIMIT or post_only) else 0.0026
        fee = cost * fee_rate

        # Check balance
        base, quote = pair.split("/")

        if side == OrderSide.BUY:
            # reduce_only buys cover an open short — they net against the
            # margin position, so the spot USD balance gate does not apply.
            if not reduce_only and self._paper_balances.get(quote, 0) < cost + fee:
                raise KrakenAPIError("Insufficient balance for paper trade")
            self._paper_balances[quote] = self._paper_balances.get(quote, 0) - cost - fee
            self._paper_balances[base] = self._paper_balances.get(base, 0) + volume
        else:
            # A SELL is either a spot exit of held inventory (long close) or a
            # margin short ENTRY of an asset we do not hold. The strategy layer
            # owns position sizing and P&L (self.capital); these paper balances
            # are bookkeeping only. The old spot-inventory check rejected every
            # short entry ("Insufficient balance"), silently making live paper
            # trading long-only — fatal in a BEAR regime. Model the short by
            # letting the base balance go negative and crediting USD proceeds.
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
