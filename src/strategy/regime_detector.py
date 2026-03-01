"""
Market regime detector.

Classifies the current market into one of three regimes:
- BULL: sustained uptrend, strategy can use wider take-profits
- BEAR: sustained downtrend, strategy should reduce exposure or sit out
- SIDEWAYS: range-bound, ideal for mean-reversion scalping

Detection method:
1. SMA slope over a configurable lookback (default 50 periods)
2. ADX-like volatility measure to separate trending from range-bound
3. Price position relative to longer-term SMA (200-period)

Each regime maps to a set of parameter adjustments applied on top of
the base scalping config.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any

from ..observability.logger import get_logger
from .signals.ema import calculate_ema

logger = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    # SMA periods
    fast_sma_period: int = 50      # For slope calculation
    slow_sma_period: int = 200     # For trend confirmation

    # Slope thresholds (% change per period)
    # Lowered from 0.05 to 0.02 — old thresholds were too aggressive and
    # classified most periods as bear (missed bull regimes for BTC, ETH, etc.)
    bull_slope_threshold: float = 0.02    # SMA rising > 0.02% per period
    bear_slope_threshold: float = -0.02   # SMA falling < -0.02% per period

    # Volatility (ATR-based)
    atr_period: int = 14
    high_volatility_threshold: float = 2.0  # ATR as % of price

    # Lookback for slope calculation
    slope_lookback: int = 10  # periods to measure SMA slope

    # Minimum data points needed
    min_data_points: int = 200


@dataclass
class RegimeResult:
    """Result of regime detection."""
    regime: MarketRegime
    confidence: float              # 0.0 to 1.0
    sma_slope_pct: float           # SMA slope as percentage per period
    price_vs_slow_sma_pct: float   # Price distance from 200 SMA
    atr_pct: float                 # ATR as percentage of price
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_trending(self) -> bool:
        """True if market is in a trending regime (bull or bear)."""
        return self.regime in (MarketRegime.BULL, MarketRegime.BEAR)


# Per-regime parameter adjustments applied on top of base scalping config
REGIME_ADJUSTMENTS: Dict[MarketRegime, Dict[str, Any]] = {
    MarketRegime.BULL: {
        "take_profit_multiplier": 1.3,     # Wider TP -- let winners run
        "stop_loss_multiplier": 1.0,       # Normal SL
        "position_scale_multiplier": 1.2,  # Slightly larger positions
        "min_confirmations_offset": 0,     # Normal entry bar
        "ema_filter_enabled": True,        # Keep filter on
        "description": "Bull: wider TP, normal SL, increased size",
    },
    MarketRegime.BEAR: {
        "take_profit_multiplier": 0.7,     # Tighter TP -- take profits fast
        "stop_loss_multiplier": 0.8,       # Tighter SL -- cut losses faster
        "position_scale_multiplier": 0.5,  # Half position size
        "min_confirmations_offset": 1,     # Require 1 extra confirmation
        "ema_filter_enabled": True,        # Keep filter on
        "description": "Bear: tight TP/SL, halved size, extra confirmation",
    },
    MarketRegime.SIDEWAYS: {
        "take_profit_multiplier": 1.0,     # Base TP
        "stop_loss_multiplier": 1.0,       # Base SL
        "position_scale_multiplier": 1.0,  # Base size
        "min_confirmations_offset": 0,     # Base confirmations
        "ema_filter_enabled": False,       # Disable EMA filter -- ranging market
        "description": "Sideways: base params, EMA filter off (ranging)",
    },
    MarketRegime.UNKNOWN: {
        "take_profit_multiplier": 1.0,
        "stop_loss_multiplier": 1.0,
        "position_scale_multiplier": 0.7,  # Conservative until we know
        "min_confirmations_offset": 1,
        "ema_filter_enabled": True,
        "description": "Unknown: conservative sizing until regime is clear",
    },
}


class RegimeDetector:
    """
    Detects current market regime from price data.

    Uses a combination of:
    - SMA slope (direction)
    - Price vs slow SMA (confirmation)
    - ATR (volatility context)
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self._last_result: Optional[RegimeResult] = None

    def detect(
        self,
        closes: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
    ) -> RegimeResult:
        """
        Detect the current market regime.

        Args:
            closes: List of closing prices (oldest first).
            highs: Optional list of high prices (for ATR).
            lows: Optional list of low prices (for ATR).

        Returns:
            RegimeResult with classification and metrics.
        """
        if len(closes) < self.config.min_data_points:
            return RegimeResult(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                sma_slope_pct=0.0,
                price_vs_slow_sma_pct=0.0,
                atr_pct=0.0,
            )

        # Calculate SMAs
        fast_sma = self._sma(closes, self.config.fast_sma_period)
        slow_sma = self._sma(closes, self.config.slow_sma_period)

        if fast_sma is None or slow_sma is None:
            return RegimeResult(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                sma_slope_pct=0.0,
                price_vs_slow_sma_pct=0.0,
                atr_pct=0.0,
            )

        # Calculate fast SMA slope over lookback window
        sma_slope_pct = self._sma_slope(closes, self.config.fast_sma_period,
                                         self.config.slope_lookback)

        # Price vs slow SMA
        current_price = closes[-1]
        price_vs_slow_sma_pct = ((current_price - slow_sma) / slow_sma) * 100

        # ATR as % of price
        atr_pct = 0.0
        if highs and lows and len(highs) >= self.config.atr_period:
            atr_pct = self._atr_percent(highs, lows, closes, self.config.atr_period)

        # Classify regime
        regime, confidence = self._classify(
            sma_slope_pct, price_vs_slow_sma_pct, atr_pct, fast_sma, slow_sma
        )

        result = RegimeResult(
            regime=regime,
            confidence=confidence,
            sma_slope_pct=sma_slope_pct,
            price_vs_slow_sma_pct=price_vs_slow_sma_pct,
            atr_pct=atr_pct,
        )

        self._last_result = result
        return result

    def get_adjustments(self, regime: Optional[MarketRegime] = None) -> Dict[str, Any]:
        """
        Get parameter adjustments for the given or last-detected regime.

        Args:
            regime: Market regime (uses last detection if None).

        Returns:
            Dictionary of parameter adjustments.
        """
        if regime is None:
            regime = self._last_result.regime if self._last_result else MarketRegime.UNKNOWN
        return REGIME_ADJUSTMENTS.get(regime, REGIME_ADJUSTMENTS[MarketRegime.UNKNOWN])

    @property
    def last_result(self) -> Optional[RegimeResult]:
        """Last detection result."""
        return self._last_result

    # ── internal helpers ──────────────────────────────────────────

    def _sma(self, prices: List[float], period: int) -> Optional[float]:
        """Simple moving average of the last *period* values."""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    def _sma_slope(self, prices: List[float], sma_period: int, lookback: int) -> float:
        """
        Compute the average per-period % change of the SMA.

        Calculates the SMA at two points (now and *lookback* periods ago)
        then returns the annualised slope.
        """
        if len(prices) < sma_period + lookback:
            return 0.0

        # Current SMA
        sma_now = sum(prices[-sma_period:]) / sma_period

        # SMA *lookback* periods ago
        end_idx = len(prices) - lookback
        start_idx = end_idx - sma_period
        if start_idx < 0:
            return 0.0
        sma_then = sum(prices[start_idx:end_idx]) / sma_period

        if sma_then == 0:
            return 0.0

        # Per-period percentage change
        total_change_pct = ((sma_now - sma_then) / sma_then) * 100
        per_period_pct = total_change_pct / lookback
        return per_period_pct

    def _atr_percent(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int,
    ) -> float:
        """Average True Range as a percentage of current price."""
        if len(highs) < period + 1:
            return 0.0

        true_ranges = []
        for i in range(-period, 0):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            true_ranges.append(tr)

        atr = sum(true_ranges) / len(true_ranges)
        current_price = closes[-1]
        return (atr / current_price) * 100 if current_price > 0 else 0.0

    def _classify(
        self,
        sma_slope_pct: float,
        price_vs_slow_sma_pct: float,
        atr_pct: float,
        fast_sma: float,
        slow_sma: float,
    ) -> tuple:
        """
        Classify regime from computed metrics.

        Returns (MarketRegime, confidence).
        """
        bull_score = 0.0
        bear_score = 0.0

        # Signal 1: SMA slope direction (strongest signal)
        if sma_slope_pct > self.config.bull_slope_threshold:
            bull_score += 0.4
        elif sma_slope_pct < self.config.bear_slope_threshold:
            bear_score += 0.4

        # Signal 2: Price above/below slow SMA (lowered from 2.0 to 1.0)
        if price_vs_slow_sma_pct > 1.0:
            bull_score += 0.3
        elif price_vs_slow_sma_pct < -1.0:
            bear_score += 0.3

        # Signal 3: Fast SMA above/below slow SMA (golden/death cross)
        if fast_sma > slow_sma:
            bull_score += 0.2
        elif fast_sma < slow_sma:
            bear_score += 0.2

        # Signal 4: Slope magnitude (stronger slope = more confident)
        slope_magnitude = abs(sma_slope_pct)
        if slope_magnitude > self.config.bull_slope_threshold * 3:
            # Strong trend
            if sma_slope_pct > 0:
                bull_score += 0.1
            else:
                bear_score += 0.1

        # Determine regime (lowered threshold from 0.5 to 0.4 for better detection)
        if bull_score >= 0.4:
            return MarketRegime.BULL, min(bull_score, 1.0)
        elif bear_score >= 0.4:
            return MarketRegime.BEAR, min(bear_score, 1.0)
        else:
            # Neither strong bull nor bear -- sideways
            # Confidence is higher when both scores are low (clear ranging)
            max_score = max(bull_score, bear_score)
            sideways_confidence = 1.0 - max_score
            return MarketRegime.SIDEWAYS, min(sideways_confidence, 1.0)

    def to_dict(self) -> dict:
        """Serialize last result for persistence."""
        if not self._last_result:
            return {}
        r = self._last_result
        return {
            "regime": r.regime.value,
            "confidence": r.confidence,
            "sma_slope_pct": r.sma_slope_pct,
            "price_vs_slow_sma_pct": r.price_vs_slow_sma_pct,
            "atr_pct": r.atr_pct,
            "timestamp": r.timestamp.isoformat(),
        }

    def from_dict(self, data: dict) -> None:
        """Restore last result from persistence."""
        if not data:
            return
        try:
            self._last_result = RegimeResult(
                regime=MarketRegime(data["regime"]),
                confidence=data["confidence"],
                sma_slope_pct=data["sma_slope_pct"],
                price_vs_slow_sma_pct=data["price_vs_slow_sma_pct"],
                atr_pct=data["atr_pct"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
            )
        except (KeyError, ValueError):
            self._last_result = None
