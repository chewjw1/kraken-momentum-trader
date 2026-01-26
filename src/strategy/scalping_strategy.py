"""
Scalping trading strategy for high-frequency, small-profit trades.

Entry conditions (require 2+ confirmations):
- RSI < 30 (short-term oversold) OR RSI divergence
- Price at/below lower Bollinger Band
- Price below VWAP (mean reversion opportunity)
- Volume spike (1.5x average)

Exit conditions:
- Take profit: 1-2% (configurable)
- Stop loss: 0.5-1% (tight)
- RSI > 70 (overbought)
- Price at upper Bollinger Band
- Time-based exit (optional)
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Tuple

from ..config.settings import get_settings
from ..observability.logger import get_logger
from .base_strategy import (
    BaseStrategy,
    MarketData,
    Position,
    Signal,
    SignalType,
)
from .signals.rsi import RSIIndicator
from .signals.ema import EMAIndicator
from .signals.volume import VolumeIndicator
from .signals.vwap import VWAPIndicator
from .signals.bollinger import BollingerBandsIndicator

logger = get_logger(__name__)


@dataclass
class ScalpingConfig:
    """Configuration for scalping strategy."""
    # RSI settings (shorter period for scalping)
    rsi_period: int = 7
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # Bollinger Bands
    bb_period: int = 20
    bb_std_dev: float = 2.0
    bb_squeeze_threshold: float = 3.0  # Lower = tighter squeeze detection

    # VWAP
    vwap_threshold_percent: float = 0.3  # Signal when price is X% from VWAP

    # Volume
    volume_spike_threshold: float = 1.5  # 1.5x average volume

    # Risk management
    take_profit_percent: float = 1.5  # Take profit at 1.5%
    stop_loss_percent: float = 0.75  # Stop loss at 0.75%

    # Entry confirmation
    min_confirmations: int = 2  # Require 2+ signals to align

    # Fees consideration
    fee_percent: float = 0.26  # Kraken taker fee
    min_profit_after_fees: float = 0.5  # Minimum net profit target


class ScalpingStrategy(BaseStrategy):
    """
    High-frequency scalping strategy.

    Combines multiple indicators for high-probability short-term trades:
    - Short-term RSI for momentum
    - Bollinger Bands for volatility and mean reversion
    - VWAP for institutional price level
    - Volume for confirmation

    Optimized for 5-15 minute candles with tight risk management.
    """

    def __init__(self, config: Optional[ScalpingConfig] = None):
        """
        Initialize scalping strategy.

        Args:
            config: Scalping configuration. Uses defaults if None.
        """
        super().__init__(name="scalping")

        self.config = config or ScalpingConfig()

        # Initialize indicators
        self.rsi = RSIIndicator(
            period=self.config.rsi_period,
            oversold=self.config.rsi_oversold,
            overbought=self.config.rsi_overbought
        )

        self.bollinger = BollingerBandsIndicator(
            period=self.config.bb_period,
            std_dev_multiplier=self.config.bb_std_dev,
            squeeze_threshold=self.config.bb_squeeze_threshold
        )

        self.vwap = VWAPIndicator(
            threshold_percent=self.config.vwap_threshold_percent
        )

        self.volume = VolumeIndicator(
            sma_period=20,
            high_volume_threshold=self.config.volume_spike_threshold
        )

        # For trend context
        self.ema_short = EMAIndicator(short_period=5, long_period=13)

        self._is_initialized = True

        logger.info(
            "Scalping strategy initialized",
            rsi_period=self.config.rsi_period,
            take_profit=self.config.take_profit_percent,
            stop_loss=self.config.stop_loss_percent,
            min_confirmations=self.config.min_confirmations
        )

    def analyze(
        self,
        market_data: MarketData,
        current_position: Optional[Position] = None
    ) -> Signal:
        """
        Analyze market data for scalping opportunities.

        Args:
            market_data: Current market data.
            current_position: Current open position, if any.

        Returns:
            Trading signal.
        """
        timestamp = datetime.now(timezone.utc)

        if not self._validate_data(market_data):
            return self._no_signal(market_data.pair, "Insufficient data", timestamp)

        # Calculate all indicators
        signals = self._calculate_signals(market_data)

        # If we have a position, check exit conditions
        if current_position:
            return self._check_exit_conditions(
                market_data, current_position, signals, timestamp
            )

        # No position - check entry conditions
        return self._check_entry_conditions(market_data, signals, timestamp)

    def _validate_data(self, market_data: MarketData) -> bool:
        """Validate market data has sufficient history."""
        min_candles = max(self.config.bb_period, self.config.rsi_period) + 5
        return len(market_data.ohlc) >= min_candles

    def _calculate_signals(self, market_data: MarketData) -> dict:
        """Calculate all indicator signals."""
        # Extract OHLCV data from candles
        closes = [c.close for c in market_data.ohlc]
        highs = [c.high for c in market_data.ohlc]
        lows = [c.low for c in market_data.ohlc]
        volumes = [c.volume for c in market_data.ohlc]

        # RSI
        rsi_result = self.rsi.calculate(closes)

        # Bollinger Bands
        bb_result = self.bollinger.calculate(closes)

        # VWAP
        vwap_result = self.vwap.calculate(highs, lows, closes, volumes)

        # Volume
        volume_result = self.volume.calculate(volumes)

        # EMA trend
        ema_result = self.ema_short.calculate(closes)

        return {
            "rsi": rsi_result,
            "bollinger": bb_result,
            "vwap": vwap_result,
            "volume": volume_result,
            "ema": ema_result,
            "current_price": closes[-1] if closes else 0,
        }

    def _check_entry_conditions(
        self,
        market_data: MarketData,
        signals: dict,
        timestamp: datetime
    ) -> Signal:
        """Check for scalping entry signals."""
        confirmations = 0
        reasons = []

        rsi = signals.get("rsi")
        bb = signals.get("bollinger")
        vwap = signals.get("vwap")
        volume = signals.get("volume")
        ema = signals.get("ema")
        current_price = signals.get("current_price", 0)

        # === LONG ENTRY CONDITIONS ===

        # 1. RSI oversold
        if rsi and rsi.is_oversold:
            confirmations += 1
            reasons.append(f"RSI oversold ({rsi.value:.1f})")

        # 2. Price at/below lower Bollinger Band
        if bb and bb.is_below_lower:
            confirmations += 1
            reasons.append(f"Price at lower BB ({bb.percent_b:.2f})")

        # 3. Price below VWAP (mean reversion opportunity)
        if vwap and vwap.is_below:
            confirmations += 1
            reasons.append(f"Below VWAP ({vwap.price_vs_vwap:.2f}%)")

        # 4. Volume confirmation
        if volume and volume.is_high_volume:
            confirmations += 1
            reasons.append(f"Volume spike ({volume.volume_ratio:.1f}x)")

        # 5. Bollinger squeeze (potential breakout)
        if bb and bb.is_squeeze:
            confirmations += 0.5  # Partial confirmation
            reasons.append("BB squeeze detected")

        # Check if we have enough confirmations
        if confirmations >= self.config.min_confirmations:
            # Verify profit potential after fees
            potential_profit = self.config.take_profit_percent
            total_fees = self.config.fee_percent * 2  # Round trip
            net_profit = potential_profit - total_fees

            if net_profit < self.config.min_profit_after_fees:
                return self._no_signal(
                    market_data.pair,
                    f"Net profit {net_profit:.2f}% below minimum"
                )

            return Signal(
                signal_type=SignalType.BUY,
                pair=market_data.pair,
                price=current_price,
                timestamp=timestamp,
                strength=min(confirmations / 4, 1.0),  # Normalize to 0-1
                reason=f"Scalp entry: {', '.join(reasons)}",
                indicators={
                    "rsi": rsi.value if rsi else None,
                    "bb_percent_b": bb.percent_b if bb else None,
                    "vwap_distance": vwap.price_vs_vwap if vwap else None,
                    "volume_ratio": volume.volume_ratio if volume else None,
                    "confirmations": confirmations,
                }
            )

        return self._no_signal(
            market_data.pair,
            f"Only {confirmations} confirmations (need {self.config.min_confirmations})",
            timestamp
        )

    def _check_exit_conditions(
        self,
        market_data: MarketData,
        position: Position,
        signals: dict,
        timestamp: datetime
    ) -> Signal:
        """Check for scalping exit signals."""
        current_price = signals.get("current_price", 0)
        entry_price = position.entry_price

        # Calculate P&L
        pnl_percent = ((current_price - entry_price) / entry_price) * 100

        rsi = signals.get("rsi")
        bb = signals.get("bollinger")
        vwap = signals.get("vwap")

        # === EXIT CONDITIONS ===

        # 1. Take profit hit
        if pnl_percent >= self.config.take_profit_percent:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                pair=market_data.pair,
                price=current_price,
                timestamp=timestamp,
                strength=1.0,
                reason=f"Take profit: {pnl_percent:.2f}%",
                indicators={"pnl_percent": pnl_percent}
            )

        # 2. Stop loss hit
        if pnl_percent <= -self.config.stop_loss_percent:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                pair=market_data.pair,
                price=current_price,
                timestamp=timestamp,
                strength=1.0,
                reason=f"Stop loss: {pnl_percent:.2f}%",
                indicators={"pnl_percent": pnl_percent}
            )

        # Minimum profit required for early exits (75% of TP target)
        min_early_exit_profit = self.config.take_profit_percent * 0.75

        # 3. RSI overbought (take profit signal) - only if decent profit
        if rsi and rsi.is_overbought and pnl_percent >= min_early_exit_profit:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                pair=market_data.pair,
                price=current_price,
                timestamp=timestamp,
                strength=0.8,
                reason=f"RSI overbought ({rsi.value:.1f}), P&L: {pnl_percent:.2f}%",
                indicators={"rsi": rsi.value, "pnl_percent": pnl_percent}
            )

        # 4. Price at upper Bollinger Band (mean reversion exit) - only if decent profit
        if bb and bb.is_above_upper and pnl_percent >= min_early_exit_profit:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                pair=market_data.pair,
                price=current_price,
                timestamp=timestamp,
                strength=0.7,
                reason=f"Price at upper BB, P&L: {pnl_percent:.2f}%",
                indicators={"bb_percent_b": bb.percent_b, "pnl_percent": pnl_percent}
            )

        # 5. Price crossed above VWAP significantly - only if decent profit
        if vwap and vwap.is_above and vwap.price_vs_vwap > 1.0 and pnl_percent >= min_early_exit_profit:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                pair=market_data.pair,
                price=current_price,
                timestamp=timestamp,
                strength=0.6,
                reason=f"Above VWAP (+{vwap.price_vs_vwap:.2f}%), P&L: {pnl_percent:.2f}%",
                indicators={"vwap_distance": vwap.price_vs_vwap, "pnl_percent": pnl_percent}
            )

        return self._no_signal(
            market_data.pair,
            f"Holding position, P&L: {pnl_percent:.2f}%",
            timestamp
        )

    def _no_signal(self, pair: str, reason: str, timestamp: datetime = None) -> Signal:
        """Create a no-action signal."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        return Signal(
            signal_type=SignalType.HOLD,
            pair=pair,
            price=0,
            timestamp=timestamp,
            strength=0,
            reason=reason,
            indicators={}
        )

    def check_trailing_stop(
        self,
        position: Position,
        current_price: float,
        peak_price: float,
        trailing_stop_active: bool
    ) -> tuple[bool, bool, str]:
        """
        Check trailing stop conditions for scalping.

        For scalping, we use tight trailing stops:
        - Activate when profit reaches 75% of take-profit target
        - Trigger when price drops 0.3% from peak

        Args:
            position: Current position.
            current_price: Current market price.
            peak_price: Highest price since entry.
            trailing_stop_active: Whether trailing stop is already active.

        Returns:
            Tuple of (should_close, should_activate_trailing, reason).
        """
        entry_price = position.entry_price
        pnl_percent = ((current_price - entry_price) / entry_price) * 100
        peak_pnl_percent = ((peak_price - entry_price) / entry_price) * 100

        # Trailing stop activation threshold (75% of take profit)
        activation_threshold = self.config.take_profit_percent * 0.75
        # Trailing stop trigger (0.3% drop from peak)
        trailing_drop_percent = 0.3

        # Check if we should activate trailing stop
        should_activate = False
        if not trailing_stop_active and pnl_percent >= activation_threshold:
            should_activate = True

        # Check if trailing stop should trigger
        if trailing_stop_active:
            drop_from_peak = ((peak_price - current_price) / peak_price) * 100
            if drop_from_peak >= trailing_drop_percent:
                return (
                    True,
                    False,
                    f"Trailing stop: dropped {drop_from_peak:.2f}% from peak, P&L: {pnl_percent:.2f}%"
                )

        return (False, should_activate, "")

    def get_risk_params(self) -> dict:
        """Get risk management parameters for this strategy."""
        return {
            "take_profit_percent": self.config.take_profit_percent,
            "stop_loss_percent": self.config.stop_loss_percent,
            "fee_percent": self.config.fee_percent,
            "net_target": self.config.take_profit_percent - (self.config.fee_percent * 2),
        }
