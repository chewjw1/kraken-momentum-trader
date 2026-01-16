"""
Momentum trading strategy implementation.

Entry conditions (all must be met):
- RSI < oversold threshold (default 30)
- Price above long-term EMA
- Volume above average (confirming signal)
- Expected gain > fee threshold

Exit conditions (any):
- Take profit hit (default 10%)
- Stop loss hit (default 5%)
- RSI > overbought threshold (default 70)
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ..config.settings import StrategyConfig, get_settings
from ..observability.logger import get_logger
from .base_strategy import (
    BaseStrategy,
    MarketData,
    Position,
    Signal,
    SignalType,
)
from .signals.ema import EMAIndicator, EMAResult
from .signals.rsi import RSIIndicator, RSIResult
from .signals.volume import VolumeIndicator, VolumeResult

logger = get_logger(__name__)


@dataclass
class MomentumSignals:
    """Container for all momentum indicator results."""
    rsi: Optional[RSIResult] = None
    ema: Optional[EMAResult] = None
    volume: Optional[VolumeResult] = None

    def all_valid(self) -> bool:
        """Check if all signals have valid data."""
        return all([self.rsi, self.ema, self.volume])


class MomentumStrategy(BaseStrategy):
    """
    Multi-indicator momentum strategy.

    Uses RSI, EMA, and Volume indicators with confirmation
    to generate trading signals. Designed to capture momentum
    moves while avoiding false signals through multi-indicator
    confirmation.
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        Initialize momentum strategy.

        Args:
            config: Strategy configuration. If None, loaded from settings.
        """
        super().__init__(name="momentum")

        if config is None:
            config = get_settings().strategy

        self.config = config

        # Initialize indicators
        self.rsi = RSIIndicator(
            period=config.rsi_period,
            oversold=config.rsi_oversold,
            overbought=config.rsi_overbought
        )
        self.ema = EMAIndicator(
            short_period=config.ema_short_period,
            long_period=config.ema_long_period
        )
        self.volume = VolumeIndicator(
            sma_period=config.volume_sma_period,
            high_volume_threshold=config.volume_threshold
        )

        # Risk parameters (from risk config)
        risk_config = get_settings().risk
        self.stop_loss_percent = risk_config.stop_loss_percent
        self.take_profit_percent = risk_config.take_profit_percent
        self.min_expected_gain = config.min_expected_gain

        self._is_initialized = True

    def analyze(
        self,
        market_data: MarketData,
        current_position: Optional[Position] = None
    ) -> Signal:
        """
        Analyze market data and generate trading signal.

        Args:
            market_data: Current market data.
            current_position: Current open position, if any.

        Returns:
            Trading signal based on momentum indicators.
        """
        # Calculate all indicators
        signals = self._calculate_signals(market_data)

        # Get current price
        current_price = (
            market_data.ticker.last if market_data.ticker
            else market_data.prices[-1] if market_data.prices
            else 0.0
        )

        timestamp = datetime.now(timezone.utc)

        # Check if we have valid data for all indicators
        if not signals.all_valid():
            return Signal(
                signal_type=SignalType.HOLD,
                pair=market_data.pair,
                strength=0.0,
                price=current_price,
                timestamp=timestamp,
                reason="Insufficient data for indicators",
                indicators=self._signals_to_dict(signals)
            )

        # If we have an open position, check for exit signals
        if current_position:
            return self._analyze_exit(
                market_data, current_position, signals, current_price, timestamp
            )

        # Otherwise, look for entry signals
        return self._analyze_entry(
            market_data, signals, current_price, timestamp
        )

    def _calculate_signals(self, market_data: MarketData) -> MomentumSignals:
        """Calculate all indicator signals."""
        signals = MomentumSignals()

        if market_data.prices:
            signals.rsi = self.rsi.calculate(market_data.prices)
            signals.ema = self.ema.calculate(market_data.prices)

        if market_data.volumes:
            signals.volume = self.volume.calculate(
                market_data.volumes,
                market_data.prices
            )

        return signals

    def _analyze_entry(
        self,
        market_data: MarketData,
        signals: MomentumSignals,
        current_price: float,
        timestamp: datetime
    ) -> Signal:
        """Analyze for entry signal."""
        # Entry conditions for LONG:
        # 1. RSI is oversold (< 30)
        # 2. Price is above long-term EMA (uptrend)
        # 3. Volume is confirming (high or increasing)

        rsi = signals.rsi
        ema = signals.ema
        volume = signals.volume

        entry_conditions = {
            "rsi_oversold": rsi.is_oversold,
            "price_above_ema": ema.price_above_ema,
            "volume_confirming": volume.is_high_volume or volume.is_increasing,
        }

        # Log signals for analysis
        logger.signal(
            signal_type="rsi",
            pair=market_data.pair,
            value=rsi.value,
            threshold=self.config.rsi_oversold
        )
        logger.signal(
            signal_type="ema_trend",
            pair=market_data.pair,
            value=ema.trend_strength,
            threshold=0.0
        )
        logger.signal(
            signal_type="volume_ratio",
            pair=market_data.pair,
            value=volume.volume_ratio,
            threshold=self.config.volume_threshold
        )

        # Calculate signal strength based on how many conditions are met
        conditions_met = sum(entry_conditions.values())
        total_conditions = len(entry_conditions)
        base_strength = conditions_met / total_conditions

        # Boost strength for stronger signals
        if rsi.value < 20:  # Very oversold
            base_strength += 0.1
        if volume.volume_ratio > 2.0:  # Very high volume
            base_strength += 0.1
        if ema.crossover_signal == "bullish":  # Recent bullish crossover
            base_strength += 0.15

        strength = min(1.0, base_strength)

        # All conditions must be met for a buy signal
        if all(entry_conditions.values()):
            reason = (
                f"BUY: RSI={rsi.value:.1f} (oversold), "
                f"Price above EMA, Volume ratio={volume.volume_ratio:.2f}"
            )
            return Signal(
                signal_type=SignalType.BUY,
                pair=market_data.pair,
                strength=strength,
                price=current_price,
                timestamp=timestamp,
                reason=reason,
                indicators=self._signals_to_dict(signals)
            )

        # Not enough conditions met
        unmet = [k for k, v in entry_conditions.items() if not v]
        reason = f"HOLD: Missing conditions: {', '.join(unmet)}"

        return Signal(
            signal_type=SignalType.HOLD,
            pair=market_data.pair,
            strength=strength,
            price=current_price,
            timestamp=timestamp,
            reason=reason,
            indicators=self._signals_to_dict(signals)
        )

    def _analyze_exit(
        self,
        market_data: MarketData,
        position: Position,
        signals: MomentumSignals,
        current_price: float,
        timestamp: datetime
    ) -> Signal:
        """Analyze for exit signal on existing position."""
        rsi = signals.rsi
        ema = signals.ema

        # Check stop loss / take profit first
        should_close, close_reason = self.should_close_position(position, market_data)
        if should_close:
            return Signal(
                signal_type=SignalType.CLOSE_LONG if position.side == "long" else SignalType.CLOSE_SHORT,
                pair=market_data.pair,
                strength=1.0,  # Stop loss / take profit are always executed
                price=current_price,
                timestamp=timestamp,
                reason=close_reason,
                indicators=self._signals_to_dict(signals)
            )

        # Check for RSI overbought exit (for long positions)
        if position.side == "long" and rsi.is_overbought:
            reason = f"SELL: RSI={rsi.value:.1f} (overbought)"
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                pair=market_data.pair,
                strength=0.8,
                price=current_price,
                timestamp=timestamp,
                reason=reason,
                indicators=self._signals_to_dict(signals)
            )

        # Check for trend reversal
        if position.side == "long" and ema.crossover_signal == "bearish":
            reason = "SELL: Bearish EMA crossover detected"
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                pair=market_data.pair,
                strength=0.7,
                price=current_price,
                timestamp=timestamp,
                reason=reason,
                indicators=self._signals_to_dict(signals)
            )

        # Hold position
        return Signal(
            signal_type=SignalType.HOLD,
            pair=market_data.pair,
            strength=0.0,
            price=current_price,
            timestamp=timestamp,
            reason=f"Hold position: RSI={rsi.value:.1f}, PnL={position.unrealized_pnl_percent:.2f}%",
            indicators=self._signals_to_dict(signals)
        )

    def get_stop_loss_price(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price."""
        if side == "long":
            return entry_price * (1 - self.stop_loss_percent / 100)
        else:
            return entry_price * (1 + self.stop_loss_percent / 100)

    def get_take_profit_price(self, entry_price: float, side: str) -> float:
        """Calculate take profit price."""
        if side == "long":
            return entry_price * (1 + self.take_profit_percent / 100)
        else:
            return entry_price * (1 - self.take_profit_percent / 100)

    def _signals_to_dict(self, signals: MomentumSignals) -> dict:
        """Convert signals to dictionary for logging."""
        result = {}

        if signals.rsi:
            result["rsi"] = {
                "value": signals.rsi.value,
                "signal": signals.rsi.signal,
                "is_oversold": signals.rsi.is_oversold,
                "is_overbought": signals.rsi.is_overbought,
            }

        if signals.ema:
            result["ema"] = {
                "short": signals.ema.short_ema,
                "long": signals.ema.long_ema,
                "price_above_ema": signals.ema.price_above_ema,
                "crossover_signal": signals.ema.crossover_signal,
                "trend_strength": signals.ema.trend_strength,
            }

        if signals.volume:
            result["volume"] = {
                "current": signals.volume.current_volume,
                "average": signals.volume.average_volume,
                "ratio": signals.volume.volume_ratio,
                "is_high": signals.volume.is_high_volume,
                "is_increasing": signals.volume.is_increasing,
            }

        return result

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self.rsi.reset()
        self.ema.reset()
        self.volume.reset()
