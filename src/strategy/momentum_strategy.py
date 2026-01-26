"""
Momentum trading strategy implementation.

Entry conditions (all must be met):
- RSI < oversold threshold (default 30)
- Price above long-term EMA
- Volume above average (confirming signal)
- Expected gain > fee threshold

Exit conditions (trailing stop strategy):
- Trailing stop activates after position gains 5%
- Exit when price drops 5% from peak
- RSI > overbought threshold (default 70)
- Bearish EMA crossover
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from ..config.settings import StrategyConfig, RiskConfig, get_settings
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

    def __init__(
        self,
        config: Optional[StrategyConfig] = None,
        risk_config: Optional[RiskConfig] = None
    ):
        """
        Initialize momentum strategy.

        Args:
            config: Strategy configuration. If None, loaded from settings.
            risk_config: Risk configuration for trailing stops and Martingale.
                        If None, loaded from global settings.
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

        # Use provided risk_config or fall back to global settings
        if risk_config is None:
            risk_config = get_settings().risk

        # Trailing stop parameters (from risk config)
        self.use_trailing_stop = risk_config.use_trailing_stop
        self.trailing_stop_percent = risk_config.trailing_stop_percent
        self.trailing_stop_activation_percent = risk_config.trailing_stop_activation_percent
        self.initial_stop_loss_percent = risk_config.initial_stop_loss_percent
        self.min_expected_gain = config.min_expected_gain

        # Martingale parameters
        self.martingale_config = risk_config.martingale
        # Handle case where martingale might be a dict (from YAML loading)
        if isinstance(self.martingale_config, dict):
            self.martingale_enabled = self.martingale_config.get("enabled", True)
            self.martingale_add_on_drop = self.martingale_config.get("add_on_drop_percent", 5.0)
            self.martingale_require_rsi = self.martingale_config.get("require_rsi_oversold", True)
            self.martingale_require_ema = self.martingale_config.get("require_ema_trend", False)
        else:
            self.martingale_enabled = self.martingale_config.enabled
            self.martingale_add_on_drop = self.martingale_config.add_on_drop_percent
            self.martingale_require_rsi = self.martingale_config.require_rsi_oversold
            self.martingale_require_ema = self.martingale_config.require_ema_trend

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
        timestamp: datetime,
        peak_price: float = None,
        trailing_stop_active: bool = False
    ) -> Signal:
        """Analyze for exit signal on existing position."""
        rsi = signals.rsi
        ema = signals.ema

        if peak_price is None:
            peak_price = position.entry_price

        # Check trailing stop first
        should_close, should_activate, close_reason = self.check_trailing_stop(
            position, current_price, peak_price, trailing_stop_active
        )

        # Store activation flag in signal metadata for caller to handle
        indicators = self._signals_to_dict(signals)
        indicators["_should_activate_trailing"] = should_activate

        if should_close:
            return Signal(
                signal_type=SignalType.CLOSE_LONG if position.side == "long" else SignalType.CLOSE_SHORT,
                pair=market_data.pair,
                strength=1.0,
                price=current_price,
                timestamp=timestamp,
                reason=close_reason,
                indicators=indicators
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
                indicators=indicators
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
                indicators=indicators
            )

        # Calculate profit for status
        profit_percent = ((current_price - position.entry_price) / position.entry_price) * 100
        peak_profit = ((peak_price - position.entry_price) / position.entry_price) * 100

        # Hold position
        status = f"RSI={rsi.value:.1f}, PnL={profit_percent:.2f}%, Peak={peak_profit:.2f}%"
        if trailing_stop_active or should_activate:
            status += " [Trailing Stop Active]"

        return Signal(
            signal_type=SignalType.HOLD,
            pair=market_data.pair,
            strength=0.0,
            price=current_price,
            timestamp=timestamp,
            reason=f"Hold position: {status}",
            indicators=indicators
        )

    def check_trailing_stop(
        self,
        position: Position,
        current_price: float,
        peak_price: float,
        trailing_stop_active: bool
    ) -> tuple[bool, bool, str]:
        """
        Check trailing stop conditions.

        Args:
            position: Current position.
            current_price: Current market price.
            peak_price: Highest price reached since entry.
            trailing_stop_active: Whether trailing stop has been activated.

        Returns:
            Tuple of (should_close, should_activate_trailing, reason).
        """
        entry_price = position.entry_price
        side = position.side

        # Calculate current profit percentage
        if side == "long":
            profit_percent = ((current_price - entry_price) / entry_price) * 100
            peak_profit_percent = ((peak_price - entry_price) / entry_price) * 100
            drop_from_peak_percent = ((peak_price - current_price) / peak_price) * 100
        else:
            profit_percent = ((entry_price - current_price) / entry_price) * 100
            peak_profit_percent = ((entry_price - peak_price) / entry_price) * 100
            drop_from_peak_percent = ((current_price - peak_price) / peak_price) * 100

        # Check if we should activate trailing stop (reached activation threshold)
        should_activate = False
        if not trailing_stop_active and peak_profit_percent >= self.trailing_stop_activation_percent:
            should_activate = True
            logger.info(
                f"Trailing stop activation threshold reached: {peak_profit_percent:.2f}% profit",
                pair=position.pair,
                activation_threshold=self.trailing_stop_activation_percent
            )

        # Check if trailing stop should trigger exit
        if trailing_stop_active or should_activate:
            if drop_from_peak_percent >= self.trailing_stop_percent:
                reason = (
                    f"Trailing stop hit: price dropped {drop_from_peak_percent:.2f}% from peak "
                    f"(peak: {peak_price:.2f}, current: {current_price:.2f})"
                )
                return True, should_activate, reason

        # Check initial stop loss if configured (for when trade goes immediately negative)
        if self.initial_stop_loss_percent > 0 and profit_percent <= -self.initial_stop_loss_percent:
            reason = f"Initial stop loss hit: {profit_percent:.2f}% loss"
            return True, should_activate, reason

        return False, should_activate, ""

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

    def should_add_to_position(
        self,
        market_data: MarketData,
        avg_entry_price: float,
        current_price: float
    ) -> tuple[bool, str]:
        """
        Check if Martingale add-on conditions are met.

        Args:
            market_data: Current market data.
            avg_entry_price: Current average entry price.
            current_price: Current market price.

        Returns:
            Tuple of (should_add, reason).
        """
        if not self.martingale_enabled:
            return False, "Martingale disabled"

        # Calculate drop from average entry
        drop_percent = ((avg_entry_price - current_price) / avg_entry_price) * 100

        if drop_percent < self.martingale_add_on_drop:
            return False, f"Price drop {drop_percent:.2f}% < trigger {self.martingale_add_on_drop}%"

        # Calculate signals for indicator checks
        signals = self._calculate_signals(market_data)

        # Check RSI if required
        if self.martingale_require_rsi:
            if not signals.rsi or not signals.rsi.is_oversold:
                rsi_val = signals.rsi.value if signals.rsi else "N/A"
                return False, f"RSI not oversold ({rsi_val})"

        # Check EMA trend if required
        if self.martingale_require_ema:
            if not signals.ema or not signals.ema.price_above_ema:
                return False, "Price not above EMA"

        # All conditions met
        reason = f"Martingale trigger: price dropped {drop_percent:.2f}% from avg entry"
        if self.martingale_require_rsi:
            reason += f", RSI={signals.rsi.value:.1f}"

        logger.info(
            f"Martingale add-on signal: {reason}",
            pair=market_data.pair,
            avg_entry=avg_entry_price,
            current_price=current_price,
            drop_percent=drop_percent
        )

        return True, reason

    def reset(self) -> None:
        """Reset strategy state."""
        super().reset()
        self.rsi.reset()
        self.ema.reset()
        self.volume.reset()
