"""
Scalping trading strategy for high-frequency, small-profit trades.

Enhanced with:
- Stochastic Oscillator for oversold/overbought confirmation
- MACD for momentum direction confirmation
- OBV for volume-price divergence detection
- ATR for dynamic stop-loss/take-profit sizing
- VWAP bands for better exit targets
- Bollinger Bandwidth for volatility regime
- EMA crossover signals for entry confirmation
- Short selling support for bearish regimes

Entry conditions (require 2+ confirmations):
Long:
- RSI < 30 (short-term oversold)
- Stochastic %K < 20 (confirms oversold)
- Price at/below lower Bollinger Band
- Price below VWAP (mean reversion opportunity)
- Volume spike (1.5x average) or OBV bullish divergence
- MACD histogram turning positive
- Bollinger squeeze (potential breakout)
- EMA bullish crossover

Short:
- RSI > 70 (overbought)
- Stochastic %K > 80 (confirms overbought)
- Price at/above upper Bollinger Band
- Price above VWAP
- OBV bearish divergence
- MACD histogram turning negative
- EMA bearish crossover

Exit conditions:
- ATR-based dynamic take profit / stop loss
- RSI overbought/oversold reversal
- Price at opposite Bollinger Band
- VWAP crossover
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
from .signals.stochastic import StochasticIndicator
from .signals.macd import MACDIndicator
from .signals.obv import OBVIndicator
from .signals.atr import ATRIndicator

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

    # Stochastic Oscillator
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # OBV
    obv_sma_period: int = 20

    # ATR for dynamic stops
    atr_period: int = 14
    atr_stop_multiplier: float = 1.5
    atr_tp_multiplier: float = 2.0
    use_atr_stops: bool = True  # Use ATR-based dynamic stops

    # Risk management (static fallbacks when ATR not available)
    take_profit_percent: float = 1.5  # Take profit at 1.5%
    stop_loss_percent: float = 0.75  # Stop loss at 0.75%

    # Entry confirmation
    min_confirmations: int = 2  # Require 2+ signals to align

    # Fees consideration
    fee_percent: float = 0.26  # Kraken taker fee
    min_profit_after_fees: float = 0.5  # Minimum net profit target

    # EMA trend filter
    ema_filter_enabled: bool = True  # Block entries during bearish trends
    ema_bearish_threshold: float = -0.5  # Block entry if EMA trend strength below this %

    # Short selling
    shorting_enabled: bool = True  # Enable short selling in bear regimes
    short_min_confirmations: int = 3  # Require more confirmations for shorts


class ScalpingStrategy(BaseStrategy):
    """
    Enhanced scalping strategy with multi-indicator confirmation.

    Combines 8 indicators for high-probability trades:
    - RSI + Stochastic: dual oscillator oversold/overbought confirmation
    - MACD: momentum direction filter
    - OBV: volume-price divergence detection
    - Bollinger Bands: volatility and mean reversion
    - VWAP: institutional price level
    - Volume: spike confirmation
    - ATR: dynamic stop-loss/take-profit sizing
    - EMA: trend direction and crossover

    Supports both long and short positions.
    """

    def __init__(self, config: Optional[ScalpingConfig] = None):
        super().__init__(name="scalping")

        self.config = config or ScalpingConfig()

        # Core oscillators
        self.rsi = RSIIndicator(
            period=self.config.rsi_period,
            oversold=self.config.rsi_oversold,
            overbought=self.config.rsi_overbought
        )

        self.stochastic = StochasticIndicator(
            k_period=self.config.stoch_k_period,
            d_period=self.config.stoch_d_period,
            oversold=self.config.stoch_oversold,
            overbought=self.config.stoch_overbought
        )

        # Momentum
        self.macd = MACDIndicator(
            fast_period=self.config.macd_fast,
            slow_period=self.config.macd_slow,
            signal_period=self.config.macd_signal
        )

        # Volume indicators
        self.obv = OBVIndicator(sma_period=self.config.obv_sma_period)
        self.volume = VolumeIndicator(
            sma_period=20,
            high_volume_threshold=self.config.volume_spike_threshold
        )

        # Bands and levels
        self.bollinger = BollingerBandsIndicator(
            period=self.config.bb_period,
            std_dev_multiplier=self.config.bb_std_dev,
            squeeze_threshold=self.config.bb_squeeze_threshold
        )

        self.vwap = VWAPIndicator(
            threshold_percent=self.config.vwap_threshold_percent
        )

        # Volatility
        self.atr = ATRIndicator(
            period=self.config.atr_period,
            stop_multiplier=self.config.atr_stop_multiplier,
            tp_multiplier=self.config.atr_tp_multiplier
        )

        # Trend
        self.ema_short = EMAIndicator(short_period=5, long_period=13)

        self._is_initialized = True

        logger.info(
            "Scalping strategy initialized (enhanced)",
            rsi_period=self.config.rsi_period,
            take_profit=self.config.take_profit_percent,
            stop_loss=self.config.stop_loss_percent,
            min_confirmations=self.config.min_confirmations,
            shorting_enabled=self.config.shorting_enabled,
            use_atr_stops=self.config.use_atr_stops,
        )

    def analyze(
        self,
        market_data: MarketData,
        current_position: Optional[Position] = None
    ) -> Signal:
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

        # No position - check entry conditions (long and short)
        return self._check_entry_conditions(market_data, signals, timestamp)

    def _validate_data(self, market_data: MarketData) -> bool:
        min_candles = max(
            self.config.bb_period,
            self.config.rsi_period,
            self.config.macd_slow + self.config.macd_signal,
            self.config.stoch_k_period + self.config.stoch_d_period,
            self.config.atr_period,
            self.config.obv_sma_period
        ) + 5
        return len(market_data.ohlc) >= min_candles

    def _calculate_signals(self, market_data: MarketData) -> dict:
        """Calculate all indicator signals."""
        closes = [c.close for c in market_data.ohlc]
        highs = [c.high for c in market_data.ohlc]
        lows = [c.low for c in market_data.ohlc]
        volumes = [c.volume for c in market_data.ohlc]

        # Core oscillators
        rsi_result = self.rsi.calculate(closes)
        stoch_result = self.stochastic.calculate(highs, lows, closes)

        # Momentum
        macd_result = self.macd.calculate(closes)

        # Volume
        volume_result = self.volume.calculate(volumes)
        obv_result = self.obv.calculate(closes, volumes)

        # Bands
        bb_result = self.bollinger.calculate(closes)
        vwap_result = self.vwap.calculate(highs, lows, closes, volumes)

        # VWAP with bands for exit targets
        vwap_bands = self.vwap.calculate_with_bands(highs, lows, closes, volumes)

        # Bollinger with trend for bandwidth analysis
        bb_trend = self.bollinger.calculate_with_trend(closes)

        # Volatility
        atr_result = self.atr.calculate(highs, lows, closes)

        # Trend
        ema_result = self.ema_short.calculate(closes)

        return {
            "rsi": rsi_result,
            "stochastic": stoch_result,
            "macd": macd_result,
            "volume": volume_result,
            "obv": obv_result,
            "bollinger": bb_result,
            "vwap": vwap_result,
            "vwap_bands": vwap_bands,
            "bb_trend": bb_trend,
            "atr": atr_result,
            "ema": ema_result,
            "current_price": closes[-1] if closes else 0,
        }

    def _get_dynamic_stops(self, signals: dict) -> Tuple[float, float]:
        """Get ATR-based dynamic stop loss and take profit percentages."""
        atr = signals.get("atr")

        if self.config.use_atr_stops and atr:
            # ATR-based dynamic stops
            stop_pct = atr.atr_percent * self.config.atr_stop_multiplier
            tp_pct = atr.atr_percent * self.config.atr_tp_multiplier

            # Clamp to reasonable bounds
            stop_pct = max(0.3, min(stop_pct, 5.0))
            tp_pct = max(0.5, min(tp_pct, 10.0))
            return stop_pct, tp_pct

        # Fallback to static config
        return self.config.stop_loss_percent, self.config.take_profit_percent

    def _check_entry_conditions(
        self,
        market_data: MarketData,
        signals: dict,
        timestamp: datetime
    ) -> Signal:
        """Check for scalping entry signals (both long and short)."""
        # Try long entry first
        long_signal = self._check_long_entry(market_data, signals, timestamp)
        if long_signal.signal_type == SignalType.BUY:
            return long_signal

        # Try short entry if enabled
        if self.config.shorting_enabled:
            short_signal = self._check_short_entry(market_data, signals, timestamp)
            if short_signal.signal_type == SignalType.SELL_SHORT:
                return short_signal

        return self._no_signal(
            market_data.pair,
            long_signal.reason,
            timestamp
        )

    def _check_long_entry(
        self,
        market_data: MarketData,
        signals: dict,
        timestamp: datetime
    ) -> Signal:
        """Check for long entry conditions."""
        confirmations = 0
        reasons = []

        rsi = signals.get("rsi")
        stoch = signals.get("stochastic")
        macd = signals.get("macd")
        bb = signals.get("bollinger")
        vwap = signals.get("vwap")
        volume = signals.get("volume")
        obv = signals.get("obv")
        ema = signals.get("ema")
        bb_trend = signals.get("bb_trend")
        current_price = signals.get("current_price", 0)

        # === LONG ENTRY CONDITIONS ===

        # 1. RSI oversold
        if rsi and rsi.is_oversold:
            confirmations += 1
            reasons.append(f"RSI oversold ({rsi.value:.1f})")

        # 2. Stochastic oversold (strong confirmation with RSI)
        if stoch and stoch.is_oversold:
            confirmations += 1
            reasons.append(f"Stoch oversold (%K={stoch.k_value:.1f})")

        # 3. Price at/below lower Bollinger Band
        if bb and bb.is_below_lower:
            confirmations += 1
            reasons.append(f"Price at lower BB ({bb.percent_b:.2f})")

        # 4. Price below VWAP (mean reversion opportunity)
        if vwap and vwap.is_below:
            confirmations += 1
            reasons.append(f"Below VWAP ({vwap.price_vs_vwap:.2f}%)")

        # 5. Volume confirmation
        if volume and volume.is_high_volume:
            confirmations += 1
            reasons.append(f"Volume spike ({volume.volume_ratio:.1f}x)")

        # 6. OBV bullish divergence (very strong signal)
        if obv and obv.divergence == "bullish_divergence":
            confirmations += 1.5  # Extra weight for divergence
            reasons.append("OBV bullish divergence")

        # 7. MACD histogram positive or turning positive
        if macd and (macd.histogram > 0 or macd.histogram_increasing):
            confirmations += 0.5
            reasons.append(f"MACD {'positive' if macd.histogram > 0 else 'turning up'}")

        # 8. Bollinger squeeze (potential breakout) + bandwidth expanding
        if bb and bb.is_squeeze:
            confirmations += 0.5
            reasons.append("BB squeeze detected")

        # 9. EMA bullish crossover
        if ema and ema.crossover_signal == "bullish":
            confirmations += 1
            reasons.append("EMA bullish crossover")

        # EMA trend filter - block entries in strong downtrends
        if self.config.ema_filter_enabled and ema:
            if ema.trend_strength < self.config.ema_bearish_threshold:
                # Don't block if we have very strong oversold signals
                strong_oversold = (
                    rsi and rsi.value < 20
                    and stoch and stoch.k_value < 15
                )
                if not strong_oversold:
                    return self._no_signal(
                        market_data.pair,
                        f"EMA bearish filter: trend strength {ema.trend_strength:.2f}% "
                        f"(threshold: {self.config.ema_bearish_threshold}%)",
                        timestamp
                    )

        # Check if we have enough confirmations
        if confirmations >= self.config.min_confirmations:
            # Get dynamic stops
            stop_pct, tp_pct = self._get_dynamic_stops(signals)

            # Verify profit potential after fees
            total_fees = self.config.fee_percent * 2
            net_profit = tp_pct - total_fees

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
                strength=min(confirmations / 5, 1.0),
                reason=f"Scalp long: {', '.join(reasons)}",
                indicators={
                    "rsi": rsi.value if rsi else None,
                    "stochastic_k": stoch.k_value if stoch else None,
                    "macd_histogram": macd.histogram if macd else None,
                    "obv_divergence": obv.divergence if obv else None,
                    "bb_percent_b": bb.percent_b if bb else None,
                    "vwap_distance": vwap.price_vs_vwap if vwap else None,
                    "volume_ratio": volume.volume_ratio if volume else None,
                    "atr_percent": signals.get("atr").atr_percent if signals.get("atr") else None,
                    "dynamic_stop": stop_pct,
                    "dynamic_tp": tp_pct,
                    "confirmations": confirmations,
                }
            )

        return self._no_signal(
            market_data.pair,
            f"Long: {confirmations:.1f} confirmations (need {self.config.min_confirmations})",
            timestamp
        )

    def _check_short_entry(
        self,
        market_data: MarketData,
        signals: dict,
        timestamp: datetime
    ) -> Signal:
        """Check for short entry conditions (bear regime)."""
        confirmations = 0
        reasons = []

        rsi = signals.get("rsi")
        stoch = signals.get("stochastic")
        macd = signals.get("macd")
        bb = signals.get("bollinger")
        vwap = signals.get("vwap")
        obv = signals.get("obv")
        ema = signals.get("ema")
        current_price = signals.get("current_price", 0)

        # === SHORT ENTRY CONDITIONS ===

        # 1. RSI overbought
        if rsi and rsi.is_overbought:
            confirmations += 1
            reasons.append(f"RSI overbought ({rsi.value:.1f})")

        # 2. Stochastic overbought
        if stoch and stoch.is_overbought:
            confirmations += 1
            reasons.append(f"Stoch overbought (%K={stoch.k_value:.1f})")

        # 3. Price at/above upper Bollinger Band
        if bb and bb.is_above_upper:
            confirmations += 1
            reasons.append(f"Price at upper BB ({bb.percent_b:.2f})")

        # 4. Price above VWAP (mean reversion short)
        if vwap and vwap.is_above and vwap.price_vs_vwap > 1.0:
            confirmations += 1
            reasons.append(f"Above VWAP (+{vwap.price_vs_vwap:.2f}%)")

        # 5. OBV bearish divergence
        if obv and obv.divergence == "bearish_divergence":
            confirmations += 1.5
            reasons.append("OBV bearish divergence")

        # 6. MACD histogram negative and falling
        if macd and macd.histogram < 0 and not macd.histogram_increasing:
            confirmations += 0.5
            reasons.append("MACD negative & falling")

        # 7. EMA bearish crossover
        if ema and ema.crossover_signal == "bearish":
            confirmations += 1
            reasons.append("EMA bearish crossover")

        # 8. EMA downtrend (trend confirmation for short)
        if ema and ema.trend_strength < -0.3:
            confirmations += 0.5
            reasons.append(f"EMA downtrend ({ema.trend_strength:.2f}%)")

        # Shorts require more confirmations (higher bar)
        if confirmations >= self.config.short_min_confirmations:
            stop_pct, tp_pct = self._get_dynamic_stops(signals)

            total_fees = self.config.fee_percent * 2
            net_profit = tp_pct - total_fees

            if net_profit < self.config.min_profit_after_fees:
                return self._no_signal(
                    market_data.pair,
                    f"Short net profit {net_profit:.2f}% below minimum"
                )

            return Signal(
                signal_type=SignalType.SELL_SHORT,
                pair=market_data.pair,
                price=current_price,
                timestamp=timestamp,
                strength=min(confirmations / 5, 1.0),
                reason=f"Scalp short: {', '.join(reasons)}",
                indicators={
                    "rsi": rsi.value if rsi else None,
                    "stochastic_k": stoch.k_value if stoch else None,
                    "macd_histogram": macd.histogram if macd else None,
                    "obv_divergence": obv.divergence if obv else None,
                    "bb_percent_b": bb.percent_b if bb else None,
                    "vwap_distance": vwap.price_vs_vwap if vwap else None,
                    "atr_percent": signals.get("atr").atr_percent if signals.get("atr") else None,
                    "dynamic_stop": stop_pct,
                    "dynamic_tp": tp_pct,
                    "confirmations": confirmations,
                    "side": "short",
                }
            )

        return self._no_signal(
            market_data.pair,
            f"Short: {confirmations:.1f} confirmations (need {self.config.short_min_confirmations})",
            timestamp
        )

    def _check_exit_conditions(
        self,
        market_data: MarketData,
        position: Position,
        signals: dict,
        timestamp: datetime
    ) -> Signal:
        """Check for exit signals (both long and short positions)."""
        if position.side == "short":
            return self._check_short_exit(market_data, position, signals, timestamp)
        return self._check_long_exit(market_data, position, signals, timestamp)

    def _check_long_exit(
        self,
        market_data: MarketData,
        position: Position,
        signals: dict,
        timestamp: datetime
    ) -> Signal:
        """Check for long position exit signals."""
        current_price = signals.get("current_price", 0)
        entry_price = position.entry_price
        pnl_percent = ((current_price - entry_price) / entry_price) * 100

        rsi = signals.get("rsi")
        stoch = signals.get("stochastic")
        bb = signals.get("bollinger")
        vwap = signals.get("vwap")
        macd = signals.get("macd")

        # Get dynamic stops
        stop_pct, tp_pct = self._get_dynamic_stops(signals)

        # 1. Dynamic take profit hit
        if pnl_percent >= tp_pct:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                pair=market_data.pair,
                price=current_price,
                timestamp=timestamp,
                strength=1.0,
                reason=f"Take profit: {pnl_percent:.2f}% (ATR target: {tp_pct:.2f}%)",
                indicators={"pnl_percent": pnl_percent, "dynamic_tp": tp_pct}
            )

        # 2. Dynamic stop loss hit
        if pnl_percent <= -stop_pct:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                pair=market_data.pair,
                price=current_price,
                timestamp=timestamp,
                strength=1.0,
                reason=f"Stop loss: {pnl_percent:.2f}% (ATR stop: -{stop_pct:.2f}%)",
                indicators={"pnl_percent": pnl_percent, "dynamic_stop": stop_pct}
            )

        # Minimum profit for early exits (75% of dynamic TP)
        min_early_exit_profit = tp_pct * 0.75

        # 3. RSI + Stochastic both overbought (strong exit signal)
        if (rsi and rsi.is_overbought and stoch and stoch.is_overbought
                and pnl_percent >= min_early_exit_profit):
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                pair=market_data.pair,
                price=current_price,
                timestamp=timestamp,
                strength=0.9,
                reason=f"RSI+Stoch overbought (RSI={rsi.value:.1f}, %K={stoch.k_value:.1f}), P&L: {pnl_percent:.2f}%",
                indicators={"rsi": rsi.value, "stoch_k": stoch.k_value, "pnl_percent": pnl_percent}
            )

        # 4. RSI overbought alone (weaker signal, needs more profit)
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

        # 5. Price at upper Bollinger Band (mean reversion exit)
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

        # 6. MACD bearish crossover with decent profit
        if macd and macd.crossover_signal == "bearish" and pnl_percent >= min_early_exit_profit:
            return Signal(
                signal_type=SignalType.CLOSE_LONG,
                pair=market_data.pair,
                price=current_price,
                timestamp=timestamp,
                strength=0.7,
                reason=f"MACD bearish crossover, P&L: {pnl_percent:.2f}%",
                indicators={"macd_histogram": macd.histogram, "pnl_percent": pnl_percent}
            )

        # 7. Price crossed above VWAP significantly
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
            f"Holding long, P&L: {pnl_percent:.2f}%",
            timestamp
        )

    def _check_short_exit(
        self,
        market_data: MarketData,
        position: Position,
        signals: dict,
        timestamp: datetime
    ) -> Signal:
        """Check for short position exit signals."""
        current_price = signals.get("current_price", 0)
        entry_price = position.entry_price
        # Short P&L is inverted: profit when price drops
        pnl_percent = ((entry_price - current_price) / entry_price) * 100

        rsi = signals.get("rsi")
        stoch = signals.get("stochastic")
        bb = signals.get("bollinger")
        vwap = signals.get("vwap")
        macd = signals.get("macd")

        stop_pct, tp_pct = self._get_dynamic_stops(signals)

        # 1. Take profit (price dropped enough)
        if pnl_percent >= tp_pct:
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                pair=market_data.pair,
                price=current_price,
                timestamp=timestamp,
                strength=1.0,
                reason=f"Short TP: {pnl_percent:.2f}% (target: {tp_pct:.2f}%)",
                indicators={"pnl_percent": pnl_percent, "dynamic_tp": tp_pct}
            )

        # 2. Stop loss (price rose against us)
        if pnl_percent <= -stop_pct:
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                pair=market_data.pair,
                price=current_price,
                timestamp=timestamp,
                strength=1.0,
                reason=f"Short SL: {pnl_percent:.2f}% (stop: -{stop_pct:.2f}%)",
                indicators={"pnl_percent": pnl_percent, "dynamic_stop": stop_pct}
            )

        min_early_exit_profit = tp_pct * 0.75

        # 3. RSI + Stochastic oversold (cover short)
        if (rsi and rsi.is_oversold and stoch and stoch.is_oversold
                and pnl_percent >= min_early_exit_profit):
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                pair=market_data.pair,
                price=current_price,
                timestamp=timestamp,
                strength=0.9,
                reason=f"RSI+Stoch oversold, cover short. P&L: {pnl_percent:.2f}%",
                indicators={"rsi": rsi.value, "stoch_k": stoch.k_value, "pnl_percent": pnl_percent}
            )

        # 4. Price at lower Bollinger Band (mean reversion exit)
        if bb and bb.is_below_lower and pnl_percent >= min_early_exit_profit:
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                pair=market_data.pair,
                price=current_price,
                timestamp=timestamp,
                strength=0.7,
                reason=f"Price at lower BB, cover short. P&L: {pnl_percent:.2f}%",
                indicators={"bb_percent_b": bb.percent_b, "pnl_percent": pnl_percent}
            )

        # 5. MACD bullish crossover (momentum turning against short)
        if macd and macd.crossover_signal == "bullish" and pnl_percent >= min_early_exit_profit:
            return Signal(
                signal_type=SignalType.CLOSE_SHORT,
                pair=market_data.pair,
                price=current_price,
                timestamp=timestamp,
                strength=0.7,
                reason=f"MACD bullish crossover, cover short. P&L: {pnl_percent:.2f}%",
                indicators={"macd_histogram": macd.histogram, "pnl_percent": pnl_percent}
            )

        return self._no_signal(
            market_data.pair,
            f"Holding short, P&L: {pnl_percent:.2f}%",
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
        - Trigger when price drops 0.3% from peak (long) or rises 0.3% from trough (short)
        """
        entry_price = position.entry_price

        if position.side == "short":
            # Short position: profit when price drops
            pnl_percent = ((entry_price - current_price) / entry_price) * 100
            peak_pnl_percent = ((entry_price - peak_price) / entry_price) * 100
        else:
            pnl_percent = ((current_price - entry_price) / entry_price) * 100
            peak_pnl_percent = ((peak_price - entry_price) / entry_price) * 100

        activation_threshold = self.config.take_profit_percent * 0.75
        trailing_drop_percent = 0.3

        should_activate = False
        if not trailing_stop_active and pnl_percent >= activation_threshold:
            should_activate = True

        if trailing_stop_active:
            if position.side == "short":
                # For shorts, trigger if price rises from trough
                rise_from_trough = ((current_price - peak_price) / peak_price) * 100
                if rise_from_trough >= trailing_drop_percent:
                    return (
                        True,
                        False,
                        f"Short trailing stop: rose {rise_from_trough:.2f}% from trough, P&L: {pnl_percent:.2f}%"
                    )
            else:
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
            "use_atr_stops": self.config.use_atr_stops,
            "atr_stop_multiplier": self.config.atr_stop_multiplier,
            "atr_tp_multiplier": self.config.atr_tp_multiplier,
            "shorting_enabled": self.config.shorting_enabled,
        }
