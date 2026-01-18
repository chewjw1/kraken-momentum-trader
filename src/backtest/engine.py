"""
Backtest engine for simulating trading strategies on historical data.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple
import uuid

from ..exchange.kraken_client import OHLC
from ..observability.metrics import MetricsTracker, Trade
from ..strategy.base_strategy import MarketData, Position, SignalType
from ..strategy.momentum_strategy import MomentumStrategy
from ..config.settings import get_settings

from .config import BacktestConfig
from .position import SimulatedPosition, CompletedTrade


@dataclass
class EquityPoint:
    """Point in the equity curve."""
    timestamp: datetime
    equity: float
    drawdown: float
    drawdown_percent: float


@dataclass
class BacktestResults:
    """Results from a backtest run."""
    config: BacktestConfig
    trades: List[CompletedTrade]
    equity_curve: List[EquityPoint]
    metrics: "PerformanceMetrics"
    start_time: datetime
    end_time: datetime
    candles_processed: int


class BacktestEngine:
    """
    Core backtest simulation engine.

    Processes historical candles one by one and simulates the trading strategy.
    Reuses existing MomentumStrategy for signal generation.
    """

    def __init__(
        self,
        config: BacktestConfig,
        strategy: Optional[MomentumStrategy] = None
    ):
        """
        Initialize the backtest engine.

        Args:
            config: Backtest configuration.
            strategy: Optional strategy instance (creates one if not provided).
        """
        self.config = config
        self.strategy = strategy or MomentumStrategy()

        # State
        self.capital = config.initial_capital
        self.peak_capital = config.initial_capital
        self.position: Optional[SimulatedPosition] = None

        # Results tracking
        self.trades: List[CompletedTrade] = []
        self.equity_curve: List[EquityPoint] = []
        self.metrics_tracker = MetricsTracker(initial_capital=config.initial_capital)

        # Trade counter for IDs
        self._trade_counter = 0

        # Martingale settings from strategy config
        settings = get_settings()
        martingale = settings.risk.martingale
        # Handle case where martingale might be a dict (from YAML loading)
        if isinstance(martingale, dict):
            self.martingale_enabled = martingale.get("enabled", True)
            self.martingale_max_entries = martingale.get("max_entries", 4)
            self.martingale_size_multiplier = martingale.get("size_multiplier", 1.25)
        else:
            self.martingale_enabled = martingale.enabled
            self.martingale_max_entries = martingale.max_entries
            self.martingale_size_multiplier = martingale.size_multiplier

        # Position sizing
        self.max_position_percent = settings.risk.max_position_percent
        self.base_position_usd = config.initial_capital * (self.max_position_percent / 100)

    def run(self, candles: List[OHLC]) -> BacktestResults:
        """
        Run the backtest on historical candles.

        Args:
            candles: List of OHLC candles sorted by timestamp.

        Returns:
            BacktestResults with all metrics and trade history.
        """
        if len(candles) < self.config.lookback_candles:
            raise ValueError(
                f"Need at least {self.config.lookback_candles} candles, got {len(candles)}"
            )

        start_time = datetime.now(timezone.utc)

        # Process each candle
        for i in range(self.config.lookback_candles, len(candles)):
            # Get lookback window for this candle
            lookback_start = i - self.config.lookback_candles
            lookback_candles = candles[lookback_start:i + 1]
            current_candle = candles[i]

            self._process_candle(lookback_candles, current_candle)

        # Close any remaining position at the end
        if self.position:
            final_candle = candles[-1]
            self._close_position(final_candle.close, final_candle.timestamp, "End of backtest")

        end_time = datetime.now(timezone.utc)

        return BacktestResults(
            config=self.config,
            trades=self.trades,
            equity_curve=self.equity_curve,
            metrics=self.metrics_tracker.get_metrics(),
            start_time=start_time,
            end_time=end_time,
            candles_processed=len(candles) - self.config.lookback_candles
        )

    def _process_candle(self, lookback_candles: List[OHLC], current_candle: OHLC) -> None:
        """
        Process a single candle.

        Args:
            lookback_candles: Historical candles for indicator calculation.
            current_candle: The current candle being processed.
        """
        # Build MarketData from candles
        market_data = MarketData.from_ohlc(
            pair=self.config.pair,
            ohlc_data=lookback_candles,
            ticker=None
        )

        current_price = current_candle.close
        candle_low = current_candle.low
        candle_high = current_candle.high
        timestamp = current_candle.timestamp

        if self.position:
            # Update peak price (use candle high for optimistic tracking)
            self.position.update_peak_price(candle_high)

            # Check trailing stop (use candle low for conservative exit check)
            should_close, should_activate, reason = self._check_trailing_stop(candle_low)

            if should_activate and not self.position.trailing_stop_active:
                self.position.activate_trailing_stop()

            if should_close:
                self._close_position(candle_low, timestamp, reason)
                self._record_equity(timestamp)
                return

            # Check Martingale add-on
            if self.martingale_enabled and self.position.num_entries < self.martingale_max_entries:
                should_add, add_reason = self.strategy.should_add_to_position(
                    market_data,
                    self.position.entry_price,
                    current_price
                )
                if should_add:
                    self._add_to_position(current_price, timestamp)

            # Get strategy signal for potential exit
            position_for_strategy = self._create_position_for_strategy(current_price)
            signal = self.strategy.analyze(market_data, position_for_strategy)

            if signal.signal_type == SignalType.CLOSE_LONG:
                self._close_position(current_price, timestamp, signal.reason)

        else:
            # No position - look for entry
            signal = self.strategy.analyze(market_data, None)

            if signal.signal_type == SignalType.BUY:
                self._open_position(current_price, timestamp, signal.reason)

        # Record equity point
        self._record_equity(timestamp)

    def _check_trailing_stop(self, check_price: float) -> Tuple[bool, bool, str]:
        """
        Check trailing stop using the strategy's logic.

        Args:
            check_price: Price to check against (typically candle low).

        Returns:
            Tuple of (should_close, should_activate, reason).
        """
        if not self.position:
            return False, False, ""

        # Create a Position object for the strategy
        position = Position(
            pair=self.position.pair,
            side=self.position.side,
            entry_price=self.position.entry_price,
            current_price=check_price,
            size=self.position.size,
            entry_time=self.position.entry_time
        )

        return self.strategy.check_trailing_stop(
            position,
            check_price,
            self.position.peak_price,
            self.position.trailing_stop_active
        )

    def _create_position_for_strategy(self, current_price: float) -> Position:
        """Create a Position object from SimulatedPosition for strategy use."""
        return Position(
            pair=self.position.pair,
            side=self.position.side,
            entry_price=self.position.entry_price,
            current_price=current_price,
            size=self.position.size,
            entry_time=self.position.entry_time
        )

    def _open_position(self, price: float, timestamp: datetime, reason: str) -> None:
        """Open a new position."""
        # Calculate position size
        position_usd = min(self.base_position_usd, self.capital * 0.95)  # Leave some buffer
        fee = position_usd * (self.config.fee_percent / 100)
        size = (position_usd - fee) / price

        self.position = SimulatedPosition(
            pair=self.config.pair,
            side="long",
            peak_price=price
        )
        self.position.add_entry(
            price=price,
            size=size,
            size_usd=position_usd,
            entry_time=timestamp,
            fee=fee
        )

        self.capital -= position_usd

    def _add_to_position(self, price: float, timestamp: datetime) -> None:
        """Add to existing position (Martingale)."""
        if not self.position:
            return

        # Calculate size with multiplier
        last_size_usd = self.position.last_entry_size_usd
        new_size_usd = min(
            last_size_usd * self.martingale_size_multiplier,
            self.capital * 0.95
        )

        if new_size_usd < 10:  # Minimum position size
            return

        fee = new_size_usd * (self.config.fee_percent / 100)
        size = (new_size_usd - fee) / price

        self.position.add_entry(
            price=price,
            size=size,
            size_usd=new_size_usd,
            entry_time=timestamp,
            fee=fee
        )

        self.capital -= new_size_usd

    def _close_position(self, price: float, timestamp: datetime, reason: str) -> None:
        """Close the current position and record the trade."""
        if not self.position:
            return

        # Calculate P&L
        exit_value = self.position.size * price
        exit_fee = exit_value * (self.config.fee_percent / 100)
        gross_pnl, net_pnl = self.position.calculate_pnl(price, exit_fee)
        pnl_percent = self.position.calculate_pnl_percent(price)

        # Update capital
        self.capital += exit_value - exit_fee

        # Record completed trade
        self._trade_counter += 1
        trade = CompletedTrade(
            trade_id=f"BT-{self._trade_counter:06d}",
            pair=self.position.pair,
            side=self.position.side,
            entry_price=self.position.entry_price,
            exit_price=price,
            size=self.position.size,
            entry_time=self.position.entry_time,
            exit_time=timestamp,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            pnl_percent=pnl_percent,
            entry_fees=self.position.total_fees,
            exit_fee=exit_fee,
            num_entries=self.position.num_entries,
            exit_reason=reason
        )
        self.trades.append(trade)

        # Record in metrics tracker
        metrics_trade = Trade(
            trade_id=trade.trade_id,
            pair=trade.pair,
            side="buy",
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            amount=trade.size,
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
            pnl=trade.net_pnl,
            pnl_percent=trade.pnl_percent,
            fees=trade.entry_fees + trade.exit_fee
        )
        self.metrics_tracker.record_trade(metrics_trade)

        # Clear position
        self.position = None

    def _record_equity(self, timestamp: datetime) -> None:
        """Record current equity point."""
        # Calculate current equity including unrealized P&L
        equity = self.capital
        if self.position:
            # We'd need current price here - simplify by using entry price
            # In practice, equity curve is more accurate when position is closed
            equity += self.position.total_cost_usd

        # Update peak
        if equity > self.peak_capital:
            self.peak_capital = equity

        # Calculate drawdown
        drawdown = self.peak_capital - equity
        drawdown_percent = (drawdown / self.peak_capital * 100) if self.peak_capital > 0 else 0

        self.equity_curve.append(EquityPoint(
            timestamp=timestamp,
            equity=equity,
            drawdown=drawdown,
            drawdown_percent=drawdown_percent
        ))

    def get_current_equity(self) -> float:
        """Get current equity value."""
        equity = self.capital
        if self.position:
            equity += self.position.total_cost_usd
        return equity
