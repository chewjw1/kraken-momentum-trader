"""
Main trading orchestrator.
Coordinates all components for automated trading.
"""

import time
from datetime import datetime, timezone
from typing import Optional

from ..config.settings import Settings, get_settings
from ..exchange.kraken_client import (
    KrakenClient,
    KrakenAPIError,
    OrderSide,
    OrderType,
)
from ..observability.logger import get_logger
from ..observability.metrics import MetricsTracker, Trade
from ..persistence.file_store import StateStore, TradeLogger
from ..risk.risk_manager import RiskManager
from ..strategy.base_strategy import MarketData, Position, SignalType
from ..strategy.momentum_strategy import MomentumStrategy
from .state_machine import TradingState, TradingStateMachine

logger = get_logger(__name__)


class Trader:
    """
    Main trading orchestrator.

    Coordinates:
    - Exchange client for market data and order execution
    - Strategy for signal generation
    - Risk manager for position sizing and safeguards
    - State machine for lifecycle management
    - Persistence for crash recovery
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[KrakenClient] = None,
        strategy: Optional[MomentumStrategy] = None,
    ):
        """
        Initialize the trader.

        Args:
            settings: Application settings. If None, loaded from config.
            client: Kraken client. If None, created from settings.
            strategy: Trading strategy. If None, creates MomentumStrategy.
        """
        self.settings = settings or get_settings()

        # Initialize exchange client
        self.client = client or KrakenClient(
            paper_trading=self.settings.trading.paper_trading
        )

        # Initialize strategy
        self.strategy = strategy or MomentumStrategy(self.settings.strategy)

        # Initialize metrics tracker
        self.metrics = MetricsTracker()

        # Initialize risk manager
        self.risk_manager = RiskManager(
            config=self.settings.risk,
            metrics=self.metrics
        )

        # Initialize state machine
        self.state_machine = TradingStateMachine()

        # Initialize persistence
        self.state_store = StateStore()
        self.trade_logger = TradeLogger()

        # Running flag
        self._running = False
        self._should_stop = False

        logger.info(
            "Trader initialized",
            paper_trading=self.settings.trading.paper_trading,
            pairs=self.settings.trading.pairs,
            strategy=self.strategy.name
        )

    def initialize(self) -> bool:
        """
        Initialize the trader and restore state.

        Returns:
            True if initialization successful.
        """
        try:
            # Try to restore previous state
            saved_state = self.state_store.load_state()
            if saved_state:
                logger.info("Restoring saved state")
                self._restore_state(saved_state)
            else:
                logger.info("No saved state found, starting fresh")

            # Update capital from exchange
            self._update_capital()

            # Check for emergency conditions
            should_exit, reason = self.risk_manager.should_emergency_exit()
            if should_exit:
                logger.risk_event(
                    event_type="startup_emergency",
                    details=reason
                )
                # Will handle emergency exit in main loop

            self.state_machine.initialize_complete()
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.state_machine.set_error(str(e))
            return False

    def run(self) -> None:
        """
        Run the main trading loop.

        Continuously:
        1. Check risk conditions
        2. Analyze market for signals
        3. Execute trades based on signals
        4. Manage open positions
        5. Persist state
        """
        if not self.initialize():
            logger.error("Failed to initialize, not starting")
            return

        self._running = True
        self._should_stop = False

        logger.info("Starting trading loop")

        try:
            while self._running and not self._should_stop:
                self._trading_iteration()

                # Sleep between iterations
                time.sleep(self.settings.trading.check_interval_seconds)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            self.state_machine.set_error(str(e))
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Signal the trading loop to stop."""
        logger.info("Stop requested")
        self._should_stop = True

    def _trading_iteration(self) -> None:
        """Execute one iteration of the trading loop."""
        try:
            # Check emergency exit conditions
            should_exit, reason = self.risk_manager.should_emergency_exit()
            if should_exit:
                self._handle_emergency_exit(reason)
                return

            # Check if trading is allowed
            risk_check = self.risk_manager.can_trade()
            if not risk_check.allowed:
                logger.info(f"Trading paused: {risk_check.reason}")
                return

            # Process each trading pair
            for pair in self.settings.trading.pairs:
                self._process_pair(pair)

            # Save state after each iteration
            self._save_state()

        except KrakenAPIError as e:
            logger.error(f"API error: {e}")
            # Continue on API errors, don't crash
        except Exception as e:
            logger.error(f"Iteration error: {e}")
            raise

    def _process_pair(self, pair: str) -> None:
        """
        Process trading logic for a single pair.

        Args:
            pair: Trading pair to process.
        """
        logger.debug(f"Processing {pair}")

        # Get market data
        market_data = self._get_market_data(pair)
        if not market_data:
            return

        # Check if we have a position in this pair
        current_position = self._get_current_position(pair, market_data)

        # Update peak price if we have a position
        if current_position and self.state_machine.has_position:
            current_price = market_data.ticker.last if market_data.ticker else 0
            self.state_machine.update_peak_price(current_price)

        # Start analysis
        self.state_machine.start_analysis()

        # Get peak price and trailing stop status for strategy
        peak_price = None
        trailing_stop_active = False
        if self.state_machine.has_position:
            pos = self.state_machine.position
            peak_price = pos.peak_price
            trailing_stop_active = pos.trailing_stop_active

        # Get signal from strategy (pass trailing stop info)
        signal = self.strategy.analyze(market_data, current_position)

        # Check if trailing stop should be activated
        if signal.indicators.get("_should_activate_trailing", False):
            self.state_machine.activate_trailing_stop()

        logger.debug(
            f"Signal for {pair}: {signal.signal_type.value}",
            strength=signal.strength,
            reason=signal.reason
        )

        # Check for Martingale add-on opportunity if we have a position
        if current_position and self.state_machine.has_position:
            current_price = market_data.ticker.last if market_data.ticker else 0
            avg_entry = self.state_machine.position.entry_price

            should_add, add_reason = self.strategy.should_add_to_position(
                market_data=market_data,
                avg_entry_price=avg_entry,
                current_price=current_price
            )

            if should_add:
                self._add_to_position(market_data, add_reason)
                self.state_machine.analysis_complete(has_signal=False)
                return

        # Handle signal
        if signal.is_actionable():
            self._handle_signal(signal, current_position, market_data)
        else:
            self.state_machine.analysis_complete(has_signal=False)

    def _get_market_data(self, pair: str) -> Optional[MarketData]:
        """
        Fetch market data for a pair.

        Args:
            pair: Trading pair.

        Returns:
            MarketData or None if fetch failed.
        """
        try:
            # Get OHLC data (last 100 candles for indicator calculation)
            ohlc = self.client.get_ohlc(pair, interval=60)

            # Get current ticker
            ticker = self.client.get_ticker(pair)

            if len(ohlc) < 50:
                logger.warning(f"Insufficient OHLC data for {pair}: {len(ohlc)} candles")
                return None

            return MarketData.from_ohlc(pair, ohlc, ticker)

        except Exception as e:
            logger.error(f"Failed to get market data for {pair}: {e}")
            return None

    def _get_current_position(
        self,
        pair: str,
        market_data: MarketData
    ) -> Optional[Position]:
        """
        Get current position for a pair if exists.

        Args:
            pair: Trading pair.
            market_data: Current market data.

        Returns:
            Position or None.
        """
        if not self.state_machine.has_position:
            return None

        pos = self.state_machine.position
        if pos.pair != pair:
            return None

        current_price = market_data.ticker.last if market_data.ticker else 0

        return Position(
            pair=pos.pair,
            side=pos.side,
            entry_price=pos.entry_price,
            current_price=current_price,
            size=pos.size,
            entry_time=pos.entry_time
        )

    def _handle_signal(
        self,
        signal,
        current_position: Optional[Position],
        market_data: MarketData
    ) -> None:
        """
        Handle a trading signal.

        Args:
            signal: Trading signal to handle.
            current_position: Current position if any.
            market_data: Current market data.
        """
        if signal.signal_type == SignalType.BUY:
            if current_position:
                logger.info("Already in position, ignoring buy signal")
                self.state_machine.analysis_complete(has_signal=False)
                return
            self._enter_position(signal, market_data)

        elif signal.signal_type in (SignalType.SELL, SignalType.CLOSE_LONG):
            if not current_position:
                logger.info("No position to close")
                self.state_machine.analysis_complete(has_signal=False)
                return
            self._exit_position(signal, current_position, market_data)

        else:
            self.state_machine.analysis_complete(has_signal=False)

    def _enter_position(self, signal, market_data: MarketData) -> None:
        """
        Enter a new position.

        Args:
            signal: Entry signal.
            market_data: Current market data.
        """
        pair = signal.pair
        price = signal.price

        # Calculate position size
        size_check = self.risk_manager.check_position_size(
            self.settings.risk.max_position_size_usd,
            pair
        )

        if not size_check.allowed:
            logger.info(f"Position size rejected: {size_check.reason}")
            self.state_machine.analysis_complete(has_signal=False)
            return

        position_size_usd = size_check.adjusted_size
        position_size_units = position_size_usd / price

        logger.info(
            f"Entering position: BUY {position_size_units:.6f} {pair} @ {price:.2f}",
            size_usd=position_size_usd,
            trailing_stop_activation=self.settings.risk.trailing_stop_activation_percent,
            trailing_stop_percent=self.settings.risk.trailing_stop_percent
        )

        self.state_machine.analysis_complete(has_signal=True)

        try:
            # Place order
            order = self.client.place_order(
                pair=pair,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                volume=position_size_units
            )

            # Update state (no stop_loss/take_profit - using trailing stop)
            self.state_machine.open_position(
                pair=pair,
                side="long",
                entry_price=order.price or price,
                size=position_size_units,
                size_usd=position_size_usd,
                order_id=order.order_id
            )

            # Update exposure
            self.risk_manager.update_exposure(position_size_usd)

            # Log trade
            self.trade_logger.log_trade(
                action="entry",
                pair=pair,
                side="buy",
                price=order.price or price,
                amount=position_size_units,
                order_id=order.order_id,
                signal_strength=signal.strength,
                signal_reason=signal.reason
            )

        except Exception as e:
            logger.error(f"Failed to place entry order: {e}")
            self.state_machine.set_error(str(e))

    def _add_to_position(self, market_data: MarketData, reason: str) -> None:
        """
        Add to existing position (Martingale).

        Args:
            market_data: Current market data.
            reason: Reason for adding.
        """
        position = self.state_machine.position
        if not position:
            logger.warning("Cannot add to position: no position exists")
            return

        pair = position.pair
        current_price = market_data.ticker.last if market_data.ticker else 0

        # Calculate add-on size based on last entry
        last_entry_size_usd = position.last_entry_size_usd
        add_on_size_usd = self.risk_manager.calculate_martingale_size(last_entry_size_usd)

        # Check if add-on is allowed
        add_check = self.risk_manager.can_add_to_position(
            additional_size_usd=add_on_size_usd,
            current_position_exposure=position.total_cost_usd,
            num_entries=position.num_entries,
            pair=pair
        )

        if not add_check.allowed:
            logger.info(f"Martingale add-on rejected: {add_check.reason}")
            return

        add_on_size_usd = add_check.adjusted_size
        add_on_size_units = add_on_size_usd / current_price

        logger.info(
            f"Martingale add-on: BUY {add_on_size_units:.6f} {pair} @ {current_price:.2f}",
            entry_num=position.num_entries + 1,
            size_usd=add_on_size_usd,
            current_avg=position.entry_price,
            reason=reason
        )

        try:
            # Place order
            order = self.client.place_order(
                pair=pair,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                volume=add_on_size_units
            )

            # Add to position
            self.state_machine.add_to_position(
                entry_price=order.price or current_price,
                size=add_on_size_units,
                size_usd=add_on_size_usd,
                order_id=order.order_id
            )

            # Update exposure
            new_exposure = position.total_cost_usd
            self.risk_manager.update_exposure(new_exposure)

            # Log trade
            self.trade_logger.log_trade(
                action="martingale_add",
                pair=pair,
                side="buy",
                price=order.price or current_price,
                amount=add_on_size_units,
                order_id=order.order_id,
                signal_strength=1.0,
                signal_reason=reason
            )

            logger.info(
                f"Martingale add-on complete: new avg entry {position.entry_price:.2f}, "
                f"total size {position.size:.6f}, entries {position.num_entries}"
            )

        except Exception as e:
            logger.error(f"Failed to place Martingale add-on order: {e}")

    def _exit_position(
        self,
        signal,
        position: Position,
        market_data: MarketData
    ) -> None:
        """
        Exit an existing position.

        Args:
            signal: Exit signal.
            position: Position to close.
            market_data: Current market data.
        """
        pair = position.pair
        price = signal.price
        size = position.size

        logger.info(
            f"Exiting position: SELL {size:.6f} {pair} @ {price:.2f}",
            reason=signal.reason
        )

        self.state_machine.start_exit(signal.reason)

        try:
            # Place exit order
            order = self.client.place_order(
                pair=pair,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                volume=size
            )

            exit_price = order.price or price

            # Calculate P&L
            pnl = (exit_price - position.entry_price) * size
            pnl_percent = ((exit_price - position.entry_price) / position.entry_price) * 100

            # Record trade result
            self.risk_manager.record_trade_result(pnl, pair)

            # Record in metrics
            trade = Trade(
                trade_id=order.order_id,
                pair=pair,
                side="long",
                entry_price=position.entry_price,
                exit_price=exit_price,
                amount=size,
                entry_time=position.entry_time,
                exit_time=datetime.now(timezone.utc),
                pnl=pnl,
                pnl_percent=pnl_percent,
                fees=order.fee
            )
            self.metrics.record_trade(trade)

            # Update state
            self.state_machine.close_position(exit_price, pnl)

            # Update exposure
            self.risk_manager.update_exposure(0)

            # Log trade
            self.trade_logger.log_trade(
                action="exit",
                pair=pair,
                side="sell",
                price=exit_price,
                amount=size,
                order_id=order.order_id,
                entry_price=position.entry_price,
                pnl=pnl,
                pnl_percent=pnl_percent,
                reason=signal.reason
            )

            # Save trade to history
            self.state_store.save_trade({
                "trade_id": order.order_id,
                "pair": pair,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "size": size,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "entry_time": position.entry_time.isoformat(),
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "reason": signal.reason
            })

            logger.info(
                f"Position closed: P&L ${pnl:.2f} ({pnl_percent:.2f}%)",
                trade_id=order.order_id
            )

        except Exception as e:
            logger.error(f"Failed to place exit order: {e}")
            self.state_machine.set_error(str(e))

    def _handle_emergency_exit(self, reason: str) -> None:
        """
        Handle emergency exit of all positions.

        Args:
            reason: Reason for emergency exit.
        """
        logger.risk_event(
            event_type="emergency_exit",
            details=reason
        )

        if self.state_machine.has_position:
            pos = self.state_machine.position

            try:
                # Emergency market sell
                order = self.client.place_order(
                    pair=pos.pair,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    volume=pos.size
                )

                exit_price = order.price or 0
                pnl = (exit_price - pos.entry_price) * pos.size

                self.state_machine.close_position(exit_price, pnl)
                self.risk_manager.update_exposure(0)

                logger.info(
                    f"Emergency exit completed: P&L ${pnl:.2f}",
                    reason=reason
                )

            except Exception as e:
                logger.error(f"Emergency exit failed: {e}")

        # Stop trading
        self.state_machine.stop(reason)
        self._should_stop = True

    def _update_capital(self) -> None:
        """Update capital from exchange balances."""
        try:
            balances = self.client.get_balances()

            # Calculate total capital (USD + value of positions)
            usd_balance = balances.get("USD")
            if usd_balance:
                capital = usd_balance.total
                self.risk_manager.set_capital(capital)
                logger.info(f"Capital updated: ${capital:.2f}")

        except Exception as e:
            logger.warning(f"Failed to update capital: {e}")

    def _save_state(self) -> None:
        """Save current state to persistence."""
        state = {
            "state_machine": self.state_machine.to_dict(),
            "risk_manager": self.risk_manager.to_dict(),
            "metrics": self.metrics.to_dict(),
        }
        self.state_store.save_state(state)

    def _restore_state(self, state: dict) -> None:
        """
        Restore state from saved data.

        Args:
            state: Saved state dictionary.
        """
        if "state_machine" in state:
            self.state_machine.from_dict(state["state_machine"])

        if "risk_manager" in state:
            self.risk_manager.from_dict(state["risk_manager"])

        if "metrics" in state:
            self.metrics = MetricsTracker.from_dict(state["metrics"])
            self.risk_manager.metrics = self.metrics

    def _shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("Shutting down trader")

        # Save final state
        self._save_state()

        # Backup data
        self.state_store.backup_all()

        # Close client
        self.client.close()

        self._running = False
        logger.info("Trader shutdown complete")

    def get_status(self) -> dict:
        """
        Get current trader status.

        Returns:
            Status dictionary.
        """
        state = self.state_machine.get_full_state()
        risk_limits = self.risk_manager.get_limits()
        metrics = self.metrics.get_metrics()

        return {
            "running": self._running,
            "state": state.current_state.value,
            "has_position": state.position is not None,
            "position": {
                "pair": state.position.pair,
                "side": state.position.side,
                "entry_price": state.position.entry_price,
                "size": state.position.size,
                "peak_price": state.position.peak_price,
                "trailing_stop_active": state.position.trailing_stop_active,
                "num_entries": state.position.num_entries,
                "total_cost_usd": state.position.total_cost_usd,
            } if state.position else None,
            "risk_limits": {
                "exposure_available": risk_limits.exposure_available,
                "max_position_size": risk_limits.max_position_size,
                "circuit_breaker": risk_limits.circuit_breaker_state.value,
            },
            "metrics": {
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "total_pnl": metrics.total_pnl,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown_percent": metrics.max_drawdown_percent,
            },
            "paper_trading": self.settings.trading.paper_trading,
        }
