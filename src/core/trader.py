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
from ..notifications.discord import DiscordNotifier, TradeNotification, DailySummaryData
from ..notifications.scheduler import DailySummaryScheduler
from ..observability.logger import get_logger
from ..observability.metrics import MetricsTracker, Trade
from ..persistence.sqlite_store import SQLiteStore
from ..persistence.file_store import TradeLogger
from ..risk.risk_manager import RiskManager
from ..strategy.base_strategy import MarketData, Position, SignalType
from ..strategy.momentum_strategy import MomentumStrategy
from ..ml.predictor import SignalPredictor
from ..ml.retrainer import AutoRetrainer
from ..ml.trainer import ModelTrainer
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

        # Initialize persistence (SQLite for all data)
        self.db = SQLiteStore(db_path=self.settings.persistence.db_path)
        self.trade_logger = TradeLogger()  # Keep append-only log for audit

        # Initialize notifications
        notif_config = self.settings.notifications
        self.notifier = DiscordNotifier(
            webhook_url=notif_config.discord.webhook_url,
            enabled=notif_config.discord.enabled
        )

        # Initialize daily summary scheduler
        self.summary_scheduler = DailySummaryScheduler(
            callback=self._send_daily_summary,
            target_hour=notif_config.daily_summary.hour,
            target_minute=notif_config.daily_summary.minute
        ) if notif_config.daily_summary.enabled else None

        # Initialize ML components
        ml_config = self.settings.ml
        db_path = self.settings.persistence.db_path

        self.ml_predictor = SignalPredictor(
            model_dir=ml_config.model_dir,
            confidence_threshold=ml_config.confidence_threshold,
            enabled=ml_config.enabled,
            db_path=db_path
        )

        # Initialize trainer and auto-retrainer
        self.ml_trainer = ModelTrainer(model_dir=ml_config.model_dir, db_path=db_path)
        self.ml_retrainer = AutoRetrainer(
            trainer=self.ml_trainer,
            predictor=self.ml_predictor,
            retrain_after_trades=ml_config.retrain_after_trades,
            min_samples_for_retrain=ml_config.min_samples_for_retrain,
            performance_threshold=ml_config.performance_threshold,
            check_interval_seconds=ml_config.check_interval_seconds
        ) if ml_config.enabled else None

        # Running flag
        self._running = False
        self._should_stop = False

        logger.info(
            "Trader initialized",
            paper_trading=self.settings.trading.paper_trading,
            pairs=self.settings.trading.pairs,
            strategy=self.strategy.name,
            notifications_enabled=notif_config.discord.enabled,
            ml_enabled=ml_config.enabled
        )

    def initialize(self) -> bool:
        """
        Initialize the trader and restore state.

        Returns:
            True if initialization successful.
        """
        try:
            # Try to restore previous state
            saved_state = self.db.load_state()
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

            # Load ML model if enabled
            if self.settings.ml.enabled:
                if self.ml_predictor.load_model():
                    logger.info("ML model loaded successfully")
                else:
                    logger.info("No ML model found - ML predictions disabled until training")

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

        # Start daily summary scheduler
        if self.summary_scheduler:
            self.summary_scheduler.start()

        # Start ML auto-retrainer
        if self.ml_retrainer:
            self.ml_retrainer.start()

        # Send startup notification
        self.notifier.notify_startup(
            pairs=self.settings.trading.pairs,
            paper_trading=self.settings.trading.paper_trading
        )

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

        # Update peak price if we have a position in this pair
        if current_position:
            current_price = market_data.ticker.last if market_data.ticker else 0
            self.state_machine.update_peak_price(pair, current_price)

        # Start analysis
        self.state_machine.start_analysis()

        # Get peak price and trailing stop status for strategy (pair-specific)
        peak_price = None
        trailing_stop_active = False
        pos = self.state_machine.get_position(pair)
        if pos:
            peak_price = pos.peak_price
            trailing_stop_active = pos.trailing_stop_active

        # Get signal from strategy (pass trailing stop info)
        signal = self.strategy.analyze(market_data, current_position)

        # Check if trailing stop should be activated (pair-specific)
        if signal.indicators.get("_should_activate_trailing", False):
            self.state_machine.activate_trailing_stop(pair)

        logger.debug(
            f"Signal for {pair}: {signal.signal_type.value}",
            strength=signal.strength,
            reason=signal.reason
        )

        # Check for Martingale add-on opportunity if we have a position in this pair
        if current_position:
            current_price = market_data.ticker.last if market_data.ticker else 0
            avg_entry = pos.entry_price

            should_add, add_reason = self.strategy.should_add_to_position(
                market_data=market_data,
                avg_entry_price=avg_entry,
                current_price=current_price
            )

            if should_add:
                self._add_to_position(pair, market_data, add_reason)
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
        pos = self.state_machine.get_position(pair)
        if not pos:
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

        # ML signal filtering
        if self.settings.ml.enabled and self.ml_predictor.is_ready:
            should_trade, ml_confidence, ml_reason = self.ml_predictor.should_take_signal(
                signal_type="buy",
                market_data=market_data
            )
            if not should_trade:
                logger.info(f"ML rejected signal for {pair}: {ml_reason}")
                self.state_machine.analysis_complete(has_signal=False)
                return
            logger.info(f"ML approved signal for {pair}: {ml_reason}")

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

            # Update total exposure across all positions
            total_exposure = self.state_machine.total_exposure_usd
            self.risk_manager.update_exposure(total_exposure)

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

            # Record trade entry for ML training
            if self.settings.ml.enabled:
                self.ml_predictor.record_trade_entry(
                    order_id=order.order_id,
                    pair=pair,
                    signal_type="buy",
                    market_data=market_data,
                    entry_price=order.price or price
                )

            # Send Discord notification
            self.notifier.notify_trade_entry(
                trade=TradeNotification(
                    pair=pair,
                    side="buy",
                    price=order.price or price,
                    size=position_size_units,
                    size_usd=position_size_usd,
                    reason=signal.reason
                ),
                paper_trading=self.settings.trading.paper_trading
            )

        except Exception as e:
            logger.error(f"Failed to place entry order: {e}")
            self.state_machine.set_error(str(e))
            self.notifier.notify_error("Entry Order Failed", str(e), pair)

    def _add_to_position(self, pair: str, market_data: MarketData, reason: str) -> None:
        """
        Add to existing position (Martingale).

        Args:
            pair: Trading pair.
            market_data: Current market data.
            reason: Reason for adding.
        """
        position = self.state_machine.get_position(pair)
        if not position:
            logger.warning(f"Cannot add to position: no position exists for {pair}")
            return

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
                pair=pair,
                entry_price=order.price or current_price,
                size=add_on_size_units,
                size_usd=add_on_size_usd,
                order_id=order.order_id
            )

            # Update total exposure across all positions
            total_exposure = self.state_machine.total_exposure_usd
            self.risk_manager.update_exposure(total_exposure)

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

            # Send Discord notification
            self.notifier.notify_martingale_add(
                trade=TradeNotification(
                    pair=pair,
                    side="buy",
                    price=order.price or current_price,
                    size=add_on_size_units,
                    size_usd=add_on_size_usd,
                    reason=reason,
                    entry_num=position.num_entries,
                    avg_entry=position.entry_price
                ),
                paper_trading=self.settings.trading.paper_trading
            )

        except Exception as e:
            logger.error(f"Failed to place Martingale add-on order: {e}")
            self.notifier.notify_error("Martingale Add-on Failed", str(e), pair)

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

        self.state_machine.start_exit(pair, signal.reason)

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
            self.state_machine.close_position(pair, exit_price, pnl)

            # Update total exposure across remaining positions
            total_exposure = self.state_machine.total_exposure_usd
            self.risk_manager.update_exposure(total_exposure)

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
            self.db.save_trade({
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

            # Record trade exit for ML training and notify retrainer
            if self.settings.ml.enabled:
                # Get entry order_id from position (first entry's order_id)
                pos_state = self.state_machine.get_position(pair)
                entry_order_id = pos_state.order_id if pos_state else order.order_id

                self.ml_predictor.record_trade_exit(
                    order_id=entry_order_id,
                    exit_price=exit_price,
                    pnl=pnl,
                    pnl_percent=pnl_percent
                )

                # Notify retrainer of completed trade
                if self.ml_retrainer:
                    self.ml_retrainer.record_trade_completed()

            # Send Discord notification
            self.notifier.notify_trade_exit(
                trade=TradeNotification(
                    pair=pair,
                    side="sell",
                    price=exit_price,
                    size=size,
                    size_usd=exit_price * size,
                    reason=signal.reason,
                    pnl=pnl,
                    pnl_percent=pnl_percent
                ),
                paper_trading=self.settings.trading.paper_trading
            )

        except Exception as e:
            logger.error(f"Failed to place exit order: {e}")
            self.state_machine.set_error(str(e))
            self.notifier.notify_error("Exit Order Failed", str(e), pair)

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

        total_pnl = 0.0
        positions_to_close = list(self.state_machine.positions.items())

        for pair, pos in positions_to_close:
            try:
                # Emergency market sell
                order = self.client.place_order(
                    pair=pair,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    volume=pos.size
                )

                exit_price = order.price or 0
                pnl = (exit_price - pos.entry_price) * pos.size
                total_pnl += pnl

                self.state_machine.close_position(pair, exit_price, pnl)

                logger.info(
                    f"Emergency exit {pair}: P&L ${pnl:.2f}",
                    reason=reason
                )

            except Exception as e:
                logger.error(f"Emergency exit failed for {pair}: {e}")

        self.risk_manager.update_exposure(0)
        logger.info(f"Emergency exit completed: Total P&L ${total_pnl:.2f}")

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
        self.db.save_state(state)

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

        # Stop daily summary scheduler
        if self.summary_scheduler:
            self.summary_scheduler.stop()

        # Stop ML auto-retrainer
        if self.ml_retrainer:
            self.ml_retrainer.stop()

        # Send shutdown notification
        self.notifier.notify_shutdown("Trader shutdown")

        # Save final state
        self._save_state()

        # Backup data
        self.db.backup_all()

        # Close client and notifier
        self.client.close()
        self.notifier.close()

        self._running = False
        logger.info("Trader shutdown complete")

    def _send_daily_summary(self) -> None:
        """Send daily summary notification."""
        try:
            metrics = self.metrics.get_metrics()
            today = datetime.now(timezone.utc)
            today_str = today.strftime("%Y-%m-%d")

            # Get today's metrics
            daily_pnl = self.metrics.get_daily_pnl(today)
            daily_trades = self.metrics.get_daily_trade_count(today)

            # Get today's wins/losses from daily_metrics
            daily = self.metrics.daily_metrics.get(today_str)
            wins = daily.wins if daily else 0
            losses = daily.losses if daily else 0

            # Get open positions info (show first position for backward compatibility)
            open_position = None
            if self.state_machine.has_position:
                # Summarize all positions
                positions = self.state_machine.positions
                if len(positions) == 1:
                    pos = next(iter(positions.values()))
                    open_position = {
                        "pair": pos.pair,
                        "size": pos.size,
                        "entry_price": pos.entry_price,
                        "num_entries": pos.num_entries
                    }
                else:
                    # Multiple positions - show summary
                    pairs = list(positions.keys())
                    total_exposure = self.state_machine.total_exposure_usd
                    open_position = {
                        "pair": f"{len(positions)} positions: {', '.join(pairs)}",
                        "size": 0,
                        "entry_price": total_exposure,
                        "num_entries": sum(p.num_entries for p in positions.values())
                    }

            summary = DailySummaryData(
                date=today_str,
                total_trades=daily_trades,
                wins=wins,
                losses=losses,
                daily_pnl=daily_pnl,
                total_pnl=metrics.total_pnl,
                win_rate=metrics.win_rate,
                current_capital=self.metrics.current_capital,
                open_position=open_position,
                sharpe_ratio=metrics.sharpe_ratio,
                max_drawdown_percent=metrics.max_drawdown_percent
            )

            self.notifier.notify_daily_summary(
                summary=summary,
                paper_trading=self.settings.trading.paper_trading
            )

            logger.info("Daily summary notification sent")

        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")

    def get_status(self) -> dict:
        """
        Get current trader status.

        Returns:
            Status dictionary.
        """
        state = self.state_machine.get_full_state()
        risk_limits = self.risk_manager.get_limits()
        metrics = self.metrics.get_metrics()

        # Build positions list
        positions_list = []
        for pair, pos in state.positions.items():
            positions_list.append({
                "pair": pos.pair,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "size": pos.size,
                "peak_price": pos.peak_price,
                "trailing_stop_active": pos.trailing_stop_active,
                "num_entries": pos.num_entries,
                "total_cost_usd": pos.total_cost_usd,
            })

        return {
            "running": self._running,
            "state": state.current_state.value,
            "has_position": len(state.positions) > 0,
            "position_count": len(state.positions),
            "positions": positions_list,
            "total_exposure_usd": self.state_machine.total_exposure_usd,
            # Keep 'position' for backward compatibility (first position)
            "position": positions_list[0] if positions_list else None,
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
            "ml": {
                "enabled": self.settings.ml.enabled,
                "model_loaded": self.ml_predictor._model_loaded if self.ml_predictor else False,
                "confidence_threshold": self.settings.ml.confidence_threshold,
                "retrainer_status": self.ml_retrainer.get_status() if self.ml_retrainer else None
            }
        }
