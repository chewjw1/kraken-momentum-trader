"""
Unit tests for trading state machine.
"""

import pytest
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.state_machine import TradingStateMachine, TradingState


class TestTradingStateMachine:
    """Tests for trading state machine."""

    def test_initial_state(self):
        """State machine should start in initializing state."""
        sm = TradingStateMachine()
        assert sm.state == TradingState.INITIALIZING
        assert not sm.has_position

    def test_initialization_complete(self):
        """Should transition from initializing to idle."""
        sm = TradingStateMachine()
        assert sm.initialize_complete()
        assert sm.state == TradingState.IDLE

    def test_analysis_cycle(self):
        """Should handle analysis cycle correctly."""
        sm = TradingStateMachine()
        sm.initialize_complete()

        # Start analysis
        assert sm.start_analysis()
        assert sm.state == TradingState.ANALYZING

        # Complete without signal
        assert sm.analysis_complete(has_signal=False)
        assert sm.state == TradingState.IDLE

    def test_entry_flow(self):
        """Should handle entry flow correctly."""
        sm = TradingStateMachine()
        sm.initialize_complete()
        sm.start_analysis()

        # Signal detected
        assert sm.analysis_complete(has_signal=True)
        assert sm.state == TradingState.ENTERING

        # Position opened
        assert sm.open_position(
            pair="BTC/USD",
            side="long",
            entry_price=50000.0,
            size=0.01,
            order_id="ORDER123",
            stop_loss=47500.0,
            take_profit=55000.0
        )
        assert sm.state == TradingState.IN_POSITION
        assert sm.has_position

    def test_exit_flow(self):
        """Should handle exit flow correctly."""
        sm = TradingStateMachine()
        sm.initialize_complete()
        sm.start_analysis()
        sm.analysis_complete(has_signal=True)
        sm.open_position(
            pair="BTC/USD",
            side="long",
            entry_price=50000.0,
            size=0.01,
            order_id="ORDER123",
            stop_loss=47500.0,
            take_profit=55000.0
        )

        # Start exit
        assert sm.start_exit("Take profit hit")
        assert sm.state == TradingState.EXITING

        # Close position
        closed = sm.close_position(exit_price=55000.0, pnl=50.0)
        assert sm.state == TradingState.IDLE
        assert not sm.has_position
        assert closed.pair == "BTC/USD"

    def test_invalid_transition(self):
        """Should reject invalid state transitions."""
        sm = TradingStateMachine()
        sm.initialize_complete()

        # Can't go directly from IDLE to IN_POSITION
        assert not sm.transition_to(TradingState.IN_POSITION)
        assert sm.state == TradingState.IDLE

    def test_error_handling(self):
        """Should handle error state correctly."""
        sm = TradingStateMachine()
        sm.initialize_complete()

        assert sm.set_error("Test error")
        assert sm.state == TradingState.ERROR

        state = sm.get_full_state()
        assert state.error_message == "Test error"

    def test_error_recovery(self):
        """Should recover from error state."""
        sm = TradingStateMachine()
        sm.initialize_complete()
        sm.set_error("Test error")

        assert sm.recover_from_error()
        assert sm.state == TradingState.IDLE

    def test_stop_and_resume(self):
        """Should handle stop and resume correctly."""
        sm = TradingStateMachine()
        sm.initialize_complete()

        assert sm.stop("Manual stop")
        assert sm.state == TradingState.STOPPED

        assert sm.resume()
        assert sm.state == TradingState.IDLE

    def test_position_data(self):
        """Should store position data correctly."""
        sm = TradingStateMachine()
        sm.initialize_complete()
        sm.start_analysis()
        sm.analysis_complete(has_signal=True)

        sm.open_position(
            pair="ETH/USD",
            side="long",
            entry_price=3000.0,
            size=0.5,
            order_id="ORDER456",
            stop_loss=2850.0,
            take_profit=3300.0
        )

        pos = sm.position
        assert pos.pair == "ETH/USD"
        assert pos.side == "long"
        assert pos.entry_price == 3000.0
        assert pos.size == 0.5
        assert pos.stop_loss == 2850.0
        assert pos.take_profit == 3300.0

    def test_pending_order(self):
        """Should track pending orders."""
        sm = TradingStateMachine()
        sm.set_pending_order("ORDER789")

        state = sm.get_full_state()
        assert state.pending_order_id == "ORDER789"

        sm.clear_pending_order()
        state = sm.get_full_state()
        assert state.pending_order_id is None

    def test_state_listener(self):
        """Should notify listeners on state changes."""
        sm = TradingStateMachine()

        transitions = []

        def listener(prev, new, reason):
            transitions.append((prev.value, new.value, reason))

        sm.add_state_listener(listener)
        sm.initialize_complete()
        sm.start_analysis()

        assert len(transitions) == 2
        assert transitions[0] == ("initializing", "idle", "Initialization complete")
        assert transitions[1] == ("idle", "analyzing", "Starting market analysis")

    def test_serialization(self):
        """State machine should serialize/deserialize correctly."""
        sm = TradingStateMachine()
        sm.initialize_complete()
        sm.start_analysis()
        sm.analysis_complete(has_signal=True)
        sm.open_position(
            pair="BTC/USD",
            side="long",
            entry_price=50000.0,
            size=0.01,
            order_id="ORDER123",
            stop_loss=47500.0,
            take_profit=55000.0
        )

        data = sm.to_dict()

        sm2 = TradingStateMachine()
        sm2.from_dict(data)

        assert sm2.state == TradingState.IN_POSITION
        assert sm2.has_position
        assert sm2.position.pair == "BTC/USD"
        assert sm2.position.entry_price == 50000.0

    def test_cannot_exit_without_position(self):
        """Should not allow exit when no position exists."""
        sm = TradingStateMachine()
        sm.initialize_complete()

        assert not sm.start_exit("No position")
        assert sm.state == TradingState.IDLE


class TestStateMachineEdgeCases:
    """Edge case tests for state machine."""

    def test_multiple_analysis_cycles(self):
        """Should handle multiple analysis cycles."""
        sm = TradingStateMachine()
        sm.initialize_complete()

        for _ in range(5):
            sm.start_analysis()
            sm.analysis_complete(has_signal=False)

        assert sm.state == TradingState.IDLE

    def test_stop_from_any_state(self):
        """Should be able to stop from any state."""
        # From IDLE
        sm = TradingStateMachine()
        sm.initialize_complete()
        assert sm.stop()

        # From ANALYZING
        sm = TradingStateMachine()
        sm.initialize_complete()
        sm.start_analysis()
        assert sm.stop()

        # From IN_POSITION
        sm = TradingStateMachine()
        sm.initialize_complete()
        sm.start_analysis()
        sm.analysis_complete(has_signal=True)
        sm.open_position(
            pair="BTC/USD",
            side="long",
            entry_price=50000.0,
            size=0.01,
            order_id="ORDER123",
            stop_loss=47500.0,
            take_profit=55000.0
        )
        assert sm.stop()

    def test_error_from_any_state(self):
        """Should be able to enter error from most states."""
        states_to_test = [
            TradingState.IDLE,
            TradingState.ANALYZING,
            TradingState.ENTERING,
            TradingState.IN_POSITION,
            TradingState.EXITING,
        ]

        for target_state in states_to_test:
            sm = TradingStateMachine()
            sm.initialize_complete()

            # Get to target state
            if target_state == TradingState.ANALYZING:
                sm.start_analysis()
            elif target_state == TradingState.ENTERING:
                sm.start_analysis()
                sm.analysis_complete(has_signal=True)
            elif target_state == TradingState.IN_POSITION:
                sm.start_analysis()
                sm.analysis_complete(has_signal=True)
                sm.open_position(
                    pair="BTC/USD",
                    side="long",
                    entry_price=50000.0,
                    size=0.01,
                    order_id="ORDER123",
                    stop_loss=47500.0,
                    take_profit=55000.0
                )
            elif target_state == TradingState.EXITING:
                sm.start_analysis()
                sm.analysis_complete(has_signal=True)
                sm.open_position(
                    pair="BTC/USD",
                    side="long",
                    entry_price=50000.0,
                    size=0.01,
                    order_id="ORDER123",
                    stop_loss=47500.0,
                    take_profit=55000.0
                )
                sm.start_exit()

            assert sm.set_error("Test error"), f"Failed from {target_state}"
