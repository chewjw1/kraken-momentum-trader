#!/usr/bin/env python3
"""
Scalping Strategy Live Trader

Runs the scalping strategy with adaptive pair management.
Separate from momentum trader - uses its own data directory.

Usage:
    python run_scalping_live.py
    python run_scalping_live.py --config config/scalping.yaml
    python run_scalping_live.py --dashboard-port 5001
"""

import argparse
import signal
import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

from dotenv import load_dotenv
load_dotenv()

import yaml

from src.exchange.kraken_client import KrakenClient
from src.strategy.scalping_strategy import ScalpingStrategy, ScalpingConfig
from src.strategy.base_strategy import MarketData, Position
from src.core.adaptive_pair_manager import AdaptivePairManager, AdaptiveConfig
from src.observability.logger import configure_logging, get_logger


class ScalpingTrader:
    """
    Live trader for scalping strategy.

    Features:
    - Multi-pair trading with adaptive management
    - Auto-disables underperforming pairs
    - Saves state for dashboard monitoring
    """

    def __init__(
        self,
        config_path: str = "config/scalping.yaml",
        data_dir: str = "data/scalping",
        paper_trading: bool = True
    ):
        self.config_path = config_path
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.paper_trading = paper_trading
        self.running = False

        # Load config
        self.config = self._load_config()

        # Initialize components
        self.client = KrakenClient(paper_trading=paper_trading)

        scalping_config = ScalpingConfig(
            take_profit_percent=self.config.get('strategy', {}).get('take_profit_percent', 4.5),
            stop_loss_percent=self.config.get('strategy', {}).get('stop_loss_percent', 2.0),
            min_confirmations=self.config.get('strategy', {}).get('min_confirmations', 2),
            fee_percent=self.config.get('fees', {}).get('taker_percent', 0.26)
        )
        self.strategy = ScalpingStrategy(scalping_config)

        adaptive_config = AdaptiveConfig(
            min_win_rate=self.config.get('adaptive', {}).get('min_win_rate', 0.35),
            min_profit_factor=self.config.get('adaptive', {}).get('min_profit_factor', 0.7),
            max_consecutive_losses=self.config.get('adaptive', {}).get('max_consecutive_losses', 5),
            cooldown_hours=self.config.get('adaptive', {}).get('cooldown_hours', 2.0),
            reenable_win_rate=self.config.get('adaptive', {}).get('reenable_win_rate', 0.45)
        )
        self.pair_manager = AdaptivePairManager(adaptive_config)

        # Trading state
        self.pairs = self.config.get('pairs', ['SOL/USD'])
        self.positions: Dict[str, dict] = {}
        self.capital = self.config.get('position', {}).get('initial_capital', 10000.0)
        self.position_size_pct = self.config.get('position', {}).get('size_percent', 20.0)

        # Metrics
        self.metrics = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'start_time': datetime.now(timezone.utc).isoformat()
        }

        # Register pairs
        for pair in self.pairs:
            self.pair_manager.register_pair(pair)

        # Load saved state
        self._load_state()

        self.logger = get_logger(__name__)
        self.logger.info(
            "Scalping trader initialized",
            pairs=self.pairs,
            paper_trading=paper_trading,
            take_profit=scalping_config.take_profit_percent,
            stop_loss=scalping_config.stop_loss_percent
        )

    def _load_config(self) -> dict:
        """Load YAML configuration."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    def _load_state(self) -> None:
        """Load saved state from disk."""
        state_file = self.data_dir / "state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                self.positions = state.get('positions', {})
                self.metrics = state.get('metrics', self.metrics)
                self.capital = state.get('capital', self.capital)
                self.pair_manager.from_dict(state.get('pair_manager', {}))
                self.logger.info("Loaded saved state")
            except Exception as e:
                self.logger.error(f"Error loading state: {e}")

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            'positions': self.positions,
            'metrics': self.metrics,
            'capital': self.capital,
            'pair_manager': self.pair_manager.to_dict(),
            'last_update': datetime.now(timezone.utc).isoformat()
        }
        state_file = self.data_dir / "state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def _get_market_data(self, pair: str) -> Optional[MarketData]:
        """Fetch market data for a pair."""
        try:
            ohlc = self.client.get_ohlc(pair, interval=60)  # 60-min candles
            if not ohlc or len(ohlc) < 25:
                return None

            return MarketData(
                pair=pair,
                ohlc=ohlc,
                prices=[c.close for c in ohlc],
                volumes=[c.volume for c in ohlc],
                ticker=self.client.get_ticker(pair)
            )
        except Exception as e:
            self.logger.error(f"Error fetching data for {pair}: {e}")
            return None

    def _process_pair(self, pair: str) -> None:
        """Process a single trading pair."""
        # Check if pair is enabled
        if not self.pair_manager.is_pair_enabled(pair):
            return

        market_data = self._get_market_data(pair)
        if not market_data:
            return

        current_price = market_data.prices[-1]

        # Check if we have a position
        if pair in self.positions:
            position_data = self.positions[pair]
            position = Position(
                pair=pair,
                side="long",
                entry_price=position_data['entry_price'],
                current_price=current_price,
                size=position_data['size'],
                entry_time=datetime.fromisoformat(position_data['entry_time'])
            )

            signal = self.strategy.analyze(market_data, position)

            if signal.signal_type.value in ("sell", "close_long"):
                # Exit position
                pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                fee_pct = self.strategy.config.fee_percent * 2
                net_pnl_pct = pnl_pct - fee_pct
                pnl_usd = position_data['size_usd'] * (net_pnl_pct / 100)

                self.capital += pnl_usd

                # Record trade
                self.pair_manager.record_trade(
                    pair=pair,
                    entry_time=position.entry_time,
                    exit_time=datetime.now(timezone.utc),
                    pnl=pnl_usd,
                    pnl_percent=net_pnl_pct
                )

                # Update metrics
                self.metrics['total_trades'] += 1
                self.metrics['total_pnl'] += pnl_usd
                if pnl_usd > 0:
                    self.metrics['wins'] += 1
                else:
                    self.metrics['losses'] += 1

                self.logger.info(
                    f"CLOSED {pair}",
                    reason=signal.reason,
                    pnl_pct=f"{net_pnl_pct:.2f}%",
                    pnl_usd=f"${pnl_usd:.2f}",
                    capital=f"${self.capital:.2f}"
                )

                del self.positions[pair]
                self._save_state()

        else:
            # Look for entry
            signal = self.strategy.analyze(market_data, None)

            if signal.signal_type.value == "buy":
                # Get position scale from adaptive manager
                scale = self.pair_manager.get_position_scale(pair)
                size_usd = self.capital * (self.position_size_pct / 100) * scale
                size = size_usd / current_price

                self.positions[pair] = {
                    'entry_price': current_price,
                    'size': size,
                    'size_usd': size_usd,
                    'entry_time': datetime.now(timezone.utc).isoformat(),
                    'reason': signal.reason
                }

                self.logger.info(
                    f"OPENED {pair}",
                    reason=signal.reason,
                    price=f"${current_price:.2f}",
                    size_usd=f"${size_usd:.2f}",
                    scale=f"{scale:.2f}x"
                )

                self._save_state()

    def run(self) -> None:
        """Run the trading loop."""
        self.running = True
        check_interval = self.config.get('trading', {}).get('check_interval_seconds', 60)

        self.logger.info("Starting scalping trader loop")

        while self.running:
            try:
                for pair in self.pairs:
                    self._process_pair(pair)
                    time.sleep(1)  # Rate limit between pairs

                self._save_state()

                # Log status
                enabled = self.pair_manager.get_enabled_pairs(self.pairs)
                self.logger.info(
                    "Cycle complete",
                    positions=len(self.positions),
                    enabled_pairs=len(enabled),
                    capital=f"${self.capital:.2f}",
                    total_pnl=f"${self.metrics['total_pnl']:.2f}"
                )

                time.sleep(check_interval)

            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(30)

    def stop(self) -> None:
        """Stop the trader."""
        self.running = False
        self._save_state()
        self.client.close()
        self.logger.info("Scalping trader stopped")

    def get_status(self) -> dict:
        """Get current status for dashboard."""
        win_rate = (self.metrics['wins'] / self.metrics['total_trades'] * 100) if self.metrics['total_trades'] > 0 else 0

        return {
            'strategy': 'scalping',
            'paper_trading': self.paper_trading,
            'capital': self.capital,
            'positions': self.positions,
            'metrics': {
                **self.metrics,
                'win_rate': win_rate
            },
            'pairs': {
                pair: self.pair_manager.get_pair_status(pair)
                for pair in self.pairs
            },
            'enabled_pairs': self.pair_manager.get_enabled_pairs(self.pairs),
            'disabled_pairs': [p for p in self.pairs if not self.pair_manager.is_pair_enabled(p)]
        }


def main():
    parser = argparse.ArgumentParser(description="Scalping Strategy Live Trader")
    parser.add_argument("--config", default="config/scalping.yaml", help="Config file path")
    parser.add_argument("--data-dir", default="data/scalping", help="Data directory")
    parser.add_argument("--live", action="store_true", help="Enable live trading (default: paper)")
    parser.add_argument("--dashboard-port", type=int, default=5001, help="Dashboard port")

    args = parser.parse_args()

    configure_logging(level="INFO", format_type="json")

    print("""
================================================================
           SCALPING STRATEGY TRADER
================================================================
  Mean-reversion scalping with adaptive pair management
  Pairs auto-disable after poor performance
================================================================
    """)

    trader = ScalpingTrader(
        config_path=args.config,
        data_dir=args.data_dir,
        paper_trading=not args.live
    )

    # Setup signal handlers
    def handle_signal(signum, frame):
        print("\nShutting down...")
        trader.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Print initial status
    status = trader.get_status()
    print(f"Paper Trading: {status['paper_trading']}")
    print(f"Capital: ${status['capital']:.2f}")
    print(f"Pairs: {', '.join(status['enabled_pairs'])}")
    print(f"Disabled: {', '.join(status['disabled_pairs']) or 'None'}")
    print()

    try:
        trader.run()
    except KeyboardInterrupt:
        trader.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
