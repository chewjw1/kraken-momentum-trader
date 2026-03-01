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
from src.strategy.regime_detector import RegimeDetector, RegimeConfig, MarketRegime
from src.core.adaptive_pair_manager import AdaptivePairManager, AdaptiveConfig
from src.risk.circuit_breaker import CircuitBreaker
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

        # Fee rate depends on order type (maker = 0.16%, taker = 0.26%)
        use_maker = self.config.get('execution', {}).get('use_maker_orders', True)
        fee_rate = self.config.get('fees', {}).get('maker_percent', 0.16) if use_maker else self.config.get('fees', {}).get('taker_percent', 0.26)

        # Default strategy config (used if per-pair not specified)
        default_config = ScalpingConfig(
            take_profit_percent=self.config.get('strategy', {}).get('take_profit_percent', 5.0),
            stop_loss_percent=self.config.get('strategy', {}).get('stop_loss_percent', 2.5),
            min_confirmations=self.config.get('strategy', {}).get('min_confirmations', 3),
            fee_percent=fee_rate
        )
        self.strategy = ScalpingStrategy(default_config)

        # Per-pair strategies (if configured)
        self.pair_strategies: Dict[str, ScalpingStrategy] = {}
        pair_params = self.config.get('pair_parameters', {})
        for pair, params in pair_params.items():
            pair_config = ScalpingConfig(
                take_profit_percent=params.get('take_profit_percent', default_config.take_profit_percent),
                stop_loss_percent=params.get('stop_loss_percent', default_config.stop_loss_percent),
                rsi_period=params.get('rsi_period', default_config.rsi_period),
                rsi_oversold=params.get('rsi_oversold', default_config.rsi_oversold),
                rsi_overbought=params.get('rsi_overbought', default_config.rsi_overbought),
                bb_period=params.get('bb_period', default_config.bb_period),
                bb_std_dev=params.get('bb_std_dev', default_config.bb_std_dev),
                vwap_threshold_percent=params.get('vwap_threshold_percent', default_config.vwap_threshold_percent),
                volume_spike_threshold=params.get('volume_spike_threshold', default_config.volume_spike_threshold),
                min_confirmations=params.get('min_confirmations', default_config.min_confirmations),
                fee_percent=fee_rate
            )
            self.pair_strategies[pair] = ScalpingStrategy(pair_config)

        adaptive_config = AdaptiveConfig(
            min_win_rate=self.config.get('adaptive', {}).get('min_win_rate', 0.35),
            min_profit_factor=self.config.get('adaptive', {}).get('min_profit_factor', 0.7),
            max_consecutive_losses=self.config.get('adaptive', {}).get('max_consecutive_losses', 5),
            cooldown_hours=self.config.get('adaptive', {}).get('cooldown_hours', 2.0),
            reenable_win_rate=self.config.get('adaptive', {}).get('reenable_win_rate', 0.45)
        )
        self.pair_manager = AdaptivePairManager(adaptive_config)

        # Trading state (set early so circuit breaker can reference initial_capital)
        self.pairs = self.config.get('pairs', ['SOL/USD'])
        self.positions: Dict[str, dict] = {}
        self.capital = self._fetch_initial_capital()
        self.initial_capital = self.capital
        self.position_size_pct = self.config.get('position', {}).get('size_percent', 20.0)

        # Circuit breaker - drawdown-based (per-pair + global)
        cb_config = self.config.get('circuit_breaker', {})
        self.circuit_breaker = CircuitBreaker(
            global_max_drawdown_pct=cb_config.get('global_max_drawdown_pct', 5.0),
            cooldown_hours=cb_config.get('cooldown_hours', 4),
            pair_max_drawdown_pct=cb_config.get('pair_max_drawdown_pct', 3.0),
            pair_cooldown_hours=cb_config.get('pair_cooldown_hours', 2.0),
            consecutive_loss_limit=cb_config.get('consecutive_loss_limit', 5),
            initial_capital=self.initial_capital,
        )

        # Regime detector - classifies bull/bear/sideways
        regime_cfg = self.config.get('regime_detector', {})
        self.regime_detector = RegimeDetector(RegimeConfig(
            fast_sma_period=regime_cfg.get('fast_sma_period', 50),
            slow_sma_period=regime_cfg.get('slow_sma_period', 200),
            bull_slope_threshold=regime_cfg.get('bull_slope_threshold', 0.05),
            bear_slope_threshold=regime_cfg.get('bear_slope_threshold', -0.05),
        ))
        self._current_regime = MarketRegime.UNKNOWN
        self._regime_adjustments: Dict[str, float] = {}

        # Store base configs for regime adjustment
        self._base_default_config = default_config
        self._base_pair_params = dict(pair_params)

        # Order execution settings
        self.use_maker_orders = self.config.get('execution', {}).get('use_maker_orders', True)
        self.maker_price_offset = self.config.get('execution', {}).get('maker_price_offset', 0.0)
        # Fee rate depends on order type
        self.fee_rate = self.config.get('fees', {}).get('maker_percent', 0.16) if self.use_maker_orders else self.config.get('fees', {}).get('taker_percent', 0.26)

        # Candle interval from config (default 60 for backward compatibility)
        self.candle_interval = self.config.get('strategy', {}).get('candle_interval', 60)

        # Metrics
        self.metrics = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'start_time': datetime.now(timezone.utc).isoformat()
        }

        # Initialize logger BEFORE loading state
        self.logger = get_logger(__name__)

        # Register pairs
        for pair in self.pairs:
            self.pair_manager.register_pair(pair)

        # Load saved state
        self._load_state()

        self.logger.info(
            "Scalping trader initialized",
            pairs=self.pairs,
            paper_trading=paper_trading,
            candle_interval=f"{self.candle_interval}m",
            take_profit=default_config.take_profit_percent,
            stop_loss=default_config.stop_loss_percent,
            use_maker_orders=self.use_maker_orders,
            fee_rate=f"{self.fee_rate}%",
            round_trip_fee=f"{self.fee_rate * 2}%",
            per_pair_configs=len(self.pair_strategies)
        )

    def _load_config(self) -> dict:
        """Load YAML configuration."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    def _fetch_initial_capital(self) -> float:
        """
        Fetch starting capital from real Kraken account balance.

        Falls back to config 'initial_capital' or $10,000 default if the
        API call fails (e.g. no credentials, network error).
        """
        fallback = self.config.get('position', {}).get('initial_capital', 10000.0)
        try:
            # Temporarily bypass paper mode to hit real API
            orig_paper = self.client.paper_trading
            self.client.paper_trading = False
            balances = self.client.get_balances()
            self.client.paper_trading = orig_paper

            usd_balance = balances.get("USD")
            if usd_balance and usd_balance.total > 0:
                print(f"  Kraken account balance: ${usd_balance.total:.2f} USD")
                return usd_balance.total

            # Sum stablecoin equivalents if no pure USD
            usdt = balances.get("USDT")
            if usdt and usdt.total > 0:
                print(f"  Kraken account balance: ${usdt.total:.2f} USDT")
                return usdt.total

            print(f"  No USD balance found on Kraken, using config fallback: ${fallback:.2f}")
            return fallback

        except Exception as e:
            print(f"  Could not fetch Kraken balance ({e}), using fallback: ${fallback:.2f}")
            return fallback

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
                if 'circuit_breaker' in state:
                    self.circuit_breaker.from_dict(state['circuit_breaker'])
                if 'regime_detector' in state:
                    self.regime_detector.from_dict(state['regime_detector'])
                if 'current_regime' in state:
                    try:
                        self._current_regime = MarketRegime(state['current_regime'])
                    except ValueError:
                        pass
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
            'circuit_breaker': self.circuit_breaker.to_dict(),
            'regime_detector': self.regime_detector.to_dict(),
            'current_regime': self._current_regime.value,
            'last_update': datetime.now(timezone.utc).isoformat()
        }
        state_file = self.data_dir / "state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def _get_market_data(self, pair: str) -> Optional[MarketData]:
        """Fetch market data for a pair."""
        try:
            ohlc = self.client.get_ohlc(pair, interval=self.candle_interval)
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

    def _update_regime(self) -> None:
        """
        Detect market regime using BTC as the reference market.

        Adjusts strategy parameters (TP, SL, position size, confirmations)
        based on the detected regime. Rebuilds strategy instances when the
        regime changes.
        """
        # Use BTC as regime reference (largest, most liquid)
        reference_pair = "BTC/USD"
        try:
            ohlc = self.client.get_ohlc(reference_pair, interval=self.candle_interval)
            if not ohlc or len(ohlc) < 200:
                return  # Not enough data yet
        except Exception as e:
            self.logger.error(f"Error fetching regime data: {e}")
            return

        closes = [c.close for c in ohlc]
        highs = [c.high for c in ohlc]
        lows = [c.low for c in ohlc]

        result = self.regime_detector.detect(closes, highs, lows)
        adjustments = self.regime_detector.get_adjustments(result.regime)

        if result.regime != self._current_regime:
            self.logger.info(
                f"REGIME CHANGE: {self._current_regime.value} -> {result.regime.value}",
                confidence=f"{result.confidence:.2f}",
                sma_slope=f"{result.sma_slope_pct:.3f}%/period",
                price_vs_200sma=f"{result.price_vs_slow_sma_pct:.1f}%",
                atr=f"{result.atr_pct:.2f}%",
                adjustments=adjustments.get('description', ''),
            )

            self._current_regime = result.regime
            self._regime_adjustments = adjustments

            # Rebuild strategies with regime-adjusted parameters
            self._rebuild_strategies_for_regime(adjustments)

    def _rebuild_strategies_for_regime(self, adjustments: Dict) -> None:
        """Rebuild strategy instances with regime-adjusted parameters."""
        fee_rate = self.fee_rate
        tp_mult = adjustments.get('take_profit_multiplier', 1.0)
        sl_mult = adjustments.get('stop_loss_multiplier', 1.0)
        conf_offset = adjustments.get('min_confirmations_offset', 0)
        ema_enabled = adjustments.get('ema_filter_enabled', True)

        # Rebuild default strategy with all indicator params preserved
        base = self._base_default_config
        adj_config = ScalpingConfig(
            take_profit_percent=base.take_profit_percent * tp_mult,
            stop_loss_percent=base.stop_loss_percent * sl_mult,
            min_confirmations=max(1, base.min_confirmations + conf_offset),
            fee_percent=fee_rate,
            ema_filter_enabled=ema_enabled,
            # Preserve all indicator params
            rsi_period=base.rsi_period,
            rsi_oversold=base.rsi_oversold,
            rsi_overbought=base.rsi_overbought,
            bb_period=base.bb_period,
            bb_std_dev=base.bb_std_dev,
            bb_squeeze_threshold=base.bb_squeeze_threshold,
            vwap_threshold_percent=base.vwap_threshold_percent,
            volume_spike_threshold=base.volume_spike_threshold,
            stoch_k_period=base.stoch_k_period,
            stoch_d_period=base.stoch_d_period,
            stoch_oversold=base.stoch_oversold,
            stoch_overbought=base.stoch_overbought,
            macd_fast=base.macd_fast,
            macd_slow=base.macd_slow,
            macd_signal=base.macd_signal,
            obv_sma_period=base.obv_sma_period,
            atr_period=base.atr_period,
            atr_stop_multiplier=base.atr_stop_multiplier,
            atr_tp_multiplier=base.atr_tp_multiplier,
            use_atr_stops=base.use_atr_stops,
            shorting_enabled=base.shorting_enabled,
            short_min_confirmations=base.short_min_confirmations,
        )
        self.strategy = ScalpingStrategy(adj_config)

        # Rebuild per-pair strategies
        self.pair_strategies = {}
        for pair, params in self._base_pair_params.items():
            base_tp = params.get('take_profit_percent', base.take_profit_percent)
            base_sl = params.get('stop_loss_percent', base.stop_loss_percent)
            base_conf = params.get('min_confirmations', base.min_confirmations)

            pair_config = ScalpingConfig(
                take_profit_percent=base_tp * tp_mult,
                stop_loss_percent=base_sl * sl_mult,
                rsi_period=params.get('rsi_period', base.rsi_period),
                rsi_oversold=params.get('rsi_oversold', base.rsi_oversold),
                rsi_overbought=params.get('rsi_overbought', base.rsi_overbought),
                bb_period=params.get('bb_period', base.bb_period),
                bb_std_dev=params.get('bb_std_dev', base.bb_std_dev),
                vwap_threshold_percent=params.get('vwap_threshold_percent', base.vwap_threshold_percent),
                volume_spike_threshold=params.get('volume_spike_threshold', base.volume_spike_threshold),
                min_confirmations=max(1, base_conf + conf_offset),
                fee_percent=fee_rate,
                ema_filter_enabled=ema_enabled,
                # New indicator params from per-pair config or base
                stoch_k_period=params.get('stoch_k_period', base.stoch_k_period),
                stoch_oversold=params.get('stoch_oversold', base.stoch_oversold),
                stoch_overbought=params.get('stoch_overbought', base.stoch_overbought),
                atr_period=params.get('atr_period', base.atr_period),
                atr_stop_multiplier=params.get('atr_stop_multiplier', base.atr_stop_multiplier),
                atr_tp_multiplier=params.get('atr_tp_multiplier', base.atr_tp_multiplier),
                use_atr_stops=params.get('use_atr_stops', base.use_atr_stops),
                shorting_enabled=base.shorting_enabled,
                short_min_confirmations=params.get('short_min_confirmations', base.short_min_confirmations),
            )
            self.pair_strategies[pair] = ScalpingStrategy(pair_config)

        self.logger.info(
            f"Strategies rebuilt for {self._current_regime.value} regime",
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            conf_offset=conf_offset,
            ema_filter=ema_enabled,
        )

    def _get_strategy_for_pair(self, pair: str) -> ScalpingStrategy:
        """Get the strategy for a specific pair (per-pair or default)."""
        return self.pair_strategies.get(pair, self.strategy)

    def _process_pair(self, pair: str) -> None:
        """Process a single trading pair."""
        # Check if pair is enabled by adaptive manager
        if not self.pair_manager.is_pair_enabled(pair):
            return

        market_data = self._get_market_data(pair)
        if not market_data:
            return

        # Use real-time ticker price, not stale candle close!
        if market_data.ticker:
            current_price = market_data.ticker.last
        else:
            current_price = market_data.prices[-1]  # Fallback to candle close

        # Get strategy for this pair (may have per-pair optimized params)
        strategy = self._get_strategy_for_pair(pair)

        # Check if we have a position
        if pair in self.positions:
            position_data = self.positions[pair]
            entry_price = position_data['entry_price']
            pnl_pct = ((current_price - entry_price) / entry_price) * 100

            self.logger.info(
                f"Checking {pair}",
                entry_price=f"${entry_price:.2f}",
                current_price=f"${current_price:.2f}",
                pnl_pct=f"{pnl_pct:.2f}%",
                tp_target=f"{strategy.config.take_profit_percent}%",
                sl_target=f"-{strategy.config.stop_loss_percent}%"
            )

            position = Position(
                pair=pair,
                side="long",
                entry_price=entry_price,
                current_price=current_price,
                size=position_data['size'],
                entry_time=datetime.fromisoformat(position_data['entry_time'])
            )

            signal = strategy.analyze(market_data, position)
            self.logger.debug(f"{pair} signal: {signal.signal_type.value} - {signal.reason}")

            if signal.signal_type.value in ("sell", "close_long"):
                # Exit position
                pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                fee_pct = strategy.config.fee_percent * 2
                net_pnl_pct = pnl_pct - fee_pct
                pnl_usd = position_data['size_usd'] * (net_pnl_pct / 100)

                self.capital += pnl_usd

                # Record trade in adaptive manager
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

                # Record trade in drawdown-based circuit breaker
                self.circuit_breaker.record_trade(pair, pnl_usd)

                self.logger.info(
                    f"CLOSED {pair}",
                    reason=signal.reason,
                    pnl_pct=f"{net_pnl_pct:.2f}%",
                    pnl_usd=f"${pnl_usd:.2f}",
                    capital=f"${self.capital:.2f}",
                    regime=self._current_regime.value,
                    circuit_breaker=self.circuit_breaker.get_state().state.value
                )

                del self.positions[pair]
                self._save_state()

        else:
            # Check global circuit breaker
            if not self.circuit_breaker.is_trading_allowed():
                cb_state = self.circuit_breaker.get_state()
                remaining = self.circuit_breaker.time_until_ready()
                self.logger.warning(
                    f"Circuit breaker OPEN - skipping {pair} entry",
                    state=cb_state.state.value,
                    reason=cb_state.trigger_reason,
                    time_remaining=str(remaining) if remaining else "manual reset needed"
                )
                return

            # Check per-pair circuit breaker
            if not self.circuit_breaker.is_pair_allowed(pair):
                ps = self.circuit_breaker.get_pair_state(pair)
                self.logger.warning(
                    f"Pair {pair} paused by circuit breaker",
                    reason=ps.pause_reason if ps else "unknown"
                )
                return

            # Look for entry
            signal = strategy.analyze(market_data, None)

            if signal.signal_type.value == "buy":
                deployed_capital = sum(p['size_usd'] for p in self.positions.values())
                available_capital = self.capital - deployed_capital

                # Scale position: adaptive manager * regime adjustment
                scale = self.pair_manager.get_position_scale(pair)
                regime_scale = self._regime_adjustments.get(
                    'position_scale_multiplier', 1.0
                )
                total_scale = scale * regime_scale

                size_usd = self.initial_capital * (self.position_size_pct / 100) * total_scale

                if size_usd > available_capital:
                    self.logger.warning(
                        f"Skipping {pair} entry - insufficient capital",
                        required=f"${size_usd:.2f}",
                        available=f"${available_capital:.2f}"
                    )
                    return

                size = size_usd / current_price

                self.positions[pair] = {
                    'entry_price': current_price,
                    'size': size,
                    'size_usd': size_usd,
                    'entry_time': datetime.now(timezone.utc).isoformat(),
                    'reason': signal.reason,
                    'regime': self._current_regime.value,
                }

                self.logger.info(
                    f"OPENED {pair}",
                    reason=signal.reason,
                    price=f"${current_price:.2f}",
                    size_usd=f"${size_usd:.2f}",
                    scale=f"{total_scale:.2f}x (adaptive={scale:.2f} regime={regime_scale:.2f})",
                    regime=self._current_regime.value,
                    available_after=f"${available_capital - size_usd:.2f}"
                )

                self._save_state()

    def run(self) -> None:
        """Run the trading loop."""
        self.running = True
        check_interval = self.config.get('trading', {}).get('check_interval_seconds', 60)

        self.logger.info("Starting scalping trader loop")

        while self.running:
            try:
                # Detect market regime and adjust strategy parameters
                self._update_regime()

                # Re-enable pairs whose cooldown has expired
                reenabled = self.pair_manager.check_reenable_pairs()
                if reenabled:
                    self.logger.info(
                        f"Re-enabled {len(reenabled)} pairs after cooldown",
                        pairs=reenabled
                    )

                # Update circuit breaker equity tracking
                self.circuit_breaker.update_equity(self.capital)

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

        regime_info = self.regime_detector.to_dict()
        regime_info['current'] = self._current_regime.value
        regime_info['adjustments'] = self._regime_adjustments

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
            'disabled_pairs': [p for p in self.pairs if not self.pair_manager.is_pair_enabled(p)],
            'circuit_breaker': self.circuit_breaker.to_dict(),
            'regime': regime_info,
        }


def main():
    parser = argparse.ArgumentParser(description="Scalping Strategy Live Trader")
    parser.add_argument("--config", default="config/scalping.yaml", help="Config file path")
    parser.add_argument("--data-dir", default="data/scalping", help="Data directory")
    parser.add_argument("--live", action="store_true", help="Enable live trading (default: paper)")
    parser.add_argument("--dashboard-port", type=int, default=44485, help="Dashboard port")

    args = parser.parse_args()

    configure_logging(level="INFO", format_type="json")

    print(f"""
================================================================
           SCALPING STRATEGY TRADER
================================================================
  Mean-reversion scalping with adaptive pair management
  Regime detection: auto-adjusts for bull/bear/sideways
  Drawdown-based circuit breaker (per-pair + global)
  Pairs auto-disable/re-enable after cooldown
  Using MAKER orders for lower fees (0.16% vs 0.26%)
  Dashboard: http://jfk21.phoebe.usbx.me:{args.dashboard_port}
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
