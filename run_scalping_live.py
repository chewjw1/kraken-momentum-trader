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
import math
import os
import signal
import sys
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, List

from dotenv import load_dotenv
load_dotenv()

import yaml

from src.exchange.kraken_client import KrakenClient, OHLC, OrderSide, OrderType
from src.strategy.scalping_strategy import ScalpingStrategy, ScalpingConfig
from src.strategy.base_strategy import MarketData, Position
from src.strategy.regime_detector import RegimeDetector, RegimeConfig, MarketRegime
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

        # Fee rate depends on order type (maker = 0.16%, taker = 0.26%)
        use_maker = self.config.get('execution', {}).get('use_maker_orders', True)
        fee_rate = self.config.get('fees', {}).get('maker_percent', 0.16) if use_maker else self.config.get('fees', {}).get('taker_percent', 0.26)

        # Paper fills must pay the CONFIGURED fees, not the client's
        # hardcoded standard-tier defaults, or paper P&L silently desyncs
        # from config on any fee change (found in the Jul 2026 fee A/B).
        self.client._paper_maker_fee = self.config.get('fees', {}).get('maker_percent', 0.16) / 100.0
        self.client._paper_taker_fee = self.config.get('fees', {}).get('taker_percent', 0.26) / 100.0

        # Read indicators section as defaults (matches backtest config construction)
        ind = self.config.get('indicators', {})
        rsi_cfg = ind.get('rsi', {})
        stoch_cfg = ind.get('stochastic', {})
        bb_cfg = ind.get('bollinger', {})
        macd_cfg = ind.get('macd', {})
        obv_cfg = ind.get('obv', {})
        atr_cfg = ind.get('atr', {})
        vwap_cfg = ind.get('vwap', {})
        vol_cfg = ind.get('volume', {})
        shorting_cfg = self.config.get('shorting', {})
        # Kraken spot shorts are margin orders; leverage >= 2 is required by
        # the exchange. Volume is still set by the strategy, so leverage only
        # affects collateral, never position size.
        self.short_leverage = int(shorting_cfg.get('leverage', 2))
        ema_filter_cfg = self.config.get('ema_filter', {})
        ema_bearish_thr = ema_filter_cfg.get('bearish_threshold', -0.5)

        # Default strategy config built from indicators section + strategy section
        default_config = ScalpingConfig(
            take_profit_percent=self.config.get('strategy', {}).get('take_profit_percent', 5.0),
            stop_loss_percent=self.config.get('strategy', {}).get('stop_loss_percent', 2.5),
            min_confirmations=self.config.get('strategy', {}).get('min_confirmations', 3),
            fee_percent=fee_rate,
            rsi_period=rsi_cfg.get('period', 10),
            rsi_oversold=rsi_cfg.get('oversold', 28.0),
            rsi_overbought=rsi_cfg.get('overbought', 70.0),
            stoch_k_period=stoch_cfg.get('k_period', 13),
            stoch_d_period=stoch_cfg.get('d_period', 3),
            stoch_oversold=stoch_cfg.get('oversold', 23.0),
            stoch_overbought=stoch_cfg.get('overbought', 75.0),
            bb_period=bb_cfg.get('period', 20),
            bb_std_dev=bb_cfg.get('std_dev', 2.0),
            bb_squeeze_threshold=bb_cfg.get('squeeze_threshold', 3.0),
            macd_fast=macd_cfg.get('fast_period', 12),
            macd_slow=macd_cfg.get('slow_period', 26),
            macd_signal=macd_cfg.get('signal_period', 9),
            obv_sma_period=obv_cfg.get('sma_period', 20),
            atr_period=atr_cfg.get('period', 14),
            atr_stop_multiplier=atr_cfg.get('stop_multiplier', 2.5),
            atr_tp_multiplier=atr_cfg.get('tp_multiplier', 2.75),
            use_atr_stops=atr_cfg.get('use_dynamic_stops', True),
            vwap_threshold_percent=vwap_cfg.get('threshold_percent', 0.5),
            vwap_anchor_candles=vwap_cfg.get('anchor_candles', 0),
            volume_spike_threshold=vol_cfg.get('spike_threshold', 1.5),
            shorting_enabled=shorting_cfg.get('enabled', True),
            short_min_confirmations=shorting_cfg.get('min_confirmations', 3),
            ema_bearish_threshold=ema_bearish_thr,
            dedup_rsi_stoch=self.config.get('strategy', {}).get('dedup_rsi_stoch', False),
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
                bb_squeeze_threshold=params.get('bb_squeeze_threshold', default_config.bb_squeeze_threshold),
                vwap_threshold_percent=params.get('vwap_threshold_percent', default_config.vwap_threshold_percent),
                vwap_anchor_candles=params.get('vwap_anchor_candles', default_config.vwap_anchor_candles),
                volume_spike_threshold=params.get('volume_spike_threshold', default_config.volume_spike_threshold),
                min_confirmations=params.get('min_confirmations', default_config.min_confirmations),
                dedup_rsi_stoch=default_config.dedup_rsi_stoch,
                fee_percent=fee_rate,
                stoch_k_period=params.get('stoch_k_period', default_config.stoch_k_period),
                stoch_d_period=params.get('stoch_d_period', default_config.stoch_d_period),
                stoch_oversold=params.get('stoch_oversold', default_config.stoch_oversold),
                stoch_overbought=params.get('stoch_overbought', default_config.stoch_overbought),
                macd_fast=params.get('macd_fast', default_config.macd_fast),
                macd_slow=params.get('macd_slow', default_config.macd_slow),
                macd_signal=params.get('macd_signal', default_config.macd_signal),
                obv_sma_period=params.get('obv_sma_period', default_config.obv_sma_period),
                atr_period=params.get('atr_period', default_config.atr_period),
                atr_stop_multiplier=params.get('atr_stop_multiplier', default_config.atr_stop_multiplier),
                atr_tp_multiplier=params.get('atr_tp_multiplier', default_config.atr_tp_multiplier),
                use_atr_stops=params.get('use_atr_stops', default_config.use_atr_stops),
                shorting_enabled=params.get('shorting_enabled', default_config.shorting_enabled),
                short_min_confirmations=params.get('short_min_confirmations', default_config.short_min_confirmations),
                ema_bearish_threshold=params.get('ema_bearish_threshold', default_config.ema_bearish_threshold),
            )
            self.pair_strategies[pair] = ScalpingStrategy(pair_config)

        adaptive_config = AdaptiveConfig(
            min_win_rate=self.config.get('adaptive', {}).get('min_win_rate', 0.35),
            min_profit_factor=self.config.get('adaptive', {}).get('min_profit_factor', 0.7),
            max_consecutive_losses=self.config.get('adaptive', {}).get('max_consecutive_losses', 5),
            cooldown_hours=self.config.get('adaptive', {}).get('cooldown_hours', 2.0),
            reenable_win_rate=self.config.get('adaptive', {}).get('reenable_win_rate', 0.45),
            confidence_scaling=self.config.get('position', {}).get('confidence_scaling', False),
            min_position_scale=self.config.get('position', {}).get('min_scale', 0.5),
            max_position_scale=self.config.get('position', {}).get('max_scale', 1.5),
        )
        self.pair_manager = AdaptivePairManager(adaptive_config)

        # Trading state (set early so circuit breaker can reference initial_capital)
        self.pairs = self.config.get('pairs', ['SOL/USD'])
        self.positions: Dict[str, dict] = {}
        self.capital = self._fetch_initial_capital()
        self.initial_capital = self.capital
        self.position_size_pct = self.config.get('position', {}).get('size_percent', 20.0)

        # Regime detector - classifies bull/bear/sideways
        regime_cfg = self.config.get('regime_detector', {})
        self.regime_detector = RegimeDetector(RegimeConfig(
            fast_sma_period=regime_cfg.get('fast_sma_period', 20),
            slow_sma_period=regime_cfg.get('slow_sma_period', 100),
            bull_slope_threshold=regime_cfg.get('bull_slope_threshold', 0.02),
            bear_slope_threshold=regime_cfg.get('bear_slope_threshold', -0.02),
        ))
        self._current_regime = MarketRegime.UNKNOWN
        self._regime_adjustments: Dict[str, float] = {}
        self._regime_check_counter = 0
        # Re-detect every N committed reference candles. This slow cadence is a
        # load-bearing whipsaw debouncer — see _update_regime docstring.
        self._regime_check_interval = regime_cfg.get('check_every_candles', 20)
        self._last_regime_candle_ts: Optional[datetime] = None

        # Entry cooldown after a regime flip (0 = disabled). Forensics across
        # Q4'24-Q1'26: trades entered within 20 candles of a flip lost -$2,010
        # net (regime-flip chasing). Counts committed reference candles.
        self.flip_entry_cooldown = int(regime_cfg.get('flip_entry_cooldown_candles', 0))
        self._candles_since_flip = 10 ** 9  # no flip seen yet -> never blocks

        # Dead-zone: regimes in which NEW entries are suppressed entirely
        # ("sit out the chop"), and regimes in which only SHORTS are
        # suppressed (e.g. grindy bear chop where shorts get squeezed and
        # bleed rollover). Exits and open positions are NEVER affected — only
        # fresh entries. Empty by default (no suppression).
        self._block_entry_regimes = set(regime_cfg.get('block_entry_regimes', []) or [])
        self._block_short_regimes = set(regime_cfg.get('block_short_regimes', []) or [])

        # Wall clock, injectable by test harnesses that simulate historical time
        # (used to detect Kraken's in-progress candle)
        self._now = lambda: datetime.now(timezone.utc)

        # Trailing stops (disabled by default — params optimized without them)
        ts_cfg = self.config.get('trailing_stops', {})
        self.trailing_stops_enabled = ts_cfg.get('enabled', False)
        self.trail_trigger_pct = ts_cfg.get('trail_trigger_pct', 0.5)
        self.trail_lock_pct = ts_cfg.get('trail_lock_pct', 0.5)

        # Store base configs for regime adjustment
        self._base_default_config = default_config
        self._base_pair_params = dict(pair_params)

        # Order execution settings
        self.use_maker_orders = self.config.get('execution', {}).get('use_maker_orders', True)
        self.maker_price_offset = self.config.get('execution', {}).get('maker_price_offset', 0.0)
        # Disaster stop: intra-candle hard loss floor (% of entry), 0 = disabled
        self.disaster_stop_pct = float(self.config.get('risk', {}).get('disaster_stop_percent', 0.0))
        # Fee rate depends on order type
        self.fee_rate = self.config.get('fees', {}).get('maker_percent', 0.16) if self.use_maker_orders else self.config.get('fees', {}).get('taker_percent', 0.26)
        # Margin rollover: Kraken charges this (% of notional) per started 4h
        # period on OPEN MARGIN positions (shorts). Longs are spot — no
        # rollover. Live charges it on the real account; paper/replay don't
        # see it natively, so we deduct it in _close_position in ALL modes so
        # the software P&L ledger matches what a multi-day short truly costs.
        self.margin_rollover_pct_4h = float(
            self.config.get('fees', {}).get('margin_rollover_percent_4h', 0.0))

        # Candle interval from config (default 60 for backward compatibility)
        self.candle_interval = self.config.get('strategy', {}).get('candle_interval', 60)

        # Per-pair candle intervals (allows mixing 4h and 12h per pair)
        self.pair_intervals: Dict[str, int] = {}
        for pair, params in pair_params.items():
            if 'candle_interval' in params:
                self.pair_intervals[pair] = params['candle_interval']

        # Correlation-aware position limits
        corr_cfg = self.config.get('correlation_limits', {})
        self.correlation_enabled = corr_cfg.get('enabled', False)
        self.correlation_groups: list[dict] = corr_cfg.get('groups', [])

        # Volume regime filter
        vol_cfg = self.config.get('volume_filter', {})
        self.volume_filter_enabled = vol_cfg.get('enabled', False)
        self.volume_filter_min_ratio = vol_cfg.get('min_ratio', 0.5)
        self.volume_filter_lookback = vol_cfg.get('lookback_candles', 30)

        # One-decision-per-candle guard. Live runner polls every 60s, but a 4h
        # strategy has only one new "frame" every 4 hours. Without this, brief
        # ticker spikes against the candle's high/low caused same-candle round
        # trips (e.g., POL: 41 trades/wk, all noise). Track the last candle
        # timestamp processed per pair; skip if unchanged.
        self._last_processed_candle_ts: Dict[str, datetime] = {}

        # Live order-lifecycle settings (only consulted when paper_trading is
        # False; paper fills are instant so none of this applies there).
        live_cfg = self.config.get('live', {})
        self.live_fill_timeout = float(live_cfg.get('fill_timeout_seconds', 120))
        self.live_fill_poll = float(live_cfg.get('fill_poll_seconds', 5))
        self.live_entry_market_fallback = bool(live_cfg.get('entry_fallback_to_market', True))
        self.live_postonly_retries = int(live_cfg.get('max_postonly_retries', 2))
        self.live_server_stop = bool(live_cfg.get('server_side_disaster_stop', True))
        self.live_flatten_unknown_margin = bool(live_cfg.get('flatten_unknown_margin', True))
        self.live_cancel_stale_orders = bool(live_cfg.get('cancel_stale_orders_on_start', True))
        self.live_drift_check_cycles = int(live_cfg.get('drift_check_every_cycles', 60))
        self._cycles_since_drift_check = 0
        self._short_blocked_pairs: set = set()

        # Concurrent pair processing (LIVE only). Sequential processing blocks
        # the whole loop on each pair's maker-fill wait (up to fill_timeout);
        # one slow fill delays every later pair's entries, exits and disaster
        # checks. With concurrent_pairs > 1, pairs are processed in a bounded
        # thread pool (Kraken HTTP calls are I/O-bound and release the GIL).
        # Default 1 = sequential (unchanged). Paper/replay always run
        # sequentially — fills are instant so there is no benefit, and it
        # preserves deterministic replay parity. Shared capital/position state
        # is guarded by _state_lock with a capital-reservation model so
        # concurrent entries cannot double-allocate the same dollars.
        self.live_concurrent_pairs = max(1, int(live_cfg.get('concurrent_pairs', 1)))
        self._state_lock = threading.RLock()
        self._reserved_usd = 0.0

        # Injectable sleep (test harnesses replace with a no-op so fill-wait
        # loops run deterministically without wall-clock delays)
        self._sleep = time.sleep

        # Metrics
        self.metrics = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'peak_capital': 0.0,
            'start_time': datetime.now(timezone.utc).isoformat()
        }

        # Per-pair indicator snapshots (updated each cycle, saved to state for dashboard)
        self.indicator_snapshots: Dict[str, dict] = {}

        # Initialize logger BEFORE loading state
        self.logger = get_logger(__name__)

        # Register pairs
        for pair in self.pairs:
            self.pair_manager.register_pair(pair)

        # Load saved state
        self._load_state()

        # LIVE ONLY: reconcile saved state against the exchange's actual
        # positions/orders. state.json can go stale while the process is down
        # (stops triggered, manual intervention, liquidation) — trusting it
        # blindly causes double entries and orphaned margin positions.
        if not paper_trading:
            self._reconcile_live_state()
            self._report_short_eligibility()

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
            orig_paper = getattr(self.client, 'paper_trading', True)
            self.client.paper_trading = False
            try:
                balances = self.client.get_balances()
            finally:
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
                # Load persisted metrics but keep this process's start_time
                # so "Uptime" reflects how long the trader has been running,
                # not how long since state.json was first created.
                loaded_metrics = state.get('metrics', {})
                process_start_time = self.metrics['start_time']
                self.metrics = {**self.metrics, **loaded_metrics,
                                'start_time': process_start_time}
                self.capital = state.get('capital', self.capital)
                self.pair_manager.from_dict(state.get('pair_manager', {}))
                if 'regime_detector' in state:
                    self.regime_detector.from_dict(state['regime_detector'])
                if 'current_regime' in state:
                    try:
                        self._current_regime = MarketRegime(state['current_regime'])
                        self._regime_adjustments = self.regime_detector.get_adjustments(self._current_regime)
                        self._rebuild_strategies_for_regime(self._regime_adjustments)
                    except ValueError:
                        pass
                self.indicator_snapshots = state.get('indicator_snapshots', {})

                if 'initial_capital' in state:
                    self.initial_capital = state['initial_capital']

                self.logger.info("Loaded saved state",
                                 capital=f"${self.capital:.2f}",
                                 initial_capital=f"${self.initial_capital:.2f}",
                                 positions=len(self.positions))

                # Reconcile paper balances with open positions.
                # On restart, paper balances are initialized from the real
                # Kraken account (which has zero crypto for paper positions).
                # Without this, exit orders fail with "Insufficient balance"
                # and positions get stuck open forever.
                if self.paper_trading and self.positions:
                    self._reconcile_paper_balances()

            except Exception as e:
                self.logger.error(f"Error loading state: {e}")

    def _reconcile_paper_balances(self) -> None:
        """Ensure paper balances reflect open positions from saved state.

        On restart the KrakenClient re-initialises paper balances from the
        real Kraken account, which holds zero crypto for simulated positions.
        This causes every exit order to fail with 'Insufficient balance',
        leaving positions stuck open indefinitely.

        Fix: for every open position, ensure the paper balance for the base
        asset is at least as large as the position size.
        """
        for pair, pos in self.positions.items():
            base = pair.split("/")[0]
            size = pos.get('size', 0)
            if size <= 0:
                continue
            current = self.client._paper_balances.get(base, 0)
            if pos.get('side') == 'short':
                # Short inventory is negative (we owe the asset). The cover is
                # a reduce_only buy, so no balance gate applies — this keeps
                # the bookkeeping consistent across restarts.
                if current > -size:
                    self.client._paper_balances[base] = -size
                    self.logger.info(
                        f"Reconciled paper balance for {base}: "
                        f"{current:.6f} -> {-size:.6f} (open SHORT in {pair})"
                    )
            elif current < size:
                self.client._paper_balances[base] = size
                self.logger.info(
                    f"Reconciled paper balance for {base}: "
                    f"{current:.6f} -> {size:.6f} (open position in {pair})"
                )

        # Also reconcile USD balance so new entry orders don't fail
        # when the real Kraken balance is smaller than the paper capital.
        deployed = sum(p.get('size_usd', 0) for p in self.positions.values())
        available_capital = self.capital - deployed
        if self.client._paper_balances.get("USD", 0) < available_capital:
            self.client._paper_balances["USD"] = available_capital
            self.logger.info(
                f"Reconciled paper USD balance to ${available_capital:.2f}"
            )

    def _save_state(self) -> None:
        """Save state to disk."""
        # Snapshot mutable shared state under the lock so a concurrent entry/
        # exit can't change dict sizes mid-serialization (RLock = safe even
        # when an outer caller already holds it).
        with self._state_lock:
            state = {
                'positions': {k: dict(v) for k, v in self.positions.items()},
                'metrics': dict(self.metrics),
                'capital': self.capital,
                'initial_capital': self.initial_capital,
                'paper_trading': self.paper_trading,
                'pair_manager': self.pair_manager.to_dict(),
                'regime_detector': self.regime_detector.to_dict(),
                'current_regime': self._current_regime.value,
                'indicator_snapshots': self.indicator_snapshots,
                'last_update': datetime.now(timezone.utc).isoformat()
            }
        state_file = self.data_dir / "state.json"
        tmp_file = self.data_dir / "state.json.tmp"
        with open(tmp_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp_file, state_file)

    # Valid Kraken API OHLC intervals (minutes)
    VALID_API_INTERVALS = {1, 5, 15, 30, 60, 240, 1440, 10080, 21600}

    def _drop_in_progress(self, candles: list[OHLC], interval_minutes: int) -> list[OHLC]:
        """Drop Kraken's in-progress candle (always the last array entry).

        Trading decisions must only see committed candles. The in-progress
        stub has partial volume and near-zero range, which poisons the
        volume filter (stub volume vs 30-candle SMA always fails) and the
        doji filter. Combined with the one-decision-per-candle guard firing
        at the first poll of a new candle, this blocked every 4h-pair entry
        in production for 18 days (May 2026)."""
        if candles and candles[-1].timestamp + timedelta(minutes=interval_minutes) > self._now():
            return candles[:-1]
        return candles

    @staticmethod
    def _aggregate_candles(candles: list[OHLC], factor: int, target_interval_minutes: int) -> list[OHLC]:
        """Aggregate smaller candles into larger ones, anchored to epoch-aligned
        boundaries (12h candles always span 00:00-12:00 / 12:00-00:00 UTC).

        Anchoring to the window's first element is wrong: Kraken serves a
        sliding ~720-candle window, so position-based grouping re-phases every
        base candle. In production this made 12h candle boundaries shift every
        4h, tripling the decision frequency for 12h pairs and evaluating
        indicators on re-phased candles (May 2026: SOL/POL decisions at
        16:00/20:00 UTC). Only complete buckets are emitted, so the trailing
        partial bucket never produces a decision candle."""
        bucket_seconds = target_interval_minutes * 60
        buckets: dict[int, list[OHLC]] = {}
        for c in candles:
            b = int(c.timestamp.timestamp()) // bucket_seconds
            buckets.setdefault(b, []).append(c)
        aggregated = []
        for b in sorted(buckets):
            group = buckets[b]
            if len(group) != factor:
                continue  # partial leading/trailing bucket
            aggregated.append(OHLC(
                timestamp=group[0].timestamp,
                open=group[0].open,
                high=max(c.high for c in group),
                low=min(c.low for c in group),
                close=group[-1].close,
                vwap=sum(c.vwap * c.volume for c in group) / max(sum(c.volume for c in group), 1e-10),
                volume=sum(c.volume for c in group),
                count=sum(c.count for c in group),
            ))
        return aggregated

    def _fetch_ohlc(self, pair: str, interval: int) -> list[OHLC]:
        """Fetch OHLC data, aggregating from a smaller interval if needed.

        Always strips the in-progress candle so strategy decisions see only
        committed data — matching the replay/backtest execution model."""
        if interval in self.VALID_API_INTERVALS:
            raw = self.client.get_ohlc(pair, interval=interval)
            return self._drop_in_progress(raw, interval)

        # Find the largest valid interval that evenly divides the target
        for base in sorted(self.VALID_API_INTERVALS, reverse=True):
            if base < interval and interval % base == 0:
                factor = interval // base
                raw = self.client.get_ohlc(pair, interval=base)
                raw = self._drop_in_progress(raw, base)
                return self._aggregate_candles(raw, factor, interval)

        # Fallback: use as-is (will likely error, same as before)
        return self.client.get_ohlc(pair, interval=interval)

    def _get_market_data(self, pair: str) -> Optional[MarketData]:
        """Fetch market data for a pair."""
        try:
            interval = self.pair_intervals.get(pair, self.candle_interval)
            ohlc = self._fetch_ohlc(pair, interval)
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

        Cadence is anchored to COMMITTED reference candles, not loop cycles:
        re-detection runs every `_regime_check_interval` new BTC candles. The
        previous cycle-count cadence meant live re-checked every ~30 minutes
        while the validated replay re-checked every 20 candles (3.3 days) —
        per-candle checking in replay turns Q4 2024 from +$1,068 into -$595
        (43 regime flips vs 4). The slow cadence is a load-bearing debouncer.
        """
        # Use BTC as regime reference (largest, most liquid)
        reference_pair = "BTC/USD"
        try:
            # _fetch_ohlc strips the in-progress candle, so regime SMAs only
            # see committed data (same execution model as the strategies)
            ohlc = self._fetch_ohlc(reference_pair, self.candle_interval)
            if not ohlc:
                return
        except Exception as e:
            self.logger.error(f"Error fetching regime data: {e}")
            return

        # Count only newly committed reference candles
        latest_ts = ohlc[-1].timestamp
        if latest_ts != self._last_regime_candle_ts:
            self._last_regime_candle_ts = latest_ts
            self._regime_check_counter += 1
            if self._candles_since_flip < 10 ** 9:
                self._candles_since_flip += 1
        if self._regime_check_counter < self._regime_check_interval:
            return
        self._regime_check_counter = 0

        if len(ohlc) < self.regime_detector.config.min_data_points:
            return  # Not enough data yet

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

            # Initial detection (UNKNOWN -> X) is bootstrapping, not a flip —
            # otherwise every process restart would freeze entries for the
            # cooldown window.
            if self._current_regime != MarketRegime.UNKNOWN:
                self._candles_since_flip = 0

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
        short_conf_offset = adjustments.get('short_confirmations_offset', 0)
        ema_enabled = adjustments.get('ema_filter_enabled', True)
        trend_short = adjustments.get('trend_short_enabled', False)
        shorting = adjustments.get('shorting_enabled', True)

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
            vwap_anchor_candles=base.vwap_anchor_candles,
            dedup_rsi_stoch=base.dedup_rsi_stoch,
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
            shorting_enabled=shorting,
            short_min_confirmations=max(1, base.short_min_confirmations + short_conf_offset),
            trend_short_enabled=trend_short,
            ema_bearish_threshold=base.ema_bearish_threshold,
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
                bb_squeeze_threshold=params.get('bb_squeeze_threshold', base.bb_squeeze_threshold),
                vwap_threshold_percent=params.get('vwap_threshold_percent', base.vwap_threshold_percent),
                vwap_anchor_candles=params.get('vwap_anchor_candles', base.vwap_anchor_candles),
                dedup_rsi_stoch=base.dedup_rsi_stoch,
                volume_spike_threshold=params.get('volume_spike_threshold', base.volume_spike_threshold),
                min_confirmations=max(1, base_conf + conf_offset),
                fee_percent=fee_rate,
                ema_filter_enabled=ema_enabled,
                stoch_k_period=params.get('stoch_k_period', base.stoch_k_period),
                stoch_d_period=params.get('stoch_d_period', base.stoch_d_period),
                stoch_oversold=params.get('stoch_oversold', base.stoch_oversold),
                stoch_overbought=params.get('stoch_overbought', base.stoch_overbought),
                macd_fast=params.get('macd_fast', base.macd_fast),
                macd_slow=params.get('macd_slow', base.macd_slow),
                macd_signal=params.get('macd_signal', base.macd_signal),
                obv_sma_period=params.get('obv_sma_period', base.obv_sma_period),
                atr_period=params.get('atr_period', base.atr_period),
                atr_stop_multiplier=params.get('atr_stop_multiplier', base.atr_stop_multiplier),
                atr_tp_multiplier=params.get('atr_tp_multiplier', base.atr_tp_multiplier),
                use_atr_stops=params.get('use_atr_stops', base.use_atr_stops),
                shorting_enabled=shorting and params.get('shorting_enabled', True),
                short_min_confirmations=max(1, params.get('short_min_confirmations', base.short_min_confirmations) + short_conf_offset),
                trend_short_enabled=trend_short,
                ema_bearish_threshold=params.get('ema_bearish_threshold', base.ema_bearish_threshold),
            )
            self.pair_strategies[pair] = ScalpingStrategy(pair_config)

        self.logger.info(
            f"Strategies rebuilt for {self._current_regime.value} regime",
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            conf_offset=conf_offset,
            ema_filter=ema_enabled,
        )

    def _execute_entry_order(self, pair: str, side: str, size: float, current_price: float) -> Optional[dict]:
        """
        Place an entry order on Kraken.

        Uses maker (limit) orders when configured for better fees.
        Falls back to market order if maker order fails.

        Returns:
            dict with order details, or None if order failed.
        """
        order_side = OrderSide.BUY if side == "long" else OrderSide.SELL
        # A short entry sells an asset we don't hold: on Kraken that must be a
        # margin order (leverage set) or the exchange rejects it. Longs stay
        # plain spot orders.
        margin_kwargs = {"leverage": self.short_leverage} if side == "short" else {}

        if not self.paper_trading:
            # Margin eligibility: not every Kraken pair supports margin. A
            # short on an ineligible pair would be rejected by the exchange —
            # block it here with a clear reason instead.
            if side == "short":
                usable = self.client.margin_leverage_for(
                    pair, order_side, self.short_leverage)
                if usable is None:
                    if pair not in self._short_blocked_pairs:
                        self._short_blocked_pairs.add(pair)
                        self.logger.warning(
                            f"SHORT BLOCKED for {pair} — no margin support on "
                            f"Kraken (paper results for this pair's shorts do "
                            f"not transfer to live)"
                        )
                    return None
                margin_kwargs = {"leverage": usable}
            return self._execute_live_order(
                pair, order_side, size, margin_kwargs,
                is_exit=False,
                allow_market_fallback=self.live_entry_market_fallback,
            )

        try:
            if self.use_maker_orders:
                order = self.client.place_maker_order(
                    pair=pair,
                    side=order_side,
                    volume=size,
                    price_offset_percent=self.maker_price_offset,
                    **margin_kwargs,
                )
            else:
                order = self.client.place_order(
                    pair=pair,
                    side=order_side,
                    order_type=OrderType.MARKET,
                    volume=size,
                    **margin_kwargs,
                )

            self.logger.info(
                f"ORDER PLACED for {pair}",
                order_id=order.order_id,
                side=order_side.value,
                order_type=order.order_type.value,
                volume=size,
                price=order.price,
                status=order.status,
            )

            fill_price = order.price or current_price
            return {
                'order_id': order.order_id,
                'fill_price': fill_price,
                'fill_volume': order.filled_volume or size,
                'fee': order.fee,
                'status': order.status,
            }

        except Exception as e:
            self.logger.error(f"ENTRY ORDER FAILED for {pair}: {e}")
            return None

    def _execute_exit_order(self, pair: str, side: str, size: float) -> Optional[dict]:
        """
        Place an exit order on Kraken.

        Uses maker orders when configured for lower fees (0.16% vs 0.26%).
        Falls back to market order if maker order fails.

        Returns:
            dict with order details, or None if order failed.
        """
        # To close: sell if long, buy if short
        order_side = OrderSide.SELL if side == "long" else OrderSide.BUY
        # Covering a short must also carry leverage so Kraken nets it against
        # the open margin position (a plain buy would be a spot purchase).
        # reduce_only guarantees the cover can only close — a size mismatch
        # can never flip us into an unintended long margin position.
        margin_kwargs = (
            {"leverage": self.short_leverage, "reduce_only": True}
            if side == "short" else {}
        )

        if not self.paper_trading:
            # Exits ALWAYS fall back to a market order: failing to exit while
            # price runs away costs more than any taker fee.
            return self._execute_live_order(
                pair, order_side, size, margin_kwargs,
                is_exit=True, allow_market_fallback=True,
            )

        try:
            if self.use_maker_orders:
                order = self.client.place_maker_order(
                    pair=pair,
                    side=order_side,
                    volume=size,
                    price_offset_percent=self.maker_price_offset,
                    **margin_kwargs,
                )
            else:
                order = self.client.place_order(
                    pair=pair,
                    side=order_side,
                    order_type=OrderType.MARKET,
                    volume=size,
                    **margin_kwargs,
                )

            self.logger.info(
                f"EXIT ORDER PLACED for {pair}",
                order_id=order.order_id,
                side=order_side.value,
                volume=size,
                status=order.status,
            )

            return {
                'order_id': order.order_id,
                'fill_price': order.price,
                'fill_volume': order.filled_volume or size,
                'fee': order.fee,
                'status': order.status,
            }

        except Exception as e:
            self.logger.error(f"EXIT ORDER FAILED for {pair}: {e}")
            return None

    # =========================================================================
    # LIVE order lifecycle (never used in paper mode — paper fills instantly)
    # =========================================================================

    def _execute_live_order(self, pair: str, order_side: OrderSide, volume: float,
                            margin_kwargs: dict, is_exit: bool,
                            allow_market_fallback: bool) -> Optional[dict]:
        """
        Place an order live and manage it to completion.

        Live maker orders return 'open' — they rest on the book. This waits
        for the fill, cancels on timeout, keeps partial entry fills, and falls
        back to a market order where allowed (always, for exits). Returns the
        same result dict shape as the paper path, with ACTUAL fill price,
        volume, and fee from the exchange.
        """
        label = "EXIT" if is_exit else "ENTRY"
        placed = None

        if self.use_maker_orders:
            # Post-only orders are REJECTED outright if price moved across our
            # limit between the ticker fetch and placement — common in fast
            # tape. place_maker_order re-fetches the ticker on each attempt.
            for attempt in range(1 + self.live_postonly_retries):
                try:
                    placed = self.client.place_maker_order(
                        pair=pair, side=order_side, volume=volume,
                        price_offset_percent=self.maker_price_offset,
                        **margin_kwargs,
                    )
                    break
                except Exception as e:
                    msg = str(e)
                    if "Post only" in msg and attempt < self.live_postonly_retries:
                        self.logger.info(
                            f"{label} post-only rejected for {pair} (price moved), retrying"
                        )
                        continue
                    if "below pair minimum" in msg:
                        self.logger.warning(f"{label} skipped for {pair}: {msg}")
                        return None
                    self.logger.error(f"{label} maker placement failed for {pair}: {e}")
                    break

        if placed is None:
            if self.use_maker_orders and not allow_market_fallback:
                return None
            return self._market_order_final(pair, order_side, volume,
                                            margin_kwargs, label)

        fill = self._await_live_fill(placed.order_id, self.live_fill_timeout)
        if fill.get('status') == 'closed':
            return self._fill_result(placed.order_id, fill, placed.price)

        # Timed out / canceled / expired: cancel remainder, take final tally.
        final = self._cancel_live_order_final(placed.order_id)
        vol_exec = final.get('vol_exec', 0.0)

        if vol_exec >= volume * 0.999:
            return self._fill_result(placed.order_id, final, placed.price)

        remaining = volume - vol_exec
        if is_exit:
            # An un-exited remainder is an open risk — market it, always.
            self.logger.warning(
                f"EXIT maker order for {pair} filled {vol_exec}/{volume} in "
                f"{self.live_fill_timeout:.0f}s — market-ordering the remainder"
            )
            mkt = self._market_order_final(pair, order_side, remaining,
                                           margin_kwargs, label)
            if mkt is None:
                if vol_exec > 0:
                    self.logger.error(
                        f"CRITICAL: {pair} exit partially filled ({vol_exec}) and market "
                        f"remainder FAILED — position partially open on exchange"
                    )
                    return self._fill_result(placed.order_id, final, placed.price)
                return None
            if vol_exec > 0:
                return self._combine_fills(
                    self._fill_result(placed.order_id, final, placed.price), mkt)
            return mkt

        # Entry path
        if vol_exec > 0:
            self.logger.warning(
                f"ENTRY maker order for {pair} partially filled "
                f"({vol_exec}/{volume}) — keeping the partial as the position"
            )
            return self._fill_result(placed.order_id, final, placed.price)
        if allow_market_fallback:
            self.logger.info(
                f"ENTRY maker order for {pair} unfilled after "
                f"{self.live_fill_timeout:.0f}s — falling back to market (taker fee)"
            )
            return self._market_order_final(pair, order_side, volume,
                                            margin_kwargs, label)
        self.logger.info(f"ENTRY maker order for {pair} unfilled — skipping entry")
        return None

    def _market_order_final(self, pair: str, order_side: OrderSide, volume: float,
                            margin_kwargs: dict, label: str) -> Optional[dict]:
        """Place a market order and confirm its fill."""
        try:
            order = self.client.place_order(
                pair=pair, side=order_side, order_type=OrderType.MARKET,
                volume=volume, **margin_kwargs,
            )
        except Exception as e:
            self.logger.error(f"{label} market order failed for {pair}: {e}")
            return None
        # Paper market orders fill synchronously; live ones near-instantly.
        if order.status == "closed":
            return {
                'order_id': order.order_id,
                'fill_price': order.price,
                'fill_volume': order.filled_volume or volume,
                'fee': order.fee,
                'status': order.status,
            }
        fill = self._await_live_fill(order.order_id, timeout=30)
        if fill.get('vol_exec', 0.0) <= 0:
            self.logger.error(
                f"CRITICAL: {label} market order {order.order_id} for {pair} "
                f"did not fill — check exchange manually"
            )
            return None
        return self._fill_result(order.order_id, fill, order.price)

    def _await_live_fill(self, order_id: str, timeout: float) -> dict:
        """Poll QueryOrders until the order reaches a terminal state."""
        poll = max(self.live_fill_poll, 0.5)
        polls = max(1, int(timeout / poll))
        last: dict = {'status': 'unknown', 'vol_exec': 0.0, 'price': 0.0, 'fee': 0.0}
        for i in range(polls):
            try:
                last = self.client.query_order(order_id)
            except Exception as e:
                self.logger.warning(f"Fill poll failed for {order_id}: {e}")
            if last.get('status') in ('closed', 'canceled', 'expired'):
                return last
            if i < polls - 1:
                self._sleep(poll)
        return last

    def _cancel_live_order_final(self, order_id: str) -> dict:
        """Cancel an order and return its final state (fills can race cancels)."""
        try:
            self.client.cancel_order(order_id)
        except Exception as e:
            self.logger.warning(f"Cancel failed for {order_id} (may have just filled): {e}")
        try:
            return self.client.query_order(order_id)
        except Exception as e:
            self.logger.error(f"Could not query final state of {order_id}: {e}")
            return {'status': 'unknown', 'vol_exec': 0.0, 'price': 0.0, 'fee': 0.0}

    @staticmethod
    def _fill_result(order_id: str, info: dict, fallback_price: Optional[float]) -> dict:
        price = info.get('price', 0.0) or 0.0
        if price <= 0 and fallback_price:
            price = fallback_price
        return {
            'order_id': order_id,
            'fill_price': price,
            'fill_volume': info.get('vol_exec', 0.0),
            'fee': info.get('fee', 0.0),
            'status': info.get('status', 'unknown'),
        }

    @staticmethod
    def _combine_fills(a: dict, b: dict) -> dict:
        """Volume-weighted combination of two partial fills of the same intent."""
        va, vb = a.get('fill_volume', 0.0), b.get('fill_volume', 0.0)
        total = va + vb
        price = ((a.get('fill_price', 0.0) * va + b.get('fill_price', 0.0) * vb) / total
                 if total > 0 else a.get('fill_price', 0.0))
        return {
            'order_id': f"{a.get('order_id', '')}+{b.get('order_id', '')}",
            'fill_price': price,
            'fill_volume': total,
            'fee': a.get('fee', 0.0) + b.get('fee', 0.0),
            'status': 'closed',
        }

    # =========================================================================
    # LIVE server-side disaster stop
    # =========================================================================

    def _place_server_stop(self, pair: str, side: str, volume: float,
                           entry_price: float) -> Optional[str]:
        """
        Rest a stop-loss order ON THE EXCHANGE at the disaster-stop level.

        The in-process disaster stop (checked every poll) remains the primary;
        this is the backstop for the cases the process can't cover: crashes,
        seedbox outages, flash moves between polls. It deliberately uses the
        wide disaster level (default 10%), NOT the strategy's ATR stops —
        normal exits stay with the validated candle-close logic.
        """
        if self.paper_trading or self.disaster_stop_pct <= 0 or not self.live_server_stop:
            return None
        try:
            if side == "long":
                stop_side = OrderSide.SELL
                trigger = entry_price * (1 - self.disaster_stop_pct / 100)
                kwargs = {}
            else:
                stop_side = OrderSide.BUY
                trigger = entry_price * (1 + self.disaster_stop_pct / 100)
                usable = self.client.margin_leverage_for(
                    pair, OrderSide.SELL, self.short_leverage) or self.short_leverage
                kwargs = {"leverage": usable, "reduce_only": True}
            trigger = round(trigger, self.client._get_pair_decimals(pair))
            order = self.client.place_order(
                pair=pair, side=stop_side, order_type=OrderType.STOP_LOSS,
                volume=volume, price=trigger, **kwargs,
            )
            self.logger.info(
                f"Server-side disaster stop placed for {pair}",
                trigger=trigger, side=stop_side.value, order_id=order.order_id,
            )
            return order.order_id
        except Exception as e:
            self.logger.error(
                f"Could not place server-side disaster stop for {pair}: {e} — "
                f"in-process disaster stop remains active"
            )
            return None

    def _resolve_server_stop(self, pair: str, position: dict) -> tuple:
        """
        Before a strategy exit: cancel the resting server-side stop, or detect
        that it already triggered while we weren't looking.

        Returns (status, fill_info):
          ('none', None)      — no stop existed
          ('cancelled', None) — stop cancelled, proceed with exit order
          ('filled', info)    — stop ALREADY EXECUTED: position is closed on
                                the exchange; settle from the stop's fill, do
                                NOT place an exit order.

        If the true state can't be determined, proceeding with the exit is
        benign: a surviving long-stop sells inventory we no longer hold (the
        exchange rejects it) and a short-stop is reduce_only against a netted
        position (no-op).
        """
        stop_id = position.get('stop_order_id')
        if self.paper_trading or not stop_id:
            return ('none', None)
        try:
            info = self.client.query_order(stop_id)
            if info.get('status') == 'closed':
                return ('filled', info)
            if info.get('status') in ('canceled', 'expired'):
                return ('none', None)
            try:
                self.client.cancel_order(stop_id)
            except Exception as e:
                self.logger.warning(f"Stop cancel failed for {pair}: {e}")
            info = self.client.query_order(stop_id)
            if info.get('status') == 'closed':
                return ('filled', info)
            return ('cancelled', None)
        except Exception as e:
            self.logger.warning(
                f"Could not resolve server stop for {pair}: {e} — proceeding with exit"
            )
            return ('cancelled', None)

    # =========================================================================
    # LIVE state reconciliation & drift detection
    # =========================================================================

    def _kraken_pair_candidates(self, pair: str) -> set:
        """All names Kraken may use for a pair ('BTC/USD' -> XBTUSD, XXBTZUSD)."""
        base, quote = pair.split("/")
        kb = "XBT" if base == "BTC" else base
        return {f"{kb}{quote}", f"X{kb}Z{quote}", f"{kb}/{quote}"}

    def _pair_from_kraken(self, kraken_name: str) -> Optional[str]:
        for p in self.pairs:
            if kraken_name in self._kraken_pair_candidates(p):
                return p
        return None

    def _reconcile_live_state(self) -> None:
        """
        LIVE startup: make saved state agree with the exchange.

        - Cancels stale resting orders carrying our BOT_USERREF (a dead
          process leaves maker orders and stops on the book). Manual orders
          (no userref tag) are never touched.
        - Drops 'ghost' positions: in state.json but absent on the exchange
          (stop triggered while down, manual close, liquidation).
        - Flattens orphaned margin SHORTS: on the exchange but not in state —
          unmonitored leverage bleeding rollover fees. Config-gated
          (live.flatten_unknown_margin). Margin longs are never ours (the bot
          opens spot longs only) so they are alerted, never touched. Spot
          balances can be personal holdings and are never sold.
        - Re-places server-side disaster stops for surviving positions.
        """
        self.logger.info("LIVE: reconciling saved state against exchange...")
        try:
            open_orders = self.client.get_open_orders_raw()
            open_positions = self.client.get_open_positions()
            balances = self.client.get_balances()
        except Exception as e:
            self.logger.error(
                f"LIVE reconciliation could not fetch exchange state: {e} — "
                f"continuing with saved state AS-IS; verify manually"
            )
            return

        cancelled = 0
        if self.live_cancel_stale_orders:
            for txid, raw in open_orders.items():
                if str(raw.get('userref', '')) == str(self.client.BOT_USERREF):
                    try:
                        self.client.cancel_order(txid)
                        cancelled += 1
                    except Exception as e:
                        self.logger.warning(f"Could not cancel stale order {txid}: {e}")

        margin_open = []
        for ptxid, p in open_positions.items():
            vol_open = float(p.get('vol', 0) or 0) - float(p.get('vol_closed', 0) or 0)
            if vol_open > 0:
                margin_open.append({
                    'txid': ptxid, 'pair': p.get('pair', ''),
                    'type': p.get('type', ''), 'vol': vol_open, 'matched': False,
                })

        ghosts = []
        for pair in list(self.positions):
            pos = self.positions[pair]
            side = pos.get('side', 'long')
            size = pos.get('size', 0)
            if side == 'short':
                cands = self._kraken_pair_candidates(pair)
                match = next(
                    (m for m in margin_open
                     if not m['matched'] and m['type'] == 'sell'
                     and m['pair'] in cands
                     and abs(m['vol'] - size) <= max(size * 0.02, 1e-9)),
                    None)
                if match:
                    match['matched'] = True
                else:
                    ghosts.append(pair)
            else:
                base = pair.split("/")[0]
                bal = balances.get(base)
                if bal is None or bal.total < size * 0.98:
                    ghosts.append(pair)

        for pair in ghosts:
            self.logger.error(
                f"CRITICAL: GHOST position {pair} in state.json but not on the "
                f"exchange (closed externally while down?) — dropping from "
                f"state. P&L for this trade is NOT recorded."
            )
            del self.positions[pair]

        for m in margin_open:
            if m['matched']:
                continue
            our_pair = self._pair_from_kraken(m['pair'])
            if m['type'] == 'sell' and self.live_flatten_unknown_margin and our_pair:
                self.logger.error(
                    f"CRITICAL: ORPHAN margin short {m['pair']} vol={m['vol']} "
                    f"on exchange but not in state — flattening (it bleeds "
                    f"rollover fees and carries unmonitored risk)"
                )
                try:
                    usable = self.client.margin_leverage_for(
                        our_pair, OrderSide.SELL, self.short_leverage) or self.short_leverage
                    self.client.place_order(
                        pair=our_pair, side=OrderSide.BUY,
                        order_type=OrderType.MARKET, volume=m['vol'],
                        leverage=usable, reduce_only=True,
                    )
                except Exception as e:
                    self.logger.error(f"CRITICAL: orphan flatten FAILED for {m['pair']}: {e}")
            else:
                self.logger.error(
                    f"CRITICAL: unknown margin position on exchange: {m['pair']} "
                    f"{m['type']} vol={m['vol']} — NOT touching it; review manually"
                )

        # Old stops were cancelled above; every surviving position needs one.
        for pair, pos in self.positions.items():
            stop_id = self._place_server_stop(
                pair, pos.get('side', 'long'), pos.get('size', 0),
                pos.get('entry_price', 0))
            if stop_id:
                pos['stop_order_id'] = stop_id
            else:
                pos.pop('stop_order_id', None)

        self._save_state()
        self.logger.info(
            "LIVE reconciliation complete",
            stale_orders_cancelled=cancelled,
            ghosts_dropped=len(ghosts),
            positions_kept=len(self.positions),
        )

    def _report_short_eligibility(self) -> None:
        """LIVE startup: which configured pairs can actually be shorted."""
        for pair in self.pairs:
            usable = self.client.margin_leverage_for(
                pair, OrderSide.SELL, self.short_leverage)
            if usable is None:
                self._short_blocked_pairs.add(pair)
                self.logger.warning(
                    f"SHORTS UNAVAILABLE for {pair} (no Kraken margin support) "
                    f"— paper/replay short P&L on this pair will not occur live"
                )
            else:
                self.logger.info(f"Shorts enabled for {pair} at {usable}x leverage")

    def _check_live_drift(self) -> None:
        """
        Periodic LIVE sanity check (read-only): alert if internal positions
        diverge from the exchange. Mutating fixes happen only at startup
        reconciliation, where there is no in-flight activity to race.
        """
        open_positions = self.client.get_open_positions()
        balances = self.client.get_balances()
        margin_by_pair: Dict[str, float] = {}
        for p in open_positions.values():
            vol_open = float(p.get('vol', 0) or 0) - float(p.get('vol_closed', 0) or 0)
            if vol_open > 0 and p.get('type') == 'sell':
                our = self._pair_from_kraken(p.get('pair', ''))
                if our:
                    margin_by_pair[our] = margin_by_pair.get(our, 0) + vol_open

        for pair, pos in self.positions.items():
            side, size = pos.get('side', 'long'), pos.get('size', 0)
            if side == 'short':
                on_exch = margin_by_pair.pop(pair, 0.0)
                if abs(on_exch - size) > size * 0.02:
                    self.logger.error(
                        f"CRITICAL DRIFT: {pair} short size {size} in state vs "
                        f"{on_exch} on exchange"
                    )
            else:
                base = pair.split("/")[0]
                bal = balances.get(base)
                if bal is None or bal.total < size * 0.98:
                    self.logger.error(
                        f"CRITICAL DRIFT: {pair} long size {size} in state but "
                        f"balance is {bal.total if bal else 0}"
                    )
        for pair, vol in margin_by_pair.items():
            self.logger.error(
                f"CRITICAL DRIFT: margin short {pair} vol={vol} on exchange "
                f"with no matching state position"
            )

    def _get_strategy_for_pair(self, pair: str) -> ScalpingStrategy:
        """Get the strategy for a specific pair (per-pair or default)."""
        return self.pair_strategies.get(pair, self.strategy)

    def _correlation_blocked(self, pair: str) -> Optional[str]:
        """Return reason string if this pair is blocked by correlation limits, else None."""
        if not self.correlation_enabled:
            return None
        for group in self.correlation_groups:
            group_pairs = group.get('pairs', [])
            if pair not in group_pairs:
                continue
            max_pos = group.get('max_positions', len(group_pairs))
            current = sum(1 for p in self.positions if p in group_pairs)
            if current >= max_pos:
                return (
                    f"correlation group '{group.get('name', '?')}' at limit "
                    f"({current}/{max_pos})"
                )
        return None

    def _volume_filter_blocked(self, market_data: MarketData) -> Optional[str]:
        """Return reason string if entry is blocked by low volume, else None."""
        if not self.volume_filter_enabled or not market_data.ohlc:
            return None
        candles = market_data.ohlc
        if len(candles) < self.volume_filter_lookback + 1:
            return None  # not enough history yet
        recent = candles[-self.volume_filter_lookback - 1:-1]
        avg_vol = sum(c.volume for c in recent) / len(recent)
        if avg_vol <= 0:
            return None
        current_vol = candles[-1].volume
        ratio = current_vol / avg_vol
        if ratio < self.volume_filter_min_ratio:
            return f"volume {ratio:.2f}x of {self.volume_filter_lookback}-candle SMA (need >= {self.volume_filter_min_ratio})"
        return None

    def _compute_indicator_snapshot(self, pair: str, strategy: 'ScalpingStrategy', market_data: MarketData, current_price: float) -> None:
        """Compute and cache indicator values for the dashboard."""
        try:
            signals = strategy._calculate_signals(market_data)
            stop_pct, tp_pct = strategy._get_dynamic_stops(signals)

            rsi = signals.get("rsi")
            stoch = signals.get("stochastic")
            macd = signals.get("macd")
            bb = signals.get("bollinger")
            vwap = signals.get("vwap")
            atr = signals.get("atr")
            ema = signals.get("ema")
            obv = signals.get("obv")
            volume = signals.get("volume")

            snapshot = {
                "price": current_price,
                "rsi": round(rsi.value, 1) if rsi else None,
                "rsi_oversold": strategy.config.rsi_oversold,
                "rsi_overbought": strategy.config.rsi_overbought,
                "stoch_k": round(stoch.k_value, 1) if stoch else None,
                "stoch_d": round(stoch.d_value, 1) if stoch and hasattr(stoch, 'd_value') else None,
                "stoch_oversold": strategy.config.stoch_oversold,
                "stoch_overbought": strategy.config.stoch_overbought,
                "macd_histogram": round(macd.histogram, 4) if macd else None,
                "macd_signal_line": round(macd.signal_line, 4) if macd and hasattr(macd, 'signal_line') else None,
                "bb_upper": round(bb.upper_band, 4) if bb else None,
                "bb_lower": round(bb.lower_band, 4) if bb else None,
                "bb_middle": round(bb.middle_band, 4) if bb else None,
                "bb_percent_b": round(bb.percent_b, 3) if bb else None,
                "vwap": round(vwap.value, 4) if vwap else None,
                "vwap_pct": round(vwap.price_vs_vwap, 2) if vwap else None,
                "atr_pct": round(atr.atr_percent, 2) if atr else None,
                "ema_trend": round(ema.trend_strength, 2) if ema else None,
                "obv_divergence": obv.divergence if obv else None,
                "volume_ratio": round(volume.volume_ratio, 1) if volume else None,
                "dynamic_tp_pct": round(tp_pct, 2),
                "dynamic_sl_pct": round(stop_pct, 2),
                "tp_price_long": round(current_price * (1 + tp_pct / 100), 4),
                "sl_price_long": round(current_price * (1 - stop_pct / 100), 4),
                "tp_price_short": round(current_price * (1 - tp_pct / 100), 4),
                "sl_price_short": round(current_price * (1 + stop_pct / 100), 4),
                "min_confirmations": strategy.config.min_confirmations,
                "updated": datetime.now(timezone.utc).isoformat(),
            }
            self.indicator_snapshots[pair] = snapshot
        except Exception as e:
            self.logger.debug(f"Could not compute indicators for {pair}: {e}")

    def _rollover_fee_pct(self, position_data: dict, exit_candle_ts) -> float:
        """Margin rollover cost as a % of notional for a held position.

        Kraken charges ~0.02%/4h on OPEN MARGIN positions; in this bot only
        shorts are margin (longs are spot, no rollover). Duration is measured
        from candle timestamps so it is correct in replay (where wall-clock
        time barely advances) as well as live. Each *started* 4h period is
        charged, matching Kraken's rollover billing.
        """
        if self.margin_rollover_pct_4h <= 0:
            return 0.0
        if position_data.get('side') != 'short':
            return 0.0
        entry_ts = position_data.get('entry_candle_ts')
        if not entry_ts:
            return 0.0
        try:
            t0 = datetime.fromisoformat(entry_ts)
            t1 = (exit_candle_ts if isinstance(exit_candle_ts, datetime)
                  else datetime.fromisoformat(exit_candle_ts))
        except (TypeError, ValueError):
            return 0.0
        held_seconds = (t1 - t0).total_seconds()
        if held_seconds <= 0:
            return 0.0
        periods = math.ceil(held_seconds / (4 * 3600))
        return periods * self.margin_rollover_pct_4h

    def _close_position(self, pair: str, exit_reason: str, current_price: float,
                        strategy_instance: 'ScalpingStrategy',
                        exit_candle_ts=None) -> bool:
        """Execute the exit order and settle P&L/metrics for an open position.

        Returns True if the position was closed, False if the exit order
        failed (position stays open, retried next cycle)."""
        position_data = self.positions[pair]
        pos_side = position_data.get('side', 'long')

        # LIVE: deal with the resting server-side stop first. If it already
        # triggered, the position is closed on the exchange — settle from the
        # stop's actual fill instead of placing a redundant exit order.
        stop_status, stop_fill = self._resolve_server_stop(pair, position_data)
        if stop_status == 'filled':
            self.logger.warning(
                f"Server-side disaster stop already executed for {pair} — "
                f"settling from its fill"
            )
            exit_order = {
                'order_id': position_data.get('stop_order_id', ''),
                'fill_price': (stop_fill.get('price') or 0.0) or current_price,
                'fill_volume': stop_fill.get('vol_exec', position_data['size']),
                'fee': stop_fill.get('fee', 0.0),
                'status': 'closed',
            }
            exit_reason = f"{exit_reason} [server stop filled]"
        else:
            exit_order = self._execute_exit_order(pair, pos_side, position_data['size'])
            if exit_order is None:
                self.logger.error(
                    f"EXIT ORDER FAILED for {pair} — position still open, will retry next cycle",
                    side=pos_side,
                    reason=exit_reason,
                )
                # We cancelled the stop but the exit failed: the position is
                # naked until the retry. Re-arm the server-side stop.
                if stop_status == 'cancelled':
                    new_stop = self._place_server_stop(
                        pair, pos_side, position_data['size'],
                        position_data.get('entry_price', 0))
                    if new_stop:
                        position_data['stop_order_id'] = new_stop
                    else:
                        position_data.pop('stop_order_id', None)
                    self._save_state()
                return False

        # Use actual fill price if available, otherwise use current_price
        exit_price = exit_order.get('fill_price') or current_price

        # Calculate P&L from actual fill prices
        entry_price = position_data['entry_price']
        if entry_price <= 0:
            pnl_pct = 0.0
        elif pos_side == "long":
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        # Use actual fee from order if available, otherwise estimate
        actual_fee = exit_order.get('fee', 0.0)
        entry_fee = position_data.get('entry_fee', 0.0)
        if actual_fee > 0 or entry_fee > 0:
            total_fee_usd = actual_fee + entry_fee
            fee_pct_actual = (total_fee_usd / position_data['size_usd']) * 100
            net_pnl_pct = pnl_pct - fee_pct_actual
        else:
            fee_pct = strategy_instance.config.fee_percent * 2
            net_pnl_pct = pnl_pct - fee_pct

        # Margin rollover (shorts only) — the cost of holding a leveraged
        # position across 4h boundaries. Live pays this to Kraken directly;
        # paper/replay deduct it here so simulated P&L matches reality.
        rollover_pct = self._rollover_fee_pct(position_data, exit_candle_ts)
        if rollover_pct > 0:
            net_pnl_pct -= rollover_pct

        pnl_usd = position_data['size_usd'] * (net_pnl_pct / 100)

        # Settle shared state atomically — capital, metrics and the positions
        # dict are read by other pair threads for sizing/serialization.
        with self._state_lock:
            self.capital += pnl_usd
            if self.capital > self.metrics.get('peak_capital', 0):
                self.metrics['peak_capital'] = self.capital

            # Record trade in adaptive manager
            self.pair_manager.record_trade(
                pair=pair,
                entry_time=datetime.fromisoformat(position_data['entry_time']),
                exit_time=datetime.now(timezone.utc),
                pnl=pnl_usd,
                pnl_percent=net_pnl_pct,
                entry_price=entry_price,
                exit_price=exit_price,
                size_usd=position_data['size_usd'],
                side=pos_side,
            )

            # Update metrics
            self.metrics['total_trades'] += 1
            self.metrics['total_pnl'] += pnl_usd
            if pnl_usd > 0:
                self.metrics['wins'] += 1
            else:
                self.metrics['losses'] += 1

            del self.positions[pair]

        self.logger.info(
            f"CLOSED {pair}",
            side=pos_side,
            reason=exit_reason,
            exit_price=f"${exit_price:.2f}",
            entry_order=position_data.get('order_id', ''),
            exit_order=exit_order.get('order_id', ''),
            pnl_pct=f"{net_pnl_pct:.2f}%",
            pnl_usd=f"${pnl_usd:.2f}",
            capital=f"${self.capital:.2f}",
            regime=self._current_regime.value,
        )

        self._save_state()
        return True

    def _process_pair(self, pair: str) -> None:
        """Process a single trading pair."""
        # Check if pair is enabled by adaptive manager
        if not self.pair_manager.is_pair_enabled(pair):
            return

        market_data = self._get_market_data(pair)
        if not market_data:
            return

        # Disaster stop: hard intra-candle loss floor, evaluated EVERY poll
        # (before the one-decision-per-candle guard). Normal ATR stops stay
        # candle-anchored — tight intra-candle stops destroy the
        # mean-reversion edge (tested) — but forensics across 3 quarters
        # showed stops filling 2x-5x past the -5% threshold (tails to -28%)
        # because they wait for the next candle. This floor only fires far
        # beyond the normal stop, capping catastrophes without touching wick
        # noise. 0 disables.
        if self.disaster_stop_pct > 0 and pair in self.positions and market_data.ticker:
            d_data = self.positions[pair]
            d_entry = d_data.get('entry_price', 0)
            d_price = market_data.ticker.last
            if d_entry > 0:
                if d_data.get('side', 'long') == 'long':
                    d_pnl = (d_price - d_entry) / d_entry * 100
                else:
                    d_pnl = (d_entry - d_price) / d_entry * 100
                if d_pnl <= -self.disaster_stop_pct:
                    self.logger.warning(
                        f"DISASTER STOP {pair}",
                        pnl=f"{d_pnl:.2f}%",
                        floor=f"-{self.disaster_stop_pct:.1f}%",
                    )
                    self._close_position(
                        pair,
                        f"Disaster stop: {d_pnl:.2f}% breached -{self.disaster_stop_pct:.1f}% floor",
                        d_price,
                        self._get_strategy_for_pair(pair),
                        exit_candle_ts=(market_data.ticker.timestamp
                                        if market_data.ticker else None),
                    )
                    return

        # One-decision-per-candle guard. A 4h strategy has one new frame every
        # 4 hours; polling every 60s on the same candle risks rapid-fire trades
        # on brief ticker excursions to the candle's high/low. Skip if the
        # latest candle hasn't advanced since our last processing of this pair.
        if market_data.ohlc:
            latest_candle_ts = market_data.ohlc[-1].timestamp
            last_seen = self._last_processed_candle_ts.get(pair)
            if last_seen == latest_candle_ts:
                return
            self._last_processed_candle_ts[pair] = latest_candle_ts

        # Use real-time ticker price, not stale candle close!
        if market_data.ticker:
            current_price = market_data.ticker.last
        else:
            current_price = market_data.prices[-1]  # Fallback to candle close

        # Get strategy for this pair (may have per-pair optimized params)
        strategy = self._get_strategy_for_pair(pair)

        # Compute indicator snapshot for dashboard
        self._compute_indicator_snapshot(pair, strategy, market_data, current_price)

        # Check if we have a position
        if pair in self.positions:
            position_data = self.positions[pair]
            entry_price = position_data['entry_price']
            pos_side = position_data.get('side', 'long')

            if entry_price <= 0:
                pnl_pct = 0.0
            elif pos_side == "long":
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100

            # --- Trailing stop logic (gated by config) ---
            strategy_instance = self._get_strategy_for_pair(pair)
            trailing_stop_hit = False
            trailing_stop = position_data.get('trailing_stop', 0.0)
            best_price = position_data.get('best_price', entry_price)
            dynamic_tp = strategy_instance.config.take_profit_percent

            if self.trailing_stops_enabled:

                if pos_side == "long":
                    if current_price > best_price:
                        best_price = current_price
                    unrealized_pct = ((best_price - entry_price) / entry_price) * 100
                else:
                    if best_price == 0.0 or current_price < best_price:
                        best_price = current_price
                    unrealized_pct = ((entry_price - best_price) / entry_price) * 100

                tp_progress = unrealized_pct / dynamic_tp if dynamic_tp > 0 else 0

                if tp_progress >= self.trail_trigger_pct:
                    if pos_side == "long":
                        trail_price = entry_price * (1 + unrealized_pct * self.trail_lock_pct / 100)
                        trailing_stop = max(trailing_stop, trail_price)
                    else:
                        trail_price = entry_price * (1 - unrealized_pct * self.trail_lock_pct / 100)
                        trailing_stop = trail_price if trailing_stop == 0 else min(trailing_stop, trail_price)

                position_data['best_price'] = best_price
                position_data['trailing_stop'] = trailing_stop

                if trailing_stop > 0:
                    if pos_side == "long" and current_price <= trailing_stop:
                        trailing_stop_hit = True
                    elif pos_side == "short" and current_price >= trailing_stop:
                        trailing_stop_hit = True

            trail_info = f"trail=${trailing_stop:.2f}" if trailing_stop > 0 else "no trail"
            self.logger.info(
                f"Checking {pair}",
                side=pos_side,
                entry_price=f"${entry_price:.2f}",
                current_price=f"${current_price:.2f}",
                best_price=f"${best_price:.2f}",
                pnl_pct=f"{pnl_pct:.2f}%",
                tp_target=f"{dynamic_tp}%",
                trailing=trail_info,
            )

            position = Position(
                pair=pair,
                side=pos_side,
                entry_price=entry_price,
                current_price=current_price,
                size=position_data['size'],
                entry_time=datetime.fromisoformat(position_data['entry_time'])
            )

            # Trailing stop takes priority over strategy signals
            signal = strategy_instance.analyze(market_data, position)
            self.logger.debug(f"{pair} signal: {signal.signal_type.value} - {signal.reason}")

            if trailing_stop_hit:
                exit_reason = f"Trailing stop hit at ${trailing_stop:.2f}"
                self.logger.info(f"TRAILING STOP triggered for {pair}", trail_price=f"${trailing_stop:.2f}")
                should_exit = True
            elif pos_side == "long":
                should_exit = signal.signal_type.value in ("sell", "close_long")
                if should_exit:
                    exit_reason = signal.reason
            else:
                should_exit = signal.signal_type.value == "close_short"
                if should_exit:
                    exit_reason = signal.reason

            if should_exit:
                self._close_position(
                    pair, exit_reason, current_price, strategy_instance,
                    exit_candle_ts=(market_data.ticker.timestamp
                                    if market_data.ticker else None),
                )

        else:
            # Look for entry
            strategy = self._get_strategy_for_pair(pair)
            signal = strategy.analyze(market_data, None)

            if signal.signal_type.value in ("buy", "sell_short"):
                # Regime-flip cooldown: forensics show entries chased into a
                # fresh flip are net losers (-$2,010 across 3 quarters).
                if (self.flip_entry_cooldown > 0
                        and self._candles_since_flip < self.flip_entry_cooldown):
                    self.logger.debug(
                        f"Skipping {pair} entry — regime flipped "
                        f"{self._candles_since_flip} candles ago "
                        f"(cooldown {self.flip_entry_cooldown})"
                    )
                    return

                # Dead-zone: sit out new entries in configured regimes.
                regime_name = self._current_regime.value
                if regime_name in self._block_entry_regimes:
                    self.logger.debug(
                        f"Skipping {pair} entry — dead-zone ({regime_name} regime)"
                    )
                    return
                if (signal.signal_type.value == "sell_short"
                        and regime_name in self._block_short_regimes):
                    self.logger.debug(
                        f"Skipping {pair} short — shorts blocked in {regime_name}"
                    )
                    return

                # Correlation cluster check
                corr_block = self._correlation_blocked(pair)
                if corr_block:
                    self.logger.debug(f"Skipping {pair} entry — {corr_block}")
                    return

                # Volume regime check
                vol_block = self._volume_filter_blocked(market_data)
                if vol_block:
                    self.logger.debug(f"Skipping {pair} entry — {vol_block}")
                    return

                pos_side = "long" if signal.signal_type.value == "buy" else "short"

                # Scale position: regime adjustment * signal strength
                regime_scale = self._regime_adjustments.get(
                    'position_scale_multiplier', 1.0
                )
                strength_scale = 0.5 + 0.5 * signal.strength
                total_scale = regime_scale * strength_scale

                if current_price <= 0:
                    self.logger.warning(f"Invalid price {current_price} for {pair}, skipping entry")
                    return

                # Reserve capital atomically: deployed (open positions) +
                # already-reserved (other pairs mid-fill) is subtracted so two
                # concurrent entries can't allocate the same dollars. The slow
                # fill-wait happens AFTER releasing the lock.
                with self._state_lock:
                    deployed_capital = sum(p['size_usd'] for p in self.positions.values())
                    available_capital = self.capital - deployed_capital - self._reserved_usd
                    size_usd = self.capital * (self.position_size_pct / 100) * total_scale
                    if size_usd > available_capital:
                        self.logger.warning(
                            f"Skipping {pair} entry - insufficient capital",
                            required=f"${size_usd:.2f}",
                            available=f"${available_capital:.2f}"
                        )
                        return
                    self._reserved_usd += size_usd
                size = size_usd / current_price

                # Execute order on Kraken (slow maker fill-wait runs outside the
                # lock so other pairs keep processing).
                order_result = self._execute_entry_order(pair, pos_side, size, current_price)

                with self._state_lock:
                    self._reserved_usd -= size_usd  # release reservation
                    if order_result is None:
                        self.logger.error(f"Skipping {pair} entry — order execution failed")
                        return
                    fill_price = order_result.get('fill_price', current_price)
                    self.positions[pair] = {
                        'entry_price': fill_price,
                        'side': pos_side,
                        'size': order_result.get('fill_volume', size),
                        'size_usd': size_usd,
                        'entry_time': datetime.now(timezone.utc).isoformat(),
                        # Candle timestamp at entry — used to measure margin hold
                        # duration for rollover (works in replay where wall-clock
                        # entry_time barely advances; live candle ts is real too).
                        'entry_candle_ts': (
                            market_data.ticker.timestamp.isoformat()
                            if market_data.ticker and market_data.ticker.timestamp
                            else datetime.now(timezone.utc).isoformat()
                        ),
                        'reason': signal.reason,
                        'regime': self._current_regime.value,
                        'best_price': fill_price,
                        'trailing_stop': 0.0,
                        'order_id': order_result.get('order_id', ''),
                        'entry_fee': order_result.get('fee', 0.0),
                    }

                # LIVE: rest a disaster stop on the exchange so the position
                # is protected even if this process dies. Slow API call kept
                # outside the lock; the resulting id is written back under it.
                stop_id = self._place_server_stop(
                    pair, pos_side, order_result.get('fill_volume', size), fill_price)
                if stop_id:
                    with self._state_lock:
                        if pair in self.positions:
                            self.positions[pair]['stop_order_id'] = stop_id

                self.logger.info(
                    f"OPENED {pair}",
                    side=pos_side,
                    reason=signal.reason,
                    price=f"${fill_price:.2f}",
                    size_usd=f"${size_usd:.2f}",
                    order_id=order_result.get('order_id', ''),
                    scale=f"{total_scale:.2f}x (strength={signal.strength:.2f} regime={regime_scale:.2f})",
                    regime=self._current_regime.value,
                    available_after=f"${available_capital - size_usd:.2f}"
                )

                self._save_state()

    def _process_pair_safe(self, pair: str) -> None:
        """Process one pair, swallowing per-pair errors so one bad pair can't
        abort the cycle (or a sibling thread)."""
        try:
            self._process_pair(pair)
        except Exception as pair_err:
            self.logger.error(f"Error processing {pair}: {pair_err}", pair=pair)

    def _process_all_pairs(self) -> None:
        """Process every pair for this cycle, sequentially or concurrently.

        Concurrency is LIVE-only and opt-in (live.concurrent_pairs > 1): the
        maker fill-wait is the only slow step, so running pairs in a bounded
        thread pool means the loop takes max(fill_waits) instead of
        sum(fill_waits). Paper/replay stay sequential — instant fills give no
        benefit and sequential order preserves deterministic replay parity.
        """
        pairs = list(self.pairs)
        if self.paper_trading or self.live_concurrent_pairs <= 1:
            for pair in pairs:
                self._process_pair_safe(pair)
                self._sleep(1)  # gentle inter-pair spacing
            return
        # Concurrent: the Kraken client's rate limiter serializes the actual
        # HTTP, but fill-waits (sleeps) overlap, which is the whole point.
        with ThreadPoolExecutor(
            max_workers=self.live_concurrent_pairs,
            thread_name_prefix="pair",
        ) as ex:
            list(ex.map(self._process_pair_safe, pairs))

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

                self._process_all_pairs()

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

                # LIVE: periodic read-only sanity check that internal state
                # still matches the exchange.
                if not self.paper_trading:
                    self._cycles_since_drift_check += 1
                    if self._cycles_since_drift_check >= self.live_drift_check_cycles:
                        self._cycles_since_drift_check = 0
                        try:
                            self._check_live_drift()
                        except Exception as drift_err:
                            self.logger.error(f"Drift check failed: {drift_err}")

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
  Adaptive pair management with auto-disable/re-enable
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

    # Start dashboard in background thread using waitress (thread-pooled WSGI server)
    from src.scalping_dashboard.app import create_app
    from waitress import serve as waitress_serve
    dashboard_app = create_app(args.data_dir)
    dashboard_thread = threading.Thread(
        target=waitress_serve,
        args=(dashboard_app,),
        kwargs={'host': '0.0.0.0', 'port': args.dashboard_port, 'threads': 4,
                'channel_timeout': 30, 'connection_limit': 100},
        daemon=True,
    )
    dashboard_thread.start()
    print(f"Dashboard running on port {args.dashboard_port}")

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
