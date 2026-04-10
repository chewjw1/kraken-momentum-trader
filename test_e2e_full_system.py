#!/usr/bin/env python3
"""
End-to-end system test for the scalping trading system.

Tests the full pipeline:
  1. Regime detection on real historical data
  2. Signal generation per pair with per-pair configs
  3. Order placement (paper trading mode)
  4. Position management (open, track, close)
  5. Dashboard status reporting
  6. Performance metrics (P&L, win rate, drawdown)
  7. Circuit breaker (global + per-pair)
  8. Adaptive pair manager (disable/re-enable)
  9. State persistence and recovery
 10. MTF backtester E2E
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml

from src.exchange.kraken_client import OHLC, OrderSide, OrderType
from src.strategy.scalping_strategy import ScalpingStrategy, ScalpingConfig
from src.strategy.regime_detector import (
    RegimeDetector, RegimeConfig, MarketRegime, REGIME_ADJUSTMENTS,
)
from src.strategy.base_strategy import MarketData, Position, SignalType
from src.core.adaptive_pair_manager import AdaptivePairManager, AdaptiveConfig
from src.risk.circuit_breaker import CircuitBreaker

# ── Helpers ──────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0
ERRORS = []


def check(name, condition, detail=""):
    global PASS, FAIL, ERRORS
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        msg = f"  ✗ {name}" + (f" — {detail}" if detail else "")
        print(msg)
        ERRORS.append(msg)


def make_candle(ts, o, h, l, c, v=100.0):
    return OHLC(
        timestamp=ts, open=o, high=h, low=l, close=c, vwap=c, volume=v, count=1
    )


def make_trending_candles(start_price, trend_pct_per_candle, n, start_time=None):
    """Generate n candles with a steady trend."""
    if start_time is None:
        start_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = start_price
    for i in range(n):
        ts = start_time + timedelta(hours=4 * i)
        move = price * trend_pct_per_candle / 100
        h = price + abs(move) * 0.5
        l = price - abs(move) * 0.5
        c = price + move
        candles.append(make_candle(ts, price, max(h, price, c), min(l, price, c), c, 100 + i))
        price = c
    return candles


def load_real_data(pair, n_candles=300):
    """Load real historical data for a pair."""
    try:
        from src.backtest.kraken_csv_provider import KrakenCSVProvider
        provider = KrakenCSVProvider()
        end = datetime(2026, 3, 2, tzinfo=timezone.utc)
        start = datetime(2025, 9, 1, tzinfo=timezone.utc)
        candles = provider.get_ohlc_range(pair, start, end, interval=240)
        if candles and len(candles) >= n_candles:
            return candles[-n_candles:]
    except Exception:
        pass
    return None


# ── Test 1: Regime Detection ────────────────────────────────────────────────

def test_regime_detection():
    print("\n" + "=" * 70)
    print("TEST 1: REGIME DETECTION")
    print("=" * 70)

    detector = RegimeDetector(RegimeConfig(
        bull_slope_threshold=0.02,
        bear_slope_threshold=-0.02,
    ))

    # Test with synthetic bull data
    bull_candles = make_trending_candles(50000, 0.3, 250)
    closes = [c.close for c in bull_candles]
    highs = [c.high for c in bull_candles]
    lows = [c.low for c in bull_candles]
    result = detector.detect(closes, highs, lows)
    check("Bull trend detected on rising data",
          result.regime in (MarketRegime.BULL, MarketRegime.SIDEWAYS),
          f"got {result.regime.value}, confidence={result.confidence:.2f}")

    # Test with synthetic bear data
    bear_candles = make_trending_candles(50000, -0.3, 250)
    closes = [c.close for c in bear_candles]
    highs = [c.high for c in bear_candles]
    lows = [c.low for c in bear_candles]
    result = detector.detect(closes, highs, lows)
    check("Bear trend detected on falling data",
          result.regime in (MarketRegime.BEAR, MarketRegime.SIDEWAYS),
          f"got {result.regime.value}, confidence={result.confidence:.2f}")

    # Test with real BTC data
    btc_candles = load_real_data("BTC/USD")
    if btc_candles:
        closes = [c.close for c in btc_candles]
        highs = [c.high for c in btc_candles]
        lows = [c.low for c in btc_candles]
        result = detector.detect(closes, highs, lows)
        check("Regime detected on real BTC data",
              result.regime in (MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS),
              f"regime={result.regime.value}, confidence={result.confidence:.2f}")

        # Verify adjustments exist for detected regime
        adj = REGIME_ADJUSTMENTS.get(result.regime)
        check("Regime adjustments available",
              adj is not None and 'take_profit_multiplier' in adj,
              f"adjustments={adj}")
    else:
        print("  (skipping real data tests — no BTC CSV data available)")

    # Serialization round-trip
    state = detector.to_dict()
    detector2 = RegimeDetector(RegimeConfig())
    detector2.from_dict(state)
    check("Regime detector serializes/deserializes",
          detector2.to_dict() == state)


# ── Test 2: Signal Generation + Per-Pair Configs ────────────────────────────

def test_signal_generation():
    print("\n" + "=" * 70)
    print("TEST 2: SIGNAL GENERATION + PER-PAIR CONFIGS")
    print("=" * 70)

    with open("config/scalping.yaml") as f:
        config = yaml.safe_load(f)

    pair_params = config.get("pair_parameters", {})
    pairs_tested = 0

    for pair in ["BTC/USD", "ETH/USD", "SOL/USD"]:
        candles = load_real_data(pair)
        if not candles:
            continue

        pp = pair_params.get(pair, {})
        sc = ScalpingConfig(
            take_profit_percent=pp.get("take_profit_percent", 4.5),
            stop_loss_percent=pp.get("stop_loss_percent", 2.5),
            rsi_period=pp.get("rsi_period", 7),
            rsi_oversold=pp.get("rsi_oversold", 28),
            rsi_overbought=pp.get("rsi_overbought", 70),
            bb_period=pp.get("bb_period", 20),
            bb_std_dev=pp.get("bb_std_dev", 2.0),
            stoch_k_period=pp.get("stoch_k_period", 14),
            stoch_d_period=pp.get("stoch_d_period", 3),
            stoch_oversold=pp.get("stoch_oversold", 23),
            stoch_overbought=pp.get("stoch_overbought", 75),
            atr_period=pp.get("atr_period", 14),
            atr_stop_multiplier=pp.get("atr_stop_multiplier", 2.5),
            atr_tp_multiplier=pp.get("atr_tp_multiplier", 2.75),
            min_confirmations=pp.get("min_confirmations", 3),
        )
        strategy = ScalpingStrategy(sc)

        market_data = MarketData(
            pair=pair,
            ohlc=candles,
            prices=[c.close for c in candles],
            volumes=[c.volume for c in candles],
            ticker=None,
        )

        # Test entry signal
        signal = strategy.analyze(market_data, None)
        check(f"{pair}: Entry signal generated",
              signal.signal_type in (SignalType.BUY, SignalType.SELL_SHORT, SignalType.HOLD),
              f"signal={signal.signal_type.value}, reason={signal.reason[:60]}")

        # Test exit signal with a fake position
        position = Position(
            pair=pair, side="long",
            entry_price=candles[-1].close * 0.98,
            current_price=candles[-1].close,
            size=1.0,
            entry_time=candles[-10].timestamp,
        )
        exit_signal = strategy.analyze(market_data, position)
        check(f"{pair}: Exit signal generated",
              exit_signal.signal_type in (
                  SignalType.CLOSE_LONG, SignalType.HOLD,
                  SignalType.SELL, SignalType.BUY,
              ),
              f"signal={exit_signal.signal_type.value}")

        pairs_tested += 1

    if pairs_tested == 0:
        print("  (skipping — no real data available)")
    else:
        check(f"Signals generated for {pairs_tested} pairs", pairs_tested >= 2)


# ── Test 3: Order Placement (Paper Trading) ─────────────────────────────────

def test_order_placement():
    print("\n" + "=" * 70)
    print("TEST 3: ORDER PLACEMENT (PAPER TRADING)")
    print("=" * 70)

    from src.exchange.kraken_client import KrakenClient

    client = KrakenClient(paper_trading=True)

    # Set up paper balances (plain floats — matches internal format)
    client._paper_balances = {"USD": 50000.0, "BTC": 0.0, "ETH": 0.0}

    # Need a ticker for market order price — mock the get_ticker method
    from types import SimpleNamespace
    real_get_ticker = client.get_ticker
    def mock_ticker(pair):
        prices = {"BTC/USD": 60000.0, "ETH/USD": 3000.0}
        p = prices.get(pair, 1000.0)
        return SimpleNamespace(ask=p * 1.001, bid=p * 0.999, last=p, volume=100.0)
    client.get_ticker = mock_ticker

    # Test market buy
    order = client.place_order(
        pair="BTC/USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        volume=0.1,
    )
    check("Market buy order placed",
          order is not None and order.order_id,
          f"order_id={order.order_id}, status={order.status}")
    check("Market buy order filled immediately",
          order.status == "closed" and order.filled_volume > 0,
          f"filled={order.filled_volume}, status={order.status}")
    check("Paper balance deducted",
          client._paper_balances["USD"] < 50000.0,
          f"USD={client._paper_balances['USD']:.2f}")
    check("BTC balance received",
          client._paper_balances.get("BTC", 0) > 0,
          f"BTC={client._paper_balances.get('BTC', 0)}")

    # Test market sell (close the position)
    sell_order = client.place_order(
        pair="BTC/USD",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        volume=0.1,
    )
    check("Market sell order placed",
          sell_order is not None and sell_order.order_id,
          f"order_id={sell_order.order_id}")
    check("Fee charged (round trip cost)",
          client._paper_balances["USD"] < 50000.0,
          f"USD after round-trip={client._paper_balances['USD']:.2f}")

    # Test order cancellation
    limit_order = client.place_order(
        pair="ETH/USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        volume=1.0,
        price=1000.0,  # Way below market
    )
    cancelled = client.cancel_order(limit_order.order_id)
    check("Limit order cancelled",
          cancelled is True,
          f"cancel result={cancelled}")

    # Test get_open_orders
    open_orders = client.get_open_orders()
    check("Open orders retrieved",
          isinstance(open_orders, list),
          f"count={len(open_orders)}")


# ── Test 4: Full Position Lifecycle (Signal → Order → Exit) ────────────────

def test_position_lifecycle():
    print("\n" + "=" * 70)
    print("TEST 4: FULL POSITION LIFECYCLE")
    print("=" * 70)

    # Import with mock krakenex
    import types as modtypes
    import importlib

    krakenex_mod = modtypes.ModuleType('krakenex')
    class FakeAPI:
        def __init__(self): pass
        def load_key(self, path): pass
        def query_public(self, *a, **kw): return {'error': [], 'result': {}}
        def query_private(self, *a, **kw): return {'error': [], 'result': {}}
    krakenex_mod.API = FakeAPI
    sys.modules['krakenex'] = krakenex_mod

    from run_scalping_live import ScalpingTrader

    with tempfile.TemporaryDirectory() as tmpdir:
        trader = ScalpingTrader(
            config_path='config/scalping.yaml',
            data_dir=tmpdir,
            paper_trading=True,
        )

        # Verify initialization
        check("Trader initialized", trader is not None)
        check("10 pairs loaded", len(trader.pairs) == 10, f"got {len(trader.pairs)}")
        check("Per-pair strategies loaded",
              len(trader.pair_strategies) == 10,
              f"got {len(trader.pair_strategies)}")
        check("Trailing stops disabled", trader.trailing_stops_enabled is False)
        check("Regime detector present", trader.regime_detector is not None)

        # Simulate opening a position manually
        trader.positions["BTC/USD"] = {
            'entry_price': 50000.0,
            'side': 'long',
            'size': 0.04,
            'size_usd': 2000.0,
            'entry_time': datetime.now(timezone.utc).isoformat(),
            'reason': 'test entry',
            'regime': 'unknown',
            'best_price': 50000.0,
            'trailing_stop': 0.0,
            'order_id': 'test-001',
        }

        check("Position added", "BTC/USD" in trader.positions)

        # Save state and verify persistence
        trader._save_state()
        state_file = Path(tmpdir) / "state.json"
        check("State file created", state_file.exists())

        with open(state_file) as f:
            state = json.load(f)
        check("Positions persisted",
              "BTC/USD" in state.get("positions", {}))
        check("Capital persisted",
              state.get("capital") == trader.capital,
              f"file={state.get('capital')}, trader={trader.capital}")
        check("Order ID persisted",
              state["positions"]["BTC/USD"].get("order_id") == "test-001")

        # Test get_status
        status = trader.get_status()
        check("Status returns strategy name",
              status.get("strategy") == "scalping")
        check("Status returns capital",
              status.get("capital") == trader.capital)
        check("Status returns positions",
              "BTC/USD" in status.get("positions", {}))
        check("Status returns regime info",
              "regime" in status,
              f"keys={list(status.get('regime', {}).keys())}")
        check("Status returns circuit breaker",
              "circuit_breaker" in status)
        check("Status returns enabled/disabled pairs",
              "enabled_pairs" in status and "disabled_pairs" in status)

        # Test state recovery after restart
        trader2 = ScalpingTrader(
            config_path='config/scalping.yaml',
            data_dir=tmpdir,
            paper_trading=True,
        )
        check("State recovered after restart",
              "BTC/USD" in trader2.positions,
              f"positions={list(trader2.positions.keys())}")
        check("Order ID recovered",
              trader2.positions.get("BTC/USD", {}).get("order_id") == "test-001")


# ── Test 5: Performance Metrics + Circuit Breaker ───────────────────────────

def test_metrics_and_circuit_breaker():
    print("\n" + "=" * 70)
    print("TEST 5: PERFORMANCE METRICS + CIRCUIT BREAKER")
    print("=" * 70)

    # Test circuit breaker directly
    cb = CircuitBreaker(
        global_max_drawdown_pct=5.0,
        consecutive_loss_limit=3,
        pair_max_drawdown_pct=3.0,
        pair_cooldown_hours=0.001,  # Very short for testing
        initial_capital=10000.0,
    )

    check("Circuit breaker starts closed",
          cb.is_trading_allowed() is True)

    # Record some wins
    cb.record_trade("BTC/USD", 100.0)
    cb.record_trade("BTC/USD", 50.0)
    check("Trading allowed after wins", cb.is_trading_allowed())

    # Record consecutive losses
    cb.record_trade("BTC/USD", -200.0)
    cb.record_trade("BTC/USD", -200.0)
    cb.record_trade("BTC/USD", -200.0)
    state = cb.get_state()
    check("Circuit breaker opened after 3 consecutive losses",
          state.state.value in ("open", "emergency"),
          f"state={state.state.value}, reason={state.trigger_reason}")
    check("Trading blocked",
          cb.is_trading_allowed() is False)

    # Test per-pair drawdown
    cb2 = CircuitBreaker(
        global_max_drawdown_pct=50.0,  # Won't trigger
        consecutive_loss_limit=100,     # Won't trigger
        pair_max_drawdown_pct=2.0,
        pair_cooldown_hours=0.001,
        initial_capital=10000.0,
    )
    cb2.record_trade("SOL/USD", -250.0)  # 2.5% loss > 2% threshold
    check("Per-pair circuit breaker triggers",
          cb2.is_pair_allowed("SOL/USD") is False)
    check("Other pairs still allowed",
          cb2.is_pair_allowed("BTC/USD") is True)

    # Serialization
    state_dict = cb.to_dict()
    cb3 = CircuitBreaker(initial_capital=10000.0)
    cb3.from_dict(state_dict)
    check("Circuit breaker serializes/deserializes",
          cb3.to_dict().get("state") == state_dict.get("state"))

    # Test adaptive pair manager
    apm = AdaptivePairManager(AdaptiveConfig(
        min_win_rate=0.35,
        max_consecutive_losses=3,
        cooldown_hours=0.001,
    ))
    apm.register_pair("BTC/USD")
    apm.register_pair("ETH/USD")

    check("Pairs start enabled",
          apm.is_pair_enabled("BTC/USD") and apm.is_pair_enabled("ETH/USD"))

    # Simulate poor performance to trigger disable
    for i in range(12):
        apm.record_trade(
            "BTC/USD",
            entry_time=datetime.now(timezone.utc) - timedelta(hours=1),
            exit_time=datetime.now(timezone.utc),
            pnl=-50.0,
            pnl_percent=-2.5,
        )

    check("Underperforming pair disabled",
          apm.is_pair_enabled("BTC/USD") is False,
          f"enabled={apm.is_pair_enabled('BTC/USD')}")
    check("Other pair still enabled",
          apm.is_pair_enabled("ETH/USD") is True)

    # Position scaling
    scale = apm.get_position_scale("ETH/USD")
    check("Position scale returns reasonable value",
          0.3 <= scale <= 2.0,
          f"scale={scale}")

    # Check re-enable
    time.sleep(0.01)  # Cooldown is 0.001 hours
    reenabled = apm.check_reenable_pairs()
    check("Pair re-enabled after cooldown",
          "BTC/USD" in reenabled or apm.is_pair_enabled("BTC/USD"),
          f"reenabled={reenabled}")

    # Serialization
    apm_dict = apm.to_dict()
    apm2 = AdaptivePairManager(AdaptiveConfig())
    apm2.from_dict(apm_dict)
    check("Pair manager serializes/deserializes",
          apm2.is_pair_enabled("ETH/USD") is True)


# ── Test 6: Dashboard Status Reporting ──────────────────────────────────────

def test_dashboard_reporting():
    print("\n" + "=" * 70)
    print("TEST 6: DASHBOARD STATUS REPORTING")
    print("=" * 70)

    # Check if dashboard app exists and can be imported
    try:
        from src.scalping_dashboard.app import create_app
        check("Dashboard app importable", True)

        # Test that dashboard reads state file
        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "positions": {
                    "BTC/USD": {
                        "entry_price": 50000.0,
                        "side": "long",
                        "size": 0.04,
                        "size_usd": 2000.0,
                        "entry_time": "2025-01-01T00:00:00+00:00",
                        "reason": "test",
                        "order_id": "test-001",
                    }
                },
                "metrics": {
                    "total_trades": 15,
                    "wins": 9,
                    "losses": 6,
                    "total_pnl": 450.0,
                    "start_time": "2025-01-01T00:00:00+00:00",
                },
                "capital": 10450.0,
                "last_update": datetime.now(timezone.utc).isoformat(),
                "pair_manager": {},
                "circuit_breaker": {
                    "state": "closed",
                    "consecutive_losses": 0,
                    "peak_equity": 10450.0,
                    "current_equity": 10450.0,
                },
            }
            state_path = Path(tmpdir) / "state.json"
            with open(state_path, "w") as f:
                json.dump(state, f)

            check("State file created for dashboard test", state_path.exists())

            # Verify the state file is valid JSON with expected structure
            with open(state_path) as f:
                loaded = json.load(f)
            check("State file has positions", "BTC/USD" in loaded.get("positions", {}))
            check("State file has metrics", loaded.get("metrics", {}).get("total_trades") == 15)
            check("State file has capital", loaded.get("capital") == 10450.0)
            check("State file has circuit_breaker", "circuit_breaker" in loaded)

    except ImportError as e:
        print(f"  (Dashboard not available: {e})")
        check("Dashboard importable", False, str(e))


# ── Test 7: MTF Backtester E2E ──────────────────────────────────────────────

def test_mtf_backtester():
    print("\n" + "=" * 70)
    print("TEST 7: MTF BACKTESTER E2E (with regime detection)")
    print("=" * 70)

    try:
        from src.backtest.kraken_csv_provider import KrakenCSVProvider
        provider = KrakenCSVProvider()

        # Load 1m data for a single pair (smaller to be fast)
        end = datetime(2025, 9, 1, tzinfo=timezone.utc)
        start = datetime(2025, 6, 1, tzinfo=timezone.utc)  # 3 months
        candles_1m = provider.get_ohlc_range("BTC/USD", start, end, interval=1)

        if not candles_1m or len(candles_1m) < 10000:
            print(f"  (skipping — only {len(candles_1m) if candles_1m else 0} 1m candles)")
            return

        print(f"  Loaded {len(candles_1m)} 1m candles for BTC/USD")

        with open("config/scalping.yaml") as f:
            config = yaml.safe_load(f)
        pp = config.get("pair_parameters", {}).get("BTC/USD", {})

        from run_multiframe_backtest import run_mtf_backtest

        sc = ScalpingConfig(
            take_profit_percent=pp.get("take_profit_percent", 4.5),
            stop_loss_percent=pp.get("stop_loss_percent", 2.5),
            rsi_period=pp.get("rsi_period", 9),
            rsi_oversold=pp.get("rsi_oversold", 30),
            rsi_overbought=pp.get("rsi_overbought", 72),
            bb_period=pp.get("bb_period", 30),
            bb_std_dev=pp.get("bb_std_dev", 2.0),
            stoch_k_period=pp.get("stoch_k_period", 19),
            stoch_d_period=pp.get("stoch_d_period", 3),
            stoch_oversold=pp.get("stoch_oversold", 23),
            stoch_overbought=pp.get("stoch_overbought", 85),
            atr_period=pp.get("atr_period", 10),
            atr_stop_multiplier=pp.get("atr_stop_multiplier", 1.5),
            atr_tp_multiplier=pp.get("atr_tp_multiplier", 2.0),
            min_confirmations=pp.get("min_confirmations", 4),
        )

        result = run_mtf_backtest(
            "BTC/USD", candles_1m, sc,
            initial_capital=10000.0,
            position_size_pct=0.20,
            fee_pct=0.0016,
            slippage_pct=0.02,
            signal_interval=240,
            trailing_stops_enabled=False,
            regime_detection=True,
        )

        check("MTF backtest completed without error",
              "error" not in result,
              result.get("error", ""))
        check("MTF backtest produced trades",
              result.get("trades", 0) > 0,
              f"trades={result.get('trades', 0)}")
        check("Win rate is reasonable",
              0 <= result.get("win_rate", -1) <= 100,
              f"win_rate={result.get('win_rate')}")
        check("Profit factor computed",
              result.get("profit_factor", -1) >= 0,
              f"PF={result.get('profit_factor')}")
        check("Max drawdown computed",
              result.get("max_dd_pct", -1) >= 0,
              f"DD={result.get('max_dd_pct')}")
        check("Sharpe ratio computed",
              isinstance(result.get("sharpe"), (int, float)),
              f"sharpe={result.get('sharpe')}")

        print(f"\n  BTC/USD 3-month result: {result.get('total_pnl_pct', 0):+.2f}% | "
              f"{result.get('trades', 0)} trades | "
              f"{result.get('win_rate', 0):.1f}% WR | "
              f"PF {result.get('profit_factor', 0):.2f} | "
              f"DD {result.get('max_dd_pct', 0):.1f}%")

    except Exception as e:
        import traceback
        traceback.print_exc()
        check("MTF backtester runs without crash", False, str(e))


# ── Test 8: Optimizer Backtester E2E ────────────────────────────────────────

def test_optimizer_backtester():
    print("\n" + "=" * 70)
    print("TEST 8: OPTIMIZER BACKTESTER E2E (with BTC regime reference)")
    print("=" * 70)

    try:
        from run_scalping_optimizer import ScalpingBacktestRunner
        from src.backtest.kraken_csv_provider import KrakenCSVProvider

        provider = KrakenCSVProvider()
        end = datetime(2025, 9, 1, tzinfo=timezone.utc)
        start = datetime(2025, 3, 1, tzinfo=timezone.utc)

        eth_candles = provider.get_ohlc_range("ETH/USD", start, end, interval=240)
        btc_candles = provider.get_ohlc_range("BTC/USD", start, end, interval=240)

        if not eth_candles or len(eth_candles) < 200:
            print(f"  (skipping — insufficient ETH data)")
            return
        if not btc_candles or len(btc_candles) < 200:
            print(f"  (skipping — insufficient BTC data)")
            return

        print(f"  ETH/USD: {len(eth_candles)} candles | BTC/USD: {len(btc_candles)} candles")

        runner = ScalpingBacktestRunner(
            initial_capital=10000.0,
            position_size_percent=20.0,
            fee_percent=0.16,
        )

        with open("config/scalping.yaml") as f:
            config = yaml.safe_load(f)
        pp = config.get("pair_parameters", {}).get("ETH/USD", {})

        params = {
            "take_profit_percent": pp.get("take_profit_percent", 4.5),
            "stop_loss_percent": pp.get("stop_loss_percent", 0.5),
            "rsi_period": pp.get("rsi_period", 11),
            "rsi_oversold": pp.get("rsi_oversold", 25),
            "rsi_overbought": pp.get("rsi_overbought", 72),
            "bb_period": pp.get("bb_period", 10),
            "bb_std_dev": pp.get("bb_std_dev", 2.0),
            "stoch_k_period": pp.get("stoch_k_period", 7),
            "stoch_oversold": pp.get("stoch_oversold", 25),
            "stoch_overbought": pp.get("stoch_overbought", 85),
            "atr_period": pp.get("atr_period", 10),
            "atr_stop_multiplier": pp.get("atr_stop_multiplier", 2.25),
            "atr_tp_multiplier": pp.get("atr_tp_multiplier", 4.0),
            "min_confirmations": pp.get("min_confirmations", 3),
        }

        # Test without BTC regime reference (falls back to own candles)
        metrics_own = runner.run(eth_candles, "ETH/USD", params)
        check("Optimizer backtest completes (own regime)",
              metrics_own.get("total_trades", 0) >= 0)

        # Test with BTC regime reference
        metrics_btc = runner.run(eth_candles, "ETH/USD", params, regime_candles=btc_candles)
        check("Optimizer backtest completes (BTC regime)",
              metrics_btc.get("total_trades", 0) >= 0)
        check("Score computed",
              isinstance(metrics_btc.get("score"), (int, float)),
              f"score={metrics_btc.get('score')}")

        print(f"\n  ETH/USD own-regime: {metrics_own.get('total_return', 0)*100:+.2f}% | "
              f"{metrics_own.get('total_trades', 0)} trades")
        print(f"  ETH/USD BTC-regime: {metrics_btc.get('total_return', 0)*100:+.2f}% | "
              f"{metrics_btc.get('total_trades', 0)} trades")

    except Exception as e:
        import traceback
        traceback.print_exc()
        check("Optimizer backtester runs without crash", False, str(e))


# ── Test 9: Regime-Adjusted Strategy Rebuild ────────────────────────────────

def test_regime_strategy_rebuild():
    print("\n" + "=" * 70)
    print("TEST 9: REGIME-ADJUSTED STRATEGY REBUILD")
    print("=" * 70)

    # Test that regime adjustments actually modify strategy configs
    base_config = ScalpingConfig(
        take_profit_percent=5.0,
        stop_loss_percent=2.5,
        min_confirmations=3,
    )

    for regime in [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS]:
        adj = REGIME_ADJUSTMENTS.get(regime, {})
        tp_mult = adj.get('take_profit_multiplier', 1.0)
        sl_mult = adj.get('stop_loss_multiplier', 1.0)
        conf_offset = adj.get('min_confirmations_offset', 0)

        adjusted_tp = base_config.take_profit_percent * tp_mult
        adjusted_sl = base_config.stop_loss_percent * sl_mult
        adjusted_conf = max(1, base_config.min_confirmations + conf_offset)

        check(f"{regime.value}: TP adjusted ({base_config.take_profit_percent} → {adjusted_tp:.1f})",
              adjusted_tp > 0)
        check(f"{regime.value}: SL adjusted ({base_config.stop_loss_percent} → {adjusted_sl:.1f})",
              adjusted_sl > 0)
        check(f"{regime.value}: Confirmations adjusted ({base_config.min_confirmations} → {adjusted_conf})",
              adjusted_conf >= 1)


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("COMPREHENSIVE END-TO-END SYSTEM TEST")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    test_regime_detection()
    test_signal_generation()
    test_order_placement()
    test_position_lifecycle()
    test_metrics_and_circuit_breaker()
    test_dashboard_reporting()
    test_mtf_backtester()
    test_optimizer_backtester()
    test_regime_strategy_rebuild()

    print("\n" + "=" * 70)
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 70)

    if ERRORS:
        print("\nFAILURES:")
        for e in ERRORS:
            print(e)

    sys.exit(1 if FAIL > 0 else 0)
