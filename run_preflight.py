#!/usr/bin/env python3
"""
PREFLIGHT GATE — run before every deploy. Exercises the REAL stack end to end.

Why this exists: every deploy so far broke something that replay/backtests and
the existing smoke harnesses could not see, because those harnesses replace the
exact components that failed in production:

    bug shipped                          why no harness caught it
    -----------------------------------  ----------------------------------------
    in-progress candle blocked entries   mock client served committed candles
    12h aggregation phase shift          mock client bypassed real get_ohlc
    regime cadence 160x too fast         replay counted candles, live counted loops
    paper shorts rejected                harnesses OVERRIDE _place_paper_order
    dashboard crash on fresh start       no harness ever rendered the template
    '--paper' flag in deploy docs        no harness ever parsed the real CLI

The fix is architectural: this gate mocks ONLY `requests.Session` (the network
boundary). Everything else — KrakenClient.__init__, request signing, response
parsing, the paper order engine, ScalpingTrader, state persistence, the Flask
dashboard, the CLI — is the same code that runs on the seedbox.

Stages:
  1. CLI contract          real argparse accepts the deploy invocation
  2. Real-stack boot       trader + client constructed for real (signed
                           private/Balance served over fake HTTP)
  3. Dashboard fresh start load_state on empty dir + StrictUndefined render
                           + real Flask GET / (the May 2026 crash class)
  4. Paper order engine    long AND short round trips + restart reconcile
                           through the real engine (the short-rejection class)
  5. Live payload contract AddOrder payloads captured at the HTTP boundary:
                           leverage/reduce_only on shorts, none on longs,
                           pair_decimals price precision (BTC=1dp, POL=5dp)
  6. Mini live loop        sliding 720 window + in-progress stubs served as
                           raw Kraken JSON through the real client; zero
                           logger.error tolerated; state.json written; restart
                           trader reloads it; dashboard renders the real state

Usage:
    python3 run_preflight.py             # default: last 240 candles x 4 polls
    python3 run_preflight.py --quick     # last 80 candles x 3 polls (~30s)
    python3 run_preflight.py --thorough  # all candles x 8 polls
"""
import base64
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "data" / "q1_2026_fresh"
CANDLE_SECONDS = 4 * 3600
KRAKEN_WINDOW = 720

# Credentials must exist (client signs private calls); secret must be base64.
os.environ.setdefault("KRAKEN_API_KEY", "preflight-key")
os.environ.setdefault("KRAKEN_API_SECRET", base64.b64encode(b"preflight-secret").decode())

from src.observability.logger import configure_logging  # noqa: E402

configure_logging(level="CRITICAL", format_type="json")

# Realistic Kraken pair_decimals (price precision). BTC/USD really is 1.
PAIR_DECIMALS = {
    "XBTUSD": 1, "ETHUSD": 2, "SOLUSD": 2, "AVAXUSD": 3, "LINKUSD": 3,
    "NEARUSD": 4, "ATOMUSD": 4, "DOTUSD": 4, "XRPUSD": 5, "POLUSD": 5,
}


# ============================================================================
# Market simulation + fake requests.Session (the ONLY mocked layer)
# ============================================================================

class MarketSim:
    """Holds raw candle rows per pair and a simulated clock position."""

    def __init__(self, data_dir: Path):
        self.data = {}      # kraken_pair_key -> list of [ts,o,h,l,c,v]
        self.idx = 0        # index of the candle currently FORMING
        self.progress = 0.5  # 0..1 fraction of the forming candle elapsed
        for f in sorted(data_dir.glob("*_4h.csv")):
            key = f.stem.replace("_4h", "").replace("_", "").replace("BTC", "XBT")
            rows = []
            with open(f) as fh:
                for row in csv.reader(fh):
                    rows.append([int(row[0])] + [float(x) for x in row[1:6]])
            self.data[key] = rows
        if not self.data:
            raise SystemExit(f"PREFLIGHT: no candle data in {data_dir}")
        self.num_candles = min(len(r) for r in self.data.values())

    def now(self) -> datetime:
        ts = next(iter(self.data.values()))[min(self.idx, self.num_candles - 1)][0]
        return datetime.fromtimestamp(
            ts + CANDLE_SECONDS * self.progress, tz=timezone.utc
        )

    def _interp_price(self, row) -> float:
        _, o, h, l, c, _ = row
        t = self.progress
        price = o + (c - o) * t
        if 0.30 <= t <= 0.35:
            price = h
        elif 0.65 <= t <= 0.70:
            price = l
        return price

    def ohlc_payload(self, pair_key: str) -> list:
        """Sliding <=720 window of raw Kraken arrays, last entry in-progress."""
        rows = self.data[pair_key]
        idx = min(self.idx, len(rows) - 1)
        window = rows[max(0, idx + 1 - KRAKEN_WINDOW):idx]
        out = []
        for ts, o, h, l, c, v in window:
            out.append([ts, str(o), str(h), str(l), str(c),
                        str((h + l + c) / 3), str(v), 100])
        # In-progress stub: partial volume/range, close = interpolated price
        ts, o, h, l, c, v = rows[idx]
        t = max(self.progress, 0.005)
        price = self._interp_price(rows[idx])
        s_high = max(o, price, h if t > 0.30 else o)
        s_low = min(o, price, l if t > 0.65 else o)
        out.append([ts, str(o), str(s_high), str(s_low), str(price),
                    str((s_high + s_low + price) / 3), str(v * t),
                    max(1, int(100 * t))])
        return out

    def ticker_payload(self, pair_key: str) -> dict:
        rows = self.data[pair_key]
        row = rows[min(self.idx, len(rows) - 1)]
        price = self._interp_price(row)
        _, o, h, l, c, v = row
        return {
            "a": [str(price * 1.0005), "1", "1.0"],
            "b": [str(price * 0.9995), "1", "1.0"],
            "c": [str(price), "0.1"],
            "v": [str(v / 6), str(v)],
            "p": [str(price), str((h + l + c) / 3)],
            "t": [100, 600],
            "l": [str(l), str(l)],
            "h": [str(h), str(h)],
            "o": str(o),
        }


class FakeResponse:
    def __init__(self, result):
        self._body = {"error": [], "result": result}

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


class FakeKrakenSession:
    """Drop-in for requests.Session serving Kraken-shaped JSON envelopes.

    Anything the code requests that isn't modeled here raises immediately —
    new API usage must be added consciously, not silently no-opped.
    """
    sim: MarketSim = None          # shared across instances (class attr)
    add_order_payloads: list = []  # captured private/AddOrder form data
    private_headers: list = []     # captured auth headers for signing checks

    def __init__(self):
        self.headers = {}

    # requests.Session API surface used by KrakenClient ----------------------
    def get(self, url, params=None, headers=None):
        return self._route(url, params or {})

    def post(self, url, data=None, headers=None):
        if headers and "API-Sign" in headers:
            FakeKrakenSession.private_headers.append(dict(headers))
        return self._route(url, data or {})

    def close(self):
        pass

    # routing -----------------------------------------------------------------
    def _route(self, url, data):
        endpoint = url.split("/0/")[-1]
        if endpoint == "public/Time":
            now = int(self.sim.now().timestamp())
            return FakeResponse({"unixtime": now, "rfc1123": "preflight"})
        if endpoint == "public/Ticker":
            pair = data["pair"]
            return FakeResponse({pair: self.sim.ticker_payload(pair)})
        if endpoint == "public/OHLC":
            pair = data["pair"]
            interval = int(data.get("interval", 0))
            assert interval == 240, (
                f"OHLC requested at interval={interval}; preflight only models "
                f"240 (4h). If this is intentional, extend MarketSim."
            )
            return FakeResponse({pair: self.sim.ohlc_payload(pair), "last": 0})
        if endpoint == "public/AssetPairs":
            pair = data["pair"]
            return FakeResponse({pair: {"pair_decimals": PAIR_DECIMALS.get(pair, 2)}})
        if endpoint == "private/Balance":
            return FakeResponse({"ZUSD": "10000.0000"})
        if endpoint == "private/AddOrder":
            FakeKrakenSession.add_order_payloads.append(dict(data))
            n = len(FakeKrakenSession.add_order_payloads)
            return FakeResponse({"txid": [f"PF-{n:04d}"], "descr": {"order": "preflight"}})
        raise AssertionError(f"PREFLIGHT: unmocked Kraken endpoint '{endpoint}'")


class NoopRateLimiter:
    def acquire_sync(self, *_a, **_k):
        pass


# ============================================================================
# Harness plumbing
# ============================================================================

FAILURES: list[str] = []
_STAGE = [0]


def stage(name):
    _STAGE[0] += 1
    print(f"\n[{_STAGE[0]}] {name}")
    print("-" * 74)


def check(ok: bool, label: str, detail: str = ""):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(f"{label}" + (f": {detail}" if detail else ""))
    return ok


class ErrorCapture:
    """Wraps trader.logger.error — a healthy run logs ZERO errors."""

    def __init__(self, logger):
        self._logger = logger
        self.errors = []
        self._orig = logger.error

    def install(self):
        def capture(msg, *a, **k):
            self.errors.append(str(msg))
            return self._orig(msg, *a, **k)
        self._logger.error = capture

    def uninstall(self):
        self._logger.error = self._orig


def build_trader(state_dir: Path):
    """Construct ScalpingTrader through its REAL __init__ (no bypasses)."""
    from run_scalping_live import ScalpingTrader
    trader = ScalpingTrader(
        config_path="config/scalping.yaml",
        data_dir=str(state_dir),
        paper_trading=True,
    )
    trader.client._rate_limiter = NoopRateLimiter()  # perf only
    trader._now = FakeKrakenSession.sim.now          # injectable clock
    return trader


# ============================================================================
# Stages
# ============================================================================

def stage_cli():
    stage("CLI contract — the deploy invocation parses")
    r = subprocess.run(
        [sys.executable, str(REPO / "run_scalping_live.py"), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    check(r.returncode == 0, "run_scalping_live.py --help exits 0",
          (r.stderr or "")[-200:] if r.returncode else "")
    # The deploy script starts the trader with NO flags (paper is the default).
    # Guard against docs/scripts drifting to flags that don't exist (--paper).
    help_text = r.stdout or ""
    check("--live" in help_text, "--live flag exists (live opt-in)")
    check("--paper" not in help_text, "no phantom --paper flag in interface")


def stage_boot(state_dir: Path):
    stage("Real-stack boot — KrakenClient.__init__ + ScalpingTrader.__init__")
    t0 = time.time()
    trader = build_trader(state_dir)
    check(True, f"trader constructed in {time.time() - t0:.1f}s "
                f"({len(trader.pairs)} pairs)")
    check(trader.client.paper_trading, "client in paper mode")
    check(abs(trader.capital - 10000.0) < 1e-6,
          "capital synced from (fake) real Kraken balance",
          f"capital=${trader.capital:.2f}")
    check(len(FakeKrakenSession.private_headers) >= 1
          and "API-Sign" in FakeKrakenSession.private_headers[0],
          "private/Balance request was signed (real auth path ran)")
    check(trader.short_leverage >= 2, "shorting.leverage configured",
          f"leverage={trader.short_leverage}")
    return trader


def stage_dashboard_fresh():
    stage("Dashboard fresh start — empty data dir must render")
    import jinja2
    from src.scalping_dashboard.app import create_app, load_state

    empty = Path(tempfile.mkdtemp(prefix="preflight_empty_"))
    try:
        state = load_state(empty)
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(
                str(REPO / "src/scalping_dashboard/templates")),
            undefined=jinja2.StrictUndefined,
        )
        try:
            env.get_template("scalping_dashboard.html").render(**state)
            check(True, "template renders default_state under StrictUndefined")
        except jinja2.exceptions.UndefinedError as e:
            check(False, "template renders default_state under StrictUndefined",
                  str(e))
        app = create_app(data_dir=str(empty))
        rc = app.test_client().get("/")
        check(rc.status_code == 200, "Flask GET / on fresh start", f"HTTP {rc.status_code}")
    finally:
        shutil.rmtree(empty, ignore_errors=True)


def stage_paper_engine(trader):
    stage("Paper order engine — LONG and SHORT round trips through real client")
    sim = FakeKrakenSession.sim
    sim.idx = max(0, sim.num_candles - 1)
    sim.progress = 0.5

    # Long round trip
    e = trader._execute_entry_order("ETH/USD", "long", 0.5, 0.0)
    check(e is not None, "LONG entry executes (spot buy)")
    x = trader._execute_exit_order("ETH/USD", "long", 0.5)
    check(x is not None, "LONG exit executes (spot sell)")

    # Short round trip — this exact call failed in production on 2026-06-11
    e = trader._execute_entry_order("NEAR/USD", "short", 300.0, 0.0)
    check(e is not None, "SHORT entry executes (margin sell, no inventory)")
    x = trader._execute_exit_order("NEAR/USD", "short", 300.0)
    check(x is not None, "SHORT cover executes (reduce_only buy)")
    near_left = abs(trader.client._paper_balances.get("NEAR", 0.0))
    check(near_left < 1e-9, "short inventory nets to zero after cover",
          f"residual={near_left}")

    # Restart simulation: fresh balances from 'real' account + reconcile,
    # then BOTH sides must still be able to exit (the May restart-bug class).
    trader.positions = {
        "ETH/USD": {"side": "long", "size": 0.5, "size_usd": 1500.0},
        "NEAR/USD": {"side": "short", "size": 300.0, "size_usd": 600.0},
    }
    trader.client._paper_balances = {"USD": 4000.0}  # simulates restart re-init
    trader._reconcile_paper_balances()
    check(trader.client._paper_balances.get("ETH", 0) >= 0.5,
          "reconcile restores long inventory after restart")
    check(trader.client._paper_balances.get("NEAR", 0) <= -300.0,
          "reconcile restores NEGATIVE short inventory after restart")
    x1 = trader._execute_exit_order("ETH/USD", "long", 0.5)
    x2 = trader._execute_exit_order("NEAR/USD", "short", 300.0)
    check(x1 is not None, "long exit succeeds post-restart")
    check(x2 is not None, "short cover succeeds post-restart")
    trader.positions = {}


def stage_live_payloads():
    stage("Live payload contract — AddOrder fields at the HTTP boundary")
    from src.exchange.kraken_client import KrakenClient, OrderSide

    FakeKrakenSession.add_order_payloads.clear()
    live = KrakenClient(
        api_key="preflight-key",
        api_secret=base64.b64encode(b"preflight-secret").decode(),
        paper_trading=False,
    )
    live._rate_limiter = NoopRateLimiter()

    def last_payload():
        return FakeKrakenSession.add_order_payloads[-1]

    # Long entry: plain spot — must NOT carry margin params
    live.place_maker_order("ETH/USD", OrderSide.BUY, 0.5)
    p = last_payload()
    check("leverage" not in p and "reduce_only" not in p,
          "long entry is plain spot (no leverage/reduce_only)", str(p))
    check(p.get("oflags") == "post", "maker order carries post-only flag")

    # Short entry: margin sell
    live.place_maker_order("NEAR/USD", OrderSide.SELL, 300.0, leverage=2)
    p = last_payload()
    check(p.get("type") == "sell" and p.get("leverage") == "2",
          "short entry carries leverage", str(p))
    check("reduce_only" not in p, "short ENTRY has no reduce_only")

    # Short cover: margin buy + reduce_only
    live.place_maker_order("NEAR/USD", OrderSide.BUY, 300.0,
                           leverage=2, reduce_only=True)
    p = last_payload()
    check(p.get("type") == "buy" and p.get("leverage") == "2"
          and p.get("reduce_only") == "true",
          "short cover carries leverage + reduce_only", str(p))

    # Price precision: BTC/USD allows exactly 1 decimal on Kraken
    live.place_maker_order("BTC/USD", OrderSide.BUY, 0.01)
    p = last_payload()
    dec = p["price"].split(".")[1] if "." in p["price"] else ""
    check(len(dec) <= 1, "BTC/USD maker price respects pair_decimals=1",
          f"price={p['price']}")

    # Price precision: POL/USD must NOT be quantized to cents
    sim = FakeKrakenSession.sim
    pol_rows = sim.data.get("POLUSD")
    if pol_rows:
        raw = sim._interp_price(pol_rows[min(sim.idx, len(pol_rows) - 1)]) * 1.0005
        live.place_maker_order("POL/USD", OrderSide.SELL, 1000.0, leverage=2)
        p = last_payload()
        sent = float(p["price"])
        check(raw > 0 and abs(sent - raw) / raw < 0.001,
              "POL/USD maker price not quantized to cents",
              f"raw={raw:.6f} sent={sent}")

    check(all("API-Sign" in h for h in FakeKrakenSession.private_headers[-5:]),
          "all AddOrder requests signed")
    live.close()


def stage_live_loop(state_dir: Path, span: int, polls: int):
    stage(f"Mini live loop — {span} candles x {polls} polls through the real client")
    sim = FakeKrakenSession.sim
    trader = build_trader(state_dir)

    cap = ErrorCapture(trader.logger)
    cap.install()
    entries = exits = 0
    start = max(0, sim.num_candles - span)
    t0 = time.time()
    try:
        for i in range(start, sim.num_candles):
            sim.idx = i
            trader._update_regime()           # unguarded: a raise fails preflight
            for poll in range(polls):
                sim.progress = (poll + 0.5) / polls
                before = set(trader.positions)
                for pair in trader.pairs:
                    trader._process_pair(pair)  # unguarded
                after = set(trader.positions)
                entries += len(after - before)
                exits += len(before - after)
            if (i - start) % 60 == 0:
                print(f"  ... candle {i - start}/{span}  positions={len(trader.positions)} "
                      f"capital=${trader.capital:.2f}")
        trader._save_state()
    finally:
        cap.uninstall()

    dur = time.time() - t0
    pnl = trader.capital - trader.initial_capital
    print(f"  loop done in {dur:.0f}s — entries={entries} exits={exits} "
          f"open={len(trader.positions)} P&L=${pnl:+.2f} regime={trader._current_regime.value}")

    check(not cap.errors, "zero logger.error during loop",
          "; ".join(cap.errors[:3]) + (f" (+{len(cap.errors) - 3} more)" if len(cap.errors) > 3 else ""))
    check(entries > 0, "at least one organic entry executed",
          "" if entries else
          "no entries — signal path may be dead (see 18-day zero-trade incident)")
    check((state_dir / "state.json").exists(), "state.json written")

    # Restart fidelity: a NEW trader on the same dir must resume cleanly
    trader2 = build_trader(state_dir)
    check(set(trader2.positions) == set(trader.positions),
          "restarted trader reloads identical open positions",
          f"{sorted(trader.positions)} vs {sorted(trader2.positions)}")
    check(abs(trader2.capital - trader.capital) < 0.01,
          "restarted trader reloads capital",
          f"${trader.capital:.2f} vs ${trader2.capital:.2f}")
    closed = 0
    for pair in list(trader2.positions):
        strat = trader2._get_strategy_for_pair(pair)
        px = float(sim.ticker_payload(pair.replace("/", "").replace("BTC", "XBT"))["c"][0])
        if trader2._close_position(pair, "Preflight restart-close", px, strat):
            closed += 1
    check(not trader2.positions,
          f"restarted trader can close all reloaded positions ({closed} closed)")

    # Dashboard against the REAL state the loop just wrote
    import jinja2
    from src.scalping_dashboard.app import create_app, load_state
    state = load_state(state_dir)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(REPO / "src/scalping_dashboard/templates")),
        undefined=jinja2.StrictUndefined,
    )
    try:
        env.get_template("scalping_dashboard.html").render(**state)
        check(True, "template renders REAL post-loop state under StrictUndefined")
    except jinja2.exceptions.UndefinedError as e:
        check(False, "template renders REAL post-loop state under StrictUndefined", str(e))
    rc = create_app(data_dir=str(state_dir)).test_client().get("/")
    check(rc.status_code == 200, "Flask GET / on real state", f"HTTP {rc.status_code}")


# ============================================================================

def main():
    mode = "default"
    if "--quick" in sys.argv:
        mode, span, polls = "quick", 80, 3
    elif "--thorough" in sys.argv:
        mode, span, polls = "thorough", 10 ** 9, 8
    else:
        span, polls = 240, 4

    print("=" * 74)
    print(f"  PREFLIGHT GATE ({mode})  —  mocks requests.Session ONLY; "
          f"all app code is real")
    print("=" * 74)

    if not DATA_DIR.exists():
        print(f"FATAL: {DATA_DIR} missing (needed for market sim)")
        return 1

    # Patch the network boundary, then everything constructs for real.
    import src.exchange.kraken_client as kc
    FakeKrakenSession.sim = MarketSim(DATA_DIR)
    FakeKrakenSession.add_order_payloads = []
    FakeKrakenSession.private_headers = []
    kc.requests.Session = FakeKrakenSession

    state_dir = Path(tempfile.mkdtemp(prefix="preflight_state_"))
    loop_dir = Path(tempfile.mkdtemp(prefix="preflight_loop_"))

    def guarded(fn, *args, **kwargs):
        """Run a stage; an uncaught exception is a preflight FAILURE, and the
        remaining stages still run so the report is complete."""
        try:
            return fn(*args, **kwargs)
        except Exception:
            import traceback
            tb = traceback.format_exc()
            print(tb)
            FAILURES.append(f"{fn.__name__} raised: {tb.strip().splitlines()[-1]}")
            return None

    try:
        guarded(stage_cli)
        trader = guarded(stage_boot, state_dir)
        guarded(stage_dashboard_fresh)
        if trader is not None:
            guarded(stage_paper_engine, trader)
        else:
            FAILURES.append("paper engine stage skipped (boot failed)")
        guarded(stage_live_payloads)
        guarded(stage_live_loop, loop_dir, span=span, polls=polls)
    finally:
        shutil.rmtree(state_dir, ignore_errors=True)
        shutil.rmtree(loop_dir, ignore_errors=True)

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"  PREFLIGHT: FAIL — {len(FAILURES)} check(s) failed. DO NOT DEPLOY.")
        for f in FAILURES:
            print(f"    * {f}")
        print("=" * 74)
        return 1
    print("  PREFLIGHT: PASS — safe to deploy.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
