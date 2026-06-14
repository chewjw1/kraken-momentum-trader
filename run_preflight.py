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
                           pair_decimals price precision (BTC=1dp, POL=5dp),
                           BOT_USERREF tagging
  6. Live order lifecycle  scripted exchange: slow/partial/never fills,
                           post-only rejections, market fallbacks, lot/min
                           volume guards, margin eligibility, server-side
                           disaster stops (cancel-before-exit + already-
                           triggered settle), startup reconciliation (ghosts,
                           orphans, stale orders), drift check
  7. Mini live loop        sliding 720 window + in-progress stubs served as
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

# Realistic Kraken AssetPairs constraints. BTC/USD price precision really is
# 1 decimal. POLUSD is given NO margin support so the short-eligibility block
# has a real negative case to catch.
ASSET_META = {
    "XBTUSD":  {"pair_decimals": 1, "lot_decimals": 8, "ordermin": 0.00005, "lev": [2, 3, 4, 5]},
    "ETHUSD":  {"pair_decimals": 2, "lot_decimals": 8, "ordermin": 0.002,   "lev": [2, 3, 4, 5]},
    "SOLUSD":  {"pair_decimals": 2, "lot_decimals": 8, "ordermin": 0.02,    "lev": [2, 3]},
    "AVAXUSD": {"pair_decimals": 3, "lot_decimals": 8, "ordermin": 0.1,     "lev": [2, 3]},
    "LINKUSD": {"pair_decimals": 3, "lot_decimals": 8, "ordermin": 0.1,     "lev": [2, 3]},
    "NEARUSD": {"pair_decimals": 4, "lot_decimals": 8, "ordermin": 1.0,     "lev": [2, 3]},
    "ATOMUSD": {"pair_decimals": 4, "lot_decimals": 8, "ordermin": 0.5,     "lev": [2, 3]},
    "DOTUSD":  {"pair_decimals": 4, "lot_decimals": 8, "ordermin": 0.2,     "lev": [2, 3]},
    "XRPUSD":  {"pair_decimals": 5, "lot_decimals": 8, "ordermin": 2.5,     "lev": [2, 3, 4, 5]},
    "POLUSD":  {"pair_decimals": 5, "lot_decimals": 8, "ordermin": 10.0,    "lev": []},
}
PAIR_DECIMALS = {k: v["pair_decimals"] for k, v in ASSET_META.items()}


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
    def __init__(self, result, errors=None):
        self._body = {"error": errors or [], "result": result}

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


class FakeKrakenSession:
    """Drop-in for requests.Session serving Kraken-shaped JSON envelopes,
    including a scriptable order book for live-lifecycle testing.

    Fill plans (consumed by LIMIT orders in placement sequence):
        ("instant",)        fill fully on placement
        ("after", n)        fill fully after n QueryOrders polls
        ("partial", frac)   fill `frac` immediately, then sit open forever
        ("never",)          sit open, zero filled
    Market orders always fill instantly; stop-loss orders rest until
    trigger_stop() is called. Anything the code requests that isn't modeled
    here raises immediately — new API usage must be added consciously.
    """
    sim: MarketSim = None          # shared across instances (class attr)
    add_order_payloads: list = []  # captured private/AddOrder form data
    private_headers: list = []     # captured auth headers for signing checks
    orders: dict = {}              # txid -> order record (live order book)
    order_seq: list = [0]
    fill_plans: list = []          # queue of plans for upcoming LIMIT orders
    postonly_rejects: list = [0]   # reject next N post-only placements
    open_positions: dict = {}      # served by private/OpenPositions
    balances: dict = {"ZUSD": "10000.0000"}

    def __init__(self):
        self.headers = {}

    @classmethod
    def reset_book(cls):
        cls.orders = {}
        cls.fill_plans = []
        cls.postonly_rejects = [0]
        cls.open_positions = {}
        cls.balances = {"ZUSD": "10000.0000"}
        cls.add_order_payloads = []

    @classmethod
    def trigger_stop(cls, txid, price):
        """Simulate a resting stop-loss executing on the exchange."""
        rec = cls.orders[txid]
        rec["status"] = "closed"
        rec["vol_exec"] = rec["vol"]
        rec["avg_price"] = price
        rec["fee"] = price * rec["vol"] * 0.0026

    # requests.Session API surface used by KrakenClient ----------------------
    def get(self, url, params=None, headers=None):
        return self._route(url, params or {})

    def post(self, url, data=None, headers=None):
        if headers and "API-Sign" in headers:
            FakeKrakenSession.private_headers.append(dict(headers))
        return self._route(url, data or {})

    def close(self):
        pass

    # order book helpers ------------------------------------------------------
    def _last_price(self, pair_key):
        try:
            return float(self.sim.ticker_payload(pair_key)["c"][0])
        except Exception:
            return 0.0

    def _fill(self, rec, frac=1.0):
        rec["vol_exec"] = rec["vol"] * frac
        px = rec["limit_price"] if rec["limit_price"] else self._last_price(rec["pair"])
        rec["avg_price"] = px
        fee_rate = 0.0016 if rec.get("post") else 0.0026
        rec["fee"] = px * rec["vol_exec"] * fee_rate
        if frac >= 0.999:
            rec["status"] = "closed"

    def _order_view(self, rec):
        return {
            "status": rec["status"],
            "vol": str(rec["vol"]),
            "vol_exec": str(rec["vol_exec"]),
            "price": str(rec["avg_price"]),
            "fee": str(rec["fee"]),
            "userref": rec["userref"],
            "descr": {"pair": rec["pair"], "type": rec["type"],
                      "ordertype": rec["ordertype"],
                      "price": str(rec["limit_price"] or 0)},
        }

    def _add_order(self, data):
        cls = FakeKrakenSession
        cls.add_order_payloads.append(dict(data))
        if data.get("oflags") == "post" and cls.postonly_rejects[0] > 0:
            cls.postonly_rejects[0] -= 1
            return FakeResponse(None, errors=["EOrder:Post only order"])
        cls.order_seq[0] += 1
        txid = f"PF-{cls.order_seq[0]:04d}"
        rec = {
            "pair": data["pair"], "type": data["type"],
            "ordertype": data["ordertype"], "vol": float(data["volume"]),
            "vol_exec": 0.0, "avg_price": 0.0, "fee": 0.0,
            "limit_price": float(data["price"]) if data.get("price") else None,
            "post": data.get("oflags") == "post",
            "userref": int(data.get("userref", 0) or 0),
            "status": "open", "queries": 0, "plan": ("instant",),
        }
        if data.get("validate"):
            return FakeResponse({"descr": {"order": "validated"}})
        if rec["ordertype"] == "market":
            self._fill(rec)                       # market: instant, always
        elif rec["ordertype"] == "stop-loss":
            pass                                  # rests until trigger_stop()
        else:                                     # limit: consume the plan queue
            plan = cls.fill_plans.pop(0) if cls.fill_plans else ("instant",)
            rec["plan"] = plan
            if plan[0] == "instant":
                self._fill(rec)
            elif plan[0] == "partial":
                self._fill(rec, plan[1])
        cls.orders[txid] = rec
        return FakeResponse({"txid": [txid], "descr": {"order": "preflight"}})

    # routing -----------------------------------------------------------------
    def _route(self, url, data):
        cls = FakeKrakenSession
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
            meta = ASSET_META.get(pair, {"pair_decimals": 2, "lot_decimals": 8,
                                         "ordermin": 0, "lev": [2]})
            return FakeResponse({pair: {
                "pair_decimals": meta["pair_decimals"],
                "lot_decimals": meta["lot_decimals"],
                "ordermin": str(meta["ordermin"]),
                "leverage_buy": meta["lev"],
                "leverage_sell": meta["lev"],
            }})
        if endpoint == "private/Balance":
            return FakeResponse(dict(cls.balances))
        if endpoint == "private/AddOrder":
            return self._add_order(data)
        if endpoint == "private/QueryOrders":
            txid = data["txid"]
            rec = cls.orders.get(txid)
            if rec is None:
                return FakeResponse(None, errors=["EOrder:Unknown order"])
            rec["queries"] += 1
            if (rec["status"] == "open" and rec["plan"][0] == "after"
                    and rec["queries"] >= rec["plan"][1]):
                self._fill(rec)
            return FakeResponse({txid: self._order_view(rec)})
        if endpoint == "private/CancelOrder":
            rec = cls.orders.get(data["txid"])
            if rec is None or rec["status"] == "closed":
                return FakeResponse(None, errors=["EOrder:Unknown order"])
            rec["status"] = "canceled"
            return FakeResponse({"count": 1})
        if endpoint == "private/OpenOrders":
            want_ref = data.get("userref")
            out = {t: self._order_view(r) for t, r in cls.orders.items()
                   if r["status"] in ("open", "pending")
                   and (want_ref is None or str(r["userref"]) == str(want_ref))}
            return FakeResponse({"open": out})
        if endpoint == "private/ClosedOrders":
            want_ref = data.get("userref")
            out = {t: self._order_view(r) for t, r in cls.orders.items()
                   if r["status"] in ("closed", "canceled")
                   and (want_ref is None or str(r["userref"]) == str(want_ref))}
            return FakeResponse({"closed": out})
        if endpoint == "private/OpenPositions":
            return FakeResponse(dict(cls.open_positions))
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
    check(all(pl.get("userref") == str(live.BOT_USERREF)
              for pl in FakeKrakenSession.add_order_payloads),
          "all live orders tagged with BOT_USERREF")
    live.close()


def stage_live_lifecycle():
    stage("LIVE order lifecycle — fills, stops, reconciliation (scripted exchange)")
    from src.exchange.kraken_client import KrakenClient
    from run_scalping_live import ScalpingTrader
    S = FakeKrakenSession
    S.reset_book()

    def live_trader(d):
        t = ScalpingTrader(config_path="config/scalping.yaml",
                           data_dir=str(d), paper_trading=False)
        t.client._rate_limiter = NoopRateLimiter()
        t._now = S.sim.now
        t._sleep = lambda s: None
        return t

    def payloads_for(pair_key, **filters):
        out = []
        for p in S.add_order_payloads:
            if p.get("pair") != pair_key:
                continue
            if all(p.get(k) == v for k, v in filters.items()):
                out.append(p)
        return out

    d1 = Path(tempfile.mkdtemp(prefix="preflight_live1_"))
    d2 = Path(tempfile.mkdtemp(prefix="preflight_live2_"))
    try:
        trader = live_trader(d1)
        check(not trader.client.paper_trading, "trader booted in LIVE mode")
        check("POL/USD" in trader._short_blocked_pairs,
              "startup eligibility report flags POL/USD (no margin support)")

        # --- fills -----------------------------------------------------------
        S.fill_plans = [("instant",)]
        r = trader._execute_entry_order("ETH/USD", "long", 0.5, 0.0)
        check(r is not None and abs(r['fill_volume'] - 0.5) < 1e-9
              and r['fill_price'] > 0 and r['fee'] > 0,
              "instant maker fill returns actual price/volume/fee", str(r))

        S.fill_plans = [("after", 3)]
        r = trader._execute_entry_order("NEAR/USD", "short", 300.0, 0.0)
        check(r is not None and abs(r['fill_volume'] - 300.0) < 1e-9,
              "slow maker fill (3 polls) awaited to completion")

        S.fill_plans = [("partial", 0.4)]
        r = trader._execute_entry_order("DOT/USD", "long", 100.0, 0.0)
        partial_rec = [rec for rec in S.orders.values()
                       if rec["pair"] == "DOTUSD" and rec["ordertype"] == "limit"][-1]
        check(r is not None and abs(r['fill_volume'] - 40.0) < 1e-6,
              "partial entry kept at filled size (40/100)", str(r))
        check(partial_rec["status"] == "canceled",
              "unfilled entry remainder cancelled on the exchange")

        S.fill_plans = [("never",)]
        n_before = len(S.add_order_payloads)
        r = trader._execute_entry_order("LINK/USD", "long", 10.0, 0.0)
        mkts = [p for p in S.add_order_payloads[n_before:]
                if p.get("ordertype") == "market"]
        check(r is not None and len(mkts) == 1 and abs(r['fill_volume'] - 10.0) < 1e-9,
              "unfilled entry falls back to market when configured")

        trader.live_entry_market_fallback = False
        S.fill_plans = [("never",)]
        n_before = len(S.add_order_payloads)
        r = trader._execute_entry_order("LINK/USD", "long", 10.0, 0.0)
        mkts = [p for p in S.add_order_payloads[n_before:]
                if p.get("ordertype") == "market"]
        check(r is None and not mkts,
              "unfilled entry skipped (no market) when fallback disabled")
        trader.live_entry_market_fallback = True

        S.fill_plans = [("partial", 0.5)]
        r = trader._execute_exit_order("ETH/USD", "long", 0.5)
        check(r is not None and abs(r['fill_volume'] - 0.5) < 1e-9
              and "+" in r['order_id'],
              "partially-filled EXIT completed via market remainder", str(r))

        S.postonly_rejects = [1]
        S.fill_plans = [("instant",)]
        n_before = len(S.add_order_payloads)
        r = trader._execute_entry_order("XRP/USD", "long", 50.0, 0.0)
        check(r is not None and len(S.add_order_payloads) - n_before == 2,
              "post-only rejection retried at fresh price")

        S.postonly_rejects = [10]
        n_before = len(S.add_order_payloads)
        r = trader._execute_entry_order("XRP/USD", "long", 50.0, 0.0)
        last = S.add_order_payloads[-1]
        check(r is not None and last.get("ordertype") == "market",
              "post-only retries exhausted -> market fallback")
        S.postonly_rejects = [0]

        # --- guards ----------------------------------------------------------
        r = trader._execute_entry_order("POL/USD", "short", 1000.0, 0.0)
        check(r is None, "short on margin-ineligible pair blocked")
        r = trader._execute_entry_order("POL/USD", "long", 5.0, 0.0)
        check(r is None, "order below pair minimum (ordermin) blocked")
        S.fill_plans = [("instant",)]
        trader._execute_entry_order("ETH/USD", "long", 0.123456789012345, 0.0)
        sent_vol = S.add_order_payloads[-1]["volume"]
        dec = sent_vol.split(".")[1] if "." in sent_vol else ""
        check(len(dec) <= 8, "volume rounded to lot_decimals", f"volume={sent_vol}")

        # --- server-side disaster stop ----------------------------------------
        def near_position():
            return {'entry_price': 1.2, 'side': 'short', 'size': 300.0,
                    'size_usd': 360.0,
                    'entry_time': S.sim.now().isoformat(), 'reason': 'pf',
                    'regime': 'bear', 'best_price': 1.2, 'trailing_stop': 0.0,
                    'order_id': 'PF-ENTRY', 'entry_fee': 0.5}

        trader.positions['NEAR/USD'] = near_position()
        stop_id = trader._place_server_stop('NEAR/USD', 'short', 300.0, 1.2)
        stop_rec = S.orders.get(stop_id, {})
        check(stop_id is not None and stop_rec.get("ordertype") == "stop-loss"
              and stop_rec.get("type") == "buy"
              and abs((stop_rec.get("limit_price") or 0) - 1.32) < 1e-6,
              "server stop: buy stop-loss at +10% of short entry",
              f"trigger={stop_rec.get('limit_price')}")
        stop_payload = payloads_for("NEARUSD", ordertype="stop-loss")[-1]
        check(stop_payload.get("leverage") == "2"
              and stop_payload.get("reduce_only") == "true",
              "server stop carries leverage + reduce_only")

        trader.positions['NEAR/USD']['stop_order_id'] = stop_id
        ok = trader._close_position('NEAR/USD', 'Preflight TP', 1.1,
                                    trader._get_strategy_for_pair('NEAR/USD'))
        check(ok and 'NEAR/USD' not in trader.positions
              and S.orders[stop_id]["status"] == "canceled",
              "normal exit cancels the resting stop first")

        trader.positions['NEAR/USD'] = near_position()
        stop_id = trader._place_server_stop('NEAR/USD', 'short', 300.0, 1.2)
        trader.positions['NEAR/USD']['stop_order_id'] = stop_id
        S.trigger_stop(stop_id, 1.35)
        n_before = len(S.add_order_payloads)
        cap_before = trader.capital
        ok = trader._close_position('NEAR/USD', 'Disaster stop', 1.30,
                                    trader._get_strategy_for_pair('NEAR/USD'))
        check(ok and 'NEAR/USD' not in trader.positions
              and len(S.add_order_payloads) == n_before
              and trader.capital < cap_before,
              "already-triggered stop settled from its fill, no duplicate exit",
              f"P&L ${trader.capital - cap_before:+.2f}")

        # --- startup reconciliation -------------------------------------------
        S.reset_book()
        ghost_state = {
            'positions': {
                'ETH/USD': {'entry_price': 2000.0, 'side': 'long', 'size': 0.5,
                            'size_usd': 1000.0, 'entry_time': S.sim.now().isoformat(),
                            'reason': 'x', 'regime': 'bull', 'best_price': 2000.0,
                            'trailing_stop': 0.0, 'order_id': 'OLD-1', 'entry_fee': 1.0},
                'NEAR/USD': {'entry_price': 1.2, 'side': 'short', 'size': 300.0,
                             'size_usd': 360.0, 'entry_time': S.sim.now().isoformat(),
                             'reason': 'x', 'regime': 'bear', 'best_price': 1.2,
                             'trailing_stop': 0.0, 'order_id': 'OLD-2', 'entry_fee': 0.5},
            },
            'capital': 9000.0, 'initial_capital': 10000.0,
            'metrics': {'total_trades': 5, 'wins': 3, 'losses': 2,
                        'total_pnl': -1000.0, 'peak_capital': 10500.0,
                        'start_time': S.sim.now().isoformat()},
        }
        (d2 / "state.json").write_text(json.dumps(ghost_state, default=str))
        S.open_positions = {
            "POS-1": {"pair": "NEARUSD", "type": "sell", "vol": "300", "vol_closed": "0"},
            "POS-2": {"pair": "DOTUSD", "type": "sell", "vol": "50", "vol_closed": "0"},
        }
        stale = {"pair": "ETHUSD", "type": "buy", "ordertype": "limit", "vol": 1.0,
                 "vol_exec": 0.0, "avg_price": 0.0, "fee": 0.0, "limit_price": 2000.0,
                 "post": True, "status": "open", "queries": 0, "plan": ("never",)}
        S.orders["STALE-BOT"] = dict(stale, userref=KrakenClient.BOT_USERREF)
        S.orders["MANUAL-1"] = dict(stale, userref=0)

        t2 = live_trader(d2)
        check('ETH/USD' not in t2.positions,
              "reconcile drops GHOST long (in state, not on exchange)")
        check('NEAR/USD' in t2.positions,
              "reconcile keeps short verified against OpenPositions")
        check(S.orders["STALE-BOT"]["status"] == "canceled",
              "stale bot order (our userref) cancelled at startup")
        check(S.orders["MANUAL-1"]["status"] == "open",
              "manual order (foreign userref) left untouched")
        flatten = payloads_for("DOTUSD", ordertype="market", type="buy")
        check(len(flatten) == 1 and flatten[0].get("reduce_only") == "true",
              "orphan margin short flattened (market reduce_only buy)")
        new_stop = t2.positions.get('NEAR/USD', {}).get('stop_order_id')
        check(bool(new_stop) and S.orders.get(new_stop, {}).get("status") == "open",
              "server stop re-armed for surviving position after restart")

        try:
            t2._check_live_drift()
            check(True, "drift check runs clean on consistent state")
        except Exception as e:
            check(False, "drift check runs clean on consistent state", str(e))
    finally:
        S.reset_book()
        S.open_positions = {}
        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)


def stage_concurrency():
    stage("Concurrent pair processing — capital reservation under contention")
    import threading
    from types import SimpleNamespace
    from datetime import datetime, timezone, timedelta
    from unittest.mock import MagicMock
    from src.exchange.kraken_client import KrakenClient, Ticker, OHLC
    from src.strategy.base_strategy import MarketData
    from run_scalping_live import ScalpingTrader

    def build(concurrent):
        oi = KrakenClient.__init__
        KrakenClient.__init__ = lambda self, **kw: None
        t = ScalpingTrader(config_path="config/scalping.yaml",
                           data_dir="data/preflight_concurrency", paper_trading=True)
        KrakenClient.__init__ = oi
        t.paper_trading = False            # force the live concurrent dispatch
        t.live_concurrent_pairs = concurrent
        t.client = MagicMock()
        t.capital = t.initial_capital = 10000.0
        t.position_size_pct = 20.0
        t.disaster_stop_pct = 0.0
        t._candles_since_flip = 10 ** 9
        t._block_entry_regimes = set()
        t._block_short_regimes = set()
        cyc = {"n": 0}

        def md(pair):
            ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=4 * cyc["n"])
            c = OHLC(timestamp=ts, open=100.0, high=101.0, low=99.0,
                     close=100.0, volume=1000.0, vwap=100.0, count=10)
            tk = Ticker(pair=pair, ask=100.1, bid=99.9, last=100.0, volume_24h=6000.0,
                        vwap_24h=100.0, trades_24h=100, low_24h=99.0, high_24h=101.0,
                        timestamp=ts)
            return MarketData(pair=pair, ohlc=[c] * 30, prices=[100.0] * 30,
                              volumes=[1000.0] * 30, ticker=tk)

        t._get_market_data = md
        t._compute_indicator_snapshot = lambda *a, **k: None
        t.pair_manager.is_pair_enabled = lambda pair: True
        t._correlation_blocked = lambda pair: None
        t._volume_filter_blocked = lambda m: None
        t._place_server_stop = lambda *a, **k: None
        t._save_state = lambda: None
        sig = SimpleNamespace(signal_type=SimpleNamespace(value="buy"),
                              strength=0.8, reason="preflight")
        t._get_strategy_for_pair = lambda pair: SimpleNamespace(analyze=lambda m, p=None: sig)
        infl = {"now": 0, "peak": 0}
        lk = threading.Lock()

        def slow(pair, side, size, price):
            with lk:
                infl["now"] += 1
                infl["peak"] = max(infl["peak"], infl["now"])
            time.sleep(0.03)
            with lk:
                infl["now"] -= 1
            return {"order_id": f"O-{pair}", "fill_price": price,
                    "fill_volume": size, "fee": price * size * 0.0016}

        t._execute_entry_order = slow
        return t, cyc, infl

    seq_positions = None
    for concurrent in (1, 10):
        t, cyc, infl = build(concurrent)
        ok_alloc = True
        for n in range(3):
            cyc["n"] = n
            t.positions.clear()
            t._last_processed_candle_ts.clear()
            t._process_all_pairs()
            deployed = sum(p["size_usd"] for p in t.positions.values())
            if deployed > t.capital + 1e-6:
                ok_alloc = False
        npos = len(t.positions)
        check(ok_alloc, f"concurrent={concurrent}: no capital over-allocation")
        check(abs(t._reserved_usd) < 1e-6,
              f"concurrent={concurrent}: reservation counter net zero")
        if concurrent == 1:
            seq_positions = npos
        else:
            check(npos == seq_positions,
                  "concurrent allocation matches sequential",
                  f"{npos} vs {seq_positions}")
            check(infl["peak"] > 1, "fills actually overlapped",
                  f"peak in-flight={infl['peak']}")
    shutil.rmtree("data/preflight_concurrency", ignore_errors=True)


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
        guarded(stage_live_lifecycle)
        guarded(stage_concurrency)
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
