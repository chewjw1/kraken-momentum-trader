#!/usr/bin/env python3
"""
Paper-vs-replay drift monitor.

Compares the live/paper trader's REALIZED performance (read from state.json)
against an idealized replay over the same period's candles, and flags
divergence.

Paper and replay run the SAME decision code, so a large gap is not a logic
difference — it is execution drift: real bid/ask spreads, slippage, maker
fill timing, candle/data gaps, or a config mismatch between the seedbox and
the repo. This is a monitoring/alerting tool to confirm live behaviour tracks
the validated simulation BEFORE flipping to real money.

Workflow:
  1. Fetch OHLCV covering the paper window into a dir of 10 `<PAIR>_4h.csv`
     files (same format run_replay.py consumes), e.g. with pull_kraken_ohlcv.py.
  2. python3 scripts/drift_monitor.py \\
         --state data/scalping/state.json --data-dir data/<window>

Exit code is 1 if any tracked metric drifts beyond tolerance (so it can gate a
deploy or fire an alert from cron), 0 otherwise.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("KRAKEN_API_KEY", "drift-monitor")
os.environ.setdefault("KRAKEN_API_SECRET", "drift-monitor")

from src.exchange.kraken_client import KrakenClient  # noqa: E402
from src.observability.logger import configure_logging  # noqa: E402
from run_replay import ReplayKrakenClient, load_candles  # noqa: E402
from run_scalping_live import ScalpingTrader  # noqa: E402


def load_paper(state_path: Path) -> dict:
    """Extract realized paper performance from a trader state.json."""
    data = json.loads(state_path.read_text())
    initial = data.get("initial_capital", 10000.0) or 10000.0
    capital = data.get("capital", initial)
    metrics = data.get("metrics", {})

    per_pair = {}
    window = []
    pm = data.get("pair_manager", {}) or {}
    for pair, pd in (pm.get("pairs", {}) or {}).items():
        trades = pd.get("trades", []) or []
        if not trades:
            continue
        per_pair[pair] = {
            "trades": len(trades),
            "pnl": sum(t.get("pnl", 0.0) for t in trades),
            "wins": sum(1 for t in trades if t.get("is_win")),
        }
        for t in trades:
            for key in ("entry_time", "exit_time"):
                try:
                    window.append(datetime.fromisoformat(t[key]))
                except (KeyError, ValueError, TypeError):
                    pass

    total_trades = metrics.get("total_trades", sum(p["trades"] for p in per_pair.values()))
    wins = metrics.get("wins", sum(p["wins"] for p in per_pair.values()))
    return {
        "initial": initial,
        "capital": capital,
        "pnl": capital - initial,
        "pnl_pct": (capital - initial) / initial * 100 if initial else 0.0,
        "trades": total_trades,
        "wins": wins,
        "win_rate": (wins / total_trades * 100) if total_trades else 0.0,
        "per_pair": per_pair,
        "window": (min(window), max(window)) if window else None,
    }


def run_replay(data_dir: Path, config_path: str) -> dict:
    """Step the real ScalpingTrader over a candle dir; return realized metrics."""
    candle_data = load_candles(data_dir)
    if not candle_data:
        raise SystemExit(f"No *_4h.csv candles found in {data_dir}")
    num_candles = len(next(iter(candle_data.values())))

    work = Path("data/drift_replay")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    client = ReplayKrakenClient(candle_data, capital=10000.0)
    oi = KrakenClient.__init__
    KrakenClient.__init__ = lambda self, **kw: None
    trader = ScalpingTrader(config_path=config_path, data_dir=str(work), paper_trading=True)
    KrakenClient.__init__ = oi
    trader.client = client
    trader.capital = trader.initial_capital = 10000.0
    client._fill_fee_rate = trader.fee_rate / 100.0

    for i in range(num_candles):
        client._current_idx = i
        try:
            trader._update_regime()
        except Exception:
            pass
        for pair in trader.pairs:
            try:
                trader._process_pair(pair)
            except Exception:
                pass

    pm = trader.pair_manager.to_dict()
    per_pair = {}
    for pair, pd in pm.get("pairs", {}).items():
        trades = pd.get("trades", []) or []
        if trades:
            per_pair[pair] = {
                "trades": len(trades),
                "pnl": sum(t.get("pnl", 0.0) for t in trades),
                "wins": sum(1 for t in trades if t.get("is_win")),
            }
    m = trader.metrics
    initial = trader.initial_capital
    shutil.rmtree(work, ignore_errors=True)
    return {
        "initial": initial,
        "capital": trader.capital,
        "pnl": trader.capital - initial,
        "pnl_pct": (trader.capital - initial) / initial * 100 if initial else 0.0,
        "trades": m.get("total_trades", 0),
        "wins": m.get("wins", 0),
        "win_rate": (m["wins"] / m["total_trades"] * 100) if m.get("total_trades") else 0.0,
        "per_pair": per_pair,
    }


def rel_drift(paper: float, sim: float) -> float:
    """Relative drift of paper vs sim, normalized by the larger magnitude."""
    denom = max(abs(paper), abs(sim), 1e-9)
    return abs(paper - sim) / denom * 100


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper-vs-replay drift monitor")
    ap.add_argument("--state", default="data/scalping/state.json",
                    help="trader state.json (default: data/scalping/state.json)")
    ap.add_argument("--data-dir", required=True,
                    help="dir of <PAIR>_4h.csv candles covering the paper window")
    ap.add_argument("--config", default="config/scalping.yaml")
    ap.add_argument("--tol-pnl-pct", type=float, default=40.0,
                    help="max relative drift %% of return before alerting")
    ap.add_argument("--tol-trades-pct", type=float, default=50.0,
                    help="max relative drift %% of trade count before alerting")
    ap.add_argument("--min-trades", type=int, default=10,
                    help="skip (PASS) if paper has fewer than this many trades")
    args = ap.parse_args()

    configure_logging(level="CRITICAL", format_type="json")

    state_path = Path(args.state)
    if not state_path.exists():
        print(f"ERROR: state file not found: {state_path}")
        return 2

    paper = load_paper(state_path)
    print("=" * 70)
    print("  PAPER vs REPLAY DRIFT MONITOR")
    print("=" * 70)
    if paper["window"]:
        print(f"  Paper trade window: {paper['window'][0].date()} -> {paper['window'][1].date()}")
    print(f"  Paper: {paper['trades']} trades, "
          f"{paper['win_rate']:.1f}% WR, P&L {paper['pnl_pct']:+.2f}% (${paper['pnl']:+,.2f})")

    if paper["trades"] < args.min_trades:
        print(f"\n  Only {paper['trades']} paper trades (< {args.min_trades}) — "
              f"too little data to judge drift. PASS (no-op).")
        return 0

    sim = run_replay(Path(args.data_dir), args.config)
    print(f"  Replay: {sim['trades']} trades, "
          f"{sim['win_rate']:.1f}% WR, P&L {sim['pnl_pct']:+.2f}% (${sim['pnl']:+,.2f})")

    print("\n  " + "-" * 66)
    print(f"  {'METRIC':<16}{'PAPER':>14}{'REPLAY':>14}{'DRIFT':>12}  STATUS")
    print("  " + "-" * 66)

    alerts = []

    def row(name, p, s, drift, tol, unit=""):
        status = "OK"
        if drift > tol:
            status = "DRIFT"
            alerts.append(f"{name}: {drift:.0f}% > {tol:.0f}% tolerance")
        print(f"  {name:<16}{p:>13.2f}{unit}{s:>13.2f}{unit}{drift:>11.0f}%  {status}")

    row("Return %", paper["pnl_pct"], sim["pnl_pct"],
        rel_drift(paper["pnl_pct"], sim["pnl_pct"]), args.tol_pnl_pct)
    row("Trade count", paper["trades"], sim["trades"],
        rel_drift(paper["trades"], sim["trades"]), args.tol_trades_pct)
    row("Win rate %", paper["win_rate"], sim["win_rate"],
        rel_drift(paper["win_rate"], sim["win_rate"]), args.tol_pnl_pct)

    # Per-pair P&L drift (only pairs that traded in BOTH)
    print("\n  Per-pair P&L (paper vs replay):")
    shared = sorted(set(paper["per_pair"]) & set(sim["per_pair"]))
    for pair in shared:
        pp = paper["per_pair"][pair]["pnl"]
        sp = sim["per_pair"][pair]["pnl"]
        d = rel_drift(pp, sp)
        flag = "  <-- DRIFT" if d > args.tol_pnl_pct and abs(pp - sp) > 50 else ""
        print(f"    {pair:<12} paper ${pp:+8.2f}   replay ${sp:+8.2f}   ({d:.0f}%){flag}")
        if flag:
            alerts.append(f"{pair} per-pair P&L: {d:.0f}% drift")
    only_paper = sorted(set(paper["per_pair"]) - set(sim["per_pair"]))
    only_sim = sorted(set(sim["per_pair"]) - set(paper["per_pair"]))
    if only_paper:
        print(f"    (traded in paper only: {', '.join(only_paper)})")
    if only_sim:
        print(f"    (traded in replay only: {', '.join(only_sim)})")

    print("\n" + "=" * 70)
    if alerts:
        print(f"  RESULT: DRIFT DETECTED — {len(alerts)} metric(s) out of tolerance:")
        for a in alerts:
            print(f"    * {a}")
        print("  Investigate spreads/slippage, data gaps, or seedbox config mismatch.")
        print("=" * 70)
        return 1
    print("  RESULT: PASS — paper tracks replay within tolerance.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
