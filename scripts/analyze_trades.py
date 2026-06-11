#!/usr/bin/env python3
"""
Trade-level forensics across replay quarters.

Replays each quarter's 4h OHLCV through the REAL ScalpingTrader (same flow
as run_replay.py) and captures every closed trade with full context:
pair, side, entry/exit timestamps+prices, P&L, hold time, regime at
entry/exit, and the actual exit reason (captured from the CLOSED log call).

Then prints aggregate tables:
  - P&L by side per quarter
  - P&L by regime-at-entry per quarter
  - P&L by pair per quarter
  - P&L by exit-reason category per quarter
  - Loss distribution histogram + clipped-loss counterfactuals
  - Top 10 worst trades
  - Hold time winners vs losers, per side
  - Regime-transition analysis (trades opened just before a regime flip,
    and trades held through a regime change)

Usage (from repo root):
    python3 scripts/analyze_trades.py
"""

import sys
import csv
import shutil
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# run_replay sets KRAKEN_API_KEY/SECRET env vars on import
from run_replay import ReplayKrakenClient, load_candles  # noqa: E402
from src.exchange.kraken_client import KrakenClient  # noqa: E402
from src.observability.logger import configure_logging  # noqa: E402
from run_scalping_live import ScalpingTrader  # noqa: E402

QUARTERS = [
    ("q4_2024", "data/q4_2024", 902.0),
    ("q1_2025", "data/q1_2025", -1707.0),
    ("q1_2026_fresh", "data/q1_2026_fresh", 2667.0),
]

INITIAL_CAPITAL = 10000.0
PCT_BUCKETS = [
    ("< -5%", -float("inf"), -5.0),
    ("-5..-3%", -5.0, -3.0),
    ("-3..-1%", -3.0, -1.0),
    ("-1..0%", -1.0, 0.0),
    ("0..1%", 0.0, 1.0),
    ("1..3%", 1.0, 3.0),
    ("3..5%", 3.0, 5.0),
    ("> 5%", 5.0, float("inf")),
]


def classify_exit(reason: str, pnl_pct: float) -> str:
    """Map the strategy's exit-reason string to a category."""
    if reason:
        r = reason.lower()
        if r.startswith("take profit") or r.startswith("short tp"):
            return "TP"
        if r.startswith("stop loss") or r.startswith("short sl"):
            return "STOP"
        if "trailing stop" in r:
            return "TRAIL"
        return "SIGNAL"
    # Fallback inference if reason wasn't captured
    if pnl_pct <= -2.5:
        return "STOP(inferred)"
    if pnl_pct >= 4.0:
        return "TP(inferred)"
    return "SIGNAL(inferred)"


def run_quarter(name: str, data_dir: str):
    """Replay one quarter through the real trader; return (trades, summary)."""
    candle_data = load_candles(Path(REPO_ROOT / data_dir))
    num_candles = len(next(iter(candle_data.values())))

    replay_dir = REPO_ROOT / "data" / "replay_analysis" / name
    if replay_dir.exists():
        shutil.rmtree(replay_dir)
    replay_dir.mkdir(parents=True, exist_ok=True)

    mock_client = ReplayKrakenClient(candle_data, capital=INITIAL_CAPITAL)

    orig_init = KrakenClient.__init__
    KrakenClient.__init__ = lambda self, **kwargs: None
    try:
        trader = ScalpingTrader(
            config_path=str(REPO_ROOT / "config" / "scalping.yaml"),
            data_dir=str(replay_dir),
            paper_trading=True,
        )
    finally:
        KrakenClient.__init__ = orig_init

    trader.client = mock_client
    trader.capital = INITIAL_CAPITAL
    trader.initial_capital = INITIAL_CAPITAL

    trades = []           # one dict per closed trade
    open_meta = {}        # pair -> {"entry_idx": int}
    regime_changes = []   # (candle_idx, from, to)
    ctx = {"idx": 0}
    last_trade_by_pair = {}

    # --- Hook 1: capture every closed trade at record_trade time.
    # At this moment trader.positions[pair] still exists (deleted later),
    # so we can read the regime stored at entry.
    orig_record = trader.pair_manager.record_trade

    def record_trade(pair, entry_time, exit_time, pnl, pnl_percent,
                     entry_price=0.0, exit_price=0.0, size_usd=0.0,
                     side="long"):
        pos = trader.positions.get(pair, {})
        meta = open_meta.get(pair, {})
        entry_idx = meta.get("entry_idx")
        exit_idx = ctx["idx"]
        pk = pair.replace("/", "_")
        candles = candle_data.get(pk, [])

        def ts(idx):
            if idx is not None and 0 <= idx < len(candles):
                return candles[idx].timestamp.strftime("%Y-%m-%d %H:%M")
            return "?"

        trade = {
            "quarter": name,
            "pair": pair,
            "side": side,
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "entry_ts": ts(entry_idx),
            "exit_ts": ts(exit_idx),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_usd": pnl,
            "pnl_pct": pnl_percent,
            "size_usd": size_usd,
            "hold_candles": (exit_idx - entry_idx) if entry_idx is not None else None,
            "regime_entry": pos.get("regime", "unknown"),
            "regime_exit": trader._current_regime.value,
            "exit_reason": None,
        }
        trades.append(trade)
        last_trade_by_pair[pair] = trade
        return orig_record(
            pair=pair, entry_time=entry_time, exit_time=exit_time,
            pnl=pnl, pnl_percent=pnl_percent, entry_price=entry_price,
            exit_price=exit_price, size_usd=size_usd, side=side,
        )

    trader.pair_manager.record_trade = record_trade

    # --- Hook 2: capture the exit reason from the "CLOSED <pair>" log call,
    # which fires right after record_trade in _process_pair.
    orig_info = trader.logger.info

    def info_hook(msg, **kwargs):
        if isinstance(msg, str) and msg.startswith("CLOSED "):
            pair = msg.split()[1]
            t = last_trade_by_pair.get(pair)
            if t is not None and t["exit_reason"] is None:
                t["exit_reason"] = kwargs.get("reason")
        return orig_info(msg, **kwargs)

    trader.logger.info = info_hook

    # --- Candle loop (same flow as run_replay.py) ---
    errors = 0
    try:
        for i in range(num_candles):
            mock_client._current_idx = i
            ctx["idx"] = i

            prev_regime = trader._current_regime
            prev_positions = set(trader.positions.keys())

            try:
                trader._update_regime()
            except Exception:
                errors += 1
            if trader._current_regime != prev_regime:
                regime_changes.append(
                    (i, prev_regime.value, trader._current_regime.value))

            for pair in trader.pairs:
                try:
                    trader._process_pair(pair)
                except Exception as e:
                    errors += 1
                    print(f"  ERROR {name} candle {i} {pair}: {e}")

            for pair in set(trader.positions.keys()) - prev_positions:
                open_meta[pair] = {"entry_idx": i}
    finally:
        # logger object is cached per module name and reused across quarters
        trader.logger.info = orig_info

    # Finalize exit reason categories
    for t in trades:
        t["exit_cat"] = classify_exit(t["exit_reason"], t["pnl_pct"])

    summary = {
        "name": name,
        "num_candles": num_candles,
        "final_capital": trader.capital,
        "total_pnl": trader.capital - INITIAL_CAPITAL,
        "metrics": dict(trader.metrics),
        "open_positions": {
            p: {"side": pos.get("side"), "entry_price": pos.get("entry_price"),
                "size_usd": pos.get("size_usd"), "regime": pos.get("regime")}
            for p, pos in trader.positions.items()
        },
        "regime_changes": regime_changes,
        "errors": errors,
    }
    return trades, summary


# ---------------------------------------------------------------------------
# Aggregation / reporting helpers
# ---------------------------------------------------------------------------

def agg(trades, key_fn):
    out = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        k = key_fn(t)
        out[k]["n"] += 1
        out[k]["pnl"] += t["pnl_usd"]
        if t["pnl_usd"] > 0:
            out[k]["wins"] += 1
    return out


def fmt_agg_row(label, d):
    wr = d["wins"] / d["n"] * 100 if d["n"] else 0.0
    return f"    {label:<14s} {d['n']:>4d} trades  {wr:>5.1f}% WR  ${d['pnl']:>+10.2f}"


def main():
    configure_logging(level="WARNING", format_type="json")

    all_trades = []
    summaries = {}
    for name, data_dir, expected in QUARTERS:
        print(f"\n=== Replaying {name} ({data_dir}) ===")
        trades, summary = run_quarter(name, data_dir)
        all_trades.extend(trades)
        summaries[name] = summary
        closed_pnl = sum(t["pnl_usd"] for t in trades)
        print(f"  Final capital ${summary['final_capital']:.2f} | "
              f"closed trades {len(trades)} | closed P&L ${closed_pnl:+.2f} | "
              f"expected ~${expected:+.0f} | open at end: "
              f"{len(summary['open_positions'])} | errors: {summary['errors']}")

    # Dump per-trade CSV for reuse
    out_csv = REPO_ROOT / "data" / "replay_analysis" / "trades_all.csv"
    fields = ["quarter", "pair", "side", "entry_ts", "exit_ts", "entry_idx",
              "exit_idx", "hold_candles", "entry_price", "exit_price",
              "size_usd", "pnl_usd", "pnl_pct", "regime_entry", "regime_exit",
              "exit_cat", "exit_reason"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_trades)
    print(f"\nPer-trade CSV written to {out_csv}")

    qnames = [q[0] for q in QUARTERS]

    # ---- Sanity check ----
    print("\n" + "=" * 78)
    print("SANITY CHECK vs known finals")
    print("=" * 78)
    for name, _, expected in QUARTERS:
        s = summaries[name]
        closed = sum(t["pnl_usd"] for t in all_trades if t["quarter"] == name)
        flag = "" if abs(s["total_pnl"] - expected) < 100 else "  <-- LARGE DIFF"
        print(f"  {name:<14s} replay P&L ${s['total_pnl']:>+9.2f}  "
              f"closed-trade sum ${closed:>+9.2f}  expected ${expected:>+9.2f}{flag}")
        if s["open_positions"]:
            for p, pos in s["open_positions"].items():
                print(f"      still open: {p} {pos['side']} @ {pos['entry_price']:.4f} (${pos['size_usd']:.0f})")

    # ---- (a) P&L by side per quarter ----
    print("\n" + "=" * 78)
    print("(a) P&L BY SIDE per quarter")
    print("=" * 78)
    for q in qnames:
        qt = [t for t in all_trades if t["quarter"] == q]
        print(f"  {q}:")
        for side, d in sorted(agg(qt, lambda t: t["side"]).items()):
            print(fmt_agg_row(side, d))
    print("  ALL QUARTERS:")
    for side, d in sorted(agg(all_trades, lambda t: t["side"]).items()):
        print(fmt_agg_row(side, d))

    # ---- (b) P&L by regime at entry per quarter ----
    print("\n" + "=" * 78)
    print("(b) P&L BY REGIME-AT-ENTRY per quarter")
    print("=" * 78)
    for q in qnames:
        qt = [t for t in all_trades if t["quarter"] == q]
        print(f"  {q}:")
        for key, d in sorted(agg(qt, lambda t: f"{t['regime_entry']}/{t['side']}").items()):
            print(fmt_agg_row(key, d))

    # ---- (c) P&L by pair per quarter ----
    print("\n" + "=" * 78)
    print("(c) P&L BY PAIR per quarter")
    print("=" * 78)
    pairs = sorted({t["pair"] for t in all_trades})
    header = f"  {'pair':<10s}" + "".join(f"{q:>22s}" for q in qnames) + f"{'TOTAL':>14s}"
    print(header)
    for p in pairs:
        row = f"  {p:<10s}"
        tot = 0.0
        for q in qnames:
            qt = [t for t in all_trades if t["quarter"] == q and t["pair"] == p]
            pnl = sum(t["pnl_usd"] for t in qt)
            tot += pnl
            row += f"{len(qt):>5d} tr ${pnl:>+9.2f}  "
        row += f"${tot:>+10.2f}"
        print(row)

    # ---- Exit reason categories ----
    print("\n" + "=" * 78)
    print("(extra) P&L BY EXIT-REASON CATEGORY per quarter")
    print("=" * 78)
    for q in qnames:
        qt = [t for t in all_trades if t["quarter"] == q]
        print(f"  {q}:")
        for key, d in sorted(agg(qt, lambda t: t["exit_cat"]).items()):
            print(fmt_agg_row(key, d))

    # ---- (d) Loss distribution + counterfactuals ----
    print("\n" + "=" * 78)
    print("(d) P&L% DISTRIBUTION (net of fees) per quarter")
    print("=" * 78)
    for q in qnames + ["ALL"]:
        qt = all_trades if q == "ALL" else [t for t in all_trades if t["quarter"] == q]
        print(f"  {q}:")
        for label, lo, hi in PCT_BUCKETS:
            bucket = [t for t in qt if lo <= t["pnl_pct"] < hi]
            pnl = sum(t["pnl_usd"] for t in bucket)
            print(f"    {label:<9s} {len(bucket):>4d} trades  ${pnl:>+10.2f}")

    print("\n  Damage from tail losses + clipped-loss counterfactuals:")
    for q in qnames:
        qt = [t for t in all_trades if t["quarter"] == q]
        actual = sum(t["pnl_usd"] for t in qt)
        worse4 = [t for t in qt if t["pnl_pct"] < -4.0]
        dmg4 = sum(t["pnl_usd"] for t in worse4)

        def clipped_total(cap):
            tot = 0.0
            for t in qt:
                if t["pnl_pct"] < -cap and t["size_usd"] > 0:
                    tot += t["size_usd"] * (-cap / 100.0)
                else:
                    tot += t["pnl_usd"]
            return tot

        print(f"  {q}: actual ${actual:>+9.2f} | trades < -4%: {len(worse4)} "
              f"totaling ${dmg4:>+9.2f} | if losses capped at -3%: "
              f"${clipped_total(3.0):>+9.2f} | at -4%: ${clipped_total(4.0):>+9.2f}")

    # ---- (e) Top 10 worst trades ----
    print("\n" + "=" * 78)
    print("(e) TOP 10 WORST TRADES (all quarters)")
    print("=" * 78)
    worst = sorted(all_trades, key=lambda t: t["pnl_usd"])[:10]
    for t in worst:
        print(f"  {t['quarter']:<13s} {t['pair']:<9s} {t['side']:<5s} "
              f"{t['entry_ts']} -> {t['exit_ts']} ({t['hold_candles']} candles) "
              f"entry {t['entry_price']:.4f} exit {t['exit_price']:.4f} "
              f"${t['pnl_usd']:+.2f} ({t['pnl_pct']:+.2f}%) "
              f"regime {t['regime_entry']}->{t['regime_exit']} [{t['exit_cat']}]")
        if t["exit_reason"]:
            print(f"      reason: {t['exit_reason']}")

    # ---- (f) Hold time analysis ----
    print("\n" + "=" * 78)
    print("(f) HOLD TIME (4h candles): winners vs losers, per side")
    print("=" * 78)

    def avg_hold(ts):
        hs = [t["hold_candles"] for t in ts if t["hold_candles"] is not None]
        return sum(hs) / len(hs) if hs else 0.0

    for q in qnames + ["ALL"]:
        qt = all_trades if q == "ALL" else [t for t in all_trades if t["quarter"] == q]
        print(f"  {q}:")
        for side in ("long", "short"):
            st = [t for t in qt if t["side"] == side]
            w = [t for t in st if t["pnl_usd"] > 0]
            l = [t for t in st if t["pnl_usd"] <= 0]
            print(f"    {side:<6s} winners: {len(w):>3d} avg {avg_hold(w):5.1f} candles | "
                  f"losers: {len(l):>3d} avg {avg_hold(l):5.1f} candles")

    # ---- (g) Regime-transition analysis ----
    print("\n" + "=" * 78)
    print("(g) REGIME-TRANSITION ANALYSIS")
    print("=" * 78)
    for q in qnames:
        s = summaries[q]
        qt = [t for t in all_trades if t["quarter"] == q]
        chg_idxs = [c[0] for c in s["regime_changes"]]
        print(f"  {q}: {len(s['regime_changes'])} regime changes")
        for idx, frm, to in s["regime_changes"]:
            print(f"      candle {idx:>3d}: {frm} -> {to}")
        pre = [t for t in qt if t["entry_idx"] is not None and
               any(0 < c - t["entry_idx"] <= 5 for c in chg_idxs)]
        pre_pnl = sum(t["pnl_usd"] for t in pre)
        pre_w = sum(1 for t in pre if t["pnl_usd"] > 0)
        print(f"    opened within 5 candles BEFORE a regime change: "
              f"{len(pre)} trades, {pre_w} wins, total ${pre_pnl:+.2f}")
        spanned = [t for t in qt if t["regime_entry"] != t["regime_exit"]]
        sp_pnl = sum(t["pnl_usd"] for t in spanned)
        sp_w = sum(1 for t in spanned if t["pnl_usd"] > 0)
        print(f"    held THROUGH a regime change (entry regime != exit regime): "
              f"{len(spanned)} trades, {sp_w} wins, total ${sp_pnl:+.2f}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
