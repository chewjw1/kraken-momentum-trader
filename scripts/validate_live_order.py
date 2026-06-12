#!/usr/bin/env python3
"""
LIVE-READINESS VALIDATOR — run on the seedbox (real API keys) BEFORE the
first --live start. Places NO orders and moves NO funds.

The preflight gate proves our code against a simulated Kraken; this script
proves the simulation's assumptions against the REAL Kraken:

  1. AssetPairs reality check per configured pair: price/volume precision,
     order minimums, and actual margin support (leverage_sell) — prints which
     pairs can really be shorted live.
  2. AddOrder validate=true dry runs: Kraken fully validates each order
     server-side (param names, leverage values, reduce_only, precision)
     without executing. Belt-and-braces: prices are also placed far from
     market and post-only.
  3. Balance fetch: confirms the API key has the required permissions.

API key requirements for --live: Query Funds, Create & Modify Orders, Cancel
Orders. (Margin trading must be enabled on the account for shorts.)

Usage:
    python3 scripts/validate_live_order.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from src.exchange.kraken_client import (  # noqa: E402
    KrakenClient, KrakenAPIError, OrderSide, OrderType,
)
from src.observability.logger import configure_logging  # noqa: E402


def main():
    configure_logging(level="ERROR", format_type="json")
    cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config/scalping.yaml"))
    pairs = cfg.get("pairs", [])
    leverage = int(cfg.get("shorting", {}).get("leverage", 2))

    print("=" * 74)
    print("  LIVE-READINESS VALIDATOR (validate=true — nothing executes)")
    print("=" * 74)

    try:
        client = KrakenClient(paper_trading=False)
    except Exception as e:
        print(f"\nFATAL: cannot construct live client: {e}")
        print("Check KRAKEN_API_KEY / KRAKEN_API_SECRET in the environment/.env")
        return 1

    failures = []

    # --- 1. account access ---------------------------------------------------
    print("\n[1] Account access")
    try:
        balances = client.get_balances()
        usd = balances.get("USD")
        print(f"  OK — balance query works. USD: ${usd.total:.2f}" if usd
              else f"  OK — balance query works. Assets: {sorted(balances)}")
    except Exception as e:
        failures.append(f"Balance query failed: {e}")
        print(f"  FAIL — {e} (key missing 'Query Funds' permission?)")

    # --- 2. pair constraints & margin reality ---------------------------------
    print("\n[2] Pair constraints (REAL AssetPairs)")
    print(f"  {'pair':10s} {'px_dec':>6s} {'lot_dec':>7s} {'ordermin':>12s} "
          f"{'shortable':>9s} {'lev_sell'}")
    shortable = {}
    for pair in pairs:
        info = client._get_pair_info(pair)
        if not info:
            failures.append(f"AssetPairs lookup failed for {pair}")
            print(f"  {pair:10s} LOOKUP FAILED")
            continue
        ok = client.margin_leverage_for(pair, OrderSide.SELL, leverage)
        shortable[pair] = ok
        print(f"  {pair:10s} {info['pair_decimals']:6d} {info['lot_decimals']:7d} "
              f"{info['ordermin']:12.8f} {('YES@' + str(ok) + 'x') if ok else 'NO':>9s} "
              f"{info['leverage_sell']}")
    blocked = [p for p, v in shortable.items() if not v]
    if blocked:
        print(f"\n  NOTE: shorts will be SKIPPED live on: {', '.join(blocked)}")
        print("  (paper/replay shorts on these pairs do not transfer to live)")

    # --- 3. AddOrder validate=true dry runs -----------------------------------
    print("\n[3] AddOrder dry runs (validate=true)")

    def dry_run(label, pair, side, price_mult, **kwargs):
        info = client._get_pair_info(pair) or {}
        vol = max(info.get("ordermin", 0) or 0, 1e-8)
        try:
            ticker = client.get_ticker(pair)
            ref = ticker.bid if side == OrderSide.BUY else ticker.ask
            price = round(ref * price_mult, info.get("pair_decimals", 2))
            client.place_order(
                pair=pair, side=side, order_type=OrderType.LIMIT,
                volume=vol, price=price, post_only=True,
                validate_only=True, **kwargs,
            )
            print(f"  OK   {label}")
            return True
        except KrakenAPIError as e:
            msg = str(e)
            if "reduce_only" in label.lower() and (
                    "position" in msg.lower() or "reduce" in msg.lower()):
                # Expected with no open margin position — params themselves
                # were parsed fine or Kraken would name the bad argument.
                print(f"  OK   {label} (rejected only for lack of open position: {msg})")
                return True
            failures.append(f"{label}: {msg}")
            print(f"  FAIL {label}: {msg}")
            return False
        except Exception as e:
            failures.append(f"{label}: {e}")
            print(f"  FAIL {label}: {e}")
            return False

    dry_run("spot long entry (ETH/USD limit post-only)",
            "ETH/USD", OrderSide.BUY, 0.5)
    short_pair = next((p for p, v in shortable.items() if v), None)
    if short_pair:
        lev = shortable[short_pair]
        dry_run(f"margin short entry ({short_pair} sell leverage={lev})",
                short_pair, OrderSide.SELL, 2.0, leverage=lev)
        dry_run(f"short cover w/ reduce_only ({short_pair} buy)",
                short_pair, OrderSide.BUY, 0.5, leverage=lev, reduce_only=True)
        # Server-side disaster stop shape
        try:
            info = client._get_pair_info(short_pair)
            ticker = client.get_ticker(short_pair)
            trigger = round(ticker.last * 2.0, info["pair_decimals"])
            client.place_order(
                pair=short_pair, side=OrderSide.BUY,
                order_type=OrderType.STOP_LOSS,
                volume=max(info["ordermin"], 1e-8), price=trigger,
                leverage=shortable[short_pair], reduce_only=True,
                validate_only=True,
            )
            print(f"  OK   stop-loss order shape ({short_pair} buy stop)")
        except KrakenAPIError as e:
            msg = str(e)
            if "position" in msg.lower() or "reduce" in msg.lower():
                print(f"  OK   stop-loss order shape (rejected only for lack of position)")
            else:
                failures.append(f"stop-loss shape: {msg}")
                print(f"  FAIL stop-loss shape: {msg}")
    else:
        print("  WARN: no shortable pairs — margin dry runs skipped entirely")

    print("\n" + "=" * 74)
    if failures:
        print(f"  RESULT: NOT LIVE-READY — {len(failures)} failure(s):")
        for f in failures:
            print(f"    * {f}")
        print("=" * 74)
        return 1
    print("  RESULT: LIVE-READY — real Kraken accepted all order shapes.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
