import sys, shutil
from pathlib import Path
sys.path.insert(0, "/home/user/kraken-momentum-trader")
from run_replay import ReplayKrakenClient, load_candles
from src.exchange.kraken_client import KrakenClient
from src.observability.logger import configure_logging
import os
os.environ["KRAKEN_API_KEY"] = "replay-test"
os.environ["KRAKEN_API_SECRET"] = "replay-test"
from run_scalping_live import ScalpingTrader

configure_logging(level="ERROR", format_type="json")
interval = int(sys.argv[1])

for q in ["q4_2024", "q1_2025", "q1_2026_fresh"]:
    candle_data = load_candles(Path(f"data/{q}"))
    num = len(next(iter(candle_data.values())))
    d = Path("data/replay_cadence_test")
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)
    mock = ReplayKrakenClient(candle_data, capital=10000.0)
    orig = KrakenClient.__init__
    KrakenClient.__init__ = lambda self, **kw: None
    t = ScalpingTrader(config_path="config/scalping.yaml", data_dir=str(d), paper_trading=True)
    KrakenClient.__init__ = orig
    t.client = mock; t.capital = 10000.0; t.initial_capital = 10000.0
    t._regime_check_interval = interval
    flips = 0; prev = t._current_regime
    for i in range(num):
        mock._current_idx = i
        try: t._update_regime()
        except Exception: pass
        if t._current_regime != prev:
            flips += 1; prev = t._current_regime
        for p in t.pairs:
            try: t._process_pair(p)
            except Exception: pass
    tt = t.metrics['total_trades']; w = t.metrics['wins']
    print(f"{q}: P&L ${t.capital-10000:+,.2f}  trades={tt} WR={(w/tt*100) if tt else 0:.0f}%  regime_flips={flips}")
