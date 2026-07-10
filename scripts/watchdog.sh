#!/usr/bin/env bash
# =============================================================================
# watchdog.sh — restart the trader if its process died (cron-driven).
#
# Restarts EXACTLY what was running before (same code, same mode read from
# state.json) — it deliberately does NOT git-pull or run the full deploy;
# unattended code changes have no place in a watchdog. Use deploy_seedbox.sh
# for real deploys.
#
# Install (crontab -e):
#   */5 * * * * /home/USER/kraken-momentum-trader/scripts/watchdog.sh >> /home/USER/kraken-momentum-trader/logs/watchdog.log 2>&1
#
# To STOP the bot on purpose without cron resurrecting it:
#   touch <repo>/DISABLE_WATCHDOG && screen -S trader -X quit
# Re-enable with: rm <repo>/DISABLE_WATCHDOG
# =============================================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# Operator kill switch — never fight a deliberate shutdown.
if [ -f DISABLE_WATCHDOG ]; then
    exit 0
fi

# Trader alive? Nothing to do.
if pgrep -f run_scalping_live.py >/dev/null 2>&1; then
    exit 0
fi

# Determine the mode of the LAST run from persisted state. If there is no
# state.json we cannot know whether the operator intended paper or live —
# refuse to guess (starting live unintentionally is unacceptable; starting
# paper when the operator thinks it's live is silently worse).
STATE=data/scalping/state.json
if [ ! -f "$STATE" ]; then
    echo "[$(stamp)] trader down but no $STATE — cannot infer mode, NOT restarting (deploy manually)"
    exit 1
fi

MODE_FLAG=""
MODE_NAME="PAPER"
if grep -q '"paper_trading": false' "$STATE"; then
    MODE_FLAG="--live"
    MODE_NAME="LIVE"
fi

mkdir -p logs
LOG="logs/trader_$(date +%Y%m%d_%H%M%S)_watchdog.log"
echo "[$(stamp)] trader DOWN — restarting [$MODE_NAME] at commit $(git rev-parse --short HEAD), log $LOG"

screen -dmS trader bash -c "cd '$PWD' && python3 run_scalping_live.py $MODE_FLAG 2>&1 | tee -a '$LOG'"

# Verify it came up (startup reconciliation can take a bit in live mode).
sleep 30
if pgrep -f run_scalping_live.py >/dev/null 2>&1; then
    echo "[$(stamp)] restart OK [$MODE_NAME]"
else
    echo "[$(stamp)] RESTART FAILED — last log lines:"
    tail -20 "$LOG" 2>/dev/null || true
    exit 1
fi
