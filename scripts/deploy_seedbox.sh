#!/usr/bin/env bash
# =============================================================================
# deploy_seedbox.sh — the ONLY supported way to deploy the trader.
#
# Every manual deploy so far has broken something the chat-pasted commands
# could not catch: a stale process kept serving the dashboard, a phantom
# --paper flag, diverged git history, fresh-start crashes. This script encodes
# those lessons:
#
#   1. fetch + reset --hard        (immune to the diverged-history fast-forward
#                                   failure of 2026-06-12)
#   2. PREFLIGHT GATE              run_preflight.py exercises the real stack;
#                                   deploy aborts on any failure
#   3. pkill by script name        kills the trader no matter how it was
#                                   started (screen/tmux/nohup) — the June 11
#                                   deploy left a 38-day-old process running
#   4. port-free verification      a bound 44485 means something survived
#   5. screen session start        detached, logs tee'd to logs/
#   6. post-start verification     process alive + "Cycle complete" in log +
#                                   dashboard HTTP 200 + zero ERROR log lines
#
# Usage:
#   ./scripts/deploy_seedbox.sh                 # deploy current branch
#   ./scripts/deploy_seedbox.sh --fresh         # also wipe data/scalping state
#   ./scripts/deploy_seedbox.sh --quick         # faster preflight (~30s)
#   DEPLOY_BRANCH=main ./scripts/deploy_seedbox.sh   # deploy another branch
# =============================================================================
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."
BRANCH="${DEPLOY_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
PORT=44485
SESSION=trader
FRESH=0
PREFLIGHT_ARGS=""
for arg in "$@"; do
    case "$arg" in
        --fresh) FRESH=1 ;;
        --quick) PREFLIGHT_ARGS="--quick" ;;
        *) echo "unknown arg: $arg (valid: --fresh --quick)"; exit 2 ;;
    esac
done

step() { echo; echo "==== [$1] $2"; }

step 1/6 "Sync to origin/$BRANCH"
git fetch origin "$BRANCH"
git reset --hard "origin/$BRANCH"
echo "  at commit: $(git log --oneline -1)"

step 2/6 "Preflight gate (real-stack e2e — aborts deploy on failure)"
python3 run_preflight.py $PREFLIGHT_ARGS

step 3/6 "Stop old trader (any launcher: screen/tmux/nohup)"
pkill -f run_scalping_live.py 2>/dev/null && echo "  killed running trader" || echo "  none running"
screen -S "$SESSION" -X quit 2>/dev/null || true
sleep 2
if pgrep -f run_scalping_live.py >/dev/null 2>&1; then
    echo "  FATAL: old trader still alive after pkill:"
    pgrep -af run_scalping_live.py
    exit 1
fi

step 4/6 "Verify port $PORT is free"
python3 - "$PORT" <<'EOF'
import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
except OSError:
    print(f"  FATAL: port {sys.argv[1]} still bound — a stale process is holding it")
    sys.exit(1)
finally:
    s.close()
print("  port free")
EOF

if [ "$FRESH" -eq 1 ]; then
    echo "  --fresh: wiping data/scalping (old paper history erased)"
    rm -rf data/scalping
fi
mkdir -p logs

step 5/6 "Start trader in screen session '$SESSION'"
LOG="logs/trader_$(date +%Y%m%d_%H%M%S).log"
screen -dmS "$SESSION" bash -c "cd '$PWD' && python3 run_scalping_live.py 2>&1 | tee -a '$LOG'"
echo "  log: $LOG"

step 6/6 "Verify startup (waiting for first full cycle, up to 180s)"
ok=0
for i in $(seq 1 90); do
    sleep 2
    if ! pgrep -f run_scalping_live.py >/dev/null 2>&1; then
        echo "  FATAL: trader process died during startup. Last log lines:"
        tail -20 "$LOG" 2>/dev/null || true
        exit 1
    fi
    if grep -q "Cycle complete" "$LOG" 2>/dev/null; then
        ok=1
        break
    fi
done
if [ "$ok" -ne 1 ]; then
    echo "  FATAL: no 'Cycle complete' within 180s. Last log lines:"
    tail -20 "$LOG" 2>/dev/null || true
    exit 1
fi
echo "  first cycle complete"

if command -v curl >/dev/null 2>&1; then
    if curl -fsS -o /dev/null --max-time 10 "http://127.0.0.1:$PORT/"; then
        echo "  dashboard responds: HTTP 200"
    else
        echo "  FATAL: dashboard not responding on :$PORT"
        exit 1
    fi
fi

ERRORS=$(grep -c '"level": "ERROR"' "$LOG" 2>/dev/null || true)
ERRORS=${ERRORS:-0}
if [ "$ERRORS" -gt 0 ]; then
    echo "  FATAL: $ERRORS ERROR line(s) in the first cycle:"
    grep '"level": "ERROR"' "$LOG" | head -5
    exit 1
fi
echo "  zero ERROR lines in first cycle"

echo
echo "=============================================================="
echo "  DEPLOY OK — commit $(git rev-parse --short HEAD) running in screen '$SESSION'"
echo "  watch:  screen -r $SESSION   (Ctrl-A D to detach)"
echo "  log:    tail -f $LOG"
echo "  web:    http://<seedbox>:$PORT"
echo "=============================================================="
