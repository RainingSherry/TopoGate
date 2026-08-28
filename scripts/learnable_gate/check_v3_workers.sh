#!/usr/bin/env bash
# Check v3 smoke workers progress
LOGDIR=${1:-/tmp/topogate_v3_worker_*}
if [ -z "$1" ]; then
    # Find latest
    LOGDIR=$(ls -td /tmp/topogate_v3_worker_* 2>/dev/null | head -1)
fi
echo "=== Monitoring $LOGDIR ==="
for w in 0 1 2; do
    if [ -f "$LOGDIR/w$w.log" ]; then
        echo ""
        echo "--- Worker $w (tail of $LOGDIR/w$w.log) ---"
        tail -n 3 "$LOGDIR/w$w.log"
        echo "    Total runs so far: $(grep -c '=== ' "$LOGDIR/w$w.log" 2>/dev/null || echo 0)"
    fi
done
echo ""
echo "=== Current results.csv (count rows) ==="
wc -l result/learnable_gate_smoke/v3_smoke/results.csv 2>/dev/null || echo "results.csv not created yet"
