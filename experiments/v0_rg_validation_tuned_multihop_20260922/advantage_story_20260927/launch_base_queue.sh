#!/usr/bin/env bash
# Run manually on yazhouwan only after REVIEW_APPROVED.json is reviewed and saved.
set -euo pipefail
cd "$(dirname "$0")"
PY=/data/luolie/conda/envs/scssl_bench_py310/bin/python
"$PY" -c 'import controller; controller.verify_approval(controller.load_queue()); controller.load_resources(); print("Frozen queue and source hashes verified")'
mkdir -p logs
nohup "$PY" controller.py run > logs/controller.stdout.log 2>&1 < /dev/null &
printf '%s\n' "$!" > logs/controller.pid
printf 'Controller PID %s; log %s\n' "$!" "$PWD/logs/controller.stdout.log"
