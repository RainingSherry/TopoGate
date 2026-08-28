#!/usr/bin/env bash
set -euo pipefail

# GPU_POOL is intentionally mirrored from run_v10_multiseed.py.
GPU_POOL=(1 4 5)  # GPU 0 and GPU 7 are forbidden
WORKER_ID="${1:-0}"

if [[ "${WORKER_ID}" -lt 0 || "${WORKER_ID}" -ge "${#GPU_POOL[@]}" ]]; then
  echo "worker_id must be 0, 1, or 2" >&2
  exit 2
fi

exec python scripts/v10_reliable_graph/run_v10_multiseed.py --worker_id "${WORKER_ID}" "${@:2}"
