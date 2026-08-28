#!/usr/bin/env bash
# Archive exact files from a plan to mgt02 with SHA-256 verification and local sidecars.
set -euo pipefail
umask 077

SOURCE_ROOT='/data/luolie/'
REMOTE_HOST='mgt02'
REMOTE_ROOT='/share/org/bd/bd_luolie'

usage() {
  cat <<'EOF'
Usage: archive_to_mgt02.sh --plan PLAN.tsv [--execute] [--delete-source] [--run-id ID]

Without --execute, only validates each plan entry and prints what would be archived.
--execute copies each valid file to mgt02 and verifies SHA-256, but keeps the source.
--execute --delete-source additionally deletes only fully verified sources and writes
<original filename>.moved.json next to each removed file.
EOF
}

plan=''
execute=0
delete_source=0
run_id="archive-$(date -u +%Y%m%dT%H%M%SZ)"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan) plan="${2:-}"; shift 2 ;;
    --execute) execute=1; shift ;;
    --delete-source) delete_source=1; shift ;;
    --run-id) run_id="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$plan" && -f "$plan" ]] || { echo "Missing plan: $plan" >&2; exit 2; }
[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid run id: $run_id" >&2; exit 2; }
if (( delete_source && ! execute )); then
  echo "--delete-source requires --execute" >&2
  exit 2
fi
command -v rsync >/dev/null || { echo 'rsync is required' >&2; exit 2; }
command -v sha256sum >/dev/null || { echo 'sha256sum is required' >&2; exit 2; }
command -v python3 >/dev/null || { echo 'python3 is required to create .moved.json safely' >&2; exit 2; }
ssh -n -o BatchMode=yes "$REMOTE_HOST" 'test -d /share/org/bd/bd_luolie && test -w /share/org/bd/bd_luolie' || {
  echo "mgt02 is unavailable or not writable" >&2
  exit 2
}

remote_quote() { printf '%q' "$1"; }
remote_sha256() {
  ssh -n -o BatchMode=yes "$REMOTE_HOST" "sha256sum -- $(remote_quote "$1")" | awk '{print $1}'
}
remote_exists() {
  ssh -n -o BatchMode=yes "$REMOTE_HOST" "test -f $(remote_quote "$1")"
}
remote_mkdir() {
  ssh -n -o BatchMode=yes "$REMOTE_HOST" "mkdir -p -- $(remote_quote "$1")"
}
remote_promote() {
  local staged="$1" final="$2"
  ssh -n -o BatchMode=yes "$REMOTE_HOST" "set -e; mkdir -p -- $(remote_quote "$(dirname -- "$final")"); test ! -e $(remote_quote "$final"); mv -- $(remote_quote "$staged") $(remote_quote "$final")"
}

total=0
preflight_ok=0
archived=0
failed=0
skipped=0

while IFS=$'\t' read -r expected_bytes expected_mtime source; do
  [[ -n "${expected_bytes:-}" ]] || continue
  [[ "$expected_bytes" == \#* ]] && continue
  total=$((total + 1))

  if [[ "$source" != "$SOURCE_ROOT"* || ! -f "$source" ]]; then
    echo "SKIP missing_or_outside_source: $source" >&2
    skipped=$((skipped + 1))
    continue
  fi
  if [[ -e "${source}.moved.json" ]]; then
    echo "SKIP sidecar_already_exists: $source" >&2
    skipped=$((skipped + 1))
    continue
  fi

  actual_bytes="$(stat -c '%s' -- "$source")"
  actual_mtime="$(stat -c '%Y' -- "$source")"
  if [[ "$actual_bytes" != "$expected_bytes" || "$actual_mtime" != "$expected_mtime" ]]; then
    echo "SKIP changed_since_plan: $source" >&2
    skipped=$((skipped + 1))
    continue
  fi
  preflight_ok=$((preflight_ok + 1))
  relative="${source#"$SOURCE_ROOT"}"
  remote_final="$REMOTE_ROOT/$relative"
  remote_ref="$REMOTE_HOST:$remote_final"
  remote_stage="$REMOTE_ROOT/.incoming/$run_id/$relative"

  if (( ! execute )); then
    printf 'PREFLIGHT_OK\t%s\t%s\n' "$actual_bytes" "$source"
    continue
  fi

  echo "HASH_SOURCE $source" >&2
  source_hash_before="$(sha256sum -- "$source" | awk '{print $1}')"
  if remote_exists "$remote_final"; then
    echo "VERIFY_EXISTING_REMOTE $remote_ref" >&2
    remote_hash="$(remote_sha256 "$remote_final")"
    if [[ "$remote_hash" != "$source_hash_before" ]]; then
      echo "FAIL remote_conflict_hash_mismatch: $source" >&2
      failed=$((failed + 1))
      continue
    fi
  else
    remote_mkdir "$(dirname -- "$remote_stage")"
    echo "RSYNC $source -> $remote_ref" >&2
    if ! rsync -aH --protect-args --partial --append-verify -- "$source" "$REMOTE_HOST:$remote_stage"; then
      echo "FAIL rsync: $source" >&2
      failed=$((failed + 1))
      continue
    fi
    echo "VERIFY_STAGED_REMOTE $remote_ref" >&2
    remote_hash="$(remote_sha256 "$remote_stage")"
    if [[ "$remote_hash" != "$source_hash_before" ]]; then
      echo "FAIL staged_hash_mismatch: $source" >&2
      failed=$((failed + 1))
      continue
    fi
    if ! remote_promote "$remote_stage" "$remote_final"; then
      echo "FAIL promote_to_final: $source" >&2
      failed=$((failed + 1))
      continue
    fi
    remote_hash="$(remote_sha256 "$remote_final")"
    if [[ "$remote_hash" != "$source_hash_before" ]]; then
      echo "FAIL final_hash_mismatch: $source" >&2
      failed=$((failed + 1))
      continue
    fi
  fi

  source_hash_after="$(sha256sum -- "$source" | awk '{print $1}')"
  actual_bytes="$(stat -c '%s' -- "$source")"
  actual_mtime="$(stat -c '%Y' -- "$source")"
  if [[ "$source_hash_after" != "$source_hash_before" || "$actual_bytes" != "$expected_bytes" || "$actual_mtime" != "$expected_mtime" ]]; then
    echo "SKIP source_changed_during_archive: $source" >&2
    skipped=$((skipped + 1))
    continue
  fi

  if (( ! delete_source )); then
    printf 'COPIED_AND_VERIFIED_SOURCE_RETAINED\t%s\t%s\n' "$actual_bytes" "$source"
    continue
  fi

  marker="${source}.moved.json"
  marker_tmp="${marker}.tmp.${run_id}.${BASHPID}"
  restore_command="rsync -aH --protect-args $(printf '%q' "$remote_ref") $(printf '%q' "$source")"
  python3 - "$marker_tmp" "$source" "$remote_ref" "$actual_bytes" "$source_hash_before" "$run_id" "$restore_command" <<'PY'
import datetime
import json
import os
import sys

(
    marker_tmp,
    original_path,
    remote,
    bytes_text,
    sha256,
    archive_run_id,
    restore_command,
) = sys.argv[1:]
payload = {
    "schema": "remote-storage-v1",
    "original_path": original_path,
    "remote": remote,
    "bytes": int(bytes_text),
    "sha256": sha256,
    "state": "archived_verified_source_deleted",
    "local_present": False,
    "moved_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "archive_run_id": archive_run_id,
    "verification": {
        "algorithm": "sha256",
        "source_sha256": sha256,
        "remote_sha256": sha256,
    },
    "restore_command": restore_command,
}
with open(marker_tmp, "x", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
  if ! rm -- "$source"; then
    echo "FAIL source_delete_failed; final sidecar not published: $source" >&2
    failed=$((failed + 1))
    continue
  fi
  mv -- "$marker_tmp" "$marker"
  printf 'ARCHIVED_AND_SOURCE_DELETED\t%s\t%s\t%s\n' "$actual_bytes" "$source_hash_before" "$source"
  archived=$((archived + 1))
done < "$plan"

printf 'SUMMARY\ttotal=%d\tpreflight_ok=%d\tarchived=%d\tfailed=%d\tskipped=%d\n' "$total" "$preflight_ok" "$archived" "$failed" "$skipped"
(( failed == 0 ))
