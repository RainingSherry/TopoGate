#!/usr/bin/env bash
# Build a non-destructive, auditable candidate plan for large model/result files.
set -euo pipefail
umask 077

usage() {
  cat <<'EOF'
Usage: make_archive_plan.sh --output PLAN.tsv ROOT [ROOT ...]

The plan has tab-separated: bytes, mtime_epoch, absolute_source_path.
It includes every regular file >=1 GiB plus recognized model/array files >=100 MiB.
The plan is a candidate snapshot only; it never moves or deletes data.
EOF
}

output=""
if [[ "${1:-}" == "--output" ]]; then
  [[ $# -ge 3 ]] || { usage >&2; exit 2; }
  output="$2"
  shift 2
else
  usage >&2
  exit 2
fi

[[ "$output" = /* ]] || { echo "Plan path must be absolute: $output" >&2; exit 2; }
[[ $# -gt 0 ]] || { usage >&2; exit 2; }

for root in "$@"; do
  [[ "$root" == /data/luolie/* ]] || { echo "Refusing root outside /data/luolie: $root" >&2; exit 2; }
  [[ -d "$root" ]] || { echo "Missing directory: $root" >&2; exit 2; }
done

mkdir -p -- "$(dirname -- "$output")"
tmp="$(mktemp "${output}.tmp.XXXXXX")"
trap 'rm -f -- "$tmp"' EXIT
printf '# remote-storage-plan-v1\n' > "$tmp"

find "$@" -xdev -type f \( \
  -size +1G -o \
  \( -size +100M \( \
    -iname '*.pt' -o -iname '*.pth' -o -iname '*.ckpt' -o -iname '*.safetensors' -o \
    -iname '*.bin' -o -iname '*.h5' -o -iname '*.hdf5' -o -iname '*.onnx' -o \
    -iname '*.npy' -o -iname '*.npz' -o -iname '*.pkl' -o -iname '*.pickle' -o -iname '*.joblib' \
  \) \) \
\) -printf '%s\t%T@\t%p\n' | \
  awk -F '\t' '{ printf "%s\t%d\t%s\n", $1, $2, $3 }' | \
  sort -n -k1,1 -k3,3 >> "$tmp"

mv -- "$tmp" "$output"
trap - EXIT
awk -F '\t' 'NR > 1 { bytes += $1; count += 1 } END { printf "plan=%s\nfiles=%d\nbytes=%d\n", FILENAME, count, bytes }' "$output"
