"""Independent-dataset availability / provenance audit (P2), plus baseline runtime audit.

Read-only and label-meaning-only: no fitting, no label values are used, no
feature arrays are printed. Files are opened with ``h5py`` or
``np.load(allow_pickle=False)`` only; object/pickled payloads are refused and
reported rather than unpickled. Independence is never asserted from names:
every candidate ends in ``available`` / ``overlap_detected`` / ``overlap_unknown``
per comparison, with the evidence recorded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import socket
import sys
import time
from typing import Any, Iterable

import numpy as np


PROTOCOL = "source_coverage_extension_audit_v1"
CANDIDATES = {
    "Pollen": r"pollen",
    "Wang": r"(^|[^a-z])wang",
    "hrvatin": r"hrvatin",
    "Shekhar": r"shekhar",
    "BaronHuman": r"baron.*human|human.*baron",
}
DEFAULT_PROFILE_DIR = "<PROJECT_ROOT>/datasets/eswa_prepared_v1"
DEFAULT_NPZ_DIR = "<PROJECT_ROOT>/datasets"
DEFAULT_REPO_ROOTS = ("<PROJECT_ROOT>", "<REMOTE_PATH>")
PACKAGES = ("numpy", "scipy", "h5py", "anndata", "scanpy", "torch", "sklearn",
            "tensorflow", "keras", "pandas")
# Metadata column names that describe study structure; values are summarized, never used.
META_COLUMN_PATTERN = re.compile(r"donor|patient|individual|batch|sample|study|accession|geo|"
                                 r"plate|run|library|tissue|organism|species|age|sex|condition",
                                 re.IGNORECASE)
LABEL_COLUMN_PATTERN = re.compile(r"cell.?type|celltype|label|cluster|annotation|class", re.IGNORECASE)
PROVENANCE_KEY_PATTERN = re.compile(r"source|path|accession|geo|doi|study|url|citation|origin|"
                                    r"label|species|organism|protocol|platform|sha|hash|gene|cell",
                                    re.IGNORECASE)
MAX_REPO_DEPTH = 3
MAX_REPO_ENTRIES = 20000
SCHNTL = {
    "repository": "https://github.com/SWJTU-ML/scHNTL-code",
    "paper_doi": "https://doi.org/10.1093/bioinformatics/btaf044",
    "readme_requirements_verified_by_parent": {"python": "3.6.4", "tensorflow": "1.12", "keras": "2.1.5"},
    "documented_outputs": ["hidden_DatasetX.tsv", "pred_DatasetX.txt"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8", "surrogatepass"))
        digest.update(b"\0")
    return digest.hexdigest()


def _decode(values) -> list[str]:
    return [value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
            for value in np.asarray(values).reshape(-1).tolist()]


def _small_scalar(value: Any):
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:500] + "...<truncated>"
    return None


def candidate_names(name: str) -> list[str]:
    lowered = name.lower()
    return [candidate for candidate, pattern in CANDIDATES.items() if re.search(pattern, lowered)]


# ------------------------------------------------------------------ readers

def read_profile(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"path": str(path), "readable": False, "error": repr(error)}
    flat = {}

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for key, item in node.items():
                walk(item, f"{prefix}{key}.")
        elif isinstance(node, list):
            if len(node) <= 20 and all(_small_scalar(item) is not None or item is None for item in node):
                flat[prefix[:-1]] = [_small_scalar(item) for item in node]
            else:
                flat[prefix[:-1]] = f"<list len={len(node)}>"
        else:
            flat[prefix[:-1]] = _small_scalar(node)

    walk(value)
    provenance = {key: item for key, item in flat.items() if PROVENANCE_KEY_PATTERN.search(key)}
    source_paths = [item for key, item in flat.items()
                    if re.search(r"source_path|original_path|raw_path", key, re.IGNORECASE) and isinstance(item, str)]
    return {"path": str(path), "readable": True, "sha256": sha256_file(path),
            "top_level_keys": sorted(value) if isinstance(value, dict) else None,
            "provenance_fields": provenance, "source_paths": source_paths}


def _h5_axis_index(group) -> tuple[list[str] | None, str]:
    """Return anndata obs/var index strings from an h5ad group without pickle."""
    import h5py

    if isinstance(group, h5py.Dataset):  # legacy structured-array layout
        names = group.dtype.names or ()
        if "index" in names:
            return _decode(group["index"]), "legacy_structured_index"
        return None, "legacy_structured_without_index"
    key = group.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else str(key)
    if key in group and isinstance(group[key], h5py.Dataset):
        return _decode(group[key][()]), f"attr_index:{key}"
    return None, "index_not_found"


def _h5_column_summary(group, name: str) -> dict:
    import h5py

    node = group[name]
    if isinstance(node, h5py.Group) and "categories" in node and "codes" in node:
        categories = _decode(node["categories"][()])
        codes = np.asarray(node["codes"][()])
        return {"encoding": "categorical", "n_categories": len(categories),
                "n_nonmissing": int(np.sum(codes >= 0)),
                "categories_sample": categories[:20] if len(categories) <= 200 else categories[:10]}
    if isinstance(node, h5py.Dataset):
        if node.dtype.kind in "OSU":
            values = _decode(node[()])
            return {"encoding": "string", "n_unique": len(set(values)), "values_sample": sorted(set(values))[:10]}
        return {"encoding": f"numeric:{node.dtype}", "shape": list(node.shape)}
    return {"encoding": "unrecognized"}


def read_h5ad(path: Path) -> dict:
    result = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    result["sha256"] = sha256_file(path)
    result["size_bytes"] = path.stat().st_size
    if importlib.util.find_spec("h5py") is None:
        result["error"] = "h5py_unavailable"
        return result
    import h5py

    try:
        with h5py.File(path, "r") as handle:
            result["top_level"] = sorted(handle.keys())
            x = handle.get("X")
            if x is not None:
                if isinstance(x, h5py.Dataset):
                    result["X"] = {"layout": "dense", "shape": list(x.shape), "dtype": str(x.dtype)}
                else:
                    shape = x.attrs.get("shape", x.attrs.get("h5sparse_shape"))
                    result["X"] = {"layout": str(x.attrs.get("encoding-type", "sparse")),
                                   "shape": None if shape is None else [int(v) for v in shape],
                                   "dtype": str(x["data"].dtype) if "data" in x else None}
            for axis in ("obs", "var"):
                if axis not in handle:
                    result[axis] = {"available": False}
                    continue
                ids, how = _h5_axis_index(handle[axis])
                entry = {"available": True, "index_source": how}
                if ids is not None:
                    entry.update({"n": len(ids), "n_unique": len(set(ids)), "ids_sha256": _hash_strings(ids),
                                  "id_examples": ids[:3]})
                    result[f"_{axis}_ids"] = ids
                group = handle[axis]
                if isinstance(group, h5py.Group):
                    columns = [key for key in group.keys() if key not in {"_index", "__categories"}]
                    entry["columns"] = sorted(columns)
                    entry["study_metadata"] = {c: _h5_column_summary(group, c) for c in columns
                                               if META_COLUMN_PATTERN.search(c)}
                    entry["label_meaning_columns"] = {c: {k: v for k, v in _h5_column_summary(group, c).items()
                                                          if k in {"encoding", "n_categories", "n_unique"}}
                                                      for c in columns if LABEL_COLUMN_PATTERN.search(c)}
                result[axis] = entry
            result["uns_keys"] = sorted(handle["uns"].keys()) if "uns" in handle else []
    except (OSError, KeyError, ValueError) as error:
        result["error"] = repr(error)
    return result


def read_npz(path: Path) -> dict:
    result = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    result["sha256"] = sha256_file(path)
    result["size_bytes"] = path.stat().st_size
    arrays, refused = {}, []
    try:
        with np.load(path, allow_pickle=False) as payload:
            for key in payload.files:
                # Header only: shape/dtype without reading (possibly huge) feature data.
                with payload.zip.open(f"{key}.npy") as member:
                    version = np.lib.format.read_magic(member)
                    shape, _, dtype = np.lib.format._read_array_header(member, version)
                if dtype.hasobject:
                    refused.append(key)  # object array: would require unpickling
                    continue
                arrays[key] = {"shape": list(shape), "dtype": str(dtype)}
                if dtype.kind in "SU" and len(shape) == 1 and re.search(r"cell|barcode|obs|name|id", key, re.I):
                    ids = _decode(payload[key])
                    arrays[key].update({"n_unique": len(set(ids)), "ids_sha256": _hash_strings(ids)})
                    result.setdefault("_ids", {})[key] = ids
    except (OSError, ValueError) as error:
        result["error"] = repr(error)
    result["arrays"] = arrays
    result["object_arrays_refused"] = refused
    return result


# --------------------------------------------------------------- comparison

def load_old_manifest(paths: list[str]) -> tuple[list[dict], list[str]]:
    """Accept CSV/JSON manifests; keep dataset name and any path/hash columns only."""
    entries, notes = [], []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            notes.append(f"missing_old_manifest:{path}")
            continue
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                keep = [c for c in (reader.fieldnames or []) if re.search(r"dataset|path|sha|hash|source", c, re.I)]
                seen = set()
                for row in reader:
                    item = {c: row.get(c) for c in keep}
                    key = json.dumps(item, sort_keys=True)
                    if key not in seen:
                        seen.add(key)
                        entries.append({"manifest": str(path), **item})
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            rows = value if isinstance(value, list) else value.get("datasets", value.get("entries", []))
            if isinstance(rows, dict):
                rows = [{"dataset": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in rows.items()]
            for row in rows if isinstance(rows, list) else []:
                if isinstance(row, dict):
                    entries.append({"manifest": str(path), **{k: _small_scalar(v) for k, v in row.items()
                                                                if re.search(r"dataset|path|sha|hash|source", k, re.I)}})
        notes.append(f"loaded_old_manifest:{path}:{sha256_file(path)}")
    return entries, notes


def _manifest_paths_and_hashes(entries: list[dict]) -> tuple[set[str], set[str], set[str]]:
    paths, hashes, names = set(), set(), set()
    for entry in entries:
        for key, value in entry.items():
            if not isinstance(value, str) or key == "manifest":
                continue
            if re.fullmatch(r"[0-9a-f]{64}", value):
                hashes.add(value)
            elif "/" in value:
                paths.add(os.path.normpath(value))
            elif re.search(r"dataset", key, re.I):
                names.add(value)
    return paths, hashes, names


def compare_ids(candidate: list[str], other: list[str]) -> dict:
    a, b = set(candidate), set(other)
    shared = len(a & b)
    return {"n_candidate": len(a), "n_other": len(b), "n_shared": shared,
            "fraction_of_candidate": shared / max(1, len(a))}


def classify(candidate: dict) -> str:
    if not candidate["files"]:
        return "not_found"
    if any(c.get("status") == "overlap_detected" for c in candidate["comparisons"]):
        return "overlap_detected"
    if not candidate["comparisons"] or any(c.get("status") == "overlap_unknown" for c in candidate["comparisons"]):
        return "available_overlap_unknown"
    return "available_no_overlap_detected_by_checked_evidence"


# ------------------------------------------------------------- environment

def package_audit() -> dict:
    report = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
    for name in PACKAGES:
        spec = importlib.util.find_spec(name)  # locates without executing the package
        entry = {"importable_spec_found": spec is not None, "origin": getattr(spec, "origin", None)}
        dist = {"sklearn": "scikit-learn"}.get(name, name)
        try:
            entry["version"] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            entry["version"] = None
        report[name] = entry
    return report


def bounded_repo_scan(roots: Iterable[str], pattern: str = r"schntl") -> dict:
    hits, visited, truncated = [], 0, False
    regex = re.compile(pattern, re.IGNORECASE)
    for root in roots:
        stack = [(Path(root), 0)]
        while stack:
            folder, depth = stack.pop()
            try:
                with os.scandir(folder) as iterator:
                    for entry in iterator:
                        visited += 1
                        if visited > MAX_REPO_ENTRIES:
                            truncated = True
                            break
                        if regex.search(entry.name):
                            hits.append(entry.path)
                        if entry.is_dir(follow_symlinks=False) and depth + 1 < MAX_REPO_DEPTH \
                                and not entry.name.startswith("."):
                            stack.append((Path(entry.path), depth + 1))
            except OSError:
                continue
            if truncated:
                break
    return {"roots": list(roots), "max_depth": MAX_REPO_DEPTH, "entries_visited": visited,
            "truncated": truncated, "matches": sorted(hits)}


def schntl_runtime_status(packages: dict, repo: dict) -> dict:
    tf, keras = packages.get("tensorflow", {}), packages.get("keras", {})
    ok = (sys.version_info[:3] == (3, 6, 4) and str(tf.get("version") or "").startswith("1.12")
          and str(keras.get("version") or "") == "2.1.5")
    return {**SCHNTL, "local_repo_candidates": repo["matches"],
            "current_interpreter_matches_requirements": ok,
            "status": ("runtime_available" if ok and repo["matches"] else
                       "blocked_exact_runtime_absent" if not ok else "blocked_official_code_not_local"),
            "note": "no baseline result is produced by this audit"}


# ------------------------------------------------------------------- driver

def run_audit(args) -> dict:
    profile_dir, npz_dir = Path(args.profile_dir), Path(args.npz_dir)
    profiles = [read_profile(p) for p in sorted(profile_dir.glob("*.profile.json"))] if profile_dir.is_dir() else []
    npz_files = sorted(npz_dir.glob("*.npz")) if npz_dir.is_dir() else []
    old_entries, manifest_notes = load_old_manifest(args.old_manifest or [])
    old_paths, old_hashes, old_names = _manifest_paths_and_hashes(old_entries)

    candidates = {}
    for name in CANDIDATES:
        candidates[name] = {"name": name, "files": [], "comparisons": [], "profile": [], "originals": []}
    for profile in profiles:
        for name in candidate_names(Path(profile["path"]).name):
            candidates[name]["profile"].append(profile)
            for source in profile.get("source_paths", []):
                if source.lower().endswith(".h5ad"):
                    candidates[name]["originals"].append(read_h5ad(Path(source)))
    for path in npz_files:
        for name in candidate_names(path.name):
            candidates[name]["files"].append(read_npz(path))
    for name, candidate in candidates.items():
        candidate["files"] += [o for o in candidate["originals"] if o.get("exists")]

    # Cell-ID universe of the old 60 datasets where IDs can be read safely.
    old_ids: dict[str, list[str]] = {}
    for path in npz_files:
        if candidate_names(path.name):
            continue
        in_manifest = (os.path.normpath(str(path)) in old_paths or path.stem in old_names)
        if old_entries and not in_manifest:
            continue
        info = read_npz(path)
        for key, ids in (info.get("_ids") or {}).items():
            old_ids[f"{path.name}:{key}"] = ids
    for profile in profiles:
        if candidate_names(Path(profile["path"]).name):
            continue
        for source in profile.get("source_paths", []):
            if source.lower().endswith(".h5ad") and (not old_entries or os.path.normpath(source) in old_paths
                                                     or Path(profile["path"]).name.split(".")[0] in old_names):
                info = read_h5ad(Path(source))
                if info.get("_obs_ids"):
                    old_ids[f"{source}:obs"] = info["_obs_ids"]

    for name, candidate in candidates.items():
        for info in candidate["files"]:
            path = os.path.normpath(info["path"])
            candidate["comparisons"].append({
                "evidence": "path_or_sha256_in_old_manifest", "file": info["path"],
                "status": ("overlap_detected" if path in old_paths or info.get("sha256") in old_hashes
                           else "no_match" if old_entries else "overlap_unknown"),
                "note": None if old_entries else "old manifest not supplied"})
            id_sets = dict(info.get("_ids") or {})
            if info.get("_obs_ids"):
                id_sets["obs"] = info["_obs_ids"]
            if not id_sets:
                candidate["comparisons"].append({"evidence": "cell_id_overlap", "file": info["path"],
                                                 "status": "overlap_unknown", "note": "no safely readable cell IDs"})
            for key, ids in id_sets.items():
                if not old_ids:
                    candidate["comparisons"].append({"evidence": "cell_id_overlap", "file": info["path"],
                                                     "status": "overlap_unknown",
                                                     "note": "no old dataset exposes safely readable cell IDs"})
                    continue
                overlaps = {other: compare_ids(ids, values) for other, values in old_ids.items()}
                shared = {k: v for k, v in overlaps.items() if v["n_shared"]}
                candidate["comparisons"].append({
                    "evidence": "cell_id_overlap", "file": info["path"], "id_key": key,
                    "old_id_sources_checked": len(old_ids), "shared": shared,
                    "status": "overlap_unknown",
                    "note": ("Shared identifiers alone do not establish shared cells: processed matrices may "
                             "use the same integer row index or reuse barcode namespaces. Require matching "
                             "study/sample namespaces or original feature-aligned matrix evidence. Absence "
                             "of shared IDs likewise does not prove independence.")})
        if name == "BaronHuman":
            candidate["required_manual_checks"] = ["GEO accession vs existing Baron (GSE84133 lists human and mouse)",
                                                   "donor identity vs existing Baron", "matrix row equality"]
        candidate["status"] = classify(candidate)
        for info in candidate["files"] + candidate["originals"]:
            info.pop("_ids", None)
            info.pop("_obs_ids", None)
            info.pop("_var_ids", None)

    packages = package_audit()
    repo = bounded_repo_scan(args.repo_root or DEFAULT_REPO_ROOTS)
    return {"protocol_id": PROTOCOL, "status": "completed", "labels_read": False,
            "label_values_used": False, "fitting_performed": False,
            "retrieval": {"host": socket.gethostname(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "script_sha256": sha256_file(Path(__file__)), "profile_dir": str(profile_dir),
                          "npz_dir": str(npz_dir), "old_manifest_notes": manifest_notes,
                          "n_profiles": len(profiles), "n_npz": len(npz_files)},
            "candidates": candidates, "packages": packages, "repo_scan": repo,
            "schntl": schntl_runtime_status(packages, repo),
            "interpretation": "overlap_unknown is not independence; filename matches are only discovery"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="read-only independent dataset / baseline availability audit")
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile-dir", default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--npz-dir", default=DEFAULT_NPZ_DIR)
    parser.add_argument("--old-manifest", action="append", help="CSV/JSON listing the old 60 datasets (repeatable)")
    parser.add_argument("--repo-root", action="append", help="bounded scan root (repeatable)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = run_audit(args)
    temporary = out / "extension_audit.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(out / "extension_audit.json")
    for name, candidate in result["candidates"].items():
        print(f"{name}: {candidate['status']} ({len(candidate['files'])} files)")
    print(f"scHNTL: {result['schntl']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
