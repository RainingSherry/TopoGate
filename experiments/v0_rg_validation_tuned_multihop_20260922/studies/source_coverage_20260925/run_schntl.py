"""Feature-only, CPU-only single-task adapter for the official scHNTL code (snapshot 61183a4).

The official ``scHNTL.py`` is executed unchanged with ``runpy`` inside a fresh
per-task working directory.  Runtime-only TF2.12/Keras2.12 compatibility, identical
to ``external/schntl_tf1_compat_smoke.py``: eager execution disabled and the legacy
Adam bound to ``keras.optimizers.Adam``.  Protocol and deviations: SCHNTL_SPEC.md.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=$R python run_schntl.py --task-json T.json --out RUN_DIR
"""
from __future__ import annotations

import os

THREADS = 2
if __name__ == "__main__":  # pin before numpy/BLAS/TensorFlow are imported
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        os.environ[_name] = str(THREADS)

import argparse  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import runpy  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

EXP = Path(__file__).resolve().parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
from common import (SAFE_ID, atomic_json, atomic_npy, canonical_sha256, directory_manifest,  # noqa: E402
                    json_safe, read_json, sha256)

PROTOCOL = "source_coverage_20260925_schntl_official_v1"
ALGORITHM = "scHNTL_official"
OFFICIAL_COMMIT = "61183a4"
OFFICIAL_DIRS = (EXP / "external" / f"SWJTU-ML-scHNTL-code-{OFFICIAL_COMMIT}",  # deployed layout
                 EXP / "external" / "scHNTL_official" / f"SWJTU-ML-scHNTL-code-{OFFICIAL_COMMIT}")
OFFICIAL_FILES = ("scHNTL.py", "utils.py", "graph_attention_layer.py", "soft_clustering_layer.py",
                  "preprocess.py", "README.md", "LICENSE")
PCA_CAP, PRE_EPOCHS, EPOCHS = 512, 100, 2000
INPUT_KINDS = ("raw_count", "log1p_expression", "general")
OFFICIAL_NORMALIZATION = {
    "raw_count": "official row scaling to 1e5 over selected raw counts, then log2(x+1)",
    "log1p_expression": "official row scaling to 1e5 applied to the provided log1p values as-is "
                        "(no expm1 inversion), then log2(x+1); double log disclosed",
    "general": "official row scaling to 1e5 applied to nonnegative general features as-is, then log2(x+1)",
}
FORBIDDEN_TASK_KEYS = ("labels_path", "label_path", "test_labels", "validation_labels_path",
                       "full_labels", "posthoc_label")
REQUIRED_TASK_KEYS = {"id", "dataset", "seed", "algorithm", "feature_path", "feature_sha256",
                      "n_clusters", "K_provenance", "input_kind", "locked_preprocessor_npz",
                      "locked_preprocessor_sha256", "locked_preprocessor_json",
                      "expected_preprocessing_hash"}
GRAPHDEF_LIMIT_BYTES = 2 ** 31  # protobuf limit; A15 and B enter the triplet loss as constants
DENSE_HOST_COPIES = 24          # declared upper estimate of simultaneous float64 N x N buffers
ARTIFACTS = ("embedding.npy", "predictions.npy", "model.h5", "selected_features.npy")


class Blocked(RuntimeError):
    """Deterministic, outcome-independent refusal; evidence goes to blocked.json."""


# ---------------------------------------------------------------- task and inputs

def load_task(task_path: Path) -> dict:
    task = read_json(task_path)
    for key in task:
        if any(bad in key for bad in FORBIDDEN_TASK_KEYS):
            raise ValueError(f"fitting task exposes a label field: {key}")
    missing = REQUIRED_TASK_KEYS - set(task)
    if missing:
        raise ValueError(f"task missing fields {sorted(missing)}")
    if task["algorithm"] != ALGORITHM:
        raise ValueError(f"task algorithm must be {ALGORITHM}")
    if task["K_provenance"] != "benchmark_oracle_from_y":
        raise ValueError("K provenance must be the explicit benchmark oracle")
    if int(task["n_clusters"]) < 2:
        raise ValueError("n_clusters must be at least 2")
    if task["input_kind"] not in INPUT_KINDS:
        raise ValueError(f"input_kind must be one of {INPUT_KINDS}")
    if not SAFE_ID.match(str(task["dataset"])):
        raise ValueError("dataset name must be a safe path component")
    int(task["seed"])
    return task


def official_source(task: dict) -> Path:
    candidates = [Path(task["official_source"])] if task.get("official_source") else list(OFFICIAL_DIRS)
    for path in candidates:
        if (path / "scHNTL.py").is_file():
            return path.resolve()
    raise FileNotFoundError(f"official scHNTL source not found in {candidates}")


def official_hashes(source: Path) -> dict[str, str]:
    return {name: sha256(source / name) for name in OFFICIAL_FILES}


def core_adapter():
    """Import the immutable core feature loader; return module, package name, file hashes."""
    try:
        from v0rg_tunable_multihop import input_adapter
    except ImportError:  # local read-only copy of the latest core
        from reference_core import input_adapter
    base = Path(input_adapter.__file__).resolve().parent
    return input_adapter, input_adapter.__name__.split(".")[0], {
        path.name: sha256(path) for path in sorted(base.glob("*.py"))}


def load_selected(task: dict, input_adapter) -> tuple[np.ndarray, np.ndarray, dict]:
    """Raw (never standardized) columns chosen by the locked preprocessor, manifest row order."""
    feature_path = Path(task["feature_path"])
    if sha256(feature_path) != task["feature_sha256"]:
        raise ValueError("feature-only matrix hash differs from task")
    loaded = input_adapter.load_npz_matrix_only(feature_path)
    if loaded.labels is not None or loaded.profile.get("labels_accessed") is not False:
        raise RuntimeError("feature loader touched labels")
    npz_path = Path(task["locked_preprocessor_npz"])
    if sha256(npz_path) != task["locked_preprocessor_sha256"]:
        raise ValueError("locked preprocessor.npz hash differs from task")
    meta = read_json(task["locked_preprocessor_json"])
    width = int(loaded.X.shape[1])
    if meta.get("fingerprint") != task["expected_preprocessing_hash"]:
        raise ValueError("locked preprocessor fingerprint differs from task")
    if meta.get("input_kind") != task["input_kind"] or int(meta.get("width", -1)) != width:
        raise ValueError("locked preprocessor input_kind/width disagree with task matrix")
    with np.load(npz_path, allow_pickle=False) as packed:
        features = np.asarray(packed["features"], dtype=np.int64).reshape(-1)
    if features.size < 2 or features[0] < 0 or features[-1] >= width or np.any(np.diff(features) <= 0):
        raise ValueError("locked feature ids must be strictly increasing column indices")
    X = loaded.X
    selected = X.tocsr()[:, features].toarray() if sp.issparse(X) else np.asarray(X)[:, features]
    return np.ascontiguousarray(selected), features, dict(loaded.profile)


def check_domain(X: np.ndarray, input_kind: str) -> dict:
    """Official normalization divides by row sums and takes log2; refuse where undefined."""
    if not np.isfinite(X).all():
        raise Blocked("non-finite selected values")
    negative = int(np.count_nonzero(X < 0))
    if negative:
        raise Blocked(f"{negative} negative selected values: official log2 row scaling is undefined; "
                      "standardized features are never substituted")
    if input_kind == "raw_count" and not np.array_equal(X, np.rint(X)):
        raise ValueError("raw_count task holds non-integer values")
    zero_rows = int(np.count_nonzero(X.sum(axis=1) <= 0))
    constant_rows = int(np.count_nonzero(X.max(axis=1) == X.min(axis=1)))
    if zero_rows or constant_rows:
        raise Blocked(f"{zero_rows} zero-total and {constant_rows} constant cells over selected features: "
                      "official scaling/Pearson undefined; cells are never filtered")
    return {"input_kind": input_kind, "official_normalization": OFFICIAL_NORMALIZATION[input_kind],
            "values_changed_by_adapter": False, "min_row_total": float(X.sum(axis=1).min()),
            "zero_fraction": float(1.0 - np.count_nonzero(X) / X.size)}


def pca_dim(n: int, d: int) -> int:
    return int(min(PCA_CAP, n - 1, d))


def dense_guard(n: int, max_dense_gib: float) -> dict:
    """Official code is dense O(N^2) memory and O(N^3) eig; refuse instead of subsampling."""
    graph_bytes = 2 * 4 * n * n
    host_bytes = DENSE_HOST_COPIES * 8 * n * n
    info = {"n": n, "graph_constant_bytes_estimate": graph_bytes, "graphdef_limit_bytes": GRAPHDEF_LIMIT_BYTES,
            "host_bytes_estimate": host_bytes, "max_dense_gib": float(max_dense_gib),
            "truncation_or_subsampling": False}
    if graph_bytes >= GRAPHDEF_LIMIT_BYTES:
        raise Blocked(f"N={n}: float32 A15 and B graph constants exceed the 2 GiB GraphDef limit")
    if host_bytes > max_dense_gib * 2 ** 30:
        raise Blocked(f"N={n}: dense host estimate {host_bytes / 2 ** 30:.1f} GiB > {max_dense_gib} GiB")
    return info


def write_official_tsv(path: Path, X: np.ndarray, features: np.ndarray) -> dict:
    """Gene-by-cell TSV with synthetic IDs; verified with the exact official reader call."""
    import pandas as pd

    if X.dtype not in (np.float32, np.float64):  # integer storage: exact widening, float_format applies
        X = X.astype(np.float64)
    integer = bool(np.array_equal(X, np.rint(X)))
    # A decimal point keeps counts float64 in pandas; official in-place row scaling would truncate ints.
    fmt, tol = ("%.1f", 0.0) if integer else (("%.9e", 1e-9) if X.dtype == np.float32 else ("%.17e", 1e-14))
    cells = [f"c{row:07d}" for row in range(X.shape[0])]
    genes = [f"f{int(column)}" for column in features]
    path.parent.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(X.T, index=genes, columns=cells).to_csv(path, sep="\t", float_format=fmt)
    back = pd.read_csv(path, index_col=0, sep="\t")
    if [str(c) for c in back.columns] != cells or [str(g) for g in back.index] != genes:
        raise RuntimeError("TSV cell/gene order changed on re-read")
    if any(dtype != np.float64 for dtype in back.dtypes):
        raise RuntimeError("official reader would not parse every column as float64")
    reference = X.astype(np.float64)
    error = np.abs(back.values.T - reference) / np.where(reference == 0, 1.0, np.abs(reference))
    max_error = float(error.max())
    if max_error > tol:
        raise RuntimeError(f"TSV round-trip relative error {max_error} > {tol}")
    return {"path": str(path), "sha256": sha256(path), "float_format": fmt, "integer_valued": integer,
            "roundtrip_max_relative_error": max_error, "cell_ids": "c{row:07d} in manifest row order",
            "gene_ids": "f{original column index}"}


# ---------------------------------------------------------------- official execution

def run_official(source: Path, cwd: Path, dataset: str, n_clusters: int, pca: int, seed: int):
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    tf.config.threading.set_intra_op_parallelism_threads(THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(THREADS)
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("GPU visible; the scHNTL baseline is CPU-only")
    import keras
    from keras.optimizers.legacy.adam import Adam
    keras.optimizers.Adam = Adam
    argv = [str(source / "scHNTL.py"), dataset, str(n_clusters), "--PCA_dim", str(pca),
            "--pre_epochs", str(PRE_EPOCHS), "--epochs", str(EPOCHS)]
    old_cwd, old_argv = os.getcwd(), sys.argv
    sys.path.insert(0, str(source))
    try:
        os.chdir(cwd)
        sys.argv = list(argv)
        random.seed(seed)
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        scope = runpy.run_path(argv[0], run_name="__main__")
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
        sys.path.remove(str(source))
    import pandas as pd
    import sklearn

    versions = {"python": platform.python_version(), "tensorflow": tf.__version__,
                "keras": keras.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                "sklearn": sklearn.__version__}
    return scope, argv, versions


def run_task(task_path: str | Path, out: str | Path, max_dense_gib: float) -> dict:
    started = time.time()
    task_path, out = Path(task_path).resolve(), Path(out).resolve()
    task = load_task(task_path)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty for the CPU-only baseline")
    if (out / "record.json").exists():
        raise RuntimeError(f"refusing to overwrite completed record in {out}")
    out.mkdir(parents=True, exist_ok=True)
    seed, n_clusters, dataset = int(task["seed"]), int(task["n_clusters"]), str(task["dataset"])
    source = official_source(task)
    official_before = official_hashes(source)
    if task.get("official_source_sha256") not in (None, official_before):
        raise ValueError("official source hashes differ from task")
    input_adapter, core_name, core_before = core_adapter()
    X, features, profile = load_selected(task, input_adapter)
    n, d = X.shape
    domain = check_domain(X, task["input_kind"])
    guard = dense_guard(n, max_dense_gib)
    pca = pca_dim(n, d)
    cwd = out / f"official_cwd_{os.getpid()}_{time.time_ns()}"  # never reuses NE cache or checkpoints
    cwd.mkdir()
    tsv = write_official_tsv(cwd / "data" / dataset / "data.tsv", X, features)
    del X
    scope, argv, versions = run_official(source, cwd, dataset, n_clusters, pca, seed)

    embedding = np.asarray(scope["hidden"], dtype=np.float32)
    predictions = np.asarray(scope["y_pred"], dtype=np.int64).reshape(-1)
    if embedding.ndim != 2 or embedding.shape[0] != n or not np.isfinite(embedding).all():
        raise RuntimeError(f"official embedding invalid: shape {embedding.shape}")
    if predictions.shape != (n,) or predictions.min() < 0 or predictions.max() >= n_clusters:
        raise RuntimeError("official predictions do not cover every row within [0, K)")
    atomic_npy(out / "embedding.npy", embedding)
    atomic_npy(out / "predictions.npy", predictions)
    atomic_npy(out / "selected_features.npy", features)
    artifacts = list(ARTIFACTS)
    if scope.get("final_pred") is not None:
        atomic_npy(out / "predictions_max_silhouette.npy", np.asarray(scope["final_pred"], dtype=np.int64))
        artifacts.append("predictions_max_silhouette.npy")
    temporary = out / f".model.{os.getpid()}.tmp.h5"  # .h5 suffix selects Keras HDF5 weights
    scope["model"].save_weights(str(temporary))
    temporary.replace(out / "model.h5")
    A, A15 = scope["A"], scope["A15"]
    official_after, core_after = official_hashes(source), core_adapter()[2]
    if official_after != official_before or core_after != core_before:
        raise RuntimeError("official or core source changed during the run")

    config = {"PCA_dim": pca, "pca_cap_rule": "min(512, n-1, n_selected_features)",
              "pre_epochs": PRE_EPOCHS, "epochs": EPOCHS, "official_args": vars(scope["args"]),
              "official_argv": argv[1:]}
    record = {
        "protocol_id": PROTOCOL, "status": "completed", "smoke": False, "epochs_override": None,
        "algorithm": ALGORITHM, "arm": ALGORITHM, "official_commit": OFFICIAL_COMMIT,
        "task_id": task["id"], "task_sha256": sha256(task_path), "dataset": dataset, "seed": seed,
        "seeding": "random.seed, np.random.seed, tf.compat.v1.set_random_seed before runpy (official has no seed arg)",
        "config": config, "config_sha256": canonical_sha256(json_safe(config)),
        "n_clusters": n_clusters, "K_provenance": task["K_provenance"],
        "feature_sha256": task["feature_sha256"], "feature_profile": profile,
        "locked_preprocessor_sha256": task["locked_preprocessor_sha256"],
        "preprocessing_hash": task["expected_preprocessing_hash"],
        "selected_feature_ids_sha256": canonical_sha256(features.tolist()),
        "n_samples": n, "n_features": d, "input_domain": domain, "dense_guard": guard, "official_tsv": tsv,
        "preprocessing_protocol": "same cells and locked selected raw columns; official native "
                                  "normalization/NE graph/PCA/global z-score; official preprocess.py QC not run",
        "runtime_shim": "tf.compat.v1.disable_eager_execution(); keras.optimizers.Adam = legacy Adam",
        "versions": versions, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
        "evaluation_mode": "transductive_feature_only", "graph_scope": "all_samples_unlabeled",
        "labels_used_during_fit": False, "labels_used_for_selection": False, "test_labels_loaded": False,
        "full_labels_loaded": False, "validation_labels_loaded": False, "official_subtype_path": None,
        "official_run": {"stop_iteration": int(scope["ite"]), "silhouette_checks": len(scope["sil_logs"]),
                         "max_silhouette": float(scope["max_sil"]), "graph_k": int(scope["K1"]),
                         "one_hop_edges": int(np.count_nonzero(A)),
                         "second_order_added_pairs": int(np.count_nonzero((A15 > 0) & (A == 0))),
                         "n_unique_predictions": int(np.unique(predictions).size),
                         "predictions_variable": "y_pred (argmax q at official stop)",
                         "working_directory": str(cwd), "outputs": directory_manifest(cwd)},
        "readout_note": "parent analyzer runs 20x KMeans(K) on embedding.npy; no labels or ARI here",
        "core_package": core_name, "source_hashes_before": core_before, "source_hashes_after": core_after,
        "official_source_dir": str(source), "official_hashes_before": official_before,
        "official_hashes_after": official_after, "runner_sha256": sha256(Path(__file__).resolve()),
        "common_sha256": sha256(EXP / "common.py"), "wall_seconds": time.time() - started,
        "artifact_sha256": {name: sha256(out / name) for name in artifacts},
    }
    atomic_json(out / "record.json", json_safe(record))
    return record


def validate_output(task_path: str | Path, out: str | Path) -> dict:
    """Parent-side check; common.validate_train_output does not apply (no ARM_SPECS arm)."""
    out = Path(out).resolve()
    record = read_json(out / "record.json")
    expect = {"status": "completed", "smoke": False, "epochs_override": None, "algorithm": ALGORITHM,
              "labels_used_during_fit": False, "test_labels_loaded": False, "full_labels_loaded": False,
              "task_sha256": sha256(Path(task_path)), "K_provenance": "benchmark_oracle_from_y"}
    bad = [key for key, value in expect.items() if record.get(key) != value]
    bad += ["source hashes changed"] * (record.get("source_hashes_before") != record.get("source_hashes_after")
                                        or record.get("official_hashes_before") != record.get("official_hashes_after"))
    bad += ["epoch budget"] * ((record.get("config") or {}).get("epochs") != EPOCHS)
    bad += [name for name in ARTIFACTS if not (out / name).is_file()
            or record.get("artifact_sha256", {}).get(name) != sha256(out / name)]
    if not bad:
        n, embedding = record["n_samples"], np.load(out / "embedding.npy")
        if embedding.shape[0] != n or not np.isfinite(embedding).all() or np.load(out / "predictions.npy").shape != (n,):
            bad.append("embedding/predictions rows or finiteness")
    if bad:
        raise RuntimeError(f"invalid scHNTL output {out}: {bad}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="run one official scHNTL baseline task (CPU)")
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-dense-gib", type=float, default=64.0, help="resource guard, not scientific")
    args = parser.parse_args()
    try:
        run_task(args.task_json, args.out, args.max_dense_gib)
    except Blocked as blocked:
        out = Path(args.out).resolve()
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / "blocked.json", {"protocol_id": PROTOCOL, "status": "blocked", "reason": str(blocked),
                                           "task_sha256": sha256(Path(args.task_json)),
                                           "labels_used_during_fit": False, "test_labels_loaded": False})
        print(f"blocked: {blocked}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
