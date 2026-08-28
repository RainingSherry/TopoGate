"""Read-only adapter to the completed C2 reconstruction probe.

Keeping the adapter narrow makes the reuse boundary explicit.  M1 may call
these objects, but it never writes to the old project and never reruns P2.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from scripts.sparse_corruption_principle_probe import c2_matrix


OLD_C2_SOURCE = Path(c2_matrix.__file__).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_manifest() -> dict[str, Any]:
    return {
        "module": str(OLD_C2_SOURCE),
        "module_sha256": sha256_file(OLD_C2_SOURCE),
        "adapter_symbols": [
            "_SmallMAE",
            "_standardize",
            "_load_h0",
            "_load_labels",
            "_clustering_acc",
            "_embedding_diagnostics",
            "_seed_everything",
        ],
        "read_only": True,
    }


SmallMAE = c2_matrix._SmallMAE
standardize = c2_matrix._standardize
load_h0 = c2_matrix._load_h0
load_labels = c2_matrix._load_labels
clustering_acc = c2_matrix._clustering_acc
embedding_diagnostics = c2_matrix._embedding_diagnostics
seed_everything = c2_matrix._seed_everything
device_or_fail = c2_matrix._device_or_fail

