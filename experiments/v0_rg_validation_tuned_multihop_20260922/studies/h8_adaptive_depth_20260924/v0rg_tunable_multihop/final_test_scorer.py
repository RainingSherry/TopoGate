"""One-time final scorer; deliberately separate from the tuning process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .run import clustering_metrics


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def score_once(locked_selection_path: str | Path, final_manifest_path: str | Path,
               prediction_path: str | Path, output_dir: str | Path) -> dict:
    """Score test labels once after an immutable validation lock exists."""
    selection = json.loads(Path(locked_selection_path).read_text(encoding="utf-8"))
    final_manifest = json.loads(Path(final_manifest_path).read_text(encoding="utf-8"))
    if selection.get("selection_locked") is not True or selection.get("test_labels_loaded") is not False:
        raise ValueError("final scoring requires an immutable validation-only selection")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    final_record = output / "final_test_metrics.json"
    if final_record.exists():
        raise FileExistsError("final test has already been scored for this output directory")
    predictions = np.asarray(np.load(prediction_path, allow_pickle=False)["predictions"], dtype=np.int64)
    sidecar = np.load(final_manifest["test_labels_path"], allow_pickle=False)
    indices = np.asarray(sidecar["indices"], dtype=np.int64)
    labels = np.asarray(sidecar["labels"], dtype=np.int64)
    if indices.size != labels.size or np.any(indices < 0) or np.any(indices >= predictions.size):
        raise ValueError("invalid final-test label sidecar")
    record = {
        "status": "completed",
        "evaluation_scope": "final_test_once",
        "test_labels_loaded": True,
        "chosen_hop_variant": selection["chosen_hop_variant"],
        "chosen_validation_delta_ari": selection["chosen_validation_delta_ari"],
        "test_labels_sha256": final_manifest["test_labels_sha256"],
        "test_metrics": clustering_metrics(labels, predictions[indices]),
    }
    _write_json(final_record, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="one-time V0_RG final test scorer")
    parser.add_argument("--locked-selection", required=True)
    parser.add_argument("--final-manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    score_once(args.locked_selection, args.final_manifest, args.predictions, args.output)


if __name__ == "__main__":
    main()
