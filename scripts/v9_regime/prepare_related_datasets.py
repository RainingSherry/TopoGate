#!/usr/bin/env python3
from __future__ import annotations

"""Prepare a fixed, auditable related-dataset bundle from downloaded sources.

The script intentionally does not download data or calculate cryptographic
hashes.  It converts the raw UCI/OpenML files already present in ``raw_dir``
and writes dense ``x``/``y`` NPZ files plus a provenance manifest.  Labels are
kept only for benchmark metadata and are never used by the text vectorizer or
numeric preprocessing.
"""

import argparse
import csv
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _save_npz(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=np.asarray(x, dtype=np.float32), y=np.asarray(y, dtype=np.int64))


def _matrix_summary(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    values = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(values)
    filled = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "n": int(values.shape[0]),
        "d": int(values.shape[1]),
        "n_clusters": int(np.unique(y).size),
        "label_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "missing_fraction": float(1.0 - np.mean(finite)),
        "zero_fraction_after_nan_to_num": float(np.mean(filled == 0.0)),
        "finite_min": float(np.min(filled)),
        "finite_max": float(np.max(filled)),
        "dtype": str(values.dtype),
    }


def _read_zip_text(path: Path, member: str, encoding: str = "utf-8") -> str:
    with zipfile.ZipFile(path) as archive:
        return archive.read(member).decode(encoding, errors="replace")


def _parse_internet_advertisements(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    text = _read_zip_text(path, "ad.data", encoding="ascii")
    rows = list(csv.reader(io.StringIO(text, newline=""), skipinitialspace=True))
    if not rows:
        raise ValueError("Internet Advertisements has no rows")
    width = len(rows[0])
    if width != 1559:
        raise ValueError(f"unexpected Internet Advertisements width: {width}")
    x = np.full((len(rows), width - 1), np.nan, dtype=np.float32)
    y = np.empty(len(rows), dtype=np.int64)
    label_map = {"nonad.": 0, "ad.": 1, "nonad": 0, "ad": 1}
    for row_index, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"row {row_index} has width {len(row)} rather than {width}")
        for column_index, raw_value in enumerate(row[:-1]):
            value = raw_value.strip()
            if value not in {"", "?"}:
                x[row_index, column_index] = float(value)
        label = row[-1].strip().lower()
        if label not in label_map:
            raise ValueError(f"unknown Internet Advertisements label: {label!r}")
        y[row_index] = label_map[label]
    return x, y, {
        "label_mapping": label_map,
        "source_expected_shape": [3279, 1558],
        "source_expected_label_counts": {"nonad.": 2821, "ad.": 458},
        "source_count_note": "Downloaded ad.data contains 2820 nonad. and 459 ad.; UCI documentation states 2821/458.",
    }


def _parse_sms_spam(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    text = _read_zip_text(path, "SMSSpamCollection", encoding="utf-8")
    labels: list[int] = []
    messages: list[str] = []
    label_map = {"ham": 0, "spam": 1}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if "\t" not in line:
            raise ValueError(f"SMS row {line_number} has no tab separator")
        raw_label, message = line.split("\t", 1)
        raw_label = raw_label.strip().lower()
        if raw_label not in label_map:
            raise ValueError(f"unknown SMS label on row {line_number}: {raw_label!r}")
        labels.append(label_map[raw_label])
        messages.append(message)
    vectorizer = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
        min_df=2,
        max_features=500,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    # Vocabulary fitting sees message text only; labels are not passed to it.
    matrix = vectorizer.fit_transform(messages).toarray().astype(np.float32)
    return matrix, np.asarray(labels, dtype=np.int64), {
        "label_mapping": label_map,
        "vectorizer": {
            "analyzer": "word",
            "lowercase": True,
            "ngram_range": [1, 2],
            "token_pattern": r"(?u)\b\w+\b",
            "min_df": 2,
            "max_features": 500,
            "sublinear_tf": True,
            "norm": "l2",
            "fit_uses_labels": False,
        },
        "raw_message_count": len(messages),
        "raw_nonzero_count": int(np.count_nonzero(matrix)),
    }


def _parse_website_phishing(path: Path) -> tuple[np.ndarray, np.ndarray]:
    text = _read_zip_text(path, "PhishingData.arff", encoding="utf-8")
    lines = text.splitlines()
    in_data = False
    rows: list[list[int]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if stripped.lower() == "@data":
            in_data = True
            continue
        if not in_data or stripped.startswith("@"):
            continue
        rows.append([int(value.strip()) for value in stripped.split(",")])
    array = np.asarray(rows, dtype=np.int64)
    if array.shape != (1353, 10):
        raise ValueError(f"unexpected Website Phishing shape: {array.shape}")
    label_values = array[:, -1]
    label_map = {-1: 0, 0: 1, 1: 2}
    y = np.asarray([label_map[int(value)] for value in label_values], dtype=np.int64)
    return array[:, :-1].astype(np.float32), y


def _parse_sparse_arff(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    attributes = [line for line in lines if line.strip().lower().startswith("@attribute ")]
    if len(attributes) < 2 or not attributes[0].strip().lower().startswith("@attribute y "):
        raise ValueError("OpenML webdata_wXa must have Y as its first ARFF attribute")
    n_features = len(attributes) - 1
    rows: list[dict[int, float]] = []
    labels: list[int] = []
    in_data = False
    label_map = {-1: 0, 1: 1}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        if line.lower() == "@data":
            in_data = True
            continue
        if not in_data:
            continue
        if not (line.startswith("{") and line.endswith("}")):
            raise ValueError(f"unsupported non-sparse row at line {line_number}")
        entries: dict[int, float] = {}
        body = line[1:-1].strip()
        if body:
            for token in re.split(r",\s*(?=\d+\s)", body):
                index_text, value_text = token.strip().split(None, 1)
                entries[int(index_text)] = float(value_text)
        if 0 not in entries:
            raise ValueError(f"missing Y entry at line {line_number}")
        raw_label = int(entries.pop(0))
        if raw_label not in label_map:
            raise ValueError(f"unknown webdata_wXa label {raw_label} at line {line_number}")
        if any(index < 1 or index > n_features for index in entries):
            raise ValueError(f"feature index outside 1..{n_features} at line {line_number}")
        rows.append(entries)
        labels.append(label_map[raw_label])
    x = np.zeros((len(rows), n_features), dtype=np.float32)
    for row_index, entries in enumerate(rows):
        for feature_index, value in entries.items():
            x[row_index, feature_index - 1] = value
    return x, np.asarray(labels, dtype=np.int64), {
        "label_mapping": label_map,
        "sparse_arff_attributes": len(attributes),
        "mean_nonzero_features_per_row": float(np.mean([len(row) for row in rows])),
        "sparse_values_materialized_as_zero": True,
    }


def prepare(raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    internet_x, internet_y, internet_meta = _parse_internet_advertisements(raw_dir / "internet_advertisements.zip")
    internet_out = output_dir / "internet_advertisements.npz"
    _save_npz(internet_out, internet_x, internet_y)
    records.append({
        "dataset_id": "external__internet_advertisements",
        "name": "internet_advertisements",
        "status": "prepared",
        "source_kind": "UCI",
        "source_identity": "uci:dataset_id=51",
        "source_version": "UCI dataset 51; downloaded 2026-08-06",
        "source_url": "https://archive.ics.uci.edu/dataset/51/internet+advertisements",
        "download_url": "https://archive.ics.uci.edu/static/public/51/internet+advertisements.zip",
        "license": "UCI dataset terms; see ad.DOCUMENTATION",
        "citation": "Nicholas Kushmerick. Learning to remove Internet advertisements. 3rd International Conference on Autonomous Agents, 1999.",
        "source_path": str((raw_dir / "internet_advertisements.zip").resolve()),
        "processed_path": str(internet_out.resolve()),
        "preprocessing": "CSV numeric fields; '?' retained as NaN; labels separated; V9 standardize_x is applied later",
        "preprocessing_uses_labels": False,
        "labels_used_during_fit": False,
        "source_fingerprint_policy": "URL/version/byte metadata recorded once; no hash recomputation",
        "matrix": _matrix_summary(internet_x, internet_y),
        **internet_meta,
    })

    sms_x, sms_y, sms_meta = _parse_sms_spam(raw_dir / "sms_spam_collection.zip")
    sms_out = output_dir / "sms_spam_collection_full_tfidf500.npz"
    _save_npz(sms_out, sms_x, sms_y)
    records.append({
        "dataset_id": "external__sms_spam_collection_full_tfidf500",
        "name": "sms_spam_collection_full_tfidf500",
        "status": "prepared",
        "source_kind": "UCI",
        "source_identity": "uci:dataset_id=228",
        "source_version": "SMS Spam Collection v1; downloaded 2026-08-06",
        "source_url": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "download_url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "license": "SMS Spam Collection v1 terms in readme",
        "citation": "Tiago A. Almeida, José M. Gómez Hidalgo, Alessandro Yamakami. Contributions to the study of SMS Spam Filtering: New Collection and Results. ACM DOCENG, 2011.",
        "source_path": str((raw_dir / "sms_spam_collection.zip").resolve()),
        "processed_path": str(sms_out.resolve()),
        "preprocessing": "unlabeled TF-IDF word/unigram+bigram matrix with fixed max_features=500",
        "preprocessing_uses_labels": False,
        "labels_used_during_fit": False,
        "source_fingerprint_policy": "URL/version/byte metadata recorded once; no hash recomputation",
        "matrix": _matrix_summary(sms_x, sms_y),
        **sms_meta,
    })

    phishing_x, phishing_y = _parse_website_phishing(raw_dir / "website_phishing.zip")
    local_phishing = Path("datasets/AHDPC/processed/website_phishing.npz").resolve()
    local_equal = False
    if local_phishing.exists():
        with np.load(local_phishing, allow_pickle=False) as data:
            local_equal = bool(np.array_equal(np.asarray(data["x"], dtype=np.float32), phishing_x) and np.array_equal(np.asarray(data["y"], dtype=np.int64), phishing_y))
    records.append({
        "dataset_id": "ahdpc_prepared__website_phishing",
        "name": "website_phishing",
        "status": "duplicate_local_prepared",
        "source_kind": "UCI",
        "source_identity": "uci:dataset_id=379",
        "source_version": "UCI dataset 379; downloaded and compared 2026-08-06",
        "source_url": "https://archive.ics.uci.edu/dataset/379/website+phishing",
        "download_url": "https://archive.ics.uci.edu/static/public/379/website+phishing.zip",
        "license": "UCI dataset terms",
        "citation": "UCI Machine Learning Repository, Website Phishing",
        "source_path": str((raw_dir / "website_phishing.zip").resolve()),
        "processed_path": str(local_phishing),
        "duplicate_of": "datasets/AHDPC/processed/website_phishing.npz",
        "array_equal_to_local": local_equal,
        "preprocessing": "UCI PhishingData.arff reduced 9-feature numeric view; no new copy created",
        "preprocessing_uses_labels": False,
        "labels_used_during_fit": False,
        "source_fingerprint_policy": "URL/version/byte metadata recorded once; no hash recomputation",
        "matrix": _matrix_summary(phishing_x, phishing_y),
        "label_mapping": {"-1": 0, "0": 1, "1": 2},
    })

    web_x, web_y, web_meta = _parse_sparse_arff(raw_dir / "openml_webdata_wXa.arff")
    web_out = output_dir / "webdata_wXa.npz"
    _save_npz(web_out, web_x, web_y)
    records.append({
        "dataset_id": "external__openml_webdata_wXa",
        "name": "webdata_wXa",
        "status": "prepared",
        "source_kind": "OpenML",
        "source_identity": "openml:did=350",
        "source_version": "OpenML dataset 350 version 1; file_id 52253; downloaded 2026-08-06",
        "source_url": "https://www.openml.org/d/350",
        "download_url": "https://www.openml.org/data/v1/download/52253/webdata_wXa.sparse_arff",
        "license": "Public (OpenML record)",
        "citation": "John C. Platt. Fast training of support vector machines using sequential minimal optimization. In Advances in Kernel Methods, 1998.",
        "source_path": str((raw_dir / "openml_webdata_wXa.arff").resolve()),
        "processed_path": str(web_out.resolve()),
        "preprocessing": "Sparse ARFF numeric features; unspecified entries materialized as zero; target Y separated",
        "preprocessing_uses_labels": False,
        "labels_used_during_fit": False,
        "source_fingerprint_policy": "OpenML DID/version/file_id recorded once; no hash recomputation",
        "matrix": _matrix_summary(web_x, web_y),
        **web_meta,
    })

    payload = {
        "bundle_id": "v9_related_datasets_2026-08-06",
        "status": "prepared_in_tmp_only",
        "research_scope": "Fixed structural/application neighbors of spambase: email/SMS spam, web advertising, phishing, and sparse web classification.",
        "raw_dir": str(raw_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "downloaded_at": "2026-08-06",
        "selection_uses_labels_or_results": False,
        "hashes_recomputed": False,
        "datasets": sorted(records, key=lambda item: item["dataset_id"]),
        "unresolved_or_excluded": [
            {
                "name": "spambase",
                "status": "existing_local_reference",
                "source_identity": "uci:dataset_id=94",
                "source_url": "https://archive.ics.uci.edu/dataset/94/spambase",
                "source_path": str(Path("datasets/spambase.npz").resolve()),
                "reason": "Existing project reference; not redownloaded to avoid duplicate source traffic.",
            }
        ],
    }
    _write_json(output_dir / "manifest.json", payload)
    # A separate matrix manifest lets build_manifest.py consume the prepared
    # NPZ paths without confusing raw archives with experiment inputs.
    v9_records = []
    for item in payload["datasets"]:
        if item.get("status") != "prepared":
            continue
        v9_records.append(
            {
                "dataset_id": item["dataset_id"],
                "name": item["name"],
                "source_kind": item["source_kind"],
                "source_path": item["processed_path"],
                "source_identity": item["source_identity"],
                "source_version": item["source_version"],
                "source_url": item["source_url"],
                "download_url": item["download_url"],
                "license": item["license"],
                "citation": item["citation"],
                "family": "text" if "sms" in item["name"] else "web_sparse",
                "preprocessing": item["preprocessing"],
                "preprocessing_uses_labels": item["preprocessing_uses_labels"],
                "labels_used_during_fit": item["labels_used_during_fit"],
                "source_fingerprint_policy": item["source_fingerprint_policy"],
            }
        )
    _write_json(
        output_dir / "v9_external_manifest.json",
        {
            "manifest_id": "v9_related_external_matrix_manifest_2026-08-06",
            "datasets": sorted(v9_records, key=lambda item: item["dataset_id"]),
            "selection_uses_labels_or_results": False,
            "hashes_recomputed": False,
        },
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = prepare(args.raw_dir, args.output_dir)
    counts = Counter(item["status"] for item in payload["datasets"])
    print(json.dumps({"output_dir": str(args.output_dir), "datasets": len(payload["datasets"]), "status": dict(counts)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
