from scripts.sparse_corruption_principle_probe.holdout_inventory import _select_maximin


def _record(path, family, n, d, sparsity, intrinsic):
    return {
        "relative_path": path,
        "source_family": family,
        "n": n,
        "d": d,
        "estimated_sparsity": sparsity,
        "estimated_intrinsic_dimension_proxy": intrinsic,
        "status": "candidate_valid",
    }


def test_holdout_selection_uses_only_label_free_feature_fields():
    records = [
        _record("a/A.h5ad", "a", 100, 1000, 0.9, 3),
        _record("b/B.h5ad", "b", 1000, 10000, 0.98, 8),
        _record("c/C.h5ad", "c", 5000, 5000, 0.75, 5),
        _record("d/D.h5ad", "d", 2000, 20000, 0.95, 12),
    ]
    selected = _select_maximin(records, target=3)
    assert len(selected) == 3
    assert [row["relative_path"] for row in selected][0] == "a/A.h5ad"
    assert all("ARI" not in row and "labels" not in row for row in selected)
