from pathlib import Path

from scripts.sparse_corruption_principle_probe.local_audit import _finite_csv


def test_csv_finite_helper_rejects_nonfinite_values(tmp_path: Path):
    path = tmp_path / "rows.csv"
    path.write_text("a,b\n1,2\n3,nan\n", encoding="utf-8")
    ok, rows, bad = _finite_csv(path)
    assert ok is False
    assert rows == 2
    assert bad == ["row1:b"]

