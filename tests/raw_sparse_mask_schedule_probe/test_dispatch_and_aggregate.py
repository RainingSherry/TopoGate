from pathlib import Path

from scripts.raw_sparse_mask_schedule_probe import aggregate, overnight, protocol


def test_queue_is_fixed_and_round_robin():
    queue = overnight._main_queue()
    assert len(queue) == 90
    assert queue[:3] == [(protocol.DATASETS[0], protocol.ARMS[0], 42), (protocol.DATASETS[0], protocol.ARMS[1], 42), (protocol.DATASETS[0], protocol.ARMS[2], 42)]
    assignments = overnight.queue_assignments([1, 2, 3])
    flattened = [cell for gpu in [1, 2, 3] for cell in assignments[gpu]]
    assert sorted(flattened) == sorted(queue)
    assert all(abs(len(assignments[1]) - len(assignments[gpu])) <= 1 for gpu in [2, 3])


def test_dispatch_without_idle_gpu_is_guarded():
    result = overnight.dispatch_main([], output_root=Path("/tmp/raw_sparse_mask_test_main"))
    assert result["status"] == "GPU_WAITING"
    assert result["launched"] == 0


def test_dispatch_rechecks_requested_gpu_occupancy(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        overnight,
        "gpu_snapshot",
        lambda: {"status": "completed_valid", "gpus": [], "legal_idle_gpus": [1]},
    )
    result = overnight.dispatch_main([1, 6], output_root=tmp_path)
    assert result["status"] == "GPU_WAITING"
    assert result["launched"] == 0
    assert result["requested_gpus"] == [1, 6]
    assert result["legal_idle_gpus"] == [1]
    assert result["occupancy_guard"] == "failed_at_dispatch"


def test_g0_fails_closed_when_main_or_svd_is_missing(tmp_path: Path):
    bundle = aggregate.collect(tmp_path)
    evaluation = aggregate.evaluate(bundle)
    assert evaluation["decision"] == "INCOMPLETE_COMPUTE"
    assert not evaluation["g0"]["passed"]
    assert "main_matrix_coverage" in evaluation["g0"]["failures"]
