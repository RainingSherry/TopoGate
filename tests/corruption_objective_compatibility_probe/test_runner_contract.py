from __future__ import annotations

import os

from scripts.corruption_objective_compatibility_probe import runner
from scripts.corruption_objective_compatibility_probe import overnight


def test_formal_runner_accepts_only_one_legal_physical_gpu(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert runner._cuda_visible_is_legal() is True
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert runner._cuda_visible_is_legal() is False


def test_cuda_preflight_does_not_fail_just_because_forbidden_devices_are_enumerated(monkeypatch) -> None:
    class Result:
        returncode = 0
        stdout = "0, H100, 80000, 0, 0\n1, H100, 80000, 0, 0\n2, H100, 80000, 0, 0\n3, H100, 80000, 0, 0\n4, H100, 80000, 0, 0\n5, H100, 80000, 0, 0\n6, H100, 80000, 0, 0\n7, H100, 80000, 0, 0\n"
        stderr = ""

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(overnight.subprocess, "run", lambda *args, **kwargs: Result())
    result = overnight._cuda_preflight()
    assert result["status"] == "ready"
    assert result["forbidden_visible"] == [0, 7]
    assert result["forbidden_requested"] == []
    assert result["idle_legal_pool"] == [1, 2, 3, 4, 5, 6]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert runner._cuda_visible_is_legal() is False
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert runner._cuda_visible_is_legal() is False
